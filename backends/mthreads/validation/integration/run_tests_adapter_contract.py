#!/usr/bin/env python3

"""Contract checks for the mthreads tools/run_tests.py policy hooks."""

from __future__ import annotations

from typing import Any

import importlib.util
import copy
import json
import math
from pathlib import Path
import re
import sys
from types import ModuleType


def load_adapter(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_flagdnn_mthreads_adapter_contract", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create mthreads adapter import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def metric(median: float) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "steady_state",
        "provider": "",
        "case": "add_perf_fp32",
        "unit": "us",
        "median": median,
        "p90": median,
        "samples": [median],
    }


def paired_metrics(case: str) -> dict[str, dict[str, Any]]:
    flagdnn = metric(2.0)
    flagdnn["provider"] = "flagdnn"
    flagdnn["case"] = case
    mudnn = metric(3.0)
    mudnn["provider"] = "mudnn"
    mudnn["case"] = case
    return {"flagdnn": flagdnn, "mudnn": mudnn}


def check_equivalent_records(adapter: ModuleType) -> None:
    public_result: dict[str, Any]
    output: dict[str, Any]
    record: dict[str, Any]
    case = "elu_backward_perf_fp32_1x1x1024_by_1x1x1024"
    record = {
        "schema_version": 4,
        "kind": "steady_state",
        "provider": "flagdnn",
        "case": case,
        "unit": "us",
        "median": 10.0,
        "p90": 10.0,
        "samples": [10.0] * 20,
        "equivalent_baseline": {
            "provider": "nvidia_equivalent",
            "platform": "nvidia",
            "target_platform": "mthreads",
            "scale_metric": "vector_fp32",
            "timing_profile": "native",
            "reference_validation": "cpu_full",
            "reference_time_us": 2.0,
            "reference_peak": 20.0,
            "target_peak": 10.0,
            "min_relative_efficiency": 0.8,
            "traffic_ratio": 1.0,
            "equivalent_time_us": 4.0,
            "limit_us": 5.0,
            "relative_efficiency": 0.4,
            "passed": False,
        },
    }
    adapter.validate_benchmark_record(record)
    for location, field, value in (
        ("record", "schema_version", 1),
        ("record", "provider", "mudnn"),
        ("record", "case", "pow_int32_1x1x16"),
        ("record", "case", "elu_backward_attributes_int32_1x1x65"),
        ("record", "case", "sigmoid_backward_attributes_fp32_1x1x65"),
        ("record", "case", "elu_backward_unknown_fp32_1x1x65"),
        ("record", "samples", [10.0] * 7),
        ("record", "samples", [float("nan")] * 20),
        ("record", "median", 9.0),
        ("record", "p90", 11.0),
        ("baseline", "target_platform", "hygon"),
        ("baseline", "timing_profile", "attention"),
        ("baseline", "reference_validation", "sanity"),
        ("baseline", "target_peak", 0.0),
        ("baseline", "reference_peak", float("inf")),
        ("baseline", "scale_metric", "vector_fp16"),
        ("baseline", "traffic_ratio", 2.0),
        ("baseline", "equivalent_time_us", 8.0),
        ("baseline", "limit_us", 10.0),
        ("baseline", "relative_efficiency", 0.8),
        ("baseline", "passed", True),
    ):
        invalid = copy.deepcopy(record)
        target = (
            invalid if location == "record" else invalid["equivalent_baseline"]
        )
        target[field] = value
        try:
            adapter.validate_benchmark_record(invalid)
        except ValueError:
            pass
        else:
            raise RuntimeError(
                f"accepted invalid equivalent {location}.{field}"
            )

    for prefix in (
        "relu_backward_attributes",
        "leaky_relu_backward_attributes",
        "swish_backward_attributes",
        "softplus_backward_perf",
        "elu_backward",
        "softplus_backward",
        "elu_backward_attributes",
        "softplus_backward_attributes",
    ):
        valid = copy.deepcopy(record)
        valid["case"] = prefix + "_bfloat16_1x1x65"
        valid["equivalent_baseline"]["scale_metric"] = (
            "hbm_triad"
            if prefix.startswith(("relu_", "leaky_relu_"))
            else "vector_fp32"
        )
        adapter.validate_benchmark_record(valid)
        invalid = copy.deepcopy(valid)
        invalid["equivalent_baseline"]["scale_metric"] = (
            "vector_fp32"
            if valid["equivalent_baseline"]["scale_metric"] == "hbm_triad"
            else "hbm_triad"
        )
        try:
            adapter.validate_benchmark_record(invalid)
        except ValueError:
            pass
        else:
            raise RuntimeError(
                "equivalent record accepted another operator's metric"
            )

    relaxed = copy.deepcopy(record)
    relaxed["equivalent_baseline"].update(
        min_relative_efficiency=0.1,
        limit_us=40.0,
        passed=True,
    )
    try:
        adapter.validate_benchmark_record(relaxed)
    except ValueError:
        pass
    else:
        raise RuntimeError(
            "self-consistent baseline relaxed the minimum below 0.8"
        )

    providers = {"flagdnn": record}
    require(
        not adapter._validate_benchmark_pairs(
            {case: providers},
            "elu_backward",
            ["elu", "elu_backward"],
        ),
        "valid equivalent record rejected",
    )
    require(
        bool(
            adapter._validate_benchmark_pairs(
                {case: {**providers, "mudnn": metric(4)}},
                "elu_backward",
                ["elu_backward"],
            )
        ),
        "equivalent record accepted fabricated reference samples",
    )
    require(
        bool(
            adapter._validate_benchmark_pairs(
                {case: providers},
                "elu",
                ["elu", "elu_backward"],
            )
        ),
        "equivalent record escaped operator boundary",
    )

    result = {"status": "passed", "exit_code": 0}
    adapter.postprocess_result(
        result=result,
        ctest_reported_status="passed",
        output=(
            "FLAGDNN_ELU_BACKWARD_BENCHMARK: PASS cases=1 executed=1 "
            "skipped=0\n"
        ),
        operator="elu_backward",
        suite="benchmark",
        records={case: providers},
        manifest_operators=["elu_backward"],
    )
    require(
        result["status"] == "passed",
        "below-baseline timing became execution failure",
    )
    failed_execution = {"status": "failed", "exit_code": 1}
    adapter.postprocess_result(
        result=failed_execution,
        ctest_reported_status="failed",
        output="missing NVIDIA baseline after an earlier measured case\n",
        operator="elu_backward",
        suite="benchmark",
        records={case: providers},
        manifest_operators=["elu_backward"],
    )
    require(
        failed_execution["status"] == "failed",
        "earlier valid timings concealed an incomplete execution",
    )
    results = {
        "elu_backward": {"benchmark": {**result, "records": {case: providers}}}
    }
    arguments = dict(
        results=results,
        suite_operators={"benchmark": ["elu_backward"]},
        suites=["benchmark"],
        state={},
        preflight_passed=True,
    )
    ungated = adapter.finalize(**arguments, min_speedup=None)
    require(not ungated["failed"], "default equivalent report enabled a gate")
    equivalent = ungated["summary"]["equivalent_performance"]
    require(
        equivalent["case_count"] == 1
        and equivalent["failed_case_count"] == 1
        and equivalent["gate_passed"] is None,
        "default report concealed the below-baseline verdict",
    )
    strict = adapter.finalize(**arguments, min_speedup=0.1)
    require(
        strict["failed"]
        and strict["summary"]["equivalent_performance"]["cases"][0][
            "required_relative_efficiency"
        ]
        == 0.8,
        "CLI threshold relaxed the baseline minimum",
    )

    public_result = {
        "elu_backward": {
            "performance": {
                "status": "Passed",
                "exit_code": 0,
                "data": {
                    "fp32": {
                        "result": "OK",
                        "details": {
                            case: {
                                "base": 0.004,
                                "gems": 0.01,
                                "equivalent_baseline": copy.deepcopy(
                                    record["equivalent_baseline"]
                                ),
                            }
                        },
                    }
                },
            }
        }
    }
    for final in (ungated, strict):
        partition = copy.deepcopy(public_result)
        report = {"result": copy.deepcopy(public_result)}
        adapter.postprocess_batch_report(
            partitions={0: partition},
            report=report,
            platform_summary=final["summary"],
            exit_code=int(final["failed"]),
        )
        for output in (partition, report["result"]):
            performance = output["elu_backward"]["performance"]
            row = performance["data"]["fp32"]["details"][case]
            require(
                row["equivalent_baseline"]["passed"] is False,
                "report changed measured baseline verdict",
            )
            require(
                row["performance_gate"]["enabled"] is final["failed"],
                "report lost opt-in gate state",
            )
            require(
                performance["status"]
                == ("Failed" if final["failed"] else "Passed"),
                "partition and merged performance gate differ",
            )

    # Exercise the real generic parser and public performance conversion.
    runner_path = (
        Path(adapter.__file__).resolve().parents[3] / "tools/run_tests.py"
    )
    runner = load_adapter(runner_path)
    parsed, errors = runner.benchmark_records(json.dumps(record), adapter)
    require(
        not errors and parsed[case] == providers,
        "generic parser lost equivalent record",
    )
    row = runner.performance_data({"records": parsed})["fp32"]["details"][case]
    require(
        row["base"] == 0.004
        and row["gems"] == 0.01
        and row["equivalent_baseline"]["passed"] is False,
        "equivalent public report fabricated a native latency",
    )

    native_case = "elu_backward_fp32_2x4x8"
    native_pair = paired_metrics(native_case)
    native_pair["mudnn"].update(median=0.2, p90=0.2, samples=[0.2])
    for include_equivalent in (False, True):
        mixed = {native_case: native_pair}
        if include_equivalent:
            fast = copy.deepcopy(record)
            fast.update(median=1.0, p90=1.0, samples=[1.0] * 20)
            fast["equivalent_baseline"].update(
                relative_efficiency=4.0, passed=True
            )
            mixed[case] = {"flagdnn": fast}
        final = adapter.finalize(
            **{
                **arguments,
                "results": {
                    "elu_backward": {
                        "benchmark": {
                            "status": "passed",
                            "records": mixed,
                        }
                    }
                },
            },
            min_speedup=0.8,
        )
        require(final["failed"], "native failure hidden by equivalent success")
        public = {
            "elu_backward": {
                "performance": {
                    "status": "Passed",
                    "exit_code": 0,
                    "data": {
                        "fp32": {
                            "result": "OK",
                            "details": {
                                native_case: {"speedup": 0.1},
                                case: {"speedup": 4.0},
                            },
                        }
                    },
                }
            }
        }
        partition = copy.deepcopy(public)
        report = {"result": copy.deepcopy(public)}
        adapter.postprocess_batch_report(
            partitions={0: partition},
            report=report,
            platform_summary=final["summary"],
            exit_code=1,
        )
        require(
            partition["elu_backward"]["performance"]["status"] == "Failed"
            and report["result"]["elu_backward"]["performance"]["status"]
            == "Failed",
            "native speedup failure missing from partition or merged report",
        )


def check_equivalent_gate_boundaries(adapter: ModuleType) -> None:
    """Latency limits must survive reporting without a second ratio verdict."""
    output: dict[str, Any]
    record: dict[str, Any]
    runner_path = (
        Path(adapter.__file__).resolve().parents[3] / "tools/run_tests.py"
    )
    runner = load_adapter(runner_path)
    case = "elu_backward_perf_fp32_1x1x1024_by_1x1x1024"
    equivalent = 0.409
    baseline_limit = equivalent / 0.8
    require(
        equivalent / baseline_limit < 0.8,
        "boundary fixture must expose division rounding below the minimum",
    )
    strict_limit = equivalent / 0.9
    scenarios = [
        (threshold, median, passed)
        for threshold in (None, 0.1, 0.8)
        for median, passed in (
            (math.nextafter(baseline_limit, 0.0), True),
            (baseline_limit, True),
            (math.nextafter(baseline_limit, math.inf), False),
        )
    ] + [
        (0.9, math.nextafter(strict_limit, 0.0), True),
        (0.9, strict_limit, True),
        (0.9, math.nextafter(strict_limit, math.inf), False),
        (0.9, baseline_limit, False),
    ]
    for threshold, median, expected_passed in scenarios:
        label = f"threshold={threshold!r} median={median!r}"
        record = {
            "schema_version": 4,
            "kind": "steady_state",
            "provider": "flagdnn",
            "case": case,
            "unit": "us",
            "median": median,
            "p90": median,
            "samples": [median] * 20,
            "equivalent_baseline": {
                "provider": "nvidia_equivalent",
                "platform": "nvidia",
                "target_platform": "mthreads",
                "scale_metric": "vector_fp32",
                "timing_profile": "native",
                "reference_validation": "cpu_full",
                "reference_time_us": equivalent,
                "reference_peak": 1.0,
                "target_peak": 1.0,
                "min_relative_efficiency": 0.8,
                "traffic_ratio": 1.0,
                "equivalent_time_us": equivalent,
                "limit_us": baseline_limit,
                "relative_efficiency": equivalent / median,
                "passed": median <= baseline_limit,
            },
        }
        parsed, errors = runner.benchmark_records(json.dumps(record), adapter)
        require(
            not errors and case in parsed, f"{label}: valid timing rejected"
        )
        final = adapter.finalize(
            results={
                "elu_backward": {
                    "benchmark": {
                        "status": "passed",
                        "records": parsed,
                    }
                }
            },
            suite_operators={"benchmark": ["elu_backward"]},
            suites=["benchmark"],
            state={},
            min_speedup=threshold,
            preflight_passed=True,
        )
        enabled = threshold is not None
        expected_failed = enabled and not expected_passed
        summary = final["summary"]["equivalent_performance"]
        require(
            final["failed"] is expected_failed
            and summary["cases"][0]["passed"] is expected_passed
            and summary["passed_case_count"] == int(expected_passed)
            and summary["failed_case_count"] == int(not expected_passed)
            and summary["gate_passed"]
            is (expected_passed if enabled else None),
            f"{label}: latency boundary changed during finalization",
        )
        require(
            summary["cases"][0]["required_relative_efficiency"]
            == (0.9 if threshold == 0.9 else 0.8),
            f"{label}: baseline or CLI minimum changed",
        )

        public = {
            "elu_backward": {
                "performance": {
                    "status": "Passed",
                    "exit_code": 0,
                    "data": runner.performance_data({"records": parsed}),
                }
            }
        }
        single = {"result": copy.deepcopy(public)}
        merged = {"result": copy.deepcopy(public)}
        partition = copy.deepcopy(public)
        require(
            adapter.postprocess_report(
                report=single,
                platform_summary=final["summary"],
                exit_code=int(expected_failed),
            ),
            f"{label}: single report skipped equivalent postprocessing",
        )
        require(
            adapter.postprocess_batch_report(
                partitions={0: partition},
                report=merged,
                platform_summary=final["summary"],
                exit_code=int(expected_failed),
            ),
            f"{label}: batch report skipped equivalent postprocessing",
        )
        for report in (single, merged):
            require(
                report["overall_status"]
                == ("failed" if expected_failed else "passed")
                and report["exit_code"] == int(expected_failed),
                f"{label}: overall report verdict differs from finalization",
            )
        for output in (single["result"], merged["result"], partition):
            performance = output["elu_backward"]["performance"]
            entry = performance["data"]["fp32"]
            row = entry["details"][case]
            gate = row["performance_gate"]
            require(
                performance["status"]
                == ("Failed" if expected_failed else "Passed")
                and performance["exit_code"] == int(expected_failed)
                and entry["result"] == ("FAIL" if expected_failed else "OK")
                and gate["enabled"] is enabled
                and gate["passed"] is (expected_passed if enabled else None),
                f"{label}: single/merged/partition gate verdict differs",
            )
            require(
                row["equivalent_baseline"] == record["equivalent_baseline"],
                f"{label}: reporting altered the measured baseline verdict",
            )


def main() -> int:
    mixed_result: dict[str, Any]
    if len(sys.argv) != 2:
        raise RuntimeError("usage: run_tests_adapter_contract.py ADAPTER")
    adapter = load_adapter(Path(sys.argv[1]).resolve())
    check_equivalent_records(adapter)
    check_equivalent_gate_boundaries(adapter)
    require(
        adapter.FILTER_REGISTERED_TESTS is False,
        "full-manifest registration gate",
    )
    require(adapter.PREFLIGHT_BY_DEFAULT is True, "preflight default")
    require(adapter.SUPPORTS_MIN_SPEEDUP is True, "speedup support")

    for suite, operator, suffixes in (
        ("functional", "matmul", ("", ".fp8")),
        ("benchmark", "matmul", ("", ".matrix", ".fp8")),
        ("benchmark", "genstats", (".matrix",)),
        ("benchmark", "add", ("", ".matrix")),
    ):
        expression = adapter.test_expression(suite, operator)
        require(expression is not None, "missing dtype/matrix test selector")
        for suffix in suffixes:
            require(
                re.fullmatch(
                    expression, f"{suite}.mthreads.{operator}{suffix}"
                )
                is not None,
                f"missing {suite} {operator}{suffix} coverage",
            )
        for invalid in (
            f"{suite}.nvidia.{operator}",
            f"{suite}.mthreads.{operator}_backward",
            f"{suite}.mthreads.{operator}.unknown",
        ):
            require(
                re.fullmatch(expression, invalid) is None,
                "test selector escaped the operator boundary",
            )
    require(
        adapter.test_expression("functional", "add") is None,
        "ordinary functional test should use the default selector",
    )

    environment = {
        "FLAGDNN_ADD_CASE": "odd_extent",
        "FLAGDNN_BENCHMARK_CASE": "add_perf_fp32",
        "MUSA_VISIBLE_DEVICES": "old",
        "CUDA_VISIBLE_DEVICES": "old-cuda",
        "HIP_VISIBLE_DEVICES": "old-hip",
        "ROCR_VISIBLE_DEVICES": "old-rocr",
        "GPU_DEVICE_ORDINAL": "old-ordinal",
    }
    adapter.configure_environment(environment, "3")
    require("FLAGDNN_ADD_CASE" not in environment, "functional filter")
    require("FLAGDNN_BENCHMARK_CASE" not in environment, "benchmark filter")
    require(environment["MUSA_VISIBLE_DEVICES"] == "3", "device mask")
    for variable in (
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        require(variable not in environment, f"stale visibility {variable}")

    inherited = {variable: "kept" for variable in adapter.VISIBILITY_VARIABLES}
    adapter.configure_environment(inherited, None)
    require(
        all(
            inherited[variable] == "kept"
            for variable in adapter.VISIBILITY_VARIABLES
        ),
        "unspecified device changed inherited visibility",
    )
    require(environment["TRITON_JIT_BACKEND"] == "MTGPU", "JIT backend")

    required = adapter.preflight_tests(("functional", "benchmark"))
    for name in (
        "integration.mthreads.cmake_configuration_contract",
        "integration.mthreads.runtime",
        "integration.mthreads.compiler_contract",
        "integration.mthreads.artifact_contract",
        "integration.mthreads.jit_add",
    ):
        require(name in required, f"missing preflight test {name}")

    functional_result: dict[str, Any] = {"status": "passed"}
    adapter.postprocess_result(
        result=functional_result,
        ctest_reported_status="passed",
        output=(
            "17: FLAGDNN_ADD_FUNCTIONAL: "
            "PASS cases=5 executed=5 skipped=0\n"
        ),
        operator="add",
        suite="functional",
        records={},
        manifest_operators=["add"],
    )
    require(functional_result["status"] == "passed", "functional accounting")

    for operator in ("conv_dgrad", "conv_fprop", "conv_wgrad"):
        convolution_result: dict[str, Any] = {"status": "passed"}
        adapter.postprocess_result(
            result=convolution_result,
            ctest_reported_status="passed",
            output=(
                "36: FLAGDNN_CONVOLUTION_FUNCTIONAL: "
                "PASS cases=27 executed=27 skipped=0\n"
            ),
            operator=operator,
            suite="functional",
            records={},
            manifest_operators=[operator],
        )
        require(
            convolution_result["status"] == "passed",
            f"shared functional convolution accounting for {operator}",
        )

    wrong_convolution_result: dict[str, Any] = {"status": "passed"}
    adapter.postprocess_result(
        result=wrong_convolution_result,
        ctest_reported_status="passed",
        output=(
            "36: FLAGDNN_CONV_DGRAD_FUNCTIONAL: "
            "PASS cases=27 executed=27 skipped=0\n"
        ),
        operator="conv_dgrad",
        suite="functional",
        records={},
        manifest_operators=["conv_dgrad"],
    )
    require(
        wrong_convolution_result["status"] == "failed",
        "direction-specific functional convolution marker rejection",
    )

    flagdnn = metric(2.0)
    flagdnn["provider"] = "flagdnn"
    mudnn = metric(3.0)
    mudnn["provider"] = "mudnn"
    records = {
        "add_perf_fp32": {
            "flagdnn": flagdnn,
            "mudnn": mudnn,
        }
    }
    benchmark_result: dict[str, Any] = {"status": "passed"}
    adapter.postprocess_result(
        result=benchmark_result,
        ctest_reported_status="passed",
        output=(
            "18: [timing] case=add_perf_fp32 method=musa_event_batch "
            "execution_count=20\n"
            "18: FLAGDNN_ADD_BENCHMARK: "
            "PASS cases=1 executed=1 skipped=0\n"
        ),
        operator="add",
        suite="benchmark",
        records=records,
        manifest_operators=["add"],
    )
    require(benchmark_result["status"] == "passed", "benchmark accounting")
    require(
        benchmark_result["timing_methods"]
        == {
            "add_perf_fp32": {
                "method": "musa_event_batch",
                "execution_count": 20,
            }
        },
        "direct Event timing method must remain in the diagnostic report",
    )

    ownership_operators = [
        "batchnorm",
        "batchnorm_inference",
        "conv_dgrad",
        "conv_fprop",
        "conv_wgrad",
    ]
    for direction in ("dgrad", "fprop", "wgrad"):
        operator = f"conv_{direction}"
        for spatial_rank in (1, 2, 3):
            case = f"conv{spatial_rank}d_{direction}_perf_fp32_case"
            convolution_benchmark: dict[str, Any] = {"status": "passed"}
            adapter.postprocess_result(
                result=convolution_benchmark,
                ctest_reported_status="passed",
                output=(
                    f"18: FLAGDNN_{operator.upper()}_BENCHMARK: "
                    "PASS cases=1 executed=1 skipped=0\n"
                ),
                operator=operator,
                suite="benchmark",
                records={case: paired_metrics(case)},
                manifest_operators=ownership_operators,
            )
            require(
                convolution_benchmark["status"] == "passed",
                f"{operator} rejected rank-qualified case {case}",
            )

    wrong_direction_case = "conv2d_fprop_perf_fp32_case"
    wrong_direction_result: dict[str, Any] = {"status": "passed"}
    adapter.postprocess_result(
        result=wrong_direction_result,
        ctest_reported_status="passed",
        output=(
            "18: FLAGDNN_CONV_WGRAD_BENCHMARK: "
            "PASS cases=1 executed=1 skipped=0\n"
        ),
        operator="conv_wgrad",
        suite="benchmark",
        records={wrong_direction_case: paired_metrics(wrong_direction_case)},
        manifest_operators=ownership_operators,
    )
    require(
        wrong_direction_result["status"] == "failed",
        "conv_wgrad accepted a conv_fprop benchmark case",
    )

    overlapping_case = "batchnorm_inference_perf_fp32_case"
    overlapping_result: dict[str, Any] = {"status": "passed"}
    adapter.postprocess_result(
        result=overlapping_result,
        ctest_reported_status="passed",
        output=(
            "18: FLAGDNN_BATCHNORM_BENCHMARK: "
            "PASS cases=1 executed=1 skipped=0\n"
        ),
        operator="batchnorm",
        suite="benchmark",
        records={overlapping_case: paired_metrics(overlapping_case)},
        manifest_operators=ownership_operators,
    )
    require(
        overlapping_result["status"] == "failed",
        "batchnorm accepted a batchnorm_inference benchmark case",
    )

    mixed_output = (
        "17: FLAGDNN_ADD_BENCHMARK: PASS cases=1 executed=1 skipped=0\n"
        "18: [skip] case=add_unavailable provider=mudnn reason=NOT_SUPPORTED\n"
        "18: FLAGDNN_ADD_BENCHMARK: SKIP cases=1 executed=0 skipped=1\n"
    )
    mixed_result = {
        "status": "failed",
        "exit_code": 0,
        "record_errors": ["skipped benchmark emitted timing provider records"],
    }
    adapter.postprocess_result(
        result=mixed_result,
        ctest_reported_status="skipped",
        output=mixed_output,
        operator="add",
        suite="benchmark",
        records=records,
        manifest_operators=["add"],
    )
    require(mixed_result["status"] == "passed", "mixed CTest group")
    require(
        mixed_result["case_skips"]
        == [
            {
                "case": "add_unavailable",
                "provider": "mudnn",
                "reason": "NOT_SUPPORTED",
            }
        ],
        "mixed group lost native skip reasons",
    )
    require(
        mixed_result["case_accounting"]["skipped"] == 1,
        "mixed group lost skips",
    )
    invalid_skip = {"status": "skipped", "exit_code": 0}
    adapter.postprocess_result(
        result=invalid_skip,
        ctest_reported_status="skipped",
        output="FLAGDNN_RNG_FUNCTIONAL: SKIP cases=1 executed=0 skipped=1\n",
        operator="rng",
        suite="functional",
        records={},
        manifest_operators=["rng"],
    )
    require(invalid_skip["status"] == "failed", "undocumented skip accepted")

    invalid_result: dict[str, Any] = {"status": "passed"}
    adapter.postprocess_result(
        result=invalid_result,
        ctest_reported_status="passed",
        output="",
        operator="add",
        suite="benchmark",
        records={"add_perf_fp32": {"flagdnn": flagdnn}},
        manifest_operators=["add"],
    )
    require(invalid_result["status"] == "failed", "invalid pair rejection")

    results = {
        "add": {
            "functional": {"status": "passed"},
            "benchmark": {"status": "passed", "records": records},
        }
    }
    passed = adapter.finalize(
        results=results,
        suite_operators={"functional": ["add"], "benchmark": ["add"]},
        suites=["functional", "benchmark"],
        state={},
        min_speedup=1.0,
        preflight_passed=True,
    )
    require(not passed["failed"], "valid finalize result")
    failed = adapter.finalize(
        results=results,
        suite_operators={"functional": ["add"], "benchmark": ["add"]},
        suites=["functional", "benchmark"],
        state={},
        min_speedup=2.0,
        preflight_passed=True,
    )
    require(failed["failed"], "speedup threshold rejection")
    print("mthreads run_tests adapter contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
