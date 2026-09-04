#!/usr/bin/env python3

"""Contract checks for the mthreads tools/run_tests.py policy hooks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
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


def metric(median: float) -> dict[str, object]:
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


def paired_metrics(case: str) -> dict[str, dict[str, object]]:
    flagdnn = metric(2.0)
    flagdnn["provider"] = "flagdnn"
    flagdnn["case"] = case
    mudnn = metric(3.0)
    mudnn["provider"] = "mudnn"
    mudnn["case"] = case
    return {"flagdnn": flagdnn, "mudnn": mudnn}


def main() -> int:
    if len(sys.argv) != 2:
        raise RuntimeError("usage: run_tests_adapter_contract.py ADAPTER")
    adapter = load_adapter(Path(sys.argv[1]).resolve())
    require(
        adapter.FILTER_REGISTERED_TESTS is False,
        "full-manifest registration gate",
    )
    require(adapter.PREFLIGHT_BY_DEFAULT is True, "preflight default")
    require(adapter.SUPPORTS_MIN_SPEEDUP is True, "speedup support")

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
    require(
        "FLAGDNN_BENCHMARK_CASE" not in environment, "benchmark filter"
    )
    require(environment["MUSA_VISIBLE_DEVICES"] == "3", "device mask")
    for variable in (
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        require(variable not in environment, f"stale visibility {variable}")

    inherited = {
        variable: "kept" for variable in adapter.VISIBILITY_VARIABLES
    }
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

    functional_result: dict[str, object] = {"status": "passed"}
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
        convolution_result: dict[str, object] = {"status": "passed"}
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

    wrong_convolution_result: dict[str, object] = {"status": "passed"}
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
    benchmark_result: dict[str, object] = {"status": "passed"}
    adapter.postprocess_result(
        result=benchmark_result,
        ctest_reported_status="passed",
        output=(
            "18: FLAGDNN_ADD_BENCHMARK: "
            "PASS cases=1 executed=1 skipped=0\n"
        ),
        operator="add",
        suite="benchmark",
        records=records,
        manifest_operators=["add"],
    )
    require(benchmark_result["status"] == "passed", "benchmark accounting")

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
            convolution_benchmark: dict[str, object] = {"status": "passed"}
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
    wrong_direction_result: dict[str, object] = {"status": "passed"}
    adapter.postprocess_result(
        result=wrong_direction_result,
        ctest_reported_status="passed",
        output=(
            "18: FLAGDNN_CONV_WGRAD_BENCHMARK: "
            "PASS cases=1 executed=1 skipped=0\n"
        ),
        operator="conv_wgrad",
        suite="benchmark",
        records={
            wrong_direction_case: paired_metrics(wrong_direction_case)
        },
        manifest_operators=ownership_operators,
    )
    require(
        wrong_direction_result["status"] == "failed",
        "conv_wgrad accepted a conv_fprop benchmark case",
    )

    overlapping_case = "batchnorm_inference_perf_fp32_case"
    overlapping_result: dict[str, object] = {"status": "passed"}
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

    invalid_result: dict[str, object] = {"status": "passed"}
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
