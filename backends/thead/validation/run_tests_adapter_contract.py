#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic contracts for the THead tools/run_tests.py policy adapter."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from typing import Any


SOURCE_ROOT = Path(__file__).resolve().parents[3]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def timing(case: str, provider: str, median: float = 2.0) -> str:
    samples = [median * 0.5, median, median * 1.5]
    return json.dumps(
        {
            "schema_version": 1,
            "kind": "steady_state",
            "provider": provider,
            "case": case,
            "unit": "us",
            "median": samples[1],
            "p90": samples[2],
            "samples": samples,
        },
        separators=(",", ":"),
    )


def pair(
    case: str, flagdnn: float = 2.0, acdnn: float = 4.0
) -> dict[str, Any]:
    return {
        "flagdnn": json.loads(timing(case, "flagdnn", flagdnn)),
        "acdnn": json.loads(timing(case, "acdnn", acdnn)),
    }


def skip_line(
    case: str,
    *,
    operator: str = "add",
    reason: str = "dtype_unsupported",
) -> str:
    return (
        f"[SKIP][acdnn] op={operator} case={case} reason={reason} "
        "sdk=2.0.0-715aa1 acdnn_header=1400 acdnn_runtime=1400 "
        "target=ppu_contract_cc80 dtype=fp16 layout=contiguous shape=16"
    )


def postprocess(
    adapter: Any,
    *,
    status: str,
    output: str,
    suite: str = "functional",
    records: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"status": status}
    adapter.postprocess_result(
        result=result,
        ctest_reported_status=status,
        output=output,
        operator="add",
        suite=suite,
        records={} if records is None else records,
        manifest_operators=["add", "add_square", "mul"],
    )
    return result


def invoke_main(runner: Any, arguments: list[str]) -> tuple[int, str, str]:
    original = sys.argv
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        sys.argv = [str(runner.__file__), *arguments]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
            stderr
        ):
            exit_code = runner.main()
    finally:
        sys.argv = original
    return exit_code, stdout.getvalue(), stderr.getvalue()


def run_native_output(
    runner: Any,
    adapter: Any,
    output: str,
    *,
    operator: str = "add",
    suite: str = "functional",
    exit_code: int = 0,
) -> dict[str, Any]:
    original_process = runner.run_process_group
    runner.run_process_group = lambda *_args: (output, "", exit_code, False)
    try:
        return runner.run_one(
            Path("."),
            operator,
            suite,
            "thead",
            {},
            30,
            False,
            [operator],
            adapter=adapter,
            capture_output=True,
        )
    finally:
        runner.run_process_group = original_process


def check_cpu_reference_accounting(runner: Any, adapter: Any) -> None:
    acdnn_pass = (
        "1: add_acdnn_fp32: FlagDNN Graph vs acDNN PASS max_abs=0 max_rel=0\n"
    )
    cpu_pass = (
        "1: add_cpu_int32: FlagDNN Graph vs CPU reference PASS"
        " fallback_reason=dtype_unsupported\n"
    )
    complete_output = (
        acdnn_pass
        + cpu_pass
        + "1: FLAGDNN_ADD_FUNCTIONAL: PASS cases=2 executed=2 skipped=0\n"
        + "100% tests passed, 0 tests failed out of 1\n"
    )
    with tempfile.TemporaryDirectory(
        prefix="flagdnn-thead-cpu-reference-contract-"
    ) as temporary_text:
        directory = Path(temporary_text)
        complete = run_native_output(runner, adapter, complete_output)
        batch = runner.batch_suite_result(
            "add", "functional", complete, directory
        )
        require(
            complete["status"] == "passed"
            and complete["case_counts"]["passed"] == 2
            and complete["case_counts"]["skipped"] == 0
            and batch["status"] == "Passed"
            and batch["total"] == 2
            and batch["passed"] == 2
            and batch["skipped"] == 0,
            "mixed acDNN/CPU reference accuracy did not pass batch accounting",
        )
        accuracy = json.loads(
            (directory / "add/accuracy_result.json").read_text(
                encoding="utf-8"
            )
        )
        require(
            len(accuracy) == 2
            and {
                record["params"].get("case"): record["result"]
                for record in accuracy.values()
            }
            == {"add_acdnn_fp32": "passed", "add_cpu_int32": "passed"},
            "CPU reference case identity was lost in accuracy_result.json",
        )

        cpu_only = run_native_output(
            runner,
            adapter,
            "1: mod_cpu_fp32: FlagDNN Graph vs CPU reference PASS"
            " fallback_reason=shape_unsupported\n"
            "1: FLAGDNN_MOD_FUNCTIONAL: PASS cases=1 executed=1 skipped=0\n"
            "100% tests passed, 0 tests failed out of 1\n",
            operator="mod",
        )
        cpu_only_batch = runner.batch_suite_result(
            "mod", "functional", cpu_only, directory
        )
        require(
            cpu_only_batch["status"] == "Passed"
            and cpu_only_batch["passed"] == 1
            and cpu_only_batch["skipped"] == 0,
            "CPU-only reference accuracy was treated as an acDNN skip",
        )

        partial = run_native_output(
            runner,
            adapter,
            acdnn_pass
            + cpu_pass
            + "1: "
            + skip_line("add_remaining_fp16")
            + "\n1: FLAGDNN_ADD_FUNCTIONAL: PASS cases=3 executed=2 skipped=1\n"
            + "100% tests passed, 0 tests failed out of 1\n",
        )
        partial_batch = runner.batch_suite_result(
            "add", "functional", partial, directory
        )
        require(
            partial_batch["status"] == "Skipped"
            and partial_batch["passed"] == 2
            and partial_batch["skipped"] == 1,
            "CPU reference PASS hid a remaining native accuracy skip",
        )

        for failed_output, failed_cases in (
            (
                acdnn_pass
                + "1: add_cpu_int32: FlagDNN Graph vs CPU reference FAIL"
                " max_abs=1\n",
                1,
            ),
            (complete_output, 0),
        ):
            failed = run_native_output(
                runner, adapter, failed_output, exit_code=1
            )
            failed_batch = runner.batch_suite_result(
                "add", "functional", failed, directory
            )
            require(
                failed["status"] == "failed"
                and failed_batch["status"] == "Failed"
                and failed_batch["failed"] == failed_cases,
                "CPU reference accounting hid a numerical or process failure",
            )

        benchmark = run_native_output(
            runner,
            adapter,
            "1: "
            + skip_line("add_perf_int32")
            + "\n1: FLAGDNN_ADD_BENCHMARK: SKIP cases=1"
            " comparable_executed=0 reference_skipped=1\n"
            "1/1 Test #1: benchmark.thead.add ...***Skipped\n"
            "100% tests passed, 0 tests failed out of 1\n",
            suite="benchmark",
        )
        benchmark_batch = runner.batch_suite_result(
            "add", "benchmark", benchmark, directory
        )
        require(
            benchmark_batch["status"] == "Skipped"
            and not benchmark_batch["data"]
            and benchmark["case_accounting"]["reference_skipped"] == 1,
            "CPU accuracy fallback changed an unsupported benchmark skip",
        )


def main() -> int:
    runner = load_module(
        "flagdnn_thead_run_tests_contract_subject",
        SOURCE_ROOT / "tools/run_tests.py",
    )
    adapter = runner.load_platform_adapter("thead")
    require(adapter is not None, "THead run_tests adapter was not loaded")
    check_cpu_reference_accounting(runner, adapter)
    require(
        adapter.DEFAULT_TIMEOUT == 21600
        and adapter.PREFLIGHT_BY_DEFAULT is True
        and adapter.SUPPORTS_MIN_SPEEDUP is True
        and adapter.FILTER_REGISTERED_TESTS is False
        and adapter.SPEEDUP_METRIC == "acdnn_median_us/flagdnn_median_us",
        "THead adapter public policy constants are incorrect",
    )

    selected = adapter.select_operator_manifests(runner.operator_manifests())
    require(
        set(selected["functional"])
        == set(runner.operator_manifests()["functional"])
        and set(selected["benchmark"]) == set(selected["functional"])
        and "relu_backward" in selected["benchmark"]
        and "rng" in selected["functional"],
        "THead must retain every public operator, including explicit"
        " capability gates",
    )
    validation_only = {"status": "passed"}
    adapter.postprocess_result(
        result=validation_only,
        ctest_reported_status="passed",
        output=(
            "add_perf_fp32_contract: FlagDNN Graph vs acDNN correctness"
            " PASS\nVALIDATION_ONLY case=add_perf_fp32_contract"
            " timing=not_collected\nFLAGDNN_ADD_BENCHMARK: PASS cases=1"
            " comparable_executed=1 reference_skipped=0\n"
        ),
        operator="add",
        suite="benchmark",
        records={},
        manifest_operators=["add"],
    )
    require(
        validation_only["status"] == "failed"
        and validation_only.get("record_errors"),
        "THead accepted validation-only output as a measured benchmark",
    )
    original_process = runner.run_process_group
    runner.run_process_group = lambda *_args: (
        "1: add_case: acDNN PASS\n"
        + "1: "
        + skip_line("add_skipped")
        + "\n"
        + "1: FLAGDNN_ADD_FUNCTIONAL: PASS cases=2 executed=1 skipped=1\n"
        + "100% tests passed, 0 tests failed out of 1\n",
        "",
        0,
        False,
    )
    try:
        actual = runner.run_one(
            Path("."),
            "add",
            "functional",
            "thead",
            {},
            30,
            False,
            ["add"],
            adapter=adapter,
        )
    finally:
        runner.run_process_group = original_process
    require(
        actual["status"] == "passed"
        and actual["case_counts"]["passed"] == 1
        and actual["case_counts"]["skipped"] == 1
        and actual["case_counts"]["total"] == 2,
        "THead execution accounting was lost in the legacy summary",
    )
    manifests = adapter.select_operator_manifests(runner.operator_manifests())
    manifest_operators = list(
        dict.fromkeys(
            operator
            for operators in manifests.values()
            for operator in operators
        )
    )
    for case_name, expected_owner in (
        ("conv1d_fprop_fp32_contract", "conv_fprop"),
        ("conv2d_dgrad_fp16_contract", "conv_dgrad"),
        ("conv3d_wgrad_bfloat16_contract", "conv_wgrad"),
    ):
        require(
            adapter.benchmark_case_operator(case_name, manifest_operators)
            == expected_owner,
            f"THead adapter did not resolve {case_name} to {expected_owner}",
        )
    require(
        adapter.benchmark_case_operator(
            "conv4d_fprop_fp32_contract", manifest_operators
        )
        is None,
        "THead adapter accepted an unsupported convolution dimensionality",
    )
    required_preflight = adapter.preflight_tests(("functional", "benchmark"))
    extended_preflight = {
        f"integration.thead.{phase}_{operation}"
        for operation in (
            "leaky_relu",
            "log",
            "cos",
            "rsqrt",
            "sin",
            "tan",
            "softplus",
            "swish",
            "gelu_approx_tanh",
            "div",
            "pow",
            "sigmoid_backward",
            "reciprocal",
        )
        for phase in ("jit", "graph", "autotune")
    }
    require(
        {
            "integration.thead.run_tests_adapter_contract",
            "integration.thead.installed_consumer",
            "integration.thead.triton_compilation",
            "integration.thead.jit_sub",
            "integration.thead.graph_sub",
            "integration.thead.autotune_sub",
            "integration.thead.jit_min",
            "integration.thead.graph_min",
            "integration.thead.autotune_min",
            "integration.thead.jit_max",
            "integration.thead.graph_max",
            "integration.thead.autotune_max",
            "integration.thead.jit_scale",
            "integration.thead.graph_scale",
            "integration.thead.autotune_scale",
            "integration.thead.jit_relu",
            "integration.thead.graph_relu",
            "integration.thead.autotune_relu",
            "integration.thead.jit_sigmoid",
            "integration.thead.graph_sigmoid",
            "integration.thead.autotune_sigmoid",
            "integration.thead.jit_tanh",
            "integration.thead.graph_tanh",
            "integration.thead.autotune_tanh",
            "integration.thead.jit_elu",
            "integration.thead.graph_elu",
            "integration.thead.autotune_elu",
            "integration.thead.jit_identity",
            "integration.thead.graph_identity",
            "integration.thead.autotune_identity",
            "integration.thead.jit_gelu",
            "integration.thead.graph_gelu",
            "integration.thead.autotune_gelu",
            "integration.thead.jit_sqrt",
            "integration.thead.graph_sqrt",
            "integration.thead.autotune_sqrt",
            "integration.thead.jit_neg",
            "integration.thead.graph_neg",
            "integration.thead.autotune_neg",
            "integration.thead.jit_abs",
            "integration.thead.graph_abs",
            "integration.thead.autotune_abs",
            "integration.thead.jit_ceil",
            "integration.thead.graph_ceil",
            "integration.thead.autotune_ceil",
            "integration.thead.jit_floor",
            "integration.thead.graph_floor",
            "integration.thead.autotune_floor",
            "integration.thead.jit_exp",
            "integration.thead.graph_exp",
            "integration.thead.autotune_exp",
        }.issubset(required_preflight)
        and extended_preflight.issubset(required_preflight),
        "THead adapter/installed-consumer contracts are not required"
        " preflight",
    )
    require(
        "integration.thead.jit_artifact_alias" not in required_preflight,
        "THead preflight still requires the retired FlagTree alias contract",
    )

    environment = {
        "CUDA_VISIBLE_DEVICES": "9",
        "HIP_VISIBLE_DEVICES": "8",
        "ROCR_VISIBLE_DEVICES": "7",
        "GPU_DEVICE_ORDINAL": "6",
        "ASCEND_RT_VISIBLE_DEVICES": "5",
        "ASCEND_VISIBLE_DEVICES": "4",
        "NPU_VISIBLE_DEVICES": "3",
        "HGGC_VISIBLE_DEVICES": "2",
        "FLAGDNN_ADD_CASE": "stale",
        "FLAGDNN_BENCHMARK_CASE": "stale",
        "KEEP_ME": "yes",
    }
    adapter.configure_environment(environment, "1")
    require(
        environment.get("CUDA_VISIBLE_DEVICES") == "1"
        and environment.get("HGGC_VISIBLE_DEVICES") == "1"
        and environment.get("KEEP_ME") == "yes"
        and not any(
            variable in environment
            for variable in adapter.VISIBILITY_VARIABLES
            if variable not in {"CUDA_VISIBLE_DEVICES", "HGGC_VISIBLE_DEVICES"}
        )
        and "FLAGDNN_ADD_CASE" not in environment
        and "FLAGDNN_BENCHMARK_CASE" not in environment,
        "THead visibility-mask normalization is unsafe",
    )
    metadata = adapter.preflight_metadata(environment)
    require(
        metadata == {"visibility_masks": {
            "CUDA_VISIBLE_DEVICES": "1", "HGGC_VISIBLE_DEVICES": "1",
        }},
        "THead preflight visibility metadata is incorrect",
    )

    # Each worker starts from the coordinator's first-device environment.
    # Re-selection must replace both aliases rather than inherit GPU 0.
    coordinator = {"CUDA_VISIBLE_DEVICES": "0", "HGGC_VISIBLE_DEVICES": "0"}
    for device in ("0", "1"):
        worker = runner.device_environment("thead", device, coordinator, adapter)
        require(
            worker["CUDA_VISIBLE_DEVICES"] == device
            and worker["HGGC_VISIBLE_DEVICES"] == device,
            "THead worker retained the coordinator's device mask",
        )
    require(
        coordinator == {"CUDA_VISIBLE_DEVICES": "0", "HGGC_VISIBLE_DEVICES": "0"},
        "THead worker mutated its shared base environment",
    )

    case = "add_perf_fp32_1x1x1024_by_1x1x1024"
    output = timing(case, "flagdnn") + "\n" + timing(case, "acdnn", 4.0)
    records, errors = runner.benchmark_records(output, adapter)
    require(
        not errors and set(records[case]) == {"flagdnn", "acdnn"},
        "generic runner rejected a complete FlagDNN/acDNN pair",
    )
    require(
        not adapter.validate_thead_benchmark_pairs(
            records, "add", manifest_operators
        ),
        "THead adapter rejected a complete provider pair",
    )

    missing_records, missing_errors = runner.benchmark_records(
        timing(case, "flagdnn"), adapter
    )
    require(
        not missing_errors
        and adapter.validate_thead_benchmark_pairs(
            missing_records, "add", manifest_operators
        ),
        "THead adapter accepted a missing acDNN provider",
    )
    _, duplicate_errors = runner.benchmark_records(
        output + "\n" + timing(case, "flagdnn"), adapter
    )
    require(duplicate_errors, "generic runner accepted a duplicate provider")

    malformed_records = [
        {
            **json.loads(timing(case, "flagdnn")),
            "samples": [1.0, 0.0, 3.0],
        },
        {**json.loads(timing(case, "flagdnn")), "median": 1.0},
        {**json.loads(timing(case, "flagdnn")), "p90": 2.0},
    ]
    for malformed in malformed_records:
        _, malformed_errors = runner.benchmark_records(
            json.dumps(malformed, separators=(",", ":")), adapter
        )
        require(
            malformed_errors,
            "generic runner accepted malformed samples/median/p90",
        )

    skipped_case = "add_perf_fp16_1x1x1024_by_1x1x1024"
    valid_skip = skip_line(skipped_case)
    entire_skip = postprocess(
        adapter,
        status="skipped",
        output=(
            valid_skip
            + "\nFLAGDNN_ADD_FUNCTIONAL: SKIP cases=1 executed=0 skipped=1"
        ),
    )
    require(
        entire_skip["status"] == "skipped"
        and not entire_skip.get("skip_record_errors")
        and not entire_skip.get("case_accounting_errors"),
        "valid THead entire-suite SKIP was rejected",
    )

    missing_skip = postprocess(
        adapter,
        status="skipped",
        output="FLAGDNN_ADD_FUNCTIONAL: SKIP cases=1 executed=0 skipped=1",
    )
    require(
        missing_skip["status"] == "failed",
        "missing acDNN skip record was accepted",
    )
    duplicate_skip = postprocess(
        adapter,
        status="skipped",
        output=(
            valid_skip
            + "\n"
            + valid_skip
            + "\nFLAGDNN_ADD_FUNCTIONAL: SKIP cases=2 executed=0 skipped=2"
        ),
    )
    require(
        duplicate_skip["status"] == "failed",
        "duplicate acDNN skip record was accepted",
    )
    wrong_owner = postprocess(
        adapter,
        status="skipped",
        output=(
            skip_line("mul_fp16_16", operator="add")
            + "\nFLAGDNN_ADD_FUNCTIONAL: SKIP cases=1 executed=0 skipped=1"
        ),
    )
    require(
        wrong_owner["status"] == "failed",
        "wrong-owner acDNN skip record was accepted",
    )

    mixed = postprocess(
        adapter,
        status="passed",
        output=(
            valid_skip
            + "\nFLAGDNN_ADD_FUNCTIONAL: PASS cases=2 executed=1 skipped=1"
        ),
    )
    require(
        mixed["status"] == "passed"
        and mixed["case_accounting"]["executed"] == 1,
        "valid mixed executed/skipped accounting was rejected",
    )
    missing_marker = postprocess(
        adapter, status="passed", output="ordinary successful output"
    )
    require(
        missing_marker["status"] == "failed",
        "missing accounting marker was accepted",
    )
    duplicate_marker = postprocess(
        adapter,
        status="passed",
        output=(
            "FLAGDNN_ADD_FUNCTIONAL: PASS cases=1 executed=1 skipped=0\n"
            "FLAGDNN_ADD_FUNCTIONAL: PASS cases=1 executed=1 skipped=0"
        ),
    )
    require(
        duplicate_marker["status"] == "failed",
        "duplicate accounting marker was accepted",
    )

    # Multiple CTest dtype categories form one public operator result. A
    # skipped
    # category must retain its accounting while other categories still
    # time pairs.
    identity_case = "identity_perf_fp32_1x1x1024"
    identity_skip = "identity_int32_1x1x16"
    aggregate_output = (
        skip_line(identity_skip, operator="identity")
        + "\nFLAGDNN_IDENTITY_BENCHMARK: PASS cases=1 comparable_executed=1"
        " reference_skipped=0"
        + "\nFLAGDNN_IDENTITY_COPY_BENCHMARK: SKIP cases=1"
        " comparable_executed=0 reference_skipped=1"
    )
    aggregate_result = {"status": "skipped"}
    adapter.postprocess_result(
        result=aggregate_result,
        ctest_reported_status="skipped",
        output=aggregate_output,
        operator="identity",
        suite="benchmark",
        records={identity_case: pair(identity_case)},
        manifest_operators=["identity"],
    )
    require(
        aggregate_result["status"] == "passed"
        and aggregate_result["case_accounting"]["cases"] == 2
        and aggregate_result["case_accounting"]["reference_skipped"] == 1,
        "mixed CTest dtype categories did not preserve timing and skip"
        " accounting",
    )
    # Exercise the shared runner too: it adds a generic record error before
    # the backend hook sees mixed passed/skipped CTest categories.
    mixed_ctest_output = (
        aggregate_output
        + "\n"
        + timing(identity_case, "flagdnn")
        + "\n"
        + timing(identity_case, "acdnn")
        + "\n2/2 Test #2: benchmark.thead.identity.copy ...***Skipped\n"
        + "100% tests passed, 0 tests failed out of 2\n"
    )

    def run_mixed_ctest(output: str, exit_code: int = 0) -> dict[str, Any]:
        previous_process = runner.run_process_group
        runner.run_process_group = lambda *_args: (
            output,
            "",
            exit_code,
            False,
        )
        try:
            with contextlib.redirect_stdout(
                io.StringIO()
            ), contextlib.redirect_stderr(io.StringIO()):
                return runner.run_one(
                    Path("."),
                    "identity",
                    "benchmark",
                    "thead",
                    {},
                    30,
                    False,
                    ["identity"],
                    adapter=adapter,
                )
        finally:
            runner.run_process_group = previous_process

    measured_mixed = run_mixed_ctest(mixed_ctest_output)
    require(
        measured_mixed["status"] == "passed"
        and not measured_mixed.get("record_errors")
        and measured_mixed["case_accounting"]["comparable_executed"] == 1
        and measured_mixed["case_accounting"]["reference_skipped"] == 1,
        "shared runner rejected valid measured mixed CTest categories",
    )
    for invalid_output, exit_code in (
        (mixed_ctest_output + timing(identity_case, "flagdnn"), 0),
        (mixed_ctest_output.replace(timing(identity_case, "acdnn"), ""), 0),
        (
            mixed_ctest_output.replace(
                "comparable_executed=1", "comparable_executed=2"
            ),
            0,
        ),
        (mixed_ctest_output.replace("op=identity", "op=add"), 0),
        (mixed_ctest_output, 1),
    ):
        require(
            run_mixed_ctest(invalid_output, exit_code)["status"] == "failed",
            "mixed category normalization hid a real benchmark failure",
        )

    for invalid in (
        aggregate_output + "\nFLAGDNN_IDENTITY_COPY_BENCHMARK: SKIP cases=1"
        " comparable_executed=0 reference_skipped=1",
        aggregate_output.replace(
            "IDENTITY_COPY_BENCHMARK", "IDENTITY_TF32_BENCHMARK"
        ),
    ):
        _, errors = adapter.validate_thead_case_accounting(
            invalid,
            "identity",
            "benchmark",
            "skipped",
            {identity_case: pair(identity_case)},
            adapter.acdnn_skip_records(invalid),
        )
        require(errors, "invalid dtype category accounting was accepted")

    catalog = adapter.load_thead_comparable_case_catalog(
        manifests["benchmark"]
    )
    required_cases = catalog["operators"]["add"]
    require(
        required_cases, "THead comparable catalog has no required Add case"
    )
    required_mul_cases = catalog["operators"]["mul"]
    required_sub_cases = catalog["operators"]["sub"]
    required_min_cases = catalog["operators"]["min"]
    required_max_cases = catalog["operators"]["max"]
    required_scale_cases = catalog["operators"]["scale"]
    required_relu_cases = catalog["operators"]["relu"]
    required_leaky_relu_cases = catalog["operators"]["leaky_relu"]
    required_sigmoid_cases = catalog["operators"]["sigmoid"]
    required_tanh_cases = catalog["operators"]["tanh"]
    required_elu_cases = catalog["operators"]["elu"]
    required_identity_cases = catalog["operators"]["identity"]
    required_gelu_cases = catalog["operators"]["gelu"]
    required_sqrt_cases = catalog["operators"]["sqrt"]
    required_neg_cases = catalog["operators"]["neg"]
    required_abs_cases = catalog["operators"]["abs"]
    required_ceil_cases = catalog["operators"]["ceil"]
    required_floor_cases = catalog["operators"]["floor"]
    required_exp_cases = catalog["operators"]["exp"]
    extended_required = {
        operation: catalog["operators"][operation]
        for operation in (
            "log",
            "cos",
            "rsqrt",
            "sin",
            "tan",
            "softplus",
            "swish",
            "gelu_approx_tanh",
        )
    }
    binary_descriptor_required = {
        operation: catalog["operators"][operation]
        for operation in ("div", "pow", "mod", "sigmoid_backward")
    }
    required_reciprocal_cases = catalog["operators"]["reciprocal"]
    required_add_square_cases = catalog["operators"]["add_square"]
    comparison_required = {
        operation: catalog["operators"][operation]
        for operation in (
            "cmp_eq",
            "cmp_neq",
            "cmp_gt",
            "cmp_ge",
            "cmp_lt",
            "cmp_le",
        )
    }
    layout_required = {
        operation: catalog["operators"][operation]
        for operation in ("reshape", "transpose", "slice")
    }
    reduction_required = catalog["operators"]["reduction"]
    batchnorm_required = catalog["operators"]["batchnorm"]
    batchnorm_inference_required = catalog["operators"]["batchnorm_inference"]
    layernorm_required = catalog["operators"]["layernorm"]
    rmsnorm_required = catalog["operators"]["rmsnorm"]
    matmul_required = catalog["operators"]["matmul"]
    convolution_required = {
        operation: catalog["operators"][operation]
        for operation in ("conv_fprop", "conv_dgrad", "conv_wgrad")
    }
    conv_bias_relu_required = catalog["operators"]["conv_bias_relu"]
    batchnorm_shapes = {
        "8x32x32x32",
        "16x64x16x16",
        "4x128x16x16",
        "8x64x56x56",
        "16x128x28x28",
        "16x256x14x14",
        "8x512x7x7",
        "32x1024x1x1",
    }
    four_dense_shapes = {
        "1x1x1000",
        "1x1x1024",
        "3x257x513",
        "8x16x32",
    }
    vector_aligned_dense_shapes = {
        "1x1x1024",
        "8x16x32",
    }
    require(
        required_mul_cases
        and set(required_sub_cases)
        >= {
            "sub_perf_fp32_1x1x1000_by_1x1x1000",
            "sub_perf_fp32_1x1x1024_by_1x1x1024",
            "sub_perf_fp32_3x257x513_by_3x257x513",
            "sub_perf_fp32_8x16x32_by_8x16x32",
        }
        and set(required_min_cases)
        >= {
            "min_perf_fp32_1x1x1000_by_1x1x1000",
            "min_perf_fp32_1x1x1024_by_1x1x1024",
            "min_perf_fp32_3x257x513_by_3x257x513",
            "min_perf_fp32_8x16x32_by_8x16x32",
        }
        and set(required_max_cases)
        >= {
            "max_perf_fp32_1x1x1000_by_1x1x1000",
            "max_perf_fp32_1x1x1024_by_1x1x1024",
            "max_perf_fp32_3x257x513_by_3x257x513",
            "max_perf_fp32_8x16x32_by_8x16x32",
        }
        and set(required_scale_cases)
        >= {
            "scale_perf_fp32_1x1x1000_by_1x1x1000",
            "scale_perf_fp32_1x1x1024_by_1x1x1024",
            "scale_perf_fp32_3x257x513_by_3x257x513",
            "scale_perf_fp32_8x16x32_by_8x16x32",
        }
        and set(required_relu_cases)
        >= {
            "relu_perf_fp32_1x1x1000",
            "relu_perf_fp32_1x1x1024",
            "relu_perf_fp32_3x257x513",
            "relu_perf_fp32_8x16x32",
        }
        and required_leaky_relu_cases
        and all(
            case.startswith("leaky_relu_perf_")
            for case in required_leaky_relu_cases
        )
        and set(required_sigmoid_cases)
        >= {
            "sigmoid_perf_fp32_1x1x1000",
            "sigmoid_perf_fp32_1x1x1024",
            "sigmoid_perf_fp32_3x257x513",
            "sigmoid_perf_fp32_8x16x32",
        }
        and set(required_tanh_cases)
        >= {
            "tanh_perf_fp32_1x1x1000",
            "tanh_perf_fp32_1x1x1024",
            "tanh_perf_fp32_3x257x513",
            "tanh_perf_fp32_8x16x32",
        }
        and set(required_elu_cases)
        >= {
            "elu_perf_fp32_1x1x1000",
            "elu_perf_fp32_1x1x1024",
            "elu_perf_fp32_3x257x513",
            "elu_perf_fp32_8x16x32",
        }
        and set(required_identity_cases)
        >= {
            "identity_perf_fp32_1x1x1",
            "identity_perf_fp32_2x3x4",
            "identity_perf_fp32_8x16x32",
            "identity_perf_fp32_64x64x64",
            "identity_perf_fp32_16x64x128",
            "identity_perf_fp32_16x256x256",
            "identity_perf_fp32_32x128x256",
            "identity_perf_fp32_4x1024x1024",
            "identity_perf_fp32_16x512x1024",
            "identity_perf_fp32_2x2048x2048",
            "identity_perf_fp32_8x1024x2048",
        }
        and set(required_gelu_cases)
        >= {
            "gelu_perf_fp32_1x1x1000",
            "gelu_perf_fp32_1x1x1024",
            "gelu_perf_fp32_3x257x513",
            "gelu_perf_fp32_8x16x32",
        }
        and set(required_sqrt_cases)
        >= {
            "sqrt_perf_fp32_1x1x1000",
            "sqrt_perf_fp32_1x1x1024",
            "sqrt_perf_fp32_3x257x513",
            "sqrt_perf_fp32_8x16x32",
        }
        and set(required_neg_cases)
        >= {
            "neg_perf_fp32_1x1x1000",
            "neg_perf_fp32_1x1x1024",
            "neg_perf_fp32_3x257x513",
            "neg_perf_fp32_8x16x32",
        }
        and set(required_abs_cases)
        >= {
            "abs_perf_fp32_1x1x1000",
            "abs_perf_fp32_1x1x1024",
            "abs_perf_fp32_3x257x513",
            "abs_perf_fp32_8x16x32",
        }
        and set(required_ceil_cases) >= {"ceil_perf_fp32_3x257x513"}
        and all(
            case.startswith("floor_perf_") for case in required_floor_cases
        )
        and set(required_exp_cases)
        >= {
            "exp_perf_fp32_1x1x1000",
            "exp_perf_fp32_1x1x1024",
            "exp_perf_fp32_3x257x513",
            "exp_perf_fp32_8x16x32",
        }
        and all(
            set(extended_required[operation])
            >= {
                f"{operation}_perf_fp32_{shape}" for shape in four_dense_shapes
            }
            for operation in (
                "log",
                "rsqrt",
                "softplus",
                "swish",
                "gelu_approx_tanh",
            )
        )
        and all(
            set(extended_required[operation])
            >= {f"{operation}_perf_fp32_3x257x513"}
            for operation in ("cos", "sin", "tan")
        )
        and all(
            set(binary_descriptor_required[operation])
            >= {
                f"{operation}_perf_fp32_{shape}_by_{shape}"
                for shape in ("1x1x1024", "8x16x32")
            }
            for operation in ("div", "pow", "sigmoid_backward")
        )
        and set(binary_descriptor_required["mod"])
        >= {
            "mod_perf_fp32_1x1x1024_by_1x1x1024",
            "mod_perf_fp32_1x1x1000_by_1x1x1000",
        }
        and set(required_reciprocal_cases) >= {"reciprocal_perf_fp32_1x1x1024"}
        and set(required_add_square_cases)
        >= {
            "add_square_perf_fp32_1x1x1000",
            "add_square_perf_fp32_1x1x1024",
            "add_square_perf_fp32_3x257x513",
            "add_square_perf_fp32_8x16x32",
        }
        and all(
            set(comparison_required[operation])
            >= {
                f"{operation}_perf_fp32_{shape}_by_{shape}"
                for shape in vector_aligned_dense_shapes
            }
            for operation in comparison_required
        )
        and set(layout_required["reshape"])
        >= {
            "reshape_perf_fp32_8x16x32_to_128x32",
            "reshape_perf_fp32_16x64x128_to_1024x128",
            "reshape_perf_fp32_16x256x256_to_4096x256",
            "reshape_perf_fp32_32x128x256_to_4096x256",
            "reshape_perf_fp32_4x1024x1024_to_4096x1024",
        }
        and set(layout_required["transpose"])
        >= {
            "transpose_perf_fp32_8x16x32",
            "transpose_perf_fp32_16x64x128",
            "transpose_perf_fp32_32x128x256",
        }
        and set(layout_required["slice"]) >= {"slice_perf_fp32_case0_8x16x32"}
        and set(reduction_required)
        >= {
            "reduction_sum_perf_fp32_axis1_keepdim_8x8x32x32",
            "reduction_avg_perf_fp32_axis1_keepdim_8x8x32x32",
            "reduction_mul_perf_fp32_axis1_keepdim_8x4x16x16",
        }
        and set(batchnorm_required)
        >= {f"batchnorm_perf_fp32_{shape}" for shape in batchnorm_shapes}
        and set(batchnorm_inference_required)
        >= {
            f"batchnorm_inference_perf_fp32_{shape}"
            for shape in batchnorm_shapes
        }
        and set(layernorm_required) >= {"layernorm_perf_fp32_1x128x768"}
        and set(rmsnorm_required) >= {"rmsnorm_perf_fp32_1x128x768"}
        and set(matmul_required)
        >= {
            "matmul_perf_fp32_4x16x32_by_4x32x24",
            "matmul_perf_fp32_8x32x64_by_8x64x32",
            "matmul_perf_fp32_32x512x512_by_32x512x512",
            "matmul_perf_fp32_16x1024x1024_by_16x1024x1024",
            "matmul_perf_fp32_8x2048x2048_by_8x2048x2048",
            "matmul_perf_fp32_4x4096x4096_by_4x4096x4096",
            "matmul_perf_fp32_16x2048x512_by_16x512x2048",
            "matmul_perf_fp32_32x1024x4096_by_32x4096x1024",
        }
        and all(
            set(convolution_required[operation])
            >= {
                f"conv2d_{operation.removeprefix('conv_')}_perf_fp32_"
                "standard_1x1_8x64x28x28_by_128x64x1x1"
            }
            for operation in ("conv_fprop", "conv_dgrad", "conv_wgrad")
        )
        and set(conv_bias_relu_required)
        >= {"conv_bias_relu_perf_fp32_x2x8x16x16_w16x8x3x3_s1x1_p1x1_d1x1"}
        and catalog["schema_version"] == 2
        and catalog["declared_operator_count"] == len(selected["benchmark"]),
        "THead comparable catalog lacks required pointwise/activation"
        " coverage",
    )

    with tempfile.TemporaryDirectory(
        prefix="flagdnn-thead-comparable-catalog-contract-"
    ) as catalog_temporary_text:
        comparable_catalog_path = (
            Path(catalog_temporary_text) / "comparable.json"
        )
        comparable_case = "add_perf_fp32_contract"
        comparable_catalog_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "platform": "thead",
                    "reference_provider": "acdnn",
                    "operators": {
                        "add": {
                            comparable_case: {
                                "status": "comparable",
                                "reason_code": "",
                                "detail": "Synthetic exact-pair qualification",
                            }
                        },
                        "mul": {
                            "mul_perf_fp32_contract": {
                                "status": "probe_required",
                                "reason_code": (
                                    "real_device_qualification_pending"
                                ),
                                "detail": "Synthetic Mul pair qualification",
                            }
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        comparable_catalog = adapter.load_thead_comparable_case_catalog(
            manifests["benchmark"], comparable_catalog_path
        )
        require(
            comparable_catalog["operators"]
            == {
                "add": [comparable_case],
                "mul": ["mul_perf_fp32_contract"],
            }
            and comparable_catalog["case_status_counts"]["comparable"] == 1
            and comparable_catalog["case_status_counts"]["probe_required"]
            == 1,
            "THead adapter rejected the canonical comparable status",
        )

    incomplete_results = {
        "add": {
            "benchmark": {
                "status": "passed",
                "records": {
                    required_case: pair(required_case)
                    for required_case in required_cases[:-1]
                },
            }
        }
    }
    coverage = adapter.thead_comparable_coverage(
        incomplete_results, ["add"], catalog
    )
    require(
        not coverage["verified"] and coverage["missing_case_count"] == 1,
        "missing required comparable pair passed coverage",
    )

    complete_results = {
        "add": {
            "benchmark": {
                "status": "passed",
                "records": {
                    required_case: pair(required_case, 2.0, 4.0)
                    for required_case in required_cases
                },
                "case_accounting": {
                    "status": "PASS",
                    "cases": len(required_cases),
                    "comparable_executed": len(required_cases),
                    "reference_skipped": 0,
                },
            }
        }
    }
    complete_coverage = adapter.thead_comparable_coverage(
        complete_results, ["add"], catalog
    )
    passed_speedup = adapter.benchmark_speedup_summary(
        complete_results, 1.5, complete_coverage
    )
    failed_speedup = adapter.benchmark_speedup_summary(
        complete_results, 2.5, complete_coverage
    )
    require(
        complete_coverage["verified"]
        and passed_speedup["gate_passed"]
        and not failed_speedup["gate_passed"],
        "THead acDNN/FlagDNN speedup threshold is incorrect",
    )

    with tempfile.TemporaryDirectory(
        prefix="flagdnn-thead-run-tests-contract-"
    ) as temporary_text:
        fake_build = Path(temporary_text) / "build"
        fake_build.mkdir()
        (fake_build / "CTestTestfile.cmake").write_text(
            "# synthetic contract\n", encoding="utf-8"
        )
        original_preflight = runner.run_preflight
        original_run_one = runner.run_one
        operator_ran = False

        def failed_preflight(**_arguments: Any) -> dict[str, Any]:
            return {
                "status": "failed",
                "duration_seconds": 0.0,
                "exit_code": 1,
                "command": [],
                "required_tests": [],
                "missing_tests": ["synthetic.contract"],
                "errors": ["synthetic preflight failure"],
                "visibility_masks": {},
            }

        def forbidden_operator(**_arguments: Any) -> dict[str, Any]:
            nonlocal operator_ran
            operator_ran = True
            raise RuntimeError("operator ran after failed THead preflight")

        runner.run_preflight = failed_preflight
        runner.run_one = forbidden_operator
        try:
            exit_code, _, _ = invoke_main(
                runner,
                [
                    "--platform",
                    "thead",
                    "--build-dir",
                    str(fake_build),
                    "--suites",
                    "functional",
                    "--ops",
                    "add",
                    "--preflight",
                ],
            )
        finally:
            runner.run_preflight = original_preflight
            runner.run_one = original_run_one
        require(
            exit_code == 1 and not operator_ran,
            "operator suite ran after a failed THead preflight",
        )

    print("THead run_tests adapter contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
