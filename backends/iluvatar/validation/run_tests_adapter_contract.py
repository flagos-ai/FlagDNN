#!/usr/bin/env python3

"""Configure-time-safe contracts for the Iluvatar run-tests adapter."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location(
        "flagdnn_iluvatar_run_tests_contract_subject", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load runner: {path}")
    module = importlib.util.module_from_spec(spec)
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
    return module


def timing(case: str, provider: str) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "kind": "steady_state",
            "provider": provider,
            "case": case,
            "unit": "us",
            "median": 2.0,
            "p90": 3.0,
            "samples": [1.0, 2.0, 3.0],
        },
        separators=(",", ":"),
    )


def invoke_main(runner, arguments: list[str]) -> tuple[int, str, str]:
    original_argv = sys.argv
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        sys.argv = [str(runner.__file__), *arguments]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
            stderr
        ):
            exit_code = runner.main()
    finally:
        sys.argv = original_argv
    return exit_code, stdout.getvalue(), stderr.getvalue()


def main() -> int:
    if len(sys.argv) != 2:
        raise RuntimeError("usage: run_tests_adapter_contract.py SOURCE_ROOT")
    source_root = Path(sys.argv[1]).resolve()
    runner = load_runner(source_root / "tools" / "run_tests.py")
    iluvatar = runner.load_platform_adapter("iluvatar")
    require(iluvatar is not None, "Iluvatar run-tests adapter is missing")
    require(
        iluvatar.PREFLIGHT_BY_DEFAULT
        and iluvatar.SUPPORTS_MIN_SPEEDUP
        and not iluvatar.FILTER_REGISTERED_TESTS
        and iluvatar.DEFAULT_TIMEOUT == 1800,
        "Iluvatar adapter defaults changed unexpectedly",
    )

    runner_source = Path(runner.__file__).read_text(encoding="utf-8").lower()
    for platform_detail in (
        "iluvatar",
        "corex_cudnn",
        "corex-cudnn",
    ):
        require(
            platform_detail not in runner_source,
            f"generic runner contains Iluvatar policy: {platform_detail}",
        )

    manifests = runner.operator_manifests()
    manifest_operators = list(
        dict.fromkeys(
            operator
            for manifest in manifests.values()
            for operator in manifest
        )
    )
    expected_preflight = {
        "integration.iluvatar.cmake_configuration_contract",
        "integration.iluvatar.dependency_boundary",
        "integration.iluvatar.reference_dependency_boundary",
        "integration.iluvatar.compiler_contract",
        "integration.iluvatar.run_tests_adapter_contract",
        "integration.iluvatar.artifact_contract",
        "integration.iluvatar.validation_contract",
        "integration.iluvatar.corex_cudnn_environment",
        "integration.iluvatar.jit",
        "integration.iluvatar.autotune",
        "integration.iluvatar.runtime",
        "integration.iluvatar.graph",
        "integration.iluvatar.installed_consumer",
        "integration.iluvatar.catalog_closure",
    }
    actual_preflight = {
        test
        for test in runner.required_preflight_tests("iluvatar")
        if test.startswith("integration.iluvatar.")
    }
    require(
        actual_preflight == expected_preflight,
        "Iluvatar preflight contract set is incomplete",
    )
    require(
        "benchmark.catalog_contract"
        not in runner.required_preflight_tests("iluvatar", ["functional"]),
        "Iluvatar functional-only preflight requires benchmark targets",
    )

    original_run_process_group = runner.run_process_group
    required_functional_preflight = runner.required_preflight_tests(
        "iluvatar", ["functional"], iluvatar
    )
    missing_preflight_test = sorted(required_functional_preflight)[0]

    def missing_iluvatar_preflight(*_arguments, **_keywords):
        catalog = {
            "tests": [
                {"name": name}
                for name in sorted(
                    required_functional_preflight - {missing_preflight_test}
                )
            ]
        }
        return json.dumps(catalog), "", 0, False

    runner.run_process_group = missing_iluvatar_preflight
    try:
        missing_preflight_result = runner.run_preflight(
            build_dir=Path("/tmp/flagdnn-build"),
            platform="iluvatar",
            environment={},
            timeout=60,
            verbose=False,
            suites=["functional"],
            adapter=iluvatar,
        )
    finally:
        runner.run_process_group = original_run_process_group
    require(
        missing_preflight_result["status"] == "failed"
        and missing_preflight_result["missing_tests"]
        == [missing_preflight_test],
        "missing Iluvatar preflight test did not fail closed",
    )

    def skipped_iluvatar_preflight(command, *_arguments, **_keywords):
        if "--show-only=json-v1" in command:
            catalog = {
                "tests": [
                    {"name": name}
                    for name in sorted(required_functional_preflight)
                ]
            }
            return json.dumps(catalog), "", 0, False
        return (
            "1/1 Test #1: integration.iluvatar.jit ***Skipped",
            "",
            0,
            False,
        )

    runner.run_process_group = skipped_iluvatar_preflight
    try:
        skipped_preflight_result = runner.run_preflight(
            build_dir=Path("/tmp/flagdnn-build"),
            platform="iluvatar",
            environment={},
            timeout=60,
            verbose=False,
            suites=["functional"],
            adapter=iluvatar,
        )
    finally:
        runner.run_process_group = original_run_process_group
    require(
        skipped_preflight_result["status"] == "skipped",
        "skipped Iluvatar preflight test did not fail closed",
    )

    base_environment = {
        "CUDA_VISIBLE_DEVICES": "7",
        "UNCHANGED": "yes",
    }
    selected_environment = runner.device_environment(
        "iluvatar", "1", base_environment, adapter=iluvatar
    )
    require(
        selected_environment["CUDA_VISIBLE_DEVICES"] == "1"
        and selected_environment["UNCHANGED"] == "yes",
        "Iluvatar device selection is not isolated",
    )
    unselected_environment = runner.device_environment(
        "iluvatar", None, base_environment, adapter=iluvatar
    )
    require(
        unselected_environment == base_environment,
        "Iluvatar omitted device selection changed the environment",
    )

    def corex_skip_line(**overrides: str) -> str:
        fields = {
            "op": "add",
            "case": "add_fp32_2x3",
            "reason": "NO_EXACT_CUDNN_PRIMITIVE",
            "cudnn_header": "7605",
            "cudnn_runtime": "7605",
            "corex": "4.4.0",
            "target": "corex_71",
            "dtype": "fp32",
            "layout": "contiguous",
            "shape": "2x3",
        }
        fields.update(overrides)
        order = (
            "op",
            "case",
            "reason",
            "cudnn_header",
            "cudnn_runtime",
            "corex",
            "target",
            "dtype",
            "layout",
            "shape",
        )
        return "[SKIP][corex-cudnn] " + " ".join(
            f"{field}={fields[field]}" for field in order
        )

    valid_skips = iluvatar.corex_cudnn_skip_records(corex_skip_line())
    require(
        len(valid_skips) == 1
        and not iluvatar.validate_iluvatar_skip_records(
            valid_skips, "add", manifest_operators
        ),
        "valid CoreX cuDNN SKIP record was rejected",
    )
    malformed_skips = iluvatar.corex_cudnn_skip_records(
        corex_skip_line().replace(
            "case=add_fp32_2x3 reason=NO_EXACT_CUDNN_PRIMITIVE",
            "reason=NO_EXACT_CUDNN_PRIMITIVE case=add_fp32_2x3",
        )
    )
    require(
        len(malformed_skips) == 1
        and iluvatar.validate_iluvatar_skip_records(
            malformed_skips, "add", manifest_operators
        ),
        "malformed CoreX cuDNN SKIP field order was accepted",
    )
    for label, invalid_records in {
        "missing": [],
        "duplicate": [valid_skips[0], valid_skips[0]],
        "operator": [{**valid_skips[0], "op": "mul"}],
        "case-owner": [{**valid_skips[0], "case": "mul_fp32_2x3"}],
        "reason": [{**valid_skips[0], "reason": "UNQUALIFIED"}],
        "header": [{**valid_skips[0], "cudnn_header": "unknown"}],
        "runtime": [{**valid_skips[0], "cudnn_runtime": "0"}],
        "corex": [{**valid_skips[0], "corex": "corex-4.4"}],
        "target": [{**valid_skips[0], "target": "cuda_80"}],
        "dtype": [{**valid_skips[0], "dtype": "int8"}],
        "layout": [{**valid_skips[0], "layout": "nhwc"}],
    }.items():
        require(
            iluvatar.validate_iluvatar_skip_records(
                invalid_records, "add", manifest_operators
            ),
            f"invalid CoreX cuDNN SKIP record was accepted: {label}",
        )

    def functional_accounting(
        status: str,
        cases: int,
        production: int,
        reference: int,
        skipped: int,
    ) -> str:
        return (
            "FLAGDNN_ADD_FUNCTIONAL: "
            f"{status} cases={cases} production_executed={production} "
            f"reference_executed={reference} reference_skipped={skipped}"
        )

    for payload, ctest_state, skips in (
        (functional_accounting("PASS", 2, 2, 2, 0), "passed", []),
        (
            functional_accounting("PASS", 2, 2, 1, 1),
            "passed",
            valid_skips,
        ),
        (
            functional_accounting("SKIP", 1, 1, 0, 1),
            "skipped",
            valid_skips,
        ),
    ):
        require(
            not iluvatar.validate_iluvatar_case_accounting(
                payload,
                "add",
                "functional",
                ctest_state,
                {},
                skips,
            ),
            "valid Iluvatar functional accounting was rejected",
        )
    for payload in (
        functional_accounting("PASS", 2, 1, 2, 0),
        functional_accounting("PASS", 2, 2, 1, 0),
        functional_accounting("SKIP", 2, 1, 0, 1),
        functional_accounting("PASS", 0, 0, 0, 0),
        functional_accounting("PASS", 1, 1, 1, 0)
        + "\n"
        + functional_accounting("PASS", 1, 1, 1, 0),
    ):
        require(
            iluvatar.validate_iluvatar_case_accounting(
                payload,
                "add",
                "functional",
                "passed",
                {},
                [],
            ),
            "invalid Iluvatar functional accounting was accepted",
        )

    benchmark_case = "add_perf_fp32_2x3_by_2x3"
    paired_records, pair_parse_errors = runner.benchmark_records(
        timing(benchmark_case, "flagdnn")
        + "\n"
        + timing(benchmark_case, "corex_cudnn"),
        iluvatar,
    )
    require(
        not pair_parse_errors
        and not iluvatar.validate_iluvatar_benchmark_pairs(
            paired_records, "add", manifest_operators
        ),
        "valid FlagDNN/CoreX cuDNN benchmark pair was rejected",
    )
    _, duplicate_provider_errors = runner.benchmark_records(
        timing(benchmark_case, "flagdnn")
        + "\n"
        + timing(benchmark_case, "flagdnn"),
        iluvatar,
    )
    require(
        duplicate_provider_errors,
        "duplicate Iluvatar benchmark provider record was accepted",
    )

    benchmark_marker = (
        "FLAGDNN_ADD_BENCHMARK: PASS cases=1 "
        "comparable_executed=1 reference_skipped=0"
    )
    require(
        not iluvatar.validate_iluvatar_case_accounting(
            benchmark_marker,
            "add",
            "benchmark",
            "passed",
            paired_records,
            [],
        ),
        "valid Iluvatar benchmark accounting was rejected",
    )
    for label, invalid_pair in (
        (
            "one-sided",
            {
                benchmark_case: {
                    "flagdnn": paired_records[benchmark_case]["flagdnn"]
                }
            },
        ),
        (
            "reverse-order",
            {
                benchmark_case: {
                    "corex_cudnn": paired_records[benchmark_case][
                        "corex_cudnn"
                    ],
                    "flagdnn": paired_records[benchmark_case]["flagdnn"],
                }
            },
        ),
        (
            "unequal-samples",
            {
                benchmark_case: {
                    "flagdnn": paired_records[benchmark_case]["flagdnn"],
                    "corex_cudnn": {
                        **paired_records[benchmark_case]["corex_cudnn"],
                        "samples": [1.0, 2.0],
                    },
                }
            },
        ),
    ):
        require(
            iluvatar.validate_iluvatar_benchmark_pairs(
                invalid_pair, "add", manifest_operators
            ),
            f"invalid Iluvatar benchmark pair was accepted: {label}",
        )

    benchmark_skip = iluvatar.corex_cudnn_skip_records(
        corex_skip_line(case=benchmark_case)
    )
    all_skip_marker = (
        "FLAGDNN_ADD_BENCHMARK: SKIP cases=1 "
        "comparable_executed=0 reference_skipped=1"
    )
    require(
        iluvatar.validate_iluvatar_case_accounting(
            all_skip_marker,
            "add",
            "benchmark",
            "skipped",
            paired_records,
            benchmark_skip,
        )
        and not iluvatar.validate_iluvatar_case_accounting(
            all_skip_marker,
            "add",
            "benchmark",
            "skipped",
            {},
            benchmark_skip,
        ),
        "Iluvatar all-reference-SKIP accounting is inconsistent",
    )

    speedup_pair = json.loads(json.dumps(paired_records[benchmark_case]))
    speedup_pair["flagdnn"]["median"] = 2.0
    speedup_pair["corex_cudnn"]["median"] = 3.0
    speedup_benchmark = {
        "status": "passed",
        "records": {benchmark_case: speedup_pair},
        "case_accounting": {
            "status": "PASS",
            "cases": 1,
            "comparable_executed": 1,
            "reference_skipped": 0,
        },
    }
    speedup_results = {"add": {"benchmark": speedup_benchmark}}
    require(
        iluvatar.iluvatar_speedup_summary(speedup_results, 1.4)["gate_passed"]
        and not iluvatar.iluvatar_speedup_summary(speedup_results, 1.6)[
            "gate_passed"
        ],
        "Iluvatar per-case speedup gate did not enforce its threshold",
    )
    failed_speedup_results = {
        "add": {"benchmark": {**speedup_benchmark, "status": "failed"}}
    }
    failed_speedup = iluvatar.iluvatar_speedup_summary(
        failed_speedup_results, 1.0
    )
    require(
        failed_speedup["case_count"] == 0
        and not failed_speedup["gate_passed"],
        "Iluvatar speedup accepted records from a failed suite",
    )

    skipped_speedup_results = {
        "add": {
            "benchmark": {
                "status": "skipped",
                "records": {},
                "case_accounting": {
                    "status": "SKIP",
                    "cases": 1,
                    "comparable_executed": 0,
                    "reference_skipped": 1,
                },
            }
        }
    }
    require(
        iluvatar.iluvatar_speedup_summary(skipped_speedup_results, None)[
            "gate_passed"
        ]
        and not iluvatar.iluvatar_speedup_summary(
            skipped_speedup_results, 1.0
        )["gate_passed"],
        "Iluvatar all-SKIP speedup coverage is inconsistent",
    )

    functional_results = {
        "add": {
            "functional": {
                "status": "passed",
                "case_accounting": {
                    "status": "PASS",
                    "cases": 1,
                    "production_executed": 1,
                    "reference_executed": 1,
                    "reference_skipped": 0,
                },
            }
        }
    }
    direct_environment: dict[str, str] = {}
    direct_state = iluvatar.prepare_run(
        environment=direct_environment, verbose=False
    )
    direct_cache = Path(direct_state["cache_path"])
    require(
        direct_cache.is_dir()
        and direct_environment["FLAGDNN_CACHE_PATH"] == str(direct_cache),
        "Iluvatar adapter did not create an isolated run cache",
    )
    direct_outcome = iluvatar.finalize(
        results=functional_results,
        suite_operators={"functional": ["add"], "benchmark": []},
        suites=["functional"],
        state=direct_state,
        min_speedup=None,
        preflight_passed=True,
    )
    require(
        not direct_outcome["failed"]
        and not direct_cache.exists()
        and not direct_outcome["summary"]["iluvatar_cache"]["preserved"],
        "Iluvatar adapter did not clean a non-verbose run cache",
    )

    preserved_environment: dict[str, str] = {}
    with contextlib.redirect_stdout(io.StringIO()):
        preserved_state = iluvatar.prepare_run(
            environment=preserved_environment, verbose=True
        )
    preserved_cache = Path(preserved_state["cache_path"])
    preserved_outcome = iluvatar.finalize(
        results=functional_results,
        suite_operators={"functional": ["add"], "benchmark": []},
        suites=["functional"],
        state=preserved_state,
        min_speedup=None,
        preflight_passed=True,
    )
    require(
        preserved_cache.is_dir()
        and preserved_outcome["summary"]["iluvatar_cache"]["preserved"],
        "Iluvatar adapter did not preserve a verbose run cache",
    )
    shutil.rmtree(preserved_cache)

    with tempfile.TemporaryDirectory(
        prefix="flagdnn-iluvatar-runner-contract-"
    ) as temporary_directory:
        contract_root = Path(temporary_directory)
        build_dir = contract_root / "build"
        build_dir.mkdir()
        (build_dir / "CTestTestfile.cmake").write_text(
            "# contract\n", encoding="utf-8"
        )
        summary_path = contract_root / "summary.json"
        observed_cache_paths: list[Path] = []
        original_run_one = runner.run_one

        def passed_functional(**arguments: Any) -> dict[str, Any]:
            cache_path = Path(arguments["environment"]["FLAGDNN_CACHE_PATH"])
            require(
                cache_path.is_dir(),
                "Iluvatar main path did not create its run cache",
            )
            observed_cache_paths.append(cache_path)
            return {
                "status": "passed",
                "duration_seconds": 0.0,
                "exit_code": 0,
                "command": [],
                "case_accounting": {
                    "status": "PASS",
                    "cases": 1,
                    "production_executed": 1,
                    "reference_executed": 1,
                    "reference_skipped": 0,
                },
            }

        runner.run_one = passed_functional
        try:
            exit_code, _, _ = invoke_main(
                runner,
                [
                    "--platform",
                    "iluvatar",
                    "--build-dir",
                    str(build_dir),
                    "--suites",
                    "functional",
                    "--ops",
                    "add",
                    "--no-preflight",
                    "--output",
                    str(summary_path),
                ],
            )
        finally:
            runner.run_one = original_run_one
        summary = json.loads(
            runner.diagnostic_path(summary_path).read_text(encoding="utf-8")
        )
        require(
            exit_code == 0
            and summary["overall_status"] == "passed"
            and summary["iluvatar_coverage"]["production"]
            == {"functional_cases": 1, "functional_executed": 1}
            and observed_cache_paths
            and not observed_cache_paths[-1].exists()
            and not summary["iluvatar_cache"]["preserved"],
            "Iluvatar functional main path lost coverage or cache cleanup",
        )

        def passed_benchmark(**arguments: Any) -> dict[str, Any]:
            cache_path = Path(arguments["environment"]["FLAGDNN_CACHE_PATH"])
            require(
                cache_path.is_dir(),
                "Iluvatar benchmark path did not create its run cache",
            )
            observed_cache_paths.append(cache_path)
            return {
                "status": "passed",
                "duration_seconds": 0.0,
                "exit_code": 0,
                "command": [],
                "records": {benchmark_case: speedup_pair},
                "case_accounting": {
                    "status": "PASS",
                    "cases": 1,
                    "comparable_executed": 1,
                    "reference_skipped": 0,
                },
            }

        runner.run_one = passed_benchmark
        try:
            speedup_exit, _, _ = invoke_main(
                runner,
                [
                    "--platform",
                    "iluvatar",
                    "--build-dir",
                    str(build_dir),
                    "--suites",
                    "benchmark",
                    "--ops",
                    "add",
                    "--no-preflight",
                    "--min-speedup",
                    "1.6",
                    "--output",
                    str(summary_path),
                ],
            )
        finally:
            runner.run_one = original_run_one
        speedup_summary = json.loads(
            runner.diagnostic_path(summary_path).read_text(encoding="utf-8")
        )
        require(
            speedup_exit == 1
            and speedup_summary["overall_status"] == "failed"
            and not speedup_summary["performance"]["gate_passed"]
            and not observed_cache_paths[-1].exists(),
            "Iluvatar benchmark main path ignored its gate or leaked cache",
        )

    print("PASS Iluvatar run-tests adapter contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
