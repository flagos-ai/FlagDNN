"""MThreads policy hooks for the repository batch-test runner."""

from __future__ import annotations

import re
from typing import Any


DEFAULT_TIMEOUT = 7200
PREFLIGHT_BY_DEFAULT = True
SUPPORTS_MIN_SPEEDUP = True

# Consume the public manifests without filtering by the discovered CTest catalog.
# This makes a missing MThreads target fail closed instead of silently shrinking an
# explicit --ops all run.
FILTER_REGISTERED_TESTS = False

SPEEDUP_METRIC = "mudnn_median_us/flagdnn_median_us"
VISIBILITY_VARIABLES = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "MUSA_VISIBLE_DEVICES",
)
CASE_FILTER_PATTERN = re.compile(r"^FLAGDNN_[A-Z0-9_]+_CASE$")
_CONVOLUTION_CASE_PATTERN = re.compile(
    r"^conv[123]d_(fprop|dgrad|wgrad)(?:_|$)"
)
_FUNCTIONAL_ACCOUNTING_ALIASES = {
    "conv_dgrad": "convolution",
    "conv_fprop": "convolution",
    "conv_wgrad": "convolution",
}


def configure_environment(
    environment: dict[str, str], device: str | None
) -> None:
    for variable in tuple(environment):
        if CASE_FILTER_PATTERN.fullmatch(variable) is not None:
            environment.pop(variable, None)
    environment["TRITON_JIT_BACKEND"] = "MTGPU"
    environment["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    if device is None:
        return
    for variable in VISIBILITY_VARIABLES:
        environment.pop(variable, None)
    environment["MUSA_VISIBLE_DEVICES"] = device


def preflight_tests(_suites: list[str] | tuple[str, ...]) -> set[str]:
    return {
        "integration.mthreads.cmake_configuration_contract",
        "integration.mthreads.dependency_boundary",
        "integration.mthreads.reference_dependency_boundary",
        "integration.mthreads.installed_consumer",
        "integration.mthreads.benchmark_reference_dependency_boundary",
        "integration.mthreads.run_tests_adapter_contract",
        "integration.mthreads.runtime",
        "integration.mthreads.compiler_contract",
        "integration.mthreads.artifact_contract",
        "integration.mthreads.jit_add",
    }


def preflight_metadata(environment: dict[str, str]) -> dict[str, Any]:
    return {
        "visibility_masks": {
            variable: environment[variable]
            for variable in VISIBILITY_VARIABLES
            if variable in environment
        }
    }


def _accounting_record(
    output: str, operator: str, suite: str
) -> tuple[dict[str, int | str] | None, list[str]]:
    marker_operator = (
        _FUNCTIONAL_ACCOUNTING_ALIASES.get(operator, operator)
        if suite == "functional"
        else operator
    )
    marker = f"FLAGDNN_{marker_operator.upper()}_{suite.upper()}"
    pattern = re.compile(
        rf"^{re.escape(marker)}:\s+"
        r"(PASS|SKIP)\s+cases=(\d+)\s+"
        r"executed=(\d+)\s+skipped=(\d+)\s*$"
    )
    records: list[tuple[str, str, str, str]] = []
    for raw_line in output.splitlines():
        line = re.sub(r"^\s*\d+:\s?", "", raw_line).strip()
        match = pattern.fullmatch(line)
        if match is not None:
            records.append(match.groups())
    if len(records) != 1:
        return None, [
            f"mthreads {suite} op={operator} must emit exactly one "
            f"{marker} accounting record; found {len(records)}"
        ]
    status, cases, executed, skipped = records[0]
    return {
        "status": status,
        "cases": int(cases),
        "executed": int(executed),
        "skipped": int(skipped),
    }, []


def _validate_accounting(
    output: str,
    operator: str,
    suite: str,
    ctest_reported_status: str,
    records: dict[str, dict[str, Any]],
) -> list[str]:
    accounting, errors = _accounting_record(output, operator, suite)
    if accounting is None:
        return errors
    cases = int(accounting["cases"])
    executed = int(accounting["executed"])
    skipped = int(accounting["skipped"])
    expected_status = (
        "SKIP" if ctest_reported_status == "skipped" else "PASS"
    )
    if accounting["status"] != expected_status:
        errors.append(
            f"mthreads accounting status={accounting['status']}; "
            f"CTest status requires {expected_status}"
        )
    if cases <= 0 or executed + skipped != cases:
        errors.append(
            "mthreads accounting requires positive cases and "
            "executed+skipped=cases"
        )
    if ctest_reported_status == "passed" and (
        executed != cases or skipped != 0
    ):
        errors.append(
            "a passed mthreads suite must execute every case without skips"
        )
    if suite == "benchmark" and len(records) != executed:
        errors.append(
            f"mthreads benchmark executed={executed} but emitted "
            f"{len(records)} timing case groups"
        )
    return errors


def _convolution_case_operator(case: str) -> str | None:
    match = _CONVOLUTION_CASE_PATTERN.match(case)
    if match is None:
        return None
    return f"conv_{match.group(1)}"


def _benchmark_case_operator(
    case: str, manifest_operators: list[str]
) -> str | None:
    matches = [
        candidate
        for candidate in dict.fromkeys(manifest_operators)
        if case == candidate or case.startswith(f"{candidate}_")
    ]
    convolution_operator = _convolution_case_operator(case)
    if (
        convolution_operator is not None
        and convolution_operator in manifest_operators
    ):
        matches.append(convolution_operator)
    if not matches:
        return None
    return max(matches, key=len)


def _validate_benchmark_pairs(
    records: dict[str, dict[str, Any]],
    operator: str,
    manifest_operators: list[str],
) -> list[str]:
    errors: list[str] = []
    if not records:
        return ["passed mthreads benchmark emitted no provider records"]
    for case, providers in records.items():
        owner = _benchmark_case_operator(case, manifest_operators)
        if owner != operator:
            errors.append(
                f"mthreads benchmark emitted case={case} owned_by={owner}; "
                "outside "
                f"operator={operator}"
            )
        if not isinstance(providers, dict) or set(providers) != {
            "flagdnn",
            "mudnn",
        }:
            actual = (
                sorted(providers)
                if isinstance(providers, dict)
                else type(providers).__name__
            )
            errors.append(
                f"mthreads benchmark case={case} providers={actual}; "
                "expected=['flagdnn', 'mudnn']"
            )
            continue
        if len(providers["flagdnn"]["samples"]) != len(
            providers["mudnn"]["samples"]
        ):
            errors.append(
                f"mthreads benchmark case={case} provider sample "
                "counts differ"
            )
    return errors


def postprocess_result(
    *,
    result: dict[str, Any],
    ctest_reported_status: str,
    output: str,
    operator: str,
    suite: str,
    records: dict[str, dict[str, Any]],
    manifest_operators: list[str],
) -> None:
    errors = _validate_accounting(
        output,
        operator,
        suite,
        ctest_reported_status,
        records,
    )
    if suite == "benchmark" and ctest_reported_status == "passed":
        errors.extend(
            _validate_benchmark_pairs(records, operator, manifest_operators)
        )
    if not errors:
        return
    result.setdefault("record_errors", []).extend(errors)
    if result.get("status") in {"passed", "skipped"}:
        result["status"] = "failed"


def _speedup_summary(
    results: dict[str, dict[str, Any]],
    threshold: float | None,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for operator, operator_results in results.items():
        benchmark = operator_results.get("benchmark")
        if (
            not isinstance(benchmark, dict)
            or benchmark.get("status") != "passed"
        ):
            continue
        records = benchmark.get("records", {})
        if not isinstance(records, dict):
            continue
        for case, providers in records.items():
            if not isinstance(providers, dict) or set(providers) != {
                "flagdnn",
                "mudnn",
            }:
                continue
            flagdnn_us = float(providers["flagdnn"]["median"])
            mudnn_us = float(providers["mudnn"]["median"])
            cases.append(
                {
                    "operator": operator,
                    "case": case,
                    "flagdnn_median_us": flagdnn_us,
                    "mudnn_median_us": mudnn_us,
                    "speedup": mudnn_us / flagdnn_us,
                }
            )
    cases.sort(key=lambda record: (record["speedup"], record["case"]))
    failures = (
        []
        if threshold is None
        else [record for record in cases if record["speedup"] < threshold]
    )
    gate_passed = threshold is None or (bool(cases) and not failures)
    return {
        "metric": SPEEDUP_METRIC,
        "threshold": threshold,
        "gate_passed": gate_passed,
        "case_count": len(cases),
        "failed_case_count": len(failures),
        "minimum_speedup": cases[0]["speedup"] if cases else None,
        "failures": failures,
    }


def finalize(
    *,
    results: dict[str, dict[str, Any]],
    suite_operators: dict[str, list[str]],
    suites: list[str],
    state: dict[str, Any],
    min_speedup: float | None,
    preflight_passed: bool,
) -> dict[str, Any]:
    del state
    selected_pairs = sum(
        len(suite_operators[suite]) for suite in suites
    )
    passed_pairs = sum(
        results.get(operator, {}).get(suite, {}).get("status") == "passed"
        for suite in suites
        for operator in suite_operators[suite]
    )
    speedup = _speedup_summary(results, min_speedup)
    return {
        "failed": (
            not preflight_passed
            or passed_pairs != selected_pairs
            or not speedup["gate_passed"]
        ),
        "summary": {"speedup": speedup},
        "coverage": {
            "selected_operator_suite_pairs": selected_pairs,
            "passed_operator_suite_pairs": passed_pairs,
            "complete": passed_pairs == selected_pairs,
        },
    }
