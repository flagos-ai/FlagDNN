"""MThreads policy hooks for the repository batch-test runner."""

from __future__ import annotations

import math
import re
from typing import Any


DEFAULT_TIMEOUT = 7200
REPORT_DEVICE = "musa"
PREFLIGHT_BY_DEFAULT = True
SUPPORTS_MIN_SPEEDUP = True

# Consume the public manifests without filtering by the discovered CTest catalog.
# This makes a missing MThreads target fail closed instead of silently shrinking an
# explicit --ops all run.
FILTER_REGISTERED_TESTS = False

SPEEDUP_METRIC = "mudnn_median_us/flagdnn_median_us"
EQUIVALENT_METRIC = "nvidia_equivalent_time_us/flagdnn_median_us"
EQUIVALENT_CASE_PATTERN = re.compile(
    r"^(?:(?:elu_backward|softplus_backward)_perf_"
    r"|(?:relu_backward|leaky_relu_backward|swish_backward)_attributes_)"
    r"(?:fp32|fp16|bfloat16)_[0-9][a-z0-9_]*$"
)
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


def test_expression(suite: str, operator: str) -> str | None:
    if suite == "functional" and operator == "matmul":
        return r"^functional\.mthreads\.matmul(\.fp8)?$"
    if suite == "benchmark":
        return (
            rf"^benchmark\.mthreads\.{re.escape(operator)}"
            r"(\.(matrix|fp8))?$"
        )
    return None


def preflight_tests(suites: list[str] | tuple[str, ...]) -> set[str]:
    required = {
        "integration.mthreads.cmake_configuration_contract",
        "integration.mthreads.dependency_boundary",
        "integration.mthreads.reference_dependency_boundary",
        "integration.mthreads.installed_consumer",
        "integration.mthreads.run_tests_adapter_contract",
        "integration.mthreads.runtime",
        "integration.mthreads.compiler_contract",
        "integration.mthreads.artifact_contract",
        "integration.mthreads.jit_add",
    }
    if "benchmark" in suites:
        required.add("integration.mthreads.equivalent_performance_baseline")
        required.add(
            "integration.mthreads.benchmark_reference_dependency_boundary"
        )
    return required


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
    marker_pattern = re.escape(marker)
    if suite == "benchmark" and operator in _FUNCTIONAL_ACCOUNTING_ALIASES:
        marker_pattern = f"(?:{marker_pattern}|FLAGDNN_CONVOLUTION_BENCHMARK)"
    pattern = re.compile(
        rf"^{marker_pattern}:\s+"
        r"(PASS|SKIP)\s+cases=(\d+)\s+"
        r"executed=(\d+)\s+skipped=(\d+)\s*$"
    )
    records: list[tuple[str, str, str, str]] = []
    for raw_line in output.splitlines():
        line = re.sub(r"^\s*\d+:\s?", "", raw_line).strip()
        match = pattern.fullmatch(line)
        if match is not None:
            records.append(match.groups())
    if not records:
        return None, [
            f"mthreads {suite} op={operator} must emit at least one "
            f"{marker} accounting record; found {len(records)}"
        ]
    return {
        "status": (
            "PASS" if any(row[0] == "PASS" for row in records) else "SKIP"
        ),
        "cases": sum(int(row[1]) for row in records),
        "executed": sum(int(row[2]) for row in records),
        "skipped": sum(int(row[3]) for row in records),
    }, []


def _skip_records(output: str) -> list[dict[str, str]]:
    matches = re.findall(
        r"(?:^|\n)(?:\s*\d+:\s*)?\[skip\] case=(\S+) "
        r"provider=mudnn reason=(\S[^\n]*)",
        output,
    )
    return [
        {"case": case, "provider": "mudnn", "reason": reason}
        for case, reason in matches
    ]


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
    expected_status = "SKIP" if ctest_reported_status == "skipped" else "PASS"
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
    if ctest_reported_status == "passed" and executed <= 0:
        errors.append("a passed mthreads suite must execute at least one case")
    if ctest_reported_status == "skipped" and executed != 0:
        errors.append("a skipped mthreads suite must not claim executed cases")
    if len(_skip_records(output)) != skipped:
        errors.append("every skipped mthreads case requires a muDNN reason")
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
    if case.startswith("matmul_mxfp8_") and "matmul_fp8" in manifest_operators:
        return "matmul_fp8"
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
        if isinstance(providers, dict) and any(
            isinstance(record, dict) and "equivalent_baseline" in record
            for record in providers.values()
        ):
            if set(providers) != {"flagdnn"}:
                errors.append(
                    f"mthreads equivalent case={case} must contain only "
                    "measured FlagDNN samples"
                )
                continue
            try:
                validate_benchmark_record(providers["flagdnn"])
                if providers["flagdnn"]["case"] != case:
                    raise ValueError("record case differs from its group")
            except ValueError as error:
                errors.append(f"mthreads equivalent case={case}: {error}")
            continue
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


def validate_benchmark_record(
    record: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Accept measured v4 samples with a separate peak-scaled reference."""
    fields = {
        "schema_version", "kind", "provider", "case", "unit", "median",
        "p90", "samples", "equivalent_baseline",
    }
    if (
        set(record) != fields
        or type(record.get("schema_version")) is not int
        or record["schema_version"] != 4
        or record["kind"] != "steady_state"
        or record["provider"] != "flagdnn"
        or record["unit"] != "us"
        or not isinstance(record["case"], str)
        or EQUIVALENT_CASE_PATTERN.fullmatch(record["case"]) is None
    ):
        raise ValueError("unsupported MThreads equivalent benchmark record")
    baseline = record["equivalent_baseline"]
    identities = {
        "provider": "nvidia_equivalent", "platform": "nvidia",
        "target_platform": "mthreads", "timing_profile": "native",
        "reference_validation": "cpu_full",
    }
    numeric = {
        "reference_time_us", "reference_peak", "target_peak",
        "min_relative_efficiency", "equivalent_time_us", "limit_us",
        "relative_efficiency", "traffic_ratio",
    }
    if (
        not isinstance(baseline, dict)
        or set(baseline) != set(identities) | numeric | {"scale_metric", "passed"}
        or any(baseline[key] != value for key, value in identities.items())
        or baseline["scale_metric"] not in ("vector_fp32", "hbm_triad")
    ):
        raise ValueError("invalid MThreads equivalent baseline identity")
    expected_metric = (
        "hbm_triad" if record["case"].startswith(
            ("relu_backward_attributes_", "leaky_relu_backward_attributes_")
        ) else "vector_fp32"
    )
    if baseline["scale_metric"] != expected_metric:
        raise ValueError("equivalent scale metric differs from the operator policy")

    def positive(value: Any) -> bool:
        try:
            return type(value) in (int, float) and math.isfinite(value) and value > 0
        except OverflowError:
            return False

    if (
        any(not positive(baseline[key]) for key in numeric)
        or not 0.8 <= baseline["min_relative_efficiency"] <= 1
        or baseline["traffic_ratio"] != 1
        or not positive(record["median"])
        or not positive(record["p90"])
        or not isinstance(record["samples"], list)
        or len(record["samples"]) != 20
        or any(not positive(value) for value in record["samples"])
    ):
        raise ValueError("invalid equivalent timings, peaks or native profile")
    ordered = sorted(record["samples"])
    for key, rank in (("median", 9), ("p90", 17)):
        if not math.isclose(record[key], ordered[rank], rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"equivalent {key} differs from measured samples")
    equivalent = baseline["reference_time_us"] * baseline["reference_peak"] / baseline["target_peak"]
    derived = {
        "equivalent_time_us": equivalent,
        "limit_us": equivalent / baseline["min_relative_efficiency"],
        "relative_efficiency": equivalent / record["median"],
    }
    for key, expected in derived.items():
        if not positive(expected) or not math.isclose(
            baseline[key], expected, rel_tol=1e-12, abs_tol=0,
        ):
            raise ValueError(f"equivalent {key} differs from the peak formula")
    if type(baseline["passed"]) is not bool or baseline["passed"] != (
        record["median"] <= baseline["limit_us"]
    ):
        raise ValueError("equivalent baseline verdict differs from measured limit")
    return {"timing": {key: record[key] for key in ("median", "p90", "samples")}}


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
    accounting, _ = _accounting_record(output, operator, suite)
    result["case_skips"] = _skip_records(output)
    timing_methods = {
        case: {"method": method, "execution_count": int(count)}
        for case, method, count in re.findall(
            r"(?:^|\n)(?:\s*\d+:\s*)?\[timing\] case=(\S+) "
            r"method=(musa_graph|musa_event_batch) execution_count=(\d+)",
            output,
        )
    }
    if timing_methods:
        result["timing_methods"] = timing_methods
    # One operator can have a native matrix plus dtype-specific CTest entries.
    # CTest's textual skip marker describes an individual entry, so a mixed
    # passed/skipped group must retain its executed cases and paired timings.
    if (
        ctest_reported_status == "skipped"
        and result.get("exit_code") == 0
        and accounting is not None
        and int(accounting["executed"]) > 0
    ):
        inherited = result.get("record_errors", [])
        remaining = [
            error
            for error in inherited
            if error != "skipped benchmark emitted timing provider records"
        ]
        if remaining:
            result["record_errors"] = remaining
        else:
            result.pop("record_errors", None)
            result["status"] = "passed"
        ctest_reported_status = "passed"
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
    if accounting is not None:
        result["case_accounting"] = accounting
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
    selected_pairs = sum(len(suite_operators[suite]) for suite in suites)
    completed_pairs = sum(
        results.get(operator, {}).get(suite, {}).get("status")
        in {"passed", "skipped"}
        for suite in suites
        for operator in suite_operators[suite]
    )
    passed_pairs = sum(
        results.get(operator, {}).get(suite, {}).get("status") == "passed"
        for suite in suites
        for operator in suite_operators[suite]
    )
    speedup = _speedup_summary(results, min_speedup)
    equivalent = _equivalent_summary(results, min_speedup)
    # A run containing only equivalent cases has no native ratio to gate.
    if not speedup["case_count"] and equivalent["case_count"]:
        speedup["gate_passed"] = True
    return {
        "failed": (
            not preflight_passed
            or completed_pairs != selected_pairs
            or not speedup["gate_passed"]
            or equivalent["gate_passed"] is False
        ),
        "summary": {"speedup": speedup, "equivalent_performance": equivalent},
        "coverage": {
            "selected_operator_suite_pairs": selected_pairs,
            "passed_operator_suite_pairs": passed_pairs,
            "complete": completed_pairs == selected_pairs,
            "skipped_operator_suite_pairs": completed_pairs - passed_pairs,
        },
    }


def _equivalent_summary(
    results: dict[str, dict[str, Any]], threshold: float | None,
) -> dict[str, Any]:
    cases = []
    for operator, suites in results.items():
        benchmark = suites.get("benchmark", {})
        if benchmark.get("status") != "passed":
            continue
        for case, providers in benchmark.get("records", {}).items():
            if set(providers) != {"flagdnn"}:
                continue
            record = providers["flagdnn"]
            if record.get("schema_version") != 4:
                continue
            validate_benchmark_record(record)
            baseline = record["equivalent_baseline"]
            minimum = baseline["min_relative_efficiency"]
            required = max(minimum, threshold or 0)
            # Preserve the validated latency boundary: dividing it back into an
            # efficiency can round below the minimum even at the exact limit.
            limit = baseline["limit_us"]
            if required > minimum:
                limit = min(limit, baseline["equivalent_time_us"] / required)
            cases.append({
                "operator": operator, "case": case,
                "flagdnn_median_us": record["median"],
                "equivalent_baseline": dict(baseline),
                "relative_efficiency": baseline["relative_efficiency"],
                "required_relative_efficiency": required,
                "passed": record["median"] <= limit,
            })
    cases.sort(key=lambda row: (row["relative_efficiency"], row["case"]))
    failures = [row for row in cases if not row["passed"]]
    return {
        "metric": EQUIVALENT_METRIC,
        "requested_min_speedup": threshold,
        "gate_enabled": threshold is not None,
        "gate_passed": not failures if threshold is not None and cases else None,
        "case_count": len(cases), "passed_case_count": len(cases) - len(failures),
        "failed_case_count": len(failures),
        "minimum_relative_efficiency": cases[0]["relative_efficiency"] if cases else None,
        "cases": cases, "failures": failures,
    }


def _apply_equivalent_gates(results: dict[str, Any], equivalent: dict[str, Any]) -> None:
    gates = {(row["operator"], row["case"]): row for row in equivalent["cases"]}
    enabled = equivalent["gate_enabled"]
    for operator, suites in results.items():
        performance = suites.get("performance", {})
        for entry in performance.get("data", {}).values():
            for case, row in entry.get("details", {}).items():
                gate = gates.get((operator, case))
                if gate is None:
                    continue
                row["performance_gate"] = {
                    "enabled": enabled, "metric": EQUIVALENT_METRIC,
                    "relative_efficiency": gate["relative_efficiency"],
                    "required_relative_efficiency": gate["required_relative_efficiency"] if enabled else None,
                    "passed": gate["passed"] if enabled else None,
                }
                if enabled and not gate["passed"]:
                    entry["result"] = "FAIL"
                    performance.update(status="Failed", exit_code=1)


def _apply_native_gates(results: dict[str, Any], speedup: dict[str, Any]) -> None:
    if speedup.get("threshold") is None:
        return
    for failure in speedup.get("failures", []):
        performance = results.get(failure["operator"], {}).get("performance", {})
        for entry in performance.get("data", {}).values():
            if failure["case"] in entry.get("details", {}):
                entry["result"] = "FAIL"
                performance.update(status="Failed", exit_code=1)


def postprocess_report(
    *, report: dict[str, Any], platform_summary: dict[str, Any],
    exit_code: int | None,
) -> bool:
    equivalent = platform_summary.get("equivalent_performance")
    has_equivalent = isinstance(equivalent, dict) and equivalent.get("case_count")
    speedup = platform_summary.get("speedup", {})
    has_native_failure = speedup.get("threshold") is not None and speedup.get("failures")
    if not has_equivalent and not has_native_failure:
        return False
    if has_equivalent:
        _apply_equivalent_gates(report["result"], equivalent)
        report["equivalent_performance"] = equivalent
    _apply_native_gates(report["result"], speedup)
    report.update(
        overall_status="passed" if exit_code == 0 else "failed",
        exit_code=exit_code,
    )
    return True


def postprocess_batch_report(
    *, partitions: dict[int, dict[str, Any]], report: dict[str, Any],
    platform_summary: dict[str, Any], exit_code: int | None,
) -> bool:
    changed = postprocess_report(
        report=report, platform_summary=platform_summary, exit_code=exit_code,
    )
    if changed:
        for partition in partitions.values():
            equivalent = platform_summary.get("equivalent_performance")
            if isinstance(equivalent, dict) and equivalent.get("case_count"):
                _apply_equivalent_gates(partition, equivalent)
            _apply_native_gates(partition, platform_summary.get("speedup", {}))
    return changed


def status_is_success(status: str) -> bool:
    return status in {"passed", "skipped"}


def select_operator_manifests(
    manifests: dict[str, list[str]]
) -> dict[str, list[str]]:
    # All Graph operators are represented, including explicit muDNN capability skips.
    return {**manifests, "benchmark": list(manifests["functional"])}
