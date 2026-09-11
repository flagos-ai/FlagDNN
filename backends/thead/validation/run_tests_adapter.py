"""THead policy hooks for the repository batch-test runner."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


COMPARABLE_CASE_CATALOG = (
    Path(__file__).resolve().parent / "benchmark" / "comparable_cases.json"
)
SPEEDUP_METRIC = "acdnn_median_us/flagdnn_median_us"
VISIBILITY_VARIABLES = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_VISIBLE_DEVICES",
    "NPU_VISIBLE_DEVICES",
    "HGGC_VISIBLE_DEVICES",
)
CASE_FILTER_PATTERN = re.compile(r"^FLAGDNN_[A-Z0-9_]+_CASE$")
MARKER_PATTERN = re.compile(
    r"^(FLAGDNN_[A-Z0-9_]+_(?:FUNCTIONAL|BENCHMARK)):\s*(.*)$"
)
FUNCTIONAL_ACCOUNTING_PATTERN = re.compile(
    r"(PASS|SKIP)\s+cases=(\d+)\s+executed=(\d+)\s+skipped=(\d+)\s*$"
)
BENCHMARK_ACCOUNTING_PATTERN = re.compile(
    r"(PASS|SKIP)\s+cases=(\d+)\s+comparable_executed=(\d+)\s+"
    r"reference_skipped=(\d+)\s*$"
)
ACDNN_SKIP_PATTERN = re.compile(
    r"^\[SKIP\]\[acdnn\]\s+"
    r"op=(?P<op>[^\s]+)\s+"
    r"case=(?P<case>[^\s]+)\s+"
    r"reason=(?P<reason>[^\s]+)\s+"
    r"sdk=(?P<sdk>[^\s]+)\s+"
    r"acdnn_header=(?P<acdnn_header>[^\s]+)\s+"
    r"acdnn_runtime=(?P<acdnn_runtime>[^\s]+)\s+"
    r"target=(?P<target>[^\s]+)\s+"
    r"dtype=(?P<dtype>[^\s]+)\s+"
    r"layout=(?P<layout>[^\s]+)\s+"
    r"shape=(?P<shape>[^\s]+)$"
)
CONVOLUTION_CASE_PATTERN = re.compile(
    r"^conv[123]d_(fprop|dgrad|wgrad)(?:_|$)"
)

# The runner applies this budget to the entire serial preflight as well as
# each operator. PPU codegen/Graph/JIT/autotune and installed-consumer checks
# exceed 30 minutes with a cold FlagTree cache; CTest still bounds each test.
DEFAULT_TIMEOUT = 7200
PREFLIGHT_BY_DEFAULT = True
SUPPORTS_MIN_SPEEDUP = True
FILTER_REGISTERED_TESTS = False
EXTENDED_UNARY_OPERATIONS = (
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


def _ctest_line(raw_line: str) -> str:
    return re.sub(r"^\s*\d+:\s?", "", raw_line).strip()


def configure_environment(
    environment: dict[str, str], device: str | None
) -> None:
    """Remove stale case filters and normalize an explicit PPU selection."""
    for variable in tuple(environment):
        if CASE_FILTER_PATTERN.fullmatch(variable) is not None:
            environment.pop(variable, None)
    if device is None:
        return
    for variable in VISIBILITY_VARIABLES:
        environment.pop(variable, None)
    environment["HGGC_VISIBLE_DEVICES"] = device


def preflight_tests(_suites: list[str] | tuple[str, ...]) -> set[str]:
    """Return contracts that must exist before an operator test may run."""
    required = {
        "integration.thead.validation_contract",
        "integration.thead.capability_contract",
        "integration.thead.acdnn_gap_contract",
        "integration.thead.acdnn_fp8_codec_contract",
        "integration.thead.catalog_closure_contract",
        "integration.thead.cmake_configuration_contract",
        "integration.thead.dependency_boundary",
        "integration.thead.reference_dependency_boundary",
        "integration.thead.artifact_contract",
        "integration.thead.compiler_contract",
        "integration.thead.triton_compat_contract",
        "integration.thead.jit",
        "integration.thead.jit_mul",
        "integration.thead.jit_sub",
        "integration.thead.jit_min",
        "integration.thead.jit_max",
        "integration.thead.jit_scale",
        "integration.thead.jit_relu",
        "integration.thead.jit_sigmoid",
        "integration.thead.jit_tanh",
        "integration.thead.jit_elu",
        "integration.thead.jit_identity",
        "integration.thead.jit_gelu",
        "integration.thead.jit_sqrt",
        "integration.thead.jit_neg",
        "integration.thead.jit_abs",
        "integration.thead.jit_ceil",
        "integration.thead.jit_floor",
        "integration.thead.jit_exp",
        "integration.thead.triton_compilation",
        "integration.thead.runtime",
        "integration.thead.graph",
        "integration.thead.graph_mul",
        "integration.thead.graph_sub",
        "integration.thead.graph_min",
        "integration.thead.graph_max",
        "integration.thead.graph_scale",
        "integration.thead.graph_relu",
        "integration.thead.graph_sigmoid",
        "integration.thead.graph_tanh",
        "integration.thead.graph_elu",
        "integration.thead.graph_identity",
        "integration.thead.graph_gelu",
        "integration.thead.graph_sqrt",
        "integration.thead.graph_neg",
        "integration.thead.graph_abs",
        "integration.thead.graph_ceil",
        "integration.thead.graph_floor",
        "integration.thead.graph_exp",
        "integration.thead.graph_capture_safety_contract",
        "integration.thead.autotune",
        "integration.thead.autotune_mul",
        "integration.thead.autotune_sub",
        "integration.thead.autotune_min",
        "integration.thead.autotune_max",
        "integration.thead.autotune_scale",
        "integration.thead.autotune_relu",
        "integration.thead.autotune_sigmoid",
        "integration.thead.autotune_tanh",
        "integration.thead.autotune_elu",
        "integration.thead.autotune_identity",
        "integration.thead.autotune_gelu",
        "integration.thead.autotune_sqrt",
        "integration.thead.autotune_neg",
        "integration.thead.autotune_abs",
        "integration.thead.autotune_ceil",
        "integration.thead.autotune_floor",
        "integration.thead.autotune_exp",
        "integration.thead.autotune_policy_contract",
        "integration.thead.jit_candidate_compatibility_contract",
        "integration.thead.run_tests_adapter_contract",
        "integration.thead.installed_consumer",
    }
    required.update(
        f"integration.thead.{phase}_{operation}"
        for operation in EXTENDED_UNARY_OPERATIONS
        for phase in ("jit", "graph", "autotune")
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


def benchmark_case_operator(
    case: str, manifest_operators: list[str]
) -> str | None:
    """Resolve a case using manifest prefixes and dimensional conv aliases."""
    unique_operators = list(dict.fromkeys(manifest_operators))
    matches = [
        candidate
        for candidate in unique_operators
        if case == candidate or case.startswith(f"{candidate}_")
    ]
    if matches:
        return max(matches, key=len)
    convolution = CONVOLUTION_CASE_PATTERN.match(case)
    if convolution is None:
        return None
    owner = f"conv_{convolution.group(1)}"
    return owner if owner in unique_operators else None


def load_thead_comparable_case_catalog(
    manifest_operators: list[str],
    path: Path = COMPARABLE_CASE_CATALOG,
) -> dict[str, Any]:
    """Load and validate cases that must produce FlagDNN/acDNN pairs."""

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    try:
        document = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=unique_object
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(
            f"Cannot load THead comparable-case catalog {path}: {error}"
        ) from error

    required_keys = {
        "schema_version",
        "platform",
        "reference_provider",
        "operators",
    }
    if not isinstance(document, dict) or set(document) != required_keys:
        actual_keys = set(document) if isinstance(document, dict) else set()
        raise RuntimeError(
            "THead comparable-case catalog fields do not match schema; "
            f"missing={sorted(required_keys - actual_keys)}; "
            f"extra={sorted(actual_keys - required_keys)}"
        )
    schema_version = document["schema_version"]
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 2
        or document["platform"] != "thead"
        or document["reference_provider"] != "acdnn"
    ):
        raise RuntimeError(
            "THead comparable-case catalog metadata violates schema version 2"
        )

    operators = document["operators"]
    if not isinstance(operators, dict) or not operators:
        raise RuntimeError(
            "THead comparable-case catalog operators must be a nonempty object"
        )
    allowed_statuses = {"comparable", "probe_required", "unsupported"}
    required_statuses = {"comparable", "probe_required"}
    normalized_cases: dict[str, dict[str, str]] = {}
    required_by_operator: dict[str, list[str]] = {}
    status_counts = {status: 0 for status in sorted(allowed_statuses)}
    for operation, cases in operators.items():
        if (
            not isinstance(operation, str)
            or not operation
            or operation not in manifest_operators
        ):
            raise RuntimeError(
                "THead comparable-case catalog contains an invalid operator: "
                f"{operation!r}"
            )
        if not isinstance(cases, dict) or not cases:
            raise RuntimeError(
                "THead comparable-case operator cases must be a nonempty "
                f"object: operator={operation}"
            )
        required_cases: list[str] = []
        for case, capability in cases.items():
            if (
                not isinstance(case, str)
                or not case
                or case != case.strip()
                or case in normalized_cases
            ):
                raise RuntimeError(
                    "THead comparable-case catalog contains an invalid or "
                    f"duplicate case name: {case!r}"
                )
            owner = benchmark_case_operator(case, manifest_operators)
            if owner != operation:
                raise RuntimeError(
                    "THead comparable-case catalog case ownership mismatch: "
                    f"case={case} owner={owner} declared_operator={operation}"
                )
            capability_keys = {"status", "reason_code", "detail"}
            if (
                not isinstance(capability, dict)
                or set(capability) != capability_keys
            ):
                actual = (
                    set(capability) if isinstance(capability, dict) else set()
                )
                raise RuntimeError(
                    "THead comparable-case capability fields do not match "
                    f"schema for case={case}; "
                    f"missing={sorted(capability_keys - actual)}; "
                    f"extra={sorted(actual - capability_keys)}"
                )
            status = capability.get("status")
            reason_code = capability.get("reason_code")
            detail = capability.get("detail")
            if (
                status not in allowed_statuses
                or not isinstance(reason_code, str)
                or reason_code != reason_code.strip()
                or not isinstance(detail, str)
                or not detail.strip()
            ):
                raise RuntimeError(
                    "THead comparable-case capability is invalid for "
                    f"case={case}"
                )
            if status == "comparable" and reason_code:
                raise RuntimeError(
                    "THead comparable benchmark case has a skip reason: "
                    f"case={case}"
                )
            if (
                status == "probe_required"
                and reason_code != "real_device_qualification_pending"
            ):
                raise RuntimeError(
                    "THead benchmark probe has no qualification reason: "
                    f"case={case}"
                )
            if status == "unsupported" and not reason_code:
                raise RuntimeError(
                    "THead unsupported benchmark case has no reason: "
                    f"case={case}"
                )
            normalized = {
                "status": status,
                "reason_code": reason_code,
                "detail": detail,
            }
            normalized_cases[case] = normalized
            status_counts[status] += 1
            if status in required_statuses:
                required_cases.append(case)
        required_by_operator[operation] = required_cases
    return {
        "schema_version": schema_version,
        "platform": "thead",
        "reference_provider": "acdnn",
        "metric": SPEEDUP_METRIC,
        "source": "repository-owned THead acDNN capability qualification",
        "catalog_path": str(path.resolve()),
        "declared_operator_count": len(required_by_operator),
        "declared_case_count": sum(
            len(cases) for cases in required_by_operator.values()
        ),
        "catalog_case_count": len(normalized_cases),
        "case_status_counts": status_counts,
        "cases": normalized_cases,
        "operators": required_by_operator,
    }


def thead_comparable_coverage(
    results: dict[str, dict[str, Any]],
    selected_benchmark_operators: list[str],
    catalog: dict[str, Any],
) -> dict[str, Any]:
    declared = catalog["operators"]
    selected_declared = [
        operator
        for operator in dict.fromkeys(selected_benchmark_operators)
        if operator in declared
    ]
    required_cases = [
        {"operator": operator, "case": case}
        for operator in selected_declared
        for case in declared[operator]
    ]
    missing: list[dict[str, str]] = []
    observed_required = 0
    for required in required_cases:
        operator = required["operator"]
        case = required["case"]
        benchmark = results.get(operator, {}).get("benchmark")
        status = (
            benchmark.get("status")
            if isinstance(benchmark, dict)
            else "not_run"
        )
        records = (
            benchmark.get("records", {})
            if isinstance(benchmark, dict)
            else {}
        )
        providers = records.get(case) if isinstance(records, dict) else None
        if (
            status == "passed"
            and isinstance(providers, dict)
            and set(providers) == {"flagdnn", "acdnn"}
        ):
            observed_required += 1
            continue
        missing.append(
            {
                "operator": operator,
                "case": case,
                "benchmark_status": str(status),
            }
        )

    observed_pairs = 0
    for operator in dict.fromkeys(selected_benchmark_operators):
        benchmark = results.get(operator, {}).get("benchmark")
        if (
            not isinstance(benchmark, dict)
            or benchmark.get("status") != "passed"
        ):
            continue
        records = benchmark.get("records", {})
        if not isinstance(records, dict):
            continue
        observed_pairs += sum(
            isinstance(providers, dict)
            and set(providers) == {"flagdnn", "acdnn"}
            for providers in records.values()
        )

    return {
        "schema_version": catalog["schema_version"],
        "metric": catalog["metric"],
        "source": catalog["source"],
        "catalog_path": catalog.get(
            "catalog_path", str(COMPARABLE_CASE_CATALOG)
        ),
        "declared_operator_count": catalog["declared_operator_count"],
        "declared_case_count": catalog["declared_case_count"],
        "selected_declared_operators": selected_declared,
        "required_cases": required_cases,
        "required_case_count": len(required_cases),
        "observed_required_case_count": observed_required,
        "observed_pair_case_count": observed_pairs,
        "extra_pair_case_count": observed_pairs - observed_required,
        "missing_case_count": len(missing),
        "missing_cases": missing,
        "verified": not missing,
    }


def validate_thead_benchmark_pairs(
    records: dict[str, dict[str, Any]],
    operator: str,
    manifest_operators: list[str],
) -> list[str]:
    errors: list[str] = []
    expected = {"flagdnn", "acdnn"}
    if not records:
        return ["passed THead benchmark emitted no provider records"]
    for case, providers in records.items():
        owner = benchmark_case_operator(case, manifest_operators)
        if owner != operator:
            errors.append(
                f"THead benchmark emitted case={case} owned_by={owner}; "
                f"expected owner={operator}"
            )
        actual = set(providers) if isinstance(providers, dict) else set()
        if actual != expected:
            errors.append(
                f"THead benchmark case={case} providers={sorted(actual)}; "
                f"expected={sorted(expected)}"
            )
    return errors


def benchmark_speedup_summary(
    results: dict[str, dict[str, Any]],
    threshold: float | None,
    comparable_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    required = None
    if comparable_coverage is not None:
        required = {
            (record["operator"], record["case"])
            for record in comparable_coverage.get("required_cases", [])
        }
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
            if required is not None and (operator, case) not in required:
                continue
            if not isinstance(providers, dict) or set(providers) != {
                "flagdnn",
                "acdnn",
            }:
                continue
            flagdnn_us = float(providers["flagdnn"]["median"])
            acdnn_us = float(providers["acdnn"]["median"])
            cases.append(
                {
                    "operator": operator,
                    "case": case,
                    "flagdnn_median_us": flagdnn_us,
                    "acdnn_median_us": acdnn_us,
                    "speedup": acdnn_us / flagdnn_us,
                }
            )
    cases.sort(key=lambda record: (record["speedup"], record["case"]))
    failures = (
        []
        if threshold is None
        else [record for record in cases if record["speedup"] < threshold]
    )
    ratio_gate_passed = threshold is None or (bool(cases) and not failures)
    coverage_gate_passed = (
        True
        if comparable_coverage is None
        else bool(comparable_coverage.get("verified"))
    )
    speedups = [record["speedup"] for record in cases]
    geometric_mean = (
        math.exp(sum(math.log(speedup) for speedup in speedups) / len(speedups))
        if speedups
        else None
    )
    return {
        "metric": SPEEDUP_METRIC,
        "threshold": threshold,
        "gate_passed": ratio_gate_passed and coverage_gate_passed,
        "ratio_gate_passed": ratio_gate_passed,
        "coverage_gate_passed": coverage_gate_passed,
        "comparable_coverage_verified": (
            None
            if comparable_coverage is None
            else bool(comparable_coverage.get("verified"))
        ),
        "case_count": len(cases),
        "passed_case_count": len(cases) - len(failures),
        "failed_case_count": len(failures),
        "minimum_speedup": min(speedups) if speedups else None,
        "median_speedup": statistics.median(speedups) if speedups else None,
        "geometric_mean_speedup": geometric_mean,
        "failures": failures,
    }


def acdnn_skip_records(output: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for raw_line in output.splitlines():
        line = _ctest_line(raw_line)
        if not line.startswith("[SKIP][acdnn]"):
            continue
        match = ACDNN_SKIP_PATTERN.fullmatch(line)
        record = {"message": line}
        if match is not None:
            record.update(match.groupdict())
        records.append(record)
    return records


def validate_thead_skip_records(
    records: list[dict[str, str]],
    operator: str,
    manifest_operators: list[str],
) -> list[str]:
    errors: list[str] = []
    required_fields = (
        "op",
        "case",
        "reason",
        "sdk",
        "acdnn_header",
        "acdnn_runtime",
        "target",
        "dtype",
        "layout",
        "shape",
    )
    matching_records = 0
    if not records:
        return [
            "skipped THead suite emitted no structured "
            f"[SKIP][acdnn] record for op={operator}"
        ]
    for index, record in enumerate(records, start=1):
        missing = [
            field
            for field in required_fields
            if not record.get(field, "").strip()
        ]
        if missing:
            errors.append(
                f"acDNN skip record {index} is missing non-empty "
                + ", ".join(missing)
            )
            continue
        if record["op"] != operator:
            errors.append(
                f"acDNN skip record {index} has op={record['op']}; "
                f"expected op={operator}"
            )
            continue
        case = record["case"]
        owner = benchmark_case_operator(case, manifest_operators)
        if owner != operator:
            errors.append(
                f"acDNN skip record {index} has case={case} "
                f"owned_by={owner}; expected owner={operator}"
            )
            continue
        matching_records += 1
    cases = [record.get("case", "") for record in records]
    if len(cases) != len(set(cases)):
        errors.append("acDNN skip records contain duplicate case names")
    if matching_records == 0:
        errors.append(
            "skipped THead suite emitted no legal skip record matching "
            f"op={operator}"
        )
    return errors


def validate_thead_case_accounting(
    output: str,
    operator: str,
    suite: str,
    ctest_reported_status: str,
    records: dict[str, dict[str, Any]],
    skip_records: list[dict[str, str]],
) -> tuple[dict[str, Any] | None, list[str]]:
    marker_records: list[tuple[str, str]] = []
    for raw_line in output.splitlines():
        match = MARKER_PATTERN.fullmatch(_ctest_line(raw_line))
        if match is not None:
            marker_records.append((match.group(1), match.group(2)))
    expected_marker = f"FLAGDNN_{operator.upper()}_{suite.upper()}"
    if len(marker_records) != 1:
        found_markers = [record[0] for record in marker_records]
        return None, [
            f"THead {suite} suite for op={operator} must emit exactly one "
            f"{expected_marker} case accounting record; found "
            f"{len(marker_records)} markers={found_markers}"
        ]

    marker, payload = marker_records[0]
    errors: list[str] = []
    if marker != expected_marker:
        errors.append(
            f"THead suite accounting marker={marker}; expected "
            f"{expected_marker}"
        )
    pattern = (
        FUNCTIONAL_ACCOUNTING_PATTERN
        if suite == "functional"
        else BENCHMARK_ACCOUNTING_PATTERN
    )
    match = pattern.fullmatch(payload)
    if match is None:
        errors.append(
            f"THead suite accounting payload for marker={marker} is malformed"
        )
        return None, errors

    reported = match.group(1)
    cases = int(match.group(2))
    executed = int(match.group(3))
    skipped = int(match.group(4))
    accounting: dict[str, Any] = {"status": reported, "cases": cases}
    if suite == "functional":
        accounting.update({"executed": executed, "skipped": skipped})
    else:
        accounting.update(
            {
                "comparable_executed": executed,
                "reference_skipped": skipped,
            }
        )

    expected_reported = (
        "SKIP" if ctest_reported_status == "skipped" else "PASS"
    )
    if reported != expected_reported:
        errors.append(
            f"THead suite accounting status={reported}; CTest status "
            f"requires {expected_reported}"
        )
    if cases <= 0:
        errors.append("THead suite accounting cases must be positive")
    if cases != executed + skipped:
        errors.append(
            f"THead suite cases={cases} but executed+skipped="
            f"{executed + skipped}"
        )
    if reported == "PASS" and executed <= 0:
        errors.append("THead PASS suite must execute at least one case")
    if reported == "SKIP" and (executed != 0 or skipped != cases):
        errors.append(
            "THead SKIP suite must execute zero cases and skip every case"
        )
    if len(skip_records) != skipped:
        errors.append(
            f"THead suite reports skipped={skipped} but emitted "
            f"{len(skip_records)} unique structured skip records"
        )
    if suite == "benchmark" and len(records) != executed:
        errors.append(
            f"THead benchmark reports comparable_executed={executed} but "
            f"emitted {len(records)} complete case record groups"
        )
    record_cases = set(records)
    skipped_cases = {
        record["case"] for record in skip_records if "case" in record
    }
    overlap = sorted(record_cases.intersection(skipped_cases))
    if overlap:
        errors.append(
            "THead suite emitted both timing and skip records for cases: "
            + ", ".join(overlap)
        )
    return accounting, errors


def prepare(
    manifests: dict[str, list[str]], suites: list[str]
) -> dict[str, Any]:
    catalog = None
    if "benchmark" in suites:
        catalog = load_thead_comparable_case_catalog(manifests["benchmark"])
    return {"comparable_catalog": catalog}


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
    skip_records = acdnn_skip_records(output)
    if skip_records:
        result["skip_records"] = skip_records
    if ctest_reported_status == "skipped" or skip_records:
        skip_errors = validate_thead_skip_records(
            skip_records, operator, manifest_operators
        )
        if skip_errors:
            result["skip_record_errors"] = skip_errors
            result["status"] = "failed"
    if suite == "benchmark" and result["status"] == "passed":
        pair_errors = validate_thead_benchmark_pairs(
            records, operator, manifest_operators
        )
        if pair_errors:
            result.setdefault("record_errors", []).extend(pair_errors)
            result["status"] = "failed"
    if ctest_reported_status in {"passed", "skipped"}:
        accounting, accounting_errors = validate_thead_case_accounting(
            output,
            operator,
            suite,
            ctest_reported_status,
            records,
            skip_records,
        )
        if accounting is not None:
            result["case_accounting"] = accounting
        if accounting_errors:
            result["case_accounting_errors"] = accounting_errors
            result["status"] = "failed"


def result_diagnostics(result: dict[str, Any]) -> list[tuple[str, str]]:
    diagnostics: list[tuple[str, str]] = []
    diagnostics.extend(
        ("acDNN skip record error", error)
        for error in result.get("skip_record_errors", [])
    )
    diagnostics.extend(
        ("THead case accounting error", error)
        for error in result.get("case_accounting_errors", [])
    )
    return diagnostics


def status_is_success(status: str) -> bool:
    return status in {"passed", "skipped"}


def finalize(
    *,
    results: dict[str, dict[str, Any]],
    suite_operators: dict[str, list[str]],
    suites: list[str],
    state: dict[str, Any],
    min_speedup: float | None,
    preflight_passed: bool,
) -> dict[str, Any]:
    failed = False
    catalog = state.get("comparable_catalog")
    comparable_coverage: dict[str, Any] | None = None
    if catalog is not None and preflight_passed:
        comparable_coverage = thead_comparable_coverage(
            results, suite_operators["benchmark"], catalog
        )
        if not comparable_coverage["verified"]:
            print(
                "comparable coverage gate: "
                f"{comparable_coverage['observed_required_case_count']}/"
                f"{comparable_coverage['required_case_count']} declared cases "
                "emitted complete FlagDNN/acDNN pairs",
                flush=True,
            )
            for missing in comparable_coverage["missing_cases"]:
                print(
                    "  FAIL missing comparable pair "
                    f"op={missing['operator']} case={missing['case']} "
                    f"benchmark_status={missing['benchmark_status']}",
                    flush=True,
                )
            failed = True

    performance = (
        benchmark_speedup_summary(
            results, min_speedup, comparable_coverage
        )
        if "benchmark" in suites
        else None
    )
    if min_speedup is not None:
        assert performance is not None
        print(
            "performance gate: "
            f"{performance['passed_case_count']}/"
            f"{performance['case_count']} declared cases meet "
            f"speedup >= {min_speedup:.6g}",
            flush=True,
        )
        for record in performance["failures"]:
            print(
                "  FAIL "
                f"op={record['operator']} case={record['case']} "
                f"flagdnn_us={record['flagdnn_median_us']:.9g} "
                f"acdnn_us={record['acdnn_median_us']:.9g} "
                f"speedup={record['speedup']:.9g}",
                flush=True,
            )
        if performance["case_count"] == 0:
            print(
                "  FAIL no declared comparable FlagDNN/acDNN benchmark "
                "case was emitted",
                flush=True,
            )
        failed = failed or not performance["gate_passed"]

    skip_record_errors = 0
    accounting_errors = 0
    reference_skips = 0
    for operator_results in results.values():
        for result in operator_results.values():
            skip_record_errors += len(result.get("skip_record_errors", []))
            accounting_errors += len(
                result.get("case_accounting_errors", [])
            )
            reference_skips += len(result.get("skip_records", []))
    failed = failed or skip_record_errors > 0 or accounting_errors > 0

    return {
        "failed": failed,
        "summary": {
            "comparable_coverage": comparable_coverage,
            "performance": performance,
        },
        "coverage": {
            "acdnn_reference_skips": reference_skips,
            "acdnn_skip_record_errors": skip_record_errors,
            "thead_case_accounting_errors": accounting_errors,
        },
    }
