"""Hygon policy hooks for the repository batch-test runner."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


COMPARABLE_CASE_CATALOG = (
    Path(__file__).resolve().parent / "benchmark" / "comparable_cases.json"
)
SPEEDUP_METRIC = "hipdnn_median_us/flagdnn_median_us"
VISIBILITY_VARIABLES = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)
CASE_FILTER_PATTERN = re.compile(r"^FLAGDNN_[A-Z0-9_]+_CASE$")
CONVOLUTION_CASE_PATTERN = re.compile(
    r"^conv[123]d_(fprop|dgrad|wgrad)(?:_|$)"
)
FUNCTIONAL_ACCOUNTING_MARKER_OVERRIDES = {
    "conv_fprop": "CONVOLUTION",
    "conv_dgrad": "CONVOLUTION",
    "conv_wgrad": "CONVOLUTION",
}

# Allow complete dtype/shape corpora and precision sub-suites with cold caches.
DEFAULT_TIMEOUT = 21600
REPORT_DEVICE = "cuda"
PREFLIGHT_BY_DEFAULT = True
SUPPORTS_MIN_SPEEDUP = True
FILTER_REGISTERED_TESTS = False


def select_operator_manifests(
    manifests: dict[str, list[str]]
) -> dict[str, list[str]]:
    additional = (
        (Path(__file__).parent / "benchmark" / "additional_operators.txt")
        .read_text()
        .split()
    )
    if set(additional) - set(manifests["functional"]) or len(
        set(additional)
    ) != len(additional):
        raise ValueError("Invalid Hygon additional benchmark operators")
    return {
        **manifests,
        "benchmark": list(dict.fromkeys(manifests["benchmark"] + additional)),
    }


def test_expression(suite: str, operator: str) -> str | None:
    if suite == "functional" and operator in {
        "matmul",
        "conv_fprop",
        "conv_dgrad",
        "conv_wgrad",
    }:
        suffix = "ieee|tf32|fp8" if operator == "matmul" else "ieee|tf32"
        return rf"^functional\.hygon\.{operator}(\.({suffix}))?$"
    if suite == "benchmark":
        return (
            rf"^benchmark\.hygon\.{re.escape(operator)}"
            r"(\.(boolean|copy|ieee|tf32|fp32_output|fp8))?$"
        )
    return None


def configure_environment(
    environment: dict[str, str], device: str | None
) -> None:
    environment.pop("FLAGDNN_INPUT_PRECISION", None)
    environment.pop("FLAGDNN_FP8_MATMUL_API", None)
    for variable in tuple(environment):
        if CASE_FILTER_PATTERN.fullmatch(variable) is not None:
            environment.pop(variable, None)
    if device is None:
        return
    for variable in VISIBILITY_VARIABLES:
        environment.pop(variable, None)
    environment["HIP_VISIBLE_DEVICES"] = device


def preflight_tests(_suites: list[str] | tuple[str, ...]) -> set[str]:
    return {
        "integration.hygon.validation_contract",
        "integration.hygon.convolution_validation_static_contract",
        "integration.hygon.cmake_configuration_contract",
        "integration.hygon.dependency_boundary",
        "integration.hygon.compiler_contract",
        "integration.hygon.reference_dependency_boundary",
        "integration.hygon.jit",
        "integration.hygon.pointwise_smoke",
        "integration.hygon.installed_consumer",
        "integration.hygon.runtime",
        "integration.hygon.graph",
        "integration.hygon.jit_candidate_compatibility_contract",
        "integration.hygon.jit_global_state_contract",
        "integration.hygon.python_runtime_lifecycle_contract",
        "integration.hygon.normalization_stability",
    }


def preflight_metadata(environment: dict[str, str]) -> dict[str, Any]:
    return {
        "visibility_masks": {
            variable: environment[variable]
            for variable in VISIBILITY_VARIABLES
            if variable in environment
        }
    }


def convolution_case_operator(case: str) -> str | None:
    match = CONVOLUTION_CASE_PATTERN.match(case)
    if match is None:
        return None
    return f"conv_{match.group(1)}"


def benchmark_case_operator(
    case: str, manifest_operators: list[str]
) -> str | None:
    if case.startswith("matmul_mxfp8_") and "matmul_fp8" in manifest_operators:
        return "matmul_fp8"
    matches = [
        candidate
        for candidate in dict.fromkeys(manifest_operators)
        if case == candidate or case.startswith(f"{candidate}_")
    ]
    convolution_operator = convolution_case_operator(case)
    if (
        convolution_operator is not None
        and convolution_operator in manifest_operators
    ):
        matches.append(convolution_operator)
    if not matches:
        return None
    return max(matches, key=len)


def load_hygon_comparable_case_catalog(
    manifest_operators: list[str],
    path: Path = COMPARABLE_CASE_CATALOG,
) -> dict[str, Any]:
    """Load the repository-owned cases that are comparable to hipDNN."""

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
            f"Cannot load Hygon comparable-case catalog {path}: {error}"
        ) from error

    required_keys = {
        "schema_version",
        "platform",
        "suite",
        "metric",
        "source",
        "declared_operator_count",
        "declared_case_count",
        "operators",
    }
    if not isinstance(document, dict) or set(document) != required_keys:
        actual_keys = set(document) if isinstance(document, dict) else set()
        raise RuntimeError(
            "Hygon comparable-case catalog fields do not match schema; "
            f"missing={sorted(required_keys - actual_keys)}; "
            f"extra={sorted(actual_keys - required_keys)}"
        )
    schema_version = document["schema_version"]
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
        or document["platform"] != "hygon"
        or document["suite"] != "benchmark"
        or document["metric"] != SPEEDUP_METRIC
        or not isinstance(document["source"], str)
        or not document["source"].strip()
    ):
        raise RuntimeError(
            "Hygon comparable-case catalog metadata violates schema version 1"
        )
    operators = document["operators"]
    if not isinstance(operators, dict):
        raise RuntimeError(
            "Hygon comparable-case catalog operators must be an object"
        )
    if not operators:
        raise RuntimeError(
            "Hygon comparable-case catalog must declare at least one operator"
        )

    normalized: dict[str, list[str]] = {}
    all_cases: set[str] = set()
    for operator, cases in operators.items():
        if not isinstance(operator, str) or operator not in manifest_operators:
            raise RuntimeError(
                "Hygon comparable-case catalog declares an unknown benchmark "
                f"operator: {operator!r}"
            )
        if not isinstance(cases, list) or not cases:
            raise RuntimeError(
                "Hygon comparable-case catalog operator must declare a "
                f"nonempty case list: {operator}"
            )
        normalized_cases: list[str] = []
        for case in cases:
            if (
                not isinstance(case, str)
                or not case.strip()
                or case != case.strip()
            ):
                raise RuntimeError(
                    "Hygon comparable-case catalog contains an invalid case "
                    f"for operator={operator}: {case!r}"
                )
            owner = benchmark_case_operator(case, manifest_operators)
            if owner != operator:
                raise RuntimeError(
                    "Hygon comparable-case catalog case ownership mismatch: "
                    f"case={case} owner={owner} declared_operator={operator}"
                )
            if case in all_cases:
                raise RuntimeError(
                    "Hygon comparable-case catalog contains a duplicate case: "
                    + case
                )
            all_cases.add(case)
            normalized_cases.append(case)
        normalized[operator] = normalized_cases

    declared_operator_count = document["declared_operator_count"]
    declared_case_count = document["declared_case_count"]
    if (
        not isinstance(declared_operator_count, int)
        or isinstance(declared_operator_count, bool)
        or declared_operator_count != len(normalized)
        or not isinstance(declared_case_count, int)
        or isinstance(declared_case_count, bool)
        or declared_case_count != len(all_cases)
    ):
        raise RuntimeError(
            "Hygon comparable-case catalog declared counts do not match its "
            "operator/case contents"
        )

    return {
        **{key: document[key] for key in required_keys - {"operators"}},
        "catalog_path": str(path.resolve()),
        "operators": normalized,
    }


def hygon_comparable_coverage(
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
    missing: list[dict[str, str]] = []
    observed_required = 0
    observed_pairs = 0
    required_cases = {
        (operator, case)
        for operator in selected_declared
        for case in declared[operator]
    }
    for operator in selected_declared:
        benchmark = results.get(operator, {}).get("benchmark")
        status = (
            benchmark.get("status")
            if isinstance(benchmark, dict)
            else "not_run"
        )
        records = (
            benchmark.get("records", {}) if isinstance(benchmark, dict) else {}
        )
        if not isinstance(records, dict):
            records = {}
        for case in declared[operator]:
            providers = records.get(case)
            if (
                status == "passed"
                and isinstance(providers, dict)
                and set(providers) == {"flagdnn", "hipdnn"}
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
            and set(providers) == {"flagdnn", "hipdnn"}
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
        "required_case_count": len(required_cases),
        "observed_required_case_count": observed_required,
        "observed_pair_case_count": observed_pairs,
        "extra_pair_case_count": observed_pairs - observed_required,
        "missing_case_count": len(missing),
        "missing_cases": missing,
        "verified": not missing,
    }


def validate_hygon_benchmark_pairs(
    records: dict[str, dict[str, Any]],
    operator: str,
    manifest_operators: list[str],
) -> list[str]:
    errors: list[str] = []
    expected = {"flagdnn", "hipdnn"}
    if not records:
        return ["passed Hygon benchmark emitted no provider records"]
    for case, providers in records.items():
        owner = benchmark_case_operator(case, manifest_operators)
        if owner != operator:
            errors.append(
                f"Hygon benchmark emitted case={case} owned_by={owner}; "
                f"expected owner={operator}"
            )
        actual = set(providers)
        if actual != expected:
            errors.append(
                f"Hygon benchmark case={case} providers={sorted(actual)}; "
                f"expected={sorted(expected)}"
            )
    return errors


def benchmark_speedup_summary(
    results: dict[str, dict[str, Any]],
    threshold: float | None,
    comparable_coverage: dict[str, Any] | None = None,
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
                "hipdnn",
            }:
                continue
            flagdnn_us = float(providers["flagdnn"]["median"])
            hipdnn_us = float(providers["hipdnn"]["median"])
            cases.append(
                {
                    "operator": operator,
                    "case": case,
                    "flagdnn_median_us": flagdnn_us,
                    "hipdnn_median_us": hipdnn_us,
                    "speedup": hipdnn_us / flagdnn_us,
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
        threshold is None
        if comparable_coverage is None
        else bool(comparable_coverage.get("verified"))
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
        "minimum_speedup": cases[0]["speedup"] if cases else None,
        "failures": failures,
    }


def validate_hygon_case_accounting(
    output: str,
    operator: str,
    suite: str,
    ctest_reported_status: str,
    records: dict[str, dict[str, Any]],
    skip_records: list[dict[str, str]],
) -> list[str]:
    marker_records: list[tuple[str, str]] = []
    marker_test_ids: list[str | None] = []
    marker_pattern = re.compile(
        r"^(FLAGDNN_[A-Z0-9_]+_(?:FUNCTIONAL|BENCHMARK)):\s*(.*)$"
    )
    accounting_pattern = re.compile(
        r"(PASS|SKIP)\s+"
        r"(?:(?:cases)=(\d+)\s+)?executed=(\d+)\s+skipped=(\d+)\s*$"
    )
    for raw_line in output.splitlines():
        line = re.sub(r"^\s*\d+:\s?", "", raw_line).strip()
        match = marker_pattern.fullmatch(line)
        if match is not None:
            marker_records.append((match.group(1), match.group(2)))
            prefix = re.match(r"^\s*(\d+):", raw_line)
            marker_test_ids.append(prefix.group(1) if prefix else None)
    marker_operator = operator.upper()
    if suite == "functional":
        marker_operator = FUNCTIONAL_ACCOUNTING_MARKER_OVERRIDES.get(
            operator, marker_operator
        )
    expected_marker = f"FLAGDNN_{marker_operator}_{suite.upper()}"
    if not marker_records:
        return [
            f"Hygon {suite} suite omitted {expected_marker} case accounting"
        ]
    errors: list[str] = []
    finished = re.findall(
        r"^\s*\d+/\d+\s+Test\s+#(\d+):\s+"
        + re.escape(f"{suite}.hygon.{operator}")
        + r"(?:\.[a-z0-9_]+)?\s+\.*\s*(?:\*\*\*)?(Passed|Skipped)\s",
        output,
        re.MULTILINE,
    )
    if finished:
        statuses = dict(finished)
        if len(marker_test_ids) != len(statuses) or set(
            marker_test_ids
        ) != set(statuses):
            errors.append(
                "Hygon must emit one accounting marker per CTest child"
            )
        for test_id, (_, payload) in zip(marker_test_ids, marker_records):
            expected = "PASS" if statuses.get(test_id) == "Passed" else "SKIP"
            if test_id in statuses and not payload.startswith(expected + " "):
                errors.append(
                    f"Hygon accounting disagrees with CTest child #{test_id}"
                )
    executed = skipped = 0
    for marker, accounting in marker_records:
        if marker != expected_marker:
            errors.append(
                f"Hygon suite accounting marker={marker}; "
                f"expected {expected_marker}"
            )
        match = accounting_pattern.fullmatch(accounting)
        if match is None:
            errors.append(
                f"Hygon suite accounting payload for marker={marker} "
                "is malformed"
            )
            continue
        reported = match.group(1)
        matched = int(match.group(2)) if match.group(2) is not None else None
        group_executed, group_skipped = int(match.group(3)), int(
            match.group(4)
        )
        executed += group_executed
        skipped += group_skipped
        if matched is None:
            errors.append(f"Hygon {suite} accounting omitted cases=<count>")
        elif matched <= 0 or matched != group_executed + group_skipped:
            errors.append(
                "Hygon suite cases must be positive and equal executed+skipped"
            )
        if reported == "PASS" and group_executed <= 0:
            errors.append("Hygon PASS suite must execute at least one case")
        if reported == "SKIP" and (
            group_executed != 0 or matched is None or group_skipped != matched
        ):
            errors.append(
                "Hygon SKIP suite must execute zero cases and skip every case"
            )
    if (ctest_reported_status == "passed") != (executed > 0):
        errors.append(
            "Hygon aggregate execution count disagrees with CTest status"
        )
    if len(skip_records) != skipped:
        errors.append(
            f"Hygon suite reports skipped={skipped} but emitted "
            f"{len(skip_records)} unique structured skip records"
        )
    if suite == "benchmark" and len(records) != executed:
        errors.append(
            f"Hygon benchmark reports executed={executed} but emitted "
            f"{len(records)} complete case record groups"
        )
    return errors


def hipdnn_skip_records(output: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    pattern = re.compile(
        r"^\[SKIP\]\[hipdnn\]\s+op=([^\s]+)\s+" r"case=([^\s]+)\s+reason=(.+)$"
    )
    for raw_line in output.splitlines():
        line = re.sub(r"^\s*\d+:\s?", "", raw_line).strip()
        if not line.startswith("[SKIP][hipdnn]"):
            continue
        match = pattern.fullmatch(line)
        record = {"message": line}
        if match is not None:
            record.update(
                {
                    "op": match.group(1),
                    "case": match.group(2),
                    "reason": match.group(3),
                }
            )
        records.append(record)
    return records


def validate_hygon_skip_records(
    records: list[dict[str, str]],
    operator: str,
    manifest_operators: list[str],
) -> list[str]:
    errors: list[str] = []
    matching_records = 0
    if not records:
        return [
            "skipped Hygon suite emitted no structured "
            f"[SKIP][hipdnn] record for op={operator}"
        ]
    for index, record in enumerate(records, start=1):
        missing = [
            key
            for key in ("op", "case", "reason")
            if not record.get(key, "").strip()
        ]
        if missing:
            errors.append(
                f"hipDNN skip record {index} is missing non-empty "
                + ", ".join(missing)
            )
            continue
        if record["op"] != operator:
            errors.append(
                f"hipDNN skip record {index} has op={record['op']}; "
                f"expected op={operator}"
            )
            continue
        case = record["case"]
        owner = benchmark_case_operator(case, manifest_operators)
        if owner != operator:
            errors.append(
                f"hipDNN skip record {index} has case={case} "
                f"owned_by={owner}; expected owner={operator}"
            )
            continue
        matching_records += 1
    cases = [record.get("case", "") for record in records]
    if len(cases) != len(set(cases)):
        errors.append("hipDNN skip records contain duplicate case names")
    if matching_records == 0:
        errors.append(
            "skipped Hygon suite emitted no legal skip record matching "
            f"op={operator}"
        )
    return errors


def prepare(
    manifests: dict[str, list[str]], suites: list[str]
) -> dict[str, Any]:
    catalog = None
    if "benchmark" in suites:
        catalog = load_hygon_comparable_case_catalog(manifests["benchmark"])
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
    # Common CTest parsing labels a group skipped when any child is skipped.
    # Hygon allows explicit capability skips alongside measured child suites.
    if ctest_reported_status == "skipped" and re.search(
        r"FLAGDNN_[A-Z0-9_]+_(?:FUNCTIONAL|BENCHMARK): PASS cases=[1-9]",
        output,
    ):
        ctest_reported_status = "passed"
        errors = result.get("record_errors", [])
        errors = [
            error
            for error in errors
            if error != "skipped benchmark emitted timing provider records"
        ]
        if errors:
            result["record_errors"] = errors
        else:
            result.pop("record_errors", None)
            result["status"] = "passed"
    skip_records = hipdnn_skip_records(output)
    if skip_records:
        result["skip_records"] = skip_records
    if ctest_reported_status == "skipped" or skip_records:
        skip_errors = validate_hygon_skip_records(
            skip_records, operator, manifest_operators
        )
        if skip_errors:
            result["skip_record_errors"] = skip_errors
            result["status"] = "failed"
    if suite == "benchmark" and result["status"] == "passed":
        pair_errors = validate_hygon_benchmark_pairs(
            records, operator, manifest_operators
        )
        if pair_errors:
            result.setdefault("record_errors", []).extend(pair_errors)
            result["status"] = "failed"
    if ctest_reported_status in {"passed", "skipped"}:
        accounting_errors = validate_hygon_case_accounting(
            output,
            operator,
            suite,
            ctest_reported_status,
            records,
            skip_records,
        )
        if accounting_errors:
            result["case_accounting_errors"] = accounting_errors
            result["status"] = "failed"


def result_diagnostics(result: dict[str, Any]) -> list[tuple[str, str]]:
    diagnostics: list[tuple[str, str]] = []
    diagnostics.extend(
        ("hipDNN skip record error", error)
        for error in result.get("skip_record_errors", [])
    )
    diagnostics.extend(
        ("Hygon case accounting error", error)
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
        comparable_coverage = hygon_comparable_coverage(
            results, suite_operators["benchmark"], catalog
        )
        if not comparable_coverage["verified"]:
            print(
                "comparable coverage gate: "
                f"{comparable_coverage['observed_required_case_count']}/"
                f"{comparable_coverage['required_case_count']} declared cases "
                "emitted complete FlagDNN/hipDNN pairs",
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
        benchmark_speedup_summary(results, min_speedup, comparable_coverage)
        if "benchmark" in suites
        else None
    )
    if min_speedup is not None:
        assert performance is not None
        failures = performance["failures"]
        print(
            "performance gate: "
            f"{performance['passed_case_count']}/"
            f"{performance['case_count']} cases meet "
            f"speedup >= {min_speedup:.6g}",
            flush=True,
        )
        for record in failures:
            print(
                "  FAIL "
                f"op={record['operator']} case={record['case']} "
                f"flagdnn_us={record['flagdnn_median_us']:.9g} "
                f"hipdnn_us={record['hipdnn_median_us']:.9g} "
                f"speedup={record['speedup']:.9g}",
                flush=True,
            )
        if performance["case_count"] == 0:
            print(
                "  FAIL no comparable FlagDNN/hipDNN benchmark case was "
                "emitted",
                flush=True,
            )
        failed = failed or not performance["gate_passed"]

    skip_record_errors = 0
    accounting_errors = 0
    reference_skips = 0
    for operator_results in results.values():
        for result in operator_results.values():
            skip_record_errors += len(result.get("skip_record_errors", []))
            accounting_errors += len(result.get("case_accounting_errors", []))
            if result["status"] in {"passed", "skipped"}:
                reference_skips += len(result.get("skip_records", []))

    return {
        "failed": failed,
        "summary": {
            "comparable_coverage": comparable_coverage,
            "performance": performance,
        },
        "coverage": {
            "hipdnn_reference_skips": reference_skips,
            "hipdnn_skip_record_errors": skip_record_errors,
            "hygon_case_accounting_errors": accounting_errors,
        },
    }
