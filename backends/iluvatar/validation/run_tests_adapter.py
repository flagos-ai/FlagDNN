"""Iluvatar policy hooks for the repository batch-test runner."""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import tempfile
from typing import Any


SPEEDUP_METRIC = "corex_cudnn_median_us/flagdnn_median_us"
CONVOLUTION_CASE_PATTERN = re.compile(
    r"^conv[123]d_(fprop|dgrad|wgrad)(?:_|$)"
)

DEFAULT_TIMEOUT = 1800
PREFLIGHT_BY_DEFAULT = True
SUPPORTS_MIN_SPEEDUP = True
FILTER_REGISTERED_TESTS = False


def configure_environment(
    environment: dict[str, str], device: str | None
) -> None:
    if device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = device


def preflight_tests(_suites: list[str] | tuple[str, ...]) -> set[str]:
    return {
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


def convolution_case_operator(case: str) -> str | None:
    match = CONVOLUTION_CASE_PATTERN.match(case)
    if match is None:
        return None
    return f"conv_{match.group(1)}"


def benchmark_case_operator(
    case: str, manifest_operators: list[str]
) -> str | None:
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


def corex_cudnn_skip_records(output: str) -> list[dict[str, str]]:
    """Parse fixed-order Iluvatar CoreX cuDNN structured SKIP records."""
    records: list[dict[str, str]] = []
    pattern = re.compile(
        r"^\[SKIP\]\[corex-cudnn\]\s+"
        r"op=([^\s]+)\s+case=([^\s]+)\s+reason=([^\s]+)\s+"
        r"cudnn_header=([^\s]+)\s+cudnn_runtime=([^\s]+)\s+"
        r"corex=([^\s]+)\s+target=([^\s]+)\s+dtype=([^\s]+)\s+"
        r"layout=([^\s]+)\s+shape=([^\s]+)$"
    )
    fields = (
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
    for raw_line in output.splitlines():
        line = re.sub(r"^\s*\d+:\s?", "", raw_line).strip()
        if not line.startswith("[SKIP][corex-cudnn]"):
            continue
        match = pattern.fullmatch(line)
        record = {"message": line}
        if match is not None:
            record.update(dict(zip(fields, match.groups(), strict=True)))
        records.append(record)
    return records


def validate_iluvatar_skip_records(
    records: list[dict[str, str]],
    operator: str,
    manifest_operators: list[str],
) -> list[str]:
    errors: list[str] = []
    if not records:
        return [
            "Iluvatar suite emitted no structured CoreX cuDNN SKIP record "
            f"for op={operator}"
        ]
    required = (
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
    legal_reasons = {
        "NO_EXACT_CUDNN_PRIMITIVE",
        "SEMANTIC_MISMATCH",
        "DTYPE_UNSUPPORTED",
        "LAYOUT_UNSUPPORTED",
        "ATTRIBUTE_UNSUPPORTED",
        "SHAPE_UNSUPPORTED",
        "CUDNN_STATUS_NOT_SUPPORTED",
        "CAPTURE_UNSUPPORTED",
    }
    legal_dtypes = {
        "fp32",
        "fp16",
        "bf16",
        "bool",
        "fp8_e4m3",
        "fp8_e5m2",
    }
    seen_cases: set[str] = set()
    for index, record in enumerate(records, start=1):
        missing = [key for key in required if not record.get(key, "").strip()]
        if missing:
            errors.append(
                f"CoreX cuDNN skip record {index} is missing non-empty "
                + ", ".join(missing)
            )
            continue
        if record["op"] != operator:
            errors.append(
                f"CoreX cuDNN skip record {index} has op={record['op']}; "
                f"expected op={operator}"
            )
        owner = benchmark_case_operator(record["case"], manifest_operators)
        if owner != operator:
            errors.append(
                f"CoreX cuDNN skip record {index} has case={record['case']} "
                f"owned_by={owner}; expected owner={operator}"
            )
        if record["case"] in seen_cases:
            errors.append(
                "CoreX cuDNN skip records contain duplicate case names"
            )
        seen_cases.add(record["case"])
        if record["reason"] not in legal_reasons:
            errors.append(
                f"CoreX cuDNN skip record {index} has illegal "
                f"reason={record['reason']}"
            )
        for version_field in ("cudnn_header", "cudnn_runtime"):
            value = record[version_field]
            if not value.isdecimal() or int(value) <= 0:
                errors.append(
                    f"CoreX cuDNN skip record {index} has invalid "
                    f"{version_field}={value}"
                )
        if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", record["corex"]) is None:
            errors.append(
                f"CoreX cuDNN skip record {index} has invalid "
                f"corex={record['corex']}"
            )
        if record["target"] != "corex_71":
            errors.append(
                f"CoreX cuDNN skip record {index} has "
                f"target={record['target']}; expected corex_71"
            )
        if record["dtype"] not in legal_dtypes:
            errors.append(
                f"CoreX cuDNN skip record {index} has invalid "
                f"dtype={record['dtype']}"
            )
        if record["layout"] not in {"contiguous", "strided"}:
            errors.append(
                f"CoreX cuDNN skip record {index} has invalid "
                f"layout={record['layout']}"
            )
    return errors


def _parse_iluvatar_case_accounting(
    output: str, operator: str, suite: str
) -> tuple[dict[str, int | str] | None, list[str]]:
    marker_pattern = re.compile(
        r"^(FLAGDNN_[A-Z0-9_]+_(?:FUNCTIONAL|BENCHMARK)):\s*(.*)$"
    )
    markers: list[tuple[str, str]] = []
    for raw_line in output.splitlines():
        line = re.sub(r"^\s*\d+:\s?", "", raw_line).strip()
        match = marker_pattern.fullmatch(line)
        if match is not None:
            markers.append((match.group(1), match.group(2)))
    expected_marker = f"FLAGDNN_{operator.upper()}_{suite.upper()}"
    if len(markers) != 1:
        return None, [
            f"Iluvatar {suite} suite for op={operator} must emit exactly one "
            f"{expected_marker} accounting record; found {len(markers)}"
        ]
    marker, payload = markers[0]
    errors: list[str] = []
    if marker != expected_marker:
        errors.append(
            f"Iluvatar suite accounting marker={marker}; "
            f"expected {expected_marker}"
        )
    if suite == "functional":
        pattern = re.compile(
            r"(PASS|SKIP)\s+cases=(\d+)\s+production_executed=(\d+)\s+"
            r"reference_executed=(\d+)\s+reference_skipped=(\d+)\s*$"
        )
        match = pattern.fullmatch(payload)
        if match is None:
            errors.append(
                f"Iluvatar functional accounting payload for marker={marker} "
                "is malformed"
            )
            return None, errors
        status, cases, production, reference, skipped = match.groups()
        return {
            "status": status,
            "cases": int(cases),
            "production_executed": int(production),
            "reference_executed": int(reference),
            "reference_skipped": int(skipped),
        }, errors
    if suite == "benchmark":
        pattern = re.compile(
            r"(PASS|SKIP)\s+cases=(\d+)\s+comparable_executed=(\d+)\s+"
            r"reference_skipped=(\d+)\s*$"
        )
        match = pattern.fullmatch(payload)
        if match is None:
            errors.append(
                f"Iluvatar benchmark accounting payload for marker={marker} "
                "is malformed"
            )
            return None, errors
        status, cases, comparable, skipped = match.groups()
        return {
            "status": status,
            "cases": int(cases),
            "comparable_executed": int(comparable),
            "reference_skipped": int(skipped),
        }, errors
    return None, [f"unknown Iluvatar suite={suite}"]


def validate_iluvatar_case_accounting(
    output: str,
    operator: str,
    suite: str,
    ctest_reported_status: str,
    records: dict[str, dict[str, Any]],
    skip_records: list[dict[str, str]],
) -> list[str]:
    accounting, errors = _parse_iluvatar_case_accounting(
        output, operator, suite
    )
    if accounting is None:
        return errors
    cases = int(accounting["cases"])
    reported_status = str(accounting["status"])
    expected_status = "SKIP" if ctest_reported_status == "skipped" else "PASS"
    if reported_status != expected_status:
        errors.append(
            f"Iluvatar accounting status={reported_status}; "
            f"CTest status requires {expected_status}"
        )
    if cases <= 0:
        errors.append("Iluvatar suite accounting cases must be positive")
    if len(skip_records) != int(accounting["reference_skipped"]):
        errors.append(
            "Iluvatar suite reference_skipped does not match unique "
            "structured CoreX cuDNN SKIP records"
        )
    skipped_cases = {
        record.get("case", "") for record in skip_records if record.get("case")
    }
    timed_skips = sorted(skipped_cases.intersection(records))
    if timed_skips:
        errors.append(
            "Iluvatar benchmark emitted timing for skipped case(s): "
            + ", ".join(timed_skips)
        )
    if suite == "functional":
        production = int(accounting["production_executed"])
        reference = int(accounting["reference_executed"])
        skipped = int(accounting["reference_skipped"])
        if production != cases:
            errors.append(
                f"Iluvatar functional cases={cases} but "
                f"production_executed={production}"
            )
        if reference + skipped != cases:
            errors.append(
                f"Iluvatar functional cases={cases} but "
                f"reference_executed+reference_skipped={reference + skipped}"
            )
        if reported_status == "PASS" and reference <= 0:
            errors.append(
                "Iluvatar functional PASS must execute a reference case"
            )
        if reported_status == "SKIP" and not (
            reference == 0 and skipped == cases and production == cases
        ):
            errors.append(
                "Iluvatar functional SKIP requires full production execution "
                "and every reference case skipped"
            )
    else:
        comparable = int(accounting["comparable_executed"])
        skipped = int(accounting["reference_skipped"])
        if comparable + skipped != cases:
            errors.append(
                f"Iluvatar benchmark cases={cases} but "
                f"comparable_executed+reference_skipped={comparable + skipped}"
            )
        if len(records) != comparable:
            errors.append(
                f"Iluvatar benchmark comparable_executed={comparable} but "
                f"emitted {len(records)} timing case groups"
            )
        if reported_status == "PASS" and comparable <= 0:
            errors.append(
                "Iluvatar benchmark PASS must execute a comparable case"
            )
        if reported_status == "SKIP" and not (
            comparable == 0 and skipped == cases
        ):
            errors.append(
                "Iluvatar benchmark SKIP requires every reference case skipped"
            )
    return errors


def validate_iluvatar_benchmark_pairs(
    records: dict[str, dict[str, Any]],
    operator: str,
    manifest_operators: list[str],
) -> list[str]:
    errors: list[str] = []
    expected_order = ["flagdnn", "corex_cudnn"]
    if not records:
        return ["passed Iluvatar benchmark emitted no provider records"]
    for case, providers in records.items():
        owner = benchmark_case_operator(case, manifest_operators)
        if owner != operator:
            errors.append(
                f"Iluvatar benchmark emitted case={case} owned_by={owner}; "
                f"expected owner={operator}"
            )
        actual_order = list(providers)
        if actual_order != expected_order:
            errors.append(
                f"Iluvatar benchmark case={case} provider_order={actual_order}; "
                f"expected={expected_order}"
            )
            continue
        flagdnn_samples = providers["flagdnn"].get("samples", [])
        reference_samples = providers["corex_cudnn"].get("samples", [])
        if len(flagdnn_samples) != len(reference_samples):
            errors.append(
                f"Iluvatar benchmark case={case} provider sample counts differ"
            )
    return errors


def iluvatar_speedup_summary(
    results: dict[str, dict[str, Any]], threshold: float | None
) -> dict[str, Any]:
    """Summarize strict CoreX-cuDNN/FlagDNN median speedups."""
    cases: list[dict[str, Any]] = []
    benchmark_results: list[dict[str, Any]] = []
    for operator, operator_results in results.items():
        benchmark = operator_results.get("benchmark")
        if not isinstance(benchmark, dict):
            continue
        benchmark_results.append(benchmark)
        if benchmark.get("status") != "passed":
            continue
        records = benchmark.get("records", {})
        if not isinstance(records, dict):
            continue
        for case, providers in records.items():
            if not isinstance(providers, dict) or list(providers) != [
                "flagdnn",
                "corex_cudnn",
            ]:
                continue
            flagdnn_us = float(providers["flagdnn"]["median"])
            reference_us = float(providers["corex_cudnn"]["median"])
            cases.append(
                {
                    "operator": operator,
                    "case": case,
                    "flagdnn_median_us": flagdnn_us,
                    "corex_cudnn_median_us": reference_us,
                    "speedup": reference_us / flagdnn_us,
                }
            )
    coverage_verified = bool(benchmark_results) and all(
        result.get("status") in {"passed", "skipped"}
        and not result.get("record_errors")
        and not result.get("skip_record_errors")
        and not result.get("case_accounting_errors")
        and isinstance(result.get("case_accounting"), dict)
        for result in benchmark_results
    )
    cases.sort(key=lambda record: (record["speedup"], record["case"]))
    failures = (
        []
        if threshold is None
        else [record for record in cases if record["speedup"] < threshold]
    )
    ratio_gate_passed = threshold is None or (bool(cases) and not failures)
    return {
        "metric": SPEEDUP_METRIC,
        "threshold": threshold,
        "gate_passed": ratio_gate_passed and coverage_verified,
        "ratio_gate_passed": ratio_gate_passed,
        "coverage_gate_passed": coverage_verified,
        "comparable_coverage_verified": coverage_verified,
        "case_count": len(cases),
        "passed_case_count": len(cases) - len(failures),
        "failed_case_count": len(failures),
        "minimum_speedup": cases[0]["speedup"] if cases else None,
        "failures": failures,
    }


def prepare_run(
    *, environment: dict[str, str], verbose: bool
) -> dict[str, Any]:
    cache_path = Path(
        tempfile.mkdtemp(prefix="flagdnn-iluvatar-run-")
    ).resolve()
    environment["FLAGDNN_CACHE_PATH"] = str(cache_path)
    if verbose:
        print(f"Iluvatar cache: {cache_path}", flush=True)
    return {
        "cache_path": str(cache_path),
        "cache_preserved": bool(verbose),
    }


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
    skip_records = corex_cudnn_skip_records(output)
    if skip_records:
        result["skip_records"] = skip_records
    if ctest_reported_status == "skipped" or skip_records:
        skip_errors = validate_iluvatar_skip_records(
            skip_records, operator, manifest_operators
        )
        if skip_errors:
            result["skip_record_errors"] = skip_errors
            result["status"] = "failed"
    if suite == "benchmark" and result["status"] == "passed":
        pair_errors = validate_iluvatar_benchmark_pairs(
            records, operator, manifest_operators
        )
        if pair_errors:
            result.setdefault("record_errors", []).extend(pair_errors)
            result["status"] = "failed"
    if ctest_reported_status in {"passed", "skipped"}:
        accounting, parse_errors = _parse_iluvatar_case_accounting(
            output, operator, suite
        )
        if accounting is not None:
            result["case_accounting"] = accounting
        accounting_errors = validate_iluvatar_case_accounting(
            output,
            operator,
            suite,
            ctest_reported_status,
            records,
            skip_records,
        )
        if parse_errors and not accounting_errors:
            accounting_errors = parse_errors
        if accounting_errors:
            result["case_accounting_errors"] = accounting_errors
            result["status"] = "failed"


def result_diagnostics(result: dict[str, Any]) -> list[tuple[str, str]]:
    diagnostics: list[tuple[str, str]] = []
    diagnostics.extend(
        ("CoreX cuDNN skip record error", error)
        for error in result.get("skip_record_errors", [])
    )
    diagnostics.extend(
        ("Iluvatar case accounting error", error)
        for error in result.get("case_accounting_errors", [])
    )
    return diagnostics


def status_is_success(status: str) -> bool:
    return status in {"passed", "skipped"}


def _cache_summary_and_cleanup(state: dict[str, Any]) -> dict[str, Any]:
    cache_value = state.get("cache_path")
    if not isinstance(cache_value, str) or not cache_value:
        raise RuntimeError("Iluvatar adapter state has no run cache path")
    cache_path = Path(cache_value).resolve()
    temporary_root = Path(tempfile.gettempdir()).resolve()
    if (
        cache_path.parent != temporary_root
        or not cache_path.name.startswith("flagdnn-iluvatar-run-")
    ):
        raise RuntimeError(
            f"Iluvatar adapter refuses unsafe cache path: {cache_path}"
        )
    preserved = bool(state.get("cache_preserved", False))
    summary = {"path": str(cache_path), "preserved": preserved}
    if not preserved:
        shutil.rmtree(cache_path, ignore_errors=True)
    return summary


def finalize(
    *,
    results: dict[str, dict[str, Any]],
    suite_operators: dict[str, list[str]],
    suites: list[str],
    state: dict[str, Any],
    min_speedup: float | None,
    preflight_passed: bool,
) -> dict[str, Any]:
    del suite_operators, preflight_passed
    complete_pairs = 0
    skip_record_errors = 0
    accounting_errors = 0
    record_errors = 0
    functional_cases = 0
    production_executed = 0
    reference_executed = 0
    functional_reference_skipped = 0
    benchmark_cases = 0
    comparable_executed = 0
    benchmark_reference_skipped = 0
    skip_reasons: dict[str, int] = {}

    for operator_results in results.values():
        for suite_name, result in operator_results.items():
            records = result.get("records", {})
            if result.get("status") == "passed" and isinstance(records, dict):
                complete_pairs += sum(
                    isinstance(providers, dict)
                    and list(providers) == ["flagdnn", "corex_cudnn"]
                    for providers in records.values()
                )
            record_errors += len(result.get("record_errors", []))
            skip_record_errors += len(
                result.get("skip_record_errors", [])
            )
            accounting_errors += len(
                result.get("case_accounting_errors", [])
            )
            if result.get("status") in {"passed", "skipped"}:
                for record in result.get("skip_records", []):
                    reason = record.get("reason")
                    if reason:
                        skip_reasons[reason] = (
                            skip_reasons.get(reason, 0) + 1
                        )
            accounting = result.get("case_accounting")
            if not isinstance(accounting, dict):
                continue
            if suite_name == "functional":
                functional_cases += int(accounting["cases"])
                production_executed += int(
                    accounting["production_executed"]
                )
                reference_executed += int(
                    accounting["reference_executed"]
                )
                functional_reference_skipped += int(
                    accounting["reference_skipped"]
                )
            elif suite_name == "benchmark":
                benchmark_cases += int(accounting["cases"])
                comparable_executed += int(
                    accounting["comparable_executed"]
                )
                benchmark_reference_skipped += int(
                    accounting["reference_skipped"]
                )

    performance = (
        iluvatar_speedup_summary(results, min_speedup)
        if "benchmark" in suites
        else None
    )
    comparable_coverage: dict[str, Any] | None = None
    failed = False
    if performance is not None:
        missing_pairs = max(0, comparable_executed - complete_pairs)
        comparable_coverage = {
            "metric": SPEEDUP_METRIC,
            "required_case_count": comparable_executed,
            "observed_pair_case_count": complete_pairs,
            "missing_case_count": missing_pairs,
            "verified": (
                performance["coverage_gate_passed"]
                and missing_pairs == 0
            ),
        }
        if not comparable_coverage["verified"]:
            print(
                "Iluvatar comparable coverage gate failed: "
                f"pairs={complete_pairs} accounted={comparable_executed}",
                flush=True,
            )
            failed = True
        if min_speedup is not None:
            print(
                "performance gate: "
                f"{performance['passed_case_count']}/"
                f"{performance['case_count']} cases meet "
                f"speedup >= {min_speedup:.6g}",
                flush=True,
            )
            for record in performance["failures"]:
                print(
                    "  FAIL "
                    f"op={record['operator']} case={record['case']} "
                    f"flagdnn_us={record['flagdnn_median_us']:.9g} "
                    "corex_cudnn_us="
                    f"{record['corex_cudnn_median_us']:.9g} "
                    f"speedup={record['speedup']:.9g}",
                    flush=True,
                )
            if performance["case_count"] == 0:
                print(
                    "  FAIL no comparable FlagDNN/CoreX cuDNN benchmark "
                    "case was emitted",
                    flush=True,
                )
        failed = failed or not performance["gate_passed"]

    iluvatar_coverage = {
        "production": {
            "functional_cases": functional_cases,
            "functional_executed": production_executed,
        },
        "reference": {
            "functional_executed": reference_executed,
            "functional_skipped": functional_reference_skipped,
            "skip_reasons": dict(sorted(skip_reasons.items())),
        },
        "benchmark": {
            "cases": benchmark_cases,
            "comparable_executed": comparable_executed,
            "reference_skipped": benchmark_reference_skipped,
            "complete_provider_pairs": complete_pairs,
        },
        "record_errors": record_errors,
        "skip_record_errors": skip_record_errors,
        "case_accounting_errors": accounting_errors,
    }
    cache_summary = _cache_summary_and_cleanup(state)
    return {
        "failed": failed,
        "summary": {
            "comparable_coverage": comparable_coverage,
            "iluvatar_coverage": iluvatar_coverage,
            "iluvatar_cache": cache_summary,
            "performance": performance,
        },
        "coverage": {
            "corex_cudnn_reference_skips": (
                functional_reference_skipped
                + benchmark_reference_skipped
            ),
            "corex_cudnn_skip_record_errors": skip_record_errors,
            "iluvatar_case_accounting_errors": accounting_errors,
        },
    }
