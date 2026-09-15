"""NVIDIA policy hooks for the repository batch-test runner."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

DEFAULT_TIMEOUT = 3600
REPORT_DEVICE = "cuda"
PREFLIGHT_BY_DEFAULT = False
SUPPORTS_MIN_SPEEDUP = True
FILTER_REGISTERED_TESTS = False

SPEEDUP_METRIC = "cudnn_gpu_median_us / flagdnn_gpu_median_us"


def select_operator_manifests(
    manifests: dict[str, list[str]],
) -> dict[str, list[str]]:
    excluded = set(
        (Path(__file__).parent / "benchmark" / "dtype_only_operators.txt")
        .read_text(encoding="utf-8")
        .split()
    )
    unknown = excluded - set(manifests["benchmark"])
    if unknown:
        raise ValueError(
            f"Unknown NVIDIA benchmark exclusions: {sorted(unknown)}"
        )
    additional = (
        (Path(__file__).parent / "benchmark" / "additional_operators.txt")
        .read_text(encoding="utf-8")
        .split()
    )
    unknown = set(additional) - set(manifests["functional"])
    if unknown or len(set(additional)) != len(additional):
        raise ValueError("Invalid NVIDIA additional benchmark operators")
    unsupported = set(
        (Path(__file__).parent / "cudnn_unsupported_operators.txt")
        .read_text(encoding="utf-8")
        .split()
    )
    unknown = unsupported - set(manifests["functional"])
    if unknown:
        raise ValueError(f"Unknown NVIDIA cuDNN exclusions: {sorted(unknown)}")
    selected = [op for op in manifests["benchmark"] if op not in unsupported]
    return {
        **manifests,
        "functional": [
            op for op in manifests["functional"] if op not in unsupported
        ],
        "benchmark": list(dict.fromkeys(selected + additional)),
    }


def configure_environment(
    environment: dict[str, str], device: str | None
) -> None:
    if device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = device


def test_expression(suite: str, operator: str) -> str | None:
    if suite == "functional" and operator in {
        "matmul",
        "conv_fprop",
        "conv_dgrad",
        "conv_wgrad",
    }:
        suffix = "ieee|tf32|fp8" if operator == "matmul" else "ieee|tf32"
        return rf"^functional\.nvidia\.{operator}(\.({suffix}))?$"
    if suite == "benchmark":
        return (
            rf"^benchmark\.nvidia\.{re.escape(operator)}"
            r"(\.(boolean|copy|ieee|tf32|fp32_output|fp8))?$"
        )
    return None


def preflight_tests(_suites: list[str] | tuple[str, ...]) -> set[str]:
    return {
        "core.autotune_policy",
        "core.library_abi",
        "core.build_environment.parallel",
        "core.build_environment.parallel_handles",
        "core.build_environment.parallel_cold_handles",
        "core.build_environment.fail",
        "core.build_environment.not_ready",
        "integration.nvidia.dependency_boundary",
        "integration.nvidia.reference_dependency_boundary",
        "integration.nvidia.runtime",
        "integration.nvidia.graph",
        "integration.nvidia.compiler_contract",
        "integration.nvidia.artifact_contract",
        "integration.nvidia.tensor_map_first",
        "integration.nvidia.short_map_first",
        "integration.nvidia.execution_contract",
        "integration.nvidia.ieee_range",
        "integration.nvidia.generation_contract",
    }


def validate_benchmark_record(
    record: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """v3 retains CUDA Graph timing and adds separate runtime costs."""
    expected = {
        "schema_version",
        "kind",
        "provider",
        "case",
        "unit",
        "median",
        "p90",
        "samples",
        "host_submit_us",
        "build_us",
        "warm_build_us",
        "workspace_bytes",
        "build_cache",
    }
    if "comparison" in record:
        raise ValueError("NVIDIA benchmark records require a cuDNN comparison")
    if set(record) != expected:
        raise ValueError("fields do not match NVIDIA benchmark schema v3")
    if (
        type(record["schema_version"]) is not int
        or record["schema_version"] != 3
        or record["kind"] != "steady_state"
        or record["unit"] != "us"
        or record["provider"] not in {"flagdnn", "cudnn"}
    ):
        raise ValueError(
            "invalid NVIDIA benchmark version, kind, unit or provider"
        )
    for name in ("build_us", "warm_build_us"):
        value = record[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    workspace = record["workspace_bytes"]
    if (
        not isinstance(workspace, int)
        or isinstance(workspace, bool)
        or workspace < 0
    ):
        raise ValueError("workspace_bytes must be a nonnegative integer")
    if not isinstance(record["build_cache"], str) or record[
        "build_cache"
    ] not in {
        "fresh_artifact_cache",
        "reuse_allowed",
        "provider_managed",
    }:
        raise ValueError("invalid build cache scope")
    return {
        "timing": {key: record[key] for key in ("median", "p90", "samples")},
        "host_submit_us": record["host_submit_us"],
    }


def postprocess_result(
    *,
    result: dict[str, Any],
    ctest_reported_status: str,
    output: str,
    operator: str,
    suite: str,
    records: dict[str, Any],
    manifest_operators: list[str],
) -> None:
    """Require actual FlagDNN and cuDNN measurements for every case."""
    del operator, manifest_operators
    if suite != "benchmark" or ctest_reported_status != "passed":
        return
    unsupported = dict(
        re.findall(
            r"^\s*(?:\d+:\s*)?([^\n]+): cuDNN UNSUPPORTED: ([^\n]+)$",
            output,
            re.MULTILINE,
        )
    )
    errors = []
    for case, providers in records.items():
        if any("comparison" in record for record in providers.values()):
            errors.append(f"{case}: unpaired benchmark records are forbidden")
        if set(providers) == {"flagdnn", "cudnn"}:
            if case in unsupported:
                errors.append(f"{case}: cuDNN is both timed and unsupported")
        elif set(providers) != {"flagdnn"} or case not in unsupported:
            errors.append(
                f"{case}: missing provider without an unsupported reason"
            )
    for case in unsupported.keys() - records.keys():
        errors.append(
            f"{case}: unsupported reference case has no FlagDNN timing"
        )
    result["reference_unsupported"] = unsupported
    errors.extend(
        f"{case}: cuDNN reference is required" for case in unsupported
    )
    if errors:
        result.setdefault("record_errors", []).extend(errors)
        result["status"] = "failed"


def finalize(
    *,
    results: dict[str, dict[str, Any]],
    suite_operators: dict[str, list[str]],
    suites: list[str],
    state: dict[str, Any],
    min_speedup: float | None,
    preflight_passed: bool,
) -> dict[str, Any]:
    """
    Gate every comparable case, never an operator average or a host timing.
    """
    del state
    selected = [
        (op, suite) for suite in suites for op in suite_operators[suite]
    ]
    passed = sum(
        results.get(op, {}).get(suite, {}).get("status") == "passed"
        for op, suite in selected
    )
    outcome: dict[str, Any] = {
        "failed": not preflight_passed or passed != len(selected),
        "summary": {},
        "coverage": {
            "selected_operator_suite_pairs": len(selected),
            "passed_operator_suite_pairs": passed,
            "complete": passed == len(selected),
        },
    }
    if "benchmark" not in suites:
        return outcome
    comparable = []
    unsupported = []
    incomplete = []
    for operator in suite_operators["benchmark"]:
        task = results.get(operator, {}).get("benchmark", {})
        records = task.get("records", {})
        if task.get("status") != "passed" or not records:
            incomplete.append(
                {
                    "operator": operator,
                    "reason": "benchmark not passed or empty",
                }
            )
            continue
        reasons = task.get("reference_unsupported", {})
        for case, providers in records.items():
            identity = {"operator": operator, "case": case}
            if any("comparison" in record for record in providers.values()):
                incomplete.append({**identity, "reason": "unpaired benchmark"})
                continue
            if set(providers) == {"flagdnn"} and case in reasons:
                unsupported.append({**identity, "reason": reasons[case]})
                continue
            if set(providers) != {"flagdnn", "cudnn"}:
                incomplete.append({**identity, "reason": "missing provider"})
                continue
            a, b = providers["flagdnn"], providers["cudnn"]
            if any(
                r.get("unit") != "us"
                or type(r.get("median")) not in (int, float)
                or not math.isfinite(r["median"])
                or r["median"] <= 0
                for r in (a, b)
            ):
                incomplete.append({**identity, "reason": "invalid GPU timing"})
                continue
            speedup = b["median"] / a["median"]
            if not math.isfinite(speedup):
                incomplete.append({**identity, "reason": "nonfinite speedup"})
                continue
            comparable.append(
                {
                    **identity,
                    "flagdnn_median_us": a["median"],
                    "cudnn_median_us": b["median"],
                    "speedup": speedup,
                }
            )
    comparable.sort(
        key=lambda row: (row["speedup"], row["operator"], row["case"])
    )
    failures = (
        []
        if min_speedup is None
        else [r for r in comparable if r["speedup"] < min_speedup]
    )
    gate_passed = (
        None
        if min_speedup is None
        else bool(comparable)
        and not failures
        and not incomplete
        and not unsupported
    )
    outcome["summary"] = {
        "performance": {
            "metric": SPEEDUP_METRIC,
            "threshold": min_speedup,
            "gate_scope": "comparable_cases_only",
            "gate_passed": gate_passed,
            "case_count": len(comparable),
            "failed_case_count": len(failures),
            "minimum_speedup": (
                comparable[0]["speedup"] if comparable else None
            ),
            "failures": failures,
        },
        "comparable_coverage": {
            "comparable_case_count": len(comparable),
            "native_case_count": 0,
            "unsupported_case_count": len(unsupported),
            "unsupported": unsupported,
            "incomplete": incomplete,
            "complete": not incomplete and not unsupported,
            "all_cases_comparable": not unsupported
            and not incomplete
            and bool(comparable),
        },
    }
    outcome["failed"] |= (
        bool(incomplete) or bool(unsupported) or gate_passed is False
    )
    return outcome
