"""Ascend policy hooks for the repository batch-test runner."""

from __future__ import annotations

import re
import math
from typing import Any

DEFAULT_TIMEOUT = 7200
REPORT_DEVICE = "npu"
PREFLIGHT_BY_DEFAULT = False
SUPPORTS_MIN_SPEEDUP = False
FILTER_REGISTERED_TESTS = False


def status_is_success(status: str) -> bool:
    # A reported native capability gap is an expected outcome of --ops all.
    return status in {"passed", "skipped"}


def configure_environment(
    environment: dict[str, str], device: str | None
) -> None:
    if device is None:
        return
    environment["ASCEND_RT_VISIBLE_DEVICES"] = device
    environment["NPU_VISIBLE_DEVICES"] = device


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_identity(provider: str, identity: Any) -> bool:
    if not isinstance(identity, dict):
        return False
    if provider == "flagdnn":
        keys = {
            "libtriton_jit_sha256",
            "compiler_identity_sha256",
            "artifact_request_sha256",
            "launch_abi",
            "selected_candidate",
        }
        return (
            set(identity) == keys
            and all(
                _sha256(identity[key])
                for key in (
                    "libtriton_jit_sha256",
                    "compiler_identity_sha256",
                    "artifact_request_sha256",
                )
            )
            and identity["launch_abi"] == "ltj_npu_raw_v1"
            and isinstance(identity["selected_candidate"], str)
            and bool(identity["selected_candidate"])
        )
    if provider == "aclnn":
        required = {"libnnopbase_sha256", "libopapi_math_sha256"}
        allowed = required | {"libopapi_nn_sha256"}
        return required <= set(identity) <= allowed and all(
            _sha256(value) for value in identity.values()
        )
    return False


def _positive_number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and value > 0
    except OverflowError:
        # JSON integers have arbitrary precision; a malformed timing record
        # must produce a record error rather than abort the whole batch.
        return False


def _validate_metric(metric: Any, sample_count: int) -> None:
    if (
        not isinstance(metric, dict)
        or set(metric) != {"median", "p90", "samples"}
        or not isinstance(metric["samples"], list)
        or len(metric["samples"]) != sample_count
        or not all(_positive_number(metric[key]) for key in ("median", "p90"))
        or not all(_positive_number(value) for value in metric["samples"])
    ):
        raise ValueError(
            "Ascend timing metric or configured sample count is invalid"
        )
    ordered = sorted(metric["samples"])
    for key, numerator, denominator in (("median", 1, 2), ("p90", 9, 10)):
        index = (sample_count * numerator + denominator - 1) // denominator - 1
        if not math.isclose(
            metric[key], ordered[index], rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(
                f"Ascend {key} differs from nearest-rank sample statistic"
            )


def validate_benchmark_record(
    record: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    required_keys = {
        "schema_version",
        "kind",
        "provider",
        "case",
        "environment",
        "provider_identity",
        "benchmark_config",
        "stream_us",
        "submit_us",
        "end_to_end_us",
    }
    equivalent = record.get("schema_version") == 6
    if equivalent:
        required_keys.add("equivalent_baseline")
    if set(record) != required_keys:
        missing = sorted(required_keys - set(record))
        extra = sorted(set(record) - required_keys)
        raise ValueError(
            f"fields do not match Ascend schema; missing={missing}; "
            f"extra={extra}"
        )
    if type(record.get("schema_version")) is not int or record[
        "schema_version"
    ] not in (2, 6):
        raise ValueError("Ascend timing record schema_version must be 2 or 6")
    if (
        record["kind"] != "steady_state"
        or not isinstance(record["case"], str)
        or not record["case"]
    ):
        raise ValueError("Ascend timing kind or case is invalid")
    provider = record.get("provider")
    if not isinstance(provider, str) or provider not in {"flagdnn", "aclnn"}:
        raise ValueError("Ascend timing provider must be flagdnn or aclnn")

    environment = record.get("environment")
    environment_keys = {
        "soc_fingerprint",
        "cann_package_version",
        "ascendcl_build_id",
        "runtime_build_id",
    }
    if (
        not isinstance(environment, dict)
        or set(environment) != environment_keys
        or any(
            not isinstance(environment[key], str) or not environment[key]
            for key in environment_keys
        )
    ):
        raise ValueError("Ascend timing environment is invalid")

    config = record.get("benchmark_config")
    config_keys = {
        "warmup_iterations",
        "sample_count",
        "iterations_per_sample",
    }
    if equivalent:
        config_keys |= {"timing_method", "graph_warmup_replays", "statistic"}
    if (
        not isinstance(config, dict)
        or set(config) != config_keys
        or any(
            not isinstance(config[key], int) or isinstance(config[key], bool)
            for key in (
                "warmup_iterations",
                "sample_count",
                "iterations_per_sample",
            )
        )
        or config["warmup_iterations"] < 0
        or config["sample_count"] < 1
        or config["iterations_per_sample"] < 1
    ):
        raise ValueError("Ascend benchmark configuration is invalid")
    if not _valid_identity(provider, record.get("provider_identity")):
        raise ValueError("Ascend provider identity is invalid")
    metrics = {
        name: record[name]
        for name in ("stream_us", "submit_us", "end_to_end_us")
    }
    for metric in metrics.values():
        _validate_metric(metric, config["sample_count"])
    if equivalent:
        _validate_equivalent(record)
    return metrics


def _validate_equivalent(record: dict[str, Any]) -> None:
    if record["provider"] != "flagdnn":
        raise ValueError(
            "equivalent timing requires a real FlagDNN measurement"
        )
    if type(
        record["benchmark_config"].get("graph_warmup_replays")
    ) is not int or record["benchmark_config"] != {
        "warmup_iterations": 10,
        "sample_count": 20,
        "iterations_per_sample": 50,
        "timing_method": "device_events_over_ascend_graph",
        "graph_warmup_replays": 1,
        "statistic": "nearest_rank_p50",
    }:
        raise ValueError("equivalent timing does not match the NVIDIA profile")
    baseline = record["equivalent_baseline"]
    identities = {
        "provider": "nvidia_equivalent",
        "reference_platform": "nvidia",
        "target_platform": "ascend",
        "reference_case": record["case"],
        "scale_metric": "hbm_copy",
        "traffic_ratio": 1,
        "reference_validation": "bit_exact",
        "timing_profile": "native",
    }
    numeric = {
        "reference_time_us",
        "reference_peak",
        "target_peak",
        "equivalent_time_us",
        "limit_us",
        "relative_efficiency",
        "min_relative_efficiency",
    }
    if (
        not isinstance(baseline, dict)
        or set(baseline) != set(identities) | numeric | {"passed"}
        or type(baseline.get("traffic_ratio")) not in (int, float)
        or any(baseline[k] != v for k, v in identities.items())
    ):
        raise ValueError("invalid Ascend equivalent baseline identity")
    if not re.fullmatch(r"identity_fp8_(?:e4m3|e5m2)_.+", record["case"]):
        raise ValueError("case is outside Ascend equivalent baseline scope")
    measured = record["stream_us"].get("median")
    if (
        not all(_positive_number(baseline[k]) for k in numeric)
        or not 0 < baseline["min_relative_efficiency"] <= 1
    ):
        raise ValueError(
            "equivalent baseline values must be finite and positive"
        )
    # Match the C++ double arithmetic even when JSON numbers use integer
    # syntax.
    expected = (
        float(baseline["reference_time_us"])
        * float(baseline["reference_peak"])
        / float(baseline["target_peak"])
    )
    expected_limit = expected / baseline["min_relative_efficiency"]
    for key, value in {
        "equivalent_time_us": expected,
        "limit_us": expected_limit,
        "relative_efficiency": expected / measured,
    }.items():
        if not math.isfinite(value) or not math.isclose(
            baseline[key], value, rel_tol=1e-8
        ):
            raise ValueError(
                f"equivalent baseline {key} does not match formula"
            )
    if (
        type(baseline["passed"]) is not bool
        or baseline["passed"] != (measured <= baseline["limit_us"])
        or baseline["passed"] != (measured <= expected_limit)
    ):
        raise ValueError(
            "equivalent baseline verdict disagrees with measured limit"
        )


def select_operator_manifests(
    manifests: dict[str, list[str]]
) -> dict[str, list[str]]:
    # Every public operator has a paired benchmark or an explicit native gate.
    return {**manifests, "benchmark": list(manifests["functional"])}


def test_expression(suite: str, operator: str) -> str:
    suffixes = (
        "dtype"
        if suite == "functional"
        else "boolean|copy|ieee|tf32|fp32_output"
    )
    return rf"^{suite}\.ascend\.{re.escape(operator)}(\.({suffixes}))?$"


def postprocess_result(
    *,
    result,
    ctest_reported_status,
    output,
    operator,
    suite,
    records,
    manifest_operators,
):
    del operator, manifest_operators
    # Retain reasons even when CTest succeeds with some unsupported dtypes.
    # The native sidecar remains auditable without enabling verbose output.
    skips = sorted(
        {
            re.sub(r"^\d+:\s*", "", line).strip()
            for line in output.splitlines()
            if "SKIP case=" in line or "CAPABILITY_UNAVAILABLE:" in line
        }
    )
    if skips:
        result["native_capability_skips"] = skips
    storage_views = sorted(
        {
            re.sub(r"^\d+:\s*", "", line).strip()
            for line in output.splitlines()
            if "ACLNN_STORAGE_VIEW case=" in line
        }
    )
    if storage_views:
        result["reference_storage_views"] = storage_views
    # CTest marks an aggregate skipped if one dtype/precision supplement was
    # skipped. Preserve successful supported cases alongside explicit skips.
    passed_test = re.search(r"\d+/\d+ Test\s+#\d+:.*?\.\.\.\s+Passed", output)
    if ctest_reported_status == "skipped" and passed_test:
        errors = result.get("record_errors", [])
        errors = [
            e
            for e in errors
            if e != "skipped benchmark emitted timing provider records"
        ]
        if errors:
            result["record_errors"] = errors
        else:
            result.pop("record_errors", None)
            result["status"] = "passed"
    if suite != "benchmark":
        return
    errors = []
    for case, providers in records.items():
        equivalent = providers.get("flagdnn", {}).get("equivalent_baseline")
        if equivalent is not None:
            if set(providers) != {"flagdnn"}:
                errors.append(
                    (
                        f"{case}: equivalent baseline cannot include "
                        f"fabricated reference samples"
                    )
                )
        elif set(providers) != {"flagdnn", "aclnn"}:
            errors.append(
                f"{case}: both FlagDNN and ACLNN measurements are required"
            )
        for record in providers.values():
            # The legacy summary consumes top-level latency in microseconds.
            # Keep all three original metrics in the native diagnostics.
            record.update(unit="us", **record["stream_us"])
    equivalent_cases = {
        case: providers["flagdnn"]["equivalent_baseline"]
        for case, providers in records.items()
        if "equivalent_baseline" in providers.get("flagdnn", {})
    }
    if equivalent_cases:
        result["equivalent_performance"] = {
            "case_count": len(equivalent_cases),
            "passed_case_count": sum(
                b["passed"] for b in equivalent_cases.values()
            ),
            "gate_enabled": False,
            "gate_passed": None,
        }
    if errors:
        result.setdefault("record_errors", []).extend(errors)
        result["status"] = "failed"
