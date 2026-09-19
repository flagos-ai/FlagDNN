"""Ascend policy hooks for the repository batch-test runner."""

from __future__ import annotations

import re
from typing import Any

DEFAULT_TIMEOUT = 7200
REPORT_DEVICE = "npu"
PREFLIGHT_BY_DEFAULT = False
SUPPORTS_MIN_SPEEDUP = False
FILTER_REGISTERED_TESTS = False


def status_is_success(status: str) -> bool:
    # A reported native capability gap is an expected outcome of --ops all.
    return status in {"passed", "skipped"}


def configure_environment(environment: dict[str, str], device: str | None) -> None:
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
    if set(record) != required_keys:
        missing = sorted(required_keys - set(record))
        extra = sorted(set(record) - required_keys)
        raise ValueError(
            f"fields do not match Ascend schema; missing={missing}; " f"extra={extra}"
        )
    if record.get("schema_version") != 2:
        raise ValueError("Ascend timing record schema_version must be 2")
    provider = record.get("provider")
    if provider not in {"flagdnn", "aclnn"}:
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
    if (
        not isinstance(config, dict)
        or set(config) != config_keys
        or any(
            not isinstance(config[key], int) or isinstance(config[key], bool)
            for key in config_keys
        )
        or config["warmup_iterations"] < 0
        or config["sample_count"] < 1
        or config["iterations_per_sample"] < 1
    ):
        raise ValueError("Ascend benchmark configuration is invalid")
    if not _valid_identity(provider, record.get("provider_identity")):
        raise ValueError("Ascend provider identity is invalid")

    metrics = {
        name: record[name] for name in ("stream_us", "submit_us", "end_to_end_us")
    }
    return metrics


def select_operator_manifests(manifests: dict[str, list[str]]) -> dict[str, list[str]]:
    # Every public operator has a paired benchmark or an explicit native gate.
    return {**manifests, "benchmark": list(manifests["functional"])}


def test_expression(suite: str, operator: str) -> str:
    suffixes = (
        "dtype" if suite == "functional" else "boolean|copy|ieee|tf32|fp32_output"
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
    skips = sorted({
        re.sub(r"^\d+:\s*", "", line).strip()
        for line in output.splitlines()
        if "SKIP case=" in line or "CAPABILITY_UNAVAILABLE:" in line
    })
    if skips:
        result["native_capability_skips"] = skips
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
        if set(providers) != {"flagdnn", "aclnn"}:
            errors.append(f"{case}: both FlagDNN and ACLNN measurements are required")
        for record in providers.values():
            # The legacy summary consumes top-level latency in microseconds.
            # Keep all three original metrics in the native diagnostics.
            record.update(unit="us", **record["stream_us"])
    if errors:
        result.setdefault("record_errors", []).extend(errors)
        result["status"] = "failed"
