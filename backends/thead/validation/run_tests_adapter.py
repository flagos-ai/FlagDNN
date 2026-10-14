"""THead policy hooks for the repository batch-test runner."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import statistics
from typing import Any


COMPARABLE_CASE_CATALOG = (
    Path(__file__).resolve().parent / "benchmark" / "comparable_cases.json"
)
SPEEDUP_METRIC = "acdnn_median_us/flagdnn_median_us"
EQUIVALENT_METRIC = "nvidia_equivalent_time_us/flagdnn_median_us"
PERFORMANCE_BASELINE_FILE = (
    Path(__file__).resolve().parents[3] / "docs" / "performance-baseline.json"
)
EQUIVALENT_SCALE_METRICS = {
    "concatenate": "hbm_copy",
    "gen_index": "hbm_write",
    "gelu_approx_tanh_backward": "vector_fp32",
}
EQUIVALENT_CASE_PATTERN = re.compile(
    r"^(?P<operator>concatenate|gen_index|gelu_approx_tanh_backward)_"
    r"(?:perf_)?(?:fp32|fp16|bfloat16|bf16|int32)_[0-9][a-z0-9_-]*$"
)
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
    r"reference_skipped=(\d+)(?:\s+equivalent_executed=(\d+))?\s*$"
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
# may compile hundreds of shapes with a cold FlagTree cache. Allow the full
# reduction and precision matrices to finish; CTest still bounds each test.
DEFAULT_TIMEOUT = 21600
REPORT_DEVICE = "cuda"
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


def select_operator_manifests(
    manifests: dict[str, list[str]],
) -> dict[str, list[str]]:
    functional = list(manifests["functional"])
    return {
        "functional": functional,
        "benchmark": list(dict.fromkeys(manifests["benchmark"] + functional)),
    }


def dtype_categories(operator: str) -> tuple[str, ...]:
    if operator in {"logical_and", "logical_or", "logical_not"}:
        return ("boolean",)
    if operator in {"identity", "reshape", "transpose", "slice"}:
        return ("copy",)
    if operator in {"matmul", "conv_fprop", "conv_dgrad", "conv_wgrad"}:
        return ("ieee", "tf32")
    if operator == "reduction":
        return ("fp32_output",)
    return ()


def test_expression(suite: str, operator: str) -> str | None:
    categories = dtype_categories(operator) if suite == "benchmark" else ()
    if not categories:
        return None
    choices = "|".join(categories)
    return rf"^benchmark\.thead\.{re.escape(operator)}(\.({choices}))?$"


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
    # The PPU CUDA compatibility stack (libalippu/PyTorch) reads the CUDA
    # mask. Keep the HGGC spelling aligned for tools using the native API.
    # Native tests and JIT code then consistently select logical device 0.
    environment["CUDA_VISIBLE_DEVICES"] = device
    environment["HGGC_VISIBLE_DEVICES"] = device


def configure_batch_environment(
    environment: dict[str, str], build_dir: Path
) -> None:
    """Reuse native artifacts across suites, GPU workers, and batch runs."""
    if not environment.get("FLAGDNN_CACHE_PATH"):
        environment["FLAGDNN_CACHE_PATH"] = str(
            build_dir / "cache" / "run_tests"
        )


def preflight_tests(_suites: list[str] | tuple[str, ...]) -> set[str]:
    """Return contracts that must exist before an operator test may run."""
    required = {
        "integration.thead.validation_contract",
        "integration.thead.capability_contract",
        "integration.thead.acdnn_gap_contract",
        "integration.thead.acdnn_fp8_codec_contract",
        "integration.thead.catalog_closure_contract",
        "integration.thead.factory_coverage_contract",
        "integration.thead.cmake_configuration_contract",
        "integration.thead.dependency_boundary",
        "integration.thead.reference_dependency_boundary",
        "integration.thead.artifact_contract",
        "integration.thead.index_contract",
        "integration.thead.index_runtime_contract",
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
    if (
        case.startswith("matmul_mxfp8_")
        or re.match(r"^matmul_ae[45]m[23]_", case)
    ) and "matmul_fp8" in manifest_operators:
        return "matmul_fp8"
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
            benchmark.get("records", {}) if isinstance(benchmark, dict) else {}
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


def equivalent_case_operator(case: str) -> str | None:
    """Return only the recorded dtype/catalog scope of the three policies."""
    match = EQUIVALENT_CASE_PATTERN.fullmatch(case)
    if match is None:
        return None
    operation = match.group("operator")
    patterns = {
        "concatenate": r"concatenate_(?:fp32|fp16|bf16)_",
        "gen_index": r"gen_index_(?:fp32|int32)_",
        "gelu_approx_tanh_backward": (
            r"gelu_approx_tanh_backward_perf_(?:fp32|fp16|bfloat16)_"
        ),
    }
    return operation if re.match(patterns[operation], case) else None


def thead_equivalent_coverage(
    results: dict[str, dict[str, Any]],
    selected_benchmark_operators: list[str],
    catalog: dict[str, Any],
) -> dict[str, Any]:
    """Require the whole selected equivalent catalog, independent of timing.

    Unmeasured concatenate dtypes are expected to emit explicit skip evidence;
    they never become required measured cases or fabricated passing records.
    """
    selected = [
        operator for operator in dict.fromkeys(selected_benchmark_operators)
        if operator in EQUIVALENT_SCALE_METRICS
    ]
    required: list[dict[str, str]] = []
    unavailable: list[dict[str, str]] = []
    for case, capability in catalog["cases"].items():
        if capability["status"] != "unsupported":
            continue
        operator = benchmark_case_operator(case, selected)
        if operator not in selected:
            continue
        item = {"operator": operator, "case": case}
        if equivalent_case_operator(case) == operator:
            required.append(item)
        elif operator == "concatenate":
            unavailable.append(item)
    required.sort(key=lambda item: (item["operator"], item["case"]))
    unavailable.sort(key=lambda item: item["case"])
    missing: list[dict[str, str]] = []
    missing_skips: list[dict[str, str]] = []
    unexpected: list[dict[str, str]] = []
    for item in required:
        benchmark = results.get(item["operator"], {}).get("benchmark", {})
        providers = benchmark.get("records", {}).get(item["case"], {})
        if (
            benchmark.get("status") == "passed"
            and set(providers) == {"flagdnn"}
            and providers["flagdnn"].get("schema_version") == 4
        ):
            continue
        missing.append({
            **item, "benchmark_status": str(benchmark.get("status", "not_run"))
        })
    for item in unavailable:
        benchmark = results.get(item["operator"], {}).get("benchmark", {})
        observed = {
            record.get("case") for record in benchmark.get("baseline_skip_records", [])
            if record.get("op") == item["operator"]
        }
        if item["case"] not in observed:
            missing_skips.append(item)
    required_names = {(item["operator"], item["case"]) for item in required}
    for operator in selected:
        for case, providers in results.get(operator, {}).get("benchmark", {}).get("records", {}).items():
            if (
                set(providers) == {"flagdnn"}
                and providers["flagdnn"].get("schema_version") == 4
                and (operator, case) not in required_names
            ):
                unexpected.append({"operator": operator, "case": case})
    missing_operators = [operator for operator in selected if operator not in catalog["operators"]]
    return {
        "source": "repository-owned THead benchmark catalog and explicit equivalent scope",
        "selected_operators": selected,
        "required_cases": required,
        "required_case_count": len(required),
        "observed_required_case_count": len(required) - len(missing),
        "missing_cases": missing,
        "missing_case_count": len(missing),
        "expected_reference_skip_cases": unavailable,
        "expected_reference_skip_case_count": len(unavailable),
        "observed_reference_skip_case_count": len(unavailable) - len(missing_skips),
        "missing_reference_skip_cases": missing_skips,
        "unexpected_cases": unexpected,
        "missing_operators": missing_operators,
        "verified": not (missing or missing_skips or unexpected or missing_operators),
    }


def validate_benchmark_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate measured FlagDNN samples and an unsampled NVIDIA estimate."""
    expected = {
        "schema_version", "kind", "provider", "case", "unit", "median",
        "p90", "samples", "equivalent_baseline",
    }
    if not isinstance(record, dict) or set(record) != expected:
        raise ValueError("fields do not match THead benchmark schema v4")
    case = record["case"]
    match = (
        EQUIVALENT_CASE_PATTERN.fullmatch(case)
        if isinstance(case, str) else None
    )
    if (
        type(record["schema_version"]) is not int
        or record["schema_version"] != 4
        or record["kind"] != "steady_state"
        or record["provider"] != "flagdnn"
        or record["unit"] != "us"
        or match is None
    ):
        raise ValueError(
            "schema v4 requires a supported measured FlagDNN case"
        )
    operation = match.group("operator")
    if equivalent_case_operator(case) != operation:
        raise ValueError("case has no qualified NVIDIA equivalent baseline")
    baseline = record["equivalent_baseline"]
    identities = {
        "provider": "nvidia_equivalent",
        "platform": "nvidia",
        "target_platform": "thead",
        "timing_profile": "native",
        "scale_metric": EQUIVALENT_SCALE_METRICS[operation],
        "reference_validation": (
            "cpu_full"
            if operation == "gelu_approx_tanh_backward"
            else "sanity"
        ),
    }
    numeric = {
        "reference_time_us", "reference_peak", "target_peak",
        "min_relative_efficiency", "equivalent_time_us", "limit_us",
        "relative_efficiency", "traffic_ratio",
    }
    if (
        not isinstance(baseline, dict)
        or set(baseline) != set(identities) | numeric | {"passed"}
        or any(baseline[key] != value for key, value in identities.items())
    ):
        raise ValueError("invalid equivalent-baseline fields or identity")

    def positive(value: Any) -> bool:
        try:
            return (
                type(value) in (int, float)
                and math.isfinite(value)
                and value > 0
            )
        except OverflowError:
            return False

    samples = record["samples"]
    if (
        any(not positive(baseline[key]) for key in numeric)
        or not positive(record["median"])
        or not positive(record["p90"])
        or not isinstance(samples, list)
        or len(samples) != 20
        or any(not positive(sample) for sample in samples)
    ):
        raise ValueError(
            "native equivalent timing requires 20 finite positive samples"
        )
    ordered = sorted(samples)
    if record["median"] != ordered[9] or record["p90"] != ordered[17]:
        raise ValueError(
            "equivalent timing quantiles disagree with measured samples"
        )
    if (
        baseline["min_relative_efficiency"] != 0.8
        or baseline["traffic_ratio"] != 1
    ):
        raise ValueError(
            "THead baseline requires minimum 0.8 and unit logical traffic"
        )
    try:
        equivalent = (
            baseline["reference_time_us"] * baseline["reference_peak"]
            / baseline["target_peak"] * baseline["traffic_ratio"]
        )
        derived = {
            "equivalent_time_us": equivalent,
            "limit_us": equivalent / baseline["min_relative_efficiency"],
            "relative_efficiency": equivalent / record["median"],
        }
    except OverflowError as error:
        raise ValueError("equivalent-baseline formula overflowed") from error
    for key, value in derived.items():
        if not positive(value) or not math.isclose(
            baseline[key], value, rel_tol=1.0e-12, abs_tol=0.0
        ):
            raise ValueError(
                f"equivalent-baseline {key} does not match formula"
            )
    if type(baseline["passed"]) is not bool or baseline["passed"] != (
        record["median"] <= baseline["limit_us"]
    ):
        raise ValueError(
            "equivalent-baseline passed disagrees with measured limit"
        )
    return {
        "timing": {key: record[key] for key in ("median", "p90", "samples")}
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
        if (
            actual == {"flagdnn"}
            and providers["flagdnn"].get("schema_version") == 4
        ):
            try:
                validate_benchmark_record(providers["flagdnn"])
                if providers["flagdnn"]["case"] != case:
                    raise ValueError("record case differs from its group")
            except ValueError as error:
                errors.append(
                    f"THead equivalent benchmark case={case}: {error}"
                )
            continue
        if actual != expected or any(
            record.get("schema_version") == 4 for record in providers.values()
        ):
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
        math.exp(
            sum(math.log(speedup) for speedup in speedups) / len(speedups)
        )
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


def equivalent_performance_summary(
    results: dict[str, dict[str, Any]],
    threshold: float | None,
) -> dict[str, Any]:
    """Keep peak-normalized efficiency separate from native acDNN speedup."""
    cases: list[dict[str, Any]] = []
    for operator, operator_results in results.items():
        benchmark = operator_results.get("benchmark")
        if (
            not isinstance(benchmark, dict)
            or benchmark.get("status") != "passed"
        ):
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
            required = (
                max(minimum, threshold) if threshold is not None else minimum
            )
            passed = baseline["passed"] and (
                required == minimum
                or baseline["relative_efficiency"] >= required
            )
            cases.append(
                {
                    "operator": operator,
                    "case": case,
                    "flagdnn_median_us": record["median"],
                    "equivalent_baseline": dict(baseline),
                    "relative_efficiency": baseline["relative_efficiency"],
                    "required_relative_efficiency": required,
                    "passed": passed,
                }
            )
    cases.sort(
        key=lambda record: (record["relative_efficiency"], record["case"])
    )
    failures = [record for record in cases if not record["passed"]]
    skips = [
        dict(record)
        for operator_results in results.values()
        for record in operator_results.get("benchmark", {}).get(
            "baseline_skip_records", []
        )
    ]
    return {
        "metric": EQUIVALENT_METRIC,
        "reference_skipped_case_count": len(skips),
        "reference_skips": skips,
        "min_speedup_semantics": (
            "opt-in relative_efficiency gate; cannot relax baseline minimum"
        ),
        "requested_min_speedup": threshold,
        "gate_enabled": threshold is not None,
        "gate_passed": (
            not failures if threshold is not None and cases else None
        ),
        "case_count": len(cases),
        "passed_case_count": len(cases) - len(failures),
        "failed_case_count": len(failures),
        "minimum_relative_efficiency": (
            cases[0]["relative_efficiency"] if cases else None
        ),
        "cases": cases,
        "failures": failures,
    }


def apply_equivalent_report_gates(
    results: dict[str, Any], equivalent: dict[str, Any]
) -> None:
    """Keep the measured baseline verdict distinct from the final CLI gate."""
    gates = {
        (record["operator"], record["case"]): record
        for record in equivalent["cases"]
    }
    enabled = equivalent.get("gate_enabled", False)
    coverage = equivalent.get("coverage") or {}
    incomplete = {}
    for field in ("missing_cases", "missing_reference_skip_cases", "unexpected_cases"):
        for item in coverage.get(field, []):
            incomplete.setdefault(item["operator"], []).append({"kind": field, **item})
    for operator in coverage.get("missing_operators", []):
        incomplete.setdefault(operator, []).append({"kind": "missing_operator"})
    for operator, result in results.items():
        performance = result.get("performance", {})
        if operator in incomplete:
            performance["coverage_errors"] = incomplete[operator]
            # Preserve an existing timeout or execution failure and its code.
            # Coverage must only turn an otherwise successful report into failure.
            if performance.get("status") in {"Passed", "Skipped", "NotFound"}:
                performance["status"] = "Failed"
                performance["exit_code"] = 1
        skips = [
            record for record in equivalent.get("reference_skips", [])
            if record.get("op") == operator
        ]
        if skips:
            performance["skipped"] = len(skips)
            performance["skip_records"] = skips
            if performance.get("status") == "Passed":
                performance["status"] = "Skipped"
        for entry in performance.get("data", {}).values():
            for case, row in entry.get("details", {}).items():
                gate = gates.get((operator, case))
                if gate is None:
                    continue
                row["performance_gate"] = {
                    "enabled": enabled,
                    "metric": equivalent["metric"],
                    "relative_efficiency": gate["relative_efficiency"],
                    "required_relative_efficiency": (
                        gate["required_relative_efficiency"]
                        if enabled
                        else None
                    ),
                    "passed": gate["passed"] if enabled else None,
                }
                if enabled and not gate["passed"]:
                    entry["result"] = "FAIL"
                    performance["status"] = "Failed"
                    performance["exit_code"] = 1


def postprocess_report(
    *,
    report: dict[str, Any],
    platform_summary: dict[str, Any],
    exit_code: int | None,
) -> bool:
    """Apply the finalized platform gate to a public single/merged report."""
    equivalent = platform_summary.get("equivalent_performance")
    if not isinstance(equivalent, dict) or not (
        equivalent.get("case_count")
        or equivalent.get("reference_skipped_case_count")
        or (equivalent.get("coverage") or {}).get("verified") is False
    ):
        return False
    apply_equivalent_report_gates(report["result"], equivalent)
    report.update(
        equivalent_performance=equivalent,
        overall_status="passed" if exit_code == 0 else "failed",
        exit_code=exit_code,
    )
    return True


def postprocess_batch_report(
    *,
    partitions: dict[int, dict[str, Any]],
    report: dict[str, Any],
    platform_summary: dict[str, Any],
    exit_code: int | None,
) -> bool:
    """Apply final gates to every worker partition and the merged report."""
    changed = postprocess_report(
        report=report, platform_summary=platform_summary, exit_code=exit_code
    )
    if changed:
        for partition in partitions.values():
            apply_equivalent_report_gates(
                partition, platform_summary["equivalent_performance"]
            )
    return changed


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


def nvidia_baseline_skip_records(output: str) -> list[dict[str, str]]:
    lines = [
        line.replace("[SKIP][nvidia_baseline]", "[SKIP][acdnn]")
        for line in output.splitlines()
        if _ctest_line(line).startswith("[SKIP][nvidia_baseline]")
    ]
    records = acdnn_skip_records("\n".join(lines))
    for record in records:
        record["message"] = record["message"].replace(
            "[SKIP][acdnn]", "[SKIP][nvidia_baseline]"
        )
    return records


def validate_nvidia_baseline_skips(records: list[dict[str, str]]) -> list[str]:
    try:
        path = Path(
            os.environ.get("FLAGDNN_PERFORMANCE_BASELINE")
            or PERFORMANCE_BASELINE_FILE
        )
        reference = json.loads(path.read_text())["reference"]
        measured = reference["cases"]
        unavailable = reference["unavailable_cases"]
        if not isinstance(measured, dict) or not isinstance(unavailable, dict):
            raise ValueError("reference case catalogs must be objects")
    except (OSError, ValueError, KeyError, TypeError) as error:
        return [f"Cannot verify NVIDIA baseline skip evidence: {error}"]
    errors = []
    for record in records:
        case = record.get("case", "")
        exclusion = unavailable.get(case, {})
        if (
            record.get("op") != "concatenate"
            or record.get("reason") != "nvidia_reference_unavailable"
            or case in measured
            or not isinstance(exclusion, dict)
            or exclusion.get("op") != "concatenate"
            or exclusion.get("status") != "excluded"
        ):
            errors.append(
                f"NVIDIA baseline skip is not a known unmeasured case: {case}"
            )
    return errors


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
    allowed_markers = {expected_marker}
    if suite == "benchmark":
        allowed_markers.update(
            f"FLAGDNN_{operator.upper()}_{category.upper()}_BENCHMARK"
            for category in dtype_categories(operator)
        )
    markers = [marker for marker, _ in marker_records]
    if not markers or len(markers) != len(set(markers)):
        return None, [
            "THead suite must emit one accounting record per unique test"
        ]
    errors: list[str] = []
    pattern = (
        FUNCTIONAL_ACCOUNTING_PATTERN
        if suite == "functional"
        else BENCHMARK_ACCOUNTING_PATTERN
    )
    cases = executed = skipped = equivalent_executed = 0
    reports_equivalent = False
    statuses = []
    for marker, payload in marker_records:
        if marker not in allowed_markers:
            errors.append(f"unexpected THead accounting marker: {marker}")
        match = pattern.fullmatch(payload)
        if match is None:
            errors.append(
                f"THead suite accounting payload for marker={marker} is"
                " malformed"
            )
            continue
        status, count, ran, omitted = match.groups()[:4]
        if suite == "benchmark" and match.group(5) is not None:
            reports_equivalent = True
            equivalent_executed += int(match.group(5))
        count, ran, omitted = int(count), int(ran), int(omitted)
        if (
            count <= 0
            or count != ran + omitted
            or (status == "PASS") != (ran > 0)
        ):
            errors.append(f"THead suite accounting is inconsistent: {marker}")
        statuses.append(status)
        cases += count
        executed += ran
        skipped += omitted
    reported = "PASS" if executed else "SKIP"
    accounting: dict[str, Any] = {"status": reported, "cases": cases}
    if suite == "functional":
        accounting.update({"executed": executed, "skipped": skipped})
    else:
        accounting.update(
            {"comparable_executed": executed, "reference_skipped": skipped}
        )
    if suite == "benchmark" and reports_equivalent:
        accounting["equivalent_executed"] = equivalent_executed
        measured_equivalent = sum(
            set(providers) == {"flagdnn"}
            and providers["flagdnn"].get("schema_version") == 4
            for providers in records.values()
        )
        if equivalent_executed != measured_equivalent:
            errors.append(
                "THead equivalent_executed does not match measured records"
            )
    expected_ctest = "skipped" if "SKIP" in statuses else "passed"
    if ctest_reported_status != expected_ctest:
        errors.append(
            f"THead accounting requires CTest status={expected_ctest}"
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
    acdnn_skips = acdnn_skip_records(output)
    baseline_skips = nvidia_baseline_skip_records(output)
    skip_records = acdnn_skips + baseline_skips
    if acdnn_skips:
        result["skip_records"] = acdnn_skips
    if baseline_skips:
        result["baseline_skip_records"] = baseline_skips
    if ctest_reported_status == "skipped" or skip_records:
        skip_errors = validate_thead_skip_records(
            skip_records, operator, manifest_operators
        )
        if baseline_skips:
            if suite != "benchmark":
                skip_errors.append(
                    "NVIDIA baseline skips require a benchmark suite"
                )
            skip_errors.extend(validate_nvidia_baseline_skips(baseline_skips))
        if skip_errors:
            result["skip_record_errors"] = skip_errors
            result["status"] = "failed"
    if (
        suite == "benchmark"
        and ctest_reported_status in {"passed", "skipped"}
        and (ctest_reported_status == "passed" or records)
    ):
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
        elif accounting is not None and accounting["status"] == "PASS":
            # The shared runner treats any skipped CTest category as an
            # entirely skipped operator and rejects its timing records.
            # Remove only that false positive after validating all pairs,
            # structured skips, and aggregate execution counts above.
            if (
                suite == "benchmark"
                and ctest_reported_status == "skipped"
                and records
                and not result.get("skip_record_errors")
            ):
                errors = result.get("record_errors", [])
                if errors and all(
                    error
                    == "skipped benchmark emitted timing provider records"
                    for error in errors
                ):
                    result.pop("record_errors")
                    result["status"] = "passed"
            if result["status"] == "skipped":
                result["status"] = "passed"


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
    failed = any(
        not status_is_success(task.get("status", "error"))
        for operator_results in results.values()
        for task in operator_results.values()
    )
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

    equivalent_coverage = None
    if catalog is not None and preflight_passed and "benchmark" in suites:
        equivalent_coverage = thead_equivalent_coverage(
            results, suite_operators["benchmark"], catalog
        )
        if not equivalent_coverage["verified"]:
            print(
                "equivalent coverage gate: "
                f"{equivalent_coverage['observed_required_case_count']}/"
                f"{equivalent_coverage['required_case_count']} required measurements; "
                f"{equivalent_coverage['observed_reference_skip_case_count']}/"
                f"{equivalent_coverage['expected_reference_skip_case_count']} "
                "unavailable references explicitly reported",
                flush=True,
            )
            for field in ("missing_cases", "missing_reference_skip_cases", "unexpected_cases"):
                for item in equivalent_coverage[field]:
                    print(f"  FAIL {field} op={item['operator']} case={item['case']}", flush=True)
            for operator in equivalent_coverage["missing_operators"]:
                print(f"  FAIL missing equivalent catalog operator={operator}", flush=True)
            failed = True

    performance = (
        benchmark_speedup_summary(results, min_speedup, comparable_coverage)
        if "benchmark" in suites
        else None
    )
    equivalent = (
        equivalent_performance_summary(results, min_speedup)
        if "benchmark" in suites else None
    )
    if equivalent is not None:
        equivalent["coverage"] = equivalent_coverage
    has_equivalent = equivalent is not None and equivalent["case_count"] > 0
    if (
        performance is not None
        and not performance["case_count"]
        and has_equivalent
    ):
        performance["gate_passed"] = None
        performance["ratio_gate_passed"] = None
    if min_speedup is not None and (
        performance is not None
        and (performance["case_count"] or not has_equivalent)
    ):
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

    if has_equivalent:
        assert equivalent is not None
        enabled = equivalent["gate_enabled"]
        print(
            "equivalent performance "
            + ("gate: " if enabled else "report (informational): ")
            + f"{equivalent['passed_case_count']}/{equivalent['case_count']} "
            "cases meet required relative efficiency",
            flush=True,
        )
        for record in equivalent["failures"]:
            print(
                f"  {'FAIL' if enabled else 'BELOW_BASELINE'} "
                f"op={record['operator']} case={record['case']} "
                f"flagdnn_us={record['flagdnn_median_us']:.9g} "
                f"relative_efficiency={record['relative_efficiency']:.9g} "
                f"required={record['required_relative_efficiency']:.9g}",
                flush=True,
            )
        if enabled:
            failed = failed or not equivalent["gate_passed"]

    skip_record_errors = 0
    accounting_errors = 0
    reference_skips = 0
    baseline_skips = 0
    for operator_results in results.values():
        for result in operator_results.values():
            skip_record_errors += len(result.get("skip_record_errors", []))
            accounting_errors += len(result.get("case_accounting_errors", []))
            reference_skips += len(result.get("skip_records", []))
            baseline_skips += len(result.get("baseline_skip_records", []))
    failed = failed or skip_record_errors > 0 or accounting_errors > 0

    return {
        "failed": failed,
        "summary": {
            "comparable_coverage": comparable_coverage,
            "performance": performance,
            "equivalent_performance": equivalent,
        },
        "coverage": {
            "acdnn_reference_skips": reference_skips,
            "nvidia_baseline_skips": baseline_skips,
            "acdnn_skip_record_errors": skip_record_errors,
            "thead_case_accounting_errors": accounting_errors,
        },
    }
