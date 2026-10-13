"""Hygon policy hooks for the repository batch-test runner."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
from typing import Any


COMPARABLE_CASE_CATALOG = (
    Path(__file__).resolve().parent / "benchmark" / "comparable_cases.json"
)
PERFORMANCE_BASELINE_FILE = (
    Path(__file__).resolve().parents[3] / "docs" / "performance-baseline.json"
)
SPEEDUP_METRIC = "hipdnn_median_us/flagdnn_median_us"
EQUIVALENT_METRIC = "nvidia_equivalent_time_us/flagdnn_median_us"
EQUIVALENT_GENERAL_OPERATORS = frozenset(
    "reshape transpose slice concatenate gen_index div pow mod "
    "cmp_eq cmp_neq cmp_gt cmp_ge cmp_lt cmp_le "
    "logical_and logical_or logical_not binary_select "
    "neg sqrt rsqrt reciprocal ceil floor exp log erf sin cos tan "
    "leaky_relu leaky_relu_backward elu_backward gelu gelu_backward "
    "gelu_approx_tanh gelu_approx_tanh_backward "
    "softplus softplus_backward".split()
)
EQUIVALENT_BFLOAT16_OPERATORS = frozenset(
    "abs add add_square elu max min mul relu relu_backward scale sigmoid "
    "sigmoid_backward sub swish swish_backward tanh tanh_backward".split()
)
EQUIVALENT_OPERATORS = (
    EQUIVALENT_GENERAL_OPERATORS
    | EQUIVALENT_BFLOAT16_OPERATORS
    | {"identity", "batchnorm", "reduction"}
)
EQUIVALENT_CASE_PATTERN = re.compile(
    # New scopes follow the measured supplement, not the complete dtype range
    # of an operator. In particular, native FP32/FP16 pointwise paths retain
    # their existing provider pairing and do not gain an unproven fallback.
    r"^(?=(?P<operator>"
    + "|".join(sorted(EQUIVALENT_OPERATORS, key=lambda op: (-len(op), op)))
    + r")_)(?:identity_(?:fp8_(?:e4m3|e5m2)|bfloat16|int32|bool)_"
    r"|identity_perf_bfloat16_|batchnorm_perf_(?:fp16|bfloat16)_"
    r"|reduction_(?:sum_bfloat16|(?:sum|avg|mul)_fp16)_to_fp32_|(?:"
    + "|".join(
        sorted(EQUIVALENT_BFLOAT16_OPERATORS, key=lambda op: (-len(op), op))
    )
    + r")_perf_bfloat16_|(?:"
    + "|".join(
        sorted(EQUIVALENT_GENERAL_OPERATORS, key=lambda op: (-len(op), op))
    )
    + r")_(?:(?:perf|attributes)_)?"
    r"(?:fp32|fp16|bfloat16|bf16|bool|int32)_"
    r"(?:case[0-9]+_)?)[0-9][a-z0-9_-]*$"
)
EQUIVALENT_SCALE_METRICS = {
    "vector_fp32",
    "hbm_copy",
    "hbm_write",
    "hbm_triad",
}
REFERENCE_VALIDATIONS = {"cpu_full", "cpu_sampled", "bit_exact", "sanity"}
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


def preflight_tests(suites: list[str] | tuple[str, ...]) -> set[str]:
    return {
        "integration.hygon.validation_contract",
        "integration.hygon.tensor_reference_contract",
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
    } | (
        {
            "integration.hygon.div_performance_baseline",
            "integration.hygon.benchmark_validation_guards",
            "integration.hygon.benchmark_timing_policy",
        }
        if "benchmark" in suites
        else set()
    )


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


def complete_benchmark(benchmark: Any) -> bool:
    """A completed equivalent failure still contains valid native pairs."""
    return (
        isinstance(benchmark, dict)
        and (
            benchmark.get("status") == "passed"
            or (
                benchmark.get("status") == "failed"
                and bool(benchmark.get("performance_errors"))
            )
        )
        and not any(
            benchmark.get(key)
            for key in (
                "record_errors",
                "case_accounting_errors",
                "skip_record_errors",
            )
        )
    )


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
    raw_required = 0
    raw_pairs = 0
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
            paired = isinstance(providers, dict) and set(providers) == {
                "flagdnn",
                "hipdnn",
            }
            raw_required += int(paired)
            if complete_benchmark(benchmark) and paired:
                observed_required += 1
                continue
            reason = (
                "pair_present_but_suite_failed"
                if paired
                else (
                    "partial_pair"
                    if providers
                    else (
                        "not_run"
                        if status in {"not_run", "not_found"}
                        else "pair_not_emitted"
                    )
                )
            )
            missing.append(
                {
                    "operator": operator,
                    "case": case,
                    "benchmark_status": str(status),
                    "reason": reason,
                }
            )

    for operator in dict.fromkeys(selected_benchmark_operators):
        benchmark = results.get(operator, {}).get("benchmark")
        if not isinstance(benchmark, dict):
            continue
        records = benchmark.get("records", {})
        if not isinstance(records, dict):
            continue
        pairs = sum(
            isinstance(providers, dict)
            and set(providers) == {"flagdnn", "hipdnn"}
            for providers in records.values()
        )
        raw_pairs += pairs
        if complete_benchmark(benchmark):
            observed_pairs += pairs

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
        "raw_complete_required_case_count": raw_required,
        "raw_complete_pair_case_count": raw_pairs,
        "excluded_complete_required_case_count": raw_required
        - observed_required,
        "extra_pair_case_count": observed_pairs - observed_required,
        "missing_case_count": len(missing),
        "missing_reason_counts": {
            reason: sum(case["reason"] == reason for case in missing)
            for reason in (
                "pair_present_but_suite_failed",
                "partial_pair",
                "pair_not_emitted",
                "not_run",
            )
        },
        "missing_cases": missing,
        "verified": not missing,
    }


def validate_benchmark_record(
    record: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Validate equivalent v4 or explicitly identified native timing v5."""
    if (
        type(record.get("schema_version")) is int
        and record["schema_version"] == 5
    ):
        expected = {
            "schema_version",
            "kind",
            "provider",
            "case",
            "unit",
            "median",
            "p90",
            "samples",
            "timing_mode",
            "timing_reason",
            "benchmark_config",
        }
        if set(record) != expected:
            raise ValueError("fields do not match Hygon benchmark schema v5")
        if (
            record["kind"] != "steady_state"
            or record["provider"] not in ("flagdnn", "hipdnn")
            or record["unit"] != "us"
            or not isinstance(record["case"], str)
            or not record["case"]
        ):
            raise ValueError(
                "schema v5 requires a native FlagDNN/hipDNN timing"
            )
        modes = {
            "hip_graph_batch": "standard",
            "hip_event_direct_batch": "hipdnn_graph_tail_workaround",
        }
        if (
            not isinstance(record["timing_mode"], str)
            or record["timing_mode"] not in modes
            or record["timing_reason"] != modes[record["timing_mode"]]
        ):
            raise ValueError(
                "schema v5 timing mode/reason combination is invalid"
            )
        config = record["benchmark_config"]
        limits = {
            "warmup_iterations": 0,
            "sample_count": 1,
            "iterations_per_sample": 1,
        }
        if (
            not isinstance(config, dict)
            or set(config) != set(limits)
            or any(
                type(config[key]) is not int or config[key] < minimum
                for key, minimum in limits.items()
            )
        ):
            raise ValueError("schema v5 benchmark config is invalid")
        if (
            not isinstance(record["samples"], list)
            or len(record["samples"]) != config["sample_count"]
        ):
            raise ValueError(
                "schema v5 sample count differs from benchmark config"
            )
        try:
            valid_numbers = all(
                type(value) in (int, float)
                and math.isfinite(value)
                and value > 0
                for value in (
                    record["median"],
                    record["p90"],
                    *record["samples"],
                )
            )
        except OverflowError:
            valid_numbers = False
        if not valid_numbers:
            raise ValueError("schema v5 timings must be finite and positive")
        return {
            "timing": {
                key: record[key] for key in ("median", "p90", "samples")
            }
        }

    # v4 stores measured samples and a derived, unsampled reference.
    expected = {
        "schema_version",
        "kind",
        "provider",
        "case",
        "unit",
        "median",
        "p90",
        "samples",
        "equivalent_baseline",
    }
    if set(record) != expected:
        raise ValueError("fields do not match Hygon benchmark schema v4")
    if (
        type(record["schema_version"]) is not int
        or record["schema_version"] != 4
        or record["kind"] != "steady_state"
        or record["provider"] != "flagdnn"
        or record["unit"] != "us"
        or not isinstance(record["case"], str)
        or EQUIVALENT_CASE_PATTERN.fullmatch(record["case"]) is None
    ):
        raise ValueError(
            "schema v4 requires a supported measured FlagDNN case"
        )
    case_operator = EQUIVALENT_CASE_PATTERN.fullmatch(record["case"]).group(
        "operator"
    )
    baseline = record["equivalent_baseline"]
    identities = {
        "provider": "nvidia_equivalent",
        "platform": "nvidia",
        "target_platform": "hygon",
        "timing_profile": (
            "batchnorm_training" if case_operator == "batchnorm" else "native"
        ),
    }
    numeric = {
        "reference_time_us",
        "reference_peak",
        "target_peak",
        "min_relative_efficiency",
        "equivalent_time_us",
        "limit_us",
        "relative_efficiency",
    }
    required = set(identities) | numeric | {"scale_metric", "passed"}
    extensions = {"traffic_ratio", "reference_validation"}
    if case_operator != "div":
        required |= extensions
    if (
        not isinstance(baseline, dict)
        or not required <= set(baseline) <= required | extensions
        or any(baseline[key] != value for key, value in identities.items())
        or not isinstance(baseline["scale_metric"], str)
        or baseline["scale_metric"] not in EQUIVALENT_SCALE_METRICS
        or not isinstance(
            baseline.get("reference_validation", "cpu_full"), str
        )
        or baseline.get("reference_validation", "cpu_full")
        not in REFERENCE_VALIDATIONS
    ):
        raise ValueError("invalid equivalent-baseline fields or identity")
    if case_operator == "identity" and (
        baseline["scale_metric"] != "hbm_copy"
        or baseline["reference_validation"]
        != (
            "cpu_full"
            if record["case"].startswith("identity_perf_")
            else "bit_exact"
        )
        or baseline["traffic_ratio"]
        != (8 if record["case"].startswith("identity_bool_") else 1)
    ):
        raise ValueError(
            "identity requires its measured validation and logical-byte copy ratio"
        )
    if case_operator in EQUIVALENT_BFLOAT16_OPERATORS | {
        "batchnorm",
        "reduction",
    } and (
        baseline["reference_validation"] != "cpu_full"
        or baseline["traffic_ratio"] != 1
    ):
        raise ValueError(
            "supplemental floating-point cases require CPU-full unit-traffic evidence"
        )

    def positive(value: Any) -> bool:
        try:
            return (
                type(value) in (int, float)
                and math.isfinite(value)
                and value > 0
            )
        except OverflowError:
            return False

    if any(not positive(baseline[key]) for key in numeric) or not positive(
        record["median"]
    ):
        raise ValueError(
            "equivalent-baseline numbers must be finite and positive"
        )
    traffic_ratio = baseline.get("traffic_ratio", 1.0)
    if not positive(traffic_ratio) or (
        baseline["scale_metric"] == "vector_fp32" and traffic_ratio != 1
    ):
        raise ValueError(
            "traffic_ratio must be finite and positive; "
            "non-unit traffic is only supported by HBM metrics"
        )
    if baseline["min_relative_efficiency"] > 1:
        raise ValueError("min_relative_efficiency must be in (0, 1]")
    sample_count = 10 if case_operator == "batchnorm" else 20
    if (
        not isinstance(record["samples"], list)
        or len(record["samples"]) != sample_count
    ):
        raise ValueError(
            f"{identities['timing_profile']} timing profile requires "
            f"{sample_count} measured samples"
        )
    try:
        equivalent = (
            baseline["reference_time_us"]
            * baseline["reference_peak"]
            / baseline["target_peak"]
            * traffic_ratio
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
        equivalent = providers.get("flagdnn", {}).get("equivalent_baseline")
        if equivalent is not None:
            match = EQUIVALENT_CASE_PATTERN.fullmatch(case)
            if (
                operator not in EQUIVALENT_OPERATORS
                or actual != {"flagdnn"}
                or match is None
                or match.group("operator") != operator
                or providers["flagdnn"].get("case") != case
            ):
                errors.append(
                    f"Hygon equivalent benchmark case={case} must have only "
                    "the measured FlagDNN provider for the matching operator"
                )
            else:
                try:
                    validate_benchmark_record(providers["flagdnn"])
                except ValueError as error:
                    errors.append(
                        f"Hygon equivalent benchmark case={case}: {error}"
                    )
            continue
        if actual != expected:
            errors.append(
                f"Hygon benchmark case={case} providers={sorted(actual)}; "
                f"expected={sorted(expected)}"
            )
            continue
        versions = [providers[key].get("schema_version") for key in expected]
        if 5 in versions:
            if versions != [5, 5]:
                errors.append(
                    f"Hygon benchmark case={case} mixes native schemas"
                )
                continue
            for provider, record in providers.items():
                try:
                    validate_benchmark_record(record)
                    if (
                        record["provider"] != provider
                        or record["case"] != case
                    ):
                        raise ValueError(
                            "record case/provider differs from its pair"
                        )
                except ValueError as error:
                    errors.append(
                        f"Hygon benchmark case={case} provider={provider}: {error}"
                    )
            for key in ("timing_mode", "timing_reason", "benchmark_config"):
                if providers["flagdnn"].get(key) != providers["hipdnn"].get(
                    key
                ):
                    errors.append(
                        f"Hygon benchmark case={case} has mismatched {key}"
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
        if not complete_benchmark(benchmark):
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
                    **{
                        key: providers["flagdnn"][key]
                        for key in (
                            "timing_mode",
                            "timing_reason",
                            "benchmark_config",
                        )
                        if key in providers["flagdnn"]
                    },
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


def equivalent_performance_summary(
    results: dict[str, dict[str, Any]],
    threshold: float | None,
) -> dict[str, Any]:
    """Keep peak-normalized efficiency separate from native hipDNN speedup."""
    cases: list[dict[str, Any]] = []
    for operator, operator_results in results.items():
        benchmark = operator_results.get("benchmark")
        if not complete_benchmark(benchmark):
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
    return {
        "metric": EQUIVALENT_METRIC,
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
    for operator, result in results.items():
        performance = result.get("performance", {})
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
    if not isinstance(equivalent, dict) or not equivalent.get("case_count"):
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
        r"(PASS|SKIP|FAIL)\s+"
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
        + r"(?:\.[a-z0-9_]+)?\s+\.*\s*(?:\*\*\*)?(Passed|Skipped|Failed)\s",
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
            expected = {
                "Passed": "PASS",
                "Skipped": "SKIP",
                "Failed": "FAIL",
            }.get(statuses.get(test_id))
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
        if reported in {"PASS", "FAIL"} and group_executed <= 0:
            errors.append(
                "Hygon PASS/FAIL suite must execute at least one case"
            )
        if reported == "FAIL" and not (
            suite == "benchmark"
            and operator in EQUIVALENT_OPERATORS
            and ctest_reported_status == "failed"
        ):
            errors.append(
                "Hygon FAIL accounting requires a failed equivalent benchmark"
            )
        if reported == "SKIP" and (
            group_executed != 0 or matched is None or group_skipped != matched
        ):
            errors.append(
                "Hygon SKIP suite must execute zero cases and skip every case"
            )
    completed_failure = (
        ctest_reported_status == "failed"
        and operator in EQUIVALENT_OPERATORS
        and suite == "benchmark"
        and executed > 0
        and any(payload.startswith("FAIL ") for _, payload in marker_records)
        and any(
            providers.get("flagdnn", {})
            .get("equivalent_baseline", {})
            .get("passed")
            is False
            for providers in records.values()
        )
    )
    if not completed_failure and (ctest_reported_status == "passed") != (
        executed > 0
    ):
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


def reference_skip_records(output: str, provider: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    prefix = f"[SKIP][{provider}]"
    pattern = re.compile(
        "^" + re.escape(prefix) + r"\s+op=([^\s]+)\s+"
        r"case=([^\s]+)\s+reason=(.+)$"
    )
    for raw_line in output.splitlines():
        line = re.sub(r"^\s*\d+:\s?", "", raw_line).strip()
        if not line.startswith(prefix):
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


def hipdnn_skip_records(output: str) -> list[dict[str, str]]:
    return reference_skip_records(output, "hipdnn")


def validate_hygon_skip_records(
    records: list[dict[str, str]],
    operator: str,
    manifest_operators: list[str],
    provider: str = "hipdnn",
) -> list[str]:
    errors: list[str] = []
    matching_records = 0
    label = "hipDNN" if provider == "hipdnn" else "NVIDIA baseline"
    if not records:
        return [
            "skipped Hygon suite emitted no structured "
            f"[SKIP][{provider}] record for op={operator}"
        ]
    for index, record in enumerate(records, start=1):
        missing = [
            key
            for key in ("op", "case", "reason")
            if not record.get(key, "").strip()
        ]
        if missing:
            errors.append(
                f"{label} skip record {index} is missing non-empty "
                + ", ".join(missing)
            )
            continue
        if record["op"] != operator:
            errors.append(
                f"{label} skip record {index} has op={record['op']}; "
                f"expected op={operator}"
            )
            continue
        case = record["case"]
        owner = benchmark_case_operator(case, manifest_operators)
        if owner != operator:
            errors.append(
                f"{label} skip record {index} has case={case} "
                f"owned_by={owner}; expected owner={operator}"
            )
            continue
        matching_records += 1
    cases = [record.get("case", "") for record in records]
    if len(cases) != len(set(cases)):
        errors.append(f"{label} skip records contain duplicate case names")
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


def validate_nvidia_baseline_skips(
    records: list[dict[str, str]], operator: str
) -> list[str]:
    """Only known unmeasured auxiliary/excluded cases may skip this gate."""
    # The 24 auxiliary logical cases now have measured NVIDIA evidence. Only
    # explicit concatenate exclusions remain eligible for a baseline skip.
    if operator != "concatenate":
        return [f"NVIDIA baseline skips are not permitted for op={operator}"]
    try:
        baseline_path = Path(
            os.environ.get("FLAGDNN_PERFORMANCE_BASELINE")
            or PERFORMANCE_BASELINE_FILE
        )
        reference = json.loads(baseline_path.read_text())["reference"]
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
        allowed = (
            isinstance(exclusion, dict)
            and exclusion.get("op") == "concatenate"
            and exclusion.get("status") == "excluded"
        )
        if case in measured or not allowed:
            errors.append(
                f"NVIDIA baseline skip is not a known unmeasured case: {case}"
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
    hipdnn_skips = hipdnn_skip_records(output)
    baseline_skips = reference_skip_records(output, "nvidia_baseline")
    skip_records = hipdnn_skips + baseline_skips
    if hipdnn_skips:
        result["skip_records"] = hipdnn_skips
    if baseline_skips:
        result["baseline_skip_records"] = baseline_skips
    if ctest_reported_status == "skipped" or skip_records:
        skip_errors = []
        if hipdnn_skips or not baseline_skips:
            skip_errors.extend(
                validate_hygon_skip_records(
                    hipdnn_skips, operator, manifest_operators
                )
            )
        if baseline_skips:
            if suite != "benchmark" or operator not in EQUIVALENT_OPERATORS:
                skip_errors.append(
                    "NVIDIA baseline skips require a supported "
                    "equivalent benchmark"
                )
            skip_errors.extend(
                validate_hygon_skip_records(
                    baseline_skips,
                    operator,
                    manifest_operators,
                    "nvidia_baseline",
                )
            )
            skip_errors.extend(
                validate_nvidia_baseline_skips(baseline_skips, operator)
            )
        names = [record.get("case", "") for record in skip_records]
        if len(names) != len(set(names)):
            skip_errors.append(
                "Hygon reference skips contain duplicate case names"
            )
        if skip_errors:
            result["skip_record_errors"] = skip_errors
            result["status"] = "failed"
    equivalent_records = (
        suite == "benchmark"
        and ctest_reported_status in {"passed", "failed"}
        and any(
            providers.get("flagdnn", {}).get("schema_version") == 4
            for providers in records.values()
        )
    )
    if suite == "benchmark" and (
        result["status"] == "passed" or equivalent_records
    ):
        pair_errors = validate_hygon_benchmark_pairs(
            records, operator, manifest_operators
        )
        if pair_errors:
            result.setdefault("record_errors", []).extend(pair_errors)
            result["status"] = "failed"
    if ctest_reported_status in {"passed", "skipped"} or equivalent_records:
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
    if equivalent_records and not any(
        result.get(key)
        for key in (
            "record_errors",
            "case_accounting_errors",
            "skip_record_errors",
        )
    ):
        failures = [
            f"{case}: measured_us={providers['flagdnn']['median']:.9g} "
            "exceeds "
            f"equivalent limit_us={baseline['limit_us']:.9g} "
            f"(relative_efficiency={baseline['relative_efficiency']:.9g}, "
            f"minimum={baseline['min_relative_efficiency']:.9g})"
            for case, providers in records.items()
            if (baseline := providers["flagdnn"].get("equivalent_baseline"))
            and not baseline["passed"]
        ]
        if failures:
            # A completed measurement below the reference is informational.
            # Preserve failures from older executables (or other real process
            # errors); never turn a nonzero execution into a successful task.
            key = (
                "performance_errors"
                if ctest_reported_status == "failed"
                else "performance_warnings"
            )
            result[key] = failures


def result_diagnostics(result: dict[str, Any]) -> list[tuple[str, str]]:
    diagnostics: list[tuple[str, str]] = []
    diagnostics.extend(
        ("Hygon reference skip record error", error)
        for error in result.get("skip_record_errors", [])
    )
    diagnostics.extend(
        ("Hygon case accounting error", error)
        for error in result.get("case_accounting_errors", [])
    )
    diagnostics.extend(
        ("Hygon equivalent performance failure", error)
        for error in result.get("performance_errors", [])
    )
    diagnostics.extend(
        ("Hygon equivalent baseline warning", warning)
        for warning in result.get("performance_warnings", [])
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
        comparable_coverage = hygon_comparable_coverage(
            results, suite_operators["benchmark"], catalog
        )
        if not comparable_coverage["verified"]:
            print(
                "comparable coverage gate: "
                f"{comparable_coverage['observed_required_case_count']}/"
                f"{comparable_coverage['required_case_count']} declared cases "
                "have complete FlagDNN/hipDNN pairs in completed suites; "
                f"raw_complete={comparable_coverage['raw_complete_required_case_count']} "
                f"excluded_suite_failed={comparable_coverage['excluded_complete_required_case_count']}",
                flush=True,
            )
            for missing in comparable_coverage["missing_cases"]:
                print(
                    "  FAIL comparable coverage "
                    f"op={missing['operator']} case={missing['case']} "
                    f"reason={missing['reason']} "
                    f"benchmark_status={missing['benchmark_status']}",
                    flush=True,
                )
            failed = True

    performance = (
        benchmark_speedup_summary(results, min_speedup, comparable_coverage)
        if "benchmark" in suites
        else None
    )
    equivalent = (
        equivalent_performance_summary(results, min_speedup)
        if "benchmark" in suites
        else None
    )
    has_equivalent = equivalent is not None and equivalent["case_count"] > 0
    if (
        performance is not None
        and not performance["case_count"]
        and has_equivalent
    ):
        # A div-only run has no native-reference speedup gate to apply.
        performance["gate_passed"] = None
        performance["ratio_gate_passed"] = None
    if min_speedup is not None and (
        performance is not None
        and (performance["case_count"] or not has_equivalent)
    ):
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
    if has_equivalent:
        assert equivalent is not None
        enabled = equivalent["gate_enabled"]
        print(
            (
                "equivalent performance gate: "
                if enabled
                else "equivalent performance report (informational): "
            )
            + f"{equivalent['passed_case_count']}/{equivalent['case_count']} "
            "cases "
            "meet baseline relative efficiency"
            + (
                f" and --min-speedup >= {min_speedup:.6g}"
                if min_speedup is not None
                else "; performance gate disabled (no --min-speedup)"
            ),
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
            valid_skips = not any(
                result.get(key)
                for key in (
                    "record_errors",
                    "case_accounting_errors",
                    "skip_record_errors",
                )
            )
            if valid_skips and (
                result["status"] in {"passed", "skipped"}
                or complete_benchmark(result)
            ):
                reference_skips += len(result.get("skip_records", []))
                baseline_skips += len(result.get("baseline_skip_records", []))

    return {
        "failed": failed,
        "summary": {
            "comparable_coverage": comparable_coverage,
            "performance": performance,
            "equivalent_performance": equivalent,
        },
        "coverage": {
            "hipdnn_reference_skips": reference_skips,
            "nvidia_baseline_skips": baseline_skips,
            "hipdnn_skip_record_errors": skip_record_errors,
            "hygon_case_accounting_errors": accounting_errors,
        },
    }
