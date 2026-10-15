#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate equivalent result schemas and optional captured native logs.

This standalone contract requires jsonschema. It does not change dependencies
of the ordinary batch runner or its mandatory preflight contracts.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[3]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def measured_record(
    case: str,
    target: str,
    metric: str,
    validation: str,
    traffic: float = 1.0,
) -> dict[str, Any]:
    batchnorm = case.startswith("batchnorm_")
    equivalent = 1.6 * traffic
    return {
        "schema_version": 4,
        "kind": "steady_state",
        "provider": "flagdnn",
        "case": case,
        "unit": "us",
        "median": 1.0,
        "p90": 1.0,
        "samples": [1.0] * (10 if batchnorm else 20),
        "equivalent_baseline": {
            "provider": "nvidia_equivalent",
            "platform": "nvidia",
            "target_platform": target,
            "scale_metric": metric,
            "timing_profile": (
                "batchnorm_training" if batchnorm else "native"
            ),
            "reference_time_us": 2.0,
            "reference_peak": 4.0,
            "target_peak": 5.0,
            "min_relative_efficiency": 0.8,
            "traffic_ratio": traffic,
            "reference_validation": validation,
            "equivalent_time_us": equivalent,
            "limit_us": equivalent / 0.8,
            "relative_efficiency": equivalent,
            "passed": True,
        },
    }


def check_synthetic(validator: Draft202012Validator) -> dict[str, int]:
    value: object
    baseline = json.loads(
        (ROOT / "docs/performance-baseline.json").read_text()
    )
    thead = load_module(
        "thead_schema_adapter",
        ROOT / "backends/thead/validation/run_tests_adapter.py",
    )
    hygon = load_module(
        "hygon_schema_adapter",
        ROOT / "backends/hygon/validation/run_tests_adapter.py",
    )
    accepted: dict[str, int] = {"thead": 0, "hygon": 0}
    last_thead: dict[str, Any] | None = None
    for case, reference in baseline["reference"]["cases"].items():
        operator = reference["op"]
        if operator in thead.EQUIVALENT_SCALE_METRICS:
            record = measured_record(
                case,
                "thead",
                thead.EQUIVALENT_SCALE_METRICS[operator],
                reference["validation"],
            )
            validator.validate(record)
            thead.validate_benchmark_record(record)
            accepted["thead"] += 1
            last_thead = record
        if hygon.EQUIVALENT_CASE_PATTERN.fullmatch(case) is not None:
            policy = baseline["platforms"]["hygon"]["scaling"][
                "case_policies"
            ][operator]
            record = measured_record(
                case,
                "hygon",
                policy["scale_metric"],
                reference["validation"],
                8.0 if case.startswith("identity_bool_") else 1.0,
            )
            validator.validate(record)
            hygon.validate_benchmark_record(record)
            accepted["hygon"] += 1
    require(
        accepted == {"thead": 115, "hygon": 1968},
        f"unexpected supported equivalent scope: {accepted}",
    )
    require(last_thead is not None, "no Thead equivalent fixtures")

    # Every explicit NVIDIA concatenation exclusion must remain invalid.
    excluded = 0
    for case, reference in baseline["reference"]["unavailable_cases"].items():
        if reference["op"] == "concatenate":
            record = measured_record(case, "thead", "hbm_copy", "sanity")
            require(
                not validator.is_valid(record),
                f"unmeasured concatenation accepted: {case}",
            )
            excluded += 1
    require(excluded == 70, "concatenation exclusion scope changed")

    # Original div v4 records predate explicit traffic/validation metadata.
    legacy = measured_record(
        "div_perf_fp32_1x1x1024_by_1x1x1024",
        "hygon",
        "vector_fp32",
        "cpu_full",
    )
    legacy["equivalent_baseline"].pop("traffic_ratio")
    legacy["equivalent_baseline"].pop("reference_validation")
    validator.validate(legacy)
    hygon.validate_benchmark_record(legacy)

    invalid = []
    for field, value in (
        ("samples", [1.0] * 19),
        ("provider", "nvidia_equivalent"),
    ):
        record = copy.deepcopy(last_thead)
        record[field] = value
        invalid.append(record)
    for case in (
        "div_perf_fp32_1x1x1024_by_1x1x1024",
        "gen_index_fp16_3_axis0",
        "gelu_approx_tanh_backward_fp32_1x1x1024_by_1x1x1024",
    ):
        record = copy.deepcopy(last_thead)
        record["case"] = case
        invalid.append(record)
    for field, value in (
        ("target_platform", "unsupported_platform"),
        ("min_relative_efficiency", 0.5),
        ("traffic_ratio", 2.0),
        ("timing_profile", "batchnorm_training"),
    ):
        record = copy.deepcopy(last_thead)
        record["equivalent_baseline"][field] = value
        invalid.append(record)
    for case, metric, validation in (
        ("concatenate_fp32_1_axis0", "hbm_write", "sanity"),
        ("concatenate_fp32_1_axis0", "hbm_copy", "cpu_full"),
        ("gen_index_int32_3_axis0", "hbm_copy", "sanity"),
        ("gen_index_int32_3_axis0", "hbm_write", "cpu_full"),
        (
            "gelu_approx_tanh_backward_perf_fp32_1x1x1024_by_1x1x1024",
            "vector_fp32",
            "sanity",
        ),
    ):
        invalid.append(measured_record(case, "thead", metric, validation))
    for record in invalid:
        require(
            not validator.is_valid(record),
            "schema accepted mismatched Thead equivalent metadata",
        )
    return {**accepted, "rejected_unmeasured_concatenate": excluded}


def check_logs(
    validator: Draft202012Validator, paths: list[Path]
) -> dict[str, int]:
    counts = {}
    for path in paths:
        count = 0
        for number, raw in enumerate(path.read_text().splitlines(), 1):
            line = re.sub(r"^\s*\d+:\s?", "", raw).strip()
            if not line.startswith("{"):
                continue
            record = json.loads(line)
            if record.get("kind") != "steady_state":
                continue
            errors = list(validator.iter_errors(record))
            require(
                not errors, f"{path}:{number} violates result schema: {errors}"
            )
            count += 1
        require(count > 0, f"{path} has no native timing records")
        counts[str(path)] = count
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    schema = json.loads((ROOT / "benchmark/result.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    synthetic = check_synthetic(validator)
    captured = check_logs(validator, args.input)
    print(json.dumps({"synthetic": synthetic, "captured": captured}))
    print("THead equivalent result schema contract: PASS")
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
