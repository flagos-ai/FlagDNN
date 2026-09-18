# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Check catalogs against shared functional and benchmark factories."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
from run_tests_adapter import dtype_categories


def dumped(
    command: list[str], environment: dict[str, str]
) -> dict[str, set[str]]:
    output = subprocess.check_output(command, env=environment, text=True)
    result: dict[str, set[str]] = {}
    for line in output.splitlines():
        operation, name = line.split("\t")
        names = result.setdefault(operation, set())
        if name in names:
            raise RuntimeError(f"duplicate factory case: {operation}/{name}")
        names.add(name)
    return result


def require_equal(
    actual: dict[str, set[str]], expected: dict[str, set[str]]
) -> None:
    if actual == expected:
        return
    differences = []
    for operation in sorted(actual.keys() | expected.keys()):
        left, right = actual.get(operation, set()), expected.get(
            operation, set()
        )
        if left != right:
            differences.append(
                f"{operation}:"
                f" missing={sorted(right-left)} extra={sorted(left-right)}"
            )
    raise RuntimeError(
        "THead catalog/factory mismatch: " + "; ".join(differences)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    binaries = args.build_dir.resolve() / "backends/thead/validation"
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    capability = json.loads((root / "capability.json").read_text())[
        "operators"
    ]
    expected = {
        operation: set(record["case_names"])
        for operation, record in capability.items()
    }
    functional = binaries / "flagdnn_test_thead_validation_contract"
    if functional.is_file():
        require_equal(
            dumped([str(functional), "--dump-cases"], environment), expected
        )
    benchmark_file = root / "benchmark/comparable_cases.json"
    benchmark = json.loads(benchmark_file.read_text())["operators"]
    if (binaries / "flagdnn_benchmark_thead_add").is_file():
        actual = {}
        for operation in expected:
            executable = binaries / f"flagdnn_benchmark_thead_{operation}"
            names = dumped([str(executable), "--dump-cases"], environment)[
                operation
            ]
            for category in dtype_categories(operation):
                env = dict(environment)
                runner_category = category
                if category in {"ieee", "tf32"}:
                    runner_category = "precision"
                    env["FLAGDNN_INPUT_PRECISION"] = (
                        "1" if category == "ieee" else "2"
                    )
                extra = dumped(
                    [
                        str(binaries / "flagdnn_benchmark_thead_dtype"),
                        "--dump-cases",
                        operation,
                        runner_category,
                    ],
                    env,
                )[operation]
                if names & extra:
                    raise RuntimeError(
                        f"duplicate benchmark workloads: {operation}"
                    )
                names |= extra
            actual[operation] = names
        require_equal(
            actual,
            {
                operation: set(records)
                for operation, records in benchmark.items()
            },
        )
    print("PASS THead complete shared-factory coverage")


if __name__ == "__main__":
    main()
