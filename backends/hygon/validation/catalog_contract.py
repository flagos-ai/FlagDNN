# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Check Hygon registrations and the policies surrounding shared shape data."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys


def load(path: Path):
    specification = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def cpp_tokens(source: str) -> str:
    source = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.DOTALL)
    return re.sub(r"\s+", "", source)


def main(source_root: Path, build_dir: Path) -> None:
    root = source_root / "backends"
    hygon = root / "hygon" / "validation"
    nvidia = root / "nvidia" / "validation"
    runner = load(source_root / "tools" / "run_tests.py")
    manifests = runner.operator_manifests()
    for filename in ("additional_operators.txt", "dtype_only_operators.txt"):
        assert (hygon / "benchmark" / filename).read_text().split() == (
            nvidia / "benchmark" / filename
        ).read_text().split(), (
            f"NVIDIA benchmark registration changed: {filename}"
        )
    nv = (nvidia / "benchmark/cases.hpp").read_text()
    hg = (hygon / "benchmark/cases.hpp").read_text()
    nv = nv.replace("NVIDIA", "HYGON").replace(
        "cudnn_benchmark_cases", "aligned_benchmark_cases"
    )
    assert cpp_tokens(nv) == cpp_tokens(
        hg
    ), "NVIDIA default benchmark selection changed"
    assert cpp_tokens(
        (nvidia / "benchmark/native_runner.cpp").read_text()
    ) == cpp_tokens(
        (hygon / "benchmark/native_runner.cpp").read_text()
    ), "NVIDIA extended benchmark generators changed"

    # The C++ probe compiles NVIDIA selectors directly. Review its local
    # pointwise capture stub whenever NVIDIA dtype selection changes.
    pointwise = (nvidia / "functional/pointwise_runner.cpp").read_text()
    selection = pointwise.split(
        "const auto type = test_case.inputs.front().data_type;", 1
    )[1]
    selection = selection.split(
        "if (test_case.mode == FLAGDNN_POINTWISE_IDENTITY", 1
    )[0]
    digest = hashlib.sha256(cpp_tokens(selection).encode()).hexdigest()
    expected = (
        (hygon / "catalog/pointwise-selection.sha256").read_text().strip()
    )
    assert (
        digest == expected
    ), "Review NVIDIA pointwise dtype selection in the catalog probe"

    listing = json.loads(
        subprocess.check_output(
            ["ctest", "--test-dir", str(build_dir), "--show-only=json-v1"],
            text=True,
        )
    )
    tests = {test["name"]: test for test in listing["tests"]}
    additional = (
        (nvidia / "benchmark/additional_operators.txt").read_text().split()
    )
    dtype_only = set(
        (nvidia / "benchmark/dtype_only_operators.txt").read_text().split()
    )
    expected_benchmarks = {
        f"benchmark.hygon.{op}"
        for op in manifests["benchmark"]
        if op not in dtype_only
    } | {f"benchmark.hygon.{op}" for op in additional}
    groups = {
        "boolean": ("logical_and", "logical_or", "logical_not"),
        "copy": ("identity", "reshape", "transpose", "slice"),
        "ieee": ("matmul", "conv_fprop", "conv_dgrad", "conv_wgrad"),
        "tf32": ("matmul", "conv_fprop", "conv_dgrad", "conv_wgrad"),
        "fp32_output": ("reduction",),
        "fp8": ("matmul",),
    }
    nv_cmake = (nvidia / "CMakeLists.txt").read_text()
    categories = (
        re.search(r"foreach\(category IN ITEMS ([^)]+)\)", nv_cmake)
        .group(1)
        .split()
    )
    assert set(categories) == set(groups) - {"fp8"}
    for category in categories:
        operators = " ".join(groups[category])
        assert f"set(dtype_operators {operators})" in nv_cmake
    expected_benchmarks |= {
        f"benchmark.hygon.{op}.{category}"
        for category, ops in groups.items()
        for op in ops
    }
    actual = {name for name in tests if name.startswith("benchmark.hygon.")}
    benchmarks_enabled = (
        "FLAGDNN_BUILD_BENCHMARKS:BOOL=ON"
        in (build_dir / "CMakeCache.txt").read_text()
    )
    if not benchmarks_enabled:
        expected_benchmarks = set()
    assert actual == expected_benchmarks, (
        f"benchmark catalog mismatch missing={expected_benchmarks - actual} "
        f"extra={actual - expected_benchmarks}"
    )
    for op in manifests["functional"]:
        # CTest omits command metadata for registered, not-yet-built targets.
        command = tests[f"functional.hygon.{op}"].get("command")
        if command is not None:
            assert Path(command[0]).name == f"flagdnn_test_hygon_{op}"
    cmake = (hygon / "CMakeLists.txt").read_text()
    functional = cmake.split("flagdnn_register_functional_suite(", 1)[1].split(
        ")", 1
    )[0]
    assert (
        "UNSUPPORTED_OPERATORS" not in functional
    ), "placeholder capability suite hides shapes"
    convolution = (hygon / "functional/convolution_runner.cpp").read_text()
    assert (
        "private_" not in convolution
        and "push_back(private" not in convolution
    )
    assert "integration.hygon.private_convolution" in tests
    assert "integration.hygon.dtype_catalog" in tests
    adapter = load(hygon / "run_tests_adapter.py")
    mixed_output = (
        "1: FLAGDNN_IDENTITY_BENCHMARK: PASS cases=1 executed=1 skipped=0\n"
        "2: [SKIP][hipdnn] op=identity "
        "case=identity_fp8_test reason=unsupported\n"
        "2: FLAGDNN_IDENTITY_BENCHMARK: SKIP cases=1 executed=0 skipped=1\n"
        "1/2 Test #1: benchmark.hygon.identity .... Passed 0.01 sec\n"
        "2/2 Test #2: benchmark.hygon.identity.copy ....***Skipped 0.01 sec\n"
    )
    records: dict[str, dict[str, dict]] = {
        "identity_fp32_test": {"flagdnn": {}, "hipdnn": {}}
    }
    result = {
        "status": "failed",
        "record_errors": ["skipped benchmark emitted timing provider records"],
    }
    adapter.postprocess_result(
        result=result,
        ctest_reported_status="skipped",
        output=mixed_output,
        operator="identity",
        suite="benchmark",
        records=records,
        manifest_operators=manifests["benchmark"],
    )
    assert result["status"] == "passed", result
    assert adapter.validate_hygon_case_accounting(
        mixed_output.replace(
            "cases=1 executed=0 skipped=1", "cases=2 executed=0 skipped=2"
        ),
        "identity",
        "benchmark",
        "passed",
        records,
        adapter.hipdnn_skip_records(mixed_output),
    ), "unreported child skips accepted"
    missing_child = "\n".join(
        line
        for line in mixed_output.splitlines()
        if "2: FLAGDNN_IDENTITY_BENCHMARK" not in line
    )
    assert adapter.validate_hygon_case_accounting(
        missing_child,
        "identity",
        "benchmark",
        "passed",
        records,
        adapter.hipdnn_skip_records(mixed_output),
    ), "missing child accounting accepted"
    functional_count = len(manifests["functional"])
    print(
        f"PASS catalog registration: functional={functional_count} "
        f"benchmark groups={len(actual)}"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
