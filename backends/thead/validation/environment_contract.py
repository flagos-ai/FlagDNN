#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Executable contracts for the THead external-environment preflight."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any


sys.dont_write_bytecode = True


_EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "ppu_sdk",
    "cuda_compat",
    "triton",
    "libtriton_jit",
    "python",
    "device",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _run(
    helper: Path,
    triton_root: Path,
    arguments: list[str],
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    search_roots = {
        Path(entry).expanduser().resolve()
        for entry in sys.path
        if isinstance(entry, str) and entry
    }
    if triton_root.resolve(strict=True) not in search_roots:
        previous_python_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(triton_root)
        if previous_python_path:
            environment["PYTHONPATH"] += os.pathsep + previous_python_path
    return subprocess.run(
        [sys.executable, str(helper), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def _load_single_json(text: str, source: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{source} did not contain one JSON value: {error}")
    _require(isinstance(value, dict), f"{source} JSON is not an object")
    return value


def _assert_canonical_file(value: Any, root: Path, name: str) -> None:
    _require(isinstance(value, str) and value, f"{name} is not a path")
    path = Path(value)
    _require(path.is_absolute(), f"{name} is not absolute: {path}")
    _require(path.is_file(), f"{name} does not exist: {path}")
    _require(path == path.resolve(strict=True), f"{name} is not canonical: {path}")
    _require(
        path.is_relative_to(root.resolve(strict=True)),
        f"{name} escaped {root}: {path}",
    )


def _git_head(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "--verify", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _release_version(release_file: Path) -> str:
    entries = [
        line.split(":", 1)[1].strip().strip("'\"")
        for line in release_file.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("version:")
    ]
    _require(len(entries) == 1 and entries[0], "invalid release.yaml fixture")
    return entries[0]


def _assert_static_document(
    document: dict[str, Any],
    sdk_root: Path,
    triton_root: Path,
    jit_root: Path,
) -> None:
    _require(
        set(document) == _EXPECTED_TOP_LEVEL_KEYS,
        f"unexpected top-level fields: {sorted(document)}",
    )
    _require(document["schema_version"] == 2, "wrong schema version")
    _require(document["device"] is None, "static probe opened a device")

    sdk = document["ppu_sdk"]
    _require(sdk["root"] == str(sdk_root.resolve(strict=True)), "wrong SDK root")
    _require(
        sdk["version"] == _release_version(sdk_root / "release.yaml"),
        "wrong SDK version",
    )
    for field in ("release_file", "hggc_library", "acdnn_header", "acdnn_library"):
        _assert_canonical_file(sdk[field], sdk_root, f"ppu_sdk.{field}")

    cuda = document["cuda_compat"]
    cuda_root = sdk_root / "CUDA_SDK"
    _require(cuda["root"] == str(cuda_root.resolve(strict=True)), "wrong CUDA root")
    for field in ("header", "driver_library", "compiler", "ptxas"):
        _assert_canonical_file(cuda[field], cuda_root, f"cuda_compat.{field}")
    _assert_canonical_file(
        cuda["ir_formatter"], sdk_root, "cuda_compat.ir_formatter"
    )

    triton = document["triton"]
    _require(
        triton["root"] == str(triton_root.resolve(strict=True)),
        "wrong Triton package root",
    )
    _require(
        isinstance(triton["module_version"], str) and triton["module_version"],
        "missing Triton module version",
    )
    _require(
        isinstance(triton["distribution_version"], str)
        and "ppu" in triton["distribution_version"].lower(),
        "Triton distribution is not PPU-qualified",
    )
    _require("nvidia" in triton["backend_catalog"], "wrong backend catalog")
    _require(
        triton["codegen_backend"] == "nvidia"
        and triton["target_backend"] == "cuda"
        and triton["binary_extension"] == "cubin"
        and triton["ppu_compatibility"] == "cuda",
        "wrong Triton PPU compatibility identity",
    )
    source_identity = triton["source_identity"]
    _require(
        source_identity.get("kind") in {"git_commit", "content_sha256"}
        and isinstance(source_identity.get("value"), str)
        and len(source_identity["value"])
        in ({40} if source_identity["kind"] == "git_commit" else {64}),
        "wrong Triton source identity",
    )
    for field in (
        "package_file",
        "distribution_metadata",
        "compiler_file",
        "driver_file",
        "compiler_frontend_file",
    ):
        _assert_canonical_file(triton[field], triton_root, f"triton.{field}")

    jit = document["libtriton_jit"]
    _require(jit["root"] == str(jit_root.resolve(strict=True)), "wrong JIT root")
    _require(jit["backend"] == "CUDA", "JIT backend is not CUDA")
    _require(
        jit["source_identity"]
        == {
            "kind": "git_commit",
            "value": _git_head(Path(jit["repository_root"])),
        },
        "wrong JIT source identity",
    )
    for field in (
        "config_file",
        "library",
        "standalone_compile",
        "gen_ssig",
    ):
        _assert_canonical_file(jit[field], jit_root, f"libtriton_jit.{field}")

    python = document["python"]
    _require(Path(python["executable"]).samefile(sys.executable), "wrong Python")
    _require(python["implementation"] == "cpython", "wrong Python implementation")
    _require(
        isinstance(python["version"], str) and python["version"],
        "missing Python version",
    )
    _require(isinstance(python["abi"], str) and python["abi"], "missing Python ABI")
    _require(
        isinstance(python["torch_version"], str) and python["torch_version"],
        "missing Torch version",
    )
    _require(Path(python["torch_file"]).is_file(), "missing Torch package file")


def main() -> int:
    validation_root = Path(__file__).resolve().parent
    helper = validation_root / "preflight_environment.py"
    schema = validation_root / "environment.schema.json"
    sdk_root = Path(os.environ.get("FLAGDNN_THEAD_PPU_SDK_ROOT", "/usr/local/PPU_SDK"))
    triton_root = Path(
        os.environ.get(
            "FLAGDNN_THEAD_TRITON_ROOT",
            "/usr/local/lib/python3.12/site-packages",
        )
    )
    jit_root = Path(
        os.environ.get(
            "FLAGDNN_THEAD_TRITON_JIT_ROOT",
            str(validation_root.parents[3] / "libtriton_jit"),
        )
    )
    bytecode_before = set(validation_root.rglob("*.pyc"))

    with tempfile.TemporaryDirectory(prefix="flagdnn-thead-environment-") as directory:
        temporary_root = Path(directory)
        output = temporary_root / "static environment.json"
        common_arguments = [
            "--sdk-root",
            str(sdk_root),
            "--triton-root",
            str(triton_root),
            "--triton-jit-root",
            str(jit_root),
            "--schema",
            str(schema),
        ]
        static_result = _run(
            helper,
            triton_root,
            ["--static", *common_arguments, "--output", str(output)],
        )
        _require(
            static_result.returncode == 0,
            "static preflight failed:\n"
            f"stdout:\n{static_result.stdout}\n"
            f"stderr:\n{static_result.stderr}",
        )
        _require(not static_result.stderr, "static preflight wrote to stderr")
        stdout_document = _load_single_json(static_result.stdout, "stdout")
        _require(output.is_file(), "static preflight did not write --output")
        output_document = _load_single_json(output.read_text(), "--output")
        _require(stdout_document == output_document, "stdout and --output differ")
        _assert_static_document(stdout_document, sdk_root, triton_root, jit_root)

        fake_smi = temporary_root / "ppu-smi-with-zero-status"
        fake_smi.write_text(
            "#!/bin/sh\n"
            "echo 'init HGML error: driver is not loaded' >&2\n"
            "exit 0\n",
            encoding="utf-8",
        )
        fake_smi.chmod(fake_smi.stat().st_mode | stat.S_IXUSR)
        device_result = _run(
            helper,
            triton_root,
            [
                "--device",
                "0",
                *common_arguments,
                "--ppu-smi",
                str(fake_smi),
            ],
        )
        _require(device_result.returncode != 0, "driver error returned success")
        _require(not device_result.stdout, "failed device probe wrote success JSON")
        error_document = _load_single_json(device_result.stderr, "device stderr")
        _require(
            error_document.get("error") == "driver_not_loaded",
            f"unstable driver diagnostic: {error_document}",
        )

    bytecode_after = set(validation_root.rglob("*.pyc"))
    _require(bytecode_after == bytecode_before, "preflight wrote Python bytecode")
    print("THead environment contract: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"THead environment contract: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error
