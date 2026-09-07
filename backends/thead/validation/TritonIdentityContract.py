#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate the installed PPU-aware Triton and CUDA-JIT identity tuple."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


sys.dont_write_bytecode = True


class IdentityError(RuntimeError):
    pass


def _inside(path_value: str | None, root: Path, description: str) -> Path:
    if not path_value:
        raise IdentityError(f"{description} has no source path")
    path = Path(path_value).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(root):
        raise IdentityError(f"{description} is outside configured root {root}: {path}")
    return path


def _distribution_metadata(root: Path) -> tuple[Path, str]:
    try:
        distribution = importlib.metadata.distribution("triton")
    except importlib.metadata.PackageNotFoundError as error:
        raise IdentityError("configured Python path has no Triton distribution") from error
    metadata_files = [
        Path(distribution.locate_file(entry)).resolve(strict=True)
        for entry in distribution.files or ()
        if entry.name == "METADATA" and entry.parent.name.endswith(".dist-info")
    ]
    if len(metadata_files) != 1 or not metadata_files[0].is_relative_to(root):
        raise IdentityError(
            "Triton distribution metadata is missing or outside configured root"
        )
    version = distribution.version
    if "ppu" not in version.lower():
        raise IdentityError(
            f"Triton distribution is not the PPU-qualified build: {version!r}"
        )
    return metadata_files[0], version


def _jit_backend(jit_root: Path) -> tuple[Path, str]:
    config = (jit_root / "build" / "TritonJITConfig.cmake").resolve(strict=True)
    matches = re.findall(
        (
            r"^[ \t]*set\([ \t]*TritonJIT_BACKEND[ \t]+"
            r"[\"']?([A-Za-z0-9_+-]+)[\"']?[ \t]*\)[ \t]*$"
        ),
        config.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if matches != ["CUDA"]:
        raise IdentityError(f"JIT backend is not exactly CUDA: {matches!r}")
    return config, matches[0]


def identify(triton_root_value: Path, jit_root_value: Path) -> dict[str, Any]:
    triton_root = triton_root_value.resolve(strict=True)
    jit_root = jit_root_value.resolve(strict=True)
    if not triton_root.is_dir() or not jit_root.is_dir():
        raise IdentityError("Triton and JIT roots must be directories")

    search_roots = {
        Path(entry).expanduser().resolve()
        for entry in sys.path
        if isinstance(entry, str) and entry
    }
    if triton_root not in search_roots:
        sys.path.insert(0, str(triton_root))
    try:
        triton = importlib.import_module("triton")
        backend_module = importlib.import_module("triton.backends")
        compiler_api = importlib.import_module("triton.backends.compiler")
        compiler_frontend = importlib.import_module("triton.compiler.compiler")
    except Exception as error:
        raise IdentityError(f"cannot import configured Triton: {error}") from error

    package_file = _inside(
        getattr(triton, "__file__", None), triton_root, "Triton package"
    )
    catalog = getattr(backend_module, "backends", None)
    if not isinstance(catalog, dict) or "nvidia" not in catalog:
        raise IdentityError(
            f"Triton backend catalog has no CUDA codegen backend: {catalog!r}"
        )
    cuda = catalog["nvidia"]
    compiler_file = _inside(
        inspect.getsourcefile(cuda.compiler), triton_root, "Triton CUDA compiler"
    )
    driver_file = _inside(
        inspect.getsourcefile(cuda.driver), triton_root, "Triton CUDA driver"
    )
    frontend_file = _inside(
        inspect.getsourcefile(compiler_frontend),
        triton_root,
        "Triton compiler frontend",
    )

    compiler_source = compiler_file.read_text(encoding="utf-8")
    ppu_markers = ("PPU_SDK", "llvm-irformatter", "--ppu-backend-options")
    if any(marker not in compiler_source for marker in ppu_markers):
        raise IdentityError("Triton CUDA compiler lacks the PPU compatibility path")

    target_type = getattr(compiler_api, "GPUTarget", None)
    if target_type is None:
        raise IdentityError("Triton has no GPUTarget API")
    try:
        backend = cuda.compiler(target_type("cuda", 80, 32))
    except Exception as error:
        raise IdentityError(f"cannot instantiate Triton CUDA compiler: {error}") from error
    binary_extension = getattr(backend, "binary_ext", None)
    if binary_extension != "cubin":
        raise IdentityError(
            f"Triton CUDA binary extension is not cubin: {binary_extension!r}"
        )

    metadata_file, distribution_version = _distribution_metadata(triton_root)
    config, jit_backend = _jit_backend(jit_root)
    module_version = getattr(triton, "__version__", None)
    if not isinstance(module_version, str) or not module_version:
        raise IdentityError("Triton module version is missing")
    return {
        "schema_version": 1,
        "triton": {
            "root": str(triton_root),
            "package_file": str(package_file),
            "module_version": module_version,
            "distribution_version": distribution_version,
            "distribution_metadata": str(metadata_file),
            "backend_catalog": sorted(catalog),
            "codegen_backend": "nvidia",
            "target_backend": "cuda",
            "compiler_file": str(compiler_file),
            "driver_file": str(driver_file),
            "compiler_frontend_file": str(frontend_file),
            "binary_extension": binary_extension,
            "ppu_compatibility": "cuda",
        },
        "libtriton_jit": {
            "root": str(jit_root),
            "config_file": str(config),
            "backend": jit_backend,
        },
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--jit-root",
        type=Path,
        default=Path(
            os.environ.get(
                "FLAGDNN_THEAD_TRITON_JIT_ROOT",
                str(Path(__file__).resolve().parents[4] / "libtriton_jit"),
            )
        ),
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    print(
        json.dumps(
            identify(arguments.root, arguments.jit_root),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (IdentityError, OSError) as error:
        print(f"THead Triton identity contract: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error
