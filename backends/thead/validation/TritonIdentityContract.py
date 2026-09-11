#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate the installed PPU-aware Triton and CUDA-JIT identity tuple."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from triton_compat import configure_triton_path, ppu_codegen_backend, ppu_distribution


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
    distribution, metadata = ppu_distribution(root)
    return metadata, distribution.version


def _jit_backend(jit_root: Path) -> tuple[Path, str]:
    # Match the source/build and install layouts accepted by the JIT resolver.
    # The identity check also runs during CMake configuration of installed JITs.
    candidates = (
        jit_root / "build/TritonJITConfig.cmake",
        jit_root / "lib/cmake/TritonJIT/TritonJITConfig.cmake",
        jit_root / "lib64/cmake/TritonJIT/TritonJITConfig.cmake",
    )
    selected = next((path for path in candidates if path.is_file()), None)
    if selected is None:
        raise IdentityError("JIT root has no supported build/install configuration")
    config = _inside(str(selected), jit_root, "libtriton_jit configuration")
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

    # Resolve and check the package before executing its native extension.
    # An absent configured package must not load an unrelated system install.
    configure_triton_path(triton_root)
    package_spec = importlib.util.find_spec("triton")
    if package_spec is None:
        raise IdentityError("configured root has no Triton package")
    _inside(package_spec.origin, triton_root, "Triton package")
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
    codegen_backend = ppu_codegen_backend(triton_root)
    catalog = getattr(backend_module, "backends", None)
    if not isinstance(catalog, dict) or codegen_backend not in catalog:
        raise IdentityError(
            f"Triton backend catalog has no CUDA codegen backend: {catalog!r}"
        )
    cuda = catalog[codegen_backend]
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
    if binary_extension != ("hgbin" if codegen_backend == "ppu" else "cubin"):
        raise IdentityError(
            f"unexpected Triton PPU binary extension: {binary_extension!r}"
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
            "codegen_backend": codegen_backend,
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
    except (RuntimeError, OSError) as error:
        print(f"THead Triton identity contract: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error
