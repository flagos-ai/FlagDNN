# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic Python and Triton identity for THead PPU code generation."""

from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import sysconfig
from types import ModuleType
from typing import Any

import torch
import triton


_FLAGDNN_CODEGEN_ENVIRONMENT = {
    "FLAGDNN_THEAD_PPU_SDK_ROOT",
    "FLAGDNN_THEAD_RESOURCE_ROOT",
    "FLAGDNN_THEAD_TRITON_JIT_ROOT",
    "FLAGDNN_THEAD_TRITON_ROOT",
}


def _required_file(path: Path, description: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"cannot resolve {description}: {error}") from error
    if not resolved.is_file():
        raise RuntimeError(f"{description} is not a file: {resolved}")
    return resolved


def _module_file(module: ModuleType) -> Path:
    origin = getattr(module, "__file__", None)
    if not isinstance(origin, str) or not origin:
        raise RuntimeError(f"Python module has no file origin: {module.__name__}")
    return _required_file(Path(origin), f"Python module {module.__name__}")


@functools.lru_cache(maxsize=None)
def _sha256_file_state(
    path_text: str, size: int, mtime_ns: int, ctime_ns: int
) -> str:
    del size, mtime_ns, ctime_ns
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    resolved = _required_file(path, "identity dependency")
    status = resolved.stat()
    return _sha256_file_state(
        str(resolved), status.st_size, status.st_mtime_ns, status.st_ctime_ns
    )


def _configured_triton_root() -> Path:
    imported_root = _module_file(triton).parent.parent
    configured = os.environ.get("FLAGDNN_THEAD_TRITON_ROOT", "")
    root = (
        Path(configured).expanduser().resolve(strict=True)
        if configured
        else imported_root
    )
    if not root.is_dir():
        raise RuntimeError(f"Triton root is not a directory: {root}")
    try:
        _module_file(triton).relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            f"imported triton escaped configured Triton root {root}: "
            f"{_module_file(triton)}"
        ) from error
    return root


def _critical_modules() -> dict[str, ModuleType]:
    names = (
        "triton.backends",
        "triton.backends.compiler",
        "triton.backends.nvidia.compiler",
        "triton.backends.nvidia.driver",
        "triton.compiler.compiler",
        "triton.runtime.jit",
        "triton.language.extra.libdevice",
        "triton._C.libtriton",
    )
    return {name: importlib.import_module(name) for name in names}


def _validate_triton(root: Path) -> dict[str, ModuleType]:
    modules = _critical_modules()
    backend_catalog = getattr(modules["triton.backends"], "backends", None)
    if not isinstance(backend_catalog, dict) or "nvidia" not in backend_catalog:
        raise RuntimeError(
            "configured Triton backend catalog has no CUDA codegen backend"
        )
    package_root = _module_file(triton).parent
    for name, module in {"triton": triton, **modules}.items():
        origin = _module_file(module)
        try:
            origin.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                f"loaded Triton module {name!r} escaped {root}: {origin}"
            ) from error
    compiler_api = modules["triton.backends.compiler"]
    target_type = getattr(compiler_api, "GPUTarget", None)
    if target_type is None:
        raise RuntimeError("Triton has no GPUTarget API")
    backend_class = backend_catalog["nvidia"].compiler
    backend = backend_class(target_type("cuda", 80, 32))
    if getattr(backend, "binary_ext", None) != "cubin":
        raise RuntimeError("Triton CUDA compiler binary extension is not cubin")
    compiler_source = _module_file(
        modules["triton.backends.nvidia.compiler"]
    ).read_text(encoding="utf-8")
    if any(
        marker not in compiler_source
        for marker in ("PPU_SDK", "llvm-irformatter", "--ppu-backend-options")
    ):
        raise RuntimeError("Triton CUDA compiler lacks the PPU compatibility path")
    distribution = importlib.metadata.distribution("triton")
    if "ppu" not in distribution.version.lower():
        raise RuntimeError(
            "configured Triton distribution is not the PPU-qualified build"
        )
    return modules


def _package_files(package_root: Path) -> tuple[Path, ...]:
    files = tuple(
        sorted(
            path.resolve()
            for path in package_root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    )
    if not files:
        raise RuntimeError(f"Triton package contains no files: {package_root}")
    return files


def _distribution_files(root: Path) -> tuple[Path, ...]:
    distribution = importlib.metadata.distribution("triton")
    files = tuple(
        sorted(
            Path(distribution.locate_file(entry)).resolve(strict=True)
            for entry in distribution.files or ()
            if entry.parent.name.endswith(".dist-info")
            and entry.name
            not in {"RECORD"}
        )
    )
    if not files or any(
        not path.is_file() or not path.is_relative_to(root) for path in files
    ):
        raise RuntimeError("Triton distribution metadata escaped its configured root")
    return files


def module_dependency_paths() -> tuple[Path, ...]:
    root = _configured_triton_root()
    modules = _validate_triton(root)
    package_root = _module_file(triton).parent
    paths: set[Path] = {
        _required_file(Path(sys.executable), "Python executable"),
        _module_file(torch),
        _module_file(torch._C),
        _module_file(triton),
        *(_module_file(module) for module in modules.values()),
        *(path.resolve() for path in _package_files(package_root)),
        *_distribution_files(root),
    }
    return tuple(sorted(paths))


def _tree_sha256(root: Path, files: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _cache_invalidating_environment() -> dict[str, str]:
    result = {
        name: value
        for name, value in os.environ.items()
        if name in _FLAGDNN_CODEGEN_ENVIRONMENT
        or name.startswith(
            (
                "PPU_",
                "TRITON_",
                "CUDA_",
                "MLIR_",
                "LLVM_",
            )
        )
        or name in {"LD_LIBRARY_PATH", "PYTHONPATH"}
    }
    extension = importlib.import_module("triton._C.libtriton")
    query = getattr(extension, "get_cache_invalidating_env_vars", None)
    if callable(query):
        values = query()
        if isinstance(values, dict):
            for name, value in values.items():
                result[f"triton_cache::{name}"] = str(value)
    return dict(sorted(result.items()))


def collect_environment_identity() -> dict[str, Any]:
    root = _configured_triton_root()
    modules = _validate_triton(root)
    package_root = _module_file(triton).parent
    package_files = _package_files(package_root)
    module_hashes = {
        "torch": sha256_file(_module_file(torch)),
        "torch._C": sha256_file(_module_file(torch._C)),
        "triton": sha256_file(_module_file(triton)),
        **{
            name: sha256_file(_module_file(module))
            for name, module in modules.items()
        },
    }
    return {
        "schema_version": 1,
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": sys.implementation.cache_tag,
            "soabi": sysconfig.get_config_var("SOABI"),
        },
        "torch": {
            "version": str(torch.__version__),
            "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
            "cuda_version": str(torch.version.cuda or ""),
            "module_sha256": module_hashes["torch"],
            "extension_sha256": module_hashes["torch._C"],
        },
        "triton": {
            "root": str(root),
            "package_root": str(package_root),
            "module_version": str(triton.__version__),
            "distribution_version": importlib.metadata.version("triton"),
            "backend_catalog": sorted(
                importlib.import_module("triton.backends").backends
            ),
            "codegen_backend": "nvidia",
            "target_backend": "cuda",
            "binary_extension": "cubin",
            "ppu_compatibility": "cuda",
            "package_tree_sha256": _tree_sha256(
                package_root, package_files
            ),
            "module_sha256": dict(sorted(module_hashes.items())),
        },
        "environment": _cache_invalidating_environment(),
    }


def canonical_environment_json() -> str:
    return json.dumps(
        collect_environment_identity(),
        sort_keys=True,
        separators=(",", ":"),
    )


def main() -> int:
    print(canonical_environment_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
