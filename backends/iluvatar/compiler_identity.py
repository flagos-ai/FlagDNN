"""Deterministic identity closure for the Iluvatar Triton provider."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from flagdnn_codegen.kernel_registry import (
    iter_kernel_candidates,
    iter_kernel_registry_sources,
    resolve_kernel_source,
    resolve_tuning_source,
)

from .python_environment_identity import (
    collect_environment_identity,
    module_dependency_paths,
)

PROVIDER_NAME = "iluvatar_triton"
PROVIDER_VERSION = "1"
ARTIFACT_SCHEMA_VERSION = 1
EXECUTION_PROGRAM_VERSION = 1
GRAPH_IR_SCHEMA_VERSION = 3


def _provider_directory() -> Path:
    return Path(__file__).resolve().parent


def _corex_root() -> Path:
    configured = os.environ.get("FLAGDNN_ILUVATAR_COREX_ROOT", "")
    candidate = Path(configured or "/usr/local/corex").expanduser().resolve()
    if not (candidate / "include/cuda.h").is_file():
        raise RuntimeError(f"selected CoreX root has no cuda.h: {candidate}")
    return candidate


def _first_file(candidates: tuple[Path, ...], description: str) -> Path:
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    raise RuntimeError(f"cannot locate {description}")


def _first_directory(candidates: tuple[Path, ...], description: str) -> Path:
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_dir():
            return resolved
    raise RuntimeError(f"cannot locate {description}")


def _jit_paths() -> dict[str, Path]:
    root_value = os.environ.get("FLAGDNN_ILUVATAR_TRITON_JIT_ROOT", "")
    config_value = os.environ.get("FLAGDNN_ILUVATAR_TRITON_JIT_DIR", "")
    library_value = os.environ.get("FLAGDNN_ILUVATAR_TRITON_JIT_LIBRARY", "")
    include_value = os.environ.get(
        "FLAGDNN_ILUVATAR_TRITON_JIT_INCLUDE_DIR", ""
    )
    scripts_value = os.environ.get(
        "FLAGDNN_ILUVATAR_TRITON_JIT_SCRIPT_DIR", ""
    )
    root = Path(root_value).expanduser() if root_value else None
    config = _first_file(
        tuple(
            candidate
            for candidate in (
                (
                    Path(config_value) / "TritonJITConfig.cmake"
                    if config_value
                    else None
                ),
                root / "build/TritonJITConfig.cmake" if root else None,
                (
                    root / "lib/cmake/TritonJIT/TritonJITConfig.cmake"
                    if root
                    else None
                ),
                Path("/usr/local/lib/cmake/TritonJIT/TritonJITConfig.cmake"),
            )
            if candidate is not None
        ),
        "IX TritonJITConfig.cmake",
    )
    build_root = (
        config.parent.parent if config.parent.name == "build" else None
    )
    install_root = (
        config.parent.parent.parent.parent
        if config.parent.name == "TritonJIT"
        else None
    )
    library = _first_file(
        tuple(
            candidate
            for candidate in (
                Path(library_value) if library_value else None,
                (
                    build_root / "build/src/libtriton_jit.so"
                    if build_root
                    else None
                ),
                (
                    install_root / "lib/libtriton_jit.so"
                    if install_root
                    else None
                ),
                Path("/usr/local/lib/libtriton_jit.so"),
            )
            if candidate is not None
        ),
        "IX libtriton_jit shared object",
    )
    include = _first_directory(
        tuple(
            candidate
            for candidate in (
                Path(include_value) if include_value else None,
                build_root / "include" if build_root else None,
                install_root / "include" if install_root else None,
                Path("/usr/local/include"),
            )
            if candidate is not None
        ),
        "IX libtriton_jit include directory",
    )
    scripts = _first_directory(
        tuple(
            candidate
            for candidate in (
                Path(scripts_value) if scripts_value else None,
                build_root / "scripts" if build_root else None,
                (
                    install_root / "share/triton_jit/scripts"
                    if install_root
                    else None
                ),
                Path("/usr/local/share/triton_jit/scripts"),
            )
            if candidate is not None
        ),
        "IX libtriton_jit script directory",
    )
    return {
        "config": config,
        "library": library,
        "ix_backend": include / "triton_jit/backends/ix_backend.h",
        "triton_jit_function": include / "triton_jit/triton_jit_function.h",
        "jit_utils": include / "triton_jit/jit_utils.h",
        "standalone_compile": scripts / "standalone_compile.py",
        "gen_ssig": scripts / "gen_ssig.py",
    }


def _corex_paths() -> dict[str, Path]:
    root = _corex_root()
    return {
        "release": root / "release-corex.txt",
        "cuda_header": root / "include/cuda.h",
        "driver": _first_file(
            (root / "lib64/libcuda.so.1", root / "lib/libcuda.so.1"),
            "CoreX Driver library",
        ),
        "clang": _first_file(
            (root / "bin/clang++", root / "bin/clang"),
            "CoreX compiler",
        ),
    }


def _kernel_dependency_paths() -> tuple[Path, ...]:
    compiler_path = _provider_directory() / "compiler.py"
    paths: set[Path] = {
        path.resolve() for path in iter_kernel_registry_sources("iluvatar")
    }
    for candidate in iter_kernel_candidates("iluvatar"):
        paths.add(resolve_kernel_source(compiler_path, candidate).resolve())
        if candidate.tuning is not None:
            paths.add(
                resolve_tuning_source(compiler_path, candidate).resolve()
            )
    return tuple(sorted(paths))


def compiler_identity_dependencies(
    target: str, execution_engine: str
) -> tuple[Path, ...]:
    _validate_request(target, execution_engine)
    paths: set[Path] = set()
    paths.update(_provider_directory().glob("*.py"))
    paths.update(_kernel_dependency_paths())
    paths.update(_corex_paths().values())
    paths.update(_jit_paths().values())
    paths.update(module_dependency_paths())
    result = tuple(sorted(path.resolve() for path in paths))
    for path in result:
        if not path.is_file():
            raise RuntimeError(
                f"compiler identity dependency is missing: {path}"
            )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _driver_version(driver: Path) -> int:
    library = ctypes.CDLL(str(driver))
    version = ctypes.c_int()
    result = int(library.cuDriverGetVersion(ctypes.byref(version)))
    if result != 0 or version.value <= 0:
        raise RuntimeError(f"cuDriverGetVersion failed with CUresult {result}")
    return int(version.value)


def _validate_request(target: str, execution_engine: str) -> None:
    if target != "corex_71":
        raise ValueError("Iluvatar compiler target must be corex_71")
    if execution_engine != "libtriton_jit":
        raise ValueError("Iluvatar supports only libtriton_jit")


def build_compiler_identity(
    target: str, execution_engine: str
) -> dict[str, Any]:
    _validate_request(target, execution_engine)
    corex = _corex_paths()
    jit = _jit_paths()
    config_text = jit["config"].read_text(encoding="utf-8")
    backend = re.search(
        r"set\s*\(\s*TritonJIT_BACKEND\s+\"?([A-Za-z0-9_]+)",
        config_text,
    )
    if backend is None or backend.group(1) != "IX":
        raise RuntimeError("selected TritonJIT package is not IX")
    clang_version = subprocess.run(
        [str(corex["clang"]), "--version"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.splitlines()[0]

    dependency_records = []
    for index, path in enumerate(
        compiler_identity_dependencies(target, execution_engine)
    ):
        dependency_records.append(
            {
                "logical_index": index,
                "basename": path.name,
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    document = {
        "schema_version": 1,
        "provider": PROVIDER_NAME,
        "provider_version": PROVIDER_VERSION,
        "backend": "iluvatar",
        "target": target,
        "warp_size": 64,
        "execution_engine": execution_engine,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "execution_program_version": EXECUTION_PROGRAM_VERSION,
        "graph_ir_schema_version": GRAPH_IR_SCHEMA_VERSION,
        "corex": {
            "release": corex["release"].read_text(encoding="utf-8").strip(),
            "driver_version": _driver_version(corex["driver"]),
            "compiler_version": clang_version,
        },
        "triton_jit_backend": "IX",
        "python_environment": collect_environment_identity(),
        "dependencies": dependency_records,
    }
    canonical = json.dumps(
        document, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "provider": PROVIDER_NAME,
        "provider_version": PROVIDER_VERSION,
        "identity_sha256": hashlib.sha256(canonical).hexdigest(),
    }
