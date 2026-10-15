# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Stable, inexpensive identity for the Python side of Hygon Triton JIT."""

from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import triton
import torch
import yaml  # type: ignore[import-untyped]


@functools.lru_cache(maxsize=None)
def _sha256_file(path: Path) -> str:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"Python environment input is not a file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_file(module: ModuleType) -> Path:
    origin = getattr(module, "__file__", None)
    if not isinstance(origin, str) or not origin:
        raise RuntimeError(
            f"Python module has no file origin: {module.__name__}"
        )
    return Path(origin).resolve()


def _distribution_identity(
    package: str, module: ModuleType
) -> list[dict[str, str]]:
    """Fingerprint wheel/editable metadata without hashing a multi-GB tree.

    RECORD distinguishes independently built packages that reuse a version
    string; direct_url.json distinguishes editable/local installations.  The
    imported module and ABI extension bytes are fingerprinted separately.
    """

    module_file = _module_file(module)
    result: list[dict[str, str]] = []
    for metadata_directory in _distribution_directories(package, module):
        distribution = importlib.metadata.Distribution.at(metadata_directory)
        name = distribution.metadata.get("Name", package)
        value = {
            "name": name,
            "version": distribution.version,
        }
        for metadata_name in ("METADATA", "RECORD", "direct_url.json"):
            text = distribution.read_text(metadata_name)
            if text is not None:
                value[f"{metadata_name.lower()}_sha256"] = hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest()
        result.append(value)
    if not result:
        raise RuntimeError(
            f"Cannot bind {package!r} distribution metadata to {module_file}"
        )
    return sorted(result, key=lambda value: (value["name"], value["version"]))


def _distribution_directories(
    package: str, module: ModuleType
) -> tuple[Path, ...]:
    module_file = _module_file(module)
    search_root = module_file.parent.parent
    relative_module = module_file.relative_to(search_root).as_posix()
    result: list[Path] = []
    for metadata_directory in sorted(search_root.glob("*.dist-info")):
        top_level = metadata_directory / "top_level.txt"
        owners = (
            set(top_level.read_text(encoding="utf-8").splitlines())
            if top_level.is_file()
            else set()
        )
        record = metadata_directory / "RECORD"
        record_owns_module = False
        if not owners and record.is_file():
            with record.open(
                "r", encoding="utf-8", errors="replace"
            ) as source:
                record_owns_module = any(
                    line.startswith(f"{relative_module},") for line in source
                )
        if package in owners or record_owns_module:
            result.append(metadata_directory.resolve())
    if not result:
        raise RuntimeError(
            f"Cannot bind {package!r} distribution metadata to {module_file}"
        )
    return tuple(result)


def _package_tree_paths(root: Path) -> tuple[Path, ...]:
    root = root.resolve()
    paths = sorted(
        (
            path.resolve()
            for path in root.rglob("*")
            if "__pycache__" not in path.parts
            and (not path.is_file() or path.suffix not in {".pyc", ".pyo"})
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not any(path.is_file() for path in paths):
        raise RuntimeError(f"Python package tree is empty: {root}")
    return (root, *paths)


def _package_tree_sha256(root: Path) -> str:
    """
    Fingerprint executable Triton package content, excluding bytecode caches.
    """

    root = root.resolve()
    digest = hashlib.sha256()
    files = [path for path in _package_tree_paths(root) if path.is_file()]
    if not files:
        raise RuntimeError(f"Python package tree is empty: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def runtime_python_environment_dependency_paths() -> tuple[Path, ...]:
    """Return files/directories whose metadata guards the cached identity."""

    critical_triton_modules = {
        name: importlib.import_module(name)
        for name in (
            "triton.compiler.compiler",
            "triton.compiler.code_generator",
            "triton.runtime.jit",
            "triton.backends.hcu.compiler_hcu",
            "triton.backends.hcu.driver",
            "triton._C.libtriton",
        )
    }
    modules: dict[str, ModuleType] = {
        "torch": torch,
        "torch._C": torch._C,
        "triton": triton,
        "yaml": yaml,
        **critical_triton_modules,
    }
    try:
        from yaml import _yaml  # type: ignore[attr-defined]

        modules["yaml._yaml"] = _yaml
    except ImportError:
        pass

    paths: set[Path] = {
        Path(sys.executable).resolve(),
        *(_module_file(module) for module in modules.values()),
    }
    triton_root = _module_file(triton).parent
    paths.update(_package_tree_paths(triton_root))
    for package, module in (
        ("torch", torch),
        ("triton", triton),
        ("yaml", yaml),
    ):
        search_root = _module_file(module).parent.parent.resolve()
        paths.add(search_root)
        for directory in _distribution_directories(package, module):
            paths.add(directory)
            for name in (
                "top_level.txt",
                "METADATA",
                "RECORD",
                "direct_url.json",
            ):
                paths.add((directory / name).resolve())
    return tuple(sorted(paths))


def _validate_loaded_package_modules(package: str, root: Path) -> None:
    """
    Reject a sys.modules package assembled from more than one environment.
    """

    root = root.resolve()

    def require_within(path: str, module_name: str) -> None:
        candidate = Path(path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                (
                    f"Loaded {module_name!r} comes from {candidate}, outside "
                    f"{root}"
                )
            ) from error

    prefix = f"{package}."
    for name, module in tuple(sys.modules.items()):
        if module is None or (name != package and not name.startswith(prefix)):
            continue
        origin = getattr(module, "__file__", None)
        if isinstance(origin, str) and origin:
            require_within(origin, name)
        paths = getattr(module, "__path__", ())
        for path in paths:
            require_within(str(path), name)


def runtime_python_environment() -> dict[str, Any]:
    """Return the ABI/content inputs that can change HCU JIT compilation."""

    critical_triton_modules = {
        name: importlib.import_module(name)
        for name in (
            "triton.compiler.compiler",
            "triton.compiler.code_generator",
            "triton.runtime.jit",
            "triton.backends.hcu.compiler_hcu",
            "triton.backends.hcu.driver",
            "triton._C.libtriton",
        )
    }
    triton_root = _module_file(triton).parent
    _validate_loaded_package_modules("triton", triton_root)
    module_files = {
        "torch": _sha256_file(_module_file(torch)),
        "torch._C": _sha256_file(_module_file(torch._C)),
        "triton": _sha256_file(_module_file(triton)),
        "yaml": _sha256_file(_module_file(yaml)),
    }
    module_files.update(
        {
            name: _sha256_file(_module_file(module))
            for name, module in critical_triton_modules.items()
        }
    )
    try:
        from yaml import _yaml  # type: ignore[attr-defined]

        module_files["yaml._yaml"] = _sha256_file(_module_file(_yaml))
    except ImportError:
        pass

    return {
        "schema_version": 2,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_cache_tag": sys.implementation.cache_tag,
        "torch_version": str(torch.__version__),
        "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "torch_hip_version": str(torch.version.hip or ""),
        "triton_version": str(triton.__version__),
        "triton_package_tree_sha256": _package_tree_sha256(triton_root),
        "yaml_version": str(yaml.__version__),
        "module_file_sha256": module_files,
        "distributions": {
            "torch": _distribution_identity("torch", torch),
            "triton": _distribution_identity("triton", triton),
            "yaml": _distribution_identity("yaml", yaml),
        },
    }


def canonical_python_environment_json() -> str:
    return json.dumps(
        runtime_python_environment(),
        sort_keys=True,
        separators=(",", ":"),
    )


def main() -> int:
    print(canonical_python_environment_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
