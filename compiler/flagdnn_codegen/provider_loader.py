"""Discover compiler providers owned by platform backend bundles.

The generic compiler knows only the provider protocol.  Platform modules live
under backends/<name>/compiler.py in both source and installed layouts.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import re
import sys
from types import ModuleType

_BACKEND_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_PROVIDER_CACHE: dict[str, ModuleType] = {}


def _backend_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    configured = os.environ.get("FLAGDNN_BACKEND_ROOT", "")
    for value in configured.split(os.pathsep):
        if value:
            roots.append(Path(value).expanduser())

    # Source layout:
    #   <root>/compiler/flagdnn_codegen/provider_loader.py
    #   <root>/backends/<name>/compiler.py
    #
    # Installed layout:
    #   <share>/flagdnn/compiler/flagdnn_codegen/provider_loader.py
    #   <share>/flagdnn/backends/<name>/compiler.py
    roots.append(Path(__file__).resolve().parents[2] / "backends")

    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return tuple(unique)


def _provider_path(backend: str) -> Path:
    if not isinstance(backend, str) or not _BACKEND_NAME.fullmatch(backend):
        raise ValueError("compiler backend name is invalid")

    searched: list[str] = []
    for root in _backend_roots():
        candidate = root / backend / "compiler.py"
        searched.append(str(candidate))
        if candidate.is_file():
            return candidate
    raise ValueError(
        f"no compiler provider is registered for backend={backend!r}; "
        f"searched {', '.join(searched)}"
    )


def _load_provider(backend: str, path: Path) -> ModuleType:
    package_name = f"_flagdnn_backend_{backend}"
    module_name = f"{package_name}.compiler"
    package = ModuleType(package_name)
    package.__file__ = str(path.parent)
    package.__package__ = package_name
    package.__path__ = [str(path.parent)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        sys.modules.pop(package_name, None)
        raise RuntimeError(f"cannot load compiler provider: {path}")

    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        sys.modules.pop(package_name, None)
        raise

    for function_name in ("compiler_identity", "compile_request"):
        if not callable(getattr(module, function_name, None)):
            raise RuntimeError(
                f"compiler provider {backend!r} does not implement "
                f"{function_name}()"
            )
    return module


def get_provider(backend: str) -> ModuleType:
    cached = _PROVIDER_CACHE.get(backend)
    if cached is not None:
        return cached
    provider = _load_provider(backend, _provider_path(backend))
    _PROVIDER_CACHE[backend] = provider
    return provider


def available_backends() -> tuple[str, ...]:
    result: set[str] = set()
    for root in _backend_roots():
        if not root.is_dir():
            continue
        for entry in root.iterdir():
            if (
                entry.is_dir()
                and _BACKEND_NAME.fullmatch(entry.name)
                and (entry / "compiler.py").is_file()
            ):
                result.add(entry.name)
    return tuple(sorted(result))
