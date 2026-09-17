"""Python package identity without initializing a GPU or importing Torch.

Compilation describes a corex_71 program. Device validation belongs to the
CoreX execution context; loading Torch here initialized the GPU again for every
Graph request, and made offline source emission depend on device availability.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import platform
import sys
import sysconfig
from typing import Any


def _module_path(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.origin:
        raise RuntimeError(f"Python module {name!r} has no file identity")
    path = Path(spec.origin).resolve()
    if not path.is_file():
        raise RuntimeError(f"Python module file does not exist: {name!r}")
    return path


def module_dependency_paths() -> tuple[Path, ...]:
    torch = _module_path("torch")
    extensions = tuple(torch.parent.glob("_C*.so"))
    if len(extensions) != 1:
        raise RuntimeError("cannot identify the selected Torch extension")
    return tuple(
        sorted(
            {
                torch,
                extensions[0],
                torch.parent / "version.py",
                Path(sys.executable).resolve(),
                _module_path("triton"),
                _module_path("yaml"),
            }
        )
    )


def _torch_version() -> str:
    source = (_module_path("torch").parent / "version.py").read_text()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == "__version__" for t in targets):
                value = ast.literal_eval(node.value)
                if isinstance(value, str):
                    return value
    raise RuntimeError("selected Torch version file has no version")


def collect_environment_identity() -> dict[str, Any]:
    import triton
    import yaml

    hashes = {
        path.name + ":" + str(index): hashlib.sha256(path.read_bytes()).hexdigest()
        for index, path in enumerate(module_dependency_paths())
    }
    return {
        "schema_version": 2,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": sys.implementation.cache_tag,
            "soabi": sysconfig.get_config_var("SOABI") or "",
            "byteorder": sys.byteorder,
        },
        "packages": {
            "torch": _torch_version(),
            "triton": str(triton.__version__),
            "pyyaml": str(yaml.__version__),
        },
        "module_file_sha256": dict(sorted(hashes.items())),
        "declared_target": {"backend": "corex", "arch": 71, "warp_size": 64},
    }
