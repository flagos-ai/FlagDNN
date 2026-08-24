"""Stable Python/Torch/Triton environment identity for Iluvatar codegen."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
import sysconfig
from typing import Any


def _module_path(name: str) -> Path:
    module = importlib.import_module(name)
    value = getattr(module, "__file__", None)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Python module {name!r} has no file identity")
    path = Path(value).resolve()
    if not path.is_file():
        raise RuntimeError(f"Python module file does not exist: {name!r}")
    return path


def module_dependency_paths() -> tuple[Path, ...]:
    return tuple(
        sorted(
            {
                _module_path("torch"),
                _module_path("torch._C"),
                _module_path("triton"),
                _module_path("yaml"),
            }
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_environment_identity() -> dict[str, Any]:
    import torch
    import triton
    import yaml

    target = triton.runtime.driver.active.get_current_target()
    module_hashes = {
        path.name + ":" + str(index): _sha256(path)
        for index, path in enumerate(module_dependency_paths())
    }
    return {
        "schema_version": 1,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": sys.implementation.cache_tag,
            "soabi": sysconfig.get_config_var("SOABI") or "",
            "byteorder": sys.byteorder,
        },
        "packages": {
            "torch": str(torch.__version__),
            "triton": str(triton.__version__),
            "pyyaml": str(yaml.__version__),
        },
        "module_file_sha256": dict(sorted(module_hashes.items())),
        "triton_target": {
            "backend": str(target.backend),
            "arch": int(target.arch),
            "warp_size": int(target.warp_size),
        },
    }


def main() -> int:
    print(
        json.dumps(
            collect_environment_identity(),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
