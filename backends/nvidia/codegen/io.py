"""Atomic artifact writes and generated-module loading."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import os


def _load_generated_module(path: Path, operation_index: int = 0):
    module_name = f"_flagdnn_generated_{os.getpid()}_{operation_index}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import generated kernel module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _compiler_entry_path() -> Path:
    import flagdnn_codegen

    return Path(flagdnn_codegen.__file__).resolve().with_name("main.py")
