# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Iluvatar codegen / io implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import importlib.util
import json
import os


def _compiler_entry_path() -> Path:
    import flagdnn_codegen

    return Path(flagdnn_codegen.__file__).resolve().with_name("main.py")


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _load_generated_module(path: Path, stage_id: int) -> Any:
    module_name = f"_flagdnn_iluvatar_generated_{os.getpid()}_{stage_id}"
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import generated kernel module: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
