# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Strict request I/O, atomic publication and stable resource paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json
import os
import tempfile
from ..dispatch.common import (
    _require_object,
)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON key: {name}")
        result[name] = value
    return result


def _load_request(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        request_bytes = path.read_bytes()
        value = json.loads(
            request_bytes,
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read compiler request: {error}") from error
    return _require_object(value, "request"), request_bytes


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    """Publish atomically without following a destination link."""

    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"artifact output is not a regular file: {path}")
        if path.read_bytes() == data:
            return
        raise ValueError(
            f"artifact output already contains different bytes: {path}"
        )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as output:
            temporary_name = output.name
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(encoded)


def compiler_entry_path() -> Path:
    return Path(__file__).resolve().parents[1] / "compiler.py"
