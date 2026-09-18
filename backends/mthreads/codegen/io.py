"""MThreads codegen io implementation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any


def _compiler_entry_path() -> Path:
    import flagdnn_codegen

    return Path(flagdnn_codegen.__file__).resolve().with_name("main.py")


def _write_and_sync(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    if path.read_bytes() != payload:
        raise RuntimeError(f"artifact self-read differs: {path.name}")


def _safe_destination(
    output_directory: Path,
    *,
    request_path: Path,
    request_bytes: bytes,
) -> tuple[Path, bool]:
    expanded = output_directory.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    if ".." in expanded.parts:
        raise ValueError("artifact output path contains parent traversal")
    if expanded.is_symlink():
        raise ValueError("artifact output directory must not be a symlink")
    destination = expanded.resolve(strict=False)
    has_core_request = False
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(
                "artifact output directory already exists and is nonempty"
            )
        mode = destination.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError("artifact output directory must not be a symlink")
        entries = list(destination.iterdir())
        if entries:
            expected_request = destination / "request.json"
            if (
                len(entries) != 1
                or entries[0] != expected_request
                or expected_request.is_symlink()
                or not expected_request.is_file()
                or request_path.resolve(strict=True)
                != expected_request.resolve(strict=True)
                or expected_request.read_bytes() != request_bytes
            ):
                raise ValueError(
                    "artifact output directory already exists and is nonempty"
                )
            has_core_request = True
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent = destination.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("artifact output parent is not a directory")
    return parent / destination.name, has_core_request


def _publish_artifact(
    output_directory: Path,
    *,
    request_path: Path,
    request_bytes: bytes,
    source_relative_path: str,
    source_bytes: bytes,
    manifest: dict[str, Any],
) -> Path:
    destination, has_core_request = _safe_destination(
        output_directory,
        request_path=request_path,
        request_bytes=request_bytes,
    )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp.",
            dir=destination.parent,
        )
    )
    published = False
    try:
        source_path = temporary / source_relative_path
        _write_and_sync(source_path, source_bytes)
        source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if (
            source_digest != manifest["source_sha256"]
            or source_digest != manifest["files"][0]["sha256"]
            or source_path.stat().st_size != manifest["files"][0]["size"]
        ):
            raise RuntimeError("artifact source verification failed")
        manifest_bytes = (
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        _write_and_sync(temporary / "manifest.json", manifest_bytes)
        if has_core_request:
            _write_and_sync(temporary / "request.json", request_bytes)
        if (
            json.loads(
                (temporary / "manifest.json").read_text(encoding="utf-8")
            )
            != manifest
        ):
            raise RuntimeError("artifact manifest self-read differs")

        for directory in (source_path.parent, temporary):
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        if has_core_request:
            backup = temporary.with_name(temporary.name + ".request-input")
            os.replace(destination, backup)
            try:
                os.replace(temporary, destination)
            except BaseException:
                os.replace(backup, destination)
                raise
            shutil.rmtree(backup)
        else:
            if destination.exists():
                destination.rmdir()
            os.replace(temporary, destination)
        parent_descriptor = os.open(
            destination.parent, os.O_RDONLY | os.O_DIRECTORY
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        published = True
        return destination
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)
