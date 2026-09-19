"""Ascend codegen io implementation."""

from __future__ import annotations

import ast
import hashlib
import os
from ..dispatch.common import (
    _SHA256,
)
from pathlib import (
    Path,
)


def _compiler_entry_path() -> Path:
    import flagdnn_codegen

    return Path(flagdnn_codegen.__file__).resolve().with_name("main.py")


def _write_immutable(path: Path, data: bytes, description: str) -> None:
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != data:
            raise ValueError(f"existing {description} violates content identity")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError:
        if not path.is_file() or path.is_symlink() or path.read_bytes() != data:
            raise ValueError(f"concurrent {description} publication disagrees")


def _materialize_source(
    *,
    output_directory: Path,
    source_bytes: bytes,
    compiler_identity_sha256: str,
) -> tuple[str, str]:
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if _SHA256.fullmatch(compiler_identity_sha256) is None:
        raise ValueError("Ascend compiler identity is not canonical SHA-256")
    filename = f"source-{compiler_identity_sha256}-{source_sha256}.py"
    _write_immutable(
        output_directory / filename, source_bytes, "materialized kernel source"
    )
    return filename, source_sha256


def _validate_non_tle_kernel_source(source_bytes: bytes) -> None:
    try:
        source = source_bytes.decode("utf-8")
        tree = ast.parse(source)
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ValueError("Ascend kernel source is not valid UTF-8 Python") from error

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "triton.experimental" and any(
                alias.name in {"tle", "*"} for alias in node.names
            ):
                raise ValueError(
                    "FlagDNN Ascend kernels must not depend on optional Triton TLE"
                )
            modules = (module,)
        else:
            continue
        if any(
            module == "triton.experimental.tle"
            or module.startswith("triton.experimental.tle.")
            for module in modules
        ):
            raise ValueError(
                "FlagDNN Ascend kernels must not depend on optional Triton TLE"
            )
