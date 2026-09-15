"""Persist exact Triton compilation outputs for libtriton_jit's cache API.

Descriptor arguments and Gluon kernels use the dependency's public cache
constructor, which loads the exact compiler outputs through libtriton_jit.
Files belong to the artifact, so clearing the global Triton cache does not
invalidate a compiled graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json

from .io import _atomic_write


def emit_jit_cache(
    compiled: Any, directory: Path, stage: int, variant: str
) -> dict[str, Any]:
    name = compiled.metadata.name
    cache = directory / f"jit_cache_{stage}_{variant}"
    cache.mkdir(exist_ok=True)
    metadata_path = Path(compiled.metadata_group[f"{name}.json"])
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    if metadata["name"] != name or metadata["num_ctas"] != 1:
        raise ValueError("compiled JIT cache metadata is incompatible")
    result: dict[str, Any] = {"directory": cache.name, "name": name}
    for kind, extension, payload in (
        ("metadata", "json", metadata_bytes),
        ("binary", "cubin", compiled.asm["cubin"]),
    ):
        file = cache / f"{name}.{extension}"
        _atomic_write(file, payload)
        result[kind] = {
            "file": file.name,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return result
