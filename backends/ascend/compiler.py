"""Ascend compiler implementation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from .dispatch.extended import handles as handles_extended
from .codegen.extended import emit as emit_extended
from .codegen.persistent import emit as emit_persistent
from .codegen.io import _compiler_entry_path, _write_immutable
from .compiler_identity import build_compiler_identity, validate_target_name
from .dispatch.common import (
    ARTIFACT_SCHEMA_VERSION,
    EXECUTION_PROGRAM_VERSION,
    LAUNCH_ABI,
    PROVIDER_NAME,
    PROVIDER_VERSION,
    SCHEMA_VERSION,
    _SEMANTIC_VERSION,
    require_object,
)
from .dispatch.selection import _validate_build_options
from .tuning_decoder import canonical_json_bytes


def compiler_identity(
    target_name: str,
    execution_engine: str = "libtriton_jit",
) -> dict[str, Any]:
    return build_compiler_identity(
        target_name,
        execution_engine,
        provider_path=Path(__file__),
        compiler_entry=_compiler_entry_path(),
        provider_name=PROVIDER_NAME,
        provider_version=PROVIDER_VERSION,
        graph_schema_version=SCHEMA_VERSION,
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        execution_program_version=EXECUTION_PROGRAM_VERSION,
        launch_abi=LAUNCH_ABI,
    )


def compile_request(
    request_path: Path,
    output_directory: Path,
    execution_engine: str = "libtriton_jit",
) -> dict[str, Any]:
    if execution_engine != "libtriton_jit":
        raise ValueError("Ascend supports only the libtriton_jit engine")
    request_bytes = request_path.read_bytes()
    request = require_object(json.loads(request_bytes), "request")
    if request.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Ascend compiler requires Graph IR schema v3")
    flagdnn_version = request.get("flagdnn_version")
    if (
        not isinstance(flagdnn_version, str)
        or _SEMANTIC_VERSION.fullmatch(flagdnn_version) is None
    ):
        raise ValueError("request FlagDNN version is invalid")
    if request.get("backend") != "ascend":
        raise ValueError("Ascend provider received another backend")
    target_name = validate_target_name(request.get("target"))
    identity = compiler_identity(target_name, execution_engine)
    if request.get("compiler_identity") != identity["identity_sha256"]:
        raise ValueError("request compiler identity does not match provider")

    enable_autotune = _validate_build_options(request.get("build_options"))
    graph = require_object(request.get("graph"), "graph")
    extended = handles_extended(graph)
    if extended:
        stages, workspace_size = emit_extended(graph, output_directory, identity)
    else:
        stages, workspace_size = emit_persistent(
            graph, target_name, identity, enable_autotune, output_directory
        )
    node_count = graph["node_count"]

    combined_source_sha256 = hashlib.sha256(
        canonical_json_bytes(
            [stage["kernel"]["source_sha256"] for stage in stages],
            "Ascend stage source hashes",
        )
    ).hexdigest()
    manifest: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "flagdnn_execution_program",
        "flagdnn_version": flagdnn_version,
        "backend": "ascend",
        "target": target_name,
        "graph_node_count": node_count,
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "source_sha256": combined_source_sha256,
        "compiler": identity,
        "workspace_size": workspace_size,
        "program": {
            "schema_version": 4 if extended else EXECUTION_PROGRAM_VERSION,
            "stage_count": len(stages),
            "stages": stages,
        },
    }
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    _write_immutable(
        output_directory / "manifest.json", manifest_bytes, "Ascend manifest"
    )
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "success",
        "backend": "ascend",
        "provider": PROVIDER_NAME,
        "node_count": node_count,
        "stage_count": len(stages),
        "target": target_name,
        "artifact_directory": str(output_directory),
        "workspace_size": workspace_size,
        "execution_engine": execution_engine,
        "launch_abi": LAUNCH_ABI,
    }
