"""MThreads compiler implementation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from flagdnn_codegen.kernel_registry import resolve_kernel_source

from .codegen.identity import (
    ARTIFACT_SCHEMA_VERSION,
    EXECUTION_PROGRAM_VERSION,
    PROVIDER_NAME,
    SUPPORTED_ENGINE,
    build_compiler_identity,
)
from .codegen.identity import (
    compiler_identity_dependencies as identity_dependencies,
)
from .codegen.io import _compiler_entry_path, _publish_artifact
from .codegen.resources import _materialize_mthreads_kernel_source

# Keep parser entry points available to existing compiler-contract consumers.
from .dispatch.graph import parse_add_request as parse_add_request
from .dispatch.graph import parse_compiler_request as parse_compiler_request
from .dispatch.graph import (
    parse_convolution_request as parse_convolution_request,
)
from .dispatch.graph import parse_matmul_request as parse_matmul_request
from .dispatch.graph import parse_pointwise_request as parse_pointwise_request
from .dispatch.graph import parse_reduction_request as parse_reduction_request
from .dispatch.selection import _plan, select_source


def compiler_identity(
    target_name: str,
    execution_engine: str = SUPPORTED_ENGINE,
) -> dict[str, Any]:
    return build_compiler_identity(target_name, execution_engine)


def compiler_identity_dependencies(
    target_name: str,
    execution_engine: str = SUPPORTED_ENGINE,
) -> tuple[Path, ...]:
    return identity_dependencies(target_name, execution_engine)


def compile_request(
    request_path: Path,
    output_directory: Path,
    execution_engine: str = SUPPORTED_ENGINE,
) -> dict[str, Any]:
    if execution_engine != SUPPORTED_ENGINE:
        raise ValueError("mthreads requires execution engine libtriton_jit")
    request_bytes = request_path.read_bytes()
    preliminary = json.loads(request_bytes)
    if not isinstance(preliminary, dict):
        raise ValueError("compiler request must be an object")
    target = preliminary.get("target")
    if not isinstance(target, str):
        raise ValueError("compiler request target is invalid")
    identity = compiler_identity(target, execution_engine)
    request = parse_compiler_request(
        request_bytes,
        expected_target=target,
        expected_identity=identity["identity_sha256"],
    )

    candidate, source_relative_path = select_source(request)
    source_path = resolve_kernel_source(_compiler_entry_path(), candidate)
    source_bytes = _materialize_mthreads_kernel_source(source_path, candidate)
    if not source_bytes or len(source_bytes) > (16 << 20):
        raise ValueError("common kernel source bytes are invalid")
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    plan = _plan(request, source_sha256)
    manifest: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "flagdnn_execution_program",
        "flagdnn_version": request.flagdnn_version,
        "backend": "mthreads",
        "target": request.target,
        "engine": execution_engine,
        "request_sha256": request.request_sha256,
        "compiler_identity": request.compiler_identity,
        "source_sha256": source_sha256,
        "workspace_size": plan.workspace_size,
        "workspace_alignment": plan.workspace_alignment,
        "external_binding_uids": list(plan.external_binding_uids),
        "program": {
            "schema_version": EXECUTION_PROGRAM_VERSION,
            "stage_count": len(plan.stages),
            "stages": [stage.to_json() for stage in plan.stages],
        },
        "files": [
            {
                "path": source_relative_path,
                "size": len(source_bytes),
                "sha256": source_sha256,
            }
        ],
    }
    artifact_directory = _publish_artifact(
        output_directory,
        request_path=request_path,
        request_bytes=request_bytes,
        source_relative_path=source_relative_path,
        source_bytes=source_bytes,
        manifest=manifest,
    )
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "success",
        "backend": "mthreads",
        "provider": PROVIDER_NAME,
        "node_count": preliminary["graph"]["node_count"],
        "stage_count": len(plan.stages),
        "target": request.target,
        "artifact_directory": str(artifact_directory),
        "workspace_size": plan.workspace_size,
        "execution_engine": execution_engine,
        "torch_loaded": "torch" in sys.modules,
    }
