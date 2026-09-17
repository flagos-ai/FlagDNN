# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Hygon compiler."""

from __future__ import annotations

from .dispatch import nn as compiler_nn
from .dispatch import extended
from .codegen.emit import (
    _compile_nn_stage,
    _compile_pointwise_stage,
    _compile_tensor_stage,
)
from .codegen.identity import build_compiler_identity
from .codegen.identity import compiler_identity_dependency_paths
from .codegen.io import (
    _atomic_write,
    _compiler_entry_path,
)
from .dispatch.common import (
    ARTIFACT_SCHEMA_VERSION,
    EXECUTION_PROGRAM_VERSION,
    HYGON_WARP_SIZE,
    LIBTRITON_JIT_GLOBAL_SCRATCH_SIZE,
    PROVIDER_NAME,
    PROVIDER_VERSION,
    SCHEMA_VERSION,
    SUPPORTED_TARGET,
    _require_list,
    _require_object,
)
from .dispatch.graph import (
    _nn_workspace_layout,
    _parse_node,
    _workspace_layout,
)
from .dispatch.pointwise import (
    POINTWISE_SCHEMAS,
)
from .dispatch.tensor_metadata import (
    _parse_tensor_table,
)
from pathlib import Path
from typing import Any
import hashlib
import json
import re
import sys
import triton


def compiler_identity(
    target_name: str, execution_engine: str = "libtriton_jit"
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
    )


def compiler_identity_dependencies(
    target_name: str, execution_engine: str = "libtriton_jit"
) -> tuple[Path, ...]:
    # Keep this protocol hook side-effect free. compiler_identity() has already
    # validated target/engine and hashed the same resolved resource selection.
    if target_name != SUPPORTED_TARGET:
        raise ValueError(
            f"Hygon provider supports only target {SUPPORTED_TARGET!r}"
        )
    if execution_engine != "libtriton_jit":
        raise ValueError("Hygon supports only the libtriton_jit engine")
    return compiler_identity_dependency_paths(
        provider_path=Path(__file__), compiler_entry=_compiler_entry_path()
    )


def compile_request(
    request_path: Path,
    output_directory: Path,
    execution_engine: str = "libtriton_jit",
) -> dict[str, Any]:
    if execution_engine != "libtriton_jit":
        raise ValueError("Hygon supports only the libtriton_jit engine")
    request_bytes = request_path.read_bytes()
    request = _require_object(json.loads(request_bytes), "request")
    if request.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported request schema_version")
    flagdnn_version = request.get("flagdnn_version")
    if (
        not isinstance(flagdnn_version, str)
        or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", flagdnn_version) is None
    ):
        raise ValueError("request FlagDNN version is invalid")
    if request.get("backend") != "hygon":
        raise ValueError("Hygon provider received another backend")
    target_name = request.get("target")
    if target_name != SUPPORTED_TARGET:
        raise ValueError(
            f"Hygon provider supports only target {SUPPORTED_TARGET!r}"
        )

    identity = compiler_identity(target_name, execution_engine)
    if request.get("compiler_identity") != identity["identity_sha256"]:
        raise ValueError("request compiler identity does not match provider")
    build_options = _require_object(
        request.get("build_options"), "build_options"
    )
    enable_autotune = build_options.get("autotune", False)
    if not isinstance(enable_autotune, bool):
        raise ValueError("build_options.autotune must be a boolean")

    graph = _require_object(request.get("graph"), "graph")
    tensor_registry = _parse_tensor_table(graph)
    nodes = _require_list(graph.get("nodes"), "graph.nodes")
    node_count = graph.get("node_count")
    if (
        isinstance(node_count, bool)
        or not isinstance(node_count, int)
        or node_count != len(nodes)
        or not 1 <= node_count <= 1024
    ):
        raise ValueError("graph node_count is invalid")

    parsed_nodes: list[dict[str, Any]] = []
    node_positions: dict[int, int] = {}
    producer_nodes: dict[int, int] = {}
    has_external_output = False
    for position, node_value in enumerate(nodes):
        node = _parse_node(node_value, position, node_count, tensor_registry)
        if node["id"] in node_positions:
            raise ValueError("graph node IDs must be unique")
        node_positions[node["id"]] = position
        for uid in node["output_uids"]:
            if uid in producer_nodes:
                raise ValueError("graph tensor has more than one producer")
            producer_nodes[uid] = node["id"]
            if not tensor_registry[uid]["virtual"]:
                has_external_output = True
        parsed_nodes.append(node)

    for node in parsed_nodes:
        position = node_positions[node["id"]]
        for uid in node["input_uids"]:
            producer = producer_nodes.get(uid)
            if tensor_registry[uid]["virtual"] and producer is None:
                raise ValueError("virtual tensor input has no producer")
            if producer is not None and node_positions[producer] >= position:
                raise ValueError(
                    "graph nodes are not in topological execution order"
                )
    if not any(not tensor["virtual"] for tensor in tensor_registry.values()):
        raise ValueError("graph has no externally bound tensors")
    if not has_external_output:
        raise ValueError("graph has no non-virtual output tensor")

    workspace, graph_workspace_size = _workspace_layout(tensor_registry)
    nn_plans, nn_workspaces, packed_workspace_size = _nn_workspace_layout(
        parsed_nodes, graph_workspace_size, max(tensor_registry)
    )
    workspace_alignment = 256
    for _, _, alignment in workspace.values():
        workspace_alignment = max(workspace_alignment, alignment)
    for named_workspace in nn_workspaces.values():
        for tensor in named_workspace.values():
            workspace_alignment = max(workspace_alignment, tensor["alignment"])
    if packed_workspace_size > 2**63 - 1 - LIBTRITON_JIT_GLOBAL_SCRATCH_SIZE:
        raise ValueError("Hygon execution workspace size overflows int64")
    workspace_size = packed_workspace_size + LIBTRITON_JIT_GLOBAL_SCRATCH_SIZE
    if workspace_size > 2**63 - workspace_alignment:
        raise ValueError("Hygon aligned workspace requirement overflows int64")
    output_directory.mkdir(parents=True, exist_ok=True)
    compiler_path = _compiler_entry_path()
    tensor_to_stage: dict[int, int] = {}
    stages: list[dict[str, Any]] = []
    for node in parsed_nodes:
        external_dependencies = sorted(
            {
                tensor_to_stage[uid]
                for uid in node["input_uids"]
                if uid in tensor_to_stage
            }
        )
        operation = node["operation"]
        if extended.handles(node):
            for expanded in extended.expand_node(node):
                stage_id = len(stages)
                stages.append(
                    _compile_tensor_stage(
                        stage_id=stage_id,
                        node=expanded,
                        dependencies=external_dependencies,
                        workspace=workspace,
                        compiler_path=compiler_path,
                        output_directory=output_directory,
                        enable_autotune=enable_autotune,
                    )
                )
                external_dependencies = [stage_id]
            producer_stage = len(stages) - 1
        elif operation in compiler_nn.SUPPORTED_OPERATIONS:
            plan = nn_plans[node["id"]]
            local_stage_ids: dict[str, int] = {}
            for configuration in plan.stages:
                stage_id = len(stages)
                dependencies = sorted(
                    set(external_dependencies).union(
                        local_stage_ids[name]
                        for name in configuration.dependencies
                    )
                )
                stages.append(
                    _compile_nn_stage(
                        stage_id=stage_id,
                        node=node,
                        configuration=configuration,
                        dependencies=dependencies,
                        workspace=workspace,
                        local_workspace=nn_workspaces[node["id"]],
                        compiler_path=compiler_path,
                        output_directory=output_directory,
                        enable_autotune=enable_autotune,
                    )
                )
                local_stage_ids[configuration.stage_name] = stage_id
            if not local_stage_ids:
                raise ValueError("Hygon NN node produced no execution stages")
            producer_stage = len(stages) - 1
        else:
            stage_id = len(stages)
            compile_stage = (
                _compile_pointwise_stage
                if operation in POINTWISE_SCHEMAS
                else _compile_tensor_stage
            )
            stages.append(
                compile_stage(
                    stage_id=stage_id,
                    node=node,
                    dependencies=external_dependencies,
                    workspace=workspace,
                    compiler_path=compiler_path,
                    output_directory=output_directory,
                    enable_autotune=enable_autotune,
                )
            )
            producer_stage = stage_id
        for uid in node["output_uids"]:
            tensor_to_stage[uid] = producer_stage

    source_hash = hashlib.sha256(
        json.dumps(
            [stage["source_sha256"] for stage in stages],
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    manifest: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "flagdnn_execution_program",
        "flagdnn_version": flagdnn_version,
        "backend": "hygon",
        "target": target_name,
        "graph_node_count": node_count,
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "source_sha256": source_hash,
        "compiler": {
            **identity,
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "torch_loaded": "torch" in sys.modules,
        },
        "workspace_size": workspace_size,
        "workspace_alignment": workspace_alignment,
        "program": {
            "schema_version": EXECUTION_PROGRAM_VERSION,
            "stage_count": len(stages),
            "stages": stages,
        },
    }
    _atomic_write(
        output_directory / "manifest.json",
        (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        ),
    )
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "success",
        "backend": "hygon",
        "provider": PROVIDER_NAME,
        "node_count": node_count,
        "stage_count": len(stages),
        "target": target_name,
        "target_backend": "hip",
        "warp_size": HYGON_WARP_SIZE,
        "artifact_directory": str(output_directory),
        "workspace_size": workspace_size,
        "workspace_alignment": workspace_alignment,
        "execution_engine": "libtriton_jit",
        "triton_version": triton.__version__,
        "torch_loaded": "torch" in sys.modules,
    }
