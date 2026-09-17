# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Iluvatar compiler implementation."""

from __future__ import annotations

from .codegen.emit import _compile_nn_stage
from .codegen.emit import _compile_pointwise_stage
from .codegen.emit import _compile_tensor_stage
from .codegen.identity import ARTIFACT_SCHEMA_VERSION
from .codegen.identity import EXECUTION_PROGRAM_VERSION
from .codegen.identity import PROVIDER_NAME
from .codegen.identity import build_compiler_identity
from .codegen.identity import compiler_identity_dependencies as _identity_dependencies
from .codegen.io import _atomic_write
from .codegen.io import _compiler_entry_path
from .dispatch import nn as nn_dispatch
from .dispatch.common import ILUVATAR_WARP_SIZE
from .dispatch.common import LIBTRITON_JIT_GLOBAL_SCRATCH_SIZE
from .dispatch.common import SCHEMA_VERSION
from .dispatch.common import SUPPORTED_TARGET
from .dispatch.common import _require_list
from .dispatch.common import _require_object
from .dispatch.graph import _fuse_conv_bias_relu
from .dispatch.graph import _nn_workspace_layout
from .dispatch.graph import _workspace_layout
from .dispatch.graph_tensor import _manifest_tensor
from .dispatch.graph_tensor import _parse_tensor_table
from .dispatch.pointwise import POINTWISE_SCHEMAS
from .dispatch.selection import _parse_node
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
    return build_compiler_identity(target_name, execution_engine)


def compiler_identity_dependencies(
    target_name: str, execution_engine: str = "libtriton_jit"
) -> tuple[Path, ...]:
    return _identity_dependencies(target_name, execution_engine)


def compile_request(
    request_path: Path,
    output_directory: Path,
    execution_engine: str = "libtriton_jit",
) -> dict[str, Any]:
    if execution_engine != "libtriton_jit":
        raise ValueError("Iluvatar supports only the libtriton_jit engine")
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
    if request.get("backend") != "iluvatar":
        raise ValueError("Iluvatar provider received another backend")
    target_name = request.get("target")
    if target_name != SUPPORTED_TARGET:
        raise ValueError(f"Iluvatar provider supports only target {SUPPORTED_TARGET!r}")

    identity = compiler_identity(target_name, execution_engine)
    if request.get("compiler_identity") != identity["identity_sha256"]:
        raise ValueError("request compiler identity does not match provider")
    build_options = _require_object(request.get("build_options"), "build_options")
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
                raise ValueError("graph nodes are not in topological execution order")
    if not any(not tensor["virtual"] for tensor in tensor_registry.values()):
        raise ValueError("graph has no externally bound tensors")
    if not has_external_output:
        raise ValueError("graph has no non-virtual output tensor")

    parsed_nodes = _fuse_conv_bias_relu(parsed_nodes, tensor_registry)
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
        raise ValueError("Iluvatar execution workspace size overflows int64")
    workspace_size = packed_workspace_size + LIBTRITON_JIT_GLOBAL_SCRATCH_SIZE
    if workspace_size > 2**63 - workspace_alignment:
        raise ValueError("Iluvatar aligned workspace requirement overflows int64")
    if workspace_size:
        workspace_size = (
            (workspace_size + workspace_alignment - 1)
            // workspace_alignment
            * workspace_alignment
        )
    else:
        workspace_alignment = 1
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
        if operation in nn_dispatch.SUPPORTED_OPERATIONS:
            plan = nn_plans[node["id"]]
            local_stage_ids: dict[str, int] = {}
            for configuration in plan.stages:
                stage_id = len(stages)
                dependencies = (
                    external_dependencies if not local_stage_ids else [stage_id - 1]
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
                raise ValueError("Iluvatar NN node produced no execution stages")
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
    external_uids = sorted(
        uid for uid, tensor in tensor_registry.items() if not tensor["virtual"]
    )
    manifest_tensors = [
        _manifest_tensor(tensor_registry[uid], workspace)
        for uid in sorted(tensor_registry)
    ]
    manifest: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "flagdnn_execution_program",
        "flagdnn_version": flagdnn_version,
        "graph_ir_schema_version": SCHEMA_VERSION,
        "backend_abi_version": 2,
        "backend": "iluvatar",
        "target": target_name,
        "warp_size": ILUVATAR_WARP_SIZE,
        "engine": "libtriton_jit",
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "source_sha256": source_hash,
        "compiler": {
            **identity,
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "torch_loaded": "torch" in sys.modules,
        },
        "external_uids": external_uids,
        "tensor_count": len(manifest_tensors),
        "tensors": manifest_tensors,
        "workspace": {
            "size": workspace_size,
            "alignment": workspace_alignment,
        },
        "program": {
            "schema_version": EXECUTION_PROGRAM_VERSION,
            "stage_count": len(stages),
            "stages": stages,
        },
    }
    _atomic_write(
        output_directory / "manifest.json",
        (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "success",
        "backend": "iluvatar",
        "provider": PROVIDER_NAME,
        "node_count": node_count,
        "stage_count": len(stages),
        "target": target_name,
        "target_backend": "corex",
        "warp_size": ILUVATAR_WARP_SIZE,
        "artifact_directory": str(output_directory),
        "workspace_size": workspace_size,
        "workspace_alignment": workspace_alignment,
        "execution_engine": "libtriton_jit",
        "triton_version": triton.__version__,
        "torch_loaded": "torch" in sys.modules,
    }
