"""NVIDIA provider entry: validate requests and assemble execution programs."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json
import re
import sys

from triton.backends.compiler import GPUTarget
import triton

from .dispatch.common import (
    ARTIFACT_SCHEMA_VERSION,
    EXECUTION_PROGRAM_VERSION,
    PROVIDER_NAME,
    PROVIDER_VERSION,
    SCHEMA_VERSION,
    _require_list,
    _require_object,
)
from .codegen.emit import _compile_graph_operation
from .dispatch.graph import (
    _expand_execution_pipelines,
    _lower_execution_groups,
    _workspace_layout,
)
from .codegen.identity import build_compiler_identity
from .codegen.io import _atomic_write, _compiler_entry_path
from .dispatch.tensor import _parse_tensor_table, _tensor_metadata
from .codegen.resources import _program_scratch_size


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
    )


def compile_request(
    request_path: Path,
    output_directory: Path,
    execution_engine: str = "libtriton_jit",
) -> dict[str, Any]:
    if execution_engine != "libtriton_jit":
        raise ValueError(
            "NVIDIA only supports the libtriton_jit execution engine"
        )
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
    if request.get("backend") != "nvidia":
        raise ValueError("NVIDIA provider received another backend")

    target_name = request.get("target")
    if not isinstance(target_name, str) or not target_name.startswith("sm_"):
        raise ValueError("NVIDIA target fingerprint is invalid")
    architecture_text = target_name[3:]
    if not architecture_text.isdigit():
        raise ValueError("NVIDIA target fingerprint is invalid")
    architecture = int(architecture_text)
    if architecture < 50 or architecture > 999:
        raise ValueError("NVIDIA SM architecture is invalid")

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
        or node_count < 1
        or node_count > 1024
    ):
        raise ValueError("graph node_count is invalid")

    parsed_nodes: list[
        tuple[
            int,
            str,
            dict[str, Any],
            list[dict[str, Any]],
            list[int],
            list[int],
        ]
    ] = []
    node_positions: dict[int, int] = {}
    producer_nodes: dict[int, int] = {}
    has_external_output = False
    for node_position, node_value in enumerate(nodes):
        graph_node = _require_object(
            node_value, f"graph.nodes[{node_position}]"
        )
        node_id = graph_node.get("id")
        if (
            isinstance(node_id, bool)
            or not isinstance(node_id, int)
            or node_id < 0
            or node_id in node_positions
        ):
            raise ValueError("graph node ID is invalid or duplicated")
        node_positions[node_id] = node_position
        operation = graph_node.get("type")
        if not isinstance(operation, str):
            raise ValueError("graph node type must be a string")
        attributes = _require_object(
            graph_node.get("attributes"),
            f"graph.nodes[{node_position}].attributes",
        )
        tensor_uids, tensor_metadata, input_count = _tensor_metadata(
            graph_node, operation, tensor_registry
        )
        input_uids = tensor_uids[:input_count]
        output_uids = tensor_uids[input_count:]
        for uid in output_uids:
            if uid in producer_nodes:
                raise ValueError("graph tensor has more than one producer")
            producer_nodes[uid] = node_id
            if not tensor_registry[uid]["virtual"]:
                has_external_output = True
        parsed_nodes.append(
            (
                node_id,
                operation,
                attributes,
                tensor_metadata,
                input_uids,
                output_uids,
            )
        )

    for node_id, _, _, _, input_uids, _ in parsed_nodes:
        node_position = node_positions[node_id]
        for uid in input_uids:
            producer = producer_nodes.get(uid)
            if tensor_registry[uid]["virtual"] and producer is None:
                raise ValueError("virtual tensor input has no producer")
            if (
                producer is not None
                and node_positions[producer] >= node_position
            ):
                raise ValueError(
                    "graph nodes are not in topological execution order"
                )

    if not any(not tensor["virtual"] for tensor in tensor_registry.values()):
        raise ValueError("graph has no externally bound tensors")
    if not has_external_output:
        raise ValueError("graph has no non-virtual output tensor")

    execution_groups = _expand_execution_pipelines(
        _lower_execution_groups(parsed_nodes, tensor_registry),
        tensor_registry,
        architecture=architecture,
    )
    workspace_layout, workspace_size = _workspace_layout(
        tensor_registry, execution_groups
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    target = GPUTarget("cuda", architecture, 32)
    tensor_to_stage: dict[int, int] = {}
    stages: list[dict[str, Any]] = []
    for stage_id, execution_group in enumerate(execution_groups):
        source_node_ids = execution_group["source_node_ids"]
        operation = execution_group["operation"]
        attributes = execution_group["parameters"]
        tensors = execution_group["tensors"]
        input_uids = execution_group["input_uids"]
        dependencies = sorted(
            {
                tensor_to_stage[uid]
                for uid in input_uids
                if uid in tensor_to_stage
            }
        )
        stage = _compile_graph_operation(
            stage_id=stage_id,
            source_node_ids=source_node_ids,
            dependencies=dependencies,
            operation=operation,
            parameters=attributes,
            tensors=tensors,
            workspace_layout=workspace_layout,
            compiler_path=_compiler_entry_path(),
            output_directory=output_directory,
            target=target,
            enable_autotune=enable_autotune,
            execution_engine=execution_engine,
        )
        stages.append(stage)
        for uid in execution_group["output_uids"]:
            tensor_to_stage[uid] = stage_id

    scratch_size = _program_scratch_size(stages)
    if scratch_size:
        workspace_size = ((workspace_size + 255) // 256) * 256 + scratch_size
    if workspace_size > (1 << 63) - 1:
        raise ValueError("execution workspace exceeds the artifact ABI")
    source_hashes = [stage["source_sha256"] for stage in stages]
    combined_source_hash = hashlib.sha256(
        json.dumps(source_hashes, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    manifest: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "flagdnn_execution_program",
        "flagdnn_version": flagdnn_version,
        "backend": "nvidia",
        "target": target_name,
        "graph_node_count": node_count,
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "source_sha256": combined_source_hash,
        "compiler": {
            **identity,
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "torch_loaded": "torch" in sys.modules,
        },
        "workspace_size": workspace_size,
        "program": {
            "schema_version": EXECUTION_PROGRAM_VERSION,
            "stage_count": len(stages),
            "stages": stages,
        },
    }
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    _atomic_write(output_directory / "manifest.json", manifest_bytes)
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "success",
        "backend": "nvidia",
        "provider": PROVIDER_NAME,
        "node_count": node_count,
        "stage_count": len(stages),
        "target": target_name,
        "artifact_directory": str(output_directory),
        "workspace_size": workspace_size,
        "execution_engine": execution_engine,
        "triton_version": triton.__version__,
        "torch_loaded": "torch" in sys.modules,
    }
