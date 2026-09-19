"""Ascend routing for extended operators and exact storage dtype plans.

The operator planners follow the same tensor/attribute contracts as NVIDIA.
Their launch ABI is emitted by the Ascend code generator, without CUDA imports.
"""

from .common import SUPPORTED_OPERATIONS
from .common import (
    BINARY_ELEMENTWISE_MODES,
    UNARY_POINTWISE_MODES,
    _tensor_storage_size,
)
from .pointwise import _pointwise_kernel_configuration
from .index import _index_kernel_configuration, _expand_concatenate_group
from .statistics import (
    _genstats_kernel_configuration,
    _bn_finalize_kernel_configuration,
)
from .normalization_extended import _extended_normalization_configuration
from .position_embedding import _rope_kernel_configuration
from .random import _rng_kernel_configuration
from .resample import _resample_kernel_configuration
from .causal_convolution import _causal_conv1d_kernel_configuration
from .moe_matmul import _moe_matmul_configuration
from .conv_backward import _convolution_backward_kernel_configuration
from .layout import _layout_kernel_configuration
from .attention_forward import _sdpa_forward_kernel_configuration
from .attention_pipeline import _expand_sdpa_backward_group
from .attention_backward import _sdpa_backward_kernel_configuration
from .reduction import configuration as reduction_configuration
from .matmul import _matmul_kernel_configuration
from .conv_fprop import _convolution_kernel_configuration
from .tensor import _parse_tensor_descriptors, _tensor_metadata


def handles(graph):
    return (
        any(
            n["type"] not in SUPPORTED_OPERATIONS
            or n["attributes"].get("input_precision", 0)
            for n in graph["nodes"]
        )
        or any(
            n["type"].startswith("reduction_")
            and len({t["data_type"] for t in graph["tensors"]}) > 1
            for n in graph["nodes"]
        )
        or (
            any(t["data_type"] == "boolean" for t in graph["tensors"])
            and any(
                n["type"] in {"identity", "reshape", "transpose", "slice"}
                for n in graph["nodes"]
            )
        )
        or any(
            t["data_type"] not in {"float32", "float16", "bfloat16", "boolean"}
            for t in graph["tensors"]
        )
    )


def configuration(operation, parameters, tensors):
    if parameters.get("input_precision", 0) not in (0, 1):
        raise ValueError("Ascend 910B supports DEFAULT/IEEE precision, not TF32")
    if operation.startswith("reduction_"):
        return "reduction", reduction_configuration(operation, parameters, tensors)
    if operation == "matmul":
        return "matmul", _matmul_kernel_configuration(parameters, tensors)
    if operation == "convolution_fprop":
        return "convolution", _convolution_kernel_configuration(parameters, tensors)
    if (
        operation in BINARY_ELEMENTWISE_MODES
        or operation in UNARY_POINTWISE_MODES
        or operation == "binary_select"
    ):
        source = (
            "unary"
            if operation.endswith("_backward")
            else (
                "binary"
                if operation in BINARY_ELEMENTWISE_MODES
                else "ternary" if operation == "binary_select" else "unary"
            )
        )
        return source, _pointwise_kernel_configuration(operation, parameters, tensors)
    if operation in {"reshape", "transpose", "slice"}:
        return "layout", _layout_kernel_configuration(operation, parameters, tensors)
    if operation in {"concatenate", "gen_index"}:
        return "unary", _index_kernel_configuration(operation, parameters, tensors)
    if operation in {
        "instancenorm",
        "adalayernorm",
        "instancenorm_backward",
        "adalayernorm_backward",
        "layernorm_backward",
        "rmsnorm_backward",
        "batchnorm_backward",
    }:
        return "normalization", _extended_normalization_configuration(
            operation, parameters, tensors
        )
    if operation in {"convolution_dgrad", "convolution_wgrad"}:
        config = _convolution_backward_kernel_configuration(
            operation, parameters, tensors
        )
        return "convolution", config
    if operation == "genstats":
        return "unary", _genstats_kernel_configuration(parameters, tensors)
    if operation == "bn_finalize":
        return "unary", _bn_finalize_kernel_configuration(parameters, tensors)
    if operation in {"rope", "rope_backward"}:
        return "unary", _rope_kernel_configuration(operation, parameters, tensors)
    if operation == "rng":
        return "unary", _rng_kernel_configuration(parameters, tensors)
    if operation == "resample":
        return "unary", _resample_kernel_configuration(parameters, tensors)
    if operation == "causal_conv1d":
        return "causal_convolution", _causal_conv1d_kernel_configuration(
            parameters, tensors
        )
    if operation in {"moe_grouped_matmul", "moe_grouped_matmul_bwd"}:
        return "matmul", _moe_matmul_configuration(operation, parameters, tensors)
    if operation == "sdpa":
        return "attention", _sdpa_forward_kernel_configuration(parameters, tensors)
    if operation == "sdpa_backward":
        return "attention", _sdpa_backward_kernel_configuration(parameters, tensors)
    raise ValueError(f"Ascend has no operator plan for {operation!r}")


def plan_graph(graph):
    tensors = _parse_tensor_descriptors(graph)
    nodes = graph["nodes"]
    if (
        type(graph.get("node_count")) is not int
        or graph["node_count"] != len(nodes)
        or not 1 <= len(nodes) <= 1024
    ):
        raise ValueError("Ascend graph node count is invalid")
    next_uid = max(tensors) + 1
    stages, producers, ids = [], {}, set()
    for node in nodes:
        node_id, op = node["id"], node["type"]
        if type(node_id) is not int or node_id in ids or not 0 <= node_id < len(nodes):
            raise ValueError("Ascend graph node ID is invalid")
        ids.add(node_id)
        uids, metadata, input_count = _tensor_metadata(node, op, tensors)
        inputs, outputs = uids[:input_count], uids[input_count:]
        if any(uid in producers for uid in outputs) or set(inputs) & set(outputs):
            raise ValueError("Ascend tensor has duplicate or aliased producers")
        if any(tensors[uid]["virtual"] and uid not in producers for uid in inputs):
            raise ValueError("Ascend virtual input has no earlier producer")
        group = dict(
            operation=op,
            parameters=dict(node["attributes"]),
            tensors=metadata,
            input_uids=inputs,
            output_uids=outputs,
            source_node_ids=[node_id],
        )
        groups = _expand_concatenate_group(group) if op == "concatenate" else [group]
        if op == "sdpa_backward":
            groups, next_uid = _expand_sdpa_backward_group(group, tensors, next_uid)
        for group in groups:
            group["dependencies"] = sorted(
                {
                    producers[uid]
                    for uid in group["input_uids"] + group["output_uids"]
                    if uid in producers
                }
            )
            group["stage_id"] = len(stages)
            group["source"], group["configuration"] = configuration(
                op, group["parameters"], group["tensors"]
            )
            stages.append(group)
            for uid in group["output_uids"]:
                producers[uid] = group["stage_id"]
        for uid in outputs:
            producers[uid] = len(stages) - 1
    if not any(not tensors[uid]["virtual"] for uid in producers):
        raise ValueError("Ascend graph has no external output")
    workspace, workspace_size = {}, 0
    for uid, tensor in sorted(tensors.items()):
        if tensor["virtual"]:
            workspace_size = (workspace_size + 255) // 256 * 256
            size = _tensor_storage_size(tensor)
            workspace[uid] = (workspace_size, size)
            workspace_size += size
    return stages, workspace, workspace_size
