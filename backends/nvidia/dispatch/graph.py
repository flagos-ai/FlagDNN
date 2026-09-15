"""Graph fusion, pipeline routing, and live-workspace allocation."""

from __future__ import annotations

from functools import partial
from typing import Any

from .attention_gradients import split_attention_gradients
from .attention_pipeline import (
    _expand_sdpa_backward_group,
    _expand_sdpa_fp8_backward_group,
    _expand_sdpa_fp8_forward_group,
)
from .index import _expand_concatenate_group
from .common import ExecutionGroup, _tensor_storage_size
from .pipeline_dgrad import _expand_dgrad_group
from .pipeline_fprop import _expand_fprop_group
from .pipeline_matmul import _expand_matmul_group
from .pipeline_wgrad import _expand_wgrad_group


def _workspace_layout(
    tensors: dict[int, dict[str, Any]],
    execution_groups: list[dict[str, Any]],
) -> tuple[dict[int, tuple[int, int]], int]:
    live_uids = {
        tensor["uid"]
        for group in execution_groups
        for tensor in group["tensors"]
    }
    alignment = 256
    offset = 0
    result: dict[int, tuple[int, int]] = {}
    for uid, tensor in tensors.items():
        if not tensor["virtual"] or uid not in live_uids:
            continue
        offset = (offset + alignment - 1) // alignment * alignment
        size = _tensor_storage_size(tensor)
        result[uid] = (offset, size)
        offset += size
    if offset != 0:
        offset = (offset + alignment - 1) // alignment * alignment
    return result, offset


def _lower_execution_groups(
    parsed_nodes: list[
        tuple[
            int,
            str,
            dict[str, Any],
            list[dict[str, Any]],
            list[int],
            list[int],
        ]
    ],
    tensor_registry: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fuse backend-supported graph patterns into executable kernel groups."""
    consumer_counts: dict[int, int] = {}
    for _, _, _, _, input_uids, _ in parsed_nodes:
        for uid in input_uids:
            consumer_counts[uid] = consumer_counts.get(uid, 0) + 1

    groups: list[dict[str, Any]] = []
    node_index = 0
    while node_index < len(parsed_nodes):
        node = parsed_nodes[node_index]
        if (
            node_index + 2 < len(parsed_nodes)
            and node[1] in {"conv2d_fprop", "convolution_fprop"}
            and parsed_nodes[node_index + 1][1] == "add"
            and parsed_nodes[node_index + 2][1] == "relu"
            and len(node[4]) == 2
            and len(node[5]) == 1
            and len(parsed_nodes[node_index + 1][4]) == 2
            and len(parsed_nodes[node_index + 1][5]) == 1
            and len(parsed_nodes[node_index + 2][4]) == 1
            and len(parsed_nodes[node_index + 2][5]) == 1
            and len(tensor_registry[node[5][0]]["dimensions"]) >= 3
        ):
            convolution = node
            bias_add = parsed_nodes[node_index + 1]
            relu = parsed_nodes[node_index + 2]
            convolution_output = convolution[5][0]
            bias_add_output = bias_add[5][0]
            bias_inputs = bias_add[4]
            bias_uid = (
                bias_inputs[1]
                if len(bias_inputs) == 2
                and bias_inputs[0] == convolution_output
                else (
                    bias_inputs[0]
                    if len(bias_inputs) == 2
                    and bias_inputs[1] == convolution_output
                    else None
                )
            )
            convolution_output_tensor = tensor_registry[convolution_output]
            bias_add_output_tensor = tensor_registry[bias_add_output]
            convolution_output_dimensions = convolution_output_tensor[
                "dimensions"
            ]
            expected_bias_dimensions = [
                1,
                convolution_output_dimensions[1],
            ] + [1] * (len(convolution_output_dimensions) - 2)
            relu_attributes = relu[2]
            is_standard_relu = (
                relu_attributes.get("negative_slope", 0) == 0
                and relu_attributes.get("relu_lower_clip", 0) == 0
                and relu_attributes.get("relu_lower_clip_slope", 0) == 0
                and relu_attributes.get("relu_upper_clip_set", False) is False
            )
            can_fuse = (
                bias_uid is not None
                and relu[4] == [bias_add_output]
                and convolution_output_tensor["virtual"]
                and bias_add_output_tensor["virtual"]
                and consumer_counts.get(convolution_output) == 1
                and consumer_counts.get(bias_add_output) == 1
                and bias_add[2].get("alpha", 1) == 1
                and tensor_registry[bias_uid]["dimensions"]
                == expected_bias_dimensions
                and is_standard_relu
            )
            if can_fuse:
                parameters = dict(convolution[2])
                parameters["_fused_bias_relu"] = True
                groups.append(
                    {
                        "source_node_ids": [
                            convolution[0],
                            bias_add[0],
                            relu[0],
                        ],
                        "operation": convolution[1],
                        "parameters": parameters,
                        "tensors": [
                            convolution[3][0],
                            convolution[3][1],
                            tensor_registry[bias_uid],
                            relu[3][-1],
                        ],
                        "input_uids": [
                            convolution[4][0],
                            convolution[4][1],
                            bias_uid,
                        ],
                        "output_uids": relu[5],
                    }
                )
                node_index += 3
                continue

        groups.append(
            {
                "source_node_ids": [node[0]],
                "operation": node[1],
                "parameters": node[2],
                "tensors": node[3],
                "input_uids": node[4],
                "output_uids": node[5],
            }
        )
        node_index += 1
    return groups


def _expand_execution_pipelines(
    groups: list[ExecutionGroup],
    tensor_registry: dict[int, dict[str, Any]],
    *,
    architecture: int = 0,
) -> list[ExecutionGroup]:
    """Expand operator plans while owning graph-wide workspace tensor UIDs."""
    result: list[ExecutionGroup] = []
    next_uid = max(tensor_registry) + 1
    expanders = {
        "convolution_fprop": _expand_fprop_group,
        "convolution_dgrad": _expand_dgrad_group,
        "convolution_wgrad": _expand_wgrad_group,
        "matmul": partial(_expand_matmul_group, architecture=architecture),
        "sdpa_backward": _expand_sdpa_backward_group,
    }
    for group in groups:
        operation = group["operation"]
        if operation == "matmul" and group["tensors"][0]["data_type"] in {
            "fp8_e4m3",
            "fp8_e5m2",
        }:
            result.append(dict(group, operation="matmul_fp8"))
        elif group["parameters"].get(
            "input_precision", 0
        ) != 0 and operation in {
            "matmul",
            "convolution_fprop",
            "convolution_dgrad",
            "convolution_wgrad",
        }:
            # Explicit precision bypasses pipelines that may pre-
            # round FP32 data.
            result.append(group)
        elif operation == "concatenate":
            result.extend(_expand_concatenate_group(group))
        elif operation == "sdpa_fp8":
            result.extend(_expand_sdpa_fp8_forward_group(group))
        elif operation == "sdpa_fp8_backward":
            result.extend(_expand_sdpa_fp8_backward_group(group))
        elif operation in expanders:
            stages, next_uid = expanders[operation](
                group, tensor_registry, next_uid
            )
            result.extend(stages)
        else:
            result.append(group)
    return split_attention_gradients(result, tensor_registry)
