# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Iluvatar dispatch / graph implementation."""

from __future__ import annotations

from . import nn as nn_dispatch
from .graph_tensor import _tensor_storage_size
from .pointwise import _pointwise_configuration
from typing import Any


def _fused_conv_bias_relu_node(
    convolution: dict[str, Any],
    bias_add: dict[str, Any],
    relu: dict[str, Any],
    tensor_registry: dict[int, dict[str, Any]],
    consumer_counts: dict[int, int],
) -> dict[str, Any] | None:
    if (
        convolution["operation"] not in ("conv2d_fprop", "convolution_fprop")
        or bias_add["operation"] != "add"
        or relu["operation"] != "relu"
        or int(convolution["derived"]["spatial_rank"]) != 2
    ):
        return None

    convolution_uid = convolution["output_uids"][0]
    biased_uid = bias_add["output_uids"][0]
    if (
        consumer_counts.get(convolution_uid) != 1
        or consumer_counts.get(biased_uid) != 1
        or not tensor_registry[convolution_uid]["virtual"]
        or not tensor_registry[biased_uid]["virtual"]
        or bias_add["input_uids"].count(convolution_uid) != 1
        or relu["input_uids"] != [biased_uid]
    ):
        return None

    bias_uid = next(uid for uid in bias_add["input_uids"] if uid != convolution_uid)
    output_uid = relu["output_uids"][0]
    bias = tensor_registry[bias_uid]
    output = tensor_registry[output_uid]
    convolution_output = tensor_registry[convolution_uid]
    biased_output = tensor_registry[biased_uid]
    out_channels = int(convolution["derived"]["out_channels"])
    if (
        bias["dimensions"] != [1, out_channels, 1, 1]
        or bias["data_type"] != convolution["derived"]["data_type"]
        or convolution_output["dimensions"] != output["dimensions"]
        or biased_output["dimensions"] != output["dimensions"]
        or convolution_output["data_type"] != output["data_type"]
        or biased_output["data_type"] != output["data_type"]
    ):
        return None

    _, _, add_constants, _ = _pointwise_configuration(bias_add)
    _, _, relu_constants, _ = _pointwise_configuration(relu)
    if (
        add_constants["ALPHA"] != 1.0
        or relu_constants["negative_slope"] != 0.0
        or relu_constants["lower_clip"] != 0.0
        or relu_constants["HAS_UPPER_CLIP"] != 0
    ):
        return None

    fused = dict(convolution)
    port_tensors = dict(convolution["port_tensors"])
    port_tensors["bias"] = bias
    port_tensors["output"] = output
    derived = dict(convolution["derived"])
    derived["result"] = output
    fused.update(
        {
            "fused_bias_relu": True,
            "source_node_ids": [
                convolution["id"],
                bias_add["id"],
                relu["id"],
            ],
            "tensors": [
                port_tensors["input"],
                port_tensors["filter"],
                bias,
                output,
            ],
            "tensor_roles": ["input", "filter", "bias", "output"],
            "port_tensors": port_tensors,
            "port_indices": {
                "input": 0,
                "filter": 1,
                "bias": 2,
                "output": 3,
            },
            "input_uids": [
                convolution["input_uids"][0],
                convolution["input_uids"][1],
                bias_uid,
            ],
            "output_uids": [output_uid],
            "derived": derived,
        }
    )
    return fused


def _fuse_conv_bias_relu(
    nodes: list[dict[str, Any]],
    tensor_registry: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    consumer_counts: dict[int, int] = {}
    for node in nodes:
        for uid in node["input_uids"]:
            consumer_counts[uid] = consumer_counts.get(uid, 0) + 1

    result: list[dict[str, Any]] = []
    index = 0
    while index < len(nodes):
        fused = None
        if index + 2 < len(nodes):
            fused = _fused_conv_bias_relu_node(
                nodes[index],
                nodes[index + 1],
                nodes[index + 2],
                tensor_registry,
                consumer_counts,
            )
        if fused is None:
            result.append(nodes[index])
            index += 1
        else:
            result.append(fused)
            index += 3
    return result


def _workspace_layout(
    tensors: dict[int, dict[str, Any]],
) -> tuple[dict[int, tuple[int, int, int]], int]:
    minimum_alignment = 256
    offset = 0
    result: dict[int, tuple[int, int, int]] = {}
    for uid, tensor in tensors.items():
        if not tensor["virtual"]:
            continue
        alignment = max(minimum_alignment, tensor["alignment"])
        offset = (offset + alignment - 1) // alignment * alignment
        size = _tensor_storage_size(tensor)
        result[uid] = (offset, size, alignment)
        offset += size
    if offset:
        offset = (
            (offset + minimum_alignment - 1) // minimum_alignment * minimum_alignment
        )
    return result, offset


def _nn_workspace_layout(
    nodes: list[dict[str, Any]],
    initial_offset: int,
    maximum_tensor_uid: int,
) -> tuple[
    dict[int, nn_dispatch.NodePlan],
    dict[int, dict[str, dict[str, int]]],
    int,
]:
    alignment = nn_dispatch.WORKSPACE_ALIGNMENT
    offset = initial_offset
    next_uid = maximum_tensor_uid + 1
    plans: dict[int, nn_dispatch.NodePlan] = {}
    workspaces: dict[int, dict[str, dict[str, int]]] = {}
    for node in nodes:
        if node["operation"] not in nn_dispatch.SUPPORTED_OPERATIONS:
            continue
        plan = nn_dispatch.plan_kernel_stages(node)
        plans[node["id"]] = plan
        node_alignment = alignment
        for tensor in plan.workspace_tensors:
            if (
                tensor.alignment < alignment
                or tensor.alignment & (tensor.alignment - 1) != 0
            ):
                raise ValueError("Iluvatar NN workspace tensor alignment is invalid")
            node_alignment = max(node_alignment, tensor.alignment)
        if offset > 2**63 - 1 - (node_alignment - 1):
            raise ValueError("Iluvatar NN workspace alignment overflows int64")
        offset = (offset + node_alignment - 1) // node_alignment * node_alignment
        node_base = offset
        named: dict[str, dict[str, int]] = {}
        for tensor in plan.workspace_tensors:
            if tensor.name in named:
                raise ValueError("duplicate Iluvatar NN workspace tensor name")
            if (
                tensor.offset % tensor.alignment != 0
                or tensor.size <= 0
                or tensor.offset > plan.workspace_size
                or tensor.size > plan.workspace_size - tensor.offset
            ):
                raise ValueError("Iluvatar NN workspace tensor layout is invalid")
            if next_uid > 2**63 - 1:
                raise ValueError("Iluvatar NN workspace tensor UID overflows int64")
            named[tensor.name] = {
                "kind": "workspace_tensor",
                "uid": next_uid,
                "offset": node_base + tensor.offset,
                "size": tensor.size,
                "alignment": tensor.alignment,
            }
            next_uid += 1
        workspaces[node["id"]] = named
        if plan.workspace_size < 0 or plan.workspace_size > 2**63 - 1 - offset:
            raise ValueError("Iluvatar NN workspace size overflows int64")
        offset += plan.workspace_size
    if offset:
        offset = (offset + alignment - 1) // alignment * alignment
    return plans, workspaces, offset
