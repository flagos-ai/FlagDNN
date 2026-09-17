# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Hygon dispatch graph."""

from __future__ import annotations

from ..dispatch import nn as compiler_nn
from . import extended
from ..dispatch import tensor as compiler_tensor
from ..dispatch.common import (
    _require_object,
)
from ..dispatch.pointwise import (
    POINTWISE_SCHEMAS,
    _parse_pointwise_node,
)
from ..dispatch.tensor_metadata import (
    _tensor_storage_size,
)
from typing import Any


SUPPORTED_OPERATIONS = (
    tuple(extended.SUPPORTED_OPERATIONS)
    + tuple(POINTWISE_SCHEMAS)
    + tuple(sorted(compiler_tensor.SUPPORTED_OPERATIONS))
    + tuple(sorted(compiler_nn.SUPPORTED_OPERATIONS))
)


def _parse_node(
    node_value: object,
    position: int,
    node_count: int,
    tensor_registry: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    node = _require_object(node_value, f"graph.nodes[{position}]")
    operation = node.get("type")
    if operation == "matmul":
        inputs = node.get("inputs", [])
        if inputs and tensor_registry.get(inputs[0].get("uid"), {}).get(
            "data_type"
        ) in {"fp8_e4m3", "fp8_e5m2"}:
            return extended.parse_node(
                node_value, position, node_count, tensor_registry
            )
    if operation in extended.SUPPORTED_OPERATIONS:
        return extended.parse_node(
            node_value, position, node_count, tensor_registry
        )
    if operation in POINTWISE_SCHEMAS:
        return _parse_pointwise_node(
            node_value, position, node_count, tensor_registry
        )
    if operation in compiler_tensor.SUPPORTED_OPERATIONS:
        return compiler_tensor.parse_node(
            node_value, position, node_count, tensor_registry
        )
    if operation in compiler_nn.SUPPORTED_OPERATIONS:
        return compiler_nn.parse_node(
            node_value, position, node_count, tensor_registry
        )
    raise ValueError(
        f"Hygon compiler does not support graph operation {operation!r}"
    )


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
            (offset + minimum_alignment - 1)
            // minimum_alignment
            * minimum_alignment
        )
    return result, offset


def _nn_workspace_layout(
    nodes: list[dict[str, Any]],
    initial_offset: int,
    maximum_tensor_uid: int,
) -> tuple[
    dict[int, compiler_nn.NodePlan],
    dict[int, dict[str, dict[str, Any]]],
    int,
]:
    alignment = compiler_nn.WORKSPACE_ALIGNMENT
    offset = initial_offset
    next_uid = maximum_tensor_uid + 1
    plans: dict[int, compiler_nn.NodePlan] = {}
    workspaces: dict[int, dict[str, dict[str, Any]]] = {}
    for node in nodes:
        if node["operation"] not in compiler_nn.SUPPORTED_OPERATIONS:
            continue
        plan = compiler_nn.plan_kernel_stages(node)
        plans[node["id"]] = plan
        node_alignment = alignment
        for tensor in plan.workspace_tensors:
            if (
                tensor.alignment < alignment
                or tensor.alignment & (tensor.alignment - 1) != 0
            ):
                raise ValueError(
                    "Hygon NN workspace tensor alignment is invalid"
                )
            node_alignment = max(node_alignment, tensor.alignment)
        if offset > 2**63 - 1 - (node_alignment - 1):
            raise ValueError("Hygon NN workspace alignment overflows int64")
        offset = (
            (offset + node_alignment - 1) // node_alignment * node_alignment
        )
        node_base = offset
        named: dict[str, dict[str, Any]] = {}
        for tensor in plan.workspace_tensors:
            if tensor.name in named:
                raise ValueError("duplicate Hygon NN workspace tensor name")
            if (
                tensor.offset % tensor.alignment != 0
                or tensor.size <= 0
                or tensor.offset > plan.workspace_size
                or tensor.size > plan.workspace_size - tensor.offset
            ):
                raise ValueError("Hygon NN workspace tensor layout is invalid")
            if next_uid > 2**63 - 1:
                raise ValueError(
                    "Hygon NN workspace tensor UID overflows int64"
                )
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
            raise ValueError("Hygon NN workspace size overflows int64")
        offset += plan.workspace_size
    if offset:
        offset = (offset + alignment - 1) // alignment * alignment
    return plans, workspaces, offset
