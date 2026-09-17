# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Iluvatar dispatch / selection implementation."""

from __future__ import annotations

from . import nn as nn_dispatch
from . import tensor as tensor_dispatch
from .common import _require_object
from .pointwise import POINTWISE_SCHEMAS
from .pointwise import _parse_pointwise_node
from typing import Any

SUPPORTED_OPERATIONS = (
    tuple(POINTWISE_SCHEMAS)
    + tuple(sorted(tensor_dispatch.SUPPORTED_OPERATIONS))
    + tuple(sorted(nn_dispatch.SUPPORTED_OPERATIONS))
)


def _parse_node(
    node_value: object,
    position: int,
    node_count: int,
    tensor_registry: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    node = _require_object(node_value, f"graph.nodes[{position}]")
    operation = node.get("type")
    if operation in POINTWISE_SCHEMAS:
        return _parse_pointwise_node(node_value, position, node_count, tensor_registry)
    if operation in tensor_dispatch.SUPPORTED_OPERATIONS:
        return tensor_dispatch.parse_node(
            node_value, position, node_count, tensor_registry
        )
    if operation in nn_dispatch.SUPPORTED_OPERATIONS:
        return nn_dispatch.parse_node(node_value, position, node_count, tensor_registry)
    raise ValueError(
        f"Iluvatar compiler does not support graph operation {operation!r}"
    )
