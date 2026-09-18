# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate graph structure before selecting an operation family."""

from __future__ import annotations

import math
from ..dispatch.common import (
    _DATA_TYPES,
    _NAME,
    _integer,
    _require_exact_fields,
    _require_list,
    _require_object,
    _string,
)
from ..dispatch.tensor import (
    _parse_tensor,
)


def _parse_ports(
    value: object,
    description: str,
    tensor_uids: set[int],
    *,
    allow_optional: bool,
) -> list[int]:
    ports = _require_list(value, description)
    if not ports:
        raise ValueError(f"{description} must not be empty")
    uids: list[int] = []
    names: set[str] = set()
    for index, port_value in enumerate(ports):
        port = _require_object(port_value, f"{description}[{index}]")
        _require_exact_fields(
            port,
            {"name", "uid"},
            {"optional"} if allow_optional else set(),
            f"{description}[{index}]",
        )
        name = _string(port["name"], f"{description}[{index}].name")
        if _NAME.fullmatch(name) is None:
            raise ValueError(f"{description} port name is invalid")
        if name in names:
            raise ValueError(f"{description} port names must be unique")
        names.add(name)
        uid = _integer(port["uid"], f"{description}[{index}].uid")
        if uid not in tensor_uids:
            raise ValueError(f"{description} references an unknown tensor uid")
        optional = port.get("optional", False)
        if not isinstance(optional, bool):
            raise ValueError(f"{description} optional flag must be a boolean")
        uids.append(uid)
    return uids


def _validate_attribute(value: object, description: str) -> None:
    if value is None:
        raise ValueError(f"{description} must not be null")
    if isinstance(value, bool) or isinstance(value, str):
        return
    if isinstance(value, int):
        if not -(2**63) <= value < 2**63:
            raise ValueError(f"{description} integer is outside int64")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{description} must be finite")
        return
    if isinstance(value, list):
        if len(value) > 1024:
            raise ValueError(f"{description} array is too large")
        for index, item in enumerate(value):
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError(f"{description}[{index}] must be an integer")
            if not -(2**63) <= item < 2**63:
                raise ValueError(f"{description}[{index}] is outside int64")
        return
    raise ValueError(f"{description} has an unsupported value type")


def _parse_graph(value: object) -> list[str]:
    graph = _require_object(value, "graph")
    _require_exact_fields(
        graph,
        {"name", "tensor_count", "tensors", "node_count", "nodes"},
        set(),
        "graph",
    )
    _string(graph["name"], "graph name", allow_empty=True)
    tensors = _require_list(graph["tensors"], "graph.tensors")
    tensor_count = _integer(graph["tensor_count"], "graph tensor_count")
    if tensor_count != len(tensors) or not 1 <= tensor_count <= 4096:
        raise ValueError("graph tensor_count is invalid")
    tensor_registry: dict[int, bool] = {}
    for index, tensor_value in enumerate(tensors):
        uid, virtual = _parse_tensor(tensor_value, index)
        if uid in tensor_registry:
            raise ValueError("graph tensor uids must be unique")
        tensor_registry[uid] = virtual

    nodes = _require_list(graph["nodes"], "graph.nodes")
    node_count = _integer(graph["node_count"], "graph node_count")
    if node_count != len(nodes) or not 1 <= node_count <= 1024:
        raise ValueError("graph node_count is invalid")
    node_ids: set[int] = set()
    operation_types: set[str] = set()
    for index, node_value in enumerate(nodes):
        node = _require_object(node_value, f"graph.nodes[{index}]")
        _require_exact_fields(
            node,
            {
                "id",
                "type",
                "name",
                "compute_data_type",
                "inputs",
                "outputs",
                "attributes",
            },
            set(),
            f"graph.nodes[{index}]",
        )
        node_id = _integer(node["id"], "node id")
        if not 0 <= node_id < node_count or node_id in node_ids:
            raise ValueError("graph node ids must be unique values in range")
        node_ids.add(node_id)
        operation = _string(node["type"], "node type")
        if _NAME.fullmatch(operation) is None:
            raise ValueError("node type is invalid")
        operation_types.add(operation)
        _string(node["name"], "node name", allow_empty=True)
        compute_type = _string(
            node["compute_data_type"], "node compute_data_type"
        )
        if compute_type not in _DATA_TYPES:
            raise ValueError("node compute_data_type is unsupported")
        _parse_ports(
            node["inputs"],
            f"graph.nodes[{index}].inputs",
            set(tensor_registry),
            allow_optional=True,
        )
        _parse_ports(
            node["outputs"],
            f"graph.nodes[{index}].outputs",
            set(tensor_registry),
            allow_optional=False,
        )
        attributes = _require_object(node["attributes"], "node attributes")
        for name, attribute in attributes.items():
            if not isinstance(name, str) or _NAME.fullmatch(name) is None:
                raise ValueError("node attribute name is invalid")
            _validate_attribute(attribute, f"node attribute {name}")
    return sorted(operation_types)
