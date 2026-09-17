# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Iluvatar dispatch / graph_tensor implementation."""

from __future__ import annotations

from .common import POINTER_TYPES
from .common import _require_list
from .common import _require_object
from typing import Any
import math


def _has_non_overlapping_strides(dimensions: list[int], strides: list[int]) -> bool:
    axes = sorted(
        (stride, dimension)
        for dimension, stride in zip(dimensions, strides)
        if dimension > 1
    )
    required_span = 1
    for stride, dimension in axes:
        if stride < required_span:
            return False
        required_span += (dimension - 1) * stride
    return True


def _storage_elements(tensor: dict[str, Any]) -> int:
    return 1 + sum(
        (dimension - 1) * stride
        for dimension, stride in zip(
            tensor["dimensions"], tensor["strides"], strict=True
        )
    )


def _is_physically_dense(tensor: dict[str, Any]) -> bool:
    return _has_non_overlapping_strides(
        tensor["dimensions"], tensor["strides"]
    ) and _storage_elements(tensor) == math.prod(tensor["dimensions"])


def _tensor_storage_size(tensor: dict[str, Any]) -> int:
    element_size = {
        "float32": 4,
        "int32": 4,
        "fp8_e8m0": 1,
        "float16": 2,
        "bfloat16": 2,
        "boolean": 1,
        "fp8_e4m3": 1,
        "fp8_e5m2": 1,
    }[tensor["data_type"]]
    size = _storage_elements(tensor) * element_size
    if size <= 0 or size > 2**63 - 1:
        raise ValueError(f"tensor {tensor['uid']} storage size is invalid")
    return size


def _parse_tensor_table(graph: dict[str, Any]) -> dict[int, dict[str, Any]]:
    tensors = _require_list(graph.get("tensors"), "graph.tensors")
    tensor_count = graph.get("tensor_count")
    if (
        isinstance(tensor_count, bool)
        or not isinstance(tensor_count, int)
        or tensor_count != len(tensors)
        or tensor_count < 1
    ):
        raise ValueError("graph tensor_count is invalid")

    result: dict[int, dict[str, Any]] = {}
    for index, value in enumerate(tensors):
        tensor = _require_object(value, f"graph.tensors[{index}]")
        uid = tensor.get("uid")
        if isinstance(uid, bool) or not isinstance(uid, int) or uid <= 0:
            raise ValueError(f"tensor UID {index} is invalid")
        if uid in result:
            raise ValueError("graph tensor UIDs must be unique")
        data_type = tensor.get("data_type")
        if not isinstance(data_type, str) or data_type not in POINTER_TYPES:
            raise ValueError(
                f"tensor {uid} has an unsupported data type: {data_type!r}"
            )
        is_virtual = tensor.get("virtual")
        if not isinstance(is_virtual, bool):
            raise ValueError(f"tensor {uid} virtual flag must be boolean")
        alignment = tensor.get("alignment", 16)
        if (
            isinstance(alignment, bool)
            or not isinstance(alignment, int)
            or alignment <= 0
            or alignment > 2**31
            or alignment & (alignment - 1) != 0
        ):
            raise ValueError(f"tensor {uid} alignment must be a positive power of two")
        dimensions = _require_list(tensor.get("dimensions"), f"tensor {uid} dimensions")
        strides = _require_list(tensor.get("strides"), f"tensor {uid} strides")
        if len(dimensions) > 8 or len(dimensions) != len(strides):
            raise ValueError(f"tensor {uid} rank is invalid")
        if any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item <= 0
            or item > 2**31 - 1
            for item in dimensions + strides
        ):
            raise ValueError(f"tensor {uid} shape or strides are invalid")
        parsed = {
            "uid": uid,
            "virtual": is_virtual,
            "alignment": alignment,
            "data_type": data_type,
            "dimensions": list(dimensions),
            "strides": list(strides),
        }
        if not _has_non_overlapping_strides(dimensions, strides):
            raise ValueError(f"tensor {uid} strides overlap")
        _tensor_storage_size(parsed)
        result[uid] = parsed
    return result


def _parse_port(
    port_value: object,
    expected_name: str,
    direction: str,
    tensor_registry: dict[int, dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    port = _require_object(port_value, f"node.{direction}")
    if port.get("name") != expected_name:
        raise ValueError(f"node {direction} port must be named {expected_name!r}")
    optional = port.get("optional", False)
    if not isinstance(optional, bool) or optional:
        raise ValueError("Iluvatar pointwise operations require every tensor port")
    uid = port.get("uid")
    if isinstance(uid, bool) or not isinstance(uid, int) or uid <= 0:
        raise ValueError(f"node {direction} UID is invalid")
    try:
        tensor = tensor_registry[uid]
    except KeyError as error:
        raise ValueError(f"node references unknown tensor UID {uid}") from error
    return uid, tensor


def _manifest_tensor(
    tensor: dict[str, Any],
    workspace: dict[int, tuple[int, int, int]],
) -> dict[str, Any]:
    result = {
        key: tensor[key]
        for key in (
            "uid",
            "data_type",
            "dimensions",
            "strides",
            "alignment",
            "virtual",
        )
    }
    result["storage_size"] = _tensor_storage_size(tensor)
    if tensor["virtual"]:
        result["workspace_offset"] = workspace[tensor["uid"]][0]
    return result
