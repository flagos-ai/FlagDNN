# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tensor layouts, strides and graph port helpers."""

from __future__ import annotations

from typing import Any
from ..dispatch.common import (
    _DATA_TYPES,
    _DATA_TYPE_BYTES,
    _integer,
    _require_exact_fields,
    _require_list,
    _require_object,
    _string,
)


def _parse_tensor(value: object, index: int) -> tuple[int, bool]:
    tensor = _require_object(value, f"graph.tensors[{index}]")
    _require_exact_fields(
        tensor,
        {"uid", "data_type", "dimensions", "strides", "alignment", "virtual"},
        set(),
        f"graph.tensors[{index}]",
    )
    uid = _integer(tensor["uid"], "tensor uid")
    if not 0 <= uid < 2**63:
        raise ValueError("tensor uid is outside the nonnegative int64 range")
    data_type = _string(tensor["data_type"], "tensor data_type")
    if data_type not in _DATA_TYPES:
        raise ValueError(f"unsupported tensor data_type: {data_type}")
    dimensions = _require_list(tensor["dimensions"], "tensor dimensions")
    strides = _require_list(tensor["strides"], "tensor strides")
    if len(dimensions) > 8:
        raise ValueError("tensor dimensions must have rank 0 through 8")
    if len(strides) != len(dimensions):
        raise ValueError("tensor strides must match dimensions rank")
    if any(
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or not 1 <= dimension <= 2**31 - 1
        for dimension in dimensions
    ):
        raise ValueError("tensor dimensions must be positive int32 values")
    if any(
        isinstance(stride, bool)
        or not isinstance(stride, int)
        or not 0 <= stride < 2**63
        for stride in strides
    ):
        raise ValueError("tensor strides must be nonnegative int64 values")
    storage_size = (
        1
        + sum(
            (dimension - 1) * stride
            for dimension, stride in zip(dimensions, strides, strict=True)
        )
    ) * _DATA_TYPE_BYTES[data_type]
    # The native JSON/artifact ABI represents sizes as signed int64 before
    # converting them to size_t.  Reject an unrepresentable view here instead
    # of emitting an artifact that the runtime must reject later.
    if storage_size > 2**63 - 1:
        raise ValueError("tensor storage size is outside positive int64 range")
    alignment = _integer(tensor["alignment"], "tensor alignment")
    if alignment <= 0 or alignment > 4096 or alignment & (alignment - 1):
        raise ValueError(
            "tensor alignment must be a power of two through 4096"
        )
    if not isinstance(tensor["virtual"], bool):
        raise ValueError("tensor virtual must be a boolean")
    return uid, tensor["virtual"]


def _dense_strides(dimensions: list[Any]) -> list[int]:
    result = [0] * len(dimensions)
    stride = 1
    for index in range(len(dimensions) - 1, -1, -1):
        result[index] = stride
        stride *= int(dimensions[index])
    return result


def _has_non_overlapping_strides(
    dimensions: list[Any], strides: list[Any]
) -> bool:
    axes = sorted(
        (
            (int(stride), int(dimension))
            for dimension, stride in zip(dimensions, strides, strict=True)
            if int(dimension) > 1
        ),
        key=lambda item: item[0],
    )
    required_stride = 1
    for stride, dimension in axes:
        if stride < required_stride:
            return False
        required_stride = stride * dimension
    return True


def _padded_pointwise_values(values: list[Any], fill: int) -> list[int]:
    if not 1 <= len(values) <= 8:
        raise ValueError("THead pointwise metadata requires rank 1 through 8")
    return [fill] * (8 - len(values)) + [int(value) for value in values]


def _same_dense_layout(tensors: list[dict[str, Any]]) -> bool:
    """Whether a physical linear walk pairs the same logical elements.

    Dense permutations (including NHWC) need no coordinate decomposition when
    all operands share their layout. Singleton strides do not affect addresses.
    Padded tensors and operands with different layouts use strided kernels.
    """
    dimensions = tensors[0]["dimensions"]
    axes = [axis for axis, size in enumerate(dimensions) if int(size) > 1]
    strides = tensors[0]["strides"]
    extent = 1
    for axis in sorted(axes, key=lambda axis: int(strides[axis])):
        if int(strides[axis]) != extent:
            return False
        extent *= int(dimensions[axis])
    return all(
        tensor["dimensions"] == dimensions
        and all(
            int(tensor["strides"][axis]) == int(strides[axis]) for axis in axes
        )
        for tensor in tensors[1:]
    )


def _named_port_uids(
    node: dict[str, Any],
    field: str,
    expected_names: tuple[str, ...],
    operation_label: str,
) -> list[int]:
    ports = _require_list(node[field], f"{operation_label} {field}")
    names = tuple(port["name"] for port in ports)
    if names != expected_names:
        raise ValueError(
            f"THead {operation_label} {field} must be ordered as "
            f"{', '.join(expected_names)}"
        )
    if any(port.get("optional", False) for port in ports):
        raise ValueError(
            f"THead {operation_label} slice does not accept optional ports"
        )
    return [int(port["uid"]) for port in ports]


def _require_integer_array(
    attributes: dict[str, Any], name: str, length: int
) -> list[int]:
    values = _require_list(attributes.get(name), f"layout attribute {name}")
    if len(values) != length or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in values
    ):
        raise ValueError(f"THead layout {name} must contain {length} integers")
    return [int(value) for value in values]
