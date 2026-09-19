"""Ascend dispatch for matmul."""

from __future__ import annotations
from .common import TensorPlan
from typing import Any
import math
from .common import (
    FLOAT_DATA_TYPES,
    TRITON_POINTER_TYPES,
    _ceil_div,
    _has_non_overlapping_strides,
    _is_row_major_contiguous,
    _require_integer,
)


def _matmul_meta(
    a: TensorPlan,
    b: TensorPlan,
    output: TensorPlan,
    *,
    batch: int,
    m: int,
    n: int,
    k: int,
) -> dict[str, int]:
    batch_dimensions = output.dimensions[:-2]
    batch_rank = len(batch_dimensions)
    if batch_rank > 6:
        raise ValueError("matmul batch rank exceeds six dimensions")
    result = {
        "BATCH": batch,
        "M": m,
        "N": n,
        "K": k,
        "INPUT_IS_FLOAT32": 1 if a.data_type == "float32" else 0,
        "GROUP_M": 8 if m >= 2048 else 1,
    }
    leading = 6 - batch_rank
    padded_dimensions = [1] * leading + list(batch_dimensions)

    def padded_batch_strides(tensor: TensorPlan) -> list[int]:
        tensor_batch_dimensions = tensor.dimensions[:-2]
        tensor_batch_strides = tensor.strides[:-2]
        tensor_leading = batch_rank - len(tensor_batch_dimensions)
        aligned_dimensions = [1] * tensor_leading + list(tensor_batch_dimensions)
        aligned_strides = [0] * tensor_leading + list(tensor_batch_strides)
        return [0] * leading + [
            0 if dimension == 1 else stride
            for dimension, stride in zip(
                aligned_dimensions, aligned_strides, strict=True
            )
        ]

    a_batch_strides = padded_batch_strides(a)
    b_batch_strides = padded_batch_strides(b)
    c_batch_strides = [0] * leading + list(output.strides[:-2])
    for axis in range(6):
        result[f"DIM_{axis}"] = padded_dimensions[axis]
        result[f"A_BATCH_STRIDE_{axis}"] = a_batch_strides[axis]
        result[f"B_BATCH_STRIDE_{axis}"] = b_batch_strides[axis]
        result[f"C_BATCH_STRIDE_{axis}"] = c_batch_strides[axis]
    result.update(
        {
            "A_STRIDE_M": a.strides[-2],
            "A_STRIDE_K": a.strides[-1],
            "B_STRIDE_K": b.strides[-2],
            "B_STRIDE_N": b.strides[-1],
            "C_STRIDE_M": output.strides[-2],
            "C_STRIDE_N": output.strides[-1],
        }
    )
    return result


def _matmul_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | bool],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    if len(tensors) != 3:
        raise ValueError("MatMul tensor count is invalid")
    a, b, output = tensors
    data_types = [tensor["data_type"] for tensor in tensors]
    broadcast_a = parameters.get("_fprop_broadcast_a", False)
    if not isinstance(broadcast_a, bool):
        raise ValueError("internal broadcast-A MatMul flag must be boolean")
    valid_data_types = len(set(data_types)) == 1 and data_types[0] in FLOAT_DATA_TYPES
    if not valid_data_types:
        raise ValueError("MatMul input/output data types must match and be floating")
    if any(
        len(tensor["dimensions"]) < 2
        or len(tensor["dimensions"]) > 8
        or not _has_non_overlapping_strides(tensor["dimensions"], tensor["strides"])
        for tensor in tensors
    ):
        raise ValueError(
            "MatMul tensors require rank [2, 8] and non-overlapping strides"
        )

    m = a["dimensions"][-2]
    k = a["dimensions"][-1]
    if k != b["dimensions"][-2]:
        raise ValueError("MatMul contraction dimensions do not match")
    n = b["dimensions"][-1]
    a_batch = a["dimensions"][:-2]
    b_batch = b["dimensions"][:-2]
    batch_rank = max(len(a_batch), len(b_batch))
    if batch_rank > 6:
        raise ValueError("MatMul batch rank exceeds six")
    batch_dimensions = [1] * batch_rank
    for trailing in range(batch_rank):
        a_dimension = a_batch[-1 - trailing] if trailing < len(a_batch) else 1
        b_dimension = b_batch[-1 - trailing] if trailing < len(b_batch) else 1
        if a_dimension != b_dimension and a_dimension != 1 and b_dimension != 1:
            raise ValueError("MatMul batch dimensions are not broadcast-compatible")
        batch_dimensions[-1 - trailing] = max(a_dimension, b_dimension)
    expected_output = [*batch_dimensions, m, n]
    if output["dimensions"] != expected_output:
        raise ValueError("MatMul output metadata is inconsistent")

    batch = math.prod(batch_dimensions)
    for name, expected in (("batch", batch), ("m", m), ("n", n), ("k", k)):
        if _require_integer(parameters, name) != expected:
            raise ValueError(f"parameters.{name} is inconsistent with MatMul metadata")

    def batch_strides(tensor: dict[str, Any]) -> list[int]:
        tensor_batch = tensor["dimensions"][:-2]
        leading = batch_rank - len(tensor_batch)
        dimensions = [1] * leading + tensor_batch
        strides = [0] * leading + tensor["strides"][:-2]
        effective = [
            0 if dimension == 1 else stride
            for dimension, stride in zip(dimensions, strides)
        ]
        return [0] * (6 - batch_rank) + effective

    padded_dimensions = [1] * (6 - batch_rank) + batch_dimensions
    a_batch_strides = batch_strides(a)
    b_batch_strides = batch_strides(b)
    output_batch_strides = [0] * (6 - batch_rank) + output["strides"][:-2]
    block_m = 32 if m < 64 else 64
    block_n = 32 if n < 64 else 64
    block_k = 32
    constants: dict[str, int | bool] = {
        "M": m,
        "N": n,
        "K": k,
        "A_STRIDE_M": a["strides"][-2],
        "A_STRIDE_K": a["strides"][-1],
        "B_STRIDE_K": b["strides"][-2],
        "B_STRIDE_N": b["strides"][-1],
        "C_STRIDE_M": output["strides"][-2],
        "C_STRIDE_N": output["strides"][-1],
        "NATIVE_TF32_RNE": False,
        "INPUT_IS_FLOAT32": data_types[0] == "float32",
        "USE_TF32": False,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "GROUP_M": 8,
    }
    for axis in range(6):
        constants[f"DIM_{axis}"] = padded_dimensions[axis]
        constants[f"A_BATCH_STRIDE_{axis}"] = a_batch_strides[axis]
        constants[f"B_BATCH_STRIDE_{axis}"] = b_batch_strides[axis]
        constants[f"C_BATCH_STRIDE_{axis}"] = output_batch_strides[axis]

    return (
        "matmul_tiled_kernel",
        {
            "a_ptr": TRITON_POINTER_TYPES[data_types[0]],
            "b_ptr": TRITON_POINTER_TYPES[data_types[1]],
            "c_ptr": TRITON_POINTER_TYPES[data_types[2]],
        },
        constants,
        (
            ((m + block_m - 1) // block_m) * ((n + block_n - 1) // block_n),
            batch,
            1,
        ),
        [("tensor", None), ("tensor", None), ("tensor", None)],
    )
