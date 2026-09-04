"""Strict tensor parsing and layout helpers for mthreads pointwise Graphs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


MAX_I32 = (1 << 31) - 1
MAX_I64 = (1 << 63) - 1
# Linear kernels may form every lane in a final 65536-element tile before
# applying its mask.  Preserve enough signed-i32 headroom for those inactive
# tail lanes so the index cannot wrap negative before pointer arithmetic.
MAX_LINEAR_ELEMENTS = MAX_I32 - ((1 << 16) - 1)
SUPPORTED_DATA_TYPES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "boolean": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
}
POINTER_TYPES = {
    "float32": "*fp32",
    "float16": "*fp16",
    "bfloat16": "*bf16",
    "boolean": "*i8",
    "fp8_e4m3": "*fp8e4nv",
    "fp8_e5m2": "*fp8e5",
}
_TENSOR_KEYS = {
    "uid",
    "data_type",
    "dimensions",
    "strides",
    "alignment",
    "virtual",
}


@dataclass(frozen=True)
class TensorSpec:
    uid: int
    data_type: str
    dimensions: tuple[int, ...]
    strides: tuple[int, ...]
    virtual: bool
    storage_size: int
    alignment: int


def require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    context: str,
) -> None:
    actual = set(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append("missing " + ", ".join(missing))
    if unknown:
        details.append("unknown " + ", ".join(unknown))
    raise ValueError(f"{context} keys are invalid ({'; '.join(details)})")


def require_object(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def require_list(value: object, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be an array")
    return value


def require_integer(
    value: object,
    context: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{context} is below its minimum")
    if maximum is not None and value > maximum:
        raise ValueError(f"{context} exceeds its maximum")
    return value


def require_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _integer_array(
    value: object,
    context: str,
    *,
    maximum_length: int,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    values = require_list(value, context)
    minimum_length = 0 if allow_empty else 1
    if not minimum_length <= len(values) <= maximum_length:
        raise ValueError(
            f"{context} length must be in "
            f"[{minimum_length}, {maximum_length}]"
        )
    return tuple(
        require_integer(
            item,
            f"{context}[{index}]",
            minimum=1,
            maximum=MAX_I32,
        )
        for index, item in enumerate(values)
    )


def _storage_size(
    dimensions: tuple[int, ...],
    strides: tuple[int, ...],
    element_size: int,
    context: str,
) -> int:
    maximum_offset = 0
    for dimension, stride in zip(dimensions, strides, strict=True):
        contribution = (dimension - 1) * stride
        if contribution > MAX_I64 - maximum_offset:
            raise ValueError(f"{context} storage span overflows int64")
        maximum_offset += contribution
    span = maximum_offset + 1
    if span > MAX_I64 // element_size:
        raise ValueError(f"{context} storage size overflows int64")
    return span * element_size


def is_non_overlapping(tensor: TensorSpec) -> bool:
    axes = sorted(
        (
            (stride, dimension)
            for dimension, stride in zip(
                tensor.dimensions, tensor.strides, strict=True
            )
            if dimension > 1
        ),
        key=lambda entry: entry[0],
    )
    required_span = 1
    for stride, dimension in axes:
        if stride < required_span:
            return False
        if stride > MAX_I64 // dimension:
            return False
        required_span = stride * dimension
    return True


def is_physically_dense(tensor: TensorSpec) -> bool:
    axes = sorted(
        (
            (stride, dimension)
            for dimension, stride in zip(
                tensor.dimensions, tensor.strides, strict=True
            )
            if dimension > 1
        ),
        key=lambda entry: entry[0],
    )
    expected = 1
    for stride, dimension in axes:
        if stride != expected:
            return False
        expected *= dimension
    return True


def is_row_major_contiguous(tensor: TensorSpec) -> bool:
    expected = 1
    for dimension, stride in zip(
        reversed(tensor.dimensions), reversed(tensor.strides), strict=True
    ):
        if stride != expected:
            return False
        expected *= dimension
    return True


def parse_tensor_table(
    graph: dict[str, Any],
) -> tuple[tuple[TensorSpec, ...], dict[int, TensorSpec]]:
    values = require_list(graph.get("tensors"), "graph.tensors")
    tensor_count = require_integer(
        graph.get("tensor_count"),
        "graph.tensor_count",
        minimum=1,
        maximum=4096,
    )
    if tensor_count != len(values):
        raise ValueError("graph.tensor_count does not match graph.tensors")

    ordered: list[TensorSpec] = []
    registry: dict[int, TensorSpec] = {}
    for index, raw_value in enumerate(values):
        context = f"graph.tensors[{index}]"
        value = require_object(raw_value, context)
        require_exact_keys(value, _TENSOR_KEYS, context)
        uid = require_integer(
            value["uid"], f"{context}.uid", minimum=0, maximum=MAX_I64
        )
        if uid in registry:
            raise ValueError("graph tensor UID is duplicated")
        data_type = value["data_type"]
        if data_type not in SUPPORTED_DATA_TYPES:
            raise ValueError(
                f"{context}.data_type is not supported by mthreads"
            )
        dimensions = _integer_array(
            value["dimensions"],
            f"{context}.dimensions",
            maximum_length=8,
            allow_empty=True,
        )
        strides = _integer_array(
            value["strides"],
            f"{context}.strides",
            maximum_length=8,
            allow_empty=True,
        )
        if len(dimensions) != len(strides):
            raise ValueError(
                f"{context} dimensions and strides have different ranks"
            )
        alignment = require_integer(
            value["alignment"],
            f"{context}.alignment",
            minimum=1,
            maximum=1 << 20,
        )
        if alignment & (alignment - 1):
            raise ValueError(f"{context}.alignment must be a power of two")
        virtual = value["virtual"]
        if not isinstance(virtual, bool):
            raise ValueError(f"{context}.virtual must be a boolean")
        tensor = TensorSpec(
            uid=uid,
            data_type=data_type,
            dimensions=dimensions,
            strides=strides,
            virtual=virtual,
            storage_size=_storage_size(
                dimensions,
                strides,
                SUPPORTED_DATA_TYPES[data_type],
                context,
            ),
            alignment=alignment,
        )
        if not is_non_overlapping(tensor):
            raise ValueError(f"{context} has overlapping strides")
        ordered.append(tensor)
        registry[uid] = tensor
    return tuple(ordered), registry


def broadcast_dimensions(
    left: TensorSpec, right: TensorSpec
) -> tuple[int, ...]:
    rank = max(len(left.dimensions), len(right.dimensions))
    result = [1] * rank
    for trailing in range(rank):
        left_dimension = (
            left.dimensions[-1 - trailing]
            if trailing < len(left.dimensions)
            else 1
        )
        right_dimension = (
            right.dimensions[-1 - trailing]
            if trailing < len(right.dimensions)
            else 1
        )
        if (
            left_dimension != right_dimension
            and left_dimension != 1
            and right_dimension != 1
        ):
            raise ValueError(
                "pointwise input shapes are not broadcast-compatible"
            )
        result[-1 - trailing] = max(left_dimension, right_dimension)
    return tuple(result)


def pointwise_constants(
    left: TensorSpec,
    right: TensorSpec,
    output: TensorSpec,
) -> tuple[int, ...]:
    rank = len(output.dimensions)
    if not 1 <= rank <= 8:
        raise ValueError("pointwise output rank must be in [1, 8]")

    def effective_strides(tensor: TensorSpec) -> list[int]:
        leading = rank - len(tensor.dimensions)
        dimensions = [1] * leading + list(tensor.dimensions)
        strides = [0] * leading + list(tensor.strides)
        return [
            0 if dimension == 1 else stride
            for dimension, stride in zip(dimensions, strides, strict=True)
        ]

    leading = 8 - rank
    dimensions = [1] * leading + list(output.dimensions)
    left_strides = [0] * leading + effective_strides(left)
    right_strides = [0] * leading + effective_strides(right)
    output_strides = [0] * leading + list(output.strides)
    return tuple(
        dimensions + left_strides + right_strides + output_strides
    )


def can_use_dense_binary(
    left: TensorSpec,
    right: TensorSpec,
    output: TensorSpec,
) -> bool:
    return (
        left.dimensions == right.dimensions == output.dimensions
        and left.strides == right.strides == output.strides
        and is_physically_dense(left)
        and is_physically_dense(right)
        and is_physically_dense(output)
    )


# Kept for the Phase 1 contract tests and installed consumers that imported
# the Add-specific helper before the binary pointwise compiler was generalized.
can_use_dense_add = can_use_dense_binary


def can_use_dense_unary(input_tensor: TensorSpec, output: TensorSpec) -> bool:
    return (
        input_tensor.dimensions == output.dimensions
        and input_tensor.strides == output.strides
        and is_physically_dense(input_tensor)
        and is_physically_dense(output)
    )


def unary_pointwise_constants(
    input_tensor: TensorSpec, output: TensorSpec
) -> tuple[int, ...]:
    if input_tensor.dimensions != output.dimensions:
        raise ValueError("unary pointwise input/output shapes must match")
    rank = len(output.dimensions)
    if not 1 <= rank <= 8:
        raise ValueError("unary pointwise output rank must be in [1, 8]")
    leading = 8 - rank
    return tuple(
        [1] * leading
        + list(output.dimensions)
        + [0] * leading
        + list(input_tensor.strides)
        + [0] * leading
        + list(output.strides)
    )


def can_use_dense_ternary(
    a: TensorSpec,
    b: TensorSpec,
    predicate: TensorSpec,
    output: TensorSpec,
) -> bool:
    tensors = (a, b, predicate, output)
    return all(
        tensor.dimensions == output.dimensions
        and tensor.strides == output.strides
        and is_physically_dense(tensor)
        for tensor in tensors
    )


def ternary_pointwise_constants(
    a: TensorSpec,
    b: TensorSpec,
    predicate: TensorSpec,
    output: TensorSpec,
) -> tuple[int, ...]:
    rank = len(output.dimensions)
    if not 1 <= rank <= 8:
        raise ValueError("ternary pointwise output rank must be in [1, 8]")

    def effective_strides(tensor: TensorSpec) -> list[int]:
        if len(tensor.dimensions) > rank:
            raise ValueError("ternary input rank exceeds output rank")
        leading = rank - len(tensor.dimensions)
        dimensions = [1] * leading + list(tensor.dimensions)
        strides = [0] * leading + list(tensor.strides)
        return [
            0 if dimension == 1 else stride
            for dimension, stride in zip(dimensions, strides, strict=True)
        ]

    leading = 8 - rank
    dimensions = [1] * leading + list(output.dimensions)
    input_strides = [
        [0] * leading + effective_strides(tensor)
        for tensor in (a, b, predicate)
    ]
    output_strides = [0] * leading + list(output.strides)
    return tuple(
        dimensions
        + input_strides[0]
        + input_strides[1]
        + input_strides[2]
        + output_strides
    )


def layout_constants(
    logical_input_dimensions: tuple[int, ...],
    logical_input_strides: tuple[int, ...],
    input_base: int,
    output: TensorSpec,
) -> tuple[int, ...]:
    input_rank = len(logical_input_dimensions)
    output_rank = len(output.dimensions)
    if (
        not 1 <= input_rank <= 8
        or len(logical_input_strides) != input_rank
        or not 1 <= output_rank <= 8
        or input_base < 0
        or input_base > MAX_I64
        or any(value <= 0 for value in logical_input_dimensions)
        or any(value <= 0 for value in logical_input_strides)
    ):
        raise ValueError("layout kernel metadata is invalid")
    input_elements = math.prod(logical_input_dimensions)
    output_elements = math.prod(output.dimensions)
    if input_elements != output_elements or output_elements > MAX_I32:
        raise ValueError("layout kernel element count is invalid")
    return tuple(
        [input_base]
        + [1] * (8 - input_rank)
        + list(logical_input_dimensions)
        + [0] * (8 - input_rank)
        + list(logical_input_strides)
        + [1] * (8 - output_rank)
        + list(output.dimensions)
        + [0] * (8 - output_rank)
        + list(output.strides)
    )


def reduction_strided_constants(
    input_tensor: TensorSpec,
    output: TensorSpec,
    axis: int,
    keep_dimensions: bool,
) -> tuple[int, ...]:
    rank = len(input_tensor.dimensions)
    if not 1 <= rank <= 8 or not 0 <= axis < rank:
        raise ValueError("reduction rank or axis is invalid")
    logical_dimensions = list(input_tensor.dimensions)
    logical_dimensions[axis] = 1
    input_strides = list(input_tensor.strides)
    reduction_stride = input_strides[axis]
    input_strides[axis] = 0
    if keep_dimensions:
        if len(output.dimensions) != rank:
            raise ValueError("keep-dimension reduction output rank differs")
        output_strides = list(output.strides)
    else:
        if len(output.dimensions) != rank - 1:
            raise ValueError("reduction output rank differs")
        output_strides = []
        output_axis = 0
        for input_axis in range(rank):
            if input_axis == axis:
                output_strides.append(0)
            else:
                output_strides.append(output.strides[output_axis])
                output_axis += 1
    output_strides[axis] = 0
    leading = 8 - rank
    return tuple(
        [reduction_stride]
        + [1] * leading
        + logical_dimensions
        + [0] * leading
        + input_strides
        + [0] * leading
        + output_strides
    )


def matmul_constants(
    a: TensorSpec,
    b: TensorSpec,
    output: TensorSpec,
) -> tuple[int, ...]:
    if any(not 2 <= len(tensor.dimensions) <= 8 for tensor in (a, b, output)):
        raise ValueError("Matmul tensor ranks must be in [2, 8]")
    m = a.dimensions[-2]
    k = a.dimensions[-1]
    n = b.dimensions[-1]
    if b.dimensions[-2] != k:
        raise ValueError("Matmul contraction dimensions do not match")
    a_batch = list(a.dimensions[:-2])
    b_batch = list(b.dimensions[:-2])
    batch_rank = max(len(a_batch), len(b_batch))
    if batch_rank > 6:
        raise ValueError("Matmul batch rank exceeds six")
    batch_dimensions = [1] * batch_rank
    for trailing in range(batch_rank):
        a_dimension = a_batch[-1 - trailing] if trailing < len(a_batch) else 1
        b_dimension = b_batch[-1 - trailing] if trailing < len(b_batch) else 1
        if (
            a_dimension != b_dimension
            and a_dimension != 1
            and b_dimension != 1
        ):
            raise ValueError("Matmul batch dimensions are not broadcastable")
        batch_dimensions[-1 - trailing] = max(a_dimension, b_dimension)
    if output.dimensions != tuple((*batch_dimensions, m, n)):
        raise ValueError("Matmul output shape is invalid")

    def batch_strides(tensor: TensorSpec) -> list[int]:
        tensor_batch = list(tensor.dimensions[:-2])
        leading = batch_rank - len(tensor_batch)
        dimensions = [1] * leading + tensor_batch
        strides = [0] * leading + list(tensor.strides[:-2])
        effective = [
            0 if dimension == 1 else stride
            for dimension, stride in zip(dimensions, strides, strict=True)
        ]
        return [0] * (6 - batch_rank) + effective

    padded_dimensions = [1] * (6 - batch_rank) + batch_dimensions
    output_batch_strides = [0] * (6 - batch_rank) + list(
        output.strides[:-2]
    )
    return tuple(
        [m, n, k]
        + padded_dimensions
        + batch_strides(a)
        + batch_strides(b)
        + output_batch_strides
        + [
            a.strides[-2],
            a.strides[-1],
            b.strides[-2],
            b.strides[-1],
            output.strides[-2],
            output.strides[-1],
            1 if a.data_type == "float32" else 0,
            1 if a.data_type == "float32" and min(m, n, k) >= 512 else 0,
        ]
    )


def element_count(tensor: TensorSpec) -> int:
    result = math.prod(tensor.dimensions)
    if result > MAX_LINEAR_ELEMENTS:
        raise ValueError(
            "tensor element count exceeds safe int32 linear index range"
        )
    return result
