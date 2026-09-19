"""Ascend dispatch for reduction."""

from __future__ import annotations
from .common import MAX_RANK, TensorPlan
import math
from .common import (
    NUMERIC_DATA_TYPES,
    REDUCTION_OPERATIONS,
    TRITON_POINTER_TYPES,
    _require_integer,
    _next_power_of_two,
    _reduction_tensor_constants,
)


def _reduction_meta(
    input_tensor: TensorPlan,
    output: TensorPlan,
    *,
    axis: int,
    keep_dimensions: int,
    outer: int,
    reduction: int,
    inner: int,
    output_elements: int,
    mode: int,
) -> dict[str, int]:
    result = {
        "RANK": len(input_tensor.dimensions),
        "OUTPUT_RANK": len(output.dimensions),
        "AXIS": axis,
        "KEEP_DIMENSIONS": keep_dimensions,
        "OUTER": outer,
        "REDUCTION_SIZE": reduction,
        "INNER": inner,
        "OUTPUT_ELEMENTS": output_elements,
        "REDUCTION_MODE": mode,
    }
    input_leading = MAX_RANK - len(input_tensor.dimensions)
    output_leading = MAX_RANK - len(output.dimensions)
    input_dimensions = [1] * input_leading + list(input_tensor.dimensions)
    input_strides = [0] * input_leading + list(input_tensor.strides)
    output_dimensions = [1] * output_leading + list(output.dimensions)
    output_strides = [0] * output_leading + list(output.strides)
    for axis_index in range(MAX_RANK):
        result[f"INPUT_DIM_{axis_index}"] = input_dimensions[axis_index]
        result[f"INPUT_STRIDE_{axis_index}"] = input_strides[axis_index]
        result[f"OUTPUT_DIM_{axis_index}"] = output_dimensions[axis_index]
        result[f"OUTPUT_STRIDE_{axis_index}"] = output_strides[axis_index]
    return result


def configuration(operation, parameters, tensors):
    tensor_data_types = [tensor["data_type"] for tensor in tensors]
    if operation in REDUCTION_OPERATIONS:
        if (
            len(tensor_data_types) != 2
            or tensor_data_types[0] not in NUMERIC_DATA_TYPES
            or (
                tensor_data_types[1] != "float32"
                and (
                    tensor_data_types[0] == "int32"
                    or tensor_data_types[0] != tensor_data_types[1]
                )
            )
        ):
            raise ValueError(
                "Reduction requires numeric input and FP32 "
                "or matching floating output"
            )
        pointer_type = TRITON_POINTER_TYPES.get(tensor_data_types[0])
        if pointer_type is None:
            raise ValueError(
                "unsupported Reduction data type: " f"{tensor_data_types[0]!r}"
            )
        outer = _require_integer(parameters, "outer")
        extent = _require_integer(parameters, "reduction", maximum=65536)
        inner = _require_integer(parameters, "inner")
        output_elements = _require_integer(parameters, "output_elements")
        input_rank = len(tensors[0]["dimensions"])
        if input_rank == 0:
            raise ValueError("Reduction input must have positive rank")
        axis = _require_integer(
            parameters,
            "axis",
            minimum=0,
            maximum=input_rank - 1,
        )
        keep_dimensions_value = _require_integer(
            parameters, "keep_dimensions", minimum=0, maximum=1
        )
        keep_dimensions = keep_dimensions_value == 1
        if math.prod(tensors[0]["dimensions"]) != outer * extent * inner:
            raise ValueError("Reduction parameters are inconsistent with input shape")
        if (
            math.prod(tensors[1]["dimensions"]) != output_elements
            or output_elements != outer * inner
        ):
            raise ValueError("Reduction parameters are inconsistent with output shape")
        strided_constants = _reduction_tensor_constants(tensors, axis, keep_dimensions)

        block_n = _next_power_of_two(extent)
        constants: dict[str, int | str] = {
            "N": extent,
            "OP": REDUCTION_OPERATIONS[operation],
            "BLOCK_M": 1,
            "BLOCK_N": block_n,
        }
        signature = {
            "x_ptr": pointer_type,
            "out_ptr": TRITON_POINTER_TYPES[tensor_data_types[1]],
            "M": "i32",
        }
        # Vectorize output lanes. Ascend's mixed-dtype strided gather is
        # reliable with one reduction coordinate per load instruction.
        block_m = 64
        constants.update(strided_constants)
        constants["BLOCK_M"] = block_m
        return (
            "reduction_strided_kernel",
            signature,
            constants,
            ((output_elements + block_m - 1) // block_m, 1, 1),
            [
                ("tensor", None),
                ("tensor", None),
                ("scalar_i32", "output_elements"),
            ],
        )
