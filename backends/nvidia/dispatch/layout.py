"""Reshape, transpose, and slice validation and kernel configurations."""

from __future__ import annotations

from typing import Any
import math

from .common import (
    ExecutionGroup,
    _has_non_overlapping_strides,
    _is_row_major_contiguous,
    _require_integer,
    _require_integer_list,
)


def _transpose_matrix_stage(
    source: dict[str, Any],
    destination: dict[str, Any],
    rows: int,
    columns: int,
    source_node_ids: list[int],
) -> ExecutionGroup:
    """
    Materialize a transpose using compact matrix views of existing storage.
    """
    input_view = {
        **source,
        "dimensions": [rows, columns],
        "strides": [columns, 1],
    }
    output_view = {
        **destination,
        "dimensions": [columns, rows],
        "strides": [rows, 1],
    }
    return {
        "operation": "transpose",
        "source_node_ids": source_node_ids,
        "parameters": {
            "rank": 2,
            "permutation": [1, 0],
            "n_elements": rows * columns,
            "input_dimensions": input_view["dimensions"],
            "input_strides": input_view["strides"],
            "output_dimensions": output_view["dimensions"],
            "output_strides": output_view["strides"],
        },
        "tensors": [input_view, output_view],
        "input_uids": [source["uid"]],
        "output_uids": [destination["uid"]],
    }


def _layout_kernel_configuration(
    operation: str,
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    if len(tensors) != 2:
        raise ValueError("layout operation tensor count is invalid")
    input_tensor, output_tensor = tensors
    data_types = [tensor["data_type"] for tensor in tensors]
    if len(set(data_types)) != 1:
        raise ValueError("layout operation input/output data types must match")
    pointer_type = {
        "float32": "*i32",
        "int32": "*i32",
        "float16": "*u16",
        "bfloat16": "*u16",
        "boolean": "*u8",
        "fp8_e4m3": "*u8",
        "fp8_e5m2": "*u8",
        "fp8_e8m0": "*u8",
    }.get(data_types[0])
    if pointer_type is None:
        raise ValueError(
            f"unsupported layout operation data type: {data_types[0]!r}"
        )
    if any(
        len(tensor["dimensions"]) > 8
        or not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        )
        for tensor in tensors
    ):
        raise ValueError(
            "layout tensors require rank at most eight and "
            "non-overlapping strides"
        )

    input_dimensions = input_tensor["dimensions"]
    input_strides = input_tensor["strides"]
    output_dimensions = output_tensor["dimensions"]
    output_strides = output_tensor["strides"]
    input_rank = len(input_dimensions)
    output_rank = len(output_dimensions)
    elements = _require_integer(parameters, "n_elements")
    if elements != math.prod(output_dimensions):
        raise ValueError(
            "parameters.n_elements is inconsistent with layout output"
        )

    def require_metadata(
        name: str,
        expected: list[int],
        *,
        minimum: int,
    ) -> list[int]:
        value = _require_integer_list(
            parameters,
            name,
            len(expected),
            minimum=minimum,
            maximum=2**63 - 1,
        )
        if value != expected:
            raise ValueError(
                f"parameters.{name} is inconsistent with tensor metadata"
            )
        return value

    require_metadata("input_dimensions", input_dimensions, minimum=1)
    require_metadata("input_strides", input_strides, minimum=1)
    require_metadata("output_dimensions", output_dimensions, minimum=1)
    require_metadata("output_strides", output_strides, minimum=1)

    input_base = 0
    logical_input_dimensions = input_dimensions
    logical_input_strides = input_strides
    if operation == "reshape":
        if (
            _require_integer(parameters, "input_rank", minimum=0, maximum=8)
            != input_rank
            or _require_integer(
                parameters, "output_rank", minimum=0, maximum=8
            )
            != output_rank
        ):
            raise ValueError("reshape rank parameters are inconsistent")
        _require_integer(parameters, "reshape_mode", minimum=2, maximum=2)
        if math.prod(input_dimensions) != elements:
            raise ValueError("reshape input/output element counts must match")
    elif operation == "transpose":
        if input_rank == 0 or input_rank != output_rank:
            raise ValueError(
                "transpose input/output ranks must match in [1, 8]"
            )
        if (
            _require_integer(parameters, "rank", minimum=1, maximum=8)
            != input_rank
        ):
            raise ValueError("transpose rank parameter is inconsistent")
        permutation = _require_integer_list(
            parameters,
            "permutation",
            input_rank,
            minimum=0,
            maximum=input_rank - 1,
        )
        if sorted(permutation) != list(range(input_rank)):
            raise ValueError(
                "transpose permutation must contain each axis once"
            )
        expected_output = [input_dimensions[axis] for axis in permutation]
        if output_dimensions != expected_output:
            raise ValueError(
                "transpose output shape does not match permutation"
            )
        logical_input_dimensions = output_dimensions
        logical_input_strides = [input_strides[axis] for axis in permutation]
        if (
            input_rank == 2
            and permutation == [1, 0]
            and elements >= 65536
            and min(input_dimensions) >= 16
            # Both built-in 16/32 tiles must fit CUDA's grid limits. Wide
            # matrices retain the flat-grid generic materialization path.
            and input_dimensions[0] <= 16 * (2**31 - 1)
            and input_dimensions[1] <= 16 * 65535
            and _is_row_major_contiguous(input_tensor)
            and _is_row_major_contiguous(output_tensor)
        ):
            rows, columns = input_dimensions
            block = 32
            return (
                "matrix_transpose_kernel",
                {"input_ptr": pointer_type, "output_ptr": pointer_type},
                {"ROWS": rows, "COLUMNS": columns, "BLOCK_SIZE": block},
                (
                    (rows + block - 1) // block,
                    (columns + block - 1) // block,
                    1,
                ),
                [("tensor", None), ("tensor", None)],
            )
    elif operation == "slice":
        if input_rank == 0 or input_rank != output_rank:
            raise ValueError("slice input/output ranks must match in [1, 8]")
        if (
            _require_integer(parameters, "rank", minimum=1, maximum=8)
            != input_rank
        ):
            raise ValueError("slice rank parameter is inconsistent")
        starts = _require_integer_list(
            parameters,
            "starts",
            input_rank,
            minimum=0,
            maximum=2**63 - 1,
        )
        limits = _require_integer_list(
            parameters,
            "limits",
            input_rank,
            minimum=1,
            maximum=2**63 - 1,
        )
        slice_strides = _require_integer_list(
            parameters,
            "slice_strides",
            input_rank,
            minimum=1,
            maximum=2**63 - 1,
        )
        expected_output: list[int] = []
        for axis in range(input_rank):
            if (
                starts[axis] >= limits[axis]
                or limits[axis] > input_dimensions[axis]
            ):
                raise ValueError("slice range is outside input shape")
            expected_output.append(
                (limits[axis] - starts[axis] + slice_strides[axis] - 1)
                // slice_strides[axis]
            )
        if output_dimensions != expected_output:
            raise ValueError(
                "slice output shape does not match slice attributes"
            )
        input_base = sum(
            start * stride for start, stride in zip(starts, input_strides)
        )
        logical_input_dimensions = output_dimensions
        logical_input_strides = [
            stride * step for stride, step in zip(input_strides, slice_strides)
        ]
    else:
        raise ValueError(f"unknown layout operation: {operation!r}")

    leading_input = 8 - len(logical_input_dimensions)
    leading_output = 8 - output_rank
    padded_input_dimensions = [1] * leading_input + logical_input_dimensions
    padded_input_strides = [0] * leading_input + logical_input_strides
    padded_output_dimensions = [1] * leading_output + output_dimensions
    padded_output_strides = [0] * leading_output + output_strides
    block = 256
    constants: dict[str, int] = {
        "INPUT_BASE": input_base,
        "BLOCK_SIZE": block,
    }
    for axis in range(8):
        constants[f"INPUT_DIM_{axis}"] = padded_input_dimensions[axis]
        constants[f"INPUT_STRIDE_{axis}"] = padded_input_strides[axis]
        constants[f"OUTPUT_DIM_{axis}"] = padded_output_dimensions[axis]
        constants[f"OUTPUT_STRIDE_{axis}"] = padded_output_strides[axis]
    return (
        "layout_copy_kernel",
        {
            "input_ptr": pointer_type,
            "output_ptr": pointer_type,
            "n_elements": "i32",
        },
        constants,
        ((elements + block - 1) // block, 1, 1),
        [
            ("tensor", None),
            ("tensor", None),
            ("scalar_i32", "n_elements"),
        ],
    )
