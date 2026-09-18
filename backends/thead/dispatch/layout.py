# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""THead dispatch layout lowering."""

from __future__ import annotations

from typing import Any
import math
from ..codegen.abi import (
    _element_count_signature,
    _tensor_argument,
    _tensor_pointer_signature,
)
from ..dispatch.common import (
    _FLOATING_DATA_TYPES,
    _PPU_WARP_SIZE,
    _integer,
    _require_exact_fields,
    _require_list,
    _require_object,
)
from ..dispatch.tensor import (
    _dense_strides,
    _named_port_uids,
    _require_integer_array,
)


def _validate_layout_graph(
    graph: dict[str, Any], operation: str
) -> dict[str, Any]:
    tensors = _require_list(graph["tensors"], "graph.tensors")
    nodes = _require_list(graph["nodes"], "graph.nodes")
    if len(nodes) != 1 or len(tensors) != 2:
        raise ValueError(
            f"THead {operation} slice requires one node and two tensors"
        )
    node = _require_object(nodes[0], f"{operation} node")
    if node["id"] != 0 or node["type"] != operation:
        raise ValueError(f"THead {operation} requires canonical node id 0")
    if node["compute_data_type"] != "float32":
        raise ValueError(
            f"THead {operation} compute data type must be float32"
        )
    input_uid = _named_port_uids(
        node, "inputs", ("input",), operation.capitalize()
    )[0]
    output_uid = _named_port_uids(
        node, "outputs", ("output",), operation.capitalize()
    )[0]
    if input_uid == output_uid:
        raise ValueError(f"THead {operation} input/output UIDs must differ")
    tensors_by_uid = {int(tensor["uid"]): tensor for tensor in tensors}
    input_tensor = tensors_by_uid[input_uid]
    output_tensor = tensors_by_uid[output_uid]
    if (
        input_tensor["data_type"]
        not in _FLOATING_DATA_TYPES
        | {"boolean", "int32", "fp8_e4m3", "fp8_e5m2", "fp8_e8m0"}
        or output_tensor["data_type"] != input_tensor["data_type"]
    ):
        raise ValueError(
            f"THead {operation} requires matching " "supported storage types"
        )
    if input_tensor["virtual"] or output_tensor["virtual"]:
        raise ValueError(f"THead {operation} requires external tensors")
    if (
        int(input_tensor["alignment"]) < 16
        or int(output_tensor["alignment"]) < 16
    ):
        raise ValueError(f"THead {operation} requires 16-byte alignment")

    input_dimensions = [int(value) for value in input_tensor["dimensions"]]
    input_strides = [int(value) for value in input_tensor["strides"]]
    output_dimensions = [int(value) for value in output_tensor["dimensions"]]
    output_strides = [int(value) for value in output_tensor["strides"]]
    attributes = _require_object(node["attributes"], f"{operation} attributes")
    n_elements = _integer(attributes.get("n_elements"), "layout n_elements")
    if n_elements != math.prod(output_dimensions):
        raise ValueError(f"THead {operation} n_elements does not match output")

    logical_input_dimensions = input_dimensions
    logical_input_strides = input_strides
    input_base = 0
    common_fields = {
        "n_elements",
        "input_dimensions",
        "input_strides",
        "output_dimensions",
        "output_strides",
    }
    if operation == "reshape":
        _require_exact_fields(
            attributes,
            common_fields | {"input_rank", "output_rank", "reshape_mode"},
            set(),
            "Reshape attributes",
        )
        if (
            _integer(attributes["input_rank"], "reshape input_rank")
            != len(input_dimensions)
            or _integer(attributes["output_rank"], "reshape output_rank")
            != len(output_dimensions)
            or _integer(attributes["reshape_mode"], "reshape mode") != 2
            or math.prod(input_dimensions) != n_elements
        ):
            raise ValueError("THead reshape metadata is inconsistent")
        if input_strides != _dense_strides(
            input_dimensions
        ) or output_strides != _dense_strides(output_dimensions):
            raise ValueError("THead reshape slice requires contiguous tensors")
    elif operation == "transpose":
        _require_exact_fields(
            attributes,
            common_fields | {"rank", "permutation"},
            set(),
            "Transpose attributes",
        )
        rank = len(input_dimensions)
        if (
            rank == 0
            or rank != len(output_dimensions)
            or _integer(attributes["rank"], "transpose rank") != rank
        ):
            raise ValueError("THead transpose ranks are inconsistent")
        permutation = _require_integer_array(attributes, "permutation", rank)
        if sorted(permutation) != list(range(rank)) or output_dimensions != [
            input_dimensions[axis] for axis in permutation
        ]:
            raise ValueError("THead transpose permutation is invalid")
        if input_strides != _dense_strides(input_dimensions):
            raise ValueError("THead transpose input must be contiguous")
        logical_input_dimensions = output_dimensions
        logical_input_strides = [input_strides[axis] for axis in permutation]
    else:
        _require_exact_fields(
            attributes,
            common_fields | {"rank", "starts", "limits", "slice_strides"},
            set(),
            "Slice attributes",
        )
        rank = len(input_dimensions)
        if (
            rank == 0
            or rank != len(output_dimensions)
            or _integer(attributes["rank"], "slice rank") != rank
        ):
            raise ValueError("THead slice ranks are inconsistent")
        starts = _require_integer_array(attributes, "starts", rank)
        limits = _require_integer_array(attributes, "limits", rank)
        steps = _require_integer_array(attributes, "slice_strides", rank)
        expected = []
        for axis in range(rank):
            if (
                starts[axis] < 0
                or limits[axis] <= starts[axis]
                or limits[axis] > input_dimensions[axis]
                or steps[axis] <= 0
            ):
                raise ValueError("THead slice range is invalid")
            expected.append(
                (limits[axis] - starts[axis] + steps[axis] - 1) // steps[axis]
            )
        if output_dimensions != expected:
            raise ValueError("THead slice output shape is invalid")
        if input_strides != _dense_strides(input_dimensions):
            raise ValueError("THead slice input must be contiguous")
        input_base = sum(
            start * stride
            for start, stride in zip(starts, input_strides, strict=True)
        )
        logical_input_dimensions = output_dimensions
        logical_input_strides = [
            stride * step
            for stride, step in zip(input_strides, steps, strict=True)
        ]

    for name, expected in (
        ("input_dimensions", input_dimensions),
        ("input_strides", input_strides),
        ("output_dimensions", output_dimensions),
        ("output_strides", output_strides),
    ):
        if _require_integer_array(attributes, name, len(expected)) != expected:
            raise ValueError(f"THead {operation} {name} metadata mismatch")
    leading_input = 8 - len(logical_input_dimensions)
    leading_output = 8 - len(output_dimensions)
    padded_input_dimensions = [1] * leading_input + logical_input_dimensions
    padded_input_strides = [0] * leading_input + logical_input_strides
    padded_output_dimensions = [1] * leading_output + output_dimensions
    padded_output_strides = [0] * leading_output + output_strides
    constants: dict[str, int] = {"INPUT_BASE": input_base}
    for axis in range(8):
        constants[f"INPUT_DIM_{axis}"] = padded_input_dimensions[axis]
        constants[f"INPUT_STRIDE_{axis}"] = padded_input_strides[axis]
        constants[f"OUTPUT_DIM_{axis}"] = padded_output_dimensions[axis]
        constants[f"OUTPUT_STRIDE_{axis}"] = padded_output_strides[axis]
    return {
        "tensors": tensors,
        "argument_tensors": [input_tensor, output_tensor],
        "n_elements": n_elements,
        "constants": constants,
    }


def _layout_variant(
    argument_tensors: list[dict[str, Any]],
    n_elements: int,
    constants: dict[str, int],
    configuration: dict[str, Any],
    variant_id: str,
) -> dict[str, Any]:
    block_size = int(configuration["META"]["BLOCK_SIZE"])
    num_warps = int(configuration["num_warps"])
    num_stages = int(configuration["num_stages"])
    ordered_constants = [constants["INPUT_BASE"]]
    ordered_constants.extend(
        constants[f"INPUT_DIM_{axis}"] for axis in range(8)
    )
    ordered_constants.extend(
        constants[f"INPUT_STRIDE_{axis}"] for axis in range(8)
    )
    ordered_constants.extend(
        constants[f"OUTPUT_DIM_{axis}"] for axis in range(8)
    )
    ordered_constants.extend(
        constants[f"OUTPUT_STRIDE_{axis}"] for axis in range(8)
    )
    return {
        "variant_id": variant_id,
        "full_signature": (
            ",".join(
                _tensor_pointer_signature(tensor, copy_bits=True)
                for tensor in argument_tensors
            )
            + ","
            + _element_count_signature(n_elements)
            + ","
            + ",".join(str(value) for value in ordered_constants)
            + f",{block_size}"
        ),
        "argument_count": 3,
        "arguments": [
            *[
                _tensor_argument(tensor, copy_bits=True)
                for tensor in argument_tensors
            ],
            {"kind": "scalar_i32", "name": "n_elements", "value": n_elements},
        ],
        "compile_options": {
            "num_warps": num_warps,
            "num_stages": num_stages,
            "maxnreg": configuration["maxnreg"],
            "ppu_compiler_options": configuration["ppu_compiler_options"],
        },
        "launch": {
            "grid": [(n_elements + block_size - 1) // block_size, 1, 1],
            "block": [num_warps * _PPU_WARP_SIZE, 1, 1],
            "shared_memory": 0,
        },
    }
