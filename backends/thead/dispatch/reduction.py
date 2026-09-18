# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""THead dispatch reduction lowering."""

from __future__ import annotations

from typing import Any
import math
from ..codegen.abi import (
    _tensor_argument,
    _tensor_pointer_signature,
)
from ..dispatch.common import (
    _FLOATING_DATA_TYPES,
    _PPU_WARP_SIZE,
    _REDUCTION_GRAPH_MODES,
    _REDUCTION_OPERATIONS,
    _integer,
    _require_exact_fields,
    _require_list,
    _require_object,
)
from ..dispatch.tensor import (
    _dense_strides,
    _has_non_overlapping_strides,
    _named_port_uids,
)


def _validate_reduction_graph(
    graph: dict[str, Any], operation: str
) -> dict[str, Any]:
    tensors = _require_list(graph["tensors"], "graph.tensors")
    nodes = _require_list(graph["nodes"], "graph.nodes")
    if len(nodes) != 1 or len(tensors) != 2:
        raise ValueError(
            f"THead {operation} requires one node and two tensors"
        )
    node = _require_object(nodes[0], f"{operation} node")
    if node["id"] != 0 or node["type"] != operation:
        raise ValueError(f"THead {operation} requires canonical node id 0")
    if node["compute_data_type"] != "float32":
        raise ValueError(f"THead {operation} compute type must be float32")
    input_uid = _named_port_uids(node, "inputs", ("input",), "Reduction")[0]
    output_uid = _named_port_uids(node, "outputs", ("output",), "Reduction")[0]
    if input_uid == output_uid:
        raise ValueError("THead reduction tensor UIDs must differ")
    tensors_by_uid = {int(tensor["uid"]): tensor for tensor in tensors}
    input_tensor = tensors_by_uid[input_uid]
    output_tensor = tensors_by_uid[output_uid]
    if input_tensor["data_type"] not in _FLOATING_DATA_TYPES | {
        "int32"
    } or output_tensor["data_type"] not in {
        input_tensor["data_type"],
        "float32",
    }:
        raise ValueError(
            "THead reduction requires floating "
            "float32/float16/bfloat16 tensors"
        )
    if (
        input_tensor["data_type"] == "int32"
        and output_tensor["data_type"] != "float32"
    ):
        raise ValueError("THead INT32 reduction requires float32 output")
    if any(tensor["virtual"] for tensor in (input_tensor, output_tensor)):
        raise ValueError("THead reduction requires external tensors")
    scalar_bytes = (
        4 if input_tensor["data_type"] in {"float32", "int32"} else 2
    )
    if any(
        int(tensor["alignment"]) < scalar_bytes
        for tensor in (input_tensor, output_tensor)
    ):
        raise ValueError("THead reduction alignment is below scalar size")
    input_dimensions = [int(value) for value in input_tensor["dimensions"]]
    output_dimensions = [int(value) for value in output_tensor["dimensions"]]
    input_strides = [int(value) for value in input_tensor["strides"]]
    output_strides = [int(value) for value in output_tensor["strides"]]
    if not _has_non_overlapping_strides(
        input_dimensions, input_strides
    ) or not (_has_non_overlapping_strides(output_dimensions, output_strides)):
        raise ValueError("THead reduction requires non-overlapping tensors")
    input_dense = input_strides == _dense_strides(input_dimensions)
    output_dense = output_strides == _dense_strides(output_dimensions)

    attributes = _require_object(node["attributes"], "Reduction attributes")
    _require_exact_fields(
        attributes,
        {
            "mode",
            "outer",
            "reduction",
            "inner",
            "output_elements",
            "axis",
            "keep_dimensions",
        },
        set(),
        "Reduction attributes",
    )
    rank = len(input_dimensions)
    mode = _integer(attributes["mode"], "Reduction mode")
    if mode != _REDUCTION_GRAPH_MODES[operation]:
        raise ValueError("THead reduction mode does not match operation")
    axis = _integer(attributes["axis"], "Reduction axis")
    keep_dimensions = _integer(
        attributes["keep_dimensions"], "Reduction keep_dimensions"
    )
    if axis < 0 or axis >= rank or keep_dimensions not in (0, 1):
        raise ValueError("THead reduction axis/keep_dimensions is invalid")
    expected_output = list(input_dimensions)
    if keep_dimensions:
        expected_output[axis] = 1
    else:
        del expected_output[axis]
    if output_dimensions != expected_output:
        raise ValueError("THead reduction output shape is invalid")
    outer = math.prod(input_dimensions[:axis])
    extent = input_dimensions[axis]
    inner = math.prod(input_dimensions[axis + 1 :])
    output_elements = outer * inner
    actual = tuple(
        _integer(attributes[name], f"Reduction {name}")
        for name in ("outer", "reduction", "inner", "output_elements")
    )
    if actual != (outer, extent, inner, output_elements):
        raise ValueError("THead reduction attributes are inconsistent")
    if extent > 65536 or output_elements > 2**31 - 1:
        raise ValueError(
            "THead reduction shape is outside the qualified slice"
        )

    if not input_dense or not output_dense:
        if keep_dimensions:
            coordinate_dimensions = list(output_dimensions)
            input_base_strides = list(input_strides)
            input_base_strides[axis] = 0
        else:
            coordinate_dimensions = list(output_dimensions)
            input_base_strides = [
                stride
                for index, stride in enumerate(input_strides)
                if index != axis
            ]
        leading = 8 - len(coordinate_dimensions)
        padded_dimensions = [1] * leading + coordinate_dimensions
        padded_input_strides = [0] * leading + input_base_strides
        padded_output_strides = [0] * leading + output_strides
        function = "reduction_strided_kernel"
        constants = {
            "N": extent,
            "REDUCTION_STRIDE": input_strides[axis],
            "OP": _REDUCTION_OPERATIONS[operation],
        }
        for padded_axis in range(8):
            constants[f"DIM_{padded_axis}"] = padded_dimensions[padded_axis]
            constants[f"INPUT_STRIDE_{padded_axis}"] = padded_input_strides[
                padded_axis
            ]
            constants[f"OUTPUT_STRIDE_{padded_axis}"] = padded_output_strides[
                padded_axis
            ]
        rows = output_elements
        scalar_name = "output_elements"
    elif inner == 1:
        function = "reduction_2d_kernel"
        constants = {
            "N": extent,
            "stride_xm": extent,
            "stride_xn": 1,
            "OP": _REDUCTION_OPERATIONS[operation],
        }
        rows = outer
        scalar_name = "outer"
    else:
        function = "reduction_3d_kernel"
        constants = {
            "N": extent,
            "I": inner,
            "stride_xo": extent * inner,
            "stride_xr": inner,
            "stride_xi": 1,
            "OP": _REDUCTION_OPERATIONS[operation],
        }
        rows = output_elements
        scalar_name = "output_elements"
    return {
        "tensors": tensors,
        "argument_tensors": [input_tensor, output_tensor],
        "function": function,
        "constants": constants,
        "rows": rows,
        "scalar_name": scalar_name,
    }


def _reduction_variant(
    plan: dict[str, Any],
    configuration: dict[str, Any],
    variant_id: str,
) -> dict[str, Any]:
    block_m = int(configuration["META"]["BLOCK_M"])
    block_n = int(configuration["META"]["BLOCK_N"])
    constants = plan["constants"]
    if plan["function"] == "reduction_2d_kernel":
        ordered_constants = [
            constants["N"],
            constants["stride_xm"],
            constants["stride_xn"],
            constants["OP"],
            block_m,
            block_n,
        ]
    elif plan["function"] == "reduction_3d_kernel":
        ordered_constants = [
            constants["N"],
            constants["I"],
            constants["stride_xo"],
            constants["stride_xr"],
            constants["stride_xi"],
            constants["OP"],
            block_m,
            block_n,
        ]
    else:
        ordered_constants = [
            constants["N"],
            constants["REDUCTION_STRIDE"],
            *[constants[f"DIM_{axis}"] for axis in range(8)],
            *[constants[f"INPUT_STRIDE_{axis}"] for axis in range(8)],
            *[constants[f"OUTPUT_STRIDE_{axis}"] for axis in range(8)],
            constants["OP"],
            block_m,
            block_n,
        ]
    rows = int(plan["rows"])
    unit_stride_2d = plan["function"] == "reduction_2d_kernel"
    unit_stride_strided = (
        plan["function"] == "reduction_strided_kernel"
        and constants["REDUCTION_STRIDE"] == 1
    )
    if unit_stride_2d and constants["OP"] != 3:
        shared_memory = block_m * 4
    elif unit_stride_strided:
        scalar_bytes = (
            4 if plan["argument_tensors"][0]["data_type"] == "float32" else 2
        )
        shared_memory = block_m * scalar_bytes
    elif constants["OP"] == 3:
        shared_memory = block_m * block_n // 2
    else:
        shared_memory = block_n * 4
    return {
        "variant_id": variant_id,
        "full_signature": (
            ",".join(
                _tensor_pointer_signature(tensor)
                for tensor in plan["argument_tensors"]
            )
            + ",i32,"
            + ",".join(str(value) for value in ordered_constants)
        ),
        "argument_count": 3,
        "arguments": [
            *[_tensor_argument(tensor) for tensor in plan["argument_tensors"]],
            {
                "kind": "scalar_i32",
                "name": plan["scalar_name"],
                "value": rows,
            },
        ],
        "compile_options": {
            "num_warps": int(configuration["num_warps"]),
            "num_stages": int(configuration["num_stages"]),
            "maxnreg": configuration["maxnreg"],
            "ppu_compiler_options": configuration["ppu_compiler_options"],
        },
        "launch": {
            "grid": [(rows + block_m - 1) // block_m, 1, 1],
            "block": [int(configuration["num_warps"]) * _PPU_WARP_SIZE, 1, 1],
            # These sizes are part of the CUDA launch ABI emitted by Triton
            # 3.5 for the qualified reduction configurations. Cumprod needs a
            # larger 3-D scratch tile at BLOCK_M=16/BLOCK_N=64.
            "shared_memory": shared_memory,
        },
    }
