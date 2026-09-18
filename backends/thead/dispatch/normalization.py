# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""THead dispatch normalization lowering."""

from __future__ import annotations

from typing import Any
import math
from ..codegen.abi import (
    _tensor_argument,
    _tensor_pointer_signature,
)
from ..dispatch.common import (
    _FLOATING_DATA_TYPES,
    _MAX_NORMALIZATION_ELEMENTS,
    _PPU_WARP_SIZE,
    _integer,
    _require_exact_fields,
    _require_list,
    _require_object,
)
from ..dispatch.tensor import (
    _dense_strides,
    _named_port_uids,
)


def _validate_normalization_graph(
    graph: dict[str, Any], operation: str
) -> dict[str, Any]:
    layernorm = operation == "layernorm"
    tensors = _require_list(graph["tensors"], "graph.tensors")
    nodes = _require_list(graph["nodes"], "graph.nodes")
    expected_count = 6 if layernorm else 5
    if len(nodes) != 1 or len(tensors) != expected_count:
        raise ValueError(
            f"THead {operation} requires one node and {expected_count} tensors"
        )
    node = _require_object(nodes[0], f"{operation} node")
    if (
        node["id"] != 0
        or node["type"] != operation
        or node["compute_data_type"] != "float32"
    ):
        raise ValueError(
            f"THead {operation} requires a canonical float32 node"
        )
    input_uids = _named_port_uids(
        node, "inputs", ("x", "scale", "bias"), operation
    )
    output_names = (
        ("y", "mean", "inv_variance") if layernorm else ("y", "inv_variance")
    )
    output_uids = _named_port_uids(node, "outputs", output_names, operation)
    if len(set(input_uids + output_uids)) != expected_count:
        raise ValueError(
            f"THead {operation} requires distinct external tensors"
        )
    tensors_by_uid = {int(tensor["uid"]): tensor for tensor in tensors}
    ordered = [tensors_by_uid[uid] for uid in input_uids + output_uids]
    if any(
        tensor["virtual"] or int(tensor["alignment"]) < 16
        for tensor in ordered
    ):
        raise ValueError(
            f"THead {operation} requires aligned external tensors"
        )
    data_type = tensors_by_uid[input_uids[0]]["data_type"]
    data_uids = set(input_uids + output_uids[:1])
    statistic_uids = set(output_uids[1:])
    if data_type not in _FLOATING_DATA_TYPES or any(
        tensors_by_uid[uid]["data_type"] != data_type for uid in data_uids
    ):
        raise ValueError(
            f"THead {operation} requires matching data and affine tensors"
        )
    if any(
        tensors_by_uid[uid]["data_type"] != "float32" for uid in statistic_uids
    ):
        raise ValueError(f"THead {operation} statistics must use float32")
    if any(
        list(tensor["strides"]) != _dense_strides(tensor["dimensions"])
        for tensor in ordered
    ):
        raise ValueError(f"THead {operation} requires contiguous tensors")

    x = tensors_by_uid[input_uids[0]]
    scale = tensors_by_uid[input_uids[1]]
    bias = tensors_by_uid[input_uids[2]]
    y = tensors_by_uid[output_uids[0]]
    dimensions = [int(value) for value in x["dimensions"]]
    scale_dimensions = [int(value) for value in scale["dimensions"]]
    if (
        list(y["dimensions"]) != dimensions
        or len(scale_dimensions) != len(dimensions)
        or list(bias["dimensions"]) != scale_dimensions
    ):
        raise ValueError(f"THead {operation} X/Y/scale shape is invalid")
    suffix_start: int | None = None
    normalized_elements = 1
    statistic_dimensions = list(dimensions)
    for axis, (dimension, scale_dimension) in enumerate(
        zip(dimensions, scale_dimensions, strict=True)
    ):
        if suffix_start is None and scale_dimension != 1:
            suffix_start = axis
        if suffix_start is None:
            if scale_dimension != 1:
                raise ValueError(
                    f"THead {operation} scale leading dimensions are invalid"
                )
            continue
        if scale_dimension != dimension:
            raise ValueError(
                f"THead {operation} scale must describe a normalized suffix"
            )
        normalized_elements *= dimension
        statistic_dimensions[axis] = 1
    if (
        suffix_start is None
        or normalized_elements > _MAX_NORMALIZATION_ELEMENTS
        or math.prod(scale_dimensions) != normalized_elements
    ):
        raise ValueError(
            f"THead {operation} normalized suffix is outside the qualified"
            " slice"
        )
    rows = math.prod(dimensions) // normalized_elements
    if rows > 2**31 - 1:
        raise ValueError(f"THead {operation} row count exceeds int32")
    if any(
        list(tensors_by_uid[uid]["dimensions"]) != statistic_dimensions
        for uid in statistic_uids
    ):
        raise ValueError(f"THead {operation} statistic shape is invalid")

    attributes = _require_object(node["attributes"], f"{operation} attributes")
    _require_exact_fields(
        attributes,
        {"rows", "normalized_elements", "epsilon", "forward_phase"},
        set(),
        f"{operation} attributes",
    )
    if (
        _integer(attributes["rows"], f"{operation} rows") != rows
        or _integer(
            attributes["normalized_elements"],
            f"{operation} normalized_elements",
        )
        != normalized_elements
    ):
        raise ValueError(
            f"THead {operation} integer attributes are inconsistent"
        )
    raw_epsilon = attributes["epsilon"]
    if (
        isinstance(raw_epsilon, bool)
        or not isinstance(raw_epsilon, (int, float))
        or not math.isfinite(raw_epsilon)
        or raw_epsilon <= 0.0
    ):
        raise ValueError(f"THead {operation} epsilon is invalid")
    if (
        _integer(attributes["forward_phase"], f"{operation} forward_phase")
        != 2
    ):
        raise ValueError(f"THead {operation} supports TRAINING phase only")

    if layernorm:
        kernel_uids = [
            input_uids[0],
            output_uids[0],
            output_uids[1],
            output_uids[2],
            input_uids[1],
            input_uids[2],
        ]
        function = "layer_norm_kernel"
    else:
        kernel_uids = [
            input_uids[0],
            output_uids[0],
            input_uids[1],
            input_uids[2],
            output_uids[1],
        ]
        function = "rms_norm_kernel"
    return {
        "tensors": tensors,
        "argument_tensors": [tensors_by_uid[uid] for uid in kernel_uids],
        "function": function,
        "rows": rows,
        "normalized_elements": normalized_elements,
        "epsilon": float(raw_epsilon),
    }


def _normalization_variant(
    plan: dict[str, Any],
    configuration: dict[str, Any],
    variant_id: str,
) -> dict[str, Any]:
    block_size = int(configuration["META"]["BLOCK_SIZE"])
    rows_per_program = int(configuration["META"]["ROWS_PER_PROGRAM"])
    num_warps = int(configuration["num_warps"])
    pointers = ",".join(
        _tensor_pointer_signature(tensor)
        for tensor in plan["argument_tensors"]
    )
    if plan["function"] == "layer_norm_kernel":
        constants: list[float | int] = [
            plan["epsilon"],
            plan["normalized_elements"],
            block_size,
            rows_per_program,
            1,
            1,
            1,
        ]
    else:
        constants = [
            plan["normalized_elements"],
            plan["epsilon"],
            block_size,
            rows_per_program,
            1,
            1,
            1,
        ]
    # libtriton_jit checks the complete static signature, including defaults.
    constants.extend(
        [0, False, False] if plan["function"] == "layer_norm_kernel" else [0]
    )
    return {
        "variant_id": variant_id,
        "full_signature": (
            pointers + ",i32," + ",".join(str(value) for value in constants)
        ),
        "argument_count": len(plan["argument_tensors"]) + 1,
        "arguments": [
            *[_tensor_argument(tensor) for tensor in plan["argument_tensors"]],
            {
                "kind": "scalar_i32",
                "name": "rows",
                "value": int(plan["rows"]),
            },
        ],
        "compile_options": {
            "num_warps": num_warps,
            "num_stages": int(configuration["num_stages"]),
            "maxnreg": configuration["maxnreg"],
            "ppu_compiler_options": configuration["ppu_compiler_options"],
        },
        "launch": {
            "grid": [
                (int(plan["rows"]) + rows_per_program - 1) // rows_per_program,
                1,
                1,
            ],
            "block": [num_warps * _PPU_WARP_SIZE, 1, 1],
            # Codegen replaces this placeholder with compiler-
            # reported resources.
            "shared_memory": 0,
        },
    }
