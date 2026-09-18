# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""THead dispatch pointwise lowering."""

from __future__ import annotations

from typing import Any
import math
from ..codegen.abi import (
    _element_count_signature,
    _tensor_argument,
    _tensor_pointer_signature,
)
from ..dispatch.common import (
    _POINTWISE_ATTRIBUTE_DEFAULTS,
    _validate_pointwise_defaults,
    _BINARY_POINTWISE_MODES,
    _COMPARISON_OPERATIONS,
    _DATA_TYPE_BYTES,
    _FLOATING_DATA_TYPES,
    _LOGICAL_OPERATIONS,
    _PPU_WARP_SIZE,
    _UNARY_POINTWISE_MODES,
    _integer,
    _require_exact_fields,
    _require_list,
    _require_object,
)
from ..dispatch.tensor import (
    _has_non_overlapping_strides,
    _named_port_uids,
    _padded_pointwise_values,
    _same_dense_layout,
)


def _validate_binary_pointwise_graph(
    graph: dict[str, Any], operation: str
) -> dict[str, Any]:
    pointwise_mode = _BINARY_POINTWISE_MODES[operation]
    operation_label = operation.capitalize()
    tensors = _require_list(graph["tensors"], "graph.tensors")
    nodes = _require_list(graph["nodes"], "graph.nodes")
    if len(nodes) != 1 or len(tensors) != 3:
        raise ValueError(
            f"THead {operation_label} slice requires one node and three"
            " tensors"
        )
    node = _require_object(nodes[0], f"{operation_label} node")
    if node["id"] != 0 or node["type"] != operation:
        raise ValueError(
            f"THead {operation_label} slice requires canonical node id 0"
        )
    expected_compute_type = (
        "boolean"
        if operation in _COMPARISON_OPERATIONS | _LOGICAL_OPERATIONS
        else "float32"
    )
    if node["compute_data_type"] != expected_compute_type:
        raise ValueError(
            f"THead {operation_label} compute data type must be "
            f"{expected_compute_type}"
        )

    input_uids = _named_port_uids(
        node, "inputs", ("left", "right"), operation_label
    )
    output_uids = _named_port_uids(
        node, "outputs", ("output",), operation_label
    )
    ordered_uids = input_uids + output_uids
    if len(set(ordered_uids)) != 3:
        raise ValueError(
            f"THead {operation_label} input/output UIDs must be distinct"
        )
    tensors_by_uid = {int(tensor["uid"]): tensor for tensor in tensors}
    ordered_tensors = [tensors_by_uid[uid] for uid in ordered_uids]
    tensor_types = [tensor["data_type"] for tensor in ordered_tensors]
    if operation in _COMPARISON_OPERATIONS:
        if (
            tensor_types[0] not in _FLOATING_DATA_TYPES
            or tensor_types[1] != tensor_types[0]
            or tensor_types[2] != "boolean"
        ):
            raise ValueError(
                f"THead {operation_label} comparison inputs must use "
                "matching float32/float16/bfloat16 types and output must be "
                "boolean"
            )
    elif operation in _LOGICAL_OPERATIONS:
        if tensor_types != ["boolean", "boolean", "boolean"]:
            raise ValueError(
                f"THead {operation_label} slice requires boolean tensors"
            )
    elif tensor_types[0] not in _FLOATING_DATA_TYPES or any(
        data_type != tensor_types[0] for data_type in tensor_types[1:]
    ):
        raise ValueError(
            f"THead {operation_label} slice requires matching "
            "float32/float16/bfloat16 tensors"
        )
    if any(tensor["virtual"] for tensor in ordered_tensors):
        raise ValueError(
            f"THead {operation_label} slice requires external tensors"
        )
    if any(int(tensor["alignment"]) < 16 for tensor in ordered_tensors):
        raise ValueError(
            f"THead {operation_label} slice requires at least 16-byte"
            " alignment"
        )

    left_dimensions = ordered_tensors[0]["dimensions"]
    right_dimensions = ordered_tensors[1]["dimensions"]
    output_dimensions = ordered_tensors[2]["dimensions"]
    if right_dimensions != left_dimensions:
        raise ValueError(
            f"THead {operation_label} broadcast is not supported in this slice"
        )
    if output_dimensions != left_dimensions:
        raise ValueError(
            f"THead {operation_label} output shape must match both inputs"
        )
    if any(
        not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        )
        for tensor in ordered_tensors
    ):
        raise ValueError(
            f"THead {operation_label} requires non-overlapping tensor strides"
        )
    strided = not _same_dense_layout(ordered_tensors)

    attributes = _require_object(
        node["attributes"], f"{operation_label} attributes"
    )
    _require_exact_fields(
        attributes,
        {"alpha", "mode", "n_elements", "pointwise_mode"},
        set(_POINTWISE_ATTRIBUTE_DEFAULTS) | {"has_upper_clip"},
        f"{operation_label} attributes",
    )
    if not operation.endswith("_backward"):
        _validate_pointwise_defaults(attributes)
    else:
        for name, default in _POINTWISE_ATTRIBUTE_DEFAULTS.items():
            if name == "relu_upper_clip_set":
                continue
            value = attributes.get(name, default)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"THead {operation_label} {name} must be finite"
                )
        upper_clip_set = attributes.get("relu_upper_clip_set", 0)
        if (
            upper_clip_set not in (0, 1)
            or attributes.get("has_upper_clip", upper_clip_set)
            != upper_clip_set
        ):
            raise ValueError(
                f"THead {operation_label} upper clip flags are inconsistent"
            )
    mode = _integer(attributes["mode"], f"{operation_label} mode")
    if mode != pointwise_mode:
        raise ValueError(
            f"THead {operation_label} mode must be {pointwise_mode}"
        )
    pointwise_mode = _integer(
        attributes["pointwise_mode"], f"{operation_label} pointwise_mode"
    )
    if pointwise_mode != mode:
        raise ValueError(
            f"THead {operation_label} pointwise_mode must match mode"
        )
    alpha = attributes["alpha"]
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(float(alpha))
        or (operation not in {"add", "sub"} and float(alpha) != 1.0)
    ):
        raise ValueError(f"THead {operation_label} alpha is not qualified")
    alpha = float(alpha)
    n_elements = _integer(
        attributes["n_elements"], f"{operation_label} n_elements"
    )
    expected_elements = math.prod(int(value) for value in left_dimensions)
    if n_elements != expected_elements or not 1 <= n_elements <= 2**31 - 1:
        raise ValueError(
            f"THead {operation_label} n_elements does not match the output"
            " shape"
        )
    return {
        "node": node,
        "tensors": tensors,
        "ordered_uids": ordered_uids,
        "n_elements": n_elements,
        "alpha": alpha,
        "activation_parameters": {
            **_POINTWISE_ATTRIBUTE_DEFAULTS,
            **attributes,
        },
        "function": (
            (
                "activation_backward_strided_kernel"
                if strided
                else "activation_backward_contiguous_kernel"
            )
            if operation.endswith("_backward")
            else (
                "binary_strided_kernel"
                if strided
                else "binary_contiguous_kernel"
            )
        ),
        "stride_constants": (
            [
                *_padded_pointwise_values(left_dimensions, 1),
                *(
                    value
                    for tensor in ordered_tensors
                    for value in _padded_pointwise_values(tensor["strides"], 0)
                ),
            ]
            if strided
            else None
        ),
    }


def _validate_binary_select_graph(graph: dict[str, Any]) -> dict[str, Any]:
    tensors = _require_list(graph["tensors"], "graph.tensors")
    nodes = _require_list(graph["nodes"], "graph.nodes")
    if len(nodes) != 1 or len(tensors) != 4:
        raise ValueError(
            "THead BinarySelect requires one node and four tensors"
        )
    node = _require_object(nodes[0], "BinarySelect node")
    if node["id"] != 0 or node["type"] != "binary_select":
        raise ValueError("THead BinarySelect requires canonical node id 0")
    if node["compute_data_type"] != "float32":
        raise ValueError(
            "THead BinarySelect compute data type must be float32"
        )
    uids = _named_port_uids(node, "inputs", ("a", "b", "t"), "BinarySelect")
    uids += _named_port_uids(node, "outputs", ("output",), "BinarySelect")
    if len(set(uids)) != 4:
        raise ValueError("THead BinarySelect tensor UIDs must be distinct")
    by_uid = {int(tensor["uid"]): tensor for tensor in tensors}
    ordered = [by_uid[uid] for uid in uids]
    dtype = ordered[0]["data_type"]
    if dtype not in _FLOATING_DATA_TYPES or [
        t["data_type"] for t in ordered
    ] != [dtype, dtype, "boolean", dtype]:
        raise ValueError(
            "THead BinarySelect requires matching floating values/output and"
            " boolean mask"
        )
    dimensions = ordered[-1]["dimensions"]
    if any(t["dimensions"] != dimensions for t in ordered):
        raise ValueError("THead BinarySelect requires matching shapes")
    if any(t["virtual"] or int(t["alignment"]) < 16 for t in ordered):
        raise ValueError(
            "THead BinarySelect requires external tensors with 16-byte"
            " alignment"
        )
    if any(
        not _has_non_overlapping_strides(t["dimensions"], t["strides"])
        for t in ordered
    ):
        raise ValueError("THead BinarySelect requires non-overlapping strides")
    attributes = _require_object(node["attributes"], "BinarySelect attributes")
    _require_exact_fields(
        attributes, {"mode", "n_elements"}, set(), "BinarySelect attributes"
    )
    if _integer(attributes["mode"], "BinarySelect mode") != 41:
        raise ValueError("THead BinarySelect mode must be 41")
    elements = _integer(attributes["n_elements"], "BinarySelect n_elements")
    if elements != math.prod(dimensions) or not 1 <= elements <= 2**31 - 1:
        raise ValueError(
            "THead BinarySelect element count does not match output"
        )
    strided = not _same_dense_layout(ordered)
    return {
        "node": node,
        "tensors": tensors,
        "ordered_uids": uids,
        "n_elements": elements,
        "alpha": 1.0,
        "function": (
            "binary_select_strided_kernel"
            if strided
            else "binary_select_tensor_kernel"
        ),
        "stride_constants": (
            [
                *_padded_pointwise_values(dimensions, 1),
                *(
                    value
                    for tensor in ordered
                    for value in _padded_pointwise_values(tensor["strides"], 0)
                ),
            ]
            if strided
            else None
        ),
    }


def _validate_unary_pointwise_graph(
    graph: dict[str, Any], operation: str
) -> dict[str, Any]:
    operation_label = operation.capitalize()
    tensors = _require_list(graph["tensors"], "graph.tensors")
    nodes = _require_list(graph["nodes"], "graph.nodes")
    if len(nodes) != 1 or len(tensors) != 2:
        raise ValueError(
            f"THead {operation_label} slice requires one node and two tensors"
        )
    node = _require_object(nodes[0], f"{operation_label} node")
    if node["id"] != 0 or node["type"] != operation:
        raise ValueError(
            f"THead {operation_label} slice requires canonical node id 0"
        )
    expected_compute_type = (
        "boolean" if operation in _LOGICAL_OPERATIONS else "float32"
    )
    if node["compute_data_type"] != expected_compute_type:
        raise ValueError(
            f"THead {operation_label} compute data type must be "
            f"{expected_compute_type}"
        )

    input_uids = _named_port_uids(node, "inputs", ("input",), operation_label)
    output_uids = _named_port_uids(
        node, "outputs", ("output",), operation_label
    )
    ordered_uids = input_uids + output_uids
    if len(set(ordered_uids)) != 2:
        raise ValueError(
            f"THead {operation_label} input/output UIDs must be distinct"
        )
    tensors_by_uid = {int(tensor["uid"]): tensor for tensor in tensors}
    ordered_tensors = [tensors_by_uid[uid] for uid in ordered_uids]
    tensor_types = [tensor["data_type"] for tensor in ordered_tensors]
    valid_types = (
        tensor_types == ["boolean", "boolean"]
        if operation in _LOGICAL_OPERATIONS
        else (
            tensor_types[0] in _FLOATING_DATA_TYPES
            or (
                operation == "identity"
                and tensor_types[0]
                in {"boolean", "int32", "fp8_e4m3", "fp8_e5m2", "fp8_e8m0"}
            )
        )
        and tensor_types[1] == tensor_types[0]
    )
    if not valid_types:
        raise ValueError(
            f"THead {operation_label} slice requires matching "
            "float32/float16/bfloat16 tensors or boolean logical tensors"
        )
    if any(tensor["virtual"] for tensor in ordered_tensors):
        raise ValueError(
            f"THead {operation_label} slice requires external tensors"
        )
    if any(int(tensor["alignment"]) < 16 for tensor in ordered_tensors):
        raise ValueError(
            f"THead {operation_label} slice requires at least 16-byte"
            " alignment"
        )

    input_dimensions = ordered_tensors[0]["dimensions"]
    output_dimensions = ordered_tensors[1]["dimensions"]
    if output_dimensions != input_dimensions:
        raise ValueError(
            f"THead {operation_label} output shape must match the input shape"
        )
    if any(
        not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        )
        for tensor in ordered_tensors
    ):
        raise ValueError(
            f"THead {operation_label} requires non-overlapping tensor strides"
        )
    strided = not _same_dense_layout(ordered_tensors)

    attributes = _require_object(
        node["attributes"], f"{operation_label} attributes"
    )
    _require_exact_fields(
        attributes,
        {
            "elu_alpha",
            "has_upper_clip",
            "lower_clip",
            "mode",
            "n_elements",
            "negative_slope",
            "relu_lower_clip",
            "relu_lower_clip_slope",
            "relu_upper_clip",
            "relu_upper_clip_set",
            "softplus_beta",
            "swish_beta",
            "upper_clip",
        },
        set(),
        f"{operation_label} attributes",
    )
    expected_defaults = {
        "elu_alpha": 1.0,
        "has_upper_clip": 0,
        "lower_clip": 0.0,
        "relu_lower_clip": 0.0,
        "relu_upper_clip": 0.0,
        "softplus_beta": 1.0,
        "swish_beta": 1.25 if operation == "swish" else 1.0,
        "upper_clip": 0.0,
    }
    for name, expected in expected_defaults.items():
        value = attributes[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) != float(expected)
        ):
            raise ValueError(
                f"THead {operation_label} slice requires default attributes"
            )
    negative_slope = attributes["negative_slope"]
    lowered_negative_slope = attributes["relu_lower_clip_slope"]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in (negative_slope, lowered_negative_slope)
    ):
        raise ValueError(
            f"THead {operation_label} negative slope must be finite"
        )
    negative_slope = float(negative_slope)
    lowered_negative_slope = float(lowered_negative_slope)
    qualified_slopes = (
        {0.0, 0.20000000298023224} if operation == "relu" else {0.0}
    )
    if (
        negative_slope != lowered_negative_slope
        or negative_slope not in qualified_slopes
    ):
        raise ValueError(
            f"THead {operation_label} negative slope is not qualified"
        )
    mode = _integer(attributes["mode"], f"{operation_label} mode")
    if mode != _UNARY_POINTWISE_MODES[operation]:
        raise ValueError(
            f"THead {operation_label} mode must be "
            f"{_UNARY_POINTWISE_MODES[operation]}"
        )
    if attributes["relu_upper_clip_set"] is not False:
        raise ValueError(
            f"THead {operation_label} slice requires default attributes"
        )
    n_elements = _integer(
        attributes["n_elements"], f"{operation_label} n_elements"
    )
    expected_elements = math.prod(int(value) for value in input_dimensions)
    if n_elements != expected_elements or not 1 <= n_elements <= 2**31 - 1:
        raise ValueError(
            f"THead {operation_label} n_elements does not match the output"
            " shape"
        )
    return {
        "node": node,
        "tensors": tensors,
        "ordered_uids": ordered_uids,
        "n_elements": n_elements,
        "function": (
            "unary_pointwise_strided_kernel"
            if strided
            else "unary_pointwise_contiguous_kernel"
        ),
        "stride_constants": (
            [
                *_padded_pointwise_values(input_dimensions, 1),
                *(
                    value
                    for tensor in ordered_tensors
                    for value in _padded_pointwise_values(tensor["strides"], 0)
                ),
                1,
            ]
            if strided
            else None
        ),
        "kernel_constants": {
            "negative_slope": negative_slope,
            "lower_clip": float(attributes["lower_clip"]),
            "upper_clip": float(attributes["upper_clip"]),
            "has_upper_clip": int(attributes["has_upper_clip"]),
            "swish_beta": float(attributes["swish_beta"]),
            "elu_alpha": float(attributes["elu_alpha"]),
            "softplus_beta": float(attributes["softplus_beta"]),
        },
    }


def _validate_add_square_graph(graph: dict[str, Any]) -> dict[str, Any]:
    tensors = _require_list(graph["tensors"], "graph.tensors")
    nodes = _require_list(graph["nodes"], "graph.nodes")
    if len(nodes) != 2 or len(tensors) != 4:
        raise ValueError("THead AddSquare requires two nodes and four tensors")
    multiply = _require_object(nodes[0], "AddSquare Mul node")
    add = _require_object(nodes[1], "AddSquare Add node")
    if multiply["id"] != 0 or multiply["type"] != "mul":
        raise ValueError("THead AddSquare requires canonical Mul node id 0")
    if add["id"] != 1 or add["type"] != "add":
        raise ValueError("THead AddSquare requires canonical Add node id 1")
    if any(node["compute_data_type"] != "float32" for node in (multiply, add)):
        raise ValueError("THead AddSquare compute data type must be float32")

    multiply_inputs = _named_port_uids(
        multiply, "inputs", ("left", "right"), "AddSquare Mul"
    )
    square_uids = _named_port_uids(
        multiply, "outputs", ("output",), "AddSquare Mul"
    )
    add_inputs = _named_port_uids(
        add, "inputs", ("left", "right"), "AddSquare Add"
    )
    output_uids = _named_port_uids(
        add, "outputs", ("output",), "AddSquare Add"
    )
    if multiply_inputs[0] != multiply_inputs[1]:
        raise ValueError("THead AddSquare Mul must square one input")
    right_uid = multiply_inputs[0]
    square_uid = square_uids[0]
    left_uid = add_inputs[0]
    output_uid = output_uids[0]
    if add_inputs[1] != square_uid:
        raise ValueError("THead AddSquare dataflow must consume Mul output")
    if len({left_uid, right_uid, square_uid, output_uid}) != 4:
        raise ValueError("THead AddSquare tensor roles must be distinct")

    tensors_by_uid = {int(tensor["uid"]): tensor for tensor in tensors}
    role_tensors = [
        tensors_by_uid[uid]
        for uid in (right_uid, left_uid, square_uid, output_uid)
    ]
    if role_tensors[0]["data_type"] not in _FLOATING_DATA_TYPES or any(
        tensor["data_type"] != role_tensors[0]["data_type"]
        for tensor in role_tensors[1:]
    ):
        raise ValueError(
            "THead AddSquare requires matching "
            "float32/float16/bfloat16 tensors"
        )
    if (
        role_tensors[0]["virtual"]
        or role_tensors[1]["virtual"]
        or not role_tensors[2]["virtual"]
        or role_tensors[3]["virtual"]
    ):
        raise ValueError(
            "THead AddSquare requires one virtual square intermediate"
        )
    if any(int(tensor["alignment"]) < 16 for tensor in role_tensors):
        raise ValueError("THead AddSquare requires 16-byte tensor alignment")
    dimensions = role_tensors[0]["dimensions"]
    if any(tensor["dimensions"] != dimensions for tensor in role_tensors[1:]):
        raise ValueError("THead AddSquare requires identical tensor shapes")
    if any(
        not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        )
        for tensor in role_tensors
    ):
        raise ValueError("THead AddSquare requires non-overlapping tensors")
    strided = not _same_dense_layout(
        [role_tensors[index] for index in (0, 1, 3)]
    )
    n_elements = math.prod(int(value) for value in dimensions)
    if not 1 <= n_elements <= 2**31 - 1:
        raise ValueError("THead AddSquare element count is outside int32")

    for node, expected_mode, label in (
        (multiply, _BINARY_POINTWISE_MODES["mul"], "Mul"),
        (add, _BINARY_POINTWISE_MODES["add"], "Add"),
    ):
        attributes = _require_object(
            node["attributes"], f"AddSquare {label} attributes"
        )
        _require_exact_fields(
            attributes,
            {"alpha", "mode", "n_elements", "pointwise_mode"},
            set(_POINTWISE_ATTRIBUTE_DEFAULTS) | {"has_upper_clip"},
            f"AddSquare {label} attributes",
        )
        _validate_pointwise_defaults(attributes)
        mode = _integer(attributes["mode"], f"AddSquare {label} mode")
        lowered_mode = _integer(
            attributes["pointwise_mode"],
            f"AddSquare {label} pointwise_mode",
        )
        if mode != expected_mode or lowered_mode != expected_mode:
            raise ValueError(
                f"THead AddSquare {label} mode must be {expected_mode}"
            )
        alpha = attributes["alpha"]
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(float(alpha))
            or float(alpha) != 1.0
        ):
            raise ValueError(f"THead AddSquare {label} alpha must equal 1")
        if (
            _integer(attributes["n_elements"], f"AddSquare {label} n_elements")
            != n_elements
        ):
            raise ValueError(
                f"THead AddSquare {label} n_elements does not match shape"
            )
    return {
        "tensors": tensors,
        "argument_tensors": [
            tensors_by_uid[right_uid],
            tensors_by_uid[left_uid],
            tensors_by_uid[output_uid],
        ],
        "virtual_tensor": tensors_by_uid[square_uid],
        "n_elements": n_elements,
        "function": (
            "add_square_strided_kernel"
            if strided
            else "add_square_contiguous_kernel"
        ),
        "stride_constants": (
            [
                *_padded_pointwise_values(dimensions, 1),
                *(
                    value
                    for tensor in (
                        tensors_by_uid[right_uid],
                        tensors_by_uid[left_uid],
                        tensors_by_uid[output_uid],
                    )
                    for value in _padded_pointwise_values(tensor["strides"], 0)
                ),
            ]
            if strided
            else None
        ),
    }


def _binary_pointwise_variant(
    argument_tensors: list[dict[str, Any]],
    n_elements: int,
    pointwise_mode: int,
    alpha: float,
    stride_constants: list[int] | None,
    configuration: dict[str, Any],
    variant_id: str,
    activation_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    activation = {
        **_POINTWISE_ATTRIBUTE_DEFAULTS,
        **(activation_parameters or {}),
    }
    backward_constants = "," + ",".join(
        str(activation[name])
        for name in (
            "relu_lower_clip_slope",
            "relu_lower_clip",
            "relu_upper_clip",
            "relu_upper_clip_set",
            "swish_beta",
            "elu_alpha",
            "softplus_beta",
        )
    )
    block_size = int(configuration["META"]["BLOCK_SIZE"])
    num_warps = int(configuration["num_warps"])
    num_stages = int(configuration["num_stages"])
    shared_memory = (
        num_warps * 4 if pointwise_mode == 23 and block_size >= 1024 else 0
    )
    if pointwise_mode == 41:
        width = _DATA_TYPE_BYTES[argument_tensors[-1]["data_type"]]
        if (
            stride_constants is not None
            and block_size > num_warps * _PPU_WARP_SIZE
        ):
            shared_memory = min(block_size * width, 2048)
        elif stride_constants is None and block_size == 1024 and width == 4:
            shared_memory = 4096
    return {
        "variant_id": variant_id,
        "full_signature": (
            ",".join(
                _tensor_pointer_signature(tensor)
                for tensor in argument_tensors
            )
            + (
                ",i32"
                if pointwise_mode == 41
                else "," + _element_count_signature(n_elements)
            )
            + (
                ""
                if stride_constants is None
                else "," + ",".join(str(value) for value in stride_constants)
            )
            + (
                f",{block_size}"
                if pointwise_mode == 41
                else f",{pointwise_mode},{alpha},{block_size}"
            )
            + (
                (
                    backward_constants
                    + (",True" if stride_constants is None else "")
                )
                if pointwise_mode == 40 or 42 <= pointwise_mode <= 48
                else ""
            )
        ),
        "argument_count": len(argument_tensors) + 1,
        "arguments": [
            *[_tensor_argument(tensor) for tensor in argument_tensors],
            {
                "kind": "scalar_i32",
                "name": "n_elements",
                "value": n_elements,
            },
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
            # Qualified from native PPU metadata for both layouts and all
            # value types, including the default 1024-element performance tile.
            "shared_memory": shared_memory,
        },
    }


def _add_square_variant(
    argument_tensors: list[dict[str, Any]],
    n_elements: int,
    stride_constants: list[int] | None,
    configuration: dict[str, Any],
    variant_id: str,
) -> dict[str, Any]:
    block_size = int(configuration["META"]["BLOCK_SIZE"])
    num_warps = int(configuration["num_warps"])
    num_stages = int(configuration["num_stages"])
    return {
        "variant_id": variant_id,
        "full_signature": (
            ",".join(
                _tensor_pointer_signature(tensor)
                for tensor in argument_tensors
            )
            + ",i32"
            + (
                ""
                if stride_constants is None
                else "," + ",".join(str(value) for value in stride_constants)
            )
            + f",{block_size}"
        ),
        "argument_count": 4,
        "arguments": [
            *[_tensor_argument(tensor) for tensor in argument_tensors],
            {
                "kind": "scalar_i32",
                "name": "n_elements",
                "value": n_elements,
            },
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


def _unary_pointwise_variant(
    argument_tensors: list[dict[str, Any]],
    n_elements: int,
    pointwise_mode: int,
    stride_constants: list[int] | None,
    kernel_constants: dict[str, float | int],
    configuration: dict[str, Any],
    variant_id: str,
) -> dict[str, Any]:
    block_size = int(configuration["META"]["BLOCK_SIZE"])
    num_warps = int(configuration["num_warps"])
    num_stages = int(configuration["num_stages"])
    return {
        "variant_id": variant_id,
        "full_signature": (
            ",".join(
                _tensor_pointer_signature(
                    tensor, copy_bits=pointwise_mode == 5
                )
                for tensor in argument_tensors
            )
            + ","
            + _element_count_signature(n_elements)
            + (
                ""
                if stride_constants is None
                else "," + ",".join(str(value) for value in stride_constants)
            )
            + ","
            f"{pointwise_mode},{kernel_constants['negative_slope']},"
            f"{kernel_constants['lower_clip']},"
            f"{kernel_constants['upper_clip']},"
            f"{kernel_constants['has_upper_clip']},"
            f"{kernel_constants['swish_beta']},"
            f"{kernel_constants['elu_alpha']},"
            f"{kernel_constants['softplus_beta']},1,{block_size}"
        ),
        "argument_count": 3,
        "arguments": [
            *[
                _tensor_argument(tensor, copy_bits=pointwise_mode == 5)
                for tensor in argument_tensors
            ],
            {
                "kind": "scalar_i32",
                "name": "n_elements",
                "value": n_elements,
            },
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
