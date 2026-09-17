# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Hygon dispatch pointwise."""

from __future__ import annotations

from ..dispatch.common import (
    FLOAT_TYPES,
    POINTER_TYPES,
    _require_integer,
    _require_list,
    _require_number,
    _require_object,
)
from ..dispatch.tensor_metadata import (
    _is_physically_dense,
)
from dataclasses import dataclass
from typing import Any
import math


POINTWISE_MODES = {
    "add": 1,
    "relu": 2,
    "sqrt": 3,
    "erf": 4,
    "identity": 5,
    "exp": 6,
    "log": 7,
    "neg": 8,
    "abs": 9,
    "ceil": 10,
    "cos": 11,
    "floor": 12,
    "rsqrt": 13,
    "sin": 14,
    "tan": 15,
    "reciprocal": 16,
    "sub": 17,
    "mul": 18,
    "div": 19,
    "min": 20,
    "max": 21,
    "mod": 22,
    "pow": 23,
    "logical_not": 24,
    "cmp_eq": 25,
    "cmp_neq": 26,
    "cmp_gt": 27,
    "cmp_ge": 28,
    "cmp_lt": 29,
    "cmp_le": 30,
    "logical_and": 31,
    "logical_or": 32,
    "sigmoid": 33,
    "tanh": 34,
    "elu": 35,
    "gelu": 36,
    "softplus": 37,
    "swish": 38,
    "gelu_approx_tanh": 39,
    "sigmoid_backward": 40,
    "relu_backward": 42,
    "tanh_backward": 43,
    "elu_backward": 44,
    "gelu_backward": 45,
    "softplus_backward": 46,
    "swish_backward": 47,
    "gelu_approx_tanh_backward": 48,
    "binary_select": 41,
}


@dataclass(frozen=True)
class PointwiseSchema:
    """Declarative Graph IR contract for one pointwise operation."""

    family: str
    input_ports: tuple[str, ...]
    output_ports: tuple[str, ...]
    mode: int
    data_type_policy: str
    allow_alpha: bool = False


def _pointwise_schemas() -> dict[str, PointwiseSchema]:
    schemas: dict[str, PointwiseSchema] = {}

    def register(
        operations: tuple[str, ...],
        *,
        family: str,
        inputs: tuple[str, ...],
        data_type_policy: str,
        allow_alpha: bool = False,
    ) -> None:
        for operation in operations:
            schemas[operation] = PointwiseSchema(
                family=family,
                input_ports=inputs,
                output_ports=("output",),
                mode=POINTWISE_MODES[operation],
                data_type_policy=data_type_policy,
                allow_alpha=allow_alpha,
            )

    register(
        ("add", "sub"),
        family="binary",
        inputs=("left", "right"),
        data_type_policy="binary_numeric",
        allow_alpha=True,
    )
    register(
        (
            "mul",
            "div",
            "min",
            "max",
            "mod",
            "pow",
            "sigmoid_backward",
            "relu_backward",
            "tanh_backward",
            "elu_backward",
            "gelu_backward",
            "softplus_backward",
            "swish_backward",
            "gelu_approx_tanh_backward",
        ),
        family="binary",
        inputs=("left", "right"),
        data_type_policy="binary_numeric",
    )
    register(
        ("cmp_eq", "cmp_neq", "cmp_gt", "cmp_ge", "cmp_lt", "cmp_le"),
        family="binary",
        inputs=("left", "right"),
        data_type_policy="binary_comparison",
    )
    register(
        ("logical_and", "logical_or"),
        family="binary",
        inputs=("left", "right"),
        data_type_policy="binary_boolean",
    )
    register(
        (
            "relu",
            "sqrt",
            "erf",
            "identity",
            "exp",
            "log",
            "neg",
            "abs",
            "ceil",
            "cos",
            "floor",
            "rsqrt",
            "sin",
            "tan",
            "reciprocal",
            "sigmoid",
            "tanh",
            "elu",
            "gelu",
            "softplus",
            "swish",
            "gelu_approx_tanh",
        ),
        family="unary",
        inputs=("input",),
        data_type_policy="unary_numeric",
    )
    register(
        ("logical_not",),
        family="unary",
        inputs=("input",),
        data_type_policy="unary_boolean",
    )
    register(
        ("binary_select",),
        family="ternary",
        inputs=("a", "b", "t"),
        data_type_policy="ternary_select",
    )
    return schemas


POINTWISE_SCHEMAS = _pointwise_schemas()


def _parse_port(
    port_value: object,
    expected_name: str,
    direction: str,
    tensor_registry: dict[int, dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    port = _require_object(port_value, f"node.{direction}")
    if port.get("name") != expected_name:
        raise ValueError(
            f"node {direction} port must be named {expected_name!r}"
        )
    optional = port.get("optional", False)
    if not isinstance(optional, bool) or optional:
        raise ValueError(
            "Hygon pointwise operations require every tensor port"
        )
    uid = port.get("uid")
    if isinstance(uid, bool) or not isinstance(uid, int) or uid <= 0:
        raise ValueError(f"node {direction} UID is invalid")
    try:
        tensor = tensor_registry[uid]
    except KeyError as error:
        raise ValueError(
            f"node references unknown tensor UID {uid}"
        ) from error
    return uid, tensor


def _parse_pointwise_node(
    node_value: object,
    position: int,
    node_count: int,
    tensor_registry: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    node = _require_object(node_value, f"graph.nodes[{position}]")
    node_id = node.get("id")
    if (
        isinstance(node_id, bool)
        or not isinstance(node_id, int)
        or node_id < 0
        or node_id >= node_count
    ):
        raise ValueError("graph node ID is invalid")
    operation = node.get("type")
    if not isinstance(operation, str) or operation not in POINTWISE_SCHEMAS:
        raise ValueError(
            f"Hygon compiler does not support graph operation {operation!r}"
        )
    schema = POINTWISE_SCHEMAS[operation]
    inputs = _require_list(node.get("inputs"), "node.inputs")
    outputs = _require_list(node.get("outputs"), "node.outputs")
    if len(inputs) != len(schema.input_ports) or len(outputs) != len(
        schema.output_ports
    ):
        raise ValueError(
            f"{schema.family} pointwise node port count is invalid"
        )

    input_uids: list[int] = []
    input_tensors: list[dict[str, Any]] = []
    for index, expected_name in enumerate(schema.input_ports):
        uid, tensor = _parse_port(
            inputs[index], expected_name, f"input[{index}]", tensor_registry
        )
        input_uids.append(uid)
        input_tensors.append(tensor)
    output_uids: list[int] = []
    output_tensors: list[dict[str, Any]] = []
    for index, expected_name in enumerate(schema.output_ports):
        uid, tensor = _parse_port(
            outputs[index], expected_name, f"output[{index}]", tensor_registry
        )
        output_uids.append(uid)
        output_tensors.append(tensor)

    compute_data_type = node.get("compute_data_type")
    if (
        not isinstance(compute_data_type, str)
        or compute_data_type not in POINTER_TYPES
    ):
        raise ValueError(
            f"node compute_data_type is unsupported: {compute_data_type!r}"
        )
    attributes = _require_object(
        node.get("attributes"), f"graph.nodes[{position}].attributes"
    )
    return {
        "id": node_id,
        "operation": operation,
        "schema": schema,
        "compute_data_type": compute_data_type,
        "parameters": attributes,
        "tensors": [*input_tensors, *output_tensors],
        "tensor_roles": [*schema.input_ports, *schema.output_ports],
        "input_uids": input_uids,
        "output_uids": output_uids,
    }


def _broadcast_dimensions(
    inputs: list[dict[str, Any]], output: dict[str, Any], family: str
) -> int:
    rank = max(len(tensor["dimensions"]) for tensor in inputs)
    if len(output["dimensions"]) != rank:
        raise ValueError(
            f"{family} pointwise output rank does not match broadcast result"
        )
    expected = [1] * rank
    for trailing in range(rank):
        result_dimension = 1
        for tensor in inputs:
            dimension = (
                tensor["dimensions"][-1 - trailing]
                if trailing < len(tensor["dimensions"])
                else 1
            )
            if result_dimension not in (1, dimension) and dimension != 1:
                raise ValueError(
                    f"{family} pointwise inputs are not broadcastable"
                )
            result_dimension = max(result_dimension, dimension)
        expected[-1 - trailing] = result_dimension
    if output["dimensions"] != expected:
        raise ValueError(
            f"{family} pointwise output shape does not match broadcast result"
        )
    return rank


def _broadcast_stride_constants(
    inputs: list[dict[str, Any]],
    output: dict[str, Any],
    prefixes: tuple[str, ...],
    family: str,
) -> dict[str, int]:
    if len(inputs) != len(prefixes):
        raise ValueError("pointwise stride schema is inconsistent")
    rank = _broadcast_dimensions(inputs, output, family)

    def effective_strides(tensor: dict[str, Any]) -> list[int]:
        leading = rank - len(tensor["dimensions"])
        dimensions = [1] * leading + tensor["dimensions"]
        strides = [0] * leading + tensor["strides"]
        return [
            0 if dimension == 1 else stride
            for dimension, stride in zip(dimensions, strides, strict=True)
        ]

    leading = 8 - rank
    constants: dict[str, int] = {
        f"DIM_{axis}": value
        for axis, value in enumerate([1] * leading + output["dimensions"])
    }
    for prefix, tensor in zip(prefixes, inputs, strict=True):
        values = [0] * leading + effective_strides(tensor)
        for axis, value in enumerate(values):
            constants[f"{prefix}_{axis}"] = value
    for axis, value in enumerate([0] * leading + output["strides"]):
        constants[f"OUTPUT_STRIDE_{axis}"] = value
    return constants


def _unary_stride_constants(
    input_tensor: dict[str, Any], output: dict[str, Any]
) -> dict[str, int]:
    if input_tensor["dimensions"] != output["dimensions"]:
        raise ValueError("unary pointwise input/output shapes must match")
    rank = len(output["dimensions"])
    leading = 8 - rank
    constants: dict[str, int] = {
        f"DIM_{axis}": value
        for axis, value in enumerate([1] * leading + output["dimensions"])
    }
    for prefix, values in (
        ("INPUT_STRIDE", [0] * leading + input_tensor["strides"]),
        ("OUTPUT_STRIDE", [0] * leading + output["strides"]),
    ):
        for axis, value in enumerate(values):
            constants[f"{prefix}_{axis}"] = value
    return constants


def _can_use_dense_pointwise_kernel(
    tensors: list[dict[str, Any]],
) -> bool:
    output = tensors[-1]
    return all(
        tensor["dimensions"] == output["dimensions"]
        and tensor["strides"] == output["strides"]
        and _is_physically_dense(tensor)
        for tensor in tensors
    )


def _validate_pointwise_data_types(
    schema: PointwiseSchema,
    operation: str,
    compute_data_type: str,
    tensors: list[dict[str, Any]],
) -> None:
    data_types = [tensor["data_type"] for tensor in tensors]
    numeric = FLOAT_TYPES | {"int32"}
    policy = schema.data_type_policy
    if policy == "binary_numeric":
        valid = len(set(data_types)) == 1 and data_types[0] in numeric
        expected_compute = {"float32"}
    elif policy == "binary_comparison":
        valid = (
            data_types[0] == data_types[1]
            and data_types[0] in numeric
            and data_types[2] == "boolean"
        )
        expected_compute = {"boolean"}
    elif policy == "binary_boolean":
        valid = data_types == ["boolean", "boolean", "boolean"]
        expected_compute = {"boolean"}
    elif policy == "unary_numeric":
        valid = data_types[0] == data_types[1] and (
            data_types[0] in POINTER_TYPES
            if operation == "identity"
            else data_types[0] in FLOAT_TYPES
        )
        expected_compute = {"float32"}
    elif policy == "unary_boolean":
        valid = data_types == ["boolean", "boolean"]
        expected_compute = {"boolean"}
    elif policy == "ternary_select":
        valid = (
            data_types[0] == data_types[1] == data_types[3]
            and data_types[0] in numeric
            and data_types[2] == "boolean"
        )
        expected_compute = {"float32"}
    else:
        raise ValueError(f"unknown pointwise data-type policy {policy!r}")
    if operation.endswith("_backward") and data_types[0] not in FLOAT_TYPES:
        valid = False
    if not valid:
        raise ValueError(
            f"{operation} tensor data types violate {policy} policy"
        )
    if compute_data_type not in expected_compute:
        raise ValueError(
            f"{operation} compute_data_type {compute_data_type!r} is not "
            f"implemented by the Hygon {policy} kernel; expected one of "
            f"{sorted(expected_compute)!r}"
        )


def _pointwise_elements(
    parameters: dict[str, Any], output: dict[str, Any], family: str
) -> int:
    elements = _require_integer(parameters, "n_elements")
    if elements != math.prod(output["dimensions"]):
        raise ValueError(
            f"parameters.n_elements is inconsistent with {family} output"
        )
    return elements


def _schema_binary_configuration(
    operation: str,
    schema: PointwiseSchema,
    compute_data_type: str,
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | float],
    tuple[int, int, int],
]:
    left, right, output = tensors
    _validate_pointwise_data_types(
        schema, operation, compute_data_type, tensors
    )
    elements = _pointwise_elements(parameters, output, schema.family)
    pointwise_mode = _require_integer(
        parameters, "pointwise_mode", minimum=1, maximum=48
    )
    if pointwise_mode != schema.mode:
        raise ValueError(
            "parameters.pointwise_mode is inconsistent with node type"
        )
    alpha = _require_number(parameters, "alpha", default=1.0)
    if left["data_type"] == "int32":
        if alpha != math.trunc(alpha) or not -(2**31) <= alpha < 2**31:
            raise ValueError("INT32 alpha must fit signed int32")
        alpha = int(alpha)
    if not schema.allow_alpha and alpha != 1.0:
        raise ValueError("pointwise alpha is supported only by add and sub")
    if operation.endswith("_backward") and any(
        tensor["dimensions"] != output["dimensions"] for tensor in tensors
    ):
        raise ValueError("activation backward tensors must have equal shapes")

    strided_constants = _broadcast_stride_constants(
        [left, right],
        output,
        ("LEFT_STRIDE", "RIGHT_STRIDE"),
        schema.family,
    )
    block_size = 256
    constants: dict[str, int | float] = {
        "OP_KIND": pointwise_mode,
        "ALPHA": alpha,
        "BLOCK_SIZE": block_size,
    }
    dense = _can_use_dense_pointwise_kernel(tensors)
    function_name = (
        "binary_contiguous_kernel" if dense else "binary_strided_kernel"
    )
    if not dense:
        constants.update(strided_constants)
    if operation.endswith("_backward"):
        function_name = (
            "activation_backward_contiguous_kernel"
            if dense
            else "activation_backward_strided_kernel"
        )
        constants.update(
            {
                "NEGATIVE_SLOPE": _require_number(
                    parameters, "relu_lower_clip_slope", default=0.0
                ),
                "LOWER_CLIP": _require_number(
                    parameters, "relu_lower_clip", default=0.0
                ),
                "UPPER_CLIP": _require_number(
                    parameters, "relu_upper_clip", default=0.0
                ),
                "HAS_UPPER_CLIP": bool(
                    _require_integer(
                        {
                            "has_upper_clip": parameters.get(
                                "has_upper_clip", 0
                            )
                        },
                        "has_upper_clip",
                        minimum=0,
                        maximum=1,
                    )
                ),
                "SWISH_BETA": _require_number(
                    parameters, "swish_beta", default=1.0
                ),
                "ELU_ALPHA": _require_number(
                    parameters, "elu_alpha", default=1.0
                ),
                "SOFTPLUS_BETA": _require_number(
                    parameters, "softplus_beta", default=1.0
                ),
            }
        )
        if constants["SOFTPLUS_BETA"] <= 0 or (
            constants["HAS_UPPER_CLIP"]
            and constants["UPPER_CLIP"] < constants["LOWER_CLIP"]
        ):
            raise ValueError("invalid activation gradient attributes")
        if dense:
            constants["MASK_TAIL"] = True
    signature = {
        "x_ptr": POINTER_TYPES[left["data_type"]],
        "y_ptr": POINTER_TYPES[right["data_type"]],
        "out_ptr": POINTER_TYPES[output["data_type"]],
        "n_elements": "i32",
    }
    return (
        function_name,
        signature,
        constants,
        ((elements + block_size - 1) // block_size, 1, 1),
    )


def _schema_unary_configuration(
    operation: str,
    schema: PointwiseSchema,
    compute_data_type: str,
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | float],
    tuple[int, int, int],
]:
    input_tensor, output = tensors
    _validate_pointwise_data_types(
        schema, operation, compute_data_type, tensors
    )
    elements = _pointwise_elements(parameters, output, schema.family)
    strided_constants = _unary_stride_constants(input_tensor, output)
    negative_slope = _require_number(parameters, "negative_slope", default=0.0)
    lower_clip = _require_number(parameters, "lower_clip", default=0.0)
    upper_clip = _require_number(parameters, "upper_clip", default=0.0)
    has_upper_clip = _require_integer(
        parameters, "has_upper_clip", minimum=0, maximum=1
    )
    swish_beta = _require_number(parameters, "swish_beta", default=1.0)
    elu_alpha = _require_number(parameters, "elu_alpha", default=1.0)
    softplus_beta = _require_number(parameters, "softplus_beta", default=1.0)
    if operation == "softplus" and softplus_beta <= 0.0:
        raise ValueError("softplus beta must be positive")

    block_size = 256
    constants: dict[str, int | float] = {
        "OPERATION": schema.mode,
        "negative_slope": negative_slope,
        "lower_clip": lower_clip,
        "upper_clip": upper_clip,
        "HAS_UPPER_CLIP": has_upper_clip,
        "SWISH_BETA": swish_beta,
        "ELU_ALPHA": elu_alpha,
        "SOFTPLUS_BETA": softplus_beta,
        "TILES_PER_PROGRAM": 1,
        "BLOCK_SIZE": block_size,
    }
    function_name = "unary_pointwise_contiguous_kernel"
    packed_factor = 1
    dense = _can_use_dense_pointwise_kernel(tensors)
    if not dense:
        function_name = "unary_pointwise_strided_kernel"
        constants.update(strided_constants)
        constants["STRIDED"] = 1
    elif operation == "identity":
        packed_factor = {
            "float16": 4,
            "bfloat16": 4,
            "float32": 2,
        }.get(input_tensor["data_type"], 1)
        if (
            packed_factor > 1
            and elements >= 4096
            and elements % packed_factor == 0
            and all(tensor["alignment"] >= 8 for tensor in tensors)
        ):
            function_name = "identity_packed_contiguous_kernel"
            constants["PACK_FACTOR"] = packed_factor
        else:
            packed_factor = 1
    signature = {
        "in_ptr": POINTER_TYPES[input_tensor["data_type"]],
        "out_ptr": POINTER_TYPES[output["data_type"]],
        "n_elements": "i32",
    }
    return (
        function_name,
        signature,
        constants,
        (
            (elements + block_size * packed_factor - 1)
            // (block_size * packed_factor),
            1,
            1,
        ),
    )


def _schema_ternary_configuration(
    operation: str,
    schema: PointwiseSchema,
    compute_data_type: str,
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | float],
    tuple[int, int, int],
]:
    left, right, predicate, output = tensors
    _validate_pointwise_data_types(
        schema, operation, compute_data_type, tensors
    )
    elements = _pointwise_elements(parameters, output, schema.family)
    strided_constants = _broadcast_stride_constants(
        [left, right, predicate],
        output,
        ("LEFT_STRIDE", "RIGHT_STRIDE", "MASK_STRIDE"),
        schema.family,
    )
    block_size = 256
    constants: dict[str, int | float] = {"BLOCK_SIZE": block_size}
    if _can_use_dense_pointwise_kernel(tensors):
        function_name = "binary_select_tensor_kernel"
        signature = {
            "input0_ptr": POINTER_TYPES[left["data_type"]],
            "input1_ptr": POINTER_TYPES[right["data_type"]],
            "mask_ptr": POINTER_TYPES[predicate["data_type"]],
            "out_ptr": POINTER_TYPES[output["data_type"]],
            "n_elements": "i32",
        }
    else:
        function_name = "binary_select_strided_kernel"
        constants.update(strided_constants)
        signature = {
            "x_ptr": POINTER_TYPES[left["data_type"]],
            "y_ptr": POINTER_TYPES[right["data_type"]],
            "t_ptr": POINTER_TYPES[predicate["data_type"]],
            "out_ptr": POINTER_TYPES[output["data_type"]],
            "n_elements": "i32",
        }
    return (
        function_name,
        signature,
        constants,
        ((elements + block_size - 1) // block_size, 1, 1),
    )


def _pointwise_configuration(
    node: dict[str, Any],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | float],
    tuple[int, int, int],
]:
    schema = node["schema"]
    configure = {
        "binary": _schema_binary_configuration,
        "unary": _schema_unary_configuration,
        "ternary": _schema_ternary_configuration,
    }.get(schema.family)
    if configure is None:
        raise ValueError(f"unknown pointwise family {schema.family!r}")
    return configure(
        node["operation"],
        schema,
        node["compute_data_type"],
        node["parameters"],
        node["tensors"],
    )
