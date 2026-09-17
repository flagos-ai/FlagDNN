# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Iluvatar dispatch / pointwise implementation."""

from __future__ import annotations

from .common import FLOAT_TYPES
from .common import POINTER_TYPES
from .common import _require_integer
from .common import _require_list
from .common import _require_number
from .common import _require_object
from .graph_tensor import _is_physically_dense
from .graph_tensor import _parse_port
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
    "binary_select": 41,
    "relu_backward": 42,
    "tanh_backward": 43,
    "elu_backward": 44,
    "gelu_backward": 45,
    "softplus_backward": 46,
    "swish_backward": 47,
    "gelu_approx_tanh_backward": 48,
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
            f"Iluvatar compiler does not support graph operation {operation!r}"
        )
    schema = POINTWISE_SCHEMAS[operation]
    inputs = _require_list(node.get("inputs"), "node.inputs")
    outputs = _require_list(node.get("outputs"), "node.outputs")
    if len(inputs) != len(schema.input_ports) or len(outputs) != len(
        schema.output_ports
    ):
        raise ValueError(f"{schema.family} pointwise node port count is invalid")

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
    if not isinstance(compute_data_type, str) or compute_data_type not in POINTER_TYPES:
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
                raise ValueError(f"{family} pointwise inputs are not broadcastable")
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
    policy = schema.data_type_policy
    if policy == "binary_numeric":
        valid = len(set(data_types)) == 1 and data_types[0] in (FLOAT_TYPES | {"int32"})
        expected_compute = {"float32", "int32"}
    elif policy == "binary_comparison":
        valid = (
            data_types[0] == data_types[1]
            and data_types[0] in (FLOAT_TYPES | {"int32"})
            and data_types[2] == "boolean"
        )
        expected_compute = {"boolean"}
    elif policy == "binary_boolean":
        valid = data_types == ["boolean", "boolean", "boolean"]
        expected_compute = {"boolean"}
    elif policy == "unary_numeric":
        allowed = set(POINTER_TYPES) if operation == "identity" else FLOAT_TYPES
        valid = data_types[0] == data_types[1] and data_types[0] in allowed
        expected_compute = {"float32", "int32"}
    elif policy == "unary_boolean":
        valid = data_types == ["boolean", "boolean"]
        expected_compute = {"boolean"}
    elif policy == "ternary_select":
        valid = (
            data_types[0] == data_types[1] == data_types[3]
            and data_types[0] in (FLOAT_TYPES | {"int32"})
            and data_types[2] == "boolean"
        )
        expected_compute = {"float32", "int32"}
    else:
        raise ValueError(f"unknown pointwise data-type policy {policy!r}")
    if not valid:
        raise ValueError(f"{operation} tensor data types violate {policy} policy")
    if compute_data_type not in expected_compute:
        raise ValueError(
            f"{operation} compute_data_type {compute_data_type!r} is not "
            f"implemented by the Iluvatar {policy} kernel; expected one of "
            f"{sorted(expected_compute)!r}"
        )


def _pointwise_elements(
    parameters: dict[str, Any], output: dict[str, Any], family: str
) -> int:
    elements = _require_integer(parameters, "n_elements")
    if elements != math.prod(output["dimensions"]):
        raise ValueError(f"parameters.n_elements is inconsistent with {family} output")
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
    _validate_pointwise_data_types(schema, operation, compute_data_type, tensors)
    elements = _pointwise_elements(parameters, output, schema.family)
    pointwise_mode_value = parameters.get("pointwise_mode", schema.mode)
    pointwise_mode = _require_integer(
        {"pointwise_mode": pointwise_mode_value},
        "pointwise_mode",
        minimum=1,
        maximum=48,
    )
    if pointwise_mode != schema.mode:
        raise ValueError("parameters.pointwise_mode is inconsistent with node type")
    alpha = _require_number(parameters, "alpha", default=1.0)
    if left["data_type"] == "int32":
        if alpha != int(alpha) or not -(2**31) <= alpha < 2**31:
            raise ValueError("INT32 alpha must be a signed integer")
        alpha = int(alpha)
    if not schema.allow_alpha and alpha != 1.0:
        raise ValueError("pointwise alpha is supported only by add and sub")
    if operation == "sigmoid_backward" and any(
        tensor["dimensions"] != output["dimensions"] for tensor in tensors
    ):
        raise ValueError("sigmoid backward tensors must have equal shapes")

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
    function_name = "binary_contiguous_kernel"
    if not _can_use_dense_pointwise_kernel(tensors):
        function_name = "binary_strided_kernel"
        constants.update(strided_constants)
    if operation.endswith("_backward"):
        if any(t["dimensions"] != output["dimensions"] for t in tensors):
            raise ValueError("activation gradient tensor shapes must match")
        if left["data_type"] not in FLOAT_TYPES:
            raise ValueError("activation gradients require floating tensors")
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
                    _require_integer(parameters, "has_upper_clip", minimum=0, maximum=1)
                ),
                "SWISH_BETA": _require_number(parameters, "swish_beta", default=1.0),
                "ELU_ALPHA": _require_number(parameters, "elu_alpha", default=1.0),
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
        function_name = function_name.replace("binary_", "activation_backward_", 1)
        if function_name == "activation_backward_contiguous_kernel":
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
    _validate_pointwise_data_types(schema, operation, compute_data_type, tensors)
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
    if not _can_use_dense_pointwise_kernel(tensors):
        function_name = "unary_pointwise_strided_kernel"
        constants.update(strided_constants)
        constants["STRIDED"] = 1
    signature = {
        "in_ptr": POINTER_TYPES[input_tensor["data_type"]],
        "out_ptr": POINTER_TYPES[output["data_type"]],
        "n_elements": "i32",
    }
    if operation == "identity":
        pointer = {
            "float32": "*i32",
            "int32": "*i32",
            "float16": "*u16",
            "bfloat16": "*u16",
            "boolean": "*u8",
            "fp8_e4m3": "*u8",
            "fp8_e5m2": "*u8",
            "fp8_e8m0": "*u8",
        }[input_tensor["data_type"]]
        signature["in_ptr"] = signature["out_ptr"] = pointer
    return (
        function_name,
        signature,
        constants,
        (
            (elements + block_size * packed_factor - 1) // (block_size * packed_factor),
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
    _validate_pointwise_data_types(schema, operation, compute_data_type, tensors)
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
