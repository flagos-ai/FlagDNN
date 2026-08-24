# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Iluvatar/DTK Triton compiler provider for FlagDNN Graph IR.

Pointwise planning is schema driven: operation arity, port names, semantic
mode, data-type policy and kernel family are declared once below.  The planner
then applies the same Graph IR, workspace, libtriton_jit and autotune contracts
to every common unary, binary and ternary pointwise operation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import triton
import yaml  # type: ignore[import-untyped]

from . import compiler_nn
from . import compiler_tensor
from .compiler_identity import (
    ARTIFACT_SCHEMA_VERSION,
    EXECUTION_PROGRAM_VERSION,
    GRAPH_IR_SCHEMA_VERSION,
    PROVIDER_NAME,
    PROVIDER_VERSION,
    build_compiler_identity,
    compiler_identity_dependencies as _identity_dependencies,
)

from flagdnn_codegen.kernel_registry import (
    materialize_kernel_source,
    resolve_kernel_source,
    resolve_tuning_source,
    select_kernel_candidate,
)

SCHEMA_VERSION = GRAPH_IR_SCHEMA_VERSION
LIBTRITON_JIT_GLOBAL_SCRATCH_SIZE = 0
ILUVATAR_WARP_SIZE = 64
MAX_I32 = 2**31 - 1
# corex_71 exposes 64 KiB of LDS. A generic WGrad tl.dot uses the two
# operand tiles modeled below.
ILUVATAR_MAX_SHARED_MEMORY_BYTES = 64 * 1024

_WGRAD_DOT_RIGHT_META = {"conv_wgrad_nd_kernel": "BLOCK_CI"}
_WGRAD_DOT_ELEMENT_BYTES = {
    "*fp16": 2,
    "*bf16": 2,
    "*fp32": 4,
}


def _triton_supports_pointer_range(version: str) -> bool:
    match = re.match(r"^(\d+)\.(\d+)(?:\.|$)", version)
    if match is None:
        raise RuntimeError(f"cannot parse Triton version {version!r}")
    return (int(match.group(1)), int(match.group(2))) >= (3, 3)


TRITON_SUPPORTS_POINTER_RANGE = _triton_supports_pointer_range(
    triton.__version__
)

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
SUPPORTED_OPERATIONS = (
    tuple(POINTWISE_SCHEMAS)
    + tuple(sorted(compiler_tensor.SUPPORTED_OPERATIONS))
    + tuple(sorted(compiler_nn.SUPPORTED_OPERATIONS))
)
POINTER_TYPES = {
    "float32": "*fp32",
    "float16": "*fp16",
    "bfloat16": "*bf16",
    "boolean": "*i8",
    "fp8_e4m3": "*fp8e4nv",
    "fp8_e5m2": "*fp8e5",
}
FLOAT_TYPES = {"float32", "float16", "bfloat16"}
SUPPORTED_TARGET = "corex_71"
FLOAT32_MAX = 3.4028234663852886e38


def _require_object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _require_integer(
    values: dict[str, Any],
    name: str,
    *,
    minimum: int = 1,
    maximum: int = 2**31 - 1,
) -> int:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"parameters.{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(
            f"parameters.{name} must be in [{minimum}, {maximum}]"
        )
    return value


def _require_number(
    values: dict[str, Any], name: str, *, default: float | None = None
) -> float:
    value = values.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"parameters.{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or abs(result) > FLOAT32_MAX:
        raise ValueError(
            f"parameters.{name} must be finite and representable as float32"
        )
    return result


def _compiler_entry_path() -> Path:
    import flagdnn_codegen

    return Path(flagdnn_codegen.__file__).resolve().with_name("main.py")


def compiler_identity(
    target_name: str, execution_engine: str = "libtriton_jit"
) -> dict[str, Any]:
    return build_compiler_identity(target_name, execution_engine)


def compiler_identity_dependencies(
    target_name: str, execution_engine: str = "libtriton_jit"
) -> tuple[Path, ...]:
    return _identity_dependencies(target_name, execution_engine)


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _load_generated_module(path: Path, stage_id: int) -> Any:
    module_name = f"_flagdnn_iluvatar_generated_{os.getpid()}_{stage_id}"
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import generated kernel module: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _has_non_overlapping_strides(
    dimensions: list[int], strides: list[int]
) -> bool:
    axes = sorted(
        (stride, dimension)
        for dimension, stride in zip(dimensions, strides)
        if dimension > 1
    )
    required_span = 1
    for stride, dimension in axes:
        if stride < required_span:
            return False
        required_span += (dimension - 1) * stride
    return True


def _storage_elements(tensor: dict[str, Any]) -> int:
    return 1 + sum(
        (dimension - 1) * stride
        for dimension, stride in zip(
            tensor["dimensions"], tensor["strides"], strict=True
        )
    )


def _is_physically_dense(tensor: dict[str, Any]) -> bool:
    return _has_non_overlapping_strides(
        tensor["dimensions"], tensor["strides"]
    ) and _storage_elements(tensor) == math.prod(tensor["dimensions"])


def _tensor_storage_size(tensor: dict[str, Any]) -> int:
    element_size = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "boolean": 1,
        "fp8_e4m3": 1,
        "fp8_e5m2": 1,
    }[tensor["data_type"]]
    size = _storage_elements(tensor) * element_size
    if size <= 0 or size > 2**63 - 1:
        raise ValueError(f"tensor {tensor['uid']} storage size is invalid")
    return size


def _parse_tensor_table(graph: dict[str, Any]) -> dict[int, dict[str, Any]]:
    tensors = _require_list(graph.get("tensors"), "graph.tensors")
    tensor_count = graph.get("tensor_count")
    if (
        isinstance(tensor_count, bool)
        or not isinstance(tensor_count, int)
        or tensor_count != len(tensors)
        or tensor_count < 1
    ):
        raise ValueError("graph tensor_count is invalid")

    result: dict[int, dict[str, Any]] = {}
    for index, value in enumerate(tensors):
        tensor = _require_object(value, f"graph.tensors[{index}]")
        uid = tensor.get("uid")
        if isinstance(uid, bool) or not isinstance(uid, int) or uid <= 0:
            raise ValueError(f"tensor UID {index} is invalid")
        if uid in result:
            raise ValueError("graph tensor UIDs must be unique")
        data_type = tensor.get("data_type")
        if not isinstance(data_type, str) or data_type not in POINTER_TYPES:
            raise ValueError(
                f"tensor {uid} has an unsupported data type: {data_type!r}"
            )
        is_virtual = tensor.get("virtual")
        if not isinstance(is_virtual, bool):
            raise ValueError(f"tensor {uid} virtual flag must be boolean")
        alignment = tensor.get("alignment", 16)
        if (
            isinstance(alignment, bool)
            or not isinstance(alignment, int)
            or alignment <= 0
            or alignment > 2**31
            or alignment & (alignment - 1) != 0
        ):
            raise ValueError(
                f"tensor {uid} alignment must be a positive power of two"
            )
        dimensions = _require_list(
            tensor.get("dimensions"), f"tensor {uid} dimensions"
        )
        strides = _require_list(tensor.get("strides"), f"tensor {uid} strides")
        if len(dimensions) > 8 or len(dimensions) != len(strides):
            raise ValueError(f"tensor {uid} rank is invalid")
        if any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item <= 0
            or item > 2**31 - 1
            for item in dimensions + strides
        ):
            raise ValueError(f"tensor {uid} shape or strides are invalid")
        parsed = {
            "uid": uid,
            "virtual": is_virtual,
            "alignment": alignment,
            "data_type": data_type,
            "dimensions": list(dimensions),
            "strides": list(strides),
        }
        if not _has_non_overlapping_strides(dimensions, strides):
            raise ValueError(f"tensor {uid} strides overlap")
        _tensor_storage_size(parsed)
        result[uid] = parsed
    return result


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
            "Iluvatar pointwise operations require every tensor port"
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
            f"Iluvatar compiler does not support graph operation {operation!r}"
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


def _parse_node(
    node_value: object,
    position: int,
    node_count: int,
    tensor_registry: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    node = _require_object(node_value, f"graph.nodes[{position}]")
    operation = node.get("type")
    if operation in POINTWISE_SCHEMAS:
        return _parse_pointwise_node(
            node_value, position, node_count, tensor_registry
        )
    if operation in compiler_tensor.SUPPORTED_OPERATIONS:
        return compiler_tensor.parse_node(
            node_value, position, node_count, tensor_registry
        )
    if operation in compiler_nn.SUPPORTED_OPERATIONS:
        return compiler_nn.parse_node(
            node_value, position, node_count, tensor_registry
        )
    raise ValueError(
        f"Iluvatar compiler does not support graph operation {operation!r}"
    )


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
    policy = schema.data_type_policy
    if policy == "binary_numeric":
        valid = len(set(data_types)) == 1 and data_types[0] in FLOAT_TYPES
        expected_compute = {"float32"}
    elif policy == "binary_comparison":
        valid = (
            data_types[0] == data_types[1]
            and data_types[0] in FLOAT_TYPES
            and data_types[2] == "boolean"
        )
        expected_compute = {"boolean"}
    elif policy == "binary_boolean":
        valid = data_types == ["boolean", "boolean", "boolean"]
        expected_compute = {"boolean"}
    elif policy == "unary_numeric":
        valid = data_types[0] == data_types[1] and data_types[0] in FLOAT_TYPES
        expected_compute = {"float32"}
    elif policy == "unary_boolean":
        valid = data_types == ["boolean", "boolean"]
        expected_compute = {"boolean"}
    elif policy == "ternary_select":
        valid = (
            data_types[0] == data_types[1] == data_types[3]
            and data_types[0] in FLOAT_TYPES
            and data_types[2] == "boolean"
        )
        expected_compute = {"float32"}
    else:
        raise ValueError(f"unknown pointwise data-type policy {policy!r}")
    if not valid:
        raise ValueError(
            f"{operation} tensor data types violate {policy} policy"
        )
    if compute_data_type not in expected_compute:
        raise ValueError(
            f"{operation} compute_data_type {compute_data_type!r} is not "
            f"implemented by the Iluvatar {policy} kernel; expected one of "
            f"{sorted(expected_compute)!r}"
        )


def _specialize_pointwise_compute_source(
    generated_bytes: bytes, node: dict[str, Any]
) -> bytes:
    """Make the Graph compute type an explicit part of generated code.

    Triton's ordinary FP16/BF16 add, subtract and multiply expressions retain
    their input precision. FlagDNN's public pointwise graphs request FP32
    compute for numeric operations, so relying on implicit Triton promotion
    silently violates the Graph contract. Keep the common source
    platform-neutral and specialize only the per-artifact materialized copy.

    Boolean pointwise operations already execute in their declared boolean
    semantics. They still receive a deterministic source marker so the
    materialized source and candidate identities encode compute_data_type.
    """

    compute_data_type = node["compute_data_type"]
    schema: PointwiseSchema = node["schema"]
    try:
        source = generated_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("pointwise kernel source must be UTF-8") from error

    marker = (
        "# FlagDNN Iluvatar compute specialization: "
        f"{schema.data_type_policy}/{compute_data_type}\n"
    )
    if source.startswith(marker):
        raise ValueError("pointwise kernel source is already specialized")

    if compute_data_type == "float32" and node["operation"] == "identity":
        # Identity performs no arithmetic.  Every supported input value is
        # already representable in FP32, and a direct copy additionally keeps
        # signed zero and NaN payload bits intact instead of round-tripping
        # through a conversion solely to annotate the compute type.
        pass
    elif compute_data_type == "float32":
        if schema.family == "binary":
            trigger_names = {"right"}
            cast_names = ("left", "right")
        elif schema.family == "unary":
            trigger_names = {"value"}
            cast_names = ("value",)
        elif schema.family == "ternary":
            trigger_names = {"right", "input1"}
            cast_names = ()
        else:
            raise ValueError(f"unknown pointwise family {schema.family!r}")

        lines: list[str] = []
        specialization_count = 0
        for line in source.splitlines(keepends=True):
            lines.append(line)
            stripped = line.lstrip()
            indentation = line[: len(line) - len(stripped)]
            loaded_name = next(
                (
                    name
                    for name in trigger_names
                    if stripped.startswith(f"{name} = tl.load(")
                ),
                None,
            )
            if loaded_name is None:
                continue
            names = cast_names
            if schema.family == "ternary":
                names = (
                    ("left", "right")
                    if loaded_name == "right"
                    else ("input0", "input1")
                )
            for name in names:
                lines.append(f"{indentation}{name} = {name}.to(tl.float32)\n")
            specialization_count += 1
        if specialization_count < 1:
            raise ValueError(
                "common pointwise kernel no longer exposes the expected "
                "load sites for FP32 specialization"
            )
        source = "".join(lines)
    elif compute_data_type != "boolean":
        # _validate_pointwise_data_types normally rejects this first. Keep a
        # defensive check here because this function is the precision
        # boundary for all future schema additions.
        raise ValueError(
            "Iluvatar pointwise materialization implements only float32 or "
            "boolean compute semantics"
        )

    return (marker + source).encode("utf-8")


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
    pointwise_mode_value = parameters.get("pointwise_mode", schema.mode)
    pointwise_mode = _require_integer(
        {"pointwise_mode": pointwise_mode_value},
        "pointwise_mode",
        minimum=1,
        maximum=41,
    )
    if pointwise_mode != schema.mode:
        raise ValueError(
            "parameters.pointwise_mode is inconsistent with node type"
        )
    alpha = _require_number(parameters, "alpha", default=1.0)
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
    if not _can_use_dense_pointwise_kernel(tensors):
        function_name = "unary_pointwise_strided_kernel"
        constants.update(strided_constants)
        constants["STRIDED"] = 1
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


def _fused_conv_bias_relu_node(
    convolution: dict[str, Any],
    bias_add: dict[str, Any],
    relu: dict[str, Any],
    tensor_registry: dict[int, dict[str, Any]],
    consumer_counts: dict[int, int],
) -> dict[str, Any] | None:
    if (
        convolution["operation"] not in ("conv2d_fprop", "convolution_fprop")
        or bias_add["operation"] != "add"
        or relu["operation"] != "relu"
        or int(convolution["derived"]["spatial_rank"]) != 2
    ):
        return None

    convolution_uid = convolution["output_uids"][0]
    biased_uid = bias_add["output_uids"][0]
    if (
        consumer_counts.get(convolution_uid) != 1
        or consumer_counts.get(biased_uid) != 1
        or not tensor_registry[convolution_uid]["virtual"]
        or not tensor_registry[biased_uid]["virtual"]
        or bias_add["input_uids"].count(convolution_uid) != 1
        or relu["input_uids"] != [biased_uid]
    ):
        return None

    bias_uid = next(
        uid for uid in bias_add["input_uids"] if uid != convolution_uid
    )
    output_uid = relu["output_uids"][0]
    bias = tensor_registry[bias_uid]
    output = tensor_registry[output_uid]
    convolution_output = tensor_registry[convolution_uid]
    biased_output = tensor_registry[biased_uid]
    out_channels = int(convolution["derived"]["out_channels"])
    if (
        bias["dimensions"] != [1, out_channels, 1, 1]
        or bias["data_type"] != convolution["derived"]["data_type"]
        or convolution_output["dimensions"] != output["dimensions"]
        or biased_output["dimensions"] != output["dimensions"]
        or convolution_output["data_type"] != output["data_type"]
        or biased_output["data_type"] != output["data_type"]
    ):
        return None

    _, _, add_constants, _ = _pointwise_configuration(bias_add)
    _, _, relu_constants, _ = _pointwise_configuration(relu)
    if (
        add_constants["ALPHA"] != 1.0
        or relu_constants["negative_slope"] != 0.0
        or relu_constants["lower_clip"] != 0.0
        or relu_constants["HAS_UPPER_CLIP"] != 0
    ):
        return None

    fused = dict(convolution)
    port_tensors = dict(convolution["port_tensors"])
    port_tensors["bias"] = bias
    port_tensors["output"] = output
    derived = dict(convolution["derived"])
    derived["result"] = output
    fused.update(
        {
            "fused_bias_relu": True,
            "source_node_ids": [
                convolution["id"],
                bias_add["id"],
                relu["id"],
            ],
            "tensors": [
                port_tensors["input"],
                port_tensors["filter"],
                bias,
                output,
            ],
            "tensor_roles": ["input", "filter", "bias", "output"],
            "port_tensors": port_tensors,
            "port_indices": {
                "input": 0,
                "filter": 1,
                "bias": 2,
                "output": 3,
            },
            "input_uids": [
                convolution["input_uids"][0],
                convolution["input_uids"][1],
                bias_uid,
            ],
            "output_uids": [output_uid],
            "derived": derived,
        }
    )
    return fused


def _fuse_conv_bias_relu(
    nodes: list[dict[str, Any]],
    tensor_registry: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    consumer_counts: dict[int, int] = {}
    for node in nodes:
        for uid in node["input_uids"]:
            consumer_counts[uid] = consumer_counts.get(uid, 0) + 1

    result: list[dict[str, Any]] = []
    index = 0
    while index < len(nodes):
        fused = None
        if index + 2 < len(nodes):
            fused = _fused_conv_bias_relu_node(
                nodes[index],
                nodes[index + 1],
                nodes[index + 2],
                tensor_registry,
                consumer_counts,
            )
        if fused is None:
            result.append(nodes[index])
            index += 1
        else:
            result.append(fused)
            index += 3
    return result


def _workspace_layout(
    tensors: dict[int, dict[str, Any]],
) -> tuple[dict[int, tuple[int, int, int]], int]:
    minimum_alignment = 256
    offset = 0
    result: dict[int, tuple[int, int, int]] = {}
    for uid, tensor in tensors.items():
        if not tensor["virtual"]:
            continue
        alignment = max(minimum_alignment, tensor["alignment"])
        offset = (offset + alignment - 1) // alignment * alignment
        size = _tensor_storage_size(tensor)
        result[uid] = (offset, size, alignment)
        offset += size
    if offset:
        offset = (
            (offset + minimum_alignment - 1)
            // minimum_alignment
            * minimum_alignment
        )
    return result, offset


def _nn_workspace_layout(
    nodes: list[dict[str, Any]],
    initial_offset: int,
    maximum_tensor_uid: int,
) -> tuple[
    dict[int, compiler_nn.NodePlan],
    dict[int, dict[str, dict[str, int]]],
    int,
]:
    alignment = compiler_nn.WORKSPACE_ALIGNMENT
    offset = initial_offset
    next_uid = maximum_tensor_uid + 1
    plans: dict[int, compiler_nn.NodePlan] = {}
    workspaces: dict[int, dict[str, dict[str, int]]] = {}
    for node in nodes:
        if node["operation"] not in compiler_nn.SUPPORTED_OPERATIONS:
            continue
        plan = compiler_nn.plan_kernel_stages(node)
        plans[node["id"]] = plan
        node_alignment = alignment
        for tensor in plan.workspace_tensors:
            if (
                tensor.alignment < alignment
                or tensor.alignment & (tensor.alignment - 1) != 0
            ):
                raise ValueError(
                    "Iluvatar NN workspace tensor alignment is invalid"
                )
            node_alignment = max(node_alignment, tensor.alignment)
        if offset > 2**63 - 1 - (node_alignment - 1):
            raise ValueError("Iluvatar NN workspace alignment overflows int64")
        offset = (
            (offset + node_alignment - 1) // node_alignment * node_alignment
        )
        node_base = offset
        named: dict[str, dict[str, int]] = {}
        for tensor in plan.workspace_tensors:
            if tensor.name in named:
                raise ValueError("duplicate Iluvatar NN workspace tensor name")
            if (
                tensor.offset % tensor.alignment != 0
                or tensor.size <= 0
                or tensor.offset > plan.workspace_size
                or tensor.size > plan.workspace_size - tensor.offset
            ):
                raise ValueError(
                    "Iluvatar NN workspace tensor layout is invalid"
                )
            if next_uid > 2**63 - 1:
                raise ValueError(
                    "Iluvatar NN workspace tensor UID overflows int64"
                )
            named[tensor.name] = {
                "kind": "workspace_tensor",
                "uid": next_uid,
                "offset": node_base + tensor.offset,
                "size": tensor.size,
                "alignment": tensor.alignment,
            }
            next_uid += 1
        workspaces[node["id"]] = named
        if plan.workspace_size < 0 or plan.workspace_size > 2**63 - 1 - offset:
            raise ValueError("Iluvatar NN workspace size overflows int64")
        offset += plan.workspace_size
    if offset:
        offset = (offset + alignment - 1) // alignment * alignment
    return plans, workspaces, offset


def _tensor_argument(
    tensor: dict[str, Any], workspace: dict[int, tuple[int, int, int]]
) -> dict[str, Any]:
    uid = tensor["uid"]
    if tensor["virtual"]:
        offset, size, _layout_alignment = workspace[uid]
        return {
            "kind": "workspace_tensor",
            "uid": uid,
            "offset": offset,
            "size": size,
            "alignment": tensor["alignment"],
        }
    return {
        "kind": "tensor",
        "uid": uid,
        "size": _tensor_storage_size(tensor),
        "alignment": tensor["alignment"],
    }


def _argument_abi(
    layout: tuple[tuple[str, str | int | None], ...],
    tensors: list[dict[str, Any]],
    tensor_roles: list[str],
    parameters: dict[str, Any],
    workspace: dict[int, tuple[int, int, int]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    tensor_index = 0
    consumed: set[int] = set()
    for kind, name in layout:
        if kind == "tensor":
            if tensor_index >= len(tensors):
                raise ValueError("kernel ABI requests too many tensors")
            argument = _tensor_argument(tensors[tensor_index], workspace)
            argument["role"] = tensor_roles[tensor_index]
            result.append(argument)
            consumed.add(tensor_index)
            tensor_index += 1
            continue
        if kind == "tensor_alias" and isinstance(name, int):
            if name < -len(tensors) or name >= len(tensors):
                raise ValueError("kernel ABI tensor alias is out of range")
            alias_index = name if name >= 0 else len(tensors) + name
            argument = _tensor_argument(tensors[alias_index], workspace)
            argument["role"] = tensor_roles[alias_index]
            result.append(argument)
            consumed.add(alias_index)
            continue
        if kind == "scalar_i32" and isinstance(name, str):
            value = parameters.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < -(2**31)
                or value > 2**31 - 1
            ):
                raise ValueError(
                    f"parameters.{name} must be representable as int32"
                )
            result.append({"kind": kind, "name": name, "value": value})
            continue
        if kind == "scalar_f32" and isinstance(name, str):
            result.append(
                {
                    "kind": kind,
                    "name": name,
                    "value": _require_number(parameters, name),
                }
            )
            continue
        raise ValueError("kernel ABI layout is invalid")
    if consumed != set(range(len(tensors))):
        raise ValueError("kernel ABI does not consume every tensor")
    result.extend(
        (
            {"kind": "global_scratch_pointer"},
            {"kind": "profile_scratch_pointer"},
        )
    )
    return result


def _nn_argument_abi(
    configuration: compiler_nn.KernelStagePlan,
    tensors: list[dict[str, Any]],
    tensor_roles: list[str],
    workspace: dict[int, tuple[int, int, int]],
    local_workspace: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for kind, payload in configuration.argument_layout:
        if kind == "tensor" and isinstance(payload, int):
            if payload < 0 or payload >= len(tensors):
                raise ValueError("Iluvatar NN kernel tensor index is invalid")
            argument = _tensor_argument(tensors[payload], workspace)
            argument["role"] = tensor_roles[payload]
            result.append(argument)
            continue
        if kind == "workspace_tensor" and isinstance(payload, str):
            try:
                argument = local_workspace[payload]
            except KeyError as error:
                raise ValueError(
                    f"Iluvatar NN workspace tensor {payload!r} is missing"
                ) from error
            argument = dict(argument)
            argument["role"] = payload
            result.append(argument)
            continue
        if kind == "scalar_i32" and isinstance(payload, str):
            value = configuration.runtime_values.get(payload)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < -(2**31)
                or value > 2**31 - 1
            ):
                raise ValueError(
                    f"Iluvatar NN runtime scalar {payload!r} is not int32"
                )
            result.append(
                {"kind": "scalar_i32", "name": payload, "value": value}
            )
            continue
        if kind == "scalar_f32" and isinstance(payload, str):
            value = configuration.runtime_values.get(payload)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or abs(float(value)) > FLOAT32_MAX
            ):
                raise ValueError(
                    f"Iluvatar NN runtime scalar {payload!r} is not float32"
                )
            result.append(
                {
                    "kind": "scalar_f32",
                    "name": payload,
                    "value": float(value),
                }
            )
            continue
        raise ValueError("Iluvatar NN kernel ABI layout is invalid")
    result.extend(
        (
            {"kind": "global_scratch_pointer"},
            {"kind": "profile_scratch_pointer"},
        )
    )
    return result


def _artifact_arguments(
    argument_abi: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Serialize only arguments bound explicitly by the FlagDNN engine."""

    result: list[dict[str, Any]] = []
    for argument in argument_abi:
        kind = argument.get("kind")
        if kind in {"global_scratch_pointer", "profile_scratch_pointer"}:
            continue
        serialized = {
            key: value
            for key, value in argument.items()
            if key not in {"role", "name"}
        }
        if kind == "workspace_tensor":
            serialized["workspace_offset"] = serialized.pop("offset")
        result.append(serialized)
    return result


def _jit_runtime_signature(
    function: Any,
    runtime_signature: dict[str, str],
    argument_abi: list[dict[str, Any]],
) -> dict[str, str]:
    if (
        len(argument_abi) < 2
        or argument_abi[-2].get("kind") != "global_scratch_pointer"
        or argument_abi[-1].get("kind") != "profile_scratch_pointer"
    ):
        raise ValueError("libtriton_jit scratch ABI is invalid")
    runtime_names = [
        name for name in function.arg_names if name in runtime_signature
    ]
    visible_arguments = argument_abi[:-2]
    if len(runtime_names) != len(visible_arguments):
        raise ValueError("kernel runtime signature and argument ABI disagree")
    result = dict(runtime_signature)
    for name, argument in zip(runtime_names, visible_arguments, strict=True):
        kind = argument.get("kind")
        if kind not in {"tensor", "workspace_tensor"}:
            continue
        token = result[name]
        if not token.startswith("*") or ":" in token:
            raise ValueError("JIT tensor argument is not a plain pointer")
        alignment = int(argument.get("alignment", 1))
        storage_size = int(argument.get("size", 0))
        specialization = "16" if alignment >= 16 else ""
        # Match IX CoreX Triton Tensor specialization exactly: buffer operations may
        # use 32-bit offsets only when the reachable storage range is proven to
        # fit in signed int32. The Iluvatar-private standalone compiler restores
        # this S marker as the tt.pointer_range=32 TTIR argument attribute.
        if TRITON_SUPPORTS_POINTER_RANGE and 0 < storage_size <= MAX_I32:
            specialization += "S"
        if specialization:
            result[name] = f"{token}:{specialization}"
    return result


def _jit_full_signature(
    function: Any,
    runtime_signature: dict[str, str],
    constants: dict[str, int | float],
) -> str:
    argument_names = list(function.arg_names)
    if set(runtime_signature).union(constants) != set(argument_names):
        raise ValueError("JIT signature does not cover every kernel argument")
    tokens: list[str] = []
    for name in argument_names:
        if name in runtime_signature:
            token = runtime_signature[name]
            if not isinstance(token, str) or not token or "," in token:
                raise ValueError("JIT runtime signature token is invalid")
            tokens.append(token)
            continue
        value = constants[name]
        if isinstance(value, bool):
            tokens.append("true" if value else "false")
        elif isinstance(value, int):
            tokens.append(str(value))
        elif isinstance(value, float) and math.isfinite(value):
            tokens.append(repr(value))
        else:
            raise ValueError("JIT constexpr must be a finite number")
    return ",".join(tokens)


def _jit_launch(grid: tuple[int, int, int], num_warps: int) -> dict[str, Any]:
    if (
        isinstance(num_warps, bool)
        or not isinstance(num_warps, int)
        or num_warps <= 0
        or (num_warps & (num_warps - 1)) != 0
        or num_warps * ILUVATAR_WARP_SIZE > 1024
    ):
        raise ValueError(
            "Iluvatar num_warps must be a positive power of two within the "
            "workgroup limit"
        )
    if any(value <= 0 or value > 2**32 - 1 for value in grid):
        raise ValueError("Iluvatar launch grid is invalid")
    return {
        "grid": list(grid),
        "block": [num_warps * ILUVATAR_WARP_SIZE, 1, 1],
        "cluster": [1, 1, 1],
        "shared_memory": 0,
        "num_ctas": 1,
        "global_scratch_size": LIBTRITON_JIT_GLOBAL_SCRATCH_SIZE,
        "profile_scratch_size": 0,
    }


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _load_pointwise_tuning(
    compiler_path: Path, candidate: Any
) -> tuple[list[dict[str, Any]], str]:
    if candidate.tuning is None:
        raise ValueError("pointwise kernel candidate has no tuning metadata")
    source = resolve_tuning_source(compiler_path, candidate)
    document = yaml.safe_load(source.read_bytes())
    if not isinstance(document, dict):
        raise ValueError("Iluvatar tuning source must be a mapping")
    entries = document.get(candidate.tuning.table)
    if not isinstance(entries, list) or not entries:
        raise ValueError("Iluvatar pointwise tuning table is missing")

    expanded: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry:
            raise ValueError("Iluvatar tuning entry must be a mapping")
        if entry.get("gen") is True:
            param_map = entry.get("param_map")
            expected_map = {
                "META": {"BLOCK_SIZE": "block_size"},
                "num_warps": "warps",
                "num_stages": "stages",
            }
            if param_map != expected_map:
                raise ValueError(
                    "Iluvatar pointwise tuning param_map is invalid"
                )
            dimensions: list[list[int]] = []
            for name in ("block_size", "warps", "stages"):
                values = entry.get(name)
                if (
                    not isinstance(values, list)
                    or not values
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value <= 0
                        for value in values
                    )
                ):
                    raise ValueError(
                        f"Iluvatar tuning parameter {name} is invalid"
                    )
                dimensions.append(values)
            for block_size, warps, stages in itertools.product(*dimensions):
                expanded.append(
                    {
                        "META": {"BLOCK_SIZE": block_size},
                        "num_warps": warps,
                        "num_stages": stages,
                    }
                )
        elif "gen" in entry:
            raise ValueError("Iluvatar tuning gen flag must be true")
        else:
            expanded.append(entry)

    expected_meta_keys = {"BLOCK_SIZE"}

    configurations: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for entry in expanded:
        meta = entry.get("META")
        if not isinstance(meta, dict) or set(meta) != expected_meta_keys:
            raise ValueError("Iluvatar pointwise tuning META is invalid")
        block_size = meta["BLOCK_SIZE"]
        num_warps = entry.get("num_warps", 4)
        num_stages = entry.get("num_stages", 1)
        if (
            isinstance(block_size, bool)
            or not isinstance(block_size, int)
            or block_size < 32
            or block_size > 65536
            or block_size & (block_size - 1) != 0
        ):
            raise ValueError("Iluvatar BLOCK_SIZE must be a power of two")
        if (
            isinstance(num_warps, bool)
            or not isinstance(num_warps, int)
            or num_warps <= 0
            or (num_warps & (num_warps - 1)) != 0
            or num_warps * ILUVATAR_WARP_SIZE > 1024
        ):
            raise ValueError(
                "Iluvatar tuning num_warps must be a positive power of two "
                "within the workgroup limit"
            )
        if (
            isinstance(num_stages, bool)
            or not isinstance(num_stages, int)
            or not 1 <= num_stages <= 32
        ):
            raise ValueError("Iluvatar tuning num_stages is invalid")
        normalized_meta = {name: int(meta[name]) for name in sorted(meta)}
        configuration = {
            "META": normalized_meta,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
        encoded = _canonical(configuration)
        if encoded not in seen:
            seen.add(encoded)
            configurations.append(configuration)
    if not 2 <= len(configurations) <= 1024:
        raise ValueError(
            "Iluvatar autotune needs between 2 and 1024 candidates"
        )
    selected_payload = {
        "schema_version": 1,
        "table": candidate.tuning.table,
        "configurations": configurations,
    }
    return (
        configurations,
        hashlib.sha256(_canonical(selected_payload)).hexdigest(),
    )


def _load_tensor_tuning(
    compiler_path: Path,
    candidate: Any,
    kernel_configuration: compiler_tensor.KernelConfiguration,
) -> tuple[list[dict[str, Any]], str]:
    tuning = candidate.tuning
    if tuning is None:
        raise ValueError("tensor kernel candidate has no tuning metadata")
    expected = kernel_configuration.tuning
    actual_contract = (
        tuning.source,
        tuning.table,
        tuning.key,
        tuning.strategy,
        tuning.warmup,
        tuning.repetitions,
    )
    expected_contract = (
        expected.source,
        expected.table,
        expected.key,
        expected.strategy,
        expected.warmup,
        expected.repetitions,
    )
    if actual_contract != expected_contract:
        raise ValueError("kernel registry and tensor tuning contract disagree")

    source = resolve_tuning_source(compiler_path, candidate)
    document = yaml.safe_load(source.read_bytes())
    if not isinstance(document, dict):
        raise ValueError("Iluvatar tuning source must be a mapping")
    entries = document.get(tuning.table)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Iluvatar tuning table {tuning.table!r} is missing")

    expected_meta = set(expected.meta_keys)
    configurations: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not entry or "gen" in entry:
            raise ValueError("Iluvatar tensor tuning entries must be explicit")
        meta = entry.get("META")
        if not isinstance(meta, dict) or set(meta) != expected_meta:
            raise ValueError("Iluvatar tensor tuning META is invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in meta.values()
        ):
            raise ValueError("Iluvatar tensor tuning META values are invalid")
        num_warps = entry.get("num_warps", 4)
        num_stages = entry.get("num_stages", 1)
        kernel_configuration.variant(
            meta, num_warps=num_warps, num_stages=num_stages
        )
        configuration = {
            "META": dict(meta),
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
        encoded = _canonical(configuration)
        if encoded not in seen:
            seen.add(encoded)
            configurations.append(configuration)
    if not 2 <= len(configurations) <= 1024:
        raise ValueError(
            "Iluvatar autotune needs between 2 and 1024 candidates"
        )
    selected_payload = {
        "schema_version": 1,
        "table": tuning.table,
        "configurations": configurations,
    }
    return (
        configurations,
        hashlib.sha256(_canonical(selected_payload)).hexdigest(),
    )


def _wgrad_dot_tuning_fits_shared_memory(
    configuration: compiler_nn.KernelStagePlan,
    meta: dict[str, Any],
) -> bool:
    right_meta = _WGRAD_DOT_RIGHT_META.get(configuration.function_name)
    if right_meta is None:
        return True
    if configuration.operation != "convolution_wgrad":
        raise ValueError("Iluvatar WGrad dot function has the wrong operation")

    token = configuration.runtime_signature.get("dy_ptr")
    element_bytes = _WGRAD_DOT_ELEMENT_BYTES.get(token)
    if element_bytes is None:
        raise ValueError(
            "Iluvatar WGrad dot stage has an invalid dy_ptr ABI token"
        )

    required_meta = ("BLOCK_M", "BLOCK_OC", right_meta)
    missing_meta = [name for name in required_meta if name not in meta]
    if missing_meta:
        raise ValueError(
            "Iluvatar WGrad dot tuning META is missing: "
            + ", ".join(missing_meta)
        )
    values = [meta[name] for name in required_meta]
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        raise ValueError("Iluvatar WGrad dot tuning META values are invalid")
    block_m, block_oc, block_right = values
    required_bytes = block_m * (block_oc + block_right) * element_bytes
    return required_bytes <= ILUVATAR_MAX_SHARED_MEMORY_BYTES


def _load_nn_tuning(
    compiler_path: Path,
    candidate: Any,
    configuration: compiler_nn.KernelStagePlan,
) -> tuple[list[dict[str, Any]], str]:
    tuning = candidate.tuning
    if tuning is None:
        raise ValueError("NN kernel candidate has no tuning metadata")
    expected = configuration.tuning
    actual_contract = (
        tuning.source,
        tuning.table,
        tuning.key,
        tuning.strategy,
        tuning.warmup,
        tuning.repetitions,
    )
    expected_contract = (
        expected.source,
        expected.table,
        expected.key,
        expected.strategy,
        expected.warmup,
        expected.repetitions,
    )
    if actual_contract != expected_contract:
        raise ValueError("kernel registry and NN tuning contract disagree")

    source = resolve_tuning_source(compiler_path, candidate)
    document = yaml.safe_load(source.read_bytes())
    if not isinstance(document, dict):
        raise ValueError("Iluvatar tuning source must be a mapping")
    entries = document.get(tuning.table)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Iluvatar tuning table {tuning.table!r} is missing")

    expected_meta = set(expected.meta_keys)
    if not expected_meta:
        raise ValueError("untunable Iluvatar NN stage requested autotuning")
    configurations: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not entry or "gen" in entry:
            raise ValueError("Iluvatar NN tuning entries must be explicit")
        raw_meta = entry.get("META")
        if not isinstance(raw_meta, dict):
            raise ValueError("Iluvatar NN tuning META is invalid")
        if not expected_meta.issubset(raw_meta):
            continue
        meta = {name: raw_meta[name] for name in expected.meta_keys}
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in meta.values()
        ):
            raise ValueError("Iluvatar NN tuning META values are invalid")
        if not _wgrad_dot_tuning_fits_shared_memory(configuration, meta):
            continue
        if (
            configuration.operation == "batchnorm"
            and configuration.function_name == "batch_norm_nchw_kernel"
            and "BLOCK_SIZE" in meta
            and meta["BLOCK_SIZE"]
            < configuration.constants.get("BLOCK_SIZE", 1)
        ):
            continue
        num_warps = entry.get("num_warps", 4)
        num_stages = entry.get("num_stages", 1)
        configuration.variant(meta, num_warps=num_warps, num_stages=num_stages)
        projected = {
            "META": meta,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
        encoded = _canonical(projected)
        if encoded not in seen:
            seen.add(encoded)
            configurations.append(projected)
    if not 2 <= len(configurations) <= 1024:
        raise ValueError(
            "Iluvatar NN autotune needs between 2 and 1024 candidates"
        )
    selected_payload = {
        "schema_version": 1,
        "table": tuning.table,
        "stage": configuration.stage_name,
        "configurations": configurations,
    }
    return (
        configurations,
        hashlib.sha256(_canonical(selected_payload)).hexdigest(),
    )


def _compile_pointwise_stage(
    *,
    stage_id: int,
    node: dict[str, Any],
    dependencies: list[int],
    workspace: dict[int, tuple[int, int, int]],
    compiler_path: Path,
    output_directory: Path,
    enable_autotune: bool,
) -> dict[str, Any]:
    operation = node["operation"]
    parameters = node["parameters"]
    tensors = node["tensors"]
    candidate = select_kernel_candidate("iluvatar", operation)
    source_relative = Path(candidate.source)
    if (
        candidate.backend != "iluvatar"
        or candidate.operation != operation
        or candidate.source_format != "module"
        or source_relative.is_absolute()
        or ".." in source_relative.parts
        or source_relative.suffix != ".py"
    ):
        raise ValueError("Iluvatar pointwise kernel candidate is invalid")
    expected_candidate_contract = {
        "common": ("common_triton", "kernels"),
        "platform": ("iluvatar_triton", "platform"),
    }.get(candidate.ownership)
    if (
        expected_candidate_contract is None
        or (
            candidate.provider,
            candidate.source_layout,
        )
        != expected_candidate_contract
    ):
        raise ValueError(
            "Iluvatar pointwise kernel ownership/provider contract is invalid"
        )
    function_name, signature, constants, default_grid = (
        _pointwise_configuration(node)
    )
    source_path = resolve_kernel_source(compiler_path, candidate)
    source_bytes = source_path.read_bytes()
    generated_bytes = materialize_kernel_source(source_path, candidate)
    generated_bytes = _specialize_pointwise_compute_source(
        generated_bytes, node
    )
    if (
        not source_bytes
        or not generated_bytes
        or len(source_bytes) > 1 << 20
        or len(generated_bytes) > 1 << 20
    ):
        raise ValueError("pointwise kernel source size is invalid")
    generated_path = output_directory / f"generated_stage_{stage_id}.py"
    _atomic_write(generated_path, generated_bytes)
    module = _load_generated_module(generated_path, stage_id)

    if function_name not in candidate.functions:
        raise RuntimeError("kernel registry and Iluvatar compiler disagree")
    function = getattr(module, function_name)
    pointwise_layout = tuple(
        [("tensor", None)] * len(tensors) + [("scalar_i32", "n_elements")]
    )
    argument_abi = _argument_abi(
        pointwise_layout, tensors, node["tensor_roles"], parameters, workspace
    )
    runtime_signature = _jit_runtime_signature(
        function, signature, argument_abi
    )
    artifact_arguments = _artifact_arguments(argument_abi)
    registry_source_hash = hashlib.sha256(source_bytes).hexdigest()
    generated_source_hash = hashlib.sha256(generated_bytes).hexdigest()
    # The materialized source includes the Graph compute precision. Use that
    # digest for execution/candidate identity; the unmodified common source
    # digest remains recorded separately for ownership/audit purposes.
    kernel_source_hash = generated_source_hash
    stage: dict[str, Any] = {
        "stage_id": stage_id,
        "kind": "kernel",
        "engine": "libtriton_jit",
        "source_node_ids": [node["id"]],
        "dependencies": dependencies,
        "operation": operation,
        "source_sha256": kernel_source_hash,
        "kernel": {
            "provider": candidate.provider,
            "ownership": candidate.ownership,
            "source": candidate.source,
            "registry_source_sha256": registry_source_hash,
            "compute_data_type": node["compute_data_type"],
            "function": function_name,
            "materialized_source": {
                "path": generated_path.name,
                "size": len(generated_bytes),
                "sha256": generated_source_hash,
            },
        },
    }

    def make_variant(
        variant_id: str,
        variant_constants: dict[str, int | float],
        num_warps: int,
        num_stages: int,
        grid: tuple[int, int, int],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "variant_id": variant_id,
            "source_sha256": kernel_source_hash,
            "full_signature": _jit_full_signature(
                function, runtime_signature, variant_constants
            ),
            "compile_options": {
                "num_warps": num_warps,
                "num_stages": num_stages,
            },
            "argument_count": len(artifact_arguments),
            "arguments": artifact_arguments,
            "launch": _jit_launch(grid, num_warps),
        }
        if config is not None:
            result["config"] = config
        return result

    if enable_autotune:
        configurations, tuning_source_hash = _load_pointwise_tuning(
            compiler_path, candidate
        )
        elements = _require_integer(parameters, "n_elements")
        variants: list[dict[str, Any]] = []
        for index, configuration in enumerate(configurations):
            variant_constants = dict(constants)
            for name, value in configuration["META"].items():
                variant_constants[name] = int(value)
            block_size = int(variant_constants["BLOCK_SIZE"])
            tiles_per_program = int(
                variant_constants.get("TILES_PER_PROGRAM", 1)
            )
            pack_factor = int(variant_constants.get("PACK_FACTOR", 1))
            elements_per_program = block_size * tiles_per_program * pack_factor
            grid = (
                (elements + elements_per_program - 1) // elements_per_program,
                1,
                1,
            )
            variants.append(
                make_variant(
                    f"config_{index}",
                    variant_constants,
                    int(configuration["num_warps"]),
                    int(configuration["num_stages"]),
                    grid,
                    configuration,
                )
            )
        base_identity = {
            "schema_version": 1,
            "backend": "iluvatar",
            "target_backend": "corex",
            "warp_size": ILUVATAR_WARP_SIZE,
            "source_sha256": tuning_source_hash,
            "kernel_ownership": candidate.ownership,
            "kernel_provider": candidate.provider,
            "operation": operation,
            "function": function_name,
            "table": candidate.tuning.table,
            "key": candidate.tuning.key,
            "key_value": elements,
            "strategy": candidate.tuning.strategy,
            "configurations": configurations,
        }
        base_candidate_identity = hashlib.sha256(
            _canonical(base_identity)
        ).hexdigest()
        rendered_identity = {
            "schema_version": 1,
            "engine": "libtriton_jit",
            "base_candidate_identity": base_candidate_identity,
            "variants": [
                {
                    "variant_id": variant["variant_id"],
                    "source_sha256": variant["source_sha256"],
                    "full_signature": variant["full_signature"],
                    "compile_options": variant["compile_options"],
                    "launch": variant["launch"],
                }
                for variant in variants
            ],
        }
        stage["variants"] = variants
        stage["tuning"] = {
            "schema_version": 1,
            "source": candidate.tuning.source,
            "source_sha256": tuning_source_hash,
            "table": candidate.tuning.table,
            "key": candidate.tuning.key,
            "strategy": candidate.tuning.strategy,
            "warmup": candidate.tuning.warmup,
            "repetitions": candidate.tuning.repetitions,
            "base_candidate_identity": base_candidate_identity,
            "candidate_identity": hashlib.sha256(
                _canonical(rendered_identity)
            ).hexdigest(),
        }
    else:
        stage["variants"] = [
            make_variant(
                "default",
                constants,
                num_warps=4,
                num_stages=1,
                grid=default_grid,
            )
        ]
    return stage


def _compile_tensor_stage(
    *,
    stage_id: int,
    node: dict[str, Any],
    dependencies: list[int],
    workspace: dict[int, tuple[int, int, int]],
    compiler_path: Path,
    output_directory: Path,
    enable_autotune: bool,
) -> dict[str, Any]:
    operation = node["operation"]
    parameters = node["parameters"]
    tensors = node["tensors"]
    configuration = compiler_tensor.kernel_configuration(node)
    candidate = select_kernel_candidate("iluvatar", operation)
    source_relative = Path(candidate.source)
    if (
        candidate.backend != "iluvatar"
        or candidate.operation != operation
        or candidate.source_format != "module"
        or source_relative.is_absolute()
        or ".." in source_relative.parts
        or source_relative.suffix != ".py"
    ):
        raise ValueError("Iluvatar tensor kernel candidate is invalid")
    expected_candidate_contract = {
        "common": ("common_triton", "kernels"),
        "platform": ("iluvatar_triton", "platform"),
    }.get(candidate.ownership)
    if (
        expected_candidate_contract is None
        or (
            candidate.provider,
            candidate.source_layout,
        )
        != expected_candidate_contract
    ):
        raise ValueError(
            "Iluvatar tensor kernel ownership/provider contract is invalid"
        )
    source_path = resolve_kernel_source(compiler_path, candidate)
    source_bytes = source_path.read_bytes()
    generated_bytes = materialize_kernel_source(source_path, candidate)
    if (
        not source_bytes
        or not generated_bytes
        or len(source_bytes) > 1 << 20
        or len(generated_bytes) > 1 << 20
    ):
        raise ValueError("Iluvatar tensor kernel source size is invalid")
    generated_path = output_directory / f"generated_stage_{stage_id}.py"
    _atomic_write(generated_path, generated_bytes)
    module = _load_generated_module(generated_path, stage_id)

    function_name = configuration.function_name
    if function_name not in candidate.functions:
        raise RuntimeError(
            "kernel registry and Iluvatar tensor planner disagree"
        )
    function = getattr(module, function_name)
    argument_abi = _argument_abi(
        configuration.argument_layout,
        tensors,
        node["tensor_roles"],
        parameters,
        workspace,
    )
    runtime_signature = _jit_runtime_signature(
        function, configuration.runtime_signature, argument_abi
    )
    artifact_arguments = _artifact_arguments(argument_abi)
    kernel_source_hash = hashlib.sha256(source_bytes).hexdigest()
    generated_source_hash = hashlib.sha256(generated_bytes).hexdigest()
    stage: dict[str, Any] = {
        "stage_id": stage_id,
        "kind": "kernel",
        "engine": "libtriton_jit",
        "source_node_ids": [node["id"]],
        "dependencies": dependencies,
        "operation": operation,
        "source_sha256": kernel_source_hash,
        "kernel": {
            "provider": candidate.provider,
            "ownership": candidate.ownership,
            "source": candidate.source,
            "function": function_name,
            "materialized_source": {
                "path": generated_path.name,
                "size": len(generated_bytes),
                "sha256": generated_source_hash,
            },
        },
    }

    def make_variant(
        variant_id: str,
        constants: dict[str, int | float | bool],
        num_warps: int,
        num_stages: int,
        grid: tuple[int, int, int],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "variant_id": variant_id,
            "source_sha256": kernel_source_hash,
            "full_signature": _jit_full_signature(
                function, runtime_signature, constants
            ),
            "compile_options": {
                "num_warps": num_warps,
                "num_stages": num_stages,
            },
            "argument_count": len(artifact_arguments),
            "arguments": artifact_arguments,
            "launch": _jit_launch(grid, num_warps),
        }
        if config is not None:
            result["config"] = config
        return result

    if enable_autotune:
        tuning = candidate.tuning
        if tuning is None:
            raise ValueError("tensor kernel candidate has no tuning metadata")
        if configuration.tuning.meta_keys == ("BLOCK_SIZE",):
            configurations, tuning_source_hash = _load_pointwise_tuning(
                compiler_path, candidate
            )
        else:
            configurations, tuning_source_hash = _load_tensor_tuning(
                compiler_path, candidate, configuration
            )
        variants: list[dict[str, Any]] = []
        for index, config in enumerate(configurations):
            constants, grid = configuration.variant(
                config["META"],
                num_warps=config["num_warps"],
                num_stages=config["num_stages"],
            )
            variants.append(
                make_variant(
                    f"config_{index}",
                    constants,
                    config["num_warps"],
                    config["num_stages"],
                    grid,
                    config,
                )
            )
        base_identity = {
            "schema_version": 1,
            "backend": "iluvatar",
            "target_backend": "corex",
            "warp_size": ILUVATAR_WARP_SIZE,
            "source_sha256": tuning_source_hash,
            "kernel_ownership": candidate.ownership,
            "kernel_provider": candidate.provider,
            "operation": operation,
            "function": function_name,
            "table": tuning.table,
            "key": tuning.key,
            "key_value": configuration.tuning_key_value,
            "strategy": tuning.strategy,
            "configurations": configurations,
        }
        base_candidate_identity = hashlib.sha256(
            _canonical(base_identity)
        ).hexdigest()
        rendered_identity = {
            "schema_version": 1,
            "engine": "libtriton_jit",
            "base_candidate_identity": base_candidate_identity,
            "variants": [
                {
                    "variant_id": variant["variant_id"],
                    "source_sha256": variant["source_sha256"],
                    "full_signature": variant["full_signature"],
                    "compile_options": variant["compile_options"],
                    "launch": variant["launch"],
                }
                for variant in variants
            ],
        }
        stage["variants"] = variants
        stage["tuning"] = {
            "schema_version": 1,
            "source": tuning.source,
            "source_sha256": tuning_source_hash,
            "table": tuning.table,
            "key": tuning.key,
            "strategy": tuning.strategy,
            "warmup": tuning.warmup,
            "repetitions": tuning.repetitions,
            "base_candidate_identity": base_candidate_identity,
            "candidate_identity": hashlib.sha256(
                _canonical(rendered_identity)
            ).hexdigest(),
        }
    else:
        stage["variants"] = [
            make_variant(
                "default",
                configuration.constants,
                configuration.default_num_warps,
                configuration.default_num_stages,
                configuration.default_grid,
            )
        ]
    return stage


def _compile_nn_stage(
    *,
    stage_id: int,
    node: dict[str, Any],
    configuration: compiler_nn.KernelStagePlan,
    dependencies: list[int],
    workspace: dict[int, tuple[int, int, int]],
    local_workspace: dict[str, dict[str, int]],
    compiler_path: Path,
    output_directory: Path,
    enable_autotune: bool,
) -> dict[str, Any]:
    operation = node["operation"]
    tensors = node["tensors"]
    candidate = select_kernel_candidate("iluvatar", operation)
    source_relative = Path(candidate.source)
    if (
        candidate.backend != "iluvatar"
        or candidate.operation != operation
        or candidate.source_format != "module"
        or source_relative.is_absolute()
        or ".." in source_relative.parts
        or source_relative.suffix != ".py"
    ):
        raise ValueError("Iluvatar NN kernel candidate is invalid")
    expected_candidate_contract = {
        "common": ("common_triton", "kernels"),
        "platform": ("iluvatar_triton", "platform"),
    }.get(candidate.ownership)
    if (
        expected_candidate_contract is None
        or (
            candidate.provider,
            candidate.source_layout,
        )
        != expected_candidate_contract
    ):
        raise ValueError(
            "Iluvatar NN kernel ownership/provider contract is invalid"
        )
    source_path = resolve_kernel_source(compiler_path, candidate)
    source_bytes = source_path.read_bytes()
    generated_bytes = materialize_kernel_source(source_path, candidate)
    if (
        not source_bytes
        or not generated_bytes
        or len(source_bytes) > 1 << 20
        or len(generated_bytes) > 1 << 20
    ):
        raise ValueError("Iluvatar NN kernel source size is invalid")
    generated_path = output_directory / f"generated_stage_{stage_id}.py"
    _atomic_write(generated_path, generated_bytes)
    module = _load_generated_module(generated_path, stage_id)

    function_name = configuration.function_name
    if function_name not in candidate.functions:
        raise RuntimeError("kernel registry and Iluvatar NN planner disagree")
    function = getattr(module, function_name)
    argument_abi = _nn_argument_abi(
        configuration,
        tensors,
        node["tensor_roles"],
        workspace,
        local_workspace,
    )
    runtime_signature = _jit_runtime_signature(
        function, configuration.runtime_signature, argument_abi
    )
    artifact_arguments = _artifact_arguments(argument_abi)
    kernel_source_hash = hashlib.sha256(source_bytes).hexdigest()
    generated_source_hash = hashlib.sha256(generated_bytes).hexdigest()
    stage: dict[str, Any] = {
        "stage_id": stage_id,
        "kind": "kernel",
        "engine": "libtriton_jit",
        "source_node_ids": list(node.get("source_node_ids", [node["id"]])),
        "dependencies": dependencies,
        "operation": operation,
        "substage": configuration.stage_name,
        "source_sha256": kernel_source_hash,
        "kernel": {
            "provider": candidate.provider,
            "ownership": candidate.ownership,
            "source": candidate.source,
            "function": function_name,
            "materialized_source": {
                "path": generated_path.name,
                "size": len(generated_bytes),
                "sha256": generated_source_hash,
            },
        },
    }

    def make_variant(
        variant_id: str,
        constants: dict[str, int | float | bool],
        num_warps: int,
        num_stages: int,
        grid: tuple[int, int, int],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "variant_id": variant_id,
            "source_sha256": kernel_source_hash,
            "full_signature": _jit_full_signature(
                function, runtime_signature, constants
            ),
            "compile_options": {
                "num_warps": num_warps,
                "num_stages": num_stages,
            },
            "argument_count": len(artifact_arguments),
            "arguments": artifact_arguments,
            "launch": _jit_launch(grid, num_warps),
        }
        if config is not None:
            result["config"] = config
        return result

    stage_autotune = (
        enable_autotune
        and bool(configuration.tuning.source)
        and bool(configuration.tuning.meta_keys)
    )
    if stage_autotune:
        tuning = configuration.tuning
        configurations, tuning_source_hash = _load_nn_tuning(
            compiler_path, candidate, configuration
        )
        variants: list[dict[str, Any]] = []
        for index, config in enumerate(configurations):
            constants, grid = configuration.variant(
                config["META"],
                num_warps=config["num_warps"],
                num_stages=config["num_stages"],
            )
            variants.append(
                make_variant(
                    f"config_{index}",
                    constants,
                    config["num_warps"],
                    config["num_stages"],
                    grid,
                    config,
                )
            )
        base_identity = {
            "schema_version": 1,
            "backend": "iluvatar",
            "target_backend": "corex",
            "warp_size": ILUVATAR_WARP_SIZE,
            "source_sha256": tuning_source_hash,
            "kernel_ownership": candidate.ownership,
            "kernel_provider": candidate.provider,
            "operation": operation,
            "substage": configuration.stage_name,
            "function": function_name,
            "table": tuning.table,
            "key": tuning.key,
            "key_value": configuration.tuning_key_value,
            "strategy": tuning.strategy,
            "configurations": configurations,
        }
        base_candidate_identity = hashlib.sha256(
            _canonical(base_identity)
        ).hexdigest()
        rendered_identity = {
            "schema_version": 1,
            "engine": "libtriton_jit",
            "base_candidate_identity": base_candidate_identity,
            "variants": [
                {
                    "variant_id": variant["variant_id"],
                    "source_sha256": variant["source_sha256"],
                    "full_signature": variant["full_signature"],
                    "compile_options": variant["compile_options"],
                    "launch": variant["launch"],
                }
                for variant in variants
            ],
        }
        stage["variants"] = variants
        stage["tuning"] = {
            "schema_version": 1,
            "source": tuning.source,
            "source_sha256": tuning_source_hash,
            "table": tuning.table,
            "key": tuning.key,
            "strategy": tuning.strategy,
            "warmup": tuning.warmup,
            "repetitions": tuning.repetitions,
            "base_candidate_identity": base_candidate_identity,
            "candidate_identity": hashlib.sha256(
                _canonical(rendered_identity)
            ).hexdigest(),
        }
    else:
        stage["variants"] = [
            make_variant(
                "default",
                configuration.constants,
                configuration.default_num_warps,
                configuration.default_num_stages,
                configuration.default_grid,
            )
        ]
    return stage


def _manifest_tensor(
    tensor: dict[str, Any],
    workspace: dict[int, tuple[int, int, int]],
) -> dict[str, Any]:
    result = {
        key: tensor[key]
        for key in (
            "uid",
            "data_type",
            "dimensions",
            "strides",
            "alignment",
            "virtual",
        )
    }
    result["storage_size"] = _tensor_storage_size(tensor)
    if tensor["virtual"]:
        result["workspace_offset"] = workspace[tensor["uid"]][0]
    return result


def compile_request(
    request_path: Path,
    output_directory: Path,
    execution_engine: str = "libtriton_jit",
) -> dict[str, Any]:
    if execution_engine != "libtriton_jit":
        raise ValueError("Iluvatar supports only the libtriton_jit engine")
    request_bytes = request_path.read_bytes()
    request = _require_object(json.loads(request_bytes), "request")
    if request.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported request schema_version")
    flagdnn_version = request.get("flagdnn_version")
    if (
        not isinstance(flagdnn_version, str)
        or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", flagdnn_version) is None
    ):
        raise ValueError("request FlagDNN version is invalid")
    if request.get("backend") != "iluvatar":
        raise ValueError("Iluvatar provider received another backend")
    target_name = request.get("target")
    if target_name != SUPPORTED_TARGET:
        raise ValueError(
            f"Iluvatar provider supports only target {SUPPORTED_TARGET!r}"
        )

    identity = compiler_identity(target_name, execution_engine)
    if request.get("compiler_identity") != identity["identity_sha256"]:
        raise ValueError("request compiler identity does not match provider")
    build_options = _require_object(
        request.get("build_options"), "build_options"
    )
    enable_autotune = build_options.get("autotune", False)
    if not isinstance(enable_autotune, bool):
        raise ValueError("build_options.autotune must be a boolean")

    graph = _require_object(request.get("graph"), "graph")
    tensor_registry = _parse_tensor_table(graph)
    nodes = _require_list(graph.get("nodes"), "graph.nodes")
    node_count = graph.get("node_count")
    if (
        isinstance(node_count, bool)
        or not isinstance(node_count, int)
        or node_count != len(nodes)
        or not 1 <= node_count <= 1024
    ):
        raise ValueError("graph node_count is invalid")

    parsed_nodes: list[dict[str, Any]] = []
    node_positions: dict[int, int] = {}
    producer_nodes: dict[int, int] = {}
    has_external_output = False
    for position, node_value in enumerate(nodes):
        node = _parse_node(node_value, position, node_count, tensor_registry)
        if node["id"] in node_positions:
            raise ValueError("graph node IDs must be unique")
        node_positions[node["id"]] = position
        for uid in node["output_uids"]:
            if uid in producer_nodes:
                raise ValueError("graph tensor has more than one producer")
            producer_nodes[uid] = node["id"]
            if not tensor_registry[uid]["virtual"]:
                has_external_output = True
        parsed_nodes.append(node)

    for node in parsed_nodes:
        position = node_positions[node["id"]]
        for uid in node["input_uids"]:
            producer = producer_nodes.get(uid)
            if tensor_registry[uid]["virtual"] and producer is None:
                raise ValueError("virtual tensor input has no producer")
            if producer is not None and node_positions[producer] >= position:
                raise ValueError(
                    "graph nodes are not in topological execution order"
                )
    if not any(not tensor["virtual"] for tensor in tensor_registry.values()):
        raise ValueError("graph has no externally bound tensors")
    if not has_external_output:
        raise ValueError("graph has no non-virtual output tensor")

    parsed_nodes = _fuse_conv_bias_relu(parsed_nodes, tensor_registry)
    workspace, graph_workspace_size = _workspace_layout(tensor_registry)
    nn_plans, nn_workspaces, packed_workspace_size = _nn_workspace_layout(
        parsed_nodes, graph_workspace_size, max(tensor_registry)
    )
    workspace_alignment = 256
    for _, _, alignment in workspace.values():
        workspace_alignment = max(workspace_alignment, alignment)
    for named_workspace in nn_workspaces.values():
        for tensor in named_workspace.values():
            workspace_alignment = max(workspace_alignment, tensor["alignment"])
    if packed_workspace_size > 2**63 - 1 - LIBTRITON_JIT_GLOBAL_SCRATCH_SIZE:
        raise ValueError("Iluvatar execution workspace size overflows int64")
    workspace_size = packed_workspace_size + LIBTRITON_JIT_GLOBAL_SCRATCH_SIZE
    if workspace_size > 2**63 - workspace_alignment:
        raise ValueError(
            "Iluvatar aligned workspace requirement overflows int64"
        )
    if workspace_size:
        workspace_size = (
            (workspace_size + workspace_alignment - 1)
            // workspace_alignment
            * workspace_alignment
        )
    else:
        workspace_alignment = 1
    output_directory.mkdir(parents=True, exist_ok=True)
    compiler_path = _compiler_entry_path()
    tensor_to_stage: dict[int, int] = {}
    stages: list[dict[str, Any]] = []
    for node in parsed_nodes:
        external_dependencies = sorted(
            {
                tensor_to_stage[uid]
                for uid in node["input_uids"]
                if uid in tensor_to_stage
            }
        )
        operation = node["operation"]
        if operation in compiler_nn.SUPPORTED_OPERATIONS:
            plan = nn_plans[node["id"]]
            local_stage_ids: dict[str, int] = {}
            for configuration in plan.stages:
                stage_id = len(stages)
                dependencies = (
                    external_dependencies
                    if not local_stage_ids
                    else [stage_id - 1]
                )
                stages.append(
                    _compile_nn_stage(
                        stage_id=stage_id,
                        node=node,
                        configuration=configuration,
                        dependencies=dependencies,
                        workspace=workspace,
                        local_workspace=nn_workspaces[node["id"]],
                        compiler_path=compiler_path,
                        output_directory=output_directory,
                        enable_autotune=enable_autotune,
                    )
                )
                local_stage_ids[configuration.stage_name] = stage_id
            if not local_stage_ids:
                raise ValueError(
                    "Iluvatar NN node produced no execution stages"
                )
            producer_stage = len(stages) - 1
        else:
            stage_id = len(stages)
            compile_stage = (
                _compile_pointwise_stage
                if operation in POINTWISE_SCHEMAS
                else _compile_tensor_stage
            )
            stages.append(
                compile_stage(
                    stage_id=stage_id,
                    node=node,
                    dependencies=external_dependencies,
                    workspace=workspace,
                    compiler_path=compiler_path,
                    output_directory=output_directory,
                    enable_autotune=enable_autotune,
                )
            )
            producer_stage = stage_id
        for uid in node["output_uids"]:
            tensor_to_stage[uid] = producer_stage

    source_hash = hashlib.sha256(
        json.dumps(
            [stage["source_sha256"] for stage in stages],
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    external_uids = sorted(
        uid for uid, tensor in tensor_registry.items() if not tensor["virtual"]
    )
    manifest_tensors = [
        _manifest_tensor(tensor_registry[uid], workspace)
        for uid in sorted(tensor_registry)
    ]
    manifest: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "flagdnn_execution_program",
        "flagdnn_version": flagdnn_version,
        "graph_ir_schema_version": SCHEMA_VERSION,
        "backend_abi_version": 2,
        "backend": "iluvatar",
        "target": target_name,
        "warp_size": ILUVATAR_WARP_SIZE,
        "engine": "libtriton_jit",
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "source_sha256": source_hash,
        "compiler": {
            **identity,
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "torch_loaded": "torch" in sys.modules,
        },
        "external_uids": external_uids,
        "tensor_count": len(manifest_tensors),
        "tensors": manifest_tensors,
        "workspace": {
            "size": workspace_size,
            "alignment": workspace_alignment,
        },
        "program": {
            "schema_version": EXECUTION_PROGRAM_VERSION,
            "stage_count": len(stages),
            "stages": stages,
        },
    }
    _atomic_write(
        output_directory / "manifest.json",
        (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        ),
    )
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "success",
        "backend": "iluvatar",
        "provider": PROVIDER_NAME,
        "node_count": node_count,
        "stage_count": len(stages),
        "target": target_name,
        "target_backend": "corex",
        "warp_size": ILUVATAR_WARP_SIZE,
        "artifact_directory": str(output_directory),
        "workspace_size": workspace_size,
        "workspace_alignment": workspace_alignment,
        "execution_engine": "libtriton_jit",
        "triton_version": triton.__version__,
        "torch_loaded": "torch" in sys.modules,
    }
