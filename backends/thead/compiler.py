# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""THead PPU compiler provider for FlagDNN Graph IR.

The provider validates the complete request before selecting a kernel.  Each
supported slice materializes an immutable execution-program artifact; all
other operation families return a typed unsupported result.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import yaml

from flagdnn_codegen import kernel_registry

from .compiler_identity import (
    PROVIDER_NAME,
    PROVIDER_VERSION,
    build_compiler_identity,
    compiler_identity_dependency_paths,
)


SCHEMA_VERSION = 3
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_DATA_TYPES = {
    "float32",
    "float16",
    "bfloat16",
    "boolean",
    "fp8_e4m3",
    "fp8_e5m2",
}
_DATA_TYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "boolean": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
}
_MAX_KERNEL_SOURCE_BYTES = 1024 * 1024
_PPU_WARP_SIZE = 32
_FIXED_BLOCK_SIZE = 256
_FIXED_NUM_WARPS = 4
_FIXED_NUM_STAGES = 1
_BINARY_POINTWISE_MODES = {
    "add": 1,
    "sub": 17,
    "mul": 18,
    "min": 20,
    "max": 21,
    "div": 19,
    "mod": 22,
    "pow": 23,
    "sigmoid_backward": 40,
    "cmp_eq": 25,
    "cmp_neq": 26,
    "cmp_gt": 27,
    "cmp_ge": 28,
    "cmp_lt": 29,
    "cmp_le": 30,
    "logical_and": 31,
    "logical_or": 32,
}
_UNARY_POINTWISE_MODES = {
    "relu": 2,
    "identity": 5,
    "sigmoid": 33,
    "tanh": 34,
    "elu": 35,
    "gelu": 36,
    "sqrt": 3,
    "neg": 8,
    "abs": 9,
    "ceil": 10,
    "floor": 12,
    "exp": 6,
    "log": 7,
    "cos": 11,
    "rsqrt": 13,
    "sin": 14,
    "tan": 15,
    "softplus": 37,
    "swish": 38,
    "gelu_approx_tanh": 39,
    "reciprocal": 16,
    "logical_not": 24,
}
_COMPARISON_OPERATIONS = {
    "cmp_eq",
    "cmp_neq",
    "cmp_gt",
    "cmp_ge",
    "cmp_lt",
    "cmp_le",
}
_LOGICAL_OPERATIONS = {"logical_not", "logical_and", "logical_or"}
_FLOATING_DATA_TYPES = {"float32", "float16", "bfloat16"}
_LAYOUT_OPERATIONS = {"reshape", "transpose", "slice"}
_REDUCTION_OPERATIONS = {
    "reduction_sum": 1,
    "reduction_avg": 2,
    "reduction_mul": 3,
}
_REDUCTION_GRAPH_MODES = {
    "reduction_sum": 0,
    "reduction_avg": 1,
    "reduction_mul": 2,
}
_BATCHNORM_OPERATIONS = {"batchnorm", "batchnorm_inference"}
_NORMALIZATION_OPERATIONS = {"layernorm", "rmsnorm"}
_MAX_NORMALIZATION_ELEMENTS = 65536
_MATMUL_OPERATIONS = {"matmul"}
_CONVOLUTION_OPERATIONS = {
    "convolution_fprop",
    "convolution_dgrad",
    "convolution_wgrad",
}
_CONV_BIAS_RELU_OPERATION_TYPES = ["add", "convolution_fprop", "relu"]
_TUNING_TABLES = {
    "binary",
    "relu",
    "reduction",
    "batch_norm",
    "layer_norm",
    "rms_norm",
    "matmul",
    "conv2d_spatial",
}
_APPROVED_PPU_COMPILER_OPTIONS = {
    "debug",
    "enable_fp_fusion",
    "enable_reflect_ftz",
    "instrumentation_mode",
    "ppu_llc_options",
    "sanitize_overflow",
}


def compiler_identity(
    target: str, execution_engine: str = "libtriton_jit"
) -> dict[str, Any]:
    return build_compiler_identity(target, execution_engine)


def compiler_identity_dependencies(
    target: str, execution_engine: str = "libtriton_jit"
) -> tuple[Path, ...]:
    return compiler_identity_dependency_paths(target, execution_engine)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON key: {name}")
        result[name] = value
    return result


def _load_request(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        request_bytes = path.read_bytes()
        value = json.loads(
            request_bytes,
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read compiler request: {error}") from error
    return _require_object(value, "request"), request_bytes


def _require_object(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object")
    return value


def _require_list(value: object, description: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{description} must be an array")
    return value


def _require_exact_fields(
    value: dict[str, Any], required: set[str], optional: set[str], description: str
) -> None:
    missing = required.difference(value)
    unknown = set(value).difference(required | optional)
    if missing:
        raise ValueError(
            f"{description} is missing fields: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise ValueError(
            f"{description} has unknown fields: {', '.join(sorted(unknown))}"
        )


def _integer(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{description} must be an integer")
    return value


def _string(value: object, description: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not value and not allow_empty):
        raise ValueError(f"{description} must be a nonempty string")
    if len(value.encode("utf-8")) > 4096:
        raise ValueError(f"{description} is too long")
    return value


def _parse_build_options(value: object) -> bool:
    options = _require_object(value, "build_options")
    _require_exact_fields(
        options, {"heuristic_modes", "autotune"}, set(), "build_options"
    )
    modes = _require_list(options["heuristic_modes"], "heuristic_modes")
    if (
        not modes
        or any(mode not in {"A", "FALLBACK"} for mode in modes)
        or len(set(modes)) != len(modes)
    ):
        raise ValueError("heuristic_modes must contain unique A/FALLBACK values")
    if not isinstance(options["autotune"], bool):
        raise ValueError("build_options.autotune must be a boolean")
    return options["autotune"]


def _parse_tensor(value: object, index: int) -> tuple[int, bool]:
    tensor = _require_object(value, f"graph.tensors[{index}]")
    _require_exact_fields(
        tensor,
        {"uid", "data_type", "dimensions", "strides", "alignment", "virtual"},
        set(),
        f"graph.tensors[{index}]",
    )
    uid = _integer(tensor["uid"], "tensor uid")
    if not 0 <= uid < 2**63:
        raise ValueError("tensor uid is outside the nonnegative int64 range")
    data_type = _string(tensor["data_type"], "tensor data_type")
    if data_type not in _DATA_TYPES:
        raise ValueError(f"unsupported tensor data_type: {data_type}")
    dimensions = _require_list(tensor["dimensions"], "tensor dimensions")
    strides = _require_list(tensor["strides"], "tensor strides")
    if len(dimensions) > 8:
        raise ValueError("tensor dimensions must have rank 0 through 8")
    if len(strides) != len(dimensions):
        raise ValueError("tensor strides must match dimensions rank")
    if any(
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or not 1 <= dimension <= 2**31 - 1
        for dimension in dimensions
    ):
        raise ValueError("tensor dimensions must be positive int32 values")
    if any(
        isinstance(stride, bool)
        or not isinstance(stride, int)
        or not 0 <= stride < 2**63
        for stride in strides
    ):
        raise ValueError("tensor strides must be nonnegative int64 values")
    storage_size = (
        1
        + sum(
            (dimension - 1) * stride
            for dimension, stride in zip(dimensions, strides, strict=True)
        )
    ) * _DATA_TYPE_BYTES[data_type]
    # The native JSON/artifact ABI represents sizes as signed int64 before
    # converting them to size_t.  Reject an unrepresentable view here instead
    # of emitting an artifact that the runtime must reject later.
    if storage_size > 2**63 - 1:
        raise ValueError("tensor storage size is outside positive int64 range")
    alignment = _integer(tensor["alignment"], "tensor alignment")
    if alignment <= 0 or alignment > 4096 or alignment & (alignment - 1):
        raise ValueError("tensor alignment must be a power of two through 4096")
    if not isinstance(tensor["virtual"], bool):
        raise ValueError("tensor virtual must be a boolean")
    return uid, tensor["virtual"]


def _parse_ports(
    value: object,
    description: str,
    tensor_uids: set[int],
    *,
    allow_optional: bool,
) -> list[int]:
    ports = _require_list(value, description)
    if not ports:
        raise ValueError(f"{description} must not be empty")
    uids: list[int] = []
    names: set[str] = set()
    for index, port_value in enumerate(ports):
        port = _require_object(port_value, f"{description}[{index}]")
        _require_exact_fields(
            port,
            {"name", "uid"},
            {"optional"} if allow_optional else set(),
            f"{description}[{index}]",
        )
        name = _string(port["name"], f"{description}[{index}].name")
        if _NAME.fullmatch(name) is None:
            raise ValueError(f"{description} port name is invalid")
        if name in names:
            raise ValueError(f"{description} port names must be unique")
        names.add(name)
        uid = _integer(port["uid"], f"{description}[{index}].uid")
        if uid not in tensor_uids:
            raise ValueError(f"{description} references an unknown tensor uid")
        optional = port.get("optional", False)
        if not isinstance(optional, bool):
            raise ValueError(f"{description} optional flag must be a boolean")
        uids.append(uid)
    return uids


def _validate_attribute(value: object, description: str) -> None:
    if value is None:
        raise ValueError(f"{description} must not be null")
    if isinstance(value, bool) or isinstance(value, str):
        return
    if isinstance(value, int):
        if not -(2**63) <= value < 2**63:
            raise ValueError(f"{description} integer is outside int64")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{description} must be finite")
        return
    if isinstance(value, list):
        if len(value) > 1024:
            raise ValueError(f"{description} array is too large")
        for index, item in enumerate(value):
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError(f"{description}[{index}] must be an integer")
            if not -(2**63) <= item < 2**63:
                raise ValueError(f"{description}[{index}] is outside int64")
        return
    raise ValueError(f"{description} has an unsupported value type")


def _parse_graph(value: object) -> list[str]:
    graph = _require_object(value, "graph")
    _require_exact_fields(
        graph,
        {"name", "tensor_count", "tensors", "node_count", "nodes"},
        set(),
        "graph",
    )
    _string(graph["name"], "graph name", allow_empty=True)
    tensors = _require_list(graph["tensors"], "graph.tensors")
    tensor_count = _integer(graph["tensor_count"], "graph tensor_count")
    if tensor_count != len(tensors) or not 1 <= tensor_count <= 4096:
        raise ValueError("graph tensor_count is invalid")
    tensor_registry: dict[int, bool] = {}
    for index, tensor_value in enumerate(tensors):
        uid, virtual = _parse_tensor(tensor_value, index)
        if uid in tensor_registry:
            raise ValueError("graph tensor uids must be unique")
        tensor_registry[uid] = virtual

    nodes = _require_list(graph["nodes"], "graph.nodes")
    node_count = _integer(graph["node_count"], "graph node_count")
    if node_count != len(nodes) or not 1 <= node_count <= 1024:
        raise ValueError("graph node_count is invalid")
    node_ids: set[int] = set()
    operation_types: set[str] = set()
    for index, node_value in enumerate(nodes):
        node = _require_object(node_value, f"graph.nodes[{index}]")
        _require_exact_fields(
            node,
            {
                "id",
                "type",
                "name",
                "compute_data_type",
                "inputs",
                "outputs",
                "attributes",
            },
            set(),
            f"graph.nodes[{index}]",
        )
        node_id = _integer(node["id"], "node id")
        if not 0 <= node_id < node_count or node_id in node_ids:
            raise ValueError("graph node ids must be unique values in range")
        node_ids.add(node_id)
        operation = _string(node["type"], "node type")
        if _NAME.fullmatch(operation) is None:
            raise ValueError("node type is invalid")
        operation_types.add(operation)
        _string(node["name"], "node name", allow_empty=True)
        compute_type = _string(
            node["compute_data_type"], "node compute_data_type"
        )
        if compute_type not in _DATA_TYPES:
            raise ValueError("node compute_data_type is unsupported")
        _parse_ports(
            node["inputs"],
            f"graph.nodes[{index}].inputs",
            set(tensor_registry),
            allow_optional=True,
        )
        _parse_ports(
            node["outputs"],
            f"graph.nodes[{index}].outputs",
            set(tensor_registry),
            allow_optional=False,
        )
        attributes = _require_object(node["attributes"], "node attributes")
        for name, attribute in attributes.items():
            if not isinstance(name, str) or _NAME.fullmatch(name) is None:
                raise ValueError("node attribute name is invalid")
            _validate_attribute(attribute, f"node attribute {name}")
    return sorted(operation_types)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    """Publish one artifact file atomically without following a destination link."""

    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"artifact output is not a regular file: {path}")
        if path.read_bytes() == data:
            return
        raise ValueError(f"artifact output already contains different bytes: {path}")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as output:
            temporary_name = output.name
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _dense_strides(dimensions: list[Any]) -> list[int]:
    result = [0] * len(dimensions)
    stride = 1
    for index in range(len(dimensions) - 1, -1, -1):
        result[index] = stride
        stride *= int(dimensions[index])
    return result


def _has_non_overlapping_strides(
    dimensions: list[Any], strides: list[Any]
) -> bool:
    axes = sorted(
        (
            (int(stride), int(dimension))
            for dimension, stride in zip(dimensions, strides, strict=True)
            if int(dimension) > 1
        ),
        key=lambda item: item[0],
    )
    required_stride = 1
    for stride, dimension in axes:
        if stride < required_stride:
            return False
        required_stride = stride * dimension
    return True


def _padded_pointwise_values(values: list[Any], fill: int) -> list[int]:
    if not 1 <= len(values) <= 8:
        raise ValueError("THead pointwise metadata requires rank 1 through 8")
    return [fill] * (8 - len(values)) + [int(value) for value in values]


def _same_dense_layout(tensors: list[dict[str, Any]]) -> bool:
    """Whether a physical linear walk pairs the same logical elements.

    Dense permutations (including NHWC) need no coordinate decomposition when
    all operands share their layout. Singleton strides do not affect addresses.
    Padded tensors and operands with different layouts still use strided kernels.
    """
    dimensions = tensors[0]["dimensions"]
    axes = [axis for axis, size in enumerate(dimensions) if int(size) > 1]
    strides = tensors[0]["strides"]
    extent = 1
    for axis in sorted(axes, key=lambda axis: int(strides[axis])):
        if int(strides[axis]) != extent:
            return False
        extent *= int(dimensions[axis])
    return all(
        tensor["dimensions"] == dimensions
        and all(int(tensor["strides"][axis]) == int(strides[axis]) for axis in axes)
        for tensor in tensors[1:]
    )


def _pointwise_block_size(n_elements: int) -> int:
    # Larger vector tiles amortize program scheduling on bandwidth-bound work.
    return 1024 if n_elements >= 65536 else _FIXED_BLOCK_SIZE


def _named_port_uids(
    node: dict[str, Any],
    field: str,
    expected_names: tuple[str, ...],
    operation_label: str,
) -> list[int]:
    ports = _require_list(node[field], f"{operation_label} {field}")
    names = tuple(port["name"] for port in ports)
    if names != expected_names:
        raise ValueError(
            f"THead {operation_label} {field} must be ordered as "
            f"{', '.join(expected_names)}"
        )
    if any(port.get("optional", False) for port in ports):
        raise ValueError(
            f"THead {operation_label} slice does not accept optional ports"
        )
    return [int(port["uid"]) for port in ports]


def _validate_binary_pointwise_graph(
    graph: dict[str, Any], operation: str
) -> dict[str, Any]:
    pointwise_mode = _BINARY_POINTWISE_MODES[operation]
    operation_label = operation.capitalize()
    tensors = _require_list(graph["tensors"], "graph.tensors")
    nodes = _require_list(graph["nodes"], "graph.nodes")
    if len(nodes) != 1 or len(tensors) != 3:
        raise ValueError(
            f"THead {operation_label} slice requires one node and three tensors"
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
    elif (
        tensor_types[0] not in _FLOATING_DATA_TYPES
        or any(data_type != tensor_types[0] for data_type in tensor_types[1:])
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
            f"THead {operation_label} slice requires at least 16-byte alignment"
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
        set(),
        f"{operation_label} attributes",
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
        or (
            operation not in {"add", "sub"}
            and float(alpha) != 1.0
        )
    ):
        raise ValueError(
            f"THead {operation_label} alpha is not qualified"
        )
    alpha = float(alpha)
    n_elements = _integer(
        attributes["n_elements"], f"{operation_label} n_elements"
    )
    expected_elements = math.prod(int(value) for value in left_dimensions)
    if n_elements != expected_elements or not 1 <= n_elements <= 2**31 - 1:
        raise ValueError(
            f"THead {operation_label} n_elements does not match the output shape"
        )
    return {
        "node": node,
        "tensors": tensors,
        "ordered_uids": ordered_uids,
        "n_elements": n_elements,
        "alpha": alpha,
        "function": (
            "binary_strided_kernel"
            if strided
            else "binary_contiguous_kernel"
        ),
        "stride_constants": (
            [
                *_padded_pointwise_values(left_dimensions, 1),
                *(
                    value
                    for tensor in ordered_tensors
                    for value in _padded_pointwise_values(
                        tensor["strides"], 0
                    )
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

    input_uids = _named_port_uids(
        node, "inputs", ("input",), operation_label
    )
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
        else tensor_types[0] in _FLOATING_DATA_TYPES
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
            f"THead {operation_label} slice requires at least 16-byte alignment"
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
            f"THead {operation_label} n_elements does not match the output shape"
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
                    for value in _padded_pointwise_values(
                        tensor["strides"], 0
                    )
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
        raise ValueError(
            "THead AddSquare requires two nodes and four tensors"
        )
    multiply = _require_object(nodes[0], "AddSquare Mul node")
    add = _require_object(nodes[1], "AddSquare Add node")
    if multiply["id"] != 0 or multiply["type"] != "mul":
        raise ValueError("THead AddSquare requires canonical Mul node id 0")
    if add["id"] != 1 or add["type"] != "add":
        raise ValueError("THead AddSquare requires canonical Add node id 1")
    if any(
        node["compute_data_type"] != "float32"
        for node in (multiply, add)
    ):
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
    if (
        role_tensors[0]["data_type"] not in _FLOATING_DATA_TYPES
        or any(
            tensor["data_type"] != role_tensors[0]["data_type"]
            for tensor in role_tensors[1:]
        )
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
    strided = not _same_dense_layout([role_tensors[index] for index in (0, 1, 3)])
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
            set(),
            f"AddSquare {label} attributes",
        )
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
        if _integer(
            attributes["n_elements"], f"AddSquare {label} n_elements"
        ) != n_elements:
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
                    for value in _padded_pointwise_values(
                        tensor["strides"], 0
                    )
                ),
            ]
            if strided
            else None
        ),
    }


def _require_integer_array(
    attributes: dict[str, Any], name: str, length: int
) -> list[int]:
    values = _require_list(attributes.get(name), f"layout attribute {name}")
    if len(values) != length or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in values
    ):
        raise ValueError(f"THead layout {name} must contain {length} integers")
    return [int(value) for value in values]


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
        raise ValueError(f"THead {operation} compute data type must be float32")
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
        input_tensor["data_type"] not in _FLOATING_DATA_TYPES
        or output_tensor["data_type"] != input_tensor["data_type"]
    ):
        raise ValueError(
            f"THead {operation} requires matching "
            "float32/float16/bfloat16 tensors"
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
        if input_strides != _dense_strides(input_dimensions) or output_strides != _dense_strides(output_dimensions):
            raise ValueError("THead reshape slice requires contiguous tensors")
    elif operation == "transpose":
        _require_exact_fields(
            attributes,
            common_fields | {"rank", "permutation"},
            set(),
            "Transpose attributes",
        )
        rank = len(input_dimensions)
        if rank == 0 or rank != len(output_dimensions) or _integer(
            attributes["rank"], "transpose rank"
        ) != rank:
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
        if rank == 0 or rank != len(output_dimensions) or _integer(
            attributes["rank"], "slice rank"
        ) != rank:
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
                (limits[axis] - starts[axis] + steps[axis] - 1)
                // steps[axis]
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
    input_uid = _named_port_uids(
        node, "inputs", ("input",), "Reduction"
    )[0]
    output_uid = _named_port_uids(
        node, "outputs", ("output",), "Reduction"
    )[0]
    if input_uid == output_uid:
        raise ValueError("THead reduction tensor UIDs must differ")
    tensors_by_uid = {int(tensor["uid"]): tensor for tensor in tensors}
    input_tensor = tensors_by_uid[input_uid]
    output_tensor = tensors_by_uid[output_uid]
    if (
        input_tensor["data_type"] not in _FLOATING_DATA_TYPES
        or output_tensor["data_type"] != input_tensor["data_type"]
    ):
        raise ValueError(
            "THead reduction requires matching "
            "float32/float16/bfloat16 tensors"
        )
    if any(tensor["virtual"] for tensor in (input_tensor, output_tensor)):
        raise ValueError("THead reduction requires external tensors")
    scalar_bytes = 4 if input_tensor["data_type"] == "float32" else 2
    if any(
        int(tensor["alignment"]) < scalar_bytes
        for tensor in (input_tensor, output_tensor)
    ):
        raise ValueError("THead reduction alignment is below scalar size")
    input_dimensions = [int(value) for value in input_tensor["dimensions"]]
    output_dimensions = [int(value) for value in output_tensor["dimensions"]]
    input_strides = [int(value) for value in input_tensor["strides"]]
    output_strides = [int(value) for value in output_tensor["strides"]]
    if not _has_non_overlapping_strides(input_dimensions, input_strides) or not (
        _has_non_overlapping_strides(output_dimensions, output_strides)
    ):
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
        raise ValueError("THead reduction shape is outside the qualified slice")

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
            constants[f"INPUT_STRIDE_{padded_axis}"] = (
                padded_input_strides[padded_axis]
            )
            constants[f"OUTPUT_STRIDE_{padded_axis}"] = (
                padded_output_strides[padded_axis]
            )
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


def _validate_batchnorm_graph(
    graph: dict[str, Any], operation: str
) -> dict[str, Any]:
    inference = operation == "batchnorm_inference"
    tensors = _require_list(graph["tensors"], "graph.tensors")
    nodes = _require_list(graph["nodes"], "graph.nodes")
    expected_count = 6 if inference else 10
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
        raise ValueError(f"THead {operation} requires a canonical fp32 node")
    if inference:
        input_names = ("x", "mean", "inv_variance", "scale", "bias")
        output_names = ("y",)
    else:
        input_names = (
            "x",
            "scale",
            "bias",
            "previous_running_mean",
            "previous_running_variance",
        )
        output_names = (
            "y",
            "mean",
            "inv_variance",
            "next_running_mean",
            "next_running_variance",
        )
    input_uids = _named_port_uids(
        node, "inputs", input_names, operation
    )
    output_uids = _named_port_uids(
        node, "outputs", output_names, operation
    )
    if len(set(input_uids + output_uids)) != expected_count:
        raise ValueError(f"THead {operation} requires distinct external tensors")
    tensors_by_uid = {int(tensor["uid"]): tensor for tensor in tensors}
    ordered = [tensors_by_uid[uid] for uid in input_uids + output_uids]
    if any(
        tensor["virtual"] or int(tensor["alignment"]) < 16
        for tensor in ordered
    ):
        raise ValueError(f"THead {operation} requires aligned external tensors")
    data_type = tensors_by_uid[input_uids[0]]["data_type"]
    if data_type not in _FLOATING_DATA_TYPES:
        raise ValueError(f"THead {operation} data type is unsupported")
    data_uids = (
        {input_uids[0], output_uids[0]}
        if inference
        else {input_uids[0], input_uids[1], input_uids[2], output_uids[0]}
    )
    statistic_uids = set(input_uids + output_uids).difference(data_uids)
    if any(tensors_by_uid[uid]["data_type"] != data_type for uid in data_uids):
        raise ValueError(f"THead {operation} requires matching data tensors")
    if any(
        tensors_by_uid[uid]["data_type"] != "float32"
        for uid in statistic_uids
    ):
        raise ValueError(f"THead {operation} statistics must use float32")
    x = tensors_by_uid[input_uids[0]]
    y = tensors_by_uid[output_uids[0]]
    dimensions = [int(value) for value in x["dimensions"]]
    if len(dimensions) < 2 or list(y["dimensions"]) != dimensions:
        raise ValueError(f"THead {operation} X/Y shape is invalid")
    if any(
        not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        )
        for tensor in (x, y)
    ):
        raise ValueError(
            f"THead {operation} requires non-overlapping X/Y tensors"
        )
    if any(
        list(tensor["strides"]) != _dense_strides(tensor["dimensions"])
        for tensor in ordered
        if int(tensor["uid"]) not in {int(x["uid"]), int(y["uid"])}
    ):
        raise ValueError(
            f"THead {operation} parameters and statistics must be contiguous"
        )
    strided = any(
        list(tensor["strides"]) != _dense_strides(tensor["dimensions"])
        for tensor in (x, y)
    )
    batch = dimensions[0]
    channels = dimensions[1]
    spatial = math.prod(dimensions[2:])
    n_elements = math.prod(dimensions)
    if any(
        value < 1 or value > 2**31 - 1
        for value in (batch, channels, spatial, n_elements)
    ):
        raise ValueError(f"THead {operation} dimensions are outside int32")
    parameter_dimensions = [int(value) for value in ordered[1]["dimensions"]]
    if math.prod(parameter_dimensions) != channels:
        raise ValueError(f"THead {operation} parameter shape is invalid")
    data_uids = {input_uids[0], output_uids[0]}
    if any(
        math.prod(tensor["dimensions"]) != channels
        for tensor in ordered
        if int(tensor["uid"]) not in data_uids
    ):
        raise ValueError(f"THead {operation} statistic shape is invalid")

    attributes = _require_object(node["attributes"], f"{operation} attributes")
    integer_fields = {
        "n_elements",
        "channels",
        "spatial",
        "rank",
    }
    if not inference:
        integer_fields.add("batch")
    real_fields = set() if inference else {"epsilon", "momentum"}
    vector_fields = {"dimensions", "x_strides", "y_strides"}
    _require_exact_fields(
        attributes,
        integer_fields | real_fields | vector_fields,
        set(),
        f"{operation} attributes",
    )
    expected_integers = {
        "n_elements": n_elements,
        "channels": channels,
        "spatial": spatial,
        "rank": len(dimensions),
    }
    if not inference:
        expected_integers["batch"] = batch
    if any(
        _integer(attributes[name], f"{operation} {name}") != expected
        for name, expected in expected_integers.items()
    ):
        raise ValueError(f"THead {operation} integer attributes are inconsistent")
    if (
        list(attributes["dimensions"]) != dimensions
        or list(attributes["x_strides"]) != list(x["strides"])
        or list(attributes["y_strides"]) != list(y["strides"])
    ):
        raise ValueError(f"THead {operation} layout attributes are inconsistent")
    epsilon = 0.0
    momentum = 0.0
    if not inference:
        raw_epsilon = attributes["epsilon"]
        raw_momentum = attributes["momentum"]
        if (
            isinstance(raw_epsilon, bool)
            or not isinstance(raw_epsilon, (int, float))
            or not math.isfinite(raw_epsilon)
            or raw_epsilon <= 0.0
            or isinstance(raw_momentum, bool)
            or not isinstance(raw_momentum, (int, float))
            or not math.isfinite(raw_momentum)
            or not 0.0 <= raw_momentum <= 1.0
        ):
            raise ValueError(f"THead {operation} real attributes are invalid")
        epsilon = float(raw_epsilon)
        momentum = float(raw_momentum)

    if inference:
        argument_tensors = [tensors_by_uid[uid] for uid in input_uids + output_uids]
        function = (
            "batch_norm_inference_kernel"
            if strided
            else "batch_norm_inference_nchw_kernel"
        )
    else:
        # Kernel ABI: x, y, previous stats, affine parameters, saved stats,
        # and next running stats.
        kernel_uids = [
            input_uids[0], output_uids[0], input_uids[3], input_uids[4],
            input_uids[1], input_uids[2], output_uids[1], output_uids[2],
            output_uids[3], output_uids[4],
        ]
        argument_tensors = [tensors_by_uid[uid] for uid in kernel_uids]
        # The NCHW specialization forms a compile-time N x spatial tile whose
        # extent is BLOCK_SIZE.  Its default qualified block is 256, so large
        # batches must use the general loop kernel even for dense layouts.
        function = (
            "batch_norm_kernel"
            if strided or batch > _FIXED_BLOCK_SIZE
            else "batch_norm_nchw_kernel"
        )
    return {
        "tensors": tensors,
        "argument_tensors": argument_tensors,
        "function": function,
        "n_elements": n_elements,
        "batch": batch,
        "channels": channels,
        "spatial": spatial,
        "epsilon": epsilon,
        "momentum": momentum,
        "stride_constants": (
            [
                *_padded_pointwise_values(dimensions, 1),
                *_padded_pointwise_values(list(x["strides"]), 0),
                *_padded_pointwise_values(list(y["strides"]), 0),
            ]
            if function in {"batch_norm_kernel", "batch_norm_inference_kernel"}
            else None
        ),
    }


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
        ("y", "mean", "inv_variance")
        if layernorm
        else ("y", "inv_variance")
    )
    output_uids = _named_port_uids(
        node, "outputs", output_names, operation
    )
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
        raise ValueError(f"THead {operation} requires aligned external tensors")
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
        tensors_by_uid[uid]["data_type"] != "float32"
        for uid in statistic_uids
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
            f"THead {operation} normalized suffix is outside the qualified slice"
        )
    rows = math.prod(dimensions) // normalized_elements
    if rows > 2**31 - 1:
        raise ValueError(f"THead {operation} row count exceeds int32")
    statistic_uids = output_uids[1:]
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
        raise ValueError(f"THead {operation} integer attributes are inconsistent")
    raw_epsilon = attributes["epsilon"]
    if (
        isinstance(raw_epsilon, bool)
        or not isinstance(raw_epsilon, (int, float))
        or not math.isfinite(raw_epsilon)
        or raw_epsilon <= 0.0
    ):
        raise ValueError(f"THead {operation} epsilon is invalid")
    if _integer(
        attributes["forward_phase"], f"{operation} forward_phase"
    ) != 2:
        raise ValueError(
            f"THead {operation} supports TRAINING phase only"
        )

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


def _validate_matmul_graph(graph: dict[str, Any]) -> dict[str, Any]:
    tensors = _require_list(graph["tensors"], "graph.tensors")
    nodes = _require_list(graph["nodes"], "graph.nodes")
    if len(nodes) != 1 or len(tensors) != 3:
        raise ValueError("THead MatMul requires one node and three tensors")
    node = _require_object(nodes[0], "MatMul node")
    if (
        node["id"] != 0
        or node["type"] != "matmul"
        or node["compute_data_type"] != "float32"
    ):
        raise ValueError("THead MatMul requires a canonical float32 node")
    input_uids = _named_port_uids(node, "inputs", ("a", "b"), "MatMul")
    output_uids = _named_port_uids(
        node, "outputs", ("output",), "MatMul"
    )
    ordered_uids = input_uids + output_uids
    if len(set(ordered_uids)) != 3:
        raise ValueError("THead MatMul tensor UIDs must be distinct")
    tensors_by_uid = {int(tensor["uid"]): tensor for tensor in tensors}
    a, b, output = [tensors_by_uid[uid] for uid in ordered_uids]
    if (
        a["data_type"] not in _FLOATING_DATA_TYPES
        or b["data_type"] != a["data_type"]
        or output["data_type"] != a["data_type"]
        or any(
            tensor["virtual"] or int(tensor["alignment"]) < 16
            for tensor in (a, b, output)
        )
    ):
        raise ValueError(
            "THead MatMul requires matching aligned external "
            "float32/float16/bfloat16 tensors"
        )
    if any(
        not 2 <= len(tensor["dimensions"]) <= 8
        or not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        )
        for tensor in (a, b, output)
    ):
        raise ValueError(
            "THead MatMul requires rank [2, 8] non-overlapping tensors"
        )

    a_dimensions = [int(value) for value in a["dimensions"]]
    b_dimensions = [int(value) for value in b["dimensions"]]
    output_dimensions = [int(value) for value in output["dimensions"]]
    m = a_dimensions[-2]
    k = a_dimensions[-1]
    if b_dimensions[-2] != k:
        raise ValueError("THead MatMul contraction dimensions do not match")
    n = b_dimensions[-1]
    a_batch = a_dimensions[:-2]
    b_batch = b_dimensions[:-2]
    batch_rank = max(len(a_batch), len(b_batch))
    if batch_rank > 6:
        raise ValueError("THead MatMul batch rank exceeds six")
    batch_dimensions = [1] * batch_rank
    for trailing in range(batch_rank):
        a_dimension = (
            a_batch[-1 - trailing] if trailing < len(a_batch) else 1
        )
        b_dimension = (
            b_batch[-1 - trailing] if trailing < len(b_batch) else 1
        )
        if (
            a_dimension != b_dimension
            and a_dimension != 1
            and b_dimension != 1
        ):
            raise ValueError(
                "THead MatMul batch dimensions are not broadcast-compatible"
            )
        batch_dimensions[-1 - trailing] = max(a_dimension, b_dimension)
    expected_output = [*batch_dimensions, m, n]
    if output_dimensions != expected_output:
        raise ValueError("THead MatMul output shape is inconsistent")
    batch = math.prod(batch_dimensions)
    if batch * m * n > 2**31 - 1:
        raise ValueError("THead MatMul output element count exceeds int32")

    attributes = _require_object(node["attributes"], "MatMul attributes")
    _require_exact_fields(
        attributes,
        {"batch", "m", "n", "k"},
        set(),
        "MatMul attributes",
    )
    encoded = {
        name: _integer(attributes[name], f"MatMul {name}")
        for name in ("batch", "m", "n", "k")
    }
    if encoded != {"batch": batch, "m": m, "n": n, "k": k}:
        raise ValueError("THead MatMul attributes disagree with tensors")

    leading = 6 - batch_rank
    padded_dimensions = [1] * leading + batch_dimensions

    def batch_strides(tensor: dict[str, Any]) -> list[int]:
        tensor_dimensions = [int(value) for value in tensor["dimensions"][:-2]]
        tensor_strides = [int(value) for value in tensor["strides"][:-2]]
        tensor_leading = batch_rank - len(tensor_dimensions)
        aligned_dimensions = [1] * tensor_leading + tensor_dimensions
        aligned_strides = [0] * tensor_leading + tensor_strides
        return [0] * leading + [
            0 if dimension == 1 else stride
            for dimension, stride in zip(
                aligned_dimensions, aligned_strides, strict=True
            )
        ]

    constants: dict[str, int] = {
        "M": m,
        "N": n,
        "K": k,
        "A_STRIDE_M": int(a["strides"][-2]),
        "A_STRIDE_K": int(a["strides"][-1]),
        "B_STRIDE_K": int(b["strides"][-2]),
        "B_STRIDE_N": int(b["strides"][-1]),
        "C_STRIDE_M": int(output["strides"][-2]),
        "C_STRIDE_N": int(output["strides"][-1]),
        "INPUT_IS_FLOAT32": 1 if a["data_type"] == "float32" else 0,
        # acDNN's FP32 reference is compared against IEEE multiplication;
        # TF32 is deliberately excluded from the qualified THead slice.
        "USE_TF32": 0,
    }
    a_batch_strides = batch_strides(a)
    b_batch_strides = batch_strides(b)
    c_batch_strides = [0] * leading + [
        int(value) for value in output["strides"][:-2]
    ]
    for axis in range(6):
        constants[f"DIM_{axis}"] = padded_dimensions[axis]
        constants[f"A_BATCH_STRIDE_{axis}"] = a_batch_strides[axis]
        constants[f"B_BATCH_STRIDE_{axis}"] = b_batch_strides[axis]
        constants[f"C_BATCH_STRIDE_{axis}"] = c_batch_strides[axis]
    return {
        "tensors": tensors,
        "argument_tensors": [a, b, output],
        "constants": constants,
        "batch": batch,
        "m": m,
        "n": n,
        "k": k,
    }


def _validate_convolution_graph(
    graph: dict[str, Any], operation: str
) -> dict[str, Any]:
    tensors = _require_list(graph["tensors"], "graph.tensors")
    nodes = _require_list(graph["nodes"], "graph.nodes")
    if len(nodes) != 1 or len(tensors) != 3:
        raise ValueError(
            f"THead {operation} requires one node and three tensors"
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
    port_names = {
        "convolution_fprop": (("input", "filter"), ("output",)),
        "convolution_dgrad": (("dy", "w"), ("dx",)),
        "convolution_wgrad": (("dy", "x"), ("dw",)),
    }[operation]
    input_uids = _named_port_uids(
        node, "inputs", port_names[0], operation
    )
    output_uids = _named_port_uids(
        node, "outputs", port_names[1], operation
    )
    argument_uids = input_uids + output_uids
    if len(set(argument_uids)) != 3:
        raise ValueError(f"THead {operation} tensor UIDs must be distinct")
    tensors_by_uid = {int(tensor["uid"]): tensor for tensor in tensors}
    arguments = [tensors_by_uid[uid] for uid in argument_uids]
    data_type = arguments[0]["data_type"]
    if (
        data_type not in _FLOATING_DATA_TYPES
        or any(tensor["data_type"] != data_type for tensor in arguments)
        or any(
            tensor["virtual"] or int(tensor["alignment"]) < 16
            for tensor in arguments
        )
    ):
        raise ValueError(
            f"THead {operation} requires aligned matching floating tensors"
        )

    attributes = _require_object(
        node["attributes"], f"{operation} attributes"
    )
    required_attributes = {
        "spatial_rank",
        "groups",
        "n_outputs",
        "pre_padding",
        "post_padding",
        "stride",
        "dilation",
    }
    if operation != "convolution_fprop":
        required_attributes.add("convolution_mode")
    _require_exact_fields(
        attributes,
        required_attributes,
        set(),
        f"{operation} attributes",
    )
    spatial_rank = _integer(
        attributes["spatial_rank"], f"{operation} spatial_rank"
    )
    if not 1 <= spatial_rank <= 3:
        raise ValueError(f"THead {operation} spatial_rank must be in [1, 3]")
    rank = spatial_rank + 2
    if any(
        len(tensor["dimensions"]) != rank
        or not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        )
        for tensor in arguments
    ):
        raise ValueError(
            f"THead {operation} requires rank spatial_rank + 2 "
            "non-overlapping tensors"
        )

    groups = _integer(attributes["groups"], f"{operation} groups")
    if groups <= 0:
        raise ValueError(f"THead {operation} groups must be positive")

    def spatial_attribute(name: str, minimum: int) -> list[int]:
        raw_values = _require_list(
            attributes[name], f"{operation} {name}"
        )
        values = [
            _integer(value, f"{operation} {name}[{index}]")
            for index, value in enumerate(raw_values)
        ]
        if len(values) != spatial_rank or any(
            value < minimum for value in values
        ):
            raise ValueError(f"THead {operation} {name} is invalid")
        return values

    pre_padding = spatial_attribute("pre_padding", 0)
    post_padding = spatial_attribute("post_padding", 0)
    stride = spatial_attribute("stride", 1)
    dilation = spatial_attribute("dilation", 1)
    convolution_mode = 0
    if operation != "convolution_fprop":
        convolution_mode = _integer(
            attributes["convolution_mode"],
            f"{operation} convolution_mode",
        )
        if convolution_mode not in (0, 1):
            raise ValueError(
                f"THead {operation} convolution_mode is invalid"
            )

    if operation == "convolution_fprop":
        image, filter_tensor, loss = arguments
    elif operation == "convolution_dgrad":
        loss, filter_tensor, image = arguments
    else:
        loss, image, filter_tensor = arguments
    image_dimensions = [int(value) for value in image["dimensions"]]
    filter_dimensions = [
        int(value) for value in filter_tensor["dimensions"]
    ]
    loss_dimensions = [int(value) for value in loss["dimensions"]]
    batch, channels = image_dimensions[:2]
    output_channels, filter_channels = filter_dimensions[:2]
    if channels % groups != 0 or output_channels % groups != 0:
        raise ValueError(
            f"THead {operation} channels must be divisible by groups"
        )
    channels_per_group = channels // groups
    outputs_per_group = output_channels // groups
    if filter_channels != channels_per_group:
        raise ValueError(
            f"THead {operation} filter channels disagree with groups"
        )
    expected_loss = [batch, output_channels]
    for axis in range(spatial_rank):
        effective_filter = (
            dilation[axis] * (filter_dimensions[axis + 2] - 1) + 1
        )
        padded_image = (
            image_dimensions[axis + 2]
            + pre_padding[axis]
            + post_padding[axis]
        )
        if padded_image < effective_filter:
            raise ValueError(
                f"THead {operation} filter exceeds padded input"
            )
        expected_loss.append(
            (padded_image - effective_filter) // stride[axis] + 1
        )
    if loss_dimensions != expected_loss:
        raise ValueError(f"THead {operation} output shape is inconsistent")
    n_outputs = _integer(
        attributes["n_outputs"], f"{operation} n_outputs"
    )
    graph_output = arguments[2]
    if n_outputs != math.prod(
        int(value) for value in graph_output["dimensions"]
    ):
        raise ValueError(
            f"THead {operation} n_outputs disagrees with output shape"
        )
    if not 1 <= n_outputs <= 2**31 - 1:
        raise ValueError(
            f"THead {operation} output element count exceeds int32"
        )

    def padded(values: list[Any], fill: int) -> list[int]:
        return [fill] * (3 - spatial_rank) + [int(value) for value in values]

    image_spatial = padded(image_dimensions[2:], 1)
    filter_spatial = padded(filter_dimensions[2:], 1)
    loss_spatial = padded(loss_dimensions[2:], 1)
    spatial_stride = padded(stride, 1)
    spatial_padding = padded(pre_padding, 0)
    spatial_dilation = padded(dilation, 1)
    image_strides = padded(image["strides"][2:], 0)
    filter_strides = padded(filter_tensor["strides"][2:], 0)
    loss_strides = padded(loss["strides"][2:], 0)
    constants: dict[str, int] = {
        "XD": image_spatial[0],
        "XH": image_spatial[1],
        "XW": image_spatial[2],
        "OD": loss_spatial[0],
        "OH": loss_spatial[1],
        "OW": loss_spatial[2],
        "KD": filter_spatial[0],
        "KH": filter_spatial[1],
        "KW": filter_spatial[2],
        "CIN_PER_GROUP": channels_per_group,
        "COUT_PER_GROUP": outputs_per_group,
        "GROUPS": groups,
        "STRIDE_D": spatial_stride[0],
        "STRIDE_H": spatial_stride[1],
        "STRIDE_W": spatial_stride[2],
        "PAD_FRONT": spatial_padding[0],
        "PAD_TOP": spatial_padding[1],
        "PAD_LEFT": spatial_padding[2],
        "DIL_D": spatial_dilation[0],
        "DIL_H": spatial_dilation[1],
        "DIL_W": spatial_dilation[2],
        "FLIP_FILTER": convolution_mode,
        "DY_STRIDE_N": int(loss["strides"][0]),
        "DY_STRIDE_C": int(loss["strides"][1]),
        "DY_STRIDE_D": loss_strides[0],
        "DY_STRIDE_H": loss_strides[1],
        "DY_STRIDE_W": loss_strides[2],
        "X_STRIDE_N": int(image["strides"][0]),
        "X_STRIDE_C": int(image["strides"][1]),
        "X_STRIDE_D": image_strides[0],
        "X_STRIDE_H": image_strides[1],
        "X_STRIDE_W": image_strides[2],
        "W_STRIDE_K": int(filter_tensor["strides"][0]),
        "W_STRIDE_C": int(filter_tensor["strides"][1]),
        "W_STRIDE_D": filter_strides[0],
        "W_STRIDE_H": filter_strides[1],
        "W_STRIDE_W": filter_strides[2],
        "INPUT_PRECISION": 1,
    }
    if operation == "convolution_fprop":
        function = "conv_fprop_nd_kernel"
        constants.update(
            {
                "Y_STRIDE_N": int(loss["strides"][0]),
                "Y_STRIDE_C": int(loss["strides"][1]),
                "Y_STRIDE_D": loss_strides[0],
                "Y_STRIDE_H": loss_strides[1],
                "Y_STRIDE_W": loss_strides[2],
            }
        )
        m = batch * math.prod(loss_spatial)
        constants["M"] = m
    elif operation == "convolution_dgrad":
        function = "conv_dgrad_nd_kernel"
        m = batch * math.prod(image_spatial)
        constants["M"] = m
    else:
        function = "conv_wgrad_nd_kernel"
        m = batch * math.prod(loss_spatial)
        constants["M"] = m
    reduction_extent = 1
    if operation == "convolution_fprop":
        reduction_extent = channels_per_group * math.prod(filter_spatial)
    elif operation == "convolution_dgrad":
        reduction_extent = outputs_per_group * math.prod(filter_spatial)
    if m > 2**31 - 1 or reduction_extent > 2**31 - 1:
        raise ValueError(
            f"THead {operation} iteration range exceeds int32"
        )
    return {
        "operation": operation,
        "tensors": tensors,
        "argument_tensors": arguments,
        "constants": constants,
        "function": function,
        "batch": batch,
        "groups": groups,
        "m": m,
        "channels_per_group": channels_per_group,
        "outputs_per_group": outputs_per_group,
        "kernel_volume": math.prod(filter_spatial),
        "n_outputs": n_outputs,
    }


def _validate_conv_bias_relu_graph(graph: dict[str, Any]) -> dict[str, Any]:
    tensors = _require_list(graph["tensors"], "graph.tensors")
    nodes = sorted(
        (_require_object(value, "ConvBiasRelu node") for value in
         _require_list(graph["nodes"], "graph.nodes")),
        key=lambda value: int(value["id"]),
    )
    if len(nodes) != 3 or len(tensors) != 6:
        raise ValueError("THead ConvBiasRelu requires three nodes and six tensors")
    convolution, add, relu = nodes
    if (
        convolution["id"] != 0
        or convolution["type"] != "convolution_fprop"
        or add["id"] != 1
        or add["type"] != "add"
        or relu["id"] != 2
        or relu["type"] != "relu"
        or any(node["compute_data_type"] != "float32" for node in nodes)
    ):
        raise ValueError(
            "THead ConvBiasRelu requires canonical float32 Conv-Add-ReLU nodes"
        )

    convolution_inputs = _named_port_uids(
        convolution, "inputs", ("input", "filter"), "ConvBiasRelu convolution"
    )
    convolution_output = _named_port_uids(
        convolution, "outputs", ("output",), "ConvBiasRelu convolution"
    )[0]
    add_inputs = _named_port_uids(
        add, "inputs", ("left", "right"), "ConvBiasRelu bias Add"
    )
    add_output = _named_port_uids(
        add, "outputs", ("output",), "ConvBiasRelu bias Add"
    )[0]
    relu_input = _named_port_uids(
        relu, "inputs", ("input",), "ConvBiasRelu ReLU"
    )[0]
    output_uid = _named_port_uids(
        relu, "outputs", ("output",), "ConvBiasRelu ReLU"
    )[0]
    if add_inputs[0] != convolution_output or relu_input != add_output:
        raise ValueError("THead ConvBiasRelu dataflow is not Conv-Add-ReLU")
    x_uid, w_uid = convolution_inputs
    bias_uid = add_inputs[1]
    if len({x_uid, w_uid, bias_uid, convolution_output, add_output,
            output_uid}) != 6:
        raise ValueError("THead ConvBiasRelu tensor roles must be distinct")

    tensors_by_uid = {int(tensor["uid"]): tensor for tensor in tensors}
    x = tensors_by_uid[x_uid]
    w = tensors_by_uid[w_uid]
    bias = tensors_by_uid[bias_uid]
    convolution_tensor = tensors_by_uid[convolution_output]
    biased_tensor = tensors_by_uid[add_output]
    output = tensors_by_uid[output_uid]
    role_tensors = [x, w, bias, convolution_tensor, biased_tensor, output]
    data_type = x["data_type"]
    if (
        data_type not in _FLOATING_DATA_TYPES
        or any(tensor["data_type"] != data_type for tensor in role_tensors)
        or any(int(tensor["alignment"]) < 16 for tensor in role_tensors)
    ):
        raise ValueError(
            "THead ConvBiasRelu requires aligned matching floating tensors"
        )
    if (
        x["virtual"]
        or w["virtual"]
        or bias["virtual"]
        or not convolution_tensor["virtual"]
        or not biased_tensor["virtual"]
        or output["virtual"]
    ):
        raise ValueError(
            "THead ConvBiasRelu requires two virtual intermediates"
        )
    if any(len(tensor["dimensions"]) != 4 for tensor in role_tensors):
        raise ValueError("THead ConvBiasRelu requires rank-four tensors")
    if any(
        tensor["dimensions"] != output["dimensions"]
        or tensor["strides"] != output["strides"]
        for tensor in (convolution_tensor, biased_tensor)
    ):
        raise ValueError(
            "THead ConvBiasRelu intermediate/output geometry must match"
        )
    output_dimensions = [int(value) for value in output["dimensions"]]
    if [int(value) for value in bias["dimensions"]] != [
        1, output_dimensions[1], 1, 1
    ]:
        raise ValueError("THead ConvBiasRelu requires a channel bias")

    def channels_last_strides(dimensions: list[Any]) -> list[int]:
        n, channels, height, width = [int(value) for value in dimensions]
        del n
        return [channels * height * width, 1, width * channels, channels]

    if any(
        [int(value) for value in tensor["strides"]]
        != channels_last_strides(tensor["dimensions"])
        for tensor in role_tensors
    ):
        raise ValueError("THead ConvBiasRelu requires channels-last strides")

    add_attributes = _require_object(
        add["attributes"], "ConvBiasRelu Add attributes"
    )
    _require_exact_fields(
        add_attributes,
        {"alpha", "mode", "n_elements", "pointwise_mode"},
        set(),
        "ConvBiasRelu Add attributes",
    )
    n_outputs = math.prod(output_dimensions)
    if (
        add_attributes["alpha"] != 1.0
        or add_attributes["mode"] != _BINARY_POINTWISE_MODES["add"]
        or add_attributes["pointwise_mode"]
        != _BINARY_POINTWISE_MODES["add"]
        or add_attributes["n_elements"] != n_outputs
    ):
        raise ValueError("THead ConvBiasRelu Add attributes are invalid")

    relu_attributes = _require_object(
        relu["attributes"], "ConvBiasRelu ReLU attributes"
    )
    expected_relu_attributes = {
        "elu_alpha": 1.0,
        "has_upper_clip": 0,
        "lower_clip": 0.0,
        "mode": _UNARY_POINTWISE_MODES["relu"],
        "n_elements": n_outputs,
        "negative_slope": 0.0,
        "relu_lower_clip": 0.0,
        "relu_lower_clip_slope": 0.0,
        "relu_upper_clip": 0.0,
        "relu_upper_clip_set": False,
        "softplus_beta": 1.0,
        "swish_beta": 1.0,
        "upper_clip": 0.0,
    }
    if relu_attributes != expected_relu_attributes:
        raise ValueError("THead ConvBiasRelu requires default ReLU attributes")

    synthetic_convolution = dict(convolution)
    synthetic_convolution["outputs"] = [{"name": "output", "uid": output_uid}]
    synthetic_tensors = []
    for tensor in (x, w, output):
        external = dict(tensor)
        external["virtual"] = False
        synthetic_tensors.append(external)
    plan = _validate_convolution_graph(
        {"tensors": synthetic_tensors, "nodes": [synthetic_convolution]},
        "convolution_fprop",
    )
    convolution_attributes = convolution["attributes"]
    if (
        convolution_attributes["groups"] != 1
        or convolution_attributes["pre_padding"]
        != convolution_attributes["post_padding"]
    ):
        raise ValueError(
            "THead ConvBiasRelu requires group one and symmetric padding"
        )
    plan.update(
        {
            "operation": "conv_bias_relu",
            "tensors": tensors,
            "argument_tensors": [x, w, bias, output],
            "virtual_tensors": [convolution_tensor, biased_tensor],
            "constants": {
                **plan["constants"],
                "BIAS_STRIDE_C": int(bias["strides"][1]),
            },
            "function": "conv2d_bias_relu_kernel",
            "source_node_ids": [0, 1, 2],
            "n_outputs": n_outputs,
        }
    )
    return plan


def _registry_sha256() -> str:
    sources = kernel_registry.iter_kernel_registry_sources("thead")
    if len(sources) != 2:
        raise ValueError("THead requires common and platform kernel registries")
    digest = hashlib.sha256()
    for source in sources:
        contents = source.read_bytes()
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def _validate_binary_pointwise_candidate(
    candidate: Any, operation: str
) -> None:
    operation_label = operation.capitalize()
    tuning = candidate.tuning
    if (
        candidate.backend != "thead"
        or candidate.operation != operation
        or candidate.provider != "common_triton"
        or candidate.ownership != "common"
        or candidate.source_layout != "kernels"
        or candidate.source_format != "module"
        or candidate.source != "binary.py"
        or candidate.functions
        != ("binary_contiguous_kernel", "binary_strided_kernel")
        or tuning is None
        or tuning.source != "common.yaml"
        or tuning.table != "binary"
        or tuning.key != "n_elements"
        or tuning.strategy != "align32"
        or tuning.warmup != 5
        or tuning.repetitions != 10
    ):
        raise ValueError(
            f"THead {operation_label} common kernel registry contract is invalid"
        )


def _validate_unary_pointwise_candidate(
    candidate: Any, operation: str
) -> None:
    operation_label = operation.capitalize()
    tuning = candidate.tuning
    if (
        candidate.backend != "thead"
        or candidate.operation != operation
        or candidate.provider != "common_triton"
        or candidate.ownership != "common"
        or candidate.source_layout != "kernels"
        or candidate.source_format != "module"
        or candidate.source != "unary.py"
        or candidate.functions
        != (
            "unary_pointwise_contiguous_kernel",
            "unary_pointwise_strided_kernel",
        )
        or tuning is None
        or tuning.source != "common.yaml"
        or tuning.table != "relu"
        or tuning.key != "n_elements"
        or tuning.strategy != "align32"
        or tuning.warmup != 5
        or tuning.repetitions != 10
    ):
        raise ValueError(
            f"THead {operation_label} common kernel registry contract is invalid"
        )


def _validate_add_square_candidate(candidate: Any) -> None:
    tuning = candidate.tuning
    if (
        candidate.backend != "thead"
        or candidate.operation != "add_square"
        or candidate.provider != "thead_triton"
        or candidate.ownership != "platform"
        or candidate.source_layout != "platform"
        or candidate.source_format != "module"
        or candidate.source != "add_square.py"
        or candidate.functions
        != ("add_square_contiguous_kernel", "add_square_strided_kernel")
        or tuning is None
        or tuning.source != "common.yaml"
        or tuning.table != "binary"
        or tuning.key != "n_elements"
        or tuning.strategy != "align32"
        or tuning.warmup != 5
        or tuning.repetitions != 10
    ):
        raise ValueError("THead AddSquare kernel registry contract is invalid")


def _validate_layout_candidate(candidate: Any, operation: str) -> None:
    tuning = candidate.tuning
    if (
        candidate.backend != "thead"
        or candidate.operation != operation
        or candidate.provider != "common_triton"
        or candidate.ownership != "common"
        or candidate.source_layout != "kernels"
        or candidate.source_format != "module"
        or candidate.source != "layout.py"
        or candidate.functions != ("layout_copy_kernel",)
        or tuning is None
        or tuning.source != "common.yaml"
        or tuning.table != "binary"
        or tuning.key != "n_elements"
        or tuning.strategy != "align32"
    ):
        raise ValueError(f"THead {operation} common kernel contract is invalid")


def _validate_reduction_candidate(candidate: Any, operation: str) -> None:
    tuning = candidate.tuning
    if (
        candidate.backend != "thead"
        or candidate.operation != operation
        or candidate.provider != "common_triton"
        or candidate.ownership != "common"
        or candidate.source_layout != "kernels"
        or candidate.source_format != "module"
        or candidate.source != "reduction.py"
        or "reduction_2d_kernel" not in candidate.functions
        or "reduction_3d_kernel" not in candidate.functions
        or "reduction_strided_kernel" not in candidate.functions
        or tuning is None
        or tuning.source != "common.yaml"
        or tuning.table != "reduction"
        or tuning.key != "output_elements"
        or tuning.strategy != "reduction"
        or tuning.warmup != 5
        or tuning.repetitions != 10
    ):
        raise ValueError(
            f"THead {operation} common reduction contract is invalid"
        )


def _validate_batchnorm_candidate(candidate: Any, operation: str) -> None:
    tuning = candidate.tuning
    expected_functions = (
        {"batch_norm_inference_nchw_kernel", "batch_norm_inference_kernel"}
        if operation == "batchnorm_inference"
        else {"batch_norm_nchw_kernel", "batch_norm_kernel"}
    )
    expected_key = "n_elements" if operation == "batchnorm_inference" else "channels"
    expected_strategy = "align32" if operation == "batchnorm_inference" else "fixed_grid"
    expected_provider = "thead_triton" if operation == "batchnorm_inference" else "common_triton"
    expected_ownership = "platform" if operation == "batchnorm_inference" else "common"
    expected_layout = "platform" if operation == "batchnorm_inference" else "kernels"
    if (
        candidate.backend != "thead"
        or candidate.operation != operation
        or candidate.provider != expected_provider
        or candidate.ownership != expected_ownership
        or candidate.source_layout != expected_layout
        or candidate.source_format != "module"
        or candidate.source != "normalization.py"
        or not expected_functions.issubset(set(candidate.functions))
        or tuning is None
        or tuning.source != "common.yaml"
        or tuning.table != "batch_norm"
        or tuning.key != expected_key
        or tuning.strategy != expected_strategy
        or tuning.warmup != 5
        or tuning.repetitions != 10
    ):
        raise ValueError(
            f"THead {operation} common normalization contract is invalid"
        )


def _validate_normalization_candidate(candidate: Any, operation: str) -> None:
    tuning = candidate.tuning
    expected_function = (
        "layer_norm_kernel" if operation == "layernorm" else "rms_norm_kernel"
    )
    expected_table = "layer_norm" if operation == "layernorm" else "rms_norm"
    if (
        candidate.backend != "thead"
        or candidate.operation != operation
        or candidate.provider != "common_triton"
        or candidate.ownership != "common"
        or candidate.source_layout != "kernels"
        or candidate.source_format != "module"
        or candidate.source != "normalization.py"
        or candidate.functions != (expected_function,)
        or tuning is None
        or tuning.source != "common.yaml"
        or tuning.table != expected_table
        or tuning.key != "normalized_elements"
        or tuning.strategy != "fixed_grid"
        or tuning.warmup != 5
        or tuning.repetitions != 10
    ):
        raise ValueError(
            f"THead {operation} common normalization contract is invalid"
        )


def _validate_matmul_candidate(candidate: Any) -> None:
    tuning = candidate.tuning
    if (
        candidate.backend != "thead"
        or candidate.operation != "matmul"
        or candidate.provider != "common_triton"
        or candidate.ownership != "common"
        or candidate.source_layout != "kernels"
        or candidate.source_format != "module"
        or candidate.source != "matmul.py"
        or candidate.functions != ("matmul_strided_kernel",)
        or tuning is None
        or tuning.source != "common.yaml"
        or tuning.table != "matmul"
        or tuning.key != "m"
        or tuning.strategy != "matmul"
        or tuning.warmup != 5
        or tuning.repetitions != 10
    ):
        raise ValueError("THead MatMul common kernel contract is invalid")


def _validate_convolution_candidate(candidate: Any, operation: str) -> None:
    tuning = candidate.tuning
    fprop = operation == "convolution_fprop"
    fused = operation == "conv_bias_relu"
    expected_function = (
        "conv2d_bias_relu_kernel"
        if fused
        else (
            "conv_fprop_nd_kernel"
            if fprop
            else (
                "conv_dgrad_nd_kernel"
                if operation == "convolution_dgrad"
                else "conv_wgrad_nd_kernel"
            )
        )
    )
    if (
        candidate.backend != "thead"
        or candidate.operation != operation
        or candidate.provider
        != ("thead_triton" if fprop or fused else "common_triton")
        or candidate.ownership != ("platform" if fprop or fused else "common")
        or candidate.source_layout != ("platform" if fprop or fused else "kernels")
        or candidate.source_format != "module"
        or candidate.source != "convolution.py"
        or expected_function not in candidate.functions
        or tuning is None
        or tuning.source != "common.yaml"
        or tuning.table != "conv2d_spatial"
        or tuning.key != "n_outputs"
        or tuning.strategy != "convolution"
        or tuning.warmup != 5
        or tuning.repetitions != 10
    ):
        raise ValueError(
            f"THead {operation} kernel registry contract is invalid"
        )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(encoded)


def _validate_ppu_option_value(value: object, description: str) -> None:
    if isinstance(value, bool) or isinstance(value, str):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    raise ValueError(f"{description} must be a boolean, integer, or string")


def _load_pointwise_tuning(
    tuning_path: Path,
    candidate: Any,
    operation: str,
) -> tuple[list[dict[str, Any]], bytes]:
    operation_label = operation.capitalize()
    tuning_bytes = tuning_path.read_bytes()
    if not tuning_bytes:
        raise ValueError(f"THead {operation_label} tuning source is empty")
    try:
        document = yaml.safe_load(tuning_bytes)
    except yaml.YAMLError as error:
        raise ValueError(
            f"THead {operation_label} tuning source is invalid: {error}"
        ) from error
    table = candidate.tuning.table
    if not isinstance(document, dict) or set(document) != _TUNING_TABLES:
        raise ValueError(
            f"THead {operation_label} tuning source has an invalid table set"
        )
    entries = document[table]
    if not isinstance(entries, list) or len(entries) != 2:
        raise ValueError(
            f"THead {operation_label} autotune requires exactly two candidates"
        )

    configurations: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict):
            raise ValueError(
                f"THead {operation_label} tuning candidate {index} "
                "must be a mapping"
            )
        _require_exact_fields(
            raw_entry,
            {
                "META",
                "num_warps",
                "num_stages",
                "maxnreg",
                "ppu_compiler_options",
            },
            set(),
            f"THead {operation_label} tuning candidate {index}",
        )
        meta = _require_object(
            raw_entry["META"], f"THead {operation_label} tuning META"
        )
        _require_exact_fields(
            meta,
            {"BLOCK_SIZE"},
            set(),
            f"THead {operation_label} tuning META",
        )
        block_size = _integer(
            meta["BLOCK_SIZE"], f"THead {operation_label} BLOCK_SIZE"
        )
        if (
            block_size < 32
            or block_size > 65536
            or block_size & (block_size - 1)
        ):
            raise ValueError(
                f"THead {operation_label} BLOCK_SIZE must be a power of two"
            )
        num_warps = _integer(
            raw_entry["num_warps"],
            f"THead {operation_label} tuning num_warps",
        )
        if (
            num_warps <= 0
            or num_warps > 32
            or num_warps & (num_warps - 1)
        ):
            raise ValueError(
                f"THead {operation_label} tuning num_warps is invalid"
            )
        num_stages = _integer(
            raw_entry["num_stages"],
            f"THead {operation_label} tuning num_stages",
        )
        if not 1 <= num_stages <= 16:
            raise ValueError(
                f"THead {operation_label} tuning num_stages is invalid"
            )

        maxnreg = raw_entry["maxnreg"]
        if maxnreg is not None:
            maxnreg = _integer(
                maxnreg, f"THead {operation_label} tuning maxnreg"
            )
            if not 1 <= maxnreg <= 2**31 - 1:
                raise ValueError(
                    f"THead {operation_label} tuning maxnreg is invalid"
                )

        raw_options = _require_object(
            raw_entry["ppu_compiler_options"],
            f"THead {operation_label} ppu_compiler_options",
        )
        unknown_options = set(raw_options).difference(
            _APPROVED_PPU_COMPILER_OPTIONS
        )
        if unknown_options:
            raise ValueError(
                f"THead {operation_label} tuning has unapproved PPU "
                "compiler options: "
                + ", ".join(sorted(unknown_options))
            )
        ppu_options: dict[str, object] = {}
        for name in sorted(raw_options):
            _validate_ppu_option_value(
                raw_options[name], f"THead PPU compiler option {name}"
            )
            ppu_options[name] = raw_options[name]

        # CUDA-backend libtriton_jit's dynamic raw-argument entry point only
        # accepts num_warps and num_stages.  Keep all PPU-specific switches in
        # the schema and candidate identity, but reject values that the current
        # ABI could otherwise silently ignore.
        if maxnreg is not None or ppu_options:
            raise ValueError(
                "THead raw-argument libtriton_jit cannot convey maxnreg or "
                "PPU compiler options"
            )

        configuration = {
            "META": {"BLOCK_SIZE": block_size},
            "maxnreg": maxnreg,
            "num_stages": num_stages,
            "num_warps": num_warps,
            "ppu_compiler_options": ppu_options,
        }
        encoded = json.dumps(
            configuration,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if encoded in seen:
            raise ValueError(
                f"THead {operation_label} tuning candidates must be unique"
            )
        seen.add(encoded)
        configurations.append(configuration)
    return configurations, tuning_bytes


def _load_normalization_tuning(
    tuning_path: Path, candidate: Any, operation: str
) -> tuple[list[dict[str, Any]], bytes]:
    tuning_bytes = tuning_path.read_bytes()
    if not tuning_bytes:
        raise ValueError(f"THead {operation} tuning source is empty")
    try:
        document = yaml.safe_load(tuning_bytes)
    except yaml.YAMLError as error:
        raise ValueError(
            f"THead {operation} tuning source is invalid: {error}"
        ) from error
    if not isinstance(document, dict) or set(document) != _TUNING_TABLES:
        raise ValueError(
            f"THead {operation} tuning source has an invalid table set"
        )
    entries = document.get(candidate.tuning.table)
    if not isinstance(entries, list) or len(entries) != 2:
        raise ValueError(
            f"THead {operation} autotune requires exactly two candidates"
        )
    configurations: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for index, raw_entry in enumerate(entries):
        entry = _require_object(
            raw_entry, f"THead {operation} tuning candidate {index}"
        )
        _require_exact_fields(
            entry,
            {
                "META",
                "num_warps",
                "num_stages",
                "maxnreg",
                "ppu_compiler_options",
            },
            set(),
            f"THead {operation} tuning candidate {index}",
        )
        meta = _require_object(
            entry["META"], f"THead {operation} tuning META"
        )
        _require_exact_fields(
            meta,
            {"BLOCK_SIZE", "ROWS_PER_PROGRAM"},
            set(),
            f"THead {operation} tuning META",
        )
        block_size = _integer(
            meta["BLOCK_SIZE"], f"THead {operation} BLOCK_SIZE"
        )
        rows_per_program = _integer(
            meta["ROWS_PER_PROGRAM"],
            f"THead {operation} ROWS_PER_PROGRAM",
        )
        num_warps = _integer(
            entry["num_warps"], f"THead {operation} num_warps"
        )
        num_stages = _integer(
            entry["num_stages"], f"THead {operation} num_stages"
        )
        options = _require_object(
            entry["ppu_compiler_options"],
            f"THead {operation} ppu_compiler_options",
        )
        if (
            block_size not in {128, 256}
            or rows_per_program != 1
            or num_warps != 4
            or num_stages != 1
            or entry["maxnreg"] is not None
            or options
        ):
            raise ValueError(
                f"THead {operation} tuning is outside the qualified PPU slice"
            )
        configuration = {
            "META": {
                "BLOCK_SIZE": block_size,
                "ROWS_PER_PROGRAM": rows_per_program,
            },
            "maxnreg": None,
            "num_stages": num_stages,
            "num_warps": num_warps,
            "ppu_compiler_options": {},
        }
        encoded = json.dumps(
            configuration, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if encoded in seen:
            raise ValueError(
                f"THead {operation} tuning candidates repeat"
            )
        seen.add(encoded)
        configurations.append(configuration)
    return configurations, tuning_bytes


def _load_reduction_tuning(
    tuning_path: Path, candidate: Any, operation: str
) -> tuple[list[dict[str, Any]], bytes]:
    tuning_bytes = tuning_path.read_bytes()
    if not tuning_bytes:
        raise ValueError("THead reduction tuning source is empty")
    try:
        document = yaml.safe_load(tuning_bytes)
    except yaml.YAMLError as error:
        raise ValueError(f"THead reduction tuning is invalid: {error}") from error
    if not isinstance(document, dict) or set(document) != _TUNING_TABLES:
        raise ValueError("THead reduction tuning table set is invalid")
    entries = document.get(candidate.tuning.table)
    if not isinstance(entries, list) or len(entries) != 2:
        raise ValueError("THead reduction autotune requires two candidates")
    configurations: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for index, raw_entry in enumerate(entries):
        entry = _require_object(
            raw_entry, f"THead {operation} tuning candidate {index}"
        )
        _require_exact_fields(
            entry,
            {
                "META",
                "num_warps",
                "num_stages",
                "maxnreg",
                "ppu_compiler_options",
            },
            set(),
            f"THead {operation} tuning candidate {index}",
        )
        meta = _require_object(entry["META"], "THead reduction tuning META")
        _require_exact_fields(
            meta,
            {"BLOCK_M", "BLOCK_N"},
            set(),
            "THead reduction tuning META",
        )
        block_m = _integer(meta["BLOCK_M"], "THead reduction BLOCK_M")
        block_n = _integer(meta["BLOCK_N"], "THead reduction BLOCK_N")
        if any(
            value < 1 or value > 65536 or value & (value - 1)
            for value in (block_m, block_n)
        ):
            raise ValueError("THead reduction blocks must be powers of two")
        num_warps = _integer(entry["num_warps"], "THead reduction num_warps")
        num_stages = _integer(
            entry["num_stages"], "THead reduction num_stages"
        )
        if (
            num_warps < 1
            or num_warps > 32
            or num_warps & (num_warps - 1)
            or not 1 <= num_stages <= 16
            or entry["maxnreg"] is not None
            or _require_object(
                entry["ppu_compiler_options"],
                "THead reduction ppu_compiler_options",
            )
        ):
            raise ValueError("THead reduction tuning options are invalid")
        configuration = {
            "META": {"BLOCK_M": block_m, "BLOCK_N": block_n},
            "maxnreg": None,
            "num_stages": num_stages,
            "num_warps": num_warps,
            "ppu_compiler_options": {},
        }
        encoded = json.dumps(
            configuration, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if encoded in seen:
            raise ValueError("THead reduction tuning candidates repeat")
        seen.add(encoded)
        configurations.append(configuration)
    return configurations, tuning_bytes


def _load_matmul_tuning(
    tuning_path: Path, candidate: Any
) -> tuple[list[dict[str, Any]], bytes]:
    tuning_bytes = tuning_path.read_bytes()
    if not tuning_bytes:
        raise ValueError("THead MatMul tuning source is empty")
    try:
        document = yaml.safe_load(tuning_bytes)
    except yaml.YAMLError as error:
        raise ValueError(f"THead MatMul tuning is invalid: {error}") from error
    if not isinstance(document, dict) or set(document) != _TUNING_TABLES:
        raise ValueError("THead MatMul tuning table set is invalid")
    entries = document.get(candidate.tuning.table)
    if not isinstance(entries, list) or len(entries) != 2:
        raise ValueError("THead MatMul autotune requires two candidates")
    configurations: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for index, raw_entry in enumerate(entries):
        entry = _require_object(
            raw_entry, f"THead MatMul tuning candidate {index}"
        )
        _require_exact_fields(
            entry,
            {
                "META",
                "num_warps",
                "num_stages",
                "maxnreg",
                "ppu_compiler_options",
            },
            set(),
            f"THead MatMul tuning candidate {index}",
        )
        meta = _require_object(entry["META"], "THead MatMul tuning META")
        _require_exact_fields(
            meta,
            {"BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M"},
            set(),
            "THead MatMul tuning META",
        )
        block_m = _integer(meta["BLOCK_M"], "THead MatMul BLOCK_M")
        block_n = _integer(meta["BLOCK_N"], "THead MatMul BLOCK_N")
        block_k = _integer(meta["BLOCK_K"], "THead MatMul BLOCK_K")
        group_m = _integer(meta["GROUP_M"], "THead MatMul GROUP_M")
        if any(
            value < 1 or value > 256 or value & (value - 1)
            for value in (block_m, block_n, block_k, group_m)
        ):
            raise ValueError(
                "THead MatMul blocks and group must be bounded powers of two"
            )
        num_warps = _integer(entry["num_warps"], "THead MatMul num_warps")
        num_stages = _integer(
            entry["num_stages"], "THead MatMul num_stages"
        )
        options = _require_object(
            entry["ppu_compiler_options"],
            "THead MatMul ppu_compiler_options",
        )
        if (
            num_warps < 1
            or num_warps > 32
            or num_warps & (num_warps - 1)
            or not 1 <= num_stages <= 16
            or entry["maxnreg"] is not None
            or options
        ):
            raise ValueError("THead MatMul tuning options are invalid")
        configuration = {
            "META": {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "GROUP_M": group_m,
            },
            "maxnreg": None,
            "num_stages": num_stages,
            "num_warps": num_warps,
            "ppu_compiler_options": {},
        }
        encoded = json.dumps(
            configuration, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if encoded in seen:
            raise ValueError("THead MatMul tuning candidates repeat")
        seen.add(encoded)
        configurations.append(configuration)
    return configurations, tuning_bytes


def _load_convolution_tuning(
    tuning_path: Path, candidate: Any, operation: str
) -> tuple[list[dict[str, Any]], bytes]:
    tuning_bytes = tuning_path.read_bytes()
    if not tuning_bytes:
        raise ValueError("THead convolution tuning source is empty")
    try:
        document = yaml.safe_load(tuning_bytes)
    except yaml.YAMLError as error:
        raise ValueError(
            f"THead convolution tuning is invalid: {error}"
        ) from error
    if not isinstance(document, dict) or set(document) != _TUNING_TABLES:
        raise ValueError("THead convolution tuning table set is invalid")
    entries = document.get(candidate.tuning.table)
    if not isinstance(entries, list) or len(entries) != 2:
        raise ValueError("THead convolution autotune requires two candidates")
    configurations: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    meta_names = {
        "BLOCK_M",
        "BLOCK_OC",
        "BLOCK_CI",
        "BLOCK_K",
        "BLOCK_HW",
    }
    for index, raw_entry in enumerate(entries):
        entry = _require_object(
            raw_entry, f"THead {operation} tuning candidate {index}"
        )
        _require_exact_fields(
            entry,
            {
                "META",
                "num_warps",
                "num_stages",
                "maxnreg",
                "ppu_compiler_options",
            },
            set(),
            f"THead {operation} tuning candidate {index}",
        )
        meta = _require_object(entry["META"], "THead convolution tuning META")
        _require_exact_fields(
            meta, meta_names, set(), "THead convolution tuning META"
        )
        parsed_meta = {
            name: _integer(meta[name], f"THead convolution {name}")
            for name in sorted(meta_names)
        }
        if any(
            value < 8 or value > 64 or value & (value - 1)
            for value in parsed_meta.values()
        ):
            raise ValueError(
                "THead convolution tiles must be bounded powers of two"
            )
        num_warps = _integer(
            entry["num_warps"], "THead convolution num_warps"
        )
        num_stages = _integer(
            entry["num_stages"], "THead convolution num_stages"
        )
        options = _require_object(
            entry["ppu_compiler_options"],
            "THead convolution ppu_compiler_options",
        )
        if (
            num_warps < 1
            or num_warps > 32
            or num_warps & (num_warps - 1)
            or not 1 <= num_stages <= 16
            or entry["maxnreg"] is not None
            or options
        ):
            raise ValueError("THead convolution tuning options are invalid")
        configuration = {
            "META": parsed_meta,
            "maxnreg": None,
            "num_stages": num_stages,
            "num_warps": num_warps,
            "ppu_compiler_options": {},
        }
        encoded = json.dumps(
            configuration, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if encoded in seen:
            raise ValueError("THead convolution tuning candidates repeat")
        seen.add(encoded)
        configurations.append(configuration)
    return configurations, tuning_bytes


def _binary_pointwise_variant(
    argument_tensors: list[dict[str, Any]],
    n_elements: int,
    pointwise_mode: int,
    alpha: float,
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
            ",".join(_tensor_pointer_signature(tensor)
                     for tensor in argument_tensors)
            + ",i32"
            + (
                ""
                if stride_constants is None
                else "," + ",".join(str(value) for value in stride_constants)
            )
            + f",{pointwise_mode},{alpha},{block_size}"
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
            ",".join(_tensor_pointer_signature(tensor)
                     for tensor in argument_tensors)
            + ",i32"
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
    ordered_constants.extend(constants[f"INPUT_DIM_{axis}"] for axis in range(8))
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
                _tensor_pointer_signature(tensor)
                for tensor in argument_tensors
            )
            + ",i32,"
            + ",".join(str(value) for value in ordered_constants)
            + f",{block_size}"
        ),
        "argument_count": 3,
        "arguments": [
            *[_tensor_argument(tensor) for tensor in argument_tensors],
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
            4
            if plan["argument_tensors"][0]["data_type"] == "float32"
            else 2
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
    return {
        "variant_id": variant_id,
        "full_signature": (
            pointers
            + ",i32,"
            + ",".join(str(value) for value in constants)
        ),
        "argument_count": len(plan["argument_tensors"]) + 1,
        "arguments": [
            *[
                _tensor_argument(tensor)
                for tensor in plan["argument_tensors"]
            ],
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
                (int(plan["rows"]) + rows_per_program - 1)
                // rows_per_program,
                1,
                1,
            ],
            "block": [num_warps * _PPU_WARP_SIZE, 1, 1],
            # CUDA-backend metadata is compiler-specialization dependent for
            # this kernel.  The qualified N=17 and N=513 specializations use
            # one scalar scratch slot per BLOCK_SIZE lane; the larger exact
            # benchmark suffixes use the 16-byte warp-partial path.
            "shared_memory": (
                block_size
                * (
                    4
                    if plan["argument_tensors"][0]["data_type"] == "float32"
                    else 2
                )
                if int(plan["normalized_elements"]) in {17, 513}
                and block_size == 256
                else 16
            ),
        },
    }


def _batchnorm_variant(
    plan: dict[str, Any],
    configuration: dict[str, Any],
    variant_id: str,
) -> dict[str, Any]:
    block_size = int(configuration["META"]["BLOCK_SIZE"])
    num_warps = int(configuration["num_warps"])
    inference_nchw = plan["function"] == "batch_norm_inference_nchw_kernel"
    inference_strided = plan["function"] == "batch_norm_inference_kernel"
    training_strided = plan["function"] == "batch_norm_kernel"
    scalar_arguments: list[dict[str, Any]] = []
    if inference_nchw:
        constants: list[float | int] = [
            plan["channels"],
            plan["spatial"],
            0.0,
            block_size,
            1,
            1,
            1,
        ]
        programs = (
            int(plan["batch"])
            * int(plan["channels"])
            * ((int(plan["spatial"]) + block_size - 1) // block_size)
        )
        shared_memory = 0
    elif inference_strided:
        constants = [
            0.0,
            block_size,
            1,
            1,
            1,
            1,
            *plan["stride_constants"],
        ]
        scalar_arguments = [
            {
                "kind": "scalar_i32",
                "name": "total_elements",
                "value": int(plan["n_elements"]),
            },
            {
                "kind": "scalar_i32",
                "name": "channels",
                "value": int(plan["channels"]),
            },
            {
                "kind": "scalar_i32",
                "name": "spatial",
                "value": int(plan["spatial"]),
            },
        ]
        programs = (int(plan["n_elements"]) + block_size - 1) // block_size
        shared_memory = 0
    else:
        constants = [
            plan["batch"],
            plan["channels"],
            plan["spatial"],
            plan["epsilon"],
            plan["momentum"],
            block_size,
            1,
            1,
            1,
            1,
            1,
        ]
        programs = int(plan["channels"])
        shared_memory = 16
        if training_strided:
            constants = [
                plan["epsilon"],
                plan["momentum"],
                block_size,
                1,
                1,
                1,
                1,
                1,
                1,
                *plan["stride_constants"],
            ]
            scalar_arguments = [
                {
                    "kind": "scalar_i32",
                    "name": "batch",
                    "value": int(plan["batch"]),
                },
                {
                    "kind": "scalar_i32",
                    "name": "channels",
                    "value": int(plan["channels"]),
                },
                {
                    "kind": "scalar_i32",
                    "name": "spatial",
                    "value": int(plan["spatial"]),
                },
            ]
    return {
        "variant_id": variant_id,
        "full_signature": (
            ",".join(
                _tensor_pointer_signature(tensor)
                for tensor in plan["argument_tensors"]
            )
            + (",i32,i32,i32" if scalar_arguments else "")
            + ","
            + ",".join(str(value) for value in constants)
        ),
        "argument_count": len(plan["argument_tensors"]) + len(scalar_arguments),
        "arguments": [
            *[
                _tensor_argument(tensor)
                for tensor in plan["argument_tensors"]
            ],
            *scalar_arguments,
        ],
        "compile_options": {
            "num_warps": num_warps,
            "num_stages": int(configuration["num_stages"]),
            "maxnreg": configuration["maxnreg"],
            "ppu_compiler_options": configuration["ppu_compiler_options"],
        },
        "launch": {
            "grid": [programs, 1, 1],
            "block": [num_warps * _PPU_WARP_SIZE, 1, 1],
            "shared_memory": shared_memory,
        },
    }


def _matmul_variant(
    plan: dict[str, Any],
    configuration: dict[str, Any],
    variant_id: str,
) -> dict[str, Any]:
    meta = configuration["META"]
    block_m = int(meta["BLOCK_M"])
    block_n = int(meta["BLOCK_N"])
    block_k = int(meta["BLOCK_K"])
    group_m = int(meta["GROUP_M"])
    constants = plan["constants"]
    ordered_constants = [
        constants["M"],
        constants["N"],
        constants["K"],
        *[constants[f"DIM_{axis}"] for axis in range(6)],
        *[constants[f"A_BATCH_STRIDE_{axis}"] for axis in range(6)],
        *[constants[f"B_BATCH_STRIDE_{axis}"] for axis in range(6)],
        *[constants[f"C_BATCH_STRIDE_{axis}"] for axis in range(6)],
        constants["A_STRIDE_M"],
        constants["A_STRIDE_K"],
        constants["B_STRIDE_K"],
        constants["B_STRIDE_N"],
        constants["C_STRIDE_M"],
        constants["C_STRIDE_N"],
        constants["INPUT_IS_FLOAT32"],
        constants["USE_TF32"],
        block_m,
        block_n,
        block_k,
        group_m,
    ]
    num_warps = int(configuration["num_warps"])
    input_data_type = plan["argument_tensors"][0]["data_type"]
    # The PPU CUDA-compatible Triton lowering widens the staged operands of
    # the 128x128 tile to four-byte shared-memory lanes even for FP16/BF16.
    # Smaller qualified tiles retain packed two-byte lanes.  This value is
    # part of the launch ABI and must match the compiled cubin metadata.
    shared_scalar_bytes = (
        4
        if input_data_type == "float32" or block_m >= 128 or block_n >= 128
        else 2
    )
    return {
        "variant_id": variant_id,
        "full_signature": (
            ",".join(
                _tensor_pointer_signature(tensor)
                for tensor in plan["argument_tensors"]
            )
            + ","
            + ",".join(str(value) for value in ordered_constants)
        ),
        "argument_count": 3,
        "arguments": [
            _tensor_argument(tensor) for tensor in plan["argument_tensors"]
        ],
        "compile_options": {
            "num_warps": num_warps,
            "num_stages": int(configuration["num_stages"]),
            "maxnreg": configuration["maxnreg"],
            "ppu_compiler_options": configuration["ppu_compiler_options"],
        },
        "launch": {
            "grid": [
                ((int(plan["m"]) + block_m - 1) // block_m)
                * ((int(plan["n"]) + block_n - 1) // block_n),
                int(plan["batch"]),
                1,
            ],
            "block": [num_warps * _PPU_WARP_SIZE, 1, 1],
            # Triton CUDA-backend lowering stages both operands in dynamic
            # shared memory for this IEEE dot-product tile.
            "shared_memory": (
                block_m * block_k + block_k * block_n
            ) * shared_scalar_bytes,
        },
    }


def _convolution_variant(
    plan: dict[str, Any],
    configuration: dict[str, Any],
    variant_id: str,
) -> dict[str, Any]:
    constants = plan["constants"]
    meta = configuration["META"]
    function = plan["function"]
    operand_bytes = (
        4
        if plan["argument_tensors"][0]["data_type"] == "float32"
        else 2
    )
    if function == "conv_fprop_nd_kernel":
        block_m = int(meta["BLOCK_M"])
        block_n = int(meta["BLOCK_OC"])
        block_k = int(meta["BLOCK_K"])
        ordered_constants = [
            *[
                constants[name]
                for name in (
                    "XD",
                    "XH",
                    "XW",
                    "OD",
                    "OH",
                    "OW",
                    "KD",
                    "KH",
                    "KW",
                    "CIN_PER_GROUP",
                    "COUT_PER_GROUP",
                    "GROUPS",
                    "STRIDE_D",
                    "STRIDE_H",
                    "STRIDE_W",
                    "PAD_FRONT",
                    "PAD_TOP",
                    "PAD_LEFT",
                    "DIL_D",
                    "DIL_H",
                    "DIL_W",
                    "X_STRIDE_N",
                    "X_STRIDE_C",
                    "X_STRIDE_D",
                    "X_STRIDE_H",
                    "X_STRIDE_W",
                    "W_STRIDE_K",
                    "W_STRIDE_C",
                    "W_STRIDE_D",
                    "W_STRIDE_H",
                    "W_STRIDE_W",
                    "Y_STRIDE_N",
                    "Y_STRIDE_C",
                    "Y_STRIDE_D",
                    "Y_STRIDE_H",
                    "Y_STRIDE_W",
                    "INPUT_PRECISION",
                    "M",
                )
            ],
            block_m,
            block_n,
            block_k,
        ]
        grid = [
            ((plan["m"] + block_m - 1) // block_m)
            * (
                (plan["outputs_per_group"] + block_n - 1)
                // block_n
            ),
            plan["groups"],
            1,
        ]
        shared_memory = (
            block_m * block_k + block_k * block_n
        ) * operand_bytes
    elif function == "conv2d_bias_relu_kernel":
        block_m = int(meta["BLOCK_HW"])
        block_n = int(meta["BLOCK_OC"])
        block_k = int(meta["BLOCK_K"])
        ordered_constants = [
            constants["XH"],
            constants["XW"],
            constants["OH"],
            constants["OW"],
            constants["CIN_PER_GROUP"],
            constants["COUT_PER_GROUP"],
            constants["GROUPS"],
            constants["STRIDE_H"],
            constants["STRIDE_W"],
            constants["PAD_TOP"],
            constants["PAD_LEFT"],
            constants["DIL_H"],
            constants["DIL_W"],
            constants["KH"],
            constants["KW"],
            constants["X_STRIDE_N"],
            constants["X_STRIDE_C"],
            constants["X_STRIDE_H"],
            constants["X_STRIDE_W"],
            constants["W_STRIDE_K"],
            constants["W_STRIDE_C"],
            constants["W_STRIDE_H"],
            constants["W_STRIDE_W"],
            constants["BIAS_STRIDE_C"],
            constants["Y_STRIDE_N"],
            constants["Y_STRIDE_C"],
            constants["Y_STRIDE_H"],
            constants["Y_STRIDE_W"],
            block_n,
            block_m,
            block_k,
            constants["INPUT_PRECISION"],
        ]
        grid = [
            ((constants["OH"] * constants["OW"] + block_m - 1) // block_m)
            * (
                (plan["outputs_per_group"] + block_n - 1)
                // block_n
            ),
            plan["batch"] * plan["groups"],
            1,
        ]
        shared_memory = (
            block_m * block_k + block_k * block_n
        ) * operand_bytes
    elif function == "conv_dgrad_nd_kernel":
        block_m = int(meta["BLOCK_M"])
        block_n = int(meta["BLOCK_CI"])
        block_k = int(meta["BLOCK_K"])
        ordered_constants = [
            *[constants[name] for name in (
                "XD", "XH", "XW", "OD", "OH", "OW", "KD", "KH", "KW",
                "CIN_PER_GROUP", "COUT_PER_GROUP", "STRIDE_D", "STRIDE_H",
                "STRIDE_W", "PAD_FRONT", "PAD_TOP", "PAD_LEFT", "DIL_D",
                "DIL_H", "DIL_W", "FLIP_FILTER", "DY_STRIDE_N",
                "DY_STRIDE_C", "DY_STRIDE_D", "DY_STRIDE_H", "DY_STRIDE_W",
                "X_STRIDE_N", "X_STRIDE_C", "X_STRIDE_D", "X_STRIDE_H",
                "X_STRIDE_W", "W_STRIDE_K", "W_STRIDE_C", "W_STRIDE_D",
                "W_STRIDE_H", "W_STRIDE_W", "INPUT_PRECISION", "M",
            )],
            block_m,
            block_n,
            block_k,
            8,
        ]
        grid = [
            ((plan["m"] + block_m - 1) // block_m)
            * (
                (plan["channels_per_group"] + block_n - 1)
                // block_n
            ),
            plan["groups"],
            1,
        ]
        shared_memory = (
            block_m * block_k + block_k * block_n
        ) * operand_bytes
    else:
        block_m = int(meta["BLOCK_M"])
        block_n = int(meta["BLOCK_OC"])
        block_k = int(meta["BLOCK_CI"])
        ordered_constants = [
            *[constants[name] for name in (
                "XD", "XH", "XW", "OD", "OH", "OW", "KD", "KH", "KW",
                "CIN_PER_GROUP", "COUT_PER_GROUP", "STRIDE_D", "STRIDE_H",
                "STRIDE_W", "PAD_FRONT", "PAD_TOP", "PAD_LEFT", "DIL_D",
                "DIL_H", "DIL_W", "FLIP_FILTER", "DY_STRIDE_N",
                "DY_STRIDE_C", "DY_STRIDE_D", "DY_STRIDE_H", "DY_STRIDE_W",
                "X_STRIDE_N", "X_STRIDE_C", "X_STRIDE_D", "X_STRIDE_H",
                "X_STRIDE_W", "W_STRIDE_K", "W_STRIDE_C", "W_STRIDE_D",
                "W_STRIDE_H", "W_STRIDE_W", "INPUT_PRECISION", "M",
            )],
            block_n,
            block_k,
            block_m,
        ]
        grid = [
            (
                (plan["outputs_per_group"] + block_n - 1)
                // block_n
            )
            * (
                (plan["channels_per_group"] + block_k - 1)
                // block_k
            ),
            plan["kernel_volume"],
            plan["groups"],
        ]
        # WGrad multiplies [BLOCK_OC, BLOCK_M] by
        # [BLOCK_M, BLOCK_CI].  Keep the launch ABI tied to those actual
        # operand tiles instead of assuming all tuning dimensions are equal.
        shared_memory = (
            block_n * block_m + block_m * block_k
        ) * operand_bytes
    num_warps = int(configuration["num_warps"])
    return {
        "variant_id": variant_id,
        "full_signature": (
            ",".join(
                _tensor_pointer_signature(tensor)
                for tensor in plan["argument_tensors"]
            )
            + ","
            + ",".join(str(value) for value in ordered_constants)
        ),
        "argument_count": len(plan["argument_tensors"]),
        "arguments": [
            _tensor_argument(tensor) for tensor in plan["argument_tensors"]
        ],
        "compile_options": {
            "num_warps": num_warps,
            "num_stages": int(configuration["num_stages"]),
            "maxnreg": configuration["maxnreg"],
            "ppu_compiler_options": configuration["ppu_compiler_options"],
        },
        "launch": {
            "grid": grid,
            "block": [num_warps * _PPU_WARP_SIZE, 1, 1],
            "shared_memory": shared_memory,
        },
    }


def _tensor_pointer_signature(tensor: dict[str, Any]) -> str:
    scalar = {
        "float32": "fp32",
        "float16": "fp16",
        "bfloat16": "bf16",
        "boolean": "i8",
        "fp8_e4m3": "fp8e4nv",
        "fp8_e5m2": "fp8e5",
    }[str(tensor["data_type"])]
    # In libtriton_jit's CUDA-compatible signature syntax ``:1`` means
    # value-equals-one specialization rather than one-byte alignment.  Leave
    # sub-16-byte pointers unspecialized and cap stronger guarantees at the
    # only supported divisibility hint.
    if int(tensor["alignment"]) < 16:
        return f"*{scalar}"
    return f"*{scalar}:16"


def _tensor_storage_size(tensor: dict[str, Any]) -> int:
    element_size = _DATA_TYPE_BYTES[str(tensor["data_type"])]
    elements = 1 + sum(
        (int(dimension) - 1) * int(stride)
        for dimension, stride in zip(
            tensor["dimensions"], tensor["strides"], strict=True
        )
    )
    return elements * element_size


def _manifest_tensor(tensor: dict[str, Any]) -> dict[str, Any]:
    return {
        "uid": tensor["uid"],
        "data_type": tensor["data_type"],
        "dimensions": tensor["dimensions"],
        "strides": tensor["strides"],
        "alignment": tensor["alignment"],
        "virtual": tensor["virtual"],
        "storage_size": _tensor_storage_size(tensor),
    }


def _tensor_argument(tensor: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "tensor",
        "uid": tensor["uid"],
        "size": _tensor_storage_size(tensor),
        "alignment": tensor["alignment"],
    }


def _compile_add_square(
    *,
    request: dict[str, Any],
    request_bytes: bytes,
    identity: dict[str, Any],
    target: str,
    output_directory: Path,
    enable_autotune: bool,
) -> dict[str, Any]:
    plan = _validate_add_square_graph(request["graph"])
    candidate = kernel_registry.select_kernel_candidate("thead", "add_square")
    _validate_add_square_candidate(candidate)
    compiler_path = Path(__file__)
    source_path = kernel_registry.resolve_kernel_source(compiler_path, candidate)
    source_bytes = kernel_registry.materialize_kernel_source(
        source_path, candidate
    )
    if not source_bytes or len(source_bytes) > _MAX_KERNEL_SOURCE_BYTES:
        raise ValueError("THead AddSquare kernel source size is invalid")
    tuning_path = kernel_registry.resolve_tuning_source(compiler_path, candidate)
    if not tuning_path.is_file():
        raise ValueError("THead AddSquare tuning source is missing")
    registry_digest = _registry_sha256()
    n_elements = int(plan["n_elements"])
    default_configuration = {
        "META": {"BLOCK_SIZE": _pointwise_block_size(n_elements)},
        "maxnreg": None,
        "num_stages": _FIXED_NUM_STAGES,
        "num_warps": _FIXED_NUM_WARPS,
        "ppu_compiler_options": {},
    }
    variants = [
        _add_square_variant(
            plan["argument_tensors"],
            n_elements,
            plan["stride_constants"],
            default_configuration,
            "default",
        )
    ]
    tuning_manifest: dict[str, Any] | None = None
    if enable_autotune:
        configurations, tuning_bytes = _load_pointwise_tuning(
            tuning_path, candidate, "add_square"
        )
        variants = [
            _add_square_variant(
                plan["argument_tensors"],
                n_elements,
                plan["stride_constants"],
                configuration,
                "block"
                + str(configuration["META"]["BLOCK_SIZE"])
                + "_w"
                + str(configuration["num_warps"])
                + "_s"
                + str(configuration["num_stages"]),
            )
            for configuration in configurations
        ]
        candidate_identity = _canonical_sha256(
            {
                "schema_version": 1,
                "backend": "thead",
                "target": target,
                "engine": "libtriton_jit",
                "compiler_identity": identity["identity_sha256"],
                "kernel": {
                    "provider": candidate.provider,
                    "ownership": candidate.ownership,
                    "operation": "add_square",
                    "function": plan["function"],
                    "registry_sha256": registry_digest,
                    "source_sha256": _sha256(source_bytes),
                },
                "tuning": {
                    "source_sha256": _sha256(tuning_bytes),
                    "table": candidate.tuning.table,
                    "key": candidate.tuning.key,
                    "key_value": n_elements,
                    "strategy": candidate.tuning.strategy,
                    "warmup": candidate.tuning.warmup,
                    "repetitions": candidate.tuning.repetitions,
                    "configurations": configurations,
                },
            }
        )
        tuning_manifest = {
            "warmup": candidate.tuning.warmup,
            "repetitions": candidate.tuning.repetitions,
            "candidate_identity": candidate_identity,
            "selection_cache": ".flagdnn-autotune-v1-stage-0.json",
        }

    # The single fused kernel consumes only the graph's external boundary
    # tensors.  Its virtual Mul output never crosses a stage boundary and
    # therefore has no backing workspace allocation.
    workspace_size = 0
    manifest_tensors = [_manifest_tensor(tensor) for tensor in plan["tensors"]]
    materialized_name = "generated_stage_0.py"
    manifest = {
        "schema_version": 1,
        "artifact_kind": "flagdnn_execution_program",
        "flagdnn_version": request["flagdnn_version"],
        "graph_ir_schema_version": SCHEMA_VERSION,
        "backend_abi_version": 2,
        "backend": "thead",
        "target": target,
        "warp_size": _PPU_WARP_SIZE,
        "engine": "libtriton_jit",
        "request_sha256": _sha256(request_bytes),
        "compiler": {
            "provider": identity["provider"],
            "provider_version": identity["provider_version"],
            "identity_sha256": identity["identity_sha256"],
        },
        "external_uids": [
            tensor["uid"] for tensor in plan["tensors"] if not tensor["virtual"]
        ],
        "tensor_count": len(plan["tensors"]),
        "tensors": manifest_tensors,
        "workspace": {"size": workspace_size, "alignment": 256},
        "program": {
            "schema_version": 1,
            "stage_count": 1,
            "stages": [
                {
                    "stage_id": 0,
                    "source_node_ids": [0, 1],
                    "dependencies": [],
                    "operation": "add_square",
                    "kernel": {
                        "provider": candidate.provider,
                        "ownership": candidate.ownership,
                        "function": plan["function"],
                        "registry_sha256": registry_digest,
                        "materialized_source": {
                            "path": materialized_name,
                            "size": len(source_bytes),
                            "sha256": _sha256(source_bytes),
                        },
                    },
                    "variants": variants,
                    "tuning": tuning_manifest,
                }
            ],
        },
    }
    manifest_bytes = json.dumps(
        manifest,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    output_directory.mkdir(parents=True, exist_ok=True)
    if output_directory.is_symlink() or not output_directory.is_dir():
        raise ValueError("THead artifact output directory is invalid")
    _atomic_write(output_directory / materialized_name, source_bytes)
    _atomic_write(output_directory / "manifest.json", manifest_bytes)
    return {
        "schema_version": 1,
        "status": "success",
        "backend": "thead",
        "provider": PROVIDER_NAME,
        "node_count": 2,
        "stage_count": 1,
        "target": target,
        "warp_size": _PPU_WARP_SIZE,
        "artifact_directory": str(output_directory),
        "workspace_size": workspace_size,
        "workspace_alignment": 256,
        "execution_engine": "libtriton_jit",
    }


def _compile_binary_pointwise(
    *,
    request: dict[str, Any],
    request_bytes: bytes,
    identity: dict[str, Any],
    target: str,
    output_directory: Path,
    enable_autotune: bool,
    operation: str,
) -> dict[str, Any]:
    operation_label = operation.capitalize()
    pointwise_mode = _BINARY_POINTWISE_MODES[operation]
    plan = _validate_binary_pointwise_graph(request["graph"], operation)
    candidate = kernel_registry.select_kernel_candidate("thead", operation)
    _validate_binary_pointwise_candidate(candidate, operation)
    compiler_path = Path(__file__)
    source_path = kernel_registry.resolve_kernel_source(compiler_path, candidate)
    source_bytes = kernel_registry.materialize_kernel_source(
        source_path, candidate
    )
    if not source_bytes or len(source_bytes) > _MAX_KERNEL_SOURCE_BYTES:
        raise ValueError(
            f"THead {operation_label} kernel source size is invalid"
        )
    tuning_path = kernel_registry.resolve_tuning_source(compiler_path, candidate)
    if not tuning_path.is_file():
        raise ValueError(f"THead {operation_label} tuning source is missing")
    registry_digest = _registry_sha256()

    tensors_by_uid = {
        int(tensor["uid"]): tensor for tensor in plan["tensors"]
    }
    argument_tensors = [
        tensors_by_uid[uid] for uid in plan["ordered_uids"]
    ]
    n_elements = int(plan["n_elements"])
    alpha = float(plan["alpha"])
    function = str(plan["function"])
    stride_constants = plan["stride_constants"]
    default_configuration = {
        "META": {"BLOCK_SIZE": _pointwise_block_size(n_elements)},
        "maxnreg": None,
        "num_stages": _FIXED_NUM_STAGES,
        "num_warps": _FIXED_NUM_WARPS,
        "ppu_compiler_options": {},
    }
    variants = [
        _binary_pointwise_variant(
            argument_tensors,
            n_elements,
            pointwise_mode,
            alpha,
            stride_constants,
            default_configuration,
            "default",
        )
    ]
    tuning_manifest: dict[str, Any] | None = None
    if enable_autotune:
        configurations, tuning_bytes = _load_pointwise_tuning(
            tuning_path, candidate, operation
        )
        variants = [
            _binary_pointwise_variant(
                argument_tensors,
                n_elements,
                pointwise_mode,
                alpha,
                stride_constants,
                configuration,
                "block"
                + str(configuration["META"]["BLOCK_SIZE"])
                + "_w"
                + str(configuration["num_warps"])
                + "_s"
                + str(configuration["num_stages"]),
            )
            for configuration in configurations
        ]
        candidate_identity = _canonical_sha256(
            {
                "schema_version": 1,
                "backend": "thead",
                "target": target,
                "engine": "libtriton_jit",
                "compiler_identity": identity["identity_sha256"],
                "kernel": {
                    "provider": candidate.provider,
                    "ownership": candidate.ownership,
                    "operation": operation,
                    "function": function,
                    "registry_sha256": registry_digest,
                    "source_sha256": _sha256(source_bytes),
                    "specialization": {"alpha": alpha},
                },
                "tuning": {
                    "source_sha256": _sha256(tuning_bytes),
                    "table": candidate.tuning.table,
                    "key": candidate.tuning.key,
                    "key_value": n_elements,
                    "strategy": candidate.tuning.strategy,
                    "warmup": candidate.tuning.warmup,
                    "repetitions": candidate.tuning.repetitions,
                    "configurations": configurations,
                },
            }
        )
        tuning_manifest = {
            "warmup": candidate.tuning.warmup,
            "repetitions": candidate.tuning.repetitions,
            "candidate_identity": candidate_identity,
            "selection_cache": ".flagdnn-autotune-v1-stage-0.json",
        }
    materialized_name = "generated_stage_0.py"
    manifest = {
        "schema_version": 1,
        "artifact_kind": "flagdnn_execution_program",
        "flagdnn_version": request["flagdnn_version"],
        "graph_ir_schema_version": SCHEMA_VERSION,
        "backend_abi_version": 2,
        "backend": "thead",
        "target": target,
        "warp_size": _PPU_WARP_SIZE,
        "engine": "libtriton_jit",
        "request_sha256": _sha256(request_bytes),
        "compiler": {
            "provider": identity["provider"],
            "provider_version": identity["provider_version"],
            "identity_sha256": identity["identity_sha256"],
        },
        "external_uids": [tensor["uid"] for tensor in plan["tensors"]],
        "tensor_count": len(plan["tensors"]),
        "tensors": [_manifest_tensor(tensor) for tensor in plan["tensors"]],
        "workspace": {"size": 0, "alignment": 256},
        "program": {
            "schema_version": 1,
            "stage_count": 1,
            "stages": [
                {
                    "stage_id": 0,
                    "source_node_ids": [0],
                    "dependencies": [],
                    "operation": operation,
                    "kernel": {
                        "provider": candidate.provider,
                        "ownership": candidate.ownership,
                        "function": function,
                        "registry_sha256": registry_digest,
                        "materialized_source": {
                            "path": materialized_name,
                            "size": len(source_bytes),
                            "sha256": _sha256(source_bytes),
                        },
                    },
                    "variants": variants,
                    "tuning": tuning_manifest,
                }
            ],
        },
    }
    manifest_bytes = json.dumps(
        manifest,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"

    output_directory.mkdir(parents=True, exist_ok=True)
    if output_directory.is_symlink() or not output_directory.is_dir():
        raise ValueError("THead artifact output directory is invalid")
    _atomic_write(output_directory / materialized_name, source_bytes)
    _atomic_write(output_directory / "manifest.json", manifest_bytes)
    return {
        "schema_version": 1,
        "status": "success",
        "backend": "thead",
        "provider": PROVIDER_NAME,
        "node_count": 1,
        "stage_count": 1,
        "target": target,
        "warp_size": _PPU_WARP_SIZE,
        "artifact_directory": str(output_directory),
        "workspace_size": 0,
        "workspace_alignment": 256,
        "execution_engine": "libtriton_jit",
    }


def _compile_unary_pointwise(
    *,
    request: dict[str, Any],
    request_bytes: bytes,
    identity: dict[str, Any],
    target: str,
    output_directory: Path,
    enable_autotune: bool,
    operation: str,
) -> dict[str, Any]:
    operation_label = operation.capitalize()
    pointwise_mode = _UNARY_POINTWISE_MODES[operation]
    plan = _validate_unary_pointwise_graph(request["graph"], operation)
    candidate = kernel_registry.select_kernel_candidate("thead", operation)
    _validate_unary_pointwise_candidate(candidate, operation)
    compiler_path = Path(__file__)
    source_path = kernel_registry.resolve_kernel_source(compiler_path, candidate)
    source_bytes = kernel_registry.materialize_kernel_source(
        source_path, candidate
    )
    if not source_bytes or len(source_bytes) > _MAX_KERNEL_SOURCE_BYTES:
        raise ValueError(
            f"THead {operation_label} kernel source size is invalid"
        )
    tuning_path = kernel_registry.resolve_tuning_source(compiler_path, candidate)
    if not tuning_path.is_file():
        raise ValueError(f"THead {operation_label} tuning source is missing")
    registry_digest = _registry_sha256()

    tensors_by_uid = {
        int(tensor["uid"]): tensor for tensor in plan["tensors"]
    }
    argument_tensors = [
        tensors_by_uid[uid] for uid in plan["ordered_uids"]
    ]
    n_elements = int(plan["n_elements"])
    function = str(plan["function"])
    stride_constants = plan["stride_constants"]
    kernel_constants = plan["kernel_constants"]
    default_configuration = {
        "META": {"BLOCK_SIZE": _pointwise_block_size(n_elements)},
        "maxnreg": None,
        "num_stages": _FIXED_NUM_STAGES,
        "num_warps": _FIXED_NUM_WARPS,
        "ppu_compiler_options": {},
    }
    variants = [
        _unary_pointwise_variant(
            argument_tensors,
            n_elements,
            pointwise_mode,
            stride_constants,
            kernel_constants,
            default_configuration,
            "default",
        )
    ]
    tuning_manifest: dict[str, Any] | None = None
    if enable_autotune:
        configurations, tuning_bytes = _load_pointwise_tuning(
            tuning_path, candidate, operation
        )
        variants = [
            _unary_pointwise_variant(
                argument_tensors,
                n_elements,
                pointwise_mode,
                stride_constants,
                kernel_constants,
                configuration,
                "block"
                + str(configuration["META"]["BLOCK_SIZE"])
                + "_w"
                + str(configuration["num_warps"])
                + "_s"
                + str(configuration["num_stages"]),
            )
            for configuration in configurations
        ]
        candidate_identity = _canonical_sha256(
            {
                "schema_version": 1,
                "backend": "thead",
                "target": target,
                "engine": "libtriton_jit",
                "compiler_identity": identity["identity_sha256"],
                "kernel": {
                    "provider": candidate.provider,
                    "ownership": candidate.ownership,
                    "operation": operation,
                    "function": function,
                    "registry_sha256": registry_digest,
                    "source_sha256": _sha256(source_bytes),
                    "specialization": kernel_constants,
                },
                "tuning": {
                    "source_sha256": _sha256(tuning_bytes),
                    "table": candidate.tuning.table,
                    "key": candidate.tuning.key,
                    "key_value": n_elements,
                    "strategy": candidate.tuning.strategy,
                    "warmup": candidate.tuning.warmup,
                    "repetitions": candidate.tuning.repetitions,
                    "configurations": configurations,
                },
            }
        )
        tuning_manifest = {
            "warmup": candidate.tuning.warmup,
            "repetitions": candidate.tuning.repetitions,
            "candidate_identity": candidate_identity,
            "selection_cache": ".flagdnn-autotune-v1-stage-0.json",
        }
    materialized_name = "generated_stage_0.py"
    manifest = {
        "schema_version": 1,
        "artifact_kind": "flagdnn_execution_program",
        "flagdnn_version": request["flagdnn_version"],
        "graph_ir_schema_version": SCHEMA_VERSION,
        "backend_abi_version": 2,
        "backend": "thead",
        "target": target,
        "warp_size": _PPU_WARP_SIZE,
        "engine": "libtriton_jit",
        "request_sha256": _sha256(request_bytes),
        "compiler": {
            "provider": identity["provider"],
            "provider_version": identity["provider_version"],
            "identity_sha256": identity["identity_sha256"],
        },
        "external_uids": [tensor["uid"] for tensor in plan["tensors"]],
        "tensor_count": len(plan["tensors"]),
        "tensors": [_manifest_tensor(tensor) for tensor in plan["tensors"]],
        "workspace": {"size": 0, "alignment": 256},
        "program": {
            "schema_version": 1,
            "stage_count": 1,
            "stages": [
                {
                    "stage_id": 0,
                    "source_node_ids": [0],
                    "dependencies": [],
                    "operation": operation,
                    "kernel": {
                        "provider": candidate.provider,
                        "ownership": candidate.ownership,
                        "function": function,
                        "registry_sha256": registry_digest,
                        "materialized_source": {
                            "path": materialized_name,
                            "size": len(source_bytes),
                            "sha256": _sha256(source_bytes),
                        },
                    },
                    "variants": variants,
                    "tuning": tuning_manifest,
                }
            ],
        },
    }
    manifest_bytes = json.dumps(
        manifest,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"

    output_directory.mkdir(parents=True, exist_ok=True)
    if output_directory.is_symlink() or not output_directory.is_dir():
        raise ValueError("THead artifact output directory is invalid")
    _atomic_write(output_directory / materialized_name, source_bytes)
    _atomic_write(output_directory / "manifest.json", manifest_bytes)
    return {
        "schema_version": 1,
        "status": "success",
        "backend": "thead",
        "provider": PROVIDER_NAME,
        "node_count": 1,
        "stage_count": 1,
        "target": target,
        "warp_size": _PPU_WARP_SIZE,
        "artifact_directory": str(output_directory),
        "workspace_size": 0,
        "workspace_alignment": 256,
        "execution_engine": "libtriton_jit",
    }


def _compile_layout(
    *,
    request: dict[str, Any],
    request_bytes: bytes,
    identity: dict[str, Any],
    target: str,
    output_directory: Path,
    enable_autotune: bool,
    operation: str,
) -> dict[str, Any]:
    plan = _validate_layout_graph(request["graph"], operation)
    candidate = kernel_registry.select_kernel_candidate("thead", operation)
    _validate_layout_candidate(candidate, operation)
    compiler_path = Path(__file__)
    source_path = kernel_registry.resolve_kernel_source(compiler_path, candidate)
    source_bytes = kernel_registry.materialize_kernel_source(
        source_path, candidate
    )
    if not source_bytes or len(source_bytes) > _MAX_KERNEL_SOURCE_BYTES:
        raise ValueError(f"THead {operation} kernel source size is invalid")
    tuning_path = kernel_registry.resolve_tuning_source(compiler_path, candidate)
    if not tuning_path.is_file():
        raise ValueError(f"THead {operation} tuning source is missing")
    registry_digest = _registry_sha256()
    n_elements = int(plan["n_elements"])
    default_configuration = {
        "META": {"BLOCK_SIZE": _FIXED_BLOCK_SIZE},
        "maxnreg": None,
        "num_stages": _FIXED_NUM_STAGES,
        "num_warps": _FIXED_NUM_WARPS,
        "ppu_compiler_options": {},
    }
    variants = [
        _layout_variant(
            plan["argument_tensors"],
            n_elements,
            plan["constants"],
            default_configuration,
            "default",
        )
    ]
    tuning_manifest: dict[str, Any] | None = None
    if enable_autotune:
        configurations, tuning_bytes = _load_pointwise_tuning(
            tuning_path, candidate, operation
        )
        variants = [
            _layout_variant(
                plan["argument_tensors"],
                n_elements,
                plan["constants"],
                configuration,
                "block"
                + str(configuration["META"]["BLOCK_SIZE"])
                + "_w"
                + str(configuration["num_warps"])
                + "_s"
                + str(configuration["num_stages"]),
            )
            for configuration in configurations
        ]
        candidate_identity = _canonical_sha256(
            {
                "schema_version": 1,
                "backend": "thead",
                "target": target,
                "engine": "libtriton_jit",
                "compiler_identity": identity["identity_sha256"],
                "kernel": {
                    "provider": candidate.provider,
                    "ownership": candidate.ownership,
                    "operation": operation,
                    "function": "layout_copy_kernel",
                    "registry_sha256": registry_digest,
                    "source_sha256": _sha256(source_bytes),
                    "specialization": plan["constants"],
                },
                "tuning": {
                    "source_sha256": _sha256(tuning_bytes),
                    "table": candidate.tuning.table,
                    "key": candidate.tuning.key,
                    "key_value": n_elements,
                    "strategy": candidate.tuning.strategy,
                    "warmup": candidate.tuning.warmup,
                    "repetitions": candidate.tuning.repetitions,
                    "configurations": configurations,
                },
            }
        )
        tuning_manifest = {
            "warmup": candidate.tuning.warmup,
            "repetitions": candidate.tuning.repetitions,
            "candidate_identity": candidate_identity,
            "selection_cache": ".flagdnn-autotune-v1-stage-0.json",
        }
    materialized_name = "generated_stage_0.py"
    manifest = {
        "schema_version": 1,
        "artifact_kind": "flagdnn_execution_program",
        "flagdnn_version": request["flagdnn_version"],
        "graph_ir_schema_version": SCHEMA_VERSION,
        "backend_abi_version": 2,
        "backend": "thead",
        "target": target,
        "warp_size": _PPU_WARP_SIZE,
        "engine": "libtriton_jit",
        "request_sha256": _sha256(request_bytes),
        "compiler": {
            "provider": identity["provider"],
            "provider_version": identity["provider_version"],
            "identity_sha256": identity["identity_sha256"],
        },
        # Graph external UID ordering is an integrity field and must match the
        # serialized Graph IR.  Kernel ABI reordering belongs only in each
        # variant's argument list.
        "external_uids": [tensor["uid"] for tensor in plan["tensors"]],
        "tensor_count": len(plan["tensors"]),
        "tensors": [_manifest_tensor(tensor) for tensor in plan["tensors"]],
        "workspace": {"size": 0, "alignment": 256},
        "program": {
            "schema_version": 1,
            "stage_count": 1,
            "stages": [
                {
                    "stage_id": 0,
                    "source_node_ids": [0],
                    "dependencies": [],
                    "operation": operation,
                    "kernel": {
                        "provider": candidate.provider,
                        "ownership": candidate.ownership,
                        "function": "layout_copy_kernel",
                        "registry_sha256": registry_digest,
                        "materialized_source": {
                            "path": materialized_name,
                            "size": len(source_bytes),
                            "sha256": _sha256(source_bytes),
                        },
                    },
                    "variants": variants,
                    "tuning": tuning_manifest,
                }
            ],
        },
    }
    manifest_bytes = json.dumps(
        manifest,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    output_directory.mkdir(parents=True, exist_ok=True)
    if output_directory.is_symlink() or not output_directory.is_dir():
        raise ValueError("THead artifact output directory is invalid")
    _atomic_write(output_directory / materialized_name, source_bytes)
    _atomic_write(output_directory / "manifest.json", manifest_bytes)
    return {
        "schema_version": 1,
        "status": "success",
        "backend": "thead",
        "provider": PROVIDER_NAME,
        "node_count": 1,
        "stage_count": 1,
        "target": target,
        "warp_size": _PPU_WARP_SIZE,
        "artifact_directory": str(output_directory),
        "workspace_size": 0,
        "workspace_alignment": 256,
        "execution_engine": "libtriton_jit",
    }


def _compile_reduction(
    *,
    request: dict[str, Any],
    request_bytes: bytes,
    identity: dict[str, Any],
    target: str,
    output_directory: Path,
    enable_autotune: bool,
    operation: str,
) -> dict[str, Any]:
    plan = _validate_reduction_graph(request["graph"], operation)
    candidate = kernel_registry.select_kernel_candidate("thead", operation)
    _validate_reduction_candidate(candidate, operation)
    compiler_path = Path(__file__)
    source_path = kernel_registry.resolve_kernel_source(compiler_path, candidate)
    source_bytes = kernel_registry.materialize_kernel_source(
        source_path, candidate
    )
    if not source_bytes or len(source_bytes) > _MAX_KERNEL_SOURCE_BYTES:
        raise ValueError("THead reduction kernel source size is invalid")
    tuning_path = kernel_registry.resolve_tuning_source(compiler_path, candidate)
    if not tuning_path.is_file():
        raise ValueError("THead reduction tuning source is missing")
    registry_digest = _registry_sha256()
    default_configuration = {
        "META": {"BLOCK_M": 8, "BLOCK_N": 32},
        "maxnreg": None,
        "num_stages": 1,
        "num_warps": 4,
        "ppu_compiler_options": {},
    }
    variants = [
        _reduction_variant(plan, default_configuration, "default")
    ]
    tuning_manifest: dict[str, Any] | None = None
    if enable_autotune:
        configurations, tuning_bytes = _load_reduction_tuning(
            tuning_path, candidate, operation
        )
        variants = [
            _reduction_variant(
                plan,
                configuration,
                "blockm"
                + str(configuration["META"]["BLOCK_M"])
                + "_blockn"
                + str(configuration["META"]["BLOCK_N"])
                + "_w"
                + str(configuration["num_warps"])
                + "_s"
                + str(configuration["num_stages"]),
            )
            for configuration in configurations
        ]
        candidate_identity = _canonical_sha256(
            {
                "schema_version": 1,
                "backend": "thead",
                "target": target,
                "engine": "libtriton_jit",
                "compiler_identity": identity["identity_sha256"],
                "kernel": {
                    "provider": candidate.provider,
                    "ownership": candidate.ownership,
                    "operation": operation,
                    "function": plan["function"],
                    "registry_sha256": registry_digest,
                    "source_sha256": _sha256(source_bytes),
                    "specialization": plan["constants"],
                },
                "tuning": {
                    "source_sha256": _sha256(tuning_bytes),
                    "table": candidate.tuning.table,
                    "key": candidate.tuning.key,
                    "key_value": plan["rows"],
                    "strategy": candidate.tuning.strategy,
                    "warmup": candidate.tuning.warmup,
                    "repetitions": candidate.tuning.repetitions,
                    "configurations": configurations,
                },
            }
        )
        tuning_manifest = {
            "warmup": candidate.tuning.warmup,
            "repetitions": candidate.tuning.repetitions,
            "candidate_identity": candidate_identity,
            "selection_cache": ".flagdnn-autotune-v1-stage-0.json",
        }
    materialized_name = "generated_stage_0.py"
    manifest = {
        "schema_version": 1,
        "artifact_kind": "flagdnn_execution_program",
        "flagdnn_version": request["flagdnn_version"],
        "graph_ir_schema_version": SCHEMA_VERSION,
        "backend_abi_version": 2,
        "backend": "thead",
        "target": target,
        "warp_size": _PPU_WARP_SIZE,
        "engine": "libtriton_jit",
        "request_sha256": _sha256(request_bytes),
        "compiler": {
            "provider": identity["provider"],
            "provider_version": identity["provider_version"],
            "identity_sha256": identity["identity_sha256"],
        },
        "external_uids": [tensor["uid"] for tensor in plan["tensors"]],
        "tensor_count": len(plan["tensors"]),
        "tensors": [_manifest_tensor(tensor) for tensor in plan["tensors"]],
        "workspace": {"size": 0, "alignment": 256},
        "program": {
            "schema_version": 1,
            "stage_count": 1,
            "stages": [
                {
                    "stage_id": 0,
                    "source_node_ids": [0],
                    "dependencies": [],
                    "operation": operation,
                    "kernel": {
                        "provider": candidate.provider,
                        "ownership": candidate.ownership,
                        "function": plan["function"],
                        "registry_sha256": registry_digest,
                        "materialized_source": {
                            "path": materialized_name,
                            "size": len(source_bytes),
                            "sha256": _sha256(source_bytes),
                        },
                    },
                    "variants": variants,
                    "tuning": tuning_manifest,
                }
            ],
        },
    }
    manifest_bytes = json.dumps(
        manifest,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    output_directory.mkdir(parents=True, exist_ok=True)
    if output_directory.is_symlink() or not output_directory.is_dir():
        raise ValueError("THead artifact output directory is invalid")
    _atomic_write(output_directory / materialized_name, source_bytes)
    _atomic_write(output_directory / "manifest.json", manifest_bytes)
    return {
        "schema_version": 1,
        "status": "success",
        "backend": "thead",
        "provider": PROVIDER_NAME,
        "node_count": 1,
        "stage_count": 1,
        "target": target,
        "warp_size": _PPU_WARP_SIZE,
        "artifact_directory": str(output_directory),
        "workspace_size": 0,
        "workspace_alignment": 256,
        "execution_engine": "libtriton_jit",
    }


def _compile_normalization(
    *,
    request: dict[str, Any],
    request_bytes: bytes,
    identity: dict[str, Any],
    target: str,
    output_directory: Path,
    enable_autotune: bool,
    operation: str,
) -> dict[str, Any]:
    plan = _validate_normalization_graph(request["graph"], operation)
    candidate = kernel_registry.select_kernel_candidate("thead", operation)
    _validate_normalization_candidate(candidate, operation)
    compiler_path = Path(__file__)
    source_path = kernel_registry.resolve_kernel_source(compiler_path, candidate)
    source_bytes = kernel_registry.materialize_kernel_source(
        source_path, candidate
    )
    if not source_bytes or len(source_bytes) > _MAX_KERNEL_SOURCE_BYTES:
        raise ValueError(f"THead {operation} kernel source size is invalid")
    tuning_path = kernel_registry.resolve_tuning_source(compiler_path, candidate)
    if not tuning_path.is_file():
        raise ValueError(f"THead {operation} tuning source is missing")
    registry_digest = _registry_sha256()
    default_configuration = {
        "META": {"BLOCK_SIZE": 256, "ROWS_PER_PROGRAM": 1},
        "maxnreg": None,
        "num_stages": 1,
        "num_warps": 4,
        "ppu_compiler_options": {},
    }
    variants = [
        _normalization_variant(plan, default_configuration, "default")
    ]
    tuning_manifest: dict[str, Any] | None = None
    if enable_autotune:
        configurations, tuning_bytes = _load_normalization_tuning(
            tuning_path, candidate, operation
        )
        if any(
            int(configuration["META"]["BLOCK_SIZE"])
            < int(plan["normalized_elements"])
            for configuration in configurations
        ):
            raise ValueError(
                f"THead {operation} tuning block is smaller than the suffix"
            )
        variants = [
            _normalization_variant(
                plan,
                configuration,
                "block"
                + str(configuration["META"]["BLOCK_SIZE"])
                + "_rows"
                + str(configuration["META"]["ROWS_PER_PROGRAM"])
                + "_w"
                + str(configuration["num_warps"])
                + "_s"
                + str(configuration["num_stages"]),
            )
            for configuration in configurations
        ]
        candidate_identity = _canonical_sha256(
            {
                "schema_version": 1,
                "backend": "thead",
                "target": target,
                "engine": "libtriton_jit",
                "compiler_identity": identity["identity_sha256"],
                "kernel": {
                    "provider": candidate.provider,
                    "ownership": candidate.ownership,
                    "operation": operation,
                    "function": plan["function"],
                    "registry_sha256": registry_digest,
                    "source_sha256": _sha256(source_bytes),
                    "specialization": {
                        "rows": plan["rows"],
                        "normalized_elements": plan["normalized_elements"],
                        "epsilon": plan["epsilon"],
                    },
                },
                "tuning": {
                    "source_sha256": _sha256(tuning_bytes),
                    "table": candidate.tuning.table,
                    "key": candidate.tuning.key,
                    "key_value": plan["normalized_elements"],
                    "strategy": candidate.tuning.strategy,
                    "warmup": candidate.tuning.warmup,
                    "repetitions": candidate.tuning.repetitions,
                    "configurations": configurations,
                },
            }
        )
        tuning_manifest = {
            "warmup": candidate.tuning.warmup,
            "repetitions": candidate.tuning.repetitions,
            "candidate_identity": candidate_identity,
            "selection_cache": ".flagdnn-autotune-v1-stage-0.json",
        }
    materialized_name = "generated_stage_0.py"
    manifest = {
        "schema_version": 1,
        "artifact_kind": "flagdnn_execution_program",
        "flagdnn_version": request["flagdnn_version"],
        "graph_ir_schema_version": SCHEMA_VERSION,
        "backend_abi_version": 2,
        "backend": "thead",
        "target": target,
        "warp_size": _PPU_WARP_SIZE,
        "engine": "libtriton_jit",
        "request_sha256": _sha256(request_bytes),
        "compiler": {
            "provider": identity["provider"],
            "provider_version": identity["provider_version"],
            "identity_sha256": identity["identity_sha256"],
        },
        "external_uids": [tensor["uid"] for tensor in plan["tensors"]],
        "tensor_count": len(plan["tensors"]),
        "tensors": [_manifest_tensor(tensor) for tensor in plan["tensors"]],
        "workspace": {"size": 0, "alignment": 256},
        "program": {
            "schema_version": 1,
            "stage_count": 1,
            "stages": [
                {
                    "stage_id": 0,
                    "source_node_ids": [0],
                    "dependencies": [],
                    "operation": operation,
                    "kernel": {
                        "provider": candidate.provider,
                        "ownership": candidate.ownership,
                        "function": plan["function"],
                        "registry_sha256": registry_digest,
                        "materialized_source": {
                            "path": materialized_name,
                            "size": len(source_bytes),
                            "sha256": _sha256(source_bytes),
                        },
                    },
                    "variants": variants,
                    "tuning": tuning_manifest,
                }
            ],
        },
    }
    manifest_bytes = json.dumps(
        manifest,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    output_directory.mkdir(parents=True, exist_ok=True)
    if output_directory.is_symlink() or not output_directory.is_dir():
        raise ValueError("THead artifact output directory is invalid")
    _atomic_write(output_directory / materialized_name, source_bytes)
    _atomic_write(output_directory / "manifest.json", manifest_bytes)
    return {
        "schema_version": 1,
        "status": "success",
        "backend": "thead",
        "provider": PROVIDER_NAME,
        "node_count": 1,
        "stage_count": 1,
        "target": target,
        "warp_size": _PPU_WARP_SIZE,
        "artifact_directory": str(output_directory),
        "workspace_size": 0,
        "workspace_alignment": 256,
        "execution_engine": "libtriton_jit",
    }


def _compile_batchnorm(
    *,
    request: dict[str, Any],
    request_bytes: bytes,
    identity: dict[str, Any],
    target: str,
    output_directory: Path,
    enable_autotune: bool,
    operation: str,
) -> dict[str, Any]:
    plan = _validate_batchnorm_graph(request["graph"], operation)
    candidate = kernel_registry.select_kernel_candidate("thead", operation)
    _validate_batchnorm_candidate(candidate, operation)
    compiler_path = Path(__file__)
    source_path = kernel_registry.resolve_kernel_source(compiler_path, candidate)
    source_bytes = kernel_registry.materialize_kernel_source(
        source_path, candidate
    )
    if not source_bytes or len(source_bytes) > _MAX_KERNEL_SOURCE_BYTES:
        raise ValueError(f"THead {operation} kernel source size is invalid")
    tuning_path = kernel_registry.resolve_tuning_source(compiler_path, candidate)
    if not tuning_path.is_file():
        raise ValueError(f"THead {operation} tuning source is missing")
    registry_digest = _registry_sha256()
    default_configuration = {
        "META": {"BLOCK_SIZE": 256},
        "maxnreg": None,
        "num_stages": 1,
        "num_warps": 4,
        "ppu_compiler_options": {},
    }
    variants = [_batchnorm_variant(plan, default_configuration, "default")]
    tuning_manifest: dict[str, Any] | None = None
    if enable_autotune:
        configurations, tuning_bytes = _load_pointwise_tuning(
            tuning_path, candidate, operation
        )
        variants = [
            _batchnorm_variant(
                plan,
                configuration,
                "block"
                + str(configuration["META"]["BLOCK_SIZE"])
                + "_w"
                + str(configuration["num_warps"])
                + "_s"
                + str(configuration["num_stages"]),
            )
            for configuration in configurations
        ]
        key_value = plan[
            "n_elements" if operation == "batchnorm_inference" else "channels"
        ]
        candidate_identity = _canonical_sha256(
            {
                "schema_version": 1,
                "backend": "thead",
                "target": target,
                "engine": "libtriton_jit",
                "compiler_identity": identity["identity_sha256"],
                "kernel": {
                    "provider": candidate.provider,
                    "ownership": candidate.ownership,
                    "operation": operation,
                    "function": plan["function"],
                    "registry_sha256": registry_digest,
                    "source_sha256": _sha256(source_bytes),
                    "specialization": {
                        name: plan[name]
                        for name in (
                            "batch",
                            "channels",
                            "spatial",
                            "epsilon",
                            "momentum",
                        )
                    },
                },
                "tuning": {
                    "source_sha256": _sha256(tuning_bytes),
                    "table": candidate.tuning.table,
                    "key": candidate.tuning.key,
                    "key_value": key_value,
                    "strategy": candidate.tuning.strategy,
                    "warmup": candidate.tuning.warmup,
                    "repetitions": candidate.tuning.repetitions,
                    "configurations": configurations,
                },
            }
        )
        tuning_manifest = {
            "warmup": candidate.tuning.warmup,
            "repetitions": candidate.tuning.repetitions,
            "candidate_identity": candidate_identity,
            "selection_cache": ".flagdnn-autotune-v1-stage-0.json",
        }
    materialized_name = "generated_stage_0.py"
    manifest = {
        "schema_version": 1,
        "artifact_kind": "flagdnn_execution_program",
        "flagdnn_version": request["flagdnn_version"],
        "graph_ir_schema_version": SCHEMA_VERSION,
        "backend_abi_version": 2,
        "backend": "thead",
        "target": target,
        "warp_size": _PPU_WARP_SIZE,
        "engine": "libtriton_jit",
        "request_sha256": _sha256(request_bytes),
        "compiler": {
            "provider": identity["provider"],
            "provider_version": identity["provider_version"],
            "identity_sha256": identity["identity_sha256"],
        },
        # Preserve Graph IR ordering here; the variant independently carries
        # the reordered kernel ABI argument list.
        "external_uids": [tensor["uid"] for tensor in plan["tensors"]],
        "tensor_count": len(plan["tensors"]),
        "tensors": [_manifest_tensor(tensor) for tensor in plan["tensors"]],
        "workspace": {"size": 0, "alignment": 256},
        "program": {
            "schema_version": 1,
            "stage_count": 1,
            "stages": [
                {
                    "stage_id": 0,
                    "source_node_ids": [0],
                    "dependencies": [],
                    "operation": operation,
                    "kernel": {
                        "provider": candidate.provider,
                        "ownership": candidate.ownership,
                        "function": plan["function"],
                        "registry_sha256": registry_digest,
                        "materialized_source": {
                            "path": materialized_name,
                            "size": len(source_bytes),
                            "sha256": _sha256(source_bytes),
                        },
                    },
                    "variants": variants,
                    "tuning": tuning_manifest,
                }
            ],
        },
    }
    manifest_bytes = json.dumps(
        manifest,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    output_directory.mkdir(parents=True, exist_ok=True)
    if output_directory.is_symlink() or not output_directory.is_dir():
        raise ValueError("THead artifact output directory is invalid")
    _atomic_write(output_directory / materialized_name, source_bytes)
    _atomic_write(output_directory / "manifest.json", manifest_bytes)
    return {
        "schema_version": 1,
        "status": "success",
        "backend": "thead",
        "provider": PROVIDER_NAME,
        "node_count": 1,
        "stage_count": 1,
        "target": target,
        "warp_size": _PPU_WARP_SIZE,
        "artifact_directory": str(output_directory),
        "workspace_size": 0,
        "workspace_alignment": 256,
        "execution_engine": "libtriton_jit",
    }


def _compile_matmul(
    *,
    request: dict[str, Any],
    request_bytes: bytes,
    identity: dict[str, Any],
    target: str,
    output_directory: Path,
    enable_autotune: bool,
) -> dict[str, Any]:
    plan = _validate_matmul_graph(request["graph"])
    candidate = kernel_registry.select_kernel_candidate("thead", "matmul")
    _validate_matmul_candidate(candidate)
    compiler_path = Path(__file__)
    source_path = kernel_registry.resolve_kernel_source(compiler_path, candidate)
    source_bytes = kernel_registry.materialize_kernel_source(
        source_path, candidate
    )
    if not source_bytes or len(source_bytes) > _MAX_KERNEL_SOURCE_BYTES:
        raise ValueError("THead MatMul kernel source size is invalid")
    tuning_path = kernel_registry.resolve_tuning_source(compiler_path, candidate)
    if not tuning_path.is_file():
        raise ValueError("THead MatMul tuning source is missing")
    registry_digest = _registry_sha256()
    large_matrix = (
        int(plan["m"]) >= 64
        and int(plan["n"]) >= 64
        and int(plan["k"]) >= 32
    )
    very_large_matrix = (
        int(plan["m"]) >= 128
        and int(plan["n"]) >= 128
        and int(plan["k"]) >= 64
    )
    default_configuration = {
        "META": {
            "BLOCK_M": 128 if very_large_matrix else (64 if large_matrix else 16),
            "BLOCK_N": 128 if very_large_matrix else (64 if large_matrix else 16),
            "BLOCK_K": 32 if large_matrix else 16,
            "GROUP_M": 8 if large_matrix else 1,
        },
        "maxnreg": None,
        "num_stages": 1,
        "num_warps": 8 if very_large_matrix else 4,
        "ppu_compiler_options": {},
    }
    variants = [_matmul_variant(plan, default_configuration, "default")]
    tuning_manifest: dict[str, Any] | None = None
    if enable_autotune:
        configurations, tuning_bytes = _load_matmul_tuning(
            tuning_path, candidate
        )
        variants = [
            _matmul_variant(
                plan,
                configuration,
                "blockm"
                + str(configuration["META"]["BLOCK_M"])
                + "_blockn"
                + str(configuration["META"]["BLOCK_N"])
                + "_blockk"
                + str(configuration["META"]["BLOCK_K"])
                + "_group"
                + str(configuration["META"]["GROUP_M"])
                + "_w"
                + str(configuration["num_warps"])
                + "_s"
                + str(configuration["num_stages"]),
            )
            for configuration in configurations
        ]
        candidate_identity = _canonical_sha256(
            {
                "schema_version": 1,
                "backend": "thead",
                "target": target,
                "engine": "libtriton_jit",
                "compiler_identity": identity["identity_sha256"],
                "kernel": {
                    "provider": candidate.provider,
                    "ownership": candidate.ownership,
                    "operation": "matmul",
                    "function": "matmul_strided_kernel",
                    "registry_sha256": registry_digest,
                    "source_sha256": _sha256(source_bytes),
                    "specialization": plan["constants"],
                },
                "tuning": {
                    "source_sha256": _sha256(tuning_bytes),
                    "table": candidate.tuning.table,
                    "key": candidate.tuning.key,
                    "key_value": plan["m"],
                    "strategy": candidate.tuning.strategy,
                    "warmup": candidate.tuning.warmup,
                    "repetitions": candidate.tuning.repetitions,
                    "configurations": configurations,
                },
            }
        )
        tuning_manifest = {
            "warmup": candidate.tuning.warmup,
            "repetitions": candidate.tuning.repetitions,
            "candidate_identity": candidate_identity,
            "selection_cache": ".flagdnn-autotune-v1-stage-0.json",
        }
    materialized_name = "generated_stage_0.py"
    manifest = {
        "schema_version": 1,
        "artifact_kind": "flagdnn_execution_program",
        "flagdnn_version": request["flagdnn_version"],
        "graph_ir_schema_version": SCHEMA_VERSION,
        "backend_abi_version": 2,
        "backend": "thead",
        "target": target,
        "warp_size": _PPU_WARP_SIZE,
        "engine": "libtriton_jit",
        "request_sha256": _sha256(request_bytes),
        "compiler": {
            "provider": identity["provider"],
            "provider_version": identity["provider_version"],
            "identity_sha256": identity["identity_sha256"],
        },
        "external_uids": [tensor["uid"] for tensor in plan["tensors"]],
        "tensor_count": len(plan["tensors"]),
        "tensors": [_manifest_tensor(tensor) for tensor in plan["tensors"]],
        "workspace": {"size": 0, "alignment": 256},
        "program": {
            "schema_version": 1,
            "stage_count": 1,
            "stages": [
                {
                    "stage_id": 0,
                    "source_node_ids": [0],
                    "dependencies": [],
                    "operation": "matmul",
                    "kernel": {
                        "provider": candidate.provider,
                        "ownership": candidate.ownership,
                        "function": "matmul_strided_kernel",
                        "registry_sha256": registry_digest,
                        "materialized_source": {
                            "path": materialized_name,
                            "size": len(source_bytes),
                            "sha256": _sha256(source_bytes),
                        },
                    },
                    "variants": variants,
                    "tuning": tuning_manifest,
                }
            ],
        },
    }
    manifest_bytes = json.dumps(
        manifest,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    output_directory.mkdir(parents=True, exist_ok=True)
    if output_directory.is_symlink() or not output_directory.is_dir():
        raise ValueError("THead artifact output directory is invalid")
    _atomic_write(output_directory / materialized_name, source_bytes)
    _atomic_write(output_directory / "manifest.json", manifest_bytes)
    return {
        "schema_version": 1,
        "status": "success",
        "backend": "thead",
        "provider": PROVIDER_NAME,
        "node_count": 1,
        "stage_count": 1,
        "target": target,
        "warp_size": _PPU_WARP_SIZE,
        "artifact_directory": str(output_directory),
        "workspace_size": 0,
        "workspace_alignment": 256,
        "execution_engine": "libtriton_jit",
    }


def _compile_convolution(
    *,
    request: dict[str, Any],
    request_bytes: bytes,
    identity: dict[str, Any],
    target: str,
    output_directory: Path,
    enable_autotune: bool,
    operation: str,
) -> dict[str, Any]:
    plan = (
        _validate_conv_bias_relu_graph(request["graph"])
        if operation == "conv_bias_relu"
        else _validate_convolution_graph(request["graph"], operation)
    )
    candidate = kernel_registry.select_kernel_candidate("thead", operation)
    _validate_convolution_candidate(candidate, operation)
    compiler_path = Path(__file__)
    source_path = kernel_registry.resolve_kernel_source(
        compiler_path, candidate
    )
    source_bytes = kernel_registry.materialize_kernel_source(
        source_path, candidate
    )
    if not source_bytes or len(source_bytes) > _MAX_KERNEL_SOURCE_BYTES:
        raise ValueError(f"THead {operation} kernel source size is invalid")
    tuning_path = kernel_registry.resolve_tuning_source(
        compiler_path, candidate
    )
    if not tuning_path.is_file():
        raise ValueError(f"THead {operation} tuning source is missing")
    registry_digest = _registry_sha256()
    default_configuration = {
        "META": {
            "BLOCK_M": 16,
            "BLOCK_OC": 16,
            "BLOCK_CI": 16,
            "BLOCK_K": 16,
            "BLOCK_HW": 16,
        },
        "maxnreg": None,
        "num_stages": 1,
        "num_warps": 4,
        "ppu_compiler_options": {},
    }
    variants = [
        _convolution_variant(plan, default_configuration, "default")
    ]
    tuning_manifest: dict[str, Any] | None = None
    if enable_autotune:
        configurations, tuning_bytes = _load_convolution_tuning(
            tuning_path, candidate, operation
        )
        variants = [
            _convolution_variant(
                plan,
                configuration,
                "tile"
                + str(configuration["META"]["BLOCK_M"])
                + "_w"
                + str(configuration["num_warps"])
                + "_s"
                + str(configuration["num_stages"]),
            )
            for configuration in configurations
        ]
        candidate_identity = _canonical_sha256(
            {
                "schema_version": 1,
                "backend": "thead",
                "target": target,
                "engine": "libtriton_jit",
                "compiler_identity": identity["identity_sha256"],
                "kernel": {
                    "provider": candidate.provider,
                    "ownership": candidate.ownership,
                    "operation": operation,
                    "function": plan["function"],
                    "registry_sha256": registry_digest,
                    "source_sha256": _sha256(source_bytes),
                    "specialization": plan["constants"],
                },
                "tuning": {
                    "source_sha256": _sha256(tuning_bytes),
                    "table": candidate.tuning.table,
                    "key": candidate.tuning.key,
                    "key_value": plan["n_outputs"],
                    "strategy": candidate.tuning.strategy,
                    "warmup": candidate.tuning.warmup,
                    "repetitions": candidate.tuning.repetitions,
                    "configurations": configurations,
                },
            }
        )
        tuning_manifest = {
            "warmup": candidate.tuning.warmup,
            "repetitions": candidate.tuning.repetitions,
            "candidate_identity": candidate_identity,
            "selection_cache": ".flagdnn-autotune-v1-stage-0.json",
        }
    materialized_name = "generated_stage_0.py"
    # ConvBiasRelu is emitted as one fused kernel.  Its convolution and bias
    # intermediates remain internal SSA values, so materializing either graph
    # virtual tensor would reserve memory that the launch ABI never uses.
    workspace_size = 0
    manifest_tensors = [_manifest_tensor(tensor) for tensor in plan["tensors"]]
    source_node_ids = plan.get("source_node_ids", [0])
    manifest = {
        "schema_version": 1,
        "artifact_kind": "flagdnn_execution_program",
        "flagdnn_version": request["flagdnn_version"],
        "graph_ir_schema_version": SCHEMA_VERSION,
        "backend_abi_version": 2,
        "backend": "thead",
        "target": target,
        "warp_size": _PPU_WARP_SIZE,
        "engine": "libtriton_jit",
        "request_sha256": _sha256(request_bytes),
        "compiler": {
            "provider": identity["provider"],
            "provider_version": identity["provider_version"],
            "identity_sha256": identity["identity_sha256"],
        },
        "external_uids": [
            tensor["uid"] for tensor in plan["tensors"]
            if not tensor["virtual"]
        ],
        "tensor_count": len(plan["tensors"]),
        "tensors": manifest_tensors,
        "workspace": {"size": workspace_size, "alignment": 256},
        "program": {
            "schema_version": 1,
            "stage_count": 1,
            "stages": [
                {
                    "stage_id": 0,
                    "source_node_ids": source_node_ids,
                    "dependencies": [],
                    "operation": operation,
                    "kernel": {
                        "provider": candidate.provider,
                        "ownership": candidate.ownership,
                        "function": plan["function"],
                        "registry_sha256": registry_digest,
                        "materialized_source": {
                            "path": materialized_name,
                            "size": len(source_bytes),
                            "sha256": _sha256(source_bytes),
                        },
                    },
                    "variants": variants,
                    "tuning": tuning_manifest,
                }
            ],
        },
    }
    manifest_bytes = json.dumps(
        manifest,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    output_directory.mkdir(parents=True, exist_ok=True)
    if output_directory.is_symlink() or not output_directory.is_dir():
        raise ValueError("THead artifact output directory is invalid")
    _atomic_write(output_directory / materialized_name, source_bytes)
    _atomic_write(output_directory / "manifest.json", manifest_bytes)
    return {
        "schema_version": 1,
        "status": "success",
        "backend": "thead",
        "provider": PROVIDER_NAME,
        "node_count": len(source_node_ids),
        "stage_count": 1,
        "target": target,
        "warp_size": _PPU_WARP_SIZE,
        "artifact_directory": str(output_directory),
        "workspace_size": workspace_size,
        "workspace_alignment": 256,
        "execution_engine": "libtriton_jit",
    }


def compile_request(
    request_path: Path,
    output_directory: Path,
    execution_engine: str = "libtriton_jit",
) -> dict[str, Any]:
    if execution_engine != "libtriton_jit":
        raise ValueError("THead supports only the libtriton_jit engine")
    request, request_bytes = _load_request(Path(request_path))
    _require_exact_fields(
        request,
        {
            "schema_version",
            "flagdnn_version",
            "backend",
            "target",
            "compiler_identity",
            "build_options",
            "graph",
        },
        set(),
        "request",
    )
    if request["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported request schema_version")
    version = _string(request["flagdnn_version"], "request FlagDNN version")
    if _VERSION.fullmatch(version) is None:
        raise ValueError("request FlagDNN version is invalid")
    if request["backend"] != "thead":
        raise ValueError("THead provider received another backend")
    target = _string(request["target"], "request target")
    identity = compiler_identity(target, execution_engine)
    requested_identity = request["compiler_identity"]
    if (
        not isinstance(requested_identity, str)
        or _SHA256.fullmatch(requested_identity) is None
        or requested_identity != identity["identity_sha256"]
    ):
        raise ValueError("request compiler identity does not match provider")
    enable_autotune = _parse_build_options(request["build_options"])
    operation_types = _parse_graph(request["graph"])
    if operation_types == _CONV_BIAS_RELU_OPERATION_TYPES:
        return _compile_convolution(
            request=request,
            request_bytes=request_bytes,
            identity=identity,
            target=target,
            output_directory=Path(output_directory),
            enable_autotune=enable_autotune,
            operation="conv_bias_relu",
        )
    if operation_types == ["add", "mul"]:
        return _compile_add_square(
            request=request,
            request_bytes=request_bytes,
            identity=identity,
            target=target,
            output_directory=Path(output_directory),
            enable_autotune=enable_autotune,
        )
    if len(operation_types) == 1 and operation_types[0] in _BINARY_POINTWISE_MODES:
        return _compile_binary_pointwise(
            request=request,
            request_bytes=request_bytes,
            identity=identity,
            target=target,
            output_directory=Path(output_directory),
            enable_autotune=enable_autotune,
            operation=operation_types[0],
        )
    if len(operation_types) == 1 and operation_types[0] in _UNARY_POINTWISE_MODES:
        return _compile_unary_pointwise(
            request=request,
            request_bytes=request_bytes,
            identity=identity,
            target=target,
            output_directory=Path(output_directory),
            enable_autotune=enable_autotune,
            operation=operation_types[0],
        )
    if len(operation_types) == 1 and operation_types[0] in _LAYOUT_OPERATIONS:
        return _compile_layout(
            request=request,
            request_bytes=request_bytes,
            identity=identity,
            target=target,
            output_directory=Path(output_directory),
            enable_autotune=enable_autotune,
            operation=operation_types[0],
        )
    if len(operation_types) == 1 and operation_types[0] in _REDUCTION_OPERATIONS:
        return _compile_reduction(
            request=request,
            request_bytes=request_bytes,
            identity=identity,
            target=target,
            output_directory=Path(output_directory),
            enable_autotune=enable_autotune,
            operation=operation_types[0],
        )
    if (
        len(operation_types) == 1
        and operation_types[0] in _NORMALIZATION_OPERATIONS
    ):
        return _compile_normalization(
            request=request,
            request_bytes=request_bytes,
            identity=identity,
            target=target,
            output_directory=Path(output_directory),
            enable_autotune=enable_autotune,
            operation=operation_types[0],
        )
    if len(operation_types) == 1 and operation_types[0] in _BATCHNORM_OPERATIONS:
        return _compile_batchnorm(
            request=request,
            request_bytes=request_bytes,
            identity=identity,
            target=target,
            output_directory=Path(output_directory),
            enable_autotune=enable_autotune,
            operation=operation_types[0],
        )
    if len(operation_types) == 1 and operation_types[0] in _MATMUL_OPERATIONS:
        return _compile_matmul(
            request=request,
            request_bytes=request_bytes,
            identity=identity,
            target=target,
            output_directory=Path(output_directory),
            enable_autotune=enable_autotune,
        )
    if (
        len(operation_types) == 1
        and operation_types[0] in _CONVOLUTION_OPERATIONS
    ):
        return _compile_convolution(
            request=request,
            request_bytes=request_bytes,
            identity=identity,
            target=target,
            output_directory=Path(output_directory),
            enable_autotune=enable_autotune,
            operation=operation_types[0],
        )
    return {
        "schema_version": 1,
        "status": "unsupported",
        "reason_code": "operation_family_not_implemented",
        "backend": "thead",
        "target": target,
        "execution_engine": execution_engine,
        "operation_types": operation_types,
    }


__all__ = (
    "PROVIDER_NAME",
    "PROVIDER_VERSION",
    "compile_request",
    "compiler_identity",
    "compiler_identity_dependencies",
)
