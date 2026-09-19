"""Ascend dispatch for tensor."""

from __future__ import annotations
import math
from .common import (
    ELEMENT_SIZES,
    GRAPH_WORKSPACE_ALIGNMENT,
    MAX_I32,
    MAX_I64,
    MAX_RANK,
    TensorPlan,
    require_list,
    require_object,
)
from typing import Any
from .common import (
    EXPECTED_OUTPUT_COUNTS,
    EXPECTED_TENSOR_ROLES,
    TRITON_POINTER_TYPES,
    _require_integer,
    _require_list,
    _require_number,
    _require_object,
    _tensor_storage_size,
)


def _has_non_overlapping_strides(
    dimensions: tuple[int, ...], strides: tuple[int, ...]
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


def _storage_elements(dimensions: tuple[int, ...], strides: tuple[int, ...]) -> int:
    result = 1
    for dimension, stride in zip(dimensions, strides):
        term = (dimension - 1) * stride
        if term > MAX_I64 - result:
            raise ValueError("tensor storage span exceeds int64 range")
        result += term
    return result


def _is_physically_dense(tensor: TensorPlan) -> bool:
    return (
        tensor.storage_size
        == math.prod(tensor.dimensions) * ELEMENT_SIZES[tensor.data_type]
    )


def _is_row_major_tensor(tensor: TensorPlan) -> bool:
    expected = 1
    for dimension, stride in zip(reversed(tensor.dimensions), reversed(tensor.strides)):
        if dimension > 1 and stride != expected:
            return False
        expected *= dimension
    return _is_physically_dense(tensor)


def _parse_tensor_table(graph: dict[str, Any]) -> dict[int, TensorPlan]:
    values = require_list(graph.get("tensors"), "graph.tensors")
    tensor_count = graph.get("tensor_count")
    if (
        isinstance(tensor_count, bool)
        or not isinstance(tensor_count, int)
        or tensor_count != len(values)
        or tensor_count < 1
    ):
        raise ValueError("graph.tensor_count is invalid")

    result: dict[int, TensorPlan] = {}
    for index, raw_value in enumerate(values):
        value = require_object(raw_value, f"graph.tensors[{index}]")
        uid = value.get("uid")
        if isinstance(uid, bool) or not isinstance(uid, int) or uid <= 0:
            raise ValueError(f"tensor UID at index {index} is invalid")
        if uid in result:
            raise ValueError("graph tensor UIDs must be unique")

        data_type = value.get("data_type")
        if data_type not in ELEMENT_SIZES:
            raise ValueError(
                "tensor "
                f"{uid} data type is unsupported by Ascend pointwise: "
                f"{data_type!r}"
            )
        virtual = value.get("virtual")
        if not isinstance(virtual, bool):
            raise ValueError(f"tensor {uid} virtual must be a boolean")
        alignment = value.get("alignment", 16)
        if (
            isinstance(alignment, bool)
            or not isinstance(alignment, int)
            or alignment <= 0
            or alignment > GRAPH_WORKSPACE_ALIGNMENT
            or alignment & (alignment - 1)
        ):
            raise ValueError(
                f"tensor {uid} alignment must be a power of two in [1, 256]"
            )

        raw_dimensions = require_list(
            value.get("dimensions"), f"tensor {uid} dimensions"
        )
        raw_strides = require_list(value.get("strides"), f"tensor {uid} strides")
        if len(raw_dimensions) != len(raw_strides) or len(raw_dimensions) > MAX_RANK:
            raise ValueError(f"tensor {uid} rank is invalid")
        if any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item <= 0
            or item > MAX_I32
            for item in raw_dimensions
        ):
            raise ValueError(f"tensor {uid} dimensions are invalid")
        if any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
            or item > MAX_I64
            for item in raw_strides
        ):
            raise ValueError(f"tensor {uid} strides are invalid")
        dimensions = tuple(raw_dimensions)
        strides = tuple(raw_strides)
        if not _has_non_overlapping_strides(dimensions, strides):
            raise ValueError(f"tensor {uid} strides overlap")
        storage_elements = _storage_elements(dimensions, strides)
        element_size = ELEMENT_SIZES[data_type]
        if storage_elements > MAX_I64 // element_size:
            raise ValueError(f"tensor {uid} storage size exceeds int64 range")
        result[uid] = TensorPlan(
            uid=uid,
            data_type=data_type,
            dimensions=dimensions,
            strides=strides,
            alignment=alignment,
            virtual=virtual,
            storage_size=storage_elements * element_size,
        )
    return result


def _parse_port(
    value: object,
    *,
    description: str,
    expected_name: str,
    tensors: dict[int, TensorPlan],
) -> TensorPlan:
    port = require_object(value, description)
    if port.get("name") != expected_name:
        raise ValueError(f"{description}.name must be {expected_name!r}")
    optional = port.get("optional", False)
    if not isinstance(optional, bool) or optional:
        raise ValueError("Ascend pointwise does not support absent optional ports")
    uid = port.get("uid")
    if isinstance(uid, bool) or not isinstance(uid, int) or uid <= 0:
        raise ValueError(f"{description}.uid is invalid")
    try:
        return tensors[uid]
    except KeyError as error:
        raise ValueError(
            f"{description} references unknown tensor UID {uid}"
        ) from error


def _parse_named_ports(
    values: list[Any],
    *,
    description: str,
    expected_names: tuple[str, ...],
    tensors: dict[int, TensorPlan],
) -> tuple[TensorPlan, ...]:
    if len(values) != len(expected_names):
        raise ValueError(f"{description} has invalid arity")
    actual_names = tuple(
        require_object(value, f"{description}[{index}]").get("name")
        for index, value in enumerate(values)
    )
    if actual_names != expected_names:
        raise ValueError(f"{description} must use canonical input order")
    expected = set(expected_names)
    parsed: dict[str, TensorPlan] = {}
    for index, raw_value in enumerate(values):
        port = require_object(raw_value, f"{description}[{index}]")
        name = port.get("name")
        if not isinstance(name, str) or name not in expected or name in parsed:
            raise ValueError(f"{description} names are invalid")
        parsed[name] = _parse_port(
            port,
            description=f"{description}[{index}]",
            expected_name=name,
            tensors=tensors,
        )
    return tuple(parsed[name] for name in expected_names)


def _effective_strides(tensor: TensorPlan, rank: int) -> list[int]:
    leading = rank - len(tensor.dimensions)
    dimensions = [1] * leading + list(tensor.dimensions)
    strides = [0] * leading + list(tensor.strides)
    return [
        0 if dimension == 1 else stride
        for dimension, stride in zip(dimensions, strides)
    ]


def _parse_tensor_descriptors(graph: dict[str, Any]) -> dict[int, dict[str, Any]]:
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
    for index, tensor_value in enumerate(tensors):
        tensor = _require_object(tensor_value, f"graph.tensors[{index}]")
        uid = tensor.get("uid")
        if isinstance(uid, bool) or not isinstance(uid, int) or uid <= 0:
            raise ValueError(f"tensor UID {index} is invalid")
        if uid in result:
            raise ValueError("graph tensor UIDs must be unique")
        data_type = tensor.get("data_type")
        if not isinstance(data_type, str) or data_type not in TRITON_POINTER_TYPES:
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
            or alignment & (alignment - 1) != 0
        ):
            raise ValueError(f"tensor {uid} alignment must be a positive power of two")
        dimensions = _require_list(tensor.get("dimensions"), f"tensor {uid} dimensions")
        strides = _require_list(tensor.get("strides"), f"tensor {uid} strides")
        if len(dimensions) != len(strides) or len(dimensions) > 8:
            raise ValueError(f"tensor {uid} rank is invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in dimensions + strides
        ):
            raise ValueError(f"tensor {uid} shape or strides are invalid")
        result[uid] = {
            "uid": uid,
            "virtual": is_virtual,
            "alignment": alignment,
            "data_type": data_type,
            "dimensions": dimensions,
            "strides": strides,
        }
    return result


def _tensor_metadata(
    node: dict[str, Any],
    operation_name: str,
    tensor_registry: dict[int, dict[str, Any]],
) -> tuple[list[int], list[dict[str, Any]], int]:
    expected_roles = EXPECTED_TENSOR_ROLES.get(operation_name)
    if expected_roles is None:
        raise ValueError(f"unsupported operation: {operation_name!r}")
    if operation_name == "moe_grouped_matmul":
        attributes = _require_object(node.get("attributes"), "node.attributes")
        mode = _require_integer(attributes, "mode", minimum=0, maximum=2)
        expected_inputs = (
            ("token", "weight", "first_token_offset")
            + (("token_index",) if mode else ())
            + (("token_ks",) if mode == 2 else ())
        )
        expected_outputs = ("output",)
    elif operation_name == "matmul_fp8":
        attributes = _require_object(node.get("attributes"), "node.attributes")
        mode = _require_integer(attributes, "scale_mode", minimum=0, maximum=2)
        expected_inputs = ("a", "b") + (("descale_a", "descale_b") if mode else ())
        expected_outputs = ("output",)
    elif operation_name == "causal_conv1d":
        attributes = _require_object(node.get("attributes"), "node.attributes")
        bias = _require_integer(attributes, "has_bias", minimum=0, maximum=1) == 1
        expected_inputs = ("input", "weight", "bias") if bias else ("input", "weight")
        expected_outputs = ("output",)
    elif operation_name == "resample":
        attributes = _require_object(node.get("attributes"), "node.attributes")
        index = (
            _require_integer(attributes, "generate_index", minimum=0, maximum=1) == 1
        )
        expected_inputs = ("input",)
        expected_outputs = ("output", "index") if index else ("output",)
    elif operation_name == "bn_finalize":
        attributes = _require_object(node.get("attributes"), "node.attributes")
        running = _require_integer(attributes, "has_running", minimum=0, maximum=1) == 1
        expected_inputs = ("sum", "sq_sum", "scale", "bias") + (
            ("previous_running_mean", "previous_running_variance") if running else ()
        )
        expected_outputs = ("eq_scale", "eq_bias", "mean", "inv_variance") + (
            ("next_running_mean", "next_running_variance") if running else ()
        )
    elif operation_name in {"gen_index", "rng"}:
        expected_inputs, expected_outputs = (), ("output",)
    elif operation_name == "concatenate":
        attributes = _require_object(node.get("attributes"), "node.attributes")
        count = _require_integer(attributes, "input_count", minimum=1, maximum=65536)
        expected_inputs = tuple(f"input_{index}" for index in range(count))
        expected_outputs = ("output",)
    elif operation_name in {
        "sdpa",
        "sdpa_backward",
        "sdpa_fp8",
        "sdpa_fp8_backward",
    }:
        attributes = _require_object(node.get("attributes"), "node.attributes")
        has_bias = _require_integer(attributes, "has_bias", minimum=0, maximum=1) == 1
        if operation_name == "sdpa":
            expected_inputs = ("q", "k", "v") + (("bias",) if has_bias else ())
            expected_outputs = ("o", "stats")
        elif operation_name == "sdpa_backward":
            has_dbias = (
                _require_integer(attributes, "has_dbias", minimum=0, maximum=1) == 1
            )
            expected_inputs = (
                "q",
                "k",
                "v",
                "o",
                "do",
                "stats",
            ) + (("bias",) if has_bias else ())
            expected_outputs = ("dq", "dk", "dv") + (("dbias",) if has_dbias else ())
        elif operation_name == "sdpa_fp8":
            expected_inputs = (
                "q",
                "k",
                "v",
                "descale_q",
                "descale_k",
                "descale_v",
                "descale_s",
                "scale_s",
                "scale_o",
            ) + (("bias",) if has_bias else ())
            expected_outputs = ("o", "stats", "amax_s", "amax_o")
        else:
            if has_bias:
                raise ValueError("FP8 SDPA backward bias is unsupported")
            has_dbias = (
                _require_integer(attributes, "has_dbias", minimum=0, maximum=1) == 1
            )
            if has_dbias:
                raise ValueError("FP8 SDPA backward dBias is unsupported")
            expected_inputs = (
                "q",
                "k",
                "v",
                "o",
                "do",
                "stats",
                "descale_q",
                "descale_k",
                "descale_v",
                "descale_o",
                "descale_do",
                "descale_s",
                "descale_dp",
                "scale_s",
                "scale_dq",
                "scale_dk",
                "scale_dv",
                "scale_dp",
            )
            expected_outputs = (
                "dq",
                "dk",
                "dv",
                "amax_dq",
                "amax_dk",
                "amax_dv",
                "amax_dp",
            )
    else:
        output_count = EXPECTED_OUTPUT_COUNTS.get(operation_name, 1)
        if output_count <= 0 or output_count >= len(expected_roles):
            raise ValueError("operation output role count is invalid")
        expected_inputs = expected_roles[:-output_count]
        expected_outputs = expected_roles[-output_count:]
    inputs = _require_list(node.get("inputs"), "node.inputs")
    outputs = _require_list(node.get("outputs"), "node.outputs")
    if len(inputs) != len(expected_inputs) or len(outputs) != len(expected_outputs):
        raise ValueError("node port count is invalid")

    tensor_uids: list[int] = []
    metadata: list[dict[str, Any]] = []
    for direction, ports, roles in (
        ("input", inputs, expected_inputs),
        ("output", outputs, expected_outputs),
    ):
        for index, expected_role in enumerate(roles):
            port = _require_object(ports[index], f"node.{direction}s[{index}]")
            if port.get("name") != expected_role:
                raise ValueError(f"{direction} port {index} does not match operation")
            optional = port.get("optional", False)
            if not isinstance(optional, bool) or optional:
                raise ValueError(
                    "the Ascend provider does not support absent optional " "ports"
                )
            uid = port.get("uid")
            if isinstance(uid, bool) or not isinstance(uid, int) or uid <= 0:
                raise ValueError(f"{direction} port UID {index} is invalid")
            try:
                tensor = tensor_registry[uid]
            except KeyError as error:
                raise ValueError(
                    f"{direction} port references unknown tensor UID {uid}"
                ) from error
            tensor_uids.append(uid)
            metadata.append(tensor)
    return tensor_uids, metadata, len(expected_inputs)
