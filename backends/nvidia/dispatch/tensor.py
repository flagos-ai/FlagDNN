"""Graph tensor parsing and runtime argument ABI construction."""

from __future__ import annotations

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
    for index, tensor_value in enumerate(tensors):
        tensor = _require_object(tensor_value, f"graph.tensors[{index}]")
        uid = tensor.get("uid")
        if isinstance(uid, bool) or not isinstance(uid, int) or uid <= 0:
            raise ValueError(f"tensor UID {index} is invalid")
        if uid in result:
            raise ValueError("graph tensor UIDs must be unique")
        data_type = tensor.get("data_type")
        if (
            not isinstance(data_type, str)
            or data_type not in TRITON_POINTER_TYPES
        ):
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
            raise ValueError(
                f"tensor {uid} alignment must be a positive power of two"
            )
        dimensions = _require_list(
            tensor.get("dimensions"), f"tensor {uid} dimensions"
        )
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
        expected_inputs = ("a", "b") + (
            ("descale_a", "descale_b") if mode else ()
        )
        expected_outputs = ("output",)
    elif operation_name == "causal_conv1d":
        attributes = _require_object(node.get("attributes"), "node.attributes")
        bias = (
            _require_integer(attributes, "has_bias", minimum=0, maximum=1) == 1
        )
        expected_inputs = (
            ("input", "weight", "bias") if bias else ("input", "weight")
        )
        expected_outputs = ("output",)
    elif operation_name == "resample":
        attributes = _require_object(node.get("attributes"), "node.attributes")
        index = (
            _require_integer(
                attributes, "generate_index", minimum=0, maximum=1
            )
            == 1
        )
        expected_inputs = ("input",)
        expected_outputs = ("output", "index") if index else ("output",)
    elif operation_name == "bn_finalize":
        attributes = _require_object(node.get("attributes"), "node.attributes")
        running = (
            _require_integer(attributes, "has_running", minimum=0, maximum=1)
            == 1
        )
        expected_inputs = ("sum", "sq_sum", "scale", "bias") + (
            ("previous_running_mean", "previous_running_variance")
            if running
            else ()
        )
        expected_outputs = ("eq_scale", "eq_bias", "mean", "inv_variance") + (
            ("next_running_mean", "next_running_variance") if running else ()
        )
    elif operation_name in {"gen_index", "rng"}:
        expected_inputs, expected_outputs = (), ("output",)
    elif operation_name == "concatenate":
        attributes = _require_object(node.get("attributes"), "node.attributes")
        count = _require_integer(
            attributes, "input_count", minimum=1, maximum=65536
        )
        expected_inputs = tuple(f"input_{index}" for index in range(count))
        expected_outputs = ("output",)
    elif operation_name in {
        "sdpa",
        "sdpa_backward",
        "sdpa_fp8",
        "sdpa_fp8_backward",
    }:
        attributes = _require_object(node.get("attributes"), "node.attributes")
        has_bias = (
            _require_integer(attributes, "has_bias", minimum=0, maximum=1) == 1
        )
        if operation_name == "sdpa":
            expected_inputs = ("q", "k", "v") + (("bias",) if has_bias else ())
            expected_outputs = ("o", "stats")
        elif operation_name == "sdpa_backward":
            has_dbias = (
                _require_integer(attributes, "has_dbias", minimum=0, maximum=1)
                == 1
            )
            expected_inputs = (
                "q",
                "k",
                "v",
                "o",
                "do",
                "stats",
            ) + (("bias",) if has_bias else ())
            expected_outputs = ("dq", "dk", "dv") + (
                ("dbias",) if has_dbias else ()
            )
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
                _require_integer(attributes, "has_dbias", minimum=0, maximum=1)
                == 1
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
    if len(inputs) != len(expected_inputs) or len(outputs) != len(
        expected_outputs
    ):
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
                raise ValueError(
                    f"{direction} port {index} does not match operation"
                )
            optional = port.get("optional", False)
            if not isinstance(optional, bool) or optional:
                raise ValueError(
                    "the NVIDIA provider does not support absent optional "
                    "ports"
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


def _build_argument_abi(
    layout: list[tuple[str, str | int | None]],
    tensors: list[dict[str, Any]],
    parameters: dict[str, Any],
    workspace_layout: dict[int, tuple[int, int]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    tensor_index = 0
    consumed_tensor_indices: set[int] = set()
    for kind, name in layout:
        if kind == "tensor_map" and isinstance(name, str):
            if tensor_index >= len(tensors):
                raise ValueError("kernel ABI requests too many TensorMaps")
            tensor = tensors[tensor_index]
            dims = tensor["dimensions"]
            if (
                tensor["virtual"]
                or tensor["data_type"] != "float32"
                or len(dims) != 3
                or tensor["strides"] != [dims[1] * dims[2], dims[2], 1]
            ):
                raise ValueError(
                    "host TensorMap requires an external "
                    "contiguous FP32 tensor"
                )
            result.append(
                {
                    "kind": "tensor_map",
                    "uid": tensor["uid"],
                    "size": _tensor_storage_size(tensor),
                    "alignment": tensor.get("alignment", 16),
                    "data_type": "tf32_rne",
                    "shape": [dims[0] * dims[1], dims[2]],
                    "strides": [dims[2], 1],
                    "block_shape": name.split(","),
                }
            )
            consumed_tensor_indices.add(tensor_index)
            tensor_index += 1
        elif kind == "tensor":
            if tensor_index >= len(tensors):
                raise ValueError(
                    "kernel ABI requests too many tensor arguments"
                )
            tensor = tensors[tensor_index]
            consumed_tensor_indices.add(tensor_index)
            uid = tensor["uid"]
            if tensor["virtual"]:
                offset, size = workspace_layout[uid]
                result.append(
                    {
                        "kind": "workspace_tensor",
                        "uid": uid,
                        "offset": offset,
                        "size": size,
                    }
                )
            else:
                result.append(
                    {
                        "kind": "tensor",
                        "uid": uid,
                        "size": _tensor_storage_size(tensor),
                        "alignment": tensor.get("alignment", 16),
                    }
                )
            tensor_index += 1
        elif kind == "tensor_alias" and isinstance(name, int):
            if name < -len(tensors) or name >= len(tensors):
                raise ValueError("kernel ABI tensor alias is out of range")
            alias_index = name if name >= 0 else len(tensors) + name
            tensor = tensors[alias_index]
            consumed_tensor_indices.add(alias_index)
            uid = tensor["uid"]
            if tensor["virtual"]:
                offset, size = workspace_layout[uid]
                result.append(
                    {
                        "kind": "workspace_tensor",
                        "uid": uid,
                        "offset": offset,
                        "size": size,
                    }
                )
            else:
                result.append(
                    {
                        "kind": "tensor",
                        "uid": uid,
                        "size": _tensor_storage_size(tensor),
                        "alignment": tensor.get("alignment", 16),
                    }
                )
        elif kind == "scalar_i32" and name is not None:
            result.append(
                {
                    "kind": "scalar_i32",
                    "name": name,
                    "value": _require_integer(
                        parameters,
                        name,
                        minimum=-(2**31),
                        maximum=2**31 - 1,
                    ),
                }
            )
        elif kind == "scalar_f32" and name is not None:
            result.append(
                {
                    "kind": "scalar_f32",
                    "name": name,
                    "value": _require_number(parameters, name),
                }
            )
        else:
            raise ValueError("kernel ABI layout is invalid")
    if consumed_tensor_indices != set(range(len(tensors))):
        raise ValueError("kernel ABI does not consume every tensor")
    result.extend(
        [
            {"kind": "global_scratch_pointer"},
            {"kind": "profile_scratch_pointer"},
        ]
    )
    return result
