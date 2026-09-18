"""Target-independent shape and port rules owned by the MThreads dispatcher.

Mirrors the public Graph contracts used by the NVIDIA dispatcher. GPU policies
and code emission remain local to MThreads.
"""

from __future__ import annotations

import math
from typing import Any

EXPECTED_TENSOR_ROLES = {
    "relu_backward": ("left", "right", "output"),
    "tanh_backward": ("left", "right", "output"),
    "elu_backward": ("left", "right", "output"),
    "gelu_backward": ("left", "right", "output"),
    "softplus_backward": ("left", "right", "output"),
    "swish_backward": ("left", "right", "output"),
    "gelu_approx_tanh_backward": ("left", "right", "output"),
    "moe_grouped_matmul": ("token", "weight", "first_token_offset", "output"),
    "moe_grouped_matmul_bwd": (
        "doutput",
        "token",
        "first_token_offset",
        "dweight",
    ),
    "matmul_fp8": ("a", "b", "output"),
    "causal_conv1d": ("input", "weight", "output"),
    "resample": ("input", "output"),
    "rng": ("output",),
    "rope": ("input", "freqs", "output"),
    "rope_backward": ("dy", "freqs", "dx"),
    "bn_finalize": (
        "sum",
        "sq_sum",
        "scale",
        "bias",
        "eq_scale",
        "eq_bias",
        "mean",
        "inv_variance",
    ),
    "instancenorm": ("x", "scale", "bias", "y", "mean", "inv_variance"),
    "adalayernorm": ("x", "scale", "bias", "y", "mean", "inv_variance"),
    "instancenorm_backward": (
        "dy",
        "x",
        "scale",
        "mean",
        "inv_variance",
        "dx",
        "dscale",
        "dbias",
    ),
    "adalayernorm_backward": (
        "dy",
        "x",
        "scale",
        "mean",
        "inv_variance",
        "dx",
        "dscale",
        "dbias",
    ),
    "layernorm_backward": (
        "dy",
        "x",
        "scale",
        "mean",
        "inv_variance",
        "dx",
        "dscale",
        "dbias",
    ),
    "rmsnorm_backward": (
        "dy",
        "x",
        "scale",
        "inv_variance",
        "dx",
        "dscale",
        "dbias",
    ),
    "batchnorm_backward": (
        "dy",
        "x",
        "scale",
        "mean",
        "inv_variance",
        "dx",
        "dscale",
        "dbias",
    ),
    "genstats": ("x", "sum", "sq_sum"),
    "gen_index": ("output",),
    "concatenate": ("output",),
    "relu": ("input", "output"),
    "add": ("left", "right", "output"),
    "reduction_sum": ("input", "output"),
    "reduction_avg": ("input", "output"),
    "reduction_mul": ("input", "output"),
    "conv2d_fprop": ("input", "filter", "output"),
    "convolution_fprop": ("input", "filter", "output"),
    "convolution_dgrad": ("dy", "w", "dx"),
    "convolution_wgrad": ("dy", "x", "dw"),
    "matmul": ("a", "b", "output"),
    "reshape": ("input", "output"),
    "transpose": ("input", "output"),
    "slice": ("input", "output"),
    "layernorm": ("x", "scale", "bias", "y", "mean", "inv_variance"),
    "rmsnorm": ("x", "scale", "bias", "y", "inv_variance"),
    "batchnorm": (
        "x",
        "scale",
        "bias",
        "previous_running_mean",
        "previous_running_variance",
        "y",
        "mean",
        "inv_variance",
        "next_running_mean",
        "next_running_variance",
    ),
    "batchnorm_inference": (
        "x",
        "mean",
        "inv_variance",
        "scale",
        "bias",
        "y",
    ),
    "sdpa": ("q", "k", "v", "bias", "o", "stats"),
    "sdpa_backward": (
        "q",
        "k",
        "v",
        "o",
        "do",
        "stats",
        "bias",
        "dq",
        "dk",
        "dv",
        "dbias",
    ),
    "sdpa_fp8": (
        "q",
        "k",
        "v",
        "descale_q",
        "descale_k",
        "descale_v",
        "descale_s",
        "scale_s",
        "scale_o",
        "bias",
        "o",
        "stats",
        "amax_s",
        "amax_o",
    ),
    "sdpa_fp8_backward": (
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
        "dq",
        "dk",
        "dv",
        "amax_dq",
        "amax_dk",
        "amax_dv",
        "amax_dp",
    ),
}

EXPECTED_OUTPUT_COUNTS = {
    "instancenorm": 3,
    "adalayernorm": 3,
    "instancenorm_backward": 3,
    "adalayernorm_backward": 3,
    "layernorm_backward": 3,
    "rmsnorm_backward": 3,
    "batchnorm_backward": 3,
    "genstats": 2,
    "layernorm": 3,
    "rmsnorm": 2,
    "batchnorm": 5,
    "sdpa": 2,
}

TRITON_POINTER_TYPES = {
    "float32": "*fp32",
    "int32": "*i32",
    "float16": "*fp16",
    "bfloat16": "*bf16",
    "boolean": "*i8",
    "fp8_e4m3": "*fp8e4nv",
    "fp8_e5m2": "*fp8e5",
    "fp8_e8m0": "*u8",
}

FLOAT_DATA_TYPES = {"float32", "float16", "bfloat16"}

NUMERIC_DATA_TYPES = FLOAT_DATA_TYPES | {"int32"}

FP8_DATA_TYPES = {"fp8_e4m3", "fp8_e5m2"}

COMPARISON_POINTWISE_OPERATIONS = {
    "cmp_eq",
    "cmp_neq",
    "cmp_gt",
    "cmp_ge",
    "cmp_lt",
    "cmp_le",
}

LOGICAL_BINARY_POINTWISE_OPERATIONS = {"logical_and", "logical_or"}

UNARY_POINTWISE_MODES = {
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
    "logical_not": 24,
    "sigmoid": 33,
    "tanh": 34,
    "elu": 35,
    "gelu": 36,
    "softplus": 37,
    "swish": 38,
    "gelu_approx_tanh": 39,
}

BINARY_POINTWISE_MODES = {
    "add": 1,
    "sub": 17,
    "mul": 18,
    "div": 19,
    "min": 20,
    "max": 21,
    "mod": 22,
    "pow": 23,
    "cmp_eq": 25,
    "cmp_neq": 26,
    "cmp_gt": 27,
    "cmp_ge": 28,
    "cmp_lt": 29,
    "cmp_le": 30,
    "logical_and": 31,
    "logical_or": 32,
    "sigmoid_backward": 40,
    "relu_backward": 42,
    "tanh_backward": 43,
    "elu_backward": 44,
    "gelu_backward": 45,
    "softplus_backward": 46,
    "swish_backward": 47,
    "gelu_approx_tanh_backward": 48,
}


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


_FLOAT32_MAX = 3.4028234663852886e38


def _require_number(
    values: dict[str, Any],
    name: str,
    *,
    default: float | None = None,
) -> float:
    value = values.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"parameters.{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or abs(result) > _FLOAT32_MAX:
        raise ValueError(
            f"parameters.{name} must be finite and representable as float32"
        )
    return result


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


def _is_physically_dense(tensor: dict[str, Any]) -> bool:
    dimensions = tensor["dimensions"]
    strides = tensor["strides"]
    return _has_non_overlapping_strides(dimensions, strides) and 1 + sum(
        (dimension - 1) * stride
        for dimension, stride in zip(dimensions, strides)
    ) == math.prod(dimensions)


def _unary_pointwise_tensor_constants(
    tensors: list[dict[str, Any]],
) -> dict[str, int | bool]:
    if len(tensors) != 2:
        raise ValueError("Pointwise tensor count is invalid")
    input_tensor, output_tensor = tensors
    if input_tensor["dimensions"] != output_tensor["dimensions"]:
        raise ValueError("Pointwise input/output shapes must match")
    for tensor in tensors:
        if not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        ):
            raise ValueError(
                "Pointwise tensors must have non-overlapping strides"
            )

    dimensions = input_tensor["dimensions"]

    same_mapping = all(
        dimension == 1 or input_stride == output_stride
        for dimension, input_stride, output_stride in zip(
            dimensions,
            input_tensor["strides"],
            output_tensor["strides"],
        )
    )
    use_strided = not (
        same_mapping
        and _is_physically_dense(input_tensor)
        and _is_physically_dense(output_tensor)
    )

    leading = 8 - len(dimensions)
    padded_dimensions = [1] * leading + dimensions
    input_strides = [0] * leading + input_tensor["strides"]
    output_strides = [0] * leading + output_tensor["strides"]
    constants: dict[str, int | bool] = {"STRIDED": use_strided}
    for axis in range(8):
        constants[f"DIM_{axis}"] = padded_dimensions[axis]
        constants[f"INPUT_STRIDE_{axis}"] = input_strides[axis]
        constants[f"OUTPUT_STRIDE_{axis}"] = output_strides[axis]
    return constants


def _binary_pointwise_tensor_constants(
    tensors: list[dict[str, Any]],
) -> dict[str, int]:
    if len(tensors) != 3:
        raise ValueError("binary pointwise tensor count is invalid")
    left, right, output = tensors
    output_dimensions = output["dimensions"]
    rank = max(len(left["dimensions"]), len(right["dimensions"]))
    if rank < 1 or rank > 8:
        raise ValueError("binary pointwise rank must be in [1, 8]")
    if len(output_dimensions) != rank:
        raise ValueError(
            "binary pointwise output rank does not match broadcast result"
        )

    broadcast_dimensions = [1] * rank
    for trailing in range(rank):
        left_dimension = (
            left["dimensions"][-1 - trailing]
            if trailing < len(left["dimensions"])
            else 1
        )
        right_dimension = (
            right["dimensions"][-1 - trailing]
            if trailing < len(right["dimensions"])
            else 1
        )
        if (
            left_dimension != right_dimension
            and left_dimension != 1
            and right_dimension != 1
        ):
            raise ValueError(
                "binary pointwise input shapes are not broadcast-compatible"
            )
        broadcast_dimensions[-1 - trailing] = max(
            left_dimension, right_dimension
        )
    if output_dimensions != broadcast_dimensions:
        raise ValueError(
            "binary pointwise output shape does not match broadcast result"
        )

    for tensor in tensors:
        if not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        ):
            raise ValueError(
                "binary pointwise tensors must have non-overlapping strides"
            )

    def effective_strides(tensor: dict[str, Any]) -> list[int]:
        leading = rank - len(tensor["dimensions"])
        dimensions = [1] * leading + tensor["dimensions"]
        strides = [0] * leading + tensor["strides"]
        return [
            0 if dimension == 1 else stride
            for dimension, stride in zip(dimensions, strides)
        ]

    leading = 8 - rank
    dimensions = [1] * leading + output_dimensions
    left_strides = [0] * leading + effective_strides(left)
    right_strides = [0] * leading + effective_strides(right)
    output_strides = [0] * leading + output["strides"]
    constants: dict[str, int] = {}
    for axis in range(8):
        constants[f"DIM_{axis}"] = dimensions[axis]
    for prefix, values in (
        ("LEFT_STRIDE", left_strides),
        ("RIGHT_STRIDE", right_strides),
        ("OUTPUT_STRIDE", output_strides),
    ):
        for axis in range(8):
            constants[f"{prefix}_{axis}"] = values[axis]
    return constants


def _can_use_dense_binary_kernel(
    tensors: list[dict[str, Any]],
) -> bool:
    if len(tensors) != 3:
        return False
    left, right, output = tensors
    if not (
        left["dimensions"] == right["dimensions"]
        and left["dimensions"] == output["dimensions"]
        and left["strides"] == right["strides"]
        and left["strides"] == output["strides"]
    ):
        return False

    return all(_is_physically_dense(tensor) for tensor in tensors)


def _ternary_pointwise_tensor_constants(
    tensors: list[dict[str, Any]],
) -> dict[str, int]:
    if len(tensors) != 4:
        raise ValueError("ternary pointwise tensor count is invalid")
    inputs = tensors[:3]
    output = tensors[3]
    rank = max(len(tensor["dimensions"]) for tensor in inputs)
    if rank < 1 or rank > 8:
        raise ValueError("ternary pointwise rank must be in [1, 8]")

    broadcast_dimensions = [1] * rank
    for tensor in inputs:
        for trailing in range(rank):
            dimension = (
                tensor["dimensions"][-1 - trailing]
                if trailing < len(tensor["dimensions"])
                else 1
            )
            current = broadcast_dimensions[-1 - trailing]
            if dimension != current and dimension != 1 and current != 1:
                raise ValueError(
                    "ternary pointwise input shapes are not "
                    "broadcast-compatible"
                )
            broadcast_dimensions[-1 - trailing] = max(current, dimension)
    if output["dimensions"] != broadcast_dimensions:
        raise ValueError(
            "ternary pointwise output shape does not match broadcast result"
        )

    for tensor in tensors:
        if not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        ):
            raise ValueError(
                "ternary pointwise tensors must have non-overlapping strides"
            )

    def effective_strides(tensor: dict[str, Any]) -> list[int]:
        leading = rank - len(tensor["dimensions"])
        dimensions = [1] * leading + tensor["dimensions"]
        strides = [0] * leading + tensor["strides"]
        return [
            0 if dimension == 1 else stride
            for dimension, stride in zip(dimensions, strides)
        ]

    leading = 8 - rank
    dimensions = [1] * leading + output["dimensions"]
    input_strides = [
        [0] * leading + effective_strides(tensor) for tensor in inputs
    ]
    output_strides = [0] * leading + output["strides"]
    constants: dict[str, int] = {}
    for axis in range(8):
        constants[f"DIM_{axis}"] = dimensions[axis]
    for prefix, values in zip(
        ("LEFT_STRIDE", "RIGHT_STRIDE", "MASK_STRIDE"),
        input_strides,
    ):
        for axis in range(8):
            constants[f"{prefix}_{axis}"] = values[axis]
    for axis in range(8):
        constants[f"OUTPUT_STRIDE_{axis}"] = output_strides[axis]
    return constants


def _can_use_dense_ternary_kernel(
    tensors: list[dict[str, Any]],
) -> bool:
    if len(tensors) != 4:
        return False
    output = tensors[-1]
    return all(
        tensor["dimensions"] == output["dimensions"]
        and tensor["strides"] == output["strides"]
        and _is_physically_dense(tensor)
        for tensor in tensors
    )


def _is_row_major_contiguous(tensor: dict[str, Any]) -> bool:
    expected = 1
    for dimension, stride in zip(
        reversed(tensor["dimensions"]), reversed(tensor["strides"])
    ):
        if stride != expected:
            return False
        expected *= dimension
    return True


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
                    "the MThreads provider does not support absent optional "
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
