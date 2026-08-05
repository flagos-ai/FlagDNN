"""CUDA/Triton compiler provider for FlagDNN Graph IR."""

from __future__ import annotations

import hashlib
import copy
import itertools
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import triton
import yaml  # type: ignore[import-untyped]
from triton.backends.compiler import GPUTarget

from .compiler_identity import build_compiler_identity

from flagdnn_codegen.kernel_registry import (
    BINARY_POINTWISE_OPERATIONS,
    KernelCandidate,
    TERNARY_POINTWISE_OPERATIONS,
    UNARY_POINTWISE_OPERATIONS,
    materialize_kernel_source,
    resolve_kernel_source,
    resolve_tuning_source,
    select_kernel_candidate,
)

SCHEMA_VERSION = 3
ARTIFACT_SCHEMA_VERSION = 4
EXECUTION_PROGRAM_VERSION = 2
PROVIDER_NAME = "nvidia_triton"
PROVIDER_VERSION = "1"
LIBTRITON_JIT_GLOBAL_SCRATCH_SIZE = 4096

EXPECTED_TENSOR_ROLES = {
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
    "layernorm": 3,
    "rmsnorm": 2,
    "batchnorm": 5,
    "sdpa": 2,
}

for _operation in UNARY_POINTWISE_OPERATIONS:
    EXPECTED_TENSOR_ROLES[_operation] = ("input", "output")
for _operation in BINARY_POINTWISE_OPERATIONS:
    EXPECTED_TENSOR_ROLES[_operation] = ("left", "right", "output")
for _operation in TERNARY_POINTWISE_OPERATIONS:
    EXPECTED_TENSOR_ROLES[_operation] = ("a", "b", "t", "output")


REDUCTION_OPERATIONS = {
    "reduction_sum": 1,
    "reduction_avg": 2,
    "reduction_mul": 3,
}

TRITON_POINTER_TYPES = {
    "float32": "*fp32",
    "float16": "*fp16",
    "bfloat16": "*bf16",
    "boolean": "*i8",
    "fp8_e4m3": "*fp8e4nv",
    "fp8_e5m2": "*fp8e5",
}

FLOAT_DATA_TYPES = {"float32", "float16", "bfloat16"}
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


def _require_integer_list(
    values: dict[str, Any],
    name: str,
    length: int,
    *,
    minimum: int,
    maximum: int = 2**31 - 1,
) -> list[int]:
    result = _require_list(values.get(name), f"parameters.{name}")
    if len(result) != length:
        raise ValueError(f"parameters.{name} must contain {length} integers")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
        for value in result
    ):
        raise ValueError(
            f"parameters.{name} values must be integers in "
            f"[{minimum}, {maximum}]"
        )
    return result


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


def _reduction_tensor_constants(
    tensors: list[dict[str, Any]], axis: int, keep_dimensions: bool
) -> dict[str, int]:
    if len(tensors) != 2:
        raise ValueError("Reduction tensor count is invalid")
    input_tensor, output_tensor = tensors
    input_dimensions = input_tensor["dimensions"]
    rank = len(input_dimensions)
    if rank == 0 or rank > 8 or axis < 0 or axis >= rank:
        raise ValueError("Reduction axis or rank is invalid")
    for tensor in tensors:
        if not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        ):
            raise ValueError(
                "Reduction tensors must have non-overlapping strides"
            )

    expected_dimensions = list(input_dimensions)
    if keep_dimensions:
        expected_dimensions[axis] = 1
    else:
        del expected_dimensions[axis]
    if output_tensor["dimensions"] != expected_dimensions:
        raise ValueError("Reduction output shape is incorrect")

    logical_dimensions = list(input_dimensions)
    logical_dimensions[axis] = 1
    input_strides = list(input_tensor["strides"])
    reduction_stride = input_strides[axis]
    input_strides[axis] = 0

    if keep_dimensions:
        output_strides = list(output_tensor["strides"])
    else:
        output_strides = []
        output_axis = 0
        for input_axis in range(rank):
            if input_axis == axis:
                output_strides.append(0)
            else:
                output_strides.append(output_tensor["strides"][output_axis])
                output_axis += 1
    output_strides[axis] = 0

    leading = 8 - rank
    dimensions = [1] * leading + logical_dimensions
    input_strides = [0] * leading + input_strides
    output_strides = [0] * leading + output_strides
    constants: dict[str, int] = {"REDUCTION_STRIDE": reduction_stride}
    for padded_axis in range(8):
        constants[f"DIM_{padded_axis}"] = dimensions[padded_axis]
        constants[f"INPUT_STRIDE_{padded_axis}"] = input_strides[padded_axis]
        constants[f"OUTPUT_STRIDE_{padded_axis}"] = output_strides[padded_axis]
    return constants


def _load_generated_module(path: Path, operation_index: int = 0):
    module_name = f"_flagdnn_generated_{os.getpid()}_{operation_index}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import generated kernel module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _convolution_im2col_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    """Configure the first stage of the NVIDIA stride-2 FProp pipeline."""
    if len(tensors) != 2:
        raise ValueError("the FProp im2col stage requires two tensors")
    input_tensor, columns = tensors
    data_type_pair = (
        input_tensor["data_type"],
        columns["data_type"],
    )
    if (
        data_type_pair
        not in {
            ("float16", "float16"),
            ("bfloat16", "bfloat16"),
            ("float32", "float16"),
            ("float32", "float32"),
        }
        or len(input_tensor["dimensions"]) != 4
        or len(columns["dimensions"]) != 3
        or not _is_row_major_contiguous(input_tensor)
        or not _has_non_overlapping_strides(
            columns["dimensions"], columns["strides"]
        )
        or columns["strides"][-1] != 1
    ):
        raise ValueError("the FProp im2col tensor metadata is invalid")
    if (
        _require_integer(parameters, "spatial_rank", minimum=1, maximum=3) != 2
        or _require_integer(parameters, "groups") != 1
        or _require_integer_list(parameters, "stride", 2, minimum=1) != [2, 2]
        or _require_integer_list(parameters, "pre_padding", 2, minimum=0)
        != [1, 1]
        or _require_integer_list(parameters, "post_padding", 2, minimum=0)
        != [1, 1]
        or _require_integer_list(parameters, "dilation", 2, minimum=1)
        != [1, 1]
    ):
        raise ValueError("the FProp im2col convolution contract is invalid")

    n, channels, input_h, input_w = input_tensor["dimensions"]
    output_h = (input_h + 1) // 2
    output_w = (input_w + 1) // 2
    output_area = output_h * output_w
    reduction_extent = channels * 9
    if (
        columns["dimensions"][:2] != [n, reduction_extent]
        or columns["dimensions"][2] < output_area
        or columns["strides"]
        != [
            reduction_extent * columns["dimensions"][2],
            columns["dimensions"][2],
            1,
        ]
    ):
        raise ValueError("the FProp im2col workspace shape is invalid")

    total = n * columns["strides"][0]
    block_size = 1024
    constants = {
        "TOTAL": total,
        "XH": input_h,
        "XW": input_w,
        "OH": output_h,
        "OW": output_w,
        "CIN_PER_GROUP": channels,
        "X_STRIDE_N": input_tensor["strides"][0],
        "X_STRIDE_C": input_tensor["strides"][1],
        "X_STRIDE_H": input_tensor["strides"][2],
        "X_STRIDE_W": input_tensor["strides"][3],
        "COL_STRIDE_N": columns["strides"][0],
        "COL_STRIDE_K": columns["strides"][1],
        "BLOCK_SIZE": block_size,
    }
    return (
        "conv2d_im2col_nchw_3x3_stride2_pad1_kernel",
        {
            "x_ptr": TRITON_POINTER_TYPES[input_tensor["data_type"]],
            "col_ptr": TRITON_POINTER_TYPES[columns["data_type"]],
        },
        constants,
        (_ceil_div(output_area, block_size), n * channels * 3, 1),
        [("tensor", None), ("tensor", None)],
    )


def _convolution_general_im2col_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | bool],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    """Configure the materialization stage for the general 2D FProp path."""
    if len(tensors) not in {3, 4}:
        raise ValueError(
            "the general FProp im2col stage requires 3 or 4 tensors"
        )
    input_tensor, weight, columns = tensors[:3]
    converted_weight = tensors[3] if len(tensors) == 4 else weight
    convert_weight = input_tensor["data_type"] == "float32"
    if (
        input_tensor["data_type"] not in {"float32", "float16", "bfloat16"}
        or weight["data_type"] != input_tensor["data_type"]
        or columns["data_type"]
        != ("float16" if convert_weight else input_tensor["data_type"])
        or converted_weight["data_type"] != columns["data_type"]
        or len(input_tensor["dimensions"]) != 4
        or len(weight["dimensions"]) != 4
        or len(columns["dimensions"]) != 3
        or not _is_row_major_contiguous(input_tensor)
        or not _is_row_major_contiguous(weight)
        or not _is_row_major_contiguous(columns)
        or not _is_row_major_contiguous(converted_weight)
    ):
        raise ValueError("the general FProp im2col tensor metadata is invalid")
    if convert_weight != (len(tensors) == 4):
        raise ValueError(
            "the FP32 FProp im2col stage requires converted weights"
        )
    if (
        _require_integer(parameters, "spatial_rank", minimum=1, maximum=3) != 2
        or _require_integer(parameters, "groups") != 1
    ):
        raise ValueError(
            "the general FProp im2col stage requires 2D group-one convolution"
        )

    stride = _require_integer_list(parameters, "stride", 2, minimum=1)
    pre_padding = _require_integer_list(
        parameters, "pre_padding", 2, minimum=0
    )
    post_padding = _require_integer_list(
        parameters, "post_padding", 2, minimum=0
    )
    dilation = _require_integer_list(parameters, "dilation", 2, minimum=1)
    n, channels, input_h, input_w = input_tensor["dimensions"]
    output_channels, filter_channels, kernel_h, kernel_w = weight["dimensions"]
    if filter_channels != channels:
        raise ValueError("the general FProp filter channel count is invalid")
    output_h = (
        input_h
        + pre_padding[0]
        + post_padding[0]
        - dilation[0] * (kernel_h - 1)
        - 1
    ) // stride[0] + 1
    output_w = (
        input_w
        + pre_padding[1]
        + post_padding[1]
        - dilation[1] * (kernel_w - 1)
        - 1
    ) // stride[1] + 1
    output_area = output_h * output_w
    reduction_extent = channels * kernel_h * kernel_w
    if columns["dimensions"] != [n, reduction_extent, output_area]:
        raise ValueError("the general FProp im2col workspace shape is invalid")
    if converted_weight["dimensions"] != weight["dimensions"]:
        raise ValueError(
            "the converted FProp weight workspace shape is invalid"
        )

    block_size = 1024
    weight_total = output_channels * reduction_extent
    pointer_abi: list[tuple[str, str | int | None]] = [
        ("tensor", None),
        ("tensor_alias", 2),
        ("tensor", None),
        ("tensor_alias", 3 if convert_weight else 1),
    ]
    return (
        "conv2d_im2col_nchw_kernel",
        {
            "x_ptr": TRITON_POINTER_TYPES[input_tensor["data_type"]],
            "col_ptr": TRITON_POINTER_TYPES[columns["data_type"]],
            "weight_ptr": TRITON_POINTER_TYPES[weight["data_type"]],
            "converted_weight_ptr": TRITON_POINTER_TYPES[
                converted_weight["data_type"]
            ],
        },
        {
            "XH": input_h,
            "XW": input_w,
            "OH": output_h,
            "OW": output_w,
            "CIN_PER_GROUP": channels,
            "KH": kernel_h,
            "KW": kernel_w,
            "STRIDE_H": stride[0],
            "STRIDE_W": stride[1],
            "PAD_TOP": pre_padding[0],
            "PAD_LEFT": pre_padding[1],
            "DIL_H": dilation[0],
            "DIL_W": dilation[1],
            "X_STRIDE_N": input_tensor["strides"][0],
            "X_STRIDE_C": input_tensor["strides"][1],
            "X_STRIDE_H": input_tensor["strides"][2],
            "X_STRIDE_W": input_tensor["strides"][3],
            "COL_STRIDE_N": columns["strides"][0],
            "COL_STRIDE_K": columns["strides"][1],
            "WEIGHT_TOTAL": weight_total,
            "WEIGHT_BLOCK": 64,
            "CONVERT_WEIGHT": convert_weight,
            "BLOCK_SIZE": block_size,
        },
        (
            _ceil_div(output_area, block_size),
            n * channels * kernel_h,
            1,
        ),
        pointer_abi,
    )


def _convolution_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | float | str | bool],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    fused_bias_relu = parameters.get("_fused_bias_relu", False)
    if not isinstance(fused_bias_relu, bool):
        raise ValueError("internal convolution fusion flag must be boolean")
    expected_tensor_count = 4 if fused_bias_relu else 3
    tensor_data_types = [tensor["data_type"] for tensor in tensors]
    if (
        len(tensor_data_types) != expected_tensor_count
        or len(set(tensor_data_types)) != 1
    ):
        raise ValueError("convolution FProp tensor data types must match")
    pointer_type = TRITON_POINTER_TYPES.get(tensor_data_types[0])
    if pointer_type is None:
        raise ValueError(
            "unsupported convolution FProp data type: "
            f"{tensor_data_types[0]!r}"
        )

    spatial_rank = _require_integer(
        parameters, "spatial_rank", minimum=1, maximum=3
    )
    tensor_rank = spatial_rank + 2
    if any(len(tensor["dimensions"]) != tensor_rank for tensor in tensors):
        raise ValueError(
            "convolution FProp tensor rank must equal spatial_rank + 2"
        )
    if any(
        dimension > 2**31 - 1
        for tensor in tensors
        for dimension in tensor["dimensions"]
    ):
        raise ValueError("convolution FProp tensor dimensions are too large")
    if any(
        not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        )
        for tensor in tensors
    ):
        raise ValueError(
            "convolution FProp tensors must have non-overlapping strides"
        )

    pre_padding = _require_integer_list(
        parameters, "pre_padding", spatial_rank, minimum=0
    )
    post_padding = _require_integer_list(
        parameters, "post_padding", spatial_rank, minimum=0
    )
    stride = _require_integer_list(
        parameters, "stride", spatial_rank, minimum=1
    )
    dilation = _require_integer_list(
        parameters, "dilation", spatial_rank, minimum=1
    )
    groups = _require_integer(parameters, "groups")
    outputs = _require_integer(parameters, "n_outputs")

    input_dimensions = tensors[0]["dimensions"]
    filter_dimensions = tensors[1]["dimensions"]
    output_tensor = tensors[-1]
    output_dimensions = output_tensor["dimensions"]
    n, c = input_dimensions[:2]
    k, filter_channels = filter_dimensions[:2]
    if c % groups != 0 or k % groups != 0:
        raise ValueError("convolution FProp channels must divide groups")
    channels_per_group = c // groups
    outputs_per_group = k // groups
    if filter_channels != channels_per_group:
        raise ValueError(
            "convolution FProp filter channels do not match input"
        )
    if fused_bias_relu:
        expected_bias_dimensions = [1, k] + [1] * spatial_rank
        if tensors[2]["dimensions"] != expected_bias_dimensions:
            raise ValueError(
                "fused convolution bias must have shape [1, K, 1, ...]"
            )

    expected_output = [n, k]
    reduction_extent = channels_per_group
    for axis in range(spatial_rank):
        input_extent = input_dimensions[axis + 2]
        filter_extent = filter_dimensions[axis + 2]
        effective_filter = dilation[axis] * (filter_extent - 1) + 1
        padded_input = input_extent + pre_padding[axis] + post_padding[axis]
        if padded_input < effective_filter:
            raise ValueError(
                "convolution FProp filter is larger than padded input"
            )
        expected_output.append(
            (padded_input - effective_filter) // stride[axis] + 1
        )
        reduction_extent *= filter_extent
    if output_dimensions != expected_output:
        raise ValueError("convolution FProp output metadata is inconsistent")
    if math.prod(output_dimensions) != outputs:
        raise ValueError("parameters.n_outputs is inconsistent with shape")
    if reduction_extent > 65536:
        raise ValueError("convolution FProp reduction extent exceeds limit")

    pointer_signature = {
        "x_ptr": pointer_type,
        "w_ptr": pointer_type,
        "bias_ptr": pointer_type,
        "y_ptr": pointer_type,
    }
    if fused_bias_relu:
        pointer_abi: list[tuple[str, str | int | None]] = [
            ("tensor", None),
            ("tensor", None),
            ("tensor", None),
            ("tensor", None),
        ]
    else:
        pointer_abi = [
            ("tensor", None),
            ("tensor", None),
            ("tensor_alias", -1),
            ("tensor", None),
        ]

    if spatial_rank == 1:
        input_l = input_dimensions[2]
        kernel_w = filter_dimensions[2]
        output_l = output_dimensions[2]
        m = n * output_l
        block_m = 32
        block_oc = 16 if outputs_per_group <= 16 else 32
        block_k = 16 if reduction_extent <= 16 else 32
        constants: dict[str, int | str | bool] = {
            "M": m,
            "XL": input_l,
            "OL": output_l,
            "DTYPE_ID": {
                "float16": 0,
                "bfloat16": 1,
                "float32": 2,
            }[tensor_data_types[0]],
            "x_stride_n": tensors[0]["strides"][0],
            "x_stride_c": tensors[0]["strides"][1],
            "x_stride_l": tensors[0]["strides"][2],
            "w_stride_o": tensors[1]["strides"][0],
            "w_stride_i": tensors[1]["strides"][1],
            "w_stride_k": tensors[1]["strides"][2],
            "bias_stride": tensors[2]["strides"][1] if fused_bias_relu else 0,
            "y_stride_n": output_tensor["strides"][0],
            "y_stride_c": output_tensor["strides"][1],
            "y_stride_l": output_tensor["strides"][2],
            "CIN_PER_GROUP": channels_per_group,
            "COUT_PER_GROUP": outputs_per_group,
            "KW": kernel_w,
            "STRIDE_W": stride[0],
            "PAD_LEFT": pre_padding[0],
            "DIL_W": dilation[0],
            "HAS_BIAS": fused_bias_relu,
            "APPLY_RELU": fused_bias_relu,
            "BLOCK_M": block_m,
            "BLOCK_OC": block_oc,
            "BLOCK_K": block_k,
            "GROUP_M": 8,
            "INPUT_PRECISION": 1 if tensor_data_types[0] == "float32" else 0,
        }
        return (
            "conv1d_gemm_kernel",
            pointer_signature,
            constants,
            (
                ((m + block_m - 1) // block_m)
                * ((outputs_per_group + block_oc - 1) // block_oc),
                groups,
                1,
            ),
            pointer_abi,
        )

    if spatial_rank == 2:
        input_h, input_w = input_dimensions[2:]
        kernel_h, kernel_w = filter_dimensions[2:]
        output_h, output_w = output_dimensions[2:]
        block_oc = 16
        block_hw = 16
        block_k = 16
        use_nchw_1x1_pad0 = (
            kernel_h == 1
            and kernel_w == 1
            and stride == [1, 1]
            and pre_padding == [0, 0]
            and post_padding == [0, 0]
            and dilation == [1, 1]
            and _is_row_major_contiguous(tensors[0])
            and _is_row_major_contiguous(tensors[1])
            and _is_row_major_contiguous(output_tensor)
        )
        if use_nchw_1x1_pad0:
            output_area = output_h * output_w
            constants = {
                "HW": output_area,
                "C_IN": c,
                "C_OUT": k,
                "CIN_PER_GROUP": channels_per_group,
                "COUT_PER_GROUP": outputs_per_group,
                "GROUPS": groups,
                "HAS_BIAS": fused_bias_relu,
                "APPLY_RELU": fused_bias_relu,
                "BIAS_STRIDE": (
                    tensors[2]["strides"][1] if fused_bias_relu else 0
                ),
                "BLOCK_OC": block_oc,
                "BLOCK_HW": block_hw,
                "BLOCK_K": block_k,
                "GROUP_M": 8,
                "DTYPE_ID": {
                    "float16": 0,
                    "bfloat16": 1,
                    "float32": 2,
                }[tensor_data_types[0]],
                "INPUT_PRECISION": (
                    1 if tensor_data_types[0] == "float32" else 0
                ),
            }
            return (
                "conv2d_1x1_nchw_pad0_kernel",
                pointer_signature,
                constants,
                (
                    ((output_area + block_hw - 1) // block_hw)
                    * ((outputs_per_group + block_oc - 1) // block_oc),
                    n * groups,
                    1,
                ),
                pointer_abi,
            )
        constants = {
            "XH": input_h,
            "XW": input_w,
            "OH": output_h,
            "OW": output_w,
            "C_IN": c,
            "C_OUT": k,
            "CIN_PER_GROUP": channels_per_group,
            "COUT_PER_GROUP": outputs_per_group,
            "GROUPS": groups,
            "STRIDE_H": stride[0],
            "STRIDE_W": stride[1],
            "PAD_TOP": pre_padding[0],
            "PAD_LEFT": pre_padding[1],
            "DIL_H": dilation[0],
            "DIL_W": dilation[1],
            "KH": kernel_h,
            "KW": kernel_w,
            "HAS_BIAS": fused_bias_relu,
            "APPLY_RELU": fused_bias_relu,
            "BIAS_STRIDE": tensors[2]["strides"][1] if fused_bias_relu else 0,
            "BLOCK_OC": block_oc,
            "BLOCK_HW": block_hw,
            "BLOCK_K": block_k,
            "GROUP_M": 8,
            "DTYPE_ID": {
                "float16": 0,
                "bfloat16": 1,
                "float32": 2,
            }[tensor_data_types[0]],
            "INPUT_PRECISION": 1 if tensor_data_types[0] == "float32" else 0,
            "X_STRIDE_N": tensors[0]["strides"][0],
            "X_STRIDE_C": tensors[0]["strides"][1],
            "X_STRIDE_H": tensors[0]["strides"][2],
            "X_STRIDE_W": tensors[0]["strides"][3],
            "W_STRIDE_K": tensors[1]["strides"][0],
            "W_STRIDE_C": tensors[1]["strides"][1],
            "W_STRIDE_R": tensors[1]["strides"][2],
            "W_STRIDE_S": tensors[1]["strides"][3],
            "Y_STRIDE_N": output_tensor["strides"][0],
            "Y_STRIDE_C": output_tensor["strides"][1],
            "Y_STRIDE_H": output_tensor["strides"][2],
            "Y_STRIDE_W": output_tensor["strides"][3],
        }
        return (
            "conv2d_spatial_nchw_kernel",
            pointer_signature,
            constants,
            (
                ((output_h * output_w + block_hw - 1) // block_hw)
                * ((outputs_per_group + block_oc - 1) // block_oc),
                n * groups,
                1,
            ),
            pointer_abi,
        )

    input_d, input_h, input_w = input_dimensions[2:]
    kernel_d, kernel_h, kernel_w = filter_dimensions[2:]
    output_d, output_h, output_w = output_dimensions[2:]
    m = n * output_d * output_h * output_w
    block_oc = 16 if outputs_per_group <= 16 else 32
    block_m = 32
    block_k = 32
    constants = {
        "M": m,
        "XD": input_d,
        "XH": input_h,
        "XW": input_w,
        "OD": output_d,
        "OH": output_h,
        "OW": output_w,
        "C_IN": c,
        "C_OUT": k,
        "CIN_PER_GROUP": channels_per_group,
        "COUT_PER_GROUP": outputs_per_group,
        "STRIDE_D": stride[0],
        "STRIDE_H": stride[1],
        "STRIDE_W": stride[2],
        "PAD_FRONT": pre_padding[0],
        "PAD_TOP": pre_padding[1],
        "PAD_LEFT": pre_padding[2],
        "DIL_D": dilation[0],
        "DIL_H": dilation[1],
        "DIL_W": dilation[2],
        "KD": kernel_d,
        "KH": kernel_h,
        "KW": kernel_w,
        "HAS_BIAS": fused_bias_relu,
        "APPLY_RELU": fused_bias_relu,
        "BIAS_STRIDE": tensors[2]["strides"][1] if fused_bias_relu else 0,
        "BLOCK_OC": block_oc,
        "BLOCK_M": block_m,
        "BLOCK_K": block_k,
        "GROUP_M": 8,
        "DTYPE_ID": {
            "float16": 0,
            "bfloat16": 1,
            "float32": 2,
        }[tensor_data_types[0]],
        "X_STRIDE_N": tensors[0]["strides"][0],
        "X_STRIDE_C": tensors[0]["strides"][1],
        "X_STRIDE_D": tensors[0]["strides"][2],
        "X_STRIDE_H": tensors[0]["strides"][3],
        "X_STRIDE_W": tensors[0]["strides"][4],
        "W_STRIDE_K": tensors[1]["strides"][0],
        "W_STRIDE_C": tensors[1]["strides"][1],
        "W_STRIDE_D": tensors[1]["strides"][2],
        "W_STRIDE_H": tensors[1]["strides"][3],
        "W_STRIDE_W": tensors[1]["strides"][4],
        "Y_STRIDE_N": output_tensor["strides"][0],
        "Y_STRIDE_C": output_tensor["strides"][1],
        "Y_STRIDE_D": output_tensor["strides"][2],
        "Y_STRIDE_H": output_tensor["strides"][3],
        "Y_STRIDE_W": output_tensor["strides"][4],
        "INPUT_PRECISION": 1 if tensor_data_types[0] == "float32" else 0,
    }
    return (
        "conv3d_spatial_ncdhw_m_kernel",
        pointer_signature,
        constants,
        (
            ((m + block_m - 1) // block_m)
            * ((outputs_per_group + block_oc - 1) // block_oc),
            groups,
            1,
        ),
        pointer_abi,
    )


def _convolution_dgrad_3d_pipeline_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | bool | str],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    stage = parameters.get("_dgrad_3d_pipeline_stage")
    if stage not in {"pack", "compute_packed", "compute_ci8_dot"}:
        raise ValueError("unknown 3D DGrad pipeline stage")
    if _require_integer(parameters, "groups") != 1:
        raise ValueError("the packed 3D DGrad pipeline requires one group")

    if stage == "pack":
        if len(tensors) != 2:
            raise ValueError("the 3D DGrad pack stage requires two tensors")
        weight, packed = tensors
        if (
            weight["data_type"] != packed["data_type"]
            or weight["data_type"] not in {"float32", "float16", "bfloat16"}
            or len(weight["dimensions"]) != 5
            or len(packed["dimensions"]) != 5
            or not _is_row_major_contiguous(weight)
            or not _is_row_major_contiguous(packed)
        ):
            raise ValueError("the 3D DGrad pack tensor metadata is invalid")
        c_out, c_in, kernel_d, kernel_h, kernel_w = weight["dimensions"]
        if packed["dimensions"] != [
            kernel_d,
            kernel_h,
            kernel_w,
            c_out,
            c_in,
        ]:
            raise ValueError("the 3D DGrad packed filter shape is invalid")
        kernel_volume = kernel_d * kernel_h * kernel_w
        block_size = 256
        pointer_type = TRITON_POINTER_TYPES[weight["data_type"]]
        return (
            "conv_dgrad3d_pack_weight_kernel",
            {"weight_ptr": pointer_type, "packed_ptr": pointer_type},
            {
                "TOTAL": c_out * c_in * kernel_volume,
                "C_OUT": c_out,
                "C_IN": c_in,
                "KERNEL_VOLUME": kernel_volume,
                "BLOCK_SIZE": block_size,
            },
            (_ceil_div(c_out * c_in, block_size), 1, 1),
            [("tensor", None), ("tensor", None)],
        )

    if len(tensors) != 3:
        raise ValueError(
            "the packed 3D DGrad compute stage requires three tensors"
        )
    loss, packed, output = tensors
    if (
        loss["data_type"] != packed["data_type"]
        or loss["data_type"] != output["data_type"]
        or loss["data_type"] not in {"float32", "float16", "bfloat16"}
        or len(loss["dimensions"]) != 5
        or len(packed["dimensions"]) != 5
        or len(output["dimensions"]) != 5
        or not all(
            _is_row_major_contiguous(tensor)
            for tensor in (loss, packed, output)
        )
    ):
        raise ValueError("the packed 3D DGrad compute metadata is invalid")

    n, c_out, loss_d, loss_h, loss_w = loss["dimensions"]
    output_n, c_in, output_d, output_h, output_w = output["dimensions"]
    kernel_d, kernel_h, kernel_w, packed_c_out, packed_c_in = packed[
        "dimensions"
    ]
    if output_n != n or packed_c_out != c_out or packed_c_in != c_in:
        raise ValueError("the packed 3D DGrad compute shape is inconsistent")
    stride = _require_integer_list(parameters, "stride", 3, minimum=1)
    padding = _require_integer_list(parameters, "pre_padding", 3, minimum=0)
    dilation = _require_integer_list(parameters, "dilation", 3, minimum=1)
    convolution_mode = _require_integer(
        parameters, "convolution_mode", minimum=0, maximum=1
    )
    pointer_type = TRITON_POINTER_TYPES[loss["data_type"]]
    pointer_signature = {
        "loss_ptr": pointer_type,
        "weight_ptr": pointer_type,
        "out_ptr": pointer_type,
    }
    pointer_abi = [("tensor", None), ("tensor", None), ("tensor", None)]
    m = n * output_d * output_h * output_w

    if stage == "compute_ci8_dot":
        if (
            loss["data_type"] != "float32"
            or c_out != 16
            or c_in != 8
            or [kernel_d, kernel_h, kernel_w] != [3, 3, 3]
            or stride != [1, 1, 1]
            or padding != [1, 1, 1]
            or dilation != [1, 1, 1]
            or convolution_mode != 0
        ):
            raise ValueError("the 3D DGrad ci8 dot shape is invalid")
        block_m = 16
        return (
            "conv_dgrad3d_pad1_3x3_fp32_ci8_dot_kernel",
            pointer_signature,
            {
                "M": m,
                "XD": output_d,
                "XH": output_h,
                "XW": output_w,
                "LOSS_D": loss_d,
                "LOSS_H": loss_h,
                "LOSS_W": loss_w,
                "loss_stride_n": loss["strides"][0],
                "loss_stride_c": loss["strides"][1],
                "loss_stride_d": loss["strides"][2],
                "loss_stride_h": loss["strides"][3],
                "loss_stride_w": loss["strides"][4],
                "out_stride_n": output["strides"][0],
                "out_stride_c": output["strides"][1],
                "out_stride_d": output["strides"][2],
                "out_stride_h": output["strides"][3],
                "out_stride_w": output["strides"][4],
                "BLOCK_M": block_m,
            },
            (_ceil_div(m, block_m), 1, 1),
            pointer_abi,
        )

    block_m = 8
    block_ci = 16
    block_co = 32
    return (
        "conv_dgrad3d_packed_kernel",
        pointer_signature,
        {
            "M": m,
            "XD": output_d,
            "XH": output_h,
            "XW": output_w,
            "LOSS_D": loss_d,
            "LOSS_H": loss_h,
            "LOSS_W": loss_w,
            "CIN_PER_GROUP": c_in,
            "COUT_PER_GROUP": c_out,
            "loss_stride_n": loss["strides"][0],
            "loss_stride_c": loss["strides"][1],
            "loss_stride_d": loss["strides"][2],
            "loss_stride_h": loss["strides"][3],
            "loss_stride_w": loss["strides"][4],
            "out_stride_n": output["strides"][0],
            "out_stride_c": output["strides"][1],
            "out_stride_d": output["strides"][2],
            "out_stride_h": output["strides"][3],
            "out_stride_w": output["strides"][4],
            "STRIDE_D": stride[0],
            "STRIDE_H": stride[1],
            "STRIDE_W": stride[2],
            "PAD_FRONT": padding[0],
            "PAD_TOP": padding[1],
            "PAD_LEFT": padding[2],
            "DIL_D": dilation[0],
            "DIL_H": dilation[1],
            "DIL_W": dilation[2],
            "KD": kernel_d,
            "KH": kernel_h,
            "KW": kernel_w,
            "FILTER_REVERSE": convolution_mode == 1,
            "INPUT_PRECISION": (1 if loss["data_type"] == "float32" else 0),
            "BLOCK_M": block_m,
            "BLOCK_CI": block_ci,
            "BLOCK_CO": block_co,
        },
        (
            _ceil_div(m, block_m) * _ceil_div(c_in, block_ci),
            1,
            1,
        ),
        pointer_abi,
    )


def _convolution_dgrad_stride2_pipeline_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | bool | str],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    stage = parameters.get("_dgrad_pipeline_stage")
    if stage not in {
        "cast_p5_loss",
        "pack",
        "zero_p5",
        "compute",
        "compute_p5_splitk",
        "compute_tile2w",
        "compute_tile4",
    }:
        raise ValueError("unknown stride-2 DGrad pipeline stage")
    groups = _require_integer(parameters, "groups")
    if groups != 1:
        raise ValueError("the packed DGrad pipeline requires one group")
    mixed_fp16 = parameters.get("_dgrad_mixed_fp16") is True

    if stage == "cast_p5_loss":
        if len(tensors) != 2:
            raise ValueError("the mixed DGrad cast stage requires two tensors")
        input_tensor, output_tensor = tensors
        if (
            not mixed_fp16
            or input_tensor["data_type"] != "float32"
            or output_tensor["data_type"] != "float16"
            or input_tensor["dimensions"] != output_tensor["dimensions"]
            or not _is_row_major_contiguous(input_tensor)
            or not _is_row_major_contiguous(output_tensor)
        ):
            raise ValueError("the mixed DGrad cast tensor metadata is invalid")
        total = math.prod(input_tensor["dimensions"])
        block_size = 256
        return (
            "cast_contiguous_kernel",
            {
                "input_ptr": TRITON_POINTER_TYPES["float32"],
                "output_ptr": TRITON_POINTER_TYPES["float16"],
            },
            {"TOTAL": total, "BLOCK_SIZE": block_size},
            (_ceil_div(total, block_size), 1, 1),
            [("tensor", None), ("tensor", None)],
        )

    if stage == "pack":
        if len(tensors) != 2:
            raise ValueError("the DGrad pack stage requires two tensors")
        weight, packed = tensors
        valid_pack_types = (
            weight["data_type"] == "float32"
            and packed["data_type"] == "float16"
            if mixed_fp16
            else weight["data_type"] == packed["data_type"]
            and weight["data_type"] in {"float32", "float16", "bfloat16"}
        )
        if (
            not valid_pack_types
            or len(weight["dimensions"]) != 4
            or len(packed["dimensions"]) != 4
            or not _is_row_major_contiguous(weight)
            or not _is_row_major_contiguous(packed)
        ):
            raise ValueError("the DGrad pack tensor metadata is invalid")
        c_out, c_in, kernel_h, kernel_w = weight["dimensions"]
        if (
            kernel_h != 3
            or kernel_w != 3
            or packed["dimensions"] != [3, 3, c_out, c_in]
        ):
            raise ValueError("the DGrad packed filter shape is invalid")
        total = c_out * c_in * 9
        block_size = 256
        return (
            "conv_dgrad2d_pack_weight_kernel",
            {
                "weight_ptr": TRITON_POINTER_TYPES[weight["data_type"]],
                "packed_ptr": TRITON_POINTER_TYPES[packed["data_type"]],
            },
            {
                "TOTAL": total,
                "C_OUT": c_out,
                "C_IN": c_in,
                "ROUND_TF32": (
                    parameters.get("_dgrad_pack_round_tf32") is True
                ),
                "BLOCK_SIZE": block_size,
            },
            (_ceil_div(c_out * c_in, block_size), 1, 1),
            [("tensor", None), ("tensor", None)],
        )

    if stage == "zero_p5":
        if len(tensors) != 1:
            raise ValueError("the P5 DGrad zero stage requires one tensor")
        output = tensors[0]
        if (
            output["data_type"] != "float32"
            or output["dimensions"] != [1, 768, 40, 40]
            or not _is_row_major_contiguous(output)
        ):
            raise ValueError("the P5 DGrad zero tensor metadata is invalid")
        total = math.prod(output["dimensions"])
        block_size = 256
        return (
            "zero_contiguous_kernel",
            {"out_ptr": TRITON_POINTER_TYPES["float32"]},
            {"TOTAL": total, "BLOCK_SIZE": block_size},
            (_ceil_div(total, block_size), 1, 1),
            [("tensor", None)],
        )

    if len(tensors) != 3:
        raise ValueError(
            "the packed DGrad compute stage requires three tensors"
        )
    loss, packed, output = tensors
    valid_compute_types = (
        loss["data_type"] == "float16"
        and packed["data_type"] == "float16"
        and output["data_type"] == "float32"
        if mixed_fp16
        else loss["data_type"] == packed["data_type"]
        and loss["data_type"] == output["data_type"]
        and loss["data_type"] in {"float32", "float16", "bfloat16"}
    )
    if (
        not valid_compute_types
        or len(loss["dimensions"]) != 4
        or len(packed["dimensions"]) != 4
        or len(output["dimensions"]) != 4
        or not all(
            _is_row_major_contiguous(tensor)
            for tensor in (loss, packed, output)
        )
    ):
        raise ValueError("the packed DGrad compute metadata is invalid")
    n, c_out, loss_h, loss_w = loss["dimensions"]
    output_n, c_in, output_h, output_w = output["dimensions"]
    if (
        output_n != n
        or packed["dimensions"] != [3, 3, c_out, c_in]
        or loss_h != (output_h + 1) // 2
        or loss_w != (output_w + 1) // 2
    ):
        raise ValueError("the packed DGrad compute shape is inconsistent")
    block_m = 32
    block_ci = 64
    block_co = 128
    common_constants: dict[str, int | bool | str] = {
        "XH": output_h,
        "XW": output_w,
        "LOSS_H": loss_h,
        "LOSS_W": loss_w,
        "CIN_PER_GROUP": c_in,
        "COUT_PER_GROUP": c_out,
        "loss_stride_n": loss["strides"][0],
        "loss_stride_c": loss["strides"][1],
        "loss_stride_h": loss["strides"][2],
        "loss_stride_w": loss["strides"][3],
        "out_stride_n": output["strides"][0],
        "out_stride_c": output["strides"][1],
        "out_stride_h": output["strides"][2],
        "out_stride_w": output["strides"][3],
        "INPUT_PRECISION": 1 if loss["data_type"] == "float32" else 0,
        "FILTER_REVERSE": (
            _require_integer(
                parameters,
                "convolution_mode",
                minimum=0,
                maximum=1,
            )
            == 1
        ),
        "BLOCK_M": block_m,
        "BLOCK_CI": block_ci,
        "BLOCK_CO": block_co,
    }
    pointer_signature = {
        "loss_ptr": TRITON_POINTER_TYPES[loss["data_type"]],
        "weight_ptr": TRITON_POINTER_TYPES[packed["data_type"]],
        "out_ptr": TRITON_POINTER_TYPES[output["data_type"]],
    }
    pointer_abi = [("tensor", None), ("tensor", None), ("tensor", None)]

    if stage == "compute_p5_splitk":
        parity_h = _require_integer(
            parameters, "_dgrad_parity_h", minimum=0, maximum=1
        )
        if (
            loss["data_type"] != "float32"
            or loss["dimensions"] != [1, 768, 20, 20]
            or packed["dimensions"] != [3, 3, 768, 768]
            or output["dimensions"] != [1, 768, 40, 40]
            or parameters.get("convolution_mode") != 0
        ):
            raise ValueError("the P5 DGrad split-K shape is invalid")
        block_m = 32
        block_ci = 64
        block_co = 64
        group_k = 2
        m = loss_h * loss_w
        split_k_blocks = _ceil_div(_ceil_div(c_out, block_co), group_k)
        return (
            "conv_dgrad2d_p5_fp32_tile2w_splitk_kernel",
            pointer_signature,
            {
                "M": m,
                "XW": output_w,
                "LOSS_H": loss_h,
                "LOSS_W": loss_w,
                "CIN_PER_GROUP": c_in,
                "COUT_PER_GROUP": c_out,
                "loss_stride_c": loss["strides"][1],
                "loss_stride_h": loss["strides"][2],
                "loss_stride_w": loss["strides"][3],
                "out_stride_c": output["strides"][1],
                "out_stride_h": output["strides"][2],
                "out_stride_w": output["strides"][3],
                "PH": parity_h,
                "GROUP_K": group_k,
                "BLOCK_M": block_m,
                "BLOCK_CI": block_ci,
                "BLOCK_CO": block_co,
            },
            (
                _ceil_div(m, block_m)
                * _ceil_div(c_in, block_ci)
                * split_k_blocks,
                1,
                1,
            ),
            pointer_abi,
        )

    if stage == "compute_tile4":
        m = n * loss_h * loss_w
        return (
            "conv_dgrad2d_stride2_pad1_3x3_packed_tile4_kernel",
            pointer_signature,
            {
                **common_constants,
                "M": m,
                "ROUND_TF32": parameters.get("_dgrad_round_tf32") is True,
            },
            (
                _ceil_div(m, block_m) * _ceil_div(c_in, block_ci),
                1,
                1,
            ),
            pointer_abi,
        )

    if stage == "compute_tile2w":
        parity_h = _require_integer(
            parameters, "_dgrad_parity_h", minimum=0, maximum=1
        )
        parity_h_count = (output_h + 1 - parity_h) // 2
        m = n * parity_h_count * loss_w
        return (
            "conv_dgrad2d_stride2_pad1_3x3_packed_tile2w_kernel",
            pointer_signature,
            {
                **common_constants,
                "M": m,
                "PARITY_H_COUNT": parity_h_count,
                "PH": parity_h,
            },
            (
                _ceil_div(m, block_m) * _ceil_div(c_in, block_ci),
                1,
                1,
            ),
            pointer_abi,
        )

    parity_h = _require_integer(
        parameters, "_dgrad_parity_h", minimum=0, maximum=1
    )
    parity_w = _require_integer(
        parameters, "_dgrad_parity_w", minimum=0, maximum=1
    )
    parity_h_count = (output_h + 1 - parity_h) // 2
    parity_w_count = (output_w + 1 - parity_w) // 2
    m = n * parity_h_count * parity_w_count
    return (
        "conv_dgrad2d_stride2_pad1_3x3_packed_parity_kernel",
        pointer_signature,
        {
            **common_constants,
            "M": m,
            "PARITY_H_COUNT": parity_h_count,
            "PARITY_W_COUNT": parity_w_count,
            "PH": parity_h,
            "PW": parity_w,
            "KH_COUNT": 1 if parity_h == 0 else 2,
            "KW_COUNT": 1 if parity_w == 0 else 2,
        },
        (
            _ceil_div(m, block_m) * _ceil_div(c_in, block_ci),
            1,
            1,
        ),
        pointer_abi,
    )


def _convolution_backward_kernel_configuration(
    operation: str,
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | bool | str],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    if operation not in {"convolution_dgrad", "convolution_wgrad"}:
        raise ValueError("unknown convolution backward operation")
    if len(tensors) != 3:
        raise ValueError("convolution backward tensor count is invalid")
    data_types = [tensor["data_type"] for tensor in tensors]
    if len(set(data_types)) != 1 or data_types[0] not in FLOAT_DATA_TYPES:
        raise ValueError(
            "convolution backward tensors must use one floating data type"
        )
    pointer_type = TRITON_POINTER_TYPES[data_types[0]]
    spatial_rank = _require_integer(
        parameters, "spatial_rank", minimum=1, maximum=3
    )
    tensor_rank = spatial_rank + 2
    if any(
        len(tensor["dimensions"]) != tensor_rank
        or any(dimension > 2**31 - 1 for dimension in tensor["dimensions"])
        or not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        )
        for tensor in tensors
    ):
        raise ValueError(
            "convolution backward tensors require rank spatial_rank + 2, "
            "int32 dimensions, and non-overlapping strides"
        )

    pre_padding = _require_integer_list(
        parameters, "pre_padding", spatial_rank, minimum=0
    )
    post_padding = _require_integer_list(
        parameters, "post_padding", spatial_rank, minimum=0
    )
    stride = _require_integer_list(
        parameters, "stride", spatial_rank, minimum=1
    )
    dilation = _require_integer_list(
        parameters, "dilation", spatial_rank, minimum=1
    )
    groups = _require_integer(parameters, "groups")
    convolution_mode = _require_integer(
        parameters, "convolution_mode", minimum=0, maximum=1
    )
    outputs = _require_integer(parameters, "n_outputs")

    dy = tensors[0]
    if operation == "convolution_dgrad":
        filter_tensor = tensors[1]
        image = tensors[2]
    else:
        image = tensors[1]
        filter_tensor = tensors[2]
    n, c = image["dimensions"][:2]
    k, filter_channels = filter_tensor["dimensions"][:2]
    if dy["dimensions"][:2] != [n, k]:
        raise ValueError(
            "convolution backward loss batch/channels are inconsistent"
        )
    if c % groups != 0 or k % groups != 0:
        raise ValueError("convolution backward channels must divide groups")
    channels_per_group = c // groups
    outputs_per_group = k // groups
    if filter_channels != channels_per_group:
        raise ValueError(
            "convolution backward filter channels are inconsistent"
        )
    expected_loss = [n, k]
    for axis in range(spatial_rank):
        image_extent = image["dimensions"][axis + 2]
        filter_extent = filter_tensor["dimensions"][axis + 2]
        effective_filter = dilation[axis] * (filter_extent - 1) + 1
        padded_image = image_extent + pre_padding[axis] + post_padding[axis]
        if padded_image < effective_filter:
            raise ValueError(
                "convolution backward filter is larger than padded image"
            )
        expected_loss.append(
            (padded_image - effective_filter) // stride[axis] + 1
        )
    if dy["dimensions"] != expected_loss:
        raise ValueError("convolution backward loss metadata is inconsistent")
    expected_output = (
        image["dimensions"]
        if operation == "convolution_dgrad"
        else filter_tensor["dimensions"]
    )
    if math.prod(expected_output) != outputs:
        raise ValueError(
            "parameters.n_outputs is inconsistent with backward output"
        )

    def padded_spatial(values: list[int], fill: int) -> list[int]:
        return [fill] * (3 - spatial_rank) + list(values)

    image_spatial = padded_spatial(image["dimensions"][2:], 1)
    filter_spatial = padded_spatial(filter_tensor["dimensions"][2:], 1)
    loss_spatial = padded_spatial(dy["dimensions"][2:], 1)
    spatial_stride = padded_spatial(stride, 1)
    spatial_padding = padded_spatial(pre_padding, 0)
    spatial_dilation = padded_spatial(dilation, 1)
    image_strides = padded_spatial(image["strides"][2:], 0)
    filter_strides = padded_spatial(filter_tensor["strides"][2:], 0)
    loss_strides = padded_spatial(dy["strides"][2:], 0)

    constants: dict[str, int | bool | str] = {
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
        "STRIDE_D": spatial_stride[0],
        "STRIDE_H": spatial_stride[1],
        "STRIDE_W": spatial_stride[2],
        "PAD_FRONT": spatial_padding[0],
        "PAD_TOP": spatial_padding[1],
        "PAD_LEFT": spatial_padding[2],
        "DIL_D": spatial_dilation[0],
        "DIL_H": spatial_dilation[1],
        "DIL_W": spatial_dilation[2],
        "FLIP_FILTER": convolution_mode == 1,
        "DY_STRIDE_N": dy["strides"][0],
        "DY_STRIDE_C": dy["strides"][1],
        "DY_STRIDE_D": loss_strides[0],
        "DY_STRIDE_H": loss_strides[1],
        "DY_STRIDE_W": loss_strides[2],
        "X_STRIDE_N": image["strides"][0],
        "X_STRIDE_C": image["strides"][1],
        "X_STRIDE_D": image_strides[0],
        "X_STRIDE_H": image_strides[1],
        "X_STRIDE_W": image_strides[2],
        "W_STRIDE_K": filter_tensor["strides"][0],
        "W_STRIDE_C": filter_tensor["strides"][1],
        "W_STRIDE_D": filter_strides[0],
        "W_STRIDE_H": filter_strides[1],
        "W_STRIDE_W": filter_strides[2],
        "INPUT_PRECISION": 1 if data_types[0] == "float32" else 0,
    }
    pointer_abi: list[tuple[str, str | int | None]] = [
        ("tensor", None),
        ("tensor", None),
        ("tensor", None),
    ]
    kernel_volume = math.prod(filter_spatial)
    if operation == "convolution_dgrad":
        m = n * math.prod(image_spatial)
        block_m = 32
        block_ci = 16 if channels_per_group <= 16 else 32
        block_k = 16 if outputs_per_group * kernel_volume <= 16 else 32
        is_contiguous_1x1_2d = (
            spatial_rank == 2
            and filter_spatial[1:] == [1, 1]
            and stride == [1, 1]
            and pre_padding == [0, 0]
            and post_padding == [0, 0]
            and dilation == [1, 1]
            and all(_is_row_major_contiguous(tensor) for tensor in tensors)
        )
        if is_contiguous_1x1_2d:
            hw = image_spatial[1] * image_spatial[2]
            block_m = 64
            block_ci = 32
            block_co = 128
            return (
                "conv_dgrad2d_1x1_nchw_kernel",
                {
                    "dy_ptr": pointer_type,
                    "w_ptr": pointer_type,
                    "dx_ptr": pointer_type,
                },
                {
                    "HW": hw,
                    "C_IN": c,
                    "C_OUT": k,
                    "CIN_PER_GROUP": channels_per_group,
                    "COUT_PER_GROUP": outputs_per_group,
                    "GROUPS": groups,
                    "INPUT_PRECISION": (
                        1 if data_types[0] == "float32" else 0
                    ),
                    "BLOCK_M": block_m,
                    "BLOCK_CI": block_ci,
                    "BLOCK_CO": block_co,
                },
                (
                    _ceil_div(hw, block_m)
                    * _ceil_div(channels_per_group, block_ci),
                    n * groups,
                    1,
                ),
                pointer_abi,
            )
        if spatial_rank == 2 and stride == [1, 1] and kernel_volume > 1:
            block_co = block_k
            stride1_constants = {
                "M": m,
                "XH": image_spatial[1],
                "XW": image_spatial[2],
                "OH": loss_spatial[1],
                "OW": loss_spatial[2],
                "CIN_PER_GROUP": channels_per_group,
                "COUT_PER_GROUP": outputs_per_group,
                "DY_STRIDE_N": dy["strides"][0],
                "DY_STRIDE_C": dy["strides"][1],
                "DY_STRIDE_H": loss_strides[1],
                "DY_STRIDE_W": loss_strides[2],
                "W_STRIDE_K": filter_tensor["strides"][0],
                "W_STRIDE_C": filter_tensor["strides"][1],
                "W_STRIDE_H": filter_strides[1],
                "W_STRIDE_W": filter_strides[2],
                "X_STRIDE_N": image["strides"][0],
                "X_STRIDE_C": image["strides"][1],
                "X_STRIDE_H": image_strides[1],
                "X_STRIDE_W": image_strides[2],
                "PAD_TOP": spatial_padding[1],
                "PAD_LEFT": spatial_padding[2],
                "DIL_H": spatial_dilation[1],
                "DIL_W": spatial_dilation[2],
                "KH": filter_spatial[1],
                "KW": filter_spatial[2],
                "FLIP_FILTER": convolution_mode == 1,
                "INPUT_PRECISION": 1 if data_types[0] == "float32" else 0,
                "BLOCK_M": block_m,
                "BLOCK_CI": block_ci,
                "BLOCK_CO": block_co,
            }
            return (
                "conv_dgrad2d_stride1_kernel",
                {
                    "dy_ptr": pointer_type,
                    "w_ptr": pointer_type,
                    "dx_ptr": pointer_type,
                },
                stride1_constants,
                (
                    ((m + block_m - 1) // block_m)
                    * ((channels_per_group + block_ci - 1) // block_ci),
                    groups,
                    1,
                ),
                pointer_abi,
            )
        constants.update(
            {
                "M": m,
                "BLOCK_M": block_m,
                "BLOCK_CI": block_ci,
                "BLOCK_K": block_k,
                "GROUP_M": 8,
            }
        )
        return (
            "conv_dgrad_nd_kernel",
            {
                "dy_ptr": pointer_type,
                "w_ptr": pointer_type,
                "dx_ptr": pointer_type,
            },
            constants,
            (
                ((m + block_m - 1) // block_m)
                * ((channels_per_group + block_ci - 1) // block_ci),
                groups,
                1,
            ),
            pointer_abi,
        )

    m = n * math.prod(loss_spatial)
    block_oc = 16 if outputs_per_group <= 16 else 32
    block_ci = 16 if channels_per_group <= 16 else 32
    block_m = 32
    constants.update(
        {
            "M": m,
            "BLOCK_OC": block_oc,
            "BLOCK_CI": block_ci,
            "BLOCK_M": block_m,
        }
    )
    return (
        "conv_wgrad_nd_kernel",
        {
            "dy_ptr": pointer_type,
            "x_ptr": pointer_type,
            "dw_ptr": pointer_type,
        },
        constants,
        (
            ((outputs_per_group + block_oc - 1) // block_oc)
            * ((channels_per_group + block_ci - 1) // block_ci),
            kernel_volume,
            groups,
        ),
        pointer_abi,
    )


def _convolution_wgrad_pipeline_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | bool | str],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    """Configure one internal stage of a split/reduce WGrad pipeline."""
    pipeline_stage = parameters.get("_wgrad_pipeline_stage")
    if pipeline_stage not in {"split", "reduce"}:
        raise ValueError("unknown convolution WGrad pipeline stage")
    spatial_rank = _require_integer(
        parameters, "spatial_rank", minimum=1, maximum=3
    )
    if spatial_rank != 2:
        raise ValueError("the WGrad split/reduce pipeline requires 2D tensors")
    groups = _require_integer(parameters, "groups")
    num_splits = _require_integer(parameters, "_wgrad_num_splits")
    kernel_h = _require_integer(parameters, "_wgrad_kernel_h")
    kernel_w = _require_integer(parameters, "_wgrad_kernel_w")

    if pipeline_stage == "split":
        if len(tensors) != 3:
            raise ValueError("the WGrad split stage requires three tensors")
        image, loss, partial = tensors
        if (
            image["data_type"] != loss["data_type"]
            or image["data_type"] not in FLOAT_DATA_TYPES
            or partial["data_type"] != "float32"
            or len(image["dimensions"]) != 4
            or len(loss["dimensions"]) != 4
        ):
            raise ValueError(
                "the WGrad split stage tensor metadata is invalid"
            )
        n, c_in, image_h, image_w = image["dimensions"]
        loss_n, c_out, loss_h, loss_w = loss["dimensions"]
        if loss_n != n or c_in % groups != 0 or c_out % groups != 0:
            raise ValueError("the WGrad split stage channels are inconsistent")
        cin_per_group = c_in // groups
        cout_per_group = c_out // groups
        cik = cin_per_group * kernel_h * kernel_w
        if partial["dimensions"] != [num_splits, c_out, cik]:
            raise ValueError("the WGrad partial workspace shape is invalid")
        stride = _require_integer_list(parameters, "stride", 2, minimum=1)
        pre_padding = _require_integer_list(
            parameters, "pre_padding", 2, minimum=0
        )
        dilation = _require_integer_list(parameters, "dilation", 2, minimum=1)
        convolution_mode = _require_integer(
            parameters, "convolution_mode", minimum=0, maximum=1
        )
        use_col_split = (
            parameters.get("_wgrad_pipeline_algorithm") == "stem_col"
        )
        block_co = 16
        block_ci = 32
        block_m = (
            128
            if use_col_split
            else 32 if image["data_type"] == "float32" else 64
        )
        constants: dict[str, int | bool | str] = {
            "M": n * loss_h * loss_w,
            "IMAGE_H": image_h,
            "IMAGE_W": image_w,
            "LOSS_H": loss_h,
            "LOSS_W": loss_w,
            "C_OUT": c_out,
            "CIN_PER_GROUP": cin_per_group,
            "COUT_PER_GROUP": cout_per_group,
            "image_stride_n": image["strides"][0],
            "image_stride_c": image["strides"][1],
            "image_stride_h": image["strides"][2],
            "image_stride_w": image["strides"][3],
            "loss_stride_n": loss["strides"][0],
            "loss_stride_c": loss["strides"][1],
            "loss_stride_h": loss["strides"][2],
            "loss_stride_w": loss["strides"][3],
            "STRIDE_H": stride[0],
            "STRIDE_W": stride[1],
            "PAD_H": pre_padding[0],
            "PAD_W": pre_padding[1],
            "DIL_H": dilation[0],
            "DIL_W": dilation[1],
            "KH": kernel_h,
            "KW": kernel_w,
            "FILTER_REVERSE": convolution_mode == 1,
            "NUM_SPLITS": num_splits,
            "BLOCK_CO": block_co,
            "BLOCK_CI": block_ci,
            "BLOCK_M": block_m,
        }
        if use_col_split:
            constants["BLOCK_N"] = constants.pop("BLOCK_CI")
            return (
                "_conv_wgrad2d_col_split_kernel",
                {
                    "image_ptr": TRITON_POINTER_TYPES[image["data_type"]],
                    "loss_ptr": TRITON_POINTER_TYPES[loss["data_type"]],
                    "partial_ptr": "*fp32",
                },
                constants,
                (
                    _ceil_div(cout_per_group, block_co)
                    * _ceil_div(cik, block_ci),
                    num_splits * groups,
                    1,
                ),
                [("tensor", None), ("tensor", None), ("tensor", None)],
            )
        return (
            "_conv_wgrad2d_3tap_split_kernel",
            {
                "image_ptr": TRITON_POINTER_TYPES[image["data_type"]],
                "loss_ptr": TRITON_POINTER_TYPES[loss["data_type"]],
                "partial_ptr": "*fp32",
            },
            constants,
            (
                _ceil_div(cout_per_group, block_co)
                * _ceil_div(cin_per_group, block_ci),
                kernel_h,
                num_splits * groups,
            ),
            [("tensor", None), ("tensor", None), ("tensor", None)],
        )

    if len(tensors) != 2:
        raise ValueError("the WGrad reduce stage requires two tensors")
    partial, output = tensors
    if (
        partial["data_type"] != "float32"
        or output["data_type"] not in FLOAT_DATA_TYPES
        or len(output["dimensions"]) != 4
    ):
        raise ValueError("the WGrad reduce stage tensor metadata is invalid")
    c_out, cin_per_group, output_kh, output_kw = output["dimensions"]
    if output_kh != kernel_h or output_kw != kernel_w or c_out % groups != 0:
        raise ValueError("the WGrad reduce stage filter shape is invalid")
    cout_per_group = c_out // groups
    cik = cin_per_group * kernel_h * kernel_w
    if partial["dimensions"] != [num_splits, c_out, cik]:
        raise ValueError("the WGrad partial workspace shape is invalid")
    block_co = 16
    block_n = 32
    constants = {
        "C_OUT": c_out,
        "CIN_PER_GROUP": cin_per_group,
        "COUT_PER_GROUP": cout_per_group,
        "out_stride_o": output["strides"][0],
        "out_stride_i": output["strides"][1],
        "out_stride_h": output["strides"][2],
        "out_stride_w": output["strides"][3],
        "KH": kernel_h,
        "KW": kernel_w,
        "NUM_SPLITS": num_splits,
        "BLOCK_CO": block_co,
        "BLOCK_N": block_n,
    }
    return (
        "_conv_wgrad2d_col_reduce_kernel",
        {
            "partial_ptr": "*fp32",
            "out_ptr": TRITON_POINTER_TYPES[output["data_type"]],
        },
        constants,
        (
            _ceil_div(cout_per_group, block_co) * _ceil_div(cik, block_n),
            groups,
            1,
        ),
        [("tensor", None), ("tensor", None)],
    )


def _convolution_wgrad_p5_pipeline_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | bool | str],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    """Configure one stage of the NVIDIA YOLO P5 WGrad pipeline."""
    pipeline_stage = parameters.get("_wgrad_pipeline_stage")
    if pipeline_stage not in {"cast_loss", "pack", "matmul"}:
        raise ValueError("unknown P5 WGrad pipeline stage")
    if _require_integer(parameters, "groups") != 1:
        raise ValueError("the P5 WGrad pipeline requires one group")
    mixed_fp16 = parameters.get("_wgrad_mixed_fp16") is True

    if pipeline_stage == "cast_loss":
        if len(tensors) != 2:
            raise ValueError(
                "the mixed P5 WGrad cast stage requires two tensors"
            )
        input_tensor, output_tensor = tensors
        if (
            not mixed_fp16
            or input_tensor["data_type"] != "float32"
            or output_tensor["data_type"] != "float16"
            or input_tensor["dimensions"] != output_tensor["dimensions"]
            or not _is_row_major_contiguous(input_tensor)
            or not _is_row_major_contiguous(output_tensor)
        ):
            raise ValueError("the mixed P5 WGrad cast metadata is invalid")
        total = math.prod(input_tensor["dimensions"])
        block_size = 256
        return (
            "cast_contiguous_kernel",
            {
                "input_ptr": TRITON_POINTER_TYPES["float32"],
                "output_ptr": TRITON_POINTER_TYPES["float16"],
            },
            {"TOTAL": total, "BLOCK_SIZE": block_size},
            (_ceil_div(total, block_size), 1, 1),
            [("tensor", None), ("tensor", None)],
        )

    if pipeline_stage == "pack":
        if len(tensors) != 2:
            raise ValueError("the P5 WGrad pack stage requires two tensors")
        image, packed = tensors
        valid_pack_types = (
            image["data_type"] == "float32"
            and packed["data_type"] == "float16"
            if mixed_fp16
            else image["data_type"] == packed["data_type"]
            and image["data_type"] in FLOAT_DATA_TYPES
        )
        if (
            not valid_pack_types
            or len(image["dimensions"]) != 4
            or image["dimensions"][0] != 1
            or image["dimensions"][2:] != [40, 40]
            or len(packed["dimensions"]) != 2
            or not _is_row_major_contiguous(packed)
        ):
            raise ValueError("the P5 WGrad pack metadata is invalid")
        c_in = image["dimensions"][1]
        cik = c_in * 9
        if packed["dimensions"] != [400, cik]:
            raise ValueError("the P5 WGrad packed image shape is invalid")
        block_m = 16
        block_n = 16
        block_k = 32
        group_m = 8
        constants: dict[str, int | bool | str] = {
            "CIN_PER_GROUP": c_in,
            "image_stride_c": image["strides"][1],
            "image_stride_h": image["strides"][2],
            "image_stride_w": image["strides"][3],
            "M": 400,
            "N": cik,
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "GROUP_M": group_m,
        }
        return (
            "_conv_wgrad2d_p5_pack_image_kernel",
            {
                "image_ptr": TRITON_POINTER_TYPES[image["data_type"]],
                "packed_ptr": TRITON_POINTER_TYPES[packed["data_type"]],
            },
            constants,
            (_ceil_div(400, block_m), _ceil_div(cik, block_n), 1),
            [("tensor", None), ("tensor", None)],
        )

    if len(tensors) != 3:
        raise ValueError("the P5 WGrad matmul stage requires three tensors")
    loss, packed, output = tensors
    valid_matmul_types = (
        loss["data_type"] == "float16"
        and packed["data_type"] == "float16"
        and output["data_type"] == "float32"
        if mixed_fp16
        else loss["data_type"] == packed["data_type"]
        and loss["data_type"] == output["data_type"]
        and loss["data_type"] in FLOAT_DATA_TYPES
    )
    if (
        not valid_matmul_types
        or len(loss["dimensions"]) != 4
        or loss["dimensions"][0] != 1
        or loss["dimensions"][2:] != [20, 20]
        or len(packed["dimensions"]) != 2
        or len(output["dimensions"]) != 4
        or output["dimensions"][2:] != [3, 3]
        or not _is_row_major_contiguous(packed)
        or not _is_row_major_contiguous(output)
    ):
        raise ValueError("the P5 WGrad matmul metadata is invalid")
    c_out = loss["dimensions"][1]
    output_c_out, c_in, _, _ = output["dimensions"]
    cik = c_in * 9
    if output_c_out != c_out or packed["dimensions"] != [400, cik]:
        raise ValueError("the P5 WGrad matmul shape is inconsistent")
    block_m = 16
    block_n = 16
    block_k = 32
    group_m = 8
    constants = {
        "M": c_out,
        "N": cik,
        "K": 400,
        "DTYPE_ID": {
            "float16": 0,
            "bfloat16": 1,
            "float32": 2,
        }[loss["data_type"]],
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "GROUP_M": group_m,
    }
    return (
        "_conv_wgrad2d_p5_mm_kernel",
        {
            "loss_ptr": TRITON_POINTER_TYPES[loss["data_type"]],
            "packed_ptr": TRITON_POINTER_TYPES[packed["data_type"]],
            "out_ptr": TRITON_POINTER_TYPES[output["data_type"]],
        },
        constants,
        (_ceil_div(c_out, block_m) * _ceil_div(cik, block_n), 1, 1),
        [("tensor", None), ("tensor", None), ("tensor", None)],
    )


def _convolution_wgrad_1x1_pipeline_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | bool | str],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    """Configure one internal stage of the NVIDIA 1x1 WGrad pipeline."""
    pipeline_stage = parameters.get("_wgrad_pipeline_stage")
    if pipeline_stage not in {"direct", "split", "reduce"}:
        raise ValueError("unknown 1x1 WGrad pipeline stage")
    groups = _require_integer(parameters, "groups")
    if pipeline_stage == "direct":
        if len(tensors) != 3:
            raise ValueError(
                "the 1x1 WGrad direct stage requires three tensors"
            )
        image, loss, output = tensors
        if (
            image["data_type"] != loss["data_type"]
            or image["data_type"] != output["data_type"]
            or image["data_type"] not in FLOAT_DATA_TYPES
            or len(image["dimensions"]) != 4
            or len(loss["dimensions"]) != 4
            or len(output["dimensions"]) != 4
        ):
            raise ValueError("the 1x1 WGrad direct tensor metadata is invalid")
        n, c_in, image_h, image_w = image["dimensions"]
        loss_n, c_out, loss_h, loss_w = loss["dimensions"]
        if (
            loss_n != n
            or loss_h != image_h
            or loss_w != image_w
            or output["dimensions"] != [c_out, c_in // groups, 1, 1]
            or c_in % groups != 0
            or c_out % groups != 0
        ):
            raise ValueError("the 1x1 WGrad direct shape is inconsistent")
        cin_per_group = c_in // groups
        cout_per_group = c_out // groups
        block_co = 16
        block_ci = 32
        block_m = 256
        constants: dict[str, int | bool | str] = {
            "BATCH_N": n,
            "HW": image_h * image_w,
            "CIN_PER_GROUP": cin_per_group,
            "COUT_PER_GROUP": cout_per_group,
            "image_stride_n": image["strides"][0],
            "image_stride_c": image["strides"][1],
            "loss_stride_n": loss["strides"][0],
            "loss_stride_c": loss["strides"][1],
            "out_stride_o": output["strides"][0],
            "out_stride_i": output["strides"][1],
            "BLOCK_CO": block_co,
            "BLOCK_CI": block_ci,
            "BLOCK_M": block_m,
        }
        pointer_type = TRITON_POINTER_TYPES[image["data_type"]]
        return (
            "_conv_wgrad2d_1x1_direct_nodiv_kernel",
            {
                "image_ptr": pointer_type,
                "loss_ptr": pointer_type,
                "out_ptr": pointer_type,
            },
            constants,
            (
                _ceil_div(cout_per_group, block_co)
                * _ceil_div(cin_per_group, block_ci),
                groups,
                1,
            ),
            [("tensor", None), ("tensor", None), ("tensor", None)],
        )

    num_splits = _require_integer(parameters, "_wgrad_num_splits")

    if pipeline_stage == "split":
        if len(tensors) != 3:
            raise ValueError(
                "the 1x1 WGrad split stage requires three tensors"
            )
        image, loss, partial = tensors
        if (
            image["data_type"] != loss["data_type"]
            or image["data_type"] not in FLOAT_DATA_TYPES
            or partial["data_type"] not in {"float16", "float32"}
            or len(image["dimensions"]) != 4
            or len(loss["dimensions"]) != 4
        ):
            raise ValueError("the 1x1 WGrad split tensor metadata is invalid")
        n, c_in, image_h, image_w = image["dimensions"]
        loss_n, c_out, loss_h, loss_w = loss["dimensions"]
        if (
            loss_n != n
            or loss_h != image_h
            or loss_w != image_w
            or c_in % groups != 0
            or c_out % groups != 0
            or num_splits % n != 0
        ):
            raise ValueError("the 1x1 WGrad split shape is inconsistent")
        cin_per_group = c_in // groups
        cout_per_group = c_out // groups
        if partial["dimensions"] != [num_splits, c_out, cin_per_group]:
            raise ValueError(
                "the 1x1 WGrad partial workspace shape is invalid"
            )
        block_co = 16
        block_ci = 64
        block_m = 256
        constants: dict[str, int | bool | str] = {
            "HW": image_h * image_w,
            "C_OUT": c_out,
            "CIN_PER_GROUP": cin_per_group,
            "COUT_PER_GROUP": cout_per_group,
            "image_stride_n": image["strides"][0],
            "image_stride_c": image["strides"][1],
            "loss_stride_n": loss["strides"][0],
            "loss_stride_c": loss["strides"][1],
            "SPLITS_PER_N": num_splits // n,
            "BLOCK_CO": block_co,
            "BLOCK_CI": block_ci,
            "BLOCK_M": block_m,
        }
        return (
            "_conv_wgrad2d_1x1_split_nodiv_kernel",
            {
                "image_ptr": TRITON_POINTER_TYPES[image["data_type"]],
                "loss_ptr": TRITON_POINTER_TYPES[loss["data_type"]],
                "partial_ptr": TRITON_POINTER_TYPES[partial["data_type"]],
            },
            constants,
            (
                _ceil_div(cout_per_group, block_co)
                * _ceil_div(cin_per_group, block_ci),
                num_splits,
                1,
            ),
            [("tensor", None), ("tensor", None), ("tensor", None)],
        )

    if len(tensors) != 2:
        raise ValueError("the 1x1 WGrad reduce stage requires two tensors")
    partial, output = tensors
    if (
        partial["data_type"] not in {"float16", "float32"}
        or output["data_type"] not in FLOAT_DATA_TYPES
        or len(output["dimensions"]) != 4
    ):
        raise ValueError("the 1x1 WGrad reduce tensor metadata is invalid")
    c_out, cin_per_group, kernel_h, kernel_w = output["dimensions"]
    if kernel_h != 1 or kernel_w != 1 or c_out % groups != 0:
        raise ValueError("the 1x1 WGrad reduce filter shape is invalid")
    cout_per_group = c_out // groups
    if partial["dimensions"] != [num_splits, c_out, cin_per_group]:
        raise ValueError("the 1x1 WGrad partial workspace shape is invalid")
    block_co = 8
    block_ci = 16 if output["data_type"] == "bfloat16" else 32
    constants = {
        "C_OUT": c_out,
        "CIN_PER_GROUP": cin_per_group,
        "COUT_PER_GROUP": cout_per_group,
        "out_stride_o": output["strides"][0],
        "out_stride_i": output["strides"][1],
        "NUM_SPLITS": num_splits,
        "BLOCK_CO": block_co,
        "BLOCK_CI": block_ci,
    }
    return (
        "_conv_wgrad2d_1x1_reduce_kernel",
        {
            "partial_ptr": TRITON_POINTER_TYPES[partial["data_type"]],
            "out_ptr": TRITON_POINTER_TYPES[output["data_type"]],
        },
        constants,
        (
            _ceil_div(cout_per_group, block_co)
            * _ceil_div(cin_per_group, block_ci),
            groups,
            1,
        ),
        [("tensor", None), ("tensor", None)],
    )


def _convolution_wgrad_stride2_pipeline_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | bool | str],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    """Configure the fixed-shape NVIDIA stride-2 WGrad pipeline."""
    pipeline_stage = parameters.get("_wgrad_pipeline_stage")
    if pipeline_stage not in {"split", "reduce"}:
        raise ValueError("unknown stride-2 WGrad pipeline stage")
    groups = _require_integer(parameters, "groups")
    num_splits = _require_integer(parameters, "_wgrad_num_splits")

    if pipeline_stage == "split":
        if len(tensors) != 3:
            raise ValueError(
                "the stride-2 WGrad split stage requires three tensors"
            )
        image, loss, partial = tensors
        if (
            image["data_type"] != loss["data_type"]
            or image["data_type"] != partial["data_type"]
            or image["data_type"] not in FLOAT_DATA_TYPES
            or len(image["dimensions"]) != 4
            or len(loss["dimensions"]) != 4
        ):
            raise ValueError("the stride-2 WGrad split metadata is invalid")
        n, c_in, image_h, image_w = image["dimensions"]
        loss_n, c_out, loss_h, loss_w = loss["dimensions"]
        if (
            loss_n != n
            or image_h != 56
            or image_w != 56
            or loss_h != 28
            or loss_w != 28
            or num_splits != n
            or c_in % groups != 0
            or c_out % groups != 0
        ):
            raise ValueError("the stride-2 WGrad split shape is inconsistent")
        cin_per_group = c_in // groups
        cout_per_group = c_out // groups
        if partial["dimensions"] != [num_splits, c_out, cin_per_group, 9]:
            raise ValueError("the stride-2 WGrad workspace shape is invalid")
        block_co = 16
        block_ci = 32
        block_hw = 128
        constants: dict[str, int | bool | str] = {
            "C_OUT": c_out,
            "CIN_PER_GROUP": cin_per_group,
            "COUT_PER_GROUP": cout_per_group,
            "image_stride_n": image["strides"][0],
            "image_stride_c": image["strides"][1],
            "image_stride_h": image["strides"][2],
            "image_stride_w": image["strides"][3],
            "loss_stride_n": loss["strides"][0],
            "loss_stride_c": loss["strides"][1],
            "loss_stride_h": loss["strides"][2],
            "loss_stride_w": loss["strides"][3],
            "BLOCK_CO": block_co,
            "BLOCK_CI": block_ci,
            "BLOCK_HW": block_hw,
        }
        pointer_type = TRITON_POINTER_TYPES[image["data_type"]]
        return (
            "_conv_wgrad2d_stride2_row4_split_kernel",
            {
                "image_ptr": pointer_type,
                "loss_ptr": pointer_type,
                "partial_ptr": pointer_type,
            },
            constants,
            (
                _ceil_div(cout_per_group, block_co)
                * _ceil_div(cin_per_group, block_ci),
                3,
                num_splits,
            ),
            [("tensor", None), ("tensor", None), ("tensor", None)],
        )

    if len(tensors) != 2:
        raise ValueError(
            "the stride-2 WGrad reduce stage requires two tensors"
        )
    partial, output = tensors
    if (
        partial["data_type"] != output["data_type"]
        or output["data_type"] not in FLOAT_DATA_TYPES
        or len(output["dimensions"]) != 4
    ):
        raise ValueError("the stride-2 WGrad reduce metadata is invalid")
    c_out, cin_per_group, kernel_h, kernel_w = output["dimensions"]
    if kernel_h != 3 or kernel_w != 3 or c_out % groups != 0:
        raise ValueError("the stride-2 WGrad reduce filter shape is invalid")
    cout_per_group = c_out // groups
    if partial["dimensions"] != [num_splits, c_out, cin_per_group, 9]:
        raise ValueError("the stride-2 WGrad workspace shape is invalid")
    block_co = 16
    block_ci = 32
    constants = {
        "C_OUT": c_out,
        "CIN_PER_GROUP": cin_per_group,
        "COUT_PER_GROUP": cout_per_group,
        "out_stride_o": output["strides"][0],
        "out_stride_i": output["strides"][1],
        "out_stride_h": output["strides"][2],
        "out_stride_w": output["strides"][3],
        "KH": kernel_h,
        "KW": kernel_w,
        "NUM_SPLITS": num_splits,
        "BLOCK_CO": block_co,
        "BLOCK_CI": block_ci,
    }
    pointer_type = TRITON_POINTER_TYPES[output["data_type"]]
    return (
        "_conv_wgrad2d_reduce_kernel",
        {"partial_ptr": pointer_type, "out_ptr": pointer_type},
        constants,
        (
            _ceil_div(cout_per_group, block_co)
            * _ceil_div(cin_per_group, block_ci),
            kernel_h * kernel_w,
            groups,
        ),
        [("tensor", None), ("tensor", None)],
    )


def _convolution_wgrad_batched_pipeline_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | bool | str],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    pipeline_stage = parameters.get("_wgrad_pipeline_stage")
    if pipeline_stage not in {"matmul", "reduce"}:
        raise ValueError("unknown batched WGrad pipeline stage")
    groups = _require_integer(parameters, "groups")
    if groups != 1:
        raise ValueError("the batched WGrad pipeline requires one group")
    num_splits = _require_integer(parameters, "_wgrad_num_splits")
    kernel_h = _require_integer(parameters, "_wgrad_kernel_h")
    kernel_w = _require_integer(parameters, "_wgrad_kernel_w")

    if pipeline_stage == "matmul":
        if len(tensors) != 3:
            raise ValueError("the batched WGrad GEMM requires three tensors")
        loss, columns, partial = tensors
        if (
            loss["data_type"] != columns["data_type"]
            or loss["data_type"] != partial["data_type"]
            or loss["data_type"] not in FLOAT_DATA_TYPES
            or len(loss["dimensions"]) != 4
            or len(columns["dimensions"]) != 3
            or len(partial["dimensions"]) != 3
            or not all(
                _is_row_major_contiguous(tensor)
                for tensor in (loss, columns, partial)
            )
        ):
            raise ValueError("the batched WGrad GEMM metadata is invalid")
        n, c_out, loss_h, loss_w = loss["dimensions"]
        columns_n, cik, padded_m = columns["dimensions"]
        m = loss_h * loss_w
        if num_splits % n != 0 or columns_n != n or padded_m < m:
            raise ValueError("the batched WGrad GEMM shape is inconsistent")
        splits_per_n = num_splits // n
        if partial["dimensions"] != [num_splits, c_out, cik]:
            raise ValueError("the batched WGrad partial shape is inconsistent")
        block_co = 16
        block_ci = 32
        block_m = 128
        pointer_type = TRITON_POINTER_TYPES[loss["data_type"]]
        return (
            "_conv_wgrad2d_batched_tma_kernel",
            {
                "loss_ptr": pointer_type,
                "columns_ptr": pointer_type,
                "partial_ptr": pointer_type,
            },
            {
                "BATCH_N": n,
                "M": m,
                "PADDED_M": padded_m,
                "C_OUT": c_out,
                "CIK": cik,
                "SPLITS_PER_N": splits_per_n,
                "INPUT_IS_FLOAT32": loss["data_type"] == "float32",
                "BLOCK_CO": block_co,
                "BLOCK_CI": block_ci,
                "BLOCK_M": block_m,
            },
            (
                _ceil_div(c_out, block_co) * _ceil_div(cik, block_ci),
                splits_per_n,
                n,
            ),
            [("tensor", None), ("tensor", None), ("tensor", None)],
        )

    if len(tensors) != 2:
        raise ValueError("the batched WGrad reduce stage requires two tensors")
    partial, output = tensors
    if (
        partial["data_type"] != output["data_type"]
        or output["data_type"] not in FLOAT_DATA_TYPES
        or len(partial["dimensions"]) != 3
        or len(output["dimensions"]) != 4
        or not _is_row_major_contiguous(partial)
    ):
        raise ValueError("the batched WGrad reduce metadata is invalid")
    c_out, cin_per_group, output_kh, output_kw = output["dimensions"]
    cik = cin_per_group * kernel_h * kernel_w
    if (
        output_kh != kernel_h
        or output_kw != kernel_w
        or partial["dimensions"] != [num_splits, c_out, cik]
    ):
        raise ValueError("the batched WGrad reduce shape is inconsistent")
    if kernel_h == 1 and kernel_w == 1 and _is_row_major_contiguous(output):
        block_m = 256
        block_n = 16
        pointer_type = TRITON_POINTER_TYPES[output["data_type"]]
        total = c_out * cik
        return (
            "_conv_wgrad2d_split_vector_reduce_kernel",
            {"partial_ptr": pointer_type, "out_ptr": pointer_type},
            {
                "TOTAL": total,
                "NUM_SPLITS": num_splits,
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
            },
            (_ceil_div(total, block_m), 1, 1),
            [("tensor", None), ("tensor", None)],
        )
    block_co = 16
    block_n = 32
    pointer_type = TRITON_POINTER_TYPES[output["data_type"]]
    return (
        "_conv_wgrad2d_col_reduce_kernel",
        {"partial_ptr": pointer_type, "out_ptr": pointer_type},
        {
            "C_OUT": c_out,
            "CIN_PER_GROUP": cin_per_group,
            "COUT_PER_GROUP": c_out,
            "out_stride_o": output["strides"][0],
            "out_stride_i": output["strides"][1],
            "out_stride_h": output["strides"][2],
            "out_stride_w": output["strides"][3],
            "KH": kernel_h,
            "KW": kernel_w,
            "NUM_SPLITS": num_splits,
            "BLOCK_CO": block_co,
            "BLOCK_N": block_n,
        },
        (_ceil_div(c_out, block_co) * _ceil_div(cik, block_n), 1, 1),
        [("tensor", None), ("tensor", None)],
    )


def _matmul_p5_pipeline_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    stage = parameters.get("_fprop_p5_matmul_stage")
    if stage not in {"split", "reduce"}:
        raise ValueError("the P5 MatMul pipeline stage is invalid")
    m = _require_integer(parameters, "m", minimum=1)
    n = _require_integer(parameters, "n", minimum=1)
    k = _require_integer(parameters, "k", minimum=1)
    splits = _require_integer(
        parameters, "_fprop_p5_splits", minimum=2, maximum=64
    )

    if stage == "split":
        if len(tensors) != 3:
            raise ValueError("the P5 split-K stage requires three tensors")
        a, b, partial = tensors
        mixed_fp16 = (
            a["data_type"] == "float32" and b["data_type"] == "float16"
        )
        if (
            not (
                mixed_fp16
                or (
                    a["data_type"] in {"float16", "bfloat16"}
                    and b["data_type"] == a["data_type"]
                )
            )
            or partial["data_type"] != "float32"
            or a["dimensions"] != [1, m, k]
            or b["dimensions"] != [1, k, n]
            or partial["dimensions"] != [splits, m, n]
            or not all(
                _is_row_major_contiguous(tensor) for tensor in (a, b, partial)
            )
        ):
            raise ValueError("the P5 split-K tensor contract is invalid")
        block_m = 64
        block_n = 64
        block_k = 64
        return (
            "matmul_p5_split_k_kernel",
            {
                "a_ptr": TRITON_POINTER_TYPES[a["data_type"]],
                "b_ptr": TRITON_POINTER_TYPES[b["data_type"]],
                "partial_ptr": TRITON_POINTER_TYPES["float32"],
            },
            {
                "M": m,
                "N": n,
                "K": k,
                "B_STRIDE_K": b["strides"][-2],
                "SPLITS": splits,
                "MIXED_FP16": mixed_fp16,
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "GROUP_M": 8,
            },
            (
                _ceil_div(m, block_m) * _ceil_div(n, block_n),
                splits,
                1,
            ),
            [("tensor", None), ("tensor", None), ("tensor", None)],
        )

    if len(tensors) != 2:
        raise ValueError("the P5 split-K reduce stage requires two tensors")
    partial, output = tensors
    if (
        partial["data_type"] != "float32"
        or output["data_type"] not in FLOAT_DATA_TYPES
        or partial["dimensions"] != [splits, m, n]
        or output["dimensions"] != [1, m, n]
        or not _is_row_major_contiguous(partial)
        or not _is_row_major_contiguous(output)
    ):
        raise ValueError("the P5 split-K reduce tensor contract is invalid")
    block_size = 1024
    total = m * n
    return (
        "matmul_p5_split_k_reduce_kernel",
        {
            "partial_ptr": TRITON_POINTER_TYPES["float32"],
            "output_ptr": TRITON_POINTER_TYPES[output["data_type"]],
        },
        {"TOTAL": total, "SPLITS": splits, "BLOCK_SIZE": block_size},
        (_ceil_div(total, block_size), 1, 1),
        [("tensor", None), ("tensor", None)],
    )


def _matmul_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
    architecture: int,
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | bool],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    if len(tensors) != 3:
        raise ValueError("MatMul tensor count is invalid")
    a, b, output = tensors
    data_types = [tensor["data_type"] for tensor in tensors]
    mixed_fp16 = parameters.get("_fprop_mixed_fp16", False)
    if not isinstance(mixed_fp16, bool):
        raise ValueError("internal mixed MatMul flag must be boolean")
    broadcast_a = parameters.get("_fprop_broadcast_a", False)
    if not isinstance(broadcast_a, bool):
        raise ValueError("internal broadcast-A MatMul flag must be boolean")
    lowp_inputs_fp32_output = parameters.get(
        "_fprop_lowp_inputs_fp32_output", False
    )
    if not isinstance(lowp_inputs_fp32_output, bool):
        raise ValueError(
            "internal low-precision-input MatMul flag must be boolean"
        )
    valid_data_types = (
        data_types[0] == data_types[1]
        and data_types[0] in {"float16", "bfloat16"}
        and data_types[2] in {data_types[0], "float32"}
        if broadcast_a
        else (
            data_types[0] == data_types[1]
            and data_types[0] in {"float16", "bfloat16"}
            and data_types[2] == "float32"
            if lowp_inputs_fp32_output
            else (
                data_types == ["float32", "float16", "float32"]
                if mixed_fp16
                else len(set(data_types)) == 1
                and data_types[0] in FLOAT_DATA_TYPES
            )
        )
    )
    if not valid_data_types:
        raise ValueError(
            "MatMul input/output data types must match and be floating"
        )
    if any(
        len(tensor["dimensions"]) < 2
        or len(tensor["dimensions"]) > 8
        or not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        )
        for tensor in tensors
    ):
        raise ValueError(
            "MatMul tensors require rank [2, 8] and non-overlapping strides"
        )

    m = a["dimensions"][-2]
    k = a["dimensions"][-1]
    if k != b["dimensions"][-2]:
        raise ValueError("MatMul contraction dimensions do not match")
    n = b["dimensions"][-1]
    a_batch = a["dimensions"][:-2]
    b_batch = b["dimensions"][:-2]
    batch_rank = max(len(a_batch), len(b_batch))
    if batch_rank > 6:
        raise ValueError("MatMul batch rank exceeds six")
    batch_dimensions = [1] * batch_rank
    for trailing in range(batch_rank):
        a_dimension = a_batch[-1 - trailing] if trailing < len(a_batch) else 1
        b_dimension = b_batch[-1 - trailing] if trailing < len(b_batch) else 1
        if (
            a_dimension != b_dimension
            and a_dimension != 1
            and b_dimension != 1
        ):
            raise ValueError(
                "MatMul batch dimensions are not broadcast-compatible"
            )
        batch_dimensions[-1 - trailing] = max(a_dimension, b_dimension)
    expected_output = [*batch_dimensions, m, n]
    if output["dimensions"] != expected_output:
        raise ValueError("MatMul output metadata is inconsistent")

    batch = math.prod(batch_dimensions)
    for name, expected in (("batch", batch), ("m", m), ("n", n), ("k", k)):
        if _require_integer(parameters, name) != expected:
            raise ValueError(
                f"parameters.{name} is inconsistent with MatMul metadata"
            )

    if broadcast_a:
        if (
            mixed_fp16
            or len(a["dimensions"]) != 3
            or len(b["dimensions"]) != 3
            or len(output["dimensions"]) != 3
            or a["dimensions"] != [1, m, k]
            or b["dimensions"] != [batch, k, n]
            or output["dimensions"] != [batch, m, n]
            or not all(
                _is_row_major_contiguous(tensor) for tensor in (a, b, output)
            )
        ):
            raise ValueError("broadcast-A MatMul metadata is inconsistent")
        block_m = 64
        block_n = 64
        block_k = 32
        return (
            "matmul_batched_broadcast_a_kernel",
            {
                "a_ptr": TRITON_POINTER_TYPES[data_types[0]],
                "b_ptr": TRITON_POINTER_TYPES[data_types[1]],
                "c_ptr": TRITON_POINTER_TYPES[data_types[2]],
            },
            {
                "BATCH": batch,
                "M": m,
                "N": n,
                "K": k,
                "INPUT_IS_FLOAT32": data_types[0] == "float32",
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "GROUP_M": 8,
            },
            (
                _ceil_div(m, block_m) * _ceil_div(n, block_n),
                batch,
                1,
            ),
            [("tensor", None), ("tensor", None), ("tensor", None)],
        )

    is_batched_contiguous = (
        not mixed_fp16
        and len(a["dimensions"]) == 3
        and len(b["dimensions"]) == 3
        and len(output["dimensions"]) == 3
        and a["dimensions"] == [batch, m, k]
        and b["dimensions"] == [batch, k, n]
        and output["dimensions"] == [batch, m, n]
        and all(_is_row_major_contiguous(tensor) for tensor in (a, b, output))
    )
    if (
        is_batched_contiguous
        and data_types[0] != "float32"
        and architecture >= 90
        and min(m, n, k) >= 512
    ):
        block_m = 128
        block_n = 128
        block_k = 64
        total_tiles = batch * _ceil_div(m, block_m) * _ceil_div(n, block_n)
        pointer_type = TRITON_POINTER_TYPES[data_types[0]]
        return (
            "matmul_batched_tma_persistent_kernel",
            {
                "a_ptr": pointer_type,
                "b_ptr": pointer_type,
                "c_ptr": pointer_type,
            },
            {
                "BATCH": batch,
                "M": m,
                "N": n,
                "K": k,
                "PERSISTENT_GRID": 128,
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "GROUP_M": 8,
                "INPUT_IS_FLOAT32": data_types[0] == "float32",
            },
            (min(total_tiles, 128), 1, 1),
            [("tensor", None), ("tensor", None), ("tensor", None)],
        )

    if is_batched_contiguous:
        block_m = 32 if m < 64 else 128
        block_n = 32 if n < 64 else 128
        block_k = 32 if k < 64 else 64
        pointer_type = TRITON_POINTER_TYPES[data_types[0]]
        return (
            "matmul_batched_contiguous_kernel",
            {
                "a_ptr": pointer_type,
                "b_ptr": pointer_type,
                "c_ptr": pointer_type,
            },
            {
                "BATCH": batch,
                "M": m,
                "N": n,
                "K": k,
                "INPUT_IS_FLOAT32": data_types[0] == "float32",
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "GROUP_M": 8,
            },
            (
                _ceil_div(m, block_m) * _ceil_div(n, block_n),
                batch,
                1,
            ),
            [("tensor", None), ("tensor", None), ("tensor", None)],
        )

    def batch_strides(tensor: dict[str, Any]) -> list[int]:
        tensor_batch = tensor["dimensions"][:-2]
        leading = batch_rank - len(tensor_batch)
        dimensions = [1] * leading + tensor_batch
        strides = [0] * leading + tensor["strides"][:-2]
        effective = [
            0 if dimension == 1 else stride
            for dimension, stride in zip(dimensions, strides)
        ]
        return [0] * (6 - batch_rank) + effective

    padded_dimensions = [1] * (6 - batch_rank) + batch_dimensions
    a_batch_strides = batch_strides(a)
    b_batch_strides = batch_strides(b)
    output_batch_strides = [0] * (6 - batch_rank) + output["strides"][:-2]
    block_m = 32 if m < 64 else 64
    block_n = 32 if n < 64 else 64
    block_k = 32
    constants: dict[str, int | bool] = {
        "M": m,
        "N": n,
        "K": k,
        "A_STRIDE_M": a["strides"][-2],
        "A_STRIDE_K": a["strides"][-1],
        "B_STRIDE_K": b["strides"][-2],
        "B_STRIDE_N": b["strides"][-1],
        "C_STRIDE_M": output["strides"][-2],
        "C_STRIDE_N": output["strides"][-1],
        "INPUT_IS_FLOAT32": data_types[0] == "float32",
        "USE_TF32": data_types[0] == "float32",
        "MIXED_FP16": mixed_fp16,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "GROUP_M": 8,
    }
    for axis in range(6):
        constants[f"DIM_{axis}"] = padded_dimensions[axis]
        constants[f"A_BATCH_STRIDE_{axis}"] = a_batch_strides[axis]
        constants[f"B_BATCH_STRIDE_{axis}"] = b_batch_strides[axis]
        constants[f"C_BATCH_STRIDE_{axis}"] = output_batch_strides[axis]

    return (
        "matmul_strided_kernel",
        {
            "a_ptr": TRITON_POINTER_TYPES[data_types[0]],
            "b_ptr": TRITON_POINTER_TYPES[data_types[1]],
            "c_ptr": TRITON_POINTER_TYPES[data_types[2]],
        },
        constants,
        (
            ((m + block_m - 1) // block_m) * ((n + block_n - 1) // block_n),
            batch,
            1,
        ),
        [("tensor", None), ("tensor", None), ("tensor", None)],
    )


def _layout_kernel_configuration(
    operation: str,
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    if len(tensors) != 2:
        raise ValueError("layout operation tensor count is invalid")
    input_tensor, output_tensor = tensors
    data_types = [tensor["data_type"] for tensor in tensors]
    if len(set(data_types)) != 1:
        raise ValueError("layout operation input/output data types must match")
    pointer_type = TRITON_POINTER_TYPES.get(data_types[0])
    if pointer_type is None:
        raise ValueError(
            f"unsupported layout operation data type: {data_types[0]!r}"
        )
    if any(
        len(tensor["dimensions"]) > 8
        or not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        )
        for tensor in tensors
    ):
        raise ValueError(
            "layout tensors require rank at most eight and "
            "non-overlapping strides"
        )

    input_dimensions = input_tensor["dimensions"]
    input_strides = input_tensor["strides"]
    output_dimensions = output_tensor["dimensions"]
    output_strides = output_tensor["strides"]
    input_rank = len(input_dimensions)
    output_rank = len(output_dimensions)
    elements = _require_integer(parameters, "n_elements")
    if elements != math.prod(output_dimensions):
        raise ValueError(
            "parameters.n_elements is inconsistent with layout output"
        )

    def require_metadata(
        name: str,
        expected: list[int],
        *,
        minimum: int,
    ) -> list[int]:
        value = _require_integer_list(
            parameters,
            name,
            len(expected),
            minimum=minimum,
            maximum=2**63 - 1,
        )
        if value != expected:
            raise ValueError(
                f"parameters.{name} is inconsistent with tensor metadata"
            )
        return value

    require_metadata("input_dimensions", input_dimensions, minimum=1)
    require_metadata("input_strides", input_strides, minimum=1)
    require_metadata("output_dimensions", output_dimensions, minimum=1)
    require_metadata("output_strides", output_strides, minimum=1)

    input_base = 0
    logical_input_dimensions = input_dimensions
    logical_input_strides = input_strides
    if operation == "reshape":
        if (
            _require_integer(parameters, "input_rank", minimum=0, maximum=8)
            != input_rank
            or _require_integer(
                parameters, "output_rank", minimum=0, maximum=8
            )
            != output_rank
        ):
            raise ValueError("reshape rank parameters are inconsistent")
        _require_integer(parameters, "reshape_mode", minimum=2, maximum=2)
        if math.prod(input_dimensions) != elements:
            raise ValueError("reshape input/output element counts must match")
    elif operation == "transpose":
        if input_rank == 0 or input_rank != output_rank:
            raise ValueError(
                "transpose input/output ranks must match in [1, 8]"
            )
        if (
            _require_integer(parameters, "rank", minimum=1, maximum=8)
            != input_rank
        ):
            raise ValueError("transpose rank parameter is inconsistent")
        permutation = _require_integer_list(
            parameters,
            "permutation",
            input_rank,
            minimum=0,
            maximum=input_rank - 1,
        )
        if sorted(permutation) != list(range(input_rank)):
            raise ValueError(
                "transpose permutation must contain each axis once"
            )
        expected_output = [input_dimensions[axis] for axis in permutation]
        if output_dimensions != expected_output:
            raise ValueError(
                "transpose output shape does not match permutation"
            )
        logical_input_dimensions = output_dimensions
        logical_input_strides = [input_strides[axis] for axis in permutation]
    elif operation == "slice":
        if input_rank == 0 or input_rank != output_rank:
            raise ValueError("slice input/output ranks must match in [1, 8]")
        if (
            _require_integer(parameters, "rank", minimum=1, maximum=8)
            != input_rank
        ):
            raise ValueError("slice rank parameter is inconsistent")
        starts = _require_integer_list(
            parameters,
            "starts",
            input_rank,
            minimum=0,
            maximum=2**63 - 1,
        )
        limits = _require_integer_list(
            parameters,
            "limits",
            input_rank,
            minimum=1,
            maximum=2**63 - 1,
        )
        slice_strides = _require_integer_list(
            parameters,
            "slice_strides",
            input_rank,
            minimum=1,
            maximum=2**63 - 1,
        )
        expected_output: list[int] = []
        for axis in range(input_rank):
            if (
                starts[axis] >= limits[axis]
                or limits[axis] > input_dimensions[axis]
            ):
                raise ValueError("slice range is outside input shape")
            expected_output.append(
                (limits[axis] - starts[axis] + slice_strides[axis] - 1)
                // slice_strides[axis]
            )
        if output_dimensions != expected_output:
            raise ValueError(
                "slice output shape does not match slice attributes"
            )
        input_base = sum(
            start * stride for start, stride in zip(starts, input_strides)
        )
        logical_input_dimensions = output_dimensions
        logical_input_strides = [
            stride * step for stride, step in zip(input_strides, slice_strides)
        ]
    else:
        raise ValueError(f"unknown layout operation: {operation!r}")

    leading_input = 8 - len(logical_input_dimensions)
    leading_output = 8 - output_rank
    padded_input_dimensions = [1] * leading_input + logical_input_dimensions
    padded_input_strides = [0] * leading_input + logical_input_strides
    padded_output_dimensions = [1] * leading_output + output_dimensions
    padded_output_strides = [0] * leading_output + output_strides
    block = 256
    constants: dict[str, int] = {
        "INPUT_BASE": input_base,
        "BLOCK_SIZE": block,
    }
    for axis in range(8):
        constants[f"INPUT_DIM_{axis}"] = padded_input_dimensions[axis]
        constants[f"INPUT_STRIDE_{axis}"] = padded_input_strides[axis]
        constants[f"OUTPUT_DIM_{axis}"] = padded_output_dimensions[axis]
        constants[f"OUTPUT_STRIDE_{axis}"] = padded_output_strides[axis]
    return (
        "layout_copy_kernel",
        {
            "input_ptr": pointer_type,
            "output_ptr": pointer_type,
            "n_elements": "i32",
        },
        constants,
        ((elements + block - 1) // block, 1, 1),
        [
            ("tensor", None),
            ("tensor", None),
            ("scalar_i32", "n_elements"),
        ],
    )


def _normalization_forward_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
    *,
    rmsnorm: bool,
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | float | str | bool],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    expected_count = 5 if rmsnorm else 6
    if len(tensors) != expected_count:
        raise ValueError("normalization tensor count is invalid")
    x, scale, bias, y = tensors[:4]
    statistics = tensors[4:]
    if (
        x["data_type"] not in FLOAT_DATA_TYPES
        or y["data_type"] != x["data_type"]
        or scale["data_type"] != x["data_type"]
        or bias["data_type"] != x["data_type"]
    ):
        raise ValueError("normalization X/Y/scale/bias data types must match")
    if x["dimensions"] != y["dimensions"]:
        raise ValueError("normalization Y shape must match X")
    if not _is_row_major_contiguous(x) or not _is_row_major_contiguous(y):
        raise ValueError("normalization X/Y must be contiguous")
    if not _is_row_major_contiguous(scale) or not _is_row_major_contiguous(
        bias
    ):
        raise ValueError("normalization scale/bias must be contiguous")

    rows = _require_integer(parameters, "rows")
    normalized_elements = _require_integer(parameters, "normalized_elements")
    if math.prod(x["dimensions"]) != rows * normalized_elements:
        raise ValueError("normalization row/extent parameters are invalid")
    if (
        math.prod(scale["dimensions"]) != normalized_elements
        or math.prod(bias["dimensions"]) != normalized_elements
    ):
        raise ValueError("normalization scale/bias size is invalid")
    for statistic in statistics:
        if (
            statistic["data_type"] != "float32"
            or math.prod(statistic["dimensions"]) != rows
            or not _is_row_major_contiguous(statistic)
        ):
            raise ValueError("normalization statistic metadata is invalid")
    epsilon = _require_number(parameters, "epsilon")
    if epsilon <= 0.0:
        raise ValueError("normalization epsilon must be positive")
    block = 1 << (normalized_elements - 1).bit_length()
    if block > 65536:
        raise ValueError("normalization extent exceeds kernel limit")
    rows_per_program = 1
    pointer_types = [
        TRITON_POINTER_TYPES.get(tensor["data_type"]) for tensor in tensors
    ]
    if any(pointer_type is None for pointer_type in pointer_types):
        raise ValueError("unsupported normalization data type")

    if rmsnorm:
        return (
            "rms_norm_kernel",
            {
                "x_ptr": pointer_types[0],
                "y_ptr": pointer_types[3],
                "weight_ptr": pointer_types[1],
                "bias_ptr": pointer_types[2],
                "inv_variance_ptr": pointer_types[4],
                "M": "i32",
            },
            {
                "N": normalized_elements,
                "eps": epsilon,
                "BLOCK_SIZE": block,
                "ROWS_PER_PROGRAM": rows_per_program,
                "HAS_WEIGHT": True,
                "HAS_BIAS": True,
                "RETURN_STATS": True,
            },
            ((rows + rows_per_program - 1) // rows_per_program, 1, 1),
            [
                ("tensor_alias", 0),
                ("tensor_alias", 3),
                ("tensor_alias", 1),
                ("tensor_alias", 2),
                ("tensor_alias", 4),
                ("scalar_i32", "rows"),
            ],
        )
    return (
        "layer_norm_kernel",
        {
            "x_ptr": pointer_types[0],
            "y_ptr": pointer_types[3],
            "mean_ptr": pointer_types[4],
            "inv_variance_ptr": pointer_types[5],
            "weight_ptr": pointer_types[1],
            "bias_ptr": pointer_types[2],
            "M": "i32",
        },
        {
            "eps": epsilon,
            "N": normalized_elements,
            "BLOCK_SIZE": block,
            "ROWS_PER_PROGRAM": rows_per_program,
            "HAS_WEIGHT": True,
            "HAS_BIAS": True,
            "RETURN_STATS": True,
        },
        ((rows + rows_per_program - 1) // rows_per_program, 1, 1),
        [
            ("tensor_alias", 0),
            ("tensor_alias", 3),
            ("tensor_alias", 4),
            ("tensor_alias", 5),
            ("tensor_alias", 1),
            ("tensor_alias", 2),
            ("scalar_i32", "rows"),
        ],
    )


def _batchnorm_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | float | str | bool],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    if len(tensors) != 10:
        raise ValueError("batchnorm tensor count is invalid")
    (
        x,
        scale,
        bias,
        previous_mean,
        previous_variance,
        y,
        mean,
        inv_variance,
        next_mean,
        next_variance,
    ) = tensors
    if (
        x["data_type"] != y["data_type"]
        or x["data_type"] not in FLOAT_DATA_TYPES
        or scale["data_type"] != x["data_type"]
        or bias["data_type"] != x["data_type"]
    ):
        raise ValueError(
            "batchnorm X/Y/scale/bias data types must match and be floating"
        )
    if x["dimensions"] != y["dimensions"]:
        raise ValueError("batchnorm Y shape must match X")
    if len(x["dimensions"]) < 2 or len(x["dimensions"]) > 8:
        raise ValueError("batchnorm X rank must be in [2, 8]")

    channels = x["dimensions"][1]
    for parameter_name, tensor in (("scale", scale), ("bias", bias)):
        if math.prod(tensor["dimensions"]) != channels:
            raise ValueError(
                f"batchnorm {parameter_name} size must match channels"
            )
        if not _is_row_major_contiguous(tensor):
            raise ValueError(f"batchnorm {parameter_name} must be contiguous")
    for statistic_name, tensor in (
        ("previous_running_mean", previous_mean),
        ("previous_running_variance", previous_variance),
        ("mean", mean),
        ("inv_variance", inv_variance),
        ("next_running_mean", next_mean),
        ("next_running_variance", next_variance),
    ):
        if tensor["data_type"] != "float32":
            raise ValueError(f"batchnorm {statistic_name} must use float32")
        if math.prod(tensor["dimensions"]) != channels:
            raise ValueError(
                f"batchnorm {statistic_name} size must match channels"
            )
        if not _is_row_major_contiguous(tensor):
            raise ValueError(f"batchnorm {statistic_name} must be contiguous")

    total_elements = _require_integer(parameters, "n_elements")
    batch = _require_integer(parameters, "batch")
    configured_channels = _require_integer(parameters, "channels")
    spatial = _require_integer(parameters, "spatial")
    rank = _require_integer(parameters, "rank")
    if total_elements != math.prod(x["dimensions"]):
        raise ValueError(
            "parameters.n_elements is inconsistent with batchnorm X"
        )
    if batch != x["dimensions"][0]:
        raise ValueError("parameters.batch is inconsistent with batchnorm X")
    if configured_channels != channels:
        raise ValueError(
            "parameters.channels is inconsistent with batchnorm X"
        )
    if spatial != math.prod(x["dimensions"][2:]):
        raise ValueError("parameters.spatial is inconsistent with batchnorm X")
    if rank != len(x["dimensions"]):
        raise ValueError("parameters.rank is inconsistent with batchnorm X")
    epsilon = _require_number(parameters, "epsilon")
    momentum = _require_number(parameters, "momentum")
    if epsilon <= 0.0:
        raise ValueError("parameters.epsilon must be positive")
    if momentum < 0.0 or momentum > 1.0:
        raise ValueError("parameters.momentum must be in [0, 1]")

    pointer_types = [
        TRITON_POINTER_TYPES.get(tensor["data_type"]) for tensor in tensors
    ]
    if any(pointer_type is None for pointer_type in pointer_types):
        raise ValueError("unsupported batchnorm data type")
    block = 256
    constants: dict[str, int | float | str | bool] = {
        "eps": epsilon,
        "momentum": momentum,
        "BLOCK_SIZE": block,
        "IS_TRAINING": True,
        "HAS_WEIGHT": True,
        "HAS_BIAS": True,
        "HAS_RUNNING_STATS": True,
        "RETURN_STATS": True,
    }
    constants.update(_unary_pointwise_tensor_constants([x, y]))
    # BatchNorm derives the parameter channel from the logical NCHW index.
    # A physically dense channels-last tensor therefore cannot use the
    # pointwise helper's physical-linear fast path even when X and Y have the
    # same mapping.
    constants["STRIDED"] = not (
        _is_row_major_contiguous(x) and _is_row_major_contiguous(y)
    )
    if (
        spatial >= 256
        and _is_row_major_contiguous(x)
        and _is_row_major_contiguous(y)
    ):
        return (
            "batch_norm_nchw_kernel",
            {
                "x_ptr": pointer_types[0],
                "y_ptr": pointer_types[5],
                "mean_ptr": pointer_types[3],
                "var_ptr": pointer_types[4],
                "weight_ptr": pointer_types[1],
                "bias_ptr": pointer_types[2],
                "saved_mean_ptr": pointer_types[6],
                "saved_inv_var_ptr": pointer_types[7],
                "next_running_mean_ptr": pointer_types[8],
                "next_running_var_ptr": pointer_types[9],
            },
            {
                "N": batch,
                "C": channels,
                "S": spatial,
                "eps": epsilon,
                "momentum": momentum,
                "BLOCK_SIZE": block,
                "IS_TRAINING": True,
                "HAS_WEIGHT": True,
                "HAS_BIAS": True,
                "HAS_RUNNING_STATS": True,
                "RETURN_STATS": True,
            },
            (channels, 1, 1),
            [
                ("tensor_alias", 0),
                ("tensor_alias", 5),
                ("tensor_alias", 3),
                ("tensor_alias", 4),
                ("tensor_alias", 1),
                ("tensor_alias", 2),
                ("tensor_alias", 6),
                ("tensor_alias", 7),
                ("tensor_alias", 8),
                ("tensor_alias", 9),
            ],
        )
    return (
        "batch_norm_kernel",
        {
            "x_ptr": pointer_types[0],
            "y_ptr": pointer_types[5],
            "mean_ptr": pointer_types[3],
            "var_ptr": pointer_types[4],
            "weight_ptr": pointer_types[1],
            "bias_ptr": pointer_types[2],
            "saved_mean_ptr": pointer_types[6],
            "saved_inv_var_ptr": pointer_types[7],
            "next_running_mean_ptr": pointer_types[8],
            "next_running_var_ptr": pointer_types[9],
            "N": "i32",
            "C": "i32",
            "S": "i32",
        },
        constants,
        (channels, 1, 1),
        [
            ("tensor_alias", 0),
            ("tensor_alias", 5),
            ("tensor_alias", 3),
            ("tensor_alias", 4),
            ("tensor_alias", 1),
            ("tensor_alias", 2),
            ("tensor_alias", 6),
            ("tensor_alias", 7),
            ("tensor_alias", 8),
            ("tensor_alias", 9),
            ("scalar_i32", "batch"),
            ("scalar_i32", "channels"),
            ("scalar_i32", "spatial"),
        ],
    )


def _batchnorm_inference_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | float | str | bool],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    if len(tensors) != 6:
        raise ValueError("batchnorm inference tensor count is invalid")
    x, mean, inv_variance, scale, bias, y = tensors
    if (
        x["data_type"] != y["data_type"]
        or x["data_type"] not in FLOAT_DATA_TYPES
    ):
        raise ValueError(
            "batchnorm inference X/Y data types must match and be floating"
        )
    if x["dimensions"] != y["dimensions"]:
        raise ValueError("batchnorm inference Y shape must match X")
    if len(x["dimensions"]) < 2 or len(x["dimensions"]) > 8:
        raise ValueError("batchnorm inference X rank must be in [2, 8]")

    channels = x["dimensions"][1]
    for parameter_name, tensor in (
        ("mean", mean),
        ("inv_variance", inv_variance),
        ("scale", scale),
        ("bias", bias),
    ):
        if tensor["data_type"] not in FLOAT_DATA_TYPES:
            raise ValueError(
                f"batchnorm inference {parameter_name} must be floating"
            )
        if math.prod(tensor["dimensions"]) != channels:
            raise ValueError(
                f"batchnorm inference {parameter_name} size must match "
                "channels"
            )
        if not _is_row_major_contiguous(tensor):
            raise ValueError(
                f"batchnorm inference {parameter_name} must be contiguous"
            )

    total_elements = _require_integer(parameters, "n_elements")
    configured_channels = _require_integer(parameters, "channels")
    spatial = _require_integer(parameters, "spatial")
    rank = _require_integer(parameters, "rank")
    if total_elements != math.prod(x["dimensions"]):
        raise ValueError(
            "parameters.n_elements is inconsistent with batchnorm inference X"
        )
    if configured_channels != channels:
        raise ValueError(
            "parameters.channels is inconsistent with batchnorm inference X"
        )
    if spatial != math.prod(x["dimensions"][2:]):
        raise ValueError(
            "parameters.spatial is inconsistent with batchnorm inference X"
        )
    if rank != len(x["dimensions"]):
        raise ValueError(
            "parameters.rank is inconsistent with batchnorm inference X"
        )

    pointer_types = [
        TRITON_POINTER_TYPES.get(tensor["data_type"]) for tensor in tensors
    ]
    if any(pointer_type is None for pointer_type in pointer_types):
        raise ValueError("unsupported batchnorm inference data type")
    block = 256
    constants: dict[str, int | float | str | bool] = {
        "eps": 0.0,
        "BLOCK_SIZE": block,
        "HAS_WEIGHT": True,
        "HAS_BIAS": True,
        "STAT_IS_INV_VARIANCE": True,
    }
    if _is_row_major_contiguous(x) and _is_row_major_contiguous(y):
        constants.update({"C": channels, "S": spatial})
        batch = total_elements // (channels * spatial)
        block_s = min(1 << (spatial - 1).bit_length(), block)
        block_c = block // block_s
        spatial_blocks = (spatial + block_s - 1) // block_s
        channel_blocks = (channels + block_c - 1) // block_c
        return (
            "batch_norm_inference_nchw_kernel",
            {
                "x_ptr": pointer_types[0],
                "mean_ptr": pointer_types[1],
                "stat_ptr": pointer_types[2],
                "weight_ptr": pointer_types[3],
                "bias_ptr": pointer_types[4],
                "y_ptr": pointer_types[5],
            },
            constants,
            (batch * channel_blocks * spatial_blocks, 1, 1),
            [("tensor", None)] * 6,
        )

    constants.update(_unary_pointwise_tensor_constants([x, y]))
    constants["STRIDED"] = not (
        _is_row_major_contiguous(x) and _is_row_major_contiguous(y)
    )
    return (
        "batch_norm_inference_kernel",
        {
            "x_ptr": pointer_types[0],
            "mean_ptr": pointer_types[1],
            "stat_ptr": pointer_types[2],
            "weight_ptr": pointer_types[3],
            "bias_ptr": pointer_types[4],
            "y_ptr": pointer_types[5],
            "total_elements": "i32",
            "C": "i32",
            "S": "i32",
        },
        constants,
        ((total_elements + block - 1) // block, 1, 1),
        [
            ("tensor", None),
            ("tensor", None),
            ("tensor", None),
            ("tensor", None),
            ("tensor", None),
            ("tensor", None),
            ("scalar_i32", "n_elements"),
            ("scalar_i32", "channels"),
            ("scalar_i32", "spatial"),
        ],
    )


def _attention_flag(parameters: dict[str, Any], name: str) -> bool:
    return _require_integer(parameters, name, minimum=0, maximum=1) == 1


def _attention_runtime_i32(
    parameters: dict[str, Any], name: str, value: int
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"SDPA runtime parameter {name} must be an integer")
    if value < -(2**31) or value > 2**31 - 1:
        raise ValueError(f"SDPA runtime parameter {name} exceeds int32")
    parameters[name] = value
    return value


def _attention_runtime_f32(
    parameters: dict[str, Any], name: str, value: float
) -> float:
    result = float(value)
    if not math.isfinite(result) or abs(result) > _FLOAT32_MAX:
        raise ValueError(f"SDPA runtime parameter {name} exceeds float32")
    parameters[name] = result
    return result


def _attention_strides(
    tensor: dict[str, Any], prefix: str, axes: str
) -> dict[str, int]:
    strides = tensor["strides"]
    if len(strides) < len(axes):
        raise ValueError(f"SDPA {prefix} tensor rank is invalid")
    return {
        f"stride_{prefix}{axis}": int(stride)
        for axis, stride in zip(axes, strides)
    }


def _attention_broadcast_strides(
    tensor: dict[str, Any], prefix: str
) -> dict[str, int]:
    dimensions = tensor["dimensions"]
    strides = tensor["strides"]
    if len(dimensions) != 4 or len(strides) != 4:
        raise ValueError("SDPA bias tensor must be rank four")
    return {
        f"stride_{prefix}b": 0 if dimensions[0] == 1 else int(strides[0]),
        f"stride_{prefix}h": 0 if dimensions[1] == 1 else int(strides[1]),
        f"stride_{prefix}m": int(strides[2]),
        f"stride_{prefix}n": int(strides[3]),
    }


def _validate_attention_base(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
    *,
    fp8: bool = False,
) -> tuple[int, int, int, int, int, int, int, int]:
    if len(tensors) < 3:
        raise ValueError("SDPA tensor count is invalid")
    q, k, v = tensors[:3]
    if any(len(tensor["dimensions"]) != 4 for tensor in (q, k, v)):
        raise ValueError("SDPA Q/K/V must be rank-four BHSD tensors")
    allowed_data_types = FP8_DATA_TYPES if fp8 else FLOAT_DATA_TYPES
    if (
        q["data_type"] not in allowed_data_types
        or k["data_type"] != q["data_type"]
        or v["data_type"] != q["data_type"]
    ):
        kind = "FP8" if fp8 else "floating"
        raise ValueError(f"SDPA Q/K/V data types must match and be {kind}")
    batch, heads, sequence_q, head_dimension = q["dimensions"]
    key_heads = k["dimensions"][1]
    value_heads = v["dimensions"][1]
    sequence_kv = k["dimensions"][2]
    value_dimension = v["dimensions"][3]
    if (
        k["dimensions"][0] != batch
        or v["dimensions"][0] != batch
        or k["dimensions"][3] != head_dimension
        or v["dimensions"][2] != sequence_kv
        or heads % key_heads != 0
        or heads % value_heads != 0
    ):
        raise ValueError("SDPA Q/K/V shapes are inconsistent")
    expected = {
        "batch": batch,
        "heads": heads,
        "key_heads": key_heads,
        "value_heads": value_heads,
        "sequence_q": sequence_q,
        "sequence_kv": sequence_kv,
        "head_dimension": head_dimension,
        "value_dimension": value_dimension,
        "q_per_k": heads // key_heads,
        "q_per_v": heads // value_heads,
    }
    for name, value in expected.items():
        if _require_integer(parameters, name) != value:
            raise ValueError(
                f"parameters.{name} is inconsistent with SDPA tensors"
            )
    if head_dimension > 256 or value_dimension > 256:
        raise ValueError(
            "SDPA head dimensions greater than 256 are unsupported"
        )
    return (
        batch,
        heads,
        key_heads,
        value_heads,
        sequence_q,
        sequence_kv,
        head_dimension,
        value_dimension,
    )


def _sdpa_forward_kernel_configuration(
    parameters: dict[str, Any], tensors: list[dict[str, Any]]
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | float | str | bool],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    has_bias = _attention_flag(parameters, "has_bias")
    if len(tensors) != (6 if has_bias else 5):
        raise ValueError("SDPA forward tensor count is invalid")
    (
        batch,
        heads,
        _,
        _,
        sequence_q,
        sequence_kv,
        head_dimension,
        value_dimension,
    ) = _validate_attention_base(parameters, tensors)
    q, k, v = tensors[:3]
    bias_index = 3 if has_bias else None
    output_index = 4 if has_bias else 3
    stats_index = output_index + 1
    output = tensors[output_index]
    stats = tensors[stats_index]
    if (
        output["data_type"] != q["data_type"]
        or output["dimensions"] != [batch, heads, sequence_q, value_dimension]
        or stats["data_type"] != "float32"
        or stats["dimensions"] != [batch, heads, sequence_q, 1]
    ):
        raise ValueError("SDPA forward output metadata is invalid")
    if has_bias:
        bias = tensors[bias_index]
        if (
            bias["data_type"] != q["data_type"]
            or bias["dimensions"][0] not in {1, batch}
            or bias["dimensions"][1] not in {1, heads}
            or bias["dimensions"][2:] != [sequence_q, sequence_kv]
        ):
            raise ValueError("SDPA bias metadata is invalid")
        bias_strides = _attention_broadcast_strides(bias, "bias_")
    else:
        bias_strides = {
            "stride_bias_b": 0,
            "stride_bias_h": 0,
            "stride_bias_m": 0,
            "stride_bias_n": 0,
        }

    runtime_values: dict[str, int | float] = {
        "qk_scale": _require_number(parameters, "attn_scale")
        * 1.4426950408889634,
        "HQ": heads,
        "SQ": sequence_q,
        "SKV": sequence_kv,
        "q_per_k": _require_integer(parameters, "q_per_k"),
        "q_per_v": _require_integer(parameters, "q_per_v"),
        "min_diag": _require_integer(
            parameters, "min_diag", minimum=-(2**31), maximum=2**31 - 1
        ),
        "max_diag": _require_integer(
            parameters, "max_diag", minimum=-(2**31), maximum=2**31 - 1
        ),
        **_attention_strides(q, "q", "bhmd"),
        **_attention_strides(k, "k", "bhnd"),
        **_attention_strides(v, "v", "bhnd"),
        **bias_strides,
        **_attention_strides(output, "o", "bhmd"),
        **_attention_strides(stats, "s", "bhm"),
    }
    for name, value in runtime_values.items():
        if isinstance(value, float):
            _attention_runtime_f32(parameters, name, value)
        else:
            _attention_runtime_i32(parameters, name, value)

    pointer_type = TRITON_POINTER_TYPES[q["data_type"]]
    runtime_signature: dict[str, str] = {
        "q_ptr": pointer_type,
        "k_ptr": pointer_type,
        "v_ptr": pointer_type,
        "bias_ptr": pointer_type,
        "o_ptr": pointer_type,
        "stats_ptr": TRITON_POINTER_TYPES["float32"],
        "qk_scale": "fp32",
    }
    for name in (
        "HQ",
        "SQ",
        "SKV",
        "q_per_k",
        "q_per_v",
        "min_diag",
        "max_diag",
        "stride_qb",
        "stride_qh",
        "stride_qm",
        "stride_qd",
        "stride_kb",
        "stride_kh",
        "stride_kn",
        "stride_kd",
        "stride_vb",
        "stride_vh",
        "stride_vn",
        "stride_vd",
        "stride_bias_b",
        "stride_bias_h",
        "stride_bias_m",
        "stride_bias_n",
        "stride_ob",
        "stride_oh",
        "stride_om",
        "stride_od",
        "stride_sb",
        "stride_sh",
        "stride_sm",
    ):
        runtime_signature[name] = "i32"

    block_m = 64
    constants: dict[str, int | float | str | bool] = {
        "HEAD_DIM": head_dimension,
        "V_DIM": value_dimension,
        "ELEM_SIZE": 4 if q["data_type"] == "float32" else 2,
        "BLOCK_M": block_m,
        "BLOCK_N": 64,
        "BLOCK_D": max(16, _next_power_of_two(head_dimension)),
        "BLOCK_DV": max(16, _next_power_of_two(value_dimension)),
        "HAS_BIAS": has_bias,
        "BANDED": _attention_flag(parameters, "banded"),
        "GENERATE_STATS": _attention_flag(parameters, "generate_stats"),
        "REVERSE_CAUSAL": _attention_flag(parameters, "reverse_causal"),
    }
    pointer_layout = [
        ("tensor_alias", 0),
        ("tensor_alias", 1),
        ("tensor_alias", 2),
        ("tensor_alias", bias_index if bias_index is not None else 0),
        ("tensor_alias", output_index),
        ("tensor_alias", stats_index),
        ("scalar_f32", "qk_scale"),
    ]
    scalar_layout = [
        ("scalar_i32", name) for name in list(runtime_signature)[7:]
    ]
    return (
        "_sdpa_fwd_kernel",
        runtime_signature,
        constants,
        (_ceil_div(sequence_q, block_m), batch * heads, 1),
        pointer_layout + scalar_layout,
    )


def _require_fp8_scale_tensor(tensor: dict[str, Any], name: str) -> None:
    if (
        tensor["data_type"] != "float32"
        or math.prod(tensor["dimensions"]) != 1
    ):
        raise ValueError(f"{name} must be a one-element float32 tensor")


def _sdpa_fp8_forward_kernel_configuration(
    parameters: dict[str, Any], tensors: list[dict[str, Any]]
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | float | str | bool],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    stage = parameters.get("_sdpa_fp8_stage")
    if stage == "zero_amax":
        if len(tensors) != 2:
            raise ValueError(
                "FP8 SDPA amax zero stage tensor count is invalid"
            )
        _require_fp8_scale_tensor(tensors[0], "FP8 SDPA amax S")
        _require_fp8_scale_tensor(tensors[1], "FP8 SDPA amax O")
        return (
            "_zero_sdpa_fp8_fwd_amax_kernel",
            {
                "amax_s_ptr": TRITON_POINTER_TYPES["float32"],
                "amax_o_ptr": TRITON_POINTER_TYPES["float32"],
            },
            {},
            (1, 1, 1),
            [("tensor_alias", 0), ("tensor_alias", 1)],
        )

    if stage != "forward":
        raise ValueError("FP8 SDPA forward pipeline stage is invalid")
    has_bias = _attention_flag(parameters, "has_bias")
    if len(tensors) != 13 + int(has_bias):
        raise ValueError("FP8 SDPA forward tensor count is invalid")
    (
        batch,
        heads,
        _,
        _,
        sequence_q,
        sequence_kv,
        head_dimension,
        value_dimension,
    ) = _validate_attention_base(parameters, tensors, fp8=True)
    q, k, v = tensors[:3]
    for index, name in enumerate(
        (
            "descale Q",
            "descale K",
            "descale V",
            "descale S",
            "scale S",
            "scale O",
        ),
        start=3,
    ):
        _require_fp8_scale_tensor(tensors[index], f"FP8 SDPA {name}")

    bias_index = 9 if has_bias else None
    output_index = 10 if has_bias else 9
    stats_index = output_index + 1
    amax_s_index = output_index + 2
    amax_o_index = output_index + 3
    output = tensors[output_index]
    stats = tensors[stats_index]
    if (
        output["data_type"] != q["data_type"]
        or output["dimensions"] != [batch, heads, sequence_q, value_dimension]
        or stats["data_type"] != "float32"
        or stats["dimensions"] != [batch, heads, sequence_q, 1]
    ):
        raise ValueError("FP8 SDPA forward output metadata is invalid")
    _require_fp8_scale_tensor(tensors[amax_s_index], "FP8 SDPA amax S")
    _require_fp8_scale_tensor(tensors[amax_o_index], "FP8 SDPA amax O")

    if has_bias:
        bias = tensors[bias_index]
        if (
            bias["data_type"] not in FLOAT_DATA_TYPES
            or len(bias["dimensions"]) != 4
            or bias["dimensions"][0] not in {1, batch}
            or bias["dimensions"][1] not in {1, heads}
            or bias["dimensions"][2:] != [sequence_q, sequence_kv]
        ):
            raise ValueError("FP8 SDPA bias metadata is invalid")
        bias_strides = _attention_broadcast_strides(bias, "bias_")
        bias_pointer_type = TRITON_POINTER_TYPES[bias["data_type"]]
    else:
        bias_strides = {
            "stride_bias_b": 0,
            "stride_bias_h": 0,
            "stride_bias_m": 0,
            "stride_bias_n": 0,
        }
        bias_pointer_type = TRITON_POINTER_TYPES[q["data_type"]]

    runtime_values: dict[str, int | float] = {
        "attn_scale": _require_number(parameters, "attn_scale"),
        "HQ": heads,
        "SQ": sequence_q,
        "SKV": sequence_kv,
        "q_per_k": _require_integer(parameters, "q_per_k"),
        "q_per_v": _require_integer(parameters, "q_per_v"),
        "min_diag": _require_integer(
            parameters, "min_diag", minimum=-(2**31), maximum=2**31 - 1
        ),
        "max_diag": _require_integer(
            parameters, "max_diag", minimum=-(2**31), maximum=2**31 - 1
        ),
        **_attention_strides(q, "q", "bhmd"),
        **_attention_strides(k, "k", "bhnd"),
        **_attention_strides(v, "v", "bhnd"),
        **bias_strides,
        **_attention_strides(output, "o", "bhmd"),
        **_attention_strides(stats, "s", "bhm"),
    }
    for name, value in runtime_values.items():
        if isinstance(value, float):
            _attention_runtime_f32(parameters, name, value)
        else:
            _attention_runtime_i32(parameters, name, value)

    pointer_type = TRITON_POINTER_TYPES[q["data_type"]]
    runtime_signature: dict[str, str] = {
        "q_ptr": pointer_type,
        "k_ptr": pointer_type,
        "v_ptr": pointer_type,
        "bias_ptr": bias_pointer_type,
        "o_ptr": pointer_type,
        "stats_ptr": TRITON_POINTER_TYPES["float32"],
        "amax_s_ptr": TRITON_POINTER_TYPES["float32"],
        "amax_o_ptr": TRITON_POINTER_TYPES["float32"],
        "descale_q_ptr": TRITON_POINTER_TYPES["float32"],
        "descale_k_ptr": TRITON_POINTER_TYPES["float32"],
        "descale_v_ptr": TRITON_POINTER_TYPES["float32"],
        "descale_s_ptr": TRITON_POINTER_TYPES["float32"],
        "scale_s_ptr": TRITON_POINTER_TYPES["float32"],
        "scale_o_ptr": TRITON_POINTER_TYPES["float32"],
        "attn_scale": "fp32",
    }
    integer_runtime_names = (
        "HQ",
        "SQ",
        "SKV",
        "q_per_k",
        "q_per_v",
        "min_diag",
        "max_diag",
        "stride_qb",
        "stride_qh",
        "stride_qm",
        "stride_qd",
        "stride_kb",
        "stride_kh",
        "stride_kn",
        "stride_kd",
        "stride_vb",
        "stride_vh",
        "stride_vn",
        "stride_vd",
        "stride_bias_b",
        "stride_bias_h",
        "stride_bias_m",
        "stride_bias_n",
        "stride_ob",
        "stride_oh",
        "stride_om",
        "stride_od",
        "stride_sb",
        "stride_sh",
        "stride_sm",
    )
    for name in integer_runtime_names:
        runtime_signature[name] = "i32"

    block_m = min(64, max(16, _next_power_of_two(sequence_q)))
    constants: dict[str, int | float | str | bool] = {
        "HEAD_DIM": head_dimension,
        "V_DIM": value_dimension,
        "BLOCK_M": block_m,
        "BLOCK_N": 64,
        "BLOCK_D": max(16, _next_power_of_two(head_dimension)),
        "BLOCK_DV": max(16, _next_power_of_two(value_dimension)),
        "HAS_BIAS": has_bias,
        "BANDED": _attention_flag(parameters, "banded"),
        "GENERATE_STATS": _attention_flag(parameters, "generate_stats"),
        "REVERSE_CAUSAL": _attention_flag(parameters, "reverse_causal"),
    }
    pointer_layout: list[tuple[str, str | int | None]] = [
        ("tensor_alias", 0),
        ("tensor_alias", 1),
        ("tensor_alias", 2),
        ("tensor_alias", bias_index if bias_index is not None else 0),
        ("tensor_alias", output_index),
        ("tensor_alias", stats_index),
        ("tensor_alias", amax_s_index),
        ("tensor_alias", amax_o_index),
        *[("tensor_alias", index) for index in range(3, 9)],
        ("scalar_f32", "attn_scale"),
    ]
    scalar_layout = [("scalar_i32", name) for name in integer_runtime_names]
    return (
        "_sdpa_fp8_fwd_kernel",
        runtime_signature,
        constants,
        (_ceil_div(sequence_q, block_m), batch * heads, 1),
        pointer_layout + scalar_layout,
    )


def _sdpa_fp8_backward_kernel_configuration(
    parameters: dict[str, Any], tensors: list[dict[str, Any]]
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | float | str | bool],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    stage = parameters.get("_sdpa_fp8_bwd_stage")
    if stage == "zero_amax":
        if len(tensors) != 4:
            raise ValueError(
                "FP8 SDPA backward amax zero stage tensor count is invalid"
            )
        for tensor, name in zip(
            tensors, ("amax dQ", "amax dK", "amax dV", "amax dP")
        ):
            _require_fp8_scale_tensor(tensor, f"FP8 SDPA backward {name}")
        return (
            "_zero_sdpa_fp8_bwd_amax_kernel",
            {
                "amax_dq_ptr": TRITON_POINTER_TYPES["float32"],
                "amax_dk_ptr": TRITON_POINTER_TYPES["float32"],
                "amax_dv_ptr": TRITON_POINTER_TYPES["float32"],
                "amax_dp_ptr": TRITON_POINTER_TYPES["float32"],
            },
            {},
            (1, 1, 1),
            [("tensor_alias", index) for index in range(4)],
        )

    if stage not in {"dq", "dkdv"}:
        raise ValueError("FP8 SDPA backward pipeline stage is invalid")
    expected_count = 16 if stage == "dq" else 22
    if len(tensors) != expected_count:
        raise ValueError(
            f"FP8 SDPA backward {stage} stage tensor count is invalid"
        )
    (
        batch,
        heads,
        key_heads,
        value_heads,
        sequence_q,
        sequence_kv,
        head_dimension,
        value_dimension,
    ) = _validate_attention_base(parameters, tensors, fp8=True)
    if key_heads != value_heads or head_dimension != value_dimension:
        raise ValueError("FP8 SDPA backward requires matching K/V heads and D")
    if head_dimension > 128:
        raise ValueError("FP8 SDPA backward head dimension exceeds 128")

    q, k, v, output, doutput, stats = tensors[:6]
    expected_output = [batch, heads, sequence_q, value_dimension]
    if (
        output["dimensions"] != expected_output
        or doutput["dimensions"] != expected_output
    ):
        raise ValueError("FP8 SDPA backward O/dO metadata is invalid")
    if (
        output["data_type"] != q["data_type"]
        or doutput["data_type"] != q["data_type"]
        or stats["data_type"] != "float32"
        or stats["dimensions"] != [batch, heads, sequence_q, 1]
    ):
        raise ValueError("FP8 SDPA backward primal metadata is invalid")

    if stage == "dq":
        dq = tensors[14]
        amax_dq = tensors[15]
        if (
            dq["data_type"] != q["data_type"]
            or dq["dimensions"] != q["dimensions"]
        ):
            raise ValueError("FP8 SDPA backward dQ metadata is invalid")
        scale_names = (
            "descale Q",
            "descale K",
            "descale V",
            "descale O",
            "descale dO",
            "descale dP",
            "scale dQ",
            "scale dP",
        )
        scale_tensors = tensors[6:14]
        amax_tensors = ((amax_dq, "amax dQ"),)
    else:
        dk, dv = tensors[17:19]
        if (
            dk["data_type"] != q["data_type"]
            or dv["data_type"] != q["data_type"]
            or dk["dimensions"] != k["dimensions"]
            or dv["dimensions"] != v["dimensions"]
        ):
            raise ValueError("FP8 SDPA backward dK/dV metadata is invalid")
        scale_names = (
            "descale Q",
            "descale K",
            "descale V",
            "descale O",
            "descale dO",
            "descale S",
            "descale dP",
            "scale S",
            "scale dK",
            "scale dV",
            "scale dP",
        )
        scale_tensors = tensors[6:17]
        amax_tensors = tuple(
            zip(tensors[19:22], ("amax dK", "amax dV", "amax dP"))
        )
    for tensor, name in zip(scale_tensors, scale_names):
        _require_fp8_scale_tensor(tensor, f"FP8 SDPA backward {name}")
    for tensor, name in amax_tensors:
        _require_fp8_scale_tensor(tensor, f"FP8 SDPA backward {name}")

    attn_scale = _attention_runtime_f32(
        parameters, "attn_scale", _require_number(parameters, "attn_scale")
    )
    del attn_scale
    for name, value in (
        ("HQ", heads),
        ("SQ", sequence_q),
        ("SKV", sequence_kv),
        (
            "min_diag",
            _require_integer(
                parameters,
                "min_diag",
                minimum=-(2**31),
                maximum=2**31 - 1,
            ),
        ),
        (
            "max_diag",
            _require_integer(
                parameters,
                "max_diag",
                minimum=-(2**31),
                maximum=2**31 - 1,
            ),
        ),
    ):
        _attention_runtime_i32(parameters, name, value)

    pointer_type = TRITON_POINTER_TYPES[q["data_type"]]
    common_constants: dict[str, int | float | str | bool] = {
        **_attention_strides(q, "q", "bhmd"),
        **_attention_strides(k, "k", "bhnd"),
        **_attention_strides(v, "v", "bhnd"),
        **_attention_strides(output, "o", "bhmd"),
        **_attention_strides(doutput, "do", "bhmd"),
        **_attention_strides(stats, "s", "bhm"),
        "HEAD_DIM": head_dimension,
        "BLOCK_M": 128,
        "BLOCK_N": 64 if stage == "dq" else 128,
        "BLOCK_D": max(16, _next_power_of_two(head_dimension)),
        "BANDED": _attention_flag(parameters, "banded"),
        "FULL_BLOCKS": False,
        "CAUSAL_TOP_LEFT": _attention_flag(parameters, "causal_top_left"),
    }
    runtime_scalars = {
        "attn_scale": "fp32",
        "HQ": "i32",
        "SQ": "i32",
        "SKV": "i32",
        "min_diag": "i32",
        "max_diag": "i32",
    }

    if stage == "dq":
        constants = {
            **common_constants,
            **_attention_strides(tensors[14], "dq", "bhmd"),
            "q_per_k": heads // key_heads,
            "q_per_v": heads // value_heads,
        }
        signature: dict[str, str] = {
            "q_ptr": pointer_type,
            "k_ptr": pointer_type,
            "v_ptr": pointer_type,
            "o_ptr": pointer_type,
            "do_ptr": pointer_type,
            "stats_ptr": TRITON_POINTER_TYPES["float32"],
            "dq_ptr": pointer_type,
            "amax_dq_ptr": TRITON_POINTER_TYPES["float32"],
            "descale_q_ptr": TRITON_POINTER_TYPES["float32"],
            "descale_k_ptr": TRITON_POINTER_TYPES["float32"],
            "descale_v_ptr": TRITON_POINTER_TYPES["float32"],
            "descale_o_ptr": TRITON_POINTER_TYPES["float32"],
            "descale_do_ptr": TRITON_POINTER_TYPES["float32"],
            "descale_dp_ptr": TRITON_POINTER_TYPES["float32"],
            "scale_dq_ptr": TRITON_POINTER_TYPES["float32"],
            "scale_dp_ptr": TRITON_POINTER_TYPES["float32"],
            **runtime_scalars,
        }
        layout: list[tuple[str, str | int | None]] = [
            *[("tensor_alias", index) for index in (0, 1, 2, 3, 4, 5)],
            ("tensor_alias", 14),
            ("tensor_alias", 15),
            *[("tensor_alias", index) for index in range(6, 14)],
            ("scalar_f32", "attn_scale"),
            ("scalar_i32", "HQ"),
            ("scalar_i32", "SQ"),
            ("scalar_i32", "SKV"),
            ("scalar_i32", "min_diag"),
            ("scalar_i32", "max_diag"),
        ]
        return (
            "_sdpa_fp8_bwd_dq_kernel",
            signature,
            constants,
            (
                _ceil_div(sequence_q, int(constants["BLOCK_M"])),
                batch * heads,
                1,
            ),
            layout,
        )

    constants = {
        **common_constants,
        **_attention_strides(tensors[17], "dk", "bhnd"),
        **_attention_strides(tensors[18], "dv", "bhnd"),
        "HKV": key_heads,
        "Q_PER": heads // key_heads,
    }
    signature = {
        "q_ptr": pointer_type,
        "k_ptr": pointer_type,
        "v_ptr": pointer_type,
        "o_ptr": pointer_type,
        "do_ptr": pointer_type,
        "stats_ptr": TRITON_POINTER_TYPES["float32"],
        "dk_ptr": pointer_type,
        "dv_ptr": pointer_type,
        "amax_dk_ptr": TRITON_POINTER_TYPES["float32"],
        "amax_dv_ptr": TRITON_POINTER_TYPES["float32"],
        "amax_dp_ptr": TRITON_POINTER_TYPES["float32"],
        "descale_q_ptr": TRITON_POINTER_TYPES["float32"],
        "descale_k_ptr": TRITON_POINTER_TYPES["float32"],
        "descale_v_ptr": TRITON_POINTER_TYPES["float32"],
        "descale_o_ptr": TRITON_POINTER_TYPES["float32"],
        "descale_do_ptr": TRITON_POINTER_TYPES["float32"],
        "descale_s_ptr": TRITON_POINTER_TYPES["float32"],
        "descale_dp_ptr": TRITON_POINTER_TYPES["float32"],
        "scale_s_ptr": TRITON_POINTER_TYPES["float32"],
        "scale_dk_ptr": TRITON_POINTER_TYPES["float32"],
        "scale_dv_ptr": TRITON_POINTER_TYPES["float32"],
        "scale_dp_ptr": TRITON_POINTER_TYPES["float32"],
        "attn_scale": "fp32",
        "SQ": "i32",
        "SKV": "i32",
        "min_diag": "i32",
        "max_diag": "i32",
    }
    layout = [
        *[("tensor_alias", index) for index in (0, 1, 2, 3, 4, 5)],
        ("tensor_alias", 17),
        ("tensor_alias", 18),
        ("tensor_alias", 19),
        ("tensor_alias", 20),
        ("tensor_alias", 21),
        *[("tensor_alias", index) for index in range(6, 17)],
        ("scalar_f32", "attn_scale"),
        ("scalar_i32", "SQ"),
        ("scalar_i32", "SKV"),
        ("scalar_i32", "min_diag"),
        ("scalar_i32", "max_diag"),
    ]
    return (
        "_sdpa_fp8_bwd_dkdv_kernel",
        signature,
        constants,
        (
            _ceil_div(sequence_kv, int(constants["BLOCK_N"])),
            batch * key_heads,
            1,
        ),
        layout,
    )


def _sdpa_backward_kernel_configuration(
    parameters: dict[str, Any], tensors: list[dict[str, Any]]
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | float | str | bool],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    stage = parameters.get("_sdpa_bwd_stage")
    if stage == "zero_dbias":
        if len(tensors) != 1:
            raise ValueError("SDPA dBias zero stage tensor count is invalid")
        elements = math.prod(tensors[0]["dimensions"])
        _attention_runtime_i32(parameters, "dbias_elements", elements)
        return (
            "_zero_contiguous_kernel",
            {
                "ptr": TRITON_POINTER_TYPES[tensors[0]["data_type"]],
                "n_elements": "i32",
            },
            {"BLOCK": 1024},
            (_ceil_div(elements, 1024), 1, 1),
            [
                ("tensor_alias", 0),
                ("scalar_i32", "dbias_elements"),
            ],
        )

    if stage == "dv":
        has_bias = _attention_flag(parameters, "has_bias")
        if len(tensors) != 5 + int(has_bias):
            raise ValueError("SDPA backward dV stage tensor count is invalid")
        q, k = tensors[:2]
        if (
            len(q["dimensions"]) != 4
            or len(k["dimensions"]) != 4
            or q["data_type"] not in FLOAT_DATA_TYPES
            or k["data_type"] != q["data_type"]
        ):
            raise ValueError("SDPA backward dV Q/K metadata is invalid")
        batch, heads, sequence_q, head_dimension = q["dimensions"]
        key_heads = k["dimensions"][1]
        sequence_kv = k["dimensions"][2]
        value_heads = _require_integer(parameters, "value_heads")
        value_dimension = _require_integer(parameters, "value_dimension")
        if (
            k["dimensions"][0] != batch
            or k["dimensions"][3] != head_dimension
            or key_heads != value_heads
            or heads % value_heads != 0
        ):
            raise ValueError("SDPA backward dV Q/K shapes are inconsistent")
        bias_index = 2 if has_bias else None
        offset = 2 + int(has_bias)
        doutput_index = offset
        stats_index = offset + 1
        dv_index = offset + 2
        doutput = tensors[doutput_index]
        stats = tensors[stats_index]
        dv = tensors[dv_index]
        if dv["dimensions"] != [
            batch,
            value_heads,
            sequence_kv,
            value_dimension,
        ]:
            raise ValueError("SDPA backward dV output metadata is invalid")
        attn_scale = _require_number(parameters, "attn_scale")
        min_diag = _require_integer(
            parameters, "min_diag", minimum=-(2**31), maximum=2**31 - 1
        )
        max_diag = _require_integer(
            parameters, "max_diag", minimum=-(2**31), maximum=2**31 - 1
        )
        for name, value in (
            ("attn_scale", attn_scale),
            ("SQ", sequence_q),
            ("SKV", sequence_kv),
            ("min_diag", min_diag),
            ("max_diag", max_diag),
        ):
            if isinstance(value, float):
                _attention_runtime_f32(parameters, name, value)
            else:
                _attention_runtime_i32(parameters, name, value)
        if has_bias:
            bias_strides = _attention_broadcast_strides(
                tensors[bias_index], "bias_"
            )
        else:
            bias_strides = {
                "stride_bias_b": 0,
                "stride_bias_h": 0,
                "stride_bias_m": 0,
                "stride_bias_n": 0,
            }
        pointer_type = TRITON_POINTER_TYPES[q["data_type"]]
        constants: dict[str, int | float | str | bool] = {
            "HKV": value_heads,
            **_attention_strides(q, "q", "bhmd"),
            **_attention_strides(k, "k", "bhnd"),
            **bias_strides,
            **_attention_strides(doutput, "do", "bhmd"),
            **_attention_strides(stats, "s", "bhm"),
            **_attention_strides(dv, "dv", "bhnd"),
            "HEAD_DIM": head_dimension,
            "V_DIM": value_dimension,
            "Q_PER": heads // value_heads,
            "BLOCK_M": 64,
            "BLOCK_N": 32,
            "BLOCK_D_FULL": max(16, _next_power_of_two(head_dimension)),
            "BLOCK_DV_OUT": 128,
            "FULL_ATTENTION": False,
            "HAS_BIAS": has_bias,
            "BANDED": _attention_flag(parameters, "banded"),
            "CAUSAL_TOP_LEFT": _attention_flag(parameters, "causal_top_left"),
        }
        signature = {
            "q_ptr": pointer_type,
            "k_ptr": pointer_type,
            "bias_ptr": pointer_type,
            "do_ptr": pointer_type,
            "stats_ptr": TRITON_POINTER_TYPES["float32"],
            "dv_ptr": pointer_type,
            "attn_scale": "fp32",
            "SQ": "i32",
            "SKV": "i32",
            "min_diag": "i32",
            "max_diag": "i32",
        }
        layout = [
            ("tensor_alias", 0),
            ("tensor_alias", 1),
            ("tensor_alias", bias_index if bias_index is not None else 0),
            ("tensor_alias", doutput_index),
            ("tensor_alias", stats_index),
            ("tensor_alias", dv_index),
            ("scalar_f32", "attn_scale"),
            ("scalar_i32", "SQ"),
            ("scalar_i32", "SKV"),
            ("scalar_i32", "min_diag"),
            ("scalar_i32", "max_diag"),
        ]
        return (
            "_sdpa_bwd_dv_kernel",
            signature,
            constants,
            (
                _ceil_div(sequence_kv, int(constants["BLOCK_N"])),
                _ceil_div(value_dimension, int(constants["BLOCK_DV_OUT"])),
                batch * value_heads,
            ),
            layout,
        )

    has_bias = _attention_flag(parameters, "has_bias")
    has_dbias = _attention_flag(parameters, "has_dbias")
    (
        batch,
        heads,
        key_heads,
        value_heads,
        sequence_q,
        sequence_kv,
        head_dimension,
        value_dimension,
    ) = _validate_attention_base(parameters, tensors)
    if key_heads != value_heads:
        raise ValueError("SDPA backward requires matching K/V head counts")
    q, k, v = tensors[:3]
    pointer_type = TRITON_POINTER_TYPES[q["data_type"]]
    attn_scale = _require_number(parameters, "attn_scale")
    min_diag = _require_integer(
        parameters, "min_diag", minimum=-(2**31), maximum=2**31 - 1
    )
    max_diag = _require_integer(
        parameters, "max_diag", minimum=-(2**31), maximum=2**31 - 1
    )
    for name, value in (
        ("attn_scale", attn_scale),
        ("SQ", sequence_q),
        ("SKV", sequence_kv),
        ("min_diag", min_diag),
        ("max_diag", max_diag),
    ):
        if isinstance(value, float):
            _attention_runtime_f32(parameters, name, value)
        else:
            _attention_runtime_i32(parameters, name, value)

    has_banded = _attention_flag(parameters, "banded")
    causal_top_left = _attention_flag(parameters, "causal_top_left")
    block_d_full = max(16, _next_power_of_two(head_dimension))
    block_dv = max(16, _next_power_of_two(value_dimension))

    def bias_constants(bias: dict[str, Any] | None) -> dict[str, int]:
        if bias is None:
            return {
                "stride_bias_b": 0,
                "stride_bias_h": 0,
                "stride_bias_m": 0,
                "stride_bias_n": 0,
            }
        return _attention_broadcast_strides(bias, "bias_")

    runtime_scalars = {
        "attn_scale": "fp32",
        "SQ": "i32",
        "SKV": "i32",
        "min_diag": "i32",
        "max_diag": "i32",
    }
    scalar_layout: list[tuple[str, str | int | None]] = [
        ("scalar_f32", "attn_scale"),
        ("scalar_i32", "SQ"),
        ("scalar_i32", "SKV"),
        ("scalar_i32", "min_diag"),
        ("scalar_i32", "max_diag"),
    ]

    if stage == "dq":
        expected_count = 8 + int(has_bias) + int(has_dbias)
        if len(tensors) != expected_count:
            raise ValueError("SDPA backward dQ stage tensor count is invalid")
        bias_index = 3 if has_bias else None
        offset = 3 + int(has_bias)
        output_index = offset
        doutput_index = offset + 1
        stats_index = offset + 2
        delta_index = offset + 3
        dq_index = offset + 4
        dbias_index = offset + 5 if has_dbias else None
        output = tensors[output_index]
        doutput = tensors[doutput_index]
        stats = tensors[stats_index]
        delta = tensors[delta_index]
        dq = tensors[dq_index]
        dbias = tensors[dbias_index] if dbias_index is not None else None
        constants: dict[str, int | float | str | bool] = {
            "HQ": heads,
            "q_per_k": heads // key_heads,
            "q_per_v": heads // value_heads,
            **_attention_strides(q, "q", "bhmd"),
            **_attention_strides(k, "k", "bhnd"),
            **_attention_strides(v, "v", "bhnd"),
            **bias_constants(tensors[bias_index] if has_bias else None),
            **_attention_strides(output, "o", "bhmd"),
            **_attention_strides(doutput, "do", "bhmd"),
            **_attention_strides(stats, "s", "bhm"),
            **_attention_strides(delta, "delta_", "bhm"),
            **_attention_strides(dq, "dq", "bhmd"),
            "HEAD_DIM": head_dimension,
            "V_DIM": value_dimension,
            "DBIAS_BATCHES": dbias["dimensions"][0] if dbias else 1,
            "DBIAS_HEADS": dbias["dimensions"][1] if dbias else 1,
            "BLOCK_M": 64,
            "BLOCK_N": 32,
            "BLOCK_D_FULL": block_d_full,
            "BLOCK_D_OUT": 128,
            "BLOCK_DV": block_dv,
            "FULL_ATTENTION": False,
            "HAS_BIAS": has_bias,
            "HAS_DBIAS": has_dbias,
            "DBIAS_REDUCE": _attention_flag(parameters, "dbias_reduce"),
            "BANDED": has_banded,
            "CAUSAL_TOP_LEFT": causal_top_left,
        }
        if dbias is None:
            constants.update(
                {
                    "stride_dbias_b": 0,
                    "stride_dbias_h": 0,
                    "stride_dbias_m": 0,
                    "stride_dbias_n": 0,
                }
            )
        else:
            constants.update(_attention_strides(dbias, "dbias_", "bhmn"))
        signature = {
            "q_ptr": pointer_type,
            "k_ptr": pointer_type,
            "v_ptr": pointer_type,
            "bias_ptr": pointer_type,
            "o_ptr": pointer_type,
            "do_ptr": pointer_type,
            "stats_ptr": TRITON_POINTER_TYPES["float32"],
            "delta_ptr": TRITON_POINTER_TYPES["float32"],
            "dq_ptr": pointer_type,
            "dbias_ptr": pointer_type,
            **runtime_scalars,
        }
        layout = [
            ("tensor_alias", 0),
            ("tensor_alias", 1),
            ("tensor_alias", 2),
            ("tensor_alias", bias_index if bias_index is not None else 0),
            ("tensor_alias", output_index),
            ("tensor_alias", doutput_index),
            ("tensor_alias", stats_index),
            ("tensor_alias", delta_index),
            ("tensor_alias", dq_index),
            ("tensor_alias", dbias_index if dbias_index is not None else 0),
            *scalar_layout,
        ]
        return (
            "_sdpa_bwd_dq_dbias_kernel",
            signature,
            constants,
            (
                _ceil_div(sequence_q, int(constants["BLOCK_M"])),
                _ceil_div(head_dimension, int(constants["BLOCK_D_OUT"])),
                batch * heads,
            ),
            layout,
        )

    bias_index = 3 if has_bias else None
    offset = 3 + int(has_bias)
    if stage == "dkdv":
        if len(tensors) != 8 + int(has_bias):
            raise ValueError(
                "SDPA backward dK/dV stage tensor count is invalid"
            )
        doutput_index = offset
        stats_index = offset + 1
        delta_index = offset + 2
        dk_index = offset + 3
        dv_index = offset + 4
        doutput = tensors[doutput_index]
        stats = tensors[stats_index]
        delta = tensors[delta_index]
        dk = tensors[dk_index]
        dv = tensors[dv_index]
        constants = {
            "HKV": key_heads,
            **_attention_strides(q, "q", "bhmd"),
            **_attention_strides(k, "k", "bhnd"),
            **_attention_strides(v, "v", "bhnd"),
            **bias_constants(tensors[bias_index] if has_bias else None),
            **_attention_strides(doutput, "do", "bhmd"),
            **_attention_strides(stats, "s", "bhm"),
            **_attention_strides(delta, "delta_", "bhm"),
            **_attention_strides(dk, "dk", "bhnd"),
            **_attention_strides(dv, "dv", "bhnd"),
            "HEAD_DIM": head_dimension,
            "Q_PER": heads // key_heads,
            "BLOCK_M": 64,
            "BLOCK_N": 32,
            "BLOCK_D_FULL": block_d_full,
            "BLOCK_D_OUT": 128,
            "FULL_ATTENTION": False,
            "HAS_BIAS": has_bias,
            "BANDED": has_banded,
            "CAUSAL_TOP_LEFT": causal_top_left,
        }
        signature = {
            "q_ptr": pointer_type,
            "k_ptr": pointer_type,
            "v_ptr": pointer_type,
            "bias_ptr": pointer_type,
            "do_ptr": pointer_type,
            "stats_ptr": TRITON_POINTER_TYPES["float32"],
            "delta_ptr": TRITON_POINTER_TYPES["float32"],
            "dk_ptr": pointer_type,
            "dv_ptr": pointer_type,
            **runtime_scalars,
        }
        layout = [
            ("tensor_alias", 0),
            ("tensor_alias", 1),
            ("tensor_alias", 2),
            ("tensor_alias", bias_index if bias_index is not None else 0),
            ("tensor_alias", doutput_index),
            ("tensor_alias", stats_index),
            ("tensor_alias", delta_index),
            ("tensor_alias", dk_index),
            ("tensor_alias", dv_index),
            *scalar_layout,
        ]
        return (
            "_sdpa_bwd_dkdv_kernel",
            signature,
            constants,
            (
                _ceil_div(sequence_kv, int(constants["BLOCK_N"])),
                _ceil_div(head_dimension, int(constants["BLOCK_D_OUT"])),
                batch * key_heads,
            ),
            layout,
        )

    if stage == "dk":
        if len(tensors) != 7 + int(has_bias):
            raise ValueError("SDPA backward dK stage tensor count is invalid")
        doutput_index = offset
        stats_index = offset + 1
        delta_index = offset + 2
        dk_index = offset + 3
        doutput = tensors[doutput_index]
        stats = tensors[stats_index]
        delta = tensors[delta_index]
        dk = tensors[dk_index]
        constants = {
            "HKV": key_heads,
            **_attention_strides(q, "q", "bhmd"),
            **_attention_strides(k, "k", "bhnd"),
            **_attention_strides(v, "v", "bhnd"),
            **bias_constants(tensors[bias_index] if has_bias else None),
            **_attention_strides(doutput, "do", "bhmd"),
            **_attention_strides(stats, "s", "bhm"),
            **_attention_strides(delta, "delta_", "bhm"),
            **_attention_strides(dk, "dk", "bhnd"),
            "HEAD_DIM": head_dimension,
            "V_DIM": value_dimension,
            "Q_PER": heads // key_heads,
            "BLOCK_M": 64,
            "BLOCK_N": 32,
            "BLOCK_D_FULL": block_d_full,
            "BLOCK_D_OUT": 128,
            "BLOCK_DV": block_dv,
            "FULL_ATTENTION": False,
            "HAS_BIAS": has_bias,
            "BANDED": has_banded,
            "CAUSAL_TOP_LEFT": causal_top_left,
        }
        signature = {
            "q_ptr": pointer_type,
            "k_ptr": pointer_type,
            "v_ptr": pointer_type,
            "bias_ptr": pointer_type,
            "do_ptr": pointer_type,
            "stats_ptr": TRITON_POINTER_TYPES["float32"],
            "delta_ptr": TRITON_POINTER_TYPES["float32"],
            "dk_ptr": pointer_type,
            **runtime_scalars,
        }
        layout = [
            ("tensor_alias", 0),
            ("tensor_alias", 1),
            ("tensor_alias", 2),
            ("tensor_alias", bias_index if bias_index is not None else 0),
            ("tensor_alias", doutput_index),
            ("tensor_alias", stats_index),
            ("tensor_alias", delta_index),
            ("tensor_alias", dk_index),
            *scalar_layout,
        ]
        return (
            "_sdpa_bwd_dk_kernel",
            signature,
            constants,
            (
                _ceil_div(sequence_kv, int(constants["BLOCK_N"])),
                _ceil_div(head_dimension, int(constants["BLOCK_D_OUT"])),
                batch * key_heads,
            ),
            layout,
        )

    raise ValueError("SDPA backward pipeline stage is invalid")


def _kernel_configuration(
    operation: str,
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
    architecture: int,
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | float | str | bool],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    tensor_data_types = [tensor["data_type"] for tensor in tensors]
    if operation == "sdpa":
        return _sdpa_forward_kernel_configuration(parameters, tensors)
    if operation == "sdpa_backward":
        return _sdpa_backward_kernel_configuration(parameters, tensors)
    if operation == "sdpa_fp8":
        return _sdpa_fp8_forward_kernel_configuration(parameters, tensors)
    if operation == "sdpa_fp8_backward":
        return _sdpa_fp8_backward_kernel_configuration(parameters, tensors)
    if operation == "layernorm":
        return _normalization_forward_kernel_configuration(
            parameters, tensors, rmsnorm=False
        )
    if operation == "rmsnorm":
        return _normalization_forward_kernel_configuration(
            parameters, tensors, rmsnorm=True
        )
    if operation == "batchnorm":
        return _batchnorm_kernel_configuration(parameters, tensors)
    if operation == "batchnorm_inference":
        return _batchnorm_inference_kernel_configuration(parameters, tensors)
    if operation in {"reshape", "transpose", "slice"}:
        return _layout_kernel_configuration(operation, parameters, tensors)
    if operation == "matmul" and "_fprop_p5_matmul_stage" in parameters:
        return _matmul_p5_pipeline_kernel_configuration(parameters, tensors)
    if operation == "matmul":
        return _matmul_kernel_configuration(parameters, tensors, architecture)
    if (
        operation == "convolution_fprop"
        and parameters.get("_fprop_pipeline_stage") == "im2col"
    ):
        if parameters.get("_fprop_pipeline_algorithm") == "general":
            return _convolution_general_im2col_kernel_configuration(
                parameters, tensors
            )
        return _convolution_im2col_kernel_configuration(parameters, tensors)
    if operation == "convolution_fprop":
        return _convolution_kernel_configuration(parameters, tensors)
    if (
        operation == "convolution_wgrad"
        and "_wgrad_pipeline_stage" in parameters
    ):
        if parameters.get("_wgrad_pipeline_algorithm") == "p5":
            return _convolution_wgrad_p5_pipeline_kernel_configuration(
                parameters, tensors
            )
        if parameters.get("_wgrad_pipeline_algorithm") == "1x1":
            return _convolution_wgrad_1x1_pipeline_kernel_configuration(
                parameters, tensors
            )
        if parameters.get("_wgrad_pipeline_algorithm") in {
            "1x1_tma",
            "stride2_im2col",
        }:
            if parameters.get("_wgrad_pipeline_stage") == "im2col":
                return _convolution_im2col_kernel_configuration(
                    parameters, tensors
                )
            return _convolution_wgrad_batched_pipeline_kernel_configuration(
                parameters, tensors
            )
        if parameters.get("_wgrad_pipeline_algorithm") == "stride2_row4":
            return _convolution_wgrad_stride2_pipeline_kernel_configuration(
                parameters, tensors
            )
        return _convolution_wgrad_pipeline_kernel_configuration(
            parameters, tensors
        )
    if (
        operation == "convolution_dgrad"
        and "_dgrad_3d_pipeline_stage" in parameters
    ):
        return _convolution_dgrad_3d_pipeline_kernel_configuration(
            parameters, tensors
        )
    if (
        operation == "convolution_dgrad"
        and "_dgrad_pipeline_stage" in parameters
    ):
        return _convolution_dgrad_stride2_pipeline_kernel_configuration(
            parameters, tensors
        )
    if operation in {"convolution_dgrad", "convolution_wgrad"}:
        return _convolution_backward_kernel_configuration(
            operation, parameters, tensors
        )
    if operation == "relu" or operation in UNARY_POINTWISE_OPERATIONS:
        if len(tensor_data_types) != 2:
            raise ValueError("unary pointwise tensor count is invalid")
        if operation == "logical_not":
            if tensor_data_types != ["boolean", "boolean"]:
                raise ValueError(
                    "logical_not input/output data types must be boolean"
                )
        elif (
            len(set(tensor_data_types)) != 1
            or tensor_data_types[0] not in FLOAT_DATA_TYPES
        ):
            raise ValueError(
                "numeric unary pointwise tensor data types must match and "
                "be floating"
            )
        input_pointer_type = TRITON_POINTER_TYPES.get(tensor_data_types[0])
        output_pointer_type = TRITON_POINTER_TYPES.get(tensor_data_types[1])
        if input_pointer_type is None or output_pointer_type is None:
            raise ValueError(
                "unsupported unary pointwise data type: "
                f"{tensor_data_types[0]!r}"
            )
        elements = _require_integer(parameters, "n_elements")
        expected_elements = math.prod(tensors[0]["dimensions"])
        if elements != expected_elements:
            raise ValueError(
                "parameters.n_elements is inconsistent with unary "
                "pointwise input"
            )
        has_upper_clip_value = parameters.get("has_upper_clip", 0)
        if (
            isinstance(has_upper_clip_value, bool)
            or not isinstance(has_upper_clip_value, int)
            or has_upper_clip_value < 0
            or has_upper_clip_value > 1
        ):
            raise ValueError(
                "parameters.has_upper_clip must be either zero or one"
            )
        has_upper_clip = has_upper_clip_value
        negative_slope = _require_number(
            parameters, "negative_slope", default=0.0
        )
        lower_clip = _require_number(parameters, "lower_clip", default=0.0)
        upper_clip = _require_number(parameters, "upper_clip", default=0.0)
        swish_beta = _require_number(parameters, "swish_beta", default=1.0)
        elu_alpha = _require_number(parameters, "elu_alpha", default=1.0)
        softplus_beta = _require_number(
            parameters, "softplus_beta", default=1.0
        )
        if softplus_beta <= 0.0:
            raise ValueError("parameters.softplus_beta must be positive")
        if has_upper_clip and upper_clip < lower_clip:
            raise ValueError(
                "parameters.upper_clip must not be less than lower_clip"
            )
        block = 256
        constants: dict[str, int | float | str | bool] = {
            "OPERATION": UNARY_POINTWISE_MODES[operation],
            "negative_slope": negative_slope,
            "lower_clip": lower_clip,
            "upper_clip": upper_clip,
            "HAS_UPPER_CLIP": bool(has_upper_clip),
            "SWISH_BETA": swish_beta,
            "ELU_ALPHA": elu_alpha,
            "SOFTPLUS_BETA": softplus_beta,
            "TILES_PER_PROGRAM": 1,
            "BLOCK_SIZE": block,
        }
        tensor_constants = _unary_pointwise_tensor_constants(tensors)
        function_name = "unary_pointwise_contiguous_kernel"
        if bool(tensor_constants["STRIDED"]):
            constants.update(tensor_constants)
            function_name = "unary_pointwise_strided_kernel"
        return (
            function_name,
            {
                "in_ptr": input_pointer_type,
                "out_ptr": output_pointer_type,
                "n_elements": "i32",
            },
            constants,
            ((elements + block - 1) // block, 1, 1),
            [
                ("tensor", None),
                ("tensor", None),
                ("scalar_i32", "n_elements"),
            ],
        )
    if operation == "add" or operation in BINARY_POINTWISE_OPERATIONS:
        if len(tensor_data_types) != 3:
            raise ValueError("binary pointwise tensor count is invalid")
        if operation in COMPARISON_POINTWISE_OPERATIONS:
            if (
                tensor_data_types[0] != tensor_data_types[1]
                or tensor_data_types[0] not in FLOAT_DATA_TYPES
                or tensor_data_types[2] != "boolean"
            ):
                raise ValueError(
                    "comparison pointwise requires matching floating inputs "
                    "and boolean output"
                )
        elif operation in LOGICAL_BINARY_POINTWISE_OPERATIONS:
            if tensor_data_types != ["boolean", "boolean", "boolean"]:
                raise ValueError(
                    "logical pointwise input/output data types must be boolean"
                )
        elif (
            len(set(tensor_data_types)) != 1
            or tensor_data_types[0] not in FLOAT_DATA_TYPES
        ):
            raise ValueError(
                "numeric binary pointwise tensor data types must match and "
                "be floating"
            )
        pointer_types = [
            TRITON_POINTER_TYPES.get(data_type)
            for data_type in tensor_data_types
        ]
        if any(pointer_type is None for pointer_type in pointer_types):
            raise ValueError(
                "unsupported binary pointwise data type: "
                f"{tensor_data_types[0]!r}"
            )
        elements = _require_integer(parameters, "n_elements")
        expected_elements = math.prod(tensors[2]["dimensions"])
        if elements != expected_elements:
            raise ValueError(
                "parameters.n_elements is inconsistent with binary "
                "pointwise output"
            )
        pointwise_mode = _require_integer(
            parameters, "pointwise_mode", minimum=1, maximum=40
        )
        if BINARY_POINTWISE_MODES.get(operation) != pointwise_mode:
            raise ValueError(
                "parameters.pointwise_mode is inconsistent with "
                "binary operation"
            )
        block = 256
        strided_constants = _binary_pointwise_tensor_constants(tensors)
        alpha = _require_number(parameters, "alpha", default=1.0)
        if operation not in ("add", "sub") and alpha != 1.0:
            raise ValueError(
                "pointwise alpha is only supported by add and sub"
            )

        constants: dict[str, int | float] = {
            "OP_KIND": pointwise_mode,
            "ALPHA": alpha,
            "BLOCK_SIZE": block,
        }
        function_name = "binary_contiguous_kernel"
        if not _can_use_dense_binary_kernel(tensors):
            constants.update(strided_constants)
            function_name = "binary_strided_kernel"
        return (
            function_name,
            {
                "x_ptr": pointer_types[0],
                "y_ptr": pointer_types[1],
                "out_ptr": pointer_types[2],
                "n_elements": "i32",
            },
            constants,
            ((elements + block - 1) // block, 1, 1),
            [
                ("tensor", None),
                ("tensor", None),
                ("tensor", None),
                ("scalar_i32", "n_elements"),
            ],
        )
    if operation in TERNARY_POINTWISE_OPERATIONS:
        if len(tensor_data_types) != 4:
            raise ValueError("ternary pointwise tensor count is invalid")
        if (
            tensor_data_types[0] != tensor_data_types[1]
            or tensor_data_types[0] != tensor_data_types[3]
            or tensor_data_types[0] not in FLOAT_DATA_TYPES
            or tensor_data_types[2] != "boolean"
        ):
            raise ValueError(
                "binary_select requires matching floating A/B/output "
                "and a boolean T predicate"
            )
        pointer_types = [
            TRITON_POINTER_TYPES.get(data_type)
            for data_type in tensor_data_types
        ]
        if any(pointer_type is None for pointer_type in pointer_types):
            raise ValueError("unsupported binary_select data type")
        elements = _require_integer(parameters, "n_elements")
        if elements != math.prod(tensors[3]["dimensions"]):
            raise ValueError(
                "parameters.n_elements is inconsistent with binary_select "
                "output"
            )
        block = 256
        constants: dict[str, int] = {"BLOCK_SIZE": block}
        function_name = "binary_select_tensor_kernel"
        signature = {
            "input0_ptr": pointer_types[0],
            "input1_ptr": pointer_types[1],
            "mask_ptr": pointer_types[2],
            "out_ptr": pointer_types[3],
            "n_elements": "i32",
        }
        if not _can_use_dense_ternary_kernel(tensors):
            constants.update(_ternary_pointwise_tensor_constants(tensors))
            function_name = "binary_select_strided_kernel"
            signature = {
                "x_ptr": pointer_types[0],
                "y_ptr": pointer_types[1],
                "t_ptr": pointer_types[2],
                "out_ptr": pointer_types[3],
                "n_elements": "i32",
            }
        return (
            function_name,
            signature,
            constants,
            ((elements + block - 1) // block, 1, 1),
            [
                ("tensor", None),
                ("tensor", None),
                ("tensor", None),
                ("tensor", None),
                ("scalar_i32", "n_elements"),
            ],
        )

    if operation in REDUCTION_OPERATIONS:
        if len(tensor_data_types) != 2 or len(set(tensor_data_types)) != 1:
            raise ValueError("Reduction tensor data types must match")
        pointer_type = TRITON_POINTER_TYPES.get(tensor_data_types[0])
        if pointer_type is None:
            raise ValueError(
                "unsupported Reduction data type: " f"{tensor_data_types[0]!r}"
            )
        outer = _require_integer(parameters, "outer")
        extent = _require_integer(parameters, "reduction", maximum=65536)
        inner = _require_integer(parameters, "inner")
        output_elements = _require_integer(parameters, "output_elements")
        input_rank = len(tensors[0]["dimensions"])
        if input_rank == 0:
            raise ValueError("Reduction input must have positive rank")
        axis = _require_integer(
            parameters,
            "axis",
            minimum=0,
            maximum=input_rank - 1,
        )
        keep_dimensions_value = _require_integer(
            parameters, "keep_dimensions", minimum=0, maximum=1
        )
        keep_dimensions = keep_dimensions_value == 1
        if math.prod(tensors[0]["dimensions"]) != outer * extent * inner:
            raise ValueError(
                "Reduction parameters are inconsistent with input shape"
            )
        if (
            math.prod(tensors[1]["dimensions"]) != output_elements
            or output_elements != outer * inner
        ):
            raise ValueError(
                "Reduction parameters are inconsistent with output shape"
            )
        strided_constants = _reduction_tensor_constants(
            tensors, axis, keep_dimensions
        )

        block_n = _next_power_of_two(extent)
        constants: dict[str, int | str] = {
            "N": extent,
            "OP": REDUCTION_OPERATIONS[operation],
            "BLOCK_M": 1,
            "BLOCK_N": block_n,
        }
        signature = {
            "x_ptr": pointer_type,
            "out_ptr": pointer_type,
            "M": "i32",
        }
        tensors_are_contiguous = all(
            _is_row_major_contiguous(tensor) for tensor in tensors
        )
        if tensors_are_contiguous and inner == 1:
            constants.update({"stride_xm": extent, "stride_xn": 1})
            return (
                "reduction_2d_kernel",
                signature,
                constants,
                (outer, 1, 1),
                [
                    ("tensor", None),
                    ("tensor", None),
                    ("scalar_i32", "outer"),
                ],
            )

        if tensors_are_contiguous:
            constants.update(
                {
                    "I": inner,
                    "stride_xo": extent * inner,
                    "stride_xr": inner,
                    "stride_xi": 1,
                }
            )
            return (
                "reduction_3d_kernel",
                signature,
                constants,
                (output_elements, 1, 1),
                [
                    ("tensor", None),
                    ("tensor", None),
                    ("scalar_i32", "output_elements"),
                ],
            )

        block_m = min(16, max(1, 65536 // block_n))
        constants.update(strided_constants)
        constants["BLOCK_M"] = block_m
        return (
            "reduction_strided_kernel",
            signature,
            constants,
            ((output_elements + block_m - 1) // block_m, 1, 1),
            [
                ("tensor", None),
                ("tensor", None),
                ("scalar_i32", "output_elements"),
            ],
        )
    # Compatibility parser for schema-v2 requests emitted by the original
    # Conv2D-only Core. New requests use the N-D branch above.
    if operation == "conv2d_fprop":
        if len(tensor_data_types) != 3 or len(set(tensor_data_types)) != 1:
            raise ValueError("Conv2D FProp tensor data types must match")
        pointer_type = TRITON_POINTER_TYPES.get(tensor_data_types[0])
        if pointer_type is None:
            raise ValueError(
                "unsupported Conv2D FProp data type: "
                f"{tensor_data_types[0]!r}"
            )
        names = ("n", "c", "h", "w", "k", "r", "s", "oh", "ow")
        dimensions = {
            name: _require_integer(parameters, name) for name in names
        }
        pad_top = _require_integer(parameters, "pad_top", minimum=0)
        pad_bottom = _require_integer(parameters, "pad_bottom", minimum=0)
        pad_left = _require_integer(parameters, "pad_left", minimum=0)
        pad_right = _require_integer(parameters, "pad_right", minimum=0)
        stride_h = _require_integer(parameters, "stride_h")
        stride_w = _require_integer(parameters, "stride_w")
        dilation_h = _require_integer(parameters, "dilation_h")
        dilation_w = _require_integer(parameters, "dilation_w")
        groups = _require_integer(parameters, "groups")
        outputs = _require_integer(parameters, "n_outputs")
        if any(len(tensor["dimensions"]) != 4 for tensor in tensors):
            raise ValueError("Conv2D FProp requires rank-4 tensors")
        if any(
            not _has_non_overlapping_strides(
                tensor["dimensions"], tensor["strides"]
            )
            for tensor in tensors
        ):
            raise ValueError(
                "Conv2D FProp tensors must have non-overlapping strides"
            )
        if dimensions["c"] % groups != 0 or dimensions["k"] % groups != 0:
            raise ValueError("Conv2D FProp channels must divide groups")
        channels_per_group = dimensions["c"] // groups
        outputs_per_group = dimensions["k"] // groups
        expected_input = [
            dimensions["n"],
            dimensions["c"],
            dimensions["h"],
            dimensions["w"],
        ]
        expected_filter = [
            dimensions["k"],
            channels_per_group,
            dimensions["r"],
            dimensions["s"],
        ]
        expected_output = [
            dimensions["n"],
            dimensions["k"],
            dimensions["oh"],
            dimensions["ow"],
        ]
        if tensors[0]["dimensions"] != expected_input:
            raise ValueError("Conv2D FProp input metadata is inconsistent")
        if tensors[1]["dimensions"] != expected_filter:
            raise ValueError("Conv2D FProp filter metadata is inconsistent")
        if tensors[2]["dimensions"] != expected_output:
            raise ValueError("Conv2D FProp output metadata is inconsistent")
        expected_oh = (
            dimensions["h"]
            + pad_top
            + pad_bottom
            - dilation_h * (dimensions["r"] - 1)
            - 1
        ) // stride_h + 1
        expected_ow = (
            dimensions["w"]
            + pad_left
            + pad_right
            - dilation_w * (dimensions["s"] - 1)
            - 1
        ) // stride_w + 1
        if expected_oh != dimensions["oh"] or expected_ow != dimensions["ow"]:
            raise ValueError("Conv2D FProp output dimensions are inconsistent")
        if outputs != (
            dimensions["n"]
            * dimensions["k"]
            * dimensions["oh"]
            * dimensions["ow"]
        ):
            raise ValueError("parameters.n_outputs is inconsistent with shape")
        block_oc = 16
        block_hw = 16
        block_k = 16
        return (
            "conv2d_spatial_nchw_kernel",
            {
                "x_ptr": pointer_type,
                "w_ptr": pointer_type,
                "bias_ptr": pointer_type,
                "y_ptr": pointer_type,
            },
            {
                "XH": dimensions["h"],
                "XW": dimensions["w"],
                "OH": dimensions["oh"],
                "OW": dimensions["ow"],
                "C_IN": dimensions["c"],
                "C_OUT": dimensions["k"],
                "CIN_PER_GROUP": channels_per_group,
                "COUT_PER_GROUP": outputs_per_group,
                "GROUPS": groups,
                "STRIDE_H": stride_h,
                "STRIDE_W": stride_w,
                "PAD_TOP": pad_top,
                "PAD_LEFT": pad_left,
                "DIL_H": dilation_h,
                "DIL_W": dilation_w,
                "KH": dimensions["r"],
                "KW": dimensions["s"],
                "HAS_BIAS": False,
                "BLOCK_OC": block_oc,
                "BLOCK_HW": block_hw,
                "BLOCK_K": block_k,
                "GROUP_M": 8,
                "DTYPE_ID": {
                    "float16": 0,
                    "bfloat16": 1,
                    "float32": 2,
                }[tensor_data_types[0]],
                "INPUT_PRECISION": 0,
                "X_STRIDE_N": tensors[0]["strides"][0],
                "X_STRIDE_C": tensors[0]["strides"][1],
                "X_STRIDE_H": tensors[0]["strides"][2],
                "X_STRIDE_W": tensors[0]["strides"][3],
                "W_STRIDE_K": tensors[1]["strides"][0],
                "W_STRIDE_C": tensors[1]["strides"][1],
                "W_STRIDE_R": tensors[1]["strides"][2],
                "W_STRIDE_S": tensors[1]["strides"][3],
                "Y_STRIDE_N": tensors[2]["strides"][0],
                "Y_STRIDE_C": tensors[2]["strides"][1],
                "Y_STRIDE_H": tensors[2]["strides"][2],
                "Y_STRIDE_W": tensors[2]["strides"][3],
            },
            (
                (
                    (dimensions["oh"] * dimensions["ow"] + block_hw - 1)
                    // block_hw
                )
                * ((outputs_per_group + block_oc - 1) // block_oc),
                dimensions["n"] * groups,
                1,
            ),
            [
                ("tensor", None),
                ("tensor", None),
                ("tensor_alias", -1),
                ("tensor", None),
            ],
        )
    raise ValueError(f"unsupported operation: {operation!r}")


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
    if operation_name in {
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
        if kind == "tensor":
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


def _tensor_storage_size(tensor: dict[str, Any]) -> int:
    element_size = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "boolean": 1,
        "fp8_e4m3": 1,
        "fp8_e5m2": 1,
    }[tensor["data_type"]]
    storage_elements = 1 + sum(
        (dimension - 1) * stride
        for dimension, stride in zip(tensor["dimensions"], tensor["strides"])
    )
    return storage_elements * element_size


def _workspace_layout(
    tensors: dict[int, dict[str, Any]],
) -> tuple[dict[int, tuple[int, int]], int]:
    alignment = 256
    offset = 0
    result: dict[int, tuple[int, int]] = {}
    for uid, tensor in tensors.items():
        if not tensor["virtual"]:
            continue
        offset = (offset + alignment - 1) // alignment * alignment
        size = _tensor_storage_size(tensor)
        result[uid] = (offset, size)
        offset += size
    if offset != 0:
        offset = (offset + alignment - 1) // alignment * alignment
    return result, offset


def _lower_execution_groups(
    parsed_nodes: list[
        tuple[
            int,
            str,
            dict[str, Any],
            list[dict[str, Any]],
            list[int],
            list[int],
        ]
    ],
    tensor_registry: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fuse backend-supported graph patterns into executable kernel groups."""
    consumer_counts: dict[int, int] = {}
    for _, _, _, _, input_uids, _ in parsed_nodes:
        for uid in input_uids:
            consumer_counts[uid] = consumer_counts.get(uid, 0) + 1

    groups: list[dict[str, Any]] = []
    node_index = 0
    while node_index < len(parsed_nodes):
        node = parsed_nodes[node_index]
        if node_index + 2 < len(parsed_nodes):
            convolution = node
            bias_add = parsed_nodes[node_index + 1]
            relu = parsed_nodes[node_index + 2]
            convolution_output = convolution[5][0]
            bias_add_output = bias_add[5][0]
            bias_inputs = bias_add[4]
            bias_uid = (
                bias_inputs[1]
                if len(bias_inputs) == 2
                and bias_inputs[0] == convolution_output
                else (
                    bias_inputs[0]
                    if len(bias_inputs) == 2
                    and bias_inputs[1] == convolution_output
                    else None
                )
            )
            convolution_output_tensor = tensor_registry[convolution_output]
            bias_add_output_tensor = tensor_registry[bias_add_output]
            convolution_output_dimensions = convolution_output_tensor[
                "dimensions"
            ]
            expected_bias_dimensions = [
                1,
                convolution_output_dimensions[1],
            ] + [1] * (len(convolution_output_dimensions) - 2)
            relu_attributes = relu[2]
            is_standard_relu = (
                relu_attributes.get("negative_slope", 0) == 0
                and relu_attributes.get("relu_lower_clip", 0) == 0
                and relu_attributes.get("relu_lower_clip_slope", 0) == 0
                and relu_attributes.get("relu_upper_clip_set", False) is False
            )
            can_fuse = (
                convolution[1] in {"conv2d_fprop", "convolution_fprop"}
                and bias_add[1] == "add"
                and relu[1] == "relu"
                and len(convolution[5]) == 1
                and len(bias_add[5]) == 1
                and len(relu[5]) == 1
                and bias_uid is not None
                and relu[4] == [bias_add_output]
                and convolution_output_tensor["virtual"]
                and bias_add_output_tensor["virtual"]
                and consumer_counts.get(convolution_output) == 1
                and consumer_counts.get(bias_add_output) == 1
                and bias_add[2].get("alpha", 1) == 1
                and tensor_registry[bias_uid]["dimensions"]
                == expected_bias_dimensions
                and is_standard_relu
            )
            if can_fuse:
                parameters = dict(convolution[2])
                parameters["_fused_bias_relu"] = True
                groups.append(
                    {
                        "source_node_ids": [
                            convolution[0],
                            bias_add[0],
                            relu[0],
                        ],
                        "operation": convolution[1],
                        "parameters": parameters,
                        "tensors": [
                            convolution[3][0],
                            convolution[3][1],
                            tensor_registry[bias_uid],
                            relu[3][-1],
                        ],
                        "input_uids": [
                            convolution[4][0],
                            convolution[4][1],
                            bias_uid,
                        ],
                        "output_uids": relu[5],
                    }
                )
                node_index += 3
                continue

        groups.append(
            {
                "source_node_ids": [node[0]],
                "operation": node[1],
                "parameters": node[2],
                "tensors": node[3],
                "input_uids": node[4],
                "output_uids": node[5],
            }
        )
        node_index += 1
    return groups


def _expand_sdpa_fp8_forward_group(
    group: dict[str, Any],
) -> list[dict[str, Any]]:
    parameters = group["parameters"]
    tensors = group["tensors"]
    has_bias = _attention_flag(parameters, "has_bias")
    if len(tensors) != 13 + int(has_bias):
        raise ValueError("FP8 SDPA forward group tensor count is invalid")
    output_index = 10 if has_bias else 9
    amax_s = tensors[output_index + 2]
    amax_o = tensors[output_index + 3]
    return [
        {
            "source_node_ids": group["source_node_ids"],
            "operation": "sdpa_fp8",
            "parameters": {
                **parameters,
                "_sdpa_fp8_stage": "zero_amax",
            },
            "tensors": [amax_s, amax_o],
            "input_uids": [],
            "output_uids": [amax_s["uid"], amax_o["uid"]],
        },
        {
            "source_node_ids": group["source_node_ids"],
            "operation": "sdpa_fp8",
            "parameters": {
                **parameters,
                "_sdpa_fp8_stage": "forward",
            },
            "tensors": tensors,
            "input_uids": [
                *group["input_uids"],
                amax_s["uid"],
                amax_o["uid"],
            ],
            "output_uids": group["output_uids"],
        },
    ]


def _expand_sdpa_fp8_backward_group(
    group: dict[str, Any],
) -> list[dict[str, Any]]:
    parameters = group["parameters"]
    tensors = group["tensors"]
    if len(tensors) != 25:
        raise ValueError("FP8 SDPA backward group tensor count is invalid")
    q, k, v, output, doutput, stats = tensors[:6]
    (
        descale_q,
        descale_k,
        descale_v,
        descale_o,
        descale_doutput,
        descale_s,
        descale_dp,
        scale_s,
        scale_dq,
        scale_dk,
        scale_dv,
        scale_dp,
    ) = tensors[6:18]
    dq, dk, dv = tensors[18:21]
    amax_dq, amax_dk, amax_dv, amax_dp = tensors[21:25]
    dq_inputs = [
        q,
        k,
        v,
        output,
        doutput,
        stats,
        descale_q,
        descale_k,
        descale_v,
        descale_o,
        descale_doutput,
        descale_dp,
        scale_dq,
        scale_dp,
    ]
    dkdv_inputs = [
        q,
        k,
        v,
        output,
        doutput,
        stats,
        descale_q,
        descale_k,
        descale_v,
        descale_o,
        descale_doutput,
        descale_s,
        descale_dp,
        scale_s,
        scale_dk,
        scale_dv,
        scale_dp,
    ]
    return [
        {
            "source_node_ids": group["source_node_ids"],
            "operation": "sdpa_fp8_backward",
            "parameters": {
                **parameters,
                "_sdpa_fp8_bwd_stage": "zero_amax",
            },
            "tensors": [amax_dq, amax_dk, amax_dv, amax_dp],
            "input_uids": [],
            "output_uids": [
                amax_dq["uid"],
                amax_dk["uid"],
                amax_dv["uid"],
                amax_dp["uid"],
            ],
        },
        {
            "source_node_ids": group["source_node_ids"],
            "operation": "sdpa_fp8_backward",
            "parameters": {
                **parameters,
                "_sdpa_fp8_bwd_stage": "dq",
            },
            "tensors": [*dq_inputs, dq, amax_dq],
            "input_uids": [
                *[tensor["uid"] for tensor in dq_inputs],
                amax_dq["uid"],
            ],
            "output_uids": [dq["uid"], amax_dq["uid"]],
        },
        {
            "source_node_ids": group["source_node_ids"],
            "operation": "sdpa_fp8_backward",
            "parameters": {
                **parameters,
                "_sdpa_fp8_bwd_stage": "dkdv",
            },
            "tensors": [
                *dkdv_inputs,
                dk,
                dv,
                amax_dk,
                amax_dv,
                amax_dp,
            ],
            "input_uids": [
                *[tensor["uid"] for tensor in dkdv_inputs],
                amax_dk["uid"],
                amax_dv["uid"],
                amax_dp["uid"],
            ],
            "output_uids": [
                dk["uid"],
                dv["uid"],
                amax_dk["uid"],
                amax_dv["uid"],
                amax_dp["uid"],
            ],
        },
    ]


def _expand_sdpa_backward_group(
    group: dict[str, Any],
    tensor_registry: dict[int, dict[str, Any]],
    next_uid: int,
) -> tuple[list[dict[str, Any]], int]:
    parameters = group["parameters"]
    tensors = group["tensors"]
    has_bias = _attention_flag(parameters, "has_bias")
    has_dbias = _attention_flag(parameters, "has_dbias")
    expected_count = 9 + int(has_bias) + int(has_dbias)
    if len(tensors) != expected_count:
        raise ValueError("SDPA backward group tensor count is invalid")

    q, k, v, output, doutput, stats = tensors[:6]
    offset = 6
    bias = tensors[offset] if has_bias else None
    offset += int(has_bias)
    dq, dk, dv = tensors[offset : offset + 3]
    offset += 3
    dbias = tensors[offset] if has_dbias else None

    batch, heads, sequence_q, _ = q["dimensions"]
    delta = {
        "uid": next_uid,
        "virtual": True,
        "alignment": 16,
        "data_type": "float32",
        "dimensions": [batch, heads, sequence_q],
        "strides": [heads * sequence_q, sequence_q, 1],
    }
    tensor_registry[next_uid] = delta
    next_uid += 1

    dbias_reduce = bool(
        dbias is not None
        and (
            dbias["dimensions"][0] != batch or dbias["dimensions"][1] != heads
        )
    )
    if dbias_reduce and not _is_row_major_contiguous(dbias):
        raise ValueError(
            "SDPA broadcast dBias reduction requires contiguous storage"
        )
    pipeline_parameters = {
        **parameters,
        "dbias_reduce": int(dbias_reduce),
    }
    common_inputs = [q, k, v]
    if bias is not None:
        common_inputs.append(bias)

    stages: list[dict[str, Any]] = []
    if dbias_reduce:
        stages.append(
            {
                "source_node_ids": group["source_node_ids"],
                "operation": "sdpa_backward",
                "parameters": {
                    **pipeline_parameters,
                    "_sdpa_bwd_stage": "zero_dbias",
                },
                "tensors": [dbias],
                "input_uids": [],
                "output_uids": [dbias["uid"]],
            }
        )

    dq_tensors = [*common_inputs, output, doutput, stats, delta, dq]
    if dbias is not None:
        dq_tensors.append(dbias)
    dq_inputs = [tensor["uid"] for tensor in common_inputs]
    dq_inputs.extend([output["uid"], doutput["uid"], stats["uid"]])
    if dbias_reduce:
        dq_inputs.append(dbias["uid"])
    dq_outputs = [delta["uid"], dq["uid"]]
    if dbias is not None:
        dq_outputs.append(dbias["uid"])
    stages.append(
        {
            "source_node_ids": group["source_node_ids"],
            "operation": "sdpa_backward",
            "parameters": {
                **pipeline_parameters,
                "_sdpa_bwd_stage": "dq",
            },
            "tensors": dq_tensors,
            "input_uids": dq_inputs,
            "output_uids": dq_outputs,
        }
    )

    if q["dimensions"][3] == v["dimensions"][3]:
        stages.append(
            {
                "source_node_ids": group["source_node_ids"],
                "operation": "sdpa_backward",
                "parameters": {
                    **pipeline_parameters,
                    "_sdpa_bwd_stage": "dkdv",
                },
                "tensors": [
                    *common_inputs,
                    doutput,
                    stats,
                    delta,
                    dk,
                    dv,
                ],
                "input_uids": [
                    *[tensor["uid"] for tensor in common_inputs],
                    doutput["uid"],
                    stats["uid"],
                    delta["uid"],
                ],
                "output_uids": [dk["uid"], dv["uid"]],
            }
        )
    else:
        stages.append(
            {
                "source_node_ids": group["source_node_ids"],
                "operation": "sdpa_backward",
                "parameters": {
                    **pipeline_parameters,
                    "_sdpa_bwd_stage": "dk",
                },
                "tensors": [
                    *common_inputs,
                    doutput,
                    stats,
                    delta,
                    dk,
                ],
                "input_uids": [
                    *[tensor["uid"] for tensor in common_inputs],
                    doutput["uid"],
                    stats["uid"],
                    delta["uid"],
                ],
                "output_uids": [dk["uid"]],
            }
        )
        dv_inputs = [q, k]
        if bias is not None:
            dv_inputs.append(bias)
        stages.append(
            {
                "source_node_ids": group["source_node_ids"],
                "operation": "sdpa_backward",
                "parameters": {
                    **pipeline_parameters,
                    "_sdpa_bwd_stage": "dv",
                },
                "tensors": [*dv_inputs, doutput, stats, dv],
                "input_uids": [
                    *[tensor["uid"] for tensor in dv_inputs],
                    doutput["uid"],
                    stats["uid"],
                ],
                "output_uids": [dv["uid"]],
            }
        )
    return stages, next_uid


def _expand_execution_pipelines(
    groups: list[dict[str, Any]],
    tensor_registry: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expand selected NVIDIA algorithms into ordered internal stages."""
    result: list[dict[str, Any]] = []
    next_uid = max(tensor_registry) + 1
    for group in groups:
        parameters = group["parameters"]
        tensors = group["tensors"]
        if group["operation"] == "sdpa_fp8":
            result.extend(_expand_sdpa_fp8_forward_group(group))
            continue
        if group["operation"] == "sdpa_fp8_backward":
            result.extend(_expand_sdpa_fp8_backward_group(group))
            continue
        if group["operation"] == "sdpa_backward":
            stages, next_uid = _expand_sdpa_backward_group(
                group, tensor_registry, next_uid
            )
            result.extend(stages)
            continue
        is_fprop_1d_im2col = (
            group["operation"] == "convolution_fprop"
            and len(tensors) == 3
            and not parameters.get("_fused_bias_relu", False)
            and tensors[0]["data_type"] in {"float16", "bfloat16"}
            and len({tensor["data_type"] for tensor in tensors}) == 1
            and all(len(tensor["dimensions"]) == 3 for tensor in tensors)
            and _is_row_major_contiguous(tensors[0])
            and _is_row_major_contiguous(tensors[1])
            and tensors[0]["dimensions"] == [8, 64, 255]
            and tensors[1]["dimensions"] == [96, 64, 5]
            and tensors[2]["dimensions"] == [8, 96, 127]
            and tensors[2]["strides"] == [96 * 127, 1, 96]
            and parameters.get("spatial_rank") == 1
            and parameters.get("groups") == 1
            and parameters.get("stride") == [2]
            and parameters.get("pre_padding") == [2]
            and parameters.get("post_padding") == [1]
            and parameters.get("dilation") == [1]
            and parameters.get("convolution_mode", 0) == 0
        )
        is_fprop_general_im2col = (
            group["operation"] == "convolution_fprop"
            and len(tensors) == 3
            and not parameters.get("_fused_bias_relu", False)
            and all(_is_row_major_contiguous(tensor) for tensor in tensors)
            and all(len(tensor["dimensions"]) == 4 for tensor in tensors)
            and tensors[0]["data_type"] in {"float32", "float16", "bfloat16"}
            and len({tensor["data_type"] for tensor in tensors}) == 1
            and parameters.get("spatial_rank") == 2
            and parameters.get("groups") == 1
            and parameters.get("convolution_mode", 0) == 0
        )
        if is_fprop_general_im2col:
            input_dimensions = tensors[0]["dimensions"]
            filter_dimensions = tensors[1]["dimensions"]
            output_dimensions = tensors[2]["dimensions"]
            is_standard_or_dilation = (
                input_dimensions[0] * input_dimensions[1] == 256
                and input_dimensions[2:] == [32, 32]
                and filter_dimensions == [64, input_dimensions[1], 3, 3]
                and output_dimensions == [input_dimensions[0], 64, 32, 32]
                and parameters.get("stride") == [1, 1]
                and parameters.get("dilation") in ([1, 1], [2, 2])
                and parameters.get("pre_padding") == parameters.get("dilation")
                and parameters.get("post_padding")
                == parameters.get("dilation")
            )
            is_asymmetric = (
                input_dimensions == [4, 32, 35, 37]
                and filter_dimensions == [48, 32, 3, 5]
                and output_dimensions == [4, 48, 35, 18]
                and parameters.get("stride") == [1, 2]
                and parameters.get("pre_padding") == [1, 0]
                and parameters.get("post_padding") == [1, 2]
                and parameters.get("dilation") == [1, 1]
            )
            is_fprop_general_im2col = is_standard_or_dilation or is_asymmetric
        if is_fprop_1d_im2col or is_fprop_general_im2col:
            input_tensor, weight, output = tensors
            n, channels = input_tensor["dimensions"][:2]
            output_channels = weight["dimensions"][0]
            output_area = math.prod(output["dimensions"][2:])
            reduction_extent = channels * math.prod(weight["dimensions"][2:])
            im2col_input = input_tensor
            im2col_weight = weight
            im2col_parameters = parameters
            if is_fprop_1d_im2col:
                input_l = input_tensor["dimensions"][2]
                kernel_w = weight["dimensions"][2]
                im2col_input = {
                    **input_tensor,
                    "dimensions": [n, channels, 1, input_l],
                    "strides": [
                        input_tensor["strides"][0],
                        input_tensor["strides"][1],
                        input_l,
                        input_tensor["strides"][2],
                    ],
                }
                im2col_weight = {
                    **weight,
                    "dimensions": [output_channels, channels, 1, kernel_w],
                    "strides": [
                        weight["strides"][0],
                        weight["strides"][1],
                        kernel_w,
                        weight["strides"][2],
                    ],
                }
                im2col_parameters = {
                    **parameters,
                    "spatial_rank": 2,
                    "stride": [1, parameters["stride"][0]],
                    "pre_padding": [0, parameters["pre_padding"][0]],
                    "post_padding": [0, parameters["post_padding"][0]],
                    "dilation": [1, parameters["dilation"][0]],
                }
            workspace_data_type = (
                "float16"
                if input_tensor["data_type"] == "float32"
                else input_tensor["data_type"]
            )
            columns = {
                "uid": next_uid,
                "virtual": True,
                "data_type": workspace_data_type,
                "dimensions": [n, reduction_extent, output_area],
                "strides": [
                    reduction_extent * output_area,
                    output_area,
                    1,
                ],
            }
            tensor_registry[next_uid] = columns
            next_uid += 1
            im2col_tensors = [im2col_input, im2col_weight, columns]
            im2col_outputs = [columns["uid"]]
            gemm_weight = weight
            if input_tensor["data_type"] == "float32":
                converted_weight = {
                    "uid": next_uid,
                    "virtual": True,
                    "data_type": "float16",
                    "dimensions": list(im2col_weight["dimensions"]),
                    "strides": list(im2col_weight["strides"]),
                }
                tensor_registry[next_uid] = converted_weight
                next_uid += 1
                im2col_tensors.append(converted_weight)
                im2col_outputs.append(converted_weight["uid"])
                gemm_weight = converted_weight

            result.append(
                {
                    "source_node_ids": group["source_node_ids"],
                    "operation": group["operation"],
                    "parameters": {
                        **im2col_parameters,
                        "_fprop_pipeline_stage": "im2col",
                        "_fprop_pipeline_algorithm": "general",
                    },
                    "tensors": im2col_tensors,
                    "input_uids": [input_tensor["uid"], weight["uid"]],
                    "output_uids": im2col_outputs,
                }
            )
            gemm_weight_view = {
                **gemm_weight,
                "dimensions": [1, output_channels, reduction_extent],
                "strides": [
                    output_channels * reduction_extent,
                    reduction_extent,
                    1,
                ],
            }
            output_view = {
                **output,
                "dimensions": [n, output_channels, output_area],
                "strides": [
                    output["strides"][0],
                    output["strides"][1],
                    output["strides"][-1],
                ],
            }
            result.append(
                {
                    "source_node_ids": group["source_node_ids"],
                    "operation": "matmul",
                    "parameters": {
                        "batch": n,
                        "m": output_channels,
                        "n": output_area,
                        "k": reduction_extent,
                        "_fprop_broadcast_a": reduction_extent == 288,
                        "_fprop_im2col_matmul": True,
                        "_fprop_lowp_inputs_fp32_output": (
                            input_tensor["data_type"] == "float32"
                        ),
                    },
                    "tensors": [gemm_weight_view, columns, output_view],
                    "input_uids": [gemm_weight["uid"], columns["uid"]],
                    "output_uids": group["output_uids"],
                }
            )
            continue

        is_fprop_stride2_im2col = (
            group["operation"] == "convolution_fprop"
            and len(tensors) == 3
            and not parameters.get("_fused_bias_relu", False)
            and all(_is_row_major_contiguous(tensor) for tensor in tensors)
            and len(tensors[0]["dimensions"]) == 4
            and len(tensors[1]["dimensions"]) == 4
            and len(tensors[2]["dimensions"]) == 4
            and tensors[1]["dimensions"][2:] == [3, 3]
            and parameters.get("spatial_rank") == 2
            and parameters.get("groups") == 1
            and parameters.get("stride") == [2, 2]
            and parameters.get("pre_padding") == [1, 1]
            and parameters.get("post_padding") == [1, 1]
            and parameters.get("dilation") == [1, 1]
            and parameters.get("convolution_mode", 0) == 0
        )
        if is_fprop_stride2_im2col:
            input_tensor, weight, output = tensors
            n, channels, input_h, input_w = input_tensor["dimensions"]
            output_channels, filter_channels, _, _ = weight["dimensions"]
            output_h = (input_h + 1) // 2
            output_w = (input_w + 1) // 2
            output_area = output_h * output_w
            reduction_extent = channels * 9
            is_p5 = (
                n == 1
                and input_h == 40
                and input_w == 40
                and channels >= 128
                and output_channels >= 256
            )
            column_leading_dimension = output_area
            shape_is_consistent = (
                filter_channels == channels
                and output["dimensions"]
                == [n, output_channels, output_h, output_w]
                and len(
                    {
                        input_tensor["data_type"],
                        weight["data_type"],
                        output["data_type"],
                    }
                )
                == 1
            )
            if shape_is_consistent:
                columns = {
                    "uid": next_uid,
                    "virtual": True,
                    "data_type": (
                        "float16"
                        if input_tensor["data_type"] == "float32"
                        else input_tensor["data_type"]
                    ),
                    "dimensions": [
                        n,
                        reduction_extent,
                        column_leading_dimension,
                    ],
                    "strides": [
                        reduction_extent * column_leading_dimension,
                        column_leading_dimension,
                        1,
                    ],
                }
                tensor_registry[next_uid] = columns
                next_uid += 1
                columns_view = {
                    **columns,
                    "dimensions": [n, reduction_extent, output_area],
                }
                weight_view = {
                    **weight,
                    "dimensions": [1, output_channels, reduction_extent],
                    "strides": [
                        output_channels * reduction_extent,
                        reduction_extent,
                        1,
                    ],
                }
                output_view = {
                    **output,
                    "dimensions": [n, output_channels, output_area],
                    "strides": [
                        output_channels * output_area,
                        output_area,
                        1,
                    ],
                }
                result.append(
                    {
                        "source_node_ids": group["source_node_ids"],
                        "operation": group["operation"],
                        "parameters": {
                            **parameters,
                            "_fprop_pipeline_stage": "im2col",
                        },
                        "tensors": [input_tensor, columns],
                        "input_uids": [input_tensor["uid"]],
                        "output_uids": [columns["uid"]],
                    }
                )
                if is_p5:
                    num_splits = 4
                    partial = {
                        "uid": next_uid,
                        "virtual": True,
                        "data_type": "float32",
                        "dimensions": [
                            num_splits,
                            output_channels,
                            output_area,
                        ],
                        "strides": [
                            output_channels * output_area,
                            output_area,
                            1,
                        ],
                    }
                    tensor_registry[next_uid] = partial
                    next_uid += 1
                    pipeline_parameters = {
                        "m": output_channels,
                        "n": output_area,
                        "k": reduction_extent,
                        "_fprop_p5_splits": num_splits,
                    }
                    result.append(
                        {
                            "source_node_ids": group["source_node_ids"],
                            "operation": "matmul",
                            "parameters": {
                                **pipeline_parameters,
                                "_fprop_p5_matmul_stage": "split",
                            },
                            "tensors": [weight_view, columns_view, partial],
                            "input_uids": [
                                weight["uid"],
                                columns["uid"],
                            ],
                            "output_uids": [partial["uid"]],
                        }
                    )
                    result.append(
                        {
                            "source_node_ids": group["source_node_ids"],
                            "operation": "matmul",
                            "parameters": {
                                **pipeline_parameters,
                                "_fprop_p5_matmul_stage": "reduce",
                            },
                            "tensors": [partial, output_view],
                            "input_uids": [partial["uid"]],
                            "output_uids": group["output_uids"],
                        }
                    )
                else:
                    result.append(
                        {
                            "source_node_ids": group["source_node_ids"],
                            "operation": "matmul",
                            "parameters": {
                                "batch": n,
                                "m": output_channels,
                                "n": output_area,
                                "k": reduction_extent,
                                "_fprop_mixed_fp16": (
                                    input_tensor["data_type"] == "float32"
                                ),
                            },
                            "tensors": [
                                weight_view,
                                columns_view,
                                output_view,
                            ],
                            "input_uids": [
                                weight["uid"],
                                columns["uid"],
                            ],
                            "output_uids": group["output_uids"],
                        }
                    )
                continue

        is_fp32_3d_ci8_dot = (
            group["operation"] == "convolution_dgrad"
            and len(tensors) == 3
            and all(tensor["data_type"] == "float32" for tensor in tensors)
            and tensors[0]["dimensions"] == [2, 16, 8, 16, 16]
            and tensors[1]["dimensions"] == [16, 8, 3, 3, 3]
            and tensors[2]["dimensions"] == [2, 8, 8, 16, 16]
            and parameters.get("spatial_rank") == 3
            and parameters.get("groups") == 1
            and parameters.get("stride") == [1, 1, 1]
            and parameters.get("pre_padding") == [1, 1, 1]
            and parameters.get("post_padding") == [1, 1, 1]
            and parameters.get("dilation") == [1, 1, 1]
            and parameters.get("convolution_mode", 0) == 0
        )
        is_lowp_3d_packed_shape = (
            group["operation"] == "convolution_dgrad"
            and len(tensors) == 3
            and tensors[0]["data_type"] in {"float16", "bfloat16"}
            and len({tensor["data_type"] for tensor in tensors}) == 1
            and tensors[0]["dimensions"] == [2, 16, 8, 16, 16]
            and tensors[1]["dimensions"] == [16, 8, 3, 3, 3]
            and tensors[2]["dimensions"] == [2, 8, 8, 16, 16]
            and parameters.get("spatial_rank") == 3
            and parameters.get("groups") == 1
            and parameters.get("stride") == [1, 1, 1]
            and parameters.get("pre_padding") == [1, 1, 1]
            and parameters.get("post_padding") == [1, 1, 1]
            and parameters.get("dilation") == [1, 1, 1]
            and parameters.get("convolution_mode", 0) == 0
        )
        is_packed_dgrad_3d = (
            group["operation"] == "convolution_dgrad"
            and len(tensors) == 3
            and all(_is_row_major_contiguous(tensor) for tensor in tensors)
            and all(len(tensor["dimensions"]) == 5 for tensor in tensors)
            and len({tensor["data_type"] for tensor in tensors}) == 1
            and (is_lowp_3d_packed_shape or is_fp32_3d_ci8_dot)
            and tensors[1]["dimensions"][0] == tensors[0]["dimensions"][1]
            and tensors[1]["dimensions"][1] == tensors[2]["dimensions"][1]
            and tensors[0]["dimensions"][0] == tensors[2]["dimensions"][0]
            and parameters.get("spatial_rank") == 3
            and parameters.get("groups") == 1
            and parameters.get("convolution_mode", 0) == 0
        )
        if is_packed_dgrad_3d:
            loss, weight, output = tensors
            c_out, c_in, kernel_d, kernel_h, kernel_w = weight["dimensions"]
            packed = {
                "uid": next_uid,
                "virtual": True,
                "data_type": weight["data_type"],
                "dimensions": [
                    kernel_d,
                    kernel_h,
                    kernel_w,
                    c_out,
                    c_in,
                ],
                "strides": [
                    kernel_h * kernel_w * c_out * c_in,
                    kernel_w * c_out * c_in,
                    c_out * c_in,
                    c_in,
                    1,
                ],
            }
            tensor_registry[next_uid] = packed
            next_uid += 1
            result.append(
                {
                    "source_node_ids": group["source_node_ids"],
                    "operation": group["operation"],
                    "parameters": {
                        **parameters,
                        "_dgrad_3d_pipeline_stage": "pack",
                    },
                    "tensors": [weight, packed],
                    "input_uids": [weight["uid"]],
                    "output_uids": [packed["uid"]],
                }
            )
            result.append(
                {
                    "source_node_ids": group["source_node_ids"],
                    "operation": group["operation"],
                    "parameters": {
                        **parameters,
                        "_dgrad_3d_pipeline_stage": (
                            "compute_ci8_dot"
                            if is_fp32_3d_ci8_dot
                            else "compute_packed"
                        ),
                    },
                    "tensors": [loss, packed, output],
                    "input_uids": [loss["uid"], packed["uid"]],
                    "output_uids": group["output_uids"],
                }
            )
            continue

        is_fp32_p5_tile4 = (
            group["operation"] == "convolution_dgrad"
            and len(tensors) == 3
            and all(tensor["data_type"] == "float32" for tensor in tensors)
            and all(len(tensor["dimensions"]) == 4 for tensor in tensors)
            and tensors[0]["dimensions"][0] == 1
            and tensors[0]["dimensions"][2:] == [20, 20]
            and tensors[1]["dimensions"][0] == tensors[0]["dimensions"][1]
            and tensors[1]["dimensions"][0] >= 512
            and tensors[1]["dimensions"][1] >= 256
            and tensors[1]["dimensions"][2:] == [3, 3]
            and tensors[2]["dimensions"]
            == [1, tensors[1]["dimensions"][1], 40, 40]
            and parameters.get("spatial_rank") == 2
            and parameters.get("groups") == 1
            and parameters.get("stride") == [2, 2]
            and parameters.get("pre_padding") == [1, 1]
            and parameters.get("post_padding") == [1, 1]
            and parameters.get("dilation") == [1, 1]
        )
        is_exact_p5_768 = (
            group["operation"] == "convolution_dgrad"
            and len(tensors) == 3
            and tensors[0]["dimensions"] == [1, 768, 20, 20]
            and tensors[1]["dimensions"] == [768, 768, 3, 3]
            and tensors[2]["dimensions"] == [1, 768, 40, 40]
            and parameters.get("spatial_rank") == 2
            and parameters.get("groups") == 1
            and parameters.get("stride") == [2, 2]
            and parameters.get("pre_padding") == [1, 1]
            and parameters.get("post_padding") == [1, 1]
            and parameters.get("dilation") == [1, 1]
            and parameters.get("convolution_mode") == 0
        )
        is_packed_dgrad_stride2 = (
            group["operation"] == "convolution_dgrad"
            and len(tensors) == 3
            and all(_is_row_major_contiguous(tensor) for tensor in tensors)
            and len(tensors[0]["dimensions"]) == 4
            and len(tensors[1]["dimensions"]) == 4
            and len(tensors[2]["dimensions"]) == 4
            and len({tensor["data_type"] for tensor in tensors}) == 1
            and tensors[0]["data_type"] in {"float32", "float16", "bfloat16"}
            and not (
                tensors[0]["data_type"] == "float32"
                and tensors[1]["dimensions"][0] >= 512
                and not is_fp32_p5_tile4
            )
            and tensors[1]["dimensions"][2:] == [3, 3]
            and parameters.get("spatial_rank") == 2
            and parameters.get("groups") == 1
            and parameters.get("stride") == [2, 2]
            and parameters.get("pre_padding") == [1, 1]
            and parameters.get("post_padding") == [1, 1]
            and parameters.get("dilation") == [1, 1]
            and parameters.get("convolution_mode") in {0, 1}
        )
        if is_packed_dgrad_stride2:
            loss, weight, output = tensors
            c_out, c_in, _, _ = weight["dimensions"]
            mixed_fp16 = is_exact_p5_768 and loss["data_type"] == "float32"
            compute_loss = loss
            if mixed_fp16:
                compute_loss = {
                    "uid": next_uid,
                    "virtual": True,
                    "data_type": "float16",
                    "dimensions": list(loss["dimensions"]),
                    "strides": list(loss["strides"]),
                }
                tensor_registry[next_uid] = compute_loss
                next_uid += 1
                result.append(
                    {
                        "source_node_ids": group["source_node_ids"],
                        "operation": group["operation"],
                        "parameters": {
                            **parameters,
                            "_dgrad_pipeline_stage": "cast_p5_loss",
                            "_dgrad_mixed_fp16": True,
                        },
                        "tensors": [loss, compute_loss],
                        "input_uids": [loss["uid"]],
                        "output_uids": [compute_loss["uid"]],
                    }
                )
            packed = {
                "uid": next_uid,
                "virtual": True,
                "data_type": (
                    "float16" if mixed_fp16 else weight["data_type"]
                ),
                "dimensions": [3, 3, c_out, c_in],
                "strides": [
                    3 * c_out * c_in,
                    c_out * c_in,
                    c_in,
                    1,
                ],
            }
            tensor_registry[next_uid] = packed
            next_uid += 1
            result.append(
                {
                    "source_node_ids": group["source_node_ids"],
                    "operation": group["operation"],
                    "parameters": {
                        **parameters,
                        "_dgrad_pipeline_stage": "pack",
                        "_dgrad_mixed_fp16": mixed_fp16,
                        "_dgrad_pack_round_tf32": (
                            is_fp32_p5_tile4 and not is_exact_p5_768
                        ),
                    },
                    "tensors": [weight, packed],
                    "input_uids": [weight["uid"]],
                    "output_uids": [packed["uid"]],
                }
            )
            if is_exact_p5_768:
                for parity_h in range(2):
                    for parity_w in range(2):
                        result.append(
                            {
                                "source_node_ids": group["source_node_ids"],
                                "operation": group["operation"],
                                "parameters": {
                                    **parameters,
                                    "_dgrad_pipeline_stage": "compute",
                                    "_dgrad_parity_h": parity_h,
                                    "_dgrad_parity_w": parity_w,
                                    "_dgrad_p5_parity": True,
                                    "_dgrad_mixed_fp16": mixed_fp16,
                                },
                                "tensors": [
                                    compute_loss,
                                    packed,
                                    output,
                                ],
                                "input_uids": [
                                    compute_loss["uid"],
                                    packed["uid"],
                                ],
                                "output_uids": group["output_uids"],
                            }
                        )
                continue
            if is_fp32_p5_tile4:
                result.append(
                    {
                        "source_node_ids": group["source_node_ids"],
                        "operation": group["operation"],
                        "parameters": {
                            **parameters,
                            "_dgrad_pipeline_stage": "compute_tile4",
                            "_dgrad_round_tf32": True,
                        },
                        "tensors": [loss, packed, output],
                        "input_uids": [loss["uid"], packed["uid"]],
                        "output_uids": group["output_uids"],
                    }
                )
                continue

            use_tile4 = (
                128 <= c_in <= 768
                and loss["dimensions"][2] * loss["dimensions"][3] <= 1024
                and (loss["data_type"] != "float32" or c_in <= 256)
            )
            if use_tile4:
                result.append(
                    {
                        "source_node_ids": group["source_node_ids"],
                        "operation": group["operation"],
                        "parameters": {
                            **parameters,
                            "_dgrad_pipeline_stage": "compute_tile4",
                        },
                        "tensors": [loss, packed, output],
                        "input_uids": [loss["uid"], packed["uid"]],
                        "output_uids": group["output_uids"],
                    }
                )
            elif loss["data_type"] == "float32":
                for parity_h in range(2):
                    result.append(
                        {
                            "source_node_ids": group["source_node_ids"],
                            "operation": group["operation"],
                            "parameters": {
                                **parameters,
                                "_dgrad_pipeline_stage": "compute_tile2w",
                                "_dgrad_parity_h": parity_h,
                                "_dgrad_small_ci": c_in < 32,
                            },
                            "tensors": [loss, packed, output],
                            "input_uids": [loss["uid"], packed["uid"]],
                            "output_uids": group["output_uids"],
                        }
                    )
            else:
                for parity_h in range(2):
                    for parity_w in range(2):
                        result.append(
                            {
                                "source_node_ids": group["source_node_ids"],
                                "operation": group["operation"],
                                "parameters": {
                                    **parameters,
                                    "_dgrad_pipeline_stage": "compute",
                                    "_dgrad_parity_h": parity_h,
                                    "_dgrad_parity_w": parity_w,
                                },
                                "tensors": [loss, packed, output],
                                "input_uids": [
                                    loss["uid"],
                                    packed["uid"],
                                ],
                                "output_uids": group["output_uids"],
                            }
                        )
            continue

        is_exact_wgrad_stem = (
            group["operation"] == "convolution_wgrad"
            and len(tensors) == 3
            and tensors[1]["dimensions"] == [1, 3, 640, 640]
            and tensors[0]["dimensions"]
            == [1, tensors[2]["dimensions"][0], 320, 320]
            and tensors[2]["dimensions"]
            == [tensors[2]["dimensions"][0], 3, 3, 3]
            and tensors[2]["dimensions"][0] in {16, 32, 64, 96}
            and all(_is_row_major_contiguous(tensor) for tensor in tensors)
            and parameters.get("spatial_rank") == 2
            and parameters.get("groups") == 1
            and parameters.get("stride") == [2, 2]
            and parameters.get("pre_padding") == [1, 1]
            and parameters.get("post_padding") == [1, 1]
            and parameters.get("dilation") == [1, 1]
            and parameters.get("convolution_mode") == 0
        )
        if is_exact_wgrad_stem:
            loss, image, output = tensors
            num_splits = 64
            c_out, cin_per_group, kernel_h, kernel_w = output["dimensions"]
            cik = cin_per_group * kernel_h * kernel_w
            partial = {
                "uid": next_uid,
                "virtual": True,
                "data_type": "float32",
                "dimensions": [num_splits, c_out, cik],
                "strides": [c_out * cik, cik, 1],
            }
            tensor_registry[next_uid] = partial
            next_uid += 1
            pipeline_parameters = {
                **parameters,
                "_wgrad_pipeline_algorithm": "stem_col",
                "_wgrad_num_splits": num_splits,
                "_wgrad_kernel_h": kernel_h,
                "_wgrad_kernel_w": kernel_w,
            }
            result.append(
                {
                    "source_node_ids": group["source_node_ids"],
                    "operation": group["operation"],
                    "parameters": {
                        **pipeline_parameters,
                        "_wgrad_pipeline_stage": "split",
                    },
                    "tensors": [image, loss, partial],
                    "input_uids": [image["uid"], loss["uid"]],
                    "output_uids": [partial["uid"]],
                }
            )
            result.append(
                {
                    "source_node_ids": group["source_node_ids"],
                    "operation": group["operation"],
                    "parameters": {
                        **pipeline_parameters,
                        "_wgrad_pipeline_stage": "reduce",
                    },
                    "tensors": [partial, output],
                    "input_uids": [partial["uid"]],
                    "output_uids": group["output_uids"],
                }
            )
            continue

        is_exact_wgrad_p5 = (
            group["operation"] == "convolution_wgrad"
            and len(tensors) == 3
            and (
                tuple(tensors[1]["dimensions"]),
                tuple(tensors[0]["dimensions"]),
                tuple(tensors[2]["dimensions"]),
            )
            in {
                (
                    (1, 128, 40, 40),
                    (1, 256, 20, 20),
                    (256, 128, 3, 3),
                ),
                (
                    (1, 256, 40, 40),
                    (1, 512, 20, 20),
                    (512, 256, 3, 3),
                ),
                (
                    (1, 512, 40, 40),
                    (1, 512, 20, 20),
                    (512, 512, 3, 3),
                ),
                (
                    (1, 768, 40, 40),
                    (1, 768, 20, 20),
                    (768, 768, 3, 3),
                ),
            }
            and all(_is_row_major_contiguous(tensor) for tensor in tensors)
            and parameters.get("spatial_rank") == 2
            and parameters.get("groups") == 1
            and parameters.get("stride") == [2, 2]
            and parameters.get("pre_padding") == [1, 1]
            and parameters.get("post_padding") == [1, 1]
            and parameters.get("dilation") == [1, 1]
            and parameters.get("convolution_mode") == 0
        )
        if is_exact_wgrad_p5:
            loss, image, output = tensors
            c_in = image["dimensions"][1]
            cik = c_in * 9
            mixed_fp16 = all(
                tensor["data_type"] == "float32"
                for tensor in (loss, image, output)
            )
            pipeline_parameters = {
                **parameters,
                "_wgrad_pipeline_algorithm": "p5",
                "_wgrad_mixed_fp16": mixed_fp16,
            }
            matmul_loss = loss
            if mixed_fp16:
                matmul_loss = {
                    "uid": next_uid,
                    "virtual": True,
                    "data_type": "float16",
                    "dimensions": list(loss["dimensions"]),
                    "strides": list(loss["strides"]),
                }
                tensor_registry[next_uid] = matmul_loss
                next_uid += 1
                result.append(
                    {
                        "source_node_ids": group["source_node_ids"],
                        "operation": group["operation"],
                        "parameters": {
                            **pipeline_parameters,
                            "_wgrad_pipeline_stage": "cast_loss",
                        },
                        "tensors": [loss, matmul_loss],
                        "input_uids": [loss["uid"]],
                        "output_uids": [matmul_loss["uid"]],
                    }
                )
            packed = {
                "uid": next_uid,
                "virtual": True,
                "data_type": "float16" if mixed_fp16 else image["data_type"],
                "dimensions": [400, cik],
                "strides": [cik, 1],
            }
            tensor_registry[next_uid] = packed
            next_uid += 1
            result.append(
                {
                    "source_node_ids": group["source_node_ids"],
                    "operation": group["operation"],
                    "parameters": {
                        **pipeline_parameters,
                        "_wgrad_pipeline_stage": "pack",
                    },
                    "tensors": [image, packed],
                    "input_uids": [image["uid"]],
                    "output_uids": [packed["uid"]],
                }
            )
            result.append(
                {
                    "source_node_ids": group["source_node_ids"],
                    "operation": group["operation"],
                    "parameters": {
                        **pipeline_parameters,
                        "_wgrad_pipeline_stage": "matmul",
                    },
                    "tensors": [matmul_loss, packed, output],
                    "input_uids": [matmul_loss["uid"], packed["uid"]],
                    "output_uids": group["output_uids"],
                }
            )
            continue

        is_exact_wgrad_stride2 = (
            group["operation"] == "convolution_wgrad"
            and len(tensors) == 3
            and tensors[0]["dimensions"] == [8, 128, 28, 28]
            and tensors[1]["dimensions"] == [8, 64, 56, 56]
            and tensors[2]["dimensions"] == [128, 64, 3, 3]
            and all(_is_row_major_contiguous(tensor) for tensor in tensors)
            and parameters.get("spatial_rank") == 2
            and parameters.get("groups") == 1
            and parameters.get("stride") == [2, 2]
            and parameters.get("pre_padding") == [1, 1]
            and parameters.get("post_padding") == [1, 1]
            and parameters.get("dilation") == [1, 1]
            and parameters.get("convolution_mode") == 0
        )
        if is_exact_wgrad_stride2:
            loss, image, output = tensors
            if image["data_type"] in FLOAT_DATA_TYPES:
                n, _, _, _ = image["dimensions"]
                _, c_out, loss_h, loss_w = loss["dimensions"]
                _, cin_per_group, kernel_h, kernel_w = output["dimensions"]
                output_area = loss_h * loss_w
                padded_area = _next_power_of_two(output_area)
                cik = cin_per_group * kernel_h * kernel_w
                columns = {
                    "uid": next_uid,
                    "virtual": True,
                    "data_type": image["data_type"],
                    "dimensions": [n, cik, padded_area],
                    "strides": [cik * padded_area, padded_area, 1],
                }
                tensor_registry[next_uid] = columns
                next_uid += 1
                partial = {
                    "uid": next_uid,
                    "virtual": True,
                    "data_type": image["data_type"],
                    "dimensions": [n, c_out, cik],
                    "strides": [c_out * cik, cik, 1],
                }
                tensor_registry[next_uid] = partial
                next_uid += 1
                pipeline_parameters = {
                    **parameters,
                    "_wgrad_pipeline_algorithm": "stride2_im2col",
                    "_wgrad_num_splits": n,
                    "_wgrad_kernel_h": kernel_h,
                    "_wgrad_kernel_w": kernel_w,
                }
                result.append(
                    {
                        "source_node_ids": group["source_node_ids"],
                        "operation": group["operation"],
                        "parameters": {
                            **pipeline_parameters,
                            "_wgrad_pipeline_stage": "im2col",
                        },
                        "tensors": [image, columns],
                        "input_uids": [image["uid"]],
                        "output_uids": [columns["uid"]],
                    }
                )
                result.append(
                    {
                        "source_node_ids": group["source_node_ids"],
                        "operation": group["operation"],
                        "parameters": {
                            **pipeline_parameters,
                            "_wgrad_pipeline_stage": "matmul",
                        },
                        "tensors": [loss, columns, partial],
                        "input_uids": [loss["uid"], columns["uid"]],
                        "output_uids": [partial["uid"]],
                    }
                )
                result.append(
                    {
                        "source_node_ids": group["source_node_ids"],
                        "operation": group["operation"],
                        "parameters": {
                            **pipeline_parameters,
                            "_wgrad_pipeline_stage": "reduce",
                        },
                        "tensors": [partial, output],
                        "input_uids": [partial["uid"]],
                        "output_uids": group["output_uids"],
                    }
                )
                continue
            num_splits = image["dimensions"][0]
            c_out, cin_per_group, kernel_h, kernel_w = output["dimensions"]
            kernel_elements = kernel_h * kernel_w
            partial = {
                "uid": next_uid,
                "virtual": True,
                "data_type": image["data_type"],
                "dimensions": [
                    num_splits,
                    c_out,
                    cin_per_group,
                    kernel_elements,
                ],
                "strides": [
                    c_out * cin_per_group * kernel_elements,
                    cin_per_group * kernel_elements,
                    kernel_elements,
                    1,
                ],
            }
            tensor_registry[next_uid] = partial
            next_uid += 1
            pipeline_parameters = {
                **parameters,
                "_wgrad_pipeline_algorithm": "stride2_row4",
                "_wgrad_num_splits": num_splits,
            }
            result.append(
                {
                    "source_node_ids": group["source_node_ids"],
                    "operation": group["operation"],
                    "parameters": {
                        **pipeline_parameters,
                        "_wgrad_pipeline_stage": "split",
                    },
                    "tensors": [image, loss, partial],
                    "input_uids": [image["uid"], loss["uid"]],
                    "output_uids": [partial["uid"]],
                }
            )
            result.append(
                {
                    "source_node_ids": group["source_node_ids"],
                    "operation": group["operation"],
                    "parameters": {
                        **pipeline_parameters,
                        "_wgrad_pipeline_stage": "reduce",
                    },
                    "tensors": [partial, output],
                    "input_uids": [partial["uid"]],
                    "output_uids": group["output_uids"],
                }
            )
            continue

        is_wgrad_pipeline_candidate = (
            group["operation"] == "convolution_wgrad"
            and len(tensors) == 3
            and all(_is_row_major_contiguous(tensor) for tensor in tensors)
            and parameters.get("spatial_rank") == 2
            and parameters.get("groups") == 1
            and parameters.get("stride") == [1, 1]
            and parameters.get("dilation") == [1, 1]
            and parameters.get("convolution_mode") == 0
        )
        is_exact_wgrad_stride1 = (
            is_wgrad_pipeline_candidate
            and tensors[0]["dimensions"] == [8, 64, 32, 32]
            and tensors[1]["dimensions"] == [8, 32, 32, 32]
            and tensors[2]["dimensions"] == [64, 32, 3, 3]
            and parameters.get("pre_padding") == [1, 1]
            and parameters.get("post_padding") == [1, 1]
        )
        is_exact_wgrad_1x1 = (
            is_wgrad_pipeline_candidate
            and tensors[0]["dimensions"] == [8, 128, 28, 28]
            and tensors[1]["dimensions"] == [8, 64, 28, 28]
            and tensors[2]["dimensions"] == [128, 64, 1, 1]
            and parameters.get("pre_padding") == [0, 0]
            and parameters.get("post_padding") == [0, 0]
        )
        if not is_exact_wgrad_stride1 and not is_exact_wgrad_1x1:
            result.append(group)
            continue

        loss, image, output = tensors
        if is_exact_wgrad_1x1 and image["data_type"] in {
            "float16",
            "bfloat16",
        }:
            n, c_in, image_h, image_w = image["dimensions"]
            c_out = output["dimensions"][0]
            image_area = image_h * image_w
            num_splits = 3 * n
            image_view = {
                **image,
                "dimensions": [n, c_in, image_area],
                "strides": [c_in * image_area, image_area, 1],
            }
            pipeline_parameters = {
                **parameters,
                "_wgrad_num_splits": num_splits,
                "_wgrad_kernel_h": 1,
                "_wgrad_kernel_w": 1,
                "_wgrad_pipeline_algorithm": "1x1_tma",
            }
            partial = {
                "uid": next_uid,
                "virtual": True,
                "data_type": image["data_type"],
                "dimensions": [num_splits, c_out, c_in],
                "strides": [c_out * c_in, c_in, 1],
            }
            tensor_registry[next_uid] = partial
            next_uid += 1
            result.append(
                {
                    "source_node_ids": group["source_node_ids"],
                    "operation": group["operation"],
                    "parameters": {
                        **pipeline_parameters,
                        "_wgrad_pipeline_stage": "matmul",
                    },
                    "tensors": [loss, image_view, partial],
                    "input_uids": [loss["uid"], image["uid"]],
                    "output_uids": [partial["uid"]],
                }
            )
            result.append(
                {
                    "source_node_ids": group["source_node_ids"],
                    "operation": group["operation"],
                    "parameters": {
                        **pipeline_parameters,
                        "_wgrad_pipeline_stage": "reduce",
                    },
                    "tensors": [partial, output],
                    "input_uids": [partial["uid"]],
                    "output_uids": group["output_uids"],
                }
            )
            continue

        pipeline_algorithm = "1x1" if is_exact_wgrad_1x1 else "3tap"
        num_splits = 32 if is_exact_wgrad_1x1 else 16
        c_out = tensors[2]["dimensions"][0]
        cin_per_group = tensors[2]["dimensions"][1]
        kernel_h = tensors[2]["dimensions"][2]
        kernel_w = tensors[2]["dimensions"][3]
        cik = cin_per_group * kernel_h * kernel_w
        partial = {
            "uid": next_uid,
            "virtual": True,
            "data_type": (
                "float16"
                if is_exact_wgrad_1x1 and tensors[1]["data_type"] == "float16"
                else "float32"
            ),
            "dimensions": [num_splits, c_out, cik],
            "strides": [c_out * cik, cik, 1],
        }
        tensor_registry[next_uid] = partial
        next_uid += 1
        pipeline_parameters = {
            **parameters,
            "_wgrad_num_splits": num_splits,
            "_wgrad_kernel_h": kernel_h,
            "_wgrad_kernel_w": kernel_w,
            "_wgrad_pipeline_algorithm": pipeline_algorithm,
        }
        result.append(
            {
                "source_node_ids": group["source_node_ids"],
                "operation": group["operation"],
                "parameters": {
                    **pipeline_parameters,
                    "_wgrad_pipeline_stage": "split",
                },
                "tensors": [image, loss, partial],
                "input_uids": [image["uid"], loss["uid"]],
                "output_uids": [partial["uid"]],
            }
        )
        result.append(
            {
                "source_node_ids": group["source_node_ids"],
                "operation": group["operation"],
                "parameters": {
                    **pipeline_parameters,
                    "_wgrad_pipeline_stage": "reduce",
                },
                "tensors": [partial, output],
                "input_uids": [partial["uid"]],
                "output_uids": group["output_uids"],
            }
        )
    return result


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _same_tensor_metadata(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left.get("alignment", 16) == right.get("alignment", 16) and all(
        left[field] == right[field]
        for field in ("virtual", "data_type", "dimensions", "strides")
    )


def _compiler_entry_path() -> Path:
    import flagdnn_codegen

    return Path(flagdnn_codegen.__file__).resolve().with_name("main.py")


def compiler_identity(
    target_name: str,
    execution_engine: str = "external_artifact",
) -> dict[str, Any]:
    return build_compiler_identity(
        target_name,
        execution_engine,
        provider_path=Path(__file__),
        compiler_entry=_compiler_entry_path(),
        provider_name=PROVIDER_NAME,
        provider_version=PROVIDER_VERSION,
        graph_schema_version=SCHEMA_VERSION,
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        execution_program_version=EXECUTION_PROGRAM_VERSION,
    )


def _canonical_tuning_value(value: Any, context: str) -> str:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{context} must contain JSON-compatible finite values"
        ) from error


def _flatten_tuning_param_map(
    value: object, path: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], str]]:
    if not isinstance(value, dict) or not value:
        raise ValueError("tuning param_map must be a nonempty mapping")
    result: list[tuple[tuple[str, ...], str]] = []
    for output_name, source in value.items():
        if not isinstance(output_name, str) or not output_name:
            raise ValueError("tuning param_map output names must be nonempty")
        output_path = (*path, output_name)
        if isinstance(source, dict):
            result.extend(_flatten_tuning_param_map(source, output_path))
        elif isinstance(source, str) and source:
            result.append((output_path, source))
        else:
            raise ValueError(
                "tuning param_map leaves must name a source parameter"
            )
    return result


def _assign_tuning_value(
    configuration: dict[str, Any],
    path: tuple[str, ...],
    value: Any,
) -> None:
    current = configuration
    for name in path[:-1]:
        existing = current.get(name)
        if existing is None:
            nested: dict[str, Any] = {}
            current[name] = nested
            current = nested
        elif isinstance(existing, dict):
            current = existing
        else:
            raise ValueError(
                f"tuning output path collides at {'.'.join(path)}"
            )
    if path[-1] in current:
        raise ValueError(f"duplicate tuning output: {'.'.join(path)}")
    current[path[-1]] = copy.deepcopy(value)


def _tuning_parameter_values(
    generated: dict[str, Any], source_name: str
) -> list[Any]:
    if source_name not in generated:
        raise ValueError(
            f"tuning param_map references missing parameter: {source_name}"
        )
    source = generated[source_name]
    values = source if isinstance(source, list) else [source]
    if not values:
        raise ValueError(f"empty tuning parameter list: {source_name}")
    unique: list[Any] = []
    seen: set[str] = set()
    for value in values:
        canonical = _canonical_tuning_value(
            value, f"tuning parameter {source_name}"
        )
        if canonical not in seen:
            seen.add(canonical)
            unique.append(value)
    return unique


def _expand_generated_tuning_entry(
    generated: dict[str, Any]
) -> list[dict[str, Any]]:
    leaves = _flatten_tuning_param_map(generated.get("param_map"))
    output_paths: set[tuple[str, ...]] = set()
    source_names: list[str] = []
    for output_path, source_name in leaves:
        if output_path in output_paths:
            raise ValueError(
                f"duplicate tuning output: {'.'.join(output_path)}"
            )
        output_paths.add(output_path)
        if source_name not in source_names:
            source_names.append(source_name)

    dimensions = [
        _tuning_parameter_values(generated, source_name)
        for source_name in source_names
    ]
    source_set = set(source_names)
    literal_fields = {
        name: value
        for name, value in generated.items()
        if name not in {"gen", "param_map"} and name not in source_set
    }

    configurations: list[dict[str, Any]] = []
    for combination in itertools.product(*dimensions):
        selected = dict(zip(source_names, combination, strict=True))
        configuration: dict[str, Any] = {}
        for output_path, source_name in leaves:
            _assign_tuning_value(
                configuration, output_path, selected[source_name]
            )
        for name, value in literal_fields.items():
            _assign_tuning_value(configuration, (name,), value)
        configurations.append(configuration)
    return configurations


def _expand_tuning_table(entries: list[object]) -> list[dict[str, Any]]:
    configurations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry_index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not entry:
            raise ValueError(
                f"tuning entry {entry_index} must be a nonempty mapping"
            )
        if entry.get("gen") is True:
            expanded = _expand_generated_tuning_entry(entry)
        elif "gen" in entry:
            raise ValueError(
                f"tuning entry {entry_index}.gen must be true when present"
            )
        else:
            expanded = [copy.deepcopy(entry)]
        for configuration in expanded:
            canonical = _canonical_tuning_value(
                configuration, f"tuning entry {entry_index}"
            )
            if canonical not in seen:
                seen.add(canonical)
                configurations.append(configuration)
    if not configurations:
        raise ValueError("kernel tuning table produced no configurations")
    return configurations


def _split_tuning_configuration(
    configuration: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_meta = configuration.get("META")
    if not isinstance(raw_meta, dict) or not raw_meta:
        raise ValueError("tuning configuration META must be nonempty")
    meta = dict(raw_meta)
    nested_options: dict[str, Any] = {}
    for option_name in ("num_warps", "num_stages"):
        if option_name not in meta:
            continue
        if option_name in configuration:
            raise ValueError(
                f"tuning configuration defines {option_name} twice"
            )
        nested_options[option_name] = meta.pop(option_name)
    if not meta:
        raise ValueError("tuning configuration META has no kernel constants")
    for name, value in meta.items():
        if not isinstance(name, str) or not name:
            raise ValueError("tuning META names must be nonempty strings")
        if isinstance(value, (dict, list)) or value is None:
            raise ValueError(f"tuning META.{name} must be a scalar")

    options: dict[str, Any] = {
        "num_warps": 4,
        "num_stages": 1,
    }
    # Preserve legacy generated search spaces whose launch options were
    # accidentally nested below META. They are compiler options, not Triton
    # constexpr parameters.
    options.update(nested_options)
    options.update(
        {
            name: value
            for name, value in configuration.items()
            if name != "META"
        }
    )
    for name, value in options.items():
        if not isinstance(name, str) or not name:
            raise ValueError("tuning option names must be nonempty strings")
        if isinstance(value, (dict, list)) or value is None:
            raise ValueError(f"tuning option {name} must be a scalar")
    for required in ("num_warps", "num_stages"):
        value = options.get(required)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"tuning configuration {required} must be positive"
            )
    return meta, options


def _load_tuning_configurations(
    compiler_path: Path,
    candidate: KernelCandidate,
    *,
    table: str | None = None,
) -> tuple[list[dict[str, Any]], str, str]:
    tuning = candidate.tuning
    if tuning is None:
        raise ValueError("kernel candidate has no tuning configuration")

    table_name = tuning.table if table is None else table

    tuning_source = resolve_tuning_source(compiler_path, candidate)
    tuning_bytes = tuning_source.read_bytes()
    document = yaml.safe_load(tuning_bytes)
    if not isinstance(document, dict):
        raise ValueError("kernel tuning source must be a YAML mapping")
    entries = document.get(table_name)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"missing tuning table: {table_name}")

    expanded_configurations = _expand_tuning_table(entries)
    configurations: list[dict[str, Any]] = []
    seen_configurations: set[str] = set()
    for raw_configuration in expanded_configurations:
        meta, options = _split_tuning_configuration(raw_configuration)
        configuration = {"META": meta, **options}
        canonical = _canonical_tuning_value(
            configuration, "normalized tuning configuration"
        )
        if canonical not in seen_configurations:
            seen_configurations.add(canonical)
            configurations.append(configuration)

    selected_tuning_payload = {
        "schema_version": 1,
        "table": table_name,
        "configurations": configurations,
    }
    selected_tuning_bytes = json.dumps(
        selected_tuning_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    tuning_source_hash = hashlib.sha256(selected_tuning_bytes).hexdigest()
    identity_payload = {
        "schema_version": 2,
        "source_sha256": tuning_source_hash,
        "kernel_ownership": candidate.ownership,
        "kernel_backend": candidate.backend,
        "table": table_name,
        "key": tuning.key,
        "strategy": tuning.strategy,
        "configurations": configurations,
    }
    canonical_identity = json.dumps(
        identity_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return (
        configurations,
        tuning_source_hash,
        hashlib.sha256(canonical_identity).hexdigest(),
    )


def _selected_tuning_table(
    candidate: KernelCandidate,
    operation: str,
    function_name: str,
    parameters: dict[str, Any],
) -> str:
    tuning = candidate.tuning
    if tuning is None:
        raise ValueError("kernel candidate has no tuning configuration")
    if function_name == "matmul_p5_split_k_reduce_kernel":
        return "unary"
    if function_name == "_conv_wgrad2d_split_vector_reduce_kernel":
        return "reduction"
    attention_tables = {
        "_sdpa_fwd_kernel": "sdpa",
        "_zero_contiguous_kernel": "sdpa_backward_zero_delta",
        "_sdpa_bwd_dq_dbias_kernel": "sdpa_backward_dq",
        "_sdpa_bwd_dkdv_kernel": "sdpa_backward_dkdv",
        "_sdpa_bwd_dk_kernel": "sdpa_backward_dk",
        "_sdpa_bwd_dv_kernel": "sdpa_backward_dv",
        "_sdpa_fp8_fwd_kernel": "sdpa_fp8",
        "_sdpa_fp8_bwd_dq_kernel": "sdpa_fp8_backward_dq",
        "_sdpa_fp8_bwd_dkdv_kernel": "sdpa_fp8_backward_dkdv",
    }
    attention_table = attention_tables.get(function_name)
    if attention_table is not None and operation in {
        "sdpa",
        "sdpa_backward",
        "sdpa_fp8",
        "sdpa_fp8_backward",
    }:
        return attention_table
    if tuning.strategy != "convolution":
        return tuning.table
    if function_name in {
        "_conv_wgrad2d_p5_pack_image_kernel",
        "_conv_wgrad2d_p5_mm_kernel",
    }:
        return "mm"
    if function_name == "_conv_wgrad2d_batched_tma_kernel":
        if parameters.get("_wgrad_pipeline_algorithm") == "1x1_tma":
            return "conv_wgrad_2d_1x1"
        return "conv_wgrad_2d"

    forward_tables = {
        "conv1d_gemm_kernel": "conv1d_gemm_v3",
        "conv2d_1x1_nchw_pad0_kernel": "conv2d_1x1",
        "conv2d_spatial_nchw_kernel": "conv2d_spatial",
        "conv2d_im2col_nchw_kernel": "unary",
        "conv2d_im2col_nchw_3x3_stride2_pad1_kernel": "unary",
        "conv3d_spatial_ncdhw_m_kernel": "conv_fprop_3d",
        "conv_dgrad2d_1x1_nchw_kernel": "conv_dgrad_2d_1x1",
        "conv_dgrad2d_stride1_kernel": "conv_dgrad_2d_stride1",
        "cast_contiguous_kernel": "batch_norm",
        "conv_dgrad2d_pack_weight_kernel": "batch_norm",
        "zero_contiguous_kernel": "batch_norm",
        "conv_dgrad2d_p5_fp32_tile2w_splitk_kernel": (
            "conv_dgrad_2d_stride2_pad1_3x3_tile2w"
        ),
        "conv_dgrad3d_pack_weight_kernel": "batch_norm",
        "conv_dgrad3d_pad1_3x3_fp32_ci8_dot_kernel": "conv_dgrad_3d",
        "conv_dgrad3d_packed_kernel": "conv_dgrad_3d_packed",
        "_conv_wgrad2d_1x1_direct_nodiv_kernel": "conv_wgrad_2d_1x1",
        "_conv_wgrad2d_1x1_split_nodiv_kernel": "conv_wgrad_2d_1x1",
        "_conv_wgrad2d_1x1_reduce_kernel": "conv_wgrad_2d_1x1",
        "_conv_wgrad2d_3tap_split_kernel": "conv_wgrad_2d",
        "_conv_wgrad2d_stride2_row4_split_kernel": "conv_wgrad_2d",
        "_conv_wgrad2d_reduce_kernel": "conv_wgrad_2d",
        "_conv_wgrad2d_col_split_kernel": "conv_wgrad_2d",
        "_conv_wgrad2d_col_reduce_kernel": "conv_wgrad_2d",
    }
    table = forward_tables.get(function_name)
    if table is not None:
        return table
    if function_name == "conv_dgrad2d_stride2_pad1_3x3_packed_parity_kernel":
        return "conv_dgrad_2d_stride2_pad1_3x3_packed_mci"
    if function_name == "conv_dgrad2d_stride2_pad1_3x3_packed_tile2w_kernel":
        if parameters.get("_dgrad_small_ci") is True:
            return "conv_dgrad_2d_stride2_pad1_3x3_tile4"
        return "conv_dgrad_2d_stride2_pad1_3x3_tile2w"
    if function_name == "conv_dgrad2d_stride2_pad1_3x3_packed_tile4_kernel":
        return "conv_dgrad_2d_stride2_pad1_3x3_tile4"
    spatial_rank = _require_integer(
        parameters, "spatial_rank", minimum=1, maximum=3
    )
    if operation == "convolution_dgrad":
        return f"conv_dgrad_{spatial_rank}d"
    if operation == "convolution_wgrad":
        return f"conv_wgrad_{spatial_rank}d"
    raise ValueError(
        "convolution tuning strategy received an unknown kernel entry"
    )


def _translated_tuning_meta(
    function_name: str, meta: dict[str, Any]
) -> dict[str, Any]:
    result = dict(meta)
    alias: tuple[str, str] | None = None
    if function_name == "conv_dgrad_nd_kernel":
        alias = ("BLOCK_CO", "BLOCK_K")
    elif function_name == "conv_wgrad_nd_kernel":
        alias = ("BLOCK_CO", "BLOCK_OC")
    elif function_name in {
        "_conv_wgrad2d_col_split_kernel",
        "_conv_wgrad2d_col_reduce_kernel",
    }:
        alias = ("BLOCK_CI", "BLOCK_N")
    if alias is not None and alias[0] in result:
        if alias[1] in result:
            raise ValueError(
                f"tuning META defines both {alias[0]} and {alias[1]}"
            )
        result[alias[1]] = result.pop(alias[0])
    if function_name == "conv_dgrad3d_pad1_3x3_fp32_ci8_dot_kernel":
        result.pop("BLOCK_CI", None)
        result.pop("BLOCK_CO", None)
    if function_name in {
        "_conv_wgrad2d_1x1_reduce_kernel",
        "_conv_wgrad2d_reduce_kernel",
        "_conv_wgrad2d_col_reduce_kernel",
    }:
        result.pop("BLOCK_M", None)
    if function_name == "_conv_wgrad2d_stride2_row4_split_kernel":
        result.pop("BLOCK_M", None)
    if function_name in {
        "conv2d_im2col_nchw_kernel",
        "conv2d_im2col_nchw_3x3_stride2_pad1_kernel",
    }:
        result.pop("TILES_PER_PROGRAM", None)
    if function_name == "matmul_p5_split_k_reduce_kernel":
        result.pop("TILES_PER_PROGRAM", None)
    if function_name == "_zero_contiguous_kernel" and "BLOCK_ZERO" in result:
        result["BLOCK"] = result.pop("BLOCK_ZERO")
        result.pop("BLOCK_M", None)
        result.pop("BLOCK_D", None)
    return result


def _positive_tuning_integer(values: dict[str, Any], name: str) -> int:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"tuning META.{name} must be a positive integer")
    return value


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _autotune_variant_grid(
    *,
    strategy: str,
    key: str,
    function_name: str,
    parameters: dict[str, Any],
    constants: dict[str, int | float | str | bool],
    meta: dict[str, Any],
    default_grid: tuple[int, int, int],
) -> tuple[int, int, int]:
    key_value = _require_integer(parameters, key)
    if strategy == "attention":
        if function_name in {"_sdpa_fwd_kernel", "_sdpa_fp8_fwd_kernel"}:
            block_m = _positive_tuning_integer(meta, "BLOCK_M")
            _positive_tuning_integer(meta, "BLOCK_N")
            return (
                _ceil_div(_require_integer(parameters, "sequence_q"), block_m),
                _require_integer(parameters, "batch")
                * _require_integer(parameters, "heads"),
                1,
            )
        if function_name == "_zero_contiguous_kernel":
            block = _positive_tuning_integer(meta, "BLOCK")
            return (
                _ceil_div(
                    _require_integer(parameters, "dbias_elements"), block
                ),
                1,
                1,
            )
        block_m = _positive_tuning_integer(meta, "BLOCK_M")
        block_n = _positive_tuning_integer(meta, "BLOCK_N")
        if function_name == "_sdpa_bwd_dq_dbias_kernel":
            block_d = _positive_tuning_integer(meta, "BLOCK_D_OUT")
            return (
                _ceil_div(_require_integer(parameters, "sequence_q"), block_m),
                _ceil_div(
                    _require_integer(parameters, "head_dimension"), block_d
                ),
                _require_integer(parameters, "batch")
                * _require_integer(parameters, "heads"),
            )
        if function_name in {
            "_sdpa_bwd_dkdv_kernel",
            "_sdpa_bwd_dk_kernel",
        }:
            block_d = _positive_tuning_integer(meta, "BLOCK_D_OUT")
            return (
                _ceil_div(
                    _require_integer(parameters, "sequence_kv"), block_n
                ),
                _ceil_div(
                    _require_integer(parameters, "head_dimension"), block_d
                ),
                _require_integer(parameters, "batch")
                * _require_integer(parameters, "key_heads"),
            )
        if function_name == "_sdpa_bwd_dv_kernel":
            block_dv = _positive_tuning_integer(meta, "BLOCK_DV_OUT")
            return (
                _ceil_div(
                    _require_integer(parameters, "sequence_kv"), block_n
                ),
                _ceil_div(
                    _require_integer(parameters, "value_dimension"), block_dv
                ),
                _require_integer(parameters, "batch")
                * _require_integer(parameters, "value_heads"),
            )
        if function_name == "_sdpa_fp8_bwd_dq_kernel":
            return (
                _ceil_div(_require_integer(parameters, "sequence_q"), block_m),
                _require_integer(parameters, "batch")
                * _require_integer(parameters, "heads"),
                1,
            )
        if function_name == "_sdpa_fp8_bwd_dkdv_kernel":
            return (
                _ceil_div(
                    _require_integer(parameters, "sequence_kv"), block_n
                ),
                _require_integer(parameters, "batch")
                * _require_integer(parameters, "key_heads"),
                1,
            )
        raise ValueError("attention tuning received an unknown kernel entry")
    if function_name == "matmul_p5_split_k_reduce_kernel":
        block_size = _positive_tuning_integer(meta, "BLOCK_SIZE")
        return (
            _ceil_div(
                _positive_tuning_integer(constants, "TOTAL"), block_size
            ),
            1,
            1,
        )
    if function_name == "_conv_wgrad2d_split_vector_reduce_kernel":
        block_m = _positive_tuning_integer(meta, "BLOCK_M")
        _positive_tuning_integer(meta, "BLOCK_N")
        return (
            _ceil_div(_positive_tuning_integer(constants, "TOTAL"), block_m),
            1,
            1,
        )
    if function_name == "zero_contiguous_kernel":
        block_size = _positive_tuning_integer(meta, "BLOCK_SIZE")
        return (
            _ceil_div(
                _positive_tuning_integer(constants, "TOTAL"), block_size
            ),
            1,
            1,
        )
    if function_name == "cast_contiguous_kernel":
        block_size = _positive_tuning_integer(meta, "BLOCK_SIZE")
        return (
            _ceil_div(
                _positive_tuning_integer(constants, "TOTAL"), block_size
            ),
            1,
            1,
        )
    if function_name == "_conv_wgrad2d_p5_pack_image_kernel":
        block_m = _positive_tuning_integer(meta, "BLOCK_M")
        block_n = _positive_tuning_integer(meta, "BLOCK_N")
        _positive_tuning_integer(meta, "BLOCK_K")
        _positive_tuning_integer(meta, "GROUP_M")
        return (
            _ceil_div(_positive_tuning_integer(constants, "M"), block_m),
            _ceil_div(_positive_tuning_integer(constants, "N"), block_n),
            1,
        )
    if function_name == "_conv_wgrad2d_p5_mm_kernel":
        block_m = _positive_tuning_integer(meta, "BLOCK_M")
        block_n = _positive_tuning_integer(meta, "BLOCK_N")
        _positive_tuning_integer(meta, "BLOCK_K")
        _positive_tuning_integer(meta, "GROUP_M")
        return (
            _ceil_div(_positive_tuning_integer(constants, "M"), block_m)
            * _ceil_div(_positive_tuning_integer(constants, "N"), block_n),
            1,
            1,
        )
    if strategy == "align32":
        block_size = _positive_tuning_integer(meta, "BLOCK_SIZE")
        if function_name == "batch_norm_inference_nchw_kernel":
            spatial = _require_integer(parameters, "spatial", minimum=1)
            channels = _require_integer(parameters, "channels", minimum=1)
            elements_per_batch = channels * spatial
            if key_value % elements_per_batch != 0:
                raise ValueError(
                    "BatchNorm Inference elements are not divisible by C*S"
                )
            block_s = min(1 << (spatial - 1).bit_length(), block_size)
            block_c = block_size // block_s
            return (
                (key_value // elements_per_batch)
                * _ceil_div(channels, block_c)
                * _ceil_div(spatial, block_s),
                1,
                1,
            )
        tiles_per_program = meta.get("TILES_PER_PROGRAM", 1)
        if (
            isinstance(tiles_per_program, bool)
            or not isinstance(tiles_per_program, int)
            or tiles_per_program <= 0
        ):
            raise ValueError(
                "tuning META.TILES_PER_PROGRAM must be a positive integer"
            )
        return (
            _ceil_div(key_value, block_size * tiles_per_program),
            1,
            1,
        )

    if strategy == "reduction":
        block_m = _positive_tuning_integer(meta, "BLOCK_M")
        _positive_tuning_integer(meta, "BLOCK_N")
        return (_ceil_div(key_value, block_m), 1, 1)

    if strategy == "fixed_grid":
        return tuple(int(value) for value in default_grid)

    if strategy == "matmul":
        block_m = _positive_tuning_integer(meta, "BLOCK_M")
        block_n = _positive_tuning_integer(meta, "BLOCK_N")
        _positive_tuning_integer(meta, "BLOCK_K")
        _positive_tuning_integer(meta, "GROUP_M")
        m = _positive_tuning_integer(constants, "M")
        n = _positive_tuning_integer(constants, "N")
        if function_name == "matmul_batched_tma_persistent_kernel":
            batch = _positive_tuning_integer(constants, "BATCH")
            persistent_grid = _positive_tuning_integer(
                constants, "PERSISTENT_GRID"
            )
            total_tiles = batch * _ceil_div(m, block_m) * _ceil_div(n, block_n)
            return (min(total_tiles, persistent_grid), 1, 1)
        return (
            _ceil_div(m, block_m) * _ceil_div(n, block_n),
            int(default_grid[1]),
            int(default_grid[2]),
        )

    if strategy != "convolution":
        raise ValueError(f"unknown kernel tuning strategy: {strategy!r}")

    if function_name in {
        "conv2d_im2col_nchw_kernel",
        "conv2d_im2col_nchw_3x3_stride2_pad1_kernel",
    }:
        block_size = _positive_tuning_integer(meta, "BLOCK_SIZE")
        output_area = _positive_tuning_integer(
            constants, "OH"
        ) * _positive_tuning_integer(constants, "OW")
        return (
            _ceil_div(output_area, block_size),
            int(default_grid[1]),
            1,
        )
    if function_name == "conv1d_gemm_kernel":
        block_m = _positive_tuning_integer(meta, "BLOCK_M")
        block_oc = _positive_tuning_integer(meta, "BLOCK_OC")
        return (
            _ceil_div(_positive_tuning_integer(constants, "M"), block_m)
            * _ceil_div(
                _positive_tuning_integer(constants, "COUT_PER_GROUP"),
                block_oc,
            ),
            int(default_grid[1]),
            int(default_grid[2]),
        )
    if function_name in {
        "conv2d_1x1_nchw_pad0_kernel",
        "conv2d_spatial_nchw_kernel",
    }:
        block_hw = _positive_tuning_integer(meta, "BLOCK_HW")
        block_oc = _positive_tuning_integer(meta, "BLOCK_OC")
        output_spatial = (
            _positive_tuning_integer(constants, "HW")
            if function_name == "conv2d_1x1_nchw_pad0_kernel"
            else (
                _positive_tuning_integer(constants, "OH")
                * _positive_tuning_integer(constants, "OW")
            )
        )
        return (
            _ceil_div(output_spatial, block_hw)
            * _ceil_div(
                _positive_tuning_integer(constants, "COUT_PER_GROUP"),
                block_oc,
            ),
            int(default_grid[1]),
            int(default_grid[2]),
        )
    if function_name == "conv3d_spatial_ncdhw_m_kernel":
        block_m = _positive_tuning_integer(meta, "BLOCK_M")
        block_oc = _positive_tuning_integer(meta, "BLOCK_OC")
        return (
            _ceil_div(_positive_tuning_integer(constants, "M"), block_m)
            * _ceil_div(
                _positive_tuning_integer(constants, "COUT_PER_GROUP"),
                block_oc,
            ),
            int(default_grid[1]),
            int(default_grid[2]),
        )
    if function_name == "conv_dgrad2d_1x1_nchw_kernel":
        block_m = _positive_tuning_integer(meta, "BLOCK_M")
        block_ci = _positive_tuning_integer(meta, "BLOCK_CI")
        return (
            _ceil_div(_positive_tuning_integer(constants, "HW"), block_m)
            * _ceil_div(
                _positive_tuning_integer(constants, "CIN_PER_GROUP"),
                block_ci,
            ),
            int(default_grid[1]),
            int(default_grid[2]),
        )
    if function_name in {
        "conv_dgrad2d_stride1_kernel",
        "conv_dgrad_nd_kernel",
        "conv_dgrad3d_packed_kernel",
    }:
        block_m = _positive_tuning_integer(meta, "BLOCK_M")
        block_ci = _positive_tuning_integer(meta, "BLOCK_CI")
        return (
            _ceil_div(_positive_tuning_integer(constants, "M"), block_m)
            * _ceil_div(
                _positive_tuning_integer(constants, "CIN_PER_GROUP"),
                block_ci,
            ),
            int(default_grid[1]),
            int(default_grid[2]),
        )
    if function_name == "conv_dgrad3d_pad1_3x3_fp32_ci8_dot_kernel":
        block_m = _positive_tuning_integer(meta, "BLOCK_M")
        return (
            _ceil_div(_positive_tuning_integer(constants, "M"), block_m),
            int(default_grid[1]),
            int(default_grid[2]),
        )
    if function_name in {
        "conv_dgrad2d_pack_weight_kernel",
        "conv_dgrad3d_pack_weight_kernel",
    }:
        block_size = _positive_tuning_integer(meta, "BLOCK_SIZE")
        return (
            _ceil_div(
                _positive_tuning_integer(constants, "C_OUT")
                * _positive_tuning_integer(constants, "C_IN"),
                block_size,
            ),
            int(default_grid[1]),
            int(default_grid[2]),
        )
    if function_name == "conv_dgrad2d_p5_fp32_tile2w_splitk_kernel":
        block_m = _positive_tuning_integer(meta, "BLOCK_M")
        block_ci = _positive_tuning_integer(meta, "BLOCK_CI")
        block_co = _positive_tuning_integer(meta, "BLOCK_CO")
        group_k = _positive_tuning_integer(constants, "GROUP_K")
        split_k_blocks = _ceil_div(
            _ceil_div(
                _positive_tuning_integer(constants, "COUT_PER_GROUP"),
                block_co,
            ),
            group_k,
        )
        return (
            _ceil_div(_positive_tuning_integer(constants, "M"), block_m)
            * _ceil_div(
                _positive_tuning_integer(constants, "CIN_PER_GROUP"),
                block_ci,
            )
            * split_k_blocks,
            1,
            1,
        )
    if function_name in {
        "conv_dgrad2d_stride2_pad1_3x3_packed_parity_kernel",
        "conv_dgrad2d_stride2_pad1_3x3_packed_tile2w_kernel",
        "conv_dgrad2d_stride2_pad1_3x3_packed_tile4_kernel",
    }:
        block_m = _positive_tuning_integer(meta, "BLOCK_M")
        block_ci = _positive_tuning_integer(meta, "BLOCK_CI")
        return (
            _ceil_div(
                _positive_tuning_integer(constants, "M"),
                block_m,
            )
            * _ceil_div(
                _positive_tuning_integer(constants, "CIN_PER_GROUP"),
                block_ci,
            ),
            int(default_grid[1]),
            int(default_grid[2]),
        )
    if function_name == "conv_wgrad_nd_kernel":
        block_oc = _positive_tuning_integer(meta, "BLOCK_OC")
        block_ci = _positive_tuning_integer(meta, "BLOCK_CI")
        return (
            _ceil_div(
                _positive_tuning_integer(constants, "COUT_PER_GROUP"),
                block_oc,
            )
            * _ceil_div(
                _positive_tuning_integer(constants, "CIN_PER_GROUP"),
                block_ci,
            ),
            int(default_grid[1]),
            int(default_grid[2]),
        )
    if function_name == "_conv_wgrad2d_3tap_split_kernel":
        block_co = _positive_tuning_integer(meta, "BLOCK_CO")
        block_ci = _positive_tuning_integer(meta, "BLOCK_CI")
        return (
            _ceil_div(
                _positive_tuning_integer(constants, "COUT_PER_GROUP"),
                block_co,
            )
            * _ceil_div(
                _positive_tuning_integer(constants, "CIN_PER_GROUP"),
                block_ci,
            ),
            int(default_grid[1]),
            int(default_grid[2]),
        )
    if function_name == "_conv_wgrad2d_batched_tma_kernel":
        block_co = _positive_tuning_integer(meta, "BLOCK_CO")
        block_ci = _positive_tuning_integer(meta, "BLOCK_CI")
        return (
            _ceil_div(
                _positive_tuning_integer(constants, "C_OUT"),
                block_co,
            )
            * _ceil_div(
                _positive_tuning_integer(constants, "CIK"),
                block_ci,
            ),
            int(default_grid[1]),
            int(default_grid[2]),
        )
    if function_name in {
        "_conv_wgrad2d_1x1_direct_nodiv_kernel",
        "_conv_wgrad2d_1x1_split_nodiv_kernel",
        "_conv_wgrad2d_1x1_reduce_kernel",
        "_conv_wgrad2d_stride2_row4_split_kernel",
        "_conv_wgrad2d_reduce_kernel",
    }:
        block_co = _positive_tuning_integer(meta, "BLOCK_CO")
        block_ci = _positive_tuning_integer(meta, "BLOCK_CI")
        return (
            _ceil_div(
                _positive_tuning_integer(constants, "COUT_PER_GROUP"),
                block_co,
            )
            * _ceil_div(
                _positive_tuning_integer(constants, "CIN_PER_GROUP"),
                block_ci,
            ),
            int(default_grid[1]),
            int(default_grid[2]),
        )
    if function_name in {
        "_conv_wgrad2d_col_split_kernel",
        "_conv_wgrad2d_col_reduce_kernel",
    }:
        block_co = _positive_tuning_integer(meta, "BLOCK_CO")
        block_n = _positive_tuning_integer(meta, "BLOCK_N")
        cik = (
            _positive_tuning_integer(constants, "CIN_PER_GROUP")
            * _positive_tuning_integer(constants, "KH")
            * _positive_tuning_integer(constants, "KW")
        )
        return (
            _ceil_div(
                _positive_tuning_integer(constants, "COUT_PER_GROUP"),
                block_co,
            )
            * _ceil_div(cik, block_n),
            int(default_grid[1]),
            int(default_grid[2]),
        )
    raise ValueError(
        "convolution tuning strategy received an unknown kernel entry"
    )


def _prepare_tuning_variants(
    *,
    compiler_path: Path,
    candidate: KernelCandidate,
    operation: str,
    function_name: str,
    parameters: dict[str, Any],
    constants: dict[str, int | float | str | bool],
    default_grid: tuple[int, int, int],
) -> tuple[
    list[
        tuple[
            dict[str, Any],
            dict[str, int | float | str | bool],
            dict[str, Any],
            tuple[int, int, int],
        ]
    ],
    str,
    str,
    str,
]:
    tuning = candidate.tuning
    if tuning is None:
        raise ValueError("kernel candidate has no tuning configuration")

    selected_table = _selected_tuning_table(
        candidate, operation, function_name, parameters
    )
    configurations, tuning_source_hash, _ = _load_tuning_configurations(
        compiler_path, candidate, table=selected_table
    )
    if function_name == "conv2d_im2col_nchw_kernel":
        for block_size in (512, 1024):
            configurations.append(
                {
                    "META": {"BLOCK_SIZE": block_size},
                    "num_warps": 2,
                    "num_stages": 1,
                }
            )
    if function_name == "matmul_batched_broadcast_a_kernel":
        for block_m, block_n, block_k, num_stages in (
            (64, 64, 32, 4),
            (64, 64, 64, 5),
            (32, 64, 64, 4),
        ):
            configurations.append(
                {
                    "META": {
                        "BLOCK_M": block_m,
                        "BLOCK_N": block_n,
                        "BLOCK_K": block_k,
                        "GROUP_M": 8,
                    },
                    "num_warps": 4,
                    "num_stages": num_stages,
                }
            )
    if (
        function_name == "matmul_strided_kernel"
        and parameters.get("_fprop_im2col_matmul") is True
    ):
        for num_warps in (4, 8):
            for num_stages in (3, 4, 5):
                configurations.append(
                    {
                        "META": {
                            "BLOCK_M": 64,
                            "BLOCK_N": 32,
                            "BLOCK_K": 64,
                            "GROUP_M": 8,
                        },
                        "num_warps": num_warps,
                        "num_stages": num_stages,
                    }
                )
    if function_name == "_conv_wgrad2d_p5_pack_image_kernel":
        for block_m, block_n in (
            (32, 32),
            (16, 64),
            (32, 64),
            (64, 32),
            (16, 128),
            (32, 128),
            (16, 256),
        ):
            configurations.append(
                {
                    "META": {
                        "BLOCK_M": block_m,
                        "BLOCK_N": block_n,
                        "BLOCK_K": 32,
                        "GROUP_M": 8,
                    },
                    "num_warps": 4,
                    "num_stages": 1,
                }
            )
    if (
        function_name == "matmul_batched_contiguous_kernel"
        and constants.get("INPUT_IS_FLOAT32") is True
    ):
        for (
            block_m,
            block_n,
            block_k,
            group_m,
            num_warps,
            num_stages,
        ) in (
            (128, 256, 32, 4, 8, 3),
            (256, 128, 32, 4, 8, 3),
        ):
            configurations.append(
                {
                    "META": {
                        "BLOCK_M": block_m,
                        "BLOCK_N": block_n,
                        "BLOCK_K": block_k,
                        "GROUP_M": group_m,
                    },
                    "num_warps": num_warps,
                    "num_stages": num_stages,
                }
            )
    if function_name == "matmul_batched_tma_persistent_kernel":
        configurations.append(
            {
                "META": {
                    "PERSISTENT_GRID": 132,
                    "BLOCK_M": 128,
                    "BLOCK_N": 256,
                    "BLOCK_K": 64,
                    "GROUP_M": 16,
                },
                "num_warps": 8,
                "num_stages": 4,
            }
        )
    if function_name == "conv_dgrad2d_p5_fp32_tile2w_splitk_kernel":
        configurations.append(
            {
                "META": {
                    "BLOCK_M": 32,
                    "BLOCK_CI": 64,
                    "BLOCK_CO": 64,
                },
                "num_warps": 4,
                "num_stages": 3,
            }
        )
    if function_name == "conv_dgrad2d_1x1_nchw_kernel":
        for block_m, block_ci, block_co, num_warps, num_stages in (
            (32, 32, 128, 4, 3),
            (64, 32, 64, 4, 3),
            (64, 64, 64, 4, 3),
            (32, 64, 64, 4, 2),
            (16, 64, 128, 4, 2),
            (16, 64, 128, 8, 2),
            (32, 64, 128, 4, 1),
            (32, 64, 128, 4, 2),
            (32, 64, 128, 4, 3),
            (32, 64, 128, 8, 1),
            (32, 64, 128, 8, 2),
            (32, 64, 128, 8, 3),
            (32, 128, 128, 8, 2),
            (32, 128, 128, 8, 3),
            (64, 64, 128, 8, 2),
            (128, 32, 64, 8, 2),
            (128, 32, 128, 8, 2),
            (128, 32, 64, 8, 3),
            (128, 32, 128, 8, 3),
        ):
            configurations.append(
                {
                    "META": {
                        "BLOCK_M": block_m,
                        "BLOCK_CI": block_ci,
                        "BLOCK_CO": block_co,
                    },
                    "num_warps": num_warps,
                    "num_stages": num_stages,
                }
            )
    if (
        function_name == "conv_dgrad2d_stride2_pad1_3x3_packed_parity_kernel"
        and parameters.get("_dgrad_p5_parity") is True
    ):
        for block_co in (128, 256):
            configurations.append(
                {
                    "META": {
                        "BLOCK_M": 64,
                        "BLOCK_CI": 64,
                        "BLOCK_CO": block_co,
                    },
                    "num_warps": 8,
                    "num_stages": 2,
                }
            )
    if (
        function_name == "conv_wgrad_nd_kernel"
        and parameters.get("spatial_rank") == 3
    ):
        configurations.append(
            {
                "META": {
                    "BLOCK_CO": 8,
                    "BLOCK_CI": 8,
                    "BLOCK_M": 128,
                },
                "num_warps": 4,
                "num_stages": 1,
            }
        )
        for block_m in (128, 256):
            for num_warps in (4, 8):
                for num_stages in (2, 3):
                    configurations.append(
                        {
                            "META": {
                                "BLOCK_CO": 8,
                                "BLOCK_CI": 8,
                                "BLOCK_M": block_m,
                            },
                            "num_warps": num_warps,
                            "num_stages": num_stages,
                        }
                    )
    key_value = _require_integer(parameters, tuning.key)
    variants: list[
        tuple[
            dict[str, Any],
            dict[str, int | float | str | bool],
            dict[str, Any],
            tuple[int, int, int],
        ]
    ] = []
    seen_configurations: set[str] = set()
    for configuration in configurations:
        raw_meta, compile_options = _split_tuning_configuration(configuration)
        if function_name in {
            "_conv_wgrad2d_1x1_reduce_kernel",
            "_conv_wgrad2d_reduce_kernel",
            "_conv_wgrad2d_col_reduce_kernel",
        }:
            compile_options = {**compile_options, "num_stages": 1}
        elif function_name == "_conv_wgrad2d_stride2_row4_split_kernel":
            compile_options = {**compile_options, "num_stages": 2}
        variant_meta = _translated_tuning_meta(function_name, raw_meta)
        unknown_meta = set(variant_meta).difference(constants)
        if unknown_meta:
            names = ", ".join(sorted(unknown_meta))
            raise ValueError(f"tuning META is not a kernel constexpr: {names}")

        normalized_configuration = {
            "META": variant_meta,
            **compile_options,
        }
        canonical = _canonical_tuning_value(
            normalized_configuration,
            "prepared tuning configuration",
        )
        if canonical in seen_configurations:
            continue
        seen_configurations.add(canonical)

        variant_constants = dict(constants)
        variant_constants.update(variant_meta)
        variant_grid = _autotune_variant_grid(
            strategy=tuning.strategy,
            key=tuning.key,
            function_name=function_name,
            parameters=parameters,
            constants=variant_constants,
            meta=variant_meta,
            default_grid=default_grid,
        )
        variants.append(
            (
                normalized_configuration,
                variant_constants,
                compile_options,
                variant_grid,
            )
        )

    if len(variants) < 1 or len(variants) > 1024:
        raise ValueError(
            "kernel tuning plan must contain between 1 and 1024 "
            f"valid candidates; got {len(variants)}"
        )

    identity_payload = {
        "schema_version": 3,
        "source_sha256": tuning_source_hash,
        "kernel_ownership": candidate.ownership,
        "kernel_backend": candidate.backend,
        "kernel_provider": candidate.provider,
        "operation": operation,
        "function": function_name,
        "table": selected_table,
        "key": tuning.key,
        "key_value": key_value,
        "strategy": tuning.strategy,
        "constants": constants,
        "default_grid": list(default_grid),
        "configurations": [variant[0] for variant in variants],
    }
    canonical_identity = json.dumps(
        identity_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return (
        variants,
        tuning_source_hash,
        hashlib.sha256(canonical_identity).hexdigest(),
        selected_table,
    )


def _libtriton_jit_runtime_signature(
    function: Any,
    runtime_signature: dict[str, str],
    argument_abi: list[dict[str, Any]],
) -> dict[str, str]:
    if (
        len(argument_abi) < 2
        or argument_abi[-2].get("kind") != "global_scratch_pointer"
        or argument_abi[-1].get("kind") != "profile_scratch_pointer"
    ):
        raise ValueError(
            "libtriton_jit argument ABI has invalid scratch slots"
        )

    runtime_names = [
        name for name in function.arg_names if name in runtime_signature
    ]
    runtime_arguments = argument_abi[:-2]
    if len(runtime_names) != len(runtime_arguments):
        raise ValueError(
            "libtriton_jit runtime signature and argument ABI disagree"
        )

    result = dict(runtime_signature)
    for name, argument in zip(runtime_names, runtime_arguments):
        kind = argument.get("kind")
        if kind not in {"tensor", "workspace_tensor"}:
            continue
        alignment = (
            16 if kind == "workspace_tensor" else argument.get("alignment", 1)
        )
        if alignment < 16:
            continue
        token = result[name]
        if not token.startswith("*") or ":" in token:
            raise ValueError(
                "libtriton_jit workspace argument is not a plain pointer"
            )
        # The execution engine enforces each external tensor's declared
        # alignment and the packed workspace's fixed 16-byte alignment.
        result[name] = f"{token}:16"
    return result


def _libtriton_jit_full_signature(
    function: Any,
    runtime_signature: dict[str, str],
    constants: dict[str, int | float | str | bool],
) -> str:
    argument_names = list(function.arg_names)
    if set(runtime_signature).union(constants) != set(argument_names):
        raise ValueError(
            "libtriton_jit signature does not cover every kernel argument"
        )

    tokens: list[str] = []
    for name in argument_names:
        if name in runtime_signature:
            token = runtime_signature[name]
            if not isinstance(token, str) or not token or "," in token:
                raise ValueError("libtriton_jit runtime signature is invalid")
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
            raise ValueError("libtriton_jit supports only numeric constexprs")
    return ",".join(tokens)


def _libtriton_jit_launch(
    grid: tuple[int, int, int] | list[int],
    num_warps: int,
    warp_size: int,
) -> dict[str, Any]:
    return {
        "grid": [int(grid[0]), int(grid[1]), int(grid[2])],
        "block": [num_warps * warp_size, 1, 1],
        "cluster": [1, 1, 1],
        "shared_memory": 0,
        "num_ctas": 1,
        "global_scratch_size": LIBTRITON_JIT_GLOBAL_SCRATCH_SIZE,
        "profile_scratch_size": 0,
    }


def _compile_graph_operation(
    *,
    stage_id: int,
    source_node_ids: list[int],
    dependencies: list[int],
    operation: str,
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
    workspace_layout: dict[int, tuple[int, int]],
    compiler_path: Path,
    output_directory: Path,
    target: GPUTarget,
    enable_autotune: bool,
    execution_engine: str = "external_artifact",
) -> dict[str, Any]:
    candidate = select_kernel_candidate("nvidia", operation)
    kernel_source_path = resolve_kernel_source(compiler_path, candidate)
    kernel_source_bytes = kernel_source_path.read_bytes()
    generated_source_bytes = materialize_kernel_source(
        kernel_source_path, candidate
    )
    if (
        not kernel_source_bytes
        or not generated_source_bytes
        or len(kernel_source_bytes) > (1 << 20)
        or len(generated_source_bytes) > (1 << 20)
    ):
        raise ValueError("kernel source size is invalid")

    generated_source = output_directory / f"generated_stage_{stage_id}.py"
    _atomic_write(generated_source, generated_source_bytes)
    module = _load_generated_module(generated_source, stage_id)
    function_name, signature, constants, grid, argument_layout = (
        _kernel_configuration(operation, parameters, tensors, int(target.arch))
    )
    if function_name not in candidate.functions:
        raise RuntimeError("kernel registry and configuration disagree")
    argument_abi = _build_argument_abi(
        argument_layout, tensors, parameters, workspace_layout
    )

    function = getattr(module, function_name)
    function.create_binder()
    full_signature = dict(signature)
    for name in constants:
        full_signature[name] = "constexpr"
    tuning_plan = None
    fixed_tuning_configuration = None
    default_compile_options: dict[str, Any] = {
        "num_warps": 4,
        "num_stages": 1,
    }
    fixed_initialization_kernels = {
        "_zero_sdpa_fp8_fwd_amax_kernel",
        "_zero_sdpa_fp8_bwd_amax_kernel",
    }
    if (
        enable_autotune
        and candidate.tuning is not None
        and function_name not in fixed_initialization_kernels
    ):
        tuning_plan = _prepare_tuning_variants(
            compiler_path=compiler_path,
            candidate=candidate,
            operation=operation,
            function_name=function_name,
            parameters=parameters,
            constants=constants,
            default_grid=grid,
        )
        if len(tuning_plan[0]) == 1:
            (
                configurations,
                tuning_source_hash,
                candidate_identity,
                selected_table,
            ) = tuning_plan
            (
                selected_configuration,
                constants,
                default_compile_options,
                grid,
            ) = configurations[0]
            tuning = candidate.tuning
            fixed_tuning_configuration = {
                "schema_version": 1,
                "mode": "fixed",
                "source": tuning.source,
                "source_sha256": tuning_source_hash,
                "table": selected_table,
                "key": tuning.key,
                "strategy": tuning.strategy,
                "candidate_identity": candidate_identity,
                "config": selected_configuration,
            }
            tuning_plan = None

    if execution_engine == "libtriton_jit":
        kernel_source_hash = hashlib.sha256(kernel_source_bytes).hexdigest()
        jit_runtime_signature = _libtriton_jit_runtime_signature(
            function, signature, argument_abi
        )
        generated_source_hash = hashlib.sha256(
            generated_source_bytes
        ).hexdigest()
        kernel_metadata = {
            "provider": candidate.provider,
            "ownership": candidate.ownership,
            "source": candidate.source,
            "function": function_name,
            "materialized_source": {
                "file": generated_source.name,
                "size": len(generated_source_bytes),
                "sha256": generated_source_hash,
            },
        }

        def make_jit_variant(
            variant_id: str,
            variant_constants: dict[str, int | float | str | bool],
            compile_options: dict[str, Any],
            variant_grid: tuple[int, int, int] | list[int],
            configuration: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            unsupported_options = set(compile_options).difference(
                {"num_warps", "num_stages"}
            )
            if unsupported_options:
                names = ", ".join(sorted(unsupported_options))
                raise ValueError(
                    "installed libtriton_jit raw API cannot consume "
                    f"compile options: {names}"
                )
            num_warps = compile_options.get("num_warps")
            num_stages = compile_options.get("num_stages")
            if (
                isinstance(num_warps, bool)
                or not isinstance(num_warps, int)
                or num_warps <= 0
                or isinstance(num_stages, bool)
                or not isinstance(num_stages, int)
                or num_stages <= 0
            ):
                raise ValueError(
                    "libtriton_jit compile options must be positive integers"
                )
            result: dict[str, Any] = {
                "variant_id": variant_id,
                "source_sha256": kernel_source_hash,
                "full_signature": _libtriton_jit_full_signature(
                    function, jit_runtime_signature, variant_constants
                ),
                "compile_options": {
                    "num_warps": num_warps,
                    "num_stages": num_stages,
                },
                "argument_abi": argument_abi,
                "launch": _libtriton_jit_launch(
                    variant_grid,
                    num_warps,
                    int(target.warp_size),
                ),
            }
            if configuration is not None:
                result["config"] = configuration
            return result

        stage: dict[str, Any] = {
            "stage_id": stage_id,
            "kind": "kernel",
            "engine": "libtriton_jit",
            "source_node_ids": source_node_ids,
            "dependencies": dependencies,
            "operation": operation,
            "source_sha256": kernel_source_hash,
            "kernel": kernel_metadata,
        }
        if tuning_plan is not None:
            tuning = candidate.tuning
            if tuning is None:
                raise RuntimeError("autotune plan lost its tuning metadata")
            (
                prepared_variants,
                tuning_source_hash,
                candidate_identity,
                selected_table,
            ) = tuning_plan
            variants: list[dict[str, Any]] = []
            for variant_index, (
                configuration,
                variant_constants,
                compile_options,
                variant_grid,
            ) in enumerate(prepared_variants):
                variants.append(
                    make_jit_variant(
                        f"config_{variant_index}",
                        variant_constants,
                        compile_options,
                        variant_grid,
                        configuration,
                    )
                )
            jit_identity_payload = {
                "schema_version": 1,
                "engine": "libtriton_jit",
                "base_candidate_identity": candidate_identity,
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
            candidate_identity = hashlib.sha256(
                json.dumps(
                    jit_identity_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            stage["variants"] = variants
            stage["tuning"] = {
                "schema_version": 1,
                "source": tuning.source,
                "source_sha256": tuning_source_hash,
                "table": selected_table,
                "key": tuning.key,
                "strategy": tuning.strategy,
                "warmup": tuning.warmup,
                "repetitions": tuning.repetitions,
                "candidate_identity": candidate_identity,
            }
        else:
            stage.update(
                make_jit_variant(
                    "default",
                    constants,
                    default_compile_options,
                    grid,
                )
            )
            if fixed_tuning_configuration is not None:
                stage["tuning_configuration"] = fixed_tuning_configuration
        return stage

    if tuning_plan is not None:
        tuning = candidate.tuning
        if tuning is None:
            raise RuntimeError("autotune plan lost its tuning metadata")
        (
            prepared_variants,
            tuning_source_hash,
            candidate_identity,
            selected_table,
        ) = tuning_plan
        kernel_source_hash = hashlib.sha256(kernel_source_bytes).hexdigest()
        variants: list[dict[str, Any]] = []
        for variant_index, (
            configuration,
            variant_constants,
            compile_options,
            variant_grid,
        ) in enumerate(prepared_variants):
            variant_source = function.ASTSource(
                fn=function,
                signature=full_signature,
                constexprs=variant_constants,
                attrs={},
            )
            compiled_variant = triton.compile(
                variant_source,
                target=target,
                options=compile_options,
            )
            binary_bytes = bytes(compiled_variant.asm["cubin"])
            binary_name = f"stage_{stage_id}_variant_{variant_index}.cubin"
            _atomic_write(output_directory / binary_name, binary_bytes)
            metadata = compiled_variant.metadata
            cluster = tuple(getattr(metadata, "cluster_dims", (1, 1, 1)))
            variants.append(
                {
                    "variant_id": f"config_{variant_index}",
                    "config": configuration,
                    "source_sha256": kernel_source_hash,
                    "binary": {
                        "file": binary_name,
                        "size": len(binary_bytes),
                        "sha256": hashlib.sha256(binary_bytes).hexdigest(),
                    },
                    "entry_symbol": compiled_variant.name,
                    "argument_abi": argument_abi,
                    "launch": {
                        "grid": list(variant_grid),
                        "block": [
                            int(metadata.num_warps) * int(target.warp_size),
                            1,
                            1,
                        ],
                        "cluster": [
                            int(cluster[0]),
                            int(cluster[1]),
                            int(cluster[2]),
                        ],
                        "shared_memory": int(metadata.shared),
                        "num_ctas": int(getattr(metadata, "num_ctas", 1)),
                        "global_scratch_size": int(
                            getattr(metadata, "global_scratch_size", 0)
                        ),
                        "profile_scratch_size": int(
                            getattr(metadata, "profile_scratch_size", 0)
                        ),
                    },
                }
            )

        return {
            "stage_id": stage_id,
            "kind": "kernel",
            "engine": "external_artifact",
            "source_node_ids": source_node_ids,
            "dependencies": dependencies,
            "operation": operation,
            "source_sha256": kernel_source_hash,
            "kernel": {
                "provider": candidate.provider,
                "ownership": candidate.ownership,
                "source": candidate.source,
                "function": function_name,
            },
            "variants": variants,
            "tuning": {
                "schema_version": 1,
                "source": tuning.source,
                "source_sha256": tuning_source_hash,
                "table": selected_table,
                "key": tuning.key,
                "strategy": tuning.strategy,
                "warmup": tuning.warmup,
                "repetitions": tuning.repetitions,
                "candidate_identity": candidate_identity,
            },
        }
    source = function.ASTSource(
        fn=function,
        signature=full_signature,
        constexprs=constants,
        attrs={},
    )
    compiled = triton.compile(
        source,
        target=target,
        options=default_compile_options,
    )

    binary_bytes = bytes(compiled.asm["cubin"])
    binary_name = f"stage_{stage_id}.cubin"
    _atomic_write(output_directory / binary_name, binary_bytes)
    metadata = compiled.metadata
    cluster = tuple(getattr(metadata, "cluster_dims", (1, 1, 1)))
    stage = {
        "stage_id": stage_id,
        "kind": "kernel",
        "engine": "external_artifact",
        "source_node_ids": source_node_ids,
        "dependencies": dependencies,
        "operation": operation,
        "source_sha256": hashlib.sha256(kernel_source_bytes).hexdigest(),
        "kernel": {
            "provider": candidate.provider,
            "ownership": candidate.ownership,
            "source": candidate.source,
            "function": function_name,
        },
        "binary": {
            "file": binary_name,
            "size": len(binary_bytes),
            "sha256": hashlib.sha256(binary_bytes).hexdigest(),
        },
        "entry_symbol": compiled.name,
        "argument_abi": argument_abi,
        "launch": {
            "grid": list(grid),
            "block": [
                int(metadata.num_warps) * int(target.warp_size),
                1,
                1,
            ],
            "cluster": [
                int(cluster[0]),
                int(cluster[1]),
                int(cluster[2]),
            ],
            "shared_memory": int(metadata.shared),
            "num_ctas": int(getattr(metadata, "num_ctas", 1)),
            "global_scratch_size": int(
                getattr(metadata, "global_scratch_size", 0)
            ),
            "profile_scratch_size": int(
                getattr(metadata, "profile_scratch_size", 0)
            ),
        },
    }
    if fixed_tuning_configuration is not None:
        stage["tuning_configuration"] = fixed_tuning_configuration
    return stage


def compile_request(
    request_path: Path,
    output_directory: Path,
    execution_engine: str = "external_artifact",
) -> dict[str, Any]:
    if execution_engine not in {"external_artifact", "libtriton_jit"}:
        raise ValueError("NVIDIA execution engine is invalid")
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
    if request.get("backend") != "nvidia":
        raise ValueError("NVIDIA provider received another backend")

    target_name = request.get("target")
    if not isinstance(target_name, str) or not target_name.startswith("sm_"):
        raise ValueError("NVIDIA target fingerprint is invalid")
    architecture_text = target_name[3:]
    if not architecture_text.isdigit():
        raise ValueError("NVIDIA target fingerprint is invalid")
    architecture = int(architecture_text)
    if architecture < 50 or architecture > 999:
        raise ValueError("NVIDIA SM architecture is invalid")

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
        or node_count < 1
        or node_count > 1024
    ):
        raise ValueError("graph node_count is invalid")

    parsed_nodes: list[
        tuple[
            int,
            str,
            dict[str, Any],
            list[dict[str, Any]],
            list[int],
            list[int],
        ]
    ] = []
    node_positions: dict[int, int] = {}
    producer_nodes: dict[int, int] = {}
    has_external_output = False
    for node_position, node_value in enumerate(nodes):
        graph_node = _require_object(
            node_value, f"graph.nodes[{node_position}]"
        )
        node_id = graph_node.get("id")
        if (
            isinstance(node_id, bool)
            or not isinstance(node_id, int)
            or node_id < 0
            or node_id in node_positions
        ):
            raise ValueError("graph node ID is invalid or duplicated")
        node_positions[node_id] = node_position
        operation = graph_node.get("type")
        if not isinstance(operation, str):
            raise ValueError("graph node type must be a string")
        attributes = _require_object(
            graph_node.get("attributes"),
            f"graph.nodes[{node_position}].attributes",
        )
        tensor_uids, tensor_metadata, input_count = _tensor_metadata(
            graph_node, operation, tensor_registry
        )
        input_uids = tensor_uids[:input_count]
        output_uids = tensor_uids[input_count:]
        for uid in output_uids:
            if uid in producer_nodes:
                raise ValueError("graph tensor has more than one producer")
            producer_nodes[uid] = node_id
            if not tensor_registry[uid]["virtual"]:
                has_external_output = True
        parsed_nodes.append(
            (
                node_id,
                operation,
                attributes,
                tensor_metadata,
                input_uids,
                output_uids,
            )
        )

    for node_id, _, _, _, input_uids, _ in parsed_nodes:
        node_position = node_positions[node_id]
        for uid in input_uids:
            producer = producer_nodes.get(uid)
            if tensor_registry[uid]["virtual"] and producer is None:
                raise ValueError("virtual tensor input has no producer")
            if (
                producer is not None
                and node_positions[producer] >= node_position
            ):
                raise ValueError(
                    "graph nodes are not in topological execution order"
                )

    if not any(not tensor["virtual"] for tensor in tensor_registry.values()):
        raise ValueError("graph has no externally bound tensors")
    if not has_external_output:
        raise ValueError("graph has no non-virtual output tensor")

    effective_execution_engine = execution_engine

    execution_groups = _expand_execution_pipelines(
        _lower_execution_groups(parsed_nodes, tensor_registry),
        tensor_registry,
    )
    workspace_layout, workspace_size = _workspace_layout(tensor_registry)
    if effective_execution_engine == "libtriton_jit":
        workspace_size += LIBTRITON_JIT_GLOBAL_SCRATCH_SIZE
    output_directory.mkdir(parents=True, exist_ok=True)
    target = GPUTarget("cuda", architecture, 32)
    tensor_to_stage: dict[int, int] = {}
    stages: list[dict[str, Any]] = []
    for stage_id, execution_group in enumerate(execution_groups):
        source_node_ids = execution_group["source_node_ids"]
        operation = execution_group["operation"]
        attributes = execution_group["parameters"]
        tensors = execution_group["tensors"]
        input_uids = execution_group["input_uids"]
        dependencies = sorted(
            {
                tensor_to_stage[uid]
                for uid in input_uids
                if uid in tensor_to_stage
            }
        )
        stage = _compile_graph_operation(
            stage_id=stage_id,
            source_node_ids=source_node_ids,
            dependencies=dependencies,
            operation=operation,
            parameters=attributes,
            tensors=tensors,
            workspace_layout=workspace_layout,
            compiler_path=_compiler_entry_path(),
            output_directory=output_directory,
            target=target,
            enable_autotune=enable_autotune,
            execution_engine=effective_execution_engine,
        )
        stages.append(stage)
        for uid in execution_group["output_uids"]:
            tensor_to_stage[uid] = stage_id

    source_hashes = [stage["source_sha256"] for stage in stages]
    combined_source_hash = hashlib.sha256(
        json.dumps(source_hashes, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    manifest: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "flagdnn_execution_program",
        "flagdnn_version": flagdnn_version,
        "backend": "nvidia",
        "target": target_name,
        "graph_node_count": node_count,
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "source_sha256": combined_source_hash,
        "compiler": {
            **identity,
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "torch_loaded": "torch" in sys.modules,
        },
        "workspace_size": workspace_size,
        "program": {
            "schema_version": EXECUTION_PROGRAM_VERSION,
            "stage_count": len(stages),
            "stages": stages,
        },
    }
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    _atomic_write(output_directory / "manifest.json", manifest_bytes)
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "success",
        "backend": "nvidia",
        "provider": PROVIDER_NAME,
        "node_count": node_count,
        "stage_count": len(stages),
        "target": target_name,
        "artifact_directory": str(output_directory),
        "workspace_size": workspace_size,
        "execution_engine": effective_execution_engine,
        "triton_version": triton.__version__,
        "torch_loaded": "torch" in sys.modules,
    }
