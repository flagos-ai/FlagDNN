"""Shared compiler contracts, tensor checks, and pointwise layout helpers."""

from __future__ import annotations

from typing import Any
from typing import NamedTuple
from typing import TypedDict
import math

from flagdnn_codegen.kernel_registry import BINARY_POINTWISE_OPERATIONS
from flagdnn_codegen.kernel_registry import TERNARY_POINTWISE_OPERATIONS
from flagdnn_codegen.kernel_registry import UNARY_POINTWISE_OPERATIONS

SCHEMA_VERSION = 3


ARTIFACT_SCHEMA_VERSION = 4


EXECUTION_PROGRAM_VERSION = 2


PROVIDER_NAME = "nvidia_triton"


PROVIDER_VERSION = "1"


LIBTRITON_JIT_GLOBAL_SCRATCH_SIZE = 4096


EXPECTED_TENSOR_ROLES = {
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


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _tensor_storage_size(tensor: dict[str, Any]) -> int:
    element_size = {
        "float32": 4,
        "int32": 4,
        "float16": 2,
        "bfloat16": 2,
        "boolean": 1,
        "fp8_e4m3": 1,
        "fp8_e5m2": 1,
        "fp8_e8m0": 1,
    }[tensor["data_type"]]
    storage_elements = 1 + sum(
        (dimension - 1) * stride
        for dimension, stride in zip(tensor["dimensions"], tensor["strides"])
    )
    return storage_elements * element_size


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


class ExecutionGroup(TypedDict):
    source_node_ids: list[int]
    operation: str
    parameters: dict[str, Any]
    tensors: list[dict[str, Any]]
    input_uids: list[int]
    output_uids: list[int]


class KernelPlan(NamedTuple):
    function_name: str
    signature: dict[str, str]
    constants: dict[str, int | float | str | bool]
    grid: tuple[int, int, int]
    argument_layout: list[tuple[str, str | int | None]]
