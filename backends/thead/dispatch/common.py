# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""THead Graph IR schema, operation modes and validation primitives."""

from __future__ import annotations

from typing import Any
import re


SCHEMA_VERSION = 3


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


_DATA_TYPES = {
    "float32",
    "float16",
    "bfloat16",
    "boolean",
    "int32",
    "fp8_e8m0",
    "fp8_e4m3",
    "fp8_e5m2",
}


_DATA_TYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "boolean": 1,
    "int32": 4,
    "fp8_e8m0": 1,
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
    "relu_backward": 42,
    "tanh_backward": 43,
    "elu_backward": 44,
    "gelu_backward": 45,
    "softplus_backward": 46,
    "swish_backward": 47,
    "gelu_approx_tanh_backward": 48,
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
    "erf": 4,
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
    "sdpa",
    "sdpa_backward_dq",
    "sdpa_fp8",
    "sdpa_fp8_backward",
}


_APPROVED_PPU_COMPILER_OPTIONS = {
    "debug",
    "enable_fp_fusion",
    "enable_reflect_ftz",
    "instrumentation_mode",
    "ppu_llc_options",
    "sanitize_overflow",
}


def _require_object(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object")
    return value


def _require_list(value: object, description: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{description} must be an array")
    return value


def _require_exact_fields(
    value: dict[str, Any],
    required: set[str],
    optional: set[str],
    description: str,
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


def _string(
    value: object, description: str, *, allow_empty: bool = False
) -> str:
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
        raise ValueError(
            "heuristic_modes must contain unique A/FALLBACK values"
        )
    if not isinstance(options["autotune"], bool):
        raise ValueError("build_options.autotune must be a boolean")
    return options["autotune"]


def _pointwise_block_size(n_elements: int) -> int:
    # Larger vector tiles amortize program scheduling on bandwidth-bound work.
    return 1024 if n_elements >= 65536 else _FIXED_BLOCK_SIZE


_POINTWISE_ATTRIBUTE_DEFAULTS = {
    "elu_alpha": 1.0,
    "relu_lower_clip": 0.0,
    "relu_lower_clip_slope": 0.0,
    "relu_upper_clip": 0.0,
    "relu_upper_clip_set": False,
    "softplus_beta": 1.0,
    "swish_beta": 1.0,
}


def _validate_pointwise_defaults(attributes: dict[str, Any]) -> None:
    for name, default in _POINTWISE_ATTRIBUTE_DEFAULTS.items():
        value = attributes.get(name, default)
        if isinstance(value, bool) != isinstance(default, bool):
            raise ValueError(f"pointwise {name} has an invalid type")
        if value != default:
            raise ValueError(
                f"pointwise {name} is not applicable to this operation"
            )
