"""MThreads dispatch common implementation."""

from __future__ import annotations

import math
import struct

from .tensor import POINTER_TYPES

_WORKSPACE_SIZE = 4096

_WORKSPACE_ALIGNMENT = 256

_BINARY_SOURCE_RELATIVE_PATH = "kernels/binary.py"

_UNARY_SOURCE_RELATIVE_PATH = "kernels/unary.py"

_IDENTITY_SOURCE_RELATIVE_PATH = "kernels/identity.py"

_TERNARY_SOURCE_RELATIVE_PATH = "kernels/ternary.py"

_LAYOUT_SOURCE_RELATIVE_PATH = "kernels/layout.py"

_REDUCTION_SOURCE_RELATIVE_PATH = "kernels/reduction.py"

_MATMUL_SOURCE_RELATIVE_PATH = "kernels/matmul.py"

_CONVOLUTION_SOURCE_RELATIVE_PATH = "kernels/convolution.py"

_COMPOSITE_SOURCE_RELATIVE_PATH = "kernels/composite.py"

_CONV_BIAS_RELU_SOURCE_RELATIVE_PATH = "kernels/conv_bias_relu.py"

_NORMALIZATION_SOURCE_RELATIVE_PATH = "kernels/normalization.py"

_ATTENTION_SOURCE_RELATIVE_PATH = "kernels/attention.py"

_SELECTION_CACHE = "tuning/stage-0.json"

_MAX_I32 = (1 << 31) - 1


def _scalar_i32_bits(value: int) -> str:
    try:
        return struct.pack("<i", value).hex()
    except struct.error as error:
        raise ValueError("runtime int32 scalar is out of range") from error


def _scalar_f32_bits(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("runtime float32 scalar must be finite")
    try:
        return struct.pack("<f", value).hex()
    except (OverflowError, struct.error) as error:
        raise ValueError("runtime float32 scalar is out of range") from error


def _pointer_token(data_type: str, alignment: int) -> str:
    token = POINTER_TYPES[data_type]
    return f"{token}:16" if alignment >= 16 else token


def _float32_token(value: float) -> str:
    token = repr(value)
    if token in {"inf", "-inf", "nan"}:
        raise ValueError("float32 constexpr must be finite")
    return token
