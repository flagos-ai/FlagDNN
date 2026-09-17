# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Iluvatar dispatch / common implementation."""

from __future__ import annotations

from ..codegen.identity import GRAPH_IR_SCHEMA_VERSION
from typing import Any
import math

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


POINTER_TYPES = {
    "float32": "*fp32",
    "int32": "*i32",
    "fp8_e8m0": "*u8",
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
        raise ValueError(f"parameters.{name} must be in [{minimum}, {maximum}]")
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
