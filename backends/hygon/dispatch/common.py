# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Hygon dispatch common."""

from __future__ import annotations

from typing import Any, Mapping
import json
import math


SCHEMA_VERSION = 3


ARTIFACT_SCHEMA_VERSION = 5


EXECUTION_PROGRAM_VERSION = 2


PROVIDER_NAME = "hygon_triton"


PROVIDER_VERSION = "1"


LIBTRITON_JIT_GLOBAL_SCRATCH_SIZE = 4096


HYGON_WARP_SIZE = 64


MAX_I32 = 2**31 - 1


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


SUPPORTED_TARGET = "gfx936"


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


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def require_ieee_precision(
    parameters: Mapping[str, Any], data_type: str
) -> None:
    """Validate explicit public precision requests before selecting kernels."""
    value = parameters.get("input_precision", 0)
    if type(value) is not int or value not in (0, 1, 2):
        raise ValueError("input_precision must be 0, 1 or 2")
    if value and data_type != "float32":
        raise ValueError("explicit IEEE/TF32 precision requires FP32 storage")
    if value == 2:
        raise ValueError(
            "Hygon supports IEEE dot products; TF32 is unsupported"
        )
