"""Attention metadata validation shared by forward and backward plans."""

from __future__ import annotations

from typing import Any
import math

from .common import (
    FLOAT_DATA_TYPES,
    FP8_DATA_TYPES,
    _FLOAT32_MAX,
    _require_integer,
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


def _require_fp8_scale_tensor(tensor: dict[str, Any], name: str) -> None:
    if (
        tensor["data_type"] != "float32"
        or math.prod(tensor["dimensions"]) != 1
    ):
        raise ValueError(f"{name} must be a one-element float32 tensor")
