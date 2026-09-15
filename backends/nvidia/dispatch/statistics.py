"""Validation and kernel plans for channel statistics."""

import math
from .common import (
    FLOAT_DATA_TYPES,
    TRITON_POINTER_TYPES,
    _require_integer,
    _has_non_overlapping_strides,
)


def _genstats_kernel_configuration(parameters, tensors):
    if len(tensors) != 3:
        raise ValueError("genstats requires an input and two outputs")
    x, total, square = tensors
    if (
        x["data_type"] not in FLOAT_DATA_TYPES
        or not 2 <= len(x["dimensions"]) <= 8
    ):
        raise ValueError("genstats requires a floating tensor of rank 2..8")
    for tensor in tensors:
        if not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        ):
            raise ValueError(
                "genstats tensors must have non-overlapping strides"
            )
    channels = _require_integer(parameters, "channels")
    reduction = _require_integer(parameters, "reduction")
    if channels != x["dimensions"][1] or channels * reduction != math.prod(
        x["dimensions"]
    ):
        raise ValueError("genstats channel/reduction sizes are inconsistent")
    shape = [1] * len(x["dimensions"])
    shape[1] = channels
    if any(
        t["dimensions"] != shape or t["data_type"] != "float32"
        for t in [total, square]
    ):
        raise ValueError("genstats outputs must be FP32 channel statistics")
    constants = {
        "CHANNELS": channels,
        "REDUCTION": reduction,
        "SPATIAL": math.prod(x["dimensions"][2:]),
        "SUM_CHANNEL_STRIDE": total["strides"][1],
        "SQ_SUM_CHANNEL_STRIDE": square["strides"][1],
        "BLOCK_SIZE": min(1024, 1 << (reduction - 1).bit_length()),
    }
    dimensions = [1] * (8 - len(x["dimensions"])) + x["dimensions"]
    strides = [0] * (8 - len(x["strides"])) + x["strides"]
    for axis in range(8):
        constants[f"DIM_{axis}"] = dimensions[axis]
        constants[f"STRIDE_{axis}"] = strides[axis]
    return (
        "genstats_kernel",
        {
            "x_ptr": TRITON_POINTER_TYPES[x["data_type"]],
            "sum_ptr": "*fp32",
            "sq_sum_ptr": "*fp32",
        },
        constants,
        (channels, 1, 1),
        [("tensor", None)] * 3,
    )


def _bn_finalize_kernel_configuration(parameters, tensors):
    running = (
        _require_integer(parameters, "has_running", minimum=0, maximum=1) == 1
    )
    if len(tensors) != (12 if running else 8):
        raise ValueError("bn_finalize tensor count is inconsistent")
    shape = tensors[0]["dimensions"]
    if not 1 <= len(shape) <= 8:
        raise ValueError("bn_finalize rank must be 1..8")
    axis = 0 if len(shape) == 1 else 1
    channels = shape[axis]
    if (
        math.prod(shape) != channels
        or _require_integer(parameters, "channels") != channels
    ):
        raise ValueError("bn_finalize requires channel-only dimensions")
    for index, tensor in enumerate(tensors):
        if tensor["dimensions"] != shape or not _has_non_overlapping_strides(
            shape, tensor["strides"]
        ):
            raise ValueError("bn_finalize shape or strides are invalid")
        if tensor["data_type"] not in FLOAT_DATA_TYPES or (
            index not in (2, 3) and tensor["data_type"] != "float32"
        ):
            raise ValueError("bn_finalize statistics require FP32")
    if tensors[2]["data_type"] != tensors[3]["data_type"]:
        raise ValueError("bn_finalize scale/bias types must match")
    count, epsilon, momentum = (
        parameters.get(k) for k in ("accum_count", "epsilon", "momentum")
    )
    if any(
        isinstance(v, bool)
        or not isinstance(v, (int, float))
        or not math.isfinite(v)
        for v in (count, epsilon, momentum)
    ):
        raise ValueError(
            "bn_finalize scalar attributes must be finite numbers"
        )
    if (
        not 1 <= count <= 2**53
        or count != math.floor(count)
        or epsilon <= 0
        or not 0 <= momentum <= 1
    ):
        raise ValueError("bn_finalize scalar attributes are invalid")
    roles = [
        "SUM",
        "SQ_SUM",
        "SCALE",
        "BIAS",
        "PREV_MEAN",
        "PREV_VAR",
        "EQ_SCALE",
        "EQ_BIAS",
        "MEAN",
        "INV",
        "NEXT_MEAN",
        "NEXT_VAR",
    ]
    abi_tensors = (
        tensors
        if running
        else tensors[:4] + [tensors[0]] * 2 + tensors[4:] + [tensors[0]] * 2
    )
    arguments = (
        [("tensor", None)] * 12
        if running
        else [("tensor", None)] * 4
        + [("tensor_alias", 0)] * 2
        + [("tensor", None)] * 4
        + [("tensor_alias", 0)] * 2
    )
    constants = {
        "CHANNELS": channels,
        "COUNT": float(count),
        "EPSILON": float(epsilon),
        "MOMENTUM": float(momentum),
        "HAS_RUNNING": running,
        "BLOCK_SIZE": 256,
    }
    for role, tensor in zip(roles, abi_tensors):
        constants[role + "_STRIDE"] = tensor["strides"][axis]
    if channels > 256 * (2**31 - 1):
        raise ValueError("bn_finalize launch grid exceeds NVIDIA limits")
    return (
        "bn_finalize_kernel",
        {
            role.lower() + "_ptr": TRITON_POINTER_TYPES[tensor["data_type"]]
            for role, tensor in zip(roles, abi_tensors)
        },
        constants,
        ((channels + 255) // 256, 1, 1),
        arguments,
    )
