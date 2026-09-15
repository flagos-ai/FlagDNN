"""Validate FP8 GEMM layouts and select the native FP8 contraction kernel."""

import math
from .common import (
    FP8_DATA_TYPES,
    FLOAT_DATA_TYPES,
    TRITON_POINTER_TYPES,
    _has_non_overlapping_strides,
    _require_integer,
)


def _fp8_matmul_configuration(parameters, tensors, architecture):
    if architecture < 90:
        raise ValueError("NVIDIA FP8 matmul requires SM90 or newer")
    mode = _require_integer(parameters, "scale_mode", minimum=0, maximum=2)
    if len(tensors) != (5 if mode else 3):
        raise ValueError("incorrect FP8 matmul tensor count")
    a, b, c = tensors[0], tensors[1], tensors[-1]
    if (
        a["data_type"] not in FP8_DATA_TYPES
        or b["data_type"] not in FP8_DATA_TYPES
        or c["data_type"] not in FLOAT_DATA_TYPES
    ):
        raise ValueError(
            "FP8 matmul requires FP8 inputs and FP32/FP16/BF16 output"
        )
    for tensor in tensors:
        if not 1 <= len(
            tensor["dimensions"]
        ) <= 8 or not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        ):
            raise ValueError(
                "FP8 matmul requires non-overlapping tensor layouts"
            )
    if len(a["dimensions"]) < 2 or len(b["dimensions"]) < 2:
        raise ValueError("FP8 matmul inputs require rank at least two")
    m, k = a["dimensions"][-2:]
    bk, n = b["dimensions"][-2:]
    ad = [1] * (8 - len(a["dimensions"])) + a["dimensions"]
    bd = [1] * (8 - len(b["dimensions"])) + b["dimensions"]
    batch = [max(x, y) for x, y in zip(ad[:-2], bd[:-2])]
    expected = batch[8 - max(len(a["dimensions"]), len(b["dimensions"])) :] + [
        m,
        n,
    ]
    if (
        k != bk
        or c["dimensions"] != expected
        or any(x != y and x != 1 and y != 1 for x, y in zip(ad[:-2], bd[:-2]))
    ):
        raise ValueError(
            "FP8 matmul contraction or batch shapes are inconsistent"
        )
    for name, value in [
        ("m", m),
        ("n", n),
        ("k", k),
        ("batch", math.prod(batch)),
    ]:
        if _require_integer(parameters, name) != value:
            raise ValueError("FP8 matmul shape attributes are inconsistent")
    constants = dict(
        M=m,
        N=n,
        K=k,
        SCALE_MODE=mode,
        BLOCK_M=16,
        BLOCK_N=32,
        BLOCK_K=32 if mode == 2 or k < 128 else 64 if k < 256 else 128,
        AM=a["strides"][-2],
        AK=a["strides"][-1],
        BK=b["strides"][-2],
        BN=b["strides"][-1],
        CM=c["strides"][-2],
        CN=c["strides"][-1],
        SAM=0,
        SAK=0,
        SBK=0,
        SBN=0,
    )
    for axis, dim in enumerate(batch):
        constants[f"D{axis}"] = dim
    for prefix, tensor in [("A", a), ("B", b), ("C", c)]:
        dims = [1] * (8 - len(tensor["dimensions"])) + tensor["dimensions"]
        strides = [0] * (8 - len(tensor["strides"])) + tensor["strides"]
        for axis in range(6):
            constants[f"{prefix}{axis}"] = (
                0 if dims[axis] == 1 else strides[axis]
            )
    for prefix in ["SA", "SB"]:
        for axis in range(6):
            constants[f"{prefix}{axis}"] = 0
    if mode:
        sa, sb = tensors[2:4]
        for prefix, tensor, operand, axis in [
            ("SA", sa, a, -1),
            ("SB", sb, b, -2),
        ]:
            if mode == 1:
                if (
                    tensor["data_type"] != "float32"
                    or math.prod(tensor["dimensions"]) != 1
                ):
                    raise ValueError("FP8 descales must be FP32 scalars")
            else:
                expected = list(operand["dimensions"])
                expected[axis] = (k + 31) // 32
                if (
                    tensor["data_type"] != "fp8_e8m0"
                    or tensor["dimensions"] != expected
                ):
                    raise ValueError(
                        "MXFP8 requires E8M0 scales per 32 "
                        "contraction elements"
                    )
                constants[prefix + ("M" if prefix == "SA" else "K")] = tensor[
                    "strides"
                ][-2]
                constants[prefix + ("K" if prefix == "SA" else "N")] = tensor[
                    "strides"
                ][-1]
                dims = [1] * (8 - len(tensor["dimensions"])) + tensor[
                    "dimensions"
                ]
                strides = [0] * (8 - len(tensor["strides"])) + tensor[
                    "strides"
                ]
                for i in range(6):
                    constants[f"{prefix}{i}"] = (
                        0 if dims[i] == 1 else strides[i]
                    )
    scale_type = "*u8" if mode == 2 else "*fp32"
    signature = dict(
        a_ptr=TRITON_POINTER_TYPES[a["data_type"]],
        b_ptr=TRITON_POINTER_TYPES[b["data_type"]],
        sa_ptr=scale_type,
        sb_ptr=scale_type,
        c_ptr=TRITON_POINTER_TYPES[c["data_type"]],
    )
    arguments = (
        [("tensor", None)] * 5
        if mode
        else [("tensor", None)] * 2
        + [("tensor_alias", -1)] * 2
        + [("tensor", None)]
    )
    grid = ((m + 15) // 16) * ((n + 31) // 32) * math.prod(batch)
    if grid > 2**31 - 1:
        raise ValueError("FP8 matmul launch grid exceeds NVIDIA limits")
    return "fp8_matmul_kernel", signature, constants, (grid, 1, 1), arguments
