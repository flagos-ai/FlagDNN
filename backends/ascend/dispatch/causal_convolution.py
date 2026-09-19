"""Ascend dispatch for causal convolution."""

import math
from .common import (
    FLOAT_DATA_TYPES,
    TRITON_POINTER_TYPES,
    _require_integer,
    _has_non_overlapping_strides,
)


def _causal_conv1d_kernel_configuration(parameters, tensors):
    bias = _require_integer(parameters, "has_bias", minimum=0, maximum=1) == 1
    activation = _require_integer(parameters, "activation", minimum=0, maximum=1)
    precision = _require_integer(parameters, "input_precision", minimum=0, maximum=2)
    dilation = _require_integer(parameters, "dilation")
    if len(tensors) != (4 if bias else 3):
        raise ValueError("causal_conv1d tensor count is inconsistent")
    x, weight = tensors[:2]
    y = tensors[-1]
    if (
        len(x["dimensions"]) != 3
        or y["dimensions"] != x["dimensions"]
        or len(weight["dimensions"]) != 2
        or weight["dimensions"][0] != x["dimensions"][1]
    ):
        raise ValueError(
            "causal_conv1d requires X[B,C,L], weight[C,K] and matching output"
        )
    if x["data_type"] not in FLOAT_DATA_TYPES or (
        precision and x["data_type"] != "float32"
    ):
        raise ValueError("causal_conv1d input dtype or precision is invalid")
    for tensor in tensors:
        if tensor["data_type"] != x["data_type"] or not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        ):
            raise ValueError(
                "causal_conv1d requires matching floating "
                "types and non-overlapping strides"
            )
    channels, length, width = (
        x["dimensions"][1],
        x["dimensions"][2],
        weight["dimensions"][1],
    )
    if bias and tensors[2]["dimensions"] != [channels]:
        raise ValueError("causal_conv1d bias requires shape [C]")
    elements = _require_integer(parameters, "n_elements")
    if elements != math.prod(x["dimensions"]) or (width - 1) * dilation > 2**63 - 1:
        raise ValueError("causal_conv1d sizes are inconsistent or too large")
    constants = {
        "ELEMENTS": elements,
        "CHANNELS": channels,
        "LENGTH": length,
        "WIDTH": width,
        "DILATION": dilation,
        "HAS_BIAS": bias,
        "SILU": bool(activation),
        "TF32": precision == 2,
        "W_C": weight["strides"][0],
        "W_K": weight["strides"][1],
        "B_C": tensors[2]["strides"][0] if bias else 0,
        "BLOCK_SIZE": 128 if width >= 16 else 256,
        "CHANNEL_TILE": width >= 16,
    }
    for role, tensor in [("X", x), ("Y", y)]:
        for axis, stride in zip("BCL", tensor["strides"]):
            constants[role + "_" + axis] = stride
    grid = (
        ((length + 127) // 128) * (elements // length)
        if width >= 16
        else (elements + 255) // 256
    )
    if grid > 2**31 - 1:
        raise ValueError("causal_conv1d launch grid exceeds Ascend limits")
    pointer = TRITON_POINTER_TYPES[x["data_type"]]
    return (
        "causal_conv1d_kernel",
        {name + "_ptr": pointer for name in ["x", "weight", "bias", "y"]},
        constants,
        (grid, 1, 1),
        [("tensor", None)] * 2
        + ([("tensor", None)] if bias else [("tensor_alias", -1)])
        + [("tensor", None)],
    )
