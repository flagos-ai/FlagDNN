# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Layout validation and launch selection for rotary position embeddings."""

import math
from .extended_common import (
    FLOAT_DATA_TYPES,
    TRITON_POINTER_TYPES,
    _has_non_overlapping_strides,
    _require_integer,
)


def _rope_kernel_configuration(operation, parameters, tensors):
    if len(tensors) != 3:
        raise ValueError("RoPE requires input, frequencies and output")
    x, freqs, y = tensors
    if len(x["dimensions"]) != 4 or y["dimensions"] != x["dimensions"]:
        raise ValueError("RoPE requires BHSD input and matching output")
    if (
        x["data_type"] not in FLOAT_DATA_TYPES
        or y["data_type"] != x["data_type"]
    ):
        raise ValueError("RoPE input/output must use the same floating type")
    for tensor in tensors:
        if not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        ):
            raise ValueError("RoPE requires non-overlapping tensor strides")
    batch, heads, sequence, dimension = x["dimensions"]
    width = _require_integer(
        parameters, "rope_dim", minimum=2, maximum=dimension
    )
    if (
        width % 2
        or freqs["dimensions"] != [sequence, 1, 1, width]
        or freqs["data_type"] != "float32"
    ):
        raise ValueError(
            "RoPE frequencies must be FP32 [S,1,1,rope_dim] "
            "with even rotation width"
        )
    elements = _require_integer(parameters, "n_elements")
    if elements != batch * heads * sequence * dimension:
        raise ValueError("RoPE element count is inconsistent")
    scale = parameters.get("output_scale")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(scale)
    ):
        raise ValueError("RoPE output scale must be finite")
    constants = {
        "ELEMENTS": elements,
        "HEADS": heads,
        "SEQUENCE": sequence,
        "DIMENSION": dimension,
        "ROPE_DIM": width,
        "SCALE": float(scale),
        "F_S": freqs["strides"][0],
        "F_D": freqs["strides"][3],
        "BACKWARD": operation == "rope_backward",
        "BLOCK_SIZE": 256,
    }
    for role, tensor in [("X", x), ("Y", y)]:
        for axis, stride in zip("BHSD", tensor["strides"]):
            constants[role + "_" + axis] = stride
    grid = (elements + 255) // 256
    if grid > 2**31 - 1:
        raise ValueError("RoPE launch grid exceeds Hygon limits")
    return (
        "rope_kernel",
        {
            "x_ptr": TRITON_POINTER_TYPES[x["data_type"]],
            "freqs_ptr": "*fp32",
            "y_ptr": TRITON_POINTER_TYPES[y["data_type"]],
        },
        constants,
        (grid, 1, 1),
        [("tensor", None)] * 3,
    )
