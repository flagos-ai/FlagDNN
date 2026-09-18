"""Resample metadata validation and spatial kernel launch selection."""

import math

from .metadata import (
    FLOAT_DATA_TYPES,
    TRITON_POINTER_TYPES,
    _has_non_overlapping_strides,
    _require_integer,
)


def _resample_kernel_configuration(parameters, tensors):
    mode = _require_integer(parameters, "mode", minimum=1, maximum=5)
    padding = _require_integer(parameters, "padding", minimum=1, maximum=3)
    index = _require_integer(
        parameters, "generate_index", minimum=0, maximum=1
    )
    align = _require_integer(parameters, "align_corners", minimum=0, maximum=1)
    if (
        len(tensors) != (3 if index else 2)
        or (index and mode != 5)
        or (align and mode != 3)
    ):
        raise ValueError("resample mode flags or tensor count are invalid")
    x, y = tensors[:2]
    shape = x["dimensions"]
    rank = len(shape)
    if (
        rank not in (3, 4, 5)
        or len(y["dimensions"]) != rank
        or shape[:2] != y["dimensions"][:2]
    ):
        raise ValueError(
            "resample requires matching NC dimensions and "
            "1..3 spatial dimensions"
        )
    if (
        x["data_type"] not in FLOAT_DATA_TYPES
        or y["data_type"] != x["data_type"]
    ):
        raise ValueError(
            "resample input/output types must match and be floating"
        )
    for tensor in tensors:
        if not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        ) or any(d > 2**31 - 1 for d in tensor["dimensions"]):
            raise ValueError(
                "resample tensor strides or MThreads dimension "
                "limits are invalid"
            )
    if index and (
        tensors[2]["dimensions"] != y["dimensions"]
        or tensors[2]["data_type"] != "int32"
    ):
        raise ValueError("resample index must have INT32 output shape")
    if (
        (mode == 3 and rank != 4)
        or (mode in (3, 4) and padding != 1)
        or (mode == 1 and padding != 3)
        or (mode == 2 and padding == 2)
    ):
        raise ValueError(
            "resample interpolation/padding combination is invalid"
        )
    spatial = rank - 2
    window, stride, pre, post = (
        parameters.get(name)
        for name in ("window", "stride", "pre_padding", "post_padding")
    )
    for values, minimum in [(window, 1), (stride, 1), (pre, 0), (post, 0)]:
        if (
            not isinstance(values, list)
            or len(values) != spatial
            or any(
                isinstance(v, bool)
                or not isinstance(v, int)
                or not minimum <= v <= 2**31 - 1
                for v in values
            )
        ):
            raise ValueError("resample spatial parameters are invalid")
    if math.prod(window) > 2**31 - 1:
        raise ValueError(
            "resample window volume exceeds MThreads kernel limits"
        )
    if mode in (3, 4):
        if (
            window != [1] * spatial
            or stride != [1] * spatial
            or pre != [0] * spatial
            or post != [0] * spatial
        ):
            raise ValueError("resize cannot use pooling windows")
    elif any(
        (d + a + b - k) // s + 1 != o
        for d, a, b, k, s, o in zip(
            shape[2:], pre, post, window, stride, y["dimensions"][2:]
        )
    ):
        raise ValueError("resample inferred output shape is inconsistent")
    elements = _require_integer(parameters, "n_elements")
    if elements != math.prod(y["dimensions"]):
        raise ValueError("resample output element count is inconsistent")
    constants = {
        "ELEMENTS": elements,
        "CHANNELS": shape[1],
        "MODE": mode,
        "PADDING": padding,
        "INDEX": bool(index),
        "ALIGN": bool(align),
        "BLOCK_SIZE": 256,
        "CHANNELS_LAST": y["strides"][1] == 1,
    }
    for prefix, values, leading in [
        ("I", shape[2:], 1),
        ("O", y["dimensions"][2:], 1),
        ("K", window, 1),
        ("S", stride, 1),
        ("P", pre, 0),
    ]:
        for axis, value in zip("DHW", [leading] * (3 - spatial) + values):
            constants[prefix + axis] = value
    for role, tensor in [
        ("X", x),
        ("Y", y),
        ("I", tensors[2] if index else y),
    ]:
        values = (
            tensor["strides"][:2] + [0] * (3 - spatial) + tensor["strides"][2:]
        )
        for axis, value in zip("NCDHW", values):
            constants[role + "_" + axis] = value
    grid = (elements + 255) // 256
    if grid > 2**31 - 1:
        raise ValueError("resample launch grid exceeds MThreads limits")
    return (
        "resample_kernel",
        {
            "x_ptr": TRITON_POINTER_TYPES[x["data_type"]],
            "y_ptr": TRITON_POINTER_TYPES[y["data_type"]],
            "index_ptr": "*i32",
        },
        constants,
        (grid, 1, 1),
        [("tensor", None)] * 2
        + ([("tensor", None)] if index else [("tensor_alias", 1)]),
    )
