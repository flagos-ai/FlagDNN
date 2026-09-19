"""Ascend dispatch convolution implementation."""

from __future__ import annotations

from .common import (
    TensorPlan,
)


def _convolution_fprop_meta(
    input_tensor: TensorPlan,
    filter_tensor: TensorPlan,
    output: TensorPlan,
    *,
    spatial_rank: int,
    groups: int,
    pre_padding: list[int],
    post_padding: list[int],
    stride: list[int],
    dilation: list[int],
) -> dict[str, int]:
    result = {
        "SPATIAL_RANK": spatial_rank,
        "GROUPS": groups,
        "INPUT_CHANNELS": input_tensor.dimensions[1],
        "OUTPUT_CHANNELS": filter_tensor.dimensions[0],
        "CHANNELS_PER_GROUP": input_tensor.dimensions[1] // groups,
    }
    for name, tensor in (
        ("INPUT", input_tensor),
        ("FILTER", filter_tensor),
        ("OUTPUT", output),
    ):
        missing_spatial = 5 - len(tensor.dimensions)
        dimensions = (
            list(tensor.dimensions[:2])
            + [1] * missing_spatial
            + list(tensor.dimensions[2:])
        )
        strides = (
            list(tensor.strides[:2]) + [0] * missing_spatial + list(tensor.strides[2:])
        )
        for axis in range(5):
            result[f"{name}_DIM_{axis}"] = dimensions[axis]
            result[f"{name}_STRIDE_{axis}"] = strides[axis]
    leading = 3 - spatial_rank
    for name, values, fill in (
        ("PRE_PADDING", pre_padding, 0),
        ("POST_PADDING", post_padding, 0),
        ("CONV_STRIDE", stride, 1),
        ("DILATION", dilation, 1),
    ):
        padded = [fill] * leading + values
        for axis in range(3):
            result[f"{name}_{axis}"] = padded[axis]
    return result
