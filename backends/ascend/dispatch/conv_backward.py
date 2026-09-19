"""Ascend dispatch for conv backward."""

from __future__ import annotations
from typing import Any
import math
from .common import (
    FLOAT_DATA_TYPES,
    TRITON_POINTER_TYPES,
    _ceil_div,
    _has_non_overlapping_strides,
    _is_row_major_contiguous,
    _require_integer,
    _require_integer_list,
)


def _convolution_backward_kernel_configuration(
    operation: str,
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | bool | str],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    if operation not in {"convolution_dgrad", "convolution_wgrad"}:
        raise ValueError("unknown convolution backward operation")
    if len(tensors) != 3:
        raise ValueError("convolution backward tensor count is invalid")
    data_types = [tensor["data_type"] for tensor in tensors]
    if len(set(data_types)) != 1 or data_types[0] not in FLOAT_DATA_TYPES:
        raise ValueError("convolution backward tensors must use one floating data type")
    pointer_type = TRITON_POINTER_TYPES[data_types[0]]
    spatial_rank = _require_integer(parameters, "spatial_rank", minimum=1, maximum=3)
    tensor_rank = spatial_rank + 2
    if any(
        len(tensor["dimensions"]) != tensor_rank
        or any(dimension > 2**31 - 1 for dimension in tensor["dimensions"])
        or not _has_non_overlapping_strides(tensor["dimensions"], tensor["strides"])
        for tensor in tensors
    ):
        raise ValueError(
            "convolution backward tensors require rank spatial_rank + 2, "
            "int32 dimensions, and non-overlapping strides"
        )

    pre_padding = _require_integer_list(
        parameters, "pre_padding", spatial_rank, minimum=0
    )
    post_padding = _require_integer_list(
        parameters, "post_padding", spatial_rank, minimum=0
    )
    stride = _require_integer_list(parameters, "stride", spatial_rank, minimum=1)
    dilation = _require_integer_list(parameters, "dilation", spatial_rank, minimum=1)
    groups = _require_integer(parameters, "groups")
    convolution_mode = _require_integer(
        parameters, "convolution_mode", minimum=0, maximum=1
    )
    outputs = _require_integer(parameters, "n_outputs")

    dy = tensors[0]
    if operation == "convolution_dgrad":
        filter_tensor = tensors[1]
        image = tensors[2]
    else:
        image = tensors[1]
        filter_tensor = tensors[2]
    n, c = image["dimensions"][:2]
    k, filter_channels = filter_tensor["dimensions"][:2]
    if dy["dimensions"][:2] != [n, k]:
        raise ValueError("convolution backward loss batch/channels are inconsistent")
    if c % groups != 0 or k % groups != 0:
        raise ValueError("convolution backward channels must divide groups")
    channels_per_group = c // groups
    outputs_per_group = k // groups
    if filter_channels != channels_per_group:
        raise ValueError("convolution backward filter channels are inconsistent")
    expected_loss = [n, k]
    for axis in range(spatial_rank):
        image_extent = image["dimensions"][axis + 2]
        filter_extent = filter_tensor["dimensions"][axis + 2]
        effective_filter = dilation[axis] * (filter_extent - 1) + 1
        padded_image = image_extent + pre_padding[axis] + post_padding[axis]
        if padded_image < effective_filter:
            raise ValueError("convolution backward filter is larger than padded image")
        expected_loss.append((padded_image - effective_filter) // stride[axis] + 1)
    if dy["dimensions"] != expected_loss:
        raise ValueError("convolution backward loss metadata is inconsistent")
    expected_output = (
        image["dimensions"]
        if operation == "convolution_dgrad"
        else filter_tensor["dimensions"]
    )
    if math.prod(expected_output) != outputs:
        raise ValueError("parameters.n_outputs is inconsistent with backward output")

    def padded_spatial(values: list[int], fill: int) -> list[int]:
        return [fill] * (3 - spatial_rank) + list(values)

    image_spatial = padded_spatial(image["dimensions"][2:], 1)
    filter_spatial = padded_spatial(filter_tensor["dimensions"][2:], 1)
    loss_spatial = padded_spatial(dy["dimensions"][2:], 1)
    spatial_stride = padded_spatial(stride, 1)
    spatial_padding = padded_spatial(pre_padding, 0)
    spatial_dilation = padded_spatial(dilation, 1)
    image_strides = padded_spatial(image["strides"][2:], 0)
    filter_strides = padded_spatial(filter_tensor["strides"][2:], 0)
    loss_strides = padded_spatial(dy["strides"][2:], 0)

    constants: dict[str, int | bool | str] = {
        "XD": image_spatial[0],
        "XH": image_spatial[1],
        "XW": image_spatial[2],
        "OD": loss_spatial[0],
        "OH": loss_spatial[1],
        "OW": loss_spatial[2],
        "KD": filter_spatial[0],
        "KH": filter_spatial[1],
        "KW": filter_spatial[2],
        "CIN_PER_GROUP": channels_per_group,
        "COUT_PER_GROUP": outputs_per_group,
        "STRIDE_D": spatial_stride[0],
        "STRIDE_H": spatial_stride[1],
        "STRIDE_W": spatial_stride[2],
        "PAD_FRONT": spatial_padding[0],
        "PAD_TOP": spatial_padding[1],
        "PAD_LEFT": spatial_padding[2],
        "DIL_D": spatial_dilation[0],
        "DIL_H": spatial_dilation[1],
        "DIL_W": spatial_dilation[2],
        "FLIP_FILTER": convolution_mode == 1,
        "DY_STRIDE_N": dy["strides"][0],
        "DY_STRIDE_C": dy["strides"][1],
        "DY_STRIDE_D": loss_strides[0],
        "DY_STRIDE_H": loss_strides[1],
        "DY_STRIDE_W": loss_strides[2],
        "X_STRIDE_N": image["strides"][0],
        "X_STRIDE_C": image["strides"][1],
        "X_STRIDE_D": image_strides[0],
        "X_STRIDE_H": image_strides[1],
        "X_STRIDE_W": image_strides[2],
        "W_STRIDE_K": filter_tensor["strides"][0],
        "W_STRIDE_C": filter_tensor["strides"][1],
        "W_STRIDE_D": filter_strides[0],
        "W_STRIDE_H": filter_strides[1],
        "W_STRIDE_W": filter_strides[2],
        "INPUT_PRECISION": 1 if data_types[0] == "float32" else 0,
    }
    pointer_abi: list[tuple[str, str | int | None]] = [
        ("tensor", None),
        ("tensor", None),
        ("tensor", None),
    ]
    kernel_volume = math.prod(filter_spatial)
    if operation == "convolution_dgrad":
        m = n * math.prod(image_spatial)
        block_m = 32
        block_ci = 16 if channels_per_group <= 16 else 32
        block_k = 16 if outputs_per_group * kernel_volume <= 16 else 32
        direct_ieee = (
            parameters.get("input_precision") == 1
            and outputs_per_group * kernel_volume <= 192
        )
        if (
            spatial_rank == 2
            and m >= 8192
            and channels_per_group <= 4
            and filter_spatial[1:] == [3, 3]
            and stride == [2, 2]
            and pre_padding == [1, 1]
            and post_padding == [1, 1]
            and dilation == [1, 1]
            and not direct_ieee
        ):
            block_m = 128
            constants.update(M=m, BLOCK_M=block_m, BLOCK_CI=1, BLOCK_K=32, GROUP_M=8)
            phase_rows = (
                n * _ceil_div(image_spatial[1], 2) * _ceil_div(image_spatial[2], 2)
            )
            return (
                "conv_dgrad_stem_vector_kernel",
                {"dy_ptr": pointer_type, "w_ptr": pointer_type, "dx_ptr": pointer_type},
                constants,
                (_ceil_div(phase_rows, block_m), 4, groups * channels_per_group),
                pointer_abi,
            )
        is_contiguous_1x1_2d = (
            spatial_rank == 2
            and filter_spatial[1:] == [1, 1]
            and stride == [1, 1]
            and pre_padding == [0, 0]
            and post_padding == [0, 0]
            and dilation == [1, 1]
            and all(_is_row_major_contiguous(tensor) for tensor in tensors)
        )
        if is_contiguous_1x1_2d and not direct_ieee:
            hw = image_spatial[1] * image_spatial[2]
            block_m = 64
            block_ci = 32
            block_co = 128
            return (
                "conv_dgrad2d_1x1_nchw_kernel",
                {
                    "dy_ptr": pointer_type,
                    "w_ptr": pointer_type,
                    "dx_ptr": pointer_type,
                },
                {
                    "HW": hw,
                    "C_IN": c,
                    "C_OUT": k,
                    "CIN_PER_GROUP": channels_per_group,
                    "COUT_PER_GROUP": outputs_per_group,
                    "GROUPS": groups,
                    "INPUT_PRECISION": (1 if data_types[0] == "float32" else 0),
                    "BLOCK_M": block_m,
                    "BLOCK_CI": block_ci,
                    "BLOCK_CO": block_co,
                },
                (
                    _ceil_div(hw, block_m) * _ceil_div(channels_per_group, block_ci),
                    n * groups,
                    1,
                ),
                pointer_abi,
            )
        if (
            spatial_rank == 2
            and stride == [1, 1]
            and kernel_volume > 1
            and not direct_ieee
        ):
            block_co = block_k
            stride1_constants = {
                "M": m,
                "XH": image_spatial[1],
                "XW": image_spatial[2],
                "OH": loss_spatial[1],
                "OW": loss_spatial[2],
                "CIN_PER_GROUP": channels_per_group,
                "COUT_PER_GROUP": outputs_per_group,
                "DY_STRIDE_N": dy["strides"][0],
                "DY_STRIDE_C": dy["strides"][1],
                "DY_STRIDE_H": loss_strides[1],
                "DY_STRIDE_W": loss_strides[2],
                "W_STRIDE_K": filter_tensor["strides"][0],
                "W_STRIDE_C": filter_tensor["strides"][1],
                "W_STRIDE_H": filter_strides[1],
                "W_STRIDE_W": filter_strides[2],
                "X_STRIDE_N": image["strides"][0],
                "X_STRIDE_C": image["strides"][1],
                "X_STRIDE_H": image_strides[1],
                "X_STRIDE_W": image_strides[2],
                "PAD_TOP": spatial_padding[1],
                "PAD_LEFT": spatial_padding[2],
                "DIL_H": spatial_dilation[1],
                "DIL_W": spatial_dilation[2],
                "KH": filter_spatial[1],
                "KW": filter_spatial[2],
                "FLIP_FILTER": convolution_mode == 1,
                "INPUT_PRECISION": 1 if data_types[0] == "float32" else 0,
                "BLOCK_M": block_m,
                "BLOCK_CI": block_ci,
                "BLOCK_CO": block_co,
            }
            return (
                "conv_dgrad2d_stride1_kernel",
                {
                    "dy_ptr": pointer_type,
                    "w_ptr": pointer_type,
                    "dx_ptr": pointer_type,
                },
                stride1_constants,
                (
                    ((m + block_m - 1) // block_m)
                    * ((channels_per_group + block_ci - 1) // block_ci),
                    groups,
                    1,
                ),
                pointer_abi,
            )
        constants.update(
            {
                "M": m,
                "BLOCK_M": block_m,
                "BLOCK_CI": block_ci,
                "BLOCK_K": block_k,
                "GROUP_M": 8,
            }
        )
        return (
            "conv_dgrad_nd_kernel",
            {
                "dy_ptr": pointer_type,
                "w_ptr": pointer_type,
                "dx_ptr": pointer_type,
            },
            constants,
            (
                ((m + block_m - 1) // block_m)
                * ((channels_per_group + block_ci - 1) // block_ci),
                groups,
                1,
            ),
            pointer_abi,
        )

    m = n * math.prod(loss_spatial)
    if _is_row_major_contiguous(filter_tensor) and (m >= 8192 or outputs >= 65536):
        # Large filters have enough tiles to fill the device at 32 x 32.
        # Retain smaller tiles for stems with only a few input channels.
        tile = 32 if outputs >= 262144 else 16
        constants.update(
            M=m,
            OUTPUT_ELEMENTS=outputs,
            BLOCK_OUTPUT=tile,
            BLOCK_OC=tile,
            BLOCK_REDUCE=64,
        )
        return (
            "conv_wgrad_contiguous_kernel",
            {"dy_ptr": pointer_type, "x_ptr": pointer_type, "dw_ptr": pointer_type},
            constants,
            (
                _ceil_div(outputs_per_group, tile)
                * _ceil_div(channels_per_group * kernel_volume, tile),
                groups,
                1,
            ),
            pointer_abi,
        )
    constants.update(M=m, OUTPUT_ELEMENTS=outputs, BLOCK_OUTPUT=8, BLOCK_REDUCE=128)
    return (
        "conv_wgrad_nd_kernel",
        {"dy_ptr": pointer_type, "x_ptr": pointer_type, "dw_ptr": pointer_type},
        constants,
        (_ceil_div(outputs, 8), 1, 1),
        pointer_abi,
    )
