"""Data-gradient convolution validation and kernel configurations."""

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


def _convolution_dgrad_3d_pipeline_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | bool | str],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    stage = parameters.get("_dgrad_3d_pipeline_stage")
    if stage not in {"pack", "compute_packed", "compute_ci8_dot"}:
        raise ValueError("unknown 3D DGrad pipeline stage")
    if _require_integer(parameters, "groups") != 1:
        raise ValueError("the packed 3D DGrad pipeline requires one group")

    if stage == "pack":
        if len(tensors) != 2:
            raise ValueError("the 3D DGrad pack stage requires two tensors")
        weight, packed = tensors
        if (
            weight["data_type"] != packed["data_type"]
            or weight["data_type"] not in {"float32", "float16", "bfloat16"}
            or len(weight["dimensions"]) != 5
            or len(packed["dimensions"]) != 5
            or not _is_row_major_contiguous(weight)
            or not _is_row_major_contiguous(packed)
        ):
            raise ValueError("the 3D DGrad pack tensor metadata is invalid")
        c_out, c_in, kernel_d, kernel_h, kernel_w = weight["dimensions"]
        if packed["dimensions"] != [
            kernel_d,
            kernel_h,
            kernel_w,
            c_out,
            c_in,
        ]:
            raise ValueError("the 3D DGrad packed filter shape is invalid")
        kernel_volume = kernel_d * kernel_h * kernel_w
        block_size = 256
        pointer_type = TRITON_POINTER_TYPES[weight["data_type"]]
        return (
            "conv_dgrad3d_pack_weight_kernel",
            {"weight_ptr": pointer_type, "packed_ptr": pointer_type},
            {
                "TOTAL": c_out * c_in * kernel_volume,
                "C_OUT": c_out,
                "C_IN": c_in,
                "KERNEL_VOLUME": kernel_volume,
                "BLOCK_SIZE": block_size,
            },
            (_ceil_div(c_out * c_in, block_size), 1, 1),
            [("tensor", None), ("tensor", None)],
        )

    if len(tensors) != 3:
        raise ValueError(
            "the packed 3D DGrad compute stage requires three tensors"
        )
    loss, packed, output = tensors
    if (
        loss["data_type"] != packed["data_type"]
        or loss["data_type"] != output["data_type"]
        or loss["data_type"] not in {"float32", "float16", "bfloat16"}
        or len(loss["dimensions"]) != 5
        or len(packed["dimensions"]) != 5
        or len(output["dimensions"]) != 5
        or not all(
            _is_row_major_contiguous(tensor)
            for tensor in (loss, packed, output)
        )
    ):
        raise ValueError("the packed 3D DGrad compute metadata is invalid")

    n, c_out, loss_d, loss_h, loss_w = loss["dimensions"]
    output_n, c_in, output_d, output_h, output_w = output["dimensions"]
    kernel_d, kernel_h, kernel_w, packed_c_out, packed_c_in = packed[
        "dimensions"
    ]
    if output_n != n or packed_c_out != c_out or packed_c_in != c_in:
        raise ValueError("the packed 3D DGrad compute shape is inconsistent")
    stride = _require_integer_list(parameters, "stride", 3, minimum=1)
    padding = _require_integer_list(parameters, "pre_padding", 3, minimum=0)
    dilation = _require_integer_list(parameters, "dilation", 3, minimum=1)
    convolution_mode = _require_integer(
        parameters, "convolution_mode", minimum=0, maximum=1
    )
    pointer_type = TRITON_POINTER_TYPES[loss["data_type"]]
    pointer_signature = {
        "loss_ptr": pointer_type,
        "weight_ptr": pointer_type,
        "out_ptr": pointer_type,
    }
    pointer_abi = [("tensor", None), ("tensor", None), ("tensor", None)]
    m = n * output_d * output_h * output_w

    if stage == "compute_ci8_dot":
        if (
            loss["data_type"] != "float32"
            or c_out != 16
            or c_in != 8
            or [kernel_d, kernel_h, kernel_w] != [3, 3, 3]
            or stride != [1, 1, 1]
            or padding != [1, 1, 1]
            or dilation != [1, 1, 1]
            or convolution_mode != 0
        ):
            raise ValueError("the 3D DGrad ci8 dot shape is invalid")
        block_m = 16
        return (
            "conv_dgrad3d_pad1_3x3_fp32_ci8_dot_kernel",
            pointer_signature,
            {
                "M": m,
                "XD": output_d,
                "XH": output_h,
                "XW": output_w,
                "LOSS_D": loss_d,
                "LOSS_H": loss_h,
                "LOSS_W": loss_w,
                "loss_stride_n": loss["strides"][0],
                "loss_stride_c": loss["strides"][1],
                "loss_stride_d": loss["strides"][2],
                "loss_stride_h": loss["strides"][3],
                "loss_stride_w": loss["strides"][4],
                "out_stride_n": output["strides"][0],
                "out_stride_c": output["strides"][1],
                "out_stride_d": output["strides"][2],
                "out_stride_h": output["strides"][3],
                "out_stride_w": output["strides"][4],
                "BLOCK_M": block_m,
            },
            (_ceil_div(m, block_m), 1, 1),
            pointer_abi,
        )

    block_m = 8
    block_ci = 16
    block_co = 32
    return (
        "conv_dgrad3d_packed_kernel",
        pointer_signature,
        {
            "M": m,
            "XD": output_d,
            "XH": output_h,
            "XW": output_w,
            "LOSS_D": loss_d,
            "LOSS_H": loss_h,
            "LOSS_W": loss_w,
            "CIN_PER_GROUP": c_in,
            "COUT_PER_GROUP": c_out,
            "loss_stride_n": loss["strides"][0],
            "loss_stride_c": loss["strides"][1],
            "loss_stride_d": loss["strides"][2],
            "loss_stride_h": loss["strides"][3],
            "loss_stride_w": loss["strides"][4],
            "out_stride_n": output["strides"][0],
            "out_stride_c": output["strides"][1],
            "out_stride_d": output["strides"][2],
            "out_stride_h": output["strides"][3],
            "out_stride_w": output["strides"][4],
            "STRIDE_D": stride[0],
            "STRIDE_H": stride[1],
            "STRIDE_W": stride[2],
            "PAD_FRONT": padding[0],
            "PAD_TOP": padding[1],
            "PAD_LEFT": padding[2],
            "DIL_D": dilation[0],
            "DIL_H": dilation[1],
            "DIL_W": dilation[2],
            "KD": kernel_d,
            "KH": kernel_h,
            "KW": kernel_w,
            "FILTER_REVERSE": convolution_mode == 1,
            "INPUT_PRECISION": (1 if loss["data_type"] == "float32" else 0),
            "BLOCK_M": block_m,
            "BLOCK_CI": block_ci,
            "BLOCK_CO": block_co,
        },
        (
            _ceil_div(m, block_m) * _ceil_div(c_in, block_ci),
            1,
            1,
        ),
        pointer_abi,
    )


def _convolution_dgrad_stride2_pipeline_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | bool | str],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    stage = parameters.get("_dgrad_pipeline_stage")
    if stage not in {
        "pack",
        "zero_p5",
        "compute",
        "compute_p5_splitk",
        "compute_tile2w",
        "compute_tile4",
    }:
        raise ValueError("unknown stride-2 DGrad pipeline stage")
    groups = _require_integer(parameters, "groups")
    if groups != 1:
        raise ValueError("the packed DGrad pipeline requires one group")

    if stage == "pack":
        if len(tensors) != 2:
            raise ValueError("the DGrad pack stage requires two tensors")
        weight, packed = tensors
        valid_pack_types = weight["data_type"] == packed[
            "data_type"
        ] and weight["data_type"] in {"float32", "float16", "bfloat16"}
        if (
            not valid_pack_types
            or len(weight["dimensions"]) != 4
            or len(packed["dimensions"]) != 4
            or not _is_row_major_contiguous(weight)
            or not _is_row_major_contiguous(packed)
        ):
            raise ValueError("the DGrad pack tensor metadata is invalid")
        c_out, c_in, kernel_h, kernel_w = weight["dimensions"]
        if (
            kernel_h != 3
            or kernel_w != 3
            or packed["dimensions"] != [3, 3, c_out, c_in]
        ):
            raise ValueError("the DGrad packed filter shape is invalid")
        total = c_out * c_in * 9
        block_size = 256
        return (
            "conv_dgrad2d_pack_weight_kernel",
            {
                "weight_ptr": TRITON_POINTER_TYPES[weight["data_type"]],
                "packed_ptr": TRITON_POINTER_TYPES[packed["data_type"]],
            },
            {
                "TOTAL": total,
                "C_OUT": c_out,
                "C_IN": c_in,
                "ROUND_TF32": (
                    parameters.get("_dgrad_pack_round_tf32") is True
                ),
                "BLOCK_SIZE": block_size,
            },
            (_ceil_div(c_out * c_in, block_size), 1, 1),
            [("tensor", None), ("tensor", None)],
        )

    if stage == "zero_p5":
        if len(tensors) != 1:
            raise ValueError("the P5 DGrad zero stage requires one tensor")
        output = tensors[0]
        if (
            output["data_type"] != "float32"
            or output["dimensions"] != [1, 768, 40, 40]
            or not _is_row_major_contiguous(output)
        ):
            raise ValueError("the P5 DGrad zero tensor metadata is invalid")
        total = math.prod(output["dimensions"])
        block_size = 256
        return (
            "zero_contiguous_kernel",
            {"out_ptr": TRITON_POINTER_TYPES["float32"]},
            {"TOTAL": total, "BLOCK_SIZE": block_size},
            (_ceil_div(total, block_size), 1, 1),
            [("tensor", None)],
        )

    if len(tensors) != 3:
        raise ValueError(
            "the packed DGrad compute stage requires three tensors"
        )
    loss, packed, output = tensors
    valid_compute_types = (
        loss["data_type"] == packed["data_type"]
        and loss["data_type"] == output["data_type"]
        and loss["data_type"] in {"float32", "float16", "bfloat16"}
    )
    if (
        not valid_compute_types
        or len(loss["dimensions"]) != 4
        or len(packed["dimensions"]) != 4
        or len(output["dimensions"]) != 4
        or not _is_row_major_contiguous(output)
    ):
        raise ValueError("the packed DGrad compute metadata is invalid")
    n, c_out, loss_h, loss_w = loss["dimensions"]
    output_n, c_in, output_h, output_w = output["dimensions"]
    k_contiguous = parameters.get("_dgrad_k_contiguous", False)
    if not isinstance(k_contiguous, bool):
        raise ValueError("DGrad K-contiguous layout flag must be boolean")
    if k_contiguous:
        if (
            stage != "compute"
            or loss["strides"]
            != [c_out * loss_h * loss_w, 1, c_out * loss_w, c_out]
            or packed["strides"] != [3 * c_out, c_out, 1, 9 * c_out]
        ):
            raise ValueError("DGrad K-contiguous layout metadata is invalid")
    elif not all(
        _is_row_major_contiguous(tensor) for tensor in (loss, packed)
    ):
        raise ValueError("the packed DGrad compute metadata is invalid")
    if (
        output_n != n
        or packed["dimensions"] != [3, 3, c_out, c_in]
        or loss_h != (output_h + 1) // 2
        or loss_w != (output_w + 1) // 2
    ):
        raise ValueError("the packed DGrad compute shape is inconsistent")
    block_m = 32
    block_ci = 64
    block_co = 128
    common_constants: dict[str, int | bool | str] = {
        "XH": output_h,
        "XW": output_w,
        "LOSS_H": loss_h,
        "LOSS_W": loss_w,
        "CIN_PER_GROUP": c_in,
        "COUT_PER_GROUP": c_out,
        "loss_stride_n": loss["strides"][0],
        "loss_stride_c": loss["strides"][1],
        "loss_stride_h": loss["strides"][2],
        "loss_stride_w": loss["strides"][3],
        "out_stride_n": output["strides"][0],
        "out_stride_c": output["strides"][1],
        "out_stride_h": output["strides"][2],
        "out_stride_w": output["strides"][3],
        "INPUT_PRECISION": 1 if loss["data_type"] == "float32" else 0,
        "FILTER_REVERSE": (
            _require_integer(
                parameters,
                "convolution_mode",
                minimum=0,
                maximum=1,
            )
            == 1
        ),
        "BLOCK_M": block_m,
        "BLOCK_CI": block_ci,
        "BLOCK_CO": block_co,
    }
    pointer_signature = {
        "loss_ptr": TRITON_POINTER_TYPES[loss["data_type"]],
        "weight_ptr": TRITON_POINTER_TYPES[packed["data_type"]],
        "out_ptr": TRITON_POINTER_TYPES[output["data_type"]],
    }
    pointer_abi = [("tensor", None), ("tensor", None), ("tensor", None)]

    if stage == "compute_p5_splitk":
        parity_h = _require_integer(
            parameters, "_dgrad_parity_h", minimum=0, maximum=1
        )
        if (
            loss["data_type"] != "float32"
            or loss["dimensions"] != [1, 768, 20, 20]
            or packed["dimensions"] != [3, 3, 768, 768]
            or output["dimensions"] != [1, 768, 40, 40]
            or parameters.get("convolution_mode") != 0
        ):
            raise ValueError("the P5 DGrad split-K shape is invalid")
        block_m = 32
        block_ci = 64
        block_co = 64
        group_k = 2
        m = loss_h * loss_w
        split_k_blocks = _ceil_div(_ceil_div(c_out, block_co), group_k)
        return (
            "conv_dgrad2d_p5_fp32_tile2w_splitk_kernel",
            pointer_signature,
            {
                "M": m,
                "XW": output_w,
                "LOSS_H": loss_h,
                "LOSS_W": loss_w,
                "CIN_PER_GROUP": c_in,
                "COUT_PER_GROUP": c_out,
                "loss_stride_c": loss["strides"][1],
                "loss_stride_h": loss["strides"][2],
                "loss_stride_w": loss["strides"][3],
                "out_stride_c": output["strides"][1],
                "out_stride_h": output["strides"][2],
                "out_stride_w": output["strides"][3],
                "PH": parity_h,
                "GROUP_K": group_k,
                "BLOCK_M": block_m,
                "BLOCK_CI": block_ci,
                "BLOCK_CO": block_co,
            },
            (
                _ceil_div(m, block_m)
                * _ceil_div(c_in, block_ci)
                * split_k_blocks,
                1,
                1,
            ),
            pointer_abi,
        )

    if stage == "compute_tile4":
        m = n * loss_h * loss_w
        return (
            "conv_dgrad2d_stride2_pad1_3x3_packed_tile4_kernel",
            pointer_signature,
            {
                **common_constants,
                "M": m,
                "ROUND_TF32": parameters.get("_dgrad_round_tf32") is True,
            },
            (
                _ceil_div(m, block_m) * _ceil_div(c_in, block_ci),
                1,
                1,
            ),
            pointer_abi,
        )

    if stage == "compute_tile2w":
        parity_h = _require_integer(
            parameters, "_dgrad_parity_h", minimum=0, maximum=1
        )
        parity_h_count = (output_h + 1 - parity_h) // 2
        m = n * parity_h_count * loss_w
        return (
            "conv_dgrad2d_stride2_pad1_3x3_packed_tile2w_kernel",
            pointer_signature,
            {
                **common_constants,
                "M": m,
                "PARITY_H_COUNT": parity_h_count,
                "PH": parity_h,
            },
            (
                _ceil_div(m, block_m) * _ceil_div(c_in, block_ci),
                1,
                1,
            ),
            pointer_abi,
        )

    parity_h = _require_integer(
        parameters, "_dgrad_parity_h", minimum=0, maximum=1
    )
    parity_w = _require_integer(
        parameters, "_dgrad_parity_w", minimum=0, maximum=1
    )
    parity_h_count = (output_h + 1 - parity_h) // 2
    parity_w_count = (output_w + 1 - parity_w) // 2
    m = n * parity_h_count * parity_w_count
    return (
        "conv_dgrad2d_stride2_pad1_3x3_packed_parity_kernel",
        pointer_signature,
        {
            **common_constants,
            "M": m,
            "PARITY_H_COUNT": parity_h_count,
            "PARITY_W_COUNT": parity_w_count,
            "PH": parity_h,
            "PW": parity_w,
            "KH_COUNT": 1 if parity_h == 0 else 2,
            "KW_COUNT": 1 if parity_w == 0 else 2,
            "WEIGHT_STRIDE_HW": packed["strides"][1],
            "WEIGHT_STRIDE_CO": packed["strides"][2],
            "WEIGHT_STRIDE_CI": packed["strides"][3],
        },
        (
            _ceil_div(m, block_m) * _ceil_div(c_in, block_ci),
            1,
            1,
        ),
        pointer_abi,
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
        raise ValueError(
            "convolution backward tensors must use one floating data type"
        )
    pointer_type = TRITON_POINTER_TYPES[data_types[0]]
    spatial_rank = _require_integer(
        parameters, "spatial_rank", minimum=1, maximum=3
    )
    tensor_rank = spatial_rank + 2
    if any(
        len(tensor["dimensions"]) != tensor_rank
        or any(dimension > 2**31 - 1 for dimension in tensor["dimensions"])
        or not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        )
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
    stride = _require_integer_list(
        parameters, "stride", spatial_rank, minimum=1
    )
    dilation = _require_integer_list(
        parameters, "dilation", spatial_rank, minimum=1
    )
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
        raise ValueError(
            "convolution backward loss batch/channels are inconsistent"
        )
    if c % groups != 0 or k % groups != 0:
        raise ValueError("convolution backward channels must divide groups")
    channels_per_group = c // groups
    outputs_per_group = k // groups
    if filter_channels != channels_per_group:
        raise ValueError(
            "convolution backward filter channels are inconsistent"
        )
    expected_loss = [n, k]
    for axis in range(spatial_rank):
        image_extent = image["dimensions"][axis + 2]
        filter_extent = filter_tensor["dimensions"][axis + 2]
        effective_filter = dilation[axis] * (filter_extent - 1) + 1
        padded_image = image_extent + pre_padding[axis] + post_padding[axis]
        if padded_image < effective_filter:
            raise ValueError(
                "convolution backward filter is larger than padded image"
            )
        expected_loss.append(
            (padded_image - effective_filter) // stride[axis] + 1
        )
    if dy["dimensions"] != expected_loss:
        raise ValueError("convolution backward loss metadata is inconsistent")
    expected_output = (
        image["dimensions"]
        if operation == "convolution_dgrad"
        else filter_tensor["dimensions"]
    )
    if math.prod(expected_output) != outputs:
        raise ValueError(
            "parameters.n_outputs is inconsistent with backward output"
        )

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
                    "INPUT_PRECISION": (
                        1 if data_types[0] == "float32" else 0
                    ),
                    "BLOCK_M": block_m,
                    "BLOCK_CI": block_ci,
                    "BLOCK_CO": block_co,
                },
                (
                    _ceil_div(hw, block_m)
                    * _ceil_div(channels_per_group, block_ci),
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
    block_oc = 16 if outputs_per_group <= 16 else 32
    block_ci = 16 if channels_per_group <= 16 else 32
    block_m = 32
    constants.update(
        {
            "M": m,
            "BLOCK_OC": block_oc,
            "BLOCK_CI": block_ci,
            "BLOCK_M": block_m,
        }
    )
    return (
        "conv_wgrad_nd_kernel",
        {
            "dy_ptr": pointer_type,
            "x_ptr": pointer_type,
            "dw_ptr": pointer_type,
        },
        constants,
        (
            ((outputs_per_group + block_oc - 1) // block_oc)
            * ((channels_per_group + block_ci - 1) // block_ci),
            kernel_volume,
            groups,
        ),
        pointer_abi,
    )
