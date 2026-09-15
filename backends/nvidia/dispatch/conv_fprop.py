"""Forward-convolution validation and individual kernel configurations."""

from __future__ import annotations

from typing import Any
import math

from .common import (
    TRITON_POINTER_TYPES,
    _ceil_div,
    _has_non_overlapping_strides,
    _is_row_major_contiguous,
    _require_integer,
    _require_integer_list,
)


def _convolution_im2col_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    """Configure the first stage of the NVIDIA stride-2 FProp pipeline."""
    if len(tensors) != 2:
        raise ValueError("the FProp im2col stage requires two tensors")
    input_tensor, columns = tensors
    data_type_pair = (
        input_tensor["data_type"],
        columns["data_type"],
    )
    if (
        data_type_pair
        not in {
            ("float16", "float16"),
            ("bfloat16", "bfloat16"),
            ("float32", "float32"),
        }
        or len(input_tensor["dimensions"]) != 4
        or len(columns["dimensions"]) != 3
        or not _is_row_major_contiguous(input_tensor)
        or not _has_non_overlapping_strides(
            columns["dimensions"], columns["strides"]
        )
        or columns["strides"][-1] != 1
    ):
        raise ValueError("the FProp im2col tensor metadata is invalid")
    if (
        _require_integer(parameters, "spatial_rank", minimum=1, maximum=3) != 2
        or _require_integer(parameters, "groups") != 1
        or _require_integer_list(parameters, "stride", 2, minimum=1) != [2, 2]
        or _require_integer_list(parameters, "pre_padding", 2, minimum=0)
        != [1, 1]
        or _require_integer_list(parameters, "post_padding", 2, minimum=0)
        != [1, 1]
        or _require_integer_list(parameters, "dilation", 2, minimum=1)
        != [1, 1]
    ):
        raise ValueError("the FProp im2col convolution contract is invalid")

    n, channels, input_h, input_w = input_tensor["dimensions"]
    output_h = (input_h + 1) // 2
    output_w = (input_w + 1) // 2
    output_area = output_h * output_w
    reduction_extent = channels * 9
    if (
        columns["dimensions"][:2] != [n, reduction_extent]
        or columns["dimensions"][2] < output_area
        or columns["strides"]
        != [
            reduction_extent * columns["dimensions"][2],
            columns["dimensions"][2],
            1,
        ]
    ):
        raise ValueError("the FProp im2col workspace shape is invalid")

    total = n * columns["strides"][0]
    block_size = 1024
    constants = {
        "TOTAL": total,
        "XH": input_h,
        "XW": input_w,
        "OH": output_h,
        "OW": output_w,
        "CIN_PER_GROUP": channels,
        "X_STRIDE_N": input_tensor["strides"][0],
        "X_STRIDE_C": input_tensor["strides"][1],
        "X_STRIDE_H": input_tensor["strides"][2],
        "X_STRIDE_W": input_tensor["strides"][3],
        "COL_STRIDE_N": columns["strides"][0],
        "COL_STRIDE_K": columns["strides"][1],
        "BLOCK_SIZE": block_size,
    }
    return (
        "conv2d_im2col_nchw_3x3_stride2_pad1_kernel",
        {
            "x_ptr": TRITON_POINTER_TYPES[input_tensor["data_type"]],
            "col_ptr": TRITON_POINTER_TYPES[columns["data_type"]],
        },
        constants,
        (_ceil_div(output_area, block_size), n * channels * 3, 1),
        [("tensor", None), ("tensor", None)],
    )


def _convolution_general_im2col_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | bool],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    """Configure the materialization stage for the general 2D FProp path."""
    if len(tensors) != 2:
        raise ValueError("the general FProp im2col stage requires 2 tensors")
    input_tensor, columns = tensors
    if (
        input_tensor["data_type"] not in {"float32", "float16", "bfloat16"}
        or columns["data_type"] != input_tensor["data_type"]
        or len(input_tensor["dimensions"]) != 4
        or len(columns["dimensions"]) != 3
        or not _is_row_major_contiguous(input_tensor)
    ):
        raise ValueError("the general FProp im2col tensor metadata is invalid")
    if (
        _require_integer(parameters, "spatial_rank", minimum=1, maximum=3) != 2
        or _require_integer(parameters, "groups") != 1
    ):
        raise ValueError(
            "the general FProp im2col stage requires 2D group-one convolution"
        )

    stride = _require_integer_list(parameters, "stride", 2, minimum=1)
    pre_padding = _require_integer_list(
        parameters, "pre_padding", 2, minimum=0
    )
    post_padding = _require_integer_list(
        parameters, "post_padding", 2, minimum=0
    )
    dilation = _require_integer_list(parameters, "dilation", 2, minimum=1)
    n, channels, input_h, input_w = input_tensor["dimensions"]
    _, filter_channels, kernel_h, kernel_w = _require_integer_list(
        parameters, "_fprop_filter_dimensions", 4, minimum=1
    )
    if filter_channels != channels:
        raise ValueError("the general FProp filter channel count is invalid")
    output_h = (
        input_h
        + pre_padding[0]
        + post_padding[0]
        - dilation[0] * (kernel_h - 1)
        - 1
    ) // stride[0] + 1
    output_w = (
        input_w
        + pre_padding[1]
        + post_padding[1]
        - dilation[1] * (kernel_w - 1)
        - 1
    ) // stride[1] + 1
    output_area = output_h * output_w
    reduction_extent = channels * kernel_h * kernel_w
    if columns["dimensions"] != [n, reduction_extent, output_area]:
        raise ValueError("the general FProp im2col workspace shape is invalid")
    transposed_columns = columns["data_type"] == "float32" and columns[
        "strides"
    ] == [
        reduction_extent * output_area,
        1,
        reduction_extent,
    ]
    if not transposed_columns and not _is_row_major_contiguous(columns):
        raise ValueError(
            "the general FProp im2col workspace strides are invalid"
        )
    block_size = 1024
    pointer_abi: list[tuple[str, str | int | None]] = [
        ("tensor", None),
        ("tensor", None),
    ]
    constants = {
        "XH": input_h,
        "XW": input_w,
        "OH": output_h,
        "OW": output_w,
        "CIN_PER_GROUP": channels,
        "KH": kernel_h,
        "KW": kernel_w,
        "STRIDE_H": stride[0],
        "STRIDE_W": stride[1],
        "PAD_TOP": pre_padding[0],
        "PAD_LEFT": pre_padding[1],
        "DIL_H": dilation[0],
        "DIL_W": dilation[1],
        "X_STRIDE_N": input_tensor["strides"][0],
        "X_STRIDE_C": input_tensor["strides"][1],
        "X_STRIDE_H": input_tensor["strides"][2],
        "X_STRIDE_W": input_tensor["strides"][3],
        "COL_STRIDE_N": columns["strides"][0],
        "COL_STRIDE_K": columns["strides"][1],
        "BLOCK_SIZE": block_size,
    }
    function = "conv2d_im2col_nchw_kernel"
    grid = (_ceil_div(output_area, block_size), n * channels * kernel_h, 1)
    if transposed_columns:
        function = "conv2d_im2col_nchw_transposed_kernel"
        for name in ("COL_STRIDE_K", "BLOCK_SIZE"):
            del constants[name]
        constants.update(BLOCK_M=32, BLOCK_K=32)
        grid = (_ceil_div(output_area, 32), _ceil_div(reduction_extent, 32), n)
    return (
        function,
        {
            "x_ptr": TRITON_POINTER_TYPES[input_tensor["data_type"]],
            "col_ptr": TRITON_POINTER_TYPES[columns["data_type"]],
        },
        constants,
        grid,
        pointer_abi,
    )


def _convolution_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | float | str | bool],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    fused_bias_relu = parameters.get("_fused_bias_relu", False)
    if not isinstance(fused_bias_relu, bool):
        raise ValueError("internal convolution fusion flag must be boolean")
    expected_tensor_count = 4 if fused_bias_relu else 3
    tensor_data_types = [tensor["data_type"] for tensor in tensors]
    if (
        len(tensor_data_types) != expected_tensor_count
        or len(set(tensor_data_types)) != 1
    ):
        raise ValueError("convolution FProp tensor data types must match")
    pointer_type = TRITON_POINTER_TYPES.get(tensor_data_types[0])
    if pointer_type is None:
        raise ValueError(
            "unsupported convolution FProp data type: "
            f"{tensor_data_types[0]!r}"
        )

    spatial_rank = _require_integer(
        parameters, "spatial_rank", minimum=1, maximum=3
    )
    tensor_rank = spatial_rank + 2
    if any(len(tensor["dimensions"]) != tensor_rank for tensor in tensors):
        raise ValueError(
            "convolution FProp tensor rank must equal spatial_rank + 2"
        )
    if any(
        dimension > 2**31 - 1
        for tensor in tensors
        for dimension in tensor["dimensions"]
    ):
        raise ValueError("convolution FProp tensor dimensions are too large")
    if any(
        not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        )
        for tensor in tensors
    ):
        raise ValueError(
            "convolution FProp tensors must have non-overlapping strides"
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
    outputs = _require_integer(parameters, "n_outputs")

    input_dimensions = tensors[0]["dimensions"]
    filter_dimensions = tensors[1]["dimensions"]
    output_tensor = tensors[-1]
    output_dimensions = output_tensor["dimensions"]
    n, c = input_dimensions[:2]
    k, filter_channels = filter_dimensions[:2]
    if c % groups != 0 or k % groups != 0:
        raise ValueError("convolution FProp channels must divide groups")
    channels_per_group = c // groups
    outputs_per_group = k // groups
    if filter_channels != channels_per_group:
        raise ValueError(
            "convolution FProp filter channels do not match input"
        )
    if fused_bias_relu:
        expected_bias_dimensions = [1, k] + [1] * spatial_rank
        if tensors[2]["dimensions"] != expected_bias_dimensions:
            raise ValueError(
                "fused convolution bias must have shape [1, K, 1, ...]"
            )

    expected_output = [n, k]
    reduction_extent = channels_per_group
    for axis in range(spatial_rank):
        input_extent = input_dimensions[axis + 2]
        filter_extent = filter_dimensions[axis + 2]
        effective_filter = dilation[axis] * (filter_extent - 1) + 1
        padded_input = input_extent + pre_padding[axis] + post_padding[axis]
        if padded_input < effective_filter:
            raise ValueError(
                "convolution FProp filter is larger than padded input"
            )
        expected_output.append(
            (padded_input - effective_filter) // stride[axis] + 1
        )
        reduction_extent *= filter_extent
    if output_dimensions != expected_output:
        raise ValueError("convolution FProp output metadata is inconsistent")
    if math.prod(output_dimensions) != outputs:
        raise ValueError("parameters.n_outputs is inconsistent with shape")
    if reduction_extent > 65536:
        raise ValueError("convolution FProp reduction extent exceeds limit")

    pointer_signature = {
        "x_ptr": pointer_type,
        "w_ptr": pointer_type,
        "bias_ptr": pointer_type,
        "y_ptr": pointer_type,
    }
    if fused_bias_relu:
        pointer_abi: list[tuple[str, str | int | None]] = [
            ("tensor", None),
            ("tensor", None),
            ("tensor", None),
            ("tensor", None),
        ]
    else:
        pointer_abi = [
            ("tensor", None),
            ("tensor", None),
            ("tensor_alias", -1),
            ("tensor", None),
        ]

    if spatial_rank == 1:
        input_l = input_dimensions[2]
        kernel_w = filter_dimensions[2]
        output_l = output_dimensions[2]
        m = n * output_l
        block_m = 32
        block_oc = 16 if outputs_per_group <= 16 else 32
        block_k = 16 if reduction_extent <= 16 else 32
        constants: dict[str, int | str | bool] = {
            "M": m,
            "XL": input_l,
            "OL": output_l,
            "DTYPE_ID": {
                "float16": 0,
                "bfloat16": 1,
                "float32": 2,
            }[tensor_data_types[0]],
            "x_stride_n": tensors[0]["strides"][0],
            "x_stride_c": tensors[0]["strides"][1],
            "x_stride_l": tensors[0]["strides"][2],
            "w_stride_o": tensors[1]["strides"][0],
            "w_stride_i": tensors[1]["strides"][1],
            "w_stride_k": tensors[1]["strides"][2],
            "bias_stride": tensors[2]["strides"][1] if fused_bias_relu else 0,
            "y_stride_n": output_tensor["strides"][0],
            "y_stride_c": output_tensor["strides"][1],
            "y_stride_l": output_tensor["strides"][2],
            "CIN_PER_GROUP": channels_per_group,
            "COUT_PER_GROUP": outputs_per_group,
            "KW": kernel_w,
            "STRIDE_W": stride[0],
            "PAD_LEFT": pre_padding[0],
            "DIL_W": dilation[0],
            "HAS_BIAS": fused_bias_relu,
            "APPLY_RELU": fused_bias_relu,
            "BLOCK_M": block_m,
            "BLOCK_OC": block_oc,
            "BLOCK_K": block_k,
            "GROUP_M": 8,
            "INPUT_PRECISION": 1 if tensor_data_types[0] == "float32" else 0,
        }
        return (
            "conv1d_gemm_kernel",
            pointer_signature,
            constants,
            (
                ((m + block_m - 1) // block_m)
                * ((outputs_per_group + block_oc - 1) // block_oc),
                groups,
                1,
            ),
            pointer_abi,
        )

    if spatial_rank == 2:
        input_h, input_w = input_dimensions[2:]
        kernel_h, kernel_w = filter_dimensions[2:]
        output_h, output_w = output_dimensions[2:]
        block_oc = 16
        block_hw = 16
        block_k = 16
        use_nchw_1x1_pad0 = (
            kernel_h == 1
            and kernel_w == 1
            and stride == [1, 1]
            and pre_padding == [0, 0]
            and post_padding == [0, 0]
            and dilation == [1, 1]
            and _is_row_major_contiguous(tensors[0])
            and _is_row_major_contiguous(tensors[1])
            and _is_row_major_contiguous(output_tensor)
        )
        if use_nchw_1x1_pad0:
            output_area = output_h * output_w
            constants = {
                "HW": output_area,
                "C_IN": c,
                "C_OUT": k,
                "CIN_PER_GROUP": channels_per_group,
                "COUT_PER_GROUP": outputs_per_group,
                "GROUPS": groups,
                "HAS_BIAS": fused_bias_relu,
                "APPLY_RELU": fused_bias_relu,
                "BIAS_STRIDE": (
                    tensors[2]["strides"][1] if fused_bias_relu else 0
                ),
                "BLOCK_OC": block_oc,
                "BLOCK_HW": block_hw,
                "BLOCK_K": block_k,
                "GROUP_M": 8,
                "DTYPE_ID": {
                    "float16": 0,
                    "bfloat16": 1,
                    "float32": 2,
                }[tensor_data_types[0]],
                "INPUT_PRECISION": (
                    1 if tensor_data_types[0] == "float32" else 0
                ),
            }
            return (
                "conv2d_1x1_nchw_pad0_kernel",
                pointer_signature,
                constants,
                (
                    ((output_area + block_hw - 1) // block_hw)
                    * ((outputs_per_group + block_oc - 1) // block_oc),
                    n * groups,
                    1,
                ),
                pointer_abi,
            )
        constants = {
            "XH": input_h,
            "XW": input_w,
            "OH": output_h,
            "OW": output_w,
            "C_IN": c,
            "C_OUT": k,
            "CIN_PER_GROUP": channels_per_group,
            "COUT_PER_GROUP": outputs_per_group,
            "GROUPS": groups,
            "STRIDE_H": stride[0],
            "STRIDE_W": stride[1],
            "PAD_TOP": pre_padding[0],
            "PAD_LEFT": pre_padding[1],
            "DIL_H": dilation[0],
            "DIL_W": dilation[1],
            "KH": kernel_h,
            "KW": kernel_w,
            "HAS_BIAS": fused_bias_relu,
            "APPLY_RELU": fused_bias_relu,
            "BIAS_STRIDE": tensors[2]["strides"][1] if fused_bias_relu else 0,
            "BLOCK_OC": block_oc,
            "BLOCK_HW": block_hw,
            "BLOCK_K": block_k,
            "GROUP_M": 8,
            "DTYPE_ID": {
                "float16": 0,
                "bfloat16": 1,
                "float32": 2,
            }[tensor_data_types[0]],
            "INPUT_PRECISION": 1 if tensor_data_types[0] == "float32" else 0,
            "X_STRIDE_N": tensors[0]["strides"][0],
            "X_STRIDE_C": tensors[0]["strides"][1],
            "X_STRIDE_H": tensors[0]["strides"][2],
            "X_STRIDE_W": tensors[0]["strides"][3],
            "W_STRIDE_K": tensors[1]["strides"][0],
            "W_STRIDE_C": tensors[1]["strides"][1],
            "W_STRIDE_R": tensors[1]["strides"][2],
            "W_STRIDE_S": tensors[1]["strides"][3],
            "Y_STRIDE_N": output_tensor["strides"][0],
            "Y_STRIDE_C": output_tensor["strides"][1],
            "Y_STRIDE_H": output_tensor["strides"][2],
            "Y_STRIDE_W": output_tensor["strides"][3],
        }
        return (
            "conv2d_spatial_nchw_kernel",
            pointer_signature,
            constants,
            (
                ((output_h * output_w + block_hw - 1) // block_hw)
                * ((outputs_per_group + block_oc - 1) // block_oc),
                n * groups,
                1,
            ),
            pointer_abi,
        )

    input_d, input_h, input_w = input_dimensions[2:]
    kernel_d, kernel_h, kernel_w = filter_dimensions[2:]
    output_d, output_h, output_w = output_dimensions[2:]
    m = n * output_d * output_h * output_w
    block_oc = 16 if outputs_per_group <= 16 else 32
    block_m = 32
    block_k = 32
    constants = {
        "M": m,
        "XD": input_d,
        "XH": input_h,
        "XW": input_w,
        "OD": output_d,
        "OH": output_h,
        "OW": output_w,
        "C_IN": c,
        "C_OUT": k,
        "CIN_PER_GROUP": channels_per_group,
        "COUT_PER_GROUP": outputs_per_group,
        "STRIDE_D": stride[0],
        "STRIDE_H": stride[1],
        "STRIDE_W": stride[2],
        "PAD_FRONT": pre_padding[0],
        "PAD_TOP": pre_padding[1],
        "PAD_LEFT": pre_padding[2],
        "DIL_D": dilation[0],
        "DIL_H": dilation[1],
        "DIL_W": dilation[2],
        "KD": kernel_d,
        "KH": kernel_h,
        "KW": kernel_w,
        "HAS_BIAS": fused_bias_relu,
        "APPLY_RELU": fused_bias_relu,
        "BIAS_STRIDE": tensors[2]["strides"][1] if fused_bias_relu else 0,
        "BLOCK_OC": block_oc,
        "BLOCK_M": block_m,
        "BLOCK_K": block_k,
        "GROUP_M": 8,
        "DTYPE_ID": {
            "float16": 0,
            "bfloat16": 1,
            "float32": 2,
        }[tensor_data_types[0]],
        "X_STRIDE_N": tensors[0]["strides"][0],
        "X_STRIDE_C": tensors[0]["strides"][1],
        "X_STRIDE_D": tensors[0]["strides"][2],
        "X_STRIDE_H": tensors[0]["strides"][3],
        "X_STRIDE_W": tensors[0]["strides"][4],
        "W_STRIDE_K": tensors[1]["strides"][0],
        "W_STRIDE_C": tensors[1]["strides"][1],
        "W_STRIDE_D": tensors[1]["strides"][2],
        "W_STRIDE_H": tensors[1]["strides"][3],
        "W_STRIDE_W": tensors[1]["strides"][4],
        "Y_STRIDE_N": output_tensor["strides"][0],
        "Y_STRIDE_C": output_tensor["strides"][1],
        "Y_STRIDE_D": output_tensor["strides"][2],
        "Y_STRIDE_H": output_tensor["strides"][3],
        "Y_STRIDE_W": output_tensor["strides"][4],
        "INPUT_PRECISION": 1 if tensor_data_types[0] == "float32" else 0,
    }
    return (
        "conv3d_spatial_ncdhw_m_kernel",
        pointer_signature,
        constants,
        (
            ((m + block_m - 1) // block_m)
            * ((outputs_per_group + block_oc - 1) // block_oc),
            groups,
            1,
        ),
        pointer_abi,
    )
