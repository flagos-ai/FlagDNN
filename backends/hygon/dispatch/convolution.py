# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Hygon dispatch convolution."""

from __future__ import annotations

from .common import require_ieee_precision
from ..dispatch.nn_common import (
    ALLOW_TF32,
    CONV_DGRAD_EXACT_COL2IM_TUNING,
    CONV_DGRAD_EXACT_GEMM_TUNING,
    CONV_DGRAD_TUNING,
    CONV_FPROP_STANDARD_3X3_TUNING,
    CONV_FPROP_TUNING,
    CONV_FPROP_YOLO_X_P5_GEMM_TUNING,
    CONV_TUNING_BY_OPERATION,
    CONV_WGRAD_TUNING,
    DGRAD_EXACT_3X3_S1_WORKSPACE_CAP,
    DTYPE_IDS,
    FLOAT_TYPES,
    GridSpec,
    KernelStagePlan,
    MAX_I32,
    MAX_I64,
    NodePlan,
    PointerArgument,
    WORKSPACE_ALIGNMENT,
    _aligned_size,
    _ceil_div,
    _checked_product,
    _is_contiguous,
    _make_stage,
    _private_workspace,
    _require_integer,
    _require_integer_list,
    _require_object,
    _same_dtype,
    _tensor_pointer,
    _workspace_pointer,
)
from typing import Any
from typing import Mapping
from typing import Sequence


def _convolution_output_dimension(
    input_size: int,
    filter_size: int,
    pre_padding: int,
    post_padding: int,
    stride: int,
    dilation: int,
) -> int:
    effective = (filter_size - 1) * dilation + 1
    padded = input_size + pre_padding + post_padding
    if padded < effective:
        raise ValueError("convolution filter is larger than padded input")
    return (padded - effective) // stride + 1


def _validate_convolution(node: Mapping[str, Any]) -> dict[str, Any]:
    operation = str(node["operation"])
    p = _require_object(node["parameters"], "node.parameters")
    ports = _require_object(node["port_tensors"], "node.port_tensors")
    spatial_rank = _require_integer(p, "spatial_rank", minimum=1, maximum=3)
    rank = spatial_rank + 2
    pre = _require_integer_list(p, "pre_padding", spatial_rank, minimum=0)
    post = _require_integer_list(p, "post_padding", spatial_rank, minimum=0)
    stride = _require_integer_list(p, "stride", spatial_rank, minimum=1)
    dilation = _require_integer_list(p, "dilation", spatial_rank, minimum=1)
    groups = _require_integer(p, "groups", minimum=1, maximum=MAX_I32)

    if operation in ("conv2d_fprop", "convolution_fprop"):
        image, weight, result = (
            ports["input"],
            ports["filter"],
            ports["output"],
        )
        if "convolution_mode" in p:
            mode = _require_integer(
                p, "convolution_mode", minimum=0, maximum=1
            )
            if mode != 0:
                raise ValueError(
                    "convolution FProp supports CROSS_CORRELATION only"
                )
        flip_filter = False
    elif operation == "convolution_dgrad":
        result, weight, image = ports["dy"], ports["w"], ports["dx"]
        flip_filter = bool(
            _require_integer(p, "convolution_mode", minimum=0, maximum=1)
        )
    else:
        result, image, weight = ports["dy"], ports["x"], ports["dw"]
        flip_filter = bool(
            _require_integer(p, "convolution_mode", minimum=0, maximum=1)
        )

    tensors = (image, weight, result)
    data_type = _same_dtype(tensors, FLOAT_TYPES, "convolution")
    require_ieee_precision(p, data_type)
    if any(len(tensor["dimensions"]) != rank for tensor in tensors):
        raise ValueError(
            "convolution tensor ranks must equal spatial_rank + 2"
        )
    image_dims = list(image["dimensions"])
    weight_dims = list(weight["dimensions"])
    result_dims = list(result["dimensions"])
    batch, in_channels = image_dims[:2]
    out_channels = weight_dims[0]
    if in_channels % groups or out_channels % groups:
        raise ValueError("convolution channels must be divisible by groups")
    in_per_group = in_channels // groups
    out_per_group = out_channels // groups
    if weight_dims[1] != in_per_group:
        raise ValueError("convolution filter channels do not match image")
    expected_result = [batch, out_channels]
    for axis in range(spatial_rank):
        expected_result.append(
            _convolution_output_dimension(
                image_dims[axis + 2],
                weight_dims[axis + 2],
                pre[axis],
                post[axis],
                stride[axis],
                dilation[axis],
            )
        )
    if result_dims != expected_result:
        raise ValueError("convolution output/loss shape is incorrect")
    n_outputs = _require_integer(p, "n_outputs", maximum=MAX_I64)
    expected_n_outputs = _checked_product(
        list(
            node["port_tensors"][
                (
                    "dx"
                    if operation == "convolution_dgrad"
                    else "dw" if operation == "convolution_wgrad" else "output"
                )
            ]["dimensions"]
        ),
        "convolution output elements",
    )
    if n_outputs != expected_n_outputs:
        raise ValueError(
            "parameters.n_outputs is inconsistent with the output tensor"
        )
    return {
        "data_type": data_type,
        "spatial_rank": spatial_rank,
        "pre_padding": pre,
        "post_padding": post,
        "stride": stride,
        "dilation": dilation,
        "groups": groups,
        "flip_filter": flip_filter,
        "image": image,
        "weight": weight,
        "result": result,
        "batch": batch,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "in_per_group": in_per_group,
        "out_per_group": out_per_group,
        "n_outputs": n_outputs,
    }


def _padded_nd(
    tensor: Mapping[str, Any], spatial_rank: int
) -> tuple[list[int], list[int]]:
    dimensions = list(tensor["dimensions"])
    strides = list(tensor["strides"])
    leading = 3 - spatial_rank
    return (
        dimensions[:2] + [1] * leading + dimensions[2:],
        strides[:2] + [0] * leading + strides[2:],
    )


def _padded_spatial(values: Sequence[int], fill: int) -> list[int]:
    return [fill] * (3 - len(values)) + list(values)


def _convolution_stage(node: Mapping[str, Any]) -> KernelStagePlan:
    operation = str(node["operation"])
    d = _require_object(node["derived"], "node.derived")
    spatial_rank = int(d["spatial_rank"])
    image = d["image"]
    weight = d["weight"]
    result = d["result"]
    image_dims, image_strides = _padded_nd(image, spatial_rank)
    weight_dims, weight_strides = _padded_nd(weight, spatial_rank)
    result_dims, result_strides = _padded_nd(result, spatial_rank)
    _, _, xd, xh, xw = image_dims
    _, _, kd, kh, kw = weight_dims
    _, _, od, oh, ow = result_dims
    stride_d, stride_h, stride_w = _padded_spatial(d["stride"], 1)
    pad_front, pad_top, pad_left = _padded_spatial(d["pre_padding"], 0)
    dil_d, dil_h, dil_w = _padded_spatial(d["dilation"], 1)
    common: dict[str, int | float | bool] = {
        "XD": xd,
        "XH": xh,
        "XW": xw,
        "OD": od,
        "OH": oh,
        "OW": ow,
        "KD": kd,
        "KH": kh,
        "KW": kw,
        "CIN_PER_GROUP": int(d["in_per_group"]),
        "COUT_PER_GROUP": int(d["out_per_group"]),
        "STRIDE_D": stride_d,
        "STRIDE_H": stride_h,
        "STRIDE_W": stride_w,
        "PAD_FRONT": pad_front,
        "PAD_TOP": pad_top,
        "PAD_LEFT": pad_left,
        "DIL_D": dil_d,
        "DIL_H": dil_h,
        "DIL_W": dil_w,
        "INPUT_PRECISION": int(ALLOW_TF32),
    }
    for prefix, strides, axes in (
        ("X", image_strides, ("N", "C", "D", "H", "W")),
        ("W", weight_strides, ("K", "C", "D", "H", "W")),
        ("Y", result_strides, ("N", "C", "D", "H", "W")),
    ):
        for axis, value in zip(axes, strides, strict=True):
            common[f"{prefix}_STRIDE_{axis}"] = value

    if operation in ("conv2d_fprop", "convolution_fprop"):
        pointers: tuple[PointerArgument, ...] = (
            _tensor_pointer(node, "x_ptr", "input"),
            _tensor_pointer(node, "w_ptr", "filter"),
            # Graph FProp has no bias port. HAS_BIAS=False makes this a safe
            # placeholder, while retaining the registry kernel's exact ABI.
            _tensor_pointer(node, "bias_ptr", "input"),
            _tensor_pointer(node, "y_ptr", "output"),
        )
        if spatial_rank == 1:
            constants = {
                "M": int(d["batch"]) * ow,
                "XL": xw,
                "OL": ow,
                "DTYPE_ID": DTYPE_IDS[str(d["data_type"])],
                "x_stride_n": image["strides"][0],
                "x_stride_c": image["strides"][1],
                "x_stride_l": image["strides"][2],
                "w_stride_o": weight["strides"][0],
                "w_stride_i": weight["strides"][1],
                "w_stride_k": weight["strides"][2],
                "bias_stride": 1,
                "y_stride_n": result["strides"][0],
                "y_stride_c": result["strides"][1],
                "y_stride_l": result["strides"][2],
                "CIN_PER_GROUP": int(d["in_per_group"]),
                "COUT_PER_GROUP": int(d["out_per_group"]),
                "KW": kw,
                "STRIDE_W": stride_w,
                "PAD_LEFT": pad_left,
                "DIL_W": dil_w,
                "HAS_BIAS": False,
                "BLOCK_M": 32,
                "BLOCK_OC": 32,
                "BLOCK_K": 32,
                "GROUP_M": 8,
                "INPUT_PRECISION": int(ALLOW_TF32),
            }
            function = "conv1d_gemm_kernel"
            grid = GridSpec(
                "conv1d",
                (
                    int(d["batch"]) * ow,
                    int(d["out_per_group"]),
                    int(d["groups"]),
                ),
            )
        elif spatial_rank == 2:
            constants = {
                "XH": xh,
                "XW": xw,
                "OH": oh,
                "OW": ow,
                "C_IN": int(d["in_channels"]),
                "C_OUT": int(d["out_channels"]),
                "CIN_PER_GROUP": int(d["in_per_group"]),
                "COUT_PER_GROUP": int(d["out_per_group"]),
                "GROUPS": int(d["groups"]),
                "STRIDE_H": stride_h,
                "STRIDE_W": stride_w,
                "PAD_TOP": pad_top,
                "PAD_LEFT": pad_left,
                "DIL_H": dil_h,
                "DIL_W": dil_w,
                "KH": kh,
                "KW": kw,
                "HAS_BIAS": False,
                "BLOCK_OC": 32,
                "BLOCK_HW": 32,
                "BLOCK_K": 32,
                "GROUP_M": 8,
                "DTYPE_ID": DTYPE_IDS[str(d["data_type"])],
                "INPUT_PRECISION": int(ALLOW_TF32),
                "X_STRIDE_N": image_strides[0],
                "X_STRIDE_C": image_strides[1],
                "X_STRIDE_H": image_strides[3],
                "X_STRIDE_W": image_strides[4],
                "W_STRIDE_K": weight_strides[0],
                "W_STRIDE_C": weight_strides[1],
                "W_STRIDE_R": weight_strides[3],
                "W_STRIDE_S": weight_strides[4],
                "Y_STRIDE_N": result_strides[0],
                "Y_STRIDE_C": result_strides[1],
                "Y_STRIDE_H": result_strides[3],
                "Y_STRIDE_W": result_strides[4],
            }
            function = "conv2d_spatial_nchw_kernel"
            grid = GridSpec(
                "conv2d",
                (
                    oh * ow,
                    int(d["out_per_group"]),
                    int(d["batch"]) * int(d["groups"]),
                ),
            )
        else:
            constants = dict(common)
            constants.update(
                {
                    "M": int(d["batch"]) * od * oh * ow,
                    "C_IN": int(d["in_channels"]),
                    "C_OUT": int(d["out_channels"]),
                    "HAS_BIAS": False,
                    "BLOCK_OC": 32,
                    "BLOCK_M": 32,
                    "BLOCK_K": 32,
                    "GROUP_M": 8,
                }
            )
            function = "conv3d_spatial_ncdhw_m_kernel"
            grid = GridSpec(
                "conv3d",
                (
                    int(d["batch"]) * od * oh * ow,
                    int(d["out_per_group"]),
                    int(d["groups"]),
                ),
            )
    else:
        loss_port = "dy"
        first_port = "w" if operation == "convolution_dgrad" else "x"
        output_port = "dx" if operation == "convolution_dgrad" else "dw"
        pointers = (
            _tensor_pointer(node, "dy_ptr", loss_port),
            _tensor_pointer(
                node,
                "w_ptr" if operation == "convolution_dgrad" else "x_ptr",
                first_port,
            ),
            _tensor_pointer(
                node,
                "dx_ptr" if operation == "convolution_dgrad" else "dw_ptr",
                output_port,
            ),
        )
        # Backward common kernels consistently call the loss tensor DY and
        # image/output tensor X, independent of which one is the graph output.
        constants = dict(common)
        for axis in ("N", "C", "D", "H", "W"):
            constants[f"DY_STRIDE_{axis}"] = constants.pop(f"Y_STRIDE_{axis}")
        constants["FLIP_FILTER"] = bool(d["flip_filter"])
        if operation == "convolution_dgrad":
            constants.update(
                {
                    "M": int(d["batch"]) * xd * xh * xw,
                    "BLOCK_M": 32,
                    "BLOCK_CI": 32,
                    "BLOCK_K": 32,
                    "GROUP_M": 8,
                }
            )
            function = "conv_dgrad_nd_kernel"
            grid = GridSpec(
                "conv_dgrad",
                (
                    int(d["batch"]) * xd * xh * xw,
                    int(d["in_per_group"]),
                    int(d["groups"]),
                ),
            )
        else:
            constants.update(
                {
                    "M": int(d["batch"]) * od * oh * ow,
                    "BLOCK_OC": 32,
                    "BLOCK_CI": 32,
                    "BLOCK_M": 32,
                }
            )
            function = "conv_wgrad_nd_kernel"
            grid = GridSpec(
                "conv_wgrad",
                (
                    int(d["out_per_group"]),
                    int(d["in_per_group"]),
                    kd * kh * kw,
                    int(d["groups"]),
                ),
            )
    return _make_stage(
        operation=operation,
        stage_name=operation,
        function_name=function,
        pointer_arguments=pointers,
        constants=constants,
        tuning=CONV_TUNING_BY_OPERATION[operation],
        tuning_key_value=int(d["n_outputs"]),
        grid_spec=grid,
        num_stages=2,
    )


def _private_fprop_plan(node: Mapping[str, Any]) -> NodePlan | None:
    operation = str(node["operation"])
    d = _require_object(node["derived"], "node.derived")
    if int(d["spatial_rank"]) != 2:
        return None
    image = d["image"]
    weight = d["weight"]
    result = d["result"]
    if not all(_is_contiguous(tensor) for tensor in (image, weight, result)):
        return None
    n, c_in, xh, xw = (int(value) for value in image["dimensions"])
    c_out, _, kh, kw = (int(value) for value in weight["dimensions"])
    _, _, oh, ow = (int(value) for value in result["dimensions"])
    groups = int(d["groups"])
    cin_per_group = int(d["in_per_group"])
    cout_per_group = int(d["out_per_group"])
    stride = list(d["stride"])
    pre = list(d["pre_padding"])
    post = list(d["post_padding"])
    dilation = list(d["dilation"])
    flip_filter = bool(d["flip_filter"])

    standard_3x3_fp32 = (
        str(d["data_type"]) == "float32"
        and n == 8
        and c_in == 32
        and c_out == 64
        and xh == 32
        and xw == 32
        and oh == 32
        and ow == 32
        and kh == 3
        and kw == 3
        and stride == [1, 1]
        and pre == [1, 1]
        and post == [1, 1]
        and dilation == [1, 1]
        and groups == 1
        and not flip_filter
    )
    if standard_3x3_fp32:
        stage = _make_stage(
            operation=operation,
            stage_name="fprop_standard_3x3",
            function_name="hygon_conv2d_fprop_standard_3x3_nchw_kernel",
            pointer_arguments=(
                _tensor_pointer(node, "x_ptr", "input"),
                _tensor_pointer(node, "w_ptr", "filter"),
                _tensor_pointer(node, "y_ptr", "output"),
            ),
            constants={
                "BLOCK_OC_S": 64,
                "BLOCK_K_S": 32,
            },
            tuning=CONV_FPROP_STANDARD_3X3_TUNING,
            tuning_key_value=int(d["n_outputs"]),
            grid_spec=GridSpec(
                "conv_private_fprop_standard_3x3", (oh, c_out, n)
            ),
            num_warps=4,
            num_stages=1,
        )
        return NodePlan(operation, (stage,))

    yolo_x_p5_fp32_gemm = (
        str(d["data_type"]) == "float32"
        and n == 1
        and c_in == 768
        and c_out == 768
        and xh == 40
        and xw == 40
        and oh == 20
        and ow == 20
        and kh == 3
        and kw == 3
        and stride == [2, 2]
        and pre == [1, 1]
        and post == [1, 1]
        and dilation == [1, 1]
        and groups == 1
        and not flip_filter
    )

    unit_1x1 = (
        kh == 1
        and kw == 1
        and stride == [1, 1]
        and pre == [0, 0]
        and post == [0, 0]
        and dilation == [1, 1]
    )
    if unit_1x1:
        block_oc, block_m, block_ci = 32, 32, 32
        stage = _make_stage(
            operation=operation,
            stage_name="fprop_1x1_nchw",
            function_name="hygon_conv2d_fprop_1x1_nchw_kernel",
            pointer_arguments=(
                _tensor_pointer(node, "x_ptr", "input"),
                _tensor_pointer(node, "w_ptr", "filter"),
                _tensor_pointer(node, "y_ptr", "output"),
            ),
            constants={
                "HW": oh * ow,
                "C_IN": c_in,
                "C_OUT": c_out,
                "CIN_PER_GROUP": cin_per_group,
                "COUT_PER_GROUP": cout_per_group,
                "GROUPS": groups,
                "BLOCK_OC": block_oc,
                "BLOCK_M": block_m,
                "BLOCK_CI": block_ci,
            },
            tuning=CONV_FPROP_TUNING,
            tuning_key_value=int(d["n_outputs"]),
            grid_spec=GridSpec(
                "conv_private_output", (oh * ow, cout_per_group, n * groups)
            ),
            num_warps=4,
            num_stages=2,
        )
        return NodePlan(operation, (stage,))

    low_ci_stride2_3x3 = (
        groups == 1
        and cin_per_group <= 3
        and kh == 3
        and kw == 3
        and stride == [2, 2]
        and pre == [1, 1]
        and post == [1, 1]
        and dilation == [1, 1]
        and n * c_in * xh * xw <= MAX_I32
        and c_out * cin_per_group * kh * kw <= MAX_I32
        and n * c_out * oh * ow <= MAX_I32
    )
    if low_ci_stride2_3x3:
        block_oc, block_m, block_k = 64, 32, 32
        stage = _make_stage(
            operation=operation,
            stage_name="fprop_stride2_low_ci",
            function_name="hygon_conv2d_fprop_stride2_low_ci_nchw_kernel",
            pointer_arguments=(
                _tensor_pointer(node, "x_ptr", "input"),
                _tensor_pointer(node, "w_ptr", "filter"),
                _tensor_pointer(node, "y_ptr", "output"),
            ),
            constants={
                "XH": xh,
                "XW": xw,
                "OH": oh,
                "OW": ow,
                "C_IN": c_in,
                "C_OUT": c_out,
                "BLOCK_OC": block_oc,
                "BLOCK_M": block_m,
                "BLOCK_K": block_k,
            },
            tuning=CONV_FPROP_TUNING,
            tuning_key_value=int(d["n_outputs"]),
            grid_spec=GridSpec("conv_private_fprop_rows", (oh, ow, c_out, n)),
            num_warps=4,
            num_stages=1,
        )
        return NodePlan(operation, (stage,))

    # The packed matrix is group-local. Keep grouped convolution on the
    # explicit-stride generic fallback until a grouped workspace ABI exists.
    if groups != 1:
        return None
    output_area = oh * ow
    reduction_extent = cin_per_group * kh * kw

    columns = _private_workspace(
        "fprop_columns",
        str(d["data_type"]),
        (n, reduction_extent, output_area),
    )
    # Bound provider-local memory so unusual graphs cannot turn this
    # optimization into an unbounded allocation policy.
    if columns.size > 512 * 1024 * 1024:
        return None
    col_stride_n, col_stride_k, col_stride_m = columns.strides
    pack_block_m, pack_block_k = 64, 16
    pack = _make_stage(
        operation=operation,
        stage_name="fprop_im2col",
        function_name="hygon_conv2d_im2col_nchw_kernel",
        pointer_arguments=(
            _tensor_pointer(node, "x_ptr", "input"),
            _workspace_pointer(
                "col_ptr", "fprop_columns", str(d["data_type"])
            ),
        ),
        constants={
            "M": output_area,
            "XH": xh,
            "XW": xw,
            "OH": oh,
            "OW": ow,
            "CIN_PER_GROUP": cin_per_group,
            "KH": kh,
            "KW": kw,
            "STRIDE_H": stride[0],
            "STRIDE_W": stride[1],
            "PAD_TOP": pre[0],
            "PAD_LEFT": pre[1],
            "DIL_H": dilation[0],
            "DIL_W": dilation[1],
            "X_STRIDE_N": int(image["strides"][0]),
            "X_STRIDE_C": int(image["strides"][1]),
            "X_STRIDE_H": int(image["strides"][2]),
            "X_STRIDE_W": int(image["strides"][3]),
            "COL_STRIDE_N": col_stride_n,
            "COL_STRIDE_K": col_stride_k,
            "COL_STRIDE_M": col_stride_m,
            "BLOCK_M": pack_block_m,
            "BLOCK_K": pack_block_k,
        },
        tuning=CONV_FPROP_TUNING,
        tuning_key_value=output_area,
        grid_spec=GridSpec(
            "conv_private_im2col", (output_area, reduction_extent, n)
        ),
        num_warps=4,
    )
    block_oc, block_m, block_k = 32, 32, 32
    gemm = _make_stage(
        operation=operation,
        stage_name="fprop_gemm",
        function_name=(
            "hygon_conv2d_fprop_yolo_x_p5_gemm_kernel"
            if yolo_x_p5_fp32_gemm
            else "hygon_conv2d_fprop_im2col_kernel"
        ),
        pointer_arguments=(
            _tensor_pointer(node, "w_ptr", "filter"),
            _workspace_pointer(
                "col_ptr", "fprop_columns", str(d["data_type"])
            ),
            _tensor_pointer(node, "y_ptr", "output"),
        ),
        constants={
            "M": output_area,
            "COUT_PER_GROUP": cout_per_group,
            "CIN_PER_GROUP": cin_per_group,
            "KH": kh,
            "KW": kw,
            "W_STRIDE_K": int(weight["strides"][0]),
            "W_STRIDE_C": int(weight["strides"][1]),
            "W_STRIDE_H": int(weight["strides"][2]),
            "W_STRIDE_W": int(weight["strides"][3]),
            "Y_STRIDE_N": int(result["strides"][0]),
            "Y_STRIDE_C": int(result["strides"][1]),
            "Y_STRIDE_H": int(result["strides"][2]),
            "Y_STRIDE_W": int(result["strides"][3]),
            "OW": ow,
            "COL_STRIDE_N": col_stride_n,
            "COL_STRIDE_K": col_stride_k,
            "COL_STRIDE_M": col_stride_m,
            **(
                {
                    "BLOCK_OC_X": block_oc,
                    "BLOCK_M_X": block_m,
                    "BLOCK_K_X": block_k,
                    "GROUP_M_X": 8,
                    "YOLO_X_P5": 1,
                }
                if yolo_x_p5_fp32_gemm
                else {
                    "BLOCK_OC": block_oc,
                    "BLOCK_M": block_m,
                    "BLOCK_K": block_k,
                    "GROUP_M": 4,
                }
            ),
        },
        tuning=(
            CONV_FPROP_YOLO_X_P5_GEMM_TUNING
            if yolo_x_p5_fp32_gemm
            else CONV_FPROP_TUNING
        ),
        tuning_key_value=int(d["n_outputs"]),
        grid_spec=GridSpec(
            (
                "conv_private_fprop_yolo_x"
                if yolo_x_p5_fp32_gemm
                else "conv_private_output"
            ),
            (output_area, cout_per_group, n),
        ),
        dependencies=("fprop_im2col",),
        num_warps=4,
        num_stages=2,
    )
    return NodePlan(
        operation,
        (pack, gemm),
        (columns,),
        _aligned_size(columns.size),
    )


def _private_dgrad_plan(node: Mapping[str, Any]) -> NodePlan | None:
    operation = str(node["operation"])
    d = _require_object(node["derived"], "node.derived")
    if int(d["spatial_rank"]) != 2:
        return None
    image = d["image"]
    weight = d["weight"]
    loss = d["result"]
    if not all(_is_contiguous(tensor) for tensor in (image, weight, loss)):
        return None
    n, c_in, xh, xw = (int(value) for value in image["dimensions"])
    c_out, _, kh, kw = (int(value) for value in weight["dimensions"])
    _, _, oh, ow = (int(value) for value in loss["dimensions"])
    groups = int(d["groups"])
    cin_per_group = int(d["in_per_group"])
    cout_per_group = int(d["out_per_group"])
    stride = list(d["stride"])
    pre = list(d["pre_padding"])
    post = list(d["post_padding"])
    dilation = list(d["dilation"])
    flip_filter = bool(d["flip_filter"])

    unit_1x1 = (
        kh == 1
        and kw == 1
        and stride == [1, 1]
        and pre == [0, 0]
        and post == [0, 0]
        and dilation == [1, 1]
    )
    if unit_1x1:
        block_m, block_ci, block_co = 32, 32, 32
        stage = _make_stage(
            operation=operation,
            stage_name="dgrad_1x1_nchw",
            function_name="hygon_conv_dgrad2d_1x1_nchw_kernel",
            pointer_arguments=(
                _tensor_pointer(node, "dy_ptr", "dy"),
                _tensor_pointer(node, "w_ptr", "w"),
                _tensor_pointer(node, "dx_ptr", "dx"),
            ),
            constants={
                "HW": xh * xw,
                "C_IN": c_in,
                "C_OUT": c_out,
                "CIN_PER_GROUP": cin_per_group,
                "COUT_PER_GROUP": cout_per_group,
                "GROUPS": groups,
                "BLOCK_M": block_m,
                "BLOCK_CI": block_ci,
                "BLOCK_CO": block_co,
            },
            tuning=CONV_DGRAD_TUNING,
            tuning_key_value=int(d["n_outputs"]),
            grid_spec=GridSpec(
                "conv_private_dgrad", (xh * xw, cin_per_group, n * groups)
            ),
            num_warps=4,
            num_stages=2,
        )
        return NodePlan(operation, (stage,))

    stride2_3x3 = (
        kh == 3
        and kw == 3
        and stride == [2, 2]
        and pre == [1, 1]
        and post == [1, 1]
        and dilation == [1, 1]
        and xh >= 2
        and xw >= 2
    )
    if stride2_3x3:
        if groups == 1:
            output_area = oh * ow
            packed_extent = c_in * kh * kw
            block_pointer_gemm = c_in > 4 and not flip_filter
            column_dimensions = (
                (n, packed_extent, output_area)
                if block_pointer_gemm
                else (n, output_area, packed_extent)
            )
            columns = _private_workspace(
                "dgrad_s2_contribution_columns",
                "float32",
                column_dimensions,
            )
            if columns.size > 512 * 1024 * 1024:
                return None
            if block_pointer_gemm:
                col_stride_n, col_stride_k, col_stride_m = columns.strides
            else:
                col_stride_n, col_stride_m, col_stride_k = columns.strides
            gemm_function = (
                "hygon_conv_dgrad2d_stride2_block_pointer_gemm_kernel"
                if block_pointer_gemm
                else "hygon_conv_dgrad2d_stride2_contribution_gemm_kernel"
            )
            gemm_constants = {
                "M": output_area,
                "C_IN": c_in,
                "C_OUT": c_out,
                "KH": kh,
                "KW": kw,
                "FLIP_FILTER": flip_filter,
                "DY_STRIDE_N": int(loss["strides"][0]),
                "DY_STRIDE_C": int(loss["strides"][1]),
                "DY_STRIDE_H": int(loss["strides"][2]),
                "DY_STRIDE_W": int(loss["strides"][3]),
                "W_STRIDE_K": int(weight["strides"][0]),
                "W_STRIDE_C": int(weight["strides"][1]),
                "W_STRIDE_H": int(weight["strides"][2]),
                "W_STRIDE_W": int(weight["strides"][3]),
                "OW": ow,
                "COL_STRIDE_N": col_stride_n,
                "COL_STRIDE_M": col_stride_m,
                "COL_STRIDE_K": col_stride_k,
                "BLOCK_M": 32,
                "BLOCK_CI": 32,
                "BLOCK_CO": 32,
            }
            if block_pointer_gemm:
                gemm_constants["GROUP_M"] = 4
            gemm = _make_stage(
                operation=operation,
                stage_name="dgrad_s2_contribution_gemm",
                function_name=gemm_function,
                pointer_arguments=(
                    _tensor_pointer(node, "dy_ptr", "dy"),
                    _tensor_pointer(node, "w_ptr", "w"),
                    _workspace_pointer(
                        "col_ptr", "dgrad_s2_contribution_columns", "float32"
                    ),
                ),
                constants=gemm_constants,
                tuning=CONV_DGRAD_TUNING,
                tuning_key_value=int(d["n_outputs"]),
                grid_spec=GridSpec(
                    "conv_private_dgrad_columns",
                    (output_area, packed_extent, n),
                ),
                num_warps=4,
                num_stages=2,
            )
            n_elements = n * c_in * xh * xw
            col2im = _make_stage(
                operation=operation,
                stage_name="dgrad_s2_contribution_col2im",
                function_name=(
                    "hygon_conv_dgrad2d_stride2_contribution_col2im_kernel"
                ),
                pointer_arguments=(
                    _workspace_pointer(
                        "col_ptr", "dgrad_s2_contribution_columns", "float32"
                    ),
                    _tensor_pointer(node, "dx_ptr", "dx"),
                ),
                constants={
                    "N_ELEMENTS": n_elements,
                    "XH": xh,
                    "XW": xw,
                    "OH": oh,
                    "OW": ow,
                    "C_IN": c_in,
                    "KH": kh,
                    "KW": kw,
                    "PAD_TOP": pre[0],
                    "PAD_LEFT": pre[1],
                    "DIL_H": dilation[0],
                    "DIL_W": dilation[1],
                    "COL_STRIDE_N": col_stride_n,
                    "COL_STRIDE_M": col_stride_m,
                    "COL_STRIDE_K": col_stride_k,
                    "X_STRIDE_N": int(image["strides"][0]),
                    "X_STRIDE_C": int(image["strides"][1]),
                    "X_STRIDE_H": int(image["strides"][2]),
                    "X_STRIDE_W": int(image["strides"][3]),
                    "BLOCK_SIZE": 256,
                },
                tuning=CONV_DGRAD_TUNING,
                tuning_key_value=int(d["n_outputs"]),
                grid_spec=GridSpec("linear", (n_elements,)),
                dependencies=("dgrad_s2_contribution_gemm",),
                num_warps=4,
            )
            return NodePlan(
                operation,
                (gemm, col2im),
                (columns,),
                _aligned_size(columns.size),
            )

        block_m, block_ci, block_co = 32, 32, 32
        stages: list[KernelStagePlan] = []
        previous: tuple[str, ...] = ()
        for parity_h in range(2):
            for parity_w in range(2):
                count_h = (xh - parity_h + 1) // 2
                count_w = (xw - parity_w + 1) // 2
                parity_m = count_h * count_w
                stage_name = f"dgrad_s2_p{parity_h}{parity_w}"
                stage = _make_stage(
                    operation=operation,
                    stage_name=stage_name,
                    function_name=("hygon_conv_dgrad2d_stride2_parity_kernel"),
                    pointer_arguments=(
                        _tensor_pointer(node, "dy_ptr", "dy"),
                        _tensor_pointer(node, "w_ptr", "w"),
                        _tensor_pointer(node, "dx_ptr", "dx"),
                    ),
                    constants={
                        "PARITY_M": parity_m,
                        "PARITY_W_COUNT": count_w,
                        "PARITY_H": parity_h,
                        "PARITY_W": parity_w,
                        "XH": xh,
                        "XW": xw,
                        "OH": oh,
                        "OW": ow,
                        "CIN_PER_GROUP": cin_per_group,
                        "COUT_PER_GROUP": cout_per_group,
                        "GROUPS": groups,
                        "KH": kh,
                        "KW": kw,
                        "PAD_TOP": pre[0],
                        "PAD_LEFT": pre[1],
                        "DIL_H": dilation[0],
                        "DIL_W": dilation[1],
                        "FLIP_FILTER": flip_filter,
                        "DY_STRIDE_N": int(loss["strides"][0]),
                        "DY_STRIDE_C": int(loss["strides"][1]),
                        "DY_STRIDE_H": int(loss["strides"][2]),
                        "DY_STRIDE_W": int(loss["strides"][3]),
                        "W_STRIDE_K": int(weight["strides"][0]),
                        "W_STRIDE_C": int(weight["strides"][1]),
                        "W_STRIDE_H": int(weight["strides"][2]),
                        "W_STRIDE_W": int(weight["strides"][3]),
                        "X_STRIDE_N": int(image["strides"][0]),
                        "X_STRIDE_C": int(image["strides"][1]),
                        "X_STRIDE_H": int(image["strides"][2]),
                        "X_STRIDE_W": int(image["strides"][3]),
                        "BLOCK_M": block_m,
                        "BLOCK_CI": block_ci,
                        "BLOCK_CO": block_co,
                    },
                    tuning=CONV_DGRAD_TUNING,
                    tuning_key_value=int(d["n_outputs"]),
                    grid_spec=GridSpec(
                        "conv_private_dgrad",
                        (parity_m, cin_per_group, n * groups),
                    ),
                    dependencies=previous,
                    num_warps=4,
                    num_stages=2,
                )
                stages.append(stage)
                previous = (stage_name,)
        return NodePlan(operation, tuple(stages))

    exact_stride1_3x3 = (
        str(d["data_type"]) in ("float32", "bfloat16")
        and kh == 3
        and kw == 3
        and stride == [1, 1]
        and pre == [1, 1]
        and post == [1, 1]
        and dilation == [1, 1]
        and groups == 1
        and not flip_filter
        and n == 8
        and c_in == 32
        and c_out == 64
        and xh == 32
        and xw == 32
        and oh == 32
        and ow == 32
    )
    if exact_stride1_3x3:
        output_area = oh * ow
        packed_extent = c_in * kh * kw
        columns = _private_workspace(
            "dgrad_s1_exact_columns",
            "float32",
            (n, packed_extent, output_area),
        )
        workspace_size = _aligned_size(columns.size)
        if (
            columns.offset != 0
            or columns.alignment != WORKSPACE_ALIGNMENT
            or columns.size != DGRAD_EXACT_3X3_S1_WORKSPACE_CAP
            or workspace_size != DGRAD_EXACT_3X3_S1_WORKSPACE_CAP
        ):
            return None
        col_stride_n, col_stride_k, col_stride_m = columns.strides

        # BLOCK_CI/BLOCK_CO are intentionally fixed: the exact GEMM has no
        # tail masks. EXACT_3X3_S1 keeps generic DGrad rows out of this route.
        gemm = _make_stage(
            operation=operation,
            stage_name="dgrad_s1_exact_gemm",
            function_name="hygon_conv_dgrad2d_exact_3x3_s1_gemm_kernel",
            pointer_arguments=(
                _tensor_pointer(node, "dy_ptr", "dy"),
                _tensor_pointer(node, "w_ptr", "w"),
                _workspace_pointer(
                    "col_ptr", "dgrad_s1_exact_columns", "float32"
                ),
            ),
            constants={
                "M": output_area,
                "C_IN": c_in,
                "C_OUT": c_out,
                "KH": kh,
                "KW": kw,
                "FLIP_FILTER": flip_filter,
                "DY_STRIDE_N": int(loss["strides"][0]),
                "DY_STRIDE_C": int(loss["strides"][1]),
                "DY_STRIDE_H": int(loss["strides"][2]),
                "DY_STRIDE_W": int(loss["strides"][3]),
                "W_STRIDE_K": int(weight["strides"][0]),
                "W_STRIDE_C": int(weight["strides"][1]),
                "W_STRIDE_H": int(weight["strides"][2]),
                "W_STRIDE_W": int(weight["strides"][3]),
                "OW": ow,
                "COL_STRIDE_N": col_stride_n,
                "COL_STRIDE_M": col_stride_m,
                "COL_STRIDE_K": col_stride_k,
                "EXACT_3X3_S1": 1,
                "BLOCK_M": 64,
                "BLOCK_CI": 32,
                "BLOCK_CO": 32,
                "GROUP_M": 16,
            },
            tuning=CONV_DGRAD_EXACT_GEMM_TUNING,
            tuning_key_value=int(d["n_outputs"]),
            grid_spec=GridSpec(
                "conv_private_dgrad_columns",
                (output_area, packed_extent, n),
            ),
            num_warps=4,
            num_stages=1,
        )
        col2im = _make_stage(
            operation=operation,
            stage_name="dgrad_s1_exact_col2im",
            function_name="hygon_conv_dgrad2d_exact_3x3_s1_col2im_kernel",
            pointer_arguments=(
                _workspace_pointer(
                    "col_ptr", "dgrad_s1_exact_columns", "float32"
                ),
                _tensor_pointer(node, "dx_ptr", "dx"),
            ),
            constants={
                "M": output_area,
                "XH": xh,
                "XW": xw,
                "OH": oh,
                "OW": ow,
                "C_IN": c_in,
                "KH": kh,
                "KW": kw,
                "PAD_TOP": pre[0],
                "PAD_LEFT": pre[1],
                "DIL_H": dilation[0],
                "DIL_W": dilation[1],
                "COL_STRIDE_N": col_stride_n,
                "COL_STRIDE_M": col_stride_m,
                "COL_STRIDE_K": col_stride_k,
                "X_STRIDE_N": int(image["strides"][0]),
                "X_STRIDE_C": int(image["strides"][1]),
                "X_STRIDE_H": int(image["strides"][2]),
                "X_STRIDE_W": int(image["strides"][3]),
                "EXACT_3X3_S1": 1,
                "BLOCK_H": 4,
            },
            tuning=CONV_DGRAD_EXACT_COL2IM_TUNING,
            tuning_key_value=int(d["n_outputs"]),
            grid_spec=GridSpec("conv_private_dgrad_fixed_2d", (xh, n * c_in)),
            dependencies=("dgrad_s1_exact_gemm",),
            num_warps=2,
            num_stages=1,
        )
        return NodePlan(
            operation,
            (gemm, col2im),
            (columns,),
            workspace_size,
        )

    if stride == [1, 1] and kh * kw <= 25:
        block_m, block_ci, block_co = 32, 32, 32
        stage = _make_stage(
            operation=operation,
            stage_name="dgrad_stride1",
            function_name="hygon_conv_dgrad2d_stride1_kernel",
            pointer_arguments=(
                _tensor_pointer(node, "dy_ptr", "dy"),
                _tensor_pointer(node, "w_ptr", "w"),
                _tensor_pointer(node, "dx_ptr", "dx"),
            ),
            constants={
                "M": n * xh * xw,
                "XH": xh,
                "XW": xw,
                "OH": oh,
                "OW": ow,
                "CIN_PER_GROUP": cin_per_group,
                "COUT_PER_GROUP": cout_per_group,
                "KH": kh,
                "KW": kw,
                "PAD_TOP": pre[0],
                "PAD_LEFT": pre[1],
                "DIL_H": dilation[0],
                "DIL_W": dilation[1],
                "FLIP_FILTER": flip_filter,
                "DY_STRIDE_N": int(loss["strides"][0]),
                "DY_STRIDE_C": int(loss["strides"][1]),
                "DY_STRIDE_H": int(loss["strides"][2]),
                "DY_STRIDE_W": int(loss["strides"][3]),
                "W_STRIDE_K": int(weight["strides"][0]),
                "W_STRIDE_C": int(weight["strides"][1]),
                "W_STRIDE_H": int(weight["strides"][2]),
                "W_STRIDE_W": int(weight["strides"][3]),
                "X_STRIDE_N": int(image["strides"][0]),
                "X_STRIDE_C": int(image["strides"][1]),
                "X_STRIDE_H": int(image["strides"][2]),
                "X_STRIDE_W": int(image["strides"][3]),
                "BLOCK_M": block_m,
                "BLOCK_CI": block_ci,
                "BLOCK_CO": block_co,
            },
            tuning=CONV_DGRAD_TUNING,
            tuning_key_value=int(d["n_outputs"]),
            grid_spec=GridSpec(
                "conv_private_dgrad", (n * xh * xw, cin_per_group, groups)
            ),
            num_warps=4,
            num_stages=2,
        )
        return NodePlan(operation, (stage,))
    return None


def _wgrad_reduce_stage(
    node: Mapping[str, Any],
    *,
    workspace_name: str,
    num_splits: int,
    cout: int,
    cin_per_group: int,
    kh: int,
    kw: int,
    partial_strides: Sequence[int],
    dependency: str,
) -> KernelStagePlan:
    operation = str(node["operation"])
    d = _require_object(node["derived"], "node.derived")
    weight = d["weight"]
    cik = cin_per_group * kh * kw
    total = cout * cik
    block_size = 256
    return _make_stage(
        operation=operation,
        stage_name="wgrad_reduce",
        function_name="hygon_conv_wgrad2d_reduce_kernel",
        pointer_arguments=(
            _workspace_pointer("partial_ptr", workspace_name, "float32"),
            _tensor_pointer(node, "dw_ptr", "dw"),
        ),
        constants={
            "TOTAL": total,
            "CIK": cik,
            "CIN_PER_GROUP": cin_per_group,
            "KH": kh,
            "KW": kw,
            "FLIP_FILTER": bool(d["flip_filter"]),
            "NUM_SPLITS": num_splits,
            "PARTIAL_STRIDE_SPLIT": int(partial_strides[0]),
            "PARTIAL_STRIDE_OC": int(partial_strides[1]),
            "PARTIAL_STRIDE_K": int(partial_strides[2]),
            "W_STRIDE_K": int(weight["strides"][0]),
            "W_STRIDE_C": int(weight["strides"][1]),
            "W_STRIDE_H": int(weight["strides"][2]),
            "W_STRIDE_W": int(weight["strides"][3]),
            "BLOCK_SIZE": block_size,
        },
        tuning=CONV_WGRAD_TUNING,
        tuning_key_value=total,
        grid_spec=GridSpec("linear", (total,)),
        dependencies=(dependency,),
        num_warps=4,
    )


def _private_wgrad_plan(node: Mapping[str, Any]) -> NodePlan | None:
    operation = str(node["operation"])
    d = _require_object(node["derived"], "node.derived")
    if int(d["spatial_rank"]) != 2:
        return None
    image = d["image"]
    weight = d["weight"]
    loss = d["result"]
    if not all(_is_contiguous(tensor) for tensor in (image, weight, loss)):
        return None
    n, c_in, xh, xw = (int(value) for value in image["dimensions"])
    c_out, _, kh, kw = (int(value) for value in weight["dimensions"])
    _, _, oh, ow = (int(value) for value in loss["dimensions"])
    groups = int(d["groups"])
    cin_per_group = int(d["in_per_group"])
    cout_per_group = int(d["out_per_group"])
    stride = list(d["stride"])
    pre = list(d["pre_padding"])
    post = list(d["post_padding"])
    dilation = list(d["dilation"])
    output_area = oh * ow
    total_rows = n * output_area

    unit_1x1 = (
        kh == 1
        and kw == 1
        and stride == [1, 1]
        and pre == [0, 0]
        and post == [0, 0]
        and dilation == [1, 1]
    )
    if unit_1x1 and total_rows >= 2048:
        num_splits = 8 if total_rows >= 4096 else 4
        partial = _private_workspace(
            "wgrad_partial",
            "float32",
            (num_splits, c_out, cin_per_group),
        )
        if partial.size > 512 * 1024 * 1024:
            return None
        block_oc, block_ci, block_m = 16, 16, 64
        split = _make_stage(
            operation=operation,
            stage_name="wgrad_1x1_split",
            function_name="hygon_conv_wgrad2d_1x1_split_kernel",
            pointer_arguments=(
                _tensor_pointer(node, "dy_ptr", "dy"),
                _tensor_pointer(node, "x_ptr", "x"),
                _workspace_pointer("partial_ptr", "wgrad_partial", "float32"),
            ),
            constants={
                "TOTAL_ROWS": total_rows,
                "ROWS_PER_SPLIT": _ceil_div(total_rows, num_splits),
                "HW": output_area,
                "C_IN": c_in,
                "C_OUT": c_out,
                "CIN_PER_GROUP": cin_per_group,
                "COUT_PER_GROUP": cout_per_group,
                "GROUPS": groups,
                "PARTIAL_STRIDE_SPLIT": int(partial.strides[0]),
                "PARTIAL_STRIDE_OC": int(partial.strides[1]),
                "PARTIAL_STRIDE_K": int(partial.strides[2]),
                "BLOCK_OC": block_oc,
                "BLOCK_CI": block_ci,
                "BLOCK_M": block_m,
            },
            tuning=CONV_WGRAD_TUNING,
            tuning_key_value=int(d["n_outputs"]),
            grid_spec=GridSpec(
                "conv_private_wgrad_1x1",
                (cout_per_group, cin_per_group, num_splits * groups),
            ),
            num_warps=4,
            num_stages=2,
        )
        reduce = _wgrad_reduce_stage(
            node,
            workspace_name="wgrad_partial",
            num_splits=num_splits,
            cout=c_out,
            cin_per_group=cin_per_group,
            kh=kh,
            kw=kw,
            partial_strides=partial.strides,
            dependency="wgrad_1x1_split",
        )
        return NodePlan(
            operation,
            (split, reduce),
            (partial,),
            _aligned_size(partial.size),
        )

    stem_split_3x3 = (
        n == 1
        and groups == 1
        and cin_per_group <= 3
        and kh == 3
        and kw == 3
        and stride == [2, 2]
        and dilation == [1, 1]
        and total_rows >= 65536
        and oh % 64 == 0
    )
    if stem_split_3x3:
        num_splits = 64
        reduction_extent = cin_per_group * kh * kw
        partial = _private_workspace(
            "wgrad_partial",
            "float32",
            (num_splits, c_out, reduction_extent),
        )
        if partial.size > 512 * 1024 * 1024:
            return None
        block_oc, block_ci_k, block_m = 64, 32, 64
        split = _make_stage(
            operation=operation,
            stage_name="wgrad_stem_split",
            function_name="hygon_conv_wgrad2d_stem_split_kernel",
            pointer_arguments=(
                _tensor_pointer(node, "dy_ptr", "dy"),
                _tensor_pointer(node, "x_ptr", "x"),
                _workspace_pointer("partial_ptr", "wgrad_partial", "float32"),
            ),
            constants={
                "OUTPUT_ROWS_PER_SPLIT": oh // num_splits,
                "OH": oh,
                "OW": ow,
                "XH": xh,
                "XW": xw,
                "COUT_PER_GROUP": cout_per_group,
                "CIN_PER_GROUP": cin_per_group,
                "KH": kh,
                "KW": kw,
                "STRIDE_H": stride[0],
                "STRIDE_W": stride[1],
                "PAD_TOP": pre[0],
                "PAD_LEFT": pre[1],
                "DIL_H": dilation[0],
                "DIL_W": dilation[1],
                "DY_STRIDE_C": int(loss["strides"][1]),
                "DY_STRIDE_H": int(loss["strides"][2]),
                "DY_STRIDE_W": int(loss["strides"][3]),
                "X_STRIDE_C": int(image["strides"][1]),
                "X_STRIDE_H": int(image["strides"][2]),
                "X_STRIDE_W": int(image["strides"][3]),
                "PARTIAL_STRIDE_SPLIT": int(partial.strides[0]),
                "PARTIAL_STRIDE_OC": int(partial.strides[1]),
                "PARTIAL_STRIDE_K": int(partial.strides[2]),
                "BLOCK_OC": block_oc,
                "BLOCK_CI_K": block_ci_k,
                "BLOCK_M": block_m,
            },
            tuning=CONV_WGRAD_TUNING,
            tuning_key_value=int(d["n_outputs"]),
            grid_spec=GridSpec(
                "conv_private_wgrad",
                (cout_per_group, reduction_extent, num_splits),
            ),
            num_warps=4,
            num_stages=2,
        )
        reduce = _wgrad_reduce_stage(
            node,
            workspace_name="wgrad_partial",
            num_splits=num_splits,
            cout=c_out,
            cin_per_group=cin_per_group,
            kh=kh,
            kw=kw,
            partial_strides=partial.strides,
            dependency="wgrad_stem_split",
        )
        return NodePlan(
            operation,
            (split, reduce),
            (partial,),
            _aligned_size(partial.size),
        )

    multirow_split_3x3 = (
        n > 1
        and groups == 1
        and cin_per_group > 3
        and kh == 3
        and kw == 3
        and stride == [2, 2]
        and pre == [1, 1]
        and post == [1, 1]
        and dilation == [1, 1]
        and not bool(d["flip_filter"])
        and ow <= 32
    )
    if multirow_split_3x3:
        num_splits = n
        reduction_extent = cin_per_group * kh * kw
        partial = _private_workspace(
            "wgrad_partial",
            "float32",
            (num_splits, c_out, reduction_extent),
        )
        if partial.size > 512 * 1024 * 1024:
            return None
        block_oc, block_ci_k, block_m = 128, 64, 64
        split = _make_stage(
            operation=operation,
            stage_name="wgrad_multirow_split",
            function_name="hygon_conv_wgrad2d_multirow_split_kernel",
            pointer_arguments=(
                _tensor_pointer(node, "dy_ptr", "dy"),
                _tensor_pointer(node, "x_ptr", "x"),
                _workspace_pointer("partial_ptr", "wgrad_partial", "float32"),
            ),
            constants={
                "OH": oh,
                "OW": ow,
                "XH": xh,
                "XW": xw,
                "COUT_PER_GROUP": cout_per_group,
                "CIN_PER_GROUP": cin_per_group,
                "KH": kh,
                "KW": kw,
                "STRIDE_H": stride[0],
                "STRIDE_W": stride[1],
                "PAD_TOP": pre[0],
                "PAD_LEFT": pre[1],
                "DIL_H": dilation[0],
                "DIL_W": dilation[1],
                "DY_STRIDE_N": int(loss["strides"][0]),
                "DY_STRIDE_C": int(loss["strides"][1]),
                "DY_STRIDE_H": int(loss["strides"][2]),
                "DY_STRIDE_W": int(loss["strides"][3]),
                "X_STRIDE_N": int(image["strides"][0]),
                "X_STRIDE_C": int(image["strides"][1]),
                "X_STRIDE_H": int(image["strides"][2]),
                "X_STRIDE_W": int(image["strides"][3]),
                "PARTIAL_STRIDE_SPLIT": int(partial.strides[0]),
                "PARTIAL_STRIDE_OC": int(partial.strides[1]),
                "PARTIAL_STRIDE_K": int(partial.strides[2]),
                "ROW_PITCH": 32,
                "BLOCK_OC": block_oc,
                "BLOCK_CI_K": block_ci_k,
                "BLOCK_M": block_m,
            },
            tuning=CONV_WGRAD_TUNING,
            tuning_key_value=int(d["n_outputs"]),
            grid_spec=GridSpec(
                "conv_private_wgrad",
                (cout_per_group, reduction_extent, num_splits),
            ),
            num_warps=8,
            num_stages=2,
        )
        reduce = _wgrad_reduce_stage(
            node,
            workspace_name="wgrad_partial",
            num_splits=num_splits,
            cout=c_out,
            cin_per_group=cin_per_group,
            kh=kh,
            kw=kw,
            partial_strides=partial.strides,
            dependency="wgrad_multirow_split",
        )
        return NodePlan(
            operation,
            (split, reduce),
            (partial,),
            _aligned_size(partial.size),
        )

    p5_rowmajor_3x3 = (
        n == 1
        and groups == 1
        and cin_per_group > 3
        and kh == 3
        and kw == 3
        and stride == [2, 2]
        and pre == [1, 1]
        and post == [1, 1]
        and dilation == [1, 1]
        and not bool(d["flip_filter"])
        and xh == 40
        and xw == 40
        and oh == 20
        and ow == 20
    )
    if p5_rowmajor_3x3:
        reduction_extent = cin_per_group * kh * kw
        # gfx936 measurements show that block pointers win for the large
        # ML/X matrices. Keep N/S and tail shapes on the existing row-major
        # kernel so this optimization cannot regress their established path.
        large_p5_block_ptr = cin_per_group >= 512 and cout_per_group >= 512
        columns = _private_workspace(
            "wgrad_rowmajor_columns",
            str(d["data_type"]),
            (total_rows, reduction_extent),
        )
        if columns.size > 512 * 1024 * 1024:
            return None
        col_stride_r, col_stride_k = columns.strides
        pack_block_m, pack_block_k = 16, 64
        pack = _make_stage(
            operation=operation,
            stage_name="wgrad_im2row",
            function_name="hygon_conv_wgrad2d_im2row_kernel",
            pointer_arguments=(
                _tensor_pointer(node, "x_ptr", "x"),
                _workspace_pointer(
                    "col_ptr",
                    "wgrad_rowmajor_columns",
                    str(d["data_type"]),
                ),
            ),
            constants={
                "M": output_area,
                "XH": xh,
                "XW": xw,
                "OW": ow,
                "CIN_PER_GROUP": cin_per_group,
                "KH": kh,
                "KW": kw,
                "STRIDE_H": stride[0],
                "STRIDE_W": stride[1],
                "PAD_TOP": pre[0],
                "PAD_LEFT": pre[1],
                "DIL_H": dilation[0],
                "DIL_W": dilation[1],
                "X_STRIDE_N": int(image["strides"][0]),
                "X_STRIDE_C": int(image["strides"][1]),
                "X_STRIDE_H": int(image["strides"][2]),
                "X_STRIDE_W": int(image["strides"][3]),
                "COL_STRIDE_R": col_stride_r,
                "COL_STRIDE_K": col_stride_k,
                "BLOCK_M": pack_block_m,
                "BLOCK_K": pack_block_k,
            },
            tuning=CONV_WGRAD_TUNING,
            tuning_key_value=total_rows * reduction_extent,
            grid_spec=GridSpec(
                "conv_private_im2col", (output_area, reduction_extent, n)
            ),
            num_warps=4,
        )
        block_oc, block_ci_k, block_m = 64, 64, 64
        if large_p5_block_ptr:
            gemm_function = "hygon_conv_wgrad2d_p5_block_ptr_kernel"
            gemm_constants = {
                "M": output_area,
                "COUT_PER_GROUP": cout_per_group,
                "REDUCTION_EXTENT": reduction_extent,
                "DY_STRIDE_C": int(loss["strides"][1]),
                "COL_STRIDE_R": col_stride_r,
                "COL_STRIDE_K": col_stride_k,
                "W_STRIDE_K": int(weight["strides"][0]),
                "BLOCK_OC": block_oc,
                "BLOCK_CI_K": block_ci_k,
                "BLOCK_M": block_m,
            }
        else:
            gemm_function = "hygon_conv_wgrad2d_rowmajor_kernel"
            gemm_constants = {
                "M": output_area,
                "COUT_PER_GROUP": cout_per_group,
                "CIN_PER_GROUP": cin_per_group,
                "KH": kh,
                "KW": kw,
                "FLIP_FILTER": bool(d["flip_filter"]),
                "DY_STRIDE_C": int(loss["strides"][1]),
                "DY_STRIDE_H": int(loss["strides"][2]),
                "DY_STRIDE_W": int(loss["strides"][3]),
                "OW": ow,
                "COL_STRIDE_R": col_stride_r,
                "COL_STRIDE_K": col_stride_k,
                "W_STRIDE_K": int(weight["strides"][0]),
                "W_STRIDE_C": int(weight["strides"][1]),
                "W_STRIDE_H": int(weight["strides"][2]),
                "W_STRIDE_W": int(weight["strides"][3]),
                "BLOCK_OC": block_oc,
                "BLOCK_CI_K": block_ci_k,
                "BLOCK_M": block_m,
            }
        gemm = _make_stage(
            operation=operation,
            stage_name="wgrad_rowmajor",
            function_name=gemm_function,
            pointer_arguments=(
                _tensor_pointer(node, "dy_ptr", "dy"),
                _workspace_pointer(
                    "col_ptr",
                    "wgrad_rowmajor_columns",
                    str(d["data_type"]),
                ),
                _tensor_pointer(node, "dw_ptr", "dw"),
            ),
            constants=gemm_constants,
            tuning=CONV_WGRAD_TUNING,
            tuning_key_value=int(d["n_outputs"]),
            grid_spec=GridSpec(
                "conv_private_wgrad",
                (cout_per_group, reduction_extent, 1),
            ),
            dependencies=("wgrad_im2row",),
            num_warps=8,
            num_stages=2,
        )
        return NodePlan(
            operation,
            (pack, gemm),
            (columns,),
            _aligned_size(columns.size),
        )

    direct_split_3x3 = (
        groups == 1
        and kh == 3
        and kw == 3
        and stride == [2, 2]
        and dilation == [1, 1]
        and total_rows >= 4096
    )
    if direct_split_3x3:
        num_splits = 64 if total_rows >= 65536 else 16
        reduction_extent = cin_per_group * kh * kw
        partial = _private_workspace(
            "wgrad_partial",
            "float32",
            (num_splits, c_out, reduction_extent),
        )
        if partial.size > 512 * 1024 * 1024:
            return None
        block_oc, block_ci_k, block_m = 16, 16, 64
        split = _make_stage(
            operation=operation,
            stage_name="wgrad_direct_split",
            function_name="hygon_conv_wgrad2d_direct_split_kernel",
            pointer_arguments=(
                _tensor_pointer(node, "dy_ptr", "dy"),
                _tensor_pointer(node, "x_ptr", "x"),
                _workspace_pointer("partial_ptr", "wgrad_partial", "float32"),
            ),
            constants={
                "TOTAL_ROWS": total_rows,
                "ROWS_PER_SPLIT": _ceil_div(total_rows, num_splits),
                "M": output_area,
                "XH": xh,
                "XW": xw,
                "OW": ow,
                "COUT_PER_GROUP": cout_per_group,
                "CIN_PER_GROUP": cin_per_group,
                "KH": kh,
                "KW": kw,
                "STRIDE_H": stride[0],
                "STRIDE_W": stride[1],
                "PAD_TOP": pre[0],
                "PAD_LEFT": pre[1],
                "DIL_H": dilation[0],
                "DIL_W": dilation[1],
                "DY_STRIDE_N": int(loss["strides"][0]),
                "DY_STRIDE_C": int(loss["strides"][1]),
                "DY_STRIDE_H": int(loss["strides"][2]),
                "DY_STRIDE_W": int(loss["strides"][3]),
                "X_STRIDE_N": int(image["strides"][0]),
                "X_STRIDE_C": int(image["strides"][1]),
                "X_STRIDE_H": int(image["strides"][2]),
                "X_STRIDE_W": int(image["strides"][3]),
                "PARTIAL_STRIDE_SPLIT": int(partial.strides[0]),
                "PARTIAL_STRIDE_OC": int(partial.strides[1]),
                "PARTIAL_STRIDE_K": int(partial.strides[2]),
                "BLOCK_OC": block_oc,
                "BLOCK_CI_K": block_ci_k,
                "BLOCK_M": block_m,
            },
            tuning=CONV_WGRAD_TUNING,
            tuning_key_value=int(d["n_outputs"]),
            grid_spec=GridSpec(
                "conv_private_wgrad",
                (cout_per_group, reduction_extent, num_splits),
            ),
            num_warps=4,
            num_stages=2,
        )
        reduce = _wgrad_reduce_stage(
            node,
            workspace_name="wgrad_partial",
            num_splits=num_splits,
            cout=c_out,
            cin_per_group=cin_per_group,
            kh=kh,
            kw=kw,
            partial_strides=partial.strides,
            dependency="wgrad_direct_split",
        )
        return NodePlan(
            operation,
            (split, reduce),
            (partial,),
            _aligned_size(partial.size),
        )

    # The current packed workspace is deliberately group-one. All other
    # layouts and ranks remain covered by conv_wgrad_nd_kernel.
    if groups != 1 or unit_1x1:
        return None
    reduction_extent = cin_per_group * kh * kw
    columns = _private_workspace(
        "wgrad_columns",
        str(d["data_type"]),
        (n, reduction_extent, output_area),
    )
    if columns.size > 512 * 1024 * 1024:
        return None
    col_stride_n, col_stride_k, col_stride_m = columns.strides
    pack_block_m, pack_block_k = 64, 16
    pack = _make_stage(
        operation=operation,
        stage_name="wgrad_im2col",
        function_name="hygon_conv2d_im2col_nchw_kernel",
        pointer_arguments=(
            _tensor_pointer(node, "x_ptr", "x"),
            _workspace_pointer(
                "col_ptr", "wgrad_columns", str(d["data_type"])
            ),
        ),
        constants={
            "M": output_area,
            "XH": xh,
            "XW": xw,
            "OH": oh,
            "OW": ow,
            "CIN_PER_GROUP": cin_per_group,
            "KH": kh,
            "KW": kw,
            "STRIDE_H": stride[0],
            "STRIDE_W": stride[1],
            "PAD_TOP": pre[0],
            "PAD_LEFT": pre[1],
            "DIL_H": dilation[0],
            "DIL_W": dilation[1],
            "X_STRIDE_N": int(image["strides"][0]),
            "X_STRIDE_C": int(image["strides"][1]),
            "X_STRIDE_H": int(image["strides"][2]),
            "X_STRIDE_W": int(image["strides"][3]),
            "COL_STRIDE_N": col_stride_n,
            "COL_STRIDE_K": col_stride_k,
            "COL_STRIDE_M": col_stride_m,
            "BLOCK_M": pack_block_m,
            "BLOCK_K": pack_block_k,
        },
        tuning=CONV_WGRAD_TUNING,
        tuning_key_value=output_area,
        grid_spec=GridSpec(
            "conv_private_im2col", (output_area, reduction_extent, n)
        ),
        num_warps=4,
    )
    parameter_elements = c_out * reduction_extent
    if total_rows >= 65536 and parameter_elements <= 131072:
        num_splits = 64
    elif total_rows >= 4096 and parameter_elements <= 131072:
        num_splits = 8
    else:
        num_splits = 1
    block_oc, block_ci_k, block_m = 16, 16, 64

    if num_splits == 1:
        gemm = _make_stage(
            operation=operation,
            stage_name="wgrad_gemm",
            function_name="hygon_conv_wgrad2d_im2col_kernel",
            pointer_arguments=(
                _tensor_pointer(node, "dy_ptr", "dy"),
                _workspace_pointer(
                    "col_ptr", "wgrad_columns", str(d["data_type"])
                ),
                _tensor_pointer(node, "dw_ptr", "dw"),
            ),
            constants={
                "N": n,
                "M": output_area,
                "COUT_PER_GROUP": cout_per_group,
                "CIN_PER_GROUP": cin_per_group,
                "KH": kh,
                "KW": kw,
                "FLIP_FILTER": bool(d["flip_filter"]),
                "DY_STRIDE_N": int(loss["strides"][0]),
                "DY_STRIDE_C": int(loss["strides"][1]),
                "DY_STRIDE_H": int(loss["strides"][2]),
                "DY_STRIDE_W": int(loss["strides"][3]),
                "OW": ow,
                "COL_STRIDE_N": col_stride_n,
                "COL_STRIDE_K": col_stride_k,
                "COL_STRIDE_M": col_stride_m,
                "W_STRIDE_K": int(weight["strides"][0]),
                "W_STRIDE_C": int(weight["strides"][1]),
                "W_STRIDE_H": int(weight["strides"][2]),
                "W_STRIDE_W": int(weight["strides"][3]),
                "BLOCK_OC": block_oc,
                "BLOCK_CI_K": block_ci_k,
                "BLOCK_M": block_m,
            },
            tuning=CONV_WGRAD_TUNING,
            tuning_key_value=int(d["n_outputs"]),
            grid_spec=GridSpec(
                "conv_private_wgrad", (cout_per_group, reduction_extent, 1)
            ),
            dependencies=("wgrad_im2col",),
            num_warps=4,
            num_stages=2,
        )
        return NodePlan(
            operation,
            (pack, gemm),
            (columns,),
            _aligned_size(columns.size),
        )

    partial_offset = _aligned_size(columns.size)
    partial = _private_workspace(
        "wgrad_partial",
        "float32",
        (num_splits, c_out, reduction_extent),
        offset=partial_offset,
    )
    workspace_size = _aligned_size(partial.offset + partial.size)
    if workspace_size > 512 * 1024 * 1024:
        return None
    split = _make_stage(
        operation=operation,
        stage_name="wgrad_split",
        function_name="hygon_conv_wgrad2d_im2col_split_kernel",
        pointer_arguments=(
            _tensor_pointer(node, "dy_ptr", "dy"),
            _workspace_pointer(
                "col_ptr", "wgrad_columns", str(d["data_type"])
            ),
            _workspace_pointer("partial_ptr", "wgrad_partial", "float32"),
        ),
        constants={
            "TOTAL_ROWS": total_rows,
            "ROWS_PER_SPLIT": _ceil_div(total_rows, num_splits),
            "M": output_area,
            "COUT_PER_GROUP": cout_per_group,
            "CIN_PER_GROUP": cin_per_group,
            "KH": kh,
            "KW": kw,
            "DY_STRIDE_N": int(loss["strides"][0]),
            "DY_STRIDE_C": int(loss["strides"][1]),
            "DY_STRIDE_H": int(loss["strides"][2]),
            "DY_STRIDE_W": int(loss["strides"][3]),
            "OW": ow,
            "COL_STRIDE_N": col_stride_n,
            "COL_STRIDE_K": col_stride_k,
            "COL_STRIDE_M": col_stride_m,
            "PARTIAL_STRIDE_SPLIT": int(partial.strides[0]),
            "PARTIAL_STRIDE_OC": int(partial.strides[1]),
            "PARTIAL_STRIDE_K": int(partial.strides[2]),
            "BLOCK_OC": block_oc,
            "BLOCK_CI_K": block_ci_k,
            "BLOCK_M": block_m,
        },
        tuning=CONV_WGRAD_TUNING,
        tuning_key_value=int(d["n_outputs"]),
        grid_spec=GridSpec(
            "conv_private_wgrad",
            (cout_per_group, reduction_extent, num_splits),
        ),
        dependencies=("wgrad_im2col",),
        num_warps=4,
        num_stages=2,
    )
    reduce = _wgrad_reduce_stage(
        node,
        workspace_name="wgrad_partial",
        num_splits=num_splits,
        cout=c_out,
        cin_per_group=cin_per_group,
        kh=kh,
        kw=kw,
        partial_strides=partial.strides,
        dependency="wgrad_split",
    )
    return NodePlan(
        operation,
        (pack, split, reduce),
        (columns, partial),
        workspace_size,
    )


def _convolution_plan(node: Mapping[str, Any]) -> NodePlan:
    operation = str(node["operation"])
    if operation in ("conv2d_fprop", "convolution_fprop"):
        private = _private_fprop_plan(node)
    elif operation == "convolution_dgrad":
        private = _private_dgrad_plan(node)
    else:
        private = _private_wgrad_plan(node)
    if private is not None:
        private.validate_dependencies()
        return private
    return NodePlan(operation, (_convolution_stage(node),))
