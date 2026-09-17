# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Iluvatar dispatch / convolution implementation."""

from __future__ import annotations

from .nn_common import CONV_TUNING_BY_OPERATION
from .nn_common import DTYPE_IDS
from .nn_common import FLOAT_TYPES
from .nn_common import GridSpec
from .nn_common import KernelStagePlan
from .nn_common import MAX_I32
from .nn_common import MAX_I64
from .nn_common import NodePlan
from .nn_common import REGISTRY_FUNCTIONS
from .nn_common import UNTUNED_TUNING
from .nn_common import WorkspaceTensor
from .nn_common import _aligned_size
from .nn_common import _ceil_div
from .nn_common import _checked_product
from .nn_common import _is_contiguous
from .nn_common import _make_stage
from .nn_common import _padded_nd
from .nn_common import _padded_spatial
from .nn_common import _require_integer
from .nn_common import _require_integer_list
from .nn_common import _require_object
from .nn_common import _same_dtype
from .nn_common import _tensor_pointer
from .nn_common import _workspace_pointer
from .nn_common import _workspace_tensor
from dataclasses import replace
from typing import Any
from typing import Mapping


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
            mode = _require_integer(p, "convolution_mode", minimum=0, maximum=1)
            if mode != 0:
                raise ValueError("convolution FProp supports CROSS_CORRELATION only")
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

    precision = p.get("input_precision", 0)
    if type(precision) is not int or precision not in (0, 1, 2):
        raise ValueError("convolution input_precision must be 0, 1 or 2")
    if precision and image["data_type"] != "float32":
        raise ValueError("explicit convolution precision requires FP32 inputs")
    tensors = (image, weight, result)
    data_type = _same_dtype(tensors, FLOAT_TYPES, "convolution")
    if any(len(tensor["dimensions"]) != rank for tensor in tensors):
        raise ValueError("convolution tensor ranks must equal spatial_rank + 2")
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
        raise ValueError("parameters.n_outputs is inconsistent with the output tensor")
    return {
        "input_precision": precision,
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
        "INPUT_PRECISION": int(d["input_precision"]),
    }
    for prefix, strides, axes in (
        ("X", image_strides, ("N", "C", "D", "H", "W")),
        ("W", weight_strides, ("K", "C", "D", "H", "W")),
        ("Y", result_strides, ("N", "C", "D", "H", "W")),
    ):
        for axis, value in zip(axes, strides, strict=True):
            common[f"{prefix}_STRIDE_{axis}"] = value

    if operation in ("conv2d_fprop", "convolution_fprop"):
        fused_bias_relu = node.get("fused_bias_relu", False)
        if not isinstance(fused_bias_relu, bool):
            raise ValueError("fused_bias_relu must be boolean")
        if fused_bias_relu and spatial_rank != 2:
            raise ValueError("Conv-Bias-ReLU fusion supports 2D convolution only")
        bias_port = "bias" if fused_bias_relu else "input"
        pointers = (
            _tensor_pointer(node, "x_ptr", "input"),
            _tensor_pointer(node, "w_ptr", "filter"),
            # Unfused FProp has no bias port. HAS_BIAS=False makes the input
            # alias a safe placeholder while retaining the exact kernel ABI.
            _tensor_pointer(node, "bias_ptr", bias_port),
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
                "INPUT_PRECISION": int(d["input_precision"]),
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
                "BATCH": int(d["batch"]),
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
                "HAS_BIAS": fused_bias_relu,
                "APPLY_RELU": fused_bias_relu,
                "BIAS_STRIDE_C": (
                    int(node["port_tensors"]["bias"]["strides"][1])
                    if fused_bias_relu
                    else 1
                ),
                "BLOCK_OC": 32,
                "BLOCK_HW": 32,
                "BLOCK_K": 32,
                "GROUP_M": 8,
                "DTYPE_ID": DTYPE_IDS[str(d["data_type"])],
                "INPUT_PRECISION": int(d["input_precision"]),
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
    stage = _make_stage(
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
    if (
        operation in ("conv2d_fprop", "convolution_fprop")
        and function == "conv2d_spatial_nchw_kernel"
        and int(d["n_outputs"]) <= 64
    ):
        small_constants = dict(stage.constants)
        small_constants.update(
            {
                "BLOCK_OC": 16,
                "BLOCK_HW": 16,
                "BLOCK_K": 16,
                "GROUP_M": 4,
            }
        )
        stage = replace(
            stage,
            constants=small_constants,
            default_grid=grid.evaluate(small_constants),
            tuning=UNTUNED_TUNING,
            default_num_warps=2,
            default_num_stages=1,
        )
        stage.validate_launch()
    return stage


def _can_use_fp32_ml_stem_fprop(node: Mapping[str, Any]) -> bool:
    if node["operation"] not in ("conv2d_fprop", "convolution_fprop"):
        return False
    derived = _require_object(node["derived"], "node.derived")
    image = _require_object(derived["image"], "node.derived.image")
    weight = _require_object(derived["weight"], "node.derived.weight")
    result = _require_object(derived["result"], "node.derived.result")
    shapes = (
        tuple(int(value) for value in image["dimensions"]),
        tuple(int(value) for value in weight["dimensions"]),
        tuple(int(value) for value in result["dimensions"]),
    )
    return (
        int(derived["spatial_rank"]) == 2
        and str(derived["data_type"]) == "float32"
        and shapes
        == (
            (1, 3, 640, 640),
            (64, 3, 3, 3),
            (1, 64, 320, 320),
        )
        and list(derived["stride"]) == [2, 2]
        and list(derived["pre_padding"]) == [1, 1]
        and list(derived["post_padding"]) == [1, 1]
        and list(derived["dilation"]) == [1, 1]
        and int(derived["groups"]) == 1
        and not bool(derived["flip_filter"])
        and not bool(node.get("fused_bias_relu", False))
        and _is_contiguous(image)
        and _is_contiguous(weight)
        and _is_contiguous(result)
    )


def _fp32_ml_stem_fprop_plan(node: Mapping[str, Any]) -> NodePlan:
    operation = str(node["operation"])
    derived = _require_object(node["derived"], "node.derived")
    stage = _make_stage(
        operation=operation,
        stage_name=f"{operation}_fp32_ml_stem",
        function_name="conv2d_fp32_ml_stem_kernel",
        pointer_arguments=(
            _tensor_pointer(node, "x_ptr", "input"),
            _tensor_pointer(node, "w_ptr", "filter"),
            _tensor_pointer(node, "y_ptr", "output"),
        ),
        constants={},
        tuning=UNTUNED_TUNING,
        tuning_key_value=int(derived["n_outputs"]),
        # 320x320 output pixels are partitioned into 400 fixed 4x64 tiles.
        grid_spec=GridSpec("fixed", (400, 1, 1)),
        num_warps=16,
        num_stages=3,
    )
    return NodePlan(operation, (stage,))


def _can_use_1x1_fprop(node: Mapping[str, Any]) -> bool:
    if node["operation"] not in ("conv2d_fprop", "convolution_fprop"):
        return False
    derived = _require_object(node["derived"], "node.derived")
    image = _require_object(derived["image"], "node.derived.image")
    weight = _require_object(derived["weight"], "node.derived.weight")
    result = _require_object(derived["result"], "node.derived.result")
    return (
        int(derived["spatial_rank"]) == 2
        and list(weight["dimensions"][-2:]) == [1, 1]
        and list(derived["stride"]) == [1, 1]
        and list(derived["pre_padding"]) == [0, 0]
        and list(derived["post_padding"]) == [0, 0]
        and list(derived["dilation"]) == [1, 1]
        and _is_contiguous(image)
        and _is_contiguous(weight)
        and _is_contiguous(result)
    )


def _one_by_one_fprop_plan(node: Mapping[str, Any]) -> NodePlan:
    derived = _require_object(node["derived"], "node.derived")
    result = _require_object(derived["result"], "node.derived.result")
    base = _convolution_stage(node)
    fused_bias_relu = bool(node.get("fused_bias_relu", False))
    constants = {
        "HW": int(result["dimensions"][2]) * int(result["dimensions"][3]),
        "C_IN": int(derived["in_channels"]),
        "C_OUT": int(derived["out_channels"]),
        "CIN_PER_GROUP": int(derived["in_per_group"]),
        "COUT_PER_GROUP": int(derived["out_per_group"]),
        "GROUPS": int(derived["groups"]),
        "SWAP_GRID": False,
        "HAS_BIAS": fused_bias_relu,
        "APPLY_RELU": fused_bias_relu,
        "BIAS_STRIDE": (
            int(node["port_tensors"]["bias"]["strides"][1]) if fused_bias_relu else 0
        ),
        "BLOCK_OC": 32,
        "BLOCK_HW": 32,
        "BLOCK_K": 32,
        "GROUP_M": 8,
    }
    grid_spec = GridSpec(
        "conv2d",
        (
            int(constants["HW"]),
            int(constants["COUT_PER_GROUP"]),
            int(derived["batch"]) * int(derived["groups"]),
        ),
    )
    stage = replace(
        base,
        function_name="conv2d_1x1_nchw_kernel",
        constants=constants,
        default_grid=grid_spec.evaluate(constants),
        grid_spec=grid_spec,
    )
    use_fixed_winner = (
        str(derived["data_type"]) in ("float32", "bfloat16")
        and int(derived["batch"]) == 8
        and int(constants["HW"]) == 784
        and int(constants["C_IN"]) == 64
        and int(constants["C_OUT"]) == 128
        and int(constants["CIN_PER_GROUP"]) == 64
        and int(constants["COUT_PER_GROUP"]) == 128
        and int(constants["GROUPS"]) == 1
    )
    if use_fixed_winner:
        exact_constants = {
            **constants,
            "BLOCK_OC": 128,
            "BLOCK_HW": 128,
            "BLOCK_K": 64,
            "GROUP_M": 8,
            "SWAP_GRID": True,
        }
        exact_grid = GridSpec("fixed", (8, 7, 1))
        stage = replace(
            stage,
            stage_name=f"{stage.stage_name}_exact_1x1",
            constants=exact_constants,
            default_grid=exact_grid.evaluate(exact_constants),
            tuning=UNTUNED_TUNING,
            default_num_warps=8,
            grid_spec=exact_grid,
            default_num_stages=1,
        )
    if stage.function_name not in REGISTRY_FUNCTIONS[stage.operation]:
        raise ValueError("1x1 FProp kernel is absent from the registry")
    stage.validate_launch()
    return NodePlan(str(node["operation"]), (stage,))


def _can_use_3x3_im2col_fprop(node: Mapping[str, Any]) -> bool:
    if node["operation"] not in ("conv2d_fprop", "convolution_fprop"):
        return False
    derived = _require_object(node["derived"], "node.derived")
    image = _require_object(derived["image"], "node.derived.image")
    weight = _require_object(derived["weight"], "node.derived.weight")
    result = _require_object(derived["result"], "node.derived.result")
    shapes = (
        tuple(int(value) for value in image["dimensions"]),
        tuple(int(value) for value in weight["dimensions"]),
        tuple(int(value) for value in result["dimensions"]),
    )
    standard = (
        shapes
        == (
            (8, 32, 32, 32),
            (64, 32, 3, 3),
            (8, 64, 32, 32),
        )
        and list(derived["stride"]) == [1, 1]
        and list(derived["pre_padding"]) == [1, 1]
        and list(derived["post_padding"]) == [1, 1]
        and list(derived["dilation"]) == [1, 1]
    )
    stride2 = (
        shapes
        == (
            (8, 64, 56, 56),
            (128, 64, 3, 3),
            (8, 128, 28, 28),
        )
        and list(derived["stride"]) == [2, 2]
        and list(derived["pre_padding"]) == [1, 1]
        and list(derived["post_padding"]) == [1, 1]
        and list(derived["dilation"]) == [1, 1]
    )
    dilation = (
        shapes
        == (
            (4, 64, 32, 32),
            (64, 64, 3, 3),
            (4, 64, 32, 32),
        )
        and list(derived["stride"]) == [1, 1]
        and list(derived["pre_padding"]) == [2, 2]
        and list(derived["post_padding"]) == [2, 2]
        and list(derived["dilation"]) == [2, 2]
    )
    stem_ml_x = (
        shapes
        in (
            (
                (1, 3, 640, 640),
                (64, 3, 3, 3),
                (1, 64, 320, 320),
            ),
            (
                (1, 3, 640, 640),
                (96, 3, 3, 3),
                (1, 96, 320, 320),
            ),
        )
        and list(derived["stride"]) == [2, 2]
        and list(derived["pre_padding"]) == [1, 1]
        and list(derived["post_padding"]) == [1, 1]
        and list(derived["dilation"]) == [1, 1]
    )
    return (
        str(derived["data_type"]) in ("float16", "bfloat16")
        and not bool(node.get("fused_bias_relu", False))
        and (standard or stride2 or dilation or stem_ml_x)
        and int(derived["groups"]) == 1
        and not bool(derived["flip_filter"])
        and _is_contiguous(image)
        and _is_contiguous(weight)
        and _is_contiguous(result)
    )


def _3x3_im2col_fprop_plan(node: Mapping[str, Any]) -> NodePlan:
    operation = str(node["operation"])
    derived = _require_object(node["derived"], "node.derived")
    image = _require_object(derived["image"], "node.derived.image")
    weight = _require_object(derived["weight"], "node.derived.weight")
    result = _require_object(derived["result"], "node.derived.result")
    batch = int(derived["batch"])
    workspace_type = str(derived["data_type"])
    packed_columns = int(derived["in_per_group"]) * 9
    output_area = int(result["dimensions"][2]) * int(result["dimensions"][3])
    is_stride2 = list(derived["stride"]) == [2, 2]
    is_stem = int(derived["in_per_group"]) == 3 and output_area == 320 * 320
    is_standard = not is_stride2 and list(derived["dilation"]) == [1, 1]
    pack_num_warps = 2 if is_standard else 8
    columns = _workspace_tensor(
        "fprop_3x3_columns",
        workspace_type,
        (batch, packed_columns, output_area),
    )

    pack_constants = {
        "XH": int(image["dimensions"][2]),
        "XW": int(image["dimensions"][3]),
        "OH": int(result["dimensions"][2]),
        "OW": int(result["dimensions"][3]),
        "CIN_PER_GROUP": int(derived["in_per_group"]),
        "KH": int(weight["dimensions"][2]),
        "KW": int(weight["dimensions"][3]),
        "STRIDE_H": int(derived["stride"][0]),
        "STRIDE_W": int(derived["stride"][1]),
        "PAD_H": int(derived["pre_padding"][0]),
        "PAD_W": int(derived["pre_padding"][1]),
        "DIL_H": int(derived["dilation"][0]),
        "DIL_W": int(derived["dilation"][1]),
        "X_STRIDE_N": int(image["strides"][0]),
        "X_STRIDE_C": int(image["strides"][1]),
        "X_STRIDE_H": int(image["strides"][2]),
        "X_STRIDE_W": int(image["strides"][3]),
        "COLUMNS_STRIDE_N": packed_columns * output_area,
        "COLUMNS_STRIDE_K": output_area,
        "BLOCK_SIZE": 1024,
    }
    pack_stage = _make_stage(
        operation=operation,
        stage_name=f"{operation}_3x3_im2col",
        function_name="conv_fprop_2d_im2col_kernel",
        pointer_arguments=(
            _tensor_pointer(node, "x_ptr", "input"),
            _workspace_pointer("columns_ptr", columns.name, workspace_type),
        ),
        constants=pack_constants,
        tuning=UNTUNED_TUNING,
        tuning_key_value=int(derived["n_outputs"]),
        grid_spec=GridSpec(
            "fixed",
            (
                _ceil_div(output_area, int(pack_constants["BLOCK_SIZE"])),
                batch * int(derived["in_per_group"]) * 3,
                1,
            ),
        ),
        num_warps=pack_num_warps,
        num_stages=1,
    )

    if is_stem:
        block_m = 64 if int(derived["out_channels"]) == 64 else 128
        block_n = 128 if int(derived["out_channels"]) == 64 else 256
        block_k = 32
        matmul_num_warps = 8
    else:
        block_m = 128 if is_stride2 else 64
        block_n = 128
        block_k = 64 if is_stride2 else 32
        matmul_num_warps = 16 if is_stride2 else 8
    matmul_constants = {
        "M": int(derived["out_channels"]),
        "N": output_area,
        "K": packed_columns,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
    }
    matmul_stage = _make_stage(
        operation=operation,
        stage_name=f"{operation}_3x3_matmul",
        function_name="conv_fprop_2d_packed_matmul_kernel",
        pointer_arguments=(
            _tensor_pointer(node, "weight_ptr", "filter"),
            _workspace_pointer("columns_ptr", columns.name, workspace_type),
            _tensor_pointer(node, "output_ptr", "output"),
        ),
        constants=matmul_constants,
        tuning=UNTUNED_TUNING,
        tuning_key_value=int(derived["n_outputs"]),
        grid_spec=GridSpec(
            "fixed",
            (
                _ceil_div(int(derived["out_channels"]), block_m)
                * _ceil_div(output_area, block_n),
                batch,
                1,
            ),
        ),
        dependencies=(pack_stage.stage_name,),
        num_warps=matmul_num_warps,
        num_stages=1,
    )
    plan = NodePlan(operation, (pack_stage, matmul_stage), (columns,), columns.size)
    plan.validate_dependencies()
    return plan


def _can_use_low_channel_3x3_fprop(node: Mapping[str, Any]) -> bool:
    if node["operation"] not in ("conv2d_fprop", "convolution_fprop"):
        return False
    derived = _require_object(node["derived"], "node.derived")
    image = _require_object(derived["image"], "node.derived.image")
    weight = _require_object(derived["weight"], "node.derived.weight")
    result = _require_object(derived["result"], "node.derived.result")
    return (
        int(derived["spatial_rank"]) == 2
        and str(derived["data_type"]) in ("float16", "bfloat16")
        and int(derived["in_per_group"]) <= 64
        and list(weight["dimensions"][-2:]) == [3, 3]
        and list(derived["stride"]) == [1, 1]
        and list(derived["pre_padding"]) == [1, 1]
        and list(derived["post_padding"]) == [1, 1]
        and list(derived["dilation"]) == [1, 1]
        and not bool(derived["flip_filter"])
        and _is_contiguous(image)
        and _is_contiguous(weight)
        and _is_contiguous(result)
    )


def _low_channel_3x3_fprop_plan(node: Mapping[str, Any]) -> NodePlan:
    derived = _require_object(node["derived"], "node.derived")
    image = _require_object(derived["image"], "node.derived.image")
    result = _require_object(derived["result"], "node.derived.result")
    base = _convolution_stage(node)
    fused_bias_relu = bool(node.get("fused_bias_relu", False))
    constants = {
        "XH": int(image["dimensions"][2]),
        "XW": int(image["dimensions"][3]),
        "OH": int(result["dimensions"][2]),
        "OW": int(result["dimensions"][3]),
        "STRIDE_H": int(derived["stride"][0]),
        "STRIDE_W": int(derived["stride"][1]),
        "C_IN": int(derived["in_channels"]),
        "C_OUT": int(derived["out_channels"]),
        "CIN_PER_GROUP": int(derived["in_per_group"]),
        "COUT_PER_GROUP": int(derived["out_per_group"]),
        "GROUPS": int(derived["groups"]),
        "HAS_BIAS": fused_bias_relu,
        "APPLY_RELU": fused_bias_relu,
        "BIAS_STRIDE": (
            int(node["port_tensors"]["bias"]["strides"][1]) if fused_bias_relu else 0
        ),
        "BLOCK_OC": 32,
        "BLOCK_HW": 32,
        "BLOCK_K": 32,
        "GROUP_M": 8,
    }
    grid_spec = GridSpec(
        "conv2d",
        (
            int(constants["OH"]) * int(constants["OW"]),
            int(constants["COUT_PER_GROUP"]),
            int(derived["batch"]) * int(derived["groups"]),
        ),
    )
    stage = replace(
        base,
        function_name="conv2d_3x3_nchw_pad1_kernel",
        constants=constants,
        default_grid=grid_spec.evaluate(constants),
        grid_spec=grid_spec,
    )
    if stage.function_name not in REGISTRY_FUNCTIONS[stage.operation]:
        raise ValueError("low-channel 3x3 FProp kernel is absent from the registry")
    stage.validate_launch()
    return NodePlan(str(node["operation"]), (stage,))


def _can_use_p5_wgrad(node: Mapping[str, Any]) -> bool:
    if node["operation"] != "convolution_wgrad":
        return False
    derived = _require_object(node["derived"], "node.derived")
    ports = _require_object(node["port_tensors"], "node.port_tensors")
    loss = _require_object(ports["dy"], "node.port_tensors.dy")
    image = _require_object(ports["x"], "node.port_tensors.x")
    weight = _require_object(ports["dw"], "node.port_tensors.dw")
    shapes = (
        tuple(int(value) for value in loss["dimensions"]),
        tuple(int(value) for value in image["dimensions"]),
        tuple(int(value) for value in weight["dimensions"]),
    )
    return (
        int(derived["spatial_rank"]) == 2
        and str(derived["data_type"]) in ("float16", "float32")
        and shapes
        in {
            ((1, 256, 20, 20), (1, 128, 40, 40), (256, 128, 3, 3)),
            ((1, 512, 20, 20), (1, 256, 40, 40), (512, 256, 3, 3)),
            ((1, 512, 20, 20), (1, 512, 40, 40), (512, 512, 3, 3)),
            ((1, 768, 20, 20), (1, 768, 40, 40), (768, 768, 3, 3)),
        }
        and int(derived["groups"]) == 1
        and list(derived["stride"]) == [2, 2]
        and list(derived["pre_padding"]) == [1, 1]
        and list(derived["post_padding"]) == [1, 1]
        and list(derived["dilation"]) == [1, 1]
        and not bool(derived["flip_filter"])
        and _is_contiguous(loss)
        and _is_contiguous(image)
        and _is_contiguous(weight)
    )


def _p5_wgrad_plan(node: Mapping[str, Any]) -> NodePlan:
    operation = str(node["operation"])
    derived = _require_object(node["derived"], "node.derived")
    loss = _require_object(derived["result"], "node.derived.result")
    image = _require_object(derived["image"], "node.derived.image")
    source_type = str(derived["data_type"])
    compute_type = "float16"
    packed_columns = int(derived["in_per_group"]) * 9
    output_area = int(loss["dimensions"][2]) * int(loss["dimensions"][3])
    cursor = 0
    workspaces: list[WorkspaceTensor] = []
    stages: list[KernelStagePlan] = []
    matmul_dependencies: list[str] = []

    loss_workspace: WorkspaceTensor | None = None
    if source_type != compute_type:
        loss_workspace = _workspace_tensor(
            "wgrad_p5_loss_fp16",
            compute_type,
            loss["dimensions"],
            offset=cursor,
        )
        workspaces.append(loss_workspace)
        cursor += _aligned_size(loss_workspace.size)
        total = _checked_product(
            list(loss["dimensions"]),
            "P5 WGrad loss elements",
        )
        cast_stage = _make_stage(
            operation=operation,
            stage_name=f"{operation}_p5_cast_loss",
            function_name="conv_dgrad_cast_contiguous_kernel",
            pointer_arguments=(
                _tensor_pointer(node, "input_ptr", "dy"),
                _workspace_pointer(
                    "output_ptr",
                    loss_workspace.name,
                    compute_type,
                ),
            ),
            constants={"TOTAL": total, "BLOCK_SIZE": 256},
            tuning=UNTUNED_TUNING,
            tuning_key_value=total,
            grid_spec=GridSpec("linear", (total,)),
            num_warps=4,
            num_stages=1,
        )
        stages.append(cast_stage)
        matmul_dependencies.append(cast_stage.stage_name)

    packed = _workspace_tensor(
        "wgrad_p5_image_packed",
        compute_type,
        (output_area, packed_columns),
        offset=cursor,
    )
    workspaces.append(packed)
    cursor += _aligned_size(packed.size)

    input_channels = int(derived["in_per_group"])
    # The four exact P5 shapes fall into three measured CoreX launch regimes.
    if input_channels == 128:
        pack_block_m = 32
        pack_block_n = 64
        pack_num_warps = 4
        block_m = 32
        block_n = 64
        block_k = 64
        matmul_num_warps = 8
    elif input_channels == 256:
        pack_block_m = 64
        pack_block_n = 256
        pack_num_warps = 16
        block_m = 64
        block_n = 256
        block_k = 32
        matmul_num_warps = 16
    else:
        pack_block_m = 64
        pack_block_n = 256
        pack_num_warps = 16
        block_m = 128
        block_n = 256
        block_k = 32
        matmul_num_warps = 16

    pack_constants = {
        "IMAGE_H": int(image["dimensions"][2]),
        "IMAGE_W": int(image["dimensions"][3]),
        "OUTPUT_H": int(loss["dimensions"][2]),
        "OUTPUT_W": int(loss["dimensions"][3]),
        "CIN_PER_GROUP": int(derived["in_per_group"]),
        "IMAGE_STRIDE_C": int(image["strides"][1]),
        "IMAGE_STRIDE_H": int(image["strides"][2]),
        "IMAGE_STRIDE_W": int(image["strides"][3]),
        "BLOCK_M": pack_block_m,
        "BLOCK_N": pack_block_n,
    }
    pack_grid = GridSpec(
        "fixed",
        (
            _ceil_div(output_area, pack_block_m),
            _ceil_div(packed_columns, pack_block_n),
            1,
        ),
    )
    pack_stage = _make_stage(
        operation=operation,
        stage_name=f"{operation}_p5_pack",
        function_name="conv_wgrad_2d_p5_pack_kernel",
        pointer_arguments=(
            _tensor_pointer(node, "image_ptr", "x"),
            _workspace_pointer("packed_ptr", packed.name, compute_type),
        ),
        constants=pack_constants,
        tuning=UNTUNED_TUNING,
        tuning_key_value=int(derived["n_outputs"]),
        grid_spec=pack_grid,
        num_warps=pack_num_warps,
        num_stages=1,
    )
    stages.append(pack_stage)
    matmul_dependencies.append(pack_stage.stage_name)

    group_m = 4
    matmul_constants = {
        "M": int(derived["out_channels"]),
        "N": packed_columns,
        "K": output_area,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "GROUP_M": group_m,
    }
    matmul_grid = GridSpec(
        "fixed",
        (
            _ceil_div(int(derived["out_channels"]), block_m)
            * _ceil_div(packed_columns, block_n),
            1,
            1,
        ),
    )
    loss_pointer = (
        _workspace_pointer("loss_ptr", loss_workspace.name, compute_type)
        if loss_workspace is not None
        else _tensor_pointer(node, "loss_ptr", "dy")
    )
    matmul_stage = _make_stage(
        operation=operation,
        stage_name=f"{operation}_p5_matmul",
        function_name="conv_wgrad_2d_p5_matmul_kernel",
        pointer_arguments=(
            loss_pointer,
            _workspace_pointer("packed_ptr", packed.name, compute_type),
            _tensor_pointer(node, "out_ptr", "dw"),
        ),
        constants=matmul_constants,
        tuning=UNTUNED_TUNING,
        tuning_key_value=int(derived["n_outputs"]),
        grid_spec=matmul_grid,
        dependencies=tuple(matmul_dependencies),
        num_warps=matmul_num_warps,
        num_stages=1,
    )
    stages.append(matmul_stage)
    plan = NodePlan(
        operation,
        tuple(stages),
        tuple(workspaces),
        cursor,
    )
    plan.validate_dependencies()
    return plan


def _can_use_col_split_wgrad(node: Mapping[str, Any]) -> bool:
    if node["operation"] != "convolution_wgrad":
        return False
    derived = _require_object(node["derived"], "node.derived")
    ports = _require_object(node["port_tensors"], "node.port_tensors")
    loss = _require_object(ports["dy"], "node.port_tensors.dy")
    image = _require_object(ports["x"], "node.port_tensors.x")
    weight = _require_object(ports["dw"], "node.port_tensors.dw")
    output_channels = int(weight["dimensions"][0])
    shapes = (
        tuple(int(value) for value in loss["dimensions"]),
        tuple(int(value) for value in image["dimensions"]),
        tuple(int(value) for value in weight["dimensions"]),
    )
    stem = (
        shapes
        == (
            (1, output_channels, 320, 320),
            (1, 3, 640, 640),
            (output_channels, 3, 3, 3),
        )
        and output_channels in (16, 32, 64, 96)
        and list(derived["stride"]) == [2, 2]
    )
    return (
        int(derived["spatial_rank"]) == 2
        and str(derived["data_type"]) in ("float32", "float16")
        and stem
        and int(derived["groups"]) == 1
        and list(derived["pre_padding"]) == [1, 1]
        and list(derived["post_padding"]) == [1, 1]
        and list(derived["dilation"]) == [1, 1]
        and not bool(derived["flip_filter"])
        and _is_contiguous(loss)
        and _is_contiguous(image)
        and _is_contiguous(weight)
    )


def _col_split_wgrad_plan(node: Mapping[str, Any]) -> NodePlan:
    operation = str(node["operation"])
    derived = _require_object(node["derived"], "node.derived")
    loss = _require_object(derived["result"], "node.derived.result")
    image = _require_object(derived["image"], "node.derived.image")
    is_stem = list(image["dimensions"]) == [1, 3, 640, 640]
    num_splits = 64 if is_stem else 16
    packed_columns = int(derived["in_per_group"]) * 9
    output_elements = int(derived["out_channels"]) * packed_columns
    partial = _workspace_tensor(
        "wgrad_col_partial",
        "float32",
        (num_splits, int(derived["out_channels"]), packed_columns),
    )
    block_co = 32 if not is_stem or int(derived["out_channels"]) >= 32 else 16
    block_n = 32
    split_constants = {
        "M": int(derived["batch"])
        * int(loss["dimensions"][2])
        * int(loss["dimensions"][3]),
        "IMAGE_H": int(image["dimensions"][2]),
        "IMAGE_W": int(image["dimensions"][3]),
        "LOSS_H": int(loss["dimensions"][2]),
        "LOSS_W": int(loss["dimensions"][3]),
        "C_OUT": int(derived["out_channels"]),
        "CIN_PER_GROUP": int(derived["in_per_group"]),
        "COUT_PER_GROUP": int(derived["out_per_group"]),
        "IMAGE_STRIDE_N": int(image["strides"][0]),
        "IMAGE_STRIDE_C": int(image["strides"][1]),
        "IMAGE_STRIDE_H": int(image["strides"][2]),
        "IMAGE_STRIDE_W": int(image["strides"][3]),
        "LOSS_STRIDE_N": int(loss["strides"][0]),
        "LOSS_STRIDE_C": int(loss["strides"][1]),
        "LOSS_STRIDE_H": int(loss["strides"][2]),
        "LOSS_STRIDE_W": int(loss["strides"][3]),
        "STRIDE_H": int(derived["stride"][0]),
        "STRIDE_W": int(derived["stride"][1]),
        "PAD_H": int(derived["pre_padding"][0]),
        "PAD_W": int(derived["pre_padding"][1]),
        "DIL_H": int(derived["dilation"][0]),
        "DIL_W": int(derived["dilation"][1]),
        "FILTER_REVERSE": bool(derived["flip_filter"]),
        "NUM_SPLITS": num_splits,
        "SOURCE_FP32": str(derived["data_type"]) == "float32",
        "BLOCK_CO": block_co,
        "BLOCK_N": block_n,
        "BLOCK_M": 128,
    }
    split_stage = _make_stage(
        operation=operation,
        stage_name=f"{operation}_col_split",
        function_name="conv_wgrad_2d_col_split_kernel",
        pointer_arguments=(
            _tensor_pointer(node, "image_ptr", "x"),
            _tensor_pointer(node, "loss_ptr", "dy"),
            _workspace_pointer("partial_ptr", partial.name),
        ),
        constants=split_constants,
        tuning=UNTUNED_TUNING,
        tuning_key_value=int(derived["n_outputs"]),
        grid_spec=GridSpec(
            "fixed",
            (
                _ceil_div(int(derived["out_per_group"]), block_co)
                * _ceil_div(packed_columns, block_n),
                num_splits,
                1,
            ),
        ),
        num_warps=8,
        num_stages=1,
    )
    reduce_stage = _make_stage(
        operation=operation,
        stage_name=f"{operation}_col_reduce",
        function_name="conv_wgrad_2d_col_reduce_kernel",
        pointer_arguments=(
            _workspace_pointer("partial_ptr", partial.name),
            _tensor_pointer(node, "out_ptr", "dw"),
        ),
        constants={
            "TOTAL": output_elements,
            "NUM_SPLITS": num_splits,
            "BLOCK_SIZE": 256,
        },
        tuning=UNTUNED_TUNING,
        tuning_key_value=output_elements,
        grid_spec=GridSpec("linear", (output_elements,)),
        dependencies=(split_stage.stage_name,),
        num_warps=4,
        num_stages=1,
    )
    plan = NodePlan(
        operation,
        (split_stage, reduce_stage),
        (partial,),
        _aligned_size(partial.size),
    )
    plan.validate_dependencies()
    return plan


def _can_use_batched_3x3_wgrad(node: Mapping[str, Any]) -> bool:
    if node["operation"] != "convolution_wgrad":
        return False
    derived = _require_object(node["derived"], "node.derived")
    ports = _require_object(node["port_tensors"], "node.port_tensors")
    loss = _require_object(ports["dy"], "node.port_tensors.dy")
    image = _require_object(ports["x"], "node.port_tensors.x")
    weight = _require_object(ports["dw"], "node.port_tensors.dw")
    shapes = (
        tuple(int(value) for value in loss["dimensions"]),
        tuple(int(value) for value in image["dimensions"]),
        tuple(int(value) for value in weight["dimensions"]),
    )
    standard = (
        shapes == ((8, 64, 32, 32), (8, 32, 32, 32), (64, 32, 3, 3))
        and list(derived["stride"]) == [1, 1]
        and list(derived["pre_padding"]) == [1, 1]
        and list(derived["post_padding"]) == [1, 1]
    )
    stride2 = (
        shapes == ((8, 128, 28, 28), (8, 64, 56, 56), (128, 64, 3, 3))
        and list(derived["stride"]) == [2, 2]
        and list(derived["pre_padding"]) == [1, 1]
        and list(derived["post_padding"]) == [1, 1]
    )
    return (
        int(derived["spatial_rank"]) == 2
        and str(derived["data_type"]) in ("float32", "float16")
        and (standard or stride2)
        and int(derived["groups"]) == 1
        and list(derived["dilation"]) == [1, 1]
        and not bool(derived["flip_filter"])
        and _is_contiguous(loss)
        and _is_contiguous(image)
        and _is_contiguous(weight)
    )


def _batched_3x3_wgrad_plan(node: Mapping[str, Any]) -> NodePlan:
    operation = str(node["operation"])
    derived = _require_object(node["derived"], "node.derived")
    loss = _require_object(derived["result"], "node.derived.result")
    image = _require_object(derived["image"], "node.derived.image")
    source_type = str(derived["data_type"])
    compute_type = "float16"
    batch = int(derived["batch"])
    output_area = int(loss["dimensions"][2]) * int(loss["dimensions"][3])
    packed_columns = int(derived["in_per_group"]) * 9
    cursor = 0
    workspaces: list[WorkspaceTensor] = []
    stages: list[KernelStagePlan] = []
    matmul_dependencies: list[str] = []

    loss_workspace: WorkspaceTensor | None = None
    if source_type != compute_type:
        loss_workspace = _workspace_tensor(
            "wgrad_batched_loss_fp16",
            compute_type,
            loss["dimensions"],
            offset=cursor,
        )
        workspaces.append(loss_workspace)
        cursor += _aligned_size(loss_workspace.size)
        loss_elements = _checked_product(
            list(loss["dimensions"]),
            "batched WGrad loss elements",
        )
        cast_stage = _make_stage(
            operation=operation,
            stage_name=f"{operation}_batched_cast_loss",
            function_name="conv_dgrad_cast_contiguous_kernel",
            pointer_arguments=(
                _tensor_pointer(node, "input_ptr", "dy"),
                _workspace_pointer(
                    "output_ptr",
                    loss_workspace.name,
                    compute_type,
                ),
            ),
            constants={"TOTAL": loss_elements, "BLOCK_SIZE": 256},
            tuning=UNTUNED_TUNING,
            tuning_key_value=loss_elements,
            grid_spec=GridSpec("linear", (loss_elements,)),
            num_warps=4,
            num_stages=1,
        )
        stages.append(cast_stage)
        matmul_dependencies.append(cast_stage.stage_name)

    packed = _workspace_tensor(
        "wgrad_batched_image_packed",
        compute_type,
        (batch, output_area, packed_columns),
        offset=cursor,
    )
    workspaces.append(packed)
    cursor += _aligned_size(packed.size)
    if int(derived["in_per_group"]) == 32:
        pack_block_m = 128
        pack_block_n = 32
        pack_num_warps = 8
    else:
        pack_block_m = 128
        pack_block_n = 64
        pack_num_warps = 16
    pack_constants = {
        "IMAGE_H": int(image["dimensions"][2]),
        "IMAGE_W": int(image["dimensions"][3]),
        "OUTPUT_H": int(loss["dimensions"][2]),
        "OUTPUT_W": int(loss["dimensions"][3]),
        "CIN_PER_GROUP": int(derived["in_per_group"]),
        "IMAGE_STRIDE_N": int(image["strides"][0]),
        "IMAGE_STRIDE_C": int(image["strides"][1]),
        "IMAGE_STRIDE_H": int(image["strides"][2]),
        "IMAGE_STRIDE_W": int(image["strides"][3]),
        "STRIDE_H": int(derived["stride"][0]),
        "STRIDE_W": int(derived["stride"][1]),
        "PAD_H": int(derived["pre_padding"][0]),
        "PAD_W": int(derived["pre_padding"][1]),
        "DIL_H": int(derived["dilation"][0]),
        "DIL_W": int(derived["dilation"][1]),
        "BLOCK_M": pack_block_m,
        "BLOCK_N": pack_block_n,
    }
    pack_stage = _make_stage(
        operation=operation,
        stage_name=f"{operation}_batched_pack",
        function_name="conv_wgrad_2d_pack_image_kernel",
        pointer_arguments=(
            _tensor_pointer(node, "image_ptr", "x"),
            _workspace_pointer("packed_ptr", packed.name, compute_type),
        ),
        constants=pack_constants,
        tuning=UNTUNED_TUNING,
        tuning_key_value=int(derived["n_outputs"]),
        grid_spec=GridSpec(
            "fixed",
            (
                _ceil_div(output_area, pack_block_m),
                _ceil_div(packed_columns, pack_block_n),
                batch,
            ),
        ),
        num_warps=pack_num_warps,
        num_stages=1,
    )
    stages.append(pack_stage)
    matmul_dependencies.append(pack_stage.stage_name)

    large_batched_shape = int(derived["in_per_group"]) == 64
    use_wide_matmul = (
        large_batched_shape
        and source_type == "float16"
        and batch == 8
        and output_area == 784
        and packed_columns == 576
        and int(derived["out_channels"]) == 128
    )
    partial = _workspace_tensor(
        "wgrad_batched_partial",
        "float32",
        (batch, int(derived["out_channels"]), packed_columns),
        offset=cursor,
    )
    workspaces.append(partial)
    cursor += _aligned_size(partial.size)
    block_m = 128 if use_wide_matmul else (64 if large_batched_shape else 32)
    block_n = 256 if use_wide_matmul else (128 if large_batched_shape else 64)
    block_k = 64
    matmul_num_warps = 16 if large_batched_shape else 8
    matmul_num_stages = 2 if large_batched_shape else 1
    group_m = 4
    matmul_constants = {
        "M": int(derived["out_channels"]),
        "N": packed_columns,
        "K": output_area,
        "LOSS_STRIDE_N": int(loss["strides"][0]),
        "LOSS_STRIDE_C": int(loss["strides"][1]),
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "GROUP_M": group_m,
    }
    loss_pointer = (
        _workspace_pointer("loss_ptr", loss_workspace.name, compute_type)
        if loss_workspace is not None
        else _tensor_pointer(node, "loss_ptr", "dy")
    )
    matmul_stage = _make_stage(
        operation=operation,
        stage_name=f"{operation}_batched_matmul",
        function_name=(
            "conv_wgrad_2d_batched_wide_kernel"
            if use_wide_matmul
            else "conv_wgrad_2d_batched_matmul_kernel"
        ),
        pointer_arguments=(
            loss_pointer,
            _workspace_pointer("packed_ptr", packed.name, compute_type),
            _workspace_pointer("partial_ptr", partial.name),
        ),
        constants=matmul_constants,
        tuning=UNTUNED_TUNING,
        tuning_key_value=int(derived["n_outputs"]),
        grid_spec=GridSpec(
            "fixed",
            (
                _ceil_div(int(derived["out_channels"]), block_m)
                * (2 if use_wide_matmul else _ceil_div(packed_columns, block_n)),
                batch,
                1,
            ),
        ),
        dependencies=tuple(matmul_dependencies),
        num_warps=matmul_num_warps,
        num_stages=matmul_num_stages,
    )
    stages.append(matmul_stage)

    output_elements = int(derived["out_channels"]) * packed_columns
    reduce_stage = _make_stage(
        operation=operation,
        stage_name=f"{operation}_batched_reduce",
        function_name="conv_wgrad_2d_batched_reduce_kernel",
        pointer_arguments=(
            _workspace_pointer("partial_ptr", partial.name),
            _tensor_pointer(node, "out_ptr", "dw"),
        ),
        constants={
            "OUTPUT_ELEMENTS": output_elements,
            "C_OUT": int(derived["out_channels"]),
            "CIN_PER_GROUP": int(derived["in_per_group"]),
            "BATCH": batch,
            "BLOCK_SIZE": 256,
        },
        tuning=UNTUNED_TUNING,
        tuning_key_value=output_elements,
        grid_spec=GridSpec("linear", (output_elements,)),
        dependencies=(matmul_stage.stage_name,),
        num_warps=4,
        num_stages=1,
    )
    stages.append(reduce_stage)
    plan = NodePlan(
        operation,
        tuple(stages),
        tuple(workspaces),
        cursor,
    )
    plan.validate_dependencies()
    return plan


def _can_use_standard_1d_3tap_wgrad(node: Mapping[str, Any]) -> bool:
    if node["operation"] != "convolution_wgrad":
        return False
    derived = _require_object(node["derived"], "node.derived")
    ports = _require_object(node["port_tensors"], "node.port_tensors")
    loss = _require_object(ports["dy"], "node.port_tensors.dy")
    image = _require_object(ports["x"], "node.port_tensors.x")
    weight = _require_object(ports["dw"], "node.port_tensors.dw")
    return (
        int(derived["spatial_rank"]) == 1
        and str(derived["data_type"]) == "float16"
        and list(loss["dimensions"]) == [16, 64, 256]
        and list(image["dimensions"]) == [16, 32, 256]
        and list(weight["dimensions"]) == [64, 32, 3]
        and int(derived["groups"]) == 1
        and list(derived["stride"]) == [1]
        and list(derived["pre_padding"]) == [1]
        and list(derived["post_padding"]) == [1]
        and list(derived["dilation"]) == [1]
        and not bool(derived["flip_filter"])
        and _is_contiguous(loss)
        and _is_contiguous(image)
        and _is_contiguous(weight)
    )


def _standard_1d_3tap_wgrad_plan(node: Mapping[str, Any]) -> NodePlan:
    operation = str(node["operation"])
    derived = _require_object(node["derived"], "node.derived")
    loss = _require_object(derived["result"], "node.derived.result")
    image = _require_object(derived["image"], "node.derived.image")
    weight = _require_object(derived["weight"], "node.derived.weight")

    num_splits = int(derived["batch"])
    splits_per_batch = 1
    block_co = 32
    block_ci = 32
    partial = _workspace_tensor(
        "wgrad_1d_3tap_partial",
        "float32",
        (
            num_splits,
            int(derived["out_channels"]),
            int(derived["in_per_group"]),
            3,
        ),
    )
    split_constants = {
        "LENGTH": int(loss["dimensions"][2]),
        "C_OUT": int(derived["out_channels"]),
        "CIN_PER_GROUP": int(derived["in_per_group"]),
        "COUT_PER_GROUP": int(derived["out_per_group"]),
        "IMAGE_STRIDE_N": int(image["strides"][0]),
        "IMAGE_STRIDE_C": int(image["strides"][1]),
        "IMAGE_STRIDE_L": int(image["strides"][2]),
        "LOSS_STRIDE_N": int(loss["strides"][0]),
        "LOSS_STRIDE_C": int(loss["strides"][1]),
        "LOSS_STRIDE_L": int(loss["strides"][2]),
        "PAD_LEFT": int(derived["pre_padding"][0]),
        "SPLITS_PER_N": splits_per_batch,
        "BLOCK_CO": block_co,
        "BLOCK_CI": block_ci,
        "BLOCK_M": 128,
    }
    split_grid = GridSpec(
        "fixed",
        (
            _ceil_div(int(derived["out_per_group"]), block_co)
            * _ceil_div(int(derived["in_per_group"]), block_ci),
            num_splits,
            3,
        ),
    )
    split_stage = _make_stage(
        operation=operation,
        stage_name=f"{operation}_1d_3tap_split",
        function_name="conv_wgrad_1d_3tap_split_kernel",
        pointer_arguments=(
            _tensor_pointer(node, "image_ptr", "x"),
            _tensor_pointer(node, "loss_ptr", "dy"),
            _workspace_pointer("partial_ptr", partial.name),
        ),
        constants=split_constants,
        tuning=UNTUNED_TUNING,
        tuning_key_value=int(derived["n_outputs"]),
        grid_spec=split_grid,
        num_warps=8,
        num_stages=1,
    )

    reduce_block_co = 16
    reduce_block_ci = 32
    reduce_constants = {
        "C_OUT": int(derived["out_channels"]),
        "CIN_PER_GROUP": int(derived["in_per_group"]),
        "COUT_PER_GROUP": int(derived["out_per_group"]),
        "OUT_STRIDE_O": int(weight["strides"][0]),
        "OUT_STRIDE_I": int(weight["strides"][1]),
        "OUT_STRIDE_K": int(weight["strides"][2]),
        "NUM_SPLITS": num_splits,
        "BLOCK_CO": reduce_block_co,
        "BLOCK_CI": reduce_block_ci,
    }
    reduce_grid = GridSpec(
        "fixed",
        (
            _ceil_div(int(derived["out_per_group"]), reduce_block_co)
            * _ceil_div(int(derived["in_per_group"]), reduce_block_ci),
            3,
            1,
        ),
    )
    reduce_stage = _make_stage(
        operation=operation,
        stage_name=f"{operation}_1d_3tap_reduce",
        function_name="conv_wgrad_1d_3tap_reduce_kernel",
        pointer_arguments=(
            _workspace_pointer("partial_ptr", partial.name),
            _tensor_pointer(node, "out_ptr", "dw"),
        ),
        constants=reduce_constants,
        tuning=UNTUNED_TUNING,
        tuning_key_value=int(derived["n_outputs"]),
        grid_spec=reduce_grid,
        dependencies=(split_stage.stage_name,),
        num_warps=4,
        num_stages=1,
    )
    plan = NodePlan(
        operation,
        (split_stage, reduce_stage),
        (partial,),
        _aligned_size(partial.size),
    )
    plan.validate_dependencies()
    return plan


def _can_use_standard_1x1_wgrad(node: Mapping[str, Any]) -> bool:
    if node["operation"] != "convolution_wgrad":
        return False
    derived = _require_object(node["derived"], "node.derived")
    ports = _require_object(node["port_tensors"], "node.port_tensors")
    loss = _require_object(ports["dy"], "node.port_tensors.dy")
    image = _require_object(ports["x"], "node.port_tensors.x")
    weight = _require_object(ports["dw"], "node.port_tensors.dw")
    return (
        int(derived["spatial_rank"]) == 2
        and str(derived["data_type"]) in ("float32", "float16")
        and list(loss["dimensions"]) == [8, 128, 28, 28]
        and list(image["dimensions"]) == [8, 64, 28, 28]
        and list(weight["dimensions"]) == [128, 64, 1, 1]
        and int(derived["groups"]) == 1
        and list(derived["stride"]) == [1, 1]
        and list(derived["pre_padding"]) == [0, 0]
        and list(derived["post_padding"]) == [0, 0]
        and list(derived["dilation"]) == [1, 1]
        and not bool(derived["flip_filter"])
        and _is_contiguous(loss)
        and _is_contiguous(image)
        and _is_contiguous(weight)
    )


def _standard_1x1_wgrad_plan(node: Mapping[str, Any]) -> NodePlan:
    operation = str(node["operation"])
    derived = _require_object(node["derived"], "node.derived")
    loss = _require_object(derived["result"], "node.derived.result")
    image = _require_object(derived["image"], "node.derived.image")
    weight = _require_object(derived["weight"], "node.derived.weight")

    num_splits = 8
    splits_per_batch = 1
    block_co = 32
    block_ci = 64
    partial = _workspace_tensor(
        "wgrad_1x1_partial",
        "float32",
        (num_splits, int(derived["out_channels"]), int(derived["in_per_group"])),
    )
    split_constants = {
        "HW": int(image["dimensions"][2]) * int(image["dimensions"][3]),
        "C_OUT": int(derived["out_channels"]),
        "CIN_PER_GROUP": int(derived["in_per_group"]),
        "COUT_PER_GROUP": int(derived["out_per_group"]),
        "IMAGE_STRIDE_N": int(image["strides"][0]),
        "IMAGE_STRIDE_C": int(image["strides"][1]),
        "LOSS_STRIDE_N": int(loss["strides"][0]),
        "LOSS_STRIDE_C": int(loss["strides"][1]),
        "SPLITS_PER_N": splits_per_batch,
        "BLOCK_CO": block_co,
        "BLOCK_CI": block_ci,
        "BLOCK_M": 128,
    }
    split_grid = GridSpec(
        "fixed",
        (
            _ceil_div(int(derived["out_per_group"]), block_co)
            * _ceil_div(int(derived["in_per_group"]), block_ci),
            num_splits,
            int(derived["groups"]),
        ),
    )
    split_stage = _make_stage(
        operation=operation,
        stage_name=f"{operation}_1x1_split",
        function_name="conv_wgrad_2d_1x1_split_kernel",
        pointer_arguments=(
            _tensor_pointer(node, "image_ptr", "x"),
            _tensor_pointer(node, "loss_ptr", "dy"),
            _workspace_pointer("partial_ptr", partial.name),
        ),
        constants=split_constants,
        tuning=UNTUNED_TUNING,
        tuning_key_value=int(derived["n_outputs"]),
        grid_spec=split_grid,
        num_warps=8,
        num_stages=1,
    )

    reduce_block_co = 8
    reduce_block_ci = 32
    reduce_constants = {
        "C_OUT": int(derived["out_channels"]),
        "CIN_PER_GROUP": int(derived["in_per_group"]),
        "COUT_PER_GROUP": int(derived["out_per_group"]),
        "OUT_STRIDE_O": int(weight["strides"][0]),
        "OUT_STRIDE_I": int(weight["strides"][1]),
        "NUM_SPLITS": num_splits,
        "BLOCK_CO": reduce_block_co,
        "BLOCK_CI": reduce_block_ci,
    }
    reduce_grid = GridSpec(
        "fixed",
        (
            _ceil_div(int(derived["out_per_group"]), reduce_block_co)
            * _ceil_div(int(derived["in_per_group"]), reduce_block_ci),
            int(derived["groups"]),
            1,
        ),
    )
    reduce_stage = _make_stage(
        operation=operation,
        stage_name=f"{operation}_1x1_reduce",
        function_name="conv_wgrad_2d_1x1_reduce_kernel",
        pointer_arguments=(
            _workspace_pointer("partial_ptr", partial.name),
            _tensor_pointer(node, "out_ptr", "dw"),
        ),
        constants=reduce_constants,
        tuning=UNTUNED_TUNING,
        tuning_key_value=int(derived["n_outputs"]),
        grid_spec=reduce_grid,
        dependencies=(split_stage.stage_name,),
        num_warps=4,
        num_stages=1,
    )
    plan = NodePlan(
        operation,
        (split_stage, reduce_stage),
        (partial,),
        _aligned_size(partial.size),
    )
    plan.validate_dependencies()
    return plan


def _can_use_1x1_dgrad(node: Mapping[str, Any]) -> bool:
    if node["operation"] != "convolution_dgrad":
        return False
    derived = _require_object(node["derived"], "node.derived")
    image = _require_object(derived["image"], "node.derived.image")
    weight = _require_object(derived["weight"], "node.derived.weight")
    result = _require_object(derived["result"], "node.derived.result")
    return (
        int(derived["spatial_rank"]) == 2
        and str(derived["data_type"]) in ("float32", "float16")
        and list(derived["stride"]) == [1, 1]
        and list(derived["pre_padding"]) == [0, 0]
        and list(derived["post_padding"]) == [0, 0]
        and list(derived["dilation"]) == [1, 1]
        and list(weight["dimensions"][-2:]) == [1, 1]
        and _is_contiguous(image)
        and _is_contiguous(weight)
        and _is_contiguous(result)
    )


def _one_by_one_dgrad_plan(node: Mapping[str, Any]) -> NodePlan:
    derived = _require_object(node["derived"], "node.derived")
    image = _require_object(derived["image"], "node.derived.image")
    base = _convolution_stage(node)
    constants = {
        "HW": int(image["dimensions"][2]) * int(image["dimensions"][3]),
        "C_IN": int(derived["in_channels"]),
        "C_OUT": int(derived["out_channels"]),
        "CIN_PER_GROUP": int(derived["in_per_group"]),
        "COUT_PER_GROUP": int(derived["out_per_group"]),
        "GROUPS": int(derived["groups"]),
        "BLOCK_M": 32,
        "BLOCK_CI": 64,
        "BLOCK_CO": 32,
    }
    grid_spec = GridSpec(
        "fixed",
        (
            _ceil_div(int(constants["HW"]), int(constants["BLOCK_M"]))
            * _ceil_div(
                int(constants["CIN_PER_GROUP"]),
                int(constants["BLOCK_CI"]),
            ),
            int(derived["batch"]) * int(derived["groups"]),
            1,
        ),
    )
    stage = replace(
        base,
        function_name="conv_dgrad_2d_1x1_kernel",
        constants=constants,
        default_grid=grid_spec.evaluate(constants),
        tuning=UNTUNED_TUNING,
        default_num_warps=8,
        default_num_stages=1,
        grid_spec=grid_spec,
    )
    if stage.function_name not in REGISTRY_FUNCTIONS[stage.operation]:
        raise ValueError("1x1 DGrad kernel is absent from the registry")
    stage.validate_launch()
    return NodePlan(str(node["operation"]), (stage,))


def _is_qualified_stride2_dgrad(node: Mapping[str, Any]) -> bool:
    if node["operation"] != "convolution_dgrad":
        return False
    derived = _require_object(node["derived"], "node.derived")
    return (
        int(derived["spatial_rank"]) == 2
        and list(derived["stride"]) == [2, 2]
        and list(derived["pre_padding"]) == [1, 1]
        and list(derived["post_padding"]) == [1, 1]
        and list(derived["dilation"]) == [1, 1]
        and not bool(derived["flip_filter"])
        and list(derived["weight"]["dimensions"][-2:]) == [3, 3]
    )


def _can_use_scatter_dgrad(node: Mapping[str, Any]) -> bool:
    if not _is_qualified_stride2_dgrad(node):
        return False
    derived = _require_object(node["derived"], "node.derived")
    return (
        str(derived["data_type"]) == "float32"
        and int(derived["in_per_group"]) <= 4
        and _is_contiguous(derived["image"])
    )


def _can_use_stride2_dgrad(node: Mapping[str, Any]) -> bool:
    if not _is_qualified_stride2_dgrad(node):
        return False
    derived = _require_object(node["derived"], "node.derived")
    return (
        str(derived["data_type"]) in ("float32", "float16")
        and int(derived["in_per_group"]) >= 16
    )


def _can_use_packed_stride2_dgrad(node: Mapping[str, Any]) -> bool:
    if not _is_qualified_stride2_dgrad(node):
        return False
    derived = _require_object(node["derived"], "node.derived")
    image = _require_object(derived["image"], "node.derived.image")
    weight = _require_object(derived["weight"], "node.derived.weight")
    result = _require_object(derived["result"], "node.derived.result")
    return (
        str(derived["data_type"]) in ("float32", "float16")
        and int(derived["groups"]) == 1
        and list(weight["dimensions"][2:]) == [3, 3]
        and int(derived["in_per_group"]) >= 64
        and int(derived["out_per_group"]) >= 128
        and _is_contiguous(image)
        and _is_contiguous(weight)
        and _is_contiguous(result)
    )


def _packed_stride2_dgrad_plan(node: Mapping[str, Any]) -> NodePlan:
    derived = _require_object(node["derived"], "node.derived")
    operation = str(node["operation"])
    image = _require_object(derived["image"], "node.derived.image")
    weight = _require_object(derived["weight"], "node.derived.weight")
    result = _require_object(derived["result"], "node.derived.result")
    source_type = str(derived["data_type"])
    compute_type = "float16" if source_type == "float32" else source_type
    cursor = 0
    workspaces: list[WorkspaceTensor] = []
    stages: list[KernelStagePlan] = []
    dependency: tuple[str, ...] = ()

    loss_workspace: WorkspaceTensor | None = None
    if source_type != compute_type:
        loss_workspace = _workspace_tensor(
            "dgrad_stride2_loss_fp16",
            compute_type,
            result["dimensions"],
            offset=cursor,
        )
        workspaces.append(loss_workspace)
        cursor += _aligned_size(loss_workspace.size)
        total = _checked_product(
            list(result["dimensions"]),
            "packed P5 DGrad loss elements",
        )
        cast_stage = _make_stage(
            operation=operation,
            stage_name=f"{operation}_cast_loss",
            function_name="conv_dgrad_cast_contiguous_kernel",
            pointer_arguments=(
                _tensor_pointer(node, "input_ptr", "dy"),
                _workspace_pointer(
                    "output_ptr",
                    loss_workspace.name,
                    compute_type,
                ),
            ),
            constants={"TOTAL": total, "BLOCK_SIZE": 256},
            tuning=UNTUNED_TUNING,
            tuning_key_value=total,
            grid_spec=GridSpec("linear", (total,)),
            num_warps=4,
            num_stages=1,
        )
        stages.append(cast_stage)
        dependency = (cast_stage.stage_name,)

    packed_workspace = _workspace_tensor(
        "dgrad_stride2_weight_packed",
        compute_type,
        (
            3,
            3,
            int(derived["out_per_group"]),
            int(derived["in_per_group"]),
        ),
        offset=cursor,
    )
    workspaces.append(packed_workspace)
    cursor += _aligned_size(packed_workspace.size)
    pair_count = int(derived["out_per_group"]) * int(derived["in_per_group"])
    pack_stage = _make_stage(
        operation=operation,
        stage_name=f"{operation}_pack_weight",
        function_name="conv_dgrad_pack_weight_3x3_kernel",
        pointer_arguments=(
            _tensor_pointer(node, "weight_ptr", "w"),
            _workspace_pointer(
                "packed_ptr",
                packed_workspace.name,
                compute_type,
            ),
        ),
        constants={
            "C_OUT": int(derived["out_per_group"]),
            "C_IN": int(derived["in_per_group"]),
            "BLOCK_SIZE": 256,
        },
        tuning=UNTUNED_TUNING,
        tuning_key_value=pair_count,
        grid_spec=GridSpec("linear", (pair_count,)),
        dependencies=dependency,
        num_warps=4,
        num_stages=1,
    )
    stages.append(pack_stage)
    dependency = (pack_stage.stage_name,)

    loss_strides = (
        loss_workspace.strides
        if loss_workspace is not None
        else tuple(int(value) for value in result["strides"])
    )
    block_m = 64
    block_ci = 64
    block_co = 128
    batch = int(derived["batch"])
    input_height = int(image["dimensions"][2])
    input_width = int(image["dimensions"][3])
    loss_height = int(result["dimensions"][2])
    loss_width = int(result["dimensions"][3])
    in_per_group = int(derived["in_per_group"])
    out_per_group = int(derived["out_per_group"])
    use_p5_splitk = (
        batch == 1
        and input_height == 40
        and input_width == 40
        and loss_height == 20
        and loss_width == 20
    )

    loss_pointer = (
        _workspace_pointer(
            "loss_ptr",
            loss_workspace.name,
            compute_type,
        )
        if loss_workspace is not None
        else _tensor_pointer(node, "loss_ptr", "dy")
    )
    if source_type == "float32" and use_p5_splitk:
        output_elements = int(derived["n_outputs"])
        zero_stage = _make_stage(
            operation=operation,
            stage_name=f"{operation}_zero_p5",
            function_name="conv_dgrad_zero_kernel",
            pointer_arguments=(_tensor_pointer(node, "dx_ptr", "dx"),),
            constants={
                "N_ELEMENTS": output_elements,
                "BLOCK_SIZE": 256,
            },
            tuning=UNTUNED_TUNING,
            tuning_key_value=output_elements,
            grid_spec=GridSpec("linear", (output_elements,)),
            dependencies=dependency,
            num_warps=4,
            num_stages=1,
        )
        stages.append(zero_stage)
        dependency = (zero_stage.stage_name,)

        split_block_m = 64
        split_block_ci = 128
        split_block_co = 16
        split_group_k = 8
        rows = loss_height * loss_width
        split_groups = _ceil_div(
            _ceil_div(out_per_group, split_block_co),
            split_group_k,
        )
        for parity_h in (0, 1):
            constants = {
                "M": rows,
                "XW": input_width,
                "LOSS_H": loss_height,
                "LOSS_W": loss_width,
                "CIN_PER_GROUP": in_per_group,
                "COUT_PER_GROUP": out_per_group,
                "loss_stride_c": int(loss_strides[1]),
                "loss_stride_h": int(loss_strides[2]),
                "loss_stride_w": int(loss_strides[3]),
                "out_stride_c": int(image["strides"][1]),
                "out_stride_h": int(image["strides"][2]),
                "out_stride_w": int(image["strides"][3]),
                "PH": parity_h,
                "GROUP_K": split_group_k,
                "BLOCK_M": split_block_m,
                "BLOCK_CI": split_block_ci,
                "BLOCK_CO": split_block_co,
            }
            split_stage = _make_stage(
                operation=operation,
                stage_name=(f"{operation}_p5_splitk_h{parity_h}"),
                function_name="conv_dgrad_2d_p5_splitk_kernel",
                pointer_arguments=(
                    loss_pointer,
                    _workspace_pointer(
                        "weight_ptr",
                        packed_workspace.name,
                        compute_type,
                    ),
                    _tensor_pointer(node, "out_ptr", "dx"),
                ),
                constants=constants,
                tuning=UNTUNED_TUNING,
                tuning_key_value=output_elements,
                grid_spec=GridSpec(
                    "fixed",
                    (
                        _ceil_div(rows, split_block_m)
                        * _ceil_div(in_per_group, split_block_ci)
                        * split_groups,
                        1,
                        1,
                    ),
                ),
                dependencies=dependency,
                num_warps=16,
                num_stages=1,
            )
            stages.append(split_stage)
            dependency = (split_stage.stage_name,)

        plan = NodePlan(
            operation,
            tuple(stages),
            tuple(workspaces),
            cursor,
        )
        plan.validate_dependencies()
        return plan

    for parity_h, parity_w in ((0, 0), (0, 1), (1, 0), (1, 1)):
        parity_h_count = (input_height + 1 - parity_h) // 2
        parity_w_count = (input_width + 1 - parity_w) // 2
        rows = batch * parity_h_count * parity_w_count
        constants = {
            "M": rows,
            "XH": input_height,
            "XW": input_width,
            "LOSS_H": loss_height,
            "LOSS_W": loss_width,
            "CIN_PER_GROUP": in_per_group,
            "COUT_PER_GROUP": out_per_group,
            "loss_stride_n": int(loss_strides[0]),
            "loss_stride_c": int(loss_strides[1]),
            "loss_stride_h": int(loss_strides[2]),
            "loss_stride_w": int(loss_strides[3]),
            "out_stride_n": int(image["strides"][0]),
            "out_stride_c": int(image["strides"][1]),
            "out_stride_h": int(image["strides"][2]),
            "out_stride_w": int(image["strides"][3]),
            "PARITY_H_COUNT": parity_h_count,
            "PARITY_W_COUNT": parity_w_count,
            "PH": parity_h,
            "PW": parity_w,
            "KH_COUNT": 1 if parity_h == 0 else 2,
            "KW_COUNT": 1 if parity_w == 0 else 2,
            "FILTER_REVERSE": False,
            "BLOCK_M": block_m,
            "BLOCK_CI": block_ci,
            "BLOCK_CO": block_co,
        }
        compute_stage = _make_stage(
            operation=operation,
            stage_name=(f"{operation}_packed_parity_{parity_h}{parity_w}"),
            function_name="conv_dgrad_2d_packed_parity_kernel",
            pointer_arguments=(
                loss_pointer,
                _workspace_pointer(
                    "weight_ptr",
                    packed_workspace.name,
                    compute_type,
                ),
                _tensor_pointer(node, "out_ptr", "dx"),
            ),
            constants=constants,
            tuning=UNTUNED_TUNING,
            tuning_key_value=int(derived["n_outputs"]),
            grid_spec=GridSpec(
                "fixed",
                (
                    _ceil_div(rows, block_m) * _ceil_div(in_per_group, block_ci),
                    1,
                    1,
                ),
            ),
            dependencies=dependency,
            num_warps=8,
            num_stages=2,
        )
        stages.append(compute_stage)
        dependency = (compute_stage.stage_name,)

    plan = NodePlan(
        operation,
        tuple(stages),
        tuple(workspaces),
        cursor,
    )
    plan.validate_dependencies()
    return plan


def _scatter_dgrad_plan(node: Mapping[str, Any]) -> NodePlan:
    derived = _require_object(node["derived"], "node.derived")
    operation = str(node["operation"])
    zero = _make_stage(
        operation=operation,
        stage_name=f"{operation}_zero",
        function_name="conv_dgrad_zero_kernel",
        pointer_arguments=(_tensor_pointer(node, "dx_ptr", "dx"),),
        constants={
            "N_ELEMENTS": int(derived["n_outputs"]),
            "BLOCK_SIZE": 256,
        },
        tuning=UNTUNED_TUNING,
        tuning_key_value=int(derived["n_outputs"]),
        grid_spec=GridSpec("linear", (int(derived["n_outputs"]),)),
        num_warps=4,
        num_stages=1,
    )

    base = _convolution_stage(node)
    constants = dict(base.constants)
    constants.update(
        {
            "BLOCK_M": 128,
            "BLOCK_CI": 4,
            "BLOCK_K": 16,
            "BLOCK_N": 32,
            "GROUP_M": 1,
        }
    )
    weight = _require_object(derived["weight"], "node.derived.weight")
    grid_spec = GridSpec(
        "conv_dgrad_scatter",
        (
            int(derived["batch"])
            * int(derived["result"]["dimensions"][2])
            * int(derived["result"]["dimensions"][3]),
            int(derived["in_per_group"])
            * int(weight["dimensions"][2])
            * int(weight["dimensions"][3]),
            int(derived["groups"]),
        ),
    )
    scatter = replace(
        base,
        stage_name=f"{operation}_scatter",
        function_name="conv_dgrad_2d_scatter_kernel",
        constants=constants,
        default_grid=grid_spec.evaluate(constants),
        tuning=UNTUNED_TUNING,
        default_num_warps=8,
        default_num_stages=1,
        grid_spec=grid_spec,
        dependencies=(zero.stage_name,),
    )
    if scatter.function_name not in REGISTRY_FUNCTIONS[operation]:
        raise ValueError("scatter DGrad kernel is absent from the registry")
    scatter.validate_launch()
    plan = NodePlan(operation, (zero, scatter))
    plan.validate_dependencies()
    return plan


def _stride2_dgrad_stage(
    node: Mapping[str, Any],
    parity_h: int,
    parity_w: int,
    dependency: tuple[str, ...],
) -> KernelStagePlan:
    base = _convolution_stage(node)
    derived = _require_object(node["derived"], "node.derived")
    image = _require_object(derived["image"], "node.derived.image")
    constants = dict(base.constants)
    constants.update(
        {
            "PARITY_H": parity_h,
            "PARITY_W": parity_w,
            "BLOCK_M": 64,
            "BLOCK_CI": 64,
            "BLOCK_K": (64 if str(derived["data_type"]) == "float16" else 32),
            "GROUP_M": 4,
        }
    )
    grid_spec = GridSpec(
        "conv_dgrad_stride2_parity",
        (
            int(derived["batch"]),
            int(image["dimensions"][2]),
            int(image["dimensions"][3]),
            int(derived["in_per_group"]),
            int(derived["groups"]),
        ),
    )
    stage_name = f"{node['operation']}_parity_{parity_h}{parity_w}"
    stage = replace(
        base,
        stage_name=stage_name,
        function_name="conv_dgrad_2d_stride2_kernel",
        constants=constants,
        default_grid=grid_spec.evaluate(constants),
        tuning=UNTUNED_TUNING,
        default_num_warps=4,
        default_num_stages=2,
        grid_spec=grid_spec,
        dependencies=dependency,
    )
    if stage.function_name not in REGISTRY_FUNCTIONS[stage.operation]:
        raise ValueError("stride-2 DGrad kernel is absent from the registry")
    stage.validate_launch()
    return stage


def _convolution_plan(node: Mapping[str, Any]) -> NodePlan:
    operation = str(node["operation"])
    # Optimized packing routes use IEEE operands. Explicit TF32 takes the
    # generic stride-aware route, rounding both operands before each dot.
    if node["derived"]["input_precision"] == 2:
        return NodePlan(operation, (_convolution_stage(node),))
    if _can_use_fp32_ml_stem_fprop(node):
        return _fp32_ml_stem_fprop_plan(node)
    if _can_use_1x1_fprop(node):
        return _one_by_one_fprop_plan(node)
    if _can_use_3x3_im2col_fprop(node):
        return _3x3_im2col_fprop_plan(node)
    if _can_use_low_channel_3x3_fprop(node):
        return _low_channel_3x3_fprop_plan(node)
    if _can_use_p5_wgrad(node):
        return _p5_wgrad_plan(node)
    if _can_use_col_split_wgrad(node):
        return _col_split_wgrad_plan(node)
    if _can_use_batched_3x3_wgrad(node):
        return _batched_3x3_wgrad_plan(node)
    if _can_use_standard_1d_3tap_wgrad(node):
        return _standard_1d_3tap_wgrad_plan(node)
    if _can_use_standard_1x1_wgrad(node):
        return _standard_1x1_wgrad_plan(node)
    if _can_use_1x1_dgrad(node):
        return _one_by_one_dgrad_plan(node)
    if _can_use_scatter_dgrad(node):
        return _scatter_dgrad_plan(node)
    if _can_use_packed_stride2_dgrad(node):
        return _packed_stride2_dgrad_plan(node)
    if _can_use_stride2_dgrad(node):
        stages: list[KernelStagePlan] = []
        dependency: tuple[str, ...] = ()
        for parity_h, parity_w in ((0, 0), (0, 1), (1, 0), (1, 1)):
            stage = _stride2_dgrad_stage(
                node,
                parity_h,
                parity_w,
                dependency,
            )
            stages.append(stage)
            dependency = (stage.stage_name,)
        plan = NodePlan(operation, tuple(stages))
        plan.validate_dependencies()
        return plan

    # All shapes outside the qualified stride-2 route retain the generic
    # correctness kernel.
    return NodePlan(operation, (_convolution_stage(node),))
