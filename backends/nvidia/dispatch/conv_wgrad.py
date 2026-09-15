"""Weight-gradient convolution validation and kernel configurations."""

from __future__ import annotations

from typing import Any

from .common import (
    FLOAT_DATA_TYPES,
    TRITON_POINTER_TYPES,
    _ceil_div,
    _is_row_major_contiguous,
    _require_integer,
    _require_integer_list,
)


def _convolution_wgrad_pipeline_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | bool | str],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    """Configure one internal stage of a split/reduce WGrad pipeline."""
    pipeline_stage = parameters.get("_wgrad_pipeline_stage")
    if pipeline_stage not in {"split", "reduce"}:
        raise ValueError("unknown convolution WGrad pipeline stage")
    spatial_rank = _require_integer(
        parameters, "spatial_rank", minimum=1, maximum=3
    )
    if spatial_rank != 2:
        raise ValueError("the WGrad split/reduce pipeline requires 2D tensors")
    groups = _require_integer(parameters, "groups")
    num_splits = _require_integer(parameters, "_wgrad_num_splits")
    kernel_h = _require_integer(parameters, "_wgrad_kernel_h")
    kernel_w = _require_integer(parameters, "_wgrad_kernel_w")

    if pipeline_stage == "split":
        if len(tensors) != 3:
            raise ValueError("the WGrad split stage requires three tensors")
        image, loss, partial = tensors
        if (
            image["data_type"] != loss["data_type"]
            or image["data_type"] not in FLOAT_DATA_TYPES
            or partial["data_type"] != "float32"
            or len(image["dimensions"]) != 4
            or len(loss["dimensions"]) != 4
        ):
            raise ValueError(
                "the WGrad split stage tensor metadata is invalid"
            )
        n, c_in, image_h, image_w = image["dimensions"]
        loss_n, c_out, loss_h, loss_w = loss["dimensions"]
        if loss_n != n or c_in % groups != 0 or c_out % groups != 0:
            raise ValueError("the WGrad split stage channels are inconsistent")
        cin_per_group = c_in // groups
        cout_per_group = c_out // groups
        cik = cin_per_group * kernel_h * kernel_w
        if partial["dimensions"] != [num_splits, c_out, cik]:
            raise ValueError("the WGrad partial workspace shape is invalid")
        stride = _require_integer_list(parameters, "stride", 2, minimum=1)
        pre_padding = _require_integer_list(
            parameters, "pre_padding", 2, minimum=0
        )
        dilation = _require_integer_list(parameters, "dilation", 2, minimum=1)
        convolution_mode = _require_integer(
            parameters, "convolution_mode", minimum=0, maximum=1
        )
        use_col_split = (
            parameters.get("_wgrad_pipeline_algorithm") == "stem_col"
        )
        block_co = 16
        block_ci = 32
        block_m = (
            128
            if use_col_split
            else 32 if image["data_type"] == "float32" else 64
        )
        constants: dict[str, int | bool | str] = {
            "M": n * loss_h * loss_w,
            "IMAGE_H": image_h,
            "IMAGE_W": image_w,
            "LOSS_H": loss_h,
            "LOSS_W": loss_w,
            "C_OUT": c_out,
            "CIN_PER_GROUP": cin_per_group,
            "COUT_PER_GROUP": cout_per_group,
            "image_stride_n": image["strides"][0],
            "image_stride_c": image["strides"][1],
            "image_stride_h": image["strides"][2],
            "image_stride_w": image["strides"][3],
            "loss_stride_n": loss["strides"][0],
            "loss_stride_c": loss["strides"][1],
            "loss_stride_h": loss["strides"][2],
            "loss_stride_w": loss["strides"][3],
            "STRIDE_H": stride[0],
            "STRIDE_W": stride[1],
            "PAD_H": pre_padding[0],
            "PAD_W": pre_padding[1],
            "DIL_H": dilation[0],
            "DIL_W": dilation[1],
            "KH": kernel_h,
            "KW": kernel_w,
            "FILTER_REVERSE": convolution_mode == 1,
            "NUM_SPLITS": num_splits,
            "BLOCK_CO": block_co,
            "BLOCK_CI": block_ci,
            "BLOCK_M": block_m,
        }
        if use_col_split:
            constants["BLOCK_N"] = constants.pop("BLOCK_CI")
            return (
                "_conv_wgrad2d_col_split_kernel",
                {
                    "image_ptr": TRITON_POINTER_TYPES[image["data_type"]],
                    "loss_ptr": TRITON_POINTER_TYPES[loss["data_type"]],
                    "partial_ptr": "*fp32",
                },
                constants,
                (
                    _ceil_div(cout_per_group, block_co)
                    * _ceil_div(cik, block_ci),
                    num_splits * groups,
                    1,
                ),
                [("tensor", None), ("tensor", None), ("tensor", None)],
            )
        return (
            "_conv_wgrad2d_3tap_split_kernel",
            {
                "image_ptr": TRITON_POINTER_TYPES[image["data_type"]],
                "loss_ptr": TRITON_POINTER_TYPES[loss["data_type"]],
                "partial_ptr": "*fp32",
            },
            constants,
            (
                _ceil_div(cout_per_group, block_co)
                * _ceil_div(cin_per_group, block_ci),
                kernel_h,
                num_splits * groups,
            ),
            [("tensor", None), ("tensor", None), ("tensor", None)],
        )

    if len(tensors) != 2:
        raise ValueError("the WGrad reduce stage requires two tensors")
    partial, output = tensors
    if (
        partial["data_type"] != "float32"
        or output["data_type"] not in FLOAT_DATA_TYPES
        or len(output["dimensions"]) != 4
    ):
        raise ValueError("the WGrad reduce stage tensor metadata is invalid")
    c_out, cin_per_group, output_kh, output_kw = output["dimensions"]
    if output_kh != kernel_h or output_kw != kernel_w or c_out % groups != 0:
        raise ValueError("the WGrad reduce stage filter shape is invalid")
    cout_per_group = c_out // groups
    cik = cin_per_group * kernel_h * kernel_w
    if partial["dimensions"] != [num_splits, c_out, cik]:
        raise ValueError("the WGrad partial workspace shape is invalid")
    block_co = 16
    block_n = 32
    constants = {
        "C_OUT": c_out,
        "CIN_PER_GROUP": cin_per_group,
        "COUT_PER_GROUP": cout_per_group,
        "out_stride_o": output["strides"][0],
        "out_stride_i": output["strides"][1],
        "out_stride_h": output["strides"][2],
        "out_stride_w": output["strides"][3],
        "KH": kernel_h,
        "KW": kernel_w,
        "NUM_SPLITS": num_splits,
        "BLOCK_CO": block_co,
        "BLOCK_N": block_n,
    }
    return (
        "_conv_wgrad2d_col_reduce_kernel",
        {
            "partial_ptr": "*fp32",
            "out_ptr": TRITON_POINTER_TYPES[output["data_type"]],
        },
        constants,
        (
            _ceil_div(cout_per_group, block_co) * _ceil_div(cik, block_n),
            groups,
            1,
        ),
        [("tensor", None), ("tensor", None)],
    )


def _convolution_wgrad_p5_pipeline_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | bool | str],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    """Configure one stage of the NVIDIA YOLO P5 WGrad pipeline."""
    pipeline_stage = parameters.get("_wgrad_pipeline_stage")
    if pipeline_stage not in {"pack", "matmul"}:
        raise ValueError("unknown P5 WGrad pipeline stage")
    if _require_integer(parameters, "groups") != 1:
        raise ValueError("the P5 WGrad pipeline requires one group")

    if pipeline_stage == "pack":
        if len(tensors) != 2:
            raise ValueError("the P5 WGrad pack stage requires two tensors")
        image, packed = tensors
        valid_pack_types = (
            image["data_type"] == packed["data_type"]
            and image["data_type"] in FLOAT_DATA_TYPES
        )
        if (
            not valid_pack_types
            or len(image["dimensions"]) != 4
            or image["dimensions"][0] != 1
            or image["dimensions"][2:] != [40, 40]
            or len(packed["dimensions"]) != 2
            or not _is_row_major_contiguous(packed)
        ):
            raise ValueError("the P5 WGrad pack metadata is invalid")
        c_in = image["dimensions"][1]
        cik = c_in * 9
        if packed["dimensions"] != [400, cik]:
            raise ValueError("the P5 WGrad packed image shape is invalid")
        block_m = 16
        block_n = 16
        block_k = 32
        group_m = 8
        constants: dict[str, int | bool | str] = {
            "CIN_PER_GROUP": c_in,
            "image_stride_c": image["strides"][1],
            "image_stride_h": image["strides"][2],
            "image_stride_w": image["strides"][3],
            "M": 400,
            "N": cik,
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "GROUP_M": group_m,
        }
        return (
            "_conv_wgrad2d_p5_pack_image_kernel",
            {
                "image_ptr": TRITON_POINTER_TYPES[image["data_type"]],
                "packed_ptr": TRITON_POINTER_TYPES[packed["data_type"]],
            },
            constants,
            (_ceil_div(400, block_m), _ceil_div(cik, block_n), 1),
            [("tensor", None), ("tensor", None)],
        )

    if len(tensors) != 3:
        raise ValueError("the P5 WGrad matmul stage requires three tensors")
    loss, packed, output = tensors
    valid_matmul_types = (
        loss["data_type"] == packed["data_type"]
        and loss["data_type"] == output["data_type"]
        and loss["data_type"] in FLOAT_DATA_TYPES
    )
    if (
        not valid_matmul_types
        or len(loss["dimensions"]) != 4
        or loss["dimensions"][0] != 1
        or loss["dimensions"][2:] != [20, 20]
        or len(packed["dimensions"]) != 2
        or len(output["dimensions"]) != 4
        or output["dimensions"][2:] != [3, 3]
        or not _is_row_major_contiguous(packed)
        or not _is_row_major_contiguous(output)
    ):
        raise ValueError("the P5 WGrad matmul metadata is invalid")
    c_out = loss["dimensions"][1]
    output_c_out, c_in, _, _ = output["dimensions"]
    cik = c_in * 9
    if output_c_out != c_out or packed["dimensions"] != [400, cik]:
        raise ValueError("the P5 WGrad matmul shape is inconsistent")
    block_m = 16
    block_n = 16
    block_k = 32
    group_m = 8
    constants = {
        "M": c_out,
        "N": cik,
        "K": 400,
        "DTYPE_ID": {
            "float16": 0,
            "bfloat16": 1,
            "float32": 2,
        }[loss["data_type"]],
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "GROUP_M": group_m,
    }
    return (
        "_conv_wgrad2d_p5_mm_kernel",
        {
            "loss_ptr": TRITON_POINTER_TYPES[loss["data_type"]],
            "packed_ptr": TRITON_POINTER_TYPES[packed["data_type"]],
            "out_ptr": TRITON_POINTER_TYPES[output["data_type"]],
        },
        constants,
        (_ceil_div(c_out, block_m) * _ceil_div(cik, block_n), 1, 1),
        [("tensor", None), ("tensor", None), ("tensor", None)],
    )


def _convolution_wgrad_1x1_pipeline_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | bool | str],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    """Configure one internal stage of the NVIDIA 1x1 WGrad pipeline."""
    pipeline_stage = parameters.get("_wgrad_pipeline_stage")
    if pipeline_stage not in {"direct", "split", "reduce"}:
        raise ValueError("unknown 1x1 WGrad pipeline stage")
    groups = _require_integer(parameters, "groups")
    if pipeline_stage == "direct":
        if len(tensors) != 3:
            raise ValueError(
                "the 1x1 WGrad direct stage requires three tensors"
            )
        image, loss, output = tensors
        if (
            image["data_type"] != loss["data_type"]
            or image["data_type"] != output["data_type"]
            or image["data_type"] not in FLOAT_DATA_TYPES
            or len(image["dimensions"]) != 4
            or len(loss["dimensions"]) != 4
            or len(output["dimensions"]) != 4
        ):
            raise ValueError("the 1x1 WGrad direct tensor metadata is invalid")
        n, c_in, image_h, image_w = image["dimensions"]
        loss_n, c_out, loss_h, loss_w = loss["dimensions"]
        if (
            loss_n != n
            or loss_h != image_h
            or loss_w != image_w
            or output["dimensions"] != [c_out, c_in // groups, 1, 1]
            or c_in % groups != 0
            or c_out % groups != 0
        ):
            raise ValueError("the 1x1 WGrad direct shape is inconsistent")
        cin_per_group = c_in // groups
        cout_per_group = c_out // groups
        block_co = 16
        block_ci = 32
        block_m = 256
        constants: dict[str, int | bool | str] = {
            "BATCH_N": n,
            "HW": image_h * image_w,
            "CIN_PER_GROUP": cin_per_group,
            "COUT_PER_GROUP": cout_per_group,
            "image_stride_n": image["strides"][0],
            "image_stride_c": image["strides"][1],
            "loss_stride_n": loss["strides"][0],
            "loss_stride_c": loss["strides"][1],
            "out_stride_o": output["strides"][0],
            "out_stride_i": output["strides"][1],
            "BLOCK_CO": block_co,
            "BLOCK_CI": block_ci,
            "BLOCK_M": block_m,
        }
        pointer_type = TRITON_POINTER_TYPES[image["data_type"]]
        return (
            "_conv_wgrad2d_1x1_direct_nodiv_kernel",
            {
                "image_ptr": pointer_type,
                "loss_ptr": pointer_type,
                "out_ptr": pointer_type,
            },
            constants,
            (
                _ceil_div(cout_per_group, block_co)
                * _ceil_div(cin_per_group, block_ci),
                groups,
                1,
            ),
            [("tensor", None), ("tensor", None), ("tensor", None)],
        )

    num_splits = _require_integer(parameters, "_wgrad_num_splits")

    if pipeline_stage == "split":
        if len(tensors) != 3:
            raise ValueError(
                "the 1x1 WGrad split stage requires three tensors"
            )
        image, loss, partial = tensors
        if (
            image["data_type"] != loss["data_type"]
            or image["data_type"] not in FLOAT_DATA_TYPES
            or partial["data_type"] not in {"float16", "float32"}
            or len(image["dimensions"]) != 4
            or len(loss["dimensions"]) != 4
        ):
            raise ValueError("the 1x1 WGrad split tensor metadata is invalid")
        n, c_in, image_h, image_w = image["dimensions"]
        loss_n, c_out, loss_h, loss_w = loss["dimensions"]
        if (
            loss_n != n
            or loss_h != image_h
            or loss_w != image_w
            or c_in % groups != 0
            or c_out % groups != 0
            or num_splits % n != 0
        ):
            raise ValueError("the 1x1 WGrad split shape is inconsistent")
        cin_per_group = c_in // groups
        cout_per_group = c_out // groups
        if partial["dimensions"] != [num_splits, c_out, cin_per_group]:
            raise ValueError(
                "the 1x1 WGrad partial workspace shape is invalid"
            )
        block_co = 16
        block_ci = 64
        block_m = 256
        constants: dict[str, int | bool | str] = {
            "HW": image_h * image_w,
            "C_OUT": c_out,
            "CIN_PER_GROUP": cin_per_group,
            "COUT_PER_GROUP": cout_per_group,
            "image_stride_n": image["strides"][0],
            "image_stride_c": image["strides"][1],
            "loss_stride_n": loss["strides"][0],
            "loss_stride_c": loss["strides"][1],
            "SPLITS_PER_N": num_splits // n,
            "BLOCK_CO": block_co,
            "BLOCK_CI": block_ci,
            "BLOCK_M": block_m,
        }
        return (
            "_conv_wgrad2d_1x1_split_nodiv_kernel",
            {
                "image_ptr": TRITON_POINTER_TYPES[image["data_type"]],
                "loss_ptr": TRITON_POINTER_TYPES[loss["data_type"]],
                "partial_ptr": TRITON_POINTER_TYPES[partial["data_type"]],
            },
            constants,
            (
                _ceil_div(cout_per_group, block_co)
                * _ceil_div(cin_per_group, block_ci),
                num_splits,
                1,
            ),
            [("tensor", None), ("tensor", None), ("tensor", None)],
        )

    if len(tensors) != 2:
        raise ValueError("the 1x1 WGrad reduce stage requires two tensors")
    partial, output = tensors
    if (
        partial["data_type"] not in {"float16", "float32"}
        or output["data_type"] not in FLOAT_DATA_TYPES
        or len(output["dimensions"]) != 4
    ):
        raise ValueError("the 1x1 WGrad reduce tensor metadata is invalid")
    c_out, cin_per_group, kernel_h, kernel_w = output["dimensions"]
    if kernel_h != 1 or kernel_w != 1 or c_out % groups != 0:
        raise ValueError("the 1x1 WGrad reduce filter shape is invalid")
    cout_per_group = c_out // groups
    if partial["dimensions"] != [num_splits, c_out, cin_per_group]:
        raise ValueError("the 1x1 WGrad partial workspace shape is invalid")
    block_co = 8
    block_ci = 16 if output["data_type"] == "bfloat16" else 32
    constants = {
        "C_OUT": c_out,
        "CIN_PER_GROUP": cin_per_group,
        "COUT_PER_GROUP": cout_per_group,
        "out_stride_o": output["strides"][0],
        "out_stride_i": output["strides"][1],
        "NUM_SPLITS": num_splits,
        "BLOCK_CO": block_co,
        "BLOCK_CI": block_ci,
    }
    return (
        "_conv_wgrad2d_1x1_reduce_kernel",
        {
            "partial_ptr": TRITON_POINTER_TYPES[partial["data_type"]],
            "out_ptr": TRITON_POINTER_TYPES[output["data_type"]],
        },
        constants,
        (
            _ceil_div(cout_per_group, block_co)
            * _ceil_div(cin_per_group, block_ci),
            groups,
            1,
        ),
        [("tensor", None), ("tensor", None)],
    )


def _convolution_wgrad_stride2_pipeline_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | bool | str],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    """Configure the fixed-shape NVIDIA stride-2 WGrad pipeline."""
    pipeline_stage = parameters.get("_wgrad_pipeline_stage")
    if pipeline_stage not in {"split", "reduce"}:
        raise ValueError("unknown stride-2 WGrad pipeline stage")
    groups = _require_integer(parameters, "groups")
    num_splits = _require_integer(parameters, "_wgrad_num_splits")

    if pipeline_stage == "split":
        if len(tensors) != 3:
            raise ValueError(
                "the stride-2 WGrad split stage requires three tensors"
            )
        image, loss, partial = tensors
        if (
            image["data_type"] != loss["data_type"]
            or image["data_type"] != partial["data_type"]
            or image["data_type"] not in FLOAT_DATA_TYPES
            or len(image["dimensions"]) != 4
            or len(loss["dimensions"]) != 4
        ):
            raise ValueError("the stride-2 WGrad split metadata is invalid")
        n, c_in, image_h, image_w = image["dimensions"]
        loss_n, c_out, loss_h, loss_w = loss["dimensions"]
        if (
            loss_n != n
            or image_h != 56
            or image_w != 56
            or loss_h != 28
            or loss_w != 28
            or num_splits != n
            or c_in % groups != 0
            or c_out % groups != 0
        ):
            raise ValueError("the stride-2 WGrad split shape is inconsistent")
        cin_per_group = c_in // groups
        cout_per_group = c_out // groups
        if partial["dimensions"] != [num_splits, c_out, cin_per_group, 9]:
            raise ValueError("the stride-2 WGrad workspace shape is invalid")
        block_co = 16
        block_ci = 32
        block_hw = 128
        constants: dict[str, int | bool | str] = {
            "C_OUT": c_out,
            "CIN_PER_GROUP": cin_per_group,
            "COUT_PER_GROUP": cout_per_group,
            "image_stride_n": image["strides"][0],
            "image_stride_c": image["strides"][1],
            "image_stride_h": image["strides"][2],
            "image_stride_w": image["strides"][3],
            "loss_stride_n": loss["strides"][0],
            "loss_stride_c": loss["strides"][1],
            "loss_stride_h": loss["strides"][2],
            "loss_stride_w": loss["strides"][3],
            "BLOCK_CO": block_co,
            "BLOCK_CI": block_ci,
            "BLOCK_HW": block_hw,
        }
        pointer_type = TRITON_POINTER_TYPES[image["data_type"]]
        return (
            "_conv_wgrad2d_stride2_row4_split_kernel",
            {
                "image_ptr": pointer_type,
                "loss_ptr": pointer_type,
                "partial_ptr": pointer_type,
            },
            constants,
            (
                _ceil_div(cout_per_group, block_co)
                * _ceil_div(cin_per_group, block_ci),
                3,
                num_splits,
            ),
            [("tensor", None), ("tensor", None), ("tensor", None)],
        )

    if len(tensors) != 2:
        raise ValueError(
            "the stride-2 WGrad reduce stage requires two tensors"
        )
    partial, output = tensors
    if (
        partial["data_type"] != output["data_type"]
        or output["data_type"] not in FLOAT_DATA_TYPES
        or len(output["dimensions"]) != 4
    ):
        raise ValueError("the stride-2 WGrad reduce metadata is invalid")
    c_out, cin_per_group, kernel_h, kernel_w = output["dimensions"]
    if kernel_h != 3 or kernel_w != 3 or c_out % groups != 0:
        raise ValueError("the stride-2 WGrad reduce filter shape is invalid")
    cout_per_group = c_out // groups
    if partial["dimensions"] != [num_splits, c_out, cin_per_group, 9]:
        raise ValueError("the stride-2 WGrad workspace shape is invalid")
    block_co = 16
    block_ci = 32
    constants = {
        "C_OUT": c_out,
        "CIN_PER_GROUP": cin_per_group,
        "COUT_PER_GROUP": cout_per_group,
        "out_stride_o": output["strides"][0],
        "out_stride_i": output["strides"][1],
        "out_stride_h": output["strides"][2],
        "out_stride_w": output["strides"][3],
        "KH": kernel_h,
        "KW": kernel_w,
        "NUM_SPLITS": num_splits,
        "BLOCK_CO": block_co,
        "BLOCK_CI": block_ci,
    }
    pointer_type = TRITON_POINTER_TYPES[output["data_type"]]
    return (
        "_conv_wgrad2d_reduce_kernel",
        {"partial_ptr": pointer_type, "out_ptr": pointer_type},
        constants,
        (
            _ceil_div(cout_per_group, block_co)
            * _ceil_div(cin_per_group, block_ci),
            kernel_h * kernel_w,
            groups,
        ),
        [("tensor", None), ("tensor", None)],
    )


def _convolution_wgrad_batched_pipeline_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | bool | str],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    pipeline_stage = parameters.get("_wgrad_pipeline_stage")
    if pipeline_stage not in {"matmul", "reduce"}:
        raise ValueError("unknown batched WGrad pipeline stage")
    groups = _require_integer(parameters, "groups")
    if groups != 1:
        raise ValueError("the batched WGrad pipeline requires one group")
    num_splits = _require_integer(parameters, "_wgrad_num_splits")
    kernel_h = _require_integer(parameters, "_wgrad_kernel_h")
    kernel_w = _require_integer(parameters, "_wgrad_kernel_w")

    if pipeline_stage == "matmul":
        if len(tensors) != 3:
            raise ValueError("the batched WGrad GEMM requires three tensors")
        loss, columns, partial = tensors
        if (
            loss["data_type"] != columns["data_type"]
            or loss["data_type"] != partial["data_type"]
            or loss["data_type"] not in FLOAT_DATA_TYPES
            or len(loss["dimensions"]) != 4
            or len(columns["dimensions"]) != 3
            or len(partial["dimensions"]) != 3
            or not all(
                _is_row_major_contiguous(tensor)
                for tensor in (loss, columns, partial)
            )
        ):
            raise ValueError("the batched WGrad GEMM metadata is invalid")
        n, c_out, loss_h, loss_w = loss["dimensions"]
        columns_n, cik, padded_m = columns["dimensions"]
        m = loss_h * loss_w
        if num_splits % n != 0 or columns_n != n or padded_m < m:
            raise ValueError("the batched WGrad GEMM shape is inconsistent")
        splits_per_n = num_splits // n
        if partial["dimensions"] != [num_splits, c_out, cik]:
            raise ValueError("the batched WGrad partial shape is inconsistent")
        short_split = (
            parameters.get("_wgrad_pipeline_algorithm") == "1x1_split"
        )
        if short_split and loss["data_type"] == "float32":
            raise ValueError(
                "the short WGrad split requires low-precision inputs"
            )
        block_co = 64 if short_split else 16
        block_ci = 32
        block_m = 128
        pointer_type = TRITON_POINTER_TYPES[loss["data_type"]]
        return (
            (
                "_conv_wgrad2d_batched_split_kernel"
                if short_split
                else "_conv_wgrad2d_batched_tma_kernel"
            ),
            {
                "loss_ptr": pointer_type,
                "columns_ptr": pointer_type,
                "partial_ptr": pointer_type,
            },
            {
                "BATCH_N": n,
                "M": m,
                "PADDED_M": padded_m,
                "C_OUT": c_out,
                "CIK": cik,
                "SPLITS_PER_N": splits_per_n,
                "INPUT_IS_FLOAT32": loss["data_type"] == "float32",
                "BLOCK_CO": block_co,
                "BLOCK_CI": block_ci,
                "BLOCK_M": block_m,
            },
            (
                _ceil_div(c_out, block_co) * _ceil_div(cik, block_ci),
                splits_per_n,
                n,
            ),
            [("tensor", None), ("tensor", None), ("tensor", None)],
        )

    if len(tensors) != 2:
        raise ValueError("the batched WGrad reduce stage requires two tensors")
    partial, output = tensors
    if (
        partial["data_type"] != output["data_type"]
        or output["data_type"] not in FLOAT_DATA_TYPES
        or len(partial["dimensions"]) != 3
        or len(output["dimensions"]) != 4
        or not _is_row_major_contiguous(partial)
    ):
        raise ValueError("the batched WGrad reduce metadata is invalid")
    c_out, cin_per_group, output_kh, output_kw = output["dimensions"]
    cik = cin_per_group * kernel_h * kernel_w
    if (
        output_kh != kernel_h
        or output_kw != kernel_w
        or partial["dimensions"] != [num_splits, c_out, cik]
    ):
        raise ValueError("the batched WGrad reduce shape is inconsistent")
    if kernel_h == 1 and kernel_w == 1 and _is_row_major_contiguous(output):
        block_m = 256
        block_n = 16
        pointer_type = TRITON_POINTER_TYPES[output["data_type"]]
        total = c_out * cik
        return (
            "_conv_wgrad2d_split_vector_reduce_kernel",
            {"partial_ptr": pointer_type, "out_ptr": pointer_type},
            {
                "TOTAL": total,
                "NUM_SPLITS": num_splits,
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
            },
            (_ceil_div(total, block_m), 1, 1),
            [("tensor", None), ("tensor", None)],
        )
    block_co = 16
    block_n = 32
    pointer_type = TRITON_POINTER_TYPES[output["data_type"]]
    return (
        "_conv_wgrad2d_col_reduce_kernel",
        {"partial_ptr": pointer_type, "out_ptr": pointer_type},
        {
            "C_OUT": c_out,
            "CIN_PER_GROUP": cin_per_group,
            "COUT_PER_GROUP": c_out,
            "out_stride_o": output["strides"][0],
            "out_stride_i": output["strides"][1],
            "out_stride_h": output["strides"][2],
            "out_stride_w": output["strides"][3],
            "KH": kernel_h,
            "KW": kernel_w,
            "NUM_SPLITS": num_splits,
            "BLOCK_CO": block_co,
            "BLOCK_N": block_n,
        },
        (_ceil_div(c_out, block_co) * _ceil_div(cik, block_n), 1, 1),
        [("tensor", None), ("tensor", None)],
    )
