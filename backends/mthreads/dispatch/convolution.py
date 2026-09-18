"""MThreads dispatch convolution implementation."""

from __future__ import annotations

import itertools
import math
from typing import Any

from .common import (
    _CONVOLUTION_SOURCE_RELATIVE_PATH,
    _SELECTION_CACHE,
    _WORKSPACE_ALIGNMENT,
    _WORKSPACE_SIZE,
    _pointer_token,
)
from .graph import ParsedConvolutionRequest
from .program import (
    AutotuneSpec,
    ExecutionPlan,
    KernelStage,
    KernelVariant,
    RuntimeArgument,
)
from .tensor import is_row_major_contiguous
from .tuning import _load_tuning


def _padded_convolution_tensor(
    tensor: Any, spatial_rank: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    leading = 3 - spatial_rank
    return (
        (*tensor.dimensions[:2], *([1] * leading), *tensor.dimensions[2:]),
        (*tensor.strides[:2], *([0] * leading), *tensor.strides[2:]),
    )


def _padded_convolution_spatial(
    values: tuple[int, ...], fill: int
) -> tuple[int, int, int]:
    return tuple((*([fill] * (3 - len(values))), *values))  # type: ignore[return-value]


def _convolution_function(request: ParsedConvolutionRequest) -> str:
    if request.operation in {"conv2d_fprop", "convolution_fprop"}:
        return {
            1: "conv1d_gemm_kernel",
            2: "conv2d_spatial_nchw_kernel",
            3: "conv3d_spatial_ncdhw_m_kernel",
        }[request.spatial_rank]
    if request.operation == "convolution_dgrad":
        return "conv_dgrad_nd_kernel"
    return "conv_wgrad_nd_kernel"


def _uses_stride2_tile4_dgrad(
    request: ParsedConvolutionRequest,
) -> bool:
    return (
        request.operation == "convolution_dgrad"
        and request.spatial_rank == 2
        and request.image.data_type in {"float32", "float16", "bfloat16"}
        and request.result.data_type == request.image.data_type
        and request.filter.data_type == request.image.data_type
        and request.filter.dimensions[2:] == (3, 3)
        and request.stride == (2, 2)
        and request.pre_padding == (1, 1)
        and request.post_padding == (1, 1)
        and request.dilation == (1, 1)
        and request.convolution_mode == 0
        and is_row_major_contiguous(request.image)
        and is_row_major_contiguous(request.result)
        and is_row_major_contiguous(request.filter)
    )


def _uses_stride2_packed2_1d_dgrad(
    request: ParsedConvolutionRequest,
) -> bool:
    return (
        request.operation == "convolution_dgrad"
        and request.spatial_rank == 1
        and request.image.data_type in {"float32", "float16", "bfloat16"}
        and request.result.data_type == request.image.data_type
        and request.filter.data_type == request.image.data_type
        and request.filter.dimensions[2:] == (5,)
        and request.stride == (2,)
        and request.pre_padding == (2,)
        and request.post_padding == (1,)
        and request.dilation == (1,)
        and request.convolution_mode == 0
        and is_row_major_contiguous(request.image)
        and is_row_major_contiguous(request.result)
        and is_row_major_contiguous(request.filter)
    )


def _uses_stride2_packed4_dgrad(
    request: ParsedConvolutionRequest,
) -> bool:
    return _uses_stride2_tile4_dgrad(request) and request.in_per_group <= 4


def _stride2_packed4_dgrad_block_m(
    request: ParsedConvolutionRequest,
    block_size: int,
) -> int:
    return (
        block_size * 4 if _uses_stride2_packed4_dgrad(request) else block_size
    )


def _convolution_runtime_arguments(
    request: ParsedConvolutionRequest,
) -> tuple[RuntimeArgument, ...]:
    if request.operation in {"conv2d_fprop", "convolution_fprop"}:
        return (
            RuntimeArgument("tensor", "input", request.image.uid, None),
            RuntimeArgument("tensor", "filter", request.filter.uid, None),
            RuntimeArgument(
                "tensor", "bias_placeholder", request.image.uid, None
            ),
            RuntimeArgument("tensor", "output", request.result.uid, None),
        )
    if request.operation == "convolution_dgrad":
        return (
            RuntimeArgument("tensor", "dy", request.result.uid, None),
            RuntimeArgument("tensor", "w", request.filter.uid, None),
            RuntimeArgument("tensor", "dx", request.image.uid, None),
        )
    return (
        RuntimeArgument("tensor", "dy", request.result.uid, None),
        RuntimeArgument("tensor", "x", request.image.uid, None),
        RuntimeArgument("tensor", "dw", request.filter.uid, None),
    )


def _convolution_full_signature(
    request: ParsedConvolutionRequest,
    *,
    function: str,
    block_size: int,
) -> str:
    image_dims, image_strides = _padded_convolution_tensor(
        request.image, request.spatial_rank
    )
    filter_dims, filter_strides = _padded_convolution_tensor(
        request.filter, request.spatial_rank
    )
    result_dims, result_strides = _padded_convolution_tensor(
        request.result, request.spatial_rank
    )
    _, _, xd, xh, xw = image_dims
    _, _, kd, kh, kw = filter_dims
    _, _, od, oh, ow = result_dims
    stride_d, stride_h, stride_w = _padded_convolution_spatial(
        request.stride, 1
    )
    pad_front, pad_top, pad_left = _padded_convolution_spatial(
        request.pre_padding, 0
    )
    dil_d, dil_h, dil_w = _padded_convolution_spatial(request.dilation, 1)
    dtype_id = {
        "float32": 0,
        "float16": 1,
        "bfloat16": 2,
    }[request.image.data_type]
    fprop_operation = request.operation in {
        "conv2d_fprop",
        "convolution_fprop",
    }
    fprop_reduction_extent = request.in_per_group * math.prod(
        request.filter.dimensions[2:]
    )
    input_precision = (
        1
        if request.image.data_type == "float32"
        and (not fprop_operation or fprop_reduction_extent >= 64)
        else 0
    )
    pointer = lambda tensor: _pointer_token(  # noqa: E731
        tensor.data_type, tensor.alignment
    )
    if request.input_precision:
        input_precision = int(request.input_precision == 2)

    if function.startswith("conv") and request.operation in {
        "conv2d_fprop",
        "convolution_fprop",
    }:
        tokens = [
            pointer(request.image),
            pointer(request.filter),
            pointer(request.image),
            pointer(request.result),
        ]
    elif request.operation == "convolution_dgrad":
        tokens = [
            pointer(request.result),
            pointer(request.filter),
            pointer(request.image),
        ]
    else:
        tokens = [
            pointer(request.result),
            pointer(request.image),
            pointer(request.filter),
        ]

    if function == "conv1d_gemm_kernel":
        constants = (
            request.batch * ow,
            xw,
            ow,
            dtype_id,
            *request.image.strides,
            *request.filter.strides,
            1,
            *request.result.strides,
            request.in_per_group,
            request.out_per_group,
            kw,
            stride_w,
            pad_left,
            dil_w,
            0,
            block_size,
            block_size,
            block_size,
            8,
            input_precision,
        )
    elif function == "conv2d_spatial_nchw_kernel":
        constants = (
            xh,
            xw,
            oh,
            ow,
            request.in_channels,
            request.out_channels,
            request.in_per_group,
            request.out_per_group,
            request.groups,
            stride_h,
            stride_w,
            pad_top,
            pad_left,
            dil_h,
            dil_w,
            kh,
            kw,
            0,
            block_size,
            block_size,
            block_size,
            8,
            dtype_id,
            input_precision,
            image_strides[0],
            image_strides[1],
            image_strides[3],
            image_strides[4],
            filter_strides[0],
            filter_strides[1],
            filter_strides[3],
            filter_strides[4],
            result_strides[0],
            result_strides[1],
            result_strides[3],
            result_strides[4],
        )
    elif function == "conv3d_spatial_ncdhw_m_kernel":
        constants = (
            request.batch * od * oh * ow,
            xd,
            xh,
            xw,
            od,
            oh,
            ow,
            request.in_channels,
            request.out_channels,
            request.in_per_group,
            request.out_per_group,
            stride_d,
            stride_h,
            stride_w,
            pad_front,
            pad_top,
            pad_left,
            dil_d,
            dil_h,
            dil_w,
            kd,
            kh,
            kw,
            0,
            block_size,
            block_size,
            block_size,
            8,
            *image_strides,
            *filter_strides,
            *result_strides,
            input_precision,
        )
    else:
        common = (
            xd,
            xh,
            xw,
            od,
            oh,
            ow,
            kd,
            kh,
            kw,
            request.in_per_group,
            request.out_per_group,
            stride_d,
            stride_h,
            stride_w,
            pad_front,
            pad_top,
            pad_left,
            dil_d,
            dil_h,
            dil_w,
            request.convolution_mode,
            *result_strides,
            *image_strides,
            *filter_strides,
            (
                int(request.input_precision == 2)
                if request.input_precision
                else int(
                    request.image.data_type == "float32"
                    and (
                        request.operation == "convolution_wgrad"
                        or _uses_stride2_tile4_dgrad(request)
                        or _uses_stride2_packed2_1d_dgrad(request)
                    )
                )
            ),
        )
        if function == "conv_dgrad_nd_kernel":
            constants = (
                *common,
                request.batch * xd * xh * xw,
                _stride2_packed4_dgrad_block_m(request, block_size),
                block_size,
                block_size,
                8,
            )
        else:
            constants = (
                *common,
                request.batch * od * oh * ow,
                block_size,
                block_size,
                block_size,
            )
    tokens.extend(str(value) for value in constants)
    return ",".join(tokens)


def _convolution_grid(
    request: ParsedConvolutionRequest,
    *,
    function: str,
    block_size: int,
) -> tuple[int, int, int]:
    image_dims, _ = _padded_convolution_tensor(
        request.image, request.spatial_rank
    )
    filter_dims, _ = _padded_convolution_tensor(
        request.filter, request.spatial_rank
    )
    result_dims, _ = _padded_convolution_tensor(
        request.result, request.spatial_rank
    )
    _, _, xd, xh, xw = image_dims
    _, _, kd, kh, kw = filter_dims
    _, _, od, oh, ow = result_dims
    ceil = lambda value: (value + block_size - 1) // block_size  # noqa: E731
    if function == "conv1d_gemm_kernel":
        return (
            ceil(request.batch * ow) * ceil(request.out_per_group),
            request.groups,
            1,
        )
    if function == "conv2d_spatial_nchw_kernel":
        return (
            ceil(oh * ow) * ceil(request.out_per_group),
            request.batch * request.groups,
            1,
        )
    if function == "conv3d_spatial_ncdhw_m_kernel":
        return (
            ceil(request.batch * od * oh * ow) * ceil(request.out_per_group),
            request.groups,
            1,
        )
    if function == "conv_dgrad_nd_kernel":
        rows = (
            request.batch * ((xw + 1) // 2)
            if _uses_stride2_packed2_1d_dgrad(request)
            else (
                request.batch * od * oh * ow
                if _uses_stride2_tile4_dgrad(request)
                else request.batch * xd * xh * xw
            )
        )
        block_m = _stride2_packed4_dgrad_block_m(request, block_size)
        channel_block = (
            block_size // 4
            if _uses_stride2_packed4_dgrad(request)
            else block_size
        )
        return (
            ((rows + block_m - 1) // block_m)
            * ((request.in_per_group + channel_block - 1) // channel_block),
            request.groups,
            1,
        )
    return (
        ceil(request.out_per_group) * ceil(request.in_per_group),
        kd * kh * kw,
        request.groups,
    )


def _uses_im2col_fprop(request: ParsedConvolutionRequest) -> bool:
    filter_area = math.prod(request.filter.dimensions[2:])
    standard_stride2_3x3 = (
        request.filter.dimensions[2:] == (3, 3)
        and request.stride == (2, 2)
        and request.pre_padding == (1, 1)
        and request.post_padding == (1, 1)
        and request.dilation == (1, 1)
    )
    stride2_3x3 = standard_stride2_3x3 and request.in_per_group >= 64
    fp32_stem = (
        standard_stride2_3x3
        and request.image.data_type == "float32"
        and request.image.dimensions == (1, 3, 640, 640)
        and request.filter.dimensions[1:] == (3, 3, 3)
        and request.out_per_group in {16, 32, 64, 96}
    )
    medium_batched = (
        request.batch >= 4
        and request.in_per_group >= 32
        and 2 <= filter_area <= 15
    )
    if not (
        request.operation in {"conv2d_fprop", "convolution_fprop"}
        and request.spatial_rank == 2
        and request.image.data_type in {"float32", "float16", "bfloat16"}
        and request.filter.data_type == request.image.data_type
        and request.result.data_type == request.image.data_type
        and request.groups == 1
        and request.convolution_mode == 0
        and is_row_major_contiguous(request.image)
        and is_row_major_contiguous(request.filter)
        and is_row_major_contiguous(request.result)
        and (stride2_3x3 or fp32_stem or medium_batched)
    ):
        return False
    return _im2col_fprop_workspace_size(request) <= 512 * 1024 * 1024


def _im2col_fprop_geometry(
    request: ParsedConvolutionRequest,
) -> tuple[int, int, tuple[int, int, int]]:
    output_area = math.prod(request.result.dimensions[2:])
    reduction_extent = request.in_per_group * math.prod(
        request.filter.dimensions[2:]
    )
    column_strides = (
        reduction_extent * output_area,
        output_area,
        1,
    )
    return output_area, reduction_extent, column_strides


def _im2col_fprop_workspace_size(
    request: ParsedConvolutionRequest,
) -> int:
    output_area, reduction_extent, _ = _im2col_fprop_geometry(request)
    element_size = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
    }[request.image.data_type]
    raw_size = request.batch * reduction_extent * output_area * element_size
    aligned_size = (
        (raw_size + _WORKSPACE_ALIGNMENT - 1)
        // _WORKSPACE_ALIGNMENT
        * _WORKSPACE_ALIGNMENT
    )
    return max(_WORKSPACE_SIZE, aligned_size)


def _im2col_fprop_mm_blocks(
    request: ParsedConvolutionRequest,
) -> tuple[int, int, int]:
    _, reduction_extent, _ = _im2col_fprop_geometry(request)
    if request.batch == 1 and reduction_extent >= 1024:
        return 64, 32, 64
    return (
        64,
        64,
        64 if request.image.data_type == "float32" else 32,
    )


def _im2col_fprop_signature(
    request: ParsedConvolutionRequest,
    *,
    function: str,
    block_size: int,
) -> str:
    output_area, reduction_extent, column_strides = _im2col_fprop_geometry(
        request
    )
    _, _, input_height, input_width = request.image.dimensions
    _, _, output_height, output_width = request.result.dimensions
    _, _, filter_height, filter_width = request.filter.dimensions
    workspace = _pointer_token(request.image.data_type, _WORKSPACE_ALIGNMENT)
    if function == "_conv_fprop2d_im2col_kernel":
        tokens = [
            _pointer_token(request.image.data_type, request.image.alignment),
            workspace,
            str(output_area),
            str(input_height),
            str(input_width),
            str(output_height),
            str(output_width),
            str(request.in_per_group),
            str(filter_height),
            str(filter_width),
            *(str(value) for value in request.stride),
            *(str(value) for value in request.pre_padding),
            *(str(value) for value in request.dilation),
            *(str(value) for value in request.image.strides),
            *(str(value) for value in column_strides),
            str(block_size),
            "32",
        ]
    elif function == "_conv_fprop2d_im2col_mm_kernel":
        block_oc, block_m, block_k = _im2col_fprop_mm_blocks(request)
        if block_oc != block_size:
            raise ValueError("im2col Fprop candidate block differs")
        tokens = [
            _pointer_token(request.filter.data_type, request.filter.alignment),
            workspace,
            _pointer_token(request.result.data_type, request.result.alignment),
            str(output_area),
            str(request.out_per_group),
            str(request.in_per_group),
            str(filter_height),
            str(filter_width),
            *(str(value) for value in request.filter.strides),
            *(str(value) for value in request.result.strides),
            str(output_width),
            *(str(value) for value in column_strides),
            (
                "1"
                if request.image.data_type == "float32"
                and request.input_precision != 1
                else "0"
            ),
            str(block_oc),
            str(block_m),
            str(block_k),
            "8",
        ]
    else:
        raise ValueError("unknown im2col Fprop stage function")
    if reduction_extent <= 0:
        raise ValueError("im2col Fprop reduction extent is invalid")
    return ",".join(tokens)


def _im2col_fprop_stage(
    request: ParsedConvolutionRequest,
    source_sha256: str,
    *,
    stage_id: int,
    function: str,
    dependencies: tuple[int, ...],
) -> KernelStage:
    target_warp = int(request.target.rsplit("-w", 1)[1])
    _, _, warmup, repetitions = _load_tuning(
        table_name="convolution",
        warp_size=target_warp,
    )
    output_area, reduction_extent, _ = _im2col_fprop_geometry(request)
    if function == "_conv_fprop2d_im2col_kernel":
        block_size, num_warps, num_stages = (64, 4, 1)
        arguments = (
            RuntimeArgument("tensor", "input", request.image.uid, None),
            RuntimeArgument("workspace", "fprop_columns", None, None),
        )
        grid = (
            ((output_area + block_size - 1) // block_size)
            * ((reduction_extent + 31) // 32),
            request.batch,
            1,
        )
    elif function == "_conv_fprop2d_im2col_mm_kernel":
        block_size, num_warps, num_stages = (64, 8, 1)
        block_oc, block_m, _ = _im2col_fprop_mm_blocks(request)
        arguments = (
            RuntimeArgument("tensor", "filter", request.filter.uid, None),
            RuntimeArgument("workspace", "fprop_columns", None, None),
            RuntimeArgument("tensor", "output", request.result.uid, None),
        )
        grid = (
            ((request.out_per_group + block_oc - 1) // block_oc)
            * ((output_area + block_m - 1) // block_m),
            request.batch,
            1,
        )
    else:
        raise ValueError("unknown im2col Fprop stage function")
    variant = KernelVariant(
        variant_id=(
            f"block-{block_size}-warps-{num_warps}-" f"stages-{num_stages}"
        ),
        source=_CONVOLUTION_SOURCE_RELATIVE_PATH,
        source_sha256=source_sha256,
        function=function,
        full_signature=_im2col_fprop_signature(
            request,
            function=function,
            block_size=block_size,
        ),
        grid=grid,
        num_warps=num_warps,
        num_stages=num_stages,
        arguments=arguments,
    )
    return KernelStage(
        id=stage_id,
        node_id=request.node_id,
        operation=request.operation,
        dependencies=dependencies,
        source=_CONVOLUTION_SOURCE_RELATIVE_PATH,
        function=function,
        variants=(variant,),
        autotune=AutotuneSpec(
            enabled=False,
            warmup=warmup,
            repetitions=repetitions,
            selection_cache=f"tuning/stage-{stage_id}.json",
        ),
    )


def _im2col_fprop_plan(
    request: ParsedConvolutionRequest,
    source_sha256: str,
) -> ExecutionPlan:
    pack = _im2col_fprop_stage(
        request,
        source_sha256,
        stage_id=0,
        function="_conv_fprop2d_im2col_kernel",
        dependencies=(),
    )
    matmul = _im2col_fprop_stage(
        request,
        source_sha256,
        stage_id=1,
        function="_conv_fprop2d_im2col_mm_kernel",
        dependencies=(0,),
    )
    return ExecutionPlan(
        stages=(pack, matmul),
        external_binding_uids=request.external_binding_uids,
        workspace_size=_im2col_fprop_workspace_size(request),
        workspace_alignment=_WORKSPACE_ALIGNMENT,
    )


def _uses_dense_stride2_dgrad(
    request: ParsedConvolutionRequest,
) -> bool:
    return (
        _uses_stride2_tile4_dgrad(request)
        and request.groups == 1
        and request.in_per_group > 4
    )


def _dense_dgrad_element_size(request: ParsedConvolutionRequest) -> int:
    return {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
    }[request.image.data_type]


def _dense_dgrad_geometry(
    request: ParsedConvolutionRequest,
) -> tuple[int, int, int, int]:
    loss_rows = request.batch * math.prod(request.result.dimensions[2:])
    packed_filter_elements = 16 * request.in_per_group * request.out_per_group
    packed_loss_elements = 4 * request.out_per_group * loss_rows
    element_size = _dense_dgrad_element_size(request)
    packed_filter_bytes = packed_filter_elements * element_size
    aligned_filter_bytes = (
        (packed_filter_bytes + _WORKSPACE_ALIGNMENT - 1)
        // _WORKSPACE_ALIGNMENT
        * _WORKSPACE_ALIGNMENT
    )
    loss_offset = aligned_filter_bytes // element_size
    return (
        loss_rows,
        packed_filter_elements,
        packed_loss_elements,
        loss_offset,
    )


def _dense_dgrad_workspace_size(
    request: ParsedConvolutionRequest,
) -> int:
    _, _, packed_loss_elements, loss_offset = _dense_dgrad_geometry(request)
    element_size = _dense_dgrad_element_size(request)
    raw_size = (loss_offset + packed_loss_elements) * element_size
    aligned_size = (
        (raw_size + _WORKSPACE_ALIGNMENT - 1)
        // _WORKSPACE_ALIGNMENT
        * _WORKSPACE_ALIGNMENT
    )
    return max(_WORKSPACE_SIZE, aligned_size)


def _dense_dgrad_signature(
    request: ParsedConvolutionRequest,
    *,
    function: str,
    block_size: int,
) -> str:
    loss_rows, _, _, loss_offset = _dense_dgrad_geometry(request)
    workspace = _pointer_token(request.image.data_type, _WORKSPACE_ALIGNMENT)
    if function == "_conv_dgrad2d_dense_pack_filter_kernel":
        tokens = [
            _pointer_token(request.filter.data_type, request.filter.alignment),
            workspace,
            str(request.in_per_group),
            str(request.out_per_group),
            *(str(value) for value in request.filter.strides),
            str(block_size),
            str(block_size),
        ]
    elif function == "_conv_dgrad2d_dense_pack_loss_kernel":
        tokens = [
            _pointer_token(request.result.data_type, request.result.alignment),
            workspace,
            str(loss_offset),
            str(loss_rows),
            str(request.result.dimensions[2]),
            str(request.result.dimensions[3]),
            str(request.out_per_group),
            *(str(value) for value in request.result.strides),
            str(block_size),
            str(block_size),
        ]
    elif function == "_conv_dgrad2d_dense_mm_kernel":
        tokens = [
            workspace,
            workspace,
            _pointer_token(request.image.data_type, request.image.alignment),
            str(loss_offset),
            str(loss_rows),
            str(request.result.dimensions[2]),
            str(request.result.dimensions[3]),
            str(request.image.dimensions[2]),
            str(request.image.dimensions[3]),
            str(request.in_per_group),
            str(request.out_per_group),
            *(str(value) for value in request.image.strides),
            (
                "1"
                if request.image.data_type == "float32"
                and request.input_precision != 1
                else "0"
            ),
            str(block_size),
            str(block_size),
            str(block_size),
            "8",
        ]
    else:
        raise ValueError("unknown dense Dgrad stage function")
    return ",".join(tokens)


def _dense_dgrad_stage(
    request: ParsedConvolutionRequest,
    source_sha256: str,
    *,
    stage_id: int,
    function: str,
    dependencies: tuple[int, ...],
) -> KernelStage:
    target_warp = int(request.target.rsplit("-w", 1)[1])
    _, _, warmup, repetitions = _load_tuning(
        table_name="convolution",
        warp_size=target_warp,
    )
    loss_rows, _, _, _ = _dense_dgrad_geometry(request)
    if function == "_conv_dgrad2d_dense_pack_filter_kernel":
        candidates = ((32, 8, 1),)
        arguments = (
            RuntimeArgument("tensor", "w", request.filter.uid, None),
            RuntimeArgument("workspace", "dgrad_dense_filter", None, None),
        )
        grid = lambda block: (  # noqa: E731
            (4 * request.in_per_group + block - 1) // block,
            (4 * request.out_per_group + block - 1) // block,
            1,
        )
        autotune = False
    elif function == "_conv_dgrad2d_dense_pack_loss_kernel":
        candidates = ((32, 8, 1),)
        arguments = (
            RuntimeArgument("tensor", "dy", request.result.uid, None),
            RuntimeArgument("workspace", "dgrad_dense_loss", None, None),
        )
        grid = lambda block: (  # noqa: E731
            (4 * request.out_per_group + block - 1) // block,
            (loss_rows + block - 1) // block,
            1,
        )
        autotune = False
    elif function == "_conv_dgrad2d_dense_mm_kernel":
        default = (64, 8, 1)
        candidates = (
            tuple((64, warps, stages) for warps in (4, 8) for stages in (1, 2))
            if request.autotune
            else (default,)
        )
        arguments = (
            RuntimeArgument("workspace", "dgrad_dense_filter", None, None),
            RuntimeArgument("workspace", "dgrad_dense_loss", None, None),
            RuntimeArgument("tensor", "dx", request.image.uid, None),
        )
        grid = lambda block: (  # noqa: E731
            ((4 * request.in_per_group + block - 1) // block)
            * ((loss_rows + block - 1) // block),
            1,
            1,
        )
        autotune = request.autotune
    else:
        raise ValueError("unknown dense Dgrad stage function")
    variants = tuple(
        KernelVariant(
            variant_id=(
                f"block-{block_size}-warps-{num_warps}-" f"stages-{num_stages}"
            ),
            source=_CONVOLUTION_SOURCE_RELATIVE_PATH,
            source_sha256=source_sha256,
            function=function,
            full_signature=_dense_dgrad_signature(
                request, function=function, block_size=block_size
            ),
            grid=grid(block_size),
            num_warps=num_warps,
            num_stages=num_stages,
            arguments=arguments,
        )
        for block_size, num_warps, num_stages in candidates
    )
    return KernelStage(
        id=stage_id,
        node_id=request.node_id,
        operation=request.operation,
        dependencies=dependencies,
        source=_CONVOLUTION_SOURCE_RELATIVE_PATH,
        function=function,
        variants=variants,
        autotune=AutotuneSpec(
            enabled=autotune,
            warmup=warmup,
            repetitions=repetitions,
            selection_cache=f"tuning/stage-{stage_id}.json",
        ),
    )


def _dense_dgrad_plan(
    request: ParsedConvolutionRequest,
    source_sha256: str,
) -> ExecutionPlan:
    stage_specs = (
        ("_conv_dgrad2d_dense_pack_filter_kernel", ()),
        ("_conv_dgrad2d_dense_pack_loss_kernel", ()),
        ("_conv_dgrad2d_dense_mm_kernel", (0, 1)),
    )
    stages = tuple(
        _dense_dgrad_stage(
            request,
            source_sha256,
            stage_id=stage_id,
            function=function,
            dependencies=dependencies,
        )
        for stage_id, (function, dependencies) in enumerate(stage_specs)
    )
    return ExecutionPlan(
        stages=stages,
        external_binding_uids=request.external_binding_uids,
        workspace_size=_dense_dgrad_workspace_size(request),
        workspace_alignment=_WORKSPACE_ALIGNMENT,
    )


def _uses_nd_packed_wgrad(request: ParsedConvolutionRequest) -> bool:
    if request.input_precision == 1:
        return False
    shape_key = (
        request.image.dimensions,
        request.result.dimensions,
        request.filter.dimensions,
        request.stride,
        request.pre_padding,
        request.post_padding,
        request.dilation,
    )
    supported_shapes = {
        (
            (16, 32, 256),
            (16, 64, 256),
            (64, 32, 3),
            (1,),
            (1,),
            (1,),
            (1,),
        ),
        (
            (2, 8, 8, 16, 16),
            (2, 16, 8, 16, 16),
            (16, 8, 3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
        ),
        (
            (1, 8, 10, 12, 14),
            (1, 12, 10, 11, 15),
            (12, 8, 2, 3, 3),
            (1, 1, 1),
            (1, 0, 1),
            (0, 1, 2),
            (1, 1, 1),
        ),
    }
    return (
        request.operation == "convolution_wgrad"
        and request.image.data_type in {"float32", "float16", "bfloat16"}
        and request.result.data_type == request.image.data_type
        and request.filter.data_type == request.image.data_type
        and shape_key in supported_shapes
        and is_row_major_contiguous(request.image)
        and is_row_major_contiguous(request.result)
        and is_row_major_contiguous(request.filter)
        and request.spatial_rank in {1, 3}
        and request.groups == 1
        and request.convolution_mode == 0
    )


def _nd_packed_wgrad_spatial3(
    values: tuple[int, ...],
    fill: int,
) -> tuple[int, int, int]:
    return (fill,) * (3 - len(values)) + values


def _nd_packed_wgrad_num_splits(
    request: ParsedConvolutionRequest,
) -> int:
    total_rows = request.batch * math.prod(request.result.dimensions[2:])
    return 16 if total_rows >= 4096 else 8


def _nd_packed_wgrad_geometry(
    request: ParsedConvolutionRequest,
) -> tuple[int, int, int, int, int]:
    output_area = math.prod(request.result.dimensions[2:])
    reduction_extent = request.in_per_group * math.prod(
        request.filter.dimensions[2:]
    )
    total_rows = request.batch * output_area
    total_weights = request.out_per_group * reduction_extent
    num_splits = _nd_packed_wgrad_num_splits(request)
    return (
        output_area,
        reduction_extent,
        total_rows,
        total_weights,
        num_splits,
    )


def _nd_packed_wgrad_partial_size(
    request: ParsedConvolutionRequest,
) -> int:
    _, _, _, total_weights, num_splits = _nd_packed_wgrad_geometry(request)
    raw_size = num_splits * total_weights * 4
    return (
        (raw_size + _WORKSPACE_ALIGNMENT - 1)
        // _WORKSPACE_ALIGNMENT
        * _WORKSPACE_ALIGNMENT
    )


def _nd_packed_wgrad_column_offset(
    request: ParsedConvolutionRequest,
) -> int:
    element_size = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
    }[request.image.data_type]
    return _nd_packed_wgrad_partial_size(request) // element_size


def _nd_packed_wgrad_workspace_size(
    request: ParsedConvolutionRequest,
) -> int:
    _, reduction_extent, total_rows, _, _ = _nd_packed_wgrad_geometry(request)
    element_size = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
    }[request.image.data_type]
    raw_size = (
        _nd_packed_wgrad_partial_size(request)
        + total_rows * reduction_extent * element_size
    )
    aligned_size = (
        (raw_size + _WORKSPACE_ALIGNMENT - 1)
        // _WORKSPACE_ALIGNMENT
        * _WORKSPACE_ALIGNMENT
    )
    return max(_WORKSPACE_SIZE, aligned_size)


def _nd_packed_wgrad_signature(
    request: ParsedConvolutionRequest,
    *,
    function: str,
    block_size: int,
) -> str:
    output_area, reduction_extent, total_rows, total_weights, num_splits = (
        _nd_packed_wgrad_geometry(request)
    )
    rows_per_split = (total_rows + num_splits - 1) // num_splits
    input_spatial = _nd_packed_wgrad_spatial3(request.image.dimensions[2:], 1)
    output_spatial = _nd_packed_wgrad_spatial3(
        request.result.dimensions[2:], 1
    )
    kernel_spatial = _nd_packed_wgrad_spatial3(
        request.filter.dimensions[2:], 1
    )
    stride = _nd_packed_wgrad_spatial3(request.stride, 1)
    padding = _nd_packed_wgrad_spatial3(request.pre_padding, 0)
    dilation = _nd_packed_wgrad_spatial3(request.dilation, 1)
    input_strides = (
        request.image.strides[:2]
        + (0,) * (3 - request.spatial_rank)
        + request.image.strides[2:]
    )
    dtype_id = {
        "float32": 0,
        "float16": 1,
        "bfloat16": 2,
    }[request.image.data_type]
    packed = _pointer_token(request.image.data_type, _WORKSPACE_ALIGNMENT)
    partial = _pointer_token("float32", _WORKSPACE_ALIGNMENT)
    if function == "_conv_wgrad_nd_im2row_kernel":
        tokens = [
            _pointer_token(request.image.data_type, request.image.alignment),
            packed,
            str(output_area),
            *(str(value) for value in input_spatial),
            str(output_spatial[1]),
            str(output_spatial[2]),
            str(request.in_per_group),
            *(str(value) for value in kernel_spatial),
            *(str(value) for value in stride),
            *(str(value) for value in padding),
            *(str(value) for value in dilation),
            *(str(value) for value in input_strides),
            str(reduction_extent),
            str(_nd_packed_wgrad_column_offset(request)),
            "1",
            str(block_size),
            str(block_size),
        ]
    elif function == "_conv_wgrad_nd_rowmajor_kernel":
        tokens = [
            _pointer_token(request.result.data_type, request.result.alignment),
            packed,
            partial,
            str(total_rows),
            str(rows_per_split),
            str(output_area),
            str(request.out_per_group),
            str(reduction_extent),
            str(request.result.strides[0]),
            str(request.result.strides[1]),
            str(request.result.strides[-1]),
            str(reduction_extent),
            str(_nd_packed_wgrad_column_offset(request)),
            "1",
            str(total_weights),
            str(reduction_extent),
            "1",
            str(dtype_id),
            str(block_size),
            str(block_size),
            str(block_size),
        ]
    elif function == "_conv_wgrad_nd_reduce_kernel":
        tokens = [
            partial,
            _pointer_token(request.filter.data_type, request.filter.alignment),
            str(total_weights),
            str(num_splits),
            str(total_weights),
            str(block_size),
        ]
    else:
        raise ValueError("unknown ND packed Wgrad stage function")
    return ",".join(tokens)


def _nd_packed_wgrad_stage(
    request: ParsedConvolutionRequest,
    source_sha256: str,
    *,
    stage_id: int,
    function: str,
    dependencies: tuple[int, ...],
) -> KernelStage:
    target_warp = int(request.target.rsplit("-w", 1)[1])
    _, _, warmup, repetitions = _load_tuning(
        table_name="convolution",
        warp_size=target_warp,
    )
    output_area, reduction_extent, _, total_weights, num_splits = (
        _nd_packed_wgrad_geometry(request)
    )
    if function == "_conv_wgrad_nd_im2row_kernel":
        default = (64, 4, 1)
        candidates = tuple(itertools.product((32, 64), (4, 8), (1,)))
        arguments = (
            RuntimeArgument("tensor", "x", request.image.uid, None),
            RuntimeArgument("workspace", "wgrad_nd_columns", None, None),
        )
        grid = lambda block: (  # noqa: E731
            ((output_area + block - 1) // block)
            * ((reduction_extent + block - 1) // block),
            request.batch,
            1,
        )
    elif function == "_conv_wgrad_nd_rowmajor_kernel":
        default = (64, 8, 1)
        candidates = tuple(itertools.product((32, 64), (4, 8), (1, 2)))
        arguments = (
            RuntimeArgument("tensor", "dy", request.result.uid, None),
            RuntimeArgument("workspace", "wgrad_nd_columns", None, None),
            RuntimeArgument("workspace", "wgrad_nd_partial", None, None),
        )
        grid = lambda block: (  # noqa: E731
            ((request.out_per_group + block - 1) // block)
            * ((reduction_extent + block - 1) // block),
            num_splits,
            1,
        )
    elif function == "_conv_wgrad_nd_reduce_kernel":
        default = (256, 4, 1)
        candidates = tuple(itertools.product((128, 256), (4, 8), (1,)))
        arguments = (
            RuntimeArgument("workspace", "wgrad_nd_partial", None, None),
            RuntimeArgument("tensor", "dw", request.filter.uid, None),
        )
        grid = lambda block: (  # noqa: E731
            (total_weights + block - 1) // block,
            1,
            1,
        )
    else:
        raise ValueError("unknown ND packed Wgrad stage function")
    selected = candidates if request.autotune else (default,)
    variants = tuple(
        KernelVariant(
            variant_id=(
                f"block-{block_size}-warps-{num_warps}-" f"stages-{num_stages}"
            ),
            source=_CONVOLUTION_SOURCE_RELATIVE_PATH,
            source_sha256=source_sha256,
            function=function,
            full_signature=_nd_packed_wgrad_signature(
                request, function=function, block_size=block_size
            ),
            grid=grid(block_size),
            num_warps=num_warps,
            num_stages=num_stages,
            arguments=arguments,
        )
        for block_size, num_warps, num_stages in selected
    )
    return KernelStage(
        id=stage_id,
        node_id=request.node_id,
        operation=request.operation,
        dependencies=dependencies,
        source=_CONVOLUTION_SOURCE_RELATIVE_PATH,
        function=function,
        variants=variants,
        autotune=AutotuneSpec(
            enabled=request.autotune,
            warmup=warmup,
            repetitions=repetitions,
            selection_cache=f"tuning/stage-{stage_id}.json",
        ),
    )


def _nd_packed_wgrad_plan(
    request: ParsedConvolutionRequest,
    source_sha256: str,
) -> ExecutionPlan:
    stage_specs = (
        ("_conv_wgrad_nd_im2row_kernel", ()),
        ("_conv_wgrad_nd_rowmajor_kernel", (0,)),
        ("_conv_wgrad_nd_reduce_kernel", (1,)),
    )
    stages = tuple(
        _nd_packed_wgrad_stage(
            request,
            source_sha256,
            stage_id=stage_id,
            function=function,
            dependencies=dependencies,
        )
        for stage_id, (function, dependencies) in enumerate(stage_specs)
    )
    return ExecutionPlan(
        stages=stages,
        external_binding_uids=request.external_binding_uids,
        workspace_size=_nd_packed_wgrad_workspace_size(request),
        workspace_alignment=_WORKSPACE_ALIGNMENT,
    )


def _uses_standard_wgrad(request: ParsedConvolutionRequest) -> bool:
    if request.input_precision == 1:
        return False
    shape_key = (
        request.image.dimensions,
        request.result.dimensions,
        request.filter.dimensions,
        request.stride,
        request.pre_padding,
        request.post_padding,
    )
    supported_shapes = {
        (
            (8, 64, 56, 56),
            (8, 128, 28, 28),
            (128, 64, 3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
        ),
        (
            (8, 32, 32, 32),
            (8, 64, 32, 32),
            (64, 32, 3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
        ),
        (
            (8, 64, 28, 28),
            (8, 128, 28, 28),
            (128, 64, 1, 1),
            (1, 1),
            (0, 0),
            (0, 0),
        ),
    }
    return (
        request.operation == "convolution_wgrad"
        and request.image.data_type in {"float32", "float16", "bfloat16"}
        and request.result.data_type == request.image.data_type
        and request.filter.data_type == request.image.data_type
        and shape_key in supported_shapes
        and is_row_major_contiguous(request.image)
        and is_row_major_contiguous(request.result)
        and is_row_major_contiguous(request.filter)
        and request.spatial_rank == 2
        and request.groups == 1
        and request.convolution_mode == 0
        and request.dilation == (1, 1)
    )


def _standard_wgrad_num_splits(request: ParsedConvolutionRequest) -> int:
    return 8 if request.filter.dimensions[2:] == (1, 1) else request.batch


def _standard_wgrad_partial_size(request: ParsedConvolutionRequest) -> int:
    kh, kw = request.filter.dimensions[2:]
    raw_size = (
        _standard_wgrad_num_splits(request)
        * request.out_per_group
        * request.in_per_group
        * kh
        * kw
        * 4
    )
    return (
        (raw_size + _WORKSPACE_ALIGNMENT - 1)
        // _WORKSPACE_ALIGNMENT
        * _WORKSPACE_ALIGNMENT
    )


def _standard_wgrad_column_offset(
    request: ParsedConvolutionRequest,
) -> int:
    element_size = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
    }[request.image.data_type]
    return _standard_wgrad_partial_size(request) // element_size


def _standard_wgrad_workspace_size(request: ParsedConvolutionRequest) -> int:
    kh, kw = request.filter.dimensions[2:]
    partial_size = _standard_wgrad_partial_size(request)
    if (kh, kw) == (1, 1):
        raw_size = partial_size
    else:
        element_size = {
            "float32": 4,
            "float16": 2,
            "bfloat16": 2,
        }[request.image.data_type]
        _, _, oh, ow = request.result.dimensions
        column_size = (
            request.batch
            * oh
            * ow
            * request.in_per_group
            * kh
            * kw
            * element_size
        )
        raw_size = partial_size + column_size
    aligned_size = (
        (raw_size + _WORKSPACE_ALIGNMENT - 1)
        // _WORKSPACE_ALIGNMENT
        * _WORKSPACE_ALIGNMENT
    )
    return max(_WORKSPACE_SIZE, aligned_size)


def _standard_wgrad_signature(
    request: ParsedConvolutionRequest,
    *,
    function: str,
    block_size: int,
) -> str:
    _, _, xh, xw = request.image.dimensions
    _, _, oh, ow = request.result.dimensions
    _, _, kh, kw = request.filter.dimensions
    total_rows = request.batch * oh * ow
    num_splits = _standard_wgrad_num_splits(request)
    rows_per_split = (total_rows + num_splits - 1) // num_splits
    partial_stride_oc = request.in_per_group * kh * kw
    partial_stride_split = request.out_per_group * partial_stride_oc
    dtype_id = {
        "float32": 0,
        "float16": 1,
        "bfloat16": 2,
    }[request.image.data_type]
    partial = _pointer_token("float32", _WORKSPACE_ALIGNMENT)
    packed = _pointer_token(request.image.data_type, _WORKSPACE_ALIGNMENT)
    pointers = [
        _pointer_token(request.result.data_type, request.result.alignment),
        _pointer_token(request.image.data_type, request.image.alignment),
        partial,
    ]
    if function == "_conv_wgrad2d_im2row_kernel":
        tokens = [
            _pointer_token(request.image.data_type, request.image.alignment),
            packed,
            str(oh * ow),
            str(xh),
            str(xw),
            str(ow),
            str(request.in_per_group),
            str(kh),
            str(kw),
            *(str(value) for value in request.stride),
            *(str(value) for value in request.pre_padding),
            *(str(value) for value in request.dilation),
            *(str(value) for value in request.image.strides),
            str(partial_stride_oc),
            str(_standard_wgrad_column_offset(request)),
            "1",
            str(block_size),
            str(block_size),
        ]
    elif function == "_conv_wgrad2d_rowmajor_kernel":
        tokens = [
            _pointer_token(request.result.data_type, request.result.alignment),
            packed,
            partial,
            str(oh * ow),
            str(request.out_per_group),
            str(request.in_per_group),
            str(kh),
            str(kw),
            *(str(value) for value in request.result.strides),
            str(_standard_wgrad_column_offset(request)),
            str(partial_stride_oc),
            "1",
            str(partial_stride_split),
            str(partial_stride_oc),
            "1",
            str(dtype_id),
            str(block_size),
            str(block_size),
            str(block_size),
        ]
    elif function == "_conv_wgrad2d_direct_split_kernel":
        tokens = [
            *pointers,
            str(total_rows),
            str(rows_per_split),
            str(oh * ow),
            str(xh),
            str(xw),
            str(ow),
            str(request.out_per_group),
            str(request.in_per_group),
            str(kh),
            str(kw),
            *(str(value) for value in request.stride),
            *(str(value) for value in request.pre_padding),
            *(str(value) for value in request.dilation),
            *(str(value) for value in request.result.strides),
            *(str(value) for value in request.image.strides),
            str(partial_stride_split),
            str(partial_stride_oc),
            "1",
            str(dtype_id),
            str(block_size),
            str(block_size),
            "64",
        ]
    elif function == "_conv_wgrad2d_1x1_split_kernel":
        tokens = [
            *pointers,
            str(total_rows),
            str(rows_per_split),
            str(oh * ow),
            str(request.image.dimensions[1]),
            str(request.result.dimensions[1]),
            str(request.in_per_group),
            str(request.out_per_group),
            str(request.groups),
            str(partial_stride_split),
            str(partial_stride_oc),
            "1",
            str(dtype_id),
            str(block_size),
            str(block_size),
            "64",
        ]
    elif function == "_conv_wgrad2d_stem_reduce_kernel":
        tokens = [
            partial,
            _pointer_token(request.filter.data_type, request.filter.alignment),
            str(request.out_per_group * partial_stride_oc),
            str(partial_stride_oc),
            str(request.in_per_group),
            str(kh),
            str(kw),
            str(num_splits),
            str(partial_stride_split),
            str(partial_stride_oc),
            "1",
            *(str(value) for value in request.filter.strides),
            str(block_size),
        ]
    else:
        raise ValueError("unknown standard Wgrad stage function")
    return ",".join(tokens)


def _standard_wgrad_stage(
    request: ParsedConvolutionRequest,
    source_sha256: str,
    *,
    stage_id: int,
    function: str,
    dependencies: tuple[int, ...],
) -> KernelStage:
    target_warp = int(request.target.rsplit("-w", 1)[1])
    _, _, warmup, repetitions = _load_tuning(
        table_name="convolution",
        warp_size=target_warp,
    )
    num_splits = _standard_wgrad_num_splits(request)
    kh, kw = request.filter.dimensions[2:]
    if function == "_conv_wgrad2d_im2row_kernel":
        default = (64, 4, 1)
        candidates = tuple(itertools.product((32, 64), (4, 8), (1,)))
        arguments = (
            RuntimeArgument("tensor", "x", request.image.uid, None),
            RuntimeArgument("workspace", "wgrad_columns", None, None),
        )
        cik = request.in_per_group * kh * kw
        _, _, oh, ow = request.result.dimensions
        grid = lambda block: (  # noqa: E731
            ((oh * ow + block - 1) // block) * ((cik + block - 1) // block),
            request.batch,
            1,
        )
    elif function == "_conv_wgrad2d_rowmajor_kernel":
        default = (64, 8, 1)
        candidates = tuple(itertools.product((32, 64), (4, 8), (1, 2)))
        arguments = (
            RuntimeArgument("tensor", "dy", request.result.uid, None),
            RuntimeArgument("workspace", "wgrad_columns", None, None),
            RuntimeArgument("workspace", "wgrad_partial", None, None),
        )
        cik = request.in_per_group * kh * kw
        grid = lambda block: (  # noqa: E731
            ((request.out_per_group + block - 1) // block)
            * ((cik + block - 1) // block),
            request.batch,
            1,
        )
    elif function == "_conv_wgrad2d_direct_split_kernel":
        default = (64, 8, 1)
        candidates = tuple(itertools.product((32, 64), (4, 8), (1, 2)))
        arguments = (
            RuntimeArgument("tensor", "dy", request.result.uid, None),
            RuntimeArgument("tensor", "x", request.image.uid, None),
            RuntimeArgument("workspace", "wgrad_partial", None, None),
        )
        cik = request.in_per_group * kh * kw
        grid = lambda block: (  # noqa: E731
            ((request.out_per_group + block - 1) // block)
            * ((cik + block - 1) // block),
            num_splits,
            1,
        )
    elif function == "_conv_wgrad2d_1x1_split_kernel":
        default = (16, 4, 2)
        candidates = tuple(itertools.product((16, 32), (4, 8), (1, 2)))
        arguments = (
            RuntimeArgument("tensor", "dy", request.result.uid, None),
            RuntimeArgument("tensor", "x", request.image.uid, None),
            RuntimeArgument("workspace", "wgrad_partial", None, None),
        )
        grid = lambda block: (  # noqa: E731
            ((request.out_per_group + block - 1) // block)
            * ((request.in_per_group + block - 1) // block),
            num_splits * request.groups,
            1,
        )
    elif function == "_conv_wgrad2d_stem_reduce_kernel":
        default = (256, 4, 1)
        candidates = tuple(itertools.product((128, 256), (4, 8), (1,)))
        arguments = (
            RuntimeArgument("workspace", "wgrad_partial", None, None),
            RuntimeArgument("tensor", "dw", request.filter.uid, None),
        )
        total = request.out_per_group * request.in_per_group * kh * kw
        grid = lambda block: (  # noqa: E731
            (total + block - 1) // block,
            1,
            1,
        )
    else:
        raise ValueError("unknown standard Wgrad stage function")
    selected = candidates if request.autotune else (default,)
    variants = tuple(
        KernelVariant(
            variant_id=(
                f"block-{block_size}-warps-{num_warps}-" f"stages-{num_stages}"
            ),
            source=_CONVOLUTION_SOURCE_RELATIVE_PATH,
            source_sha256=source_sha256,
            function=function,
            full_signature=_standard_wgrad_signature(
                request, function=function, block_size=block_size
            ),
            grid=grid(block_size),
            num_warps=num_warps,
            num_stages=num_stages,
            arguments=arguments,
        )
        for block_size, num_warps, num_stages in selected
    )
    return KernelStage(
        id=stage_id,
        node_id=request.node_id,
        operation=request.operation,
        dependencies=dependencies,
        source=_CONVOLUTION_SOURCE_RELATIVE_PATH,
        function=function,
        variants=variants,
        autotune=AutotuneSpec(
            enabled=request.autotune,
            warmup=warmup,
            repetitions=repetitions,
            selection_cache=f"tuning/stage-{stage_id}.json",
        ),
    )


def _standard_wgrad_plan(
    request: ParsedConvolutionRequest,
    source_sha256: str,
) -> ExecutionPlan:
    if request.filter.dimensions[2:] == (1, 1):
        stage_specs = (
            ("_conv_wgrad2d_1x1_split_kernel", ()),
            ("_conv_wgrad2d_stem_reduce_kernel", (0,)),
        )
    else:
        stage_specs = (
            ("_conv_wgrad2d_im2row_kernel", ()),
            ("_conv_wgrad2d_rowmajor_kernel", (0,)),
            ("_conv_wgrad2d_stem_reduce_kernel", (1,)),
        )
    stages = tuple(
        _standard_wgrad_stage(
            request,
            source_sha256,
            stage_id=stage_id,
            function=function,
            dependencies=dependencies,
        )
        for stage_id, (function, dependencies) in enumerate(stage_specs)
    )
    return ExecutionPlan(
        stages=stages,
        external_binding_uids=request.external_binding_uids,
        workspace_size=_standard_wgrad_workspace_size(request),
        workspace_alignment=_WORKSPACE_ALIGNMENT,
    )


def _uses_stem_wgrad(request: ParsedConvolutionRequest) -> bool:
    if request.input_precision == 1:
        return False
    return (
        request.operation == "convolution_wgrad"
        and request.image.data_type in {"float32", "float16", "bfloat16"}
        and request.result.data_type == request.image.data_type
        and request.filter.data_type == request.image.data_type
        and request.image.dimensions == (1, 3, 640, 640)
        and request.result.dimensions[0] == 1
        and request.result.dimensions[1] in {16, 32, 64, 96}
        and request.result.dimensions[2:] == (320, 320)
        and request.filter.dimensions
        == (request.result.dimensions[1], 3, 3, 3)
        and is_row_major_contiguous(request.image)
        and is_row_major_contiguous(request.result)
        and is_row_major_contiguous(request.filter)
        and request.spatial_rank == 2
        and request.groups == 1
        and request.convolution_mode == 0
        and request.stride == (2, 2)
        and request.pre_padding == (1, 1)
        and request.post_padding == (1, 1)
        and request.dilation == (1, 1)
    )


def _stem_wgrad_workspace_size(request: ParsedConvolutionRequest) -> int:
    raw_size = 64 * request.out_per_group * request.in_per_group * 9 * 4
    aligned_size = (
        (raw_size + _WORKSPACE_ALIGNMENT - 1)
        // _WORKSPACE_ALIGNMENT
        * _WORKSPACE_ALIGNMENT
    )
    return max(_WORKSPACE_SIZE, aligned_size)


def _stem_wgrad_signature(
    request: ParsedConvolutionRequest,
    *,
    function: str,
    block_size: int,
) -> str:
    dtype_id = {
        "float32": 0,
        "float16": 1,
        "bfloat16": 2,
    }[request.image.data_type]
    partial_stride_split = request.out_per_group * request.in_per_group * 9
    partial_stride_oc = request.in_per_group * 9
    partial = _pointer_token("float32", _WORKSPACE_ALIGNMENT)
    if function == "_conv_wgrad2d_stem_split_kernel":
        tokens = [
            _pointer_token(request.result.data_type, request.result.alignment),
            _pointer_token(request.image.data_type, request.image.alignment),
            partial,
            "5",
            "320",
            "320",
            "640",
            "640",
            str(request.out_per_group),
            str(request.in_per_group),
            "3",
            "3",
            "2",
            "2",
            "1",
            "1",
            "1",
            "1",
            str(request.result.strides[1]),
            str(request.result.strides[2]),
            str(request.result.strides[3]),
            str(request.image.strides[1]),
            str(request.image.strides[2]),
            str(request.image.strides[3]),
            str(partial_stride_split),
            str(partial_stride_oc),
            "1",
            str(dtype_id),
            str(block_size),
            "32",
            "64",
        ]
    elif function == "_conv_wgrad2d_stem_reduce_kernel":
        tokens = [
            partial,
            _pointer_token(request.filter.data_type, request.filter.alignment),
            str(request.out_per_group * request.in_per_group * 9),
            str(request.in_per_group * 9),
            str(request.in_per_group),
            "3",
            "3",
            "64",
            str(partial_stride_split),
            str(partial_stride_oc),
            "1",
            str(request.filter.strides[0]),
            str(request.filter.strides[1]),
            str(request.filter.strides[2]),
            str(request.filter.strides[3]),
            str(block_size),
        ]
    else:
        raise ValueError("unknown stem Wgrad stage function")
    return ",".join(tokens)


def _stem_wgrad_stage(
    request: ParsedConvolutionRequest,
    source_sha256: str,
    *,
    stage_id: int,
    function: str,
    dependencies: tuple[int, ...],
) -> KernelStage:
    target_warp = int(request.target.rsplit("-w", 1)[1])
    _, _, warmup, repetitions = _load_tuning(
        table_name="convolution",
        warp_size=target_warp,
    )
    if function == "_conv_wgrad2d_stem_split_kernel":
        default = (64, 4, 2)
        candidates = tuple(itertools.product((32, 64), (4, 8), (1, 2)))
        arguments = (
            RuntimeArgument("tensor", "dy", request.result.uid, None),
            RuntimeArgument("tensor", "x", request.image.uid, None),
            RuntimeArgument("workspace", "wgrad_partial", None, None),
        )
        grid = lambda block: (  # noqa: E731
            (request.out_per_group + block - 1) // block,
            64,
            1,
        )
    elif function == "_conv_wgrad2d_stem_reduce_kernel":
        default = (256, 4, 1)
        candidates = tuple(itertools.product((128, 256), (4, 8), (1,)))
        arguments = (
            RuntimeArgument("workspace", "wgrad_partial", None, None),
            RuntimeArgument("tensor", "dw", request.filter.uid, None),
        )
        total = request.out_per_group * request.in_per_group * 9
        grid = lambda block: (  # noqa: E731
            (total + block - 1) // block,
            1,
            1,
        )
    else:
        raise ValueError("unknown stem Wgrad stage function")
    selected = candidates if request.autotune else (default,)
    variants = tuple(
        KernelVariant(
            variant_id=(
                f"block-{block_size}-warps-{num_warps}-" f"stages-{num_stages}"
            ),
            source=_CONVOLUTION_SOURCE_RELATIVE_PATH,
            source_sha256=source_sha256,
            function=function,
            full_signature=_stem_wgrad_signature(
                request,
                function=function,
                block_size=block_size,
            ),
            grid=grid(block_size),
            num_warps=num_warps,
            num_stages=num_stages,
            arguments=arguments,
        )
        for block_size, num_warps, num_stages in selected
    )
    return KernelStage(
        id=stage_id,
        node_id=request.node_id,
        operation=request.operation,
        dependencies=dependencies,
        source=_CONVOLUTION_SOURCE_RELATIVE_PATH,
        function=function,
        variants=variants,
        autotune=AutotuneSpec(
            enabled=request.autotune,
            warmup=warmup,
            repetitions=repetitions,
            selection_cache=f"tuning/stage-{stage_id}.json",
        ),
    )


def _stem_wgrad_plan(
    request: ParsedConvolutionRequest,
    source_sha256: str,
) -> ExecutionPlan:
    split = _stem_wgrad_stage(
        request,
        source_sha256,
        stage_id=0,
        function="_conv_wgrad2d_stem_split_kernel",
        dependencies=(),
    )
    reduce = _stem_wgrad_stage(
        request,
        source_sha256,
        stage_id=1,
        function="_conv_wgrad2d_stem_reduce_kernel",
        dependencies=(0,),
    )
    return ExecutionPlan(
        stages=(split, reduce),
        external_binding_uids=request.external_binding_uids,
        workspace_size=_stem_wgrad_workspace_size(request),
        workspace_alignment=_WORKSPACE_ALIGNMENT,
    )


def _uses_p5_wgrad(request: ParsedConvolutionRequest) -> bool:
    if request.input_precision == 1:
        return False
    return (
        request.operation == "convolution_wgrad"
        and request.image.data_type in {"float32", "float16", "bfloat16"}
        and request.result.data_type == request.image.data_type
        and request.filter.data_type == request.image.data_type
        and request.image.dimensions[0] == 1
        and request.image.dimensions[2:] == (40, 40)
        and request.result.dimensions[0] == 1
        and request.result.dimensions[2:] == (20, 20)
        and request.filter.dimensions[2:] == (3, 3)
        and is_row_major_contiguous(request.image)
        and is_row_major_contiguous(request.result)
        and is_row_major_contiguous(request.filter)
        and request.spatial_rank == 2
        and request.groups == 1
        and request.convolution_mode == 0
        and request.stride == (2, 2)
        and request.pre_padding == (1, 1)
        and request.post_padding == (1, 1)
        and request.dilation == (1, 1)
    )


def _p5_wgrad_workspace_size(request: ParsedConvolutionRequest) -> int:
    element_size = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
    }[request.image.data_type]
    raw_size = 400 * request.in_per_group * 9 * element_size
    aligned_size = (
        (raw_size + _WORKSPACE_ALIGNMENT - 1)
        // _WORKSPACE_ALIGNMENT
        * _WORKSPACE_ALIGNMENT
    )
    return max(_WORKSPACE_SIZE, aligned_size)


def _p5_wgrad_signature(
    request: ParsedConvolutionRequest,
    *,
    function: str,
    block_size: int,
) -> str:
    dtype_id = {
        "float32": 0,
        "float16": 1,
        "bfloat16": 2,
    }[request.image.data_type]
    workspace = _pointer_token(request.image.data_type, _WORKSPACE_ALIGNMENT)
    if function == "_conv_wgrad2d_p5_pack_image_kernel":
        tokens = [
            _pointer_token(request.image.data_type, request.image.alignment),
            workspace,
            str(request.in_per_group),
            str(request.image.strides[1]),
            str(request.image.strides[2]),
            str(request.image.strides[3]),
            "400",
            str(request.in_per_group * 9),
            str(block_size),
            str(block_size),
            str(block_size),
            "8",
        ]
    elif function == "_conv_wgrad2d_p5_mm_kernel":
        tokens = [
            _pointer_token(request.result.data_type, request.result.alignment),
            workspace,
            _pointer_token(request.filter.data_type, request.filter.alignment),
            str(request.out_per_group),
            str(request.in_per_group * 9),
            "400",
            str(dtype_id),
            str(block_size),
            str(block_size),
            str(block_size),
            "8",
        ]
    else:
        raise ValueError("unknown P5 Wgrad stage function")
    return ",".join(tokens)


def _p5_wgrad_stage(
    request: ParsedConvolutionRequest,
    source_sha256: str,
    *,
    stage_id: int,
    function: str,
    dependencies: tuple[int, ...],
) -> KernelStage:
    target_warp = int(request.target.rsplit("-w", 1)[1])
    default, candidates, warmup, repetitions = _load_tuning(
        table_name="convolution",
        warp_size=target_warp,
    )
    stage_candidates = candidates
    if (
        function == "_conv_wgrad2d_p5_mm_kernel"
        and request.image.data_type != "float32"
    ):
        stage_candidates = (
            *candidates,
            *((64, warps, stages) for warps in (4, 8) for stages in (1, 2)),
        )
    selected = stage_candidates if request.autotune else (default,)
    if function == "_conv_wgrad2d_p5_pack_image_kernel":
        arguments = (
            RuntimeArgument("tensor", "x", request.image.uid, None),
            RuntimeArgument("workspace", "packed_image", None, None),
        )
        grid = lambda block: (  # noqa: E731
            (400 + block - 1) // block,
            (request.in_per_group * 9 + block - 1) // block,
            1,
        )
    elif function == "_conv_wgrad2d_p5_mm_kernel":
        arguments = (
            RuntimeArgument("tensor", "dy", request.result.uid, None),
            RuntimeArgument("workspace", "packed_image", None, None),
            RuntimeArgument("tensor", "dw", request.filter.uid, None),
        )
        grid = lambda block: (  # noqa: E731
            ((request.out_per_group + block - 1) // block)
            * ((request.in_per_group * 9 + block - 1) // block),
            1,
            1,
        )
    else:
        raise ValueError("unknown P5 Wgrad stage function")
    variants = tuple(
        KernelVariant(
            variant_id=(
                f"block-{block_size}-warps-{num_warps}-" f"stages-{num_stages}"
            ),
            source=_CONVOLUTION_SOURCE_RELATIVE_PATH,
            source_sha256=source_sha256,
            function=function,
            full_signature=_p5_wgrad_signature(
                request,
                function=function,
                block_size=block_size,
            ),
            grid=grid(block_size),
            num_warps=num_warps,
            num_stages=num_stages,
            arguments=arguments,
        )
        for block_size, num_warps, num_stages in selected
    )
    return KernelStage(
        id=stage_id,
        node_id=request.node_id,
        operation=request.operation,
        dependencies=dependencies,
        source=_CONVOLUTION_SOURCE_RELATIVE_PATH,
        function=function,
        variants=variants,
        autotune=AutotuneSpec(
            enabled=request.autotune,
            warmup=warmup,
            repetitions=repetitions,
            selection_cache=f"tuning/stage-{stage_id}.json",
        ),
    )


def _p5_wgrad_plan(
    request: ParsedConvolutionRequest,
    source_sha256: str,
) -> ExecutionPlan:
    pack = _p5_wgrad_stage(
        request,
        source_sha256,
        stage_id=0,
        function="_conv_wgrad2d_p5_pack_image_kernel",
        dependencies=(),
    )
    matmul = _p5_wgrad_stage(
        request,
        source_sha256,
        stage_id=1,
        function="_conv_wgrad2d_p5_mm_kernel",
        dependencies=(0,),
    )
    return ExecutionPlan(
        stages=(pack, matmul),
        external_binding_uids=request.external_binding_uids,
        workspace_size=_p5_wgrad_workspace_size(request),
        workspace_alignment=_WORKSPACE_ALIGNMENT,
    )


def _convolution_plan(
    request: ParsedConvolutionRequest,
    source_sha256: str,
) -> ExecutionPlan:
    if _uses_dense_stride2_dgrad(request):
        return _dense_dgrad_plan(request, source_sha256)
    if _uses_im2col_fprop(request):
        return _im2col_fprop_plan(request, source_sha256)
    function = _convolution_function(request)
    if _uses_nd_packed_wgrad(request):
        return _nd_packed_wgrad_plan(request, source_sha256)
    if _uses_standard_wgrad(request):
        return _standard_wgrad_plan(request, source_sha256)
    if _uses_stem_wgrad(request):
        return _stem_wgrad_plan(request, source_sha256)
    if _uses_p5_wgrad(request):
        return _p5_wgrad_plan(request, source_sha256)
    target_warp = int(request.target.rsplit("-w", 1)[1])
    default, candidates, warmup, repetitions = _load_tuning(
        table_name="convolution",
        warp_size=target_warp,
    )
    selected = candidates if request.autotune else (default,)
    arguments = _convolution_runtime_arguments(request)
    variants = tuple(
        KernelVariant(
            variant_id=(
                f"block-{block_size}-warps-{num_warps}-" f"stages-{num_stages}"
            ),
            source=_CONVOLUTION_SOURCE_RELATIVE_PATH,
            source_sha256=source_sha256,
            function=function,
            full_signature=_convolution_full_signature(
                request,
                function=function,
                block_size=block_size,
            ),
            grid=_convolution_grid(
                request,
                function=function,
                block_size=block_size,
            ),
            num_warps=num_warps,
            num_stages=num_stages,
            arguments=arguments,
        )
        for block_size, num_warps, num_stages in selected
    )
    stage = KernelStage(
        id=0,
        node_id=request.node_id,
        operation=request.operation,
        dependencies=(),
        source=_CONVOLUTION_SOURCE_RELATIVE_PATH,
        function=function,
        variants=variants,
        autotune=AutotuneSpec(
            enabled=request.autotune,
            warmup=warmup,
            repetitions=repetitions,
            selection_cache=_SELECTION_CACHE,
        ),
    )
    return ExecutionPlan(
        stages=(stage,),
        external_binding_uids=request.external_binding_uids,
        workspace_size=_WORKSPACE_SIZE,
        workspace_alignment=_WORKSPACE_ALIGNMENT,
    )
