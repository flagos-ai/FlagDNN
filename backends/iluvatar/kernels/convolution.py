# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""N-D convolution kernels refactored from ``flag_dnn.ops.conv*``.

The public compiler ABI uses logical dimensions and explicit tensor strides,
so these kernels do not depend on Torch layouts, packing caches, or Python
dispatch helpers.
"""

import triton
import triton.language as tl


@triton.jit
def conv1d_gemm_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M: tl.constexpr,
    XL: tl.constexpr,
    OL: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    x_stride_n: tl.constexpr,
    x_stride_c: tl.constexpr,
    x_stride_l: tl.constexpr,
    w_stride_o: tl.constexpr,
    w_stride_i: tl.constexpr,
    w_stride_k: tl.constexpr,
    bias_stride: tl.constexpr,
    y_stride_n: tl.constexpr,
    y_stride_c: tl.constexpr,
    y_stride_l: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    tile = tl.program_id(0)
    group = tl.program_id(1).to(tl.int64)
    tiles_oc = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    tile_m = tile // tiles_oc
    tile_oc = tile % tiles_oc
    row_offsets = tl.arange(0, BLOCK_M)
    rows = tile_m * BLOCK_M + row_offsets
    output_channels = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    if OL % BLOCK_M == 0:
        tiles_per_batch: tl.constexpr = OL // BLOCK_M
        batch = tile_m // tiles_per_batch
        output_l = (tile_m - batch * tiles_per_batch) * BLOCK_M + row_offsets
    else:
        batch = rows // OL
        output_l = rows % OL
    reduction_base = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_OC), dtype=tl.float32)
    reduction_extent: tl.constexpr = CIN_PER_GROUP * KW

    for start in tl.static_range(0, reduction_extent, BLOCK_K):
        reduction = start + reduction_base
        input_channel = reduction // KW
        kernel_w = reduction - input_channel * KW
        input_l = output_l[:, None] * STRIDE_W - PAD_LEFT + kernel_w[None, :] * DIL_W
        input_ptrs = (
            x_ptr
            + batch[:, None] * x_stride_n
            + (group * CIN_PER_GROUP + input_channel[None, :]) * x_stride_c
            + input_l * x_stride_l
        )
        spatial_mask = (input_l >= 0) & (input_l < XL)
        if M % BLOCK_M == 0 and reduction_extent % BLOCK_K == 0:
            safe_input_l = tl.maximum(0, tl.minimum(input_l, XL - 1))
            safe_input_ptrs = input_ptrs + (safe_input_l - input_l) * x_stride_l
            input_values = tl.load(safe_input_ptrs)
            input_values = tl.where(spatial_mask, input_values, 0.0)
        elif M % BLOCK_M == 0:
            input_values = tl.load(
                input_ptrs,
                mask=(reduction[None, :] < reduction_extent) & spatial_mask,
                other=0.0,
            )
        elif reduction_extent % BLOCK_K == 0:
            input_values = tl.load(
                input_ptrs,
                mask=(rows[:, None] < M) & spatial_mask,
                other=0.0,
            )
        else:
            input_values = tl.load(
                input_ptrs,
                mask=(rows[:, None] < M)
                & (reduction[None, :] < reduction_extent)
                & spatial_mask,
                other=0.0,
            )
        weight_ptrs = (
            w_ptr
            + (group * COUT_PER_GROUP + output_channels[:, None]) * w_stride_o
            + input_channel[None, :] * w_stride_i
            + kernel_w[None, :] * w_stride_k
        )
        if COUT_PER_GROUP % BLOCK_OC == 0 and reduction_extent % BLOCK_K == 0:
            weights = tl.load(weight_ptrs)
        elif COUT_PER_GROUP % BLOCK_OC == 0:
            weights = tl.load(
                weight_ptrs,
                mask=reduction[None, :] < reduction_extent,
                other=0.0,
            )
        elif reduction_extent % BLOCK_K == 0:
            weights = tl.load(
                weight_ptrs,
                mask=output_channels[:, None] < COUT_PER_GROUP,
                other=0.0,
            )
        else:
            weights = tl.load(
                weight_ptrs,
                mask=(output_channels[:, None] < COUT_PER_GROUP)
                & (reduction[None, :] < reduction_extent),
                other=0.0,
            )
        accumulator += tl.dot(input_values, tl.trans(weights), input_precision="ieee")

    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + (group * COUT_PER_GROUP + output_channels) * bias_stride,
            mask=output_channels < COUT_PER_GROUP,
            other=0.0,
        )
        accumulator += bias[None, :]
    output_ptrs = (
        y_ptr
        + batch[:, None] * y_stride_n
        + (group * COUT_PER_GROUP + output_channels[None, :]) * y_stride_c
        + output_l[:, None] * y_stride_l
    )
    output = accumulator.to(y_ptr.dtype.element_ty)
    if M % BLOCK_M == 0 and COUT_PER_GROUP % BLOCK_OC == 0:
        tl.store(output_ptrs, output)
    elif M % BLOCK_M == 0:
        tl.store(
            output_ptrs,
            output,
            mask=output_channels[None, :] < COUT_PER_GROUP,
        )
    elif COUT_PER_GROUP % BLOCK_OC == 0:
        tl.store(output_ptrs, output, mask=rows[:, None] < M)
    else:
        tl.store(
            output_ptrs,
            output,
            mask=(rows[:, None] < M) & (output_channels[None, :] < COUT_PER_GROUP),
        )


@triton.jit
def _conv2d_1x1_nchw_exact_tile(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    batch,
    group,
    output_start,
    output_channels,
    HW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    APPLY_RELU: tl.constexpr,
    BIAS_STRIDE: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    TILE_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    output_hw = output_start + tl.arange(0, TILE_HW)
    reduction_base = tl.arange(0, BLOCK_K)
    channel_mask = output_channels < COUT_PER_GROUP
    accumulator = tl.zeros((BLOCK_OC, TILE_HW), dtype=tl.float32)

    for start in range(0, CIN_PER_GROUP, BLOCK_K):
        input_channels = start + reduction_base
        reduction_mask = input_channels < CIN_PER_GROUP
        global_input_channels = group * CIN_PER_GROUP + input_channels
        input_ptrs = (
            x_ptr
            + batch * (C_IN * HW)
            + global_input_channels[:, None] * HW
            + output_hw[None, :]
        )
        if CIN_PER_GROUP % BLOCK_K == 0:
            input_values = tl.load(input_ptrs)
        else:
            input_values = tl.load(
                input_ptrs,
                mask=reduction_mask[:, None],
                other=0.0,
            )
        weight_ptrs = (
            w_ptr
            + (group * COUT_PER_GROUP + output_channels[:, None]) * CIN_PER_GROUP
            + input_channels[None, :]
        )
        if COUT_PER_GROUP % BLOCK_OC == 0 and CIN_PER_GROUP % BLOCK_K == 0:
            weights = tl.load(weight_ptrs)
        elif COUT_PER_GROUP % BLOCK_OC == 0:
            weights = tl.load(
                weight_ptrs,
                mask=reduction_mask[None, :],
                other=0.0,
            )
        elif CIN_PER_GROUP % BLOCK_K == 0:
            weights = tl.load(
                weight_ptrs,
                mask=channel_mask[:, None],
                other=0.0,
            )
        else:
            weights = tl.load(
                weight_ptrs,
                mask=channel_mask[:, None] & reduction_mask[None, :],
                other=0.0,
            )
        accumulator += tl.dot(weights, input_values, input_precision="ieee")

    global_output_channels = group * COUT_PER_GROUP + output_channels
    if HAS_BIAS:
        if COUT_PER_GROUP % BLOCK_OC == 0:
            bias = tl.load(bias_ptr + global_output_channels * BIAS_STRIDE)
        else:
            bias = tl.load(
                bias_ptr + global_output_channels * BIAS_STRIDE,
                mask=channel_mask,
                other=0.0,
            )
        accumulator += bias[:, None]
    if APPLY_RELU:
        accumulator = tl.maximum(accumulator, 0.0)
    output_ptrs = (
        y_ptr
        + batch * (C_OUT * HW)
        + global_output_channels[:, None] * HW
        + output_hw[None, :]
    )
    output = accumulator.to(y_ptr.dtype.element_ty)
    if COUT_PER_GROUP % BLOCK_OC == 0:
        tl.store(output_ptrs, output)
    else:
        tl.store(output_ptrs, output, mask=channel_mask[:, None])


@triton.jit
def conv2d_1x1_nchw_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    HW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    SWAP_GRID: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    APPLY_RELU: tl.constexpr,
    BIAS_STRIDE: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    if SWAP_GRID:
        tile = pid1
        batch = pid0.to(tl.int64)
        group = 0
    else:
        tile = pid0
        batch_group = pid1.to(tl.int64)
        batch = batch_group // GROUPS
        group = batch_group - batch * GROUPS

    tiles_oc = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    tile_hw = tile // tiles_oc
    tile_oc = tile - tile_hw * tiles_oc
    output_channels = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    if (
        HW == 784
        and C_IN == 64
        and C_OUT == 128
        and CIN_PER_GROUP == 64
        and COUT_PER_GROUP == 128
        and GROUPS == 1
        and SWAP_GRID
        and HW % BLOCK_HW == 16
    ):
        if tile_hw == 0:
            _conv2d_1x1_nchw_exact_tile(
                x_ptr,
                w_ptr,
                bias_ptr,
                y_ptr,
                batch,
                group,
                HW - 16,
                output_channels,
                HW,
                C_IN,
                C_OUT,
                CIN_PER_GROUP,
                COUT_PER_GROUP,
                HAS_BIAS,
                APPLY_RELU,
                BIAS_STRIDE,
                BLOCK_OC,
                16,
                BLOCK_K,
            )
        else:
            _conv2d_1x1_nchw_exact_tile(
                x_ptr,
                w_ptr,
                bias_ptr,
                y_ptr,
                batch,
                group,
                (tile_hw - 1) * BLOCK_HW,
                output_channels,
                HW,
                C_IN,
                C_OUT,
                CIN_PER_GROUP,
                COUT_PER_GROUP,
                HAS_BIAS,
                APPLY_RELU,
                BIAS_STRIDE,
                BLOCK_OC,
                BLOCK_HW,
                BLOCK_K,
            )
        return

    output_hw = tile_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    reduction_base = tl.arange(0, BLOCK_K)
    output_mask = output_hw < HW
    channel_mask = output_channels < COUT_PER_GROUP
    accumulator = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for start in range(0, CIN_PER_GROUP, BLOCK_K):
        input_channels = start + reduction_base
        reduction_mask = input_channels < CIN_PER_GROUP
        global_input_channels = group * CIN_PER_GROUP + input_channels
        input_ptrs = (
            x_ptr
            + batch * (C_IN * HW)
            + global_input_channels[:, None] * HW
            + output_hw[None, :]
        )
        if CIN_PER_GROUP % BLOCK_K == 0 and HW % BLOCK_HW == 0:
            input_values = tl.load(input_ptrs)
        elif CIN_PER_GROUP % BLOCK_K == 0:
            input_values = tl.load(
                input_ptrs,
                mask=output_mask[None, :],
                other=0.0,
            )
        elif HW % BLOCK_HW == 0:
            input_values = tl.load(
                input_ptrs,
                mask=reduction_mask[:, None],
                other=0.0,
            )
        else:
            input_values = tl.load(
                input_ptrs,
                mask=reduction_mask[:, None] & output_mask[None, :],
                other=0.0,
            )
        weight_ptrs = (
            w_ptr
            + (group * COUT_PER_GROUP + output_channels[:, None]) * CIN_PER_GROUP
            + input_channels[None, :]
        )
        if COUT_PER_GROUP % BLOCK_OC == 0 and CIN_PER_GROUP % BLOCK_K == 0:
            weights = tl.load(weight_ptrs)
        elif COUT_PER_GROUP % BLOCK_OC == 0:
            weights = tl.load(
                weight_ptrs,
                mask=reduction_mask[None, :],
                other=0.0,
            )
        elif CIN_PER_GROUP % BLOCK_K == 0:
            weights = tl.load(
                weight_ptrs,
                mask=channel_mask[:, None],
                other=0.0,
            )
        else:
            weights = tl.load(
                weight_ptrs,
                mask=channel_mask[:, None] & reduction_mask[None, :],
                other=0.0,
            )
        accumulator += tl.dot(weights, input_values, input_precision="ieee")

    global_output_channels = group * COUT_PER_GROUP + output_channels
    if HAS_BIAS:
        if COUT_PER_GROUP % BLOCK_OC == 0:
            bias = tl.load(bias_ptr + global_output_channels * BIAS_STRIDE)
        else:
            bias = tl.load(
                bias_ptr + global_output_channels * BIAS_STRIDE,
                mask=channel_mask,
                other=0.0,
            )
        accumulator += bias[:, None]
    if APPLY_RELU:
        accumulator = tl.maximum(accumulator, 0.0)
    output_ptrs = (
        y_ptr
        + batch * (C_OUT * HW)
        + global_output_channels[:, None] * HW
        + output_hw[None, :]
    )
    output = accumulator.to(y_ptr.dtype.element_ty)
    if COUT_PER_GROUP % BLOCK_OC == 0 and HW % BLOCK_HW == 0:
        tl.store(output_ptrs, output)
    elif COUT_PER_GROUP % BLOCK_OC == 0:
        tl.store(output_ptrs, output, mask=output_mask[None, :])
    elif HW % BLOCK_HW == 0:
        tl.store(output_ptrs, output, mask=channel_mask[:, None])
    else:
        tl.store(
            output_ptrs,
            output,
            mask=channel_mask[:, None] & output_mask[None, :],
        )


@triton.jit
def conv_fprop_2d_im2col_kernel(
    x_ptr,
    columns_ptr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    COLUMNS_STRIDE_N: tl.constexpr,
    COLUMNS_STRIDE_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    output_area: tl.constexpr = OH * OW
    plane = tl.program_id(1).to(tl.int64)
    planes_per_batch: tl.constexpr = CIN_PER_GROUP * KH
    batch = plane // planes_per_batch
    plane_in_batch = plane - batch * planes_per_batch
    input_channel = plane_in_batch // KH
    kernel_h = plane_in_batch - input_channel * KH
    output_hw = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_h = output_hw // OW
    output_w = output_hw - output_h * OW
    input_h = output_h * STRIDE_H - PAD_H + kernel_h * DIL_H
    valid_hw = output_hw < output_area
    valid_h = valid_hw & (input_h >= 0) & (input_h < XH)
    column_base = (
        batch * COLUMNS_STRIDE_N
        + (input_channel * KH * KW + kernel_h * KW) * COLUMNS_STRIDE_K
    )

    for kernel_w in tl.static_range(0, KW):
        input_w = output_w * STRIDE_W - PAD_W + kernel_w * DIL_W
        valid = valid_h & (input_w >= 0) & (input_w < XW)
        values = tl.load(
            x_ptr
            + batch * X_STRIDE_N
            + input_channel * X_STRIDE_C
            + input_h * X_STRIDE_H
            + input_w * X_STRIDE_W,
            mask=valid,
            other=0.0,
        )
        tl.store(
            columns_ptr + column_base + kernel_w * COLUMNS_STRIDE_K + output_hw,
            values,
            mask=valid_hw,
        )


@triton.jit
def conv_fprop_2d_packed_matmul_kernel(
    weight_ptr,
    columns_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    tiles_n = tl.cdiv(N, BLOCK_N)
    tile_m = tile // tiles_n
    tile_n = tile - tile_m * tiles_n
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    reduction = tl.arange(0, BLOCK_K)
    weight_ptrs = weight_ptr + rows[:, None] * K + reduction[None, :]
    column_ptrs = (
        columns_ptr + batch * K * N + reduction[:, None] * N + columns[None, :]
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for reduction_start in tl.range(0, K, BLOCK_K):
        reduction_offsets = reduction_start + reduction
        if M % BLOCK_M == 0 and K % BLOCK_K == 0:
            weights = tl.load(weight_ptrs)
        else:
            weights = tl.load(
                weight_ptrs,
                mask=(rows[:, None] < M) & (reduction_offsets[None, :] < K),
                other=0.0,
            )
        if K % BLOCK_K == 0 and N % BLOCK_N == 0:
            packed = tl.load(column_ptrs)
        else:
            packed = tl.load(
                column_ptrs,
                mask=(reduction_offsets[:, None] < K) & (columns[None, :] < N),
                other=0.0,
            )
        accumulator += tl.dot(weights, packed, input_precision="ieee")
        weight_ptrs += BLOCK_K
        column_ptrs += BLOCK_K * N

    output_ptrs = output_ptr + batch * M * N + rows[:, None] * N + columns[None, :]
    output = accumulator.to(output_ptr.dtype.element_ty)
    if M % BLOCK_M == 0 and N % BLOCK_N == 0:
        tl.store(output_ptrs, output)
    else:
        tl.store(
            output_ptrs,
            output,
            mask=(rows[:, None] < M) & (columns[None, :] < N),
        )


@triton.jit
def conv2d_3x3_nchw_pad1_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    APPLY_RELU: tl.constexpr,
    BIAS_STRIDE: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    tile = tl.program_id(0)
    batch_group = tl.program_id(1).to(tl.int64)
    batch = batch_group // GROUPS
    group = batch_group - batch * GROUPS
    tiles_oc = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    tile_hw = tile // tiles_oc
    tile_oc = tile - tile_hw * tiles_oc

    output_hw = tile_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    output_channels = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    output_h = output_hw // OW
    output_w = output_hw - output_h * OW
    output_mask = output_hw < OH * OW
    channel_mask = output_channels < COUT_PER_GROUP
    reduction_base = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    for kernel_h in tl.static_range(0, 3):
        input_h = output_h * STRIDE_H - 1 + kernel_h
        valid_h = (input_h >= 0) & (input_h < XH)
        for kernel_w in tl.static_range(0, 3):
            input_w = output_w * STRIDE_W - 1 + kernel_w
            input_mask = output_mask & valid_h & (input_w >= 0) & (input_w < XW)
            for input_start in range(0, CIN_PER_GROUP, BLOCK_K):
                input_channels = input_start + reduction_base
                reduction_mask = input_channels < CIN_PER_GROUP
                global_input_channels = group * CIN_PER_GROUP + input_channels
                input_values = tl.load(
                    x_ptr
                    + batch * (C_IN * XH * XW)
                    + global_input_channels[None, :] * (XH * XW)
                    + input_h[:, None] * XW
                    + input_w[:, None],
                    mask=input_mask[:, None] & reduction_mask[None, :],
                    other=0.0,
                )
                weights = tl.load(
                    w_ptr
                    + (
                        (group * COUT_PER_GROUP + output_channels[None, :])
                        * CIN_PER_GROUP
                        + input_channels[:, None]
                    )
                    * 9
                    + kernel_h * 3
                    + kernel_w,
                    mask=reduction_mask[:, None] & channel_mask[None, :],
                    other=0.0,
                )
                accumulator += tl.dot(
                    input_values,
                    weights,
                    input_precision="ieee",
                )

    global_output_channels = group * COUT_PER_GROUP + output_channels
    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + global_output_channels * BIAS_STRIDE,
            mask=channel_mask,
            other=0.0,
        )
        accumulator += bias[None, :]
    if APPLY_RELU:
        accumulator = tl.maximum(accumulator, 0.0)
    tl.store(
        y_ptr
        + batch * (C_OUT * OH * OW)
        + global_output_channels[None, :] * (OH * OW)
        + output_hw[:, None],
        accumulator.to(y_ptr.dtype.element_ty),
        mask=output_mask[:, None] & channel_mask[None, :],
    )


@triton.jit
def conv2d_fp32_ml_stem_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
):
    # Exact FP32 NCHW YOLO-ML stem: 1x3x640x640, K64x3x3x3, stride 2/pad 1.
    tile_hw = tl.program_id(0)
    spatial_offsets = tl.arange(0, 256)
    reduction = tl.arange(0, 32)
    output_channels = tl.arange(0, 64)

    tile_row_group = tile_hw // 5
    tile_column = tile_hw - tile_row_group * 5
    row_in_tile = spatial_offsets // 64
    column_in_tile = spatial_offsets - row_in_tile * 64
    output_h = tile_row_group * 4 + row_in_tile
    output_w = tile_column * 64 + column_in_tile
    output_w = tl.max_contiguous(output_w, 64)
    output_hw = output_h * 320 + output_w
    output_hw = tl.max_contiguous(output_hw, 64)

    input_channel = reduction // 9
    kernel_hw = reduction - input_channel * 9
    kernel_h = kernel_hw // 3
    kernel_w = kernel_hw - kernel_h * 3
    reduction_mask = reduction < 27
    spatial_mask = (
        ((output_h[:, None] > 0) | (kernel_h[None, :] > 0))
        & ((output_w[:, None] > 0) | (kernel_w[None, :] > 0))
    )
    # Factor input NCHW offsets into one spatial term and one reduction term.
    spatial_base = output_h * 1280 + output_w * 2 - 641
    kernel_offset = input_channel * 409600 + kernel_h * 640 + kernel_w
    input_values = tl.load(
        x_ptr + spatial_base[:, None] + kernel_offset[None, :],
        mask=reduction_mask[None, :] & spatial_mask,
        other=0.0,
    )
    weights = tl.load(
        w_ptr + output_channels[None, :] * 27 + reduction[:, None],
        mask=reduction_mask[:, None],
        other=0.0,
    )
    accumulator = tl.dot(input_values, weights, input_precision="ieee")
    tl.store(
        y_ptr + output_channels[None, :] * 102400 + output_hw[:, None],
        accumulator,
    )


@triton.jit
def conv2d_spatial_nchw_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    BATCH: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    APPLY_RELU: tl.constexpr,
    BIAS_STRIDE_C: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    W_STRIDE_K: tl.constexpr,
    W_STRIDE_C: tl.constexpr,
    W_STRIDE_R: tl.constexpr,
    W_STRIDE_S: tl.constexpr,
    Y_STRIDE_N: tl.constexpr,
    Y_STRIDE_C: tl.constexpr,
    Y_STRIDE_H: tl.constexpr,
    Y_STRIDE_W: tl.constexpr,
):
    tile = tl.program_id(0)
    batch_group = tl.program_id(1).to(tl.int64)
    batch = batch_group // GROUPS
    group = batch_group % GROUPS
    tiles_oc = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    tile_hw = tile // tiles_oc
    tile_oc = tile % tiles_oc
    spatial_offsets = tl.arange(0, BLOCK_HW)
    output_channels = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    exact_stem: tl.constexpr = (
        XH == 640
        and XW == 640
        and OH == 320
        and OW == 320
        and C_IN == 3
        and (C_OUT == 64 or C_OUT == 96)
        and BATCH == 1
        and CIN_PER_GROUP == 3
        and COUT_PER_GROUP == C_OUT
        and GROUPS == 1
        and STRIDE_H == 2
        and STRIDE_W == 2
        and PAD_TOP == 1
        and PAD_LEFT == 1
        and DIL_H == 1
        and DIL_W == 1
        and KH == 3
        and KW == 3
        and W_STRIDE_C == 9
        and W_STRIDE_R == 3
        and W_STRIDE_S == 1
        and Y_STRIDE_H == 320
        and Y_STRIDE_W == 1
        and (
            (C_OUT == 64 and BLOCK_OC == 64)
            or (C_OUT == 96 and (BLOCK_OC == 64 or BLOCK_OC == 128))
        )
        and (BLOCK_HW == 64 or BLOCK_HW == 128 or BLOCK_HW == 256)
        and BLOCK_K == 32
    )
    exact_x_stem: tl.constexpr = (
        exact_stem
        and C_OUT == 96
        and BLOCK_OC == 128
        and not HAS_BIAS
        and not APPLY_RELU
    )
    if exact_stem:
        tiles_per_row_group: tl.constexpr = 5
        rows_per_tile: tl.constexpr = BLOCK_HW // 64
        columns_per_row: tl.constexpr = 64
        tile_row_group = tile_hw // tiles_per_row_group
        tile_column = tile_hw - tile_row_group * tiles_per_row_group
        row_in_tile = spatial_offsets // columns_per_row
        column_in_tile = spatial_offsets - row_in_tile * columns_per_row
        output_h = tile_row_group * rows_per_tile + row_in_tile
        output_w = tile_column * columns_per_row + column_in_tile
        output_hw = output_h * OW + output_w
    else:
        output_hw = tile_hw * BLOCK_HW + spatial_offsets
        if OW % BLOCK_HW == 0:
            tiles_per_row: tl.constexpr = OW // BLOCK_HW
            output_h = tile_hw // tiles_per_row
            output_w = (tile_hw - output_h * tiles_per_row) * BLOCK_HW + spatial_offsets
        else:
            output_h = output_hw // OW
            output_w = output_hw % OW
    reduction_extent: tl.constexpr = CIN_PER_GROUP * KH * KW
    reduction_offsets = tl.arange(0, BLOCK_K)
    if exact_x_stem:
        output_channels_64 = tl.arange(0, 64)
        output_channels_32 = 64 + tl.arange(0, 32)
        accumulator_64 = tl.zeros((BLOCK_HW, 64), dtype=tl.float32)
        accumulator_32 = tl.zeros((BLOCK_HW, 32), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    for reduction_start in range(0, reduction_extent, BLOCK_K):
        reduction = reduction_start + reduction_offsets
        input_channel = reduction // (KH * KW)
        kernel_hw = reduction % (KH * KW)
        kernel_h = kernel_hw // KW
        kernel_w = kernel_hw % KW
        input_h = output_h[:, None] * STRIDE_H - PAD_TOP + kernel_h[None, :] * DIL_H
        input_w = output_w[:, None] * STRIDE_W - PAD_LEFT + kernel_w[None, :] * DIL_W
        if exact_stem:
            spatial_mask = (input_h >= 0) & (input_w >= 0)
            input_ptrs = (
                x_ptr
                + input_channel[None, :] * X_STRIDE_C
                + input_h * X_STRIDE_H
                + input_w * X_STRIDE_W
            )
        else:
            spatial_mask = (
                (input_h >= 0) & (input_h < XH) & (input_w >= 0) & (input_w < XW)
            )
            input_ptrs = (
                x_ptr
                + batch[:, None] * X_STRIDE_N
                + (group * CIN_PER_GROUP + input_channel[None, :]) * X_STRIDE_C
                + input_h * X_STRIDE_H
                + input_w * X_STRIDE_W
            )
        if OH * OW % BLOCK_HW == 0 and reduction_extent % BLOCK_K == 0:
            input_values = tl.load(input_ptrs, mask=spatial_mask, other=0.0)
        elif OH * OW % BLOCK_HW == 0:
            input_values = tl.load(
                input_ptrs,
                mask=(reduction[None, :] < reduction_extent) & spatial_mask,
                other=0.0,
            )
        elif reduction_extent % BLOCK_K == 0:
            input_values = tl.load(
                input_ptrs,
                mask=(output_hw[:, None] < OH * OW) & spatial_mask,
                other=0.0,
            )
        else:
            input_values = tl.load(
                input_ptrs,
                mask=(output_hw[:, None] < OH * OW)
                & (reduction[None, :] < reduction_extent)
                & spatial_mask,
                other=0.0,
            )
        if exact_x_stem:
            reduction_mask = reduction < reduction_extent
            weights_64 = tl.load(
                w_ptr + output_channels_64[None, :] * W_STRIDE_K + reduction[:, None],
                mask=reduction_mask[:, None],
                other=0.0,
            )
            weights_32 = tl.load(
                w_ptr + output_channels_32[None, :] * W_STRIDE_K + reduction[:, None],
                mask=reduction_mask[:, None],
                other=0.0,
            )
            accumulator_64 += tl.dot(input_values, weights_64, input_precision="ieee")
            accumulator_32 += tl.dot(input_values, weights_32, input_precision="ieee")
        else:
            if exact_stem:
                weight_ptrs = (
                    w_ptr + output_channels[None, :] * W_STRIDE_K + reduction[:, None]
                )
            else:
                weight_ptrs = (
                    w_ptr
                    + (group * COUT_PER_GROUP + output_channels[None, :]) * W_STRIDE_K
                    + input_channel[:, None] * W_STRIDE_C
                    + kernel_h[:, None] * W_STRIDE_R
                    + kernel_w[:, None] * W_STRIDE_S
                )
            if reduction_extent % BLOCK_K == 0 and COUT_PER_GROUP % BLOCK_OC == 0:
                weights = tl.load(weight_ptrs)
            elif reduction_extent % BLOCK_K == 0:
                weights = tl.load(
                    weight_ptrs,
                    mask=output_channels[None, :] < COUT_PER_GROUP,
                    other=0.0,
                )
            elif COUT_PER_GROUP % BLOCK_OC == 0:
                weights = tl.load(
                    weight_ptrs,
                    mask=reduction[:, None] < reduction_extent,
                    other=0.0,
                )
            else:
                weights = tl.load(
                    weight_ptrs,
                    mask=(reduction[:, None] < reduction_extent)
                    & (output_channels[None, :] < COUT_PER_GROUP),
                    other=0.0,
                )
            accumulator += tl.dot(input_values, weights, input_precision="ieee")

    if exact_x_stem:
        output_ptrs_64 = (
            y_ptr + output_channels_64[None, :] * Y_STRIDE_C + output_hw[:, None]
        )
        output_ptrs_32 = (
            y_ptr + output_channels_32[None, :] * Y_STRIDE_C + output_hw[:, None]
        )
        tl.store(output_ptrs_64, accumulator_64.to(y_ptr.dtype.element_ty))
        tl.store(output_ptrs_32, accumulator_32.to(y_ptr.dtype.element_ty))
    else:
        if HAS_BIAS:
            bias = tl.load(
                bias_ptr + (group * COUT_PER_GROUP + output_channels) * BIAS_STRIDE_C,
                mask=output_channels < COUT_PER_GROUP,
                other=0.0,
            )
            accumulator += bias[None, :]
        if APPLY_RELU:
            accumulator = tl.maximum(accumulator, 0.0)
        if exact_stem:
            output_ptrs = (
                y_ptr + output_channels[None, :] * Y_STRIDE_C + output_hw[:, None]
            )
        else:
            output_ptrs = (
                y_ptr
                + batch * Y_STRIDE_N
                + (group * COUT_PER_GROUP + output_channels[None, :]) * Y_STRIDE_C
                + output_h[:, None] * Y_STRIDE_H
                + output_w[:, None] * Y_STRIDE_W
            )
        output = accumulator.to(y_ptr.dtype.element_ty)
        if OH * OW % BLOCK_HW == 0 and COUT_PER_GROUP % BLOCK_OC == 0:
            tl.store(output_ptrs, output)
        elif OH * OW % BLOCK_HW == 0:
            tl.store(
                output_ptrs,
                output,
                mask=output_channels[None, :] < COUT_PER_GROUP,
            )
        elif COUT_PER_GROUP % BLOCK_OC == 0:
            tl.store(output_ptrs, output, mask=output_hw[:, None] < OH * OW)
        else:
            tl.store(
                output_ptrs,
                output,
                mask=(output_hw[:, None] < OH * OW)
                & (output_channels[None, :] < COUT_PER_GROUP),
            )


@triton.jit
def conv3d_spatial_ncdhw_m_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M: tl.constexpr,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_FRONT: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_D: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    W_STRIDE_K: tl.constexpr,
    W_STRIDE_C: tl.constexpr,
    W_STRIDE_D: tl.constexpr,
    W_STRIDE_H: tl.constexpr,
    W_STRIDE_W: tl.constexpr,
    Y_STRIDE_N: tl.constexpr,
    Y_STRIDE_C: tl.constexpr,
    Y_STRIDE_D: tl.constexpr,
    Y_STRIDE_H: tl.constexpr,
    Y_STRIDE_W: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    tile = tl.program_id(0)
    group = tl.program_id(1).to(tl.int64)
    tiles_oc = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    tile_m = tile // tiles_oc
    tile_oc = tile % tiles_oc
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    output_channels = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    output_volume: tl.constexpr = OD * OH * OW
    batch = rows // output_volume
    spatial = rows % output_volume
    output_d = spatial // (OH * OW)
    output_hw = spatial % (OH * OW)
    output_h = output_hw // OW
    output_w = output_hw % OW
    reduction_base = tl.arange(0, BLOCK_K)
    kernel_volume: tl.constexpr = KD * KH * KW
    reduction_extent: tl.constexpr = CIN_PER_GROUP * kernel_volume
    accumulator = tl.zeros((BLOCK_M, BLOCK_OC), dtype=tl.float32)

    for start in range(0, reduction_extent, BLOCK_K):
        reduction = start + reduction_base
        input_channel = reduction // kernel_volume
        kernel_spatial = reduction - input_channel * kernel_volume
        kernel_d = kernel_spatial // (KH * KW)
        kernel_hw = kernel_spatial - kernel_d * (KH * KW)
        kernel_h = kernel_hw // KW
        kernel_w = kernel_hw - kernel_h * KW
        input_d = output_d[:, None] * STRIDE_D - PAD_FRONT + kernel_d[None, :] * DIL_D
        input_h = output_h[:, None] * STRIDE_H - PAD_TOP + kernel_h[None, :] * DIL_H
        input_w = output_w[:, None] * STRIDE_W - PAD_LEFT + kernel_w[None, :] * DIL_W
        input_values = tl.load(
            x_ptr
            + batch[:, None] * X_STRIDE_N
            + (group * CIN_PER_GROUP + input_channel[None, :]) * X_STRIDE_C
            + input_d * X_STRIDE_D
            + input_h * X_STRIDE_H
            + input_w * X_STRIDE_W,
            mask=(rows[:, None] < M)
            & (reduction[None, :] < reduction_extent)
            & (input_d >= 0)
            & (input_d < XD)
            & (input_h >= 0)
            & (input_h < XH)
            & (input_w >= 0)
            & (input_w < XW),
            other=0.0,
        )
        weights = tl.load(
            w_ptr
            + (group * COUT_PER_GROUP + output_channels[:, None]) * W_STRIDE_K
            + input_channel[None, :] * W_STRIDE_C
            + kernel_d[None, :] * W_STRIDE_D
            + kernel_h[None, :] * W_STRIDE_H
            + kernel_w[None, :] * W_STRIDE_W,
            mask=(output_channels[:, None] < COUT_PER_GROUP)
            & (reduction[None, :] < reduction_extent),
            other=0.0,
        )
        accumulator += tl.dot(input_values, tl.trans(weights), input_precision="ieee")

    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + group * COUT_PER_GROUP + output_channels,
            mask=output_channels < COUT_PER_GROUP,
            other=0.0,
        )
        accumulator += bias[None, :]
    tl.store(
        y_ptr
        + batch[:, None] * Y_STRIDE_N
        + (group * COUT_PER_GROUP + output_channels[None, :]) * Y_STRIDE_C
        + output_d[:, None] * Y_STRIDE_D
        + output_h[:, None] * Y_STRIDE_H
        + output_w[:, None] * Y_STRIDE_W,
        accumulator.to(y_ptr.dtype.element_ty),
        mask=(rows[:, None] < M) & (output_channels[None, :] < COUT_PER_GROUP),
    )


@triton.jit
def conv_dgrad_2d_1x1_kernel(
    dy_ptr,
    w_ptr,
    dx_ptr,
    HW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    tile = tl.program_id(0)
    batch_group = tl.program_id(1).to(tl.int64)
    batch = batch_group // GROUPS
    group = batch_group - batch * GROUPS
    tiles_ci = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    tile_m = tile // tiles_ci
    tile_ci = tile - tile_m * tiles_ci

    spatial = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    input_channels = tile_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = spatial < HW
    mask_ci = input_channels < CIN_PER_GROUP
    accumulator = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for channel_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
        output_channels = channel_start + tl.arange(0, BLOCK_CO)
        mask_co = output_channels < COUT_PER_GROUP
        global_co = group * COUT_PER_GROUP + output_channels
        losses = tl.load(
            dy_ptr + batch * (C_OUT * HW) + global_co[None, :] * HW + spatial[:, None],
            mask=mask_m[:, None] & mask_co[None, :],
            other=0.0,
        )
        weights = tl.load(
            w_ptr + global_co[:, None] * CIN_PER_GROUP + input_channels[None, :],
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        accumulator = tl.dot(
            losses,
            weights,
            accumulator,
            input_precision="ieee",
        )

    global_ci = group * CIN_PER_GROUP + input_channels
    tl.store(
        dx_ptr + batch * (C_IN * HW) + global_ci[None, :] * HW + spatial[:, None],
        accumulator.to(dx_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_ci[None, :],
    )


@triton.jit
def conv_dgrad_2d_p5_splitk_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XW: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    PH: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    num_m_blocks: tl.constexpr = (M + BLOCK_M - 1) // BLOCK_M
    num_k_blocks: tl.constexpr = (COUT_PER_GROUP + BLOCK_CO - 1) // BLOCK_CO
    num_split_k: tl.constexpr = (num_k_blocks + GROUP_K - 1) // GROUP_K
    pid = tl.program_id(0)
    pid_m = pid % num_m_blocks
    pid_tmp = pid // num_m_blocks
    pid_k_group = pid_tmp % num_split_k
    pid_ci = pid_tmp // num_split_k

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    input_channels = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = rows < M
    mask_ci = input_channels < CIN_PER_GROUP

    output_h = rows // LOSS_W
    output_w = rows - output_h * LOSS_W
    input_h = output_h * 2 + PH
    input_w0 = output_w * 2
    loss_base = output_h * loss_stride_h + output_w * loss_stride_w
    valid_h1 = output_h + 1 < LOSS_H
    valid_w1 = output_w + 1 < LOSS_W

    accumulator0 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    accumulator1 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    for k_inner in tl.static_range(0, GROUP_K):
        k_block = pid_k_group * GROUP_K + k_inner
        output_channels = k_block * BLOCK_CO + tl.arange(0, BLOCK_CO)
        mask_co = output_channels < COUT_PER_GROUP
        common_mask = mask_m[:, None] & mask_co[None, :]
        weight_mask = mask_co[:, None] & mask_ci[None, :]
        loss00 = tl.load(
            loss_ptr + output_channels[None, :] * loss_stride_c + loss_base[:, None],
            mask=common_mask,
            other=0.0,
        )
        loss01 = tl.load(
            loss_ptr
            + output_channels[None, :] * loss_stride_c
            + (loss_base + loss_stride_w)[:, None],
            mask=common_mask & valid_w1[:, None],
            other=0.0,
        )
        if PH == 0:
            weight11 = tl.load(
                weight_ptr
                + (
                    ((1 * 3 + 1) * COUT_PER_GROUP + output_channels[:, None])
                    * CIN_PER_GROUP
                )
                + input_channels[None, :],
                mask=weight_mask,
                other=0.0,
            )
            weight12 = tl.load(
                weight_ptr
                + (
                    ((1 * 3 + 2) * COUT_PER_GROUP + output_channels[:, None])
                    * CIN_PER_GROUP
                )
                + input_channels[None, :],
                mask=weight_mask,
                other=0.0,
            )
            weight10 = tl.load(
                weight_ptr
                + (
                    ((1 * 3) * COUT_PER_GROUP + output_channels[:, None])
                    * CIN_PER_GROUP
                )
                + input_channels[None, :],
                mask=weight_mask,
                other=0.0,
            )
            accumulator0 = tl.dot(
                loss00,
                weight11,
                accumulator0,
                input_precision="ieee",
            )
            accumulator1 = tl.dot(
                loss00,
                weight12,
                accumulator1,
                input_precision="ieee",
            )
            accumulator1 = tl.dot(
                loss01,
                weight10,
                accumulator1,
                input_precision="ieee",
            )
        else:
            loss10 = tl.load(
                loss_ptr
                + output_channels[None, :] * loss_stride_c
                + (loss_base + loss_stride_h)[:, None],
                mask=common_mask & valid_h1[:, None],
                other=0.0,
            )
            loss11 = tl.load(
                loss_ptr
                + output_channels[None, :] * loss_stride_c
                + (loss_base + loss_stride_h + loss_stride_w)[:, None],
                mask=common_mask & valid_h1[:, None] & valid_w1[:, None],
                other=0.0,
            )
            weight21 = tl.load(
                weight_ptr
                + (
                    ((2 * 3 + 1) * COUT_PER_GROUP + output_channels[:, None])
                    * CIN_PER_GROUP
                )
                + input_channels[None, :],
                mask=weight_mask,
                other=0.0,
            )
            weight01 = tl.load(
                weight_ptr
                + (
                    ((0 * 3 + 1) * COUT_PER_GROUP + output_channels[:, None])
                    * CIN_PER_GROUP
                )
                + input_channels[None, :],
                mask=weight_mask,
                other=0.0,
            )
            weight22 = tl.load(
                weight_ptr
                + (
                    ((2 * 3 + 2) * COUT_PER_GROUP + output_channels[:, None])
                    * CIN_PER_GROUP
                )
                + input_channels[None, :],
                mask=weight_mask,
                other=0.0,
            )
            weight20 = tl.load(
                weight_ptr
                + (
                    ((2 * 3) * COUT_PER_GROUP + output_channels[:, None])
                    * CIN_PER_GROUP
                )
                + input_channels[None, :],
                mask=weight_mask,
                other=0.0,
            )
            weight02 = tl.load(
                weight_ptr
                + (
                    ((0 * 3 + 2) * COUT_PER_GROUP + output_channels[:, None])
                    * CIN_PER_GROUP
                )
                + input_channels[None, :],
                mask=weight_mask,
                other=0.0,
            )
            weight00 = tl.load(
                weight_ptr
                + output_channels[:, None] * CIN_PER_GROUP
                + input_channels[None, :],
                mask=weight_mask,
                other=0.0,
            )
            accumulator0 = tl.dot(
                loss00,
                weight21,
                accumulator0,
                input_precision="ieee",
            )
            accumulator0 = tl.dot(
                loss10,
                weight01,
                accumulator0,
                input_precision="ieee",
            )
            accumulator1 = tl.dot(
                loss00,
                weight22,
                accumulator1,
                input_precision="ieee",
            )
            accumulator1 = tl.dot(
                loss01,
                weight20,
                accumulator1,
                input_precision="ieee",
            )
            accumulator1 = tl.dot(
                loss10,
                weight02,
                accumulator1,
                input_precision="ieee",
            )
            accumulator1 = tl.dot(
                loss11,
                weight00,
                accumulator1,
                input_precision="ieee",
            )

    output0 = (
        out_ptr
        + input_channels[None, :] * out_stride_c
        + input_h[:, None] * out_stride_h
        + input_w0[:, None] * out_stride_w
    )
    active = mask_m[:, None] & mask_ci[None, :]
    tl.atomic_add(output0, accumulator0, mask=active)
    tl.atomic_add(output0 + out_stride_w, accumulator1, mask=active)


@triton.jit
def conv_dgrad_cast_contiguous_kernel(
    input_ptr,
    output_ptr,
    TOTAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = offsets < TOTAL
    values = tl.load(input_ptr + offsets, mask=active, other=0.0)
    tl.store(
        output_ptr + offsets,
        values.to(output_ptr.dtype.element_ty),
        mask=active,
    )


@triton.jit
def conv_dgrad_pack_weight_3x3_kernel(
    weight_ptr,
    packed_ptr,
    C_OUT: tl.constexpr,
    C_IN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pair_count: tl.constexpr = C_OUT * C_IN
    pair = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    kernel_position = tl.arange(0, 16)
    mask = (pair[:, None] < pair_count) & (kernel_position[None, :] < 9)
    values = tl.load(
        weight_ptr + pair[:, None] * 9 + kernel_position[None, :],
        mask=mask,
        other=0.0,
    )
    tl.store(
        packed_ptr + kernel_position[:, None] * pair_count + pair[None, :],
        tl.trans(values),
        mask=tl.trans(mask),
    )


@triton.jit
def conv_dgrad_2d_packed_parity_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    PARITY_H_COUNT: tl.constexpr,
    PARITY_W_COUNT: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    KH_COUNT: tl.constexpr,
    KW_COUNT: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci < CIN_PER_GROUP

    parity_spatial: tl.constexpr = PARITY_H_COUNT * PARITY_W_COUNT
    batch = offs_m // parity_spatial
    spatial = offs_m - batch * parity_spatial
    loss_h_base = spatial // PARITY_W_COUNT
    loss_w_base = spatial - loss_h_base * PARITY_W_COUNT
    input_h = loss_h_base * 2 + PH
    input_w = loss_w_base * 2 + PW

    accumulator = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    for kh_index in tl.static_range(0, KH_COUNT):
        if PH == 0:
            kernel_h = 1
            loss_h = loss_h_base
        else:
            kernel_h = kh_index * 2
            loss_h = loss_h_base + (1 if kh_index == 0 else 0)
        valid_h = loss_h < LOSS_H
        weight_h = 2 - kernel_h if FILTER_REVERSE else kernel_h

        for kw_index in tl.static_range(0, KW_COUNT):
            if PW == 0:
                kernel_w = 1
                loss_w = loss_w_base
            else:
                kernel_w = kw_index * 2
                loss_w = loss_w_base + (1 if kw_index == 0 else 0)
            valid_w = loss_w < LOSS_W
            weight_w = 2 - kernel_w if FILTER_REVERSE else kernel_w

            for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
                output_channels = co_start + tl.arange(0, BLOCK_CO)
                mask_co = output_channels < COUT_PER_GROUP
                losses = tl.load(
                    loss_ptr
                    + batch[:, None] * loss_stride_n
                    + output_channels[None, :] * loss_stride_c
                    + loss_h[:, None] * loss_stride_h
                    + loss_w[:, None] * loss_stride_w,
                    mask=(
                        mask_m[:, None]
                        & mask_co[None, :]
                        & valid_h[:, None]
                        & valid_w[:, None]
                    ),
                    other=0.0,
                )
                weights = tl.load(
                    weight_ptr
                    + (
                        (
                            (weight_h * 3 + weight_w) * COUT_PER_GROUP
                            + output_channels[:, None]
                        )
                        * CIN_PER_GROUP
                    )
                    + offs_ci[None, :],
                    mask=mask_co[:, None] & mask_ci[None, :],
                    other=0.0,
                )
                accumulator = tl.dot(
                    losses,
                    weights,
                    accumulator,
                    input_precision="ieee",
                )

    tl.store(
        out_ptr
        + batch[:, None] * out_stride_n
        + offs_ci[None, :] * out_stride_c
        + input_h[:, None] * out_stride_h
        + input_w[:, None] * out_stride_w,
        accumulator.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_ci[None, :],
    )


@triton.jit
def conv_dgrad_zero_kernel(
    dx_ptr,
    N_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(dx_ptr + offsets, 0.0, mask=offsets < N_ELEMENTS)


@triton.jit
def conv_dgrad_2d_scatter_kernel(
    dy_ptr,
    w_ptr,
    dx_ptr,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_FRONT: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    FLIP_FILTER: tl.constexpr,
    DY_STRIDE_N: tl.constexpr,
    DY_STRIDE_C: tl.constexpr,
    DY_STRIDE_D: tl.constexpr,
    DY_STRIDE_H: tl.constexpr,
    DY_STRIDE_W: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_D: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    W_STRIDE_K: tl.constexpr,
    W_STRIDE_C: tl.constexpr,
    W_STRIDE_D: tl.constexpr,
    W_STRIDE_H: tl.constexpr,
    W_STRIDE_W: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile = tl.program_id(0)
    group = tl.program_id(1).to(tl.int64)
    contribution_columns: tl.constexpr = CIN_PER_GROUP * KH * KW
    tiles_n = tl.cdiv(contribution_columns, BLOCK_N)
    tile_m = tile // tiles_n
    tile_n = tile % tiles_n
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    batch_size: tl.constexpr = M // (XH * XW)
    output_volume: tl.constexpr = OH * OW
    batch = rows // output_volume
    output_spatial = rows % output_volume
    output_h = output_spatial // OW
    output_w = output_spatial % OW
    input_channel = columns // (KH * KW)
    kernel_spatial = columns % (KH * KW)
    kernel_h = kernel_spatial // KW
    kernel_w = kernel_spatial % KW
    weight_h = KH - 1 - kernel_h if FLIP_FILTER else kernel_h
    weight_w = KW - 1 - kernel_w if FLIP_FILTER else kernel_w
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    reduction_offsets = tl.arange(0, BLOCK_K)

    for start in range(0, COUT_PER_GROUP, BLOCK_K):
        output_channels = start + reduction_offsets
        losses = tl.load(
            dy_ptr
            + batch[:, None] * DY_STRIDE_N
            + (group * COUT_PER_GROUP + output_channels[None, :]) * DY_STRIDE_C
            + output_h[:, None] * DY_STRIDE_H
            + output_w[:, None] * DY_STRIDE_W,
            mask=(rows[:, None] < batch_size * output_volume)
            & (output_channels[None, :] < COUT_PER_GROUP),
            other=0.0,
        )
        weights = tl.load(
            w_ptr
            + (group * COUT_PER_GROUP + output_channels[:, None]) * W_STRIDE_K
            + input_channel[None, :] * W_STRIDE_C
            + weight_h[None, :] * W_STRIDE_H
            + weight_w[None, :] * W_STRIDE_W,
            mask=(output_channels[:, None] < COUT_PER_GROUP)
            & (columns[None, :] < contribution_columns),
            other=0.0,
        )
        accumulator += tl.dot(losses, weights, input_precision="ieee")

    input_h = output_h[:, None] * STRIDE_H - PAD_TOP + kernel_h[None, :] * DIL_H
    input_w = output_w[:, None] * STRIDE_W - PAD_LEFT + kernel_w[None, :] * DIL_W
    active = (
        (rows[:, None] < batch_size * output_volume)
        & (columns[None, :] < contribution_columns)
        & (input_h >= 0)
        & (input_h < XH)
        & (input_w >= 0)
        & (input_w < XW)
    )
    tl.atomic_add(
        dx_ptr
        + batch[:, None] * X_STRIDE_N
        + (group * CIN_PER_GROUP + input_channel[None, :]) * X_STRIDE_C
        + input_h * X_STRIDE_H
        + input_w * X_STRIDE_W,
        accumulator,
        mask=active,
    )


@triton.jit
def conv_dgrad_2d_stride2_kernel(
    dy_ptr,
    w_ptr,
    dx_ptr,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_FRONT: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    FLIP_FILTER: tl.constexpr,
    DY_STRIDE_N: tl.constexpr,
    DY_STRIDE_C: tl.constexpr,
    DY_STRIDE_D: tl.constexpr,
    DY_STRIDE_H: tl.constexpr,
    DY_STRIDE_W: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_D: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    W_STRIDE_K: tl.constexpr,
    W_STRIDE_C: tl.constexpr,
    W_STRIDE_D: tl.constexpr,
    W_STRIDE_H: tl.constexpr,
    W_STRIDE_W: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    M: tl.constexpr,
    PARITY_H: tl.constexpr,
    PARITY_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    tile = tl.program_id(0)
    group = tl.program_id(1).to(tl.int64)
    tiles_ci = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    tile_m = tile // tiles_ci
    tile_ci = tile % tiles_ci
    parity_height: tl.constexpr = (XH + 1 - PARITY_H) // 2
    parity_width: tl.constexpr = (XW + 1 - PARITY_W) // 2
    parity_volume: tl.constexpr = parity_height * parity_width
    batch_size: tl.constexpr = M // (XH * XW)
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    batch = rows // parity_volume
    parity_spatial = rows % parity_volume
    input_h = PARITY_H + 2 * (parity_spatial // parity_width)
    input_w = PARITY_W + 2 * (parity_spatial % parity_width)
    input_channels = tile_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    active_rows = rows < batch_size * parity_volume
    accumulator = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    kernel_h_start: tl.constexpr = 1 - PARITY_H
    kernel_w_start: tl.constexpr = 1 - PARITY_W
    kernel_h_count: tl.constexpr = 1 + PARITY_H
    kernel_w_count: tl.constexpr = 1 + PARITY_W

    kernel_count: tl.constexpr = kernel_h_count * kernel_w_count
    reduction_extent: tl.constexpr = kernel_count * COUT_PER_GROUP
    reduction_base = tl.arange(0, BLOCK_K)

    for start in range(0, reduction_extent, BLOCK_K):
        reduction = start + reduction_base
        kernel_index = reduction // COUT_PER_GROUP
        output_channels = reduction % COUT_PER_GROUP
        kernel_h_index = kernel_index // kernel_w_count
        kernel_w_index = kernel_index % kernel_w_count
        kernel_h = kernel_h_start + 2 * kernel_h_index
        kernel_w = kernel_w_start + 2 * kernel_w_index
        numerator_h = input_h[:, None] + PAD_TOP - kernel_h[None, :] * DIL_H
        numerator_w = input_w[:, None] + PAD_LEFT - kernel_w[None, :] * DIL_W
        output_h = numerator_h // STRIDE_H
        output_w = numerator_w // STRIDE_W
        valid = (
            active_rows[:, None]
            & (reduction[None, :] < reduction_extent)
            & (output_h >= 0)
            & (output_h < OH)
            & (output_w >= 0)
            & (output_w < OW)
        )
        losses = tl.load(
            dy_ptr
            + batch[:, None] * DY_STRIDE_N
            + (group * COUT_PER_GROUP + output_channels[None, :]) * DY_STRIDE_C
            + output_h * DY_STRIDE_H
            + output_w * DY_STRIDE_W,
            mask=valid,
            other=0.0,
        )
        weight_h = KH - 1 - kernel_h if FLIP_FILTER else kernel_h
        weight_w = KW - 1 - kernel_w if FLIP_FILTER else kernel_w
        weights = tl.load(
            w_ptr
            + (group * COUT_PER_GROUP + output_channels[:, None]) * W_STRIDE_K
            + input_channels[None, :] * W_STRIDE_C
            + weight_h[:, None] * W_STRIDE_H
            + weight_w[:, None] * W_STRIDE_W,
            mask=(reduction[:, None] < reduction_extent)
            & (input_channels[None, :] < CIN_PER_GROUP),
            other=0.0,
        )
        accumulator += tl.dot(losses, weights, input_precision="ieee")

    tl.store(
        dx_ptr
        + batch[:, None] * X_STRIDE_N
        + (group * CIN_PER_GROUP + input_channels[None, :]) * X_STRIDE_C
        + input_h[:, None] * X_STRIDE_H
        + input_w[:, None] * X_STRIDE_W,
        accumulator.to(dx_ptr.dtype.element_ty),
        mask=active_rows[:, None] & (input_channels[None, :] < CIN_PER_GROUP),
    )


@triton.jit
def conv_dgrad_nd_kernel(
    dy_ptr,
    w_ptr,
    dx_ptr,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_FRONT: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    FLIP_FILTER: tl.constexpr,
    DY_STRIDE_N: tl.constexpr,
    DY_STRIDE_C: tl.constexpr,
    DY_STRIDE_D: tl.constexpr,
    DY_STRIDE_H: tl.constexpr,
    DY_STRIDE_W: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_D: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    W_STRIDE_K: tl.constexpr,
    W_STRIDE_C: tl.constexpr,
    W_STRIDE_D: tl.constexpr,
    W_STRIDE_H: tl.constexpr,
    W_STRIDE_W: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    tile = tl.program_id(0)
    group = tl.program_id(1).to(tl.int64)
    tiles_ci = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    tile_m = tile // tiles_ci
    tile_ci = tile % tiles_ci
    row_offsets = tl.arange(0, BLOCK_M)
    rows = tile_m * BLOCK_M + row_offsets
    input_channels = tile_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    input_volume: tl.constexpr = XD * XH * XW
    input_plane: tl.constexpr = XH * XW
    if input_volume % BLOCK_M == 0:
        tiles_per_batch: tl.constexpr = input_volume // BLOCK_M
        batch = tile_m // tiles_per_batch
        spatial_tile = tile_m - batch * tiles_per_batch
        if input_plane % BLOCK_M == 0:
            tiles_per_plane: tl.constexpr = input_plane // BLOCK_M
            input_d = spatial_tile // tiles_per_plane
            tile_in_plane = spatial_tile - input_d * tiles_per_plane
            input_hw = tile_in_plane * BLOCK_M + row_offsets
            if BLOCK_M % XW == 0:
                input_h = tile_in_plane * (BLOCK_M // XW) + row_offsets // XW
                input_w = row_offsets % XW
            else:
                input_h = input_hw // XW
                input_w = input_hw % XW
        else:
            spatial = spatial_tile * BLOCK_M + row_offsets
            input_d = spatial // input_plane
            input_hw = spatial % input_plane
            input_h = input_hw // XW
            input_w = input_hw % XW
    else:
        batch = rows // input_volume
        spatial = rows % input_volume
        input_d = spatial // input_plane
        input_hw = spatial % input_plane
        input_h = input_hw // XW
        input_w = input_hw % XW
    kernel_volume: tl.constexpr = KD * KH * KW
    reduction_extent: tl.constexpr = COUT_PER_GROUP * kernel_volume
    reduction_base = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    exact_symmetric_3d: tl.constexpr = (
        XD == 8
        and XH == 16
        and XW == 16
        and OD == 8
        and OH == 16
        and OW == 16
        and KD == 3
        and KH == 3
        and KW == 3
        and CIN_PER_GROUP == 8
        and COUT_PER_GROUP == 16
        and STRIDE_D == 1
        and STRIDE_H == 1
        and STRIDE_W == 1
        and PAD_FRONT == 1
        and PAD_TOP == 1
        and PAD_LEFT == 1
        and DIL_D == 1
        and DIL_H == 1
        and DIL_W == 1
        and not FLIP_FILTER
        and DY_STRIDE_N == 32768
        and DY_STRIDE_C == 2048
        and DY_STRIDE_D == 256
        and DY_STRIDE_H == 16
        and DY_STRIDE_W == 1
        and X_STRIDE_N == 16384
        and X_STRIDE_C == 2048
        and X_STRIDE_D == 256
        and X_STRIDE_H == 16
        and X_STRIDE_W == 1
        and W_STRIDE_K == 216
        and W_STRIDE_C == 27
        and W_STRIDE_D == 9
        and W_STRIDE_H == 3
        and W_STRIDE_W == 1
        and M == 4096
        and BLOCK_K == 32
    )

    for start in range(0, reduction_extent, BLOCK_K):
        reduction = start + reduction_base
        if exact_symmetric_3d:
            output_channel = reduction_base % COUT_PER_GROUP
            kernel_spatial = start // COUT_PER_GROUP + reduction_base // COUT_PER_GROUP
        else:
            output_channel = reduction // kernel_volume
            kernel_spatial = reduction % kernel_volume
        kernel_d = kernel_spatial // (KH * KW)
        kernel_hw = kernel_spatial % (KH * KW)
        kernel_h = kernel_hw // KW
        kernel_w = kernel_hw % KW
        numerator_d = input_d[:, None] + PAD_FRONT - kernel_d[None, :] * DIL_D
        numerator_h = input_h[:, None] + PAD_TOP - kernel_h[None, :] * DIL_H
        numerator_w = input_w[:, None] + PAD_LEFT - kernel_w[None, :] * DIL_W
        output_d = numerator_d // STRIDE_D
        output_h = numerator_h // STRIDE_H
        output_w = numerator_w // STRIDE_W
        valid = (
            (rows[:, None] < M)
            & (reduction[None, :] < reduction_extent)
            & (numerator_d % STRIDE_D == 0)
            & (numerator_h % STRIDE_H == 0)
            & (numerator_w % STRIDE_W == 0)
            & (output_d >= 0)
            & (output_d < OD)
            & (output_h >= 0)
            & (output_h < OH)
            & (output_w >= 0)
            & (output_w < OW)
        )
        losses = tl.load(
            dy_ptr
            + batch[:, None] * DY_STRIDE_N
            + (group * COUT_PER_GROUP + output_channel[None, :]) * DY_STRIDE_C
            + output_d * DY_STRIDE_D
            + output_h * DY_STRIDE_H
            + output_w * DY_STRIDE_W,
            mask=valid,
            other=0.0,
        )
        weight_d = KD - 1 - kernel_d if FLIP_FILTER else kernel_d
        weight_h = KH - 1 - kernel_h if FLIP_FILTER else kernel_h
        weight_w = KW - 1 - kernel_w if FLIP_FILTER else kernel_w
        weights = tl.load(
            w_ptr
            + (group * COUT_PER_GROUP + output_channel[:, None]) * W_STRIDE_K
            + input_channels[None, :] * W_STRIDE_C
            + weight_d[:, None] * W_STRIDE_D
            + weight_h[:, None] * W_STRIDE_H
            + weight_w[:, None] * W_STRIDE_W,
            mask=(reduction[:, None] < reduction_extent)
            & (input_channels[None, :] < CIN_PER_GROUP),
            other=0.0,
        )
        accumulator += tl.dot(losses, weights, input_precision="ieee")

    tl.store(
        dx_ptr
        + batch[:, None] * X_STRIDE_N
        + (group * CIN_PER_GROUP + input_channels[None, :]) * X_STRIDE_C
        + input_d[:, None] * X_STRIDE_D
        + input_h[:, None] * X_STRIDE_H
        + input_w[:, None] * X_STRIDE_W,
        accumulator.to(dx_ptr.dtype.element_ty),
        mask=(rows[:, None] < M) & (input_channels[None, :] < CIN_PER_GROUP),
    )


@triton.jit
def conv_wgrad_2d_p5_pack_kernel(
    image_ptr,
    packed_ptr,
    IMAGE_H: tl.constexpr,
    IMAGE_W: tl.constexpr,
    OUTPUT_H: tl.constexpr,
    OUTPUT_W: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    IMAGE_STRIDE_C: tl.constexpr,
    IMAGE_STRIDE_H: tl.constexpr,
    IMAGE_STRIDE_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile_m = tl.program_id(0)
    tile_n = tl.program_id(1)
    output_area: tl.constexpr = OUTPUT_H * OUTPUT_W
    packed_columns: tl.constexpr = CIN_PER_GROUP * 9
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = rows < output_area
    mask_n = columns < packed_columns

    output_h = rows // OUTPUT_W
    output_w = rows - output_h * OUTPUT_W
    input_channel = columns // 9
    kernel_position = columns - input_channel * 9
    kernel_h = kernel_position // 3
    kernel_w = kernel_position - kernel_h * 3
    safe_input_channel = tl.where(mask_n, input_channel, 0)
    input_h = output_h[:, None] * 2 - 1 + kernel_h[None, :]
    input_w = output_w[:, None] * 2 - 1 + kernel_w[None, :]
    valid = (
        mask_m[:, None]
        & mask_n[None, :]
        & (input_h >= 0)
        & (input_h < IMAGE_H)
        & (input_w >= 0)
        & (input_w < IMAGE_W)
    )
    safe_input_h = tl.where(valid, input_h, 0)
    safe_input_w = tl.where(valid, input_w, 0)
    values = tl.load(
        image_ptr
        + safe_input_channel[None, :] * IMAGE_STRIDE_C
        + safe_input_h * IMAGE_STRIDE_H
        + safe_input_w * IMAGE_STRIDE_W,
        mask=valid,
        other=0.0,
    )
    tl.store(
        packed_ptr + rows[:, None] * packed_columns + columns[None, :],
        values,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def conv_wgrad_2d_p5_matmul_kernel(
    loss_ptr,
    packed_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    tile = tl.program_id(0)
    tiles_m = tl.cdiv(M, BLOCK_M)
    tiles_n = tl.cdiv(N, BLOCK_N)
    tiles_per_group = GROUP_M * tiles_n
    group = tile // tiles_per_group
    first_tile_m = group * GROUP_M
    group_m = tl.minimum(tiles_m - first_tile_m, GROUP_M)
    tile_in_group = tile - group * tiles_per_group
    tile_m = first_tile_m + tile_in_group % group_m
    tile_n = tile_in_group // group_m

    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    reduction = tl.arange(0, BLOCK_K)
    loss_ptrs = loss_ptr + rows[:, None] * K + reduction[None, :]
    packed_ptrs = packed_ptr + reduction[:, None] * N + columns[None, :]
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for reduction_start in tl.range(0, K, BLOCK_K):
        reduction_offsets = reduction_start + reduction
        losses = tl.load(
            loss_ptrs,
            mask=(rows[:, None] < M) & (reduction_offsets[None, :] < K),
            other=0.0,
        )
        packed = tl.load(
            packed_ptrs,
            mask=(reduction_offsets[:, None] < K) & (columns[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(losses, packed, input_precision="ieee")
        loss_ptrs += BLOCK_K
        packed_ptrs += BLOCK_K * N

    tl.store(
        out_ptr + rows[:, None] * N + columns[None, :],
        accumulator.to(out_ptr.dtype.element_ty),
        mask=(rows[:, None] < M) & (columns[None, :] < N),
    )


@triton.jit
def conv_wgrad_2d_pack_image_kernel(
    image_ptr,
    packed_ptr,
    IMAGE_H: tl.constexpr,
    IMAGE_W: tl.constexpr,
    OUTPUT_H: tl.constexpr,
    OUTPUT_W: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    IMAGE_STRIDE_N: tl.constexpr,
    IMAGE_STRIDE_C: tl.constexpr,
    IMAGE_STRIDE_H: tl.constexpr,
    IMAGE_STRIDE_W: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile_m = tl.program_id(0)
    tile_n = tl.program_id(1)
    batch = tl.program_id(2).to(tl.int64)
    output_area: tl.constexpr = OUTPUT_H * OUTPUT_W
    packed_columns: tl.constexpr = CIN_PER_GROUP * 9
    row_offsets = tl.arange(0, BLOCK_M)
    column_offsets = tl.arange(0, BLOCK_N)
    exact_stride2: tl.constexpr = (
        IMAGE_H == 56
        and IMAGE_W == 56
        and OUTPUT_H == 28
        and OUTPUT_W == 28
        and CIN_PER_GROUP == 64
        and IMAGE_STRIDE_N == 200704
        and IMAGE_STRIDE_C == 3136
        and IMAGE_STRIDE_H == 56
        and IMAGE_STRIDE_W == 1
        and STRIDE_H == 2
        and STRIDE_W == 2
        and PAD_H == 1
        and PAD_W == 1
        and DIL_H == 1
        and DIL_W == 1
        and BLOCK_M == 128
        and BLOCK_N == 64
    )
    if exact_stride2:
        row_in_tile = row_offsets // 32
        output_h = tile_m * 4 + row_in_tile
        output_w = row_offsets - row_in_tile * 32
        rows = output_h * OUTPUT_W + output_w
        columns = tile_n * BLOCK_N + column_offsets
        mask_m = (output_h < OUTPUT_H) & (output_w < OUTPUT_W)
        mask_n = columns < packed_columns
        kernel_position = tile_n + column_offsets // BLOCK_N
        input_channel = column_offsets
        kernel_h = kernel_position // 3
        kernel_w = kernel_position - kernel_h * 3
    else:
        rows = tile_m * BLOCK_M + row_offsets
        columns = tile_n * BLOCK_N + column_offsets
        mask_m = rows < output_area
        mask_n = columns < packed_columns
        output_h = rows // OUTPUT_W
        output_w = rows - output_h * OUTPUT_W
        kernel_position = columns // CIN_PER_GROUP
        input_channel = columns - kernel_position * CIN_PER_GROUP
        kernel_h = kernel_position // 3
        kernel_w = kernel_position - kernel_h * 3
    safe_input_channel = tl.where(mask_n, input_channel, 0)
    input_h = output_h[:, None] * STRIDE_H - PAD_H + kernel_h[None, :] * DIL_H
    input_w = output_w[:, None] * STRIDE_W - PAD_W + kernel_w[None, :] * DIL_W
    valid = (
        mask_m[:, None]
        & mask_n[None, :]
        & (input_h >= 0)
        & (input_h < IMAGE_H)
        & (input_w >= 0)
        & (input_w < IMAGE_W)
    )
    safe_input_h = tl.where(valid, input_h, 0)
    safe_input_w = tl.where(valid, input_w, 0)
    values = tl.load(
        image_ptr
        + batch * IMAGE_STRIDE_N
        + safe_input_channel[None, :] * IMAGE_STRIDE_C
        + safe_input_h * IMAGE_STRIDE_H
        + safe_input_w * IMAGE_STRIDE_W,
        mask=valid,
        other=0.0,
    )
    tl.store(
        packed_ptr
        + (batch * output_area + rows[:, None]) * packed_columns
        + columns[None, :],
        values,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def conv_wgrad_2d_batched_matmul_kernel(
    loss_ptr,
    packed_ptr,
    partial_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    LOSS_STRIDE_N: tl.constexpr,
    LOSS_STRIDE_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    tile = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    tiles_m = tl.cdiv(M, BLOCK_M)
    tiles_n = tl.cdiv(N, BLOCK_N)
    tiles_per_group = GROUP_M * tiles_n
    group = tile // tiles_per_group
    first_tile_m = group * GROUP_M
    group_m = tl.minimum(tiles_m - first_tile_m, GROUP_M)
    tile_in_group = tile - group * tiles_per_group
    tile_m = first_tile_m + tile_in_group % group_m
    tile_n = tile_in_group // group_m

    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    reduction = tl.arange(0, BLOCK_K)
    loss_ptrs = (
        loss_ptr
        + batch * LOSS_STRIDE_N
        + rows[:, None] * LOSS_STRIDE_C
        + reduction[None, :]
    )
    packed_ptrs = packed_ptr + batch * K * N + reduction[:, None] * N + columns[None, :]
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for reduction_start in tl.range(0, K, BLOCK_K):
        reduction_offsets = reduction_start + reduction
        losses = tl.load(
            loss_ptrs,
            mask=(rows[:, None] < M) & (reduction_offsets[None, :] < K),
            other=0.0,
        )
        packed = tl.load(
            packed_ptrs,
            mask=(reduction_offsets[:, None] < K) & (columns[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(losses, packed, input_precision="ieee")
        loss_ptrs += BLOCK_K
        packed_ptrs += BLOCK_K * N

    tl.store(
        partial_ptr + batch * M * N + rows[:, None] * N + columns[None, :],
        accumulator,
        mask=(rows[:, None] < M) & (columns[None, :] < N),
    )


@triton.jit
def conv_wgrad_2d_batched_wide_kernel(
    loss_ptr,
    packed_ptr,
    partial_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    LOSS_STRIDE_N: tl.constexpr,
    LOSS_STRIDE_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    tile = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    tiles_m = tl.cdiv(M, BLOCK_M)
    tiles_n: tl.constexpr = 2
    tiles_per_group = GROUP_M * tiles_n
    group = tile // tiles_per_group
    first_tile_m = group * GROUP_M
    group_m = tl.minimum(tiles_m - first_tile_m, GROUP_M)
    tile_in_group = tile - group * tiles_per_group
    tile_m = first_tile_m + tile_in_group % group_m
    tile_n = tile_in_group // group_m

    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    wide_columns = tile_n * (BLOCK_N + BLOCK_N // 8) + tl.arange(0, BLOCK_N)
    narrow_columns = (
        tile_n * (BLOCK_N + BLOCK_N // 8) + BLOCK_N + tl.arange(0, BLOCK_N // 8)
    )
    reduction = tl.arange(0, BLOCK_K)
    loss_ptrs = (
        loss_ptr
        + batch * LOSS_STRIDE_N
        + rows[:, None] * LOSS_STRIDE_C
        + reduction[None, :]
    )
    packed_batch = packed_ptr + batch * K * N
    wide_ptrs = packed_batch + reduction[:, None] * N + wide_columns[None, :]
    narrow_ptrs = packed_batch + reduction[:, None] * N + narrow_columns[None, :]
    accumulator_wide = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    accumulator_narrow = tl.zeros((BLOCK_M, BLOCK_N // 8), dtype=tl.float32)
    for reduction_start in tl.range(0, K, BLOCK_K):
        reduction_offsets = reduction_start + reduction
        active_reduction = reduction_offsets < K
        losses = tl.load(
            loss_ptrs,
            mask=(rows[:, None] < M) & active_reduction[None, :],
            other=0.0,
        )
        packed_wide = tl.load(
            wide_ptrs,
            mask=active_reduction[:, None],
            other=0.0,
        )
        packed_narrow = tl.load(
            narrow_ptrs,
            mask=active_reduction[:, None],
            other=0.0,
        )
        accumulator_wide += tl.dot(
            losses,
            packed_wide,
            input_precision="ieee",
        )
        accumulator_narrow += tl.dot(
            losses,
            packed_narrow,
            input_precision="ieee",
        )
        loss_ptrs += BLOCK_K
        wide_ptrs += BLOCK_K * N
        narrow_ptrs += BLOCK_K * N

    partial_batch = partial_ptr + batch * M * N
    tl.store(
        partial_batch + rows[:, None] * N + wide_columns[None, :],
        accumulator_wide,
        mask=rows[:, None] < M,
    )
    tl.store(
        partial_batch + rows[:, None] * N + narrow_columns[None, :],
        accumulator_narrow,
        mask=rows[:, None] < M,
    )


@triton.jit
def conv_wgrad_2d_batched_reduce_kernel(
    partial_ptr,
    out_ptr,
    OUTPUT_ELEMENTS: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    BATCH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = offsets < OUTPUT_ELEMENTS
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for batch in tl.static_range(0, BATCH):
        accumulator += tl.load(
            partial_ptr + batch * OUTPUT_ELEMENTS + offsets,
            mask=active,
            other=0.0,
        )
    packed_columns: tl.constexpr = CIN_PER_GROUP * 9
    output_channel = offsets // packed_columns
    packed_column = offsets - output_channel * packed_columns
    kernel_position = packed_column // CIN_PER_GROUP
    input_channel = packed_column - kernel_position * CIN_PER_GROUP
    output_offsets = (
        output_channel * CIN_PER_GROUP * 9 + input_channel * 9 + kernel_position
    )
    tl.store(
        out_ptr + output_offsets,
        accumulator.to(out_ptr.dtype.element_ty),
        mask=active & (output_channel < C_OUT),
    )


@triton.jit
def conv_wgrad_2d_col_split_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    M: tl.constexpr,
    IMAGE_H: tl.constexpr,
    IMAGE_W: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    IMAGE_STRIDE_N: tl.constexpr,
    IMAGE_STRIDE_C: tl.constexpr,
    IMAGE_STRIDE_H: tl.constexpr,
    IMAGE_STRIDE_W: tl.constexpr,
    LOSS_STRIDE_N: tl.constexpr,
    LOSS_STRIDE_C: tl.constexpr,
    LOSS_STRIDE_H: tl.constexpr,
    LOSS_STRIDE_W: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SOURCE_FP32: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    tile = tl.program_id(0)
    split = tl.program_id(1).to(tl.int64)
    packed_columns: tl.constexpr = CIN_PER_GROUP * 9
    tiles_n = tl.cdiv(packed_columns, BLOCK_N)
    tile_co = tile // tiles_n
    tile_n = tile - tile_co * tiles_n
    output_channels = tile_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    columns = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    input_channels = columns // 9
    kernel_position = columns - input_channels * 9
    kernel_h = kernel_position // 3
    kernel_w = kernel_position - kernel_h * 3
    image_kernel_h = 2 - kernel_h if FILTER_REVERSE else kernel_h
    image_kernel_w = 2 - kernel_w if FILTER_REVERSE else kernel_w
    mask_co = output_channels < COUT_PER_GROUP
    mask_n = columns < packed_columns

    split_size: tl.constexpr = tl.cdiv(M, NUM_SPLITS)
    split_begin = split * split_size
    split_end = tl.minimum(split_begin + split_size, M)
    accumulator = tl.zeros((BLOCK_CO, BLOCK_N), dtype=tl.float32)
    for row_start in tl.range(split_begin, split_end, BLOCK_M):
        rows = row_start + tl.arange(0, BLOCK_M)
        mask_m = rows < split_end
        safe_rows = tl.where(mask_m, rows, 0)
        output_w = safe_rows % LOSS_W
        quotient = safe_rows // LOSS_W
        output_h = quotient % LOSS_H
        batch = quotient // LOSS_H
        input_h = output_h[:, None] * STRIDE_H - PAD_H + image_kernel_h[None, :] * DIL_H
        input_w = output_w[:, None] * STRIDE_W - PAD_W + image_kernel_w[None, :] * DIL_W
        valid = (
            mask_m[:, None]
            & mask_n[None, :]
            & (input_h >= 0)
            & (input_h < IMAGE_H)
            & (input_w >= 0)
            & (input_w < IMAGE_W)
        )
        safe_h = tl.where(valid, input_h, 0)
        safe_w = tl.where(valid, input_w, 0)
        losses = tl.load(
            loss_ptr
            + batch[None, :] * LOSS_STRIDE_N
            + output_channels[:, None] * LOSS_STRIDE_C
            + output_h[None, :] * LOSS_STRIDE_H
            + output_w[None, :] * LOSS_STRIDE_W,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        images = tl.load(
            image_ptr
            + batch[:, None] * IMAGE_STRIDE_N
            + input_channels[None, :] * IMAGE_STRIDE_C
            + safe_h * IMAGE_STRIDE_H
            + safe_w * IMAGE_STRIDE_W,
            mask=valid,
            other=0.0,
        )
        if SOURCE_FP32:
            losses = losses.to(tl.float16)
            images = images.to(tl.float16)
        accumulator += tl.dot(losses, images, input_precision="ieee")

    tl.store(
        partial_ptr
        + (split * C_OUT + output_channels[:, None]) * packed_columns
        + columns[None, :],
        accumulator,
        mask=mask_co[:, None] & mask_n[None, :],
    )


@triton.jit
def conv_wgrad_2d_col_reduce_kernel(
    partial_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = offsets < TOTAL
    accumulator0 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    accumulator1 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    accumulator2 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    accumulator3 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    accumulator4 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    accumulator5 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    accumulator6 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    accumulator7 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for split in tl.static_range(0, NUM_SPLITS, 8):
        accumulator0 += tl.load(
            partial_ptr + split * TOTAL + offsets,
            mask=active,
            other=0.0,
        )
        accumulator1 += tl.load(
            partial_ptr + (split + 1) * TOTAL + offsets,
            mask=active,
            other=0.0,
        )
        accumulator2 += tl.load(
            partial_ptr + (split + 2) * TOTAL + offsets,
            mask=active,
            other=0.0,
        )
        accumulator3 += tl.load(
            partial_ptr + (split + 3) * TOTAL + offsets,
            mask=active,
            other=0.0,
        )
        accumulator4 += tl.load(
            partial_ptr + (split + 4) * TOTAL + offsets,
            mask=active,
            other=0.0,
        )
        accumulator5 += tl.load(
            partial_ptr + (split + 5) * TOTAL + offsets,
            mask=active,
            other=0.0,
        )
        accumulator6 += tl.load(
            partial_ptr + (split + 6) * TOTAL + offsets,
            mask=active,
            other=0.0,
        )
        accumulator7 += tl.load(
            partial_ptr + (split + 7) * TOTAL + offsets,
            mask=active,
            other=0.0,
        )
    accumulator = (
        (accumulator0 + accumulator1)
        + (accumulator2 + accumulator3)
        + (accumulator4 + accumulator5)
        + (accumulator6 + accumulator7)
    )
    tl.store(
        out_ptr + offsets,
        accumulator.to(out_ptr.dtype.element_ty),
        mask=active,
    )


@triton.jit
def conv_wgrad_1d_3tap_split_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    LENGTH: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    IMAGE_STRIDE_N: tl.constexpr,
    IMAGE_STRIDE_C: tl.constexpr,
    IMAGE_STRIDE_L: tl.constexpr,
    LOSS_STRIDE_N: tl.constexpr,
    LOSS_STRIDE_C: tl.constexpr,
    LOSS_STRIDE_L: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    SPLITS_PER_N: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    tile = tl.program_id(0)
    split = tl.program_id(1)
    kernel_l = tl.program_id(2).to(tl.int64)
    tiles_ci = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    tile_co = tile // tiles_ci
    tile_ci = tile - tile_co * tiles_ci
    output_channels = tile_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    input_channels = tile_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = output_channels < COUT_PER_GROUP
    mask_ci = input_channels < CIN_PER_GROUP

    batch = split // SPLITS_PER_N
    split_in_batch = split - batch * SPLITS_PER_N
    split_size: tl.constexpr = tl.cdiv(LENGTH, SPLITS_PER_N)
    loss_begin = split_in_batch * split_size
    loss_end = tl.minimum(loss_begin + split_size, LENGTH)
    accumulator = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for loss_start in tl.range(loss_begin, loss_end, BLOCK_M):
        loss_l = loss_start + tl.arange(0, BLOCK_M)
        mask_m = loss_l < loss_end
        image_l = loss_l - PAD_LEFT + kernel_l
        valid_image = (image_l >= 0) & (image_l < LENGTH)
        safe_image_l = tl.where(valid_image, image_l, 0)
        losses = tl.load(
            loss_ptr
            + batch * LOSS_STRIDE_N
            + output_channels[:, None] * LOSS_STRIDE_C
            + loss_l[None, :] * LOSS_STRIDE_L,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        images = tl.load(
            image_ptr
            + batch * IMAGE_STRIDE_N
            + input_channels[None, :] * IMAGE_STRIDE_C
            + safe_image_l[:, None] * IMAGE_STRIDE_L,
            mask=mask_m[:, None] & valid_image[:, None] & mask_ci[None, :],
            other=0.0,
        )
        accumulator += tl.dot(losses, images, input_precision="ieee")

    tl.store(
        partial_ptr
        + (
            (split * C_OUT + output_channels[:, None]) * CIN_PER_GROUP
            + input_channels[None, :]
        )
        * 3
        + kernel_l,
        accumulator,
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def conv_wgrad_1d_3tap_reduce_kernel(
    partial_ptr,
    out_ptr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    OUT_STRIDE_O: tl.constexpr,
    OUT_STRIDE_I: tl.constexpr,
    OUT_STRIDE_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    tile = tl.program_id(0)
    kernel_l = tl.program_id(1).to(tl.int64)
    tiles_ci = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    tile_co = tile // tiles_ci
    tile_ci = tile - tile_co * tiles_ci
    output_channels = tile_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    input_channels = tile_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = output_channels < COUT_PER_GROUP
    mask_ci = input_channels < CIN_PER_GROUP
    accumulator = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for split in tl.static_range(0, NUM_SPLITS):
        accumulator += tl.load(
            partial_ptr
            + (
                (split * C_OUT + output_channels[:, None]) * CIN_PER_GROUP
                + input_channels[None, :]
            )
            * 3
            + kernel_l,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )

    tl.store(
        out_ptr
        + output_channels[:, None] * OUT_STRIDE_O
        + input_channels[None, :] * OUT_STRIDE_I
        + kernel_l * OUT_STRIDE_K,
        accumulator.to(out_ptr.dtype.element_ty),
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def conv_wgrad_2d_1x1_split_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    HW: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    IMAGE_STRIDE_N: tl.constexpr,
    IMAGE_STRIDE_C: tl.constexpr,
    LOSS_STRIDE_N: tl.constexpr,
    LOSS_STRIDE_C: tl.constexpr,
    SPLITS_PER_N: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    tile = tl.program_id(0)
    split = tl.program_id(1)
    group = tl.program_id(2).to(tl.int64)
    tiles_ci = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    tile_co = tile // tiles_ci
    tile_ci = tile - tile_co * tiles_ci
    output_channels = tile_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    input_channels = tile_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = output_channels < COUT_PER_GROUP
    mask_ci = input_channels < CIN_PER_GROUP
    global_co = group * COUT_PER_GROUP + output_channels
    global_ci = group * CIN_PER_GROUP + input_channels

    batch = split // SPLITS_PER_N
    split_in_batch = split - batch * SPLITS_PER_N
    split_size: tl.constexpr = tl.cdiv(HW, SPLITS_PER_N)
    spatial_begin = split_in_batch * split_size
    spatial_end = tl.minimum(spatial_begin + split_size, HW)
    accumulator = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for spatial_start in tl.range(spatial_begin, spatial_end, BLOCK_M):
        spatial = spatial_start + tl.arange(0, BLOCK_M)
        mask_m = spatial < spatial_end
        safe_spatial = tl.where(mask_m, spatial, 0)
        losses = tl.load(
            loss_ptr
            + batch * LOSS_STRIDE_N
            + global_co[:, None] * LOSS_STRIDE_C
            + safe_spatial[None, :],
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        images = tl.load(
            image_ptr
            + batch * IMAGE_STRIDE_N
            + global_ci[None, :] * IMAGE_STRIDE_C
            + safe_spatial[:, None],
            mask=mask_m[:, None] & mask_ci[None, :],
            other=0.0,
        )
        accumulator += tl.dot(losses, images, input_precision="ieee")
    tl.store(
        partial_ptr
        + split * C_OUT * CIN_PER_GROUP
        + global_co[:, None] * CIN_PER_GROUP
        + input_channels[None, :],
        accumulator,
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def conv_wgrad_2d_1x1_reduce_kernel(
    partial_ptr,
    out_ptr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    OUT_STRIDE_O: tl.constexpr,
    OUT_STRIDE_I: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    tile = tl.program_id(0)
    group = tl.program_id(1).to(tl.int64)
    tiles_ci = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    tile_co = tile // tiles_ci
    tile_ci = tile - tile_co * tiles_ci
    output_channels = tile_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    input_channels = tile_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = output_channels < COUT_PER_GROUP
    mask_ci = input_channels < CIN_PER_GROUP
    global_co = group * COUT_PER_GROUP + output_channels

    accumulator = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for split in tl.static_range(0, NUM_SPLITS):
        accumulator += tl.load(
            partial_ptr
            + split * C_OUT * CIN_PER_GROUP
            + global_co[:, None] * CIN_PER_GROUP
            + input_channels[None, :],
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
    tl.store(
        out_ptr
        + global_co[:, None] * OUT_STRIDE_O
        + input_channels[None, :] * OUT_STRIDE_I,
        accumulator.to(out_ptr.dtype.element_ty),
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def conv_wgrad_nd_kernel(
    dy_ptr,
    x_ptr,
    dw_ptr,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_FRONT: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    FLIP_FILTER: tl.constexpr,
    DY_STRIDE_N: tl.constexpr,
    DY_STRIDE_C: tl.constexpr,
    DY_STRIDE_D: tl.constexpr,
    DY_STRIDE_H: tl.constexpr,
    DY_STRIDE_W: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_D: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    W_STRIDE_K: tl.constexpr,
    W_STRIDE_C: tl.constexpr,
    W_STRIDE_D: tl.constexpr,
    W_STRIDE_H: tl.constexpr,
    W_STRIDE_W: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    M: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    tile = tl.program_id(0)
    filter_spatial = tl.program_id(1).to(tl.int64)
    group = tl.program_id(2).to(tl.int64)
    tiles_ci = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    tile_oc = tile // tiles_ci
    tile_ci = tile % tiles_ci
    output_channels = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    input_channels = tile_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    kernel_d = filter_spatial // (KH * KW)
    kernel_hw = filter_spatial % (KH * KW)
    kernel_h = kernel_hw // KW
    kernel_w = kernel_hw % KW
    effective_d = KD - 1 - kernel_d if FLIP_FILTER else kernel_d
    effective_h = KH - 1 - kernel_h if FLIP_FILTER else kernel_h
    effective_w = KW - 1 - kernel_w if FLIP_FILTER else kernel_w
    loss_volume: tl.constexpr = OD * OH * OW
    accumulator = tl.zeros((BLOCK_OC, BLOCK_CI), dtype=tl.float32)

    for start in range(0, M, BLOCK_M):
        rows = start + tl.arange(0, BLOCK_M)
        batch = rows // loss_volume
        spatial = rows % loss_volume
        output_d = spatial // (OH * OW)
        output_hw = spatial % (OH * OW)
        output_h = output_hw // OW
        output_w = output_hw % OW
        input_d = output_d * STRIDE_D - PAD_FRONT + effective_d * DIL_D
        input_h = output_h * STRIDE_H - PAD_TOP + effective_h * DIL_H
        input_w = output_w * STRIDE_W - PAD_LEFT + effective_w * DIL_W
        active_rows = (
            (rows < M)
            & (input_d >= 0)
            & (input_d < XD)
            & (input_h >= 0)
            & (input_h < XH)
            & (input_w >= 0)
            & (input_w < XW)
        )
        losses = tl.load(
            dy_ptr
            + (group * COUT_PER_GROUP + output_channels[:, None]) * DY_STRIDE_C
            + batch[None, :] * DY_STRIDE_N
            + output_d[None, :] * DY_STRIDE_D
            + output_h[None, :] * DY_STRIDE_H
            + output_w[None, :] * DY_STRIDE_W,
            mask=(output_channels[:, None] < COUT_PER_GROUP) & active_rows[None, :],
            other=0.0,
        )
        inputs = tl.load(
            x_ptr
            + batch[:, None] * X_STRIDE_N
            + (group * CIN_PER_GROUP + input_channels[None, :]) * X_STRIDE_C
            + input_d[:, None] * X_STRIDE_D
            + input_h[:, None] * X_STRIDE_H
            + input_w[:, None] * X_STRIDE_W,
            mask=active_rows[:, None] & (input_channels[None, :] < CIN_PER_GROUP),
            other=0.0,
        )
        accumulator += tl.dot(losses, inputs, input_precision="ieee")

    tl.store(
        dw_ptr
        + (group * COUT_PER_GROUP + output_channels[:, None]) * W_STRIDE_K
        + input_channels[None, :] * W_STRIDE_C
        + kernel_d * W_STRIDE_D
        + kernel_h * W_STRIDE_H
        + kernel_w * W_STRIDE_W,
        accumulator.to(dw_ptr.dtype.element_ty),
        mask=(output_channels[:, None] < COUT_PER_GROUP)
        & (input_channels[None, :] < CIN_PER_GROUP),
    )
