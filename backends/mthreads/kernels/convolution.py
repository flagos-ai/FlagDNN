# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""MThreads convolution kernels with MTGPU-specific compatibility fixes.

This implementation retains the common kernel's tiled dot-product reduction,
while expressing convolution-mode filter reversal as branch-free integer
coordinate mapping to avoid the failing MTGPU code shape.
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
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    output_channels = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    reduction_base = tl.arange(0, BLOCK_K)
    batch = rows // OL
    output_l = rows % OL
    accumulator = tl.zeros((BLOCK_M, BLOCK_OC), dtype=tl.float32)
    reduction_extent: tl.constexpr = CIN_PER_GROUP * KW

    for start in range(0, reduction_extent, BLOCK_K):
        reduction = start + reduction_base
        input_channel = reduction // KW
        kernel_w = reduction % KW
        input_l = (
            output_l[:, None] * STRIDE_W - PAD_LEFT + kernel_w[None, :] * DIL_W
        )
        input_ptrs = (
            x_ptr
            + batch[:, None] * x_stride_n
            + (group * CIN_PER_GROUP + input_channel[None, :]) * x_stride_c
            + input_l * x_stride_l
        )
        input_values = tl.load(
            input_ptrs,
            mask=(rows[:, None] < M)
            & (reduction[None, :] < reduction_extent)
            & (input_l >= 0)
            & (input_l < XL),
            other=0.0,
        )
        weight_ptrs = (
            w_ptr
            + (group * COUT_PER_GROUP + output_channels[:, None]) * w_stride_o
            + input_channel[None, :] * w_stride_i
            + kernel_w[None, :] * w_stride_k
        )
        weights = tl.load(
            weight_ptrs,
            mask=(output_channels[:, None] < COUT_PER_GROUP)
            & (reduction[None, :] < reduction_extent),
            other=0.0,
        )
        if INPUT_PRECISION == 1:
            accumulator += tl.dot(
                input_values, tl.trans(weights), input_precision="tf32"
            )
        else:
            accumulator += tl.dot(
                input_values, tl.trans(weights), input_precision="ieee"
            )

    if HAS_BIAS:
        bias = tl.load(
            bias_ptr
            + (group * COUT_PER_GROUP + output_channels) * bias_stride,
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
    tl.store(
        output_ptrs,
        accumulator.to(y_ptr.dtype.element_ty),
        mask=(rows[:, None] < M) & (output_channels[None, :] < COUT_PER_GROUP),
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
    output_hw = tile_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    output_channels = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    output_h = output_hw // OW
    output_w = output_hw % OW
    reduction_base = tl.arange(0, BLOCK_K)
    reduction_extent: tl.constexpr = CIN_PER_GROUP * KH * KW
    accumulator = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    for start in range(0, reduction_extent, BLOCK_K):
        reduction = start + reduction_base
        input_channel = reduction // (KH * KW)
        kernel_hw = reduction % (KH * KW)
        kernel_h = kernel_hw // KW
        kernel_w = kernel_hw % KW
        input_h = (
            output_h[:, None] * STRIDE_H - PAD_TOP + kernel_h[None, :] * DIL_H
        )
        input_w = (
            output_w[:, None] * STRIDE_W - PAD_LEFT + kernel_w[None, :] * DIL_W
        )
        input_values = tl.load(
            x_ptr
            + batch * X_STRIDE_N
            + (group * CIN_PER_GROUP + input_channel[None, :]) * X_STRIDE_C
            + input_h * X_STRIDE_H
            + input_w * X_STRIDE_W,
            mask=(output_hw[:, None] < OH * OW)
            & (reduction[None, :] < reduction_extent)
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
            + kernel_h[None, :] * W_STRIDE_R
            + kernel_w[None, :] * W_STRIDE_S,
            mask=(output_channels[:, None] < COUT_PER_GROUP)
            & (reduction[None, :] < reduction_extent),
            other=0.0,
        )
        if INPUT_PRECISION == 1:
            accumulator += tl.dot(
                input_values, tl.trans(weights), input_precision="tf32"
            )
        else:
            accumulator += tl.dot(
                input_values, tl.trans(weights), input_precision="ieee"
            )

    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + group * COUT_PER_GROUP + output_channels,
            mask=output_channels < COUT_PER_GROUP,
            other=0.0,
        )
        accumulator += bias[None, :]
    tl.store(
        y_ptr
        + batch * Y_STRIDE_N
        + (group * COUT_PER_GROUP + output_channels[None, :]) * Y_STRIDE_C
        + output_h[:, None] * Y_STRIDE_H
        + output_w[:, None] * Y_STRIDE_W,
        accumulator.to(y_ptr.dtype.element_ty),
        mask=(output_hw[:, None] < OH * OW)
        & (output_channels[None, :] < COUT_PER_GROUP),
    )


@triton.jit
def _conv_fprop2d_im2col_kernel(
    x_ptr,
    columns_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    COL_STRIDE_N: tl.constexpr,
    COL_STRIDE_K: tl.constexpr,
    COL_STRIDE_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    tiles_m: tl.constexpr = tl.cdiv(M, BLOCK_M)
    tile_k = tile // tiles_m
    tile_m = tile - tile_k * tiles_m
    output_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    packed_k = tile_k * BLOCK_K + tl.arange(0, BLOCK_K)
    reduction_extent: tl.constexpr = CIN_PER_GROUP * KH * KW

    output_h = output_m // OW
    output_w = output_m - output_h * OW
    input_channel = packed_k // (KH * KW)
    kernel_hw = packed_k - input_channel * (KH * KW)
    kernel_h = kernel_hw // KW
    kernel_w = kernel_hw - kernel_h * KW
    input_h = (
        output_h[None, :] * STRIDE_H
        - PAD_TOP
        + kernel_h[:, None] * DIL_H
    )
    input_w = (
        output_w[None, :] * STRIDE_W
        - PAD_LEFT
        + kernel_w[:, None] * DIL_W
    )
    valid = (
        (packed_k[:, None] < reduction_extent)
        & (output_m[None, :] < M)
        & (input_h >= 0)
        & (input_h < XH)
        & (input_w >= 0)
        & (input_w < XW)
    )
    values = tl.load(
        x_ptr
        + batch * X_STRIDE_N
        + input_channel[:, None] * X_STRIDE_C
        + input_h * X_STRIDE_H
        + input_w * X_STRIDE_W,
        mask=valid,
        other=0.0,
    )
    tl.store(
        columns_ptr
        + batch * COL_STRIDE_N
        + packed_k[:, None] * COL_STRIDE_K
        + output_m[None, :] * COL_STRIDE_M,
        values,
        mask=(packed_k[:, None] < reduction_extent)
        & (output_m[None, :] < M),
    )


@triton.jit
def _conv_fprop2d_im2col_mm_kernel(
    w_ptr,
    columns_ptr,
    y_ptr,
    M: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    W_STRIDE_K: tl.constexpr,
    W_STRIDE_C: tl.constexpr,
    W_STRIDE_H: tl.constexpr,
    W_STRIDE_W: tl.constexpr,
    Y_STRIDE_N: tl.constexpr,
    Y_STRIDE_C: tl.constexpr,
    Y_STRIDE_H: tl.constexpr,
    Y_STRIDE_W: tl.constexpr,
    OW: tl.constexpr,
    COL_STRIDE_N: tl.constexpr,
    COL_STRIDE_K: tl.constexpr,
    COL_STRIDE_M: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    tile = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    tiles_m = tl.cdiv(M, BLOCK_M)
    tile_oc = tile // tiles_m
    tile_m = tile - tile_oc * tiles_m
    output_channels = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    output_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    reduction = tl.arange(0, BLOCK_K)
    reduction_extent: tl.constexpr = CIN_PER_GROUP * KH * KW
    weight_ptrs = (
        w_ptr
        + output_channels[:, None] * W_STRIDE_K
        + reduction[None, :]
    )
    column_ptrs = (
        columns_ptr
        + batch * COL_STRIDE_N
        + reduction[:, None] * COL_STRIDE_K
        + output_m[None, :] * COL_STRIDE_M
    )
    accumulator = tl.zeros((BLOCK_OC, BLOCK_M), dtype=tl.float32)
    for start in range(0, reduction_extent, BLOCK_K):
        reduction_offsets = start + reduction
        weights = tl.load(
            weight_ptrs,
            mask=(output_channels[:, None] < COUT_PER_GROUP)
            & (reduction_offsets[None, :] < reduction_extent),
            other=0.0,
        )
        columns = tl.load(
            column_ptrs,
            mask=(reduction_offsets[:, None] < reduction_extent)
            & (output_m[None, :] < M),
            other=0.0,
        )
        if INPUT_PRECISION == 1:
            accumulator += tl.dot(
                weights, columns, input_precision="tf32"
            )
        else:
            accumulator += tl.dot(
                weights, columns, input_precision="ieee"
            )
        weight_ptrs += BLOCK_K
        column_ptrs += BLOCK_K * COL_STRIDE_K

    output_h = output_m // OW
    output_w = output_m - output_h * OW
    tl.store(
        y_ptr
        + batch * Y_STRIDE_N
        + output_channels[:, None] * Y_STRIDE_C
        + output_h[None, :] * Y_STRIDE_H
        + output_w[None, :] * Y_STRIDE_W,
        accumulator.to(y_ptr.dtype.element_ty),
        mask=(output_channels[:, None] < COUT_PER_GROUP)
        & (output_m[None, :] < M),
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
        kernel_spatial = reduction % kernel_volume
        kernel_d = kernel_spatial // (KH * KW)
        kernel_hw = kernel_spatial % (KH * KW)
        kernel_h = kernel_hw // KW
        kernel_w = kernel_hw % KW
        input_d = (
            output_d[:, None] * STRIDE_D
            - PAD_FRONT
            + kernel_d[None, :] * DIL_D
        )
        input_h = (
            output_h[:, None] * STRIDE_H - PAD_TOP + kernel_h[None, :] * DIL_H
        )
        input_w = (
            output_w[:, None] * STRIDE_W - PAD_LEFT + kernel_w[None, :] * DIL_W
        )
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
        if INPUT_PRECISION == 1:
            accumulator += tl.dot(
                input_values, tl.trans(weights), input_precision="tf32"
            )
        else:
            accumulator += tl.dot(
                input_values, tl.trans(weights), input_precision="ieee"
            )

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
def _conv_dgrad_nd_impl(
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
    exact_stride2_2d: tl.constexpr = (
        XD == 1
        and OD == 1
        and KD == 1
        and KH == 3
        and KW == 3
        and STRIDE_D == 1
        and STRIDE_H == 2
        and STRIDE_W == 2
        and PAD_FRONT == 0
        and PAD_TOP == 1
        and PAD_LEFT == 1
        and DIL_D == 1
        and DIL_H == 1
        and DIL_W == 1
        and FLIP_FILTER == 0
    )
    if exact_stride2_2d:
        # Each loss position owns a disjoint 2x2 input-gradient tile.  The
        # four accumulators need 1, 2, 2, and 4 filter taps respectively,
        # avoiding the generic kernel's stride divisibility masks and the
        # three quarters of dot products that those masks discard.
        tile = tl.program_id(0)
        group = tl.program_id(1).to(tl.int64)
        batch_count: tl.constexpr = M // (XH * XW)
        loss_rows: tl.constexpr = batch_count * OH * OW
        tiles_m: tl.constexpr = tl.cdiv(loss_rows, BLOCK_M)
        tile_ci = tile // tiles_m
        tile_m = tile - tile_ci * tiles_m
        rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        input_channels = tile_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
        active_rows = rows < loss_rows
        loss_area: tl.constexpr = OH * OW
        batch = rows // loss_area
        loss_hw = rows - batch * loss_area
        loss_h = loss_hw // OW
        loss_w = loss_hw - loss_h * OW
        input_h0 = loss_h * 2
        input_w0 = loss_w * 2
        input_h1 = input_h0 + 1
        input_w1 = input_w0 + 1
        loss_h1 = loss_h + 1
        loss_w1 = loss_w + 1
        active_h1 = loss_h1 < OH
        active_w1 = loss_w1 < OW
        active_ci = input_channels < CIN_PER_GROUP
        accumulator00 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
        accumulator01 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
        accumulator10 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
        accumulator11 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

        for start in range(0, COUT_PER_GROUP, BLOCK_K):
            output_channels = start + tl.arange(0, BLOCK_K)
            active_co = output_channels < COUT_PER_GROUP
            loss_base = (
                dy_ptr
                + batch[:, None] * DY_STRIDE_N
                + (group * COUT_PER_GROUP + output_channels[None, :])
                * DY_STRIDE_C
                + loss_h[:, None] * DY_STRIDE_H
                + loss_w[:, None] * DY_STRIDE_W
            )
            loss_mask = active_rows[:, None] & active_co[None, :]
            loss00 = tl.load(loss_base, mask=loss_mask, other=0.0)
            loss01 = tl.load(
                loss_base + DY_STRIDE_W,
                mask=loss_mask & active_w1[:, None],
                other=0.0,
            )
            loss10 = tl.load(
                loss_base + DY_STRIDE_H,
                mask=loss_mask & active_h1[:, None],
                other=0.0,
            )
            loss11 = tl.load(
                loss_base + DY_STRIDE_H + DY_STRIDE_W,
                mask=(
                    loss_mask
                    & active_h1[:, None]
                    & active_w1[:, None]
                ),
                other=0.0,
            )
            weight_base = (
                w_ptr
                + (group * COUT_PER_GROUP + output_channels[:, None])
                * W_STRIDE_K
                + input_channels[None, :] * W_STRIDE_C
            )
            weight_mask = active_co[:, None] & active_ci[None, :]
            weight00 = tl.load(weight_base, mask=weight_mask, other=0.0)
            weight01 = tl.load(
                weight_base + W_STRIDE_W, mask=weight_mask, other=0.0
            )
            weight02 = tl.load(
                weight_base + 2 * W_STRIDE_W,
                mask=weight_mask,
                other=0.0,
            )
            weight10 = tl.load(
                weight_base + W_STRIDE_H, mask=weight_mask, other=0.0
            )
            weight11 = tl.load(
                weight_base + W_STRIDE_H + W_STRIDE_W,
                mask=weight_mask,
                other=0.0,
            )
            weight12 = tl.load(
                weight_base + W_STRIDE_H + 2 * W_STRIDE_W,
                mask=weight_mask,
                other=0.0,
            )
            weight20 = tl.load(
                weight_base + 2 * W_STRIDE_H,
                mask=weight_mask,
                other=0.0,
            )
            weight21 = tl.load(
                weight_base + 2 * W_STRIDE_H + W_STRIDE_W,
                mask=weight_mask,
                other=0.0,
            )
            weight22 = tl.load(
                weight_base + 2 * W_STRIDE_H + 2 * W_STRIDE_W,
                mask=weight_mask,
                other=0.0,
            )
            if INPUT_PRECISION == 1:
                accumulator00 = tl.dot(
                    loss00,
                    weight11,
                    accumulator00,
                    input_precision="tf32",
                )
                accumulator01 = tl.dot(
                    loss00,
                    weight12,
                    accumulator01,
                    input_precision="tf32",
                )
                accumulator01 = tl.dot(
                    loss01,
                    weight10,
                    accumulator01,
                    input_precision="tf32",
                )
                accumulator10 = tl.dot(
                    loss00,
                    weight21,
                    accumulator10,
                    input_precision="tf32",
                )
                accumulator10 = tl.dot(
                    loss10,
                    weight01,
                    accumulator10,
                    input_precision="tf32",
                )
                accumulator11 = tl.dot(
                    loss00,
                    weight22,
                    accumulator11,
                    input_precision="tf32",
                )
                accumulator11 = tl.dot(
                    loss01,
                    weight20,
                    accumulator11,
                    input_precision="tf32",
                )
                accumulator11 = tl.dot(
                    loss10,
                    weight02,
                    accumulator11,
                    input_precision="tf32",
                )
                accumulator11 = tl.dot(
                    loss11,
                    weight00,
                    accumulator11,
                    input_precision="tf32",
                )
            else:
                accumulator00 = tl.dot(
                    loss00,
                    weight11,
                    accumulator00,
                    input_precision="ieee",
                )
                accumulator01 = tl.dot(
                    loss00,
                    weight12,
                    accumulator01,
                    input_precision="ieee",
                )
                accumulator01 = tl.dot(
                    loss01,
                    weight10,
                    accumulator01,
                    input_precision="ieee",
                )
                accumulator10 = tl.dot(
                    loss00,
                    weight21,
                    accumulator10,
                    input_precision="ieee",
                )
                accumulator10 = tl.dot(
                    loss10,
                    weight01,
                    accumulator10,
                    input_precision="ieee",
                )
                accumulator11 = tl.dot(
                    loss00,
                    weight22,
                    accumulator11,
                    input_precision="ieee",
                )
                accumulator11 = tl.dot(
                    loss01,
                    weight20,
                    accumulator11,
                    input_precision="ieee",
                )
                accumulator11 = tl.dot(
                    loss10,
                    weight02,
                    accumulator11,
                    input_precision="ieee",
                )
                accumulator11 = tl.dot(
                    loss11,
                    weight00,
                    accumulator11,
                    input_precision="ieee",
                )

        output_base = (
            dx_ptr
            + batch[:, None] * X_STRIDE_N
            + (group * CIN_PER_GROUP + input_channels[None, :]) * X_STRIDE_C
        )
        output_mask = active_rows[:, None] & active_ci[None, :]
        tl.store(
            output_base
            + input_h0[:, None] * X_STRIDE_H
            + input_w0[:, None] * X_STRIDE_W,
            accumulator00.to(dx_ptr.dtype.element_ty),
            mask=(
                output_mask
                & (input_h0[:, None] < XH)
                & (input_w0[:, None] < XW)
            ),
        )
        tl.store(
            output_base
            + input_h0[:, None] * X_STRIDE_H
            + input_w1[:, None] * X_STRIDE_W,
            accumulator01.to(dx_ptr.dtype.element_ty),
            mask=(
                output_mask
                & (input_h0[:, None] < XH)
                & (input_w1[:, None] < XW)
            ),
        )
        tl.store(
            output_base
            + input_h1[:, None] * X_STRIDE_H
            + input_w0[:, None] * X_STRIDE_W,
            accumulator10.to(dx_ptr.dtype.element_ty),
            mask=(
                output_mask
                & (input_h1[:, None] < XH)
                & (input_w0[:, None] < XW)
            ),
        )
        tl.store(
            output_base
            + input_h1[:, None] * X_STRIDE_H
            + input_w1[:, None] * X_STRIDE_W,
            accumulator11.to(dx_ptr.dtype.element_ty),
            mask=(
                output_mask
                & (input_h1[:, None] < XH)
                & (input_w1[:, None] < XW)
            ),
        )
    else:
        tile = tl.program_id(0)
        group = tl.program_id(1).to(tl.int64)
        tiles_ci = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
        tile_m = tile // tiles_ci
        tile_ci = tile - tile_m * tiles_ci
        rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        input_channels = tile_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
        input_volume: tl.constexpr = XD * XH * XW
        batch = rows // input_volume
        spatial = rows - batch * input_volume
        input_d = spatial // (XH * XW)
        input_hw = spatial - input_d * (XH * XW)
        input_h = input_hw // XW
        input_w = input_hw - input_h * XW
        kernel_volume: tl.constexpr = KD * KH * KW
        reduction_extent: tl.constexpr = COUT_PER_GROUP * kernel_volume
        reduction_base = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

        for start in range(0, reduction_extent, BLOCK_K):
            reduction = start + reduction_base
            output_channel = reduction // kernel_volume
            kernel_spatial = reduction - output_channel * kernel_volume
            kernel_d = kernel_spatial // (KH * KW)
            kernel_hw = kernel_spatial - kernel_d * (KH * KW)
            kernel_h = kernel_hw // KW
            kernel_w = kernel_hw - kernel_h * KW
            numerator_d = (
                input_d[:, None] + PAD_FRONT - kernel_d[None, :] * DIL_D
            )
            numerator_h = (
                input_h[:, None] + PAD_TOP - kernel_h[None, :] * DIL_H
            )
            numerator_w = (
                input_w[:, None] + PAD_LEFT - kernel_w[None, :] * DIL_W
            )
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
                + (group * COUT_PER_GROUP + output_channel[None, :])
                * DY_STRIDE_C
                + output_d * DY_STRIDE_D
                + output_h * DY_STRIDE_H
                + output_w * DY_STRIDE_W,
                mask=valid,
                other=0.0,
            )
            weight_d = kernel_d + FLIP_FILTER * (KD - 1 - 2 * kernel_d)
            weight_h = kernel_h + FLIP_FILTER * (KH - 1 - 2 * kernel_h)
            weight_w = kernel_w + FLIP_FILTER * (KW - 1 - 2 * kernel_w)
            weights = tl.load(
                w_ptr
                + (group * COUT_PER_GROUP + output_channel[:, None])
                * W_STRIDE_K
                + input_channels[None, :] * W_STRIDE_C
                + weight_d[:, None] * W_STRIDE_D
                + weight_h[:, None] * W_STRIDE_H
                + weight_w[:, None] * W_STRIDE_W,
                mask=(reduction[:, None] < reduction_extent)
                & (input_channels[None, :] < CIN_PER_GROUP),
                other=0.0,
            )
            if INPUT_PRECISION == 1:
                accumulator += tl.dot(
                    losses, weights, input_precision="tf32"
                )
            else:
                accumulator += tl.dot(
                    losses, weights, input_precision="ieee"
                )

        tl.store(
            dx_ptr
            + batch[:, None] * X_STRIDE_N
            + (group * CIN_PER_GROUP + input_channels[None, :]) * X_STRIDE_C
            + input_d[:, None] * X_STRIDE_D
            + input_h[:, None] * X_STRIDE_H
            + input_w[:, None] * X_STRIDE_W,
            accumulator.to(dx_ptr.dtype.element_ty),
            mask=(rows[:, None] < M)
            & (input_channels[None, :] < CIN_PER_GROUP),
        )



@triton.jit
def _conv_dgrad_dot(
    left,
    right,
    accumulator,
    INPUT_PRECISION: tl.constexpr,
):
    if INPUT_PRECISION == 1:
        return tl.dot(
            left,
            right,
            accumulator,
            input_precision="tf32",
        )
    return tl.dot(
        left,
        right,
        accumulator,
        input_precision="ieee",
    )


@triton.jit
def _conv_dgrad_stride2_parity(
    loss_origin,
    weight_origin,
    active_rows,
    active_ci,
    active_h1,
    active_w1,
    group,
    COUT_PER_GROUP: tl.constexpr,
    DY_STRIDE_C: tl.constexpr,
    DY_STRIDE_H: tl.constexpr,
    DY_STRIDE_W: tl.constexpr,
    W_STRIDE_K: tl.constexpr,
    W_STRIDE_H: tl.constexpr,
    W_STRIDE_W: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PARITY: tl.constexpr,
):
    parity_h: tl.constexpr = PARITY // 2
    parity_w: tl.constexpr = PARITY - parity_h * 2
    taps_h: tl.constexpr = parity_h + 1
    taps_w: tl.constexpr = parity_w + 1
    tap_count: tl.constexpr = taps_h * taps_w
    accumulator = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for start in range(0, COUT_PER_GROUP, BLOCK_K):
        reduction = tl.arange(0, tap_count * BLOCK_K)
        neighbor_h = reduction // (taps_w * BLOCK_K)
        remaining = reduction - neighbor_h * (taps_w * BLOCK_K)
        neighbor_w = remaining // BLOCK_K
        output_channels = start + remaining - neighbor_w * BLOCK_K
        active_co = output_channels < COUT_PER_GROUP
        losses = tl.load(
            loss_origin[:, None]
            + (group * COUT_PER_GROUP + output_channels[None, :])
            * DY_STRIDE_C
            + neighbor_h[None, :] * DY_STRIDE_H
            + neighbor_w[None, :] * DY_STRIDE_W,
            mask=(
                active_rows[:, None]
                & active_co[None, :]
                & (
                    (neighbor_h[None, :] == 0)
                    | active_h1[:, None]
                )
                & (
                    (neighbor_w[None, :] == 0)
                    | active_w1[:, None]
                )
            ),
            other=0.0,
        )
        weight_h = 1 + parity_h - 2 * parity_h * neighbor_h
        weight_w = 1 + parity_w - 2 * parity_w * neighbor_w
        weights = tl.load(
            weight_origin[None, :]
            + (group * COUT_PER_GROUP + output_channels[:, None])
            * W_STRIDE_K
            + weight_h[:, None] * W_STRIDE_H
            + weight_w[:, None] * W_STRIDE_W,
            mask=active_co[:, None] & active_ci[None, :],
            other=0.0,
        )
        accumulator = _conv_dgrad_dot(
            losses, weights, accumulator, INPUT_PRECISION
        )

    return accumulator


@triton.jit
def _conv_dgrad2d_pack_filter_kernel(
    filter_ptr,
    packed_ptr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    W_STRIDE_K: tl.constexpr,
    W_STRIDE_C: tl.constexpr,
    W_STRIDE_H: tl.constexpr,
    W_STRIDE_W: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CIT: tl.constexpr,
):
    output_channels = (
        tl.program_id(0) * BLOCK_CO + tl.arange(0, BLOCK_CO)
    )
    packed_columns = (
        tl.program_id(1) * BLOCK_CIT + tl.arange(0, BLOCK_CIT)
    )
    input_channel = packed_columns // 9
    tap = packed_columns - input_channel * 9
    kernel_h = tap // 3
    kernel_w = tap - kernel_h * 3
    active_co = output_channels < COUT_PER_GROUP
    active_column = packed_columns < 9 * CIN_PER_GROUP
    values = tl.load(
        filter_ptr
        + output_channels[:, None] * W_STRIDE_K
        + input_channel[None, :] * W_STRIDE_C
        + kernel_h[None, :] * W_STRIDE_H
        + kernel_w[None, :] * W_STRIDE_W,
        mask=active_co[:, None] & active_column[None, :],
        other=0.0,
    )
    tl.store(
        packed_ptr
        + packed_columns[None, :] * COUT_PER_GROUP
        + output_channels[:, None],
        values,
        mask=active_co[:, None] & active_column[None, :],
    )


@triton.jit
def _conv_dgrad2d_packed_parity_kernel(
    loss_ptr,
    packed_filter_ptr,
    output_ptr,
    M: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    DY_STRIDE_N: tl.constexpr,
    DY_STRIDE_C: tl.constexpr,
    DY_STRIDE_H: tl.constexpr,
    DY_STRIDE_W: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    PARITY: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile = tl.program_id(0)
    tiles_m: tl.constexpr = tl.cdiv(M, BLOCK_M)
    tile_ci = tile // tiles_m
    tile_m = tile - tile_ci * tiles_m
    input_channels = tile_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    active_ci = input_channels < CIN_PER_GROUP
    active_rows = rows < M
    loss_area: tl.constexpr = OH * OW
    batch = rows // loss_area
    loss_hw = rows - batch * loss_area
    loss_h = loss_hw // OW
    loss_w = loss_hw - loss_h * OW
    parity_h: tl.constexpr = PARITY // 2
    parity_w: tl.constexpr = PARITY - parity_h * 2
    accumulator = tl.zeros((BLOCK_CI, BLOCK_M), dtype=tl.float32)

    for start in range(0, COUT_PER_GROUP, BLOCK_K):
        output_channels = start + tl.arange(0, BLOCK_K)
        active_co = output_channels < COUT_PER_GROUP
        for neighbor_h in tl.static_range(0, parity_h + 1):
            for neighbor_w in tl.static_range(0, parity_w + 1):
                weight_h = (
                    1 + parity_h - 2 * parity_h * neighbor_h
                )
                weight_w = (
                    1 + parity_w - 2 * parity_w * neighbor_w
                )
                tap = weight_h * 3 + weight_w
                weights = tl.load(
                    packed_filter_ptr
                    + (input_channels[:, None] * 9 + tap)
                    * COUT_PER_GROUP
                    + output_channels[None, :],
                    mask=active_ci[:, None] & active_co[None, :],
                    other=0.0,
                )
                neighbor_loss_h = loss_h + neighbor_h
                neighbor_loss_w = loss_w + neighbor_w
                losses = tl.load(
                    loss_ptr
                    + batch[None, :] * DY_STRIDE_N
                    + output_channels[:, None] * DY_STRIDE_C
                    + neighbor_loss_h[None, :] * DY_STRIDE_H
                    + neighbor_loss_w[None, :] * DY_STRIDE_W,
                    mask=(
                        active_co[:, None]
                        & active_rows[None, :]
                        & (neighbor_loss_h[None, :] < OH)
                        & (neighbor_loss_w[None, :] < OW)
                    ),
                    other=0.0,
                )
                accumulator = _conv_dgrad_dot(
                    weights, losses, accumulator, INPUT_PRECISION
                )

    output_h = loss_h * 2 + parity_h
    output_w = loss_w * 2 + parity_w
    tl.store(
        output_ptr
        + batch[None, :] * X_STRIDE_N
        + input_channels[:, None] * X_STRIDE_C
        + output_h[None, :] * X_STRIDE_H
        + output_w[None, :] * X_STRIDE_W,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=(
            active_ci[:, None]
            & active_rows[None, :]
            & (output_h[None, :] < XH)
            & (output_w[None, :] < XW)
        ),
    )


@triton.jit
def _conv_dgrad2d_packed_parity_split_kernel(
    loss_ptr,
    packed_filter_ptr,
    partial_ptr,
    M: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    DY_STRIDE_N: tl.constexpr,
    DY_STRIDE_C: tl.constexpr,
    DY_STRIDE_H: tl.constexpr,
    DY_STRIDE_W: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    PARITY: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile = tl.program_id(0)
    split = tl.program_id(1).to(tl.int64)
    tiles_m: tl.constexpr = tl.cdiv(M, BLOCK_M)
    tile_ci = tile // tiles_m
    tile_m = tile - tile_ci * tiles_m
    input_channels = tile_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    active_ci = input_channels < CIN_PER_GROUP
    active_rows = rows < M
    loss_area: tl.constexpr = OH * OW
    batch = rows // loss_area
    loss_hw = rows - batch * loss_area
    loss_h = loss_hw // OW
    loss_w = loss_hw - loss_h * OW
    parity_h: tl.constexpr = PARITY // 2
    parity_w: tl.constexpr = PARITY - parity_h * 2
    channels_per_split: tl.constexpr = tl.cdiv(
        COUT_PER_GROUP, NUM_SPLITS
    )
    split_start = split * channels_per_split
    split_end = tl.minimum(split_start + channels_per_split, COUT_PER_GROUP)
    accumulator = tl.zeros((BLOCK_CI, BLOCK_M), dtype=tl.float32)

    for local_start in range(0, channels_per_split, BLOCK_K):
        output_channels = (
            split_start + local_start + tl.arange(0, BLOCK_K)
        )
        active_co = output_channels < split_end
        for neighbor_h in tl.static_range(0, parity_h + 1):
            for neighbor_w in tl.static_range(0, parity_w + 1):
                weight_h = (
                    1 + parity_h - 2 * parity_h * neighbor_h
                )
                weight_w = (
                    1 + parity_w - 2 * parity_w * neighbor_w
                )
                tap = weight_h * 3 + weight_w
                weights = tl.load(
                    packed_filter_ptr
                    + (input_channels[:, None] * 9 + tap)
                    * COUT_PER_GROUP
                    + output_channels[None, :],
                    mask=active_ci[:, None] & active_co[None, :],
                    other=0.0,
                )
                neighbor_loss_h = loss_h + neighbor_h
                neighbor_loss_w = loss_w + neighbor_w
                losses = tl.load(
                    loss_ptr
                    + batch[None, :] * DY_STRIDE_N
                    + output_channels[:, None] * DY_STRIDE_C
                    + neighbor_loss_h[None, :] * DY_STRIDE_H
                    + neighbor_loss_w[None, :] * DY_STRIDE_W,
                    mask=(
                        active_co[:, None]
                        & active_rows[None, :]
                        & (neighbor_loss_h[None, :] < OH)
                        & (neighbor_loss_w[None, :] < OW)
                    ),
                    other=0.0,
                )
                accumulator = _conv_dgrad_dot(
                    weights, losses, accumulator, INPUT_PRECISION
                )

    parity_stride: tl.constexpr = NUM_SPLITS * CIN_PER_GROUP * M
    split_stride: tl.constexpr = CIN_PER_GROUP * M
    tl.store(
        partial_ptr
        + PARITY * parity_stride
        + split * split_stride
        + input_channels[:, None] * M
        + rows[None, :],
        accumulator,
        mask=active_ci[:, None] & active_rows[None, :],
    )


@triton.jit
def _conv_dgrad2d_packed_reduce_kernel(
    partial_ptr,
    output_ptr,
    M: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    parity_stride: tl.constexpr = NUM_SPLITS * CIN_PER_GROUP * M
    split_stride: tl.constexpr = CIN_PER_GROUP * M
    parity_area: tl.constexpr = CIN_PER_GROUP * M
    total: tl.constexpr = 4 * parity_area
    linear = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    parity = linear // parity_area
    parity_offset = linear - parity * parity_area
    input_channel = parity_offset // M
    row = parity_offset - input_channel * M
    active = linear < total
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for split in tl.static_range(0, NUM_SPLITS):
        accumulator += tl.load(
            partial_ptr
            + parity * parity_stride
            + split * split_stride
            + input_channel * M
            + row,
            mask=active,
            other=0.0,
        )

    loss_area: tl.constexpr = OH * OW
    batch = row // loss_area
    loss_hw = row - batch * loss_area
    loss_h = loss_hw // OW
    loss_w = loss_hw - loss_h * OW
    parity_h = parity // 2
    parity_w = parity - parity_h * 2
    output_h = loss_h * 2 + parity_h
    output_w = loss_w * 2 + parity_w
    tl.store(
        output_ptr
        + batch * X_STRIDE_N
        + input_channel * X_STRIDE_C
        + output_h * X_STRIDE_H
        + output_w * X_STRIDE_W,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=(
            active
            & (output_h < XH)
            & (output_w < XW)
        ),
    )


@triton.jit
def _conv_dgrad2d_dense_pack_filter_kernel(
    filter_ptr,
    packed_ptr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    W_STRIDE_K: tl.constexpr,
    W_STRIDE_C: tl.constexpr,
    W_STRIDE_H: tl.constexpr,
    W_STRIDE_W: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    matrix_rows = (
        tl.program_id(0) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    )
    packed_k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    parity = matrix_rows // CIN_PER_GROUP
    input_channel = matrix_rows - parity * CIN_PER_GROUP
    neighbor = packed_k // COUT_PER_GROUP
    output_channel = packed_k - neighbor * COUT_PER_GROUP
    parity_h = parity // 2
    parity_w = parity - parity_h * 2
    neighbor_h = neighbor // 2
    neighbor_w = neighbor - neighbor_h * 2
    allowed = (
        (neighbor_h[None, :] <= parity_h[:, None])
        & (neighbor_w[None, :] <= parity_w[:, None])
    )
    weight_h = (
        1
        + parity_h[:, None]
        - 2 * parity_h[:, None] * neighbor_h[None, :]
    )
    weight_w = (
        1
        + parity_w[:, None]
        - 2 * parity_w[:, None] * neighbor_w[None, :]
    )
    active_row = matrix_rows < 4 * CIN_PER_GROUP
    active_k = packed_k < 4 * COUT_PER_GROUP
    active = active_row[:, None] & active_k[None, :]
    values = tl.load(
        filter_ptr
        + output_channel[None, :] * W_STRIDE_K
        + input_channel[:, None] * W_STRIDE_C
        + weight_h * W_STRIDE_H
        + weight_w * W_STRIDE_W,
        mask=active & allowed,
        other=0.0,
    )
    tl.store(
        packed_ptr
        + matrix_rows[:, None] * (4 * COUT_PER_GROUP)
        + packed_k[None, :],
        values,
        mask=active,
    )


@triton.jit
def _conv_dgrad2d_dense_pack_loss_kernel(
    loss_ptr,
    packed_ptr,
    PACKED_OFFSET: tl.constexpr,
    M: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    DY_STRIDE_N: tl.constexpr,
    DY_STRIDE_C: tl.constexpr,
    DY_STRIDE_H: tl.constexpr,
    DY_STRIDE_W: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    packed_k = tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)
    rows = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    neighbor = packed_k // COUT_PER_GROUP
    output_channel = packed_k - neighbor * COUT_PER_GROUP
    neighbor_h = neighbor // 2
    neighbor_w = neighbor - neighbor_h * 2
    loss_area: tl.constexpr = OH * OW
    batch = rows // loss_area
    loss_hw = rows - batch * loss_area
    loss_h = loss_hw // OW
    loss_w = loss_hw - loss_h * OW
    neighbor_loss_h = loss_h[None, :] + neighbor_h[:, None]
    neighbor_loss_w = loss_w[None, :] + neighbor_w[:, None]
    active_k = packed_k < 4 * COUT_PER_GROUP
    active_rows = rows < M
    active = active_k[:, None] & active_rows[None, :]
    valid = (
        active
        & (neighbor_loss_h < OH)
        & (neighbor_loss_w < OW)
    )
    values = tl.load(
        loss_ptr
        + batch[None, :] * DY_STRIDE_N
        + output_channel[:, None] * DY_STRIDE_C
        + neighbor_loss_h * DY_STRIDE_H
        + neighbor_loss_w * DY_STRIDE_W,
        mask=valid,
        other=0.0,
    )
    tl.store(
        packed_ptr
        + PACKED_OFFSET
        + packed_k[:, None] * M
        + rows[None, :],
        values,
        mask=active,
    )


@triton.jit
def _conv_dgrad2d_dense_mm_kernel(
    packed_filter_ptr,
    packed_loss_ptr,
    output_ptr,
    PACKED_LOSS_OFFSET: tl.constexpr,
    M: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    matrix_rows: tl.constexpr = 4 * CIN_PER_GROUP
    reduction_extent: tl.constexpr = 4 * COUT_PER_GROUP
    tile = tl.program_id(0)
    tiles_row: tl.constexpr = tl.cdiv(matrix_rows, BLOCK_ROW)
    tiles_m: tl.constexpr = tl.cdiv(M, BLOCK_M)
    tiles_per_group: tl.constexpr = GROUP_M * tiles_m
    group = tile // tiles_per_group
    first_tile_row = group * GROUP_M
    group_rows = tl.minimum(tiles_row - first_tile_row, GROUP_M)
    tile_in_group = tile - group * tiles_per_group
    tile_row = first_tile_row + tile_in_group % group_rows
    tile_m = tile_in_group // group_rows
    matrix_row = tile_row * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    columns = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    reduction = tl.arange(0, BLOCK_K)
    filter_ptrs = (
        packed_filter_ptr
        + matrix_row[:, None] * reduction_extent
        + reduction[None, :]
    )
    loss_ptrs = (
        packed_loss_ptr
        + PACKED_LOSS_OFFSET
        + reduction[:, None] * M
        + columns[None, :]
    )
    accumulator = tl.zeros((BLOCK_ROW, BLOCK_M), dtype=tl.float32)
    for start in range(0, reduction_extent, BLOCK_K):
        reduction_offsets = start + reduction
        weights = tl.load(
            filter_ptrs,
            mask=(matrix_row[:, None] < matrix_rows)
            & (reduction_offsets[None, :] < reduction_extent),
            other=0.0,
        )
        losses = tl.load(
            loss_ptrs,
            mask=(reduction_offsets[:, None] < reduction_extent)
            & (columns[None, :] < M),
            other=0.0,
        )
        accumulator = _conv_dgrad_dot(
            weights, losses, accumulator, INPUT_PRECISION
        )
        filter_ptrs += BLOCK_K
        loss_ptrs += BLOCK_K * M

    parity = matrix_row // CIN_PER_GROUP
    input_channel = matrix_row - parity * CIN_PER_GROUP
    loss_area: tl.constexpr = OH * OW
    batch = columns // loss_area
    loss_hw = columns - batch * loss_area
    loss_h = loss_hw // OW
    loss_w = loss_hw - loss_h * OW
    parity_h = parity // 2
    parity_w = parity - parity_h * 2
    output_h = loss_h[None, :] * 2 + parity_h[:, None]
    output_w = loss_w[None, :] * 2 + parity_w[:, None]
    tl.store(
        output_ptr
        + batch[None, :] * X_STRIDE_N
        + input_channel[:, None] * X_STRIDE_C
        + output_h * X_STRIDE_H
        + output_w * X_STRIDE_W,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=(
            (matrix_row[:, None] < matrix_rows)
            & (columns[None, :] < M)
            & (output_h < XH)
            & (output_w < XW)
        ),
    )


@triton.jit
def _conv_dgrad_concat_experiment(
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
    exact_stride2_2d: tl.constexpr = (
        XD == 1
        and OD == 1
        and KD == 1
        and KH == 3
        and KW == 3
        and STRIDE_D == 1
        and STRIDE_H == 2
        and STRIDE_W == 2
        and PAD_FRONT == 0
        and PAD_TOP == 1
        and PAD_LEFT == 1
        and DIL_D == 1
        and DIL_H == 1
        and DIL_W == 1
        and FLIP_FILTER == 0
    )
    if exact_stride2_2d:
        tile = tl.program_id(0)
        group = tl.program_id(1).to(tl.int64)
        batch_count: tl.constexpr = M // (XH * XW)
        loss_rows: tl.constexpr = batch_count * OH * OW
        tiles_m: tl.constexpr = tl.cdiv(loss_rows, BLOCK_M)
        tile_ci = tile // tiles_m
        tile_m = tile - tile_ci * tiles_m
        rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        input_channels = tile_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
        active_rows = rows < loss_rows
        active_ci = input_channels < CIN_PER_GROUP
        loss_area: tl.constexpr = OH * OW
        batch = rows // loss_area
        loss_hw = rows - batch * loss_area
        loss_h = loss_hw // OW
        loss_w = loss_hw - loss_h * OW
        input_h0 = loss_h * 2
        input_w0 = loss_w * 2
        input_h1 = input_h0 + 1
        input_w1 = input_w0 + 1
        active_h1 = loss_h + 1 < OH
        active_w1 = loss_w + 1 < OW
        loss_origin = (
            dy_ptr
            + batch * DY_STRIDE_N
            + loss_h * DY_STRIDE_H
            + loss_w * DY_STRIDE_W
        )
        weight_origin = w_ptr + input_channels * W_STRIDE_C
        accumulator00 = tl.zeros(
            (BLOCK_M, BLOCK_CI), dtype=tl.float32
        )
        accumulator01 = tl.zeros(
            (BLOCK_M, BLOCK_CI), dtype=tl.float32
        )
        accumulator10 = tl.zeros(
            (BLOCK_M, BLOCK_CI), dtype=tl.float32
        )
        accumulator11 = tl.zeros(
            (BLOCK_M, BLOCK_CI), dtype=tl.float32
        )

        for start in range(0, COUT_PER_GROUP, BLOCK_K):
            reduction00 = tl.arange(0, BLOCK_K)
            output_channels00 = start + reduction00
            active_co00 = output_channels00 < COUT_PER_GROUP
            losses00 = tl.load(
                loss_origin[:, None]
                + (group * COUT_PER_GROUP + output_channels00[None, :])
                * DY_STRIDE_C,
                mask=active_rows[:, None] & active_co00[None, :],
                other=0.0,
            )
            weights00 = tl.load(
                weight_origin[None, :]
                + (group * COUT_PER_GROUP + output_channels00[:, None])
                * W_STRIDE_K
                + W_STRIDE_H
                + W_STRIDE_W,
                mask=active_co00[:, None] & active_ci[None, :],
                other=0.0,
            )
            accumulator00 = _conv_dgrad_dot(
                losses00, weights00, accumulator00, INPUT_PRECISION
            )

            reduction01 = tl.arange(0, 2 * BLOCK_K)
            tap_w01 = reduction01 // BLOCK_K
            output_channels01 = (
                start + reduction01 - tap_w01 * BLOCK_K
            )
            active_co01 = output_channels01 < COUT_PER_GROUP
            losses01 = tl.load(
                loss_origin[:, None]
                + (group * COUT_PER_GROUP + output_channels01[None, :])
                * DY_STRIDE_C
                + tap_w01[None, :] * DY_STRIDE_W,
                mask=(
                    active_rows[:, None]
                    & active_co01[None, :]
                    & (
                        (tap_w01[None, :] == 0)
                        | active_w1[:, None]
                    )
                ),
                other=0.0,
            )
            weights01 = tl.load(
                weight_origin[None, :]
                + (group * COUT_PER_GROUP + output_channels01[:, None])
                * W_STRIDE_K
                + W_STRIDE_H
                + (2 - 2 * tap_w01[:, None]) * W_STRIDE_W,
                mask=active_co01[:, None] & active_ci[None, :],
                other=0.0,
            )
            accumulator01 = _conv_dgrad_dot(
                losses01, weights01, accumulator01, INPUT_PRECISION
            )

            reduction10 = tl.arange(0, 2 * BLOCK_K)
            tap_h10 = reduction10 // BLOCK_K
            output_channels10 = (
                start + reduction10 - tap_h10 * BLOCK_K
            )
            active_co10 = output_channels10 < COUT_PER_GROUP
            losses10 = tl.load(
                loss_origin[:, None]
                + (group * COUT_PER_GROUP + output_channels10[None, :])
                * DY_STRIDE_C
                + tap_h10[None, :] * DY_STRIDE_H,
                mask=(
                    active_rows[:, None]
                    & active_co10[None, :]
                    & (
                        (tap_h10[None, :] == 0)
                        | active_h1[:, None]
                    )
                ),
                other=0.0,
            )
            weights10 = tl.load(
                weight_origin[None, :]
                + (group * COUT_PER_GROUP + output_channels10[:, None])
                * W_STRIDE_K
                + (2 - 2 * tap_h10[:, None]) * W_STRIDE_H
                + W_STRIDE_W,
                mask=active_co10[:, None] & active_ci[None, :],
                other=0.0,
            )
            accumulator10 = _conv_dgrad_dot(
                losses10, weights10, accumulator10, INPUT_PRECISION
            )

            reduction11 = tl.arange(0, 4 * BLOCK_K)
            tap_h11 = reduction11 // (2 * BLOCK_K)
            remaining11 = reduction11 - tap_h11 * (2 * BLOCK_K)
            tap_w11 = remaining11 // BLOCK_K
            output_channels11 = (
                start + remaining11 - tap_w11 * BLOCK_K
            )
            active_co11 = output_channels11 < COUT_PER_GROUP
            losses11 = tl.load(
                loss_origin[:, None]
                + (group * COUT_PER_GROUP + output_channels11[None, :])
                * DY_STRIDE_C
                + tap_h11[None, :] * DY_STRIDE_H
                + tap_w11[None, :] * DY_STRIDE_W,
                mask=(
                    active_rows[:, None]
                    & active_co11[None, :]
                    & (
                        (tap_h11[None, :] == 0)
                        | active_h1[:, None]
                    )
                    & (
                        (tap_w11[None, :] == 0)
                        | active_w1[:, None]
                    )
                ),
                other=0.0,
            )
            weights11 = tl.load(
                weight_origin[None, :]
                + (group * COUT_PER_GROUP + output_channels11[:, None])
                * W_STRIDE_K
                + (2 - 2 * tap_h11[:, None]) * W_STRIDE_H
                + (2 - 2 * tap_w11[:, None]) * W_STRIDE_W,
                mask=active_co11[:, None] & active_ci[None, :],
                other=0.0,
            )
            accumulator11 = _conv_dgrad_dot(
                losses11, weights11, accumulator11, INPUT_PRECISION
            )

        output_base = (
            dx_ptr
            + batch[:, None] * X_STRIDE_N
            + (group * CIN_PER_GROUP + input_channels[None, :])
            * X_STRIDE_C
        )
        output_mask = active_rows[:, None] & active_ci[None, :]
        tl.store(
            output_base
            + input_h0[:, None] * X_STRIDE_H
            + input_w0[:, None] * X_STRIDE_W,
            accumulator00.to(dx_ptr.dtype.element_ty),
            mask=output_mask,
        )
        tl.store(
            output_base
            + input_h0[:, None] * X_STRIDE_H
            + input_w1[:, None] * X_STRIDE_W,
            accumulator01.to(dx_ptr.dtype.element_ty),
            mask=output_mask & (input_w1[:, None] < XW),
        )
        tl.store(
            output_base
            + input_h1[:, None] * X_STRIDE_H
            + input_w0[:, None] * X_STRIDE_W,
            accumulator10.to(dx_ptr.dtype.element_ty),
            mask=output_mask & (input_h1[:, None] < XH),
        )
        tl.store(
            output_base
            + input_h1[:, None] * X_STRIDE_H
            + input_w1[:, None] * X_STRIDE_W,
            accumulator11.to(dx_ptr.dtype.element_ty),
            mask=(
                output_mask
                & (input_h1[:, None] < XH)
                & (input_w1[:, None] < XW)
            ),
        )
    else:
        _conv_dgrad_nd_impl(
            dy_ptr,
            w_ptr,
            dx_ptr,
            XD,
            XH,
            XW,
            OD,
            OH,
            OW,
            KD,
            KH,
            KW,
            CIN_PER_GROUP,
            COUT_PER_GROUP,
            STRIDE_D,
            STRIDE_H,
            STRIDE_W,
            PAD_FRONT,
            PAD_TOP,
            PAD_LEFT,
            DIL_D,
            DIL_H,
            DIL_W,
            FLIP_FILTER,
            DY_STRIDE_N,
            DY_STRIDE_C,
            DY_STRIDE_D,
            DY_STRIDE_H,
            DY_STRIDE_W,
            X_STRIDE_N,
            X_STRIDE_C,
            X_STRIDE_D,
            X_STRIDE_H,
            X_STRIDE_W,
            W_STRIDE_K,
            W_STRIDE_C,
            W_STRIDE_D,
            W_STRIDE_H,
            W_STRIDE_W,
            INPUT_PRECISION,
            M,
            BLOCK_M,
            BLOCK_CI,
            BLOCK_K,
            GROUP_M,
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
    packed_stride2_2d: tl.constexpr = (
        XD == 1
        and OD == 1
        and KD == 1
        and KH == 3
        and KW == 3
        and STRIDE_D == 1
        and STRIDE_H == 2
        and STRIDE_W == 2
        and PAD_FRONT == 0
        and PAD_TOP == 1
        and PAD_LEFT == 1
        and DIL_D == 1
        and DIL_H == 1
        and DIL_W == 1
        and FLIP_FILTER == 0
    )
    packed_stride2_1d: tl.constexpr = (
        XD == 1
        and XH == 1
        and OD == 1
        and OH == 1
        and KD == 1
        and KH == 1
        and KW == 5
        and STRIDE_D == 1
        and STRIDE_H == 1
        and STRIDE_W == 2
        and PAD_FRONT == 0
        and PAD_TOP == 0
        and PAD_LEFT == 2
        and DIL_D == 1
        and DIL_H == 1
        and DIL_W == 1
        and FLIP_FILTER == 0
    )
    if packed_stride2_1d:
        tile = tl.program_id(0)
        group = tl.program_id(1).to(tl.int64)
        batch_count: tl.constexpr = M // XW
        base_width: tl.constexpr = tl.cdiv(XW, 2)
        base_rows: tl.constexpr = batch_count * base_width
        tiles_m: tl.constexpr = tl.cdiv(base_rows, BLOCK_M)
        tile_ci = tile // tiles_m
        tile_m = tile - tile_ci * tiles_m
        rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        columns = tl.arange(0, 2 * BLOCK_CI)
        parity = columns - (columns // 2) * 2
        input_channels = tile_ci * BLOCK_CI + columns // 2
        active_rows = rows < base_rows
        active_ci = input_channels < CIN_PER_GROUP
        batch = rows // base_width
        base_w = rows - batch * base_width
        accumulator = tl.zeros(
            (BLOCK_M, 2 * BLOCK_CI), dtype=tl.float32
        )

        for start in range(0, COUT_PER_GROUP, BLOCK_K):
            reduction = tl.arange(0, 4 * BLOCK_K)
            neighbor = reduction // BLOCK_K
            active_neighbor = neighbor < 3
            output_channels = start + reduction - neighbor * BLOCK_K
            neighbor_output_w = (
                base_w[:, None] + neighbor[None, :] - 1
            )
            active_co = output_channels < COUT_PER_GROUP
            losses = tl.load(
                dy_ptr
                + batch[:, None] * DY_STRIDE_N
                + (group * COUT_PER_GROUP + output_channels[None, :])
                * DY_STRIDE_C
                + neighbor_output_w * DY_STRIDE_W,
                mask=(
                    active_rows[:, None]
                    & active_co[None, :]
                    & active_neighbor[None, :]
                    & (neighbor_output_w >= 0)
                    & (neighbor_output_w < OW)
                ),
                other=0.0,
            )
            allowed = neighbor[:, None] >= parity[None, :]
            weight_w = (
                4
                + parity[None, :]
                - 2 * neighbor[:, None]
            )
            weights = tl.load(
                w_ptr
                + (group * COUT_PER_GROUP + output_channels[:, None])
                * W_STRIDE_K
                + input_channels[None, :] * W_STRIDE_C
                + weight_w * W_STRIDE_W,
                mask=(
                    active_co[:, None]
                    & active_ci[None, :]
                    & allowed
                    & active_neighbor[:, None]
                ),
                other=0.0,
            )
            accumulator = _conv_dgrad_dot(
                losses, weights, accumulator, INPUT_PRECISION
            )

        output_w = base_w[:, None] * 2 + parity[None, :]
        tl.store(
            dx_ptr
            + batch[:, None] * X_STRIDE_N
            + (group * CIN_PER_GROUP + input_channels[None, :])
            * X_STRIDE_C
            + output_w * X_STRIDE_W,
            accumulator.to(dx_ptr.dtype.element_ty),
            mask=(
                active_rows[:, None]
                & active_ci[None, :]
                & (output_w < XW)
            ),
        )
    elif packed_stride2_2d and CIN_PER_GROUP <= 4:
        tile = tl.program_id(0)
        group = tl.program_id(1).to(tl.int64)
        batch_count: tl.constexpr = M // (XH * XW)
        loss_rows: tl.constexpr = batch_count * OH * OW
        small_channels: tl.constexpr = CIN_PER_GROUP <= 4
        if small_channels:
            output_block: tl.constexpr = BLOCK_CI
            channels_per_tile: tl.constexpr = BLOCK_CI // 4
        else:
            output_block: tl.constexpr = 4 * BLOCK_CI
            channels_per_tile: tl.constexpr = BLOCK_CI
        tiles_m: tl.constexpr = tl.cdiv(loss_rows, BLOCK_M)
        tile_ci = tile // tiles_m
        tile_m = tile - tile_ci * tiles_m
        rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        columns = tl.arange(0, output_block)
        parity = columns - (columns // 4) * 4
        parity_h = parity // 2
        parity_w = parity - parity_h * 2
        input_channels = tile_ci * channels_per_tile + columns // 4
        active_rows = rows < loss_rows
        active_ci = input_channels < CIN_PER_GROUP
        loss_area: tl.constexpr = OH * OW
        batch = rows // loss_area
        loss_hw = rows - batch * loss_area
        loss_h = loss_hw // OW
        loss_w = loss_hw - loss_h * OW
        input_h0 = loss_h * 2
        input_w0 = loss_w * 2
        active_h1 = loss_h + 1 < OH
        active_w1 = loss_w + 1 < OW
        loss_origin = (
            dy_ptr
            + batch * DY_STRIDE_N
            + loss_h * DY_STRIDE_H
            + loss_w * DY_STRIDE_W
        )
        accumulator = tl.zeros((BLOCK_M, output_block), dtype=tl.float32)

        for start in range(0, COUT_PER_GROUP, BLOCK_K):
            reduction = tl.arange(0, 4 * BLOCK_K)
            neighbor = reduction // BLOCK_K
            neighbor_h = neighbor // 2
            neighbor_w = neighbor - neighbor_h * 2
            output_channels = start + reduction - neighbor * BLOCK_K
            active_co = output_channels < COUT_PER_GROUP
            losses = tl.load(
                loss_origin[:, None]
                + (group * COUT_PER_GROUP + output_channels[None, :])
                * DY_STRIDE_C
                + neighbor_h[None, :] * DY_STRIDE_H
                + neighbor_w[None, :] * DY_STRIDE_W,
                mask=(
                    active_rows[:, None]
                    & active_co[None, :]
                    & (
                        (neighbor_h[None, :] == 0)
                        | active_h1[:, None]
                    )
                    & (
                        (neighbor_w[None, :] == 0)
                        | active_w1[:, None]
                    )
                ),
                other=0.0,
            )
            allowed = (
                (neighbor_h[:, None] <= parity_h[None, :])
                & (neighbor_w[:, None] <= parity_w[None, :])
            )
            weight_h = (
                1
                + parity_h[None, :]
                - 2 * parity_h[None, :] * neighbor_h[:, None]
            )
            weight_w = (
                1
                + parity_w[None, :]
                - 2 * parity_w[None, :] * neighbor_w[:, None]
            )
            weights = tl.load(
                w_ptr
                + (group * COUT_PER_GROUP + output_channels[:, None])
                * W_STRIDE_K
                + input_channels[None, :] * W_STRIDE_C
                + weight_h * W_STRIDE_H
                + weight_w * W_STRIDE_W,
                mask=(
                    active_co[:, None]
                    & active_ci[None, :]
                    & allowed
                ),
                other=0.0,
            )
            accumulator = _conv_dgrad_dot(
                losses, weights, accumulator, INPUT_PRECISION
            )

        output_h = input_h0[:, None] + parity_h[None, :]
        output_w = input_w0[:, None] + parity_w[None, :]
        tl.store(
            dx_ptr
            + batch[:, None] * X_STRIDE_N
            + (group * CIN_PER_GROUP + input_channels[None, :])
            * X_STRIDE_C
            + output_h * X_STRIDE_H
            + output_w * X_STRIDE_W,
            accumulator.to(dx_ptr.dtype.element_ty),
            mask=(
                active_rows[:, None]
                & active_ci[None, :]
                & (output_h < XH)
                & (output_w < XW)
            ),
        )
    elif packed_stride2_2d:
        tile = tl.program_id(0)
        group = tl.program_id(1).to(tl.int64)
        batch_count: tl.constexpr = M // (XH * XW)
        loss_rows: tl.constexpr = batch_count * OH * OW
        tiles_m: tl.constexpr = tl.cdiv(loss_rows, BLOCK_M)
        tile_ci = tile // tiles_m
        tile_m = tile - tile_ci * tiles_m
        rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        input_channels = tile_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
        active_rows = rows < loss_rows
        active_ci = input_channels < CIN_PER_GROUP
        loss_area: tl.constexpr = OH * OW
        batch = rows // loss_area
        loss_hw = rows - batch * loss_area
        loss_h = loss_hw // OW
        loss_w = loss_hw - loss_h * OW
        input_h0 = loss_h * 2
        input_w0 = loss_w * 2
        input_h1 = input_h0 + 1
        input_w1 = input_w0 + 1
        active_h1 = loss_h + 1 < OH
        active_w1 = loss_w + 1 < OW
        loss_origin = (
            dy_ptr
            + batch * DY_STRIDE_N
            + loss_h * DY_STRIDE_H
            + loss_w * DY_STRIDE_W
        )
        weight_origin = w_ptr + input_channels * W_STRIDE_C
        output_base = (
            dx_ptr
            + batch[:, None] * X_STRIDE_N
            + (group * CIN_PER_GROUP + input_channels[None, :])
            * X_STRIDE_C
        )
        output_mask = active_rows[:, None] & active_ci[None, :]

        accumulator = _conv_dgrad_stride2_parity(
            loss_origin,
            weight_origin,
            active_rows,
            active_ci,
            active_h1,
            active_w1,
            group,
            COUT_PER_GROUP,
            DY_STRIDE_C,
            DY_STRIDE_H,
            DY_STRIDE_W,
            W_STRIDE_K,
            W_STRIDE_H,
            W_STRIDE_W,
            INPUT_PRECISION,
            BLOCK_M,
            BLOCK_CI,
            BLOCK_K,
            0,
        )
        tl.store(
            output_base
            + input_h0[:, None] * X_STRIDE_H
            + input_w0[:, None] * X_STRIDE_W,
            accumulator.to(dx_ptr.dtype.element_ty),
            mask=output_mask,
        )

        accumulator = _conv_dgrad_stride2_parity(
            loss_origin,
            weight_origin,
            active_rows,
            active_ci,
            active_h1,
            active_w1,
            group,
            COUT_PER_GROUP,
            DY_STRIDE_C,
            DY_STRIDE_H,
            DY_STRIDE_W,
            W_STRIDE_K,
            W_STRIDE_H,
            W_STRIDE_W,
            INPUT_PRECISION,
            BLOCK_M,
            BLOCK_CI,
            BLOCK_K,
            1,
        )
        tl.store(
            output_base
            + input_h0[:, None] * X_STRIDE_H
            + input_w1[:, None] * X_STRIDE_W,
            accumulator.to(dx_ptr.dtype.element_ty),
            mask=output_mask & (input_w1[:, None] < XW),
        )

        accumulator = _conv_dgrad_stride2_parity(
            loss_origin,
            weight_origin,
            active_rows,
            active_ci,
            active_h1,
            active_w1,
            group,
            COUT_PER_GROUP,
            DY_STRIDE_C,
            DY_STRIDE_H,
            DY_STRIDE_W,
            W_STRIDE_K,
            W_STRIDE_H,
            W_STRIDE_W,
            INPUT_PRECISION,
            BLOCK_M,
            BLOCK_CI,
            BLOCK_K,
            2,
        )
        tl.store(
            output_base
            + input_h1[:, None] * X_STRIDE_H
            + input_w0[:, None] * X_STRIDE_W,
            accumulator.to(dx_ptr.dtype.element_ty),
            mask=output_mask & (input_h1[:, None] < XH),
        )

        accumulator = _conv_dgrad_stride2_parity(
            loss_origin,
            weight_origin,
            active_rows,
            active_ci,
            active_h1,
            active_w1,
            group,
            COUT_PER_GROUP,
            DY_STRIDE_C,
            DY_STRIDE_H,
            DY_STRIDE_W,
            W_STRIDE_K,
            W_STRIDE_H,
            W_STRIDE_W,
            INPUT_PRECISION,
            BLOCK_M,
            BLOCK_CI,
            BLOCK_K,
            3,
        )
        tl.store(
            output_base
            + input_h1[:, None] * X_STRIDE_H
            + input_w1[:, None] * X_STRIDE_W,
            accumulator.to(dx_ptr.dtype.element_ty),
            mask=(
                output_mask
                & (input_h1[:, None] < XH)
                & (input_w1[:, None] < XW)
            ),
        )
    else:
        _conv_dgrad_nd_impl(
            dy_ptr, w_ptr, dx_ptr,
            XD, XH, XW, OD, OH, OW, KD, KH, KW,
            CIN_PER_GROUP, COUT_PER_GROUP,
            STRIDE_D, STRIDE_H, STRIDE_W,
            PAD_FRONT, PAD_TOP, PAD_LEFT,
            DIL_D, DIL_H, DIL_W, FLIP_FILTER,
            DY_STRIDE_N, DY_STRIDE_C, DY_STRIDE_D,
            DY_STRIDE_H, DY_STRIDE_W,
            X_STRIDE_N, X_STRIDE_C, X_STRIDE_D,
            X_STRIDE_H, X_STRIDE_W,
            W_STRIDE_K, W_STRIDE_C, W_STRIDE_D,
            W_STRIDE_H, W_STRIDE_W,
            INPUT_PRECISION, M, BLOCK_M, BLOCK_CI, BLOCK_K, GROUP_M,
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

    # Compute each cross-correlation gradient tile with its native sampling
    # coordinate, then reverse only the destination coordinate for mathematical
    # convolution. Keeping reversal out of the dot-product load address avoids
    # the failing MTGPU code shape while preserving identical dW semantics.
    weight_d = kernel_d + FLIP_FILTER * (KD - 1 - 2 * kernel_d)
    weight_h = kernel_h + FLIP_FILTER * (KH - 1 - 2 * kernel_h)
    weight_w = kernel_w + FLIP_FILTER * (KW - 1 - 2 * kernel_w)
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
        input_d = output_d * STRIDE_D - PAD_FRONT + kernel_d * DIL_D
        input_h = output_h * STRIDE_H - PAD_TOP + kernel_h * DIL_H
        input_w = output_w * STRIDE_W - PAD_LEFT + kernel_w * DIL_W
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
            mask=(output_channels[:, None] < COUT_PER_GROUP)
            & active_rows[None, :],
            other=0.0,
        )
        inputs = tl.load(
            x_ptr
            + batch[:, None] * X_STRIDE_N
            + (group * CIN_PER_GROUP + input_channels[None, :])
            * X_STRIDE_C
            + input_d[:, None] * X_STRIDE_D
            + input_h[:, None] * X_STRIDE_H
            + input_w[:, None] * X_STRIDE_W,
            mask=active_rows[:, None]
            & (input_channels[None, :] < CIN_PER_GROUP),
            other=0.0,
        )
        if INPUT_PRECISION == 1:
            accumulator += tl.dot(losses, inputs, input_precision="tf32")
        else:
            accumulator += tl.dot(losses, inputs, input_precision="ieee")

    tl.store(
        dw_ptr
        + (group * COUT_PER_GROUP + output_channels[:, None]) * W_STRIDE_K
        + input_channels[None, :] * W_STRIDE_C
        + weight_d * W_STRIDE_D
        + weight_h * W_STRIDE_H
        + weight_w * W_STRIDE_W,
        accumulator.to(dw_ptr.dtype.element_ty),
        mask=(output_channels[:, None] < COUT_PER_GROUP)
        & (input_channels[None, :] < CIN_PER_GROUP),
    )


@triton.jit
def _conv_wgrad2d_stem_split_kernel(
    dy_ptr,
    x_ptr,
    partial_ptr,
    OUTPUT_ROWS_PER_SPLIT: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    DY_STRIDE_C: tl.constexpr,
    DY_STRIDE_H: tl.constexpr,
    DY_STRIDE_W: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    PARTIAL_STRIDE_SPLIT: tl.constexpr,
    PARTIAL_STRIDE_OC: tl.constexpr,
    PARTIAL_STRIDE_K: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_CI_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    tile = tl.program_id(0)
    split = tl.program_id(1).to(tl.int64)
    reduction_extent: tl.constexpr = CIN_PER_GROUP * KH * KW
    tiles_k: tl.constexpr = tl.cdiv(reduction_extent, BLOCK_CI_K)
    tile_oc = tile // tiles_k
    tile_k = tile - tile_oc * tiles_k
    output_channels = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    packed_k = tile_k * BLOCK_CI_K + tl.arange(0, BLOCK_CI_K)
    input_channel = packed_k // (KH * KW)
    kernel_hw = packed_k - input_channel * (KH * KW)
    kernel_h = kernel_hw // KW
    kernel_w = kernel_hw - kernel_h * KW
    output_row_base = split * OUTPUT_ROWS_PER_SPLIT
    accumulator = tl.zeros((BLOCK_OC, BLOCK_CI_K), dtype=tl.float32)

    for local_h in tl.static_range(0, OUTPUT_ROWS_PER_SPLIT):
        output_h = output_row_base + local_h
        input_h = output_h * STRIDE_H - PAD_TOP + kernel_h * DIL_H
        for start_w in range(0, OW, BLOCK_M):
            output_w = start_w + tl.arange(0, BLOCK_M)
            active = (output_h < OH) & (output_w < OW)
            losses = tl.load(
                dy_ptr
                + output_channels[:, None] * DY_STRIDE_C
                + output_h * DY_STRIDE_H
                + output_w[None, :] * DY_STRIDE_W,
                mask=(output_channels[:, None] < COUT_PER_GROUP)
                & active[None, :],
                other=0.0,
            )
            input_w = (
                output_w[:, None] * STRIDE_W
                - PAD_LEFT
                + kernel_w[None, :] * DIL_W
            )
            inputs = tl.load(
                x_ptr
                + input_channel[None, :] * X_STRIDE_C
                + input_h[None, :] * X_STRIDE_H
                + input_w * X_STRIDE_W,
                mask=active[:, None]
                & (packed_k[None, :] < reduction_extent)
                & (input_h[None, :] >= 0)
                & (input_h[None, :] < XH)
                & (input_w >= 0)
                & (input_w < XW),
                other=0.0,
            )
            if DTYPE_ID == 0:
                accumulator += tl.dot(losses, inputs, input_precision="tf32")
            else:
                accumulator += tl.dot(losses, inputs, input_precision="ieee")

    tl.store(
        partial_ptr
        + split * PARTIAL_STRIDE_SPLIT
        + output_channels[:, None] * PARTIAL_STRIDE_OC
        + packed_k[None, :] * PARTIAL_STRIDE_K,
        accumulator,
        mask=(output_channels[:, None] < COUT_PER_GROUP)
        & (packed_k[None, :] < reduction_extent),
    )

@triton.jit
def _conv_wgrad2d_direct_split_kernel(
    dy_ptr,
    x_ptr,
    partial_ptr,
    TOTAL_ROWS: tl.constexpr,
    ROWS_PER_SPLIT: tl.constexpr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OW: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    DY_STRIDE_N: tl.constexpr,
    DY_STRIDE_C: tl.constexpr,
    DY_STRIDE_H: tl.constexpr,
    DY_STRIDE_W: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    PARTIAL_STRIDE_SPLIT: tl.constexpr,
    PARTIAL_STRIDE_OC: tl.constexpr,
    PARTIAL_STRIDE_K: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_CI_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    tile = tl.program_id(0)
    split = tl.program_id(1).to(tl.int64)
    reduction_extent: tl.constexpr = CIN_PER_GROUP * KH * KW
    tiles_k: tl.constexpr = tl.cdiv(reduction_extent, BLOCK_CI_K)
    tile_oc = tile // tiles_k
    tile_k = tile - tile_oc * tiles_k
    output_channels = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    packed_k = tile_k * BLOCK_CI_K + tl.arange(0, BLOCK_CI_K)
    input_channel = packed_k // (KH * KW)
    kernel_hw = packed_k - input_channel * (KH * KW)
    kernel_h = kernel_hw // KW
    kernel_w = kernel_hw - kernel_h * KW
    row_base = split * ROWS_PER_SPLIT
    accumulator = tl.zeros((BLOCK_OC, BLOCK_CI_K), dtype=tl.float32)

    for start in range(0, ROWS_PER_SPLIT, BLOCK_M):
        split_row = start + tl.arange(0, BLOCK_M)
        linear_row = row_base + split_row
        batch = linear_row // M
        output_m = linear_row - batch * M
        output_h = output_m // OW
        output_w = output_m - output_h * OW
        active = (split_row < ROWS_PER_SPLIT) & (linear_row < TOTAL_ROWS)
        losses = tl.load(
            dy_ptr
            + batch[None, :] * DY_STRIDE_N
            + output_channels[:, None] * DY_STRIDE_C
            + output_h[None, :] * DY_STRIDE_H
            + output_w[None, :] * DY_STRIDE_W,
            mask=(output_channels[:, None] < COUT_PER_GROUP)
            & active[None, :],
            other=0.0,
        )
        input_h = (
            output_h[:, None] * STRIDE_H
            - PAD_TOP
            + kernel_h[None, :] * DIL_H
        )
        input_w = (
            output_w[:, None] * STRIDE_W
            - PAD_LEFT
            + kernel_w[None, :] * DIL_W
        )
        inputs = tl.load(
            x_ptr
            + batch[:, None] * X_STRIDE_N
            + input_channel[None, :] * X_STRIDE_C
            + input_h * X_STRIDE_H
            + input_w * X_STRIDE_W,
            mask=active[:, None]
            & (packed_k[None, :] < reduction_extent)
            & (input_h >= 0)
            & (input_h < XH)
            & (input_w >= 0)
            & (input_w < XW),
            other=0.0,
        )
        if DTYPE_ID == 0:
            accumulator += tl.dot(losses, inputs, input_precision="tf32")
        else:
            accumulator += tl.dot(losses, inputs, input_precision="ieee")

    tl.store(
        partial_ptr
        + split * PARTIAL_STRIDE_SPLIT
        + output_channels[:, None] * PARTIAL_STRIDE_OC
        + packed_k[None, :] * PARTIAL_STRIDE_K,
        accumulator,
        mask=(output_channels[:, None] < COUT_PER_GROUP)
        & (packed_k[None, :] < reduction_extent),
    )


@triton.jit
def _conv_wgrad2d_1x1_split_kernel(
    dy_ptr,
    x_ptr,
    partial_ptr,
    TOTAL_ROWS: tl.constexpr,
    ROWS_PER_SPLIT: tl.constexpr,
    HW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    PARTIAL_STRIDE_SPLIT: tl.constexpr,
    PARTIAL_STRIDE_OC: tl.constexpr,
    PARTIAL_STRIDE_K: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    tile = tl.program_id(0)
    split_group = tl.program_id(1).to(tl.int64)
    split = split_group // GROUPS
    group = split_group - split * GROUPS
    tiles_ci: tl.constexpr = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    tile_oc = tile // tiles_ci
    tile_ci = tile - tile_oc * tiles_ci
    output_channels = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    input_channels = tile_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    row_base = split * ROWS_PER_SPLIT
    accumulator = tl.zeros((BLOCK_OC, BLOCK_CI), dtype=tl.float32)

    for start in range(0, ROWS_PER_SPLIT, BLOCK_M):
        split_row = start + tl.arange(0, BLOCK_M)
        linear_row = row_base + split_row
        batch = linear_row // HW
        spatial = linear_row - batch * HW
        global_oc = group * COUT_PER_GROUP + output_channels
        global_ci = group * CIN_PER_GROUP + input_channels
        active = (split_row < ROWS_PER_SPLIT) & (linear_row < TOTAL_ROWS)
        losses = tl.load(
            dy_ptr
            + batch[None, :] * (C_OUT * HW)
            + global_oc[:, None] * HW
            + spatial[None, :],
            mask=(output_channels[:, None] < COUT_PER_GROUP)
            & active[None, :],
            other=0.0,
        )
        inputs = tl.load(
            x_ptr
            + batch[:, None] * (C_IN * HW)
            + global_ci[None, :] * HW
            + spatial[:, None],
            mask=active[:, None]
            & (input_channels[None, :] < CIN_PER_GROUP),
            other=0.0,
        )
        if DTYPE_ID == 0:
            accumulator += tl.dot(losses, inputs, input_precision="tf32")
        else:
            accumulator += tl.dot(losses, inputs, input_precision="ieee")

    global_oc = group * COUT_PER_GROUP + output_channels
    tl.store(
        partial_ptr
        + split * PARTIAL_STRIDE_SPLIT
        + global_oc[:, None] * PARTIAL_STRIDE_OC
        + input_channels[None, :] * PARTIAL_STRIDE_K,
        accumulator,
        mask=(output_channels[:, None] < COUT_PER_GROUP)
        & (input_channels[None, :] < CIN_PER_GROUP),
    )



@triton.jit
def _conv_wgrad2d_stem_reduce_kernel(
    partial_ptr,
    dw_ptr,
    TOTAL: tl.constexpr,
    CIK: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    PARTIAL_STRIDE_SPLIT: tl.constexpr,
    PARTIAL_STRIDE_OC: tl.constexpr,
    PARTIAL_STRIDE_K: tl.constexpr,
    W_STRIDE_K: tl.constexpr,
    W_STRIDE_C: tl.constexpr,
    W_STRIDE_H: tl.constexpr,
    W_STRIDE_W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_channel = linear // CIK
    packed_k = linear - output_channel * CIK
    active = linear < TOTAL
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for split in tl.static_range(0, NUM_SPLITS):
        accumulator += tl.load(
            partial_ptr
            + split * PARTIAL_STRIDE_SPLIT
            + output_channel * PARTIAL_STRIDE_OC
            + packed_k * PARTIAL_STRIDE_K,
            mask=active,
            other=0.0,
        )

    input_channel = packed_k // (KH * KW)
    kernel_hw = packed_k - input_channel * (KH * KW)
    kernel_h = kernel_hw // KW
    kernel_w = kernel_hw - kernel_h * KW
    tl.store(
        dw_ptr
        + output_channel * W_STRIDE_K
        + input_channel * W_STRIDE_C
        + kernel_h * W_STRIDE_H
        + kernel_w * W_STRIDE_W,
        accumulator.to(dw_ptr.dtype.element_ty),
        mask=active & (input_channel < CIN_PER_GROUP),
    )


@triton.jit
def _conv_wgrad2d_im2row_kernel(
    x_ptr,
    col_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OW: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    COL_STRIDE_R: tl.constexpr,
    COL_OFFSET: tl.constexpr,
    COL_STRIDE_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    tiles_m: tl.constexpr = tl.cdiv(M, BLOCK_M)
    tile_k = tile // tiles_m
    tile_m = tile - tile_k * tiles_m
    output_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    packed_k = tile_k * BLOCK_K + tl.arange(0, BLOCK_K)
    reduction_extent: tl.constexpr = CIN_PER_GROUP * KH * KW
    output_h = output_m // OW
    output_w = output_m - output_h * OW
    input_channel = packed_k // (KH * KW)
    kernel_hw = packed_k - input_channel * (KH * KW)
    kernel_h = kernel_hw // KW
    kernel_w = kernel_hw - kernel_h * KW
    input_h = (
        output_h[:, None] * STRIDE_H
        - PAD_TOP
        + kernel_h[None, :] * DIL_H
    )
    input_w = (
        output_w[:, None] * STRIDE_W
        - PAD_LEFT
        + kernel_w[None, :] * DIL_W
    )
    active = (
        (output_m[:, None] < M)
        & (packed_k[None, :] < reduction_extent)
    )
    valid = (
        active
        & (input_h >= 0)
        & (input_h < XH)
        & (input_w >= 0)
        & (input_w < XW)
    )
    values = tl.load(
        x_ptr
        + batch * X_STRIDE_N
        + input_channel[None, :] * X_STRIDE_C
        + input_h * X_STRIDE_H
        + input_w * X_STRIDE_W,
        mask=valid,
        other=0.0,
    )
    rows = batch * M + output_m
    tl.store(
        col_ptr
        + COL_OFFSET
        + rows[:, None] * COL_STRIDE_R
        + packed_k[None, :] * COL_STRIDE_K,
        values,
        mask=active,
    )


@triton.jit
def _conv_wgrad2d_rowmajor_kernel(
    dy_ptr,
    col_ptr,
    partial_ptr,
    M: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    DY_STRIDE_N: tl.constexpr,
    DY_STRIDE_C: tl.constexpr,
    DY_STRIDE_H: tl.constexpr,
    DY_STRIDE_W: tl.constexpr,
    COL_OFFSET: tl.constexpr,
    COL_STRIDE_R: tl.constexpr,
    COL_STRIDE_K: tl.constexpr,
    PARTIAL_STRIDE_SPLIT: tl.constexpr,
    PARTIAL_STRIDE_OC: tl.constexpr,
    PARTIAL_STRIDE_K: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_CI_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    tile = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    reduction_extent: tl.constexpr = CIN_PER_GROUP * KH * KW
    tiles_k: tl.constexpr = tl.cdiv(reduction_extent, BLOCK_CI_K)
    tile_oc = tile // tiles_k
    tile_k = tile - tile_oc * tiles_k
    output_channels = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    packed_k = tile_k * BLOCK_CI_K + tl.arange(0, BLOCK_CI_K)
    accumulator = tl.zeros((BLOCK_OC, BLOCK_CI_K), dtype=tl.float32)

    reduction_block: tl.constexpr = 256 if BLOCK_M == 32 else 128
    for start in range(0, M, reduction_block):
        output_m = start + tl.arange(0, reduction_block)
        active = output_m < M
        losses = tl.load(
            dy_ptr
            + batch * DY_STRIDE_N
            + output_channels[:, None] * DY_STRIDE_C
            + output_m[None, :] * DY_STRIDE_W,
            mask=(output_channels[:, None] < COUT_PER_GROUP)
            & active[None, :],
            other=0.0,
        )
        linear_row = batch * M + output_m
        columns = tl.load(
            col_ptr
            + COL_OFFSET
            + linear_row[:, None] * COL_STRIDE_R
            + packed_k[None, :] * COL_STRIDE_K,
            mask=active[:, None]
            & (packed_k[None, :] < reduction_extent),
            other=0.0,
        )
        if DTYPE_ID == 0:
            accumulator += tl.dot(losses, columns, input_precision="tf32")
        else:
            accumulator += tl.dot(losses, columns, input_precision="ieee")

    tl.store(
        partial_ptr
        + batch * PARTIAL_STRIDE_SPLIT
        + output_channels[:, None] * PARTIAL_STRIDE_OC
        + packed_k[None, :] * PARTIAL_STRIDE_K,
        accumulator,
        mask=(output_channels[:, None] < COUT_PER_GROUP)
        & (packed_k[None, :] < reduction_extent),
    )


@triton.jit
def _conv_wgrad_nd_im2row_kernel(
    x_ptr,
    col_ptr,
    M: tl.constexpr,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_D: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    COL_STRIDE_R: tl.constexpr,
    COL_OFFSET: tl.constexpr,
    COL_STRIDE_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    tiles_m: tl.constexpr = tl.cdiv(M, BLOCK_M)
    tile_k = tile // tiles_m
    tile_m = tile - tile_k * tiles_m
    output_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    packed_k = tile_k * BLOCK_K + tl.arange(0, BLOCK_K)
    reduction_extent: tl.constexpr = CIN_PER_GROUP * KD * KH * KW

    output_plane: tl.constexpr = OH * OW
    output_d = output_m // output_plane
    output_hw = output_m - output_d * output_plane
    output_h = output_hw // OW
    output_w = output_hw - output_h * OW
    kernel_volume: tl.constexpr = KD * KH * KW
    input_channel = packed_k // kernel_volume
    kernel_linear = packed_k - input_channel * kernel_volume
    kernel_plane: tl.constexpr = KH * KW
    kernel_d = kernel_linear // kernel_plane
    kernel_hw = kernel_linear - kernel_d * kernel_plane
    kernel_h = kernel_hw // KW
    kernel_w = kernel_hw - kernel_h * KW
    input_d = output_d[:, None] * STRIDE_D - PAD_D + kernel_d[None, :] * DIL_D
    input_h = output_h[:, None] * STRIDE_H - PAD_H + kernel_h[None, :] * DIL_H
    input_w = output_w[:, None] * STRIDE_W - PAD_W + kernel_w[None, :] * DIL_W
    active = (
        (output_m[:, None] < M)
        & (packed_k[None, :] < reduction_extent)
    )
    valid = (
        active
        & (input_d >= 0)
        & (input_d < XD)
        & (input_h >= 0)
        & (input_h < XH)
        & (input_w >= 0)
        & (input_w < XW)
    )
    values = tl.load(
        x_ptr
        + batch * X_STRIDE_N
        + input_channel[None, :] * X_STRIDE_C
        + input_d * X_STRIDE_D
        + input_h * X_STRIDE_H
        + input_w * X_STRIDE_W,
        mask=valid,
        other=0.0,
    )
    rows = batch * M + output_m
    tl.store(
        col_ptr
        + COL_OFFSET
        + rows[:, None] * COL_STRIDE_R
        + packed_k[None, :] * COL_STRIDE_K,
        values,
        mask=active,
    )


@triton.jit
def _conv_wgrad_nd_rowmajor_kernel(
    dy_ptr,
    col_ptr,
    partial_ptr,
    TOTAL_ROWS: tl.constexpr,
    ROWS_PER_SPLIT: tl.constexpr,
    M: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    REDUCTION_EXTENT: tl.constexpr,
    DY_STRIDE_N: tl.constexpr,
    DY_STRIDE_C: tl.constexpr,
    DY_STRIDE_SPATIAL: tl.constexpr,
    COL_STRIDE_R: tl.constexpr,
    COL_OFFSET: tl.constexpr,
    COL_STRIDE_K: tl.constexpr,
    PARTIAL_STRIDE_SPLIT: tl.constexpr,
    PARTIAL_STRIDE_OC: tl.constexpr,
    PARTIAL_STRIDE_K: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_CI_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    tile = tl.program_id(0)
    split = tl.program_id(1).to(tl.int64)
    tiles_k: tl.constexpr = tl.cdiv(REDUCTION_EXTENT, BLOCK_CI_K)
    tile_oc = tile // tiles_k
    tile_k = tile - tile_oc * tiles_k
    output_channels = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    packed_k = tile_k * BLOCK_CI_K + tl.arange(0, BLOCK_CI_K)
    accumulator = tl.zeros((BLOCK_OC, BLOCK_CI_K), dtype=tl.float32)

    reduction_block: tl.constexpr = 256 if BLOCK_M == 32 else 128
    row_start = split * ROWS_PER_SPLIT
    for start in range(0, ROWS_PER_SPLIT, reduction_block):
        split_row = start + tl.arange(0, reduction_block)
        linear_row = row_start + split_row
        batch = linear_row // M
        output_m = linear_row - batch * M
        active = (split_row < ROWS_PER_SPLIT) & (linear_row < TOTAL_ROWS)
        losses = tl.load(
            dy_ptr
            + batch[None, :] * DY_STRIDE_N
            + output_channels[:, None] * DY_STRIDE_C
            + output_m[None, :] * DY_STRIDE_SPATIAL,
            mask=(output_channels[:, None] < COUT_PER_GROUP)
            & active[None, :],
            other=0.0,
        )
        columns = tl.load(
            col_ptr
            + COL_OFFSET
            + linear_row[:, None] * COL_STRIDE_R
            + packed_k[None, :] * COL_STRIDE_K,
            mask=active[:, None]
            & (packed_k[None, :] < REDUCTION_EXTENT),
            other=0.0,
        )
        if DTYPE_ID == 0:
            accumulator += tl.dot(losses, columns, input_precision="tf32")
        else:
            accumulator += tl.dot(losses, columns, input_precision="ieee")

    tl.store(
        partial_ptr
        + split * PARTIAL_STRIDE_SPLIT
        + output_channels[:, None] * PARTIAL_STRIDE_OC
        + packed_k[None, :] * PARTIAL_STRIDE_K,
        accumulator,
        mask=(output_channels[:, None] < COUT_PER_GROUP)
        & (packed_k[None, :] < REDUCTION_EXTENT),
    )


@triton.jit
def _conv_wgrad_nd_reduce_kernel(
    partial_ptr,
    dw_ptr,
    TOTAL: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    PARTIAL_STRIDE_SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < TOTAL
    accumulator = tl.zeros((BLOCK,), dtype=tl.float32)
    for split in range(0, NUM_SPLITS):
        accumulator += tl.load(
            partial_ptr + split * PARTIAL_STRIDE_SPLIT + offsets,
            mask=active,
            other=0.0,
        )
    tl.store(
        dw_ptr + offsets,
        accumulator.to(dw_ptr.dtype.element_ty),
        mask=active,
    )


@triton.jit
def _conv_wgrad2d_p5_pack_image_kernel(
    image_ptr,
    packed_ptr,
    CIN_PER_GROUP: tl.constexpr,
    IMAGE_STRIDE_C: tl.constexpr,
    IMAGE_STRIDE_H: tl.constexpr,
    IMAGE_STRIDE_W: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    tile_m = tl.program_id(0)
    tile_n = tl.program_id(1)
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    active_rows = rows < M
    active_columns = columns < N

    output_h = rows // 20
    output_w = rows % 20
    input_channel = columns // 9
    kernel_spatial = columns % 9
    kernel_h = kernel_spatial // 3
    kernel_w = kernel_spatial % 3
    safe_channel = tl.where(active_columns, input_channel, 0)
    input_h = output_h[:, None] * 2 - 1 + kernel_h[None, :]
    input_w = output_w[:, None] * 2 - 1 + kernel_w[None, :]
    valid = (
        active_rows[:, None]
        & active_columns[None, :]
        & (input_h >= 0)
        & (input_h < 40)
        & (input_w >= 0)
        & (input_w < 40)
    )
    safe_h = tl.where(valid, input_h, 0)
    safe_w = tl.where(valid, input_w, 0)
    values = tl.load(
        image_ptr
        + safe_channel[None, :] * IMAGE_STRIDE_C
        + safe_h * IMAGE_STRIDE_H
        + safe_w * IMAGE_STRIDE_W,
        mask=valid,
        other=0.0,
    )
    tl.store(
        packed_ptr + rows[:, None] * N + columns[None, :],
        values,
        mask=active_rows[:, None] & active_columns[None, :],
    )


@triton.jit
def _conv_wgrad2d_p5_mm_kernel(
    loss_ptr,
    packed_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    DTYPE_ID: tl.constexpr,
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
    for start in range(0, K, BLOCK_K):
        reduction_offsets = start + reduction
        loss = tl.load(
            loss_ptrs,
            mask=(rows[:, None] < M)
            & (reduction_offsets[None, :] < K),
            other=0.0,
        )
        packed = tl.load(
            packed_ptrs,
            mask=(reduction_offsets[:, None] < K)
            & (columns[None, :] < N),
            other=0.0,
        )
        if DTYPE_ID == 0:
            accumulator += tl.dot(loss, packed, input_precision="tf32")
        else:
            accumulator += tl.dot(loss, packed, input_precision="ieee")
        loss_ptrs += BLOCK_K
        packed_ptrs += BLOCK_K * N

    tl.store(
        output_ptr + rows[:, None] * N + columns[None, :],
        accumulator.to(output_ptr.dtype.element_ty),
        mask=(rows[:, None] < M) & (columns[None, :] < N),
    )
