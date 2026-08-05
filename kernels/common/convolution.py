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
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    input_channels = tile_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    input_volume: tl.constexpr = XD * XH * XW
    batch = rows // input_volume
    spatial = rows % input_volume
    input_d = spatial // (XH * XW)
    input_hw = spatial % (XH * XW)
    input_h = input_hw // XW
    input_w = input_hw % XW
    kernel_volume: tl.constexpr = KD * KH * KW
    reduction_extent: tl.constexpr = COUT_PER_GROUP * kernel_volume
    reduction_base = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for start in range(0, reduction_extent, BLOCK_K):
        reduction = start + reduction_base
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
            mask=(output_channels[:, None] < COUT_PER_GROUP)
            & active_rows[None, :],
            other=0.0,
        )
        inputs = tl.load(
            x_ptr
            + batch[:, None] * X_STRIDE_N
            + (group * CIN_PER_GROUP + input_channels[None, :]) * X_STRIDE_C
            + input_d[:, None] * X_STRIDE_D
            + input_h[:, None] * X_STRIDE_H
            + input_w[:, None] * X_STRIDE_W,
            mask=active_rows[:, None]
            & (input_channels[None, :] < CIN_PER_GROUP),
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


# -----------------------------------------------------------------------------
# Kernel algorithm variants. Native dispatch selects only registry-declared
# entry points; auxiliary kernels remain available to explicit compiler policy.
# -----------------------------------------------------------------------------


@triton.jit
def conv1d_depthwise_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    XL,
    OL,
    C_IN,
    DTYPE_ID,
    x_stride_n,
    x_stride_c,
    x_stride_l,
    w_stride_o,
    w_stride_k,
    bias_stride,
    y_stride_n,
    y_stride_c,
    y_stride_l,
    STRIDE_W: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_W: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_n = tl.program_id(2)

    # tl.assume(x_stride_c > 0)
    # tl.assume(x_stride_l > 0)
    # tl.assume(w_stride_o > 0)
    # tl.assume(w_stride_k > 0)
    # tl.assume(y_stride_c > 0)
    # tl.assume(y_stride_l > 0)

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_l = offs_l < OL
    mask_c = offs_c < C_IN

    acc = tl.zeros((BLOCK_C, BLOCK_L), dtype=tl.float32)
    x_base = x_ptr + pid_n * x_stride_n
    y_base = y_ptr + pid_n * y_stride_n

    for kw in tl.static_range(0, KW):
        iw = offs_l * STRIDE_W - PAD_LEFT + kw * DIL_W
        valid_l = mask_l & (iw >= 0) & (iw < XL)
        x = tl.load(
            x_base + offs_c[:, None] * x_stride_c + iw[None, :] * x_stride_l,
            mask=mask_c[:, None] & valid_l[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_c * w_stride_o + kw * w_stride_k,
            mask=mask_c,
            other=0.0,
        )
        acc += x * w[:, None]

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_c * bias_stride, mask=mask_c, other=0.0)
        acc += bias[:, None]

    if ACTIVATION == "silu":
        acc *= tl.sigmoid(acc)

    tl.store(
        y_base + offs_c[:, None] * y_stride_c + offs_l[None, :] * y_stride_l,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_c[:, None] & mask_l[None, :],
    )


@triton.jit
def conv1d_conv1d_gemm_kernel_variant(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
    XL,
    OL,
    DTYPE_ID,
    x_stride_n,
    x_stride_c,
    x_stride_l,
    w_stride_o,
    w_stride_i,
    w_stride_k,
    bias_stride,
    y_stride_n,
    y_stride_c,
    y_stride_l,
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
):
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)

    # tl.assume(x_stride_c > 0)
    # tl.assume(x_stride_l > 0)
    # tl.assume(w_stride_o > 0)
    # tl.assume(w_stride_i > 0)
    # tl.assume(w_stride_k > 0)
    # tl.assume(y_stride_c > 0)
    # tl.assume(y_stride_l > 0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_m = offs_m < M
    mask_oc = offs_oc < COUT_PER_GROUP

    batch = offs_m // OL
    ow = offs_m % OL
    oc_global = pid_g * COUT_PER_GROUP + offs_oc

    acc = tl.zeros((BLOCK_M, BLOCK_OC), dtype=tl.float32)
    k_total: tl.constexpr = CIN_PER_GROUP * KW

    for k0 in range(0, k_total, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        ci = offs_k // KW
        kw = offs_k % KW
        mask_k = offs_k < k_total
        ic_global = pid_g * CIN_PER_GROUP + ci
        iw = ow[:, None] * STRIDE_W - PAD_LEFT + kw[None, :] * DIL_W
        valid_x = mask_m[:, None] & mask_k[None, :] & (iw >= 0) & (iw < XL)

        x = tl.load(
            x_ptr
            + batch[:, None] * x_stride_n
            + ic_global[None, :] * x_stride_c
            + iw * x_stride_l,
            mask=valid_x,
            other=0.0,
        )
        w = tl.load(
            w_ptr
            + oc_global[None, :] * w_stride_o
            + ci[:, None] * w_stride_i
            + kw[:, None] * w_stride_k,
            mask=mask_k[:, None] & mask_oc[None, :],
            other=0.0,
        )
        acc = tl.dot(x, w, acc)

    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + oc_global * bias_stride, mask=mask_oc, other=0.0
        )
        acc += bias[None, :]

    tl.store(
        y_ptr
        + batch[:, None] * y_stride_n
        + oc_global[None, :] * y_stride_c
        + ow[:, None] * y_stride_l,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_oc[None, :],
    )


@triton.jit
def conv1d_general_fp64_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    total_elements,
    XL,
    OL,
    C_OUT,
    x_stride_n,
    x_stride_c,
    x_stride_l,
    w_stride_o,
    w_stride_i,
    w_stride_k,
    bias_stride,
    y_stride_n,
    y_stride_c,
    y_stride_l,
    COUT_PER_GROUP: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    ow = offsets % OL
    oc = (offsets // OL) % C_OUT
    batch = offsets // (C_OUT * OL)
    group = oc // COUT_PER_GROUP

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + oc * bias_stride, mask=mask, other=0.0).to(
            tl.float64
        )

    for kw in tl.static_range(0, KW):
        iw = ow * STRIDE_W - PAD_LEFT + kw * DIL_W
        valid = mask & (iw >= 0) & (iw < XL)
        for ci in tl.static_range(0, CIN_PER_GROUP):
            ic = group * CIN_PER_GROUP + ci
            x = tl.load(
                x_ptr + batch * x_stride_n + ic * x_stride_c + iw * x_stride_l,
                mask=valid,
                other=0.0,
            ).to(tl.float64)
            weight = tl.load(
                w_ptr + oc * w_stride_o + ci * w_stride_i + kw * w_stride_k,
                mask=mask,
                other=0.0,
            ).to(tl.float64)
            acc += x * weight

    tl.store(
        y_ptr + batch * y_stride_n + oc * y_stride_c + ow * y_stride_l,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _conv2d_winograd_f2_weight_transform_3x3_kernel(
    w_ptr,
    u_ptr,
    TOTAL: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    mask = offs < TOTAL

    group_area = COUT_PER_GROUP * CIN_PER_GROUP
    g = offs // group_area
    rem = offs - g * group_area
    oc = rem // CIN_PER_GROUP
    ic = rem - oc * CIN_PER_GROUP
    oc_global = g * COUT_PER_GROUP + oc

    base = ((oc_global * CIN_PER_GROUP + ic) * 3) * 3
    g00 = tl.load(w_ptr + base + 0, mask=mask, other=0.0)
    g01 = tl.load(w_ptr + base + 1, mask=mask, other=0.0)
    g02 = tl.load(w_ptr + base + 2, mask=mask, other=0.0)
    g10 = tl.load(w_ptr + base + 3, mask=mask, other=0.0)
    g11 = tl.load(w_ptr + base + 4, mask=mask, other=0.0)
    g12 = tl.load(w_ptr + base + 5, mask=mask, other=0.0)
    g20 = tl.load(w_ptr + base + 6, mask=mask, other=0.0)
    g21 = tl.load(w_ptr + base + 7, mask=mask, other=0.0)
    g22 = tl.load(w_ptr + base + 8, mask=mask, other=0.0)

    half = 0.5
    t00 = g00
    t01 = g01
    t02 = g02
    t10 = (g00 + g10 + g20) * half
    t11 = (g01 + g11 + g21) * half
    t12 = (g02 + g12 + g22) * half
    t20 = (g00 - g10 + g20) * half
    t21 = (g01 - g11 + g21) * half
    t22 = (g02 - g12 + g22) * half
    t30 = g20
    t31 = g21
    t32 = g22

    u00 = t00
    u01 = (t00 + t01 + t02) * half
    u02 = (t00 - t01 + t02) * half
    u03 = t02
    u10 = t10
    u11 = (t10 + t11 + t12) * half
    u12 = (t10 - t11 + t12) * half
    u13 = t12
    u20 = t20
    u21 = (t20 + t21 + t22) * half
    u22 = (t20 - t21 + t22) * half
    u23 = t22
    u30 = t30
    u31 = (t30 + t31 + t32) * half
    u32 = (t30 - t31 + t32) * half
    u33 = t32

    out_base = (g * COUT_PER_GROUP + oc) * CIN_PER_GROUP + ic
    stride_p = GROUPS * COUT_PER_GROUP * CIN_PER_GROUP
    tl.store(u_ptr + 0 * stride_p + out_base, u00, mask=mask)
    tl.store(u_ptr + 1 * stride_p + out_base, u01, mask=mask)
    tl.store(u_ptr + 2 * stride_p + out_base, u02, mask=mask)
    tl.store(u_ptr + 3 * stride_p + out_base, u03, mask=mask)
    tl.store(u_ptr + 4 * stride_p + out_base, u10, mask=mask)
    tl.store(u_ptr + 5 * stride_p + out_base, u11, mask=mask)
    tl.store(u_ptr + 6 * stride_p + out_base, u12, mask=mask)
    tl.store(u_ptr + 7 * stride_p + out_base, u13, mask=mask)
    tl.store(u_ptr + 8 * stride_p + out_base, u20, mask=mask)
    tl.store(u_ptr + 9 * stride_p + out_base, u21, mask=mask)
    tl.store(u_ptr + 10 * stride_p + out_base, u22, mask=mask)
    tl.store(u_ptr + 11 * stride_p + out_base, u23, mask=mask)
    tl.store(u_ptr + 12 * stride_p + out_base, u30, mask=mask)
    tl.store(u_ptr + 13 * stride_p + out_base, u31, mask=mask)
    tl.store(u_ptr + 14 * stride_p + out_base, u32, mask=mask)
    tl.store(u_ptr + 15 * stride_p + out_base, u33, mask=mask)


@triton.jit
def conv2d_spatial_nchw_3x3_stride2_pad1_im2col_kernel(
    x_ptr,
    col_ptr,
    TOTAL: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    HW = OH * OW
    kernel_area = 9

    k = offsets // HW
    hw = offsets - k * HW
    ic = k // kernel_area
    rem = k - ic * kernel_area
    kh = rem // 3
    kw = rem - kh * 3

    oh_o = hw // OW
    ow_o = hw - oh_o * OW
    ih = oh_o * 2 - 1 + kh
    iw = ow_o * 2 - 1 + kw

    in_range = offsets < TOTAL
    valid = in_range & (ih >= 0) & (ih < XH) & (iw >= 0) & (iw < XW)
    x = tl.load(
        x_ptr + ic * (XH * XW) + ih * XW + iw,
        mask=valid,
        other=0.0,
    )
    tl.store(col_ptr + offsets, x, mask=in_range)


@triton.jit
def conv2d_1x1_nchw_kernel(
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
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DTYPE_ID: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_bg = tl.program_id(1)

    batch_idx = pid_bg // GROUPS
    group_idx = pid_bg - batch_idx * GROUPS

    HW = OH * OW
    num_pid_m = tl.cdiv(HW, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_hw = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k = tl.arange(0, BLOCK_K)

    mask_hw = offs_hw < HW
    mask_oc = offs_oc < COUT_PER_GROUP

    oh = offs_hw // OW
    ow = offs_hw - oh * OW
    ih = oh * STRIDE_H - PAD_TOP
    iw = ow * STRIDE_W - PAD_LEFT
    valid_hw = mask_hw & (ih >= 0) & (ih < XH) & (iw >= 0) & (iw < XW)

    x_batch_base = batch_idx * (C_IN * XH * XW)
    y_batch_base = batch_idx * (C_OUT * HW)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for k0 in range(0, CIN_PER_GROUP, BLOCK_K):
        ic_local = k0 + offs_k
        mask_k = ic_local < CIN_PER_GROUP
        ic_global = group_idx * CIN_PER_GROUP + ic_local

        x_ptrs = (
            x_ptr
            + x_batch_base
            + ic_global[:, None] * (XH * XW)
            + ih[None, :] * XW
            + iw[None, :]
        )
        x = tl.load(
            x_ptrs, mask=mask_k[:, None] & valid_hw[None, :], other=0.0
        )

        # Packed [G, CoutG, CinG]
        w_ptrs = (
            w_ptr
            + (group_idx * COUT_PER_GROUP + offs_oc[:, None]) * CIN_PER_GROUP
            + ic_local[None, :]
        )
        w = tl.load(w_ptrs, mask=mask_oc[:, None] & mask_k[None, :], other=0.0)
        acc = tl.dot(w, x, acc, input_precision="tf32")

    oc_global = group_idx * COUT_PER_GROUP + offs_oc
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[:, None]

    y_ptrs = y_ptr + y_batch_base + oc_global[:, None] * HW + offs_hw[None, :]
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_hw[None, :],
    )


@triton.jit
def conv2d_1x1_nchw_pad0_kernel(
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
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DTYPE_ID: tl.constexpr,
):
    # Specialized NCHW 1x1, stride=1, padding=0.  It removes all oh/ow -> ih/iw
    # address arithmetic and boundary predicates from the hot path.  The weak
    # 1x1 benchmark cases are all in this form.
    pid = tl.program_id(0)
    pid_bg = tl.program_id(1)

    batch_idx = pid_bg // GROUPS
    group_idx = pid_bg - batch_idx * GROUPS

    num_pid_m = tl.cdiv(HW, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_hw = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_hw = offs_hw < HW
    mask_oc = offs_oc < COUT_PER_GROUP

    x_batch_base = batch_idx * (C_IN * HW)
    y_batch_base = batch_idx * (C_OUT * HW)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for k0 in range(0, CIN_PER_GROUP, BLOCK_K):
        ic_local = k0 + offs_k_base
        mask_k = ic_local < CIN_PER_GROUP
        ic_global = group_idx * CIN_PER_GROUP + ic_local

        x = tl.load(
            x_ptr + x_batch_base + ic_global[:, None] * HW + offs_hw[None, :],
            mask=mask_k[:, None] & mask_hw[None, :],
            other=0.0,
        )

        w = tl.load(
            w_ptr
            + (group_idx * COUT_PER_GROUP + offs_oc[:, None]) * CIN_PER_GROUP
            + ic_local[None, :],
            mask=mask_oc[:, None] & mask_k[None, :],
            other=0.0,
        )
        acc = tl.dot(w, x, acc, input_precision="tf32")

    oc_global = group_idx * COUT_PER_GROUP + offs_oc
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[:, None]

    tl.store(
        y_ptr + y_batch_base + oc_global[:, None] * HW + offs_hw[None, :],
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_hw[None, :],
    )


@triton.jit
def conv2d_1x1_nchw_m_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
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
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DTYPE_ID: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)

    HW = OH * OW

    num_pid_m = tl.cdiv(M, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_oc = offs_oc < COUT_PER_GROUP

    batch_idx = offs_m // HW
    hw = offs_m - batch_idx * HW
    oh = hw // OW
    ow = hw - oh * OW

    ih = oh * STRIDE_H - PAD_TOP
    iw = ow * STRIDE_W - PAD_LEFT
    valid_hw = mask_m & (ih >= 0) & (ih < XH) & (iw >= 0) & (iw < XW)

    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    for k0 in range(0, CIN_PER_GROUP, BLOCK_K):
        offs_k = k0 + offs_k_base
        mask_k = offs_k < CIN_PER_GROUP
        ic_global = pid_g * CIN_PER_GROUP + offs_k

        x_ptrs = (
            x_ptr
            + batch_idx[:, None] * (C_IN * XH * XW)
            + ic_global[None, :] * (XH * XW)
            + ih[:, None] * XW
            + iw[:, None]
        )
        x = tl.load(
            x_ptrs,
            mask=valid_hw[:, None] & mask_k[None, :],
            other=0.0,
        )

        # Packed [G, CinG, CoutG]
        w_ptrs = (
            w_ptr
            + pid_g * (CIN_PER_GROUP * COUT_PER_GROUP)
            + offs_k[:, None] * COUT_PER_GROUP
            + offs_oc[None, :]
        )
        w = tl.load(
            w_ptrs,
            mask=mask_k[:, None] & mask_oc[None, :],
            other=0.0,
        )

        acc = tl.dot(x, w, acc, input_precision="tf32")

    oc_global = pid_g * COUT_PER_GROUP + offs_oc

    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[None, :]

    y_ptrs = (
        y_ptr
        + batch_idx[:, None] * (C_OUT * OH * OW)
        + oc_global[None, :] * (OH * OW)
        + hw[:, None]
    )
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_oc[None, :],
    )


@triton.jit
def conv2d_1x1_nchw_m_oc_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
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
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DTYPE_ID: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)

    HW = OH * OW

    num_pid_m = tl.cdiv(M, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_oc = offs_oc < COUT_PER_GROUP

    batch_idx = offs_m // HW
    hw = offs_m - batch_idx * HW
    oh = hw // OW
    ow = hw - oh * OW

    ih = oh * STRIDE_H - PAD_TOP
    iw = ow * STRIDE_W - PAD_LEFT
    valid_hw = mask_m & (ih >= 0) & (ih < XH) & (iw >= 0) & (iw < XW)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for k0 in range(0, CIN_PER_GROUP, BLOCK_K):
        offs_k = k0 + offs_k_base
        mask_k = offs_k < CIN_PER_GROUP
        ic_global = pid_g * CIN_PER_GROUP + offs_k

        x_ptrs = (
            x_ptr
            + batch_idx[None, :] * (C_IN * XH * XW)
            + ic_global[:, None] * (XH * XW)
            + ih[None, :] * XW
            + iw[None, :]
        )
        x = tl.load(
            x_ptrs,
            mask=mask_k[:, None] & valid_hw[None, :],
            other=0.0,
        )

        # Packed [G, CoutG, CinG]
        w_ptrs = (
            w_ptr
            + (pid_g * COUT_PER_GROUP + offs_oc[:, None]) * CIN_PER_GROUP
            + offs_k[None, :]
        )
        w = tl.load(
            w_ptrs,
            mask=mask_oc[:, None] & mask_k[None, :],
            other=0.0,
        )

        acc = tl.dot(w, x, acc, input_precision="tf32")

    oc_global = pid_g * COUT_PER_GROUP + offs_oc

    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[:, None]

    y_ptrs = (
        y_ptr
        + batch_idx[None, :] * (C_OUT * OH * OW)
        + oc_global[:, None] * (OH * OW)
        + hw[None, :]
    )
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_m[None, :],
    )


@triton.jit
def conv2d_1x1_nchw_m_pad0_oc_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
    HW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DTYPE_ID: tl.constexpr,
):
    # 1x1, stride=1, padding=0, NCHW.  This is the M-spanning version of
    # conv2d_1x1_nchw_pad0_kernel: M = N * H * W.  It keeps contiguous HW
    # stores like the old pad0 kernel, but aggregates all batches into each
    # GEMM tile so mid-channel shapes such as 128->256, 28x28 do not launch
    # many small per-batch GEMMs.
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_oc = offs_oc < COUT_PER_GROUP

    batch_idx = offs_m // HW
    hw = offs_m - batch_idx * HW

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for k0 in range(0, CIN_PER_GROUP, BLOCK_K):
        offs_k = k0 + offs_k_base
        mask_k = offs_k < CIN_PER_GROUP
        ic_global = pid_g * CIN_PER_GROUP + offs_k

        x = tl.load(
            x_ptr
            + batch_idx[None, :] * (C_IN * HW)
            + ic_global[:, None] * HW
            + hw[None, :],
            mask=mask_k[:, None] & mask_m[None, :],
            other=0.0,
        )

        # Packed [G, CoutG, CinG].
        w = tl.load(
            w_ptr
            + (pid_g * COUT_PER_GROUP + offs_oc[:, None]) * CIN_PER_GROUP
            + offs_k[None, :],
            mask=mask_oc[:, None] & mask_k[None, :],
            other=0.0,
        )

        acc = tl.dot(w, x, acc, input_precision="tf32")

    oc_global = pid_g * COUT_PER_GROUP + offs_oc

    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[:, None]

    tl.store(
        y_ptr
        + batch_idx[None, :] * (C_OUT * HW)
        + oc_global[:, None] * HW
        + hw[None, :],
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_m[None, :],
    )


@triton.jit
def conv2d_conv2d_spatial_nchw_kernel_variant(
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
):
    pid = tl.program_id(0)
    pid_bg = tl.program_id(1)

    batch_idx = pid_bg // GROUPS
    group_idx = pid_bg - batch_idx * GROUPS

    HW = OH * OW
    KDIM = CIN_PER_GROUP * KH * KW
    KERNEL_AREA = KH * KW

    num_pid_m = tl.cdiv(HW, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_hw = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_hw = offs_hw < HW
    mask_oc = offs_oc < COUT_PER_GROUP

    oh = offs_hw // OW
    ow = offs_hw - oh * OW

    x_batch_base = batch_idx * (C_IN * XH * XW)
    y_batch_base = batch_idx * (C_OUT * HW)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for k0 in range(0, KDIM, BLOCK_K):
        offs_k = k0 + offs_k_base
        mask_k = offs_k < KDIM

        ic_local = offs_k // KERNEL_AREA
        rem_k = offs_k - ic_local * KERNEL_AREA
        kh_idx = rem_k // KW
        kw_idx = rem_k - kh_idx * KW
        ic_global = group_idx * CIN_PER_GROUP + ic_local

        ih = oh[None, :] * STRIDE_H - PAD_TOP + kh_idx[:, None] * DIL_H
        iw = ow[None, :] * STRIDE_W - PAD_LEFT + kw_idx[:, None] * DIL_W
        valid = (
            mask_hw[None, :]
            & mask_k[:, None]
            & (ih >= 0)
            & (ih < XH)
            & (iw >= 0)
            & (iw < XW)
        )

        x_ptrs = (
            x_ptr
            + x_batch_base
            + ic_global[:, None] * (XH * XW)
            + ih * XW
            + iw
        )
        x = tl.load(x_ptrs, mask=valid, other=0.0)

        # Contiguous OIHW flattened as [G, CoutG, CinG*KH*KW].
        w_ptrs = (
            w_ptr
            + (group_idx * COUT_PER_GROUP + offs_oc[:, None]) * KDIM
            + offs_k[None, :]
        )
        w = tl.load(w_ptrs, mask=mask_oc[:, None] & mask_k[None, :], other=0.0)
        acc = tl.dot(w, x, acc, input_precision="tf32")

    oc_global = group_idx * COUT_PER_GROUP + offs_oc
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[:, None]

    y_ptrs = y_ptr + y_batch_base + oc_global[:, None] * HW + offs_hw[None, :]
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_hw[None, :],
    )


@triton.jit
def conv2d_spatial_nchw_packed_khw_kernel(
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
):
    pid = tl.program_id(0)
    pid_bg = tl.program_id(1)

    batch_idx = pid_bg // GROUPS
    group_idx = pid_bg - batch_idx * GROUPS

    HW = OH * OW

    num_pid_m = tl.cdiv(HW, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_hw = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_hw = offs_hw < HW
    mask_oc = offs_oc < COUT_PER_GROUP

    oh = offs_hw // OW
    ow = offs_hw - oh * OW

    x_batch_base = batch_idx * (C_IN * XH * XW)
    y_batch_base = batch_idx * (C_OUT * HW)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    # Static kh/kw loops remove div/mod by KH*KW from the hot K loop.
    for kh in tl.static_range(0, KH):
        ih = oh * STRIDE_H - PAD_TOP + kh * DIL_H
        valid_h = (ih >= 0) & (ih < XH)

        for kw in tl.static_range(0, KW):
            iw = ow * STRIDE_W - PAD_LEFT + kw * DIL_W
            valid_hw = mask_hw & valid_h & (iw >= 0) & (iw < XW)

            for k0 in range(0, CIN_PER_GROUP, BLOCK_K):
                ic_local = k0 + offs_k_base
                mask_k = ic_local < CIN_PER_GROUP
                ic_global = group_idx * CIN_PER_GROUP + ic_local

                x_ptrs = (
                    x_ptr
                    + x_batch_base
                    + ic_global[:, None] * (XH * XW)
                    + ih[None, :] * XW
                    + iw[None, :]
                )
                x = tl.load(
                    x_ptrs,
                    mask=mask_k[:, None] & valid_hw[None, :],
                    other=0.0,
                )

                # Packed [G, KH, KW, CoutG, CinG].
                w_ptrs = (
                    w_ptr
                    + (
                        (
                            ((group_idx * KH + kh) * KW + kw) * COUT_PER_GROUP
                            + offs_oc[:, None]
                        )
                        * CIN_PER_GROUP
                    )
                    + ic_local[None, :]
                )
                w = tl.load(
                    w_ptrs,
                    mask=mask_oc[:, None] & mask_k[None, :],
                    other=0.0,
                )

                acc = tl.dot(w, x, acc, input_precision="tf32")

    oc_global = group_idx * COUT_PER_GROUP + offs_oc
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[:, None]

    y_ptrs = y_ptr + y_batch_base + oc_global[:, None] * HW + offs_hw[None, :]
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_hw[None, :],
    )


@triton.jit
def conv2d_spatial_nchw_m_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
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
):
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)

    HW = OH * OW
    KDIM = CIN_PER_GROUP * KH * KW
    KERNEL_AREA = KH * KW

    num_pid_m = tl.cdiv(M, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_oc = offs_oc < COUT_PER_GROUP

    batch_idx = offs_m // HW
    hw = offs_m - batch_idx * HW
    oh = hw // OW
    ow = hw - oh * OW

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for k0 in range(0, KDIM, BLOCK_K):
        offs_k = k0 + offs_k_base
        mask_k = offs_k < KDIM

        ic_local = offs_k // KERNEL_AREA
        rem_k = offs_k - ic_local * KERNEL_AREA
        kh_idx = rem_k // KW
        kw_idx = rem_k - kh_idx * KW
        ic_global = pid_g * CIN_PER_GROUP + ic_local

        ih = oh[None, :] * STRIDE_H - PAD_TOP + kh_idx[:, None] * DIL_H
        iw = ow[None, :] * STRIDE_W - PAD_LEFT + kw_idx[:, None] * DIL_W
        valid = (
            mask_k[:, None]
            & mask_m[None, :]
            & (ih >= 0)
            & (ih < XH)
            & (iw >= 0)
            & (iw < XW)
        )

        x_ptrs = (
            x_ptr
            + batch_idx[None, :] * (C_IN * XH * XW)
            + ic_global[:, None] * (XH * XW)
            + ih * XW
            + iw
        )
        x = tl.load(x_ptrs, mask=valid, other=0.0)

        # Contiguous OIHW flattened as [G, CoutG, CinG*KH*KW].
        w_ptrs = (
            w_ptr
            + (pid_g * COUT_PER_GROUP + offs_oc[:, None]) * KDIM
            + offs_k[None, :]
        )
        w = tl.load(w_ptrs, mask=mask_oc[:, None] & mask_k[None, :], other=0.0)
        acc = tl.dot(w, x, acc, input_precision="tf32")

    oc_global = pid_g * COUT_PER_GROUP + offs_oc
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[:, None]

    y_ptrs = (
        y_ptr
        + batch_idx[None, :] * (C_OUT * OH * OW)
        + oc_global[:, None] * (OH * OW)
        + hw[None, :]
    )
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_m[None, :],
    )


@triton.jit
def conv2d_spatial_nchw_m_packed_khw_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
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
):
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)

    HW = OH * OW

    num_pid_m = tl.cdiv(M, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_oc = offs_oc < COUT_PER_GROUP

    batch_idx = offs_m // HW
    hw = offs_m - batch_idx * HW
    oh = hw // OW
    ow = hw - oh * OW

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    # Static kh/kw loops remove div/mod by KH*KW from the K loop while M spans
    # all batch items.  Accumulator orientation is OC x M so NCHW stores are
    # contiguous along the HW dimension.
    for kh in tl.static_range(0, KH):
        ih = oh * STRIDE_H - PAD_TOP + kh * DIL_H
        valid_h = (ih >= 0) & (ih < XH)

        for kw in tl.static_range(0, KW):
            iw = ow * STRIDE_W - PAD_LEFT + kw * DIL_W
            valid_hw = mask_m & valid_h & (iw >= 0) & (iw < XW)

            for k0 in range(0, CIN_PER_GROUP, BLOCK_K):
                ic_local = k0 + offs_k_base
                mask_k = ic_local < CIN_PER_GROUP
                ic_global = pid_g * CIN_PER_GROUP + ic_local

                x_ptrs = (
                    x_ptr
                    + batch_idx[None, :] * (C_IN * XH * XW)
                    + ic_global[:, None] * (XH * XW)
                    + ih[None, :] * XW
                    + iw[None, :]
                )
                x = tl.load(
                    x_ptrs,
                    mask=mask_k[:, None] & valid_hw[None, :],
                    other=0.0,
                )

                # Packed [G, KH, KW, CoutG, CinG].
                w_ptrs = (
                    w_ptr
                    + (
                        (
                            ((pid_g * KH + kh) * KW + kw) * COUT_PER_GROUP
                            + offs_oc[:, None]
                        )
                        * CIN_PER_GROUP
                    )
                    + ic_local[None, :]
                )
                w = tl.load(
                    w_ptrs,
                    mask=mask_oc[:, None] & mask_k[None, :],
                    other=0.0,
                )

                acc = tl.dot(w, x, acc, input_precision="tf32")

    oc_global = pid_g * COUT_PER_GROUP + offs_oc
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[:, None]

    y_ptrs = (
        y_ptr
        + batch_idx[None, :] * (C_OUT * OH * OW)
        + oc_global[:, None] * (OH * OW)
        + hw[None, :]
    )
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_m[None, :],
    )


@triton.jit
def _winograd_f2_load_d(
    x_ptr,
    batch_idx,
    offs_ic,
    tile_oh,
    tile_ow,
    mask_k,
    mask_t,
    C_IN: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    R: tl.constexpr,
    S: tl.constexpr,
):
    ih = tile_oh - 1 + R
    iw = tile_ow - 1 + S
    valid = mask_t & (ih >= 0) & (ih < XH) & (iw >= 0) & (iw < XW)
    ih_safe = tl.where(valid, ih, 0)
    iw_safe = tl.where(valid, iw, 0)
    return tl.load(
        x_ptr
        + batch_idx[None, :] * (C_IN * XH * XW)
        + offs_ic[:, None] * (XH * XW)
        + ih_safe[None, :] * XW
        + iw_safe[None, :],
        mask=mask_k[:, None] & valid[None, :],
        other=0.0,
    )


@triton.jit
def conv2d_spatial_nchw_winograd_f2_3x3_kernel(
    x_ptr,
    u_ptr,
    bias_ptr,
    y_ptr,
    NUM_TILES: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_TILE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)

    tiles_h = tl.cdiv(OH, 2)
    tiles_w = tl.cdiv(OW, 2)
    tiles_per_n = tiles_h * tiles_w

    num_pid_t = tl.cdiv(NUM_TILES, BLOCK_TILE)
    pid_t = pid % num_pid_t
    pid_oc = pid // num_pid_t

    offs_t = pid_t * BLOCK_TILE + tl.arange(0, BLOCK_TILE)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_t = offs_t < NUM_TILES
    mask_oc = offs_oc < COUT_PER_GROUP

    batch_idx = offs_t // tiles_per_n
    tile_rem = offs_t - batch_idx * tiles_per_n
    tile_h = tile_rem // tiles_w
    tile_w = tile_rem - tile_h * tiles_w
    out_h0 = tile_h * 2
    out_w0 = tile_w * 2

    acc00 = tl.zeros((BLOCK_OC, BLOCK_TILE), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_OC, BLOCK_TILE), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_OC, BLOCK_TILE), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_OC, BLOCK_TILE), dtype=tl.float32)

    stride_p = GROUPS * COUT_PER_GROUP * CIN_PER_GROUP
    group_u_base = pid_g * COUT_PER_GROUP * CIN_PER_GROUP

    for k0 in range(0, CIN_PER_GROUP, BLOCK_K):
        offs_k = k0 + offs_k_base
        mask_k = offs_k < CIN_PER_GROUP
        ic_global = pid_g * CIN_PER_GROUP + offs_k

        for p in tl.static_range(0, 16):
            pi = p // 4
            pj = p - pi * 4

            if pi == 0:
                ra0 = 0
                ra1 = 2
                rs0 = 1.0
                rs1 = -1.0
            elif pi == 1:
                ra0 = 1
                ra1 = 2
                rs0 = 1.0
                rs1 = 1.0
            elif pi == 2:
                ra0 = 1
                ra1 = 2
                rs0 = -1.0
                rs1 = 1.0
            else:
                ra0 = 1
                ra1 = 3
                rs0 = 1.0
                rs1 = -1.0

            if pj == 0:
                cb0 = 0
                cb1 = 2
                cs0 = 1.0
                cs1 = -1.0
            elif pj == 1:
                cb0 = 1
                cb1 = 2
                cs0 = 1.0
                cs1 = 1.0
            elif pj == 2:
                cb0 = 1
                cb1 = 2
                cs0 = -1.0
                cs1 = 1.0
            else:
                cb0 = 1
                cb1 = 3
                cs0 = 1.0
                cs1 = -1.0

            d00 = _winograd_f2_load_d(
                x_ptr,
                batch_idx,
                ic_global,
                out_h0,
                out_w0,
                mask_k,
                mask_t,
                C_IN,
                XH,
                XW,
                R=ra0,
                S=cb0,
            )
            d01 = _winograd_f2_load_d(
                x_ptr,
                batch_idx,
                ic_global,
                out_h0,
                out_w0,
                mask_k,
                mask_t,
                C_IN,
                XH,
                XW,
                R=ra0,
                S=cb1,
            )
            d10 = _winograd_f2_load_d(
                x_ptr,
                batch_idx,
                ic_global,
                out_h0,
                out_w0,
                mask_k,
                mask_t,
                C_IN,
                XH,
                XW,
                R=ra1,
                S=cb0,
            )
            d11 = _winograd_f2_load_d(
                x_ptr,
                batch_idx,
                ic_global,
                out_h0,
                out_w0,
                mask_k,
                mask_t,
                C_IN,
                XH,
                XW,
                R=ra1,
                S=cb1,
            )
            v = (
                (rs0 * cs0) * d00
                + (rs0 * cs1) * d01
                + (rs1 * cs0) * d10
                + (rs1 * cs1) * d11
            )
            v = v.to(x_ptr.dtype.element_ty)

            u = tl.load(
                u_ptr
                + p * stride_p
                + group_u_base
                + offs_oc[:, None] * CIN_PER_GROUP
                + offs_k[None, :],
                mask=mask_oc[:, None] & mask_k[None, :],
                other=0.0,
            )
            prod = tl.dot(u, v, input_precision="tf32")

            if pi <= 2 and pj <= 2:
                acc00 += prod
            if pi <= 2:
                if pj == 1:
                    acc01 += prod
                elif pj != 0:
                    acc01 -= prod
            if pj <= 2:
                if pi == 1:
                    acc10 += prod
                elif pi != 0:
                    acc10 -= prod
            if pi != 0 and pj != 0:
                if pi == 1:
                    if pj == 1:
                        acc11 += prod
                    else:
                        acc11 -= prod
                else:
                    if pj == 1:
                        acc11 -= prod
                    else:
                        acc11 += prod

    oc_global = pid_g * COUT_PER_GROUP + offs_oc
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc00 += bias[:, None]
        acc01 += bias[:, None]
        acc10 += bias[:, None]
        acc11 += bias[:, None]

    y_base = batch_idx[None, :] * (C_OUT * OH * OW) + oc_global[:, None] * (
        OH * OW
    )
    mask_base = mask_oc[:, None] & mask_t[None, :]

    out_hw00 = out_h0 * OW + out_w0
    tl.store(
        y_ptr + y_base + out_hw00[None, :],
        acc00.to(y_ptr.dtype.element_ty),
        mask=mask_base & (out_h0[None, :] < OH) & (out_w0[None, :] < OW),
    )

    out_hw01 = out_h0 * OW + (out_w0 + 1)
    tl.store(
        y_ptr + y_base + out_hw01[None, :],
        acc01.to(y_ptr.dtype.element_ty),
        mask=mask_base & (out_h0[None, :] < OH) & ((out_w0 + 1)[None, :] < OW),
    )

    out_hw10 = (out_h0 + 1) * OW + out_w0
    tl.store(
        y_ptr + y_base + out_hw10[None, :],
        acc10.to(y_ptr.dtype.element_ty),
        mask=mask_base & ((out_h0 + 1)[None, :] < OH) & (out_w0[None, :] < OW),
    )

    out_hw11 = (out_h0 + 1) * OW + (out_w0 + 1)
    tl.store(
        y_ptr + y_base + out_hw11[None, :],
        acc11.to(y_ptr.dtype.element_ty),
        mask=mask_base
        & ((out_h0 + 1)[None, :] < OH)
        & ((out_w0 + 1)[None, :] < OW),
    )


@triton.jit
def conv2d_spatial_nchw_3x3_interior_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M_INT: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DTYPE_ID: tl.constexpr,
):
    # 3x3, stride=1, padding=dilation interior.  All input coordinates are
    # valid here, so the hot loop has no boundary comparisons.  Border elements
    # are produced by conv2d_spatial_nchw_3x3_border_kernel.
    pid = tl.program_id(0)
    pid_bg = tl.program_id(1)

    batch_idx = pid_bg // GROUPS
    group_idx = pid_bg - batch_idx * GROUPS

    # INT_H = OH - 2 * DIL_H
    INT_W = OW - 2 * DIL_W

    num_pid_m = tl.cdiv(M_INT, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_i = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_i = offs_i < M_INT
    mask_oc = offs_oc < COUT_PER_GROUP

    oh_i = offs_i // INT_W + DIL_H
    ow_i = offs_i - (oh_i - DIL_H) * INT_W + DIL_W
    out_hw = oh_i * OW + ow_i

    x_batch_base = batch_idx * (C_IN * XH * XW)
    y_batch_base = batch_idx * (C_OUT * OH * OW)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for kh in tl.static_range(0, 3):
        ih = oh_i - PAD_TOP + kh * DIL_H
        for kw in tl.static_range(0, 3):
            iw = ow_i - PAD_LEFT + kw * DIL_W
            for k0 in range(0, CIN_PER_GROUP, BLOCK_K):
                ic_local = k0 + offs_k_base
                mask_k = ic_local < CIN_PER_GROUP
                ic_global = group_idx * CIN_PER_GROUP + ic_local

                x = tl.load(
                    x_ptr
                    + x_batch_base
                    + ic_global[:, None] * (XH * XW)
                    + ih[None, :] * XW
                    + iw[None, :],
                    mask=mask_k[:, None] & mask_i[None, :],
                    other=0.0,
                )

                # Packed [G, KH, KW, CoutG, CinG].
                w = tl.load(
                    w_ptr
                    + (
                        ((group_idx * 3 + kh) * 3 + kw) * COUT_PER_GROUP
                        + offs_oc[:, None]
                    )
                    * CIN_PER_GROUP
                    + ic_local[None, :],
                    mask=mask_oc[:, None] & mask_k[None, :],
                    other=0.0,
                )
                acc = tl.dot(w, x, acc, input_precision="tf32")

    oc_global = group_idx * COUT_PER_GROUP + offs_oc
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[:, None]

    tl.store(
        y_ptr
        + y_batch_base
        + oc_global[:, None] * (OH * OW)
        + out_hw[None, :],
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_i[None, :],
    )


@triton.jit
def conv2d_spatial_nchw_3x3_border_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M_BORDER: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DTYPE_ID: tl.constexpr,
):
    # Produces only the border band of the same 3x3/padding=dilation case.
    pid = tl.program_id(0)
    pid_bg = tl.program_id(1)

    batch_idx = pid_bg // GROUPS
    group_idx = pid_bg - batch_idx * GROUPS

    num_pid_m = tl.cdiv(M_BORDER, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_b = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_b = offs_b < M_BORDER
    mask_oc = offs_oc < COUT_PER_GROUP

    TOP = DIL_H * OW
    BOTTOM = DIL_H * OW
    SIDE_W = 2 * DIL_W
    # MID_H = OH - 2 * DIL_H

    in_top = offs_b < TOP
    in_bottom = (offs_b >= TOP) & (offs_b < TOP + BOTTOM)
    rem_bottom = offs_b - TOP
    rem_side = offs_b - TOP - BOTTOM

    side_row = rem_side // SIDE_W + DIL_H
    side_col_tmp = rem_side - (side_row - DIL_H) * SIDE_W
    side_col = tl.where(
        side_col_tmp < DIL_W, side_col_tmp, OW - DIL_W + (side_col_tmp - DIL_W)
    )

    out_hw_top = offs_b
    out_hw_bottom = (OH - DIL_H) * OW + rem_bottom
    out_hw_side = side_row * OW + side_col
    out_hw = tl.where(
        in_top, out_hw_top, tl.where(in_bottom, out_hw_bottom, out_hw_side)
    )

    oh_o = out_hw // OW
    ow_o = out_hw - oh_o * OW

    x_batch_base = batch_idx * (C_IN * XH * XW)
    y_batch_base = batch_idx * (C_OUT * OH * OW)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for kh in tl.static_range(0, 3):
        ih = oh_o * 1 - PAD_TOP + kh * DIL_H
        valid_h = (ih >= 0) & (ih < XH)
        for kw in tl.static_range(0, 3):
            iw = ow_o * 1 - PAD_LEFT + kw * DIL_W
            valid_hw = mask_b & valid_h & (iw >= 0) & (iw < XW)
            for k0 in range(0, CIN_PER_GROUP, BLOCK_K):
                ic_local = k0 + offs_k_base
                mask_k = ic_local < CIN_PER_GROUP
                ic_global = group_idx * CIN_PER_GROUP + ic_local

                x = tl.load(
                    x_ptr
                    + x_batch_base
                    + ic_global[:, None] * (XH * XW)
                    + ih[None, :] * XW
                    + iw[None, :],
                    mask=mask_k[:, None] & valid_hw[None, :],
                    other=0.0,
                )
                w = tl.load(
                    w_ptr
                    + (
                        ((group_idx * 3 + kh) * 3 + kw) * COUT_PER_GROUP
                        + offs_oc[:, None]
                    )
                    * CIN_PER_GROUP
                    + ic_local[None, :],
                    mask=mask_oc[:, None] & mask_k[None, :],
                    other=0.0,
                )
                acc = tl.dot(w, x, acc, input_precision="tf32")

    oc_global = group_idx * COUT_PER_GROUP + offs_oc
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[:, None]

    tl.store(
        y_ptr
        + y_batch_base
        + oc_global[:, None] * (OH * OW)
        + out_hw[None, :],
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_b[None, :],
    )


@triton.jit
def conv2d_spatial_nchw_3x3_interior_m_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M_TOTAL,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DTYPE_ID: tl.constexpr,
):
    # Global-M version of conv2d_spatial_nchw_3x3_interior_kernel.
    # M_TOTAL = N * (OH - 2*DIL_H) * (OW - 2*DIL_W).  This keeps small
    # 14x14/7x7 high-channel cases from degenerating into 25-wide per-batch
    # GEMMs while preserving the no-boundary-check interior hot loop.
    pid = tl.program_id(0)
    group_idx = tl.program_id(1)

    INT_H = OH - 2 * DIL_H
    INT_W = OW - 2 * DIL_W
    M_PER_N = INT_H * INT_W

    num_pid_m = tl.cdiv(M_TOTAL, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M_TOTAL
    mask_oc = offs_oc < COUT_PER_GROUP

    batch_idx = offs_m // M_PER_N
    i_local = offs_m - batch_idx * M_PER_N
    oh_i = i_local // INT_W + DIL_H
    ow_i = i_local - (oh_i - DIL_H) * INT_W + DIL_W
    out_hw = oh_i * OW + ow_i

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for kh in tl.static_range(0, 3):
        ih = oh_i - PAD_TOP + kh * DIL_H
        for kw in tl.static_range(0, 3):
            iw = ow_i - PAD_LEFT + kw * DIL_W
            for k0 in range(0, CIN_PER_GROUP, BLOCK_K):
                ic_local = k0 + offs_k_base
                mask_k = ic_local < CIN_PER_GROUP
                ic_global = group_idx * CIN_PER_GROUP + ic_local

                x = tl.load(
                    x_ptr
                    + batch_idx[None, :] * (C_IN * XH * XW)
                    + ic_global[:, None] * (XH * XW)
                    + ih[None, :] * XW
                    + iw[None, :],
                    mask=mask_k[:, None] & mask_m[None, :],
                    other=0.0,
                )

                w = tl.load(
                    w_ptr
                    + (
                        ((group_idx * 3 + kh) * 3 + kw) * COUT_PER_GROUP
                        + offs_oc[:, None]
                    )
                    * CIN_PER_GROUP
                    + ic_local[None, :],
                    mask=mask_oc[:, None] & mask_k[None, :],
                    other=0.0,
                )
                acc = tl.dot(w, x, acc, input_precision="tf32")

    oc_global = group_idx * COUT_PER_GROUP + offs_oc
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[:, None]

    tl.store(
        y_ptr
        + batch_idx[None, :] * (C_OUT * OH * OW)
        + oc_global[:, None] * (OH * OW)
        + out_hw[None, :],
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_m[None, :],
    )


@triton.jit
def conv2d_spatial_nchw_3x3_border_m_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M_TOTAL,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DTYPE_ID: tl.constexpr,
):
    # Global-M version of the border-band kernel.
    pid = tl.program_id(0)
    group_idx = tl.program_id(1)

    TOP = DIL_H * OW
    BOTTOM = DIL_H * OW
    SIDE_W = 2 * DIL_W
    MID_H = OH - 2 * DIL_H
    M_PER_N = OH * OW - MID_H * (OW - 2 * DIL_W)

    num_pid_m = tl.cdiv(M_TOTAL, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M_TOTAL
    mask_oc = offs_oc < COUT_PER_GROUP

    batch_idx = offs_m // M_PER_N
    offs_b = offs_m - batch_idx * M_PER_N

    in_top = offs_b < TOP
    in_bottom = (offs_b >= TOP) & (offs_b < TOP + BOTTOM)
    rem_bottom = offs_b - TOP
    rem_side = offs_b - TOP - BOTTOM

    side_row = rem_side // SIDE_W + DIL_H
    side_col_tmp = rem_side - (side_row - DIL_H) * SIDE_W
    side_col = tl.where(
        side_col_tmp < DIL_W, side_col_tmp, OW - DIL_W + (side_col_tmp - DIL_W)
    )

    out_hw_top = offs_b
    out_hw_bottom = (OH - DIL_H) * OW + rem_bottom
    out_hw_side = side_row * OW + side_col
    out_hw = tl.where(
        in_top, out_hw_top, tl.where(in_bottom, out_hw_bottom, out_hw_side)
    )

    oh_o = out_hw // OW
    ow_o = out_hw - oh_o * OW

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for kh in tl.static_range(0, 3):
        ih = oh_o - PAD_TOP + kh * DIL_H
        valid_h = (ih >= 0) & (ih < XH)
        for kw in tl.static_range(0, 3):
            iw = ow_o - PAD_LEFT + kw * DIL_W
            valid_hw = mask_m & valid_h & (iw >= 0) & (iw < XW)
            for k0 in range(0, CIN_PER_GROUP, BLOCK_K):
                ic_local = k0 + offs_k_base
                mask_k = ic_local < CIN_PER_GROUP
                ic_global = group_idx * CIN_PER_GROUP + ic_local

                x = tl.load(
                    x_ptr
                    + batch_idx[None, :] * (C_IN * XH * XW)
                    + ic_global[:, None] * (XH * XW)
                    + ih[None, :] * XW
                    + iw[None, :],
                    mask=mask_k[:, None] & valid_hw[None, :],
                    other=0.0,
                )
                w = tl.load(
                    w_ptr
                    + (
                        ((group_idx * 3 + kh) * 3 + kw) * COUT_PER_GROUP
                        + offs_oc[:, None]
                    )
                    * CIN_PER_GROUP
                    + ic_local[None, :],
                    mask=mask_oc[:, None] & mask_k[None, :],
                    other=0.0,
                )
                acc = tl.dot(w, x, acc, input_precision="tf32")

    oc_global = group_idx * COUT_PER_GROUP + offs_oc
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[:, None]

    tl.store(
        y_ptr
        + batch_idx[None, :] * (C_OUT * OH * OW)
        + oc_global[:, None] * (OH * OW)
        + out_hw[None, :],
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_m[None, :],
    )


@triton.jit
def depthwise_conv2d_nchw_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    DTYPE_ID: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_hw = offs_hw < M
    mask_c = offs_c < C_IN

    oh = offs_hw // OW
    ow = offs_hw - oh * OW

    x_batch_base = pid_n * (C_IN * XH * XW)
    y_batch_base = pid_n * (C_IN * OH * OW)

    acc = tl.zeros((BLOCK_C, BLOCK_HW), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        ih = oh * STRIDE_H - PAD_TOP + kh * DIL_H
        valid_h = (ih >= 0) & (ih < XH)

        for kw in tl.static_range(0, KW):
            iw = ow * STRIDE_W - PAD_LEFT + kw * DIL_W
            valid_hw = mask_hw & valid_h & (iw >= 0) & (iw < XW)

            x_ptrs = (
                x_ptr
                + x_batch_base
                + offs_c[:, None] * (XH * XW)
                + ih[None, :] * XW
                + iw[None, :]
            )
            x = tl.load(
                x_ptrs,
                mask=mask_c[:, None] & valid_hw[None, :],
                other=0.0,
            )

            # Packed [KH, KW, C]
            w = tl.load(
                w_ptr + (kh * KW + kw) * C_IN + offs_c,
                mask=mask_c,
                other=0.0,
            )
            acc += w[:, None] * x

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
        acc += bias[:, None]

    y_ptrs = (
        y_ptr + y_batch_base + offs_c[:, None] * (OH * OW) + offs_hw[None, :]
    )
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_c[:, None] & mask_hw[None, :],
    )


@triton.jit
def depthwise_conv2d_nchw_c1_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    DTYPE_ID: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    c = tl.program_id(1)
    n = tl.program_id(2)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < M

    oh = offs_hw // OW
    ow = offs_hw - oh * OW

    x_base = x_ptr + n * (C_IN * XH * XW) + c * (XH * XW)
    y_base = y_ptr + n * (C_IN * OH * OW) + c * (OH * OW)

    acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        ih = oh * STRIDE_H - PAD_TOP + kh * DIL_H
        valid_h = (ih >= 0) & (ih < XH)

        for kw in tl.static_range(0, KW):
            iw = ow * STRIDE_W - PAD_LEFT + kw * DIL_W
            valid_hw = mask_hw & valid_h & (iw >= 0) & (iw < XW)

            x = tl.load(
                x_base + ih * XW + iw,
                mask=valid_hw,
                other=0.0,
            )
            ww = tl.load(w_ptr + (kh * KW + kw) * C_IN + c)
            acc += x * ww

    if HAS_BIAS:
        acc += tl.load(bias_ptr + c)

    tl.store(
        y_base + offs_hw,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_hw,
    )


@triton.jit
def conv2d_1x1_cl_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
    XH,
    XW,
    OH,
    OW,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    y_stride_n,
    y_stride_c,
    y_stride_h,
    y_stride_w,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DTYPE_ID: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)

    tl.assume(x_stride_n > 0)
    tl.assume(x_stride_c > 0)
    tl.assume(x_stride_h > 0)
    tl.assume(x_stride_w > 0)
    tl.assume(y_stride_n > 0)
    tl.assume(y_stride_c > 0)
    tl.assume(y_stride_h > 0)
    tl.assume(y_stride_w > 0)

    num_pid_m = tl.cdiv(M, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_n = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_m = offs_m < M
    mask_n = offs_n < COUT_PER_GROUP

    HW = OH * OW
    batch_idx = offs_m // HW
    rem = offs_m - batch_idx * HW
    oh = rem // OW
    ow = rem - oh * OW
    ih = oh * STRIDE_H - PAD_TOP
    iw = ow * STRIDE_W - PAD_LEFT
    valid_hw = mask_m & (ih >= 0) & (ih < XH) & (iw >= 0) & (iw < XW)

    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    for k0 in range(0, CIN_PER_GROUP, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < CIN_PER_GROUP
        ic_global = pid_g * CIN_PER_GROUP + offs_k

        a_ptrs = (
            x_ptr
            + batch_idx[:, None] * x_stride_n
            + ic_global[None, :] * x_stride_c
            + ih[:, None] * x_stride_h
            + iw[:, None] * x_stride_w
        )
        a = tl.load(
            a_ptrs, mask=valid_hw[:, None] & mask_k[None, :], other=0.0
        )

        # Packed [G, CinG, CoutG]
        w_ptrs = (
            w_ptr
            + pid_g * (CIN_PER_GROUP * COUT_PER_GROUP)
            + offs_k[:, None] * COUT_PER_GROUP
            + offs_n[None, :]
        )
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, w, acc, input_precision="tf32")

    oc_global = pid_g * COUT_PER_GROUP + offs_n
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_n, other=0.0)
        acc += bias[None, :]

    y_ptrs = (
        y_ptr
        + batch_idx[:, None] * y_stride_n
        + oc_global[None, :] * y_stride_c
        + oh[:, None] * y_stride_h
        + ow[:, None] * y_stride_w
    )
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def conv2d_spatial_cl_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
    XH,
    XW,
    OH,
    OW,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    y_stride_n,
    y_stride_c,
    y_stride_h,
    y_stride_w,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
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
):
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)

    tl.assume(x_stride_n > 0)
    tl.assume(x_stride_c > 0)
    tl.assume(x_stride_h > 0)
    tl.assume(x_stride_w > 0)
    tl.assume(y_stride_n > 0)
    tl.assume(y_stride_c > 0)
    tl.assume(y_stride_h > 0)
    tl.assume(y_stride_w > 0)

    KDIM = CIN_PER_GROUP * KH * KW
    KERNEL_AREA = KH * KW

    num_pid_m = tl.cdiv(M, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_n = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < COUT_PER_GROUP

    HW = OH * OW
    batch_idx = offs_m // HW
    rem = offs_m - batch_idx * HW
    oh = rem // OW
    ow = rem - oh * OW

    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    for k0 in range(0, KDIM, BLOCK_K):
        offs_k = k0 + offs_k_base
        mask_k = offs_k < KDIM

        ic_local = offs_k // KERNEL_AREA
        rem_k = offs_k - ic_local * KERNEL_AREA
        kh_idx = rem_k // KW
        kw_idx = rem_k - kh_idx * KW
        ic_global = pid_g * CIN_PER_GROUP + ic_local

        ih = oh[:, None] * STRIDE_H - PAD_TOP + kh_idx[None, :] * DIL_H
        iw = ow[:, None] * STRIDE_W - PAD_LEFT + kw_idx[None, :] * DIL_W
        valid = (
            mask_m[:, None]
            & mask_k[None, :]
            & (ih >= 0)
            & (ih < XH)
            & (iw >= 0)
            & (iw < XW)
        )

        x_ptrs = (
            x_ptr
            + batch_idx[:, None] * x_stride_n
            + ic_global[None, :] * x_stride_c
            + ih * x_stride_h
            + iw * x_stride_w
        )
        a = tl.load(x_ptrs, mask=valid, other=0.0)

        # Packed [G, CinG, KH, KW, CoutG].
        w_ptrs = (
            w_ptr
            + pid_g * (KDIM * COUT_PER_GROUP)
            + offs_k[:, None] * COUT_PER_GROUP
            + offs_n[None, :]
        )
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, w, acc, input_precision="tf32")

    oc_global = pid_g * COUT_PER_GROUP + offs_n
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_n, other=0.0)
        acc += bias[None, :]

    y_ptrs = (
        y_ptr
        + batch_idx[:, None] * y_stride_n
        + oc_global[None, :] * y_stride_c
        + oh[:, None] * y_stride_h
        + ow[:, None] * y_stride_w
    )
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def depthwise_conv2d_cl_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
    XH,
    XW,
    OH,
    OW,
    C_IN,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    y_stride_n,
    y_stride_c,
    y_stride_h,
    y_stride_w,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    DTYPE_ID: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_n = tl.program_id(2)

    tl.assume(x_stride_n > 0)
    tl.assume(x_stride_c > 0)
    tl.assume(x_stride_h > 0)
    tl.assume(x_stride_w > 0)
    tl.assume(y_stride_n > 0)
    tl.assume(y_stride_c > 0)
    tl.assume(y_stride_h > 0)
    tl.assume(y_stride_w > 0)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_hw = offs_hw < M
    mask_c = offs_c < C_IN

    oh = offs_hw // OW
    ow = offs_hw - oh * OW

    acc = tl.zeros((BLOCK_HW, BLOCK_C), dtype=tl.float32)

    x_base = x_ptr + pid_n * x_stride_n
    y_base = y_ptr + pid_n * y_stride_n

    for kh in tl.static_range(0, KH):
        ih = oh * STRIDE_H - PAD_TOP + kh * DIL_H
        valid_h = (ih >= 0) & (ih < XH)

        for kw in tl.static_range(0, KW):
            iw = ow * STRIDE_W - PAD_LEFT + kw * DIL_W
            valid_hw = mask_hw & valid_h & (iw >= 0) & (iw < XW)

            x_ptrs = (
                x_base
                + ih[:, None] * x_stride_h
                + iw[:, None] * x_stride_w
                + offs_c[None, :] * x_stride_c
            )
            x = tl.load(
                x_ptrs,
                mask=valid_hw[:, None] & mask_c[None, :],
                other=0.0,
            )

            # Packed [KH, KW, C]
            w = tl.load(
                w_ptr + (kh * KW + kw) * C_IN + offs_c,
                mask=mask_c,
                other=0.0,
            )
            acc += x * w[None, :]

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
        acc += bias[None, :]

    y_ptrs = (
        y_base
        + oh[:, None] * y_stride_h
        + ow[:, None] * y_stride_w
        + offs_c[None, :] * y_stride_c
    )
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_hw[:, None] & mask_c[None, :],
    )


@triton.jit
def conv2d_fp64_im2col_nchw_kernel(
    x_ptr,
    col_ptr,
    TOTAL: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    N: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    # Materialize columns as [G, N, K, HW] contiguous.  Each [K, HW] slice is a
    # column-major matrix of shape HW x K for the old fp64 im2col helper.
    pid = tl.program_id(0)
    offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    mask = offs < TOTAL

    HW = OH * OW
    KERNEL_AREA = KH * KW
    KDIM = CIN_PER_GROUP * KERNEL_AREA

    hw = offs % HW
    k = (offs // HW) % KDIM
    bn = offs // (KDIM * HW)
    group_idx = bn // N
    batch_idx = bn - group_idx * N

    ow = hw % OW
    oh = hw // OW

    ic_local = k // KERNEL_AREA
    rem = k - ic_local * KERNEL_AREA
    kh_idx = rem // KW
    kw_idx = rem - kh_idx * KW
    ic_global = group_idx * CIN_PER_GROUP + ic_local

    ih = oh * STRIDE_H - PAD_TOP + kh_idx * DIL_H
    iw = ow * STRIDE_W - PAD_LEFT + kw_idx * DIL_W
    valid = mask & (ih >= 0) & (ih < XH) & (iw >= 0) & (iw < XW)

    x = tl.load(
        x_ptr
        + batch_idx * (C_IN * XH * XW)
        + ic_global * (XH * XW)
        + ih * XW
        + iw,
        mask=valid,
        other=0.0,
    )
    tl.store(col_ptr + offs, x, mask=mask)


@triton.jit
def conv2d_add_bias_nchw_kernel(
    y_ptr,
    bias_ptr,
    TOTAL: tl.constexpr,
    HW: tl.constexpr,
    C_OUT: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    mask = offs < TOTAL
    oc = (offs // HW) % C_OUT
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + oc, mask=mask, other=0.0)
    tl.store(y_ptr + offs, y + b, mask=mask)


@triton.jit
def depthwise_conv2d_fp64_nchw_c1_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    c = tl.program_id(1)
    n = tl.program_id(2)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < M

    oh = offs_hw // OW
    ow = offs_hw - oh * OW

    x_base = x_ptr + n * (C_IN * XH * XW) + c * (XH * XW)
    y_base = y_ptr + n * (C_IN * OH * OW) + c * (OH * OW)

    acc = tl.zeros((BLOCK_HW,), dtype=tl.float64)

    for kh in tl.static_range(0, KH):
        ih = oh * STRIDE_H - PAD_TOP + kh * DIL_H
        valid_h = (ih >= 0) & (ih < XH)

        for kw in tl.static_range(0, KW):
            iw = ow * STRIDE_W - PAD_LEFT + kw * DIL_W
            valid_hw = mask_hw & valid_h & (iw >= 0) & (iw < XW)

            x = tl.load(x_base + ih * XW + iw, mask=valid_hw, other=0.0).to(
                tl.float64
            )
            ww = tl.load(w_ptr + (kh * KW + kw) * C_IN + c).to(tl.float64)
            acc += x * ww

    if HAS_BIAS:
        acc += tl.load(bias_ptr + c).to(tl.float64)

    tl.store(y_base + offs_hw, acc, mask=mask_hw)


@triton.jit
def conv2d_fp64_nchw_m_tile_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
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
):
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)

    HW = OH * OW

    num_pid_m = tl.cdiv(M, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)

    mask_m = offs_m < M
    mask_oc = offs_oc < COUT_PER_GROUP

    batch_idx = offs_m // HW
    hw = offs_m - batch_idx * HW
    oh = hw // OW
    ow = hw - oh * OW

    oc_global = pid_g * COUT_PER_GROUP + offs_oc
    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float64)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0).to(
            tl.float64
        )
        acc += bias[:, None]

    # FP64 cannot rely on Tensor Core tl.dot in a portable way here.
    for kh in tl.static_range(0, KH):
        ih = oh * STRIDE_H - PAD_TOP + kh * DIL_H
        valid_h = (ih >= 0) & (ih < XH)

        for kw in tl.static_range(0, KW):
            iw = ow * STRIDE_W - PAD_LEFT + kw * DIL_W
            valid_hw = mask_m & valid_h & (iw >= 0) & (iw < XW)

            for k0 in range(0, CIN_PER_GROUP, BLOCK_K):
                for kk in tl.static_range(0, BLOCK_K):
                    ic_local = k0 + kk
                    mask_k = ic_local < CIN_PER_GROUP
                    ic_global = pid_g * CIN_PER_GROUP + ic_local

                    x = tl.load(
                        x_ptr
                        + batch_idx * (C_IN * XH * XW)
                        + ic_global * (XH * XW)
                        + ih * XW
                        + iw,
                        mask=valid_hw & mask_k,
                        other=0.0,
                    ).to(tl.float64)

                    w = tl.load(
                        w_ptr
                        + oc_global * (CIN_PER_GROUP * KH * KW)
                        + (ic_local * KH + kh) * KW
                        + kw,
                        mask=mask_oc & mask_k,
                        other=0.0,
                    ).to(tl.float64)

                    acc += w[:, None] * x[None, :]

    y_ptrs = (
        y_ptr
        + batch_idx[None, :] * (C_OUT * OH * OW)
        + oc_global[:, None] * (OH * OW)
        + hw[None, :]
    )
    tl.store(y_ptrs, acc, mask=mask_oc[:, None] & mask_m[None, :])


@triton.jit
def conv2d_fp64_vector_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    total_elements,
    KDIM: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_e = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    mask_e = offs_e < total_elements

    ow = offs_e % OW
    oh = (offs_e // OW) % OH
    oc = (offs_e // (OH * OW)) % C_OUT
    batch = offs_e // (C_OUT * OH * OW)

    group = oc // COUT_PER_GROUP
    kernel_area = KH * KW

    acc = tl.zeros((BLOCK_E,), dtype=tl.float64)

    if HAS_BIAS:
        b = tl.load(bias_ptr + oc, mask=mask_e, other=0.0).to(tl.float64)
        acc += b

    offs_k_base = tl.arange(0, BLOCK_K)

    for k0 in range(0, KDIM, BLOCK_K):
        offs_k = k0 + offs_k_base
        mask_k = offs_k < KDIM

        ic_local = offs_k // kernel_area
        rem = offs_k - ic_local * kernel_area
        kh_idx = rem // KW
        kw_idx = rem - kh_idx * KW

        ic_global = group[None, :] * CIN_PER_GROUP + ic_local[:, None]

        ih = oh[None, :] * STRIDE_H - PAD_TOP + kh_idx[:, None] * DIL_H
        iw = ow[None, :] * STRIDE_W - PAD_LEFT + kw_idx[:, None] * DIL_W

        valid = (
            mask_k[:, None]
            & mask_e[None, :]
            & (ih >= 0)
            & (ih < XH)
            & (iw >= 0)
            & (iw < XW)
        )

        x_ptrs = (
            x_ptr
            + batch[None, :] * (C_IN * XH * XW)
            + ic_global * (XH * XW)
            + ih * XW
            + iw
        )
        x = tl.load(x_ptrs, mask=valid, other=0.0).to(tl.float64)

        w_ptrs = (
            w_ptr + oc[None, :] * (CIN_PER_GROUP * KH * KW) + offs_k[:, None]
        )
        ww = tl.load(
            w_ptrs,
            mask=mask_k[:, None] & mask_e[None, :],
            other=0.0,
        ).to(tl.float64)

        acc += tl.sum(x * ww, axis=0)

    y_ptrs = y_ptr + batch * (C_OUT * OH * OW) + oc * (OH * OW) + oh * OW + ow
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask_e)


@triton.jit
def conv3d_conv3d_spatial_ncdhw_m_kernel_variant(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
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
):
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)

    DHW = OD * OH * OW
    XHW = XH * XW
    XCDHW = C_IN * XD * XHW
    YCDHW = C_OUT * DHW
    KERNEL_VOLUME = KD * KH * KW
    KDIM = CIN_PER_GROUP * KERNEL_VOLUME

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_oc = offs_oc < COUT_PER_GROUP

    batch_idx = offs_m // DHW
    spatial = offs_m - batch_idx * DHW
    od = spatial // (OH * OW)
    rem_ohw = spatial - od * (OH * OW)
    oh = rem_ohw // OW
    ow = rem_ohw - oh * OW

    acc = tl.zeros((BLOCK_OC, BLOCK_M), dtype=tl.float32)

    for k0 in range(0, KDIM, BLOCK_K):
        offs_k = k0 + offs_k_base
        mask_k = offs_k < KDIM

        ic_local = offs_k // KERNEL_VOLUME
        rem_kernel = offs_k - ic_local * KERNEL_VOLUME
        kd_idx = rem_kernel // (KH * KW)
        rem_hw = rem_kernel - kd_idx * (KH * KW)
        kh_idx = rem_hw // KW
        kw_idx = rem_hw - kh_idx * KW
        ic_global = pid_g * CIN_PER_GROUP + ic_local

        id_in = od[None, :] * STRIDE_D - PAD_FRONT + kd_idx[:, None] * DIL_D
        ih = oh[None, :] * STRIDE_H - PAD_TOP + kh_idx[:, None] * DIL_H
        iw = ow[None, :] * STRIDE_W - PAD_LEFT + kw_idx[:, None] * DIL_W
        valid = (
            mask_k[:, None]
            & mask_m[None, :]
            & (id_in >= 0)
            & (id_in < XD)
            & (ih >= 0)
            & (ih < XH)
            & (iw >= 0)
            & (iw < XW)
        )

        x_ptrs = (
            x_ptr
            + batch_idx[None, :] * XCDHW
            + ic_global[:, None] * (XD * XHW)
            + id_in * XHW
            + ih * XW
            + iw
        )
        x = tl.load(x_ptrs, mask=valid, other=0.0)

        w_ptrs = (
            w_ptr
            + (pid_g * COUT_PER_GROUP + offs_oc[:, None]) * KDIM
            + offs_k[None, :]
        )
        w = tl.load(w_ptrs, mask=mask_oc[:, None] & mask_k[None, :], other=0.0)
        acc = tl.dot(w, x, acc, input_precision="tf32")

    oc_global = pid_g * COUT_PER_GROUP + offs_oc
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[:, None]

    y_ptrs = (
        y_ptr
        + batch_idx[None, :] * YCDHW
        + oc_global[:, None] * DHW
        + spatial[None, :]
    )
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_m[None, :],
    )


@triton.jit
def conv3d_fp64_ncdhw_m_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
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
):
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)

    DHW = OD * OH * OW
    XHW = XH * XW
    XCDHW = C_IN * XD * XHW
    YCDHW = C_OUT * DHW

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)

    mask_m = offs_m < M
    mask_oc = offs_oc < COUT_PER_GROUP

    batch_idx = offs_m // DHW
    spatial = offs_m - batch_idx * DHW
    od = spatial // (OH * OW)
    rem_ohw = spatial - od * (OH * OW)
    oh = rem_ohw // OW
    ow = rem_ohw - oh * OW

    oc_global = pid_g * COUT_PER_GROUP + offs_oc
    acc = tl.zeros((BLOCK_OC, BLOCK_M), dtype=tl.float64)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0).to(
            tl.float64
        )
        acc += bias[:, None]

    for kd in tl.static_range(0, KD):
        id_in = od * STRIDE_D - PAD_FRONT + kd * DIL_D
        valid_d = (id_in >= 0) & (id_in < XD)
        for kh in tl.static_range(0, KH):
            ih = oh * STRIDE_H - PAD_TOP + kh * DIL_H
            valid_h = valid_d & (ih >= 0) & (ih < XH)
            for kw in tl.static_range(0, KW):
                iw = ow * STRIDE_W - PAD_LEFT + kw * DIL_W
                valid_spatial = mask_m & valid_h & (iw >= 0) & (iw < XW)
                for c0 in range(0, CIN_PER_GROUP, BLOCK_K):
                    for kk in tl.static_range(0, BLOCK_K):
                        ic_local = c0 + kk
                        mask_k = ic_local < CIN_PER_GROUP
                        ic_global = pid_g * CIN_PER_GROUP + ic_local

                        x = tl.load(
                            x_ptr
                            + batch_idx * XCDHW
                            + ic_global * (XD * XHW)
                            + id_in * XHW
                            + ih * XW
                            + iw,
                            mask=valid_spatial & mask_k,
                            other=0.0,
                        ).to(tl.float64)
                        w = tl.load(
                            w_ptr
                            + oc_global * (CIN_PER_GROUP * KD * KH * KW)
                            + ((ic_local * KD + kd) * KH + kh) * KW
                            + kw,
                            mask=mask_oc & mask_k,
                            other=0.0,
                        ).to(tl.float64)
                        acc += w[:, None] * x[None, :]

    y_ptrs = (
        y_ptr
        + batch_idx[None, :] * YCDHW
        + oc_global[:, None] * DHW
        + spatial[None, :]
    )
    tl.store(y_ptrs, acc, mask=mask_oc[:, None] & mask_m[None, :])


@triton.jit
def _conv_dgrad1d_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    X_LEN: tl.constexpr,
    LOSS_LEN: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_l: tl.constexpr,
    weight_stride_o: tl.constexpr,
    weight_stride_i: tl.constexpr,
    weight_stride_k: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_l: tl.constexpr,
    STRIDE_L: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_L: tl.constexpr,
    KL: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    n_idx = offs_m // X_LEN
    x_idx = offs_m - n_idx * X_LEN
    ci = group * CIN_PER_GROUP + offs_ci_rel

    acc = tl.zeros((BLOCK_CI, BLOCK_M), dtype=tl.float32)

    for kl in tl.static_range(0, KL):
        loss_l_num = x_idx + PAD_LEFT - kl * DIL_L
        loss_l = loss_l_num // STRIDE_L
        valid_l = (loss_l_num >= 0) & (loss_l < LOSS_LEN)
        if STRIDE_L != 1:
            valid_l = valid_l & ((loss_l_num % STRIDE_L) == 0)
        safe_l = tl.where(valid_l, loss_l, 0)
        weight_l = KL - 1 - kl if FILTER_REVERSE else kl

        for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
            offs_co_rel = co_start + tl.arange(0, BLOCK_CO)
            co = group * COUT_PER_GROUP + offs_co_rel
            mask_co = offs_co_rel < COUT_PER_GROUP

            loss = tl.load(
                loss_ptr
                + n_idx[None, :] * loss_stride_n
                + co[:, None] * loss_stride_c
                + safe_l[None, :] * loss_stride_l,
                mask=mask_co[:, None] & mask_m[None, :] & valid_l[None, :],
                other=0.0,
            )
            weight = tl.load(
                weight_ptr
                + co[None, :] * weight_stride_o
                + offs_ci_rel[:, None] * weight_stride_i
                + weight_l * weight_stride_k,
                mask=mask_ci[:, None] & mask_co[None, :],
                other=0.0,
            )
            acc += tl.dot(weight, loss, out_dtype=tl.float32)

    tl.store(
        out_ptr
        + n_idx[None, :] * out_stride_n
        + ci[:, None] * out_stride_c
        + x_idx[None, :] * out_stride_l,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_ci[:, None] & mask_m[None, :],
    )


@triton.jit
def _conv_dgrad1d_mci_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    X_LEN: tl.constexpr,
    LOSS_LEN: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_l: tl.constexpr,
    weight_stride_o: tl.constexpr,
    weight_stride_i: tl.constexpr,
    weight_stride_k: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_l: tl.constexpr,
    STRIDE_L: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_L: tl.constexpr,
    KL: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    n_idx = offs_m // X_LEN
    x_idx = offs_m - n_idx * X_LEN
    ci = group * CIN_PER_GROUP + offs_ci_rel

    acc = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for kl in tl.static_range(0, KL):
        loss_l_num = x_idx + PAD_LEFT - kl * DIL_L
        loss_l = loss_l_num // STRIDE_L
        valid_l = (loss_l_num >= 0) & (loss_l < LOSS_LEN)
        if STRIDE_L != 1:
            valid_l = valid_l & ((loss_l_num % STRIDE_L) == 0)
        safe_l = tl.where(valid_l, loss_l, 0)
        weight_l = KL - 1 - kl if FILTER_REVERSE else kl

        for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
            offs_co_rel = co_start + tl.arange(0, BLOCK_CO)
            co = group * COUT_PER_GROUP + offs_co_rel
            mask_co = offs_co_rel < COUT_PER_GROUP

            loss = tl.load(
                loss_ptr
                + n_idx[:, None] * loss_stride_n
                + co[None, :] * loss_stride_c
                + safe_l[:, None] * loss_stride_l,
                mask=mask_m[:, None] & mask_co[None, :] & valid_l[:, None],
                other=0.0,
            )
            weight = tl.load(
                weight_ptr
                + co[:, None] * weight_stride_o
                + offs_ci_rel[None, :] * weight_stride_i
                + weight_l * weight_stride_k,
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            acc += tl.dot(loss, weight, out_dtype=tl.float32)

    tl.store(
        out_ptr
        + n_idx[:, None] * out_stride_n
        + ci[None, :] * out_stride_c
        + x_idx[:, None] * out_stride_l,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_dgrad2d_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    weight_stride_o: tl.constexpr,
    weight_stride_i: tl.constexpr,
    weight_stride_h: tl.constexpr,
    weight_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    spatial = XH * XW
    n_idx = offs_m // spatial
    spatial_idx = offs_m - n_idx * spatial
    xh = spatial_idx // XW
    xw = spatial_idx - xh * XW
    ci = group * CIN_PER_GROUP + offs_ci_rel

    acc = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        loss_h_num = xh + PAD_TOP - kh * DIL_H
        loss_h = loss_h_num // STRIDE_H
        valid_h = (loss_h_num >= 0) & (loss_h < LOSS_H)
        if STRIDE_H != 1:
            valid_h = valid_h & ((loss_h_num % STRIDE_H) == 0)
        safe_h = tl.where(valid_h, loss_h, 0)
        weight_h = KH - 1 - kh if FILTER_REVERSE else kh

        for kw in tl.static_range(0, KW):
            loss_w_num = xw + PAD_LEFT - kw * DIL_W
            loss_w = loss_w_num // STRIDE_W
            valid_w = (loss_w_num >= 0) & (loss_w < LOSS_W)
            if STRIDE_W != 1:
                valid_w = valid_w & ((loss_w_num % STRIDE_W) == 0)
            valid_hw = valid_h & valid_w
            safe_w = tl.where(valid_w, loss_w, 0)
            weight_w = KW - 1 - kw if FILTER_REVERSE else kw

            for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
                offs_co_rel = co_start + tl.arange(0, BLOCK_CO)
                co = group * COUT_PER_GROUP + offs_co_rel
                mask_co = offs_co_rel < COUT_PER_GROUP

                loss = tl.load(
                    loss_ptr
                    + n_idx[:, None] * loss_stride_n
                    + co[None, :] * loss_stride_c
                    + safe_h[:, None] * loss_stride_h
                    + safe_w[:, None] * loss_stride_w,
                    mask=(
                        mask_m[:, None] & mask_co[None, :] & valid_hw[:, None]
                    ),
                    other=0.0,
                )
                weight = tl.load(
                    weight_ptr
                    + co[:, None] * weight_stride_o
                    + offs_ci_rel[None, :] * weight_stride_i
                    + weight_h * weight_stride_h
                    + weight_w * weight_stride_w,
                    mask=mask_co[:, None] & mask_ci[None, :],
                    other=0.0,
                )
                acc += tl.dot(loss, weight, out_dtype=tl.float32)

    tl.store(
        out_ptr
        + n_idx[:, None] * out_stride_n
        + ci[None, :] * out_stride_c
        + xh[:, None] * out_stride_h
        + xw[:, None] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_dgrad2d_1x1_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    weight_stride_o: tl.constexpr,
    weight_stride_i: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    spatial = XH * XW
    n_idx = offs_m // spatial
    spatial_idx = offs_m - n_idx * spatial
    ci = group * CIN_PER_GROUP + offs_ci_rel

    acc = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
        offs_co_rel = co_start + tl.arange(0, BLOCK_CO)
        co = group * COUT_PER_GROUP + offs_co_rel
        mask_co = offs_co_rel < COUT_PER_GROUP

        loss = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + co[None, :] * loss_stride_c
            + spatial_idx[:, None],
            mask=mask_m[:, None] & mask_co[None, :],
            other=0.0,
        )
        weight = tl.load(
            weight_ptr
            + co[:, None] * weight_stride_o
            + offs_ci_rel[None, :] * weight_stride_i,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        acc += tl.dot(
            loss,
            weight,
            out_dtype=tl.float32,
            input_precision="tf32",
        )

    tl.store(
        out_ptr
        + n_idx[:, None] * out_stride_n
        + ci[None, :] * out_stride_c
        + spatial_idx[:, None],
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_dgrad2d_1x1_strided_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    weight_stride_o: tl.constexpr,
    weight_stride_i: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    spatial = XH * XW
    n_idx = offs_m // spatial
    spatial_idx = offs_m - n_idx * spatial
    xh = spatial_idx // XW
    xw = spatial_idx - xh * XW
    ci = group * CIN_PER_GROUP + offs_ci_rel

    acc = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
        offs_co_rel = co_start + tl.arange(0, BLOCK_CO)
        co = group * COUT_PER_GROUP + offs_co_rel
        mask_co = offs_co_rel < COUT_PER_GROUP

        loss = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + co[None, :] * loss_stride_c
            + xh[:, None] * loss_stride_h
            + xw[:, None] * loss_stride_w,
            mask=mask_m[:, None] & mask_co[None, :],
            other=0.0,
        )
        weight = tl.load(
            weight_ptr
            + co[:, None] * weight_stride_o
            + offs_ci_rel[None, :] * weight_stride_i,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        acc += tl.dot(
            loss,
            weight,
            out_dtype=tl.float32,
            input_precision="tf32",
        )

    tl.store(
        out_ptr
        + n_idx[:, None] * out_stride_n
        + ci[None, :] * out_stride_c
        + xh[:, None] * out_stride_h
        + xw[:, None] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_dgrad2d_stride1_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    weight_stride_o: tl.constexpr,
    weight_stride_i: tl.constexpr,
    weight_stride_h: tl.constexpr,
    weight_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    spatial = XH * XW
    n_idx = offs_m // spatial
    spatial_idx = offs_m - n_idx * spatial
    xh = spatial_idx // XW
    xw = spatial_idx - xh * XW
    ci = group * CIN_PER_GROUP + offs_ci_rel

    acc = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        loss_h = xh + PAD_TOP - kh * DIL_H
        valid_h = (loss_h >= 0) & (loss_h < LOSS_H)
        safe_h = tl.where(valid_h, loss_h, 0)
        weight_h = KH - 1 - kh if FILTER_REVERSE else kh

        for kw in tl.static_range(0, KW):
            loss_w = xw + PAD_LEFT - kw * DIL_W
            valid_w = (loss_w >= 0) & (loss_w < LOSS_W)
            valid_hw = valid_h & valid_w
            safe_w = tl.where(valid_w, loss_w, 0)
            weight_w = KW - 1 - kw if FILTER_REVERSE else kw

            for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
                offs_co_rel = co_start + tl.arange(0, BLOCK_CO)
                co = group * COUT_PER_GROUP + offs_co_rel
                mask_co = offs_co_rel < COUT_PER_GROUP

                loss = tl.load(
                    loss_ptr
                    + n_idx[:, None] * loss_stride_n
                    + co[None, :] * loss_stride_c
                    + safe_h[:, None] * loss_stride_h
                    + safe_w[:, None] * loss_stride_w,
                    mask=(
                        mask_m[:, None] & mask_co[None, :] & valid_hw[:, None]
                    ),
                    other=0.0,
                )
                weight = tl.load(
                    weight_ptr
                    + co[:, None] * weight_stride_o
                    + offs_ci_rel[None, :] * weight_stride_i
                    + weight_h * weight_stride_h
                    + weight_w * weight_stride_w,
                    mask=mask_co[:, None] & mask_ci[None, :],
                    other=0.0,
                )
                acc += tl.dot(loss, weight, out_dtype=tl.float32)

    tl.store(
        out_ptr
        + n_idx[:, None] * out_stride_n
        + ci[None, :] * out_stride_c
        + xh[:, None] * out_stride_h
        + xw[:, None] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_dgrad2d_stride2_pad1_3x3_packed_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
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
    FILTER_REVERSE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    parity = tl.program_id(1)
    group = tl.program_id(2)
    ph = parity // 2
    pw = parity - ph * 2

    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    parity_h_count = (XH + 1 - ph) // 2
    parity_w_count = (XW + 1 - pw) // 2
    parity_spatial = parity_h_count * parity_w_count
    m_parity = N * parity_spatial

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < m_parity
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    n_idx = offs_m // parity_spatial
    spatial_idx = offs_m - n_idx * parity_spatial
    yh = spatial_idx // parity_w_count
    yw = spatial_idx - yh * parity_w_count
    xh = yh * 2 + ph
    xw = yw * 2 + pw
    ci = group * CIN_PER_GROUP + offs_ci_rel

    acc = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for kh_i in tl.static_range(0, 2):
        kh = tl.where(ph == 0, 1, kh_i * 2)
        loss_h = tl.where(ph == 0, yh, yh + (1 if kh_i == 0 else 0))
        valid_h = loss_h < LOSS_H
        if kh_i == 1:
            valid_h = valid_h & (ph != 0)
        weight_h = 2 - kh if FILTER_REVERSE else kh

        for kw_i in tl.static_range(0, 2):
            kw = tl.where(pw == 0, 1, kw_i * 2)
            loss_w = tl.where(pw == 0, yw, yw + (1 if kw_i == 0 else 0))
            valid_w = loss_w < LOSS_W
            if kw_i == 1:
                valid_w = valid_w & (pw != 0)
            valid_hw = valid_h & valid_w
            weight_w = 2 - kw if FILTER_REVERSE else kw

            for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
                offs_co_rel = co_start + tl.arange(0, BLOCK_CO)
                co = group * COUT_PER_GROUP + offs_co_rel
                mask_co = offs_co_rel < COUT_PER_GROUP

                loss = tl.load(
                    loss_ptr
                    + n_idx[:, None] * loss_stride_n
                    + co[None, :] * loss_stride_c
                    + loss_h[:, None] * loss_stride_h
                    + loss_w[:, None] * loss_stride_w,
                    mask=(
                        mask_m[:, None] & mask_co[None, :] & valid_hw[:, None]
                    ),
                    other=0.0,
                )
                weight = tl.load(
                    weight_ptr
                    + (
                        (
                            ((group * 3 + weight_h) * 3 + weight_w)
                            * COUT_PER_GROUP
                            + offs_co_rel[:, None]
                        )
                        * CIN_PER_GROUP
                    )
                    + offs_ci_rel[None, :],
                    mask=mask_co[:, None] & mask_ci[None, :],
                    other=0.0,
                )
                acc += tl.dot(
                    loss, weight, out_dtype=tl.float32, input_precision="tf32"
                )

    tl.store(
        out_ptr
        + n_idx[:, None] * out_stride_n
        + ci[None, :] * out_stride_c
        + xh[:, None] * out_stride_h
        + xw[:, None] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_dgrad2d_stride2_pad1_3x3_packed_mci_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
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
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    parity_spatial = PARITY_H_COUNT * PARITY_W_COUNT
    n_idx = offs_m // parity_spatial
    spatial_idx = offs_m - n_idx * parity_spatial
    yh = spatial_idx // PARITY_W_COUNT
    yw = spatial_idx - yh * PARITY_W_COUNT
    xh = yh * 2 + PH
    xw = yw * 2 + PW
    ci = offs_ci_rel

    acc = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for kh_i in tl.static_range(0, KH_COUNT):
        if PH == 0:
            kh = 1
            loss_h = yh
        else:
            kh = kh_i * 2
            loss_h = yh + (1 if kh_i == 0 else 0)
        valid_h = loss_h < LOSS_H
        weight_h = 2 - kh if FILTER_REVERSE else kh

        for kw_i in tl.static_range(0, KW_COUNT):
            if PW == 0:
                kw = 1
                loss_w = yw
            else:
                kw = kw_i * 2
                loss_w = yw + (1 if kw_i == 0 else 0)
            valid_w = loss_w < LOSS_W
            valid_hw = valid_h & valid_w
            weight_w = 2 - kw if FILTER_REVERSE else kw

            for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
                offs_co_rel = co_start + tl.arange(0, BLOCK_CO)
                mask_co = offs_co_rel < COUT_PER_GROUP

                loss = tl.load(
                    loss_ptr
                    + n_idx[:, None] * loss_stride_n
                    + offs_co_rel[None, :] * loss_stride_c
                    + loss_h[:, None] * loss_stride_h
                    + loss_w[:, None] * loss_stride_w,
                    mask=(
                        mask_m[:, None] & mask_co[None, :] & valid_hw[:, None]
                    ),
                    other=0.0,
                )
                weight = tl.load(
                    weight_ptr
                    + (
                        (
                            (weight_h * 3 + weight_w) * COUT_PER_GROUP
                            + offs_co_rel[:, None]
                        )
                        * CIN_PER_GROUP
                    )
                    + offs_ci_rel[None, :],
                    mask=mask_co[:, None] & mask_ci[None, :],
                    other=0.0,
                )
                acc += tl.dot(
                    loss, weight, out_dtype=tl.float32, input_precision="tf32"
                )

    tl.store(
        out_ptr
        + n_idx[:, None] * out_stride_n
        + ci[None, :] * out_stride_c
        + xh[:, None] * out_stride_h
        + xw[:, None] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_dgrad2d_stride2_pad1_3x3_tile2w_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
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
    PH: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    parity_spatial = PARITY_H_COUNT * LOSS_W
    n_idx = offs_m // parity_spatial
    spatial_idx = offs_m - n_idx * parity_spatial
    yh = spatial_idx // LOSS_W
    yw = spatial_idx - yh * LOSS_W
    xh = yh * 2 + PH
    xw0 = yw * 2
    xw1 = xw0 + 1
    yh1 = yh + 1
    yw1 = yw + 1
    ci = offs_ci_rel

    valid0 = mask_m & (xh < XH) & (xw0 < XW)
    valid1 = mask_m & (xh < XH) & (xw1 < XW)
    valid_yh1 = yh1 < LOSS_H
    valid_yw1 = yw1 < LOSS_W

    if FILTER_REVERSE:
        w0 = 2
        w1 = 1
        w2 = 0
    else:
        w0 = 0
        w1 = 1
        w2 = 2

    acc0 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
        offs_co_rel = co_start + tl.arange(0, BLOCK_CO)
        mask_co = offs_co_rel < COUT_PER_GROUP

        loss00 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + offs_co_rel[None, :] * loss_stride_c
            + yh[:, None] * loss_stride_h
            + yw[:, None] * loss_stride_w,
            mask=mask_m[:, None] & mask_co[None, :],
            other=0.0,
        )
        loss01 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + offs_co_rel[None, :] * loss_stride_c
            + yh[:, None] * loss_stride_h
            + yw1[:, None] * loss_stride_w,
            mask=mask_m[:, None] & mask_co[None, :] & valid_yw1[:, None],
            other=0.0,
        )

        if PH == 0:
            weight11 = tl.load(
                weight_ptr
                + (
                    ((w1 * 3 + w1) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight12 = tl.load(
                weight_ptr
                + (
                    ((w1 * 3 + w2) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight10 = tl.load(
                weight_ptr
                + (
                    ((w1 * 3 + w0) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            acc0 += tl.dot(
                loss00, weight11, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss00, weight12, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss01, weight10, out_dtype=tl.float32, input_precision="tf32"
            )
        else:
            loss10 = tl.load(
                loss_ptr
                + n_idx[:, None] * loss_stride_n
                + offs_co_rel[None, :] * loss_stride_c
                + yh1[:, None] * loss_stride_h
                + yw[:, None] * loss_stride_w,
                mask=mask_m[:, None] & mask_co[None, :] & valid_yh1[:, None],
                other=0.0,
            )
            loss11 = tl.load(
                loss_ptr
                + n_idx[:, None] * loss_stride_n
                + offs_co_rel[None, :] * loss_stride_c
                + yh1[:, None] * loss_stride_h
                + yw1[:, None] * loss_stride_w,
                mask=mask_m[:, None]
                & mask_co[None, :]
                & valid_yh1[:, None]
                & valid_yw1[:, None],
                other=0.0,
            )
            weight21 = tl.load(
                weight_ptr
                + (
                    ((w2 * 3 + w1) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight01 = tl.load(
                weight_ptr
                + (
                    ((w0 * 3 + w1) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight22 = tl.load(
                weight_ptr
                + (
                    ((w2 * 3 + w2) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight20 = tl.load(
                weight_ptr
                + (
                    ((w2 * 3 + w0) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight02 = tl.load(
                weight_ptr
                + (
                    ((w0 * 3 + w2) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight00 = tl.load(
                weight_ptr
                + (
                    ((w0 * 3 + w0) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            acc0 += tl.dot(
                loss00, weight21, out_dtype=tl.float32, input_precision="tf32"
            )
            acc0 += tl.dot(
                loss10, weight01, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss00, weight22, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss01, weight20, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss10, weight02, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss11, weight00, out_dtype=tl.float32, input_precision="tf32"
            )

    out_base = (
        out_ptr + n_idx[:, None] * out_stride_n + ci[None, :] * out_stride_c
    )
    tl.store(
        out_base + xh[:, None] * out_stride_h + xw0[:, None] * out_stride_w,
        acc0.to(out_ptr.dtype.element_ty),
        mask=valid0[:, None] & mask_ci[None, :],
    )
    tl.store(
        out_base + xh[:, None] * out_stride_h + xw1[:, None] * out_stride_w,
        acc1.to(out_ptr.dtype.element_ty),
        mask=valid1[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_dgrad2d_stride2_pad1_3x3_tile2w_splitk_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
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
    PH: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    GROUP_K: tl.constexpr,
    SPLIT_K_BLOCKS: tl.constexpr,
    K_OFFSET: tl.constexpr,
    STORE: tl.constexpr,
):
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid = tl.program_id(0)
    pid_k_rel = pid % SPLIT_K_BLOCKS
    pid_k_group = pid_k_rel + K_OFFSET
    pid_tmp = pid // SPLIT_K_BLOCKS
    pid_m = pid_tmp % num_m_blocks
    pid_ci = pid_tmp // num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    parity_spatial = PARITY_H_COUNT * LOSS_W
    n_idx = offs_m // parity_spatial
    spatial_idx = offs_m - n_idx * parity_spatial
    yh = spatial_idx // LOSS_W
    yw = spatial_idx - yh * LOSS_W
    xh = yh * 2 + PH
    xw0 = yw * 2
    xw1 = xw0 + 1
    yh1 = yh + 1
    yw1 = yw + 1
    ci = offs_ci_rel

    valid0 = mask_m & (xh < XH) & (xw0 < XW)
    valid1 = mask_m & (xh < XH) & (xw1 < XW)
    valid_yh1 = yh1 < LOSS_H
    valid_yw1 = yw1 < LOSS_W

    if FILTER_REVERSE:
        w0 = 2
        w1 = 1
        w2 = 0
    else:
        w0 = 0
        w1 = 1
        w2 = 2

    acc0 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for k_inner in tl.static_range(0, GROUP_K):
        pid_k = pid_k_group * GROUP_K + k_inner
        offs_co_rel = pid_k * BLOCK_CO + tl.arange(0, BLOCK_CO)
        mask_co = offs_co_rel < COUT_PER_GROUP

        loss00 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + offs_co_rel[None, :] * loss_stride_c
            + yh[:, None] * loss_stride_h
            + yw[:, None] * loss_stride_w,
            mask=mask_m[:, None] & mask_co[None, :],
            other=0.0,
        )
        loss01 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + offs_co_rel[None, :] * loss_stride_c
            + yh[:, None] * loss_stride_h
            + yw1[:, None] * loss_stride_w,
            mask=mask_m[:, None] & mask_co[None, :] & valid_yw1[:, None],
            other=0.0,
        )

        if PH == 0:
            weight11 = tl.load(
                weight_ptr
                + (
                    ((w1 * 3 + w1) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight12 = tl.load(
                weight_ptr
                + (
                    ((w1 * 3 + w2) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight10 = tl.load(
                weight_ptr
                + (
                    ((w1 * 3 + w0) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            acc0 += tl.dot(
                loss00, weight11, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss00, weight12, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss01, weight10, out_dtype=tl.float32, input_precision="tf32"
            )
        else:
            loss10 = tl.load(
                loss_ptr
                + n_idx[:, None] * loss_stride_n
                + offs_co_rel[None, :] * loss_stride_c
                + yh1[:, None] * loss_stride_h
                + yw[:, None] * loss_stride_w,
                mask=mask_m[:, None] & mask_co[None, :] & valid_yh1[:, None],
                other=0.0,
            )
            loss11 = tl.load(
                loss_ptr
                + n_idx[:, None] * loss_stride_n
                + offs_co_rel[None, :] * loss_stride_c
                + yh1[:, None] * loss_stride_h
                + yw1[:, None] * loss_stride_w,
                mask=(
                    mask_m[:, None]
                    & mask_co[None, :]
                    & valid_yh1[:, None]
                    & valid_yw1[:, None]
                ),
                other=0.0,
            )
            weight21 = tl.load(
                weight_ptr
                + (
                    ((w2 * 3 + w1) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight01 = tl.load(
                weight_ptr
                + (
                    ((w0 * 3 + w1) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight22 = tl.load(
                weight_ptr
                + (
                    ((w2 * 3 + w2) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight20 = tl.load(
                weight_ptr
                + (
                    ((w2 * 3 + w0) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight02 = tl.load(
                weight_ptr
                + (
                    ((w0 * 3 + w2) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight00 = tl.load(
                weight_ptr
                + (
                    ((w0 * 3 + w0) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            acc0 += tl.dot(
                loss00, weight21, out_dtype=tl.float32, input_precision="tf32"
            )
            acc0 += tl.dot(
                loss10, weight01, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss00, weight22, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss01, weight20, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss10, weight02, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss11, weight00, out_dtype=tl.float32, input_precision="tf32"
            )

    out_base = (
        out_ptr + n_idx[:, None] * out_stride_n + ci[None, :] * out_stride_c
    )
    ptr0 = out_base + xh[:, None] * out_stride_h + xw0[:, None] * out_stride_w
    ptr1 = out_base + xh[:, None] * out_stride_h + xw1[:, None] * out_stride_w
    mask = mask_m[:, None] & mask_ci[None, :]
    if STORE:
        tl.store(ptr0, acc0, mask=valid0[:, None] & mask_ci[None, :])
        tl.store(ptr1, acc1, mask=valid1[:, None] & mask_ci[None, :])
    else:
        tl.atomic_add(ptr0, acc0, sem="relaxed", mask=mask)
        tl.atomic_add(ptr1, acc1, sem="relaxed", mask=mask)


@triton.jit
def _conv_dgrad2d_stride2_pad1_3x3_p5_tile2w_splitk_kernel(
    loss,
    weight,
    out,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
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
    PH: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    GROUP_K: tl.constexpr,
    SPLIT_K_BLOCKS: tl.constexpr,
    K_OFFSET: tl.constexpr,
    STORE: tl.constexpr,
):
    num_m_blocks = tl.cdiv(400, BLOCK_M)
    pid = tl.program_id(0)
    pid_m = pid % num_m_blocks
    pid_tmp = pid // num_m_blocks
    pid_k_rel = pid_tmp % SPLIT_K_BLOCKS
    pid_k_group = pid_k_rel + K_OFFSET
    pid_ci = pid_tmp // SPLIT_K_BLOCKS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < 400
    mask_ci = offs_ci < 768

    yh = offs_m // 20
    yw = offs_m - yh * 20
    x_base = (yh * 2 + PH) * 40 + yw * 2
    loss_base = yh * 20 + yw
    valid_yh1 = yh < 19
    valid_yw1 = yw < 19

    acc0 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for k_inner in tl.static_range(0, GROUP_K):
        pid_k = pid_k_group * GROUP_K + k_inner
        offs_co = pid_k * BLOCK_CO + tl.arange(0, BLOCK_CO)
        mask_co = offs_co < 768
        loss00 = tl.load(
            loss + offs_co[None, :] * 400 + loss_base[:, None],
            mask=mask_m[:, None] & mask_co[None, :],
            other=0.0,
        )
        loss01 = tl.load(
            loss + offs_co[None, :] * 400 + (loss_base + 1)[:, None],
            mask=mask_m[:, None] & mask_co[None, :] & valid_yw1[:, None],
            other=0.0,
        )
        if PH == 0:
            w11 = tl.load(
                weight
                + ((1 * 3 + 1) * 768 + offs_co[:, None]) * 768
                + offs_ci[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            w12 = tl.load(
                weight
                + ((1 * 3 + 2) * 768 + offs_co[:, None]) * 768
                + offs_ci[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            w10 = tl.load(
                weight
                + ((1 * 3 + 0) * 768 + offs_co[:, None]) * 768
                + offs_ci[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            acc0 += tl.dot(
                loss00, w11, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss00, w12, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss01, w10, out_dtype=tl.float32, input_precision="tf32"
            )
        else:
            loss10 = tl.load(
                loss + offs_co[None, :] * 400 + (loss_base + 20)[:, None],
                mask=mask_m[:, None] & mask_co[None, :] & valid_yh1[:, None],
                other=0.0,
            )
            loss11 = tl.load(
                loss + offs_co[None, :] * 400 + (loss_base + 21)[:, None],
                mask=mask_m[:, None]
                & mask_co[None, :]
                & valid_yh1[:, None]
                & valid_yw1[:, None],
                other=0.0,
            )
            w21 = tl.load(
                weight
                + ((2 * 3 + 1) * 768 + offs_co[:, None]) * 768
                + offs_ci[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            w01 = tl.load(
                weight
                + ((0 * 3 + 1) * 768 + offs_co[:, None]) * 768
                + offs_ci[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            w22 = tl.load(
                weight
                + ((2 * 3 + 2) * 768 + offs_co[:, None]) * 768
                + offs_ci[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            w20 = tl.load(
                weight
                + ((2 * 3 + 0) * 768 + offs_co[:, None]) * 768
                + offs_ci[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            w02 = tl.load(
                weight
                + ((0 * 3 + 2) * 768 + offs_co[:, None]) * 768
                + offs_ci[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            w00 = tl.load(
                weight
                + ((0 * 3 + 0) * 768 + offs_co[:, None]) * 768
                + offs_ci[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            acc0 += tl.dot(
                loss00, w21, out_dtype=tl.float32, input_precision="tf32"
            )
            acc0 += tl.dot(
                loss10, w01, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss00, w22, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss01, w20, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss10, w02, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss11, w00, out_dtype=tl.float32, input_precision="tf32"
            )

    ptr0 = out + offs_ci[None, :] * 1600 + x_base[:, None]
    ptr1 = ptr0 + 1
    mask = mask_m[:, None] & mask_ci[None, :]
    if STORE:
        tl.store(ptr0, acc0, mask=mask)
        tl.store(ptr1, acc1, mask=mask)
    else:
        tl.atomic_add(ptr0, acc0, sem="relaxed", mask=mask)
        tl.atomic_add(ptr1, acc1, sem="relaxed", mask=mask)


@triton.jit
def _conv_dgrad2d_p5_zero_kernel(
    out,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(
        out + offs, tl.zeros((BLOCK,), dtype=tl.float32), mask=offs < TOTAL
    )


@triton.jit
def _conv_dgrad2d_stride2_pad1_3x3_tile4_splitk_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    weight_stride_o: tl.constexpr,
    weight_stride_i: tl.constexpr,
    weight_stride_h: tl.constexpr,
    weight_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    GROUP_K: tl.constexpr,
    SPLIT_K_BLOCKS: tl.constexpr,
    K_OFFSET: tl.constexpr,
    STORE: tl.constexpr,
):
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid = tl.program_id(0)
    pid_k_rel = pid % SPLIT_K_BLOCKS
    pid_k_group = pid_k_rel + K_OFFSET
    pid_tmp = pid // SPLIT_K_BLOCKS
    pid_m = pid_tmp % num_m_blocks
    pid_ci = pid_tmp // num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    n_idx = offs_m // (LOSS_H * LOSS_W)
    spatial = offs_m - n_idx * (LOSS_H * LOSS_W)
    yh = spatial // LOSS_W
    yw = spatial - yh * LOSS_W
    xh0 = yh * 2
    xw0 = yw * 2
    xh1 = xh0 + 1
    xw1 = xw0 + 1
    yh1 = yh + 1
    yw1 = yw + 1
    ci = offs_ci_rel
    valid_yh1 = yh1 < LOSS_H
    valid_yw1 = yw1 < LOSS_W

    if FILTER_REVERSE:
        w0 = 2
        w1 = 1
        w2 = 0
    else:
        w0 = 0
        w1 = 1
        w2 = 2

    acc00 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for k_inner in tl.static_range(0, GROUP_K):
        pid_k = pid_k_group * GROUP_K + k_inner
        offs_co_rel = pid_k * BLOCK_CO + tl.arange(0, BLOCK_CO)
        mask_co = offs_co_rel < COUT_PER_GROUP

        loss00 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + offs_co_rel[None, :] * loss_stride_c
            + yh[:, None] * loss_stride_h
            + yw[:, None] * loss_stride_w,
            mask=mask_m[:, None] & mask_co[None, :],
            other=0.0,
        )
        loss01 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + offs_co_rel[None, :] * loss_stride_c
            + yh[:, None] * loss_stride_h
            + yw1[:, None] * loss_stride_w,
            mask=mask_m[:, None] & mask_co[None, :] & valid_yw1[:, None],
            other=0.0,
        )
        loss10 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + offs_co_rel[None, :] * loss_stride_c
            + yh1[:, None] * loss_stride_h
            + yw[:, None] * loss_stride_w,
            mask=mask_m[:, None] & mask_co[None, :] & valid_yh1[:, None],
            other=0.0,
        )
        loss11 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + offs_co_rel[None, :] * loss_stride_c
            + yh1[:, None] * loss_stride_h
            + yw1[:, None] * loss_stride_w,
            mask=mask_m[:, None]
            & mask_co[None, :]
            & valid_yh1[:, None]
            & valid_yw1[:, None],
            other=0.0,
        )

        weight11 = tl.load(
            weight_ptr
            + offs_co_rel[:, None] * weight_stride_o
            + ci[None, :] * weight_stride_i
            + w1 * weight_stride_h
            + w1 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight12 = tl.load(
            weight_ptr
            + offs_co_rel[:, None] * weight_stride_o
            + ci[None, :] * weight_stride_i
            + w1 * weight_stride_h
            + w2 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight10 = tl.load(
            weight_ptr
            + offs_co_rel[:, None] * weight_stride_o
            + ci[None, :] * weight_stride_i
            + w1 * weight_stride_h
            + w0 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight21 = tl.load(
            weight_ptr
            + offs_co_rel[:, None] * weight_stride_o
            + ci[None, :] * weight_stride_i
            + w2 * weight_stride_h
            + w1 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight01 = tl.load(
            weight_ptr
            + offs_co_rel[:, None] * weight_stride_o
            + ci[None, :] * weight_stride_i
            + w0 * weight_stride_h
            + w1 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight22 = tl.load(
            weight_ptr
            + offs_co_rel[:, None] * weight_stride_o
            + ci[None, :] * weight_stride_i
            + w2 * weight_stride_h
            + w2 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight20 = tl.load(
            weight_ptr
            + offs_co_rel[:, None] * weight_stride_o
            + ci[None, :] * weight_stride_i
            + w2 * weight_stride_h
            + w0 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight02 = tl.load(
            weight_ptr
            + offs_co_rel[:, None] * weight_stride_o
            + ci[None, :] * weight_stride_i
            + w0 * weight_stride_h
            + w2 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight00 = tl.load(
            weight_ptr
            + offs_co_rel[:, None] * weight_stride_o
            + ci[None, :] * weight_stride_i
            + w0 * weight_stride_h
            + w0 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )

        acc00 += tl.dot(
            loss00, weight11, out_dtype=tl.float32, input_precision="tf32"
        )
        acc01 += tl.dot(
            loss00, weight12, out_dtype=tl.float32, input_precision="tf32"
        )
        acc01 += tl.dot(
            loss01, weight10, out_dtype=tl.float32, input_precision="tf32"
        )
        acc10 += tl.dot(
            loss00, weight21, out_dtype=tl.float32, input_precision="tf32"
        )
        acc10 += tl.dot(
            loss10, weight01, out_dtype=tl.float32, input_precision="tf32"
        )
        acc11 += tl.dot(
            loss00, weight22, out_dtype=tl.float32, input_precision="tf32"
        )
        acc11 += tl.dot(
            loss01, weight20, out_dtype=tl.float32, input_precision="tf32"
        )
        acc11 += tl.dot(
            loss10, weight02, out_dtype=tl.float32, input_precision="tf32"
        )
        acc11 += tl.dot(
            loss11, weight00, out_dtype=tl.float32, input_precision="tf32"
        )

    out_base = (
        out_ptr + n_idx[:, None] * out_stride_n + ci[None, :] * out_stride_c
    )
    ptr00 = (
        out_base + xh0[:, None] * out_stride_h + xw0[:, None] * out_stride_w
    )
    ptr01 = (
        out_base + xh0[:, None] * out_stride_h + xw1[:, None] * out_stride_w
    )
    ptr10 = (
        out_base + xh1[:, None] * out_stride_h + xw0[:, None] * out_stride_w
    )
    ptr11 = (
        out_base + xh1[:, None] * out_stride_h + xw1[:, None] * out_stride_w
    )
    mask = mask_m[:, None] & mask_ci[None, :]
    if STORE:
        tl.store(ptr00, acc00, mask=mask)
        tl.store(ptr01, acc01, mask=mask)
        tl.store(ptr10, acc10, mask=mask)
        tl.store(ptr11, acc11, mask=mask)
    else:
        tl.atomic_add(ptr00, acc00, sem="relaxed", mask=mask)
        tl.atomic_add(ptr01, acc01, sem="relaxed", mask=mask)
        tl.atomic_add(ptr10, acc10, sem="relaxed", mask=mask)
        tl.atomic_add(ptr11, acc11, sem="relaxed", mask=mask)


@triton.jit
def _conv_dgrad2d_stride2_pad1_3x3_tile4_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    weight_stride_o: tl.constexpr,
    weight_stride_i: tl.constexpr,
    weight_stride_h: tl.constexpr,
    weight_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    offs_bco = tl.arange(0, BLOCK_CO)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    loss_spatial = LOSS_H * LOSS_W
    n_idx = offs_m // loss_spatial
    spatial_idx = offs_m - n_idx * loss_spatial
    yh = spatial_idx // LOSS_W
    yw = spatial_idx - yh * LOSS_W
    xh0 = yh * 2
    xw0 = yw * 2
    xh1 = xh0 + 1
    xw1 = xw0 + 1
    yh1 = yh + 1
    yw1 = yw + 1

    valid00 = mask_m & (xh0 < XH) & (xw0 < XW)
    valid01 = mask_m & (xh0 < XH) & (xw1 < XW)
    valid10 = mask_m & (xh1 < XH) & (xw0 < XW)
    valid11 = mask_m & (xh1 < XH) & (xw1 < XW)
    valid_yh1 = yh1 < LOSS_H
    valid_yw1 = yw1 < LOSS_W

    ci = group * CIN_PER_GROUP + offs_ci_rel
    if FILTER_REVERSE:
        w0 = 2
        w1 = 1
        w2 = 0
    else:
        w0 = 0
        w1 = 1
        w2 = 2

    acc00 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
        offs_co_rel = co_start + offs_bco
        co = group * COUT_PER_GROUP + offs_co_rel
        mask_co = offs_co_rel < COUT_PER_GROUP

        loss00 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + co[None, :] * loss_stride_c
            + yh[:, None] * loss_stride_h
            + yw[:, None] * loss_stride_w,
            mask=mask_m[:, None] & mask_co[None, :],
            other=0.0,
        )
        loss01 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + co[None, :] * loss_stride_c
            + yh[:, None] * loss_stride_h
            + yw1[:, None] * loss_stride_w,
            mask=(mask_m[:, None] & mask_co[None, :] & valid_yw1[:, None]),
            other=0.0,
        )
        loss10 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + co[None, :] * loss_stride_c
            + yh1[:, None] * loss_stride_h
            + yw[:, None] * loss_stride_w,
            mask=(mask_m[:, None] & mask_co[None, :] & valid_yh1[:, None]),
            other=0.0,
        )
        loss11 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + co[None, :] * loss_stride_c
            + yh1[:, None] * loss_stride_h
            + yw1[:, None] * loss_stride_w,
            mask=(
                mask_m[:, None]
                & mask_co[None, :]
                & valid_yh1[:, None]
                & valid_yw1[:, None]
            ),
            other=0.0,
        )

        weight11 = tl.load(
            weight_ptr
            + co[:, None] * weight_stride_o
            + offs_ci_rel[None, :] * weight_stride_i
            + w1 * weight_stride_h
            + w1 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight12 = tl.load(
            weight_ptr
            + co[:, None] * weight_stride_o
            + offs_ci_rel[None, :] * weight_stride_i
            + w1 * weight_stride_h
            + w2 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight10 = tl.load(
            weight_ptr
            + co[:, None] * weight_stride_o
            + offs_ci_rel[None, :] * weight_stride_i
            + w1 * weight_stride_h
            + w0 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight21 = tl.load(
            weight_ptr
            + co[:, None] * weight_stride_o
            + offs_ci_rel[None, :] * weight_stride_i
            + w2 * weight_stride_h
            + w1 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight01 = tl.load(
            weight_ptr
            + co[:, None] * weight_stride_o
            + offs_ci_rel[None, :] * weight_stride_i
            + w0 * weight_stride_h
            + w1 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight22 = tl.load(
            weight_ptr
            + co[:, None] * weight_stride_o
            + offs_ci_rel[None, :] * weight_stride_i
            + w2 * weight_stride_h
            + w2 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight20 = tl.load(
            weight_ptr
            + co[:, None] * weight_stride_o
            + offs_ci_rel[None, :] * weight_stride_i
            + w2 * weight_stride_h
            + w0 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight02 = tl.load(
            weight_ptr
            + co[:, None] * weight_stride_o
            + offs_ci_rel[None, :] * weight_stride_i
            + w0 * weight_stride_h
            + w2 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight00 = tl.load(
            weight_ptr
            + co[:, None] * weight_stride_o
            + offs_ci_rel[None, :] * weight_stride_i
            + w0 * weight_stride_h
            + w0 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )

        acc00 += tl.dot(
            loss00, weight11, out_dtype=tl.float32, input_precision="tf32"
        )
        acc01 += tl.dot(
            loss00, weight12, out_dtype=tl.float32, input_precision="tf32"
        )
        acc01 += tl.dot(
            loss01, weight10, out_dtype=tl.float32, input_precision="tf32"
        )
        acc10 += tl.dot(
            loss00, weight21, out_dtype=tl.float32, input_precision="tf32"
        )
        acc10 += tl.dot(
            loss10, weight01, out_dtype=tl.float32, input_precision="tf32"
        )
        acc11 += tl.dot(
            loss00, weight22, out_dtype=tl.float32, input_precision="tf32"
        )
        acc11 += tl.dot(
            loss01, weight20, out_dtype=tl.float32, input_precision="tf32"
        )
        acc11 += tl.dot(
            loss10, weight02, out_dtype=tl.float32, input_precision="tf32"
        )
        acc11 += tl.dot(
            loss11, weight00, out_dtype=tl.float32, input_precision="tf32"
        )

    out_base = (
        out_ptr + n_idx[:, None] * out_stride_n + ci[None, :] * out_stride_c
    )
    tl.store(
        out_base + xh0[:, None] * out_stride_h + xw0[:, None] * out_stride_w,
        acc00.to(out_ptr.dtype.element_ty),
        mask=valid00[:, None] & mask_ci[None, :],
    )
    tl.store(
        out_base + xh0[:, None] * out_stride_h + xw1[:, None] * out_stride_w,
        acc01.to(out_ptr.dtype.element_ty),
        mask=valid01[:, None] & mask_ci[None, :],
    )
    tl.store(
        out_base + xh1[:, None] * out_stride_h + xw0[:, None] * out_stride_w,
        acc10.to(out_ptr.dtype.element_ty),
        mask=valid10[:, None] & mask_ci[None, :],
    )
    tl.store(
        out_base + xh1[:, None] * out_stride_h + xw1[:, None] * out_stride_w,
        acc11.to(out_ptr.dtype.element_ty),
        mask=valid11[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_dgrad2d_stride2_pad1_3x3_merged_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    weight_stride_o: tl.constexpr,
    weight_stride_i: tl.constexpr,
    weight_stride_h: tl.constexpr,
    weight_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    parity = tl.program_id(1)
    ph = parity // 2
    pw = parity - ph * 2

    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    parity_h_count = (XH + 1 - ph) // 2
    parity_w_count = (XW + 1 - pw) // 2
    parity_spatial = parity_h_count * parity_w_count
    m_parity = N * parity_spatial

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < m_parity
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    n_idx = offs_m // parity_spatial
    spatial_idx = offs_m - n_idx * parity_spatial
    yh = spatial_idx // parity_w_count
    yw = spatial_idx - yh * parity_w_count
    xh = yh * 2 + ph
    xw = yw * 2 + pw
    ci = offs_ci_rel

    acc = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for kh_i in tl.static_range(0, 2):
        kh = tl.where(ph == 0, 1, kh_i * 2)
        loss_h = tl.where(ph == 0, yh, yh + (1 if kh_i == 0 else 0))
        valid_h = loss_h < LOSS_H
        if kh_i == 1:
            valid_h = valid_h & (ph != 0)
        weight_h = 2 - kh if FILTER_REVERSE else kh

        for kw_i in tl.static_range(0, 2):
            kw = tl.where(pw == 0, 1, kw_i * 2)
            loss_w = tl.where(pw == 0, yw, yw + (1 if kw_i == 0 else 0))
            valid_w = loss_w < LOSS_W
            if kw_i == 1:
                valid_w = valid_w & (pw != 0)
            valid_hw = valid_h & valid_w
            weight_w = 2 - kw if FILTER_REVERSE else kw

            for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
                offs_co_rel = co_start + tl.arange(0, BLOCK_CO)
                mask_co = offs_co_rel < COUT_PER_GROUP

                loss = tl.load(
                    loss_ptr
                    + n_idx[:, None] * loss_stride_n
                    + offs_co_rel[None, :] * loss_stride_c
                    + loss_h[:, None] * loss_stride_h
                    + loss_w[:, None] * loss_stride_w,
                    mask=(
                        mask_m[:, None] & mask_co[None, :] & valid_hw[:, None]
                    ),
                    other=0.0,
                )
                weight = tl.load(
                    weight_ptr
                    + offs_co_rel[:, None] * weight_stride_o
                    + offs_ci_rel[None, :] * weight_stride_i
                    + weight_h * weight_stride_h
                    + weight_w * weight_stride_w,
                    mask=mask_co[:, None] & mask_ci[None, :],
                    other=0.0,
                )
                acc += tl.dot(loss, weight, out_dtype=tl.float32)

    tl.store(
        out_ptr
        + n_idx[:, None] * out_stride_n
        + ci[None, :] * out_stride_c
        + xh[:, None] * out_stride_h
        + xw[:, None] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_dgrad2d_stride2_pad1_3x3_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    weight_stride_o: tl.constexpr,
    weight_stride_i: tl.constexpr,
    weight_stride_h: tl.constexpr,
    weight_stride_w: tl.constexpr,
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
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    parity_spatial = PARITY_H_COUNT * PARITY_W_COUNT
    n_idx = offs_m // parity_spatial
    spatial_idx = offs_m - n_idx * parity_spatial
    yh = spatial_idx // PARITY_W_COUNT
    yw = spatial_idx - yh * PARITY_W_COUNT
    xh = yh * 2 + PH
    xw = yw * 2 + PW
    ci = offs_ci_rel

    acc = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for kh_i in tl.static_range(0, KH_COUNT):
        if PH == 0:
            kh = 1
            loss_h = yh
        else:
            kh = kh_i * 2
            loss_h = yh + (1 if kh_i == 0 else 0)
        valid_h = loss_h < LOSS_H
        weight_h = 2 - kh if FILTER_REVERSE else kh

        for kw_i in tl.static_range(0, KW_COUNT):
            if PW == 0:
                kw = 1
                loss_w = yw
            else:
                kw = kw_i * 2
                loss_w = yw + (1 if kw_i == 0 else 0)
            valid_w = loss_w < LOSS_W
            valid_hw = valid_h & valid_w
            weight_w = 2 - kw if FILTER_REVERSE else kw

            for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
                offs_co_rel = co_start + tl.arange(0, BLOCK_CO)
                mask_co = offs_co_rel < COUT_PER_GROUP

                loss = tl.load(
                    loss_ptr
                    + n_idx[:, None] * loss_stride_n
                    + offs_co_rel[None, :] * loss_stride_c
                    + loss_h[:, None] * loss_stride_h
                    + loss_w[:, None] * loss_stride_w,
                    mask=(
                        mask_m[:, None] & mask_co[None, :] & valid_hw[:, None]
                    ),
                    other=0.0,
                )
                weight = tl.load(
                    weight_ptr
                    + offs_co_rel[:, None] * weight_stride_o
                    + offs_ci_rel[None, :] * weight_stride_i
                    + weight_h * weight_stride_h
                    + weight_w * weight_stride_w,
                    mask=mask_co[:, None] & mask_ci[None, :],
                    other=0.0,
                )
                acc += tl.dot(loss, weight, out_dtype=tl.float32)

    tl.store(
        out_ptr
        + n_idx[:, None] * out_stride_n
        + ci[None, :] * out_stride_c
        + xh[:, None] * out_stride_h
        + xw[:, None] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_dgrad3d_pad1_3x3_fp32_split_dot_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M_PART: tl.constexpr,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_D: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_d: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    weight_stride_o: tl.constexpr,
    weight_stride_i: tl.constexpr,
    weight_stride_d: tl.constexpr,
    weight_stride_h: tl.constexpr,
    weight_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_d: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    PART: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci = tl.arange(0, 8)
    offs_co = tl.arange(0, 16)
    mask_m = offs_m < M_PART

    int_d = XD - 2
    int_h = XH - 2
    int_w = XW - 2
    if PART == 0:
        part_spatial = int_d * int_h * int_w
        n_idx = offs_m // part_spatial
        part_idx = offs_m - n_idx * part_spatial
        xd = part_idx // (int_h * int_w) + 1
        rem = part_idx - (xd - 1) * (int_h * int_w)
        xh = rem // int_w + 1
        xw = rem - (xh - 1) * int_w + 1
    else:
        d_faces = 2 * XH * XW
        h_faces = int_d * 2 * XW
        part_spatial = d_faces + h_faces + int_d * int_h * 2
        n_idx = offs_m // part_spatial
        part_idx = offs_m - n_idx * part_spatial

        in_d = part_idx < d_faces
        in_h = (part_idx >= d_faces) & (part_idx < d_faces + h_faces)

        d_side = part_idx // (XH * XW)
        d_rem = part_idx - d_side * (XH * XW)
        xd_d = tl.where(d_side == 0, 0, XD - 1)
        xh_d = d_rem // XW
        xw_d = d_rem - xh_d * XW

        h_idx = part_idx - d_faces
        h_d = h_idx // (2 * XW)
        h_rem = h_idx - h_d * (2 * XW)
        h_side = h_rem // XW
        xd_h = h_d + 1
        xh_h = tl.where(h_side == 0, 0, XH - 1)
        xw_h = h_rem - h_side * XW

        w_idx = part_idx - d_faces - h_faces
        w_pair = w_idx // 2
        w_side = w_idx - w_pair * 2
        xd_w = w_pair // int_h + 1
        xh_w = w_pair - (xd_w - 1) * int_h + 1
        xw_w = tl.where(w_side == 0, 0, XW - 1)

        xd = tl.where(in_d, xd_d, tl.where(in_h, xd_h, xd_w))
        xh = tl.where(in_d, xh_d, tl.where(in_h, xh_h, xh_w))
        xw = tl.where(in_d, xw_d, tl.where(in_h, xw_h, xw_w))

    acc = tl.zeros((BLOCK_M, 8), dtype=tl.float32)

    for kd in tl.static_range(0, 3):
        ld = xd + 1 - kd
        if PART == 0:
            valid_d = tl.full((BLOCK_M,), True, dtype=tl.int1)
            safe_d = ld
        else:
            valid_d = (ld >= 0) & (ld < LOSS_D)
            safe_d = tl.where(valid_d, ld, 0)
        for kh in tl.static_range(0, 3):
            lh = xh + 1 - kh
            if PART == 0:
                valid_h = tl.full((BLOCK_M,), True, dtype=tl.int1)
                safe_h = lh
            else:
                valid_h = (lh >= 0) & (lh < LOSS_H)
                safe_h = tl.where(valid_h, lh, 0)
            for kw in tl.static_range(0, 3):
                lw = xw + 1 - kw
                if PART == 0:
                    valid_dhw = tl.full((BLOCK_M,), True, dtype=tl.int1)
                    safe_w = lw
                else:
                    valid_w = (lw >= 0) & (lw < LOSS_W)
                    safe_w = tl.where(valid_w, lw, 0)
                    valid_dhw = valid_d & valid_h & valid_w

                loss = tl.load(
                    loss_ptr
                    + n_idx[:, None] * loss_stride_n
                    + offs_co[None, :] * loss_stride_c
                    + safe_d[:, None] * loss_stride_d
                    + safe_h[:, None] * loss_stride_h
                    + safe_w[:, None] * loss_stride_w,
                    mask=mask_m[:, None] & valid_dhw[:, None],
                    other=0.0,
                )
                weight = tl.load(
                    weight_ptr
                    + offs_co[:, None] * weight_stride_o
                    + offs_ci[None, :] * weight_stride_i
                    + kd * weight_stride_d
                    + kh * weight_stride_h
                    + kw * weight_stride_w,
                )
                acc += tl.dot(
                    loss,
                    weight,
                    out_dtype=tl.float32,
                    input_precision="tf32",
                )

    tl.store(
        out_ptr
        + n_idx[:, None] * out_stride_n
        + offs_ci[None, :] * out_stride_c
        + xd[:, None] * out_stride_d
        + xh[:, None] * out_stride_h
        + xw[:, None] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None],
    )


@triton.jit
def _conv_dgrad3d_small_fp32_ci8_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_D: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_d: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    weight_stride_o: tl.constexpr,
    weight_stride_i: tl.constexpr,
    weight_stride_d: tl.constexpr,
    weight_stride_h: tl.constexpr,
    weight_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_d: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    PAD_FRONT: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci = tl.arange(0, 8)
    mask_m = offs_m < M

    spatial_hw = XH * XW
    spatial = XD * spatial_hw
    n_idx = offs_m // spatial
    spatial_idx = offs_m - n_idx * spatial
    xd = spatial_idx // spatial_hw
    rem = spatial_idx - xd * spatial_hw
    xh = rem // XW
    xw = rem - xh * XW

    acc = tl.zeros((BLOCK_M, 8), dtype=tl.float32)

    for kd in tl.static_range(0, KD):
        loss_d = xd + PAD_FRONT - kd
        valid_d = (loss_d >= 0) & (loss_d < LOSS_D)
        safe_d = tl.where(valid_d, loss_d, 0)
        for kh in tl.static_range(0, KH):
            loss_h = xh + PAD_TOP - kh
            valid_h = (loss_h >= 0) & (loss_h < LOSS_H)
            safe_h = tl.where(valid_h, loss_h, 0)
            for kw in tl.static_range(0, KW):
                loss_w = xw + PAD_LEFT - kw
                valid_w = (loss_w >= 0) & (loss_w < LOSS_W)
                safe_w = tl.where(valid_w, loss_w, 0)
                valid_dhw = valid_d & valid_h & valid_w

                for co in tl.static_range(0, COUT_PER_GROUP):
                    loss = tl.load(
                        loss_ptr
                        + n_idx * loss_stride_n
                        + co * loss_stride_c
                        + safe_d * loss_stride_d
                        + safe_h * loss_stride_h
                        + safe_w * loss_stride_w,
                        mask=mask_m & valid_dhw,
                        other=0.0,
                    )
                    weight = tl.load(
                        weight_ptr
                        + co * weight_stride_o
                        + offs_ci * weight_stride_i
                        + kd * weight_stride_d
                        + kh * weight_stride_h
                        + kw * weight_stride_w,
                    )
                    acc += loss[:, None] * weight[None, :]

    tl.store(
        out_ptr
        + n_idx[:, None] * out_stride_n
        + offs_ci[None, :] * out_stride_c
        + xd[:, None] * out_stride_d
        + xh[:, None] * out_stride_h
        + xw[:, None] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None],
    )


@triton.jit
def _conv_dgrad3d_pad1_3x3_fp32_ci8_dot_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_D: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_d: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_d: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci = tl.arange(0, 16)
    offs_co = tl.arange(0, 16)
    mask_m = offs_m < M
    mask_ci = offs_ci < 8

    spatial_hw = XH * XW
    spatial = XD * spatial_hw
    n_idx = offs_m // spatial
    spatial_idx = offs_m - n_idx * spatial
    xd = spatial_idx // spatial_hw
    rem = spatial_idx - xd * spatial_hw
    xh = rem // XW
    xw = rem - xh * XW

    acc = tl.zeros((16, BLOCK_M), dtype=tl.float32)

    for kd in tl.static_range(0, 3):
        loss_d = xd + 1 - kd
        valid_d = (loss_d >= 0) & (loss_d < LOSS_D)
        safe_d = tl.where(valid_d, loss_d, 0)
        for kh in tl.static_range(0, 3):
            loss_h = xh + 1 - kh
            valid_h = (loss_h >= 0) & (loss_h < LOSS_H)
            safe_h = tl.where(valid_h, loss_h, 0)
            for kw in tl.static_range(0, 3):
                loss_w = xw + 1 - kw
                valid_w = (loss_w >= 0) & (loss_w < LOSS_W)
                safe_w = tl.where(valid_w, loss_w, 0)
                valid_dhw = valid_d & valid_h & valid_w

                loss = tl.load(
                    loss_ptr
                    + n_idx[None, :] * loss_stride_n
                    + offs_co[:, None] * loss_stride_c
                    + safe_d[None, :] * loss_stride_d
                    + safe_h[None, :] * loss_stride_h
                    + safe_w[None, :] * loss_stride_w,
                    mask=mask_m[None, :] & valid_dhw[None, :],
                    other=0.0,
                )
                weight = tl.load(
                    weight_ptr
                    + (((kd * 3 + kh) * 3 + kw) * 16 + offs_co[None, :]) * 8
                    + offs_ci[:, None],
                    mask=mask_ci[:, None],
                    other=0.0,
                )
                acc += tl.dot(
                    weight,
                    loss,
                    out_dtype=tl.float32,
                    input_precision="tf32",
                )

    tl.store(
        out_ptr
        + n_idx[None, :] * out_stride_n
        + offs_ci[:, None] * out_stride_c
        + xd[None, :] * out_stride_d
        + xh[None, :] * out_stride_h
        + xw[None, :] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_ci[:, None] & mask_m[None, :],
    )


@triton.jit
def _conv_dgrad3d_packed_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_D: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_d: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_d: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
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
    FILTER_REVERSE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    spatial_hw = XH * XW
    spatial = XD * spatial_hw
    n_idx = offs_m // spatial
    spatial_idx = offs_m - n_idx * spatial
    xd = spatial_idx // spatial_hw
    rem = spatial_idx - xd * spatial_hw
    xh = rem // XW
    xw = rem - xh * XW
    ci = group * CIN_PER_GROUP + offs_ci_rel

    acc = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for kd in tl.static_range(0, KD):
        loss_d_num = xd + PAD_FRONT - kd * DIL_D
        loss_d = loss_d_num // STRIDE_D
        valid_d = (loss_d_num >= 0) & (loss_d < LOSS_D)
        if STRIDE_D != 1:
            valid_d = valid_d & ((loss_d_num % STRIDE_D) == 0)
        safe_d = tl.where(valid_d, loss_d, 0)
        weight_d = KD - 1 - kd if FILTER_REVERSE else kd

        for kh in tl.static_range(0, KH):
            loss_h_num = xh + PAD_TOP - kh * DIL_H
            loss_h = loss_h_num // STRIDE_H
            valid_h = (loss_h_num >= 0) & (loss_h < LOSS_H)
            if STRIDE_H != 1:
                valid_h = valid_h & ((loss_h_num % STRIDE_H) == 0)
            safe_h = tl.where(valid_h, loss_h, 0)
            weight_h = KH - 1 - kh if FILTER_REVERSE else kh

            for kw in tl.static_range(0, KW):
                loss_w_num = xw + PAD_LEFT - kw * DIL_W
                loss_w = loss_w_num // STRIDE_W
                valid_w = (loss_w_num >= 0) & (loss_w < LOSS_W)
                if STRIDE_W != 1:
                    valid_w = valid_w & ((loss_w_num % STRIDE_W) == 0)
                valid_dhw = valid_d & valid_h & valid_w
                safe_w = tl.where(valid_w, loss_w, 0)
                weight_w = KW - 1 - kw if FILTER_REVERSE else kw

                for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
                    offs_co_rel = co_start + tl.arange(0, BLOCK_CO)
                    co = group * COUT_PER_GROUP + offs_co_rel
                    mask_co = offs_co_rel < COUT_PER_GROUP

                    loss = tl.load(
                        loss_ptr
                        + n_idx[:, None] * loss_stride_n
                        + co[None, :] * loss_stride_c
                        + safe_d[:, None] * loss_stride_d
                        + safe_h[:, None] * loss_stride_h
                        + safe_w[:, None] * loss_stride_w,
                        mask=(
                            mask_m[:, None]
                            & mask_co[None, :]
                            & valid_dhw[:, None]
                        ),
                        other=0.0,
                    )
                    weight = tl.load(
                        weight_ptr
                        + (
                            (
                                (
                                    ((group * KD + weight_d) * KH + weight_h)
                                    * KW
                                    + weight_w
                                )
                                * COUT_PER_GROUP
                                + offs_co_rel[:, None]
                            )
                            * CIN_PER_GROUP
                        )
                        + offs_ci_rel[None, :],
                        mask=mask_co[:, None] & mask_ci[None, :],
                        other=0.0,
                    )
                    acc += tl.dot(
                        loss,
                        weight,
                        out_dtype=tl.float32,
                        input_precision="tf32",
                    )

    tl.store(
        out_ptr
        + n_idx[:, None] * out_stride_n
        + ci[None, :] * out_stride_c
        + xd[:, None] * out_stride_d
        + xh[:, None] * out_stride_h
        + xw[:, None] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_dgrad3d_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_D: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_d: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    weight_stride_o: tl.constexpr,
    weight_stride_i: tl.constexpr,
    weight_stride_d: tl.constexpr,
    weight_stride_h: tl.constexpr,
    weight_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_d: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
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
    FILTER_REVERSE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    spatial_hw = XH * XW
    spatial = XD * spatial_hw
    n_idx = offs_m // spatial
    spatial_idx = offs_m - n_idx * spatial
    xd = spatial_idx // spatial_hw
    rem = spatial_idx - xd * spatial_hw
    xh = rem // XW
    xw = rem - xh * XW
    ci = group * CIN_PER_GROUP + offs_ci_rel

    acc = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for kd in tl.static_range(0, KD):
        loss_d_num = xd + PAD_FRONT - kd * DIL_D
        loss_d = loss_d_num // STRIDE_D
        valid_d = (loss_d_num >= 0) & (loss_d < LOSS_D)
        if STRIDE_D != 1:
            valid_d = valid_d & ((loss_d_num % STRIDE_D) == 0)
        safe_d = tl.where(valid_d, loss_d, 0)
        weight_d = KD - 1 - kd if FILTER_REVERSE else kd

        for kh in tl.static_range(0, KH):
            loss_h_num = xh + PAD_TOP - kh * DIL_H
            loss_h = loss_h_num // STRIDE_H
            valid_h = (loss_h_num >= 0) & (loss_h < LOSS_H)
            if STRIDE_H != 1:
                valid_h = valid_h & ((loss_h_num % STRIDE_H) == 0)
            safe_h = tl.where(valid_h, loss_h, 0)
            weight_h = KH - 1 - kh if FILTER_REVERSE else kh

            for kw in tl.static_range(0, KW):
                loss_w_num = xw + PAD_LEFT - kw * DIL_W
                loss_w = loss_w_num // STRIDE_W
                valid_w = (loss_w_num >= 0) & (loss_w < LOSS_W)
                if STRIDE_W != 1:
                    valid_w = valid_w & ((loss_w_num % STRIDE_W) == 0)
                valid_dhw = valid_d & valid_h & valid_w
                safe_w = tl.where(valid_w, loss_w, 0)
                weight_w = KW - 1 - kw if FILTER_REVERSE else kw

                for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
                    offs_co_rel = co_start + tl.arange(0, BLOCK_CO)
                    co = group * COUT_PER_GROUP + offs_co_rel
                    mask_co = offs_co_rel < COUT_PER_GROUP

                    loss = tl.load(
                        loss_ptr
                        + n_idx[:, None] * loss_stride_n
                        + co[None, :] * loss_stride_c
                        + safe_d[:, None] * loss_stride_d
                        + safe_h[:, None] * loss_stride_h
                        + safe_w[:, None] * loss_stride_w,
                        mask=(
                            mask_m[:, None]
                            & mask_co[None, :]
                            & valid_dhw[:, None]
                        ),
                        other=0.0,
                    )
                    weight = tl.load(
                        weight_ptr
                        + co[:, None] * weight_stride_o
                        + offs_ci_rel[None, :] * weight_stride_i
                        + weight_d * weight_stride_d
                        + weight_h * weight_stride_h
                        + weight_w * weight_stride_w,
                        mask=mask_co[:, None] & mask_ci[None, :],
                        other=0.0,
                    )
                    acc += tl.dot(
                        loss,
                        weight,
                        out_dtype=tl.float32,
                        input_precision="tf32",
                    )

    tl.store(
        out_ptr
        + n_idx[:, None] * out_stride_n
        + ci[None, :] * out_stride_c
        + xd[:, None] * out_stride_d
        + xh[:, None] * out_stride_h
        + xw[:, None] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_wgrad1d_kernel(
    image_ptr,
    loss_ptr,
    out_ptr,
    M: tl.constexpr,
    IMAGE_LEN: tl.constexpr,
    LOSS_LEN: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_l: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_l: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    out_stride_k: tl.constexpr,
    STRIDE_L: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_L: tl.constexpr,
    KL: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    k = tl.program_id(1)
    group = tl.program_id(2)

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + offs_ci_rel
    image_k = KL - 1 - k if FILTER_REVERSE else k

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for m_start in tl.range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        safe_m = tl.where(mask_m, offs_m, 0)
        n_idx = safe_m // LOSS_LEN
        loss_l = safe_m - n_idx * LOSS_LEN
        image_l = loss_l * STRIDE_L - PAD_LEFT + image_k * DIL_L
        valid_l = (image_l >= 0) & (image_l < IMAGE_LEN)
        safe_l = tl.where(valid_l, image_l, 0)

        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + co[:, None] * loss_stride_c
            + loss_l[None, :] * loss_stride_l,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_l[:, None] * image_stride_l,
            mask=mask_m[:, None] & mask_ci[None, :] & valid_l[:, None],
            other=0.0,
        )
        acc += tl.dot(
            loss,
            image,
            out_dtype=tl.float32,
            input_precision="tf32",
        )

    tl.store(
        out_ptr
        + co[:, None] * out_stride_o
        + offs_ci_rel[None, :] * out_stride_i
        + k * out_stride_k,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_wgrad1d_split_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    M: tl.constexpr,
    IMAGE_LEN: tl.constexpr,
    LOSS_LEN: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_l: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_l: tl.constexpr,
    STRIDE_L: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_L: tl.constexpr,
    KL: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    k = tl.program_id(1)
    split_group = tl.program_id(2)
    split = split_group % NUM_SPLITS
    group = split_group // NUM_SPLITS

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + offs_ci_rel
    image_k = KL - 1 - k if FILTER_REVERSE else k

    split_size = tl.cdiv(M, NUM_SPLITS)
    split_begin = split * split_size
    split_end = tl.minimum(split_begin + split_size, M)

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for m_start in tl.range(split_begin, split_end, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < split_end
        safe_m = tl.where(mask_m, offs_m, 0)
        n_idx = safe_m // LOSS_LEN
        loss_l = safe_m - n_idx * LOSS_LEN
        image_l = loss_l * STRIDE_L - PAD_LEFT + image_k * DIL_L
        valid_l = (image_l >= 0) & (image_l < IMAGE_LEN)
        safe_l = tl.where(valid_l, image_l, 0)

        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + co[:, None] * loss_stride_c
            + loss_l[None, :] * loss_stride_l,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_l[:, None] * image_stride_l,
            mask=mask_m[:, None] & mask_ci[None, :] & valid_l[:, None],
            other=0.0,
        )
        acc += tl.dot(
            loss,
            image,
            out_dtype=tl.float32,
            input_precision="tf32",
        )

    tl.store(
        partial_ptr
        + (
            (split * C_OUT + co[:, None]) * CIN_PER_GROUP
            + offs_ci_rel[None, :]
        )
        * KL
        + k,
        acc,
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_wgrad1d_reduce_kernel(
    partial_ptr,
    out_ptr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    out_stride_k: tl.constexpr,
    KL: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    pid = tl.program_id(0)
    k = tl.program_id(1)
    group = tl.program_id(2)

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for split in tl.static_range(0, NUM_SPLITS):
        acc += tl.load(
            partial_ptr
            + (
                (split * C_OUT + co[:, None]) * CIN_PER_GROUP
                + offs_ci_rel[None, :]
            )
            * KL
            + k,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )

    tl.store(
        out_ptr
        + co[:, None] * out_stride_o
        + offs_ci_rel[None, :] * out_stride_i
        + k * out_stride_k,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_wgrad1d_col_split_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    M: tl.constexpr,
    IMAGE_LEN: tl.constexpr,
    LOSS_LEN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_l: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_l: tl.constexpr,
    STRIDE_L: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_L: tl.constexpr,
    KL: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    split_group = tl.program_id(1)
    split = split_group % NUM_SPLITS
    group = split_group // NUM_SPLITS

    cik = CIN_PER_GROUP * KL
    num_n_blocks = tl.cdiv(cik, BLOCK_N)
    pid_co = pid // num_n_blocks
    pid_n = pid - pid_co * num_n_blocks

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_ci_rel = offs_n // KL
    k = offs_n - offs_ci_rel * KL
    image_k = KL - 1 - k if FILTER_REVERSE else k

    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_n = offs_n < cik
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + offs_ci_rel

    split_size = tl.cdiv(M, NUM_SPLITS)
    split_begin = split * split_size
    split_end = tl.minimum(split_begin + split_size, M)

    acc = tl.zeros((BLOCK_CO, BLOCK_N), dtype=tl.float32)
    for m_start in tl.range(split_begin, split_end, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < split_end
        safe_m = tl.where(mask_m, offs_m, 0)
        n_idx = safe_m // LOSS_LEN
        loss_l = safe_m - n_idx * LOSS_LEN

        image_l = (
            loss_l[:, None] * STRIDE_L - PAD_LEFT + image_k[None, :] * DIL_L
        )
        valid_l = (image_l >= 0) & (image_l < IMAGE_LEN)
        safe_l = tl.where(valid_l, image_l, 0)

        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + co[:, None] * loss_stride_c
            + loss_l[None, :] * loss_stride_l,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_l * image_stride_l,
            mask=mask_m[:, None] & mask_n[None, :] & valid_l,
            other=0.0,
        )
        acc += tl.dot(
            loss, image, out_dtype=tl.float32, input_precision="tf32"
        )

    tl.store(
        partial_ptr + (split * C_OUT + co[:, None]) * cik + offs_n[None, :],
        acc,
        mask=mask_co[:, None] & mask_n[None, :],
    )


@triton.jit
def _conv_wgrad1d_3tap_split_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    M: tl.constexpr,
    IMAGE_LEN: tl.constexpr,
    LOSS_LEN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_l: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_l: tl.constexpr,
    STRIDE_L: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_L: tl.constexpr,
    KL: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    split_group = tl.program_id(1)
    split = split_group % NUM_SPLITS
    group = split_group // NUM_SPLITS

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + offs_ci_rel

    image_k0 = KL - 1 if FILTER_REVERSE else 0
    image_k1 = KL - 2 if FILTER_REVERSE else 1
    image_k2 = KL - 3 if FILTER_REVERSE else 2

    split_size = tl.cdiv(M, NUM_SPLITS)
    split_begin = split * split_size
    split_end = tl.minimum(split_begin + split_size, M)

    acc0 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for m_start in tl.range(split_begin, split_end, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < split_end
        safe_m = tl.where(mask_m, offs_m, 0)
        n_idx = safe_m // LOSS_LEN
        loss_l = safe_m - n_idx * LOSS_LEN

        image_l0 = loss_l * STRIDE_L - PAD_LEFT + image_k0 * DIL_L
        image_l1 = loss_l * STRIDE_L - PAD_LEFT + image_k1 * DIL_L
        image_l2 = loss_l * STRIDE_L - PAD_LEFT + image_k2 * DIL_L
        valid0 = (image_l0 >= 0) & (image_l0 < IMAGE_LEN)
        valid1 = (image_l1 >= 0) & (image_l1 < IMAGE_LEN)
        valid2 = (image_l2 >= 0) & (image_l2 < IMAGE_LEN)
        safe_l0 = tl.where(valid0, image_l0, 0)
        safe_l1 = tl.where(valid1, image_l1, 0)
        safe_l2 = tl.where(valid2, image_l2, 0)

        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + co[:, None] * loss_stride_c
            + loss_l[None, :] * loss_stride_l,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image0 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_l0[:, None] * image_stride_l,
            mask=mask_m[:, None] & mask_ci[None, :] & valid0[:, None],
            other=0.0,
        )
        image1 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_l1[:, None] * image_stride_l,
            mask=mask_m[:, None] & mask_ci[None, :] & valid1[:, None],
            other=0.0,
        )
        image2 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_l2[:, None] * image_stride_l,
            mask=mask_m[:, None] & mask_ci[None, :] & valid2[:, None],
            other=0.0,
        )
        acc0 += tl.dot(
            loss, image0, out_dtype=tl.float32, input_precision="tf32"
        )
        acc1 += tl.dot(
            loss, image1, out_dtype=tl.float32, input_precision="tf32"
        )
        acc2 += tl.dot(
            loss, image2, out_dtype=tl.float32, input_precision="tf32"
        )

    base = (
        (split * C_OUT + co[:, None]) * CIN_PER_GROUP + offs_ci_rel[None, :]
    ) * KL
    mask = mask_co[:, None] & mask_ci[None, :]
    tl.store(partial_ptr + base + 0, acc0, mask=mask)
    tl.store(partial_ptr + base + 1, acc1, mask=mask)
    tl.store(partial_ptr + base + 2, acc2, mask=mask)


@triton.jit
def _conv_wgrad1d_col_reduce_kernel(
    partial_ptr,
    out_ptr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    out_stride_k: tl.constexpr,
    KL: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)

    cik = CIN_PER_GROUP * KL
    num_n_blocks = tl.cdiv(cik, BLOCK_N)
    pid_co = pid // num_n_blocks
    pid_n = pid - pid_co * num_n_blocks

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_ci_rel = offs_n // KL
    k = offs_n - offs_ci_rel * KL

    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_n = offs_n < cik
    co = group * COUT_PER_GROUP + offs_co_rel

    acc = tl.zeros((BLOCK_CO, BLOCK_N), dtype=tl.float32)
    for split in tl.static_range(0, NUM_SPLITS):
        acc += tl.load(
            partial_ptr
            + (split * C_OUT + co[:, None]) * cik
            + offs_n[None, :],
            mask=mask_co[:, None] & mask_n[None, :],
            other=0.0,
        )

    tl.store(
        out_ptr
        + co[:, None] * out_stride_o
        + offs_ci_rel[None, :] * out_stride_i
        + k[None, :] * out_stride_k,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_co[:, None] & mask_n[None, :],
    )


@triton.jit
def _conv_wgrad2d_kernel(
    image_ptr,
    loss_ptr,
    out_ptr,
    M: tl.constexpr,
    IMAGE_H: tl.constexpr,
    IMAGE_W: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_h: tl.constexpr,
    image_stride_w: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    k = tl.program_id(1)
    group = tl.program_id(2)

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    kh = k // KW
    kw = k - kh * KW
    image_kh = KH - 1 - kh if FILTER_REVERSE else kh
    image_kw = KW - 1 - kw if FILTER_REVERSE else kw

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + offs_ci_rel

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for m_start in tl.range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        safe_m = tl.where(mask_m, offs_m, 0)
        loss_w = safe_m % LOSS_W
        tmp = safe_m // LOSS_W
        loss_h = tmp % LOSS_H
        n_idx = tmp // LOSS_H

        image_h = loss_h * STRIDE_H - PAD_H + image_kh * DIL_H
        image_w = loss_w * STRIDE_W - PAD_W + image_kw * DIL_W
        valid_hw = (
            (image_h >= 0)
            & (image_h < IMAGE_H)
            & (image_w >= 0)
            & (image_w < IMAGE_W)
        )
        safe_h = tl.where(valid_hw, image_h, 0)
        safe_w = tl.where(valid_hw, image_w, 0)

        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + co[:, None] * loss_stride_c
            + loss_h[None, :] * loss_stride_h
            + loss_w[None, :] * loss_stride_w,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_h[:, None] * image_stride_h
            + safe_w[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid_hw[:, None],
            other=0.0,
        )
        acc += tl.dot(
            loss,
            image,
            out_dtype=tl.float32,
            input_precision="tf32",
        )

    tl.store(
        out_ptr
        + co[:, None] * out_stride_o
        + offs_ci_rel[None, :] * out_stride_i
        + kh * out_stride_h
        + kw * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_wgrad2d_1x1_kernel(
    image_ptr,
    loss_ptr,
    out_ptr,
    M: tl.constexpr,
    HW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + offs_ci_rel

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for m_start in tl.range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        safe_m = tl.where(mask_m, offs_m, 0)
        n_idx = safe_m // HW
        hw_idx = safe_m - n_idx * HW

        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + co[:, None] * loss_stride_c
            + hw_idx[None, :],
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + hw_idx[:, None],
            mask=mask_m[:, None] & mask_ci[None, :],
            other=0.0,
        )
        acc += tl.dot(
            loss,
            image,
            out_dtype=tl.float32,
            input_precision="tf32",
        )

    tl.store(
        out_ptr
        + co[:, None] * out_stride_o
        + offs_ci_rel[None, :] * out_stride_i,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_wgrad2d_1x1_split_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    M: tl.constexpr,
    HW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    group = tl.program_id(2)

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + offs_ci_rel

    split_size = tl.cdiv(M, NUM_SPLITS)
    split_begin = split * split_size
    split_end = tl.minimum(split_begin + split_size, M)

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for m_start in tl.range(split_begin, split_end, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < split_end
        safe_m = tl.where(mask_m, offs_m, 0)
        n_idx = safe_m // HW
        hw_idx = safe_m - n_idx * HW

        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + co[:, None] * loss_stride_c
            + hw_idx[None, :],
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + hw_idx[:, None],
            mask=mask_m[:, None] & mask_ci[None, :],
            other=0.0,
        )
        acc += tl.dot(
            loss,
            image,
            out_dtype=tl.float32,
            input_precision="tf32",
        )

    tl.store(
        partial_ptr
        + split * C_OUT * CIN_PER_GROUP
        + co[:, None] * CIN_PER_GROUP
        + offs_ci_rel[None, :],
        acc,
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_wgrad2d_1x1_split_nodiv_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    HW: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    SPLITS_PER_N: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    n_idx = split // SPLITS_PER_N
    split_in_n = split - n_idx * SPLITS_PER_N
    split_size = tl.cdiv(HW, SPLITS_PER_N)
    hw_begin = split_in_n * split_size
    hw_end = tl.minimum(hw_begin + split_size, HW)

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for hw_start in tl.range(hw_begin, hw_end, BLOCK_M):
        hw = hw_start + tl.arange(0, BLOCK_M)
        mask_m = hw < hw_end
        safe_hw = tl.where(mask_m, hw, 0)
        loss = tl.load(
            loss_ptr
            + n_idx * loss_stride_n
            + offs_co_rel[:, None] * loss_stride_c
            + safe_hw[None, :],
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx * image_stride_n
            + offs_ci_rel[None, :] * image_stride_c
            + safe_hw[:, None],
            mask=mask_m[:, None] & mask_ci[None, :],
            other=0.0,
        )
        acc += tl.dot(
            loss,
            image,
            out_dtype=tl.float32,
            input_precision="tf32",
        )

    tl.store(
        partial_ptr
        + split * C_OUT * CIN_PER_GROUP
        + offs_co_rel[:, None] * CIN_PER_GROUP
        + offs_ci_rel[None, :],
        acc,
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_wgrad_zero_kernel(out_ptr, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(
        out_ptr + offs,
        tl.zeros((BLOCK,), dtype=tl.float32),
        mask=offs < TOTAL,
    )


@triton.jit
def _conv_wgrad2d_1x1_atomic_nodiv_kernel(
    image_ptr,
    loss_ptr,
    out_ptr,
    HW: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    SPLITS_PER_N: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    n_idx = split // SPLITS_PER_N
    split_in_n = split - n_idx * SPLITS_PER_N
    split_size = tl.cdiv(HW, SPLITS_PER_N)
    hw_begin = split_in_n * split_size
    hw_end = tl.minimum(hw_begin + split_size, HW)

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for hw_start in tl.range(hw_begin, hw_end, BLOCK_M):
        hw = hw_start + tl.arange(0, BLOCK_M)
        mask_m = hw < hw_end
        safe_hw = tl.where(mask_m, hw, 0)
        loss = tl.load(
            loss_ptr
            + n_idx * loss_stride_n
            + offs_co_rel[:, None] * loss_stride_c
            + safe_hw[None, :],
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx * image_stride_n
            + offs_ci_rel[None, :] * image_stride_c
            + safe_hw[:, None],
            mask=mask_m[:, None] & mask_ci[None, :],
            other=0.0,
        )
        acc += tl.dot(
            loss,
            image,
            out_dtype=tl.float32,
            input_precision="tf32x3",
        )

    tl.atomic_add(
        out_ptr
        + offs_co_rel[:, None] * out_stride_o
        + offs_ci_rel[None, :] * out_stride_i,
        acc,
        sem="relaxed",
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_wgrad2d_1x1_reduce_kernel(
    partial_ptr,
    out_ptr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for split in tl.static_range(0, NUM_SPLITS):
        acc += tl.load(
            partial_ptr
            + split * C_OUT * CIN_PER_GROUP
            + co[:, None] * CIN_PER_GROUP
            + offs_ci_rel[None, :],
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )

    tl.store(
        out_ptr
        + co[:, None] * out_stride_o
        + offs_ci_rel[None, :] * out_stride_i,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_wgrad2d_split_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    M: tl.constexpr,
    IMAGE_H: tl.constexpr,
    IMAGE_W: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_h: tl.constexpr,
    image_stride_w: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    k = tl.program_id(1)
    split_group = tl.program_id(2)
    split = split_group % NUM_SPLITS
    group = split_group // NUM_SPLITS

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    kh = k // KW
    kw = k - kh * KW
    image_kh = KH - 1 - kh if FILTER_REVERSE else kh
    image_kw = KW - 1 - kw if FILTER_REVERSE else kw

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + offs_ci_rel

    split_size = tl.cdiv(M, NUM_SPLITS)
    split_begin = split * split_size
    split_end = tl.minimum(split_begin + split_size, M)

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for m_start in tl.range(split_begin, split_end, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < split_end
        safe_m = tl.where(mask_m, offs_m, 0)
        loss_w = safe_m % LOSS_W
        tmp = safe_m // LOSS_W
        loss_h = tmp % LOSS_H
        n_idx = tmp // LOSS_H

        image_h = loss_h * STRIDE_H - PAD_H + image_kh * DIL_H
        image_w = loss_w * STRIDE_W - PAD_W + image_kw * DIL_W
        valid_hw = (
            (image_h >= 0)
            & (image_h < IMAGE_H)
            & (image_w >= 0)
            & (image_w < IMAGE_W)
        )
        safe_h = tl.where(valid_hw, image_h, 0)
        safe_w = tl.where(valid_hw, image_w, 0)

        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + co[:, None] * loss_stride_c
            + loss_h[None, :] * loss_stride_h
            + loss_w[None, :] * loss_stride_w,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_h[:, None] * image_stride_h
            + safe_w[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid_hw[:, None],
            other=0.0,
        )
        acc += tl.dot(
            loss,
            image,
            out_dtype=tl.float32,
            input_precision="tf32",
        )

    k_elems = KH * KW
    tl.store(
        partial_ptr
        + (
            (split * C_OUT + co[:, None]) * CIN_PER_GROUP
            + offs_ci_rel[None, :]
        )
        * k_elems
        + k,
        acc,
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_wgrad2d_3tap_split_kernel(
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
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_h: tl.constexpr,
    image_stride_w: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    kh = tl.program_id(1)
    split_group = tl.program_id(2)
    split = split_group % NUM_SPLITS
    group = split_group // NUM_SPLITS

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    image_kh = KH - 1 - kh if FILTER_REVERSE else kh
    image_kw0 = KW - 1 if FILTER_REVERSE else 0
    image_kw1 = KW - 2 if FILTER_REVERSE else 1
    image_kw2 = KW - 3 if FILTER_REVERSE else 2

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + offs_ci_rel

    split_size = tl.cdiv(M, NUM_SPLITS)
    split_begin = split * split_size
    split_end = tl.minimum(split_begin + split_size, M)

    acc0 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for m_start in tl.range(split_begin, split_end, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < split_end
        safe_m = tl.where(mask_m, offs_m, 0)
        loss_w = safe_m % LOSS_W
        tmp = safe_m // LOSS_W
        loss_h = tmp % LOSS_H
        n_idx = tmp // LOSS_H

        image_h = loss_h * STRIDE_H - PAD_H + image_kh * DIL_H
        image_w0 = loss_w * STRIDE_W - PAD_W + image_kw0 * DIL_W
        image_w1 = loss_w * STRIDE_W - PAD_W + image_kw1 * DIL_W
        image_w2 = loss_w * STRIDE_W - PAD_W + image_kw2 * DIL_W
        valid_h = (image_h >= 0) & (image_h < IMAGE_H)
        valid0 = valid_h & (image_w0 >= 0) & (image_w0 < IMAGE_W)
        valid1 = valid_h & (image_w1 >= 0) & (image_w1 < IMAGE_W)
        valid2 = valid_h & (image_w2 >= 0) & (image_w2 < IMAGE_W)
        safe_h = tl.where(valid_h, image_h, 0)
        safe_w0 = tl.where(valid0, image_w0, 0)
        safe_w1 = tl.where(valid1, image_w1, 0)
        safe_w2 = tl.where(valid2, image_w2, 0)

        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + co[:, None] * loss_stride_c
            + loss_h[None, :] * loss_stride_h
            + loss_w[None, :] * loss_stride_w,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image0 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_h[:, None] * image_stride_h
            + safe_w0[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid0[:, None],
            other=0.0,
        )
        image1 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_h[:, None] * image_stride_h
            + safe_w1[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid1[:, None],
            other=0.0,
        )
        image2 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_h[:, None] * image_stride_h
            + safe_w2[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid2[:, None],
            other=0.0,
        )
        acc0 += tl.dot(
            loss, image0, out_dtype=tl.float32, input_precision="tf32"
        )
        acc1 += tl.dot(
            loss, image1, out_dtype=tl.float32, input_precision="tf32"
        )
        acc2 += tl.dot(
            loss, image2, out_dtype=tl.float32, input_precision="tf32"
        )

    base = (
        (split * C_OUT + co[:, None]) * CIN_PER_GROUP + offs_ci_rel[None, :]
    ) * (KH * KW) + kh * KW
    mask = mask_co[:, None] & mask_ci[None, :]
    tl.store(partial_ptr + base + 0, acc0, mask=mask)
    tl.store(partial_ptr + base + 1, acc1, mask=mask)
    tl.store(partial_ptr + base + 2, acc2, mask=mask)


@triton.jit
def _conv_wgrad1d_3tap_nodiv_split_v6_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    LOSS_LEN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_l: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_l: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    KL: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    SPLITS_PER_N: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    split_group = tl.program_id(1)
    split = split_group
    group = 0
    n_idx = split // SPLITS_PER_N
    split_in_n = split - n_idx * SPLITS_PER_N
    split_size = tl.cdiv(LOSS_LEN, SPLITS_PER_N)
    l_begin = split_in_n * split_size
    l_end = tl.minimum(l_begin + split_size, LOSS_LEN)

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + offs_ci_rel

    image_k0 = KL - 1 if FILTER_REVERSE else 0
    image_k1 = KL - 2 if FILTER_REVERSE else 1
    image_k2 = KL - 3 if FILTER_REVERSE else 2

    acc0 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for l_start in tl.range(l_begin, l_end, BLOCK_M):
        loss_l = l_start + tl.arange(0, BLOCK_M)
        mask_m = loss_l < l_end
        image_l0 = loss_l - PAD_LEFT + image_k0
        image_l1 = loss_l - PAD_LEFT + image_k1
        image_l2 = loss_l - PAD_LEFT + image_k2
        valid0 = (image_l0 >= 0) & (image_l0 < LOSS_LEN)
        valid1 = (image_l1 >= 0) & (image_l1 < LOSS_LEN)
        valid2 = (image_l2 >= 0) & (image_l2 < LOSS_LEN)
        safe_l0 = tl.where(valid0, image_l0, 0)
        safe_l1 = tl.where(valid1, image_l1, 0)
        safe_l2 = tl.where(valid2, image_l2, 0)

        loss = tl.load(
            loss_ptr
            + n_idx * loss_stride_n
            + co[:, None] * loss_stride_c
            + loss_l[None, :] * loss_stride_l,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image0 = tl.load(
            image_ptr
            + n_idx * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_l0[:, None] * image_stride_l,
            mask=mask_m[:, None] & mask_ci[None, :] & valid0[:, None],
            other=0.0,
        )
        image1 = tl.load(
            image_ptr
            + n_idx * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_l1[:, None] * image_stride_l,
            mask=mask_m[:, None] & mask_ci[None, :] & valid1[:, None],
            other=0.0,
        )
        image2 = tl.load(
            image_ptr
            + n_idx * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_l2[:, None] * image_stride_l,
            mask=mask_m[:, None] & mask_ci[None, :] & valid2[:, None],
            other=0.0,
        )
        acc0 += tl.dot(
            loss, image0, out_dtype=tl.float32, input_precision="tf32"
        )
        acc1 += tl.dot(
            loss, image1, out_dtype=tl.float32, input_precision="tf32"
        )
        acc2 += tl.dot(
            loss, image2, out_dtype=tl.float32, input_precision="tf32"
        )

    base = (
        (split * C_OUT + co[:, None]) * CIN_PER_GROUP + offs_ci_rel[None, :]
    ) * KL
    mask = mask_co[:, None] & mask_ci[None, :]
    tl.store(
        partial_ptr + base + 0,
        acc0.to(partial_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        partial_ptr + base + 1,
        acc1.to(partial_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        partial_ptr + base + 2,
        acc2.to(partial_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _conv_wgrad1d_s1_3tap_nsplit_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    LOSS_LEN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_l: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_l: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    KL: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    split_group = tl.program_id(1)
    n_idx = split_group % NUM_SPLITS
    group = split_group // NUM_SPLITS

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + offs_ci_rel

    image_k0 = KL - 1 if FILTER_REVERSE else 0
    image_k1 = KL - 2 if FILTER_REVERSE else 1
    image_k2 = KL - 3 if FILTER_REVERSE else 2

    acc0 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for l_start in tl.range(0, LOSS_LEN, BLOCK_M):
        loss_l = l_start + tl.arange(0, BLOCK_M)
        mask_m = loss_l < LOSS_LEN
        image_l0 = loss_l - PAD_LEFT + image_k0
        image_l1 = loss_l - PAD_LEFT + image_k1
        image_l2 = loss_l - PAD_LEFT + image_k2
        valid0 = (image_l0 >= 0) & (image_l0 < LOSS_LEN)
        valid1 = (image_l1 >= 0) & (image_l1 < LOSS_LEN)
        valid2 = (image_l2 >= 0) & (image_l2 < LOSS_LEN)
        safe_l0 = tl.where(valid0, image_l0, 0)
        safe_l1 = tl.where(valid1, image_l1, 0)
        safe_l2 = tl.where(valid2, image_l2, 0)

        loss = tl.load(
            loss_ptr
            + n_idx * loss_stride_n
            + co[:, None] * loss_stride_c
            + loss_l[None, :] * loss_stride_l,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image0 = tl.load(
            image_ptr
            + n_idx * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_l0[:, None] * image_stride_l,
            mask=mask_m[:, None] & mask_ci[None, :] & valid0[:, None],
            other=0.0,
        )
        image1 = tl.load(
            image_ptr
            + n_idx * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_l1[:, None] * image_stride_l,
            mask=mask_m[:, None] & mask_ci[None, :] & valid1[:, None],
            other=0.0,
        )
        image2 = tl.load(
            image_ptr
            + n_idx * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_l2[:, None] * image_stride_l,
            mask=mask_m[:, None] & mask_ci[None, :] & valid2[:, None],
            other=0.0,
        )
        acc0 += tl.dot(
            loss, image0, out_dtype=tl.float32, input_precision="tf32"
        )
        acc1 += tl.dot(
            loss, image1, out_dtype=tl.float32, input_precision="tf32"
        )
        acc2 += tl.dot(
            loss, image2, out_dtype=tl.float32, input_precision="tf32"
        )

    base = (
        (n_idx * C_OUT + co[:, None]) * CIN_PER_GROUP + offs_ci_rel[None, :]
    ) * KL
    mask = mask_co[:, None] & mask_ci[None, :]
    tl.store(partial_ptr + base + 0, acc0, mask=mask)
    tl.store(partial_ptr + base + 1, acc1, mask=mask)
    tl.store(partial_ptr + base + 2, acc2, mask=mask)


@triton.jit
def _conv_wgrad2d_stride2_3tap_split_kernel(
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
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_h: tl.constexpr,
    image_stride_w: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    kh = tl.program_id(1)
    split = tl.program_id(2)
    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    split_size = tl.cdiv(M, NUM_SPLITS)
    split_begin = split * split_size
    split_end = tl.minimum(split_begin + split_size, M)
    acc0 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)

    for m_start in tl.range(split_begin, split_end, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < split_end
        safe_m = tl.where(mask_m, offs_m, 0)
        loss_w = safe_m % LOSS_W
        tmp = safe_m // LOSS_W
        loss_h = tmp % LOSS_H
        n_idx = tmp // LOSS_H

        image_h = loss_h * 2 - 1 + kh
        image_w0 = loss_w * 2 - 1
        image_w1 = loss_w * 2
        image_w2 = loss_w * 2 + 1
        valid_h = (image_h >= 0) & (image_h < IMAGE_H)
        valid0 = valid_h & (image_w0 >= 0) & (image_w0 < IMAGE_W)
        valid1 = valid_h & (image_w1 >= 0) & (image_w1 < IMAGE_W)
        valid2 = valid_h & (image_w2 >= 0) & (image_w2 < IMAGE_W)
        safe_h = tl.where(valid_h, image_h, 0)
        safe_w0 = tl.where(valid0, image_w0, 0)
        safe_w1 = tl.where(valid1, image_w1, 0)
        safe_w2 = tl.where(valid2, image_w2, 0)
        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + offs_co_rel[:, None] * loss_stride_c
            + loss_h[None, :] * loss_stride_h
            + loss_w[None, :] * loss_stride_w,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image0 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + offs_ci_rel[None, :] * image_stride_c
            + safe_h[:, None] * image_stride_h
            + safe_w0[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid0[:, None],
            other=0.0,
        )
        image1 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + offs_ci_rel[None, :] * image_stride_c
            + safe_h[:, None] * image_stride_h
            + safe_w1[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid1[:, None],
            other=0.0,
        )
        image2 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + offs_ci_rel[None, :] * image_stride_c
            + safe_h[:, None] * image_stride_h
            + safe_w2[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid2[:, None],
            other=0.0,
        )
        acc0 += tl.dot(
            loss, image0, out_dtype=tl.float32, input_precision="tf32"
        )
        acc1 += tl.dot(
            loss, image1, out_dtype=tl.float32, input_precision="tf32"
        )
        acc2 += tl.dot(
            loss, image2, out_dtype=tl.float32, input_precision="tf32"
        )

    base = (
        (split * C_OUT + offs_co_rel[:, None]) * CIN_PER_GROUP
        + offs_ci_rel[None, :]
    ) * 9 + kh * 3
    mask = mask_co[:, None] & mask_ci[None, :]
    tl.store(partial_ptr + base + 0, acc0, mask=mask)
    tl.store(partial_ptr + base + 1, acc1, mask=mask)
    tl.store(partial_ptr + base + 2, acc2, mask=mask)


@triton.jit
def _conv_wgrad2d_stride2_row4_split_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_h: tl.constexpr,
    image_stride_w: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    kh = tl.program_id(1)
    n_idx = tl.program_id(2)
    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    ci = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    offs = tl.arange(0, BLOCK_HW)
    row_off = offs // 28
    loss_w = offs - row_off * 28
    valid_base = (row_off < 4) & (loss_w < 28)
    mask_co = co < COUT_PER_GROUP
    mask_ci = ci < CIN_PER_GROUP

    acc0 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for loss_h_base in tl.static_range(0, 28, 4):
        loss_h = loss_h_base + row_off
        valid_loss = valid_base & (loss_h < 28)
        image_h = loss_h * 2 - 1 + kh
        valid_h = (image_h >= 0) & (image_h < 56) & valid_loss
        image_w0 = loss_w * 2 - 1
        image_w1 = loss_w * 2
        image_w2 = loss_w * 2 + 1
        valid0 = valid_h & (image_w0 >= 0) & (image_w0 < 56)
        valid1 = valid_h & (image_w1 >= 0) & (image_w1 < 56)
        valid2 = valid_h & (image_w2 >= 0) & (image_w2 < 56)
        safe_w0 = tl.where(valid0, image_w0, 0)
        safe_w1 = tl.where(valid1, image_w1, 0)
        safe_w2 = tl.where(valid2, image_w2, 0)

        loss = tl.load(
            loss_ptr
            + n_idx * loss_stride_n
            + co[:, None] * loss_stride_c
            + loss_h[None, :] * loss_stride_h
            + loss_w[None, :] * loss_stride_w,
            mask=mask_co[:, None] & valid_loss[None, :],
            other=0.0,
        )
        image0 = tl.load(
            image_ptr
            + n_idx * image_stride_n
            + ci[None, :] * image_stride_c
            + image_h[:, None] * image_stride_h
            + safe_w0[:, None] * image_stride_w,
            mask=mask_ci[None, :] & valid0[:, None],
            other=0.0,
        )
        image1 = tl.load(
            image_ptr
            + n_idx * image_stride_n
            + ci[None, :] * image_stride_c
            + image_h[:, None] * image_stride_h
            + safe_w1[:, None] * image_stride_w,
            mask=mask_ci[None, :] & valid1[:, None],
            other=0.0,
        )
        image2 = tl.load(
            image_ptr
            + n_idx * image_stride_n
            + ci[None, :] * image_stride_c
            + image_h[:, None] * image_stride_h
            + safe_w2[:, None] * image_stride_w,
            mask=mask_ci[None, :] & valid2[:, None],
            other=0.0,
        )
        acc0 += tl.dot(
            loss, image0, out_dtype=tl.float32, input_precision="tf32"
        )
        acc1 += tl.dot(
            loss, image1, out_dtype=tl.float32, input_precision="tf32"
        )
        acc2 += tl.dot(
            loss, image2, out_dtype=tl.float32, input_precision="tf32"
        )

    base = (
        (n_idx * C_OUT + co[:, None]) * CIN_PER_GROUP + ci[None, :]
    ) * 9 + kh * 3
    mask = mask_co[:, None] & mask_ci[None, :]
    tl.store(partial_ptr + base + 0, acc0, mask=mask)
    tl.store(partial_ptr + base + 1, acc1, mask=mask)
    tl.store(partial_ptr + base + 2, acc2, mask=mask)


@triton.jit
def _conv_wgrad2d_stride2_3tap_atomic_kernel(
    image_ptr,
    loss_ptr,
    out_ptr,
    M: tl.constexpr,
    IMAGE_H: tl.constexpr,
    IMAGE_W: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_h: tl.constexpr,
    image_stride_w: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    kh = tl.program_id(1)
    split = tl.program_id(2)
    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks
    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co < COUT_PER_GROUP
    mask_ci = offs_ci < CIN_PER_GROUP
    split_size = tl.cdiv(M, NUM_SPLITS)
    split_begin = split * split_size
    split_end = tl.minimum(split_begin + split_size, M)
    acc0 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for m_start in tl.range(split_begin, split_end, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < split_end
        safe_m = tl.where(mask_m, offs_m, 0)
        loss_w = safe_m % LOSS_W
        tmp = safe_m // LOSS_W
        loss_h = tmp % LOSS_H
        n_idx = tmp // LOSS_H
        image_h = loss_h * 2 - 1 + kh
        image_w0 = loss_w * 2 - 1
        image_w1 = loss_w * 2
        image_w2 = loss_w * 2 + 1
        valid_h = (image_h >= 0) & (image_h < IMAGE_H)
        valid0 = valid_h & (image_w0 >= 0) & (image_w0 < IMAGE_W)
        valid1 = valid_h & (image_w1 >= 0) & (image_w1 < IMAGE_W)
        valid2 = valid_h & (image_w2 >= 0) & (image_w2 < IMAGE_W)
        safe_h = tl.where(valid_h, image_h, 0)
        safe_w0 = tl.where(valid0, image_w0, 0)
        safe_w1 = tl.where(valid1, image_w1, 0)
        safe_w2 = tl.where(valid2, image_w2, 0)
        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + offs_co[:, None] * loss_stride_c
            + loss_h[None, :] * loss_stride_h
            + loss_w[None, :] * loss_stride_w,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        img0 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + offs_ci[None, :] * image_stride_c
            + safe_h[:, None] * image_stride_h
            + safe_w0[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid0[:, None],
            other=0.0,
        )
        img1 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + offs_ci[None, :] * image_stride_c
            + safe_h[:, None] * image_stride_h
            + safe_w1[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid1[:, None],
            other=0.0,
        )
        img2 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + offs_ci[None, :] * image_stride_c
            + safe_h[:, None] * image_stride_h
            + safe_w2[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid2[:, None],
            other=0.0,
        )
        acc0 += tl.dot(
            loss, img0, out_dtype=tl.float32, input_precision="tf32"
        )
        acc1 += tl.dot(
            loss, img1, out_dtype=tl.float32, input_precision="tf32"
        )
        acc2 += tl.dot(
            loss, img2, out_dtype=tl.float32, input_precision="tf32"
        )
    mask = mask_co[:, None] & mask_ci[None, :]
    base = (
        out_ptr
        + offs_co[:, None] * out_stride_o
        + offs_ci[None, :] * out_stride_i
        + kh * out_stride_h
    )
    tl.atomic_add(base + 0 * out_stride_w, acc0, sem="relaxed", mask=mask)
    tl.atomic_add(base + 1 * out_stride_w, acc1, sem="relaxed", mask=mask)
    tl.atomic_add(base + 2 * out_stride_w, acc2, sem="relaxed", mask=mask)


@triton.jit
def _conv_wgrad2d_stride2_p5_row4_tail_direct_kernel(
    image_ptr,
    loss_ptr,
    out_ptr,
    COUT_PER_GROUP: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_h: tl.constexpr,
    image_stride_w: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_K_MAIN: tl.constexpr,
    BLOCK_K_TAIL: tl.constexpr,
):
    pid = tl.program_id(0)
    kh = tl.program_id(1)
    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks
    co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    ci = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = co < COUT_PER_GROUP
    mask_ci = ci < CIN_PER_GROUP

    offs_main = tl.arange(0, BLOCK_K_MAIN)
    row_main = offs_main // 16
    loss_w_main = offs_main - row_main * 16

    offs_tail = tl.arange(0, BLOCK_K_TAIL)
    row_tail = offs_tail // 4
    loss_w_tail = 16 + offs_tail - row_tail * 4

    acc0 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for loss_h_base in tl.static_range(0, 20, 4):
        loss_h_main = loss_h_base + row_main
        image_h_main = loss_h_main * 2 - 1 + kh
        valid_h_main = (image_h_main >= 0) & (image_h_main < 40)
        image_w0_main = loss_w_main * 2 - 1
        image_w1_main = loss_w_main * 2
        image_w2_main = loss_w_main * 2 + 1
        valid0_main = (
            valid_h_main & (image_w0_main >= 0) & (image_w0_main < 40)
        )
        valid1_main = (
            valid_h_main & (image_w1_main >= 0) & (image_w1_main < 40)
        )
        valid2_main = (
            valid_h_main & (image_w2_main >= 0) & (image_w2_main < 40)
        )
        safe_w0_main = tl.where(valid0_main, image_w0_main, 0)
        safe_w1_main = tl.where(valid1_main, image_w1_main, 0)
        safe_w2_main = tl.where(valid2_main, image_w2_main, 0)
        loss_main = tl.load(
            loss_ptr
            + co[:, None] * loss_stride_c
            + loss_h_main[None, :] * loss_stride_h
            + loss_w_main[None, :] * loss_stride_w,
            mask=mask_co[:, None],
            other=0.0,
        )
        img0_main = tl.load(
            image_ptr
            + ci[None, :] * image_stride_c
            + image_h_main[:, None] * image_stride_h
            + safe_w0_main[:, None] * image_stride_w,
            mask=mask_ci[None, :] & valid0_main[:, None],
            other=0.0,
        )
        img1_main = tl.load(
            image_ptr
            + ci[None, :] * image_stride_c
            + image_h_main[:, None] * image_stride_h
            + safe_w1_main[:, None] * image_stride_w,
            mask=mask_ci[None, :] & valid1_main[:, None],
            other=0.0,
        )
        img2_main = tl.load(
            image_ptr
            + ci[None, :] * image_stride_c
            + image_h_main[:, None] * image_stride_h
            + safe_w2_main[:, None] * image_stride_w,
            mask=mask_ci[None, :] & valid2_main[:, None],
            other=0.0,
        )
        acc0 += tl.dot(
            loss_main, img0_main, out_dtype=tl.float32, input_precision="tf32"
        )
        acc1 += tl.dot(
            loss_main, img1_main, out_dtype=tl.float32, input_precision="tf32"
        )
        acc2 += tl.dot(
            loss_main, img2_main, out_dtype=tl.float32, input_precision="tf32"
        )

        loss_h_tail = loss_h_base + row_tail
        image_h_tail = loss_h_tail * 2 - 1 + kh
        valid_h_tail = (image_h_tail >= 0) & (image_h_tail < 40)
        image_w0_tail = loss_w_tail * 2 - 1
        image_w1_tail = loss_w_tail * 2
        image_w2_tail = loss_w_tail * 2 + 1
        valid0_tail = (
            valid_h_tail & (image_w0_tail >= 0) & (image_w0_tail < 40)
        )
        valid1_tail = (
            valid_h_tail & (image_w1_tail >= 0) & (image_w1_tail < 40)
        )
        valid2_tail = (
            valid_h_tail & (image_w2_tail >= 0) & (image_w2_tail < 40)
        )
        loss_tail = tl.load(
            loss_ptr
            + co[:, None] * loss_stride_c
            + loss_h_tail[None, :] * loss_stride_h
            + loss_w_tail[None, :] * loss_stride_w,
            mask=mask_co[:, None],
            other=0.0,
        )
        img0_tail = tl.load(
            image_ptr
            + ci[None, :] * image_stride_c
            + image_h_tail[:, None] * image_stride_h
            + image_w0_tail[:, None] * image_stride_w,
            mask=mask_ci[None, :] & valid0_tail[:, None],
            other=0.0,
        )
        img1_tail = tl.load(
            image_ptr
            + ci[None, :] * image_stride_c
            + image_h_tail[:, None] * image_stride_h
            + image_w1_tail[:, None] * image_stride_w,
            mask=mask_ci[None, :] & valid1_tail[:, None],
            other=0.0,
        )
        img2_tail = tl.load(
            image_ptr
            + ci[None, :] * image_stride_c
            + image_h_tail[:, None] * image_stride_h
            + image_w2_tail[:, None] * image_stride_w,
            mask=mask_ci[None, :] & valid2_tail[:, None],
            other=0.0,
        )
        acc0 += tl.dot(
            loss_tail, img0_tail, out_dtype=tl.float32, input_precision="tf32"
        )
        acc1 += tl.dot(
            loss_tail, img1_tail, out_dtype=tl.float32, input_precision="tf32"
        )
        acc2 += tl.dot(
            loss_tail, img2_tail, out_dtype=tl.float32, input_precision="tf32"
        )

    mask = mask_co[:, None] & mask_ci[None, :]
    base = (
        out_ptr
        + co[:, None] * out_stride_o
        + ci[None, :] * out_stride_i
        + kh * out_stride_h
    )
    tl.store(
        base + 0 * out_stride_w,
        acc0.to(out_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        base + 1 * out_stride_w,
        acc1.to(out_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        base + 2 * out_stride_w,
        acc2.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _conv_wgrad2d_stride2_p5_pack_image_kernel(
    image_ptr,
    packed_ptr,
    CIN_PER_GROUP: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_h: tl.constexpr,
    image_stride_w: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    cik = CIN_PER_GROUP * 9

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < 400
    mask_n = offs_n < cik

    loss_h = offs_m // 20
    loss_w = offs_m - loss_h * 20
    ci = offs_n // 9
    kpos = offs_n - ci * 9
    kh = kpos // 3
    kw = kpos - kh * 3
    safe_ci = tl.where(mask_n, ci, 0)

    image_h = loss_h[:, None] * 2 - 1 + kh[None, :]
    image_w = loss_w[:, None] * 2 - 1 + kw[None, :]
    valid = (
        mask_m[:, None]
        & mask_n[None, :]
        & (image_h >= 0)
        & (image_h < 40)
        & (image_w >= 0)
        & (image_w < 40)
    )
    safe_h = tl.where(valid, image_h, 0)
    safe_w = tl.where(valid, image_w, 0)
    values = tl.load(
        image_ptr
        + safe_ci[None, :] * image_stride_c
        + safe_h * image_stride_h
        + safe_w * image_stride_w,
        mask=valid,
        other=0.0,
    )
    tl.store(
        packed_ptr + offs_m[:, None] * cik + offs_n[None, :],
        values,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _conv_wgrad2d_stride2_p5_flat_ptr_mm_tf32_kernel(
    loss_ptr,
    packed_ptr,
    out_ptr,
    M,
    N,
    K,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid - group_id * num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    loss_ptrs = loss_ptr + offs_m[:, None] * K + offs_k[None, :]
    packed_ptrs = packed_ptr + offs_k[:, None] * N + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in tl.range(0, K, BLOCK_K):
        k_offsets = k_start + offs_k
        loss = tl.load(
            loss_ptrs,
            mask=(offs_m[:, None] < M) & (k_offsets[None, :] < K),
            other=0.0,
        )
        packed = tl.load(
            packed_ptrs,
            mask=(k_offsets[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(loss, packed, input_precision="tf32")
        loss_ptrs += BLOCK_K
        packed_ptrs += BLOCK_K * N

    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _conv_wgrad2d_reduce_kernel(
    partial_ptr,
    out_ptr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    pid = tl.program_id(0)
    k = tl.program_id(1)
    group = tl.program_id(2)

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    kh = k // KW
    kw = k - kh * KW
    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel
    k_elems = KH * KW

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for split in tl.static_range(0, NUM_SPLITS):
        acc += tl.load(
            partial_ptr
            + (
                (split * C_OUT + co[:, None]) * CIN_PER_GROUP
                + offs_ci_rel[None, :]
            )
            * k_elems
            + k,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )

    tl.store(
        out_ptr
        + co[:, None] * out_stride_o
        + offs_ci_rel[None, :] * out_stride_i
        + kh * out_stride_h
        + kw * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_wgrad2d_col_split_kernel(
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
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_h: tl.constexpr,
    image_stride_w: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    split_group = tl.program_id(1)
    split = split_group % NUM_SPLITS
    group = split_group // NUM_SPLITS

    k_elems = KH * KW
    cik = CIN_PER_GROUP * k_elems
    num_n_blocks = tl.cdiv(cik, BLOCK_N)
    pid_co = pid // num_n_blocks
    pid_n = pid - pid_co * num_n_blocks

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_ci_rel = offs_n // k_elems
    rem = offs_n - offs_ci_rel * k_elems
    kh = rem // KW
    kw = rem - kh * KW
    image_kh = KH - 1 - kh if FILTER_REVERSE else kh
    image_kw = KW - 1 - kw if FILTER_REVERSE else kw

    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_n = offs_n < cik
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + offs_ci_rel

    split_size = tl.cdiv(M, NUM_SPLITS)
    split_begin = split * split_size
    split_end = tl.minimum(split_begin + split_size, M)

    acc = tl.zeros((BLOCK_CO, BLOCK_N), dtype=tl.float32)
    for m_start in tl.range(split_begin, split_end, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < split_end
        safe_m = tl.where(mask_m, offs_m, 0)
        loss_w = safe_m % LOSS_W
        tmp = safe_m // LOSS_W
        loss_h = tmp % LOSS_H
        n_idx = tmp // LOSS_H

        image_h = (
            loss_h[:, None] * STRIDE_H - PAD_H + image_kh[None, :] * DIL_H
        )
        image_w = (
            loss_w[:, None] * STRIDE_W - PAD_W + image_kw[None, :] * DIL_W
        )
        valid_hw = (
            (image_h >= 0)
            & (image_h < IMAGE_H)
            & (image_w >= 0)
            & (image_w < IMAGE_W)
        )
        safe_h = tl.where(valid_hw, image_h, 0)
        safe_w = tl.where(valid_hw, image_w, 0)

        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + co[:, None] * loss_stride_c
            + loss_h[None, :] * loss_stride_h
            + loss_w[None, :] * loss_stride_w,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_h * image_stride_h
            + safe_w * image_stride_w,
            mask=mask_m[:, None] & mask_n[None, :] & valid_hw,
            other=0.0,
        )
        acc += tl.dot(
            loss, image, out_dtype=tl.float32, input_precision="tf32"
        )

    tl.store(
        partial_ptr + (split * C_OUT + co[:, None]) * cik + offs_n[None, :],
        acc,
        mask=mask_co[:, None] & mask_n[None, :],
    )


@triton.jit
def _conv_wgrad2d_col_direct_strided_v8_kernel(
    image_ptr,
    loss_ptr,
    out_ptr,
    M: tl.constexpr,
    IMAGE_H: tl.constexpr,
    IMAGE_W: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_h: tl.constexpr,
    image_stride_w: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)
    k_elems = KH * KW
    cik = CIN_PER_GROUP * k_elems
    num_n_blocks = tl.cdiv(cik, BLOCK_N)
    pid_co = pid // num_n_blocks
    pid_n = pid - pid_co * num_n_blocks

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_ci_rel = offs_n // k_elems
    rem = offs_n - offs_ci_rel * k_elems
    kh = rem // KW
    kw = rem - kh * KW
    image_kh = KH - 1 - kh if FILTER_REVERSE else kh
    image_kw = KW - 1 - kw if FILTER_REVERSE else kw

    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_n = offs_n < cik
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + offs_ci_rel

    acc = tl.zeros((BLOCK_CO, BLOCK_N), dtype=tl.float32)
    for m_start in tl.range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        safe_m = tl.where(mask_m, offs_m, 0)
        loss_w = safe_m % LOSS_W
        tmp = safe_m // LOSS_W
        loss_h = tmp % LOSS_H
        n_idx = tmp // LOSS_H
        image_h = (
            loss_h[:, None] * STRIDE_H - PAD_H + image_kh[None, :] * DIL_H
        )
        image_w = (
            loss_w[:, None] * STRIDE_W - PAD_W + image_kw[None, :] * DIL_W
        )
        valid_hw = (
            (image_h >= 0)
            & (image_h < IMAGE_H)
            & (image_w >= 0)
            & (image_w < IMAGE_W)
        )
        safe_h = tl.where(valid_hw, image_h, 0)
        safe_w = tl.where(valid_hw, image_w, 0)
        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + co[:, None] * loss_stride_c
            + loss_h[None, :] * loss_stride_h
            + loss_w[None, :] * loss_stride_w,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_h * image_stride_h
            + safe_w * image_stride_w,
            mask=mask_m[:, None] & mask_n[None, :] & valid_hw,
            other=0.0,
        )
        acc += tl.dot(
            loss, image, out_dtype=tl.float32, input_precision="tf32"
        )

    tl.store(
        out_ptr
        + co[:, None] * out_stride_o
        + offs_ci_rel[None, :] * out_stride_i
        + kh[None, :] * out_stride_h
        + kw[None, :] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_co[:, None] & mask_n[None, :],
    )


@triton.jit
def _conv_wgrad2d_col_reduce_kernel(
    partial_ptr,
    out_ptr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)

    k_elems = KH * KW
    cik = CIN_PER_GROUP * k_elems
    num_n_blocks = tl.cdiv(cik, BLOCK_N)
    pid_co = pid // num_n_blocks
    pid_n = pid - pid_co * num_n_blocks

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_ci_rel = offs_n // k_elems
    rem = offs_n - offs_ci_rel * k_elems
    kh = rem // KW
    kw = rem - kh * KW

    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_n = offs_n < cik
    co = group * COUT_PER_GROUP + offs_co_rel

    acc = tl.zeros((BLOCK_CO, BLOCK_N), dtype=tl.float32)
    for split in tl.static_range(0, NUM_SPLITS):
        acc += tl.load(
            partial_ptr
            + (split * C_OUT + co[:, None]) * cik
            + offs_n[None, :],
            mask=mask_co[:, None] & mask_n[None, :],
            other=0.0,
        )

    tl.store(
        out_ptr
        + co[:, None] * out_stride_o
        + offs_ci_rel[None, :] * out_stride_i
        + kh[None, :] * out_stride_h
        + kw[None, :] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_co[:, None] & mask_n[None, :],
    )


@triton.jit
def _conv_wgrad3d_kernel(
    image_ptr,
    loss_ptr,
    out_ptr,
    M: tl.constexpr,
    IMAGE_D: tl.constexpr,
    IMAGE_H: tl.constexpr,
    IMAGE_W: tl.constexpr,
    LOSS_D: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_d: tl.constexpr,
    image_stride_h: tl.constexpr,
    image_stride_w: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_d: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    out_stride_d: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    k = tl.program_id(1)
    group = tl.program_id(2)

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    kw = k % KW
    tmp_k = k // KW
    kh = tmp_k % KH
    kd = tmp_k // KH
    image_kd = KD - 1 - kd if FILTER_REVERSE else kd
    image_kh = KH - 1 - kh if FILTER_REVERSE else kh
    image_kw = KW - 1 - kw if FILTER_REVERSE else kw

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + offs_ci_rel

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for m_start in tl.range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        safe_m = tl.where(mask_m, offs_m, 0)
        loss_w = safe_m % LOSS_W
        tmp = safe_m // LOSS_W
        loss_h = tmp % LOSS_H
        tmp = tmp // LOSS_H
        loss_d = tmp % LOSS_D
        n_idx = tmp // LOSS_D

        image_d = loss_d * STRIDE_D - PAD_D + image_kd * DIL_D
        image_h = loss_h * STRIDE_H - PAD_H + image_kh * DIL_H
        image_w = loss_w * STRIDE_W - PAD_W + image_kw * DIL_W
        valid_dhw = (
            (image_d >= 0)
            & (image_d < IMAGE_D)
            & (image_h >= 0)
            & (image_h < IMAGE_H)
            & (image_w >= 0)
            & (image_w < IMAGE_W)
        )
        safe_d = tl.where(valid_dhw, image_d, 0)
        safe_h = tl.where(valid_dhw, image_h, 0)
        safe_w = tl.where(valid_dhw, image_w, 0)

        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + co[:, None] * loss_stride_c
            + loss_d[None, :] * loss_stride_d
            + loss_h[None, :] * loss_stride_h
            + loss_w[None, :] * loss_stride_w,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_d[:, None] * image_stride_d
            + safe_h[:, None] * image_stride_h
            + safe_w[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid_dhw[:, None],
            other=0.0,
        )
        acc += tl.dot(
            loss,
            image,
            out_dtype=tl.float32,
            input_precision="tf32",
        )

    tl.store(
        out_ptr
        + co[:, None] * out_stride_o
        + offs_ci_rel[None, :] * out_stride_i
        + kd * out_stride_d
        + kh * out_stride_h
        + kw * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_wgrad3d_split_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    M: tl.constexpr,
    IMAGE_D: tl.constexpr,
    IMAGE_H: tl.constexpr,
    IMAGE_W: tl.constexpr,
    LOSS_D: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_d: tl.constexpr,
    image_stride_h: tl.constexpr,
    image_stride_w: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_d: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    k = tl.program_id(1)
    split_group = tl.program_id(2)
    split = split_group % NUM_SPLITS
    group = split_group // NUM_SPLITS

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    kw = k % KW
    tmp_k = k // KW
    kh = tmp_k % KH
    kd = tmp_k // KH
    image_kd = KD - 1 - kd if FILTER_REVERSE else kd
    image_kh = KH - 1 - kh if FILTER_REVERSE else kh
    image_kw = KW - 1 - kw if FILTER_REVERSE else kw

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + offs_ci_rel

    split_size = tl.cdiv(M, NUM_SPLITS)
    split_begin = split * split_size
    split_end = tl.minimum(split_begin + split_size, M)

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for m_start in tl.range(split_begin, split_end, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < split_end
        safe_m = tl.where(mask_m, offs_m, 0)
        loss_w = safe_m % LOSS_W
        tmp = safe_m // LOSS_W
        loss_h = tmp % LOSS_H
        tmp = tmp // LOSS_H
        loss_d = tmp % LOSS_D
        n_idx = tmp // LOSS_D

        image_d = loss_d * STRIDE_D - PAD_D + image_kd * DIL_D
        image_h = loss_h * STRIDE_H - PAD_H + image_kh * DIL_H
        image_w = loss_w * STRIDE_W - PAD_W + image_kw * DIL_W
        valid_dhw = (
            (image_d >= 0)
            & (image_d < IMAGE_D)
            & (image_h >= 0)
            & (image_h < IMAGE_H)
            & (image_w >= 0)
            & (image_w < IMAGE_W)
        )
        safe_d = tl.where(valid_dhw, image_d, 0)
        safe_h = tl.where(valid_dhw, image_h, 0)
        safe_w = tl.where(valid_dhw, image_w, 0)

        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + co[:, None] * loss_stride_c
            + loss_d[None, :] * loss_stride_d
            + loss_h[None, :] * loss_stride_h
            + loss_w[None, :] * loss_stride_w,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_d[:, None] * image_stride_d
            + safe_h[:, None] * image_stride_h
            + safe_w[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid_dhw[:, None],
            other=0.0,
        )
        acc += tl.dot(
            loss,
            image,
            out_dtype=tl.float32,
            input_precision="tf32",
        )

    k_elems = KD * KH * KW
    tl.store(
        partial_ptr
        + (
            (split * C_OUT + co[:, None]) * CIN_PER_GROUP
            + offs_ci_rel[None, :]
        )
        * k_elems
        + k,
        acc,
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_wgrad3d_kw3_split_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    M: tl.constexpr,
    IMAGE_D: tl.constexpr,
    IMAGE_H: tl.constexpr,
    IMAGE_W: tl.constexpr,
    LOSS_D: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_d: tl.constexpr,
    image_stride_h: tl.constexpr,
    image_stride_w: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_d: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    plane = tl.program_id(1)
    split_group = tl.program_id(2)
    split = split_group % NUM_SPLITS
    group = split_group // NUM_SPLITS

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    kh = plane % KH
    kd = plane // KH
    image_kd = KD - 1 - kd if FILTER_REVERSE else kd
    image_kh = KH - 1 - kh if FILTER_REVERSE else kh
    image_kw0 = KW - 1 if FILTER_REVERSE else 0
    image_kw1 = KW - 2 if FILTER_REVERSE else 1
    image_kw2 = KW - 3 if FILTER_REVERSE else 2

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + offs_ci_rel

    split_size = tl.cdiv(M, NUM_SPLITS)
    split_begin = split * split_size
    split_end = tl.minimum(split_begin + split_size, M)

    acc0 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for m_start in tl.range(split_begin, split_end, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < split_end
        safe_m = tl.where(mask_m, offs_m, 0)
        loss_w = safe_m % LOSS_W
        tmp = safe_m // LOSS_W
        loss_h = tmp % LOSS_H
        tmp = tmp // LOSS_H
        loss_d = tmp % LOSS_D
        n_idx = tmp // LOSS_D

        image_d = loss_d * STRIDE_D - PAD_D + image_kd * DIL_D
        image_h = loss_h * STRIDE_H - PAD_H + image_kh * DIL_H
        image_w0 = loss_w * STRIDE_W - PAD_W + image_kw0 * DIL_W
        image_w1 = loss_w * STRIDE_W - PAD_W + image_kw1 * DIL_W
        image_w2 = loss_w * STRIDE_W - PAD_W + image_kw2 * DIL_W
        valid_dh = (
            (image_d >= 0)
            & (image_d < IMAGE_D)
            & (image_h >= 0)
            & (image_h < IMAGE_H)
        )
        valid0 = valid_dh & (image_w0 >= 0) & (image_w0 < IMAGE_W)
        valid1 = valid_dh & (image_w1 >= 0) & (image_w1 < IMAGE_W)
        valid2 = valid_dh & (image_w2 >= 0) & (image_w2 < IMAGE_W)
        safe_d = tl.where(valid_dh, image_d, 0)
        safe_h = tl.where(valid_dh, image_h, 0)
        safe_w0 = tl.where(valid0, image_w0, 0)
        safe_w1 = tl.where(valid1, image_w1, 0)
        safe_w2 = tl.where(valid2, image_w2, 0)

        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + co[:, None] * loss_stride_c
            + loss_d[None, :] * loss_stride_d
            + loss_h[None, :] * loss_stride_h
            + loss_w[None, :] * loss_stride_w,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image0 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_d[:, None] * image_stride_d
            + safe_h[:, None] * image_stride_h
            + safe_w0[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid0[:, None],
            other=0.0,
        )
        image1 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_d[:, None] * image_stride_d
            + safe_h[:, None] * image_stride_h
            + safe_w1[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid1[:, None],
            other=0.0,
        )
        image2 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_d[:, None] * image_stride_d
            + safe_h[:, None] * image_stride_h
            + safe_w2[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid2[:, None],
            other=0.0,
        )
        acc0 += tl.dot(
            loss, image0, out_dtype=tl.float32, input_precision="tf32"
        )
        acc1 += tl.dot(
            loss, image1, out_dtype=tl.float32, input_precision="tf32"
        )
        acc2 += tl.dot(
            loss, image2, out_dtype=tl.float32, input_precision="tf32"
        )

    k_elems = KD * KH * KW
    k_base = (kd * KH + kh) * KW
    base = (
        (split * C_OUT + co[:, None]) * CIN_PER_GROUP + offs_ci_rel[None, :]
    ) * k_elems + k_base
    mask = mask_co[:, None] & mask_ci[None, :]
    tl.store(partial_ptr + base + 0, acc0, mask=mask)
    tl.store(partial_ptr + base + 1, acc1, mask=mask)
    tl.store(partial_ptr + base + 2, acc2, mask=mask)


@triton.jit
def _conv_wgrad3d_reduce_kernel(
    partial_ptr,
    out_ptr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    out_stride_d: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    pid = tl.program_id(0)
    k = tl.program_id(1)
    group = tl.program_id(2)

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    kw = k % KW
    tmp_k = k // KW
    kh = tmp_k % KH
    kd = tmp_k // KH
    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel
    k_elems = KD * KH * KW

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for split in tl.static_range(0, NUM_SPLITS):
        acc += tl.load(
            partial_ptr
            + (
                (split * C_OUT + co[:, None]) * CIN_PER_GROUP
                + offs_ci_rel[None, :]
            )
            * k_elems
            + k,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )

    tl.store(
        out_ptr
        + co[:, None] * out_stride_o
        + offs_ci_rel[None, :] * out_stride_i
        + kd * out_stride_d
        + kh * out_stride_h
        + kw * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_wgrad3d_col_split_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    M: tl.constexpr,
    IMAGE_D: tl.constexpr,
    IMAGE_H: tl.constexpr,
    IMAGE_W: tl.constexpr,
    LOSS_D: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_d: tl.constexpr,
    image_stride_h: tl.constexpr,
    image_stride_w: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_d: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    split_group = tl.program_id(1)
    split = split_group % NUM_SPLITS
    group = split_group // NUM_SPLITS

    k_elems = KD * KH * KW
    cik = CIN_PER_GROUP * k_elems
    num_n_blocks = tl.cdiv(cik, BLOCK_N)
    pid_co = pid // num_n_blocks
    pid_n = pid - pid_co * num_n_blocks

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_ci_rel = offs_n // k_elems
    rem0 = offs_n - offs_ci_rel * k_elems
    kw = rem0 % KW
    tmp_k = rem0 // KW
    kh = tmp_k % KH
    kd = tmp_k // KH
    image_kd = KD - 1 - kd if FILTER_REVERSE else kd
    image_kh = KH - 1 - kh if FILTER_REVERSE else kh
    image_kw = KW - 1 - kw if FILTER_REVERSE else kw

    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_n = offs_n < cik
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + offs_ci_rel

    split_size = tl.cdiv(M, NUM_SPLITS)
    split_begin = split * split_size
    split_end = tl.minimum(split_begin + split_size, M)

    acc = tl.zeros((BLOCK_CO, BLOCK_N), dtype=tl.float32)
    for m_start in tl.range(split_begin, split_end, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < split_end
        safe_m = tl.where(mask_m, offs_m, 0)
        loss_w = safe_m % LOSS_W
        tmp = safe_m // LOSS_W
        loss_h = tmp % LOSS_H
        tmp = tmp // LOSS_H
        loss_d = tmp % LOSS_D
        n_idx = tmp // LOSS_D

        image_d = (
            loss_d[:, None] * STRIDE_D - PAD_D + image_kd[None, :] * DIL_D
        )
        image_h = (
            loss_h[:, None] * STRIDE_H - PAD_H + image_kh[None, :] * DIL_H
        )
        image_w = (
            loss_w[:, None] * STRIDE_W - PAD_W + image_kw[None, :] * DIL_W
        )
        valid_dhw = (
            (image_d >= 0)
            & (image_d < IMAGE_D)
            & (image_h >= 0)
            & (image_h < IMAGE_H)
            & (image_w >= 0)
            & (image_w < IMAGE_W)
        )
        safe_d = tl.where(valid_dhw, image_d, 0)
        safe_h = tl.where(valid_dhw, image_h, 0)
        safe_w = tl.where(valid_dhw, image_w, 0)

        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + co[:, None] * loss_stride_c
            + loss_d[None, :] * loss_stride_d
            + loss_h[None, :] * loss_stride_h
            + loss_w[None, :] * loss_stride_w,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_d * image_stride_d
            + safe_h * image_stride_h
            + safe_w * image_stride_w,
            mask=mask_m[:, None] & mask_n[None, :] & valid_dhw,
            other=0.0,
        )
        acc += tl.dot(
            loss, image, out_dtype=tl.float32, input_precision="tf32"
        )

    tl.store(
        partial_ptr + (split * C_OUT + co[:, None]) * cik + offs_n[None, :],
        acc,
        mask=mask_co[:, None] & mask_n[None, :],
    )


@triton.jit
def _conv_wgrad3d_col_direct_strided_v8_kernel(
    image_ptr,
    loss_ptr,
    out_ptr,
    M: tl.constexpr,
    IMAGE_D: tl.constexpr,
    IMAGE_H: tl.constexpr,
    IMAGE_W: tl.constexpr,
    LOSS_D: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_d: tl.constexpr,
    image_stride_h: tl.constexpr,
    image_stride_w: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_d: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    out_stride_d: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)
    k_elems = KD * KH * KW
    cik = CIN_PER_GROUP * k_elems
    num_n_blocks = tl.cdiv(cik, BLOCK_N)
    pid_co = pid // num_n_blocks
    pid_n = pid - pid_co * num_n_blocks

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_ci_rel = offs_n // k_elems
    rem0 = offs_n - offs_ci_rel * k_elems
    kw = rem0 % KW
    tmp_k = rem0 // KW
    kh = tmp_k % KH
    kd = tmp_k // KH
    image_kd = KD - 1 - kd if FILTER_REVERSE else kd
    image_kh = KH - 1 - kh if FILTER_REVERSE else kh
    image_kw = KW - 1 - kw if FILTER_REVERSE else kw

    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_n = offs_n < cik
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + offs_ci_rel

    acc = tl.zeros((BLOCK_CO, BLOCK_N), dtype=tl.float32)
    for m_start in tl.range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        safe_m = tl.where(mask_m, offs_m, 0)
        loss_w = safe_m % LOSS_W
        tmp = safe_m // LOSS_W
        loss_h = tmp % LOSS_H
        tmp = tmp // LOSS_H
        loss_d = tmp % LOSS_D
        n_idx = tmp // LOSS_D
        image_d = (
            loss_d[:, None] * STRIDE_D - PAD_D + image_kd[None, :] * DIL_D
        )
        image_h = (
            loss_h[:, None] * STRIDE_H - PAD_H + image_kh[None, :] * DIL_H
        )
        image_w = (
            loss_w[:, None] * STRIDE_W - PAD_W + image_kw[None, :] * DIL_W
        )
        valid_dhw = (
            (image_d >= 0)
            & (image_d < IMAGE_D)
            & (image_h >= 0)
            & (image_h < IMAGE_H)
            & (image_w >= 0)
            & (image_w < IMAGE_W)
        )
        safe_d = tl.where(valid_dhw, image_d, 0)
        safe_h = tl.where(valid_dhw, image_h, 0)
        safe_w = tl.where(valid_dhw, image_w, 0)
        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + co[:, None] * loss_stride_c
            + loss_d[None, :] * loss_stride_d
            + loss_h[None, :] * loss_stride_h
            + loss_w[None, :] * loss_stride_w,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_d * image_stride_d
            + safe_h * image_stride_h
            + safe_w * image_stride_w,
            mask=mask_m[:, None] & mask_n[None, :] & valid_dhw,
            other=0.0,
        )
        acc += tl.dot(
            loss, image, out_dtype=tl.float32, input_precision="tf32"
        )

    tl.store(
        out_ptr
        + co[:, None] * out_stride_o
        + offs_ci_rel[None, :] * out_stride_i
        + kd[None, :] * out_stride_d
        + kh[None, :] * out_stride_h
        + kw[None, :] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_co[:, None] & mask_n[None, :],
    )


@triton.jit
def _conv_wgrad3d_col_reduce_kernel(
    partial_ptr,
    out_ptr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    out_stride_d: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)

    k_elems = KD * KH * KW
    cik = CIN_PER_GROUP * k_elems
    num_n_blocks = tl.cdiv(cik, BLOCK_N)
    pid_co = pid // num_n_blocks
    pid_n = pid - pid_co * num_n_blocks

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_ci_rel = offs_n // k_elems
    rem0 = offs_n - offs_ci_rel * k_elems
    kw = rem0 % KW
    tmp_k = rem0 // KW
    kh = tmp_k % KH
    kd = tmp_k // KH

    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_n = offs_n < cik
    co = group * COUT_PER_GROUP + offs_co_rel

    acc = tl.zeros((BLOCK_CO, BLOCK_N), dtype=tl.float32)
    for split in tl.static_range(0, NUM_SPLITS):
        acc += tl.load(
            partial_ptr
            + (split * C_OUT + co[:, None]) * cik
            + offs_n[None, :],
            mask=mask_co[:, None] & mask_n[None, :],
            other=0.0,
        )

    tl.store(
        out_ptr
        + co[:, None] * out_stride_o
        + offs_ci_rel[None, :] * out_stride_i
        + kd[None, :] * out_stride_d
        + kh[None, :] * out_stride_h
        + kw[None, :] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_co[:, None] & mask_n[None, :],
    )


@triton.jit
def _conv_wgrad1d_valid_nsplit_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    IMAGE_LEN: tl.constexpr,
    LOSS_LEN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_l: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_l: tl.constexpr,
    STRIDE_L: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_L: tl.constexpr,
    KL: tl.constexpr,
    SPLITS_PER_N: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    k = tl.program_id(1)
    split = tl.program_id(2)

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    n_idx = split // SPLITS_PER_N
    split_in_n = split - n_idx * SPLITS_PER_N
    image_k = k

    valid_begin = (PAD_LEFT - image_k * DIL_L + STRIDE_L - 1) // STRIDE_L
    valid_begin = tl.maximum(valid_begin, 0)
    valid_end = (IMAGE_LEN - 1 + PAD_LEFT - image_k * DIL_L) // STRIDE_L + 1
    valid_end = tl.minimum(valid_end, LOSS_LEN)
    valid_len = tl.maximum(valid_end - valid_begin, 0)
    split_size = tl.cdiv(valid_len, SPLITS_PER_N)
    l_begin = valid_begin + split_in_n * split_size
    l_end = tl.minimum(l_begin + split_size, valid_end)

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for l_start in tl.range(l_begin, l_end, BLOCK_M):
        loss_l = l_start + tl.arange(0, BLOCK_M)
        mask_m = loss_l < l_end
        safe_l = tl.where(mask_m, loss_l, valid_begin)
        image_l = safe_l * STRIDE_L - PAD_LEFT + image_k * DIL_L
        loss = tl.load(
            loss_ptr
            + n_idx * loss_stride_n
            + offs_co_rel[:, None] * loss_stride_c
            + safe_l[None, :] * loss_stride_l,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx * image_stride_n
            + offs_ci_rel[None, :] * image_stride_c
            + image_l[:, None] * image_stride_l,
            mask=mask_m[:, None] & mask_ci[None, :],
            other=0.0,
        )
        acc += tl.dot(
            loss, image, out_dtype=tl.float32, input_precision="tf32"
        )

    tl.store(
        partial_ptr
        + (
            (split * C_OUT + offs_co_rel[:, None]) * CIN_PER_GROUP
            + offs_ci_rel[None, :]
        )
        * KL
        + k,
        acc,
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_wgrad2d_valid_nsplit_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    IMAGE_H: tl.constexpr,
    IMAGE_W: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_h: tl.constexpr,
    image_stride_w: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SPLITS_PER_N: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    k = tl.program_id(1)
    split = tl.program_id(2)

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    kh = k // KW
    kw = k - kh * KW
    image_kh = kh
    image_kw = kw
    n_idx = split // SPLITS_PER_N
    split_in_n = split - n_idx * SPLITS_PER_N

    h_begin = (PAD_H - image_kh * DIL_H + STRIDE_H - 1) // STRIDE_H
    h_begin = tl.maximum(h_begin, 0)
    h_end = (IMAGE_H - 1 + PAD_H - image_kh * DIL_H) // STRIDE_H + 1
    h_end = tl.minimum(h_end, LOSS_H)
    w_begin = (PAD_W - image_kw * DIL_W + STRIDE_W - 1) // STRIDE_W
    w_begin = tl.maximum(w_begin, 0)
    w_end = (IMAGE_W - 1 + PAD_W - image_kw * DIL_W) // STRIDE_W + 1
    w_end = tl.minimum(w_end, LOSS_W)
    valid_h = tl.maximum(h_end - h_begin, 0)
    valid_w = tl.maximum(w_end - w_begin, 0)
    valid_area = valid_h * valid_w
    split_size = tl.cdiv(valid_area, SPLITS_PER_N)
    area_begin = split_in_n * split_size
    area_end = tl.minimum(area_begin + split_size, valid_area)

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for area_start in tl.range(area_begin, area_end, BLOCK_M):
        area = area_start + tl.arange(0, BLOCK_M)
        mask_m = area < area_end
        safe_area = tl.where(mask_m, area, 0)
        rel_h = safe_area // valid_w
        rel_w = safe_area - rel_h * valid_w
        loss_h = h_begin + rel_h
        loss_w = w_begin + rel_w
        image_h = loss_h * STRIDE_H - PAD_H + image_kh * DIL_H
        image_w = loss_w * STRIDE_W - PAD_W + image_kw * DIL_W
        loss = tl.load(
            loss_ptr
            + n_idx * loss_stride_n
            + offs_co_rel[:, None] * loss_stride_c
            + loss_h[None, :] * loss_stride_h
            + loss_w[None, :] * loss_stride_w,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx * image_stride_n
            + offs_ci_rel[None, :] * image_stride_c
            + image_h[:, None] * image_stride_h
            + image_w[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :],
            other=0.0,
        )
        acc += tl.dot(
            loss, image, out_dtype=tl.float32, input_precision="tf32"
        )

    k_elems = KH * KW
    tl.store(
        partial_ptr
        + (
            (split * C_OUT + offs_co_rel[:, None]) * CIN_PER_GROUP
            + offs_ci_rel[None, :]
        )
        * k_elems
        + k,
        acc,
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_wgrad3d_valid_nsplit_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    IMAGE_D: tl.constexpr,
    IMAGE_H: tl.constexpr,
    IMAGE_W: tl.constexpr,
    LOSS_D: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_d: tl.constexpr,
    image_stride_h: tl.constexpr,
    image_stride_w: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_d: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SPLITS_PER_N: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    k = tl.program_id(1)
    split = tl.program_id(2)

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    kw = k % KW
    tmp_k = k // KW
    kh = tmp_k % KH
    kd = tmp_k // KH
    image_kd = kd
    image_kh = kh
    image_kw = kw
    n_idx = split // SPLITS_PER_N
    split_in_n = split - n_idx * SPLITS_PER_N

    d_begin = (PAD_D - image_kd * DIL_D + STRIDE_D - 1) // STRIDE_D
    d_begin = tl.maximum(d_begin, 0)
    d_end = (IMAGE_D - 1 + PAD_D - image_kd * DIL_D) // STRIDE_D + 1
    d_end = tl.minimum(d_end, LOSS_D)
    h_begin = (PAD_H - image_kh * DIL_H + STRIDE_H - 1) // STRIDE_H
    h_begin = tl.maximum(h_begin, 0)
    h_end = (IMAGE_H - 1 + PAD_H - image_kh * DIL_H) // STRIDE_H + 1
    h_end = tl.minimum(h_end, LOSS_H)
    w_begin = (PAD_W - image_kw * DIL_W + STRIDE_W - 1) // STRIDE_W
    w_begin = tl.maximum(w_begin, 0)
    w_end = (IMAGE_W - 1 + PAD_W - image_kw * DIL_W) // STRIDE_W + 1
    w_end = tl.minimum(w_end, LOSS_W)
    valid_d = tl.maximum(d_end - d_begin, 0)
    valid_h = tl.maximum(h_end - h_begin, 0)
    valid_w = tl.maximum(w_end - w_begin, 0)
    valid_hw = valid_h * valid_w
    valid_vol = valid_d * valid_hw
    split_size = tl.cdiv(valid_vol, SPLITS_PER_N)
    vol_begin = split_in_n * split_size
    vol_end = tl.minimum(vol_begin + split_size, valid_vol)

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for vol_start in tl.range(vol_begin, vol_end, BLOCK_M):
        vol = vol_start + tl.arange(0, BLOCK_M)
        mask_m = vol < vol_end
        safe_vol = tl.where(mask_m, vol, 0)
        rel_d = safe_vol // valid_hw
        rem = safe_vol - rel_d * valid_hw
        rel_h = rem // valid_w
        rel_w = rem - rel_h * valid_w
        loss_d = d_begin + rel_d
        loss_h = h_begin + rel_h
        loss_w = w_begin + rel_w
        image_d = loss_d * STRIDE_D - PAD_D + image_kd * DIL_D
        image_h = loss_h * STRIDE_H - PAD_H + image_kh * DIL_H
        image_w = loss_w * STRIDE_W - PAD_W + image_kw * DIL_W
        loss = tl.load(
            loss_ptr
            + n_idx * loss_stride_n
            + offs_co_rel[:, None] * loss_stride_c
            + loss_d[None, :] * loss_stride_d
            + loss_h[None, :] * loss_stride_h
            + loss_w[None, :] * loss_stride_w,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx * image_stride_n
            + offs_ci_rel[None, :] * image_stride_c
            + image_d[:, None] * image_stride_d
            + image_h[:, None] * image_stride_h
            + image_w[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :],
            other=0.0,
        )
        acc += tl.dot(
            loss, image, out_dtype=tl.float32, input_precision="tf32"
        )

    k_elems = KD * KH * KW
    tl.store(
        partial_ptr
        + (
            (split * C_OUT + offs_co_rel[:, None]) * CIN_PER_GROUP
            + offs_ci_rel[None, :]
        )
        * k_elems
        + k,
        acc,
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_wgrad1d_col_direct_nodiv_v5_kernel(
    image_ptr,
    loss_ptr,
    out_ptr,
    IMAGE_LEN: tl.constexpr,
    LOSS_LEN: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_l: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_l: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    out_stride_k: tl.constexpr,
    STRIDE_L: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_L: tl.constexpr,
    KL: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    BATCH_N: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)
    cik = CIN_PER_GROUP * KL
    num_n_blocks = tl.cdiv(cik, BLOCK_N)
    pid_co = pid // num_n_blocks
    pid_n = pid - pid_co * num_n_blocks
    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ci_rel = offs_n // KL
    k = offs_n - ci_rel * KL
    image_k = KL - 1 - k if FILTER_REVERSE else k
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_n = offs_n < cik
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + ci_rel
    acc = tl.zeros((BLOCK_CO, BLOCK_N), dtype=tl.float32)
    for n_idx in tl.static_range(0, BATCH_N):
        for l_start in tl.range(0, LOSS_LEN, BLOCK_M):
            loss_l = l_start + tl.arange(0, BLOCK_M)
            mask_l = loss_l < LOSS_LEN
            image_l = (
                loss_l[:, None] * STRIDE_L
                - PAD_LEFT
                + image_k[None, :] * DIL_L
            )
            valid_l = (image_l >= 0) & (image_l < IMAGE_LEN)
            safe_l = tl.where(valid_l, image_l, 0)
            loss = tl.load(
                loss_ptr
                + n_idx * loss_stride_n
                + co[:, None] * loss_stride_c
                + loss_l[None, :] * loss_stride_l,
                mask=mask_co[:, None] & mask_l[None, :],
                other=0.0,
            )
            image = tl.load(
                image_ptr
                + n_idx * image_stride_n
                + ci[None, :] * image_stride_c
                + safe_l * image_stride_l,
                mask=mask_l[:, None] & mask_n[None, :] & valid_l,
                other=0.0,
            )
            acc += tl.dot(
                loss, image, out_dtype=tl.float32, input_precision="tf32"
            )
    tl.store(
        out_ptr
        + co[:, None] * out_stride_o
        + ci_rel[None, :] * out_stride_i
        + k[None, :] * out_stride_k,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_co[:, None] & mask_n[None, :],
    )


@triton.jit
def _conv_wgrad2d_col_rowsplit_v5_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    ROWS: tl.constexpr,
    IMAGE_H: tl.constexpr,
    IMAGE_W: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_OUT: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_h: tl.constexpr,
    image_stride_w: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    NUM_ROW_SPLITS: tl.constexpr,
    GROUP_ROWS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    split_group = tl.program_id(1)
    split = split_group % NUM_ROW_SPLITS
    group = split_group // NUM_ROW_SPLITS
    k_elems = KH * KW
    cik = CIN_PER_GROUP * k_elems
    num_n_blocks = tl.cdiv(cik, BLOCK_N)
    pid_co = pid // num_n_blocks
    pid_n = pid - pid_co * num_n_blocks
    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ci_rel = offs_n // k_elems
    rem = offs_n - ci_rel * k_elems
    kh = rem // KW
    kw = rem - kh * KW
    image_kh = KH - 1 - kh if FILTER_REVERSE else kh
    image_kw = KW - 1 - kw if FILTER_REVERSE else kw
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_n = offs_n < cik
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + ci_rel
    offs_w = tl.arange(0, BLOCK_W)
    mask_w = offs_w < LOSS_W
    acc = tl.zeros((BLOCK_CO, BLOCK_N), dtype=tl.float32)
    row_base = split * GROUP_ROWS
    for rr in tl.static_range(0, GROUP_ROWS):
        row = row_base + rr
        valid_row = row < ROWS
        n_idx = row // LOSS_H
        loss_h = row - n_idx * LOSS_H
        image_h = loss_h * STRIDE_H - PAD_H + image_kh * DIL_H
        image_w = (
            offs_w[:, None] * STRIDE_W - PAD_W + image_kw[None, :] * DIL_W
        )
        valid_h = (image_h >= 0) & (image_h < IMAGE_H)
        valid_w = (image_w >= 0) & (image_w < IMAGE_W)
        valid = valid_row & mask_w[:, None] & valid_h[None, :] & valid_w
        safe_h = tl.where(valid_h, image_h, 0)
        safe_w = tl.where(valid_w, image_w, 0)
        loss = tl.load(
            loss_ptr
            + n_idx * loss_stride_n
            + co[:, None] * loss_stride_c
            + loss_h * loss_stride_h
            + offs_w[None, :] * loss_stride_w,
            mask=valid_row & mask_co[:, None] & mask_w[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_h[None, :] * image_stride_h
            + safe_w * image_stride_w,
            mask=mask_n[None, :] & valid,
            other=0.0,
        )
        acc += tl.dot(
            loss, image, out_dtype=tl.float32, input_precision="tf32"
        )
    tl.store(
        partial_ptr + (split * C_OUT + co[:, None]) * cik + offs_n[None, :],
        acc,
        mask=mask_co[:, None] & mask_n[None, :],
    )


@triton.jit
def _conv_wgrad3d_col_rowsplit_v5_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    ROWS: tl.constexpr,
    IMAGE_D: tl.constexpr,
    IMAGE_H: tl.constexpr,
    IMAGE_W: tl.constexpr,
    LOSS_D: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_OUT: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_d: tl.constexpr,
    image_stride_h: tl.constexpr,
    image_stride_w: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_d: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    NUM_ROW_SPLITS: tl.constexpr,
    GROUP_ROWS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    split_group = tl.program_id(1)
    split = split_group % NUM_ROW_SPLITS
    group = split_group // NUM_ROW_SPLITS
    k_elems = KD * KH * KW
    cik = CIN_PER_GROUP * k_elems
    num_n_blocks = tl.cdiv(cik, BLOCK_N)
    pid_co = pid // num_n_blocks
    pid_n = pid - pid_co * num_n_blocks
    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ci_rel = offs_n // k_elems
    rem0 = offs_n - ci_rel * k_elems
    kw = rem0 % KW
    tmp_k = rem0 // KW
    kh = tmp_k % KH
    kd = tmp_k // KH
    image_kd = KD - 1 - kd if FILTER_REVERSE else kd
    image_kh = KH - 1 - kh if FILTER_REVERSE else kh
    image_kw = KW - 1 - kw if FILTER_REVERSE else kw
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_n = offs_n < cik
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + ci_rel
    offs_w = tl.arange(0, BLOCK_W)
    mask_w = offs_w < LOSS_W
    acc = tl.zeros((BLOCK_CO, BLOCK_N), dtype=tl.float32)
    row_base = split * GROUP_ROWS
    for rr in tl.static_range(0, GROUP_ROWS):
        row = row_base + rr
        valid_row = row < ROWS
        loss_h = row % LOSS_H
        tmp = row // LOSS_H
        loss_d = tmp % LOSS_D
        n_idx = tmp // LOSS_D
        image_d = loss_d * STRIDE_D - PAD_D + image_kd * DIL_D
        image_h = loss_h * STRIDE_H - PAD_H + image_kh * DIL_H
        image_w = (
            offs_w[:, None] * STRIDE_W - PAD_W + image_kw[None, :] * DIL_W
        )
        valid_dh = (
            (image_d >= 0)
            & (image_d < IMAGE_D)
            & (image_h >= 0)
            & (image_h < IMAGE_H)
        )
        valid_w = (image_w >= 0) & (image_w < IMAGE_W)
        valid = valid_row & mask_w[:, None] & valid_dh[None, :] & valid_w
        safe_d = tl.where(valid_dh, image_d, 0)
        safe_h = tl.where(valid_dh, image_h, 0)
        safe_w = tl.where(valid_w, image_w, 0)
        loss = tl.load(
            loss_ptr
            + n_idx * loss_stride_n
            + co[:, None] * loss_stride_c
            + loss_d * loss_stride_d
            + loss_h * loss_stride_h
            + offs_w[None, :] * loss_stride_w,
            mask=valid_row & mask_co[:, None] & mask_w[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_d[None, :] * image_stride_d
            + safe_h[None, :] * image_stride_h
            + safe_w * image_stride_w,
            mask=mask_n[None, :] & valid,
            other=0.0,
        )
        acc += tl.dot(
            loss, image, out_dtype=tl.float32, input_precision="tf32"
        )
    tl.store(
        partial_ptr + (split * C_OUT + co[:, None]) * cik + offs_n[None, :],
        acc,
        mask=mask_co[:, None] & mask_n[None, :],
    )


@triton.jit
def _conv_wgrad3d_valid_col_direct_kernel(
    image_ptr,
    loss_ptr,
    out_ptr,
    IMAGE_D: tl.constexpr,
    IMAGE_H: tl.constexpr,
    IMAGE_W: tl.constexpr,
    LOSS_D: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_d: tl.constexpr,
    image_stride_h: tl.constexpr,
    image_stride_w: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_d: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    out_stride_d: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    co_rel = tl.program_id(0)
    pid_n = tl.program_id(1)
    group = tl.program_id(2)

    k_elems = KD * KH * KW
    cik = CIN_PER_GROUP * k_elems
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ci_rel = offs_n // k_elems
    rem = offs_n - ci_rel * k_elems
    kw = rem % KW
    tmp_k = rem // KW
    kh = tmp_k % KH
    kd = tmp_k // KH
    mask_n = offs_n < cik

    d_begin = (PAD_D - kd * DIL_D + STRIDE_D - 1) // STRIDE_D
    d_begin = tl.maximum(d_begin, 0)
    d_end = (IMAGE_D - 1 + PAD_D - kd * DIL_D) // STRIDE_D + 1
    d_end = tl.minimum(d_end, LOSS_D)
    h_begin = (PAD_H - kh * DIL_H + STRIDE_H - 1) // STRIDE_H
    h_begin = tl.maximum(h_begin, 0)
    h_end = (IMAGE_H - 1 + PAD_H - kh * DIL_H) // STRIDE_H + 1
    h_end = tl.minimum(h_end, LOSS_H)
    w_begin = (PAD_W - kw * DIL_W + STRIDE_W - 1) // STRIDE_W
    w_begin = tl.maximum(w_begin, 0)
    w_end = (IMAGE_W - 1 + PAD_W - kw * DIL_W) // STRIDE_W + 1
    w_end = tl.minimum(w_end, LOSS_W)
    valid_d = tl.maximum(d_end - d_begin, 0)
    valid_h = tl.maximum(h_end - h_begin, 0)
    valid_w = tl.maximum(w_end - w_begin, 0)
    valid_hw = valid_h * valid_w
    valid_vol = valid_d * valid_hw

    co = group * COUT_PER_GROUP + co_rel
    ci = group * CIN_PER_GROUP + ci_rel
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    max_vol = LOSS_D * LOSS_H * LOSS_W
    for vol_start in tl.range(0, max_vol, BLOCK_M):
        vol = vol_start + tl.arange(0, BLOCK_M)
        mask_m = vol < max_vol
        valid_mn = (
            mask_m[:, None]
            & mask_n[None, :]
            & (vol[:, None] < valid_vol[None, :])
        )
        safe_vol = tl.where(valid_mn, vol[:, None], 0)
        rel_d = safe_vol // valid_hw[None, :]
        rem_vol = safe_vol - rel_d * valid_hw[None, :]
        rel_h = rem_vol // valid_w[None, :]
        rel_w = rem_vol - rel_h * valid_w[None, :]
        loss_d = d_begin[None, :] + rel_d
        loss_h = h_begin[None, :] + rel_h
        loss_w = w_begin[None, :] + rel_w
        image_d = loss_d * STRIDE_D - PAD_D + kd[None, :] * DIL_D
        image_h = loss_h * STRIDE_H - PAD_H + kh[None, :] * DIL_H
        image_w = loss_w * STRIDE_W - PAD_W + kw[None, :] * DIL_W

        loss = tl.load(
            loss_ptr
            + co * loss_stride_c
            + loss_d * loss_stride_d
            + loss_h * loss_stride_h
            + loss_w * loss_stride_w,
            mask=valid_mn,
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + ci[None, :] * image_stride_c
            + image_d * image_stride_d
            + image_h * image_stride_h
            + image_w * image_stride_w,
            mask=valid_mn,
            other=0.0,
        )
        acc += tl.sum(loss.to(tl.float32) * image.to(tl.float32), axis=0)

    tl.store(
        out_ptr
        + co * out_stride_o
        + ci_rel * out_stride_i
        + kd * out_stride_d
        + kh * out_stride_h
        + kw * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_n,
    )


@triton.jit
def _conv_wgrad1d_reduce3_kernel(
    partial_ptr,
    out_ptr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    out_stride_k: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)
    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks
    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel
    mask = mask_co[:, None] & mask_ci[None, :]
    acc0 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for split in tl.static_range(0, NUM_SPLITS):
        base = (
            (split * C_OUT + co[:, None]) * CIN_PER_GROUP
            + offs_ci_rel[None, :]
        ) * 3
        acc0 += tl.load(partial_ptr + base + 0, mask=mask, other=0.0).to(
            tl.float32
        )
        acc1 += tl.load(partial_ptr + base + 1, mask=mask, other=0.0).to(
            tl.float32
        )
        acc2 += tl.load(partial_ptr + base + 2, mask=mask, other=0.0).to(
            tl.float32
        )
    out_base = (
        out_ptr
        + co[:, None] * out_stride_o
        + offs_ci_rel[None, :] * out_stride_i
    )
    tl.store(
        out_base + 0 * out_stride_k,
        acc0.to(out_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        out_base + 1 * out_stride_k,
        acc1.to(out_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        out_base + 2 * out_stride_k,
        acc2.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _conv_wgrad2d_reduce3tap_kernel(
    partial_ptr,
    out_ptr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    pid = tl.program_id(0)
    kh = tl.program_id(1)
    group = tl.program_id(2)
    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks
    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel
    mask = mask_co[:, None] & mask_ci[None, :]
    acc0 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    k_base = kh * 3
    for split in tl.static_range(0, NUM_SPLITS):
        base = (
            (split * C_OUT + co[:, None]) * CIN_PER_GROUP
            + offs_ci_rel[None, :]
        ) * 9 + k_base
        acc0 += tl.load(partial_ptr + base + 0, mask=mask, other=0.0).to(
            tl.float32
        )
        acc1 += tl.load(partial_ptr + base + 1, mask=mask, other=0.0).to(
            tl.float32
        )
        acc2 += tl.load(partial_ptr + base + 2, mask=mask, other=0.0).to(
            tl.float32
        )
    out_base = (
        out_ptr
        + co[:, None] * out_stride_o
        + offs_ci_rel[None, :] * out_stride_i
        + kh * out_stride_h
    )
    tl.store(
        out_base + 0 * out_stride_w,
        acc0.to(out_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        out_base + 1 * out_stride_w,
        acc1.to(out_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        out_base + 2 * out_stride_w,
        acc2.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _conv_wgrad3d_kw3_atomic_kernel(
    image_ptr,
    loss_ptr,
    out_ptr,
    M: tl.constexpr,
    IMAGE_D: tl.constexpr,
    IMAGE_H: tl.constexpr,
    IMAGE_W: tl.constexpr,
    LOSS_D: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_d: tl.constexpr,
    image_stride_h: tl.constexpr,
    image_stride_w: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_d: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    out_stride_d: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    plane = tl.program_id(1)
    split = tl.program_id(2)
    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks
    kd = plane // KH
    kh = plane - kd * KH
    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co < COUT_PER_GROUP
    mask_ci = offs_ci < CIN_PER_GROUP
    split_size = tl.cdiv(M, NUM_SPLITS)
    split_begin = split * split_size
    split_end = tl.minimum(split_begin + split_size, M)
    acc0 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for m_start in tl.range(split_begin, split_end, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < split_end
        safe_m = tl.where(mask_m, offs_m, 0)
        loss_w = safe_m % LOSS_W
        tmp = safe_m // LOSS_W
        loss_h = tmp % LOSS_H
        tmp = tmp // LOSS_H
        loss_d = tmp % LOSS_D
        n_idx = tmp // LOSS_D
        image_d = loss_d * STRIDE_D - PAD_D + kd * DIL_D
        image_h = loss_h * STRIDE_H - PAD_H + kh * DIL_H
        image_w0 = loss_w * STRIDE_W - PAD_W + 0 * DIL_W
        image_w1 = loss_w * STRIDE_W - PAD_W + 1 * DIL_W
        image_w2 = loss_w * STRIDE_W - PAD_W + 2 * DIL_W
        valid_dh = (
            (image_d >= 0)
            & (image_d < IMAGE_D)
            & (image_h >= 0)
            & (image_h < IMAGE_H)
        )
        valid0 = valid_dh & (image_w0 >= 0) & (image_w0 < IMAGE_W)
        valid1 = valid_dh & (image_w1 >= 0) & (image_w1 < IMAGE_W)
        valid2 = valid_dh & (image_w2 >= 0) & (image_w2 < IMAGE_W)
        safe_d = tl.where(valid_dh, image_d, 0)
        safe_h = tl.where(valid_dh, image_h, 0)
        safe_w0 = tl.where(valid0, image_w0, 0)
        safe_w1 = tl.where(valid1, image_w1, 0)
        safe_w2 = tl.where(valid2, image_w2, 0)
        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + offs_co[:, None] * loss_stride_c
            + loss_d[None, :] * loss_stride_d
            + loss_h[None, :] * loss_stride_h
            + loss_w[None, :] * loss_stride_w,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        img0 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + offs_ci[None, :] * image_stride_c
            + safe_d[:, None] * image_stride_d
            + safe_h[:, None] * image_stride_h
            + safe_w0[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid0[:, None],
            other=0.0,
        )
        img1 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + offs_ci[None, :] * image_stride_c
            + safe_d[:, None] * image_stride_d
            + safe_h[:, None] * image_stride_h
            + safe_w1[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid1[:, None],
            other=0.0,
        )
        img2 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + offs_ci[None, :] * image_stride_c
            + safe_d[:, None] * image_stride_d
            + safe_h[:, None] * image_stride_h
            + safe_w2[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid2[:, None],
            other=0.0,
        )
        acc0 += tl.dot(
            loss, img0, out_dtype=tl.float32, input_precision="tf32"
        )
        acc1 += tl.dot(
            loss, img1, out_dtype=tl.float32, input_precision="tf32"
        )
        acc2 += tl.dot(
            loss, img2, out_dtype=tl.float32, input_precision="tf32"
        )
    mask = mask_co[:, None] & mask_ci[None, :]
    base = (
        out_ptr
        + offs_co[:, None] * out_stride_o
        + offs_ci[None, :] * out_stride_i
        + kd * out_stride_d
        + kh * out_stride_h
    )
    tl.atomic_add(base + 0 * out_stride_w, acc0, sem="relaxed", mask=mask)
    tl.atomic_add(base + 1 * out_stride_w, acc1, sem="relaxed", mask=mask)
    tl.atomic_add(base + 2 * out_stride_w, acc2, sem="relaxed", mask=mask)
