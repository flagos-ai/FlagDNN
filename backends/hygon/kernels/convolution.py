# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Hygon convolution kernels with strict-IEEE optimized NCHW paths.

The public compiler ABI uses logical dimensions and explicit tensor strides,
so these kernels do not depend on Torch layouts, packing caches, or Python
dispatch helpers. The first five entry points preserve the common-provider ABI
as a correctness fallback. The ``hygon_*`` entry points are backend-owned and
selected only after the compiler proves their layout and shape preconditions.
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
def hygon_conv2d_im2col_nchw_kernel(
    x_ptr,
    col_ptr,
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
    """Pack one NCHW image into [N, C*KH*KW, OH*OW]."""
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
        output_h[None, :] * STRIDE_H - PAD_TOP + kernel_h[:, None] * DIL_H
    )
    input_w = (
        output_w[None, :] * STRIDE_W - PAD_LEFT + kernel_w[:, None] * DIL_W
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
        col_ptr
        + batch * COL_STRIDE_N
        + packed_k[:, None] * COL_STRIDE_K
        + output_m[None, :] * COL_STRIDE_M,
        values,
        mask=(packed_k[:, None] < reduction_extent) & (output_m[None, :] < M),
    )


@triton.jit
def hygon_conv2d_fprop_stride2_low_ci_nchw_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Strict-IEEE direct 3x3 stride-2 FProp for low-channel NCHW stems."""
    tile = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    tiles_w: tl.constexpr = tl.cdiv(OW, BLOCK_M)
    tiles_oc: tl.constexpr = tl.cdiv(C_OUT, BLOCK_OC)
    spatial_tile = tile // tiles_oc
    tile_oc = tile - spatial_tile * tiles_oc
    output_h = spatial_tile // tiles_w
    tile_w = spatial_tile - output_h * tiles_w
    output_w = tile_w * BLOCK_M + tl.arange(0, BLOCK_M)
    output_channels = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    reduction_base = tl.arange(0, BLOCK_K)
    reduction_extent: tl.constexpr = C_IN * 3 * 3
    accumulator = tl.zeros((BLOCK_M, BLOCK_OC), dtype=tl.float32)
    for start in range(0, reduction_extent, BLOCK_K):
        reduction = start + reduction_base
        input_channel = reduction // 9
        kernel_hw = reduction - input_channel * 9
        kernel_h = kernel_hw // 3
        kernel_w = kernel_hw - kernel_h * 3
        input_h = output_h * 2 - 1 + kernel_h
        input_w = output_w[:, None] * 2 - 1 + kernel_w[None, :]
        valid_reduction = reduction < reduction_extent

        inputs = tl.load(
            x_ptr
            + batch * (C_IN * XH * XW)
            + input_channel[None, :] * (XH * XW)
            + input_h[None, :] * XW
            + input_w,
            mask=(output_h < OH)
            & (output_w[:, None] < OW)
            & valid_reduction[None, :]
            & (input_h[None, :] >= 0)
            & (input_h[None, :] < XH)
            & (input_w >= 0)
            & (input_w < XW),
            other=0.0,
        )
        weights = tl.load(
            w_ptr
            + output_channels[:, None] * reduction_extent
            + reduction[None, :],
            mask=(output_channels[:, None] < C_OUT) & valid_reduction[None, :],
            other=0.0,
        )
        accumulator += tl.dot(
            inputs, tl.trans(weights), input_precision="ieee"
        )
    output_m = output_h * OW + output_w
    tl.store(
        y_ptr
        + batch * (C_OUT * OH * OW)
        + output_channels[None, :] * (OH * OW)
        + output_m[:, None],
        accumulator.to(y_ptr.dtype.element_ty),
        mask=(output_h < OH)
        & (output_w[:, None] < OW)
        & (output_channels[None, :] < C_OUT),
    )


@triton.jit
def hygon_conv2d_fprop_standard_3x3_nchw_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    BLOCK_OC_S: tl.constexpr,
    BLOCK_K_S: tl.constexpr,
):
    """Strict-shape FP32 NCHW direct FProp for the standard 3x3 case."""
    tile = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    # The planner route proves these exact extents.  The *_S tuning names
    # isolate this entry point without introducing a shape-changing marker.
    c_in: tl.constexpr = 32
    c_out: tl.constexpr = 64
    image_extent: tl.constexpr = 32 * 32
    reduction_extent: tl.constexpr = c_in * 3 * 3
    tiles_oc: tl.constexpr = tl.cdiv(c_out, BLOCK_OC_S)
    output_h = tile // tiles_oc
    tile_oc = tile - output_h * tiles_oc
    output_channels = tile_oc * BLOCK_OC_S + tl.arange(0, BLOCK_OC_S)
    output_w = tl.arange(0, 32)
    reduction_base = tl.arange(0, BLOCK_K_S)
    accumulator = tl.zeros((BLOCK_OC_S, 32), dtype=tl.float32)

    for start in range(0, reduction_extent, BLOCK_K_S):
        reduction = start + reduction_base
        input_channel = reduction // 9
        kernel_hw = reduction - input_channel * 9
        kernel_h = kernel_hw // 3
        kernel_w = kernel_hw - kernel_h * 3
        input_h = output_h - 1 + kernel_h
        input_w = output_w[None, :] - 1 + kernel_w[:, None]
        active_reduction = reduction < reduction_extent
        weights = tl.load(
            w_ptr
            + output_channels[:, None] * reduction_extent
            + reduction[None, :],
            mask=(output_channels[:, None] < c_out)
            & active_reduction[None, :],
            other=0.0,
        )
        inputs = tl.load(
            x_ptr
            + batch * (c_in * image_extent)
            + input_channel[:, None] * image_extent
            + input_h[:, None] * 32
            + input_w,
            mask=active_reduction[:, None]
            & (input_h[:, None] >= 0)
            & (input_h[:, None] < 32)
            & (input_w >= 0)
            & (input_w < 32),
            other=0.0,
        )
        accumulator += tl.dot(weights, inputs, input_precision="ieee")

    tl.store(
        y_ptr
        + batch * (c_out * image_extent)
        + output_channels[:, None] * image_extent
        + output_h * 32
        + output_w[None, :],
        accumulator.to(y_ptr.dtype.element_ty),
        mask=output_channels[:, None] < c_out,
    )


@triton.jit
def hygon_conv2d_fprop_im2col_kernel(
    w_ptr,
    col_ptr,
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
    BLOCK_OC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Strict-IEEE FProp GEMM consuming the local im2col matrix."""
    tile = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    tiles_m = tl.cdiv(M, BLOCK_M)
    tiles_oc = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    tiles_per_group = GROUP_M * tiles_m
    group = tile // tiles_per_group
    first_oc = group * GROUP_M
    group_oc = min(tiles_oc - first_oc, GROUP_M)
    tile_oc = first_oc + (tile % group_oc)
    tile_m = (tile % tiles_per_group) // group_oc
    output_channels = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    output_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    reduction_base = tl.arange(0, BLOCK_K)
    reduction_extent: tl.constexpr = CIN_PER_GROUP * KH * KW
    accumulator = tl.zeros((BLOCK_OC, BLOCK_M), dtype=tl.float32)

    weight_ptrs = (
        w_ptr
        + output_channels[:, None] * reduction_extent
        + reduction_base[None, :]
    )
    column_ptrs = (
        col_ptr
        + batch * COL_STRIDE_N
        + reduction_base[:, None] * COL_STRIDE_K
        + output_m[None, :] * COL_STRIDE_M
    )

    for start in range(0, reduction_extent, BLOCK_K):
        remaining = reduction_extent - start
        weights = tl.load(
            weight_ptrs,
            mask=(output_channels[:, None] < COUT_PER_GROUP)
            & (reduction_base[None, :] < remaining),
            other=0.0,
        )
        columns = tl.load(
            column_ptrs,
            mask=(reduction_base[:, None] < remaining)
            & (output_m[None, :] < M),
            other=0.0,
        )
        accumulator += tl.dot(weights, columns, input_precision="ieee")
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
def hygon_conv2d_fprop_yolo_x_p5_gemm_kernel(
    w_ptr,
    col_ptr,
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
    YOLO_X_P5: tl.constexpr,
    BLOCK_OC_X: tl.constexpr,
    BLOCK_M_X: tl.constexpr,
    BLOCK_K_X: tl.constexpr,
    GROUP_M_X: tl.constexpr,
):
    """Strict-shape gfx936 GEMM for the FP32 YOLO-X P5 FProp route."""
    tile = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    tiles_m = tl.cdiv(M, BLOCK_M_X)
    tiles_oc = tl.cdiv(COUT_PER_GROUP, BLOCK_OC_X)
    tiles_per_group = GROUP_M_X * tiles_m
    group = tile // tiles_per_group
    first_oc = group * GROUP_M_X
    group_oc = min(tiles_oc - first_oc, GROUP_M_X)
    tile_oc = first_oc + (tile % group_oc)
    tile_m = (tile % tiles_per_group) // group_oc
    output_channels = tile_oc * BLOCK_OC_X + tl.arange(0, BLOCK_OC_X)
    output_m = tile_m * BLOCK_M_X + tl.arange(0, BLOCK_M_X)
    reduction_base = tl.arange(0, BLOCK_K_X)
    reduction_extent: tl.constexpr = CIN_PER_GROUP * KH * KW
    accumulator = tl.zeros((BLOCK_OC_X, BLOCK_M_X), dtype=tl.float32)

    weight_ptrs = (
        w_ptr
        + output_channels[:, None] * reduction_extent
        + reduction_base[None, :]
    )
    column_ptrs = (
        col_ptr
        + batch * COL_STRIDE_N
        + reduction_base[:, None] * COL_STRIDE_K
        + output_m[None, :] * COL_STRIDE_M
    )

    for start in range(0, reduction_extent, BLOCK_K_X):
        remaining = reduction_extent - start
        weights = tl.load(
            weight_ptrs,
            mask=(output_channels[:, None] < COUT_PER_GROUP)
            & (reduction_base[None, :] < remaining),
            other=0.0,
        )
        columns = tl.load(
            column_ptrs,
            mask=(reduction_base[:, None] < remaining)
            & (output_m[None, :] < M),
            other=0.0,
        )
        accumulator += tl.dot(weights, columns, input_precision="ieee")
        weight_ptrs += BLOCK_K_X
        column_ptrs += BLOCK_K_X * COL_STRIDE_K

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
def hygon_conv2d_fprop_1x1_nchw_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    HW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    """Division-free contiguous NCHW 1x1 FProp matrix product."""
    tile = tl.program_id(0)
    batch_group = tl.program_id(1).to(tl.int64)
    batch = batch_group // GROUPS
    group = batch_group - batch * GROUPS
    tiles_m: tl.constexpr = tl.cdiv(HW, BLOCK_M)
    tile_oc = tile // tiles_m
    tile_m = tile - tile_oc * tiles_m
    output_channels = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    spatial = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    reduction_base = tl.arange(0, BLOCK_CI)
    accumulator = tl.zeros((BLOCK_OC, BLOCK_M), dtype=tl.float32)

    for start in range(0, CIN_PER_GROUP, BLOCK_CI):
        input_channels = start + reduction_base
        global_ci = group * CIN_PER_GROUP + input_channels
        global_oc = group * COUT_PER_GROUP + output_channels
        weights = tl.load(
            w_ptr
            + global_oc[:, None] * CIN_PER_GROUP
            + input_channels[None, :],
            mask=(output_channels[:, None] < COUT_PER_GROUP)
            & (input_channels[None, :] < CIN_PER_GROUP),
            other=0.0,
        )
        inputs = tl.load(
            x_ptr
            + batch * (C_IN * HW)
            + global_ci[:, None] * HW
            + spatial[None, :],
            mask=(input_channels[:, None] < CIN_PER_GROUP)
            & (spatial[None, :] < HW),
            other=0.0,
        )
        accumulator += tl.dot(weights, inputs, input_precision="ieee")

    global_oc = group * COUT_PER_GROUP + output_channels
    tl.store(
        y_ptr
        + batch * (C_OUT * HW)
        + global_oc[:, None] * HW
        + spatial[None, :],
        accumulator.to(y_ptr.dtype.element_ty),
        mask=(output_channels[:, None] < COUT_PER_GROUP)
        & (spatial[None, :] < HW),
    )


@triton.jit
def hygon_conv_dgrad2d_1x1_nchw_kernel(
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
    """Division-free contiguous NCHW 1x1 DGrad matrix product."""
    tile = tl.program_id(0)
    batch_group = tl.program_id(1).to(tl.int64)
    batch = batch_group // GROUPS
    group = batch_group - batch * GROUPS
    tiles_m: tl.constexpr = tl.cdiv(HW, BLOCK_M)
    tile_ci = tile // tiles_m
    tile_m = tile - tile_ci * tiles_m
    spatial = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    input_channels = tile_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    accumulator = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for start in range(0, COUT_PER_GROUP, BLOCK_CO):
        output_channels = start + tl.arange(0, BLOCK_CO)
        global_oc = group * COUT_PER_GROUP + output_channels
        losses = tl.load(
            dy_ptr
            + batch * (C_OUT * HW)
            + global_oc[None, :] * HW
            + spatial[:, None],
            mask=(spatial[:, None] < HW)
            & (output_channels[None, :] < COUT_PER_GROUP),
            other=0.0,
        )
        weights = tl.load(
            w_ptr
            + global_oc[:, None] * CIN_PER_GROUP
            + input_channels[None, :],
            mask=(output_channels[:, None] < COUT_PER_GROUP)
            & (input_channels[None, :] < CIN_PER_GROUP),
            other=0.0,
        )
        accumulator += tl.dot(losses, weights, input_precision="ieee")

    global_ci = group * CIN_PER_GROUP + input_channels
    tl.store(
        dx_ptr
        + batch * (C_IN * HW)
        + global_ci[None, :] * HW
        + spatial[:, None],
        accumulator.to(dx_ptr.dtype.element_ty),
        mask=(spatial[:, None] < HW)
        & (input_channels[None, :] < CIN_PER_GROUP),
    )


@triton.jit
def hygon_conv_dgrad2d_stride1_kernel(
    dy_ptr,
    w_ptr,
    dx_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    FLIP_FILTER: tl.constexpr,
    DY_STRIDE_N: tl.constexpr,
    DY_STRIDE_C: tl.constexpr,
    DY_STRIDE_H: tl.constexpr,
    DY_STRIDE_W: tl.constexpr,
    W_STRIDE_K: tl.constexpr,
    W_STRIDE_C: tl.constexpr,
    W_STRIDE_H: tl.constexpr,
    W_STRIDE_W: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    """2D stride-one DGrad with static filter loops and IEEE dot."""
    tile = tl.program_id(0)
    group = tl.program_id(1).to(tl.int64)
    tiles_m: tl.constexpr = tl.cdiv(M, BLOCK_M)
    tile_ci = tile // tiles_m
    tile_m = tile - tile_ci * tiles_m
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    input_channels = tile_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    input_area: tl.constexpr = XH * XW
    batch = rows // input_area
    input_hw = rows - batch * input_area
    input_h = input_hw // XW
    input_w = input_hw - input_h * XW
    accumulator = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for kernel_h in tl.static_range(0, KH):
        output_h = input_h + PAD_TOP - kernel_h * DIL_H
        for kernel_w in tl.static_range(0, KW):
            output_w = input_w + PAD_LEFT - kernel_w * DIL_W
            active = (
                (rows < M)
                & (output_h >= 0)
                & (output_h < OH)
                & (output_w >= 0)
                & (output_w < OW)
            )
            weight_h = KH - 1 - kernel_h if FLIP_FILTER else kernel_h
            weight_w = KW - 1 - kernel_w if FLIP_FILTER else kernel_w
            for co_start in range(0, COUT_PER_GROUP, BLOCK_CO):
                output_channels = co_start + tl.arange(0, BLOCK_CO)
                losses = tl.load(
                    dy_ptr
                    + batch[:, None] * DY_STRIDE_N
                    + (group * COUT_PER_GROUP + output_channels[None, :])
                    * DY_STRIDE_C
                    + output_h[:, None] * DY_STRIDE_H
                    + output_w[:, None] * DY_STRIDE_W,
                    mask=active[:, None]
                    & (output_channels[None, :] < COUT_PER_GROUP),
                    other=0.0,
                )
                weights = tl.load(
                    w_ptr
                    + (group * COUT_PER_GROUP + output_channels[:, None])
                    * W_STRIDE_K
                    + input_channels[None, :] * W_STRIDE_C
                    + weight_h * W_STRIDE_H
                    + weight_w * W_STRIDE_W,
                    mask=(output_channels[:, None] < COUT_PER_GROUP)
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
        mask=(rows[:, None] < M) & (input_channels[None, :] < CIN_PER_GROUP),
    )


@triton.jit
def hygon_conv_dgrad2d_exact_3x3_s1_gemm_kernel(
    dy_ptr,
    w_ptr,
    col_ptr,
    M: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FLIP_FILTER: tl.constexpr,
    DY_STRIDE_N: tl.constexpr,
    DY_STRIDE_C: tl.constexpr,
    DY_STRIDE_H: tl.constexpr,
    DY_STRIDE_W: tl.constexpr,
    W_STRIDE_K: tl.constexpr,
    W_STRIDE_C: tl.constexpr,
    W_STRIDE_H: tl.constexpr,
    W_STRIDE_W: tl.constexpr,
    OW: tl.constexpr,
    COL_STRIDE_N: tl.constexpr,
    COL_STRIDE_M: tl.constexpr,
    COL_STRIDE_K: tl.constexpr,
    EXACT_3X3_S1: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """No-tail GEMM with FP32 accumulation for the exact fixed-shape route."""
    tile = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    packed_extent: tl.constexpr = C_IN * KH * KW
    tiles_m: tl.constexpr = M // BLOCK_M
    tiles_ci: tl.constexpr = packed_extent // BLOCK_CI
    tiles_per_group: tl.constexpr = GROUP_M * tiles_ci
    group = tile // tiles_per_group
    first_tile_m = group * GROUP_M
    group_m = tl.minimum(tiles_m - first_tile_m, GROUP_M)
    tile_in_group = tile - group * tiles_per_group
    tile_m = first_tile_m + tile_in_group % group_m
    tile_ci = tile_in_group // group_m

    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    packed_k = tile_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    accumulator = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    for co_start in tl.static_range(0, C_OUT, BLOCK_CO):
        output_channels = co_start + tl.arange(0, BLOCK_CO)
        losses = tl.load(
            dy_ptr
            + batch * DY_STRIDE_N
            + rows[:, None] * DY_STRIDE_W
            + output_channels[None, :] * DY_STRIDE_C
        )
        weights = tl.load(
            w_ptr
            + output_channels[:, None] * W_STRIDE_K
            + packed_k[None, :] * W_STRIDE_W
        )
        accumulator = tl.dot(
            losses, weights, accumulator, input_precision="ieee"
        )

    tl.store(
        col_ptr
        + batch * COL_STRIDE_N
        + rows[:, None] * COL_STRIDE_M
        + packed_k[None, :] * COL_STRIDE_K,
        accumulator,
    )


@triton.jit
def hygon_conv_dgrad2d_exact_3x3_s1_col2im_kernel(
    col_ptr,
    dx_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    COL_STRIDE_N: tl.constexpr,
    COL_STRIDE_M: tl.constexpr,
    COL_STRIDE_K: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    EXACT_3X3_S1: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Division-free FP32-workspace gather for the exact width-32 route."""
    row_block = tl.program_id(0)
    batch_channel = tl.program_id(1).to(tl.int64)
    batch = batch_channel // C_IN
    input_channel = batch_channel - batch * C_IN
    input_h = row_block * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    input_w = tl.arange(0, 32)[None, :]
    active_h = input_h < XH
    accumulator = tl.zeros((BLOCK_H, 32), dtype=tl.float32)

    for kernel_h in tl.static_range(0, KH):
        output_h = input_h + PAD_TOP - kernel_h * DIL_H
        valid_h = active_h & (output_h >= 0) & (output_h < OH)
        for kernel_w in tl.static_range(0, KW):
            output_w = input_w + PAD_LEFT - kernel_w * DIL_W
            valid = valid_h & (output_w >= 0) & (output_w < OW)
            column_m = output_h * OW + output_w
            column_k = input_channel * (KH * KW) + kernel_h * KW + kernel_w
            accumulator += tl.load(
                col_ptr
                + batch * COL_STRIDE_N
                + column_m * COL_STRIDE_M
                + column_k * COL_STRIDE_K,
                mask=valid,
                other=0.0,
            )

    tl.store(
        dx_ptr
        + batch * X_STRIDE_N
        + input_channel * X_STRIDE_C
        + input_h * X_STRIDE_H
        + input_w * X_STRIDE_W,
        accumulator.to(dx_ptr.dtype.element_ty),
        mask=active_h,
    )


@triton.jit
def hygon_conv_dgrad2d_stride2_contribution_gemm_kernel(
    dy_ptr,
    w_ptr,
    col_ptr,
    M: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FLIP_FILTER: tl.constexpr,
    DY_STRIDE_N: tl.constexpr,
    DY_STRIDE_C: tl.constexpr,
    DY_STRIDE_H: tl.constexpr,
    DY_STRIDE_W: tl.constexpr,
    W_STRIDE_K: tl.constexpr,
    W_STRIDE_C: tl.constexpr,
    W_STRIDE_H: tl.constexpr,
    W_STRIDE_W: tl.constexpr,
    OW: tl.constexpr,
    COL_STRIDE_N: tl.constexpr,
    COL_STRIDE_M: tl.constexpr,
    COL_STRIDE_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    """Form all useful group-one 3x3 contributions in one IEEE GEMM."""
    tile = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    tiles_m: tl.constexpr = tl.cdiv(M, BLOCK_M)
    tile_k = tile // tiles_m
    tile_m = tile - tile_k * tiles_m
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    packed_k = tile_k * BLOCK_CI + tl.arange(0, BLOCK_CI)
    output_channels = tl.arange(0, BLOCK_CO)
    packed_extent: tl.constexpr = C_IN * KH * KW
    accumulator = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    output_h = rows // OW
    output_w = rows - output_h * OW
    input_channel = packed_k // (KH * KW)
    kernel_linear = packed_k - input_channel * (KH * KW)
    kernel_h = kernel_linear // KW
    kernel_w = kernel_linear - kernel_h * KW
    weight_h = KH - 1 - kernel_h if FLIP_FILTER else kernel_h
    weight_w = KW - 1 - kernel_w if FLIP_FILTER else kernel_w

    for co_start in range(0, C_OUT, BLOCK_CO):
        global_oc = co_start + output_channels
        losses = tl.load(
            dy_ptr
            + batch * DY_STRIDE_N
            + global_oc[None, :] * DY_STRIDE_C
            + output_h[:, None] * DY_STRIDE_H
            + output_w[:, None] * DY_STRIDE_W,
            mask=(rows[:, None] < M) & (global_oc[None, :] < C_OUT),
            other=0.0,
        )
        weights = tl.load(
            w_ptr
            + global_oc[:, None] * W_STRIDE_K
            + input_channel[None, :] * W_STRIDE_C
            + weight_h[None, :] * W_STRIDE_H
            + weight_w[None, :] * W_STRIDE_W,
            mask=(global_oc[:, None] < C_OUT)
            & (packed_k[None, :] < packed_extent),
            other=0.0,
        )
        accumulator += tl.dot(losses, weights, input_precision="ieee")

    tl.store(
        col_ptr
        + batch * COL_STRIDE_N
        + rows[:, None] * COL_STRIDE_M
        + packed_k[None, :] * COL_STRIDE_K,
        accumulator,
        mask=(rows[:, None] < M) & (packed_k[None, :] < packed_extent),
    )


@triton.jit
def hygon_conv_dgrad2d_stride2_block_pointer_gemm_kernel(
    dy_ptr,
    w_ptr,
    col_ptr,
    M: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FLIP_FILTER: tl.constexpr,
    DY_STRIDE_N: tl.constexpr,
    DY_STRIDE_C: tl.constexpr,
    DY_STRIDE_H: tl.constexpr,
    DY_STRIDE_W: tl.constexpr,
    W_STRIDE_K: tl.constexpr,
    W_STRIDE_C: tl.constexpr,
    W_STRIDE_H: tl.constexpr,
    W_STRIDE_W: tl.constexpr,
    OW: tl.constexpr,
    COL_STRIDE_N: tl.constexpr,
    COL_STRIDE_M: tl.constexpr,
    COL_STRIDE_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Contiguous group-one contribution GEMM using block pointers."""
    tile = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    packed_extent: tl.constexpr = C_IN * KH * KW
    tiles_m = tl.cdiv(M, BLOCK_M)
    tiles_ci = tl.cdiv(packed_extent, BLOCK_CI)
    if GROUP_M == 1:
        tile_m = tile // tiles_ci
        tile_ci = tile - tile_m * tiles_ci
    elif GROUP_M >= tiles_m:
        tile_m = tile % tiles_m
        tile_ci = tile // tiles_m
    else:
        tiles_per_group = GROUP_M * tiles_ci
        group = tile // tiles_per_group
        first_tile_m = group * GROUP_M
        group_m = tl.minimum(tiles_m - first_tile_m, GROUP_M)
        tile_in_group = tile - group * tiles_per_group
        tile_m = first_tile_m + tile_in_group % group_m
        tile_ci = tile_in_group // group_m

    loss_block = tl.make_block_ptr(
        base=dy_ptr + batch * DY_STRIDE_N,
        shape=(M, C_OUT),
        strides=(DY_STRIDE_W, DY_STRIDE_C),
        offsets=(tile_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_CO),
        order=(0, 1),
    )
    weight_block = tl.make_block_ptr(
        base=w_ptr,
        shape=(C_OUT, packed_extent),
        strides=(W_STRIDE_K, W_STRIDE_W),
        offsets=(0, tile_ci * BLOCK_CI),
        block_shape=(BLOCK_CO, BLOCK_CI),
        order=(1, 0),
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    for _ in tl.range(0, C_OUT, BLOCK_CO):
        losses = tl.load(
            loss_block, boundary_check=(0, 1), padding_option="zero"
        )
        weights = tl.load(
            weight_block, boundary_check=(0, 1), padding_option="zero"
        )
        accumulator = tl.dot(
            losses, weights, accumulator, input_precision="ieee"
        )
        loss_block = tl.advance(loss_block, (0, BLOCK_CO))
        weight_block = tl.advance(weight_block, (BLOCK_CO, 0))

    output_block = tl.make_block_ptr(
        base=col_ptr + batch * COL_STRIDE_N,
        shape=(M, packed_extent),
        strides=(COL_STRIDE_M, COL_STRIDE_K),
        offsets=(tile_m * BLOCK_M, tile_ci * BLOCK_CI),
        block_shape=(BLOCK_M, BLOCK_CI),
        order=(0, 1),
    )
    tl.store(output_block, accumulator, boundary_check=(0, 1))


@triton.jit
def hygon_conv_dgrad2d_stride2_contribution_col2im_kernel(
    col_ptr,
    dx_ptr,
    N_ELEMENTS: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    COL_STRIDE_N: tl.constexpr,
    COL_STRIDE_M: tl.constexpr,
    COL_STRIDE_K: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Gather the FP32 contribution matrix into contiguous NCHW dx."""
    elements = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active_element = elements < N_ELEMENTS
    input_w = elements % XW
    remaining = elements // XW
    input_h = remaining % XH
    remaining = remaining // XH
    input_channel = remaining % C_IN
    batch = remaining // C_IN
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for kernel_h in tl.static_range(0, KH):
        numerator_h = input_h + PAD_TOP - kernel_h * DIL_H
        output_h = numerator_h // 2
        valid_h = (
            active_element
            & (numerator_h % 2 == 0)
            & (output_h >= 0)
            & (output_h < OH)
        )
        for kernel_w in tl.static_range(0, KW):
            numerator_w = input_w + PAD_LEFT - kernel_w * DIL_W
            output_w = numerator_w // 2
            valid = (
                valid_h
                & (numerator_w % 2 == 0)
                & (output_w >= 0)
                & (output_w < OW)
            )
            column_m = output_h * OW + output_w
            column_k = input_channel * (KH * KW) + kernel_h * KW + kernel_w
            accumulator += tl.load(
                col_ptr
                + batch * COL_STRIDE_N
                + column_m * COL_STRIDE_M
                + column_k * COL_STRIDE_K,
                mask=valid,
                other=0.0,
            )

    tl.store(
        dx_ptr
        + batch * X_STRIDE_N
        + input_channel * X_STRIDE_C
        + input_h * X_STRIDE_H
        + input_w * X_STRIDE_W,
        accumulator.to(dx_ptr.dtype.element_ty),
        mask=active_element,
    )


@triton.jit
def hygon_conv_dgrad2d_stride2_parity_kernel(
    dy_ptr,
    w_ptr,
    dx_ptr,
    PARITY_M: tl.constexpr,
    PARITY_W_COUNT: tl.constexpr,
    PARITY_H: tl.constexpr,
    PARITY_W: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    FLIP_FILTER: tl.constexpr,
    DY_STRIDE_N: tl.constexpr,
    DY_STRIDE_C: tl.constexpr,
    DY_STRIDE_H: tl.constexpr,
    DY_STRIDE_W: tl.constexpr,
    W_STRIDE_K: tl.constexpr,
    W_STRIDE_C: tl.constexpr,
    W_STRIDE_H: tl.constexpr,
    W_STRIDE_W: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    """Stride-2 DGrad for one input parity, avoiding three quarters of dots."""
    tile = tl.program_id(0)
    batch_group = tl.program_id(1).to(tl.int64)
    batch = batch_group // GROUPS
    group = batch_group - batch * GROUPS
    tiles_m: tl.constexpr = tl.cdiv(PARITY_M, BLOCK_M)
    tile_ci = tile // tiles_m
    tile_m = tile - tile_ci * tiles_m
    compact = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    input_h = PARITY_H + (compact // PARITY_W_COUNT) * 2
    input_w = PARITY_W + (compact % PARITY_W_COUNT) * 2
    input_channels = tile_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    accumulator = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for kernel_h in tl.static_range(0, KH):
        if (PARITY_H + PAD_TOP - kernel_h * DIL_H) % 2 == 0:
            numerator_h = input_h + PAD_TOP - kernel_h * DIL_H
            output_h = numerator_h // 2
            for kernel_w in tl.static_range(0, KW):
                if (PARITY_W + PAD_LEFT - kernel_w * DIL_W) % 2 == 0:
                    numerator_w = input_w + PAD_LEFT - kernel_w * DIL_W
                    output_w = numerator_w // 2
                    active = (
                        (compact < PARITY_M)
                        & (input_h < XH)
                        & (input_w < XW)
                        & (output_h >= 0)
                        & (output_h < OH)
                        & (output_w >= 0)
                        & (output_w < OW)
                    )
                    weight_h = KH - 1 - kernel_h if FLIP_FILTER else kernel_h
                    weight_w = KW - 1 - kernel_w if FLIP_FILTER else kernel_w
                    for co_start in range(0, COUT_PER_GROUP, BLOCK_CO):
                        output_channels = co_start + tl.arange(0, BLOCK_CO)
                        losses = tl.load(
                            dy_ptr
                            + batch * DY_STRIDE_N
                            + (
                                group * COUT_PER_GROUP
                                + output_channels[None, :]
                            )
                            * DY_STRIDE_C
                            + output_h[:, None] * DY_STRIDE_H
                            + output_w[:, None] * DY_STRIDE_W,
                            mask=active[:, None]
                            & (output_channels[None, :] < COUT_PER_GROUP),
                            other=0.0,
                        )
                        weights = tl.load(
                            w_ptr
                            + (
                                group * COUT_PER_GROUP
                                + output_channels[:, None]
                            )
                            * W_STRIDE_K
                            + input_channels[None, :] * W_STRIDE_C
                            + weight_h * W_STRIDE_H
                            + weight_w * W_STRIDE_W,
                            mask=(output_channels[:, None] < COUT_PER_GROUP)
                            & (input_channels[None, :] < CIN_PER_GROUP),
                            other=0.0,
                        )
                        accumulator += tl.dot(
                            losses, weights, input_precision="ieee"
                        )

    tl.store(
        dx_ptr
        + batch * X_STRIDE_N
        + (group * CIN_PER_GROUP + input_channels[None, :]) * X_STRIDE_C
        + input_h[:, None] * X_STRIDE_H
        + input_w[:, None] * X_STRIDE_W,
        accumulator.to(dx_ptr.dtype.element_ty),
        mask=(compact[:, None] < PARITY_M)
        & (input_h[:, None] < XH)
        & (input_w[:, None] < XW)
        & (input_channels[None, :] < CIN_PER_GROUP),
    )


@triton.jit
def hygon_conv_wgrad2d_im2col_kernel(
    dy_ptr,
    col_ptr,
    dw_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FLIP_FILTER: tl.constexpr,
    DY_STRIDE_N: tl.constexpr,
    DY_STRIDE_C: tl.constexpr,
    DY_STRIDE_H: tl.constexpr,
    DY_STRIDE_W: tl.constexpr,
    OW: tl.constexpr,
    COL_STRIDE_N: tl.constexpr,
    COL_STRIDE_K: tl.constexpr,
    COL_STRIDE_M: tl.constexpr,
    W_STRIDE_K: tl.constexpr,
    W_STRIDE_C: tl.constexpr,
    W_STRIDE_H: tl.constexpr,
    W_STRIDE_W: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_CI_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Group-one WGrad GEMM over a backend-local im2col matrix."""
    tile = tl.program_id(0)
    reduction_extent: tl.constexpr = CIN_PER_GROUP * KH * KW
    tiles_k: tl.constexpr = tl.cdiv(reduction_extent, BLOCK_CI_K)
    tile_oc = tile // tiles_k
    tile_k = tile - tile_oc * tiles_k
    output_channels = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    packed_k = tile_k * BLOCK_CI_K + tl.arange(0, BLOCK_CI_K)
    accumulator = tl.zeros((BLOCK_OC, BLOCK_CI_K), dtype=tl.float32)

    for batch in tl.static_range(0, N):
        for start in range(0, M, BLOCK_M):
            output_m = start + tl.arange(0, BLOCK_M)
            output_h = output_m // OW
            output_w = output_m - output_h * OW
            losses = tl.load(
                dy_ptr
                + batch * DY_STRIDE_N
                + output_channels[:, None] * DY_STRIDE_C
                + output_h[None, :] * DY_STRIDE_H
                + output_w[None, :] * DY_STRIDE_W,
                mask=(output_channels[:, None] < COUT_PER_GROUP)
                & (output_m[None, :] < M),
                other=0.0,
            )
            columns = tl.load(
                col_ptr
                + batch * COL_STRIDE_N
                + packed_k[:, None] * COL_STRIDE_K
                + output_m[None, :] * COL_STRIDE_M,
                mask=(packed_k[:, None] < reduction_extent)
                & (output_m[None, :] < M),
                other=0.0,
            )
            accumulator += tl.dot(
                losses, tl.trans(columns), input_precision="ieee"
            )

    input_channel = packed_k // (KH * KW)
    effective_hw = packed_k - input_channel * (KH * KW)
    effective_h = effective_hw // KW
    effective_w = effective_hw - effective_h * KW
    kernel_h = KH - 1 - effective_h if FLIP_FILTER else effective_h
    kernel_w = KW - 1 - effective_w if FLIP_FILTER else effective_w
    tl.store(
        dw_ptr
        + output_channels[:, None] * W_STRIDE_K
        + input_channel[None, :] * W_STRIDE_C
        + kernel_h[None, :] * W_STRIDE_H
        + kernel_w[None, :] * W_STRIDE_W,
        accumulator.to(dw_ptr.dtype.element_ty),
        mask=(output_channels[:, None] < COUT_PER_GROUP)
        & (packed_k[None, :] < reduction_extent),
    )


@triton.jit
def hygon_conv_wgrad2d_im2col_split_kernel(
    dy_ptr,
    col_ptr,
    partial_ptr,
    TOTAL_ROWS: tl.constexpr,
    ROWS_PER_SPLIT: tl.constexpr,
    M: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    DY_STRIDE_N: tl.constexpr,
    DY_STRIDE_C: tl.constexpr,
    DY_STRIDE_H: tl.constexpr,
    DY_STRIDE_W: tl.constexpr,
    OW: tl.constexpr,
    COL_STRIDE_N: tl.constexpr,
    COL_STRIDE_K: tl.constexpr,
    COL_STRIDE_M: tl.constexpr,
    PARTIAL_STRIDE_SPLIT: tl.constexpr,
    PARTIAL_STRIDE_OC: tl.constexpr,
    PARTIAL_STRIDE_K: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_CI_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Parallel split-K WGrad stage for large N*OH*OW reductions."""
    tile = tl.program_id(0)
    split = tl.program_id(1).to(tl.int64)
    reduction_extent: tl.constexpr = CIN_PER_GROUP * KH * KW
    tiles_k: tl.constexpr = tl.cdiv(reduction_extent, BLOCK_CI_K)
    tile_oc = tile // tiles_k
    tile_k = tile - tile_oc * tiles_k
    output_channels = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    packed_k = tile_k * BLOCK_CI_K + tl.arange(0, BLOCK_CI_K)
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
            mask=(output_channels[:, None] < COUT_PER_GROUP) & active[None, :],
            other=0.0,
        )
        columns = tl.load(
            col_ptr
            + batch[:, None] * COL_STRIDE_N
            + packed_k[None, :] * COL_STRIDE_K
            + output_m[:, None] * COL_STRIDE_M,
            mask=active[:, None] & (packed_k[None, :] < reduction_extent),
            other=0.0,
        )
        accumulator += tl.dot(losses, columns, input_precision="ieee")

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
def hygon_conv_wgrad2d_stem_split_kernel(
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
    BLOCK_OC: tl.constexpr,
    BLOCK_CI_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """N=1 low-CI WGrad split by complete output rows."""
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
def hygon_conv_wgrad2d_im2row_kernel(
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
    COL_STRIDE_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Pack NCHW input into backend-local row-major [N*M, C*KH*KW]."""
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
        output_h[:, None] * STRIDE_H - PAD_TOP + kernel_h[None, :] * DIL_H
    )
    input_w = (
        output_w[:, None] * STRIDE_W - PAD_LEFT + kernel_w[None, :] * DIL_W
    )
    active = (output_m[:, None] < M) & (packed_k[None, :] < reduction_extent)
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
        + rows[:, None] * COL_STRIDE_R
        + packed_k[None, :] * COL_STRIDE_K,
        values,
        mask=active,
    )


@triton.jit
def hygon_conv_wgrad2d_rowmajor_kernel(
    dy_ptr,
    col_ptr,
    dw_ptr,
    M: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FLIP_FILTER: tl.constexpr,
    DY_STRIDE_C: tl.constexpr,
    DY_STRIDE_H: tl.constexpr,
    DY_STRIDE_W: tl.constexpr,
    OW: tl.constexpr,
    COL_STRIDE_R: tl.constexpr,
    COL_STRIDE_K: tl.constexpr,
    W_STRIDE_K: tl.constexpr,
    W_STRIDE_C: tl.constexpr,
    W_STRIDE_H: tl.constexpr,
    W_STRIDE_W: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_CI_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """N=1 WGrad GEMM over row-major columns without a transpose."""
    tile = tl.program_id(0)
    reduction_extent: tl.constexpr = CIN_PER_GROUP * KH * KW
    tiles_oc: tl.constexpr = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    tile_k = tile // tiles_oc
    tile_oc = tile - tile_k * tiles_oc
    output_channels = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    packed_k = tile_k * BLOCK_CI_K + tl.arange(0, BLOCK_CI_K)
    accumulator = tl.zeros((BLOCK_OC, BLOCK_CI_K), dtype=tl.float32)

    for start in range(0, M, BLOCK_M):
        output_m = start + tl.arange(0, BLOCK_M)
        output_h = output_m // OW
        output_w = output_m - output_h * OW
        losses = tl.load(
            dy_ptr
            + output_channels[:, None] * DY_STRIDE_C
            + output_h[None, :] * DY_STRIDE_H
            + output_w[None, :] * DY_STRIDE_W,
            mask=(output_channels[:, None] < COUT_PER_GROUP)
            & (output_m[None, :] < M),
            other=0.0,
        )
        columns = tl.load(
            col_ptr
            + output_m[:, None] * COL_STRIDE_R
            + packed_k[None, :] * COL_STRIDE_K,
            mask=(output_m[:, None] < M)
            & (packed_k[None, :] < reduction_extent),
            other=0.0,
        )
        accumulator += tl.dot(losses, columns, input_precision="ieee")

    input_channel = packed_k // (KH * KW)
    effective_hw = packed_k - input_channel * (KH * KW)
    effective_h = effective_hw // KW
    effective_w = effective_hw - effective_h * KW
    kernel_h = KH - 1 - effective_h if FLIP_FILTER else effective_h
    kernel_w = KW - 1 - effective_w if FLIP_FILTER else effective_w
    tl.store(
        dw_ptr
        + output_channels[:, None] * W_STRIDE_K
        + input_channel[None, :] * W_STRIDE_C
        + kernel_h[None, :] * W_STRIDE_H
        + kernel_w[None, :] * W_STRIDE_W,
        accumulator.to(dw_ptr.dtype.element_ty),
        mask=(output_channels[:, None] < COUT_PER_GROUP)
        & (packed_k[None, :] < reduction_extent),
    )


@triton.jit
def hygon_conv_wgrad2d_p5_block_ptr_kernel(
    dy_ptr,
    col_ptr,
    dw_ptr,
    M: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    REDUCTION_EXTENT: tl.constexpr,
    DY_STRIDE_C: tl.constexpr,
    COL_STRIDE_R: tl.constexpr,
    COL_STRIDE_K: tl.constexpr,
    W_STRIDE_K: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_CI_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Large N=1 P5 WGrad GEMM with contiguous matrix block pointers."""
    tile = tl.program_id(0)
    tiles_oc: tl.constexpr = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    tile_k = tile // tiles_oc
    tile_oc = tile - tile_k * tiles_oc
    accumulator = tl.zeros((BLOCK_OC, BLOCK_CI_K), dtype=tl.float32)

    loss_block = tl.make_block_ptr(
        base=dy_ptr,
        shape=(COUT_PER_GROUP, M),
        strides=(DY_STRIDE_C, 1),
        offsets=(tile_oc * BLOCK_OC, 0),
        block_shape=(BLOCK_OC, BLOCK_M),
        order=(1, 0),
    )
    column_block = tl.make_block_ptr(
        base=col_ptr,
        shape=(M, REDUCTION_EXTENT),
        strides=(COL_STRIDE_R, COL_STRIDE_K),
        offsets=(0, tile_k * BLOCK_CI_K),
        block_shape=(BLOCK_M, BLOCK_CI_K),
        order=(1, 0),
    )
    for _ in range(0, M, BLOCK_M):
        losses = tl.load(
            loss_block,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        columns = tl.load(
            column_block,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        accumulator += tl.dot(losses, columns, input_precision="ieee")
        loss_block = tl.advance(loss_block, (0, BLOCK_M))
        column_block = tl.advance(column_block, (BLOCK_M, 0))

    output_block = tl.make_block_ptr(
        base=dw_ptr,
        shape=(COUT_PER_GROUP, REDUCTION_EXTENT),
        strides=(W_STRIDE_K, 1),
        offsets=(tile_oc * BLOCK_OC, tile_k * BLOCK_CI_K),
        block_shape=(BLOCK_OC, BLOCK_CI_K),
        order=(1, 0),
    )
    tl.store(
        output_block,
        accumulator.to(dw_ptr.dtype.element_ty),
        boundary_check=(0, 1),
    )


@triton.jit
def hygon_conv_wgrad2d_multirow_split_kernel(
    dy_ptr,
    x_ptr,
    partial_ptr,
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
    ROW_PITCH: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_CI_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """One-batch split WGrad with several output rows in each dot."""
    tile = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    reduction_extent: tl.constexpr = CIN_PER_GROUP * KH * KW
    tiles_oc: tl.constexpr = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    tile_k = tile // tiles_oc
    tile_oc = tile - tile_k * tiles_oc
    output_channels = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    packed_k = tile_k * BLOCK_CI_K + tl.arange(0, BLOCK_CI_K)
    input_channel = packed_k // (KH * KW)
    kernel_hw = packed_k - input_channel * (KH * KW)
    kernel_h = kernel_hw // KW
    kernel_w = kernel_hw - kernel_h * KW
    compact = tl.arange(0, BLOCK_M)
    delta_h = compact // ROW_PITCH
    output_w = compact - delta_h * ROW_PITCH
    rows_per_dot: tl.constexpr = BLOCK_M // ROW_PITCH
    accumulator = tl.zeros((BLOCK_OC, BLOCK_CI_K), dtype=tl.float32)

    for output_row_base in range(0, OH, rows_per_dot):
        output_h = output_row_base + delta_h
        active = (output_h < OH) & (output_w < OW)
        losses = tl.load(
            dy_ptr
            + batch * DY_STRIDE_N
            + output_channels[:, None] * DY_STRIDE_C
            + output_h[None, :] * DY_STRIDE_H
            + output_w[None, :] * DY_STRIDE_W,
            mask=(output_channels[:, None] < COUT_PER_GROUP) & active[None, :],
            other=0.0,
        )
        input_h = (
            output_h[:, None] * STRIDE_H - PAD_TOP + kernel_h[None, :] * DIL_H
        )
        input_w = (
            output_w[:, None] * STRIDE_W - PAD_LEFT + kernel_w[None, :] * DIL_W
        )
        inputs = tl.load(
            x_ptr
            + batch * X_STRIDE_N
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
        accumulator += tl.dot(losses, inputs, input_precision="ieee")

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
def hygon_conv_wgrad2d_direct_split_kernel(
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
    BLOCK_OC: tl.constexpr,
    BLOCK_CI_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Group-one split WGrad that gathers NCHW inputs without im2col."""
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
            mask=(output_channels[:, None] < COUT_PER_GROUP) & active[None, :],
            other=0.0,
        )
        input_h = (
            output_h[:, None] * STRIDE_H - PAD_TOP + kernel_h[None, :] * DIL_H
        )
        input_w = (
            output_w[:, None] * STRIDE_W - PAD_LEFT + kernel_w[None, :] * DIL_W
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
def hygon_conv_wgrad2d_1x1_split_kernel(
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
    BLOCK_OC: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Division-free split-K WGrad for contiguous 1x1 NCHW tensors."""
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
            mask=(output_channels[:, None] < COUT_PER_GROUP) & active[None, :],
            other=0.0,
        )
        inputs = tl.load(
            x_ptr
            + batch[:, None] * (C_IN * HW)
            + global_ci[None, :] * HW
            + spatial[:, None],
            mask=active[:, None] & (input_channels[None, :] < CIN_PER_GROUP),
            other=0.0,
        )
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
def hygon_conv_wgrad2d_reduce_kernel(
    partial_ptr,
    dw_ptr,
    TOTAL: tl.constexpr,
    CIK: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FLIP_FILTER: tl.constexpr,
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
    """Reduce FP32 WGrad partials and restore the public filter layout."""
    linear = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_channel = linear // CIK
    packed_k = linear - output_channel * CIK
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for split in tl.static_range(0, NUM_SPLITS):
        accumulator += tl.load(
            partial_ptr
            + split * PARTIAL_STRIDE_SPLIT
            + output_channel * PARTIAL_STRIDE_OC
            + packed_k * PARTIAL_STRIDE_K,
            mask=linear < TOTAL,
            other=0.0,
        )

    input_channel = packed_k // (KH * KW)
    effective_hw = packed_k - input_channel * (KH * KW)
    effective_h = effective_hw // KW
    effective_w = effective_hw - effective_h * KW
    kernel_h = KH - 1 - effective_h if FLIP_FILTER else effective_h
    kernel_w = KW - 1 - effective_w if FLIP_FILTER else effective_w
    tl.store(
        dw_ptr
        + output_channel * W_STRIDE_K
        + input_channel * W_STRIDE_C
        + kernel_h * W_STRIDE_H
        + kernel_w * W_STRIDE_W,
        accumulator.to(dw_ptr.dtype.element_ty),
        mask=(linear < TOTAL) & (input_channel < CIN_PER_GROUP),
    )
