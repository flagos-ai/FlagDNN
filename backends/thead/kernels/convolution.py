# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""THead-only convolution entry points with exact Graph boundary ABIs."""

import triton
import triton.language as tl


@triton.jit
def conv_fprop_nd_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
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
    GROUPS: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_FRONT: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
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
    M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """N-D cross-correlation over explicit logical tensor strides."""

    tile = tl.program_id(0)
    group = tl.program_id(1).to(tl.int64)
    tiles_oc = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    tile_m = tile // tiles_oc
    tile_oc = tile % tiles_oc
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    output_channels = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    output_volume: tl.constexpr = OD * OH * OW
    batch = rows // output_volume
    output_spatial = rows % output_volume
    output_d = output_spatial // (OH * OW)
    output_hw = output_spatial % (OH * OW)
    output_h = output_hw // OW
    output_w = output_hw % OW
    kernel_volume: tl.constexpr = KD * KH * KW
    reduction_extent: tl.constexpr = CIN_PER_GROUP * kernel_volume
    reduction_base = tl.arange(0, BLOCK_K)
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
            output_h[:, None] * STRIDE_H
            - PAD_TOP
            + kernel_h[None, :] * DIL_H
        )
        input_w = (
            output_w[:, None] * STRIDE_W
            - PAD_LEFT
            + kernel_w[None, :] * DIL_W
        )
        reduction_active = reduction < reduction_extent
        inputs = tl.load(
            x_ptr
            + batch[:, None] * X_STRIDE_N
            + (group * CIN_PER_GROUP + input_channel[None, :]) * X_STRIDE_C
            + input_d * X_STRIDE_D
            + input_h * X_STRIDE_H
            + input_w * X_STRIDE_W,
            mask=(rows[:, None] < M)
            & reduction_active[None, :]
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
            + (group * COUT_PER_GROUP + output_channels[:, None])
            * W_STRIDE_K
            + input_channel[None, :] * W_STRIDE_C
            + kernel_d[None, :] * W_STRIDE_D
            + kernel_h[None, :] * W_STRIDE_H
            + kernel_w[None, :] * W_STRIDE_W,
            mask=(output_channels[:, None] < COUT_PER_GROUP)
            & reduction_active[None, :],
            other=0.0,
        )
        accumulator += tl.dot(inputs, tl.trans(weights), input_precision="ieee")

    tl.store(
        y_ptr
        + batch[:, None] * Y_STRIDE_N
        + (group * COUT_PER_GROUP + output_channels[None, :]) * Y_STRIDE_C
        + output_d[:, None] * Y_STRIDE_D
        + output_h[:, None] * Y_STRIDE_H
        + output_w[:, None] * Y_STRIDE_W,
        accumulator.to(y_ptr.dtype.element_ty),
        mask=(rows[:, None] < M)
        & (output_channels[None, :] < COUT_PER_GROUP),
    )
@triton.jit
def conv2d_bias_relu_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
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
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    W_STRIDE_K: tl.constexpr,
    W_STRIDE_C: tl.constexpr,
    W_STRIDE_R: tl.constexpr,
    W_STRIDE_S: tl.constexpr,
    BIAS_STRIDE_C: tl.constexpr,
    Y_STRIDE_N: tl.constexpr,
    Y_STRIDE_C: tl.constexpr,
    Y_STRIDE_H: tl.constexpr,
    Y_STRIDE_W: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
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
            output_h[:, None] * STRIDE_H
            - PAD_TOP
            + kernel_h[None, :] * DIL_H
        )
        input_w = (
            output_w[:, None] * STRIDE_W
            - PAD_LEFT
            + kernel_w[None, :] * DIL_W
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
            + (group * COUT_PER_GROUP + output_channels[:, None])
            * W_STRIDE_K
            + input_channel[None, :] * W_STRIDE_C
            + kernel_h[None, :] * W_STRIDE_R
            + kernel_w[None, :] * W_STRIDE_S,
            mask=(output_channels[:, None] < COUT_PER_GROUP)
            & (reduction[None, :] < reduction_extent),
            other=0.0,
        )
        accumulator += tl.dot(
            input_values,
            tl.trans(weights),
            input_precision="ieee",
        )

    global_channels = group * COUT_PER_GROUP + output_channels
    bias = tl.load(
        bias_ptr + global_channels * BIAS_STRIDE_C,
        mask=output_channels < COUT_PER_GROUP,
        other=0.0,
    )
    activated = tl.maximum(accumulator + bias[None, :], 0.0)
    tl.store(
        y_ptr
        + batch * Y_STRIDE_N
        + global_channels[None, :] * Y_STRIDE_C
        + output_h[:, None] * Y_STRIDE_H
        + output_w[:, None] * Y_STRIDE_W,
        activated.to(y_ptr.dtype.element_ty),
        mask=(output_hw[:, None] < OH * OW)
        & (output_channels[None, :] < COUT_PER_GROUP),
    )
