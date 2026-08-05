# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA convolution kernels with Native ABI baselines and optimized variants.

The public compiler ABI uses logical dimensions and explicit tensor strides,
so these kernels do not depend on Torch layouts, packing caches, or Python
dispatch helpers.
"""

import triton
import triton.language as tl


@triton.jit
def _round_fp32_to_tf32_rne(x):
    """Use NVIDIA's native round-to-nearest FP32-to-TF32 conversion."""
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .b32 tf32;
            cvt.rna.tf32.f32 tf32, $1;
            mov.b32 $0, tf32;
        }
        """,
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def conv2d_im2col_nchw_3x3_stride2_pad1_kernel(
    x_ptr,
    col_ptr,
    TOTAL: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    COL_STRIDE_N: tl.constexpr,
    COL_STRIDE_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Materialize NCHW 3x3/stride-2/pad-1 patches for batched GEMM."""
    output_area: tl.constexpr = OH * OW
    plane = tl.program_id(1)
    planes_per_batch: tl.constexpr = CIN_PER_GROUP * 3
    batch = plane // planes_per_batch
    plane_in_batch = plane - batch * planes_per_batch
    input_channel = plane_in_batch // 3
    kernel_h = plane_in_batch - input_channel * 3
    output_hw = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_h = output_hw // OW
    output_w = output_hw - output_h * OW
    input_h = output_h * 2 - 1 + kernel_h
    valid_hw = output_hw < output_area
    valid_h = valid_hw & (input_h >= 0) & (input_h < XH)
    column_base = (
        batch * COL_STRIDE_N
        + (input_channel * 9 + kernel_h * 3) * COL_STRIDE_K
    )

    for kernel_w in tl.static_range(0, 3):
        input_w = output_w * 2 - 1 + kernel_w
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
        column_offsets = column_base + kernel_w * COL_STRIDE_K + output_hw
        tl.store(
            col_ptr + column_offsets,
            values,
            mask=valid_hw & (column_offsets < TOTAL),
        )


@triton.jit
def conv2d_im2col_nchw_kernel(
    x_ptr,
    col_ptr,
    weight_ptr,
    converted_weight_ptr,
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
    WEIGHT_TOTAL: tl.constexpr,
    WEIGHT_BLOCK: tl.constexpr,
    CONVERT_WEIGHT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Materialize general NCHW patches and optionally convert FP32 weights.

    Weight conversion shares the im2col launch so the mixed-FP16 path does
    not pay for a third kernel.  The following GEMM launch is the global
    synchronization point before consuming either workspace tensor.
    """
    output_area: tl.constexpr = OH * OW
    plane = tl.program_id(1)
    planes_per_batch: tl.constexpr = CIN_PER_GROUP * KH
    batch = plane // planes_per_batch
    plane_in_batch = plane - batch * planes_per_batch
    input_channel = plane_in_batch // KH
    kernel_h = plane_in_batch - input_channel * KH
    output_hw = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_h = output_hw // OW
    output_w = output_hw - output_h * OW
    input_h = output_h * STRIDE_H - PAD_TOP + kernel_h * DIL_H
    valid_hw = output_hw < output_area
    valid_h = valid_hw & (input_h >= 0) & (input_h < XH)
    column_base = (
        batch * COL_STRIDE_N
        + (input_channel * KH * KW + kernel_h * KW) * COL_STRIDE_K
    )

    for kernel_w in tl.static_range(0, KW):
        input_w = output_w * STRIDE_W - PAD_LEFT + kernel_w * DIL_W
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
            col_ptr + column_base + kernel_w * COL_STRIDE_K + output_hw,
            values,
            mask=valid_hw,
        )

    if CONVERT_WEIGHT:
        linear_program = tl.program_id(1) * tl.num_programs(0) + tl.program_id(
            0
        )
        weight_start = linear_program * WEIGHT_BLOCK
        if weight_start < WEIGHT_TOTAL:
            weight_offsets = weight_start + tl.arange(0, WEIGHT_BLOCK)
            weight_mask = weight_offsets < WEIGHT_TOTAL
            weight_values = tl.load(
                weight_ptr + weight_offsets,
                mask=weight_mask,
                other=0.0,
            )
            tl.store(
                converted_weight_ptr + weight_offsets,
                weight_values,
                mask=weight_mask,
            )


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
    APPLY_RELU: tl.constexpr,
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
        if DTYPE_ID == 2 and INPUT_PRECISION == 1:
            accumulator += tl.dot(
                input_values, tl.trans(weights), input_precision="tf32x3"
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
    if APPLY_RELU:
        accumulator = tl.maximum(accumulator, 0.0)
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
    APPLY_RELU: tl.constexpr,
    BIAS_STRIDE: tl.constexpr,
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
    output_area: tl.constexpr = OH * OW
    reduction_extent: tl.constexpr = CIN_PER_GROUP * KH * KW
    kernel_area: tl.constexpr = KH * KW

    tiles_hw = tl.cdiv(output_area, BLOCK_HW)
    tiles_oc = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    tiles_per_group = GROUP_M * tiles_oc
    tile_group = tile // tiles_per_group
    first_tile_hw = tile_group * GROUP_M
    group_size_hw = min(tiles_hw - first_tile_hw, GROUP_M)
    tile_in_group = tile % tiles_per_group
    tile_hw = first_tile_hw + tile_in_group % group_size_hw
    tile_oc = tile_in_group // group_size_hw

    output_hw = tile_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    output_channels = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    output_h = output_hw // OW
    output_w = output_hw % OW
    reduction_base = tl.arange(0, BLOCK_K)
    output_mask = output_hw < output_area
    channel_mask = output_channels < COUT_PER_GROUP
    accumulator = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for start in range(0, reduction_extent, BLOCK_K):
        reduction = start + reduction_base
        reduction_mask = reduction < reduction_extent
        input_channel = reduction // kernel_area
        kernel_hw = reduction - input_channel * kernel_area
        kernel_h = kernel_hw // KW
        kernel_w = kernel_hw - kernel_h * KW
        input_h = (
            output_h[None, :] * STRIDE_H - PAD_TOP + kernel_h[:, None] * DIL_H
        )
        input_w = (
            output_w[None, :] * STRIDE_W - PAD_LEFT + kernel_w[:, None] * DIL_W
        )
        input_values = tl.load(
            x_ptr
            + batch * X_STRIDE_N
            + (group * CIN_PER_GROUP + input_channel[:, None]) * X_STRIDE_C
            + input_h * X_STRIDE_H
            + input_w * X_STRIDE_W,
            mask=output_mask[None, :]
            & reduction_mask[:, None]
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
            mask=channel_mask[:, None] & reduction_mask[None, :],
            other=0.0,
        )
        if DTYPE_ID == 2 and INPUT_PRECISION == 1:
            accumulator += tl.dot(
                weights, input_values, input_precision="tf32x3"
            )
        else:
            accumulator += tl.dot(
                weights, input_values, input_precision="ieee"
            )

    if HAS_BIAS:
        bias = tl.load(
            bias_ptr
            + (group * COUT_PER_GROUP + output_channels) * BIAS_STRIDE,
            mask=channel_mask,
            other=0.0,
        )
        accumulator += bias[:, None]
    if APPLY_RELU:
        accumulator = tl.maximum(accumulator, 0.0)
    tl.store(
        y_ptr
        + batch * Y_STRIDE_N
        + (group * COUT_PER_GROUP + output_channels[:, None]) * Y_STRIDE_C
        + output_h[None, :] * Y_STRIDE_H
        + output_w[None, :] * Y_STRIDE_W,
        accumulator.to(y_ptr.dtype.element_ty),
        mask=channel_mask[:, None] & output_mask[None, :],
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
    APPLY_RELU: tl.constexpr,
    BIAS_STRIDE: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """NCHW 1x1 stride-one/pad-zero convolution without spatial indexing."""
    tile = tl.program_id(0)
    batch_group = tl.program_id(1).to(tl.int64)
    batch = batch_group // GROUPS
    group = batch_group % GROUPS

    tiles_hw = tl.cdiv(HW, BLOCK_HW)
    tiles_oc = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    tiles_per_group = GROUP_M * tiles_oc
    tile_group = tile // tiles_per_group
    first_tile_hw = tile_group * GROUP_M
    group_size_hw = min(tiles_hw - first_tile_hw, GROUP_M)
    tile_in_group = tile % tiles_per_group
    tile_hw = first_tile_hw + tile_in_group % group_size_hw
    tile_oc = tile_in_group // group_size_hw

    output_hw = tile_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    output_channels = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    reduction_base = tl.arange(0, BLOCK_K)
    output_mask = output_hw < HW
    channel_mask = output_channels < COUT_PER_GROUP
    accumulator = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for start in range(0, CIN_PER_GROUP, BLOCK_K):
        input_channels = start + reduction_base
        reduction_mask = input_channels < CIN_PER_GROUP
        global_input_channels = group * CIN_PER_GROUP + input_channels
        input_values = tl.load(
            x_ptr
            + batch * (C_IN * HW)
            + global_input_channels[:, None] * HW
            + output_hw[None, :],
            mask=reduction_mask[:, None] & output_mask[None, :],
            other=0.0,
        )
        weights = tl.load(
            w_ptr
            + (group * COUT_PER_GROUP + output_channels[:, None])
            * CIN_PER_GROUP
            + input_channels[None, :],
            mask=channel_mask[:, None] & reduction_mask[None, :],
            other=0.0,
        )
        if DTYPE_ID == 2 and INPUT_PRECISION == 1:
            accumulator += tl.dot(
                weights, input_values, input_precision="tf32x3"
            )
        else:
            accumulator += tl.dot(
                weights, input_values, input_precision="ieee"
            )

    global_output_channels = group * COUT_PER_GROUP + output_channels
    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + global_output_channels * BIAS_STRIDE,
            mask=channel_mask,
            other=0.0,
        )
        accumulator += bias[:, None]
    if APPLY_RELU:
        accumulator = tl.maximum(accumulator, 0.0)
    tl.store(
        y_ptr
        + batch * (C_OUT * HW)
        + global_output_channels[:, None] * HW
        + output_hw[None, :],
        accumulator.to(y_ptr.dtype.element_ty),
        mask=channel_mask[:, None] & output_mask[None, :],
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
    APPLY_RELU: tl.constexpr,
    BIAS_STRIDE: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DTYPE_ID: tl.constexpr,
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
        if DTYPE_ID == 2 and INPUT_PRECISION == 1:
            accumulator += tl.dot(
                input_values, tl.trans(weights), input_precision="tf32x3"
            )
        else:
            accumulator += tl.dot(
                input_values, tl.trans(weights), input_precision="ieee"
            )

    if HAS_BIAS:
        bias = tl.load(
            bias_ptr
            + (group * COUT_PER_GROUP + output_channels) * BIAS_STRIDE,
            mask=output_channels < COUT_PER_GROUP,
            other=0.0,
        )
        accumulator += bias[None, :]
    if APPLY_RELU:
        accumulator = tl.maximum(accumulator, 0.0)
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
def conv_dgrad2d_1x1_nchw_kernel(
    dy_ptr,
    w_ptr,
    dx_ptr,
    HW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    """NCHW 1x1 DGrad as a division-free matrix product."""
    tile = tl.program_id(0)
    batch_group = tl.program_id(1).to(tl.int64)
    batch = batch_group // GROUPS
    group = batch_group - batch * GROUPS
    tiles_m = tl.cdiv(HW, BLOCK_M)
    tile_ci = tile // tiles_m
    tile_m = tile - tile_ci * tiles_m

    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci = tile_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < HW
    mask_ci = offs_ci < CIN_PER_GROUP
    accumulator = tl.zeros((BLOCK_CI, BLOCK_M), dtype=tl.float32)

    for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
        offs_co = co_start + tl.arange(0, BLOCK_CO)
        mask_co = offs_co < COUT_PER_GROUP
        global_co = group * COUT_PER_GROUP + offs_co
        losses = tl.load(
            dy_ptr
            + batch * (C_OUT * HW)
            + global_co[:, None] * HW
            + offs_m[None, :],
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        weights = tl.load(
            w_ptr + global_co[:, None] * CIN_PER_GROUP + offs_ci[None, :],
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        if INPUT_PRECISION == 1:
            accumulator = tl.dot(
                tl.trans(weights),
                losses,
                accumulator,
                input_precision="tf32",
            )
        else:
            accumulator = tl.dot(
                tl.trans(weights),
                losses,
                accumulator,
                input_precision="ieee",
            )

    global_ci = group * CIN_PER_GROUP + offs_ci
    tl.store(
        dx_ptr
        + batch * (C_IN * HW)
        + global_ci[:, None] * HW
        + offs_m[None, :],
        accumulator.to(dx_ptr.dtype.element_ty),
        mask=mask_ci[:, None] & mask_m[None, :],
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
        if INPUT_PRECISION == 1:
            accumulator += tl.dot(losses, weights, input_precision="tf32")
        else:
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
def conv_dgrad2d_stride1_kernel(
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
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FLIP_FILTER: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    """2D stride-one dgrad with static filter loops."""
    tile = tl.program_id(0)
    group = tl.program_id(1).to(tl.int64)
    tiles_m = tl.cdiv(M, BLOCK_M)
    tile_ci = tile // tiles_m
    tile_m = tile - tile_ci * tiles_m

    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    input_channels = tile_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    row_mask = rows < M
    channel_mask = input_channels < CIN_PER_GROUP
    input_area: tl.constexpr = XH * XW
    batch = rows // input_area
    input_spatial = rows - batch * input_area
    input_h = input_spatial // XW
    input_w = input_spatial - input_h * XW
    global_input_channels = group * CIN_PER_GROUP + input_channels
    accumulator = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for kernel_h in tl.static_range(0, KH):
        output_h = input_h + PAD_TOP - kernel_h * DIL_H
        valid_h = (output_h >= 0) & (output_h < OH)
        safe_h = tl.where(valid_h, output_h, 0)
        weight_h = KH - 1 - kernel_h if FLIP_FILTER else kernel_h
        for kernel_w in tl.static_range(0, KW):
            output_w = input_w + PAD_LEFT - kernel_w * DIL_W
            valid_w = (output_w >= 0) & (output_w < OW)
            valid_spatial = valid_h & valid_w
            safe_w = tl.where(valid_w, output_w, 0)
            weight_w = KW - 1 - kernel_w if FLIP_FILTER else kernel_w
            for channel_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
                output_channels = channel_start + tl.arange(0, BLOCK_CO)
                global_output_channels = (
                    group * COUT_PER_GROUP + output_channels
                )
                output_channel_mask = output_channels < COUT_PER_GROUP
                losses = tl.load(
                    dy_ptr
                    + batch[:, None] * DY_STRIDE_N
                    + global_output_channels[None, :] * DY_STRIDE_C
                    + safe_h[:, None] * DY_STRIDE_H
                    + safe_w[:, None] * DY_STRIDE_W,
                    mask=row_mask[:, None]
                    & output_channel_mask[None, :]
                    & valid_spatial[:, None],
                    other=0.0,
                )
                weights = tl.load(
                    w_ptr
                    + global_output_channels[:, None] * W_STRIDE_K
                    + input_channels[None, :] * W_STRIDE_C
                    + weight_h * W_STRIDE_H
                    + weight_w * W_STRIDE_W,
                    mask=output_channel_mask[:, None] & channel_mask[None, :],
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
        + global_input_channels[None, :] * X_STRIDE_C
        + input_h[:, None] * X_STRIDE_H
        + input_w[:, None] * X_STRIDE_W,
        accumulator.to(dx_ptr.dtype.element_ty),
        mask=row_mask[:, None] & channel_mask[None, :],
    )


@triton.jit
def cast_contiguous_kernel(
    input_ptr,
    output_ptr,
    TOTAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Convert one contiguous internal pipeline tensor."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = offsets < TOTAL
    values = tl.load(input_ptr + offsets, mask=active, other=0.0)
    tl.store(
        output_ptr + offsets,
        values.to(output_ptr.dtype.element_ty),
        mask=active,
    )


@triton.jit
def conv_dgrad2d_pack_weight_kernel(
    weight_ptr,
    packed_ptr,
    TOTAL: tl.constexpr,
    C_OUT: tl.constexpr,
    C_IN: tl.constexpr,
    ROUND_TF32: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Transpose OI-by-HW weights into HW-by-OI with coalesced traffic."""
    pair_count: tl.constexpr = C_OUT * C_IN
    pair = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    kernel_position = tl.arange(0, 16)
    mask = (pair[:, None] < pair_count) & (kernel_position[None, :] < 9)
    values = tl.load(
        weight_ptr + pair[:, None] * 9 + kernel_position[None, :],
        mask=mask,
        other=0.0,
    )
    if ROUND_TF32:
        values = _round_fp32_to_tf32_rne(values)
    tl.store(
        packed_ptr + kernel_position[:, None] * pair_count + pair[None, :],
        tl.trans(values),
        mask=tl.trans(mask),
    )


@triton.jit
def zero_contiguous_kernel(
    out_ptr,
    TOTAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Initialize one contiguous internal or output tensor to zero."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(
        out_ptr + offsets,
        tl.zeros((BLOCK_SIZE,), dtype=tl.float32),
        mask=offsets < TOTAL,
    )


@triton.jit
def conv_dgrad2d_p5_fp32_tile2w_splitk_kernel(
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
    """FP32 P5 DGrad split-K stage producing both width parities."""
    num_m_blocks: tl.constexpr = (M + BLOCK_M - 1) // BLOCK_M
    num_k_blocks: tl.constexpr = (COUT_PER_GROUP + BLOCK_CO - 1) // BLOCK_CO
    num_split_k: tl.constexpr = (num_k_blocks + GROUP_K - 1) // GROUP_K
    pid = tl.program_id(0)
    pid_m = pid % num_m_blocks
    pid_tmp = pid // num_m_blocks
    pid_k_group = pid_tmp % num_split_k
    pid_ci = pid_tmp // num_split_k

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci < CIN_PER_GROUP

    yh = offs_m // LOSS_W
    yw = offs_m - yh * LOSS_W
    xh = yh * 2 + PH
    xw0 = yw * 2
    loss_base = yh * loss_stride_h + yw * loss_stride_w
    valid_yh1 = yh + 1 < LOSS_H
    valid_yw1 = yw + 1 < LOSS_W

    acc0 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    for k_inner in tl.static_range(0, GROUP_K):
        pid_k = pid_k_group * GROUP_K + k_inner
        offs_co = pid_k * BLOCK_CO + tl.arange(0, BLOCK_CO)
        mask_co = offs_co < COUT_PER_GROUP
        common_mask = mask_m[:, None] & mask_co[None, :]
        weight_mask = mask_co[:, None] & mask_ci[None, :]
        loss00 = tl.load(
            loss_ptr + offs_co[None, :] * loss_stride_c + loss_base[:, None],
            mask=common_mask,
            other=0.0,
        )
        loss01 = tl.load(
            loss_ptr
            + offs_co[None, :] * loss_stride_c
            + (loss_base + loss_stride_w)[:, None],
            mask=common_mask & valid_yw1[:, None],
            other=0.0,
        )
        if PH == 0:
            weight11 = tl.load(
                weight_ptr
                + (
                    ((1 * 3 + 1) * COUT_PER_GROUP + offs_co[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci[None, :],
                mask=weight_mask,
                other=0.0,
            )
            weight12 = tl.load(
                weight_ptr
                + (
                    ((1 * 3 + 2) * COUT_PER_GROUP + offs_co[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci[None, :],
                mask=weight_mask,
                other=0.0,
            )
            weight10 = tl.load(
                weight_ptr
                + (
                    ((1 * 3) * COUT_PER_GROUP + offs_co[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci[None, :],
                mask=weight_mask,
                other=0.0,
            )
            acc0 = tl.dot(loss00, weight11, acc0, input_precision="tf32")
            acc1 = tl.dot(loss00, weight12, acc1, input_precision="tf32")
            acc1 = tl.dot(loss01, weight10, acc1, input_precision="tf32")
        else:
            loss10 = tl.load(
                loss_ptr
                + offs_co[None, :] * loss_stride_c
                + (loss_base + loss_stride_h)[:, None],
                mask=common_mask & valid_yh1[:, None],
                other=0.0,
            )
            loss11 = tl.load(
                loss_ptr
                + offs_co[None, :] * loss_stride_c
                + (loss_base + loss_stride_h + loss_stride_w)[:, None],
                mask=(common_mask & valid_yh1[:, None] & valid_yw1[:, None]),
                other=0.0,
            )
            weight21 = tl.load(
                weight_ptr
                + (
                    ((2 * 3 + 1) * COUT_PER_GROUP + offs_co[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci[None, :],
                mask=weight_mask,
                other=0.0,
            )
            weight01 = tl.load(
                weight_ptr
                + ((1 * COUT_PER_GROUP + offs_co[:, None]) * CIN_PER_GROUP)
                + offs_ci[None, :],
                mask=weight_mask,
                other=0.0,
            )
            weight22 = tl.load(
                weight_ptr
                + (
                    ((2 * 3 + 2) * COUT_PER_GROUP + offs_co[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci[None, :],
                mask=weight_mask,
                other=0.0,
            )
            weight20 = tl.load(
                weight_ptr
                + (
                    ((2 * 3) * COUT_PER_GROUP + offs_co[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci[None, :],
                mask=weight_mask,
                other=0.0,
            )
            weight02 = tl.load(
                weight_ptr
                + ((2 * COUT_PER_GROUP + offs_co[:, None]) * CIN_PER_GROUP)
                + offs_ci[None, :],
                mask=weight_mask,
                other=0.0,
            )
            weight00 = tl.load(
                weight_ptr
                + offs_co[:, None] * CIN_PER_GROUP
                + offs_ci[None, :],
                mask=weight_mask,
                other=0.0,
            )
            acc0 = tl.dot(loss00, weight21, acc0, input_precision="tf32")
            acc0 = tl.dot(loss10, weight01, acc0, input_precision="tf32")
            acc1 = tl.dot(loss00, weight22, acc1, input_precision="tf32")
            acc1 = tl.dot(loss01, weight20, acc1, input_precision="tf32")
            acc1 = tl.dot(loss10, weight02, acc1, input_precision="tf32")
            acc1 = tl.dot(loss11, weight00, acc1, input_precision="tf32")

    out0 = (
        out_ptr
        + offs_ci[None, :] * out_stride_c
        + xh[:, None] * out_stride_h
        + xw0[:, None] * out_stride_w
    )
    mask = mask_m[:, None] & mask_ci[None, :]
    tl.atomic_add(out0, acc0, sem="relaxed", mask=mask)
    tl.atomic_add(out0 + out_stride_w, acc1, sem="relaxed", mask=mask)


@triton.jit
def conv_dgrad3d_pack_weight_kernel(
    weight_ptr,
    packed_ptr,
    TOTAL: tl.constexpr,
    C_OUT: tl.constexpr,
    C_IN: tl.constexpr,
    KERNEL_VOLUME: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Transpose OI-by-KDHW weights into KDHW-by-OI."""
    pair_count: tl.constexpr = C_OUT * C_IN
    pair = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    kernel_position = tl.arange(0, 32)
    mask = (pair[:, None] < pair_count) & (
        kernel_position[None, :] < KERNEL_VOLUME
    )
    values = tl.load(
        weight_ptr + pair[:, None] * KERNEL_VOLUME + kernel_position[None, :],
        mask=mask,
        other=0.0,
    )
    tl.store(
        packed_ptr + kernel_position[:, None] * pair_count + pair[None, :],
        tl.trans(values),
        mask=tl.trans(mask),
    )


@triton.jit
def conv_dgrad2d_stride2_pad1_3x3_packed_parity_kernel(
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
    INPUT_PRECISION: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    """One constexpr input-parity slice of packed stride-2 DGrad."""
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

    accumulator = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    for kh_index in tl.static_range(0, KH_COUNT):
        if PH == 0:
            kh = 1
            loss_h = yh
        else:
            kh = kh_index * 2
            loss_h = yh + (1 if kh_index == 0 else 0)
        valid_h = loss_h < LOSS_H
        weight_h = 2 - kh if FILTER_REVERSE else kh

        for kw_index in tl.static_range(0, KW_COUNT):
            if PW == 0:
                kw = 1
                loss_w = yw
            else:
                kw = kw_index * 2
                loss_w = yw + (1 if kw_index == 0 else 0)
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
                if INPUT_PRECISION == 1:
                    accumulator = tl.dot(
                        loss,
                        weight,
                        accumulator,
                        input_precision="tf32",
                    )
                else:
                    accumulator = tl.dot(
                        loss,
                        weight,
                        accumulator,
                        input_precision="ieee",
                    )

    tl.store(
        out_ptr
        + n_idx[:, None] * out_stride_n
        + offs_ci_rel[None, :] * out_stride_c
        + xh[:, None] * out_stride_h
        + xw[:, None] * out_stride_w,
        accumulator.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_ci[None, :],
    )


@triton.jit
def conv_dgrad2d_stride2_pad1_3x3_packed_tile2w_kernel(
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
    PH: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    """One constexpr height parity producing both output-width parities."""
    pid = tl.program_id(0)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci < CIN_PER_GROUP

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

    valid0 = mask_m & (xh < XH) & (xw0 < XW)
    valid1 = mask_m & (xh < XH) & (xw1 < XW)
    valid_yh1 = yh1 < LOSS_H
    valid_yw1 = yw1 < LOSS_W

    if FILTER_REVERSE:
        w0: tl.constexpr = 2
        w1: tl.constexpr = 1
        w2: tl.constexpr = 0
    else:
        w0: tl.constexpr = 0
        w1: tl.constexpr = 1
        w2: tl.constexpr = 2

    acc0 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
        offs_co = co_start + tl.arange(0, BLOCK_CO)
        mask_co = offs_co < COUT_PER_GROUP
        common_mask = mask_m[:, None] & mask_co[None, :]
        loss00 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + offs_co[None, :] * loss_stride_c
            + yh[:, None] * loss_stride_h
            + yw[:, None] * loss_stride_w,
            mask=common_mask,
            other=0.0,
        )
        loss01 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + offs_co[None, :] * loss_stride_c
            + yh[:, None] * loss_stride_h
            + yw1[:, None] * loss_stride_w,
            mask=common_mask & valid_yw1[:, None],
            other=0.0,
        )

        if PH == 0:
            weight11 = tl.load(
                weight_ptr
                + (
                    ((w1 * 3 + w1) * COUT_PER_GROUP + offs_co[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight12 = tl.load(
                weight_ptr
                + (
                    ((w1 * 3 + w2) * COUT_PER_GROUP + offs_co[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight10 = tl.load(
                weight_ptr
                + (
                    ((w1 * 3 + w0) * COUT_PER_GROUP + offs_co[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            if INPUT_PRECISION == 1:
                acc0 = tl.dot(loss00, weight11, acc0, input_precision="tf32")
                acc1 = tl.dot(loss00, weight12, acc1, input_precision="tf32")
                acc1 = tl.dot(loss01, weight10, acc1, input_precision="tf32")
            else:
                acc0 = tl.dot(loss00, weight11, acc0, input_precision="ieee")
                acc1 = tl.dot(loss00, weight12, acc1, input_precision="ieee")
                acc1 = tl.dot(loss01, weight10, acc1, input_precision="ieee")
        else:
            loss10 = tl.load(
                loss_ptr
                + n_idx[:, None] * loss_stride_n
                + offs_co[None, :] * loss_stride_c
                + yh1[:, None] * loss_stride_h
                + yw[:, None] * loss_stride_w,
                mask=common_mask & valid_yh1[:, None],
                other=0.0,
            )
            loss11 = tl.load(
                loss_ptr
                + n_idx[:, None] * loss_stride_n
                + offs_co[None, :] * loss_stride_c
                + yh1[:, None] * loss_stride_h
                + yw1[:, None] * loss_stride_w,
                mask=(common_mask & valid_yh1[:, None] & valid_yw1[:, None]),
                other=0.0,
            )
            weight21 = tl.load(
                weight_ptr
                + (
                    ((w2 * 3 + w1) * COUT_PER_GROUP + offs_co[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight01 = tl.load(
                weight_ptr
                + (
                    ((w0 * 3 + w1) * COUT_PER_GROUP + offs_co[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight22 = tl.load(
                weight_ptr
                + (
                    ((w2 * 3 + w2) * COUT_PER_GROUP + offs_co[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight20 = tl.load(
                weight_ptr
                + (
                    ((w2 * 3 + w0) * COUT_PER_GROUP + offs_co[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight02 = tl.load(
                weight_ptr
                + (
                    ((w0 * 3 + w2) * COUT_PER_GROUP + offs_co[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight00 = tl.load(
                weight_ptr
                + (
                    ((w0 * 3 + w0) * COUT_PER_GROUP + offs_co[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            if INPUT_PRECISION == 1:
                acc0 = tl.dot(loss00, weight21, acc0, input_precision="tf32")
                acc0 = tl.dot(loss10, weight01, acc0, input_precision="tf32")
                acc1 = tl.dot(loss00, weight22, acc1, input_precision="tf32")
                acc1 = tl.dot(loss01, weight20, acc1, input_precision="tf32")
                acc1 = tl.dot(loss10, weight02, acc1, input_precision="tf32")
                acc1 = tl.dot(loss11, weight00, acc1, input_precision="tf32")
            else:
                acc0 = tl.dot(loss00, weight21, acc0, input_precision="ieee")
                acc0 = tl.dot(loss10, weight01, acc0, input_precision="ieee")
                acc1 = tl.dot(loss00, weight22, acc1, input_precision="ieee")
                acc1 = tl.dot(loss01, weight20, acc1, input_precision="ieee")
                acc1 = tl.dot(loss10, weight02, acc1, input_precision="ieee")
                acc1 = tl.dot(loss11, weight00, acc1, input_precision="ieee")

    out_base = (
        out_ptr
        + n_idx[:, None] * out_stride_n
        + offs_ci[None, :] * out_stride_c
        + xh[:, None] * out_stride_h
    )
    tl.store(
        out_base + xw0[:, None] * out_stride_w,
        acc0.to(out_ptr.dtype.element_ty),
        mask=valid0[:, None] & mask_ci[None, :],
    )
    tl.store(
        out_base + xw1[:, None] * out_stride_w,
        acc1.to(out_ptr.dtype.element_ty),
        mask=valid1[:, None] & mask_ci[None, :],
    )


@triton.jit
def conv_dgrad2d_stride2_pad1_3x3_packed_tile4_kernel(
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
    INPUT_PRECISION: tl.constexpr,
    ROUND_TF32: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    """Produce one packed 2x2 DGrad output tile per loss position."""
    pid = tl.program_id(0)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci < CIN_PER_GROUP

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

    if FILTER_REVERSE:
        w0: tl.constexpr = 2
        w1: tl.constexpr = 1
        w2: tl.constexpr = 0
    else:
        w0: tl.constexpr = 0
        w1: tl.constexpr = 1
        w2: tl.constexpr = 2

    acc00 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
        offs_co = co_start + tl.arange(0, BLOCK_CO)
        mask_co = offs_co < COUT_PER_GROUP
        common_mask = mask_m[:, None] & mask_co[None, :]
        weight_mask = mask_co[:, None] & mask_ci[None, :]
        loss00 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + offs_co[None, :] * loss_stride_c
            + yh[:, None] * loss_stride_h
            + yw[:, None] * loss_stride_w,
            mask=common_mask,
            other=0.0,
        )
        loss01 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + offs_co[None, :] * loss_stride_c
            + yh[:, None] * loss_stride_h
            + yw1[:, None] * loss_stride_w,
            mask=common_mask & valid_yw1[:, None],
            other=0.0,
        )
        loss10 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + offs_co[None, :] * loss_stride_c
            + yh1[:, None] * loss_stride_h
            + yw[:, None] * loss_stride_w,
            mask=common_mask & valid_yh1[:, None],
            other=0.0,
        )
        loss11 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + offs_co[None, :] * loss_stride_c
            + yh1[:, None] * loss_stride_h
            + yw1[:, None] * loss_stride_w,
            mask=(common_mask & valid_yh1[:, None] & valid_yw1[:, None]),
            other=0.0,
        )

        weight11 = tl.load(
            weight_ptr
            + (
                ((w1 * 3 + w1) * COUT_PER_GROUP + offs_co[:, None])
                * CIN_PER_GROUP
            )
            + offs_ci[None, :],
            mask=weight_mask,
            other=0.0,
        )
        weight12 = tl.load(
            weight_ptr
            + (
                ((w1 * 3 + w2) * COUT_PER_GROUP + offs_co[:, None])
                * CIN_PER_GROUP
            )
            + offs_ci[None, :],
            mask=weight_mask,
            other=0.0,
        )
        weight10 = tl.load(
            weight_ptr
            + (
                ((w1 * 3 + w0) * COUT_PER_GROUP + offs_co[:, None])
                * CIN_PER_GROUP
            )
            + offs_ci[None, :],
            mask=weight_mask,
            other=0.0,
        )
        weight21 = tl.load(
            weight_ptr
            + (
                ((w2 * 3 + w1) * COUT_PER_GROUP + offs_co[:, None])
                * CIN_PER_GROUP
            )
            + offs_ci[None, :],
            mask=weight_mask,
            other=0.0,
        )
        weight01 = tl.load(
            weight_ptr
            + (
                ((w0 * 3 + w1) * COUT_PER_GROUP + offs_co[:, None])
                * CIN_PER_GROUP
            )
            + offs_ci[None, :],
            mask=weight_mask,
            other=0.0,
        )
        weight22 = tl.load(
            weight_ptr
            + (
                ((w2 * 3 + w2) * COUT_PER_GROUP + offs_co[:, None])
                * CIN_PER_GROUP
            )
            + offs_ci[None, :],
            mask=weight_mask,
            other=0.0,
        )
        weight20 = tl.load(
            weight_ptr
            + (
                ((w2 * 3 + w0) * COUT_PER_GROUP + offs_co[:, None])
                * CIN_PER_GROUP
            )
            + offs_ci[None, :],
            mask=weight_mask,
            other=0.0,
        )
        weight02 = tl.load(
            weight_ptr
            + (
                ((w0 * 3 + w2) * COUT_PER_GROUP + offs_co[:, None])
                * CIN_PER_GROUP
            )
            + offs_ci[None, :],
            mask=weight_mask,
            other=0.0,
        )
        weight00 = tl.load(
            weight_ptr
            + (
                ((w0 * 3 + w0) * COUT_PER_GROUP + offs_co[:, None])
                * CIN_PER_GROUP
            )
            + offs_ci[None, :],
            mask=weight_mask,
            other=0.0,
        )

        if INPUT_PRECISION == 1:
            if ROUND_TF32:
                loss00 = _round_fp32_to_tf32_rne(loss00)
                loss01 = _round_fp32_to_tf32_rne(loss01)
                loss10 = _round_fp32_to_tf32_rne(loss10)
                loss11 = _round_fp32_to_tf32_rne(loss11)
            acc00 = tl.dot(loss00, weight11, acc00, input_precision="tf32")
            acc01 = tl.dot(loss00, weight12, acc01, input_precision="tf32")
            acc01 = tl.dot(loss01, weight10, acc01, input_precision="tf32")
            acc10 = tl.dot(loss00, weight21, acc10, input_precision="tf32")
            acc10 = tl.dot(loss10, weight01, acc10, input_precision="tf32")
            acc11 = tl.dot(loss00, weight22, acc11, input_precision="tf32")
            acc11 = tl.dot(loss01, weight20, acc11, input_precision="tf32")
            acc11 = tl.dot(loss10, weight02, acc11, input_precision="tf32")
            acc11 = tl.dot(loss11, weight00, acc11, input_precision="tf32")
        else:
            acc00 = tl.dot(loss00, weight11, acc00, input_precision="ieee")
            acc01 = tl.dot(loss00, weight12, acc01, input_precision="ieee")
            acc01 = tl.dot(loss01, weight10, acc01, input_precision="ieee")
            acc10 = tl.dot(loss00, weight21, acc10, input_precision="ieee")
            acc10 = tl.dot(loss10, weight01, acc10, input_precision="ieee")
            acc11 = tl.dot(loss00, weight22, acc11, input_precision="ieee")
            acc11 = tl.dot(loss01, weight20, acc11, input_precision="ieee")
            acc11 = tl.dot(loss10, weight02, acc11, input_precision="ieee")
            acc11 = tl.dot(loss11, weight00, acc11, input_precision="ieee")

    out_base = (
        out_ptr
        + n_idx[:, None] * out_stride_n
        + offs_ci[None, :] * out_stride_c
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
def conv_dgrad3d_pad1_3x3_fp32_ci8_dot_kernel(
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
    """Specialized FP32 3x3x3 DGrad for C_OUT=16 and C_IN=8."""
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci = tl.arange(0, 16)
    offs_co = tl.arange(0, 16)
    mask_m = offs_m < M
    mask_ci = offs_ci < 8

    spatial_hw: tl.constexpr = XH * XW
    spatial: tl.constexpr = XD * spatial_hw
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
def conv_dgrad3d_packed_kernel(
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
    INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    """Packed KDH​W-by-OI implicit-GEMM 3D DGrad."""
    pid = tl.program_id(0)
    group = tl.program_id(1)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    spatial_hw: tl.constexpr = XH * XW
    spatial: tl.constexpr = XD * spatial_hw
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
                                ((group * KD + weight_d) * KH + weight_h) * KW
                                + weight_w
                            )
                            * COUT_PER_GROUP
                            + offs_co_rel[:, None]
                        )
                        * CIN_PER_GROUP
                        + offs_ci_rel[None, :],
                        mask=mask_co[:, None] & mask_ci[None, :],
                        other=0.0,
                    )
                    if INPUT_PRECISION == 1:
                        acc += tl.dot(
                            loss,
                            weight,
                            out_dtype=tl.float32,
                            input_precision="tf32",
                        )
                    else:
                        acc += tl.dot(
                            loss,
                            weight,
                            out_dtype=tl.float32,
                            input_precision="ieee",
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
        if INPUT_PRECISION == 1:
            accumulator += tl.dot(losses, inputs, input_precision="tf32")
        else:
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
def _conv2d_spatial_nchw_packed_khw_kernel(
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
    """NVIDIA TF32 implicit-GEMM with pre-packed [G, KH, KW, OC, IC]."""
    pid = tl.program_id(0)
    pid_bg = tl.program_id(1)

    batch_idx = pid_bg // GROUPS
    group_idx = pid_bg - batch_idx * GROUPS
    output_hw = OH * OW

    num_pid_m = tl.cdiv(output_hw, BLOCK_HW)
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
    mask_hw = offs_hw < output_hw
    mask_oc = offs_oc < COUT_PER_GROUP

    oh = offs_hw // OW
    ow = offs_hw - oh * OW
    x_batch_base = batch_idx * (C_IN * XH * XW)
    y_batch_base = batch_idx * (C_OUT * output_hw)
    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

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
                weight = tl.load(
                    w_ptrs,
                    mask=mask_oc[:, None] & mask_k[None, :],
                    other=0.0,
                )
                acc = tl.dot(weight, x, acc, input_precision="tf32")

    oc_global = group_idx * COUT_PER_GROUP + offs_oc
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[:, None]
    y_ptrs = (
        y_ptr
        + y_batch_base
        + oc_global[:, None] * output_hw
        + offs_hw[None, :]
    )
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_hw[None, :],
    )


@triton.jit
def _conv_fprop3d_ncdhw_kernel(
    x_ptr,
    w_ptr,
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
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_FRONT: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Hopper NCDHW implicit GEMM for the benchmark 3D FProp family."""
    pid = tl.program_id(0)
    output_dhw = OD * OH * OW
    input_hw = XH * XW
    input_cdhw = C_IN * XD * input_hw
    output_cdhw = C_OUT * output_dhw
    kernel_volume = KD * KH * KW
    reduction_size = C_IN * kernel_volume

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(C_OUT, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid - group_id * num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_oc = offs_oc < C_OUT

    batch = offs_m // output_dhw
    spatial = offs_m - batch * output_dhw
    od = spatial // (OH * OW)
    rem_hw = spatial - od * (OH * OW)
    oh = rem_hw // OW
    ow = rem_hw - oh * OW
    acc = tl.zeros((BLOCK_OC, BLOCK_M), dtype=tl.float32)

    for k_start in range(0, reduction_size, BLOCK_K):
        offs_k = k_start + offs_k_base
        mask_k = offs_k < reduction_size
        ic = offs_k // kernel_volume
        rem_kernel = offs_k - ic * kernel_volume
        kd = rem_kernel // (KH * KW)
        rem_kernel_hw = rem_kernel - kd * (KH * KW)
        kh = rem_kernel_hw // KW
        kw = rem_kernel_hw - kh * KW

        input_d = od[None, :] * STRIDE_D - PAD_FRONT + kd[:, None] * DIL_D
        input_h = oh[None, :] * STRIDE_H - PAD_TOP + kh[:, None] * DIL_H
        input_w = ow[None, :] * STRIDE_W - PAD_LEFT + kw[:, None] * DIL_W
        valid = (
            mask_k[:, None]
            & mask_m[None, :]
            & (input_d >= 0)
            & (input_d < XD)
            & (input_h >= 0)
            & (input_h < XH)
            & (input_w >= 0)
            & (input_w < XW)
        )
        x = tl.load(
            x_ptr
            + batch[None, :] * input_cdhw
            + ic[:, None] * (XD * input_hw)
            + input_d * input_hw
            + input_h * XW
            + input_w,
            mask=valid,
            other=0.0,
        )
        weight = tl.load(
            w_ptr + offs_oc[:, None] * reduction_size + offs_k[None, :],
            mask=mask_oc[:, None] & mask_k[None, :],
            other=0.0,
        )
        acc = tl.dot(weight, x, acc, input_precision="tf32")

    tl.store(
        y_ptr
        + batch[None, :] * output_cdhw
        + offs_oc[:, None] * output_dhw
        + spatial[None, :],
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_m[None, :],
    )


@triton.jit
def _conv_wgrad1d_3tap_nodiv_split_kernel(
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
    """H100 fixed-shape 3-tap split convolution kernel."""
    pid = tl.program_id(0)
    split = tl.program_id(1)
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
        partial_ptr + base,
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
def _conv_wgrad1d_col_direct_nodiv_kernel(
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
    """H100 fixed-shape one-launch CIK convolution kernel."""
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
    for n_idx in tl.range(0, BATCH_N):
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
        acc0 += tl.load(partial_ptr + base, mask=mask, other=0.0).to(
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
        out_base,
        acc0.to(out_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        out_base + out_stride_k,
        acc1.to(out_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        out_base + 2 * out_stride_k,
        acc2.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _conv_wgrad2d_1x1_direct_nodiv_kernel(
    image_ptr,
    loss_ptr,
    out_ptr,
    BATCH_N: tl.constexpr,
    HW: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
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
    for n_idx in tl.static_range(0, BATCH_N):
        for hw_start in tl.range(0, HW, BLOCK_M):
            hw = hw_start + tl.arange(0, BLOCK_M)
            mask_m = hw < HW
            safe_hw = tl.where(mask_m, hw, 0)
            loss = tl.load(
                loss_ptr
                + n_idx * loss_stride_n
                + co[:, None] * loss_stride_c
                + safe_hw[None, :],
                mask=mask_co[:, None] & mask_m[None, :],
                other=0.0,
            )
            image = tl.load(
                image_ptr
                + n_idx * image_stride_n
                + ci[None, :] * image_stride_c
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
        out_ptr
        + co[:, None] * out_stride_o
        + offs_ci_rel[None, :] * out_stride_i,
        acc.to(out_ptr.dtype.element_ty),
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
def _conv_wgrad2d_batched_tma_kernel(
    loss_ptr,
    columns_ptr,
    partial_ptr,
    BATCH_N: tl.constexpr,
    M: tl.constexpr,
    PADDED_M: tl.constexpr,
    C_OUT: tl.constexpr,
    CIK: tl.constexpr,
    SPLITS_PER_N: tl.constexpr,
    INPUT_IS_FLOAT32: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Batched WGrad GEMM over contiguous image rows or materialized patches."""
    loss_desc = tl.make_tensor_descriptor(
        loss_ptr,
        shape=[BATCH_N * C_OUT, M],
        strides=[M, 1],
        block_shape=[BLOCK_CO, BLOCK_M],
    )
    columns_desc = tl.make_tensor_descriptor(
        columns_ptr,
        shape=[BATCH_N * CIK, M],
        strides=[PADDED_M, 1],
        block_shape=[BLOCK_CI, BLOCK_M],
    )
    tile = tl.program_id(0)
    split_in_n = tl.program_id(1)
    batch = tl.program_id(2)
    split = batch * SPLITS_PER_N + split_in_n
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    if SPLITS_PER_N == 3:
        boundary1: tl.constexpr = num_m_blocks // 3
        boundary2: tl.constexpr = (2 * num_m_blocks) // 3
        first_m_block = tl.where(
            split_in_n == 0,
            0,
            tl.where(split_in_n == 1, boundary1, boundary2),
        )
        last_m_block = tl.where(
            split_in_n == 0,
            boundary1,
            tl.where(split_in_n == 1, boundary2, num_m_blocks),
        )
    else:
        first_m_block = (split_in_n * num_m_blocks) // SPLITS_PER_N
        last_m_block = ((split_in_n + 1) * num_m_blocks) // SPLITS_PER_N
    num_ci_blocks = tl.cdiv(CIK, BLOCK_CI)
    tile_co = tile // num_ci_blocks
    tile_ci = tile - tile_co * num_ci_blocks
    co_begin = tile_co * BLOCK_CO
    ci_begin = tile_ci * BLOCK_CI

    accumulator = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for m_block in tl.range(first_m_block, last_m_block):
        m_start = m_block * BLOCK_M
        loss = loss_desc.load([batch * C_OUT + co_begin, m_start])
        columns = columns_desc.load([batch * CIK + ci_begin, m_start])
        accumulator = tl.dot(
            loss,
            tl.trans(columns),
            acc=accumulator,
            out_dtype=tl.float32,
            input_precision="tf32" if INPUT_IS_FLOAT32 else "ieee",
        )
    co = co_begin + tl.arange(0, BLOCK_CO)
    ci = ci_begin + tl.arange(0, BLOCK_CI)
    partial_offsets = (
        partial_ptr + split * C_OUT * CIK + co[:, None] * CIK + ci[None, :]
    )
    partial = accumulator.to(partial_ptr.dtype.element_ty)
    output_mask = (co[:, None] < C_OUT) & (ci[None, :] < CIK)
    if C_OUT % BLOCK_CO == 0 and CIK % BLOCK_CI == 0:
        tl.store(partial_offsets, partial)
    else:
        tl.store(partial_offsets, partial, mask=output_mask)


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
    tl.store(partial_ptr + base, acc0, mask=mask)
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
        image0 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + offs_ci[None, :] * image_stride_c
            + safe_h[:, None] * image_stride_h
            + safe_w0[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid0[:, None],
            other=0.0,
        )
        image1 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + offs_ci[None, :] * image_stride_c
            + safe_h[:, None] * image_stride_h
            + safe_w1[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid1[:, None],
            other=0.0,
        )
        image2 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + offs_ci[None, :] * image_stride_c
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
    mask = mask_co[:, None] & mask_ci[None, :]
    base = (
        out_ptr
        + offs_co[:, None] * out_stride_o
        + offs_ci[None, :] * out_stride_i
        + kh * out_stride_h
    )
    tl.atomic_add(base, acc0, sem="relaxed", mask=mask)
    tl.atomic_add(base + out_stride_w, acc1, sem="relaxed", mask=mask)
    tl.atomic_add(base + 2 * out_stride_w, acc2, sem="relaxed", mask=mask)


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
    tl.store(partial_ptr + base, acc0, mask=mask)
    tl.store(partial_ptr + base + 1, acc1, mask=mask)
    tl.store(partial_ptr + base + 2, acc2, mask=mask)


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
def _conv_wgrad2d_split_vector_reduce_kernel(
    partial_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Reduce contiguous WGrad partials in parallel across split tiles."""
    offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    output_mask = offsets < TOTAL
    accumulator0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator3 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator4 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator5 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator6 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator7 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for split in tl.static_range(0, NUM_SPLITS, 8):
        if TOTAL % BLOCK_M == 0:
            value0 = tl.load(partial_ptr + split * TOTAL + offsets)
            value1 = tl.load(partial_ptr + (split + 1) * TOTAL + offsets)
            value2 = tl.load(partial_ptr + (split + 2) * TOTAL + offsets)
            value3 = tl.load(partial_ptr + (split + 3) * TOTAL + offsets)
            value4 = tl.load(partial_ptr + (split + 4) * TOTAL + offsets)
            value5 = tl.load(partial_ptr + (split + 5) * TOTAL + offsets)
            value6 = tl.load(partial_ptr + (split + 6) * TOTAL + offsets)
            value7 = tl.load(partial_ptr + (split + 7) * TOTAL + offsets)
        else:
            value0 = tl.load(
                partial_ptr + split * TOTAL + offsets,
                mask=output_mask,
                other=0.0,
            )
            value1 = tl.load(
                partial_ptr + (split + 1) * TOTAL + offsets,
                mask=output_mask,
                other=0.0,
            )
            value2 = tl.load(
                partial_ptr + (split + 2) * TOTAL + offsets,
                mask=output_mask,
                other=0.0,
            )
            value3 = tl.load(
                partial_ptr + (split + 3) * TOTAL + offsets,
                mask=output_mask,
                other=0.0,
            )
            value4 = tl.load(
                partial_ptr + (split + 4) * TOTAL + offsets,
                mask=output_mask,
                other=0.0,
            )
            value5 = tl.load(
                partial_ptr + (split + 5) * TOTAL + offsets,
                mask=output_mask,
                other=0.0,
            )
            value6 = tl.load(
                partial_ptr + (split + 6) * TOTAL + offsets,
                mask=output_mask,
                other=0.0,
            )
            value7 = tl.load(
                partial_ptr + (split + 7) * TOTAL + offsets,
                mask=output_mask,
                other=0.0,
            )
        accumulator0 += value0.to(tl.float32)
        accumulator1 += value1.to(tl.float32)
        accumulator2 += value2.to(tl.float32)
        accumulator3 += value3.to(tl.float32)
        accumulator4 += value4.to(tl.float32)
        accumulator5 += value5.to(tl.float32)
        accumulator6 += value6.to(tl.float32)
        accumulator7 += value7.to(tl.float32)
    accumulator = (
        (accumulator0 + accumulator1)
        + (accumulator2 + accumulator3)
        + (accumulator4 + accumulator5)
        + (accumulator6 + accumulator7)
    )
    output = accumulator.to(out_ptr.dtype.element_ty)
    if TOTAL % BLOCK_M == 0:
        tl.store(out_ptr + offsets, output)
    else:
        tl.store(out_ptr + offsets, output, mask=output_mask)


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
def _conv_wgrad2d_p5_pack_image_kernel(
    image_ptr,
    packed_ptr,
    CIN_PER_GROUP: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_h: tl.constexpr,
    image_stride_w: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    cik = N

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
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
def _conv_wgrad2d_p5_mm_kernel(
    loss_ptr,
    packed_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
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
