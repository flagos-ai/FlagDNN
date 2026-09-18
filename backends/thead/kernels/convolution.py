# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""THead-only convolution entry points with exact Graph boundary ABIs."""

import triton
import triton.language as tl


@triton.jit
def _stride_offset(index, stride: tl.constexpr, extent: tl.constexpr):
    # Widen before multiplication: converting an overflowing int32 product
    # at the pointer addition is too late. Small layouts keep their codegen.
    wide: tl.constexpr = (extent - 1) * stride > 2147483647
    return index.to(tl.int64 if wide else index.dtype) * stride


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

    wide_coordinates: tl.constexpr = (
        OD * STRIDE_D + KD * DIL_D + PAD_FRONT > 2147483647
        or OH * STRIDE_H + KH * DIL_H + PAD_TOP > 2147483647
        or OW * STRIDE_W + KW * DIL_W + PAD_LEFT > 2147483647
    )
    tile = tl.program_id(0)
    group = tl.program_id(1).to(tl.int64)
    tiles_oc = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    tile_m = tile // tiles_oc
    tile_oc = tile % tiles_oc
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if wide_coordinates:
        rows = rows.to(tl.int64)
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
    if wide_coordinates:
        reduction_base = reduction_base.to(tl.int64)
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
        reduction_active = reduction < reduction_extent
        inputs = tl.load(
            x_ptr
            + _stride_offset(batch[:, None], X_STRIDE_N, M // (OD * OH * OW))
            + (group * CIN_PER_GROUP + input_channel[None, :]) * X_STRIDE_C
            + _stride_offset(input_d, X_STRIDE_D, XD)
            + _stride_offset(input_h, X_STRIDE_H, XH)
            + _stride_offset(input_w, X_STRIDE_W, XW),
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
            + (group * COUT_PER_GROUP + output_channels[:, None]) * W_STRIDE_K
            + _stride_offset(input_channel[None, :], W_STRIDE_C, CIN_PER_GROUP)
            + _stride_offset(kernel_d[None, :], W_STRIDE_D, KD)
            + _stride_offset(kernel_h[None, :], W_STRIDE_H, KH)
            + _stride_offset(kernel_w[None, :], W_STRIDE_W, KW),
            mask=(output_channels[:, None] < COUT_PER_GROUP)
            & reduction_active[None, :],
            other=0.0,
        )
        accumulator += tl.dot(
            inputs,
            tl.trans(weights),
            input_precision="tf32" if INPUT_PRECISION == 2 else "ieee",
        )

    tl.store(
        y_ptr
        + _stride_offset(batch[:, None], Y_STRIDE_N, M // (OD * OH * OW))
        + (group * COUT_PER_GROUP + output_channels[None, :]) * Y_STRIDE_C
        + _stride_offset(output_d[:, None], Y_STRIDE_D, OD)
        + _stride_offset(output_h[:, None], Y_STRIDE_H, OH)
        + _stride_offset(output_w[:, None], Y_STRIDE_W, OW),
        accumulator.to(y_ptr.dtype.element_ty),
        mask=(rows[:, None] < M) & (output_channels[None, :] < COUT_PER_GROUP),
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
    wide_coordinates: tl.constexpr = (
        OH * STRIDE_H + KH * DIL_H + PAD_TOP > 2147483647
        or OW * STRIDE_W + KW * DIL_W + PAD_LEFT > 2147483647
    )
    tile = tl.program_id(0)
    batch_group = tl.program_id(1).to(tl.int64)
    batch = batch_group // GROUPS
    group = batch_group % GROUPS
    tiles_oc = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    tile_hw = tile // tiles_oc
    tile_oc = tile % tiles_oc
    output_hw = tile_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    output_channels = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    if wide_coordinates:
        output_hw = output_hw.to(tl.int64)
    output_h = output_hw // OW
    output_w = output_hw % OW
    reduction_base = tl.arange(0, BLOCK_K)
    if wide_coordinates:
        reduction_base = reduction_base.to(tl.int64)
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
            + _stride_offset(input_h, X_STRIDE_H, XH)
            + _stride_offset(input_w, X_STRIDE_W, XW),
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
            + _stride_offset(input_channel[None, :], W_STRIDE_C, CIN_PER_GROUP)
            + _stride_offset(kernel_h[None, :], W_STRIDE_R, KH)
            + _stride_offset(kernel_w[None, :], W_STRIDE_S, KW),
            mask=(output_channels[:, None] < COUT_PER_GROUP)
            & (reduction[None, :] < reduction_extent),
            other=0.0,
        )
        accumulator += tl.dot(
            input_values,
            tl.trans(weights),
            input_precision="tf32" if INPUT_PRECISION == 2 else "ieee",
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
        + _stride_offset(output_h[:, None], Y_STRIDE_H, OH)
        + _stride_offset(output_w[:, None], Y_STRIDE_W, OW),
        activated.to(y_ptr.dtype.element_ty),
        mask=(output_hw[:, None] < OH * OW)
        & (output_channels[None, :] < COUT_PER_GROUP),
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
    wide_coordinates: tl.constexpr = (
        OD * STRIDE_D + KD * DIL_D + PAD_FRONT > 2147483647
        or OH * STRIDE_H + KH * DIL_H + PAD_TOP > 2147483647
        or OW * STRIDE_W + KW * DIL_W + PAD_LEFT > 2147483647
    )
    # One weight per CTA avoids mostly masked matrix tiles for small CIN.
    # Each CTA owns its result; no atomics or cross-launch initialization.
    if CIN_PER_GROUP <= 8 and M >= 4096 and BLOCK_M >= 256:
        weight = tl.program_id(0).to(tl.int64)
        group = tl.program_id(1).to(tl.int64)
        kernel_w = weight % KW
        kernel_h = (weight // KW) % KH
        kernel_d = (weight // (KW * KH)) % KD
        input_channel = (weight // (KW * KH * KD)) % CIN_PER_GROUP
        output_channel = weight // (KW * KH * KD * CIN_PER_GROUP)
        effective_d = KD - 1 - kernel_d if FLIP_FILTER else kernel_d
        effective_h = KH - 1 - kernel_h if FLIP_FILTER else kernel_h
        effective_w = KW - 1 - kernel_w if FLIP_FILTER else kernel_w
        offsets = tl.arange(0, BLOCK_M)
        accumulator = tl.zeros((BLOCK_M,), tl.float32)
        for start in range(0, M, BLOCK_M):
            rows = start + offsets
            if wide_coordinates:
                rows = rows.to(tl.int64)
            batch = rows // (OD * OH * OW)
            od = rows // (OH * OW) % OD
            oh = rows // OW % OH
            ow = rows % OW
            xd = od * STRIDE_D - PAD_FRONT + effective_d * DIL_D
            xh = oh * STRIDE_H - PAD_TOP + effective_h * DIL_H
            xw = ow * STRIDE_W - PAD_LEFT + effective_w * DIL_W
            active = (
                (rows < M)
                & (xd >= 0)
                & (xd < XD)
                & (xh >= 0)
                & (xh < XH)
                & (xw >= 0)
                & (xw < XW)
            )
            losses = tl.load(
                dy_ptr
                + _stride_offset(batch, DY_STRIDE_N, M // (OD * OH * OW))
                + (group * COUT_PER_GROUP + output_channel) * DY_STRIDE_C
                + _stride_offset(od, DY_STRIDE_D, OD)
                + _stride_offset(oh, DY_STRIDE_H, OH)
                + _stride_offset(ow, DY_STRIDE_W, OW),
                active,
                0,
            ).to(tl.float32)
            inputs = tl.load(
                x_ptr
                + _stride_offset(batch, X_STRIDE_N, M // (OD * OH * OW))
                + (group * CIN_PER_GROUP + input_channel) * X_STRIDE_C
                + _stride_offset(xd, X_STRIDE_D, XD)
                + _stride_offset(xh, X_STRIDE_H, XH)
                + _stride_offset(xw, X_STRIDE_W, XW),
                active,
                0,
            ).to(tl.float32)
            accumulator += losses * inputs
        value = tl.sum(accumulator, 0)
        tl.store(
            dw_ptr
            + (group * COUT_PER_GROUP + output_channel) * W_STRIDE_K
            + _stride_offset(input_channel, W_STRIDE_C, CIN_PER_GROUP)
            + _stride_offset(kernel_d, W_STRIDE_D, KD)
            + _stride_offset(kernel_h, W_STRIDE_H, KH)
            + _stride_offset(kernel_w, W_STRIDE_W, KW),
            value,
        )
    else:
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
            if wide_coordinates:
                rows = rows.to(tl.int64)
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
                + (group * COUT_PER_GROUP + output_channels[:, None])
                * DY_STRIDE_C
                + _stride_offset(
                    batch[None, :], DY_STRIDE_N, M // (OD * OH * OW)
                )
                + _stride_offset(output_d[None, :], DY_STRIDE_D, OD)
                + _stride_offset(output_h[None, :], DY_STRIDE_H, OH)
                + _stride_offset(output_w[None, :], DY_STRIDE_W, OW),
                mask=(output_channels[:, None] < COUT_PER_GROUP)
                & active_rows[None, :],
                other=0.0,
            )
            inputs = tl.load(
                x_ptr
                + _stride_offset(
                    batch[:, None], X_STRIDE_N, M // (OD * OH * OW)
                )
                + (group * CIN_PER_GROUP + input_channels[None, :])
                * X_STRIDE_C
                + _stride_offset(input_d[:, None], X_STRIDE_D, XD)
                + _stride_offset(input_h[:, None], X_STRIDE_H, XH)
                + _stride_offset(input_w[:, None], X_STRIDE_W, XW),
                mask=active_rows[:, None]
                & (input_channels[None, :] < CIN_PER_GROUP),
                other=0.0,
            )
            accumulator += tl.dot(
                losses,
                inputs,
                input_precision="tf32" if INPUT_PRECISION == 2 else "ieee",
            )

        tl.store(
            dw_ptr
            + (group * COUT_PER_GROUP + output_channels[:, None]) * W_STRIDE_K
            + _stride_offset(
                input_channels[None, :], W_STRIDE_C, CIN_PER_GROUP
            )
            + _stride_offset(kernel_d, W_STRIDE_D, KD)
            + _stride_offset(kernel_h, W_STRIDE_H, KH)
            + _stride_offset(kernel_w, W_STRIDE_W, KW),
            accumulator.to(dw_ptr.dtype.element_ty),
            mask=(output_channels[:, None] < COUT_PER_GROUP)
            & (input_channels[None, :] < CIN_PER_GROUP),
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
    wide_coordinates: tl.constexpr = (
        XD + PAD_FRONT + KD * DIL_D > 2147483647
        or XH + PAD_TOP + KH * DIL_H > 2147483647
        or XW + PAD_LEFT + KW * DIL_W > 2147483647
    )
    # Four spatial parity groups omit invalid stride-two filter terms.
    # Small CIN uses only real channels instead of mostly masked matrix tiles.
    if (
        CIN_PER_GROUP <= 8
        and COUT_PER_GROUP >= 16
        and M >= 4096
        and BLOCK_M == 128
        and BLOCK_K == 16
        and XD == 1
        and OD == 1
        and KD == 1
        and KH == 3
        and KW == 3
        and STRIDE_H == 2
        and STRIDE_W == 2
        and DIL_H == 1
        and DIL_W == 1
        and XH % 2 == 0
        and XW % 2 == 0
        and X_STRIDE_N == CIN_PER_GROUP * XH * XW
        and X_STRIDE_C == XH * XW
        and X_STRIDE_H == XW
        and X_STRIDE_W == 1
        and DY_STRIDE_N == COUT_PER_GROUP * OH * OW
        and DY_STRIDE_C == OH * OW
        and DY_STRIDE_H == OW
        and DY_STRIDE_W == 1
        and W_STRIDE_K == CIN_PER_GROUP * 9
        and W_STRIDE_C == 9
        and W_STRIDE_H == 3
        and W_STRIDE_W == 1
    ):
        positions = (tl.program_id(0) // 4) * BLOCK_M + tl.arange(0, BLOCK_M)
        phase = tl.program_id(0) % 4
        ci = tl.program_id(1).to(tl.int64)
        group = tl.program_id(2).to(tl.int64)
        batch = positions // (XH * XW // 4)
        if wide_coordinates:
            positions = positions.to(tl.int64)
        xd = tl.full((BLOCK_M,), 0, tl.int64 if wide_coordinates else tl.int32)
        xh = 2 * (positions // (XW // 2) % (XH // 2)) + phase // 2
        xw = 2 * (positions % (XW // 2)) + phase % 2
        rows = batch * XH * XW + xh * XW + xw
        if wide_coordinates:
            rows = rows.to(tl.int64)
        co_offsets = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_K, BLOCK_M), tl.float32)
        for spatial in tl.static_range(KD * KH * KW):
            kd = spatial // (KH * KW)
            kh = spatial // KW % KH
            kw = spatial % KW
            ed = KD - 1 - kd if FLIP_FILTER else kd
            eh = KH - 1 - kh if FLIP_FILTER else kh
            ew = KW - 1 - kw if FLIP_FILTER else kw
            if (phase // 2 + PAD_TOP - eh * DIL_H) % 2 == 0 and (
                phase % 2 + PAD_LEFT - ew * DIL_W
            ) % 2 == 0:
                od_num = xd + PAD_FRONT - ed * DIL_D
                oh_num = xh + PAD_TOP - eh * DIL_H
                ow_num = xw + PAD_LEFT - ew * DIL_W
                od = od_num // STRIDE_D
                oh = oh_num // STRIDE_H
                ow = ow_num // STRIDE_W
                active = (
                    (rows < M)
                    & (od_num % STRIDE_D == 0)
                    & (oh_num % STRIDE_H == 0)
                    & (ow_num % STRIDE_W == 0)
                    & (od >= 0)
                    & (od < OD)
                    & (oh >= 0)
                    & (oh < OH)
                    & (ow >= 0)
                    & (ow < OW)
                )
                for start in range(0, COUT_PER_GROUP, BLOCK_K):
                    co = start + co_offsets
                    loss = tl.load(
                        dy_ptr
                        + _stride_offset(
                            batch[None, :], DY_STRIDE_N, M // (XD * XH * XW)
                        )
                        + (group * COUT_PER_GROUP + co[:, None]) * DY_STRIDE_C
                        + _stride_offset(od[None, :], DY_STRIDE_D, OD)
                        + _stride_offset(oh[None, :], DY_STRIDE_H, OH)
                        + _stride_offset(ow[None, :], DY_STRIDE_W, OW),
                        (co[:, None] < COUT_PER_GROUP) & active[None, :],
                        0,
                    ).to(tl.float32)
                    weight = tl.load(
                        w_ptr
                        + (group * COUT_PER_GROUP + co) * W_STRIDE_K
                        + _stride_offset(ci, W_STRIDE_C, CIN_PER_GROUP)
                        + _stride_offset(kd, W_STRIDE_D, KD)
                        + _stride_offset(kh, W_STRIDE_H, KH)
                        + _stride_offset(kw, W_STRIDE_W, KW),
                        co < COUT_PER_GROUP,
                        0,
                    ).to(tl.float32)
                    acc += loss * weight[:, None]
        result = tl.sum(acc, 0)
        tl.store(
            dx_ptr
            + _stride_offset(batch, X_STRIDE_N, M // (XD * XH * XW))
            + (group * CIN_PER_GROUP + ci) * X_STRIDE_C
            + _stride_offset(xd, X_STRIDE_D, XD)
            + _stride_offset(xh, X_STRIDE_H, XH)
            + _stride_offset(xw, X_STRIDE_W, XW),
            result,
            rows < M,
        )
    else:
        tile = tl.program_id(0)
        group = tl.program_id(1).to(tl.int64)
        tiles_ci = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
        tile_m = tile // tiles_ci
        tile_ci = tile % tiles_ci
        rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        if wide_coordinates:
            rows = rows.to(tl.int64)
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
        if wide_coordinates:
            reduction_base = reduction_base.to(tl.int64)
        accumulator = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

        for start in range(0, reduction_extent, BLOCK_K):
            reduction = start + reduction_base
            output_channel = reduction // kernel_volume
            kernel_spatial = reduction % kernel_volume
            kernel_d = kernel_spatial // (KH * KW)
            kernel_hw = kernel_spatial % (KH * KW)
            kernel_h = kernel_hw // KW
            kernel_w = kernel_hw % KW
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
                + _stride_offset(
                    batch[:, None], DY_STRIDE_N, M // (XD * XH * XW)
                )
                + (group * COUT_PER_GROUP + output_channel[None, :])
                * DY_STRIDE_C
                + _stride_offset(output_d, DY_STRIDE_D, OD)
                + _stride_offset(output_h, DY_STRIDE_H, OH)
                + _stride_offset(output_w, DY_STRIDE_W, OW),
                mask=valid,
                other=0.0,
            )
            weight_d = KD - 1 - kernel_d if FLIP_FILTER else kernel_d
            weight_h = KH - 1 - kernel_h if FLIP_FILTER else kernel_h
            weight_w = KW - 1 - kernel_w if FLIP_FILTER else kernel_w
            weights = tl.load(
                w_ptr
                + (group * COUT_PER_GROUP + output_channel[:, None])
                * W_STRIDE_K
                + _stride_offset(
                    input_channels[None, :], W_STRIDE_C, CIN_PER_GROUP
                )
                + _stride_offset(weight_d[:, None], W_STRIDE_D, KD)
                + _stride_offset(weight_h[:, None], W_STRIDE_H, KH)
                + _stride_offset(weight_w[:, None], W_STRIDE_W, KW),
                mask=(reduction[:, None] < reduction_extent)
                & (input_channels[None, :] < CIN_PER_GROUP),
                other=0.0,
            )
            accumulator += tl.dot(
                losses,
                weights,
                input_precision="tf32" if INPUT_PRECISION == 2 else "ieee",
            )

        tl.store(
            dx_ptr
            + _stride_offset(batch[:, None], X_STRIDE_N, M // (XD * XH * XW))
            + (group * CIN_PER_GROUP + input_channels[None, :]) * X_STRIDE_C
            + _stride_offset(input_d[:, None], X_STRIDE_D, XD)
            + _stride_offset(input_h[:, None], X_STRIDE_H, XH)
            + _stride_offset(input_w[:, None], X_STRIDE_W, XW),
            accumulator.to(dx_ptr.dtype.element_ty),
            mask=(rows[:, None] < M)
            & (input_channels[None, :] < CIN_PER_GROUP),
        )
