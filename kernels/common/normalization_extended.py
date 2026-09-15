# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Normalization over arbitrary axes, with deterministic affine gradients."""
import triton
import triton.language as tl


@triton.jit
def _coordinates(group, reduced, dims: tl.constexpr, mask: tl.constexpr):
    reduced = reduced.to(tl.int64)
    # Keep block-shaped addresses even when the reduction contains one element
    # and all logical coordinates originate from the scalar group index.
    group = group.to(tl.int64) + tl.zeros(reduced.shape, tl.int64)
    if mask & 128:
        c7 = reduced % dims[7]
        reduced //= dims[7]
    else:
        c7 = group % dims[7]
        group //= dims[7]
    if mask & 64:
        c6 = reduced % dims[6]
        reduced //= dims[6]
    else:
        c6 = group % dims[6]
        group //= dims[6]
    if mask & 32:
        c5 = reduced % dims[5]
        reduced //= dims[5]
    else:
        c5 = group % dims[5]
        group //= dims[5]
    if mask & 16:
        c4 = reduced % dims[4]
        reduced //= dims[4]
    else:
        c4 = group % dims[4]
        group //= dims[4]
    if mask & 8:
        c3 = reduced % dims[3]
        reduced //= dims[3]
    else:
        c3 = group % dims[3]
        group //= dims[3]
    if mask & 4:
        c2 = reduced % dims[2]
        reduced //= dims[2]
    else:
        c2 = group % dims[2]
        group //= dims[2]
    if mask & 2:
        c1 = reduced % dims[1]
        reduced //= dims[1]
    else:
        c1 = group % dims[1]
        group //= dims[1]
    if mask & 1:
        c0 = reduced % dims[0]
        reduced //= dims[0]
    else:
        c0 = group % dims[0]
        group //= dims[0]
    return (c0, c1, c2, c3, c4, c5, c6, c7)


@triton.jit
def _offset(coords, strides: tl.constexpr):
    result = coords[0] * strides[0]
    for axis in tl.static_range(1, 8):
        result += coords[axis] * strides[axis]
    return result


@triton.jit
def extended_normalization_forward(
    x_ptr,
    scale_ptr,
    bias_ptr,
    y_ptr,
    mean_ptr,
    inv_ptr,
    GROUPS: tl.constexpr,
    REDUCTION: tl.constexpr,
    AXES: tl.constexpr,
    PARAM_AXES: tl.constexpr,
    PARAMETERS: tl.constexpr,
    AFFINE_REDUCTION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPSILON: tl.constexpr,
    DIM_0: tl.constexpr,
    DIM_1: tl.constexpr,
    DIM_2: tl.constexpr,
    DIM_3: tl.constexpr,
    DIM_4: tl.constexpr,
    DIM_5: tl.constexpr,
    DIM_6: tl.constexpr,
    DIM_7: tl.constexpr,
    X_STRIDE_0: tl.constexpr,
    X_STRIDE_1: tl.constexpr,
    X_STRIDE_2: tl.constexpr,
    X_STRIDE_3: tl.constexpr,
    X_STRIDE_4: tl.constexpr,
    X_STRIDE_5: tl.constexpr,
    X_STRIDE_6: tl.constexpr,
    X_STRIDE_7: tl.constexpr,
    SCALE_STRIDE_0: tl.constexpr,
    SCALE_STRIDE_1: tl.constexpr,
    SCALE_STRIDE_2: tl.constexpr,
    SCALE_STRIDE_3: tl.constexpr,
    SCALE_STRIDE_4: tl.constexpr,
    SCALE_STRIDE_5: tl.constexpr,
    SCALE_STRIDE_6: tl.constexpr,
    SCALE_STRIDE_7: tl.constexpr,
    BIAS_STRIDE_0: tl.constexpr,
    BIAS_STRIDE_1: tl.constexpr,
    BIAS_STRIDE_2: tl.constexpr,
    BIAS_STRIDE_3: tl.constexpr,
    BIAS_STRIDE_4: tl.constexpr,
    BIAS_STRIDE_5: tl.constexpr,
    BIAS_STRIDE_6: tl.constexpr,
    BIAS_STRIDE_7: tl.constexpr,
    Y_STRIDE_0: tl.constexpr,
    Y_STRIDE_1: tl.constexpr,
    Y_STRIDE_2: tl.constexpr,
    Y_STRIDE_3: tl.constexpr,
    Y_STRIDE_4: tl.constexpr,
    Y_STRIDE_5: tl.constexpr,
    Y_STRIDE_6: tl.constexpr,
    Y_STRIDE_7: tl.constexpr,
    MEAN_STRIDE_0: tl.constexpr,
    MEAN_STRIDE_1: tl.constexpr,
    MEAN_STRIDE_2: tl.constexpr,
    MEAN_STRIDE_3: tl.constexpr,
    MEAN_STRIDE_4: tl.constexpr,
    MEAN_STRIDE_5: tl.constexpr,
    MEAN_STRIDE_6: tl.constexpr,
    MEAN_STRIDE_7: tl.constexpr,
    INV_STRIDE_0: tl.constexpr,
    INV_STRIDE_1: tl.constexpr,
    INV_STRIDE_2: tl.constexpr,
    INV_STRIDE_3: tl.constexpr,
    INV_STRIDE_4: tl.constexpr,
    INV_STRIDE_5: tl.constexpr,
    INV_STRIDE_6: tl.constexpr,
    INV_STRIDE_7: tl.constexpr,
):
    dims: tl.constexpr = (
        DIM_0,
        DIM_1,
        DIM_2,
        DIM_3,
        DIM_4,
        DIM_5,
        DIM_6,
        DIM_7,
    )
    x_strides: tl.constexpr = (
        X_STRIDE_0,
        X_STRIDE_1,
        X_STRIDE_2,
        X_STRIDE_3,
        X_STRIDE_4,
        X_STRIDE_5,
        X_STRIDE_6,
        X_STRIDE_7,
    )
    scale_strides: tl.constexpr = (
        SCALE_STRIDE_0,
        SCALE_STRIDE_1,
        SCALE_STRIDE_2,
        SCALE_STRIDE_3,
        SCALE_STRIDE_4,
        SCALE_STRIDE_5,
        SCALE_STRIDE_6,
        SCALE_STRIDE_7,
    )
    bias_strides: tl.constexpr = (
        BIAS_STRIDE_0,
        BIAS_STRIDE_1,
        BIAS_STRIDE_2,
        BIAS_STRIDE_3,
        BIAS_STRIDE_4,
        BIAS_STRIDE_5,
        BIAS_STRIDE_6,
        BIAS_STRIDE_7,
    )
    y_strides: tl.constexpr = (
        Y_STRIDE_0,
        Y_STRIDE_1,
        Y_STRIDE_2,
        Y_STRIDE_3,
        Y_STRIDE_4,
        Y_STRIDE_5,
        Y_STRIDE_6,
        Y_STRIDE_7,
    )
    mean_strides: tl.constexpr = (
        MEAN_STRIDE_0,
        MEAN_STRIDE_1,
        MEAN_STRIDE_2,
        MEAN_STRIDE_3,
        MEAN_STRIDE_4,
        MEAN_STRIDE_5,
        MEAN_STRIDE_6,
        MEAN_STRIDE_7,
    )
    inv_strides: tl.constexpr = (
        INV_STRIDE_0,
        INV_STRIDE_1,
        INV_STRIDE_2,
        INV_STRIDE_3,
        INV_STRIDE_4,
        INV_STRIDE_5,
        INV_STRIDE_6,
        INV_STRIDE_7,
    )
    group = tl.program_id(0).to(tl.int64)
    lanes = tl.arange(0, BLOCK_SIZE)
    partial = tl.zeros((BLOCK_SIZE,), tl.float32)
    for start in range(0, REDUCTION, BLOCK_SIZE):
        reduced = start + lanes
        coords = _coordinates(group, reduced, dims, AXES)
        x = tl.load(
            x_ptr + _offset(coords, x_strides), reduced < REDUCTION, other=0
        ).to(tl.float32)
        partial += x
    mean = tl.sum(partial, 0) / REDUCTION
    variance = tl.zeros((BLOCK_SIZE,), tl.float32)
    for start in range(0, REDUCTION, BLOCK_SIZE):
        reduced = start + lanes
        coords = _coordinates(group, reduced, dims, AXES)
        x = tl.load(
            x_ptr + _offset(coords, x_strides), reduced < REDUCTION, other=0
        ).to(tl.float32)
        centered = x - mean
        variance += tl.where(reduced < REDUCTION, centered * centered, 0.0)
    inverse = tl.rsqrt(tl.sum(variance, 0) / REDUCTION + EPSILON)
    statistic_coords = _coordinates(
        group, tl.full((), 0, tl.int64), dims, AXES
    )
    tl.store(mean_ptr + _offset(statistic_coords, mean_strides), mean)
    tl.store(inv_ptr + _offset(statistic_coords, inv_strides), inverse)
    for start in range(0, REDUCTION, BLOCK_SIZE):
        reduced = start + lanes
        coords = _coordinates(group, reduced, dims, AXES)
        valid = reduced < REDUCTION
        x = tl.load(x_ptr + _offset(coords, x_strides), valid, other=0).to(
            tl.float32
        )
        scale = tl.load(
            scale_ptr + _offset(coords, scale_strides), valid, other=0
        ).to(tl.float32)
        bias = tl.load(
            bias_ptr + _offset(coords, bias_strides), valid, other=0
        ).to(tl.float32)
        y = (x - mean) * inverse * scale + bias
        tl.store(y_ptr + _offset(coords, y_strides), y, valid)


@triton.jit
def extended_normalization_backward(
    dy_ptr,
    x_ptr,
    scale_ptr,
    mean_ptr,
    inv_ptr,
    dx_ptr,
    dscale_ptr,
    dbias_ptr,
    GROUPS: tl.constexpr,
    REDUCTION: tl.constexpr,
    AXES: tl.constexpr,
    PARAM_AXES: tl.constexpr,
    PARAMETERS: tl.constexpr,
    AFFINE_REDUCTION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RMS: tl.constexpr,
    DIM_0: tl.constexpr,
    DIM_1: tl.constexpr,
    DIM_2: tl.constexpr,
    DIM_3: tl.constexpr,
    DIM_4: tl.constexpr,
    DIM_5: tl.constexpr,
    DIM_6: tl.constexpr,
    DIM_7: tl.constexpr,
    DY_STRIDE_0: tl.constexpr,
    DY_STRIDE_1: tl.constexpr,
    DY_STRIDE_2: tl.constexpr,
    DY_STRIDE_3: tl.constexpr,
    DY_STRIDE_4: tl.constexpr,
    DY_STRIDE_5: tl.constexpr,
    DY_STRIDE_6: tl.constexpr,
    DY_STRIDE_7: tl.constexpr,
    X_STRIDE_0: tl.constexpr,
    X_STRIDE_1: tl.constexpr,
    X_STRIDE_2: tl.constexpr,
    X_STRIDE_3: tl.constexpr,
    X_STRIDE_4: tl.constexpr,
    X_STRIDE_5: tl.constexpr,
    X_STRIDE_6: tl.constexpr,
    X_STRIDE_7: tl.constexpr,
    SCALE_STRIDE_0: tl.constexpr,
    SCALE_STRIDE_1: tl.constexpr,
    SCALE_STRIDE_2: tl.constexpr,
    SCALE_STRIDE_3: tl.constexpr,
    SCALE_STRIDE_4: tl.constexpr,
    SCALE_STRIDE_5: tl.constexpr,
    SCALE_STRIDE_6: tl.constexpr,
    SCALE_STRIDE_7: tl.constexpr,
    MEAN_STRIDE_0: tl.constexpr,
    MEAN_STRIDE_1: tl.constexpr,
    MEAN_STRIDE_2: tl.constexpr,
    MEAN_STRIDE_3: tl.constexpr,
    MEAN_STRIDE_4: tl.constexpr,
    MEAN_STRIDE_5: tl.constexpr,
    MEAN_STRIDE_6: tl.constexpr,
    MEAN_STRIDE_7: tl.constexpr,
    INV_STRIDE_0: tl.constexpr,
    INV_STRIDE_1: tl.constexpr,
    INV_STRIDE_2: tl.constexpr,
    INV_STRIDE_3: tl.constexpr,
    INV_STRIDE_4: tl.constexpr,
    INV_STRIDE_5: tl.constexpr,
    INV_STRIDE_6: tl.constexpr,
    INV_STRIDE_7: tl.constexpr,
    DX_STRIDE_0: tl.constexpr,
    DX_STRIDE_1: tl.constexpr,
    DX_STRIDE_2: tl.constexpr,
    DX_STRIDE_3: tl.constexpr,
    DX_STRIDE_4: tl.constexpr,
    DX_STRIDE_5: tl.constexpr,
    DX_STRIDE_6: tl.constexpr,
    DX_STRIDE_7: tl.constexpr,
    DSCALE_STRIDE_0: tl.constexpr,
    DSCALE_STRIDE_1: tl.constexpr,
    DSCALE_STRIDE_2: tl.constexpr,
    DSCALE_STRIDE_3: tl.constexpr,
    DSCALE_STRIDE_4: tl.constexpr,
    DSCALE_STRIDE_5: tl.constexpr,
    DSCALE_STRIDE_6: tl.constexpr,
    DSCALE_STRIDE_7: tl.constexpr,
    DBIAS_STRIDE_0: tl.constexpr,
    DBIAS_STRIDE_1: tl.constexpr,
    DBIAS_STRIDE_2: tl.constexpr,
    DBIAS_STRIDE_3: tl.constexpr,
    DBIAS_STRIDE_4: tl.constexpr,
    DBIAS_STRIDE_5: tl.constexpr,
    DBIAS_STRIDE_6: tl.constexpr,
    DBIAS_STRIDE_7: tl.constexpr,
):
    dims: tl.constexpr = (
        DIM_0,
        DIM_1,
        DIM_2,
        DIM_3,
        DIM_4,
        DIM_5,
        DIM_6,
        DIM_7,
    )
    dy_strides: tl.constexpr = (
        DY_STRIDE_0,
        DY_STRIDE_1,
        DY_STRIDE_2,
        DY_STRIDE_3,
        DY_STRIDE_4,
        DY_STRIDE_5,
        DY_STRIDE_6,
        DY_STRIDE_7,
    )
    x_strides: tl.constexpr = (
        X_STRIDE_0,
        X_STRIDE_1,
        X_STRIDE_2,
        X_STRIDE_3,
        X_STRIDE_4,
        X_STRIDE_5,
        X_STRIDE_6,
        X_STRIDE_7,
    )
    scale_strides: tl.constexpr = (
        SCALE_STRIDE_0,
        SCALE_STRIDE_1,
        SCALE_STRIDE_2,
        SCALE_STRIDE_3,
        SCALE_STRIDE_4,
        SCALE_STRIDE_5,
        SCALE_STRIDE_6,
        SCALE_STRIDE_7,
    )
    mean_strides: tl.constexpr = (
        MEAN_STRIDE_0,
        MEAN_STRIDE_1,
        MEAN_STRIDE_2,
        MEAN_STRIDE_3,
        MEAN_STRIDE_4,
        MEAN_STRIDE_5,
        MEAN_STRIDE_6,
        MEAN_STRIDE_7,
    )
    inv_strides: tl.constexpr = (
        INV_STRIDE_0,
        INV_STRIDE_1,
        INV_STRIDE_2,
        INV_STRIDE_3,
        INV_STRIDE_4,
        INV_STRIDE_5,
        INV_STRIDE_6,
        INV_STRIDE_7,
    )
    dx_strides: tl.constexpr = (
        DX_STRIDE_0,
        DX_STRIDE_1,
        DX_STRIDE_2,
        DX_STRIDE_3,
        DX_STRIDE_4,
        DX_STRIDE_5,
        DX_STRIDE_6,
        DX_STRIDE_7,
    )
    dscale_strides: tl.constexpr = (
        DSCALE_STRIDE_0,
        DSCALE_STRIDE_1,
        DSCALE_STRIDE_2,
        DSCALE_STRIDE_3,
        DSCALE_STRIDE_4,
        DSCALE_STRIDE_5,
        DSCALE_STRIDE_6,
        DSCALE_STRIDE_7,
    )
    dbias_strides: tl.constexpr = (
        DBIAS_STRIDE_0,
        DBIAS_STRIDE_1,
        DBIAS_STRIDE_2,
        DBIAS_STRIDE_3,
        DBIAS_STRIDE_4,
        DBIAS_STRIDE_5,
        DBIAS_STRIDE_6,
        DBIAS_STRIDE_7,
    )
    program = tl.program_id(0).to(tl.int64)
    lanes = tl.arange(0, BLOCK_SIZE)
    if program < GROUPS:
        statistic_coords = _coordinates(
            program, tl.full((), 0, tl.int64), dims, AXES
        )
        if RMS:
            mean = 0.0
        else:
            mean = tl.load(
                mean_ptr + _offset(statistic_coords, mean_strides)
            ).to(tl.float32)
        inverse = tl.load(inv_ptr + _offset(statistic_coords, inv_strides)).to(
            tl.float32
        )
        sum_gradient = tl.zeros((BLOCK_SIZE,), tl.float32)
        sum_gradient_normalized = tl.zeros((BLOCK_SIZE,), tl.float32)
        for start in range(0, REDUCTION, BLOCK_SIZE):
            reduced = start + lanes
            coords = _coordinates(program, reduced, dims, AXES)
            valid = reduced < REDUCTION
            x = tl.load(x_ptr + _offset(coords, x_strides), valid, other=0).to(
                tl.float32
            )
            dy = tl.load(
                dy_ptr + _offset(coords, dy_strides), valid, other=0
            ).to(tl.float32)
            scale = tl.load(
                scale_ptr + _offset(coords, scale_strides), valid, other=0
            ).to(tl.float32)
            gradient = dy * scale
            normalized = (x - mean) * inverse
            sum_gradient += gradient
            sum_gradient_normalized += gradient * normalized
        gradient_mean = tl.sum(sum_gradient, 0) / REDUCTION
        gradient_normalized_mean = (
            tl.sum(sum_gradient_normalized, 0) / REDUCTION
        )
        for start in range(0, REDUCTION, BLOCK_SIZE):
            reduced = start + lanes
            coords = _coordinates(program, reduced, dims, AXES)
            valid = reduced < REDUCTION
            x = tl.load(x_ptr + _offset(coords, x_strides), valid, other=0).to(
                tl.float32
            )
            dy = tl.load(
                dy_ptr + _offset(coords, dy_strides), valid, other=0
            ).to(tl.float32)
            scale = tl.load(
                scale_ptr + _offset(coords, scale_strides), valid, other=0
            ).to(tl.float32)
            gradient = (
                dy * scale - (x - mean) * inverse * gradient_normalized_mean
            )
            if not RMS:
                gradient -= gradient_mean
            tl.store(
                dx_ptr + _offset(coords, dx_strides), gradient * inverse, valid
            )
    else:
        parameter = program - GROUPS
        sum_scale = tl.zeros((BLOCK_SIZE,), tl.float32)
        sum_bias = tl.zeros((BLOCK_SIZE,), tl.float32)
        for start in range(0, AFFINE_REDUCTION, BLOCK_SIZE):
            reduced = start + lanes
            coords = _coordinates(parameter, reduced, dims, PARAM_AXES)
            valid = reduced < AFFINE_REDUCTION
            x = tl.load(x_ptr + _offset(coords, x_strides), valid, other=0).to(
                tl.float32
            )
            dy = tl.load(
                dy_ptr + _offset(coords, dy_strides), valid, other=0
            ).to(tl.float32)
            if RMS:
                mean = 0.0
            else:
                mean = tl.load(
                    mean_ptr + _offset(coords, mean_strides), valid, other=0
                ).to(tl.float32)
            inverse = tl.load(
                inv_ptr + _offset(coords, inv_strides), valid, other=0
            ).to(tl.float32)
            sum_scale += dy * (x - mean) * inverse
            sum_bias += dy
        coords = _coordinates(
            parameter, tl.full((), 0, tl.int64), dims, PARAM_AXES
        )
        tl.store(
            dscale_ptr + _offset(coords, dscale_strides), tl.sum(sum_scale, 0)
        )
        tl.store(
            dbias_ptr + _offset(coords, dbias_strides), tl.sum(sum_bias, 0)
        )


@triton.jit
def compact_normalization_forward(
    x_ptr,
    scale_ptr,
    bias_ptr,
    y_ptr,
    mean_ptr,
    inv_ptr,
    REDUCTION: tl.constexpr,
    CHANNELS: tl.constexpr,
    ROWS_PER_BATCH: tl.constexpr,
    AFFINE_MODE: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK_SIZE)
    valid = columns < REDUCTION
    x = tl.load(x_ptr + row * REDUCTION + columns, valid, other=0).to(
        tl.float32
    )
    mean = tl.sum(x, 0) / REDUCTION
    centered = tl.where(valid, x - mean, 0.0)
    inverse = tl.rsqrt(tl.sum(centered * centered, 0) / REDUCTION + EPSILON)
    if AFFINE_MODE == 0:
        scale = tl.load(scale_ptr + row % CHANNELS).to(tl.float32)
        bias = tl.load(bias_ptr + row % CHANNELS).to(tl.float32)
    else:
        affine = columns
        if AFFINE_MODE == 2:
            affine += row // ROWS_PER_BATCH * REDUCTION
        scale = tl.load(scale_ptr + affine, valid, other=0).to(tl.float32)
        bias = tl.load(bias_ptr + affine, valid, other=0).to(tl.float32)
    tl.store(
        y_ptr + row * REDUCTION + columns,
        centered * inverse * scale + bias,
        valid,
    )
    tl.store(mean_ptr + row, mean)
    tl.store(inv_ptr + row, inverse)


@triton.jit
def compact_batchnorm_backward(
    dy_ptr,
    x_ptr,
    scale_ptr,
    mean_ptr,
    inv_ptr,
    dx_ptr,
    dscale_ptr,
    dbias_ptr,
    CHANNELS: tl.constexpr,
    SPATIAL: tl.constexpr,
    REDUCTION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    channel = tl.program_id(0).to(tl.int64)
    reduced = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    valid = reduced < REDUCTION
    offset = (
        reduced // SPATIAL * (CHANNELS * SPATIAL)
        + channel * SPATIAL
        + reduced % SPATIAL
    )
    mean = tl.load(mean_ptr + channel)
    inverse = tl.load(inv_ptr + channel)
    scale = tl.load(scale_ptr + channel).to(tl.float32)
    x = tl.load(x_ptr + offset, valid, other=0).to(tl.float32)
    dy = tl.load(dy_ptr + offset, valid, other=0).to(tl.float32)
    normalized = (x - mean) * inverse
    sum_dy = tl.sum(dy, 0)
    sum_normalized = tl.sum(dy * normalized, 0)
    dx = (
        dy - sum_dy / REDUCTION - normalized * (sum_normalized / REDUCTION)
    ) * (scale * inverse)
    tl.store(dx_ptr + offset, dx, valid)
    tl.store(dscale_ptr + channel, sum_normalized)
    tl.store(dbias_ptr + channel, sum_dy)
