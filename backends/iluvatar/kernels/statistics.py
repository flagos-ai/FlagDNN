# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Channel statistics with FP32 accumulation for arbitrary tensor strides."""

import triton
import triton.language as tl


@triton.jit
def genstats_kernel(
    x_ptr,
    sum_ptr,
    sq_sum_ptr,
    CHANNELS: tl.constexpr,
    REDUCTION: tl.constexpr,
    SPATIAL: tl.constexpr,
    SUM_CHANNEL_STRIDE: tl.constexpr,
    SQ_SUM_CHANNEL_STRIDE: tl.constexpr,
    DIM_0: tl.constexpr,
    DIM_1: tl.constexpr,
    DIM_2: tl.constexpr,
    DIM_3: tl.constexpr,
    DIM_4: tl.constexpr,
    DIM_5: tl.constexpr,
    DIM_6: tl.constexpr,
    DIM_7: tl.constexpr,
    STRIDE_0: tl.constexpr,
    STRIDE_1: tl.constexpr,
    STRIDE_2: tl.constexpr,
    STRIDE_3: tl.constexpr,
    STRIDE_4: tl.constexpr,
    STRIDE_5: tl.constexpr,
    STRIDE_6: tl.constexpr,
    STRIDE_7: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    channel = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    partial_sum = tl.zeros((BLOCK_SIZE,), tl.float32)
    partial_square = tl.zeros((BLOCK_SIZE,), tl.float32)
    for start in range(0, REDUCTION, BLOCK_SIZE):
        reduced = start + offsets
        logical = (
            reduced.to(tl.int64) // SPATIAL * CHANNELS + channel
        ) * SPATIAL + reduced % SPATIAL
        physical = tl.zeros((BLOCK_SIZE,), tl.int64)
        physical += (logical % DIM_7) * STRIDE_7
        logical //= DIM_7
        physical += (logical % DIM_6) * STRIDE_6
        logical //= DIM_6
        physical += (logical % DIM_5) * STRIDE_5
        logical //= DIM_5
        physical += (logical % DIM_4) * STRIDE_4
        logical //= DIM_4
        physical += (logical % DIM_3) * STRIDE_3
        logical //= DIM_3
        physical += (logical % DIM_2) * STRIDE_2
        logical //= DIM_2
        physical += (logical % DIM_1) * STRIDE_1
        logical //= DIM_1
        physical += (logical % DIM_0) * STRIDE_0
        logical //= DIM_0
        x = tl.load(x_ptr + physical, reduced < REDUCTION, other=0).to(tl.float32)
        partial_sum += x
        partial_square += x * x
    tl.store(sum_ptr + channel * SUM_CHANNEL_STRIDE, tl.sum(partial_sum, 0))
    tl.store(sq_sum_ptr + channel * SQ_SUM_CHANNEL_STRIDE, tl.sum(partial_square, 0))


@triton.jit
def bn_finalize_kernel(
    sum_ptr,
    sq_sum_ptr,
    scale_ptr,
    bias_ptr,
    prev_mean_ptr,
    prev_var_ptr,
    eq_scale_ptr,
    eq_bias_ptr,
    mean_ptr,
    inv_ptr,
    next_mean_ptr,
    next_var_ptr,
    SUM_STRIDE: tl.constexpr,
    SQ_SUM_STRIDE: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    BIAS_STRIDE: tl.constexpr,
    PREV_MEAN_STRIDE: tl.constexpr,
    PREV_VAR_STRIDE: tl.constexpr,
    EQ_SCALE_STRIDE: tl.constexpr,
    EQ_BIAS_STRIDE: tl.constexpr,
    MEAN_STRIDE: tl.constexpr,
    INV_STRIDE: tl.constexpr,
    NEXT_MEAN_STRIDE: tl.constexpr,
    NEXT_VAR_STRIDE: tl.constexpr,
    CHANNELS: tl.constexpr,
    COUNT: tl.constexpr,
    EPSILON: tl.constexpr,
    MOMENTUM: tl.constexpr,
    HAS_RUNNING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    channel = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = channel < CHANNELS
    total = tl.load(sum_ptr + channel * SUM_STRIDE, valid, other=0).to(tl.float32)
    square = tl.load(sq_sum_ptr + channel * SQ_SUM_STRIDE, valid, other=0).to(
        tl.float32
    )
    scale = tl.load(scale_ptr + channel * SCALE_STRIDE, valid, other=0).to(tl.float32)
    bias = tl.load(bias_ptr + channel * BIAS_STRIDE, valid, other=0).to(tl.float32)
    # CoreX 4.4/corex_71 does not implement reliable FP64 arithmetic.
    # Match the declared FP32 compute type for the moments and inverse.
    mean = total / COUNT
    variance = tl.maximum(square / COUNT - mean * mean, 0.0)
    inverse = tl.rsqrt((variance + EPSILON).to(tl.float32))
    equivalent_scale = scale * inverse
    tl.store(eq_scale_ptr + channel * EQ_SCALE_STRIDE, equivalent_scale, valid)
    tl.store(
        eq_bias_ptr + channel * EQ_BIAS_STRIDE,
        bias - mean * equivalent_scale,
        valid,
    )
    tl.store(mean_ptr + channel * MEAN_STRIDE, mean, valid)
    tl.store(inv_ptr + channel * INV_STRIDE, inverse, valid)
    if HAS_RUNNING:
        previous_mean = tl.load(
            prev_mean_ptr + channel * PREV_MEAN_STRIDE, valid, other=0
        ).to(tl.float32)
        previous_variance = tl.load(
            prev_var_ptr + channel * PREV_VAR_STRIDE, valid, other=0
        ).to(tl.float32)
        unbiased_variance = variance * (COUNT / (COUNT - 1.0) if COUNT > 1.0 else 0.0)
        tl.store(
            next_mean_ptr + channel * NEXT_MEAN_STRIDE,
            (1.0 - MOMENTUM) * previous_mean + MOMENTUM * mean,
            valid,
        )
        tl.store(
            next_var_ptr + channel * NEXT_VAR_STRIDE,
            (1.0 - MOMENTUM) * previous_variance + MOMENTUM * unbiased_variance,
            valid,
        )
