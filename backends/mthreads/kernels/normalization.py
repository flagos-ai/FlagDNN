# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""MThreads BatchNorm-inference kernels.

The MTGPU Triton frontend cannot lower the common NCHW kernel's nested
compile-time channel/spatial tile arithmetic.  Keep the public kernel ABI and
use a two-dimensional launch instead: X indexes one batch/channel pair and Y
indexes one contiguous spatial tile.  The generic explicit-stride entry point
retains the common implementation's semantics.
"""

import triton
import triton.language as tl


@triton.jit
def batch_norm_inference_nchw_kernel(
    x_ptr,
    mean_ptr,
    stat_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    C: tl.constexpr,
    S: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    STAT_IS_INV_VARIANCE: tl.constexpr,
):
    batch_channel = tl.program_id(0).to(tl.int64)
    channel = batch_channel % C
    batch = batch_channel // C
    spatial = (
        tl.program_id(1).to(tl.int64) * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)
    )
    active = spatial < S
    offsets = (batch * C + channel) * S + spatial

    values = tl.load(x_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channel).to(tl.float32)
    statistic = tl.load(stat_ptr + channel).to(tl.float32)
    inv_variance = (
        statistic if STAT_IS_INV_VARIANCE else tl.rsqrt(statistic + eps)
    )
    weight = (
        tl.load(weight_ptr + channel).to(tl.float32) if HAS_WEIGHT else 1.0
    )
    bias = tl.load(bias_ptr + channel).to(tl.float32) if HAS_BIAS else 0.0
    result = (values - mean) * inv_variance * weight + bias
    tl.store(
        y_ptr + offsets,
        result.to(y_ptr.dtype.element_ty),
        mask=active,
    )


@triton.jit
def batch_norm_inference_kernel(
    x_ptr,
    mean_ptr,
    stat_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    total_elements,
    C,
    S,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    STAT_IS_INV_VARIANCE: tl.constexpr,
    STRIDED: tl.constexpr,
    DIM_0: tl.constexpr,
    DIM_1: tl.constexpr,
    DIM_2: tl.constexpr,
    DIM_3: tl.constexpr,
    DIM_4: tl.constexpr,
    DIM_5: tl.constexpr,
    DIM_6: tl.constexpr,
    DIM_7: tl.constexpr,
    INPUT_STRIDE_0: tl.constexpr,
    INPUT_STRIDE_1: tl.constexpr,
    INPUT_STRIDE_2: tl.constexpr,
    INPUT_STRIDE_3: tl.constexpr,
    INPUT_STRIDE_4: tl.constexpr,
    INPUT_STRIDE_5: tl.constexpr,
    INPUT_STRIDE_6: tl.constexpr,
    INPUT_STRIDE_7: tl.constexpr,
    OUTPUT_STRIDE_0: tl.constexpr,
    OUTPUT_STRIDE_1: tl.constexpr,
    OUTPUT_STRIDE_2: tl.constexpr,
    OUTPUT_STRIDE_3: tl.constexpr,
    OUTPUT_STRIDE_4: tl.constexpr,
    OUTPUT_STRIDE_5: tl.constexpr,
    OUTPUT_STRIDE_6: tl.constexpr,
    OUTPUT_STRIDE_7: tl.constexpr,
):
    logical = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(
        0, BLOCK_SIZE
    )
    active = logical < total_elements
    channel = (logical // S) % C
    if STRIDED:
        remaining = logical
        input_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        output_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        coordinate = remaining % DIM_7
        remaining //= DIM_7
        input_offsets += coordinate * INPUT_STRIDE_7
        output_offsets += coordinate * OUTPUT_STRIDE_7
        coordinate = remaining % DIM_6
        remaining //= DIM_6
        input_offsets += coordinate * INPUT_STRIDE_6
        output_offsets += coordinate * OUTPUT_STRIDE_6
        coordinate = remaining % DIM_5
        remaining //= DIM_5
        input_offsets += coordinate * INPUT_STRIDE_5
        output_offsets += coordinate * OUTPUT_STRIDE_5
        coordinate = remaining % DIM_4
        remaining //= DIM_4
        input_offsets += coordinate * INPUT_STRIDE_4
        output_offsets += coordinate * OUTPUT_STRIDE_4
        coordinate = remaining % DIM_3
        remaining //= DIM_3
        input_offsets += coordinate * INPUT_STRIDE_3
        output_offsets += coordinate * OUTPUT_STRIDE_3
        coordinate = remaining % DIM_2
        remaining //= DIM_2
        input_offsets += coordinate * INPUT_STRIDE_2
        output_offsets += coordinate * OUTPUT_STRIDE_2
        coordinate = remaining % DIM_1
        remaining //= DIM_1
        input_offsets += coordinate * INPUT_STRIDE_1
        output_offsets += coordinate * OUTPUT_STRIDE_1
        coordinate = remaining % DIM_0
        input_offsets += coordinate * INPUT_STRIDE_0
        output_offsets += coordinate * OUTPUT_STRIDE_0
    else:
        input_offsets = logical
        output_offsets = logical

    values = tl.load(x_ptr + input_offsets, mask=active, other=0.0).to(
        tl.float32
    )
    mean = tl.load(mean_ptr + channel, mask=active, other=0.0).to(tl.float32)
    statistic = tl.load(stat_ptr + channel, mask=active, other=0.0).to(
        tl.float32
    )
    inv_variance = (
        statistic if STAT_IS_INV_VARIANCE else tl.rsqrt(statistic + eps)
    )
    weight = (
        tl.load(weight_ptr + channel, mask=active, other=1.0).to(tl.float32)
        if HAS_WEIGHT
        else 1.0
    )
    bias = (
        tl.load(bias_ptr + channel, mask=active, other=0.0).to(tl.float32)
        if HAS_BIAS
        else 0.0
    )
    result = (values - mean) * inv_variance * weight + bias
    tl.store(
        y_ptr + output_offsets,
        result.to(y_ptr.dtype.element_ty),
        mask=active,
    )
