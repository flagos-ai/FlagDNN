# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""THead normalization kernels for PPU-specific Triton limitations."""

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
    """Pack small channels; retain scalar parameter loads for one channel."""
    BLOCK_S: tl.constexpr = min(triton.next_power_of_2(S), BLOCK_SIZE)
    BLOCK_C: tl.constexpr = BLOCK_SIZE // BLOCK_S
    if BLOCK_C == 1:
        # A [1, BLOCK_S] PPU layout duplicates masked parameter loads.
        # The original scalar loads avoid that overhead for unpacked tiles.
        program = tl.program_id(0).to(tl.int64)
        spatial_blocks: tl.constexpr = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
        spatial_block = program % spatial_blocks
        remaining = program // spatial_blocks
        channel = remaining % C
        batch = remaining // C
        spatial = spatial_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
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

    else:
        program = tl.program_id(0).to(tl.int64)
        SPATIAL_BLOCKS: tl.constexpr = (S + BLOCK_S - 1) // BLOCK_S
        CHANNEL_BLOCKS: tl.constexpr = (C + BLOCK_C - 1) // BLOCK_C

        spatial_block = program % SPATIAL_BLOCKS
        remaining = program // SPATIAL_BLOCKS
        channel_block = remaining % CHANNEL_BLOCKS
        batch = remaining // CHANNEL_BLOCKS
        channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)[:, None]
        spatial = spatial_block * BLOCK_S + tl.arange(0, BLOCK_S)[None, :]
        channel_active = channels < C
        active = channel_active & (spatial < S)
        offsets = (batch * C + channels) * S + spatial

        values = tl.load(x_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + channels, mask=channel_active, other=0.0).to(
            tl.float32
        )
        statistic = tl.load(
            stat_ptr + channels, mask=channel_active, other=0.0
        ).to(tl.float32)
        inv_variance = (
            statistic if STAT_IS_INV_VARIANCE else tl.rsqrt(statistic + eps)
        )
        weight = (
            tl.load(weight_ptr + channels, mask=channel_active, other=1.0).to(
                tl.float32
            )
            if HAS_WEIGHT
            else 1.0
        )
        bias = (
            tl.load(bias_ptr + channels, mask=channel_active, other=0.0).to(
                tl.float32
            )
            if HAS_BIAS
            else 0.0
        )
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
    """Apply inference BatchNorm to an explicit non-overlapping layout.

    This platform-owned spelling keeps the same frontend ABI as the common
    strided kernel while avoiding the PPU constexpr issue in the common NCHW
    specialization.
    """

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

    values = tl.load(
        x_ptr + input_offsets, mask=active, other=0.0
    ).to(tl.float32)
    mean = tl.load(mean_ptr + channel, mask=active, other=0.0).to(tl.float32)
    statistic = tl.load(
        stat_ptr + channel, mask=active, other=0.0
    ).to(tl.float32)
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
