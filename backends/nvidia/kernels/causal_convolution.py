# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA causal convolution with TF32 rounding and channel tiling."""
import triton
import triton.language as tl


@triton.jit
def _tf32(value):
    bits = value.to(tl.uint32, bitcast=True)
    rounded = ((bits + 0xFFF + ((bits >> 13) & 1)) & 0xFFFFE000).to(
        tl.float32, bitcast=True
    )
    return tl.where((bits & 0x7F800000) == 0x7F800000, value, rounded)


@triton.jit
def causal_conv1d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    ELEMENTS: tl.constexpr,
    CHANNELS: tl.constexpr,
    LENGTH: tl.constexpr,
    WIDTH: tl.constexpr,
    DILATION: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
    TF32: tl.constexpr,
    X_B: tl.constexpr,
    X_C: tl.constexpr,
    X_L: tl.constexpr,
    W_C: tl.constexpr,
    W_K: tl.constexpr,
    B_C: tl.constexpr,
    Y_B: tl.constexpr,
    Y_C: tl.constexpr,
    Y_L: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CHANNEL_TILE: tl.constexpr = False,
):
    if CHANNEL_TILE:
        tiles: tl.constexpr = triton.cdiv(LENGTH, BLOCK_SIZE)
        series = tl.program_id(0).to(tl.int64) // tiles
        position = tl.program_id(0).to(
            tl.int64
        ) % tiles * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        channel = series % CHANNELS
        batch = series // CHANNELS
        valid = position < LENGTH
    else:
        logical = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(
            0, BLOCK_SIZE
        )
        position = logical % LENGTH
        channel = logical // LENGTH % CHANNELS
        batch = logical // (LENGTH * CHANNELS)
        valid = logical < ELEMENTS
    accumulator = tl.zeros((BLOCK_SIZE,), tl.float32)
    for tap in tl.range(0, WIDTH, loop_unroll_factor=8 if CHANNEL_TILE else 1):
        source = position - (WIDTH - 1 - tap) * DILATION
        x = tl.load(
            x_ptr + batch * X_B + channel * X_C + source * X_L,
            valid & (source >= 0),
            other=0,
        ).to(tl.float32)
        if CHANNEL_TILE:
            w = tl.load(weight_ptr + channel * W_C + tap * W_K).to(tl.float32)
        else:
            w = tl.load(
                weight_ptr + channel * W_C + tap * W_K, valid, other=0
            ).to(tl.float32)
        if TF32:
            x, w = _tf32(x), _tf32(w)
        accumulator += x * w
    if HAS_BIAS:
        if CHANNEL_TILE:
            accumulator += tl.load(bias_ptr + channel * B_C).to(tl.float32)
        else:
            accumulator += tl.load(
                bias_ptr + channel * B_C, valid, other=0
            ).to(tl.float32)
    if SILU:
        accumulator *= tl.sigmoid(accumulator)
    tl.store(
        y_ptr + batch * Y_B + channel * Y_C + position * Y_L,
        accumulator,
        valid,
    )
