# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Depthwise causal convolution with FP32 accumulation and fused epilogue."""
import triton
import triton.language as tl


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
):
    logical = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(
        0, BLOCK_SIZE
    )
    position = logical % LENGTH
    channel = logical // LENGTH % CHANNELS
    batch = logical // (LENGTH * CHANNELS)
    valid = logical < ELEMENTS
    accumulator = tl.zeros((BLOCK_SIZE,), tl.float32)
    for tap in range(WIDTH):
        source = position - (WIDTH - 1 - tap) * DILATION
        x = tl.load(
            x_ptr + batch * X_B + channel * X_C + source * X_L,
            valid & (source >= 0),
            other=0,
        ).to(tl.float32)
        w = tl.load(weight_ptr + channel * W_C + tap * W_K, valid, other=0).to(
            tl.float32
        )
        accumulator += x * w
    if HAS_BIAS:
        accumulator += tl.load(bias_ptr + channel * B_C, valid, other=0).to(
            tl.float32
        )
    if SILU:
        accumulator *= tl.sigmoid(accumulator)
    tl.store(
        y_ptr + batch * Y_B + channel * Y_C + position * Y_L,
        accumulator,
        valid,
    )
