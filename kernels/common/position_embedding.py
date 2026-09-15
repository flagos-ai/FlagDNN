# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Halved rotary embeddings and their transpose Jacobian."""
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def rope_kernel(
    x_ptr,
    freqs_ptr,
    y_ptr,
    ELEMENTS: tl.constexpr,
    HEADS: tl.constexpr,
    SEQUENCE: tl.constexpr,
    DIMENSION: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    SCALE: tl.constexpr,
    X_B: tl.constexpr,
    X_H: tl.constexpr,
    X_S: tl.constexpr,
    X_D: tl.constexpr,
    Y_B: tl.constexpr,
    Y_H: tl.constexpr,
    Y_S: tl.constexpr,
    Y_D: tl.constexpr,
    F_S: tl.constexpr,
    F_D: tl.constexpr,
    BACKWARD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(
        0, BLOCK_SIZE
    )
    d = index % DIMENSION
    position = index // DIMENSION % SEQUENCE
    head = index // (DIMENSION * SEQUENCE) % HEADS
    batch = index // (DIMENSION * SEQUENCE * HEADS)
    source = batch * X_B + head * X_H + position * X_S + d * X_D
    destination = batch * Y_B + head * Y_H + position * Y_S + d * Y_D
    active = d >= DIMENSION - ROPE_DIM
    relative = d - (DIMENSION - ROPE_DIM)
    low = relative < ROPE_DIM // 2
    partner = tl.where(low, relative + ROPE_DIM // 2, relative - ROPE_DIM // 2)
    valid = index < ELEMENTS
    x = tl.load(x_ptr + source, valid, other=0).to(tl.float32)
    paired = tl.load(
        x_ptr + source + (partner - relative) * X_D, valid & active, other=0
    ).to(tl.float32)
    angle = tl.load(
        freqs_ptr + position * F_S + relative * F_D, valid & active, other=0
    ).to(tl.float32)
    if BACKWARD:
        paired_angle = tl.load(
            freqs_ptr + position * F_S + partner * F_D, valid & active, other=0
        ).to(tl.float32)
        sine = libdevice.sin(paired_angle)
        sign = tl.where(low, 1.0, -1.0)
    else:
        sine = libdevice.sin(angle)
        sign = tl.where(low, -1.0, 1.0)
    result = x * libdevice.cos(angle) + sign * paired * sine
    tl.store(y_ptr + destination, tl.where(active, result, x) * SCALE, valid)
