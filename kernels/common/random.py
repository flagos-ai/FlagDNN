# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Counter-based Philox random numbers with layout-independent streams."""

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def rng_kernel(
    output_ptr,
    N_ELEMENTS: tl.constexpr,
    SEED: tl.constexpr,
    OFFSET: tl.constexpr,
    DISTRIBUTION: tl.constexpr,
    PROBABILITY: tl.constexpr,
    UNIFORM_BITS: tl.constexpr,
    DIM_0: tl.constexpr,
    DIM_1: tl.constexpr,
    DIM_2: tl.constexpr,
    DIM_3: tl.constexpr,
    DIM_4: tl.constexpr,
    DIM_5: tl.constexpr,
    DIM_6: tl.constexpr,
    DIM_7: tl.constexpr,
    OUTPUT_STRIDE_0: tl.constexpr,
    OUTPUT_STRIDE_1: tl.constexpr,
    OUTPUT_STRIDE_2: tl.constexpr,
    OUTPUT_STRIDE_3: tl.constexpr,
    OUTPUT_STRIDE_4: tl.constexpr,
    OUTPUT_STRIDE_5: tl.constexpr,
    OUTPUT_STRIDE_6: tl.constexpr,
    OUTPUT_STRIDE_7: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    logical = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(
        0, BLOCK_SIZE
    )
    remaining = logical
    physical = tl.zeros((BLOCK_SIZE,), tl.int64)
    physical += (remaining % DIM_7) * OUTPUT_STRIDE_7
    remaining //= DIM_7
    physical += (remaining % DIM_6) * OUTPUT_STRIDE_6
    remaining //= DIM_6
    physical += (remaining % DIM_5) * OUTPUT_STRIDE_5
    remaining //= DIM_5
    physical += (remaining % DIM_4) * OUTPUT_STRIDE_4
    remaining //= DIM_4
    physical += (remaining % DIM_3) * OUTPUT_STRIDE_3
    remaining //= DIM_3
    physical += (remaining % DIM_2) * OUTPUT_STRIDE_2
    remaining //= DIM_2
    physical += (remaining % DIM_1) * OUTPUT_STRIDE_1
    remaining //= DIM_1
    physical += (remaining % DIM_0) * OUTPUT_STRIDE_0
    remaining //= DIM_0
    first, second, _, _ = tl.randint4x(SEED, (logical + OFFSET).to(tl.uint64))
    if DISTRIBUTION == 1:
        value = (first >> (32 - UNIFORM_BITS)).to(tl.float32) * (
            2.0 ** (-UNIFORM_BITS)
        )
    elif DISTRIBUTION == 2:
        # Open-closed first uniform prevents log(0); second is in [0,1).
        u1 = ((first >> 8).to(tl.float32) + 1.0) * (2.0**-24)
        u2 = (second >> 8).to(tl.float32) * (2.0**-24)
        value = tl.sqrt(-2.0 * tl.log(u1)) * libdevice.cos(
            6.283185307179586 * u2
        )
    else:
        uniform = (first >> 8).to(tl.float32) * (2.0**-24)
        value = (uniform < PROBABILITY).to(tl.float32)
    tl.store(output_ptr + physical, value, logical < N_ELEMENTS)
