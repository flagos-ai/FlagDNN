# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Fused THead kernel for the canonical ``left + right * right`` graph."""

import triton
import triton.language as tl


@triton.jit
def add_square_contiguous_kernel(
    right_ptr,
    left_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    right = tl.load(right_ptr + offsets, mask=mask, other=0.0)
    left = tl.load(left_ptr + offsets, mask=mask, other=0.0)
    if left.dtype == tl.int32:
        result = left + right * right
    else:
        result = left.to(tl.float32) + right.to(tl.float32) * right.to(tl.float32)
    tl.store(
        output_ptr + offsets,
        result.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def add_square_strided_kernel(
    right_ptr,
    left_ptr,
    output_ptr,
    n_elements,
    DIM_0: tl.constexpr,
    DIM_1: tl.constexpr,
    DIM_2: tl.constexpr,
    DIM_3: tl.constexpr,
    DIM_4: tl.constexpr,
    DIM_5: tl.constexpr,
    DIM_6: tl.constexpr,
    DIM_7: tl.constexpr,
    RIGHT_STRIDE_0: tl.constexpr,
    RIGHT_STRIDE_1: tl.constexpr,
    RIGHT_STRIDE_2: tl.constexpr,
    RIGHT_STRIDE_3: tl.constexpr,
    RIGHT_STRIDE_4: tl.constexpr,
    RIGHT_STRIDE_5: tl.constexpr,
    RIGHT_STRIDE_6: tl.constexpr,
    RIGHT_STRIDE_7: tl.constexpr,
    LEFT_STRIDE_0: tl.constexpr,
    LEFT_STRIDE_1: tl.constexpr,
    LEFT_STRIDE_2: tl.constexpr,
    LEFT_STRIDE_3: tl.constexpr,
    LEFT_STRIDE_4: tl.constexpr,
    LEFT_STRIDE_5: tl.constexpr,
    LEFT_STRIDE_6: tl.constexpr,
    LEFT_STRIDE_7: tl.constexpr,
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
    """Fuse ``left + right * right`` for explicit non-overlapping layouts."""

    logical = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(
        0, BLOCK_SIZE
    )
    active = logical < n_elements
    remaining = logical
    right_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    left_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    output_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

    coordinate = remaining % DIM_7
    remaining //= DIM_7
    right_offsets += coordinate * RIGHT_STRIDE_7
    left_offsets += coordinate * LEFT_STRIDE_7
    output_offsets += coordinate * OUTPUT_STRIDE_7
    coordinate = remaining % DIM_6
    remaining //= DIM_6
    right_offsets += coordinate * RIGHT_STRIDE_6
    left_offsets += coordinate * LEFT_STRIDE_6
    output_offsets += coordinate * OUTPUT_STRIDE_6
    coordinate = remaining % DIM_5
    remaining //= DIM_5
    right_offsets += coordinate * RIGHT_STRIDE_5
    left_offsets += coordinate * LEFT_STRIDE_5
    output_offsets += coordinate * OUTPUT_STRIDE_5
    coordinate = remaining % DIM_4
    remaining //= DIM_4
    right_offsets += coordinate * RIGHT_STRIDE_4
    left_offsets += coordinate * LEFT_STRIDE_4
    output_offsets += coordinate * OUTPUT_STRIDE_4
    coordinate = remaining % DIM_3
    remaining //= DIM_3
    right_offsets += coordinate * RIGHT_STRIDE_3
    left_offsets += coordinate * LEFT_STRIDE_3
    output_offsets += coordinate * OUTPUT_STRIDE_3
    coordinate = remaining % DIM_2
    remaining //= DIM_2
    right_offsets += coordinate * RIGHT_STRIDE_2
    left_offsets += coordinate * LEFT_STRIDE_2
    output_offsets += coordinate * OUTPUT_STRIDE_2
    coordinate = remaining % DIM_1
    remaining //= DIM_1
    right_offsets += coordinate * RIGHT_STRIDE_1
    left_offsets += coordinate * LEFT_STRIDE_1
    output_offsets += coordinate * OUTPUT_STRIDE_1
    coordinate = remaining % DIM_0
    right_offsets += coordinate * RIGHT_STRIDE_0
    left_offsets += coordinate * LEFT_STRIDE_0
    output_offsets += coordinate * OUTPUT_STRIDE_0

    right = tl.load(right_ptr + right_offsets, mask=active, other=0.0)
    left = tl.load(left_ptr + left_offsets, mask=active, other=0.0)
    if left.dtype == tl.int32:
        result = left + right * right
    else:
        result = left.to(tl.float32) + right.to(tl.float32) * right.to(tl.float32)
    tl.store(
        output_ptr + output_offsets,
        result.to(output_ptr.dtype.element_ty),
        mask=active,
    )
