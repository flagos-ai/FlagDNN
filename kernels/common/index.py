# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Index generation and byte-preserving concatenation kernels."""

import triton
import triton.language as tl


@triton.jit
def gen_index_kernel(
    output_ptr,
    N_ELEMENTS: tl.constexpr,
    AXIS_EXTENT: tl.constexpr,
    INNER: tl.constexpr,
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
    value = (logical // INNER) % AXIS_EXTENT
    tl.store(
        output_ptr + physical,
        value.to(output_ptr.dtype.element_ty),
        logical < N_ELEMENTS,
    )


@triton.jit
def concatenate_copy_kernel(
    input_ptr,
    output_ptr,
    N_ELEMENTS: tl.constexpr,
    OUTPUT_BASE: tl.constexpr,
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
    BLOCK_SIZE: tl.constexpr,
):
    logical = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(
        0, BLOCK_SIZE
    )
    remaining = logical
    source = tl.zeros((BLOCK_SIZE,), tl.int64)
    destination = tl.full((BLOCK_SIZE,), OUTPUT_BASE, tl.int64)
    coordinate = remaining % DIM_7
    remaining //= DIM_7
    source += coordinate * INPUT_STRIDE_7
    destination += coordinate * OUTPUT_STRIDE_7
    coordinate = remaining % DIM_6
    remaining //= DIM_6
    source += coordinate * INPUT_STRIDE_6
    destination += coordinate * OUTPUT_STRIDE_6
    coordinate = remaining % DIM_5
    remaining //= DIM_5
    source += coordinate * INPUT_STRIDE_5
    destination += coordinate * OUTPUT_STRIDE_5
    coordinate = remaining % DIM_4
    remaining //= DIM_4
    source += coordinate * INPUT_STRIDE_4
    destination += coordinate * OUTPUT_STRIDE_4
    coordinate = remaining % DIM_3
    remaining //= DIM_3
    source += coordinate * INPUT_STRIDE_3
    destination += coordinate * OUTPUT_STRIDE_3
    coordinate = remaining % DIM_2
    remaining //= DIM_2
    source += coordinate * INPUT_STRIDE_2
    destination += coordinate * OUTPUT_STRIDE_2
    coordinate = remaining % DIM_1
    remaining //= DIM_1
    source += coordinate * INPUT_STRIDE_1
    destination += coordinate * OUTPUT_STRIDE_1
    coordinate = remaining % DIM_0
    remaining //= DIM_0
    source += coordinate * INPUT_STRIDE_0
    destination += coordinate * OUTPUT_STRIDE_0
    value = tl.load(input_ptr + source, logical < N_ELEMENTS, other=0)
    tl.store(output_ptr + destination, value, logical < N_ELEMENTS)
