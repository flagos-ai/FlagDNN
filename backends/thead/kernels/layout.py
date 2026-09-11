# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import triton
import triton.language as tl

@triton.jit
def layout_copy_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    INPUT_BASE: tl.constexpr,
    INPUT_DIM_0: tl.constexpr,
    INPUT_DIM_1: tl.constexpr,
    INPUT_DIM_2: tl.constexpr,
    INPUT_DIM_3: tl.constexpr,
    INPUT_DIM_4: tl.constexpr,
    INPUT_DIM_5: tl.constexpr,
    INPUT_DIM_6: tl.constexpr,
    INPUT_DIM_7: tl.constexpr,
    INPUT_STRIDE_0: tl.constexpr,
    INPUT_STRIDE_1: tl.constexpr,
    INPUT_STRIDE_2: tl.constexpr,
    INPUT_STRIDE_3: tl.constexpr,
    INPUT_STRIDE_4: tl.constexpr,
    INPUT_STRIDE_5: tl.constexpr,
    INPUT_STRIDE_6: tl.constexpr,
    INPUT_STRIDE_7: tl.constexpr,
    OUTPUT_DIM_0: tl.constexpr,
    OUTPUT_DIM_1: tl.constexpr,
    OUTPUT_DIM_2: tl.constexpr,
    OUTPUT_DIM_3: tl.constexpr,
    OUTPUT_DIM_4: tl.constexpr,
    OUTPUT_DIM_5: tl.constexpr,
    OUTPUT_DIM_6: tl.constexpr,
    OUTPUT_DIM_7: tl.constexpr,
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
    # Transpose may preserve the physical strides of a view.  When input
    # and output address the same dense range, copy it in physical order.
    if (
        INPUT_BASE == 0
        and INPUT_DIM_0 == OUTPUT_DIM_0
        and INPUT_DIM_1 == OUTPUT_DIM_1
        and INPUT_DIM_2 == OUTPUT_DIM_2
        and INPUT_DIM_3 == OUTPUT_DIM_3
        and INPUT_DIM_4 == OUTPUT_DIM_4
        and INPUT_DIM_5 == OUTPUT_DIM_5
        and INPUT_DIM_6 == OUTPUT_DIM_6
        and INPUT_DIM_7 == OUTPUT_DIM_7
        and INPUT_STRIDE_0 == OUTPUT_STRIDE_0
        and INPUT_STRIDE_1 == OUTPUT_STRIDE_1
        and INPUT_STRIDE_2 == OUTPUT_STRIDE_2
        and INPUT_STRIDE_3 == OUTPUT_STRIDE_3
        and INPUT_STRIDE_4 == OUTPUT_STRIDE_4
        and INPUT_STRIDE_5 == OUTPUT_STRIDE_5
        and INPUT_STRIDE_6 == OUTPUT_STRIDE_6
        and INPUT_STRIDE_7 == OUTPUT_STRIDE_7
        and 1 + (INPUT_DIM_0 - 1) * INPUT_STRIDE_0 + (INPUT_DIM_1 - 1) * INPUT_STRIDE_1 + (INPUT_DIM_2 - 1) * INPUT_STRIDE_2 + (INPUT_DIM_3 - 1) * INPUT_STRIDE_3 + (INPUT_DIM_4 - 1) * INPUT_STRIDE_4 + (INPUT_DIM_5 - 1) * INPUT_STRIDE_5 + (INPUT_DIM_6 - 1) * INPUT_STRIDE_6 + (INPUT_DIM_7 - 1) * INPUT_STRIDE_7 == INPUT_DIM_0 * INPUT_DIM_1 * INPUT_DIM_2 * INPUT_DIM_3 * INPUT_DIM_4 * INPUT_DIM_5 * INPUT_DIM_6 * INPUT_DIM_7
    ):
        offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        values = tl.load(input_ptr + offsets, offsets < n_elements, 0)
        tl.store(output_ptr + offsets, values, offsets < n_elements)
    else:
        logical = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(
            0, BLOCK_SIZE
        )
        active = logical < n_elements
        input_remaining = logical
        output_remaining = logical
        input_offsets = tl.full((BLOCK_SIZE,), INPUT_BASE, dtype=tl.int64)
        output_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

        coordinate = input_remaining % INPUT_DIM_7
        input_remaining //= INPUT_DIM_7
        input_offsets += coordinate * INPUT_STRIDE_7
        coordinate = input_remaining % INPUT_DIM_6
        input_remaining //= INPUT_DIM_6
        input_offsets += coordinate * INPUT_STRIDE_6
        coordinate = input_remaining % INPUT_DIM_5
        input_remaining //= INPUT_DIM_5
        input_offsets += coordinate * INPUT_STRIDE_5
        coordinate = input_remaining % INPUT_DIM_4
        input_remaining //= INPUT_DIM_4
        input_offsets += coordinate * INPUT_STRIDE_4
        coordinate = input_remaining % INPUT_DIM_3
        input_remaining //= INPUT_DIM_3
        input_offsets += coordinate * INPUT_STRIDE_3
        coordinate = input_remaining % INPUT_DIM_2
        input_remaining //= INPUT_DIM_2
        input_offsets += coordinate * INPUT_STRIDE_2
        coordinate = input_remaining % INPUT_DIM_1
        input_remaining //= INPUT_DIM_1
        input_offsets += coordinate * INPUT_STRIDE_1
        input_offsets += (input_remaining % INPUT_DIM_0) * INPUT_STRIDE_0

        coordinate = output_remaining % OUTPUT_DIM_7
        output_remaining //= OUTPUT_DIM_7
        output_offsets += coordinate * OUTPUT_STRIDE_7
        coordinate = output_remaining % OUTPUT_DIM_6
        output_remaining //= OUTPUT_DIM_6
        output_offsets += coordinate * OUTPUT_STRIDE_6
        coordinate = output_remaining % OUTPUT_DIM_5
        output_remaining //= OUTPUT_DIM_5
        output_offsets += coordinate * OUTPUT_STRIDE_5
        coordinate = output_remaining % OUTPUT_DIM_4
        output_remaining //= OUTPUT_DIM_4
        output_offsets += coordinate * OUTPUT_STRIDE_4
        coordinate = output_remaining % OUTPUT_DIM_3
        output_remaining //= OUTPUT_DIM_3
        output_offsets += coordinate * OUTPUT_STRIDE_3
        coordinate = output_remaining % OUTPUT_DIM_2
        output_remaining //= OUTPUT_DIM_2
        output_offsets += coordinate * OUTPUT_STRIDE_2
        coordinate = output_remaining % OUTPUT_DIM_1
        output_remaining //= OUTPUT_DIM_1
        output_offsets += coordinate * OUTPUT_STRIDE_1
        output_offsets += (output_remaining % OUTPUT_DIM_0) * OUTPUT_STRIDE_0

        value = tl.load(input_ptr + input_offsets, mask=active, other=0.0)
        tl.store(output_ptr + output_offsets, value, mask=active)
