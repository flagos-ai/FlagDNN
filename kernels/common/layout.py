# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Materialization kernel for reshape, transpose, and slice Graph nodes.

The Python APIs in ``flag_dnn.ops`` express these operations as views.  Native
Graph execution may need a physical output, so lowering supplies one common
logical-shape/stride ABI consumed here.
"""

import triton
import triton.language as tl


@triton.jit
def matrix_transpose_kernel(
    input_ptr,
    output_ptr,
    ROWS: tl.constexpr,
    COLUMNS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Materialize a compact matrix transpose with coalesced reads and writes."""
    row = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    column = tl.program_id(1).to(tl.int64) * BLOCK_SIZE + tl.arange(
        0, BLOCK_SIZE
    )
    active = (row[:, None] < ROWS) & (column[None, :] < COLUMNS)
    value = tl.load(
        input_ptr + row[:, None] * COLUMNS + column[None, :], active, other=0
    )
    tl.store(
        output_ptr + column[:, None] * ROWS + row[None, :],
        tl.trans(value),
        tl.trans(active),
    )


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
