"""Ascend kernels for layout."""

import triton
import triton.language as tl

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


@triton.jit
def layout_copy_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    INPUT_BASE: tl.constexpr,
    ELEMENT_SIZE_BYTES: tl.constexpr,
    LAYOUT_MODE: tl.constexpr,
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
    WORKER_COUNT: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    start = program_id * BLOCK_SIZE
    if LAYOUT_MODE == 1:
        copy_block_size: tl.constexpr = BLOCK_SIZE * 8
        linear_start = program_id * copy_block_size
        worker_stride = WORKER_COUNT * copy_block_size
        while linear_start < n_elements:
            logical = linear_start + tl.arange(0, copy_block_size)
            active = logical < n_elements
            value = tl.load(input_ptr + INPUT_BASE + logical, mask=active, other=0)
            tl.store(output_ptr + logical, value, mask=active)
            linear_start += worker_stride

    if LAYOUT_MODE == 2:
        if OUTPUT_DIM_7 >= 16:
            outer_rows = n_elements // OUTPUT_DIM_7
            row = program_id
            while row < outer_rows:
                output_remaining = row
                output_row_offset = row * 0
                coordinate = output_remaining % OUTPUT_DIM_6
                output_remaining = output_remaining // OUTPUT_DIM_6
                output_row_offset += coordinate * OUTPUT_STRIDE_6
                coordinate = output_remaining % OUTPUT_DIM_5
                output_remaining = output_remaining // OUTPUT_DIM_5
                output_row_offset += coordinate * OUTPUT_STRIDE_5
                coordinate = output_remaining % OUTPUT_DIM_4
                output_remaining = output_remaining // OUTPUT_DIM_4
                output_row_offset += coordinate * OUTPUT_STRIDE_4
                coordinate = output_remaining % OUTPUT_DIM_3
                output_remaining = output_remaining // OUTPUT_DIM_3
                output_row_offset += coordinate * OUTPUT_STRIDE_3
                coordinate = output_remaining % OUTPUT_DIM_2
                output_remaining = output_remaining // OUTPUT_DIM_2
                output_row_offset += coordinate * OUTPUT_STRIDE_2
                coordinate = output_remaining % OUTPUT_DIM_1
                output_remaining = output_remaining // OUTPUT_DIM_1
                output_row_offset += coordinate * OUTPUT_STRIDE_1
                output_row_offset += (output_remaining % OUTPUT_DIM_0) * OUTPUT_STRIDE_0

                column_start = row * 0
                while column_start < OUTPUT_DIM_7:
                    column = column_start + tl.arange(0, BLOCK_SIZE)
                    active = column < OUTPUT_DIM_7
                    output_offsets = output_row_offset + column * OUTPUT_STRIDE_7
                    input_offsets = INPUT_BASE + output_offsets
                    value = tl.load(input_ptr + input_offsets, mask=active, other=0)
                    tl.store(output_ptr + output_offsets, value, mask=active)
                    column_start += BLOCK_SIZE
                row += WORKER_COUNT
        else:
            # Short rows leave almost every lane idle when the full tuning
            # block is used for columns.  Keep row coordinates scalar, use
            # i32 division, and widen only the final tensor offsets.
            column_block: tl.constexpr = 16
            outer_rows = (n_elements // OUTPUT_DIM_7).to(tl.int64)
            row = tl.program_id(0).to(tl.int64)
            while row < outer_rows:
                row_i32 = row.to(tl.int32)
                output_remaining_i32 = row_i32
                coordinate_i32 = output_remaining_i32 % OUTPUT_DIM_6
                output_remaining_i32 //= OUTPUT_DIM_6
                output_row_offset = coordinate_i32.to(tl.int64) * OUTPUT_STRIDE_6
                coordinate_i32 = output_remaining_i32 % OUTPUT_DIM_5
                output_remaining_i32 //= OUTPUT_DIM_5
                output_row_offset += coordinate_i32.to(tl.int64) * OUTPUT_STRIDE_5
                coordinate_i32 = output_remaining_i32 % OUTPUT_DIM_4
                output_remaining_i32 //= OUTPUT_DIM_4
                output_row_offset += coordinate_i32.to(tl.int64) * OUTPUT_STRIDE_4
                coordinate_i32 = output_remaining_i32 % OUTPUT_DIM_3
                output_remaining_i32 //= OUTPUT_DIM_3
                output_row_offset += coordinate_i32.to(tl.int64) * OUTPUT_STRIDE_3
                coordinate_i32 = output_remaining_i32 % OUTPUT_DIM_2
                output_remaining_i32 //= OUTPUT_DIM_2
                output_row_offset += coordinate_i32.to(tl.int64) * OUTPUT_STRIDE_2
                coordinate_i32 = output_remaining_i32 % OUTPUT_DIM_1
                output_remaining_i32 //= OUTPUT_DIM_1
                output_row_offset += coordinate_i32.to(tl.int64) * OUTPUT_STRIDE_1
                coordinate_i32 = output_remaining_i32 % OUTPUT_DIM_0
                output_row_offset += coordinate_i32.to(tl.int64) * OUTPUT_STRIDE_0

                columns = tl.arange(0, column_block)
                active = columns < OUTPUT_DIM_7
                output_offsets = output_row_offset + columns * OUTPUT_STRIDE_7
                value = tl.load(
                    input_ptr + INPUT_BASE + output_offsets,
                    mask=active,
                    other=0,
                )
                tl.store(output_ptr + output_offsets, value, mask=active)
                row += WORKER_COUNT

    if LAYOUT_MODE != 0:
        start = n_elements
    while start < n_elements:
        logical = start + tl.arange(0, BLOCK_SIZE)
        active = logical < n_elements
        input_remaining = logical
        output_remaining = logical
        input_offsets = tl.full((BLOCK_SIZE,), INPUT_BASE, dtype=tl.int64)
        output_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

        coordinate = input_remaining % INPUT_DIM_7
        input_remaining = input_remaining // INPUT_DIM_7
        input_offsets += coordinate * INPUT_STRIDE_7
        coordinate = input_remaining % INPUT_DIM_6
        input_remaining = input_remaining // INPUT_DIM_6
        input_offsets += coordinate * INPUT_STRIDE_6
        coordinate = input_remaining % INPUT_DIM_5
        input_remaining = input_remaining // INPUT_DIM_5
        input_offsets += coordinate * INPUT_STRIDE_5
        coordinate = input_remaining % INPUT_DIM_4
        input_remaining = input_remaining // INPUT_DIM_4
        input_offsets += coordinate * INPUT_STRIDE_4
        coordinate = input_remaining % INPUT_DIM_3
        input_remaining = input_remaining // INPUT_DIM_3
        input_offsets += coordinate * INPUT_STRIDE_3
        coordinate = input_remaining % INPUT_DIM_2
        input_remaining = input_remaining // INPUT_DIM_2
        input_offsets += coordinate * INPUT_STRIDE_2
        coordinate = input_remaining % INPUT_DIM_1
        input_remaining = input_remaining // INPUT_DIM_1
        input_offsets += coordinate * INPUT_STRIDE_1
        input_offsets += (input_remaining % INPUT_DIM_0) * INPUT_STRIDE_0

        coordinate = output_remaining % OUTPUT_DIM_7
        output_remaining = output_remaining // OUTPUT_DIM_7
        output_offsets += coordinate * OUTPUT_STRIDE_7
        coordinate = output_remaining % OUTPUT_DIM_6
        output_remaining = output_remaining // OUTPUT_DIM_6
        output_offsets += coordinate * OUTPUT_STRIDE_6
        coordinate = output_remaining % OUTPUT_DIM_5
        output_remaining = output_remaining // OUTPUT_DIM_5
        output_offsets += coordinate * OUTPUT_STRIDE_5
        coordinate = output_remaining % OUTPUT_DIM_4
        output_remaining = output_remaining // OUTPUT_DIM_4
        output_offsets += coordinate * OUTPUT_STRIDE_4
        coordinate = output_remaining % OUTPUT_DIM_3
        output_remaining = output_remaining // OUTPUT_DIM_3
        output_offsets += coordinate * OUTPUT_STRIDE_3
        coordinate = output_remaining % OUTPUT_DIM_2
        output_remaining = output_remaining // OUTPUT_DIM_2
        output_offsets += coordinate * OUTPUT_STRIDE_2
        coordinate = output_remaining % OUTPUT_DIM_1
        output_remaining = output_remaining // OUTPUT_DIM_1
        output_offsets += coordinate * OUTPUT_STRIDE_1
        output_offsets += (output_remaining % OUTPUT_DIM_0) * OUTPUT_STRIDE_0

        # The compiler gives layout kernels an integer pointer signature whose
        # width is independently bound to ELEMENT_SIZE_BYTES. Loading through
        # that raw signature preserves NaN payloads and signed zero bits.
        value = tl.load(input_ptr + input_offsets, mask=active, other=0)
        tl.store(output_ptr + output_offsets, value, mask=active)
        start += WORKER_COUNT * BLOCK_SIZE


# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0


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
    column = tl.program_id(1).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
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
def layout_strided_copy_kernel(
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
    logical = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
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
