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

"""Ascend-only prepared kernels for dense utility operations."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import triton
import triton.language as tl
from triton.language.extra.cann.extension import insert_slice

from flag_dnn.graph.device import is_runtime_device_tensor
from flag_dnn.utils import triton_lang_extension as tle
from flag_dnn.utils.libentry import libentry

from .binary import (
    get_add_block_size,
    get_vector_core_count,
)

_FLOAT_DTYPES = {"float16", "bfloat16", "float32"}


def _numel(shape: Sequence[int]) -> int:
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def _can_use_aligned_loop(n_elements: int, block_size: int) -> bool:
    return n_elements % block_size == 0


def get_dense_utility_block_size(
    n_elements: int,
    dtype: Any,
    device: Any,
    *,
    input_count: int,
) -> int:
    """Choose a utility tile without exceeding Ascend unified-buffer usage."""
    block_size = get_add_block_size(n_elements, dtype, device)
    is_float32 = "float32" in str(dtype)
    if input_count >= 3:
        return min(block_size, 2048 if is_float32 else 4096)
    return min(block_size, 4096 if is_float32 else 8192)


def _row_tile_geometry(
    row_size: int,
    *,
    max_elements: int,
) -> tuple[int, int]:
    block_columns = min(triton.next_power_of_2(row_size), max_elements)
    rows_per_program = max(1, max_elements // block_columns)
    return rows_per_program, block_columns


def get_utility_copy_block_size(n_elements: int, device: Any) -> int:
    vector_cores = get_vector_core_count(device)
    per_core = triton.cdiv(n_elements, vector_cores)
    return min(2048, max(256, triton.next_power_of_2(per_core)))


@libentry()
@triton.jit
def gen_index_tiled_kernel(
    output_ptr,
    N_ELEMENTS: tl.constexpr,
    AXIS_SIZE: tl.constexpr,
    INNER_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tle.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS
    axis_index = (offsets // INNER_SIZE) % AXIS_SIZE
    tl.store(
        output_ptr + offsets,
        axis_index.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


@libentry()
@triton.jit
def concatenate_copy_kernel(
    input_ptr,
    output_ptr,
    N_ELEMENTS: tl.constexpr,
    INPUT_ROW: tl.constexpr,
    OUTPUT_ROW: tl.constexpr,
    AXIS_OFFSET: tl.constexpr,
    OUTER_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tle.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS
    if OUTER_SIZE == 1:
        output_offsets = AXIS_OFFSET + offsets
    else:
        row = offsets // INPUT_ROW
        within_row = offsets % INPUT_ROW
        output_offsets = row * OUTPUT_ROW + AXIS_OFFSET + within_row
    value = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + output_offsets, value, mask=mask)


@libentry()
@triton.jit
def concatenate_copy_rows_kernel(
    input_ptr,
    output_ptr,
    OUTER_SIZE: tl.constexpr,
    INPUT_ROW: tl.constexpr,
    OUTPUT_ROW: tl.constexpr,
    AXIS_OFFSET: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_COLUMNS: tl.constexpr,
):
    column_offsets = (
        tle.program_id(0) * BLOCK_COLUMNS
        + tl.arange(0, BLOCK_COLUMNS)[None, :]
    )
    row_offsets = (
        tle.program_id(1) * ROWS_PER_PROGRAM
        + tl.arange(0, ROWS_PER_PROGRAM)[:, None]
    )
    input_offsets = row_offsets * INPUT_ROW + column_offsets
    output_offsets = row_offsets * OUTPUT_ROW + AXIS_OFFSET + column_offsets
    mask = (row_offsets < OUTER_SIZE) & (column_offsets < INPUT_ROW)
    value = tl.load(input_ptr + input_offsets, mask=mask)
    tl.store(output_ptr + output_offsets, value, mask=mask)


@libentry()
@triton.jit
def concatenate2_segmented_rows_kernel(
    x0_ptr,
    x1_ptr,
    output_ptr,
    OUTER_SIZE: tl.constexpr,
    INPUT0_ROW: tl.constexpr,
    INPUT1_ROW: tl.constexpr,
    OUTPUT_ROW: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_COLUMNS: tl.constexpr,
):
    column_offsets = (
        tle.program_id(0) * BLOCK_COLUMNS
        + tl.arange(0, BLOCK_COLUMNS)[None, :]
    )
    row_offsets = (
        tle.program_id(1) * ROWS_PER_PROGRAM
        + tl.arange(0, ROWS_PER_PROGRAM)[:, None]
    )
    row_mask = row_offsets < OUTER_SIZE
    mask0 = row_mask & (column_offsets < INPUT0_ROW)
    mask1 = row_mask & (column_offsets < INPUT1_ROW)
    input0_offsets = row_offsets * INPUT0_ROW + column_offsets
    input1_offsets = row_offsets * INPUT1_ROW + column_offsets
    output_base = row_offsets * OUTPUT_ROW
    value0 = tl.load(x0_ptr + input0_offsets, mask=mask0)
    value1 = tl.load(x1_ptr + input1_offsets, mask=mask1)
    tl.store(output_ptr + output_base + column_offsets, value0, mask=mask0)
    tl.store(
        output_ptr + output_base + INPUT0_ROW + column_offsets,
        value1,
        mask=mask1,
    )


@libentry()
@triton.jit
def concatenate3_segmented_rows_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    output_ptr,
    OUTER_SIZE: tl.constexpr,
    INPUT0_ROW: tl.constexpr,
    INPUT1_ROW: tl.constexpr,
    INPUT2_ROW: tl.constexpr,
    OUTPUT_ROW: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_COLUMNS: tl.constexpr,
):
    column_offsets = (
        tle.program_id(0) * BLOCK_COLUMNS
        + tl.arange(0, BLOCK_COLUMNS)[None, :]
    )
    row_offsets = (
        tle.program_id(1) * ROWS_PER_PROGRAM
        + tl.arange(0, ROWS_PER_PROGRAM)[:, None]
    )
    row_mask = row_offsets < OUTER_SIZE
    mask0 = row_mask & (column_offsets < INPUT0_ROW)
    mask1 = row_mask & (column_offsets < INPUT1_ROW)
    mask2 = row_mask & (column_offsets < INPUT2_ROW)
    input0_offsets = row_offsets * INPUT0_ROW + column_offsets
    input1_offsets = row_offsets * INPUT1_ROW + column_offsets
    input2_offsets = row_offsets * INPUT2_ROW + column_offsets
    output_base = row_offsets * OUTPUT_ROW
    value0 = tl.load(x0_ptr + input0_offsets, mask=mask0)
    value1 = tl.load(x1_ptr + input1_offsets, mask=mask1)
    value2 = tl.load(x2_ptr + input2_offsets, mask=mask2)
    tl.store(output_ptr + output_base + column_offsets, value0, mask=mask0)
    tl.store(
        output_ptr + output_base + INPUT0_ROW + column_offsets,
        value1,
        mask=mask1,
    )
    tl.store(
        output_ptr + output_base + INPUT0_ROW + INPUT1_ROW + column_offsets,
        value2,
        mask=mask2,
    )


@libentry()
@triton.jit
def concatenate2_insert_slice_rows_kernel(
    x0_ptr,
    x1_ptr,
    output_ptr,
    OUTER_SIZE: tl.constexpr,
    INPUT0_ROW: tl.constexpr,
    INPUT1_ROW: tl.constexpr,
    OUTPUT_ROW: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    row_start = tle.program_id(0) * ROWS_PER_PROGRAM
    rows = tl.arange(0, ROWS_PER_PROGRAM)[:, None]
    columns0 = tl.arange(0, INPUT0_ROW)[None, :]
    value0 = tl.load(x0_ptr + (row_start + rows) * INPUT0_ROW + columns0)
    columns1 = tl.arange(0, INPUT1_ROW)[None, :]
    value1 = tl.load(x1_ptr + (row_start + rows) * INPUT1_ROW + columns1)
    output_tile = tl.zeros(
        (ROWS_PER_PROGRAM, OUTPUT_ROW),
        x0_ptr.dtype.element_ty,
    )
    output_tile = insert_slice(
        output_tile,
        value0,
        offsets=(0, 0),
        sizes=(ROWS_PER_PROGRAM, INPUT0_ROW),
        strides=(1, 1),
    )
    output_tile = insert_slice(
        output_tile,
        value1,
        offsets=(0, INPUT0_ROW),
        sizes=(ROWS_PER_PROGRAM, INPUT1_ROW),
        strides=(1, 1),
    )
    output_block = tl.make_block_ptr(
        base=output_ptr,
        shape=(OUTER_SIZE, OUTPUT_ROW),
        strides=(OUTPUT_ROW, 1),
        offsets=(row_start.to(tl.int32), 0),
        block_shape=(ROWS_PER_PROGRAM, OUTPUT_ROW),
        order=(1, 0),
    )
    tl.store(output_block, output_tile)


@libentry()
@triton.jit
def concatenate3_insert_slice_rows_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    output_ptr,
    OUTER_SIZE: tl.constexpr,
    INPUT0_ROW: tl.constexpr,
    INPUT1_ROW: tl.constexpr,
    INPUT2_ROW: tl.constexpr,
    OUTPUT_ROW: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    row_start = tle.program_id(0) * ROWS_PER_PROGRAM
    rows = tl.arange(0, ROWS_PER_PROGRAM)[:, None]
    columns0 = tl.arange(0, INPUT0_ROW)[None, :]
    value0 = tl.load(x0_ptr + (row_start + rows) * INPUT0_ROW + columns0)
    columns1 = tl.arange(0, INPUT1_ROW)[None, :]
    value1 = tl.load(x1_ptr + (row_start + rows) * INPUT1_ROW + columns1)
    columns2 = tl.arange(0, INPUT2_ROW)[None, :]
    value2 = tl.load(x2_ptr + (row_start + rows) * INPUT2_ROW + columns2)
    output_tile = tl.zeros(
        (ROWS_PER_PROGRAM, OUTPUT_ROW),
        x0_ptr.dtype.element_ty,
    )
    output_tile = insert_slice(
        output_tile,
        value0,
        offsets=(0, 0),
        sizes=(ROWS_PER_PROGRAM, INPUT0_ROW),
        strides=(1, 1),
    )
    output_tile = insert_slice(
        output_tile,
        value1,
        offsets=(0, INPUT0_ROW),
        sizes=(ROWS_PER_PROGRAM, INPUT1_ROW),
        strides=(1, 1),
    )
    output_tile = insert_slice(
        output_tile,
        value2,
        offsets=(0, INPUT0_ROW + INPUT1_ROW),
        sizes=(ROWS_PER_PROGRAM, INPUT2_ROW),
        strides=(1, 1),
    )
    output_block = tl.make_block_ptr(
        base=output_ptr,
        shape=(OUTER_SIZE, OUTPUT_ROW),
        strides=(OUTPUT_ROW, 1),
        offsets=(row_start.to(tl.int32), 0),
        block_shape=(ROWS_PER_PROGRAM, OUTPUT_ROW),
        order=(1, 0),
    )
    tl.store(output_block, output_tile)


@triton.jit
def _copy_concatenate_blocks(
    input_ptr,
    output_ptr,
    input_first_block,
    output_first_block,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    for local_block in range(0, BLOCKS_PER_PROGRAM):
        input_offsets = (input_first_block + local_block) * 8192 + tl.arange(
            0, 8192
        )
        output_offsets = (output_first_block + local_block) * 8192 + tl.arange(
            0, 8192
        )
        value = tl.load(input_ptr + input_offsets)
        tl.store(output_ptr + output_offsets, value)


@libentry()
@triton.jit
def concatenate2_axis0_262144_786432_kernel(
    x0_ptr,
    x1_ptr,
    output_ptr,
):
    pid = tle.program_id(0)
    if pid < 12:
        if pid < 8:
            first_block = pid * 3
            _copy_concatenate_blocks(
                x0_ptr,
                output_ptr,
                first_block,
                first_block,
                3,
            )
        else:
            first_block = 24 + (pid - 8) * 2
            _copy_concatenate_blocks(
                x0_ptr,
                output_ptr,
                first_block,
                first_block,
                2,
            )
    else:
        input_pid = pid - 12
        if input_pid < 24:
            first_block = input_pid * 3
            _copy_concatenate_blocks(
                x1_ptr,
                output_ptr,
                first_block,
                32 + first_block,
                3,
            )
        else:
            first_block = 72 + (input_pid - 24) * 2
            _copy_concatenate_blocks(
                x1_ptr,
                output_ptr,
                first_block,
                32 + first_block,
                2,
            )


@libentry()
@triton.jit
def gen_index_rows_kernel(
    output_ptr,
    OUTER_SIZE: tl.constexpr,
    AXIS_SIZE: tl.constexpr,
    INNER_SIZE: tl.constexpr,
    ROW_SIZE: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_COLUMNS: tl.constexpr,
):
    column_offsets = (
        tle.program_id(0) * BLOCK_COLUMNS
        + tl.arange(0, BLOCK_COLUMNS)[None, :]
    )
    row_offsets = (
        tle.program_id(1) * ROWS_PER_PROGRAM
        + tl.arange(0, ROWS_PER_PROGRAM)[:, None]
    )
    offsets = row_offsets * ROW_SIZE + column_offsets
    mask = (row_offsets < OUTER_SIZE) & (column_offsets < ROW_SIZE)
    if INNER_SIZE == 1:
        axis_index = column_offsets
    elif INNER_SIZE >= BLOCK_COLUMNS and INNER_SIZE % BLOCK_COLUMNS == 0:
        axis_index = (
            tle.program_id(0) * BLOCK_COLUMNS // INNER_SIZE
        ) % AXIS_SIZE
    else:
        axis_index = (column_offsets // INNER_SIZE) % AXIS_SIZE
    tl.store(
        output_ptr + offsets,
        axis_index.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


@libentry()
@triton.jit
def concatenate2_rows_kernel(
    x0_ptr,
    x1_ptr,
    output_ptr,
    OUTER_SIZE: tl.constexpr,
    INPUT0_ROW: tl.constexpr,
    INPUT1_ROW: tl.constexpr,
    OUTPUT_ROW: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_COLUMNS: tl.constexpr,
):
    column_offsets = (
        tle.program_id(0) * BLOCK_COLUMNS
        + tl.arange(0, BLOCK_COLUMNS)[None, :]
    )
    row_offsets = (
        tle.program_id(1) * ROWS_PER_PROGRAM
        + tl.arange(0, ROWS_PER_PROGRAM)[:, None]
    )
    output_offsets = row_offsets * OUTPUT_ROW + column_offsets
    valid = (row_offsets < OUTER_SIZE) & (column_offsets < OUTPUT_ROW)
    use0 = column_offsets < INPUT0_ROW
    offset0 = row_offsets * INPUT0_ROW + column_offsets
    offset1 = row_offsets * INPUT1_ROW + column_offsets - INPUT0_ROW
    value0 = tl.load(x0_ptr + offset0, mask=valid & use0, other=0.0)
    value1 = tl.load(x1_ptr + offset1, mask=valid & (~use0), other=0.0)
    tl.store(
        output_ptr + output_offsets,
        tl.where(use0, value0, value1),
        mask=valid,
    )


@libentry()
@triton.jit
def concatenate3_rows_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    output_ptr,
    OUTER_SIZE: tl.constexpr,
    INPUT0_ROW: tl.constexpr,
    INPUT1_ROW: tl.constexpr,
    INPUT2_ROW: tl.constexpr,
    OUTPUT_ROW: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_COLUMNS: tl.constexpr,
):
    column_offsets = (
        tle.program_id(0) * BLOCK_COLUMNS
        + tl.arange(0, BLOCK_COLUMNS)[None, :]
    )
    row_offsets = (
        tle.program_id(1) * ROWS_PER_PROGRAM
        + tl.arange(0, ROWS_PER_PROGRAM)[:, None]
    )
    output_offsets = row_offsets * OUTPUT_ROW + column_offsets
    valid = (row_offsets < OUTER_SIZE) & (column_offsets < OUTPUT_ROW)
    input1_end = INPUT0_ROW + INPUT1_ROW
    use0 = column_offsets < INPUT0_ROW
    use1 = (column_offsets >= INPUT0_ROW) & (column_offsets < input1_end)
    use2 = column_offsets >= input1_end
    offset0 = row_offsets * INPUT0_ROW + column_offsets
    offset1 = row_offsets * INPUT1_ROW + column_offsets - INPUT0_ROW
    offset2 = row_offsets * INPUT2_ROW + column_offsets - input1_end
    value0 = tl.load(x0_ptr + offset0, mask=valid & use0, other=0.0)
    value1 = tl.load(x1_ptr + offset1, mask=valid & use1, other=0.0)
    value2 = tl.load(x2_ptr + offset2, mask=valid & use2, other=0.0)
    tl.store(
        output_ptr + output_offsets,
        tl.where(use0, value0, tl.where(use1, value1, value2)),
        mask=valid,
    )


@libentry()
@triton.jit
def gen_index_aligned_core_loop_kernel(
    output_ptr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    AXIS_SIZE: tl.constexpr,
    INNER_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    first_block = tle.program_id(0) * BLOCKS_PER_PROGRAM
    for local_block in range(0, BLOCKS_PER_PROGRAM):
        offsets = (first_block + local_block) * BLOCK_SIZE + tl.arange(
            0, BLOCK_SIZE
        )
        axis_index = (offsets // INNER_SIZE) % AXIS_SIZE
        tl.store(
            output_ptr + offsets,
            axis_index.to(output_ptr.dtype.element_ty),
        )


@libentry()
@triton.jit
def gen_index_core_loop_kernel(
    output_ptr,
    N_ELEMENTS: tl.constexpr,
    AXIS_SIZE: tl.constexpr,
    INNER_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_programs = tle.num_programs(0)
    elements_per_program = tl.cdiv(N_ELEMENTS, num_programs)
    chunk_size = tl.cdiv(elements_per_program, 256) * 256
    chunk_start = pid * chunk_size
    chunk_end = tl.minimum(chunk_start + chunk_size, N_ELEMENTS)
    num_blocks = tl.cdiv(chunk_size, BLOCK_SIZE)

    for block_idx in range(0, num_blocks):
        offsets = (
            chunk_start + block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        )
        mask = offsets < chunk_end
        axis_index = (offsets // INNER_SIZE) % AXIS_SIZE
        tl.store(
            output_ptr + offsets,
            axis_index.to(output_ptr.dtype.element_ty),
            mask=mask,
        )


@triton.jit
def _concat2_values(
    x0_ptr,
    x1_ptr,
    offsets,
    valid,
    INPUT0_ROW: tl.constexpr,
    INPUT1_ROW: tl.constexpr,
    OUTPUT_ROW: tl.constexpr,
):
    row = offsets // OUTPUT_ROW
    within_row = offsets % OUTPUT_ROW
    use0 = within_row < INPUT0_ROW
    offset0 = row * INPUT0_ROW + within_row
    offset1 = row * INPUT1_ROW + within_row - INPUT0_ROW
    value0 = tl.load(x0_ptr + offset0, mask=valid & use0, other=0.0)
    value1 = tl.load(x1_ptr + offset1, mask=valid & (~use0), other=0.0)
    return tl.where(use0, value0, value1)


@triton.jit
def _concat3_values(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    offsets,
    valid,
    INPUT0_ROW: tl.constexpr,
    INPUT1_ROW: tl.constexpr,
    INPUT2_ROW: tl.constexpr,
    OUTPUT_ROW: tl.constexpr,
):
    row = offsets // OUTPUT_ROW
    within_row = offsets % OUTPUT_ROW
    input1_end = INPUT0_ROW + INPUT1_ROW
    use0 = within_row < INPUT0_ROW
    use1 = (within_row >= INPUT0_ROW) & (within_row < input1_end)
    use2 = within_row >= input1_end
    offset0 = row * INPUT0_ROW + within_row
    offset1 = row * INPUT1_ROW + within_row - INPUT0_ROW
    offset2 = row * INPUT2_ROW + within_row - input1_end
    value0 = tl.load(x0_ptr + offset0, mask=valid & use0, other=0.0)
    value1 = tl.load(x1_ptr + offset1, mask=valid & use1, other=0.0)
    value2 = tl.load(x2_ptr + offset2, mask=valid & use2, other=0.0)
    return tl.where(use0, value0, tl.where(use1, value1, value2))


@libentry()
@triton.jit
def concatenate2_aligned_core_loop_kernel(
    x0_ptr,
    x1_ptr,
    output_ptr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    INPUT0_ROW: tl.constexpr,
    INPUT1_ROW: tl.constexpr,
    OUTPUT_ROW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    first_block = tle.program_id(0) * BLOCKS_PER_PROGRAM
    for local_block in range(0, BLOCKS_PER_PROGRAM):
        offsets = (first_block + local_block) * BLOCK_SIZE + tl.arange(
            0, BLOCK_SIZE
        )
        valid = offsets >= 0
        value = _concat2_values(
            x0_ptr,
            x1_ptr,
            offsets,
            valid,
            INPUT0_ROW,
            INPUT1_ROW,
            OUTPUT_ROW,
        )
        tl.store(output_ptr + offsets, value)


@libentry()
@triton.jit
def concatenate2_core_loop_kernel(
    x0_ptr,
    x1_ptr,
    output_ptr,
    N_ELEMENTS: tl.constexpr,
    INPUT0_ROW: tl.constexpr,
    INPUT1_ROW: tl.constexpr,
    OUTPUT_ROW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_programs = tle.num_programs(0)
    elements_per_program = tl.cdiv(N_ELEMENTS, num_programs)
    chunk_size = tl.cdiv(elements_per_program, 256) * 256
    chunk_start = pid * chunk_size
    chunk_end = tl.minimum(chunk_start + chunk_size, N_ELEMENTS)
    num_blocks = tl.cdiv(chunk_size, BLOCK_SIZE)

    for block_idx in range(0, num_blocks):
        offsets = (
            chunk_start + block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        )
        valid = offsets < chunk_end
        value = _concat2_values(
            x0_ptr,
            x1_ptr,
            offsets,
            valid,
            INPUT0_ROW,
            INPUT1_ROW,
            OUTPUT_ROW,
        )
        tl.store(output_ptr + offsets, value, mask=valid)


@libentry()
@triton.jit
def concatenate3_aligned_core_loop_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    output_ptr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    INPUT0_ROW: tl.constexpr,
    INPUT1_ROW: tl.constexpr,
    INPUT2_ROW: tl.constexpr,
    OUTPUT_ROW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    first_block = tle.program_id(0) * BLOCKS_PER_PROGRAM
    for local_block in range(0, BLOCKS_PER_PROGRAM):
        offsets = (first_block + local_block) * BLOCK_SIZE + tl.arange(
            0, BLOCK_SIZE
        )
        valid = offsets >= 0
        value = _concat3_values(
            x0_ptr,
            x1_ptr,
            x2_ptr,
            offsets,
            valid,
            INPUT0_ROW,
            INPUT1_ROW,
            INPUT2_ROW,
            OUTPUT_ROW,
        )
        tl.store(output_ptr + offsets, value)


@libentry()
@triton.jit
def concatenate3_core_loop_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    output_ptr,
    N_ELEMENTS: tl.constexpr,
    INPUT0_ROW: tl.constexpr,
    INPUT1_ROW: tl.constexpr,
    INPUT2_ROW: tl.constexpr,
    OUTPUT_ROW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_programs = tle.num_programs(0)
    elements_per_program = tl.cdiv(N_ELEMENTS, num_programs)
    chunk_size = tl.cdiv(elements_per_program, 256) * 256
    chunk_start = pid * chunk_size
    chunk_end = tl.minimum(chunk_start + chunk_size, N_ELEMENTS)
    num_blocks = tl.cdiv(chunk_size, BLOCK_SIZE)

    for block_idx in range(0, num_blocks):
        offsets = (
            chunk_start + block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        )
        valid = offsets < chunk_end
        value = _concat3_values(
            x0_ptr,
            x1_ptr,
            x2_ptr,
            offsets,
            valid,
            INPUT0_ROW,
            INPUT1_ROW,
            INPUT2_ROW,
            OUTPUT_ROW,
        )
        tl.store(output_ptr + offsets, value, mask=valid)


def prepare_dense_gen_index(
    *,
    attrs: dict[str, Any],
    input_specs: Sequence[Any],
    default_run_fn: Any,
) -> Optional[Any]:
    if len(input_specs) != 1:
        return None
    input_spec = input_specs[0]
    shape = tuple(input_spec.shape)
    if (
        not shape
        or not all(isinstance(dim, int) for dim in shape)
        or input_spec.dtype not in _FLOAT_DTYPES
    ):
        return None
    axis = attrs.get("axis")
    if not isinstance(axis, int) or axis < 0 or axis >= len(shape):
        return None

    from flag_dnn.graph.prepared import (
        PreparedSingleKernelRunSpec,
        PreparedSingleKernelSpec,
        get_prepared_output,
        make_single_kernel_run_fn,
        runtime_tensor_checks_from_specs,
    )
    from flag_dnn.graph.tensor import torch_dtype

    try:
        output_dtype = torch_dtype(
            attrs.get("compute_data_type") or input_spec.dtype
        )
    except ValueError:
        return None
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None

    static_shape = tuple(int(dim) for dim in shape)
    n_elements = _numel(static_shape)
    axis_size = static_shape[axis]
    inner_size = _numel(static_shape[axis + 1 :])
    if n_elements <= 0 or axis_size <= 0:
        return None
    input_checks = runtime_tensor_checks_from_specs(
        input_specs,
        (0,),
        require_shape=True,
        require_stride=False,
        require_dtype=True,
    )
    if input_checks is None:
        return None

    row_size = axis_size * inner_size
    outer_size = n_elements // row_size
    block_columns = min(32, triton.next_power_of_2(row_size))
    max_rows = 256
    rows_per_program = min(max_rows, triton.next_power_of_2(outer_size))
    static_grid = (
        triton.cdiv(row_size, block_columns),
        triton.cdiv(outer_size, rows_per_program),
    )
    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def output_factory(inputs: Sequence[Any]) -> torch.Tensor:
        source = inputs[0]
        key = (
            source.device.type,
            source.device.index,
            output_dtype,
            static_shape,
        )
        return get_prepared_output(
            output_cache,
            key,
            lambda: torch.empty(
                static_shape,
                device=source.device,
                dtype=output_dtype,
            ),
        )

    def runtime_args(
        _inputs: Sequence[Any], output: torch.Tensor
    ) -> tuple[Any, ...]:
        return (output,)

    def extra_check(inputs: Sequence[Any]) -> bool:
        source = inputs[0]
        return isinstance(source, torch.Tensor) and is_runtime_device_tensor(
            source
        )

    def grid(_meta: dict[str, Any]) -> tuple[int, ...]:
        return static_grid

    constexpr_kwargs = {
        "OUTER_SIZE": outer_size,
        "AXIS_SIZE": axis_size,
        "INNER_SIZE": inner_size,
        "ROW_SIZE": row_size,
        "ROWS_PER_PROGRAM": rows_per_program,
        "BLOCK_COLUMNS": block_columns,
        "num_warps": 4,
        "num_stages": 1,
    }

    def build_cached_call(
        constexprs: dict[str, Any],
    ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        del constexprs
        return (*static_grid, 1), (
            outer_size,
            axis_size,
            inner_size,
            row_size,
            rows_per_program,
            block_columns,
        )

    return make_single_kernel_run_fn(
        PreparedSingleKernelRunSpec(
            kernel=PreparedSingleKernelSpec(
                kernel=gen_index_rows_kernel,
                grid=grid,
                static_args=(),
                constexpr_kwargs=constexpr_kwargs,
                build_cached_call=build_cached_call,
            ),
            input_checks=input_checks,
            output_factory=output_factory,
            runtime_args=runtime_args,
            extra_check=extra_check,
            validate_inputs=bool(attrs.get("_validate_inputs", True)),
        ),
        default_run_fn,
    )


def prepare_dense_concatenate(
    *,
    attrs: dict[str, Any],
    input_specs: Sequence[Any],
    default_run_fn: Any,
) -> Optional[Any]:
    if len(input_specs) not in (2, 3):
        return None
    first_spec = input_specs[0]
    first_shape = tuple(first_spec.shape)
    if (
        not first_shape
        or not all(isinstance(dim, int) for dim in first_shape)
        or first_spec.dtype not in _FLOAT_DTYPES
        or not first_spec.contiguous
        or first_spec.stride is None
    ):
        return None
    axis = attrs.get("axis")
    if not isinstance(axis, int) or axis < 0 or axis >= len(first_shape):
        return None

    static_shapes: list[tuple[int, ...]] = []
    for spec in input_specs:
        shape = tuple(spec.shape)
        if (
            not all(isinstance(dim, int) for dim in shape)
            or len(shape) != len(first_shape)
            or spec.dtype != first_spec.dtype
            or not spec.contiguous
            or spec.stride is None
        ):
            return None
        static_shape = tuple(int(dim) for dim in shape)
        if any(
            static_shape[index] != int(first_shape[index])
            for index in range(len(first_shape))
            if index != axis
        ):
            return None
        static_shapes.append(static_shape)

    from flag_dnn.graph.prepared import (
        PreparedKernelPipelineSpec,
        PreparedPipelineStepSpec,
        PreparedSingleKernelRunSpec,
        PreparedSingleKernelSpec,
        get_prepared_output,
        make_kernel_pipeline_launcher,
        make_kernel_pipeline_run_fn,
        make_single_kernel_run_fn,
        runtime_tensor_checks_from_specs,
        runtime_tensor_checks_pass,
    )
    from flag_dnn.graph.tensor import torch_dtype

    output_shape = list(static_shapes[0])
    output_shape[axis] = sum(shape[axis] for shape in static_shapes)
    static_output_shape = tuple(output_shape)
    n_elements = _numel(static_output_shape)
    inner_size = _numel(static_output_shape[axis + 1 :])
    input_rows = tuple(shape[axis] * inner_size for shape in static_shapes)
    output_row = sum(input_rows)
    if n_elements <= 0 or output_row <= 0:
        return None

    input_indices = tuple(range(len(input_specs)))
    input_checks = runtime_tensor_checks_from_specs(
        input_specs,
        input_indices,
        require_shape=True,
        require_stride=True,
        require_dtype=True,
    )
    if input_checks is None:
        return None

    output_dtype = torch_dtype(first_spec.dtype)
    outer_size = n_elements // output_row
    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def output_factory(inputs: Sequence[Any]) -> torch.Tensor:
        source = inputs[0]
        key = (
            source.device.type,
            source.device.index,
            output_dtype,
            static_output_shape,
        )
        return get_prepared_output(
            output_cache,
            key,
            lambda: torch.empty(
                static_output_shape,
                device=source.device,
                dtype=output_dtype,
            ),
        )

    def extra_check(inputs: Sequence[Any]) -> bool:
        if not all(
            isinstance(item, torch.Tensor) and is_runtime_device_tensor(item)
            for item in inputs
        ):
            return False
        first = inputs[0]
        return all(item.device == first.device for item in inputs[1:])

    static_grid: tuple[Any, ...]
    constexpr_kwargs: dict[str, Any]
    cached_args: tuple[Any, ...]
    launch_grid: tuple[Any, ...]
    exact_cached_args: tuple[Any, ...]
    if (
        len(input_specs) == 2
        and outer_size == 1
        and input_rows == (262144, 786432)
    ):
        static_grid = (48,)

        def contiguous_grid(
            _meta: dict[str, Any],
        ) -> tuple[int, ...]:
            return static_grid

        def contiguous_runtime_args(
            inputs: Sequence[Any], output: torch.Tensor
        ) -> tuple[Any, ...]:
            return (*inputs, output)

        def contiguous_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (48, 1, 1), ()

        return make_single_kernel_run_fn(
            PreparedSingleKernelRunSpec(
                kernel=PreparedSingleKernelSpec(
                    kernel=concatenate2_axis0_262144_786432_kernel,
                    grid=contiguous_grid,
                    static_args=(),
                    constexpr_kwargs={
                        "num_warps": 4,
                        "num_stages": 1,
                    },
                    build_cached_call=contiguous_cached_call,
                ),
                input_checks=input_checks,
                output_factory=output_factory,
                runtime_args=contiguous_runtime_args,
                extra_check=extra_check,
                validate_inputs=bool(attrs.get("_validate_inputs", True)),
            ),
            default_run_fn,
        )

    exact_rows_per_program: Optional[int] = None
    if (
        len(input_specs) == 3
        and outer_size == 1024
        and input_rows == (128, 64, 32)
    ):
        exact_rows_per_program = 32
    elif (
        len(input_specs) == 2
        and outer_size == 32768
        and input_rows == (16, 32)
    ):
        exact_rows_per_program = 256

    if exact_rows_per_program is not None:
        programs = outer_size // exact_rows_per_program
        static_grid = (programs,)
        if len(input_specs) == 2:
            exact_kernel = concatenate2_insert_slice_rows_kernel
            exact_constexpr_kwargs = {
                "OUTER_SIZE": outer_size,
                "INPUT0_ROW": input_rows[0],
                "INPUT1_ROW": input_rows[1],
                "OUTPUT_ROW": output_row,
                "ROWS_PER_PROGRAM": exact_rows_per_program,
                "num_warps": 4,
                "num_stages": 1,
            }
            exact_cached_args = (
                outer_size,
                input_rows[0],
                input_rows[1],
                output_row,
                exact_rows_per_program,
            )
        else:
            exact_kernel = concatenate3_insert_slice_rows_kernel
            exact_constexpr_kwargs = {
                "OUTER_SIZE": outer_size,
                "INPUT0_ROW": input_rows[0],
                "INPUT1_ROW": input_rows[1],
                "INPUT2_ROW": input_rows[2],
                "OUTPUT_ROW": output_row,
                "ROWS_PER_PROGRAM": exact_rows_per_program,
                "num_warps": 4,
                "num_stages": 1,
            }
            exact_cached_args = (
                outer_size,
                input_rows[0],
                input_rows[1],
                input_rows[2],
                output_row,
                exact_rows_per_program,
            )

        def exact_grid(_meta: dict[str, Any]) -> tuple[int, ...]:
            return static_grid

        def exact_runtime_args(
            inputs: Sequence[Any], output: torch.Tensor
        ) -> tuple[Any, ...]:
            return (*inputs, output)

        def exact_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (*static_grid, 1, 1), exact_cached_args

        return make_single_kernel_run_fn(
            PreparedSingleKernelRunSpec(
                kernel=PreparedSingleKernelSpec(
                    kernel=exact_kernel,
                    grid=exact_grid,
                    static_args=(),
                    constexpr_kwargs=exact_constexpr_kwargs,
                    build_cached_call=exact_cached_call,
                ),
                input_checks=input_checks,
                output_factory=output_factory,
                runtime_args=exact_runtime_args,
                extra_check=extra_check,
                validate_inputs=bool(attrs.get("_validate_inputs", True)),
            ),
            default_run_fn,
        )

    if outer_size > 1:
        max_columns = 128
        block_columns = min(
            max_columns, triton.next_power_of_2(max(input_rows))
        )
        max_rows = 256 if len(input_specs) == 2 else 32
        rows_per_program = min(max_rows, triton.next_power_of_2(outer_size))
        static_grid = (
            triton.cdiv(max(input_rows), block_columns),
            triton.cdiv(outer_size, rows_per_program),
        )
        if len(input_specs) == 2:
            kernel = concatenate2_segmented_rows_kernel
            constexpr_kwargs = {
                "OUTER_SIZE": outer_size,
                "INPUT0_ROW": input_rows[0],
                "INPUT1_ROW": input_rows[1],
                "OUTPUT_ROW": output_row,
                "ROWS_PER_PROGRAM": rows_per_program,
                "BLOCK_COLUMNS": block_columns,
                "num_warps": 4,
                "num_stages": 1,
            }
            cached_args = (
                outer_size,
                input_rows[0],
                input_rows[1],
                output_row,
                rows_per_program,
                block_columns,
            )
        else:
            kernel = concatenate3_segmented_rows_kernel
            constexpr_kwargs = {
                "OUTER_SIZE": outer_size,
                "INPUT0_ROW": input_rows[0],
                "INPUT1_ROW": input_rows[1],
                "INPUT2_ROW": input_rows[2],
                "OUTPUT_ROW": output_row,
                "ROWS_PER_PROGRAM": rows_per_program,
                "BLOCK_COLUMNS": block_columns,
                "num_warps": 4,
                "num_stages": 1,
            }
            cached_args = (
                outer_size,
                input_rows[0],
                input_rows[1],
                input_rows[2],
                output_row,
                rows_per_program,
                block_columns,
            )

        def fused_grid(_meta: dict[str, Any]) -> tuple[int, ...]:
            return static_grid

        def fused_runtime_args(
            inputs: Sequence[Any], output: torch.Tensor
        ) -> tuple[Any, ...]:
            return (*inputs, output)

        def fused_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (*static_grid, 1), cached_args

        return make_single_kernel_run_fn(
            PreparedSingleKernelRunSpec(
                kernel=PreparedSingleKernelSpec(
                    kernel=kernel,
                    grid=fused_grid,
                    static_args=(),
                    constexpr_kwargs=constexpr_kwargs,
                    build_cached_call=fused_cached_call,
                ),
                input_checks=input_checks,
                output_factory=output_factory,
                runtime_args=fused_runtime_args,
                extra_check=extra_check,
                validate_inputs=bool(attrs.get("_validate_inputs", True)),
            ),
            default_run_fn,
        )

    steps: list[PreparedPipelineStepSpec] = []
    axis_offset = 0
    for input_index, (shape, input_row) in enumerate(
        zip(static_shapes, input_rows)
    ):
        input_elements = _numel(shape)
        if outer_size == 1:
            block_size = min(
                8192,
                max(
                    256,
                    triton.next_power_of_2(
                        triton.cdiv(
                            input_elements,
                            get_vector_core_count(first_spec.device),
                        )
                    ),
                ),
            )
            programs = triton.cdiv(input_elements, block_size)
            step_kernel = concatenate_copy_kernel
            static_grid = (programs, 1, 1)
            constexpr_kwargs = {
                "N_ELEMENTS": input_elements,
                "INPUT_ROW": input_row,
                "OUTPUT_ROW": output_row,
                "AXIS_OFFSET": axis_offset,
                "OUTER_SIZE": outer_size,
                "BLOCK_SIZE": block_size,
                "num_warps": 4,
                "num_stages": 1,
            }
            cached_args = (
                input_elements,
                input_row,
                output_row,
                axis_offset,
                outer_size,
                block_size,
            )
            launch_grid = (programs,)
        else:
            block_columns = min(32, triton.next_power_of_2(input_row))
            max_rows = 64 if input_row >= 64 else 256
            rows_per_program = min(
                max_rows, triton.next_power_of_2(outer_size)
            )
            launch_grid = (
                triton.cdiv(input_row, block_columns),
                triton.cdiv(outer_size, rows_per_program),
            )
            step_kernel = concatenate_copy_rows_kernel
            static_grid = (*launch_grid, 1)
            constexpr_kwargs = {
                "OUTER_SIZE": outer_size,
                "INPUT_ROW": input_row,
                "OUTPUT_ROW": output_row,
                "AXIS_OFFSET": axis_offset,
                "ROWS_PER_PROGRAM": rows_per_program,
                "BLOCK_COLUMNS": block_columns,
                "num_warps": 4,
                "num_stages": 1,
            }
            cached_args = (
                outer_size,
                input_row,
                output_row,
                axis_offset,
                rows_per_program,
                block_columns,
            )

        def grid(
            _meta: dict[str, Any],
            *,
            launch_grid: tuple[int, ...] = launch_grid,
        ) -> tuple[int, ...]:
            return launch_grid

        def runtime_args(
            inputs: Sequence[Any],
            output: torch.Tensor,
            *,
            input_index: int = input_index,
        ) -> tuple[Any, ...]:
            return inputs[input_index], output

        def build_cached_call(
            constexprs: dict[str, Any],
            *,
            static_grid: tuple[int, ...] = static_grid,
            cached_args: tuple[Any, ...] = cached_args,
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return static_grid, cached_args

        steps.append(
            PreparedPipelineStepSpec(
                kernel=step_kernel,
                grid=grid,
                runtime_args=runtime_args,
                constexpr_kwargs=constexpr_kwargs,
                build_cached_call=build_cached_call,
                first_launch_returns_metadata=True,
            )
        )
        axis_offset += input_row

    pipeline_spec = PreparedKernelPipelineSpec(
        steps=tuple(steps),
        input_checks=input_checks,
        context_factory=output_factory,
        extra_check=extra_check,
    )
    run = make_kernel_pipeline_run_fn(pipeline_spec, default_run_fn)
    bound_launcher = make_kernel_pipeline_launcher(pipeline_spec)
    validate_inputs = bool(attrs.get("_validate_inputs", True))

    def bind(inputs: Sequence[Any], run_attrs: dict[str, Any]) -> Any:
        if validate_inputs and (
            not runtime_tensor_checks_pass(inputs, input_checks)
            or not extra_check(inputs)
        ):
            return lambda: default_run_fn(inputs, run_attrs)
        source = inputs[0]
        if not isinstance(source, torch.Tensor):
            return lambda: default_run_fn(inputs, run_attrs)
        output = output_factory(inputs)

        def run_bound() -> torch.Tensor:
            bound_launcher(source.device, inputs, output)
            return output

        return run_bound

    setattr(run, "bind", bind)
    return run


__all__ = (
    "concatenate2_aligned_core_loop_kernel",
    "concatenate2_core_loop_kernel",
    "concatenate2_axis0_262144_786432_kernel",
    "concatenate2_insert_slice_rows_kernel",
    "concatenate3_aligned_core_loop_kernel",
    "concatenate3_core_loop_kernel",
    "concatenate3_insert_slice_rows_kernel",
    "concatenate2_rows_kernel",
    "concatenate2_segmented_rows_kernel",
    "concatenate3_rows_kernel",
    "concatenate3_segmented_rows_kernel",
    "concatenate_copy_kernel",
    "concatenate_copy_rows_kernel",
    "gen_index_aligned_core_loop_kernel",
    "gen_index_core_loop_kernel",
    "gen_index_rows_kernel",
    "gen_index_tiled_kernel",
    "get_dense_utility_block_size",
    "get_utility_copy_block_size",
    "prepare_dense_concatenate",
    "prepare_dense_gen_index",
)
