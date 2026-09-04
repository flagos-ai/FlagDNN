# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""MThreads-specific identity materialization kernels."""

import triton
import triton.language as tl


@triton.jit
def identity_contiguous_packed_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    PACK_SIZE: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy aligned dense tensors as 64-bit words with a scalar tail."""

    packed_input = input_ptr.to(tl.pointer_type(tl.uint64), bitcast=True)
    packed_output = output_ptr.to(tl.pointer_type(tl.uint64), bitcast=True)
    word_count = n_elements // PACK_SIZE
    program_id = tl.program_id(0)
    program_base = program_id * BLOCK_SIZE * TILES_PER_PROGRAM

    for tile_index in tl.static_range(TILES_PER_PROGRAM):
        word_offsets = (
            program_base
            + tile_index * BLOCK_SIZE
            + tl.arange(0, BLOCK_SIZE)
        )
        active = word_offsets < word_count
        values = tl.load(packed_input + word_offsets, mask=active, other=0)
        tl.store(packed_output + word_offsets, values, mask=active)

    tail_offsets = word_count * PACK_SIZE + tl.arange(0, PACK_SIZE)
    tail_active = (program_id == 0) & (tail_offsets < n_elements)
    tail = tl.load(input_ptr + tail_offsets, mask=tail_active, other=0)
    tl.store(output_ptr + tail_offsets, tail, mask=tail_active)


@triton.jit
def identity_contiguous_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    TILES_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program_base = tl.program_id(0) * BLOCK_SIZE * TILES_PER_PROGRAM
    for tile_index in tl.static_range(TILES_PER_PROGRAM):
        offsets = (
            program_base
            + tile_index * BLOCK_SIZE
            + tl.arange(0, BLOCK_SIZE)
        )
        active = offsets < n_elements
        values = tl.load(input_ptr + offsets, mask=active, other=0)
        tl.store(output_ptr + offsets, values, mask=active)


@triton.jit
def identity_strided_kernel(
    input_ptr,
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
    logical = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = logical < n_elements
    remaining = logical
    input_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    output_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    coordinate = remaining % DIM_7
    remaining //= DIM_7
    input_offsets += coordinate * INPUT_STRIDE_7
    output_offsets += coordinate * OUTPUT_STRIDE_7
    coordinate = remaining % DIM_6
    remaining //= DIM_6
    input_offsets += coordinate * INPUT_STRIDE_6
    output_offsets += coordinate * OUTPUT_STRIDE_6
    coordinate = remaining % DIM_5
    remaining //= DIM_5
    input_offsets += coordinate * INPUT_STRIDE_5
    output_offsets += coordinate * OUTPUT_STRIDE_5
    coordinate = remaining % DIM_4
    remaining //= DIM_4
    input_offsets += coordinate * INPUT_STRIDE_4
    output_offsets += coordinate * OUTPUT_STRIDE_4
    coordinate = remaining % DIM_3
    remaining //= DIM_3
    input_offsets += coordinate * INPUT_STRIDE_3
    output_offsets += coordinate * OUTPUT_STRIDE_3
    coordinate = remaining % DIM_2
    remaining //= DIM_2
    input_offsets += coordinate * INPUT_STRIDE_2
    output_offsets += coordinate * OUTPUT_STRIDE_2
    coordinate = remaining % DIM_1
    remaining //= DIM_1
    input_offsets += coordinate * INPUT_STRIDE_1
    output_offsets += coordinate * OUTPUT_STRIDE_1
    coordinate = remaining % DIM_0
    input_offsets += coordinate * INPUT_STRIDE_0
    output_offsets += coordinate * OUTPUT_STRIDE_0

    values = tl.load(input_ptr + input_offsets, mask=active, other=0)
    tl.store(output_ptr + output_offsets, values, mask=active)
