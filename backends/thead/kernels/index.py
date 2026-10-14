# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Single-stage concatenation; typed tensor storage is copied bit for bit."""

import triton
import triton.language as tl


@triton.jit
def _concatenate_region(
    input_ptr,
    output_ptr,
    logical,
    N_ELEMENTS: tl.constexpr,
    OUTPUT_DIMS: tl.constexpr,
    INPUT_STRIDES: tl.constexpr,
    OUTPUT_STRIDES: tl.constexpr,
    AXIS: tl.constexpr,
    BEGIN: tl.constexpr,
    END: tl.constexpr,
    STORAGE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    remaining = logical
    source = tl.zeros((BLOCK_SIZE,), tl.int64)
    destination = tl.zeros((BLOCK_SIZE,), tl.int64)
    active = logical < N_ELEMENTS
    for axis in tl.static_range(len(OUTPUT_DIMS) - 1, -1, -1):
        coordinate = remaining % OUTPUT_DIMS[axis]
        remaining //= OUTPUT_DIMS[axis]
        destination += coordinate * OUTPUT_STRIDES[axis]
        if axis == AXIS:
            in_region = (coordinate >= BEGIN) & (coordinate < END)
            active = active & in_region
            # Subtract before multiplying: singleton source dimensions may
            # have huge valid strides. Masked lanes also use a valid index.
            local = tl.where(in_region, coordinate - BEGIN, 0)
            source += local * INPUT_STRIDES[axis]
        else:
            source += coordinate * INPUT_STRIDES[axis]
    value = tl.load(
        input_ptr.to(tl.pointer_type(STORAGE)) + source,
        mask=active,
        other=0,
    )
    tl.store(
        output_ptr.to(tl.pointer_type(STORAGE)) + destination,
        value,
        mask=active,
    )


@triton.jit
def concatenate_kernel(
    input_ptr,
    output_ptr,
    N_ELEMENTS: tl.constexpr,
    OUTPUT_DIMS: tl.constexpr,
    INPUT_STRIDES: tl.constexpr,
    OUTPUT_STRIDES: tl.constexpr,
    AXIS: tl.constexpr,
    STORAGE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Canonical one-input entry; codegen expands its region calls and ABI."""
    logical = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(
        0, BLOCK_SIZE
    )
    _concatenate_region(
        input_ptr, output_ptr, logical, N_ELEMENTS, OUTPUT_DIMS,
        INPUT_STRIDES, OUTPUT_STRIDES, AXIS, 0, OUTPUT_DIMS[AXIS],
        STORAGE, BLOCK_SIZE,
    )
