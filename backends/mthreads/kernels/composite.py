# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""MThreads-private fused kernels for multi-node FlagDNN graphs."""

import triton
import triton.language as tl


@triton.jit
def add_square_tensor_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    COMPUTE_FLOAT32: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
):
    program_id = tl.program_id(0)
    base = program_id * BLOCK_SIZE * TILES_PER_PROGRAM
    for tile in tl.static_range(TILES_PER_PROGRAM):
        offsets = base + tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        left = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        right = tl.load(b_ptr + offsets, mask=mask, other=0.0)
        if COMPUTE_FLOAT32:
            left = left.to(tl.float32)
            right = right.to(tl.float32)
        result = left + right * right
        tl.store(
            out_ptr + offsets,
            result.to(out_ptr.dtype.element_ty),
            mask=mask,
        )
