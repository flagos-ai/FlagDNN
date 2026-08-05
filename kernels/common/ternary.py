# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Platform-neutral Triton kernel for binary-select/where.

The kernel applies C++ Graph broadcast and stride metadata directly while
sharing one compile-time operation implementation across compatible backends.
"""

import triton
import triton.language as tl


@triton.jit
def binary_select_strided_kernel(
    x_ptr,
    y_ptr,
    t_ptr,
    out_ptr,
    n_elements,
    DIM_0: tl.constexpr,
    DIM_1: tl.constexpr,
    DIM_2: tl.constexpr,
    DIM_3: tl.constexpr,
    DIM_4: tl.constexpr,
    DIM_5: tl.constexpr,
    DIM_6: tl.constexpr,
    DIM_7: tl.constexpr,
    LEFT_STRIDE_0: tl.constexpr,
    LEFT_STRIDE_1: tl.constexpr,
    LEFT_STRIDE_2: tl.constexpr,
    LEFT_STRIDE_3: tl.constexpr,
    LEFT_STRIDE_4: tl.constexpr,
    LEFT_STRIDE_5: tl.constexpr,
    LEFT_STRIDE_6: tl.constexpr,
    LEFT_STRIDE_7: tl.constexpr,
    RIGHT_STRIDE_0: tl.constexpr,
    RIGHT_STRIDE_1: tl.constexpr,
    RIGHT_STRIDE_2: tl.constexpr,
    RIGHT_STRIDE_3: tl.constexpr,
    RIGHT_STRIDE_4: tl.constexpr,
    RIGHT_STRIDE_5: tl.constexpr,
    RIGHT_STRIDE_6: tl.constexpr,
    RIGHT_STRIDE_7: tl.constexpr,
    MASK_STRIDE_0: tl.constexpr,
    MASK_STRIDE_1: tl.constexpr,
    MASK_STRIDE_2: tl.constexpr,
    MASK_STRIDE_3: tl.constexpr,
    MASK_STRIDE_4: tl.constexpr,
    MASK_STRIDE_5: tl.constexpr,
    MASK_STRIDE_6: tl.constexpr,
    MASK_STRIDE_7: tl.constexpr,
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
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(
        0, BLOCK_SIZE
    )
    active = offsets < n_elements
    remaining = offsets
    left_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    right_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    mask_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    output_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

    coordinate = remaining % DIM_7
    remaining //= DIM_7
    left_offsets += coordinate * LEFT_STRIDE_7
    right_offsets += coordinate * RIGHT_STRIDE_7
    mask_offsets += coordinate * MASK_STRIDE_7
    output_offsets += coordinate * OUTPUT_STRIDE_7
    coordinate = remaining % DIM_6
    remaining //= DIM_6
    left_offsets += coordinate * LEFT_STRIDE_6
    right_offsets += coordinate * RIGHT_STRIDE_6
    mask_offsets += coordinate * MASK_STRIDE_6
    output_offsets += coordinate * OUTPUT_STRIDE_6
    coordinate = remaining % DIM_5
    remaining //= DIM_5
    left_offsets += coordinate * LEFT_STRIDE_5
    right_offsets += coordinate * RIGHT_STRIDE_5
    mask_offsets += coordinate * MASK_STRIDE_5
    output_offsets += coordinate * OUTPUT_STRIDE_5
    coordinate = remaining % DIM_4
    remaining //= DIM_4
    left_offsets += coordinate * LEFT_STRIDE_4
    right_offsets += coordinate * RIGHT_STRIDE_4
    mask_offsets += coordinate * MASK_STRIDE_4
    output_offsets += coordinate * OUTPUT_STRIDE_4
    coordinate = remaining % DIM_3
    remaining //= DIM_3
    left_offsets += coordinate * LEFT_STRIDE_3
    right_offsets += coordinate * RIGHT_STRIDE_3
    mask_offsets += coordinate * MASK_STRIDE_3
    output_offsets += coordinate * OUTPUT_STRIDE_3
    coordinate = remaining % DIM_2
    remaining //= DIM_2
    left_offsets += coordinate * LEFT_STRIDE_2
    right_offsets += coordinate * RIGHT_STRIDE_2
    mask_offsets += coordinate * MASK_STRIDE_2
    output_offsets += coordinate * OUTPUT_STRIDE_2
    coordinate = remaining % DIM_1
    remaining //= DIM_1
    left_offsets += coordinate * LEFT_STRIDE_1
    right_offsets += coordinate * RIGHT_STRIDE_1
    mask_offsets += coordinate * MASK_STRIDE_1
    output_offsets += coordinate * OUTPUT_STRIDE_1
    coordinate = remaining % DIM_0
    left_offsets += coordinate * LEFT_STRIDE_0
    right_offsets += coordinate * RIGHT_STRIDE_0
    mask_offsets += coordinate * MASK_STRIDE_0
    output_offsets += coordinate * OUTPUT_STRIDE_0

    left = tl.load(x_ptr + left_offsets, mask=active, other=0.0)
    right = tl.load(y_ptr + right_offsets, mask=active, other=0.0)
    predicate = tl.load(t_ptr + mask_offsets, mask=active, other=0)
    result = tl.where(predicate != 0, left, right)
    tl.store(
        out_ptr + output_offsets,
        result.to(out_ptr.dtype.element_ty),
        mask=active,
    )


# -----------------------------------------------------------------------------
# Kernel algorithm variants. Native dispatch selects only registry-declared
# entry points; auxiliary kernels remain available to explicit compiler policy.
# -----------------------------------------------------------------------------


@triton.jit
def binary_select_tensor_kernel(
    input0_ptr,
    input1_ptr,
    mask_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = offsets < n_elements

    input0 = tl.load(input0_ptr + offsets, mask=active)
    input1 = tl.load(input1_ptr + offsets, mask=active)
    mask_value = tl.load(mask_ptr + offsets, mask=active, other=0)
    result = tl.where(mask_value != 0, input0, input1)
    tl.store(
        out_ptr + offsets, result.to(out_ptr.dtype.element_ty), mask=active
    )


@triton.jit
def binary_select_broadcast_kernel(
    input0_ptr,
    input1_ptr,
    mask_ptr,
    out_ptr,
    n_elements,
    s1,
    s2,
    s3,
    s4,
    s5,
    sx0,
    sx1,
    sx2,
    sx3,
    sx4,
    sx5,
    sy0,
    sy1,
    sy2,
    sy3,
    sy4,
    sy5,
    sm0,
    sm1,
    sm2,
    sm3,
    sm4,
    sm5,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = offsets < n_elements

    idx5 = offsets % s5
    rem4 = offsets // s5

    idx4 = rem4 % s4
    rem3 = rem4 // s4

    idx3 = rem3 % s3
    rem2 = rem3 // s3

    idx2 = rem2 % s2
    rem1 = rem2 // s2

    idx1 = rem1 % s1
    idx0 = rem1 // s1

    input0_off = (
        idx0 * sx0
        + idx1 * sx1
        + idx2 * sx2
        + idx3 * sx3
        + idx4 * sx4
        + idx5 * sx5
    )
    input1_off = (
        idx0 * sy0
        + idx1 * sy1
        + idx2 * sy2
        + idx3 * sy3
        + idx4 * sy4
        + idx5 * sy5
    )
    mask_off = (
        idx0 * sm0
        + idx1 * sm1
        + idx2 * sm2
        + idx3 * sm3
        + idx4 * sm4
        + idx5 * sm5
    )

    input0 = tl.load(input0_ptr + input0_off, mask=active)
    input1 = tl.load(input1_ptr + input1_off, mask=active)
    mask_value = tl.load(mask_ptr + mask_off, mask=active, other=0)
    result = tl.where(mask_value != 0, input0, input1)
    tl.store(
        out_ptr + offsets, result.to(out_ptr.dtype.element_ty), mask=active
    )
