# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""MTGPU-specialized Triton kernels for binary pointwise operations."""

import triton
import triton.language as tl
from triton.language.extra import libdevice


# Stable FlagDNN pointwise mode values. These are FlagDNN-owned semantic IDs,
# not platform-library enum values.
POINTWISE_ADD = tl.constexpr(1)
POINTWISE_SUB = tl.constexpr(17)
POINTWISE_MUL = tl.constexpr(18)
POINTWISE_DIV = tl.constexpr(19)
POINTWISE_MIN = tl.constexpr(20)
POINTWISE_MAX = tl.constexpr(21)
POINTWISE_MOD = tl.constexpr(22)
POINTWISE_POW = tl.constexpr(23)
POINTWISE_CMP_EQ = tl.constexpr(25)
POINTWISE_CMP_NEQ = tl.constexpr(26)
POINTWISE_CMP_GT = tl.constexpr(27)
POINTWISE_CMP_GE = tl.constexpr(28)
POINTWISE_CMP_LT = tl.constexpr(29)
POINTWISE_CMP_LE = tl.constexpr(30)
POINTWISE_LOGICAL_AND = tl.constexpr(31)
POINTWISE_LOGICAL_OR = tl.constexpr(32)
POINTWISE_SIGMOID_BWD = tl.constexpr(40)


@triton.jit
def _apply_binary_operation(
    left,
    right,
    OP_KIND: tl.constexpr,
    ALPHA: tl.constexpr,
):
    if OP_KIND == POINTWISE_ADD:
        result = left + ALPHA * right
    elif OP_KIND == POINTWISE_SUB:
        result = left - ALPHA * right
    elif OP_KIND == POINTWISE_MUL:
        result = left * right
    elif OP_KIND == POINTWISE_SIGMOID_BWD:
        sigmoid = tl.sigmoid(right.to(tl.float32))
        result = left.to(tl.float32) * sigmoid * (1.0 - sigmoid)
    elif OP_KIND == POINTWISE_DIV:
        result = left / right
    elif OP_KIND == POINTWISE_MIN:
        result = tl.minimum(left, right)
    elif OP_KIND == POINTWISE_MAX:
        result = tl.maximum(left, right)
    elif OP_KIND == POINTWISE_MOD:
        result = libdevice.fmod(left.to(tl.float32), right.to(tl.float32))
    elif OP_KIND == POINTWISE_POW:
        result = libdevice.pow(left.to(tl.float32), right.to(tl.float32))
    elif OP_KIND == POINTWISE_CMP_EQ:
        result = left == right
    elif OP_KIND == POINTWISE_CMP_NEQ:
        result = left != right
    elif OP_KIND == POINTWISE_CMP_GT:
        result = left > right
    elif OP_KIND == POINTWISE_CMP_GE:
        result = left >= right
    elif OP_KIND == POINTWISE_CMP_LT:
        result = left < right
    elif OP_KIND == POINTWISE_CMP_LE:
        result = left <= right
    elif OP_KIND == POINTWISE_LOGICAL_AND:
        result = (left != 0) & (right != 0)
    elif OP_KIND == POINTWISE_LOGICAL_OR:
        result = (left != 0) | (right != 0)
    return result


@triton.jit
def binary_contiguous_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    OP_KIND: tl.constexpr,
    ALPHA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # n_elements is an i32 runtime scalar and Graph validation caps every
    # reachable dense offset below INT32_MAX. Avoiding i64 linear arithmetic
    # is materially cheaper on MTGPU.
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = offsets < n_elements
    left = tl.load(x_ptr + offsets, mask=active, other=0.0)
    right = tl.load(y_ptr + offsets, mask=active, other=0.0)
    result = _apply_binary_operation(left, right, OP_KIND, ALPHA)
    tl.store(
        out_ptr + offsets,
        result.to(out_ptr.dtype.element_ty),
        mask=active,
    )


@triton.jit
def binary_strided_kernel(
    x_ptr,
    y_ptr,
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
    OUTPUT_STRIDE_0: tl.constexpr,
    OUTPUT_STRIDE_1: tl.constexpr,
    OUTPUT_STRIDE_2: tl.constexpr,
    OUTPUT_STRIDE_3: tl.constexpr,
    OUTPUT_STRIDE_4: tl.constexpr,
    OUTPUT_STRIDE_5: tl.constexpr,
    OUTPUT_STRIDE_6: tl.constexpr,
    OUTPUT_STRIDE_7: tl.constexpr,
    OP_KIND: tl.constexpr,
    ALPHA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = offsets < n_elements
    remaining = offsets
    left_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    right_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    output_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

    coordinate = remaining % DIM_7
    remaining //= DIM_7
    left_offsets += coordinate * LEFT_STRIDE_7
    right_offsets += coordinate * RIGHT_STRIDE_7
    output_offsets += coordinate * OUTPUT_STRIDE_7
    coordinate = remaining % DIM_6
    remaining //= DIM_6
    left_offsets += coordinate * LEFT_STRIDE_6
    right_offsets += coordinate * RIGHT_STRIDE_6
    output_offsets += coordinate * OUTPUT_STRIDE_6
    coordinate = remaining % DIM_5
    remaining //= DIM_5
    left_offsets += coordinate * LEFT_STRIDE_5
    right_offsets += coordinate * RIGHT_STRIDE_5
    output_offsets += coordinate * OUTPUT_STRIDE_5
    coordinate = remaining % DIM_4
    remaining //= DIM_4
    left_offsets += coordinate * LEFT_STRIDE_4
    right_offsets += coordinate * RIGHT_STRIDE_4
    output_offsets += coordinate * OUTPUT_STRIDE_4
    coordinate = remaining % DIM_3
    remaining //= DIM_3
    left_offsets += coordinate * LEFT_STRIDE_3
    right_offsets += coordinate * RIGHT_STRIDE_3
    output_offsets += coordinate * OUTPUT_STRIDE_3
    coordinate = remaining % DIM_2
    remaining //= DIM_2
    left_offsets += coordinate * LEFT_STRIDE_2
    right_offsets += coordinate * RIGHT_STRIDE_2
    output_offsets += coordinate * OUTPUT_STRIDE_2
    coordinate = remaining % DIM_1
    remaining //= DIM_1
    left_offsets += coordinate * LEFT_STRIDE_1
    right_offsets += coordinate * RIGHT_STRIDE_1
    output_offsets += coordinate * OUTPUT_STRIDE_1
    coordinate = remaining % DIM_0
    left_offsets += coordinate * LEFT_STRIDE_0
    right_offsets += coordinate * RIGHT_STRIDE_0
    output_offsets += coordinate * OUTPUT_STRIDE_0

    left = tl.load(x_ptr + left_offsets, mask=active, other=0.0)
    right = tl.load(y_ptr + right_offsets, mask=active, other=0.0)
    result = _apply_binary_operation(left, right, OP_KIND, ALPHA)
    tl.store(
        out_ptr + output_offsets,
        result.to(out_ptr.dtype.element_ty),
        mask=active,
    )
