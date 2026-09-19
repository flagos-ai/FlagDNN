"""Ascend kernels for ternary."""

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
def binary_select_contiguous_kernel(
    x_ptr,
    y_ptr,
    t_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    WORKER_COUNT: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    vector_block_size: tl.constexpr = BLOCK_SIZE * 4
    start = program_id * vector_block_size
    worker_stride = WORKER_COUNT * vector_block_size
    while start < n_elements:
        offsets = start + tl.arange(0, vector_block_size)
        active = offsets < n_elements
        left = tl.load(x_ptr + offsets, mask=active, other=0.0)
        right = tl.load(y_ptr + offsets, mask=active, other=0.0)
        predicate = tl.load(t_ptr + offsets, mask=active, other=0)
        result = tl.where(predicate != 0, left, right)
        tl.store(
            out_ptr + offsets,
            result.to(out_ptr.dtype.element_ty),
            mask=active,
        )
        start += worker_stride


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
    WORKER_COUNT: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    start = program_id * BLOCK_SIZE
    while start < n_elements:
        offsets = start + tl.arange(0, BLOCK_SIZE)
        active = offsets < n_elements
        remaining = offsets
        left_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        right_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        predicate_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        output_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

        coordinate = remaining % DIM_7
        remaining = remaining // DIM_7
        left_offsets += coordinate * LEFT_STRIDE_7
        right_offsets += coordinate * RIGHT_STRIDE_7
        predicate_offsets += coordinate * MASK_STRIDE_7
        output_offsets += coordinate * OUTPUT_STRIDE_7
        coordinate = remaining % DIM_6
        remaining = remaining // DIM_6
        left_offsets += coordinate * LEFT_STRIDE_6
        right_offsets += coordinate * RIGHT_STRIDE_6
        predicate_offsets += coordinate * MASK_STRIDE_6
        output_offsets += coordinate * OUTPUT_STRIDE_6
        coordinate = remaining % DIM_5
        remaining = remaining // DIM_5
        left_offsets += coordinate * LEFT_STRIDE_5
        right_offsets += coordinate * RIGHT_STRIDE_5
        predicate_offsets += coordinate * MASK_STRIDE_5
        output_offsets += coordinate * OUTPUT_STRIDE_5
        coordinate = remaining % DIM_4
        remaining = remaining // DIM_4
        left_offsets += coordinate * LEFT_STRIDE_4
        right_offsets += coordinate * RIGHT_STRIDE_4
        predicate_offsets += coordinate * MASK_STRIDE_4
        output_offsets += coordinate * OUTPUT_STRIDE_4
        coordinate = remaining % DIM_3
        remaining = remaining // DIM_3
        left_offsets += coordinate * LEFT_STRIDE_3
        right_offsets += coordinate * RIGHT_STRIDE_3
        predicate_offsets += coordinate * MASK_STRIDE_3
        output_offsets += coordinate * OUTPUT_STRIDE_3
        coordinate = remaining % DIM_2
        remaining = remaining // DIM_2
        left_offsets += coordinate * LEFT_STRIDE_2
        right_offsets += coordinate * RIGHT_STRIDE_2
        predicate_offsets += coordinate * MASK_STRIDE_2
        output_offsets += coordinate * OUTPUT_STRIDE_2
        coordinate = remaining % DIM_1
        remaining = remaining // DIM_1
        left_offsets += coordinate * LEFT_STRIDE_1
        right_offsets += coordinate * RIGHT_STRIDE_1
        predicate_offsets += coordinate * MASK_STRIDE_1
        output_offsets += coordinate * OUTPUT_STRIDE_1
        coordinate = remaining % DIM_0
        left_offsets += coordinate * LEFT_STRIDE_0
        right_offsets += coordinate * RIGHT_STRIDE_0
        predicate_offsets += coordinate * MASK_STRIDE_0
        output_offsets += coordinate * OUTPUT_STRIDE_0

        left = tl.load(x_ptr + left_offsets, mask=active, other=0.0)
        right = tl.load(y_ptr + right_offsets, mask=active, other=0.0)
        predicate = tl.load(t_ptr + predicate_offsets, mask=active, other=0)
        result = tl.where(predicate != 0, left, right)
        tl.store(
            out_ptr + output_offsets,
            result.to(out_ptr.dtype.element_ty),
            mask=active,
        )
        start += WORKER_COUNT * BLOCK_SIZE


# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0


@triton.jit
def binary_select_elementwise_strided_kernel(
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
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
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
    tl.store(out_ptr + offsets, result.to(out_ptr.dtype.element_ty), mask=active)
