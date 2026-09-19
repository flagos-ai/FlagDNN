"""Ascend kernels for reduction."""

import triton
import triton.language as tl

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");


REDUCTION_SUM = tl.constexpr(0)
REDUCTION_AVG = tl.constexpr(1)
REDUCTION_MUL = tl.constexpr(2)


@triton.jit
def _multiply(left, right):
    return left * right


@triton.jit
def _combine(accumulator, values, REDUCTION_MODE: tl.constexpr):
    if REDUCTION_MODE == REDUCTION_MUL:
        return accumulator * tl.reduce(values, axis=0, combine_fn=_multiply)
    return accumulator + tl.sum(values, axis=0)


@triton.jit
def reduction_3d_persistent_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    RANK: tl.constexpr,
    OUTPUT_RANK: tl.constexpr,
    AXIS: tl.constexpr,
    KEEP_DIMENSIONS: tl.constexpr,
    OUTER: tl.constexpr,
    REDUCTION_SIZE: tl.constexpr,
    INNER: tl.constexpr,
    OUTPUT_ELEMENTS: tl.constexpr,
    REDUCTION_MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WORKER_COUNT: tl.constexpr,
):
    if REDUCTION_SIZE <= 8:
        output_block: tl.constexpr = 64
        output_base = tl.program_id(0).to(tl.int64) * output_block
        output_lanes = tl.arange(0, output_block).to(tl.int64)
        while output_base < n_elements:
            output_indices = output_base + output_lanes
            output_mask = output_indices < n_elements
            if INNER % output_block == 0:
                outer_indices = output_base // INNER
                inner_indices = output_base % INNER + output_lanes
            else:
                outer_indices = output_indices // INNER
                inner_indices = output_indices % INNER
            if (
                REDUCTION_MODE == REDUCTION_MUL
                or input_ptr.dtype.element_ty == tl.float32
            ):
                if REDUCTION_MODE == REDUCTION_MUL:
                    result = tl.full((output_block,), 1.0, tl.float32)
                else:
                    result = tl.zeros((output_block,), tl.float32)
                for reduction_index in tl.static_range(0, REDUCTION_SIZE):
                    values = tl.load(
                        input_ptr
                        + outer_indices * REDUCTION_SIZE * INNER
                        + inner_indices
                        + reduction_index * INNER,
                        mask=output_mask,
                        other=(1.0 if REDUCTION_MODE == REDUCTION_MUL else 0.0),
                    ).to(tl.float32)
                    if REDUCTION_MODE == REDUCTION_MUL:
                        result *= values
                    else:
                        result += values
            else:
                reduction_lanes = tl.arange(0, 8).to(tl.int64)
                reduction_active = reduction_lanes < REDUCTION_SIZE
                values = tl.load(
                    input_ptr
                    + outer_indices[:, None] * REDUCTION_SIZE * INNER
                    + inner_indices[:, None]
                    + reduction_lanes[None, :] * INNER,
                    mask=output_mask[:, None] & reduction_active[None, :],
                    other=0.0,
                ).to(tl.float32)
                result = tl.sum(values, axis=1)
            if REDUCTION_MODE == REDUCTION_AVG:
                result /= REDUCTION_SIZE
            tl.store(
                output_ptr + output_indices,
                result.to(output_ptr.dtype.element_ty),
                mask=output_mask,
            )
            output_base += output_block * WORKER_COUNT
        return

    output_index = tl.program_id(0).to(tl.int64)
    while output_index < n_elements:
        outer_index = output_index // INNER
        inner_index = output_index % INNER
        input_base = outer_index * REDUCTION_SIZE * INNER + inner_index
        if REDUCTION_MODE == REDUCTION_MUL:
            accumulator = 1.0
        else:
            accumulator = 0.0
        reduction_start = output_index * 0
        while reduction_start < REDUCTION_SIZE:
            reduction_offsets = reduction_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)
            active = reduction_offsets < REDUCTION_SIZE
            values = tl.load(
                input_ptr + input_base + reduction_offsets * INNER,
                mask=active,
                other=1.0 if REDUCTION_MODE == REDUCTION_MUL else 0.0,
            ).to(tl.float32)
            accumulator = _combine(accumulator, values, REDUCTION_MODE)
            reduction_start += BLOCK_SIZE
        if REDUCTION_MODE == REDUCTION_AVG:
            accumulator /= REDUCTION_SIZE
        tl.store(
            output_ptr + output_index,
            accumulator.to(output_ptr.dtype.element_ty),
        )
        output_index += WORKER_COUNT


@triton.jit
def reduction_strided_persistent_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    RANK: tl.constexpr,
    OUTPUT_RANK: tl.constexpr,
    AXIS: tl.constexpr,
    KEEP_DIMENSIONS: tl.constexpr,
    OUTER: tl.constexpr,
    REDUCTION_SIZE: tl.constexpr,
    INNER: tl.constexpr,
    OUTPUT_ELEMENTS: tl.constexpr,
    REDUCTION_MODE: tl.constexpr,
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
    output_index = tl.program_id(0).to(tl.int64)
    while output_index < n_elements:
        output_remaining = output_index
        output_offset = output_index * 0
        coordinate = output_remaining % OUTPUT_DIM_7
        output_remaining //= OUTPUT_DIM_7
        output_offset += coordinate * OUTPUT_STRIDE_7
        coordinate = output_remaining % OUTPUT_DIM_6
        output_remaining //= OUTPUT_DIM_6
        output_offset += coordinate * OUTPUT_STRIDE_6
        coordinate = output_remaining % OUTPUT_DIM_5
        output_remaining //= OUTPUT_DIM_5
        output_offset += coordinate * OUTPUT_STRIDE_5
        coordinate = output_remaining % OUTPUT_DIM_4
        output_remaining //= OUTPUT_DIM_4
        output_offset += coordinate * OUTPUT_STRIDE_4
        coordinate = output_remaining % OUTPUT_DIM_3
        output_remaining //= OUTPUT_DIM_3
        output_offset += coordinate * OUTPUT_STRIDE_3
        coordinate = output_remaining % OUTPUT_DIM_2
        output_remaining //= OUTPUT_DIM_2
        output_offset += coordinate * OUTPUT_STRIDE_2
        coordinate = output_remaining % OUTPUT_DIM_1
        output_remaining //= OUTPUT_DIM_1
        output_offset += coordinate * OUTPUT_STRIDE_1
        output_offset += (output_remaining % OUTPUT_DIM_0) * OUTPUT_STRIDE_0

        reduction_start = output_index * 0
        if REDUCTION_MODE == REDUCTION_MUL:
            accumulator = 1.0
        else:
            accumulator = 0.0
        while reduction_start < REDUCTION_SIZE:
            reduction_offsets = reduction_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)
            active = reduction_offsets < REDUCTION_SIZE
            input_linear = (
                (output_index // INNER) * REDUCTION_SIZE * INNER
                + reduction_offsets * INNER
                + output_index % INNER
            )
            input_remaining = input_linear
            input_offset = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
            input_coordinate = input_remaining % INPUT_DIM_7
            input_remaining //= INPUT_DIM_7
            input_offset += input_coordinate * INPUT_STRIDE_7
            input_coordinate = input_remaining % INPUT_DIM_6
            input_remaining //= INPUT_DIM_6
            input_offset += input_coordinate * INPUT_STRIDE_6
            input_coordinate = input_remaining % INPUT_DIM_5
            input_remaining //= INPUT_DIM_5
            input_offset += input_coordinate * INPUT_STRIDE_5
            input_coordinate = input_remaining % INPUT_DIM_4
            input_remaining //= INPUT_DIM_4
            input_offset += input_coordinate * INPUT_STRIDE_4
            input_coordinate = input_remaining % INPUT_DIM_3
            input_remaining //= INPUT_DIM_3
            input_offset += input_coordinate * INPUT_STRIDE_3
            input_coordinate = input_remaining % INPUT_DIM_2
            input_remaining //= INPUT_DIM_2
            input_offset += input_coordinate * INPUT_STRIDE_2
            input_coordinate = input_remaining % INPUT_DIM_1
            input_remaining //= INPUT_DIM_1
            input_offset += input_coordinate * INPUT_STRIDE_1
            input_coordinate = input_remaining % INPUT_DIM_0
            input_offset += input_coordinate * INPUT_STRIDE_0
            values = tl.load(
                input_ptr + input_offset,
                mask=active,
                other=1.0 if REDUCTION_MODE == REDUCTION_MUL else 0.0,
            ).to(tl.float32)
            accumulator = _combine(accumulator, values, REDUCTION_MODE)
            reduction_start += BLOCK_SIZE
        if REDUCTION_MODE == REDUCTION_AVG:
            accumulator /= REDUCTION_SIZE
        tl.store(
            output_ptr + output_offset,
            accumulator.to(output_ptr.dtype.element_ty),
        )
        output_index += WORKER_COUNT


# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0


@triton.jit
def _reduce_block(values, OP: tl.constexpr, BLOCK_N: tl.constexpr):
    if OP == 3:
        products = tl.cumprod(values, axis=1)
        last = tl.arange(0, BLOCK_N) == (BLOCK_N - 1)
        return tl.sum(tl.where(last[None, :], products, 0.0), axis=1)
    return tl.sum(values, axis=1)


@triton.jit
def reduction_2d_kernel(
    x_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    OP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    active_rows = rows < M
    columns = tl.arange(0, BLOCK_N)
    if OP == 3:
        result = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
    else:
        result = tl.zeros((BLOCK_M,), dtype=tl.float32)
    other: tl.constexpr = 1.0 if OP == 3 else 0.0

    for start in range(0, N, BLOCK_N):
        reduction_offsets = start + columns
        active = active_rows[:, None] & (reduction_offsets[None, :] < N)
        values = tl.load(
            x_ptr + rows[:, None] * stride_xm + reduction_offsets[None, :] * stride_xn,
            mask=active,
            other=other,
        ).to(tl.float32)
        reduced = _reduce_block(values, OP, BLOCK_N)
        if OP == 3:
            result *= reduced
        else:
            result += reduced

    if OP == 2:
        result /= N
    tl.store(out_ptr + rows, result, mask=active_rows)


@triton.jit
def reduction_3d_kernel(
    x_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    I: tl.constexpr,
    stride_xo: tl.constexpr,
    stride_xr: tl.constexpr,
    stride_xi: tl.constexpr,
    OP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    active_rows = rows < M
    outer = rows // I
    inner = rows % I
    columns = tl.arange(0, BLOCK_N)
    if OP == 3:
        result = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
    else:
        result = tl.zeros((BLOCK_M,), dtype=tl.float32)
    other: tl.constexpr = 1.0 if OP == 3 else 0.0

    for start in range(0, N, BLOCK_N):
        reduction_offsets = start + columns
        active = active_rows[:, None] & (reduction_offsets[None, :] < N)
        values = tl.load(
            x_ptr
            + outer[:, None] * stride_xo
            + reduction_offsets[None, :] * stride_xr
            + inner[:, None] * stride_xi,
            mask=active,
            other=other,
        ).to(tl.float32)
        reduced = _reduce_block(values, OP, BLOCK_N)
        if OP == 3:
            result *= reduced
        else:
            result += reduced

    if OP == 2:
        result /= N
    tl.store(out_ptr + rows, result, mask=active_rows)


@triton.jit
def reduction_strided_kernel(
    x_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    REDUCTION_STRIDE: tl.constexpr,
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
    OP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    active_rows = rows < M
    remaining = rows
    input_offsets = tl.zeros((BLOCK_M,), dtype=tl.int64)
    output_offsets = tl.zeros((BLOCK_M,), dtype=tl.int64)

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

    if OP == 3:
        result = tl.full((BLOCK_M,), 1.0, tl.float32)
    else:
        result = tl.full((BLOCK_M,), 0.0, tl.float32)
    for reduction_index in range(N):
        value = tl.load(
            x_ptr + input_offsets + reduction_index * REDUCTION_STRIDE,
            active_rows,
            1.0 if OP == 3 else 0.0,
        ).to(tl.float32)
        if OP == 3:
            result *= value
        else:
            result += value
    if OP == 2:
        result /= N
    tl.store(out_ptr + output_offsets, result, active_rows)
