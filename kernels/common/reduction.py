# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Reusable reduction kernels extracted from ``flag_dnn.ops.reduction``."""

import triton
import triton.language as tl


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
            x_ptr
            + rows[:, None] * stride_xm
            + reduction_offsets[None, :] * stride_xn,
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
            + input_offsets[:, None]
            + reduction_offsets[None, :] * REDUCTION_STRIDE,
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
    tl.store(out_ptr + output_offsets, result, mask=active_rows)
