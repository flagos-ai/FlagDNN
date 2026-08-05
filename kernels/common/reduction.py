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


# -----------------------------------------------------------------------------
# Kernel algorithm variants. Native dispatch selects only registry-declared
# entry points; auxiliary kernels remain available to explicit compiler policy.
# -----------------------------------------------------------------------------


@triton.jit
def _mean_kernel_2d_loop_store(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    IS_FP64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float64 if IS_FP64 else tl.float32)

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        mask = m_mask[:, None] & n_mask[None, :]
        x_ptrs = (
            x_ptr
            + m_offsets[:, None] * stride_xm
            + n_offsets[None, :] * stride_xn
        )

        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x = x.to(tl.float64 if IS_FP64 else tl.float32)
        acc += tl.sum(x, axis=1)

    tl.store(out_ptr + m_offsets, acc / N, mask=m_mask)


@triton.jit
def _mean_kernel_3d_loop_store(
    x_ptr,
    out_ptr,
    M,
    N,
    I,
    stride_xo,
    stride_xr,
    stride_xi,
    IS_FP64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    o_idx = m_offsets // I
    i_idx = m_offsets % I
    base_ptrs = x_ptr + (o_idx * stride_xo + i_idx * stride_xi)

    acc = tl.zeros((BLOCK_M,), dtype=tl.float64 if IS_FP64 else tl.float32)

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        mask = m_mask[:, None] & n_mask[None, :]
        x_ptrs = base_ptrs[:, None] + n_offsets[None, :] * stride_xr

        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x = x.to(tl.float64 if IS_FP64 else tl.float32)
        acc += tl.sum(x, axis=1)

    tl.store(out_ptr + m_offsets, acc / N, mask=m_mask)


@triton.jit
def _mean_kernel_dim0_small_r_store(
    x_ptr,
    out_ptr,
    R,
    I,
    stride_xr,
    stride_xi,
    IS_FP64: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid_i = tl.program_id(0)

    i_offsets = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    i_mask = i_offsets < I

    base_ptrs = x_ptr + i_offsets * stride_xi
    acc = tl.zeros((BLOCK_I,), dtype=tl.float64 if IS_FP64 else tl.float32)

    for r in range(0, R):
        x = tl.load(base_ptrs + r * stride_xr, mask=i_mask, other=0.0)
        x = x.to(tl.float64 if IS_FP64 else tl.float32)
        acc += x

    tl.store(out_ptr + i_offsets, acc / R, mask=i_mask)


@triton.jit
def _mean_kernel_2d_atomic(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    IS_FP64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N

    mask = m_mask[:, None] & n_mask[None, :]
    x_ptrs = (
        x_ptr + m_offsets[:, None] * stride_xm + n_offsets[None, :] * stride_xn
    )

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    x = x.to(tl.float64 if IS_FP64 else tl.float32)

    sum_vals = tl.sum(x, axis=1) / N
    tl.atomic_add(out_ptr + m_offsets, sum_vals, mask=m_mask)


@triton.jit
def _mean_kernel_3d_atomic(
    x_ptr,
    out_ptr,
    M,
    N,
    I,
    stride_xo,
    stride_xr,
    stride_xi,
    IS_FP64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N

    o_idx = m_offsets // I
    i_idx = m_offsets % I

    mask = m_mask[:, None] & n_mask[None, :]
    x_ptrs = x_ptr + (
        o_idx[:, None] * stride_xo
        + n_offsets[None, :] * stride_xr
        + i_idx[:, None] * stride_xi
    )

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    x = x.to(tl.float64 if IS_FP64 else tl.float32)

    sum_vals = tl.sum(x, axis=1) / N
    tl.atomic_add(out_ptr + m_offsets, sum_vals, mask=m_mask)


@triton.jit
def _prod_combine(a, b):
    return a * b


@triton.jit
def _prod_kernel_1row_split_stage1(
    x_ptr,
    partial_ptr,
    N,
    stride_xn,
    IS_FP64: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    n_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    x = tl.load(x_ptr + n_offsets * stride_xn, mask=n_mask, other=1.0)
    x = x.to(tl.float64 if IS_FP64 else tl.float32)

    part_val = tl.reduce(x, axis=0, combine_fn=_prod_combine)
    tl.store(partial_ptr + pid, part_val)


@triton.jit
def _prod_kernel_1row_finalize(
    partial_ptr,
    out_ptr,
    N,
    stride_pn,
    IS_FP64: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    acc = tl.full((1,), 1.0, dtype=tl.float64 if IS_FP64 else tl.float32)

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        x = tl.load(
            partial_ptr + n_offsets * stride_pn, mask=n_mask, other=1.0
        )
        x = x.to(tl.float64 if IS_FP64 else tl.float32)

        acc *= tl.reduce(x, axis=0, combine_fn=_prod_combine)

    tl.store(out_ptr + tl.arange(0, 1), acc)


@triton.jit
def _prod_kernel_2d_loop_store(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    IS_FP64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    acc = tl.full((BLOCK_M,), 1.0, dtype=tl.float64 if IS_FP64 else tl.float32)

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        mask = m_mask[:, None] & n_mask[None, :]
        x_ptrs = (
            x_ptr
            + m_offsets[:, None] * stride_xm
            + n_offsets[None, :] * stride_xn
        )

        x = tl.load(x_ptrs, mask=mask, other=1.0)
        x = x.to(tl.float64 if IS_FP64 else tl.float32)

        acc *= tl.reduce(x, axis=1, combine_fn=_prod_combine)

    tl.store(out_ptr + m_offsets, acc, mask=m_mask)


@triton.jit
def _prod_kernel_3d_loop_store(
    x_ptr,
    out_ptr,
    M,
    N,
    I,
    stride_xo,
    stride_xr,
    stride_xi,
    IS_FP64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    o_idx = m_offsets // I
    i_idx = m_offsets % I
    base_ptrs = x_ptr + (o_idx * stride_xo + i_idx * stride_xi)

    acc = tl.full((BLOCK_M,), 1.0, dtype=tl.float64 if IS_FP64 else tl.float32)

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        mask = m_mask[:, None] & n_mask[None, :]
        x_ptrs = base_ptrs[:, None] + n_offsets[None, :] * stride_xr

        x = tl.load(x_ptrs, mask=mask, other=1.0)
        x = x.to(tl.float64 if IS_FP64 else tl.float32)

        acc *= tl.reduce(x, axis=1, combine_fn=_prod_combine)

    tl.store(out_ptr + m_offsets, acc, mask=m_mask)


@triton.jit
def _prod_kernel_2d_split_stage1(
    x_ptr,
    partial_ptr,
    M,
    N,
    PARTIAL_N,
    stride_xm,
    stride_xn,
    IS_FP64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid // PARTIAL_N
    pid_n = pid % PARTIAL_N

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N
    mask = m_mask[:, None] & n_mask[None, :]

    x_ptrs = (
        x_ptr + m_offsets[:, None] * stride_xm + n_offsets[None, :] * stride_xn
    )
    x = tl.load(x_ptrs, mask=mask, other=1.0)
    x = x.to(tl.float64 if IS_FP64 else tl.float32)

    part_vals = tl.reduce(x, axis=1, combine_fn=_prod_combine)

    partial_ptrs = partial_ptr + m_offsets * PARTIAL_N + pid_n
    tl.store(partial_ptrs, part_vals, mask=m_mask)


@triton.jit
def _prod_kernel_3d_split_stage1(
    x_ptr,
    partial_ptr,
    M,
    N,
    I,
    PARTIAL_N,
    stride_xo,
    stride_xr,
    stride_xi,
    IS_FP64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid // PARTIAL_N
    pid_n = pid % PARTIAL_N

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N
    mask = m_mask[:, None] & n_mask[None, :]

    o_idx = m_offsets // I
    i_idx = m_offsets % I

    x_ptrs = x_ptr + (
        o_idx[:, None] * stride_xo
        + n_offsets[None, :] * stride_xr
        + i_idx[:, None] * stride_xi
    )

    x = tl.load(x_ptrs, mask=mask, other=1.0)
    x = x.to(tl.float64 if IS_FP64 else tl.float32)

    part_vals = tl.reduce(x, axis=1, combine_fn=_prod_combine)

    partial_ptrs = partial_ptr + m_offsets * PARTIAL_N + pid_n
    tl.store(partial_ptrs, part_vals, mask=m_mask)


@triton.jit
def _prod_kernel_dim0_small_r_store(
    x_ptr,
    out_ptr,
    R,
    I,
    stride_xr,
    stride_xi,
    IS_FP64: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid_i = tl.program_id(0)

    i_offsets = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    i_mask = i_offsets < I

    base_ptrs = x_ptr + i_offsets * stride_xi
    acc = tl.full((BLOCK_I,), 1.0, dtype=tl.float64 if IS_FP64 else tl.float32)

    for r in range(0, R):
        x = tl.load(base_ptrs + r * stride_xr, mask=i_mask, other=1.0)
        x = x.to(tl.float64 if IS_FP64 else tl.float32)
        acc *= x

    tl.store(out_ptr + i_offsets, acc, mask=i_mask)


@triton.jit
def _reduction_2d_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    OP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    if OP == "MIN":
        acc = tl.full((BLOCK_M,), float("inf"), dtype=tl.float32)
    elif OP == "MUL" or OP == "MUL_NO_ZEROS":
        acc = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
    elif OP == "AMAX" or (
        OP == "ADD" or OP == "AVG" or OP == "NORM1" or OP == "NORM2"
    ):
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    else:
        acc = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N
        ptrs = (
            x_ptr
            + m_offsets[:, None] * stride_xm
            + n_offsets[None, :] * stride_xn
        )
        mask = m_mask[:, None] & n_mask[None, :]
        if OP == "ADD" or OP == "AVG" or OP == "NORM1" or OP == "NORM2":
            vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
            if OP == "NORM1":
                vals = tl.abs(vals)
            elif OP == "NORM2":
                vals *= vals
            acc += tl.sum(vals, axis=1)
        elif OP == "MUL" or OP == "MUL_NO_ZEROS":
            vals = tl.load(ptrs, mask=mask, other=1.0).to(tl.float32)
            if OP == "MUL_NO_ZEROS":
                vals = tl.where(vals == 0.0, 1.0, vals)
            products = tl.cumprod(vals, axis=1)
            last = tl.arange(0, BLOCK_N) == (BLOCK_N - 1)
            chunk = tl.sum(tl.where(last[None, :], products, 0.0), axis=1)
            acc *= chunk
        elif OP == "MIN":
            vals = tl.load(ptrs, mask=mask, other=float("inf"))
            vals = vals.to(tl.float32)
            acc = tl.minimum(acc, tl.min(vals, axis=1))
        else:
            other = 0.0 if OP == "AMAX" else -float("inf")
            vals = tl.load(ptrs, mask=mask, other=other)
            vals = vals.to(tl.float32)
            if OP == "AMAX":
                vals = tl.abs(vals)
            acc = tl.maximum(acc, tl.max(vals, axis=1))

    if OP == "AVG":
        acc /= N
    elif OP == "NORM2":
        acc = tl.sqrt(acc)
    tl.store(out_ptr + m_offsets, acc, mask=m_mask)


@triton.jit
def _reduction_3d_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    I,
    stride_xo,
    stride_xr,
    stride_xi,
    OP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M
    o_idx = m_offsets // I
    i_idx = m_offsets % I
    base_ptrs = x_ptr + o_idx * stride_xo + i_idx * stride_xi

    if OP == "MIN":
        acc = tl.full((BLOCK_M,), float("inf"), dtype=tl.float32)
    elif OP == "MUL" or OP == "MUL_NO_ZEROS":
        acc = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
    elif OP == "AMAX" or (
        OP == "ADD" or OP == "AVG" or OP == "NORM1" or OP == "NORM2"
    ):
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    else:
        acc = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N
        ptrs = base_ptrs[:, None] + n_offsets[None, :] * stride_xr
        mask = m_mask[:, None] & n_mask[None, :]
        if OP == "ADD" or OP == "AVG" or OP == "NORM1" or OP == "NORM2":
            vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
            if OP == "NORM1":
                vals = tl.abs(vals)
            elif OP == "NORM2":
                vals *= vals
            acc += tl.sum(vals, axis=1)
        elif OP == "MUL" or OP == "MUL_NO_ZEROS":
            vals = tl.load(ptrs, mask=mask, other=1.0).to(tl.float32)
            if OP == "MUL_NO_ZEROS":
                vals = tl.where(vals == 0.0, 1.0, vals)
            products = tl.cumprod(vals, axis=1)
            last = tl.arange(0, BLOCK_N) == (BLOCK_N - 1)
            chunk = tl.sum(tl.where(last[None, :], products, 0.0), axis=1)
            acc *= chunk
        elif OP == "MIN":
            vals = tl.load(ptrs, mask=mask, other=float("inf"))
            vals = vals.to(tl.float32)
            acc = tl.minimum(acc, tl.min(vals, axis=1))
        else:
            other = 0.0 if OP == "AMAX" else -float("inf")
            vals = tl.load(ptrs, mask=mask, other=other)
            vals = vals.to(tl.float32)
            if OP == "AMAX":
                vals = tl.abs(vals)
            acc = tl.maximum(acc, tl.max(vals, axis=1))

    if OP == "AVG":
        acc /= N
    elif OP == "NORM2":
        acc = tl.sqrt(acc)
    tl.store(out_ptr + m_offsets, acc, mask=m_mask)


@triton.jit
def _sum_kernel_2d_atomic_fp64(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N

    mask = m_mask[:, None] & n_mask[None, :]
    x_ptrs = (
        x_ptr + m_offsets[:, None] * stride_xm + n_offsets[None, :] * stride_xn
    )

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    sum_vals = tl.sum(x, axis=1, dtype=tl.float64)

    tl.atomic_add(out_ptr + m_offsets, sum_vals, mask=m_mask)


@triton.jit
def _sum_kernel_3d_atomic_fp64(
    x_ptr,
    out_ptr,
    M,
    N,
    I,
    stride_xo,
    stride_xr,
    stride_xi,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N

    o_idx = m_offsets // I
    i_idx = m_offsets % I

    mask = m_mask[:, None] & n_mask[None, :]
    x_ptrs = x_ptr + (
        o_idx[:, None] * stride_xo
        + n_offsets[None, :] * stride_xr
        + i_idx[:, None] * stride_xi
    )

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    sum_vals = tl.sum(x, axis=1, dtype=tl.float64)

    tl.atomic_add(out_ptr + m_offsets, sum_vals, mask=m_mask)


@triton.jit
def _sum_kernel_2d_loop_fp64(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float64)

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        mask = m_mask[:, None] & n_mask[None, :]
        x_ptrs = (
            x_ptr
            + m_offsets[:, None] * stride_xm
            + n_offsets[None, :] * stride_xn
        )

        x = tl.load(x_ptrs, mask=mask, other=0.0)
        acc += tl.sum(x, axis=1, dtype=tl.float64)

    out_ptrs = out_ptr + m_offsets
    tl.store(out_ptrs, acc, mask=m_mask)


@triton.jit
def _sum_kernel_3d_loop_fp64(
    x_ptr,
    out_ptr,
    M,
    N,
    I,
    stride_xo,
    stride_xr,
    stride_xi,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    o_idx = m_offsets // I
    i_idx = m_offsets % I
    base_ptrs = x_ptr + (o_idx * stride_xo + i_idx * stride_xi)

    acc = tl.zeros((BLOCK_M,), dtype=tl.float64)

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        mask = m_mask[:, None] & n_mask[None, :]
        x_ptrs = base_ptrs[:, None] + n_offsets[None, :] * stride_xr

        x = tl.load(x_ptrs, mask=mask, other=0.0)
        acc += tl.sum(x, axis=1, dtype=tl.float64)

    out_ptrs = out_ptr + m_offsets
    tl.store(out_ptrs, acc, mask=m_mask)


@triton.jit
def _sum_kernel_3d_loop_transpose(
    x_ptr,
    out_ptr,
    M,
    N,
    I,
    stride_xo,
    stride_xr,
    stride_xi,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    o_idx = m_offsets // I
    i_idx = m_offsets % I
    base_ptrs = x_ptr + o_idx * stride_xo + i_idx * stride_xi

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        # 注意这里把 tile 组织成 [BLOCK_N, BLOCK_M]
        mask = n_mask[:, None] & m_mask[None, :]
        x_ptrs = base_ptrs[None, :] + n_offsets[:, None] * stride_xr

        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x = x.to(tl.float32)

        # 沿着 n 维归约，输出还是 BLOCK_M 个结果
        acc += tl.sum(x, axis=0)

    tl.store(out_ptr + m_offsets, acc, mask=m_mask)


@triton.jit
def _sum_kernel_2d_loop(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        mask = m_mask[:, None] & n_mask[None, :]
        x_ptrs = (
            x_ptr
            + m_offsets[:, None] * stride_xm
            + n_offsets[None, :] * stride_xn
        )

        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x = x.to(tl.float32)

        acc += tl.sum(x, axis=1)

    out_ptrs = out_ptr + m_offsets
    tl.store(out_ptrs, acc, mask=m_mask)


@triton.jit
def _sum_kernel_3d_loop(
    x_ptr,
    out_ptr,
    M,
    N,
    I,
    stride_xo,
    stride_xr,
    stride_xi,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    o_idx = m_offsets // I
    i_idx = m_offsets % I
    base_ptrs = x_ptr + (o_idx * stride_xo + i_idx * stride_xi)

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        mask = m_mask[:, None] & n_mask[None, :]
        x_ptrs = base_ptrs[:, None] + n_offsets[None, :] * stride_xr

        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x = x.to(tl.float32)

        acc += tl.sum(x, axis=1)

    out_ptrs = out_ptr + m_offsets
    tl.store(out_ptrs, acc, mask=m_mask)


@triton.jit
def _sum_kernel_2d_atomic(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N

    mask = m_mask[:, None] & n_mask[None, :]
    x_ptrs = (
        x_ptr + m_offsets[:, None] * stride_xm + n_offsets[None, :] * stride_xn
    )

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    x = x.to(tl.float32)

    sum_vals = tl.sum(x, axis=1)

    out_ptrs = out_ptr + m_offsets
    tl.atomic_add(out_ptrs, sum_vals, mask=m_mask)


@triton.jit
def _sum_kernel_3d_atomic(
    x_ptr,
    out_ptr,
    M,
    N,
    I,
    stride_xo,
    stride_xr,
    stride_xi,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N

    o_idx = m_offsets // I
    i_idx = m_offsets % I

    mask = m_mask[:, None] & n_mask[None, :]
    x_ptrs = x_ptr + (
        o_idx[:, None] * stride_xo
        + n_offsets[None, :] * stride_xr
        + i_idx[:, None] * stride_xi
    )

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    x = x.to(tl.float32)

    sum_vals = tl.sum(x, axis=1)

    out_ptrs = out_ptr + m_offsets
    tl.atomic_add(out_ptrs, sum_vals, mask=m_mask)
