# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Numerically stable Hygon-specialized normalization kernels.

LayerNorm uses a centered second moment for a resident row and mergeable
Welford statistics for a row that spans multiple tiles. BatchNorm uses the
same mergeable Welford formulation for both the contiguous NCHW and generic
paths. This avoids the catastrophic cancellation of ``E[x**2] - E[x]**2``
without changing the public graph ABI or the one-program-per-row/channel
ownership model.
"""

import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    x_ptr,
    y_ptr,
    mean_ptr,
    inv_variance_ptr,
    weight_ptr,
    bias_ptr,
    M,
    eps: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RETURN_STATS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    row_active = rows < M
    inv_n: tl.constexpr = 1.0 / N

    if BLOCK_SIZE >= N:
        columns = tl.arange(0, BLOCK_SIZE)[None, :]
        column_active = columns < N
        active = row_active[:, None] & column_active
        values = tl.load(
            x_ptr + rows[:, None] * N + columns,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        mean = tl.sum(values, axis=1) * inv_n
        centered = tl.where(active, values - mean[:, None], 0.0)
        variance = tl.sum(centered * centered, axis=1) * inv_n
        inv_variance = tl.rsqrt(variance + eps)
        if RETURN_STATS:
            tl.store(mean_ptr + rows, mean, mask=row_active)
            tl.store(
                inv_variance_ptr + rows,
                inv_variance,
                mask=row_active,
            )
        normalized = centered * inv_variance[:, None]
        if HAS_WEIGHT:
            weight = tl.load(
                weight_ptr + columns,
                mask=column_active,
                other=0.0,
            ).to(tl.float32)
            normalized *= weight
        if HAS_BIAS:
            bias = tl.load(
                bias_ptr + columns,
                mask=column_active,
                other=0.0,
            ).to(tl.float32)
            normalized += bias
        tl.store(
            y_ptr + rows[:, None] * N + columns,
            normalized.to(y_ptr.dtype.element_ty),
            mask=active,
        )
    else:
        running_mean = tl.zeros((ROWS_PER_PROGRAM,), dtype=tl.float32)
        running_m2 = tl.zeros((ROWS_PER_PROGRAM,), dtype=tl.float32)
        running_count = tl.zeros((1,), dtype=tl.float32)
        for offset in range(0, N, BLOCK_SIZE):
            columns = offset + tl.arange(0, BLOCK_SIZE)[None, :]
            column_active = columns < N
            active = row_active[:, None] & column_active
            values = tl.load(
                x_ptr + rows[:, None] * N + columns,
                mask=active,
                other=0.0,
            ).to(tl.float32)
            chunk_count = tl.sum(column_active.to(tl.float32), axis=1)
            chunk_mean = tl.sum(values, axis=1) / chunk_count
            centered = tl.where(active, values - chunk_mean[:, None], 0.0)
            chunk_m2 = tl.sum(centered * centered, axis=1)

            merged_count = running_count + chunk_count
            delta = chunk_mean - running_mean
            running_m2 += chunk_m2 + (
                delta * delta * running_count * chunk_count / merged_count
            )
            running_mean += delta * chunk_count / merged_count
            running_count = merged_count

        mean = running_mean
        variance = running_m2 * inv_n
        inv_variance = tl.rsqrt(variance + eps)
        if RETURN_STATS:
            tl.store(mean_ptr + rows, mean, mask=row_active)
            tl.store(
                inv_variance_ptr + rows,
                inv_variance,
                mask=row_active,
            )

        for offset in range(0, N, BLOCK_SIZE):
            columns = offset + tl.arange(0, BLOCK_SIZE)[None, :]
            column_active = columns < N
            active = row_active[:, None] & column_active
            values = tl.load(
                x_ptr + rows[:, None] * N + columns,
                mask=active,
                other=0.0,
            ).to(tl.float32)
            normalized = (values - mean[:, None]) * inv_variance[:, None]
            if HAS_WEIGHT:
                weight = tl.load(
                    weight_ptr + columns,
                    mask=column_active,
                    other=0.0,
                ).to(tl.float32)
                normalized *= weight
            if HAS_BIAS:
                bias = tl.load(
                    bias_ptr + columns,
                    mask=column_active,
                    other=0.0,
                ).to(tl.float32)
                normalized += bias
            tl.store(
                y_ptr + rows[:, None] * N + columns,
                normalized.to(y_ptr.dtype.element_ty),
                mask=active,
            )


@triton.jit
def _logical_offset(
    logical,
    BLOCK_SIZE: tl.constexpr,
    DIM_0: tl.constexpr,
    DIM_1: tl.constexpr,
    DIM_2: tl.constexpr,
    DIM_3: tl.constexpr,
    DIM_4: tl.constexpr,
    DIM_5: tl.constexpr,
    DIM_6: tl.constexpr,
    DIM_7: tl.constexpr,
    STRIDE_0: tl.constexpr,
    STRIDE_1: tl.constexpr,
    STRIDE_2: tl.constexpr,
    STRIDE_3: tl.constexpr,
    STRIDE_4: tl.constexpr,
    STRIDE_5: tl.constexpr,
    STRIDE_6: tl.constexpr,
    STRIDE_7: tl.constexpr,
):
    remaining = logical
    offset = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    coordinate = remaining % DIM_7
    remaining //= DIM_7
    offset += coordinate * STRIDE_7
    coordinate = remaining % DIM_6
    remaining //= DIM_6
    offset += coordinate * STRIDE_6
    coordinate = remaining % DIM_5
    remaining //= DIM_5
    offset += coordinate * STRIDE_5
    coordinate = remaining % DIM_4
    remaining //= DIM_4
    offset += coordinate * STRIDE_4
    coordinate = remaining % DIM_3
    remaining //= DIM_3
    offset += coordinate * STRIDE_3
    coordinate = remaining % DIM_2
    remaining //= DIM_2
    offset += coordinate * STRIDE_2
    coordinate = remaining % DIM_1
    remaining //= DIM_1
    offset += coordinate * STRIDE_1
    offset += (remaining % DIM_0) * STRIDE_0
    return offset


@triton.jit
def batch_norm_nchw_kernel(
    x_ptr,
    y_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    saved_mean_ptr,
    saved_inv_var_ptr,
    next_running_mean_ptr,
    next_running_var_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_TRAINING: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RUNNING_STATS: tl.constexpr,
    RETURN_STATS: tl.constexpr,
):
    channel = tl.program_id(0).to(tl.int64)
    batch_block: tl.constexpr = triton.next_power_of_2(N)
    spatial_block: tl.constexpr = BLOCK_SIZE // batch_block
    batches = tl.arange(0, batch_block)[:, None]
    batch_active = batches < N
    batch_mean = 0.0
    batch_m2 = 0.0
    sample_count = 0.0

    # Keeping batch as the outer tile dimension makes each row a contiguous
    # spatial transaction.  Large Hygon-only BLOCK_SIZE candidates amortize
    # the loop/address overhead without changing the one-channel ownership.
    for start in range(0, S, spatial_block):
        spatial = start + tl.arange(0, spatial_block)[None, :]
        active = batch_active & (spatial < S)
        offsets = (batches * C + channel) * S + spatial
        values = tl.load(x_ptr + offsets, mask=active, other=0.0).to(
            tl.float32
        )
        flat_values = tl.reshape(values, (BLOCK_SIZE,))
        flat_active = tl.reshape(active, (BLOCK_SIZE,))
        chunk_count = tl.sum(flat_active.to(tl.float32), axis=0)
        chunk_mean = tl.sum(flat_values, axis=0) / chunk_count
        centered = tl.where(flat_active, flat_values - chunk_mean, 0.0)
        chunk_m2 = tl.sum(centered * centered, axis=0)

        merged_count = sample_count + chunk_count
        delta = chunk_mean - batch_mean
        batch_m2 += (
            chunk_m2
            + delta * delta * sample_count * chunk_count / merged_count
        )
        batch_mean += delta * chunk_count / merged_count
        sample_count = merged_count

    count: tl.constexpr = N * S
    variance = batch_m2 / sample_count
    inv_variance = tl.rsqrt(variance + eps)
    if RETURN_STATS:
        tl.store(saved_mean_ptr + channel, batch_mean)
        tl.store(saved_inv_var_ptr + channel, inv_variance)
    if HAS_RUNNING_STATS:
        previous_mean = tl.load(mean_ptr + channel).to(tl.float32)
        previous_variance = tl.load(var_ptr + channel).to(tl.float32)
        unbiased = batch_m2 / (count - 1) if count > 1 else variance
        tl.store(
            next_running_mean_ptr + channel,
            previous_mean * (1.0 - momentum) + batch_mean * momentum,
        )
        tl.store(
            next_running_var_ptr + channel,
            previous_variance * (1.0 - momentum) + unbiased * momentum,
        )

    weight = (
        tl.load(weight_ptr + channel).to(tl.float32) if HAS_WEIGHT else 1.0
    )
    bias = tl.load(bias_ptr + channel).to(tl.float32) if HAS_BIAS else 0.0
    for start in range(0, S, spatial_block):
        spatial = start + tl.arange(0, spatial_block)[None, :]
        active = batch_active & (spatial < S)
        offsets = (batches * C + channel) * S + spatial
        values = tl.load(x_ptr + offsets, mask=active, other=0.0).to(
            tl.float32
        )
        normalized = (values - batch_mean) * inv_variance * weight + bias
        tl.store(
            y_ptr + offsets,
            normalized.to(y_ptr.dtype.element_ty),
            mask=active,
        )


@triton.jit
def batch_norm_kernel(
    x_ptr,
    y_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    saved_mean_ptr,
    saved_inv_var_ptr,
    next_running_mean_ptr,
    next_running_var_ptr,
    N,
    C,
    S,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_TRAINING: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RUNNING_STATS: tl.constexpr,
    RETURN_STATS: tl.constexpr,
    STRIDED: tl.constexpr,
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
):
    channel = tl.program_id(0).to(tl.int64)
    count = N * S
    batch_mean = 0.0
    batch_m2 = 0.0
    sample_count = 0.0

    for start in range(0, count, BLOCK_SIZE):
        item = start + tl.arange(0, BLOCK_SIZE)
        active = item < count
        logical = (item // S) * C * S + channel * S + item % S
        if STRIDED:
            input_offsets = _logical_offset(
                logical,
                BLOCK_SIZE,
                DIM_0,
                DIM_1,
                DIM_2,
                DIM_3,
                DIM_4,
                DIM_5,
                DIM_6,
                DIM_7,
                INPUT_STRIDE_0,
                INPUT_STRIDE_1,
                INPUT_STRIDE_2,
                INPUT_STRIDE_3,
                INPUT_STRIDE_4,
                INPUT_STRIDE_5,
                INPUT_STRIDE_6,
                INPUT_STRIDE_7,
            )
        else:
            input_offsets = logical
        values = tl.load(x_ptr + input_offsets, mask=active, other=0.0).to(
            tl.float32
        )
        chunk_count = tl.sum(active.to(tl.float32), axis=0)
        chunk_mean = tl.sum(values, axis=0) / chunk_count
        centered = tl.where(active, values - chunk_mean, 0.0)
        chunk_m2 = tl.sum(centered * centered, axis=0)

        merged_count = sample_count + chunk_count
        delta = chunk_mean - batch_mean
        batch_m2 += (
            chunk_m2
            + delta * delta * sample_count * chunk_count / merged_count
        )
        batch_mean += delta * chunk_count / merged_count
        sample_count = merged_count

    variance = batch_m2 / sample_count
    inv_variance = tl.rsqrt(variance + eps)
    if RETURN_STATS:
        tl.store(saved_mean_ptr + channel, batch_mean)
        tl.store(saved_inv_var_ptr + channel, inv_variance)
    if HAS_RUNNING_STATS:
        previous_mean = tl.load(mean_ptr + channel).to(tl.float32)
        previous_variance = tl.load(var_ptr + channel).to(tl.float32)
        unbiased = tl.where(
            sample_count > 1.0,
            batch_m2 / tl.maximum(sample_count - 1.0, 1.0),
            variance,
        )
        tl.store(
            next_running_mean_ptr + channel,
            previous_mean * (1.0 - momentum) + batch_mean * momentum,
        )
        tl.store(
            next_running_var_ptr + channel,
            previous_variance * (1.0 - momentum) + unbiased * momentum,
        )

    weight = (
        tl.load(weight_ptr + channel).to(tl.float32) if HAS_WEIGHT else 1.0
    )
    bias = tl.load(bias_ptr + channel).to(tl.float32) if HAS_BIAS else 0.0
    for start in range(0, count, BLOCK_SIZE):
        item = start + tl.arange(0, BLOCK_SIZE)
        active = item < count
        logical = (item // S) * C * S + channel * S + item % S
        if STRIDED:
            input_offsets = _logical_offset(
                logical,
                BLOCK_SIZE,
                DIM_0,
                DIM_1,
                DIM_2,
                DIM_3,
                DIM_4,
                DIM_5,
                DIM_6,
                DIM_7,
                INPUT_STRIDE_0,
                INPUT_STRIDE_1,
                INPUT_STRIDE_2,
                INPUT_STRIDE_3,
                INPUT_STRIDE_4,
                INPUT_STRIDE_5,
                INPUT_STRIDE_6,
                INPUT_STRIDE_7,
            )
            output_offsets = _logical_offset(
                logical,
                BLOCK_SIZE,
                DIM_0,
                DIM_1,
                DIM_2,
                DIM_3,
                DIM_4,
                DIM_5,
                DIM_6,
                DIM_7,
                OUTPUT_STRIDE_0,
                OUTPUT_STRIDE_1,
                OUTPUT_STRIDE_2,
                OUTPUT_STRIDE_3,
                OUTPUT_STRIDE_4,
                OUTPUT_STRIDE_5,
                OUTPUT_STRIDE_6,
                OUTPUT_STRIDE_7,
            )
        else:
            input_offsets = logical
            output_offsets = logical
        values = tl.load(x_ptr + input_offsets, mask=active, other=0.0).to(
            tl.float32
        )
        normalized = (values - batch_mean) * inv_variance * weight + bias
        tl.store(
            y_ptr + output_offsets,
            normalized.to(y_ptr.dtype.element_ty),
            mask=active,
        )
