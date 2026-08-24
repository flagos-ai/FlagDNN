# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compiler-safe normalization kernels from ``flag_dnn.ops``."""

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
        sum_squares = tl.sum(values * values, axis=1)
        variance = tl.maximum(sum_squares * inv_n - mean * mean, 0.0)
        inv_variance = tl.rsqrt(variance + eps)
        if RETURN_STATS:
            tl.store(mean_ptr + rows, mean, mask=row_active)
            tl.store(
                inv_variance_ptr + rows,
                inv_variance,
                mask=row_active,
            )
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
    else:
        sum_values = tl.zeros((ROWS_PER_PROGRAM,), dtype=tl.float32)
        sum_squares = tl.zeros((ROWS_PER_PROGRAM,), dtype=tl.float32)
        for offset in range(0, N, BLOCK_SIZE):
            columns = offset + tl.arange(0, BLOCK_SIZE)[None, :]
            column_active = columns < N
            active = row_active[:, None] & column_active
            values = tl.load(
                x_ptr + rows[:, None] * N + columns,
                mask=active,
                other=0.0,
            ).to(tl.float32)
            sum_values += tl.sum(values, axis=1)
            sum_squares += tl.sum(values * values, axis=1)

        mean = sum_values * inv_n
        variance = tl.maximum(sum_squares * inv_n - mean * mean, 0.0)
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
def rms_norm_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    inv_variance_ptr,
    M,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RETURN_STATS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    row_active = rows < M

    if BLOCK_SIZE >= N:
        columns = tl.arange(0, BLOCK_SIZE)[None, :]
        column_active = columns < N
        active = row_active[:, None] & column_active
        values = tl.load(
            x_ptr + rows[:, None] * N + columns,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        inv_variance = tl.rsqrt(tl.sum(values * values, axis=1) / N + eps)
        if RETURN_STATS:
            tl.store(
                inv_variance_ptr + rows,
                inv_variance,
                mask=row_active,
            )
        normalized = values * inv_variance[:, None]
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
        sum_squares = tl.zeros((ROWS_PER_PROGRAM,), dtype=tl.float32)
        for offset in range(0, N, BLOCK_SIZE):
            columns = offset + tl.arange(0, BLOCK_SIZE)[None, :]
            active = row_active[:, None] & (columns < N)
            values = tl.load(
                x_ptr + rows[:, None] * N + columns,
                mask=active,
                other=0.0,
            ).to(tl.float32)
            sum_squares += tl.sum(values * values, axis=1)

        inv_variance = tl.rsqrt(sum_squares / N + eps)
        if RETURN_STATS:
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
            normalized = values * inv_variance[:, None]
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
    channel = tl.program_id(0)
    batch_block: tl.constexpr = triton.next_power_of_2(N)
    spatial_block: tl.constexpr = BLOCK_SIZE // batch_block
    batches = tl.arange(0, batch_block)[:, None]
    batch_active = batches < N
    sum_values = 0.0
    sum_squares = 0.0

    for start in range(0, S, spatial_block):
        spatial = start + tl.arange(0, spatial_block)[None, :]
        active = batch_active & (spatial < S)
        offsets = (batches * C + channel) * S + spatial
        values = tl.load(x_ptr + offsets, mask=active, other=0.0).to(
            tl.float32
        )
        flat_values = tl.reshape(values, (BLOCK_SIZE,))
        sum_values += tl.sum(flat_values, axis=0)
        sum_squares += tl.sum(flat_values * flat_values, axis=0)

    count: tl.constexpr = N * S
    batch_mean = sum_values / count
    variance = tl.maximum(sum_squares / count - batch_mean * batch_mean, 0.0)
    inv_variance = tl.rsqrt(variance + eps)
    if RETURN_STATS:
        tl.store(saved_mean_ptr + channel, batch_mean)
        tl.store(saved_inv_var_ptr + channel, inv_variance)
    if HAS_RUNNING_STATS:
        previous_mean = tl.load(mean_ptr + channel).to(tl.float32)
        previous_variance = tl.load(var_ptr + channel).to(tl.float32)
        unbiased = variance * count / (count - 1) if count > 1 else variance
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
    sum_values = 0.0
    sum_squares = 0.0

    for start in range(0, count, BLOCK_SIZE):
        item = start + tl.arange(0, BLOCK_SIZE)
        active = item < count
        logical = (item // S) * C * S + channel * S + item % S
        if STRIDED:
            remaining = logical
            input_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
            coordinate = remaining % DIM_7
            remaining //= DIM_7
            input_offsets += coordinate * INPUT_STRIDE_7
            coordinate = remaining % DIM_6
            remaining //= DIM_6
            input_offsets += coordinate * INPUT_STRIDE_6
            coordinate = remaining % DIM_5
            remaining //= DIM_5
            input_offsets += coordinate * INPUT_STRIDE_5
            coordinate = remaining % DIM_4
            remaining //= DIM_4
            input_offsets += coordinate * INPUT_STRIDE_4
            coordinate = remaining % DIM_3
            remaining //= DIM_3
            input_offsets += coordinate * INPUT_STRIDE_3
            coordinate = remaining % DIM_2
            remaining //= DIM_2
            input_offsets += coordinate * INPUT_STRIDE_2
            coordinate = remaining % DIM_1
            remaining //= DIM_1
            input_offsets += coordinate * INPUT_STRIDE_1
            input_offsets += (remaining % DIM_0) * INPUT_STRIDE_0
        else:
            input_offsets = logical
        values = tl.load(x_ptr + input_offsets, mask=active, other=0.0).to(
            tl.float32
        )
        sum_values += tl.sum(values, axis=0)
        sum_squares += tl.sum(values * values, axis=0)

    batch_mean = sum_values / count
    variance = tl.maximum(sum_squares / count - batch_mean * batch_mean, 0.0)
    inv_variance = tl.rsqrt(variance + eps)
    if RETURN_STATS:
        tl.store(saved_mean_ptr + channel, batch_mean)
        tl.store(saved_inv_var_ptr + channel, inv_variance)
    if HAS_RUNNING_STATS:
        previous_mean = tl.load(mean_ptr + channel).to(tl.float32)
        previous_variance = tl.load(var_ptr + channel).to(tl.float32)
        unbiased = tl.where(
            count > 1, variance * count / (count - 1), variance
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
            remaining = logical
            input_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
            output_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
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


@triton.jit
def batch_norm_inference_nchw_kernel(
    x_ptr,
    mean_ptr,
    stat_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    STAT_IS_INV_VARIANCE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(
        0, BLOCK_SIZE
    )
    active = offsets < N * C * S
    channel = (offsets // S) % C
    values = tl.load(x_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channel, mask=active, other=0.0).to(tl.float32)
    statistic = tl.load(stat_ptr + channel, mask=active, other=0.0).to(
        tl.float32
    )
    inv_variance = (
        statistic if STAT_IS_INV_VARIANCE else tl.rsqrt(statistic + eps)
    )
    weight = (
        tl.load(weight_ptr + channel, mask=active, other=1.0).to(tl.float32)
        if HAS_WEIGHT
        else 1.0
    )
    bias = (
        tl.load(bias_ptr + channel, mask=active, other=0.0).to(tl.float32)
        if HAS_BIAS
        else 0.0
    )
    result = (values - mean) * inv_variance * weight + bias
    tl.store(
        y_ptr + offsets,
        result.to(y_ptr.dtype.element_ty),
        mask=active,
    )


@triton.jit
def batch_norm_inference_kernel(
    x_ptr,
    mean_ptr,
    stat_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    total_elements,
    C,
    S,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    STAT_IS_INV_VARIANCE: tl.constexpr,
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
    logical = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(
        0, BLOCK_SIZE
    )
    active = logical < total_elements
    channel = (logical // S) % C
    if STRIDED:
        remaining = logical
        input_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        output_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
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
    else:
        input_offsets = logical
        output_offsets = logical

    values = tl.load(x_ptr + input_offsets, mask=active, other=0.0).to(
        tl.float32
    )
    mean = tl.load(mean_ptr + channel, mask=active, other=0.0).to(tl.float32)
    statistic = tl.load(stat_ptr + channel, mask=active, other=0.0).to(
        tl.float32
    )
    inv_variance = (
        statistic if STAT_IS_INV_VARIANCE else tl.rsqrt(statistic + eps)
    )
    weight = (
        tl.load(weight_ptr + channel, mask=active, other=1.0).to(tl.float32)
        if HAS_WEIGHT
        else 1.0
    )
    bias = (
        tl.load(bias_ptr + channel, mask=active, other=0.0).to(tl.float32)
        if HAS_BIAS
        else 0.0
    )
    result = (values - mean) * inv_variance * weight + bias
    tl.store(
        y_ptr + output_offsets,
        result.to(y_ptr.dtype.element_ty),
        mask=active,
    )
