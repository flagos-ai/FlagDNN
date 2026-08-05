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
    C: tl.constexpr,
    S: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    STAT_IS_INV_VARIANCE: tl.constexpr,
):
    program = tl.program_id(0).to(tl.int64)
    BLOCK_S: tl.constexpr = triton.next_power_of_2(S)
    if BLOCK_S > BLOCK_SIZE:
        BLOCK_S = BLOCK_SIZE
    BLOCK_C: tl.constexpr = BLOCK_SIZE // BLOCK_S
    SPATIAL_BLOCKS: tl.constexpr = (S + BLOCK_S - 1) // BLOCK_S
    CHANNEL_BLOCKS: tl.constexpr = (C + BLOCK_C - 1) // BLOCK_C

    spatial_block = program % SPATIAL_BLOCKS
    remaining = program // SPATIAL_BLOCKS
    channel_block = remaining % CHANNEL_BLOCKS
    batch = remaining // CHANNEL_BLOCKS
    channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)[:, None]
    spatial = spatial_block * BLOCK_S + tl.arange(0, BLOCK_S)[None, :]
    channel_active = channels < C
    active = channel_active & (spatial < S)
    offsets = (batch * C + channels) * S + spatial

    values = tl.load(x_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channels, mask=channel_active, other=0.0).to(
        tl.float32
    )
    statistic = tl.load(
        stat_ptr + channels, mask=channel_active, other=0.0
    ).to(tl.float32)
    inv_variance = (
        statistic if STAT_IS_INV_VARIANCE else tl.rsqrt(statistic + eps)
    )
    weight = (
        tl.load(weight_ptr + channels, mask=channel_active, other=1.0).to(
            tl.float32
        )
        if HAS_WEIGHT
        else 1.0
    )
    bias = (
        tl.load(bias_ptr + channels, mask=channel_active, other=0.0).to(
            tl.float32
        )
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


# -----------------------------------------------------------------------------
# Kernel algorithm variants. Native dispatch selects only registry-declared
# entry points; auxiliary kernels remain available to explicit compiler policy.
# -----------------------------------------------------------------------------


@triton.jit
def batch_norm_batch_norm_inference_kernel_variant(
    x_ptr,
    y_ptr,
    mean_ptr,
    stat_ptr,
    weight_ptr,
    bias_ptr,
    total_elements,
    C,
    S,
    eps,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    STAT_IS_INV_VARIANCE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    c_idx = (offsets // S) % C

    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    mean = tl.load(mean_ptr + c_idx, mask=mask).to(tl.float32)
    stat = tl.load(stat_ptr + c_idx, mask=mask).to(tl.float32)
    weight = (
        tl.load(weight_ptr + c_idx, mask=mask).to(tl.float32)
        if HAS_WEIGHT
        else 1.0
    )
    bias = (
        tl.load(bias_ptr + c_idx, mask=mask).to(tl.float32)
        if HAS_BIAS
        else 0.0
    )

    rstd = stat if STAT_IS_INV_VARIANCE else 1.0 / tl.sqrt(stat + eps)
    y = (x - mean) * rstd * weight + bias
    tl.store(y_ptr + offsets, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def batch_norm_fused_kernel_optimized_(
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
    eps,
    momentum,
    BLOCK_SIZE: tl.constexpr,
    IS_TRAINING: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RUNNING_STATS: tl.constexpr,
    RETURN_STATS: tl.constexpr,
):
    c = tl.program_id(0)
    M = N * S

    stride_gap = S * (C - 1)
    base_x_ptr = x_ptr + c * S
    base_y_ptr = y_ptr + c * S

    if IS_TRAINING:
        sum_x = 0.0
        sum_x2 = 0.0

        for i_offset in range(0, M, BLOCK_SIZE):
            i = i_offset + tl.arange(0, BLOCK_SIZE)
            mask = i < M
            mem_ptrs = base_x_ptr + i + (i // S) * stride_gap
            x = tl.load(mem_ptrs, mask=mask, other=0.0).to(tl.float32)
            sum_x += tl.sum(x, axis=0)
            sum_x2 += tl.sum(x * x, axis=0)

        mean = sum_x / M
        var = (sum_x2 / M) - (mean * mean)
        var = tl.maximum(var, 0.0)
        rstd = 1.0 / tl.sqrt(var + eps)

        if RETURN_STATS:
            tl.store(saved_mean_ptr + c, mean)
            tl.store(saved_inv_var_ptr + c, rstd)

        if HAS_RUNNING_STATS:
            rm = tl.load(mean_ptr + c).to(tl.float32)
            rv = tl.load(var_ptr + c).to(tl.float32)
            unbiased_var = var * (M / (M - 1)) if M > 1 else var
            new_rm = rm * (1.0 - momentum) + mean * momentum
            new_rv = rv * (1.0 - momentum) + unbiased_var * momentum
            if RETURN_STATS:
                tl.store(next_running_mean_ptr + c, new_rm)
                tl.store(next_running_var_ptr + c, new_rv)
            else:
                tl.store(mean_ptr + c, new_rm.to(mean_ptr.dtype.element_ty))
                tl.store(var_ptr + c, new_rv.to(var_ptr.dtype.element_ty))
    else:
        mean = tl.load(mean_ptr + c).to(tl.float32)
        var = tl.load(var_ptr + c).to(tl.float32)
        rstd = 1.0 / tl.sqrt(var + eps)

    weight = tl.load(weight_ptr + c).to(tl.float32) if HAS_WEIGHT else 1.0
    bias = tl.load(bias_ptr + c).to(tl.float32) if HAS_BIAS else 0.0

    for i_offset in range(0, M, BLOCK_SIZE):
        i = i_offset + tl.arange(0, BLOCK_SIZE)
        mask = i < M
        mem_ptrs = base_x_ptr + i + (i // S) * stride_gap
        x = tl.load(mem_ptrs, mask=mask).to(tl.float32)
        y = (x - mean) * rstd * weight + bias
        out_ptrs = base_y_ptr + i + (i // S) * stride_gap
        tl.store(out_ptrs, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def layer_norm_layer_norm_kernel_variant(
    x_ptr,
    y_ptr,
    mean_ptr,
    rstd_ptr,
    weight_ptr,
    bias_ptr,
    M,
    eps,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RETURN_STATS: tl.constexpr,
):
    row_idx = tl.program_id(0)

    x_row_ptr = x_ptr + row_idx * N
    y_row_ptr = y_ptr + row_idx * N
    inv_n: tl.constexpr = 1.0 / N

    if BLOCK_SIZE >= N:
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) * inv_n
        sum_x2 = tl.sum(x * x, axis=0)
        var = tl.maximum((sum_x2 * inv_n) - (mean * mean), 0.0)
        rstd = tl.math.rsqrt(var + eps)
        if RETURN_STATS:
            tl.store(mean_ptr + row_idx, mean)
            tl.store(rstd_ptr + row_idx, rstd)
        x_hat = (x - mean) * rstd
        if HAS_WEIGHT:
            weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(
                tl.float32
            )
            x_hat = x_hat * weight

        if HAS_BIAS:
            bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(
                tl.float32
            )
            x_hat = x_hat + bias

        y = x_hat.to(x_ptr.dtype.element_ty)
        tl.store(y_row_ptr + cols, y, mask=mask)
    else:
        sum_x = 0.0
        sum_x2 = 0.0
        for offset in range(0, N, BLOCK_SIZE):
            cols = offset + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            sum_x += tl.sum(x, axis=0)
            sum_x2 += tl.sum(x * x, axis=0)

        mean = sum_x * inv_n
        var = tl.maximum((sum_x2 * inv_n) - (mean * mean), 0.0)
        rstd = tl.math.rsqrt(var + eps)
        if RETURN_STATS:
            tl.store(mean_ptr + row_idx, mean)
            tl.store(rstd_ptr + row_idx, rstd)

        for offset in range(0, N, BLOCK_SIZE):
            cols = offset + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            x_hat = (x - mean) * rstd
            if HAS_WEIGHT:
                weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(
                    tl.float32
                )
                x_hat = x_hat * weight
            if HAS_BIAS:
                bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(
                    tl.float32
                )
                x_hat = x_hat + bias
            y = x_hat.to(x_ptr.dtype.element_ty)
            tl.store(y_row_ptr + cols, y, mask=mask)


@triton.jit
def rms_norm_rms_norm_kernel_variant(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    rstd_ptr,
    M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RETURN_STATS: tl.constexpr,
):
    row_idx = tl.program_id(0)

    x_row_ptr = x_ptr + row_idx * N
    y_row_ptr = y_ptr + row_idx * N

    sum_squares = 0.0
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sum_squares += tl.sum(x * x, axis=0)

    rrms = tl.math.rsqrt((sum_squares / N) + eps)
    if RETURN_STATS:
        tl.store(rstd_ptr + row_idx, rrms)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = x * rrms

        if HAS_WEIGHT:
            weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(
                tl.float32
            )
            x_hat = x_hat * weight

        if HAS_BIAS:
            bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(
                tl.float32
            )
            x_hat = x_hat + bias

        y = x_hat.to(x_ptr.dtype.element_ty)
        tl.store(y_row_ptr + cols, y, mask=mask)


@triton.jit
def _hadamard_sign(cols, k: tl.constexpr):
    bits = cols & k
    parity = (
        (bits & 1) ^ ((bits >> 1) & 1) ^ ((bits >> 2) & 1) ^ ((bits >> 3) & 1)
    )
    return tl.where(parity == 0, 1.0, -1.0)


@triton.jit
def _rmsnorm_rht_rows_kernel(
    x_ptr,
    w_ptr,
    o_ptr,
    row_amax_ptr,
    eps,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    x_row = x_ptr + row * N
    o_row = o_ptr + row * N

    sum_squares = 0.0
    for start in range(0, N, BLOCK_N):
        offsets = start + cols
        mask = offsets < N
        x = tl.load(x_row + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_squares += tl.sum(x * x, axis=0)

    rrms = tl.rsqrt(sum_squares / N + eps)
    max_abs = 0.0

    for start in range(0, N, BLOCK_N):
        offsets = start + cols
        mask = offsets < N
        had_cols = offsets & 15
        group_base = offsets - had_cols
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k in tl.static_range(0, 16):
            source_cols = group_base + k
            x = tl.load(x_row + source_cols, mask=mask, other=0.0).to(
                tl.float32
            )
            w = tl.load(w_ptr + source_cols, mask=mask, other=0.0).to(
                tl.float32
            )
            sign = _hadamard_sign(had_cols, k)
            acc += x * rrms * w * sign

        out = acc * 0.25
        max_abs = tl.maximum(max_abs, tl.max(tl.abs(out), axis=0))
        tl.store(o_row + offsets, out.to(o_ptr.dtype.element_ty), mask=mask)

    tl.store(row_amax_ptr + row, max_abs)


@triton.jit
def _rows_to_cta_amax_kernel(
    row_amax_ptr,
    amax_ptr,
    ROWS_PER_CTA: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    cta = tl.program_id(0)
    offsets = cta * ROWS_PER_CTA + tl.arange(0, BLOCK_R)
    mask = tl.arange(0, BLOCK_R) < ROWS_PER_CTA
    values = tl.load(row_amax_ptr + offsets, mask=mask, other=0.0)
    tl.store(amax_ptr + cta, tl.max(values, axis=0))
