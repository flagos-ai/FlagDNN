"""Ascend kernels for normalization."""

import triton
import triton.language as tl

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");


@triton.jit
def _batchnorm_training_tensor_offset(
    logical_index,
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
    remaining = logical_index
    offset = logical_index * 0
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
    coordinate = remaining % DIM_0
    offset += coordinate * STRIDE_0
    return offset


@triton.jit
def batchnorm_training_persistent_kernel(
    x_ptr,
    scale_ptr,
    bias_ptr,
    previous_running_mean_ptr,
    previous_running_variance_ptr,
    y_ptr,
    mean_ptr,
    inv_variance_ptr,
    next_running_mean_ptr,
    next_running_variance_ptr,
    n_elements,
    RANK: tl.constexpr,
    BATCH: tl.constexpr,
    CHANNELS: tl.constexpr,
    SPATIAL: tl.constexpr,
    REDUCTION_ELEMENTS: tl.constexpr,
    EPSILON: tl.constexpr,
    MOMENTUM: tl.constexpr,
    DIM_0: tl.constexpr,
    DIM_1: tl.constexpr,
    DIM_2: tl.constexpr,
    DIM_3: tl.constexpr,
    DIM_4: tl.constexpr,
    DIM_5: tl.constexpr,
    DIM_6: tl.constexpr,
    DIM_7: tl.constexpr,
    X_STRIDE_0: tl.constexpr,
    X_STRIDE_1: tl.constexpr,
    X_STRIDE_2: tl.constexpr,
    X_STRIDE_3: tl.constexpr,
    X_STRIDE_4: tl.constexpr,
    X_STRIDE_5: tl.constexpr,
    X_STRIDE_6: tl.constexpr,
    X_STRIDE_7: tl.constexpr,
    Y_STRIDE_0: tl.constexpr,
    Y_STRIDE_1: tl.constexpr,
    Y_STRIDE_2: tl.constexpr,
    Y_STRIDE_3: tl.constexpr,
    Y_STRIDE_4: tl.constexpr,
    Y_STRIDE_5: tl.constexpr,
    Y_STRIDE_6: tl.constexpr,
    Y_STRIDE_7: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WORKER_COUNT: tl.constexpr,
):
    dense_x: tl.constexpr = (
        (DIM_7 == 1 or X_STRIDE_7 == 1)
        and (DIM_6 == 1 or X_STRIDE_6 == DIM_7)
        and (DIM_5 == 1 or X_STRIDE_5 == DIM_6 * DIM_7)
        and (DIM_4 == 1 or X_STRIDE_4 == DIM_5 * DIM_6 * DIM_7)
        and (DIM_3 == 1 or X_STRIDE_3 == DIM_4 * DIM_5 * DIM_6 * DIM_7)
        and (DIM_2 == 1 or X_STRIDE_2 == DIM_3 * DIM_4 * DIM_5 * DIM_6 * DIM_7)
        and (DIM_1 == 1 or X_STRIDE_1 == DIM_2 * DIM_3 * DIM_4 * DIM_5 * DIM_6 * DIM_7)
        and (
            DIM_0 == 1
            or X_STRIDE_0 == DIM_1 * DIM_2 * DIM_3 * DIM_4 * DIM_5 * DIM_6 * DIM_7
        )
    )
    dense_y: tl.constexpr = (
        (DIM_7 == 1 or Y_STRIDE_7 == 1)
        and (DIM_6 == 1 or Y_STRIDE_6 == DIM_7)
        and (DIM_5 == 1 or Y_STRIDE_5 == DIM_6 * DIM_7)
        and (DIM_4 == 1 or Y_STRIDE_4 == DIM_5 * DIM_6 * DIM_7)
        and (DIM_3 == 1 or Y_STRIDE_3 == DIM_4 * DIM_5 * DIM_6 * DIM_7)
        and (DIM_2 == 1 or Y_STRIDE_2 == DIM_3 * DIM_4 * DIM_5 * DIM_6 * DIM_7)
        and (DIM_1 == 1 or Y_STRIDE_1 == DIM_2 * DIM_3 * DIM_4 * DIM_5 * DIM_6 * DIM_7)
        and (
            DIM_0 == 1
            or Y_STRIDE_0 == DIM_1 * DIM_2 * DIM_3 * DIM_4 * DIM_5 * DIM_6 * DIM_7
        )
    )
    channel_tile: tl.constexpr = 8
    if (
        dense_x
        and dense_y
        and SPATIAL == 1
        and BATCH <= BLOCK_SIZE
        and CHANNELS >= WORKER_COUNT * channel_tile
    ):
        channel_base = tl.program_id(0).to(tl.int64) * channel_tile
        channel_lane = tl.arange(0, channel_tile).to(tl.int64)
        channel_batch_block: tl.constexpr = triton.next_power_of_2(BATCH)
        batches = tl.arange(0, channel_batch_block)[:, None].to(tl.int64)
        batch_mask = batches < BATCH
        while channel_base < CHANNELS:
            channels = channel_base + channel_lane
            channel_mask = channels < CHANNELS
            mask = batch_mask & channel_mask[None, :]
            offset = batches * CHANNELS + channels[None, :]
            value = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
            channel_mean = tl.sum(value, axis=0) / REDUCTION_ELEMENTS
            centered = tl.where(mask, value - channel_mean[None, :], 0.0)
            variance = tl.sum(centered * centered, axis=0) / REDUCTION_ELEMENTS
            variance_with_epsilon = variance + EPSILON
            channel_inv_variance = tl.rsqrt(variance_with_epsilon)
            channel_inv_variance *= 1.5 - (
                0.5
                * variance_with_epsilon
                * channel_inv_variance
                * channel_inv_variance
            )
            tl.store(mean_ptr + channels, channel_mean, mask=channel_mask)
            tl.store(
                inv_variance_ptr + channels,
                channel_inv_variance,
                mask=channel_mask,
            )

            previous_mean = tl.load(
                previous_running_mean_ptr + channels,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            previous_variance = tl.load(
                previous_running_variance_ptr + channels,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            unbiased_variance = tl.where(
                REDUCTION_ELEMENTS > 1,
                variance * REDUCTION_ELEMENTS / (REDUCTION_ELEMENTS - 1),
                variance,
            )
            tl.store(
                next_running_mean_ptr + channels,
                previous_mean * (1.0 - MOMENTUM) + channel_mean * MOMENTUM,
                mask=channel_mask,
            )
            tl.store(
                next_running_variance_ptr + channels,
                previous_variance * (1.0 - MOMENTUM) + unbiased_variance * MOMENTUM,
                mask=channel_mask,
            )

            channel_scale = tl.load(
                scale_ptr + channels, mask=channel_mask, other=0.0
            ).to(tl.float32)
            channel_bias = tl.load(
                bias_ptr + channels, mask=channel_mask, other=0.0
            ).to(tl.float32)
            result = (
                centered * channel_inv_variance[None, :] * channel_scale[None, :]
                + channel_bias[None, :]
            )
            tl.store(
                y_ptr + offset,
                result.to(y_ptr.dtype.element_ty),
                mask=mask,
            )
            channel_base += WORKER_COUNT * channel_tile
        return

    channel = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    while channel < CHANNELS:
        if dense_x and dense_y and BATCH <= BLOCK_SIZE:
            batch_block: tl.constexpr = triton.next_power_of_2(BATCH)
            spatial_block: tl.constexpr = BLOCK_SIZE // batch_block
            batches = tl.arange(0, batch_block)[:, None].to(tl.int64)
            batch_mask = batches < BATCH
            value_sum = 0.0
            for spatial_base in range(0, SPATIAL, spatial_block):
                spatial = spatial_base + tl.arange(0, spatial_block)[None, :].to(
                    tl.int64
                )
                mask = batch_mask & (spatial < SPATIAL)
                offset = (batches * CHANNELS + channel) * SPATIAL + spatial
                value = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
                value_sum += tl.sum(tl.reshape(value, (BLOCK_SIZE,)), axis=0)
            channel_mean = value_sum / REDUCTION_ELEMENTS

            square_sum = 0.0
            for spatial_base in range(0, SPATIAL, spatial_block):
                spatial = spatial_base + tl.arange(0, spatial_block)[None, :].to(
                    tl.int64
                )
                mask = batch_mask & (spatial < SPATIAL)
                offset = (batches * CHANNELS + channel) * SPATIAL + spatial
                value = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
                centered = tl.where(mask, value - channel_mean, 0.0)
                square_sum += tl.sum(
                    tl.reshape(centered * centered, (BLOCK_SIZE,)), axis=0
                )
            variance = square_sum / REDUCTION_ELEMENTS
        else:
            value_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            reduction_base = tl.zeros((), dtype=tl.int64)
            while reduction_base < REDUCTION_ELEMENTS:
                reduction_index = reduction_base + lane
                mask = reduction_index < REDUCTION_ELEMENTS
                logical_index = (
                    (reduction_index // SPATIAL) * CHANNELS * SPATIAL
                    + channel * SPATIAL
                    + reduction_index % SPATIAL
                )
                if dense_x:
                    x_offset = logical_index
                else:
                    x_offset = _batchnorm_training_tensor_offset(
                        logical_index,
                        DIM_0,
                        DIM_1,
                        DIM_2,
                        DIM_3,
                        DIM_4,
                        DIM_5,
                        DIM_6,
                        DIM_7,
                        X_STRIDE_0,
                        X_STRIDE_1,
                        X_STRIDE_2,
                        X_STRIDE_3,
                        X_STRIDE_4,
                        X_STRIDE_5,
                        X_STRIDE_6,
                        X_STRIDE_7,
                    )
                value = tl.load(x_ptr + x_offset, mask=mask, other=0.0).to(tl.float32)
                value_sum += value
                reduction_base += BLOCK_SIZE
            channel_mean = tl.sum(value_sum, axis=0) / REDUCTION_ELEMENTS

            square_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            reduction_base = tl.zeros((), dtype=tl.int64)
            while reduction_base < REDUCTION_ELEMENTS:
                reduction_index = reduction_base + lane
                mask = reduction_index < REDUCTION_ELEMENTS
                logical_index = (
                    (reduction_index // SPATIAL) * CHANNELS * SPATIAL
                    + channel * SPATIAL
                    + reduction_index % SPATIAL
                )
                if dense_x:
                    x_offset = logical_index
                else:
                    x_offset = _batchnorm_training_tensor_offset(
                        logical_index,
                        DIM_0,
                        DIM_1,
                        DIM_2,
                        DIM_3,
                        DIM_4,
                        DIM_5,
                        DIM_6,
                        DIM_7,
                        X_STRIDE_0,
                        X_STRIDE_1,
                        X_STRIDE_2,
                        X_STRIDE_3,
                        X_STRIDE_4,
                        X_STRIDE_5,
                        X_STRIDE_6,
                        X_STRIDE_7,
                    )
                value = tl.load(x_ptr + x_offset, mask=mask, other=0.0).to(tl.float32)
                centered = tl.where(mask, value - channel_mean, 0.0)
                square_sum += centered * centered
                reduction_base += BLOCK_SIZE
            variance = tl.sum(square_sum, axis=0) / REDUCTION_ELEMENTS
        variance_with_epsilon = variance + EPSILON
        channel_inv_variance = tl.rsqrt(variance_with_epsilon)
        channel_inv_variance *= 1.5 - (
            0.5 * variance_with_epsilon * channel_inv_variance * channel_inv_variance
        )
        tl.store(mean_ptr + channel, channel_mean)
        tl.store(inv_variance_ptr + channel, channel_inv_variance)

        previous_mean = tl.load(previous_running_mean_ptr + channel).to(tl.float32)
        previous_variance = tl.load(previous_running_variance_ptr + channel).to(
            tl.float32
        )
        unbiased_variance = tl.where(
            REDUCTION_ELEMENTS > 1,
            variance * REDUCTION_ELEMENTS / (REDUCTION_ELEMENTS - 1),
            variance,
        )
        tl.store(
            next_running_mean_ptr + channel,
            previous_mean * (1.0 - MOMENTUM) + channel_mean * MOMENTUM,
        )
        tl.store(
            next_running_variance_ptr + channel,
            previous_variance * (1.0 - MOMENTUM) + unbiased_variance * MOMENTUM,
        )

        channel_scale = tl.load(scale_ptr + channel).to(tl.float32)
        channel_bias = tl.load(bias_ptr + channel).to(tl.float32)
        if dense_x and dense_y and BATCH <= BLOCK_SIZE:
            for spatial_base in range(0, SPATIAL, spatial_block):
                spatial = spatial_base + tl.arange(0, spatial_block)[None, :].to(
                    tl.int64
                )
                mask = batch_mask & (spatial < SPATIAL)
                offset = (batches * CHANNELS + channel) * SPATIAL + spatial
                value = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
                result = (
                    value - channel_mean
                ) * channel_inv_variance * channel_scale + channel_bias
                tl.store(
                    y_ptr + offset,
                    result.to(y_ptr.dtype.element_ty),
                    mask=mask,
                )
        else:
            reduction_base = tl.zeros((), dtype=tl.int64)
            while reduction_base < REDUCTION_ELEMENTS:
                reduction_index = reduction_base + lane
                mask = reduction_index < REDUCTION_ELEMENTS
                logical_index = (
                    (reduction_index // SPATIAL) * CHANNELS * SPATIAL
                    + channel * SPATIAL
                    + reduction_index % SPATIAL
                )
                if dense_x:
                    x_offset = logical_index
                else:
                    x_offset = _batchnorm_training_tensor_offset(
                        logical_index,
                        DIM_0,
                        DIM_1,
                        DIM_2,
                        DIM_3,
                        DIM_4,
                        DIM_5,
                        DIM_6,
                        DIM_7,
                        X_STRIDE_0,
                        X_STRIDE_1,
                        X_STRIDE_2,
                        X_STRIDE_3,
                        X_STRIDE_4,
                        X_STRIDE_5,
                        X_STRIDE_6,
                        X_STRIDE_7,
                    )
                if dense_y:
                    y_offset = logical_index
                else:
                    y_offset = _batchnorm_training_tensor_offset(
                        logical_index,
                        DIM_0,
                        DIM_1,
                        DIM_2,
                        DIM_3,
                        DIM_4,
                        DIM_5,
                        DIM_6,
                        DIM_7,
                        Y_STRIDE_0,
                        Y_STRIDE_1,
                        Y_STRIDE_2,
                        Y_STRIDE_3,
                        Y_STRIDE_4,
                        Y_STRIDE_5,
                        Y_STRIDE_6,
                        Y_STRIDE_7,
                    )
                value = tl.load(x_ptr + x_offset, mask=mask, other=0.0).to(tl.float32)
                result = (
                    value - channel_mean
                ) * channel_inv_variance * channel_scale + channel_bias
                tl.store(
                    y_ptr + y_offset,
                    result.to(y_ptr.dtype.element_ty),
                    mask=mask,
                )
                reduction_base += BLOCK_SIZE
        channel += WORKER_COUNT


@triton.jit
def layernorm_persistent_kernel(
    x_ptr,
    scale_ptr,
    bias_ptr,
    y_ptr,
    mean_ptr,
    inv_variance_ptr,
    n_elements,
    ROWS: tl.constexpr,
    NORMALIZED_ELEMENTS: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WORKER_COUNT: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    rows_per_worker: tl.constexpr = ROWS // WORKER_COUNT
    extra_rows: tl.constexpr = ROWS % WORKER_COUNT
    extra_before = tl.minimum(program_id, extra_rows)
    row = program_id * rows_per_worker + extra_before
    row_limit = row + rows_per_worker + (program_id < extra_rows).to(tl.int64)
    if BLOCK_SIZE == 1024 and ROWS >= WORKER_COUNT * 2:
        row_tile: tl.constexpr = 8 if ROWS >= WORKER_COUNT * 8 else 4
        row_lane = tl.arange(0, row_tile).to(tl.int64)
        normalized_lane = tl.arange(0, BLOCK_SIZE).to(tl.int64)
        parameter_mask = normalized_lane < NORMALIZED_ELEMENTS
        scale = tl.load(
            scale_ptr + normalized_lane,
            mask=parameter_mask,
            other=0.0,
        )[None, :]
        bias = tl.load(
            bias_ptr + normalized_lane,
            mask=parameter_mask,
            other=0.0,
        )[None, :]
        while row < row_limit:
            rows = row + row_lane
            row_mask = rows < row_limit
            normalized_index = normalized_lane[None, :]
            linear_index = rows[:, None] * NORMALIZED_ELEMENTS + normalized_index
            mask = row_mask[:, None] & parameter_mask[None, :]
            value = tl.load(x_ptr + linear_index, mask=mask, other=0.0).to(tl.float32)
            mean = tl.sum(value, axis=1) / NORMALIZED_ELEMENTS
            centered = tl.where(mask, value - mean[:, None], 0.0)
            inv_variance = tl.rsqrt(
                tl.sum(centered * centered, axis=1) / NORMALIZED_ELEMENTS + EPSILON
            )
            tl.store(mean_ptr + rows, mean, mask=row_mask)
            tl.store(inv_variance_ptr + rows, inv_variance, mask=row_mask)
            result = centered * inv_variance[:, None] * scale + bias
            tl.store(
                y_ptr + linear_index,
                result.to(y_ptr.dtype.element_ty),
                mask=mask,
            )
            row += row_tile
        return
    lane = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    if BLOCK_SIZE >= NORMALIZED_ELEMENTS:
        parameter_mask = lane < NORMALIZED_ELEMENTS
        scale = tl.load(scale_ptr + lane, mask=parameter_mask, other=0.0)
        bias = tl.load(bias_ptr + lane, mask=parameter_mask, other=0.0)
    while row < row_limit:
        if BLOCK_SIZE >= NORMALIZED_ELEMENTS:
            normalized_index = lane
            linear_index = row * NORMALIZED_ELEMENTS + normalized_index
            mask = (normalized_index < NORMALIZED_ELEMENTS) & (
                linear_index < n_elements
            )
            value = tl.load(x_ptr + linear_index, mask=mask, other=0.0).to(tl.float32)
            mean = tl.sum(value, axis=0) / NORMALIZED_ELEMENTS
            centered = tl.where(mask, value - mean, 0.0)
            inv_variance = tl.rsqrt(
                tl.sum(centered * centered, axis=0) / NORMALIZED_ELEMENTS + EPSILON
            )
            tl.store(mean_ptr + row, mean)
            tl.store(inv_variance_ptr + row, inv_variance)
            result = centered * inv_variance * scale + bias
            tl.store(
                y_ptr + linear_index,
                result.to(y_ptr.dtype.element_ty),
                mask=mask,
            )
        else:
            value_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            normalized_base = tl.zeros((), dtype=tl.int64)
            while normalized_base < NORMALIZED_ELEMENTS:
                normalized_index = lane + normalized_base
                linear_index = row * NORMALIZED_ELEMENTS + normalized_index
                mask = (normalized_index < NORMALIZED_ELEMENTS) & (
                    linear_index < n_elements
                )
                value = tl.load(x_ptr + linear_index, mask=mask, other=0.0).to(
                    tl.float32
                )
                value_sum += value
                normalized_base += BLOCK_SIZE
            mean = tl.sum(value_sum, axis=0) / NORMALIZED_ELEMENTS
            tl.store(mean_ptr + row, mean)

            square_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            normalized_base = tl.zeros((), dtype=tl.int64)
            while normalized_base < NORMALIZED_ELEMENTS:
                normalized_index = lane + normalized_base
                linear_index = row * NORMALIZED_ELEMENTS + normalized_index
                mask = (normalized_index < NORMALIZED_ELEMENTS) & (
                    linear_index < n_elements
                )
                value = tl.load(x_ptr + linear_index, mask=mask, other=0.0).to(
                    tl.float32
                )
                centered = tl.where(mask, value - mean, 0.0)
                square_sum += centered * centered
                normalized_base += BLOCK_SIZE
            inv_variance = tl.rsqrt(
                tl.sum(square_sum, axis=0) / NORMALIZED_ELEMENTS + EPSILON
            )
            tl.store(inv_variance_ptr + row, inv_variance)

            normalized_base = tl.zeros((), dtype=tl.int64)
            while normalized_base < NORMALIZED_ELEMENTS:
                normalized_index = lane + normalized_base
                linear_index = row * NORMALIZED_ELEMENTS + normalized_index
                mask = (normalized_index < NORMALIZED_ELEMENTS) & (
                    linear_index < n_elements
                )
                value = tl.load(x_ptr + linear_index, mask=mask, other=0.0).to(
                    tl.float32
                )
                scale = tl.load(scale_ptr + normalized_index, mask=mask, other=0.0).to(
                    tl.float32
                )
                bias = tl.load(bias_ptr + normalized_index, mask=mask, other=0.0).to(
                    tl.float32
                )
                result = (value - mean) * inv_variance * scale + bias
                tl.store(
                    y_ptr + linear_index,
                    result.to(y_ptr.dtype.element_ty),
                    mask=mask,
                )
                normalized_base += BLOCK_SIZE
        row += 1


@triton.jit
def rmsnorm_persistent_kernel(
    x_ptr,
    scale_ptr,
    bias_ptr,
    y_ptr,
    inv_variance_ptr,
    n_elements,
    ROWS: tl.constexpr,
    NORMALIZED_ELEMENTS: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WORKER_COUNT: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    rows_per_worker: tl.constexpr = ROWS // WORKER_COUNT
    extra_rows: tl.constexpr = ROWS % WORKER_COUNT
    extra_before = tl.minimum(program_id, extra_rows)
    row = program_id * rows_per_worker + extra_before
    row_limit = row + rows_per_worker + (program_id < extra_rows).to(tl.int64)
    lane = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    if BLOCK_SIZE >= NORMALIZED_ELEMENTS:
        parameter_mask = lane < NORMALIZED_ELEMENTS
        scale = tl.load(scale_ptr + lane, mask=parameter_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + lane, mask=parameter_mask, other=0.0).to(tl.float32)
    while row < row_limit:
        if BLOCK_SIZE >= NORMALIZED_ELEMENTS:
            normalized_index = lane
            linear_index = row * NORMALIZED_ELEMENTS + normalized_index
            mask = (normalized_index < NORMALIZED_ELEMENTS) & (
                linear_index < n_elements
            )
            value = tl.load(x_ptr + linear_index, mask=mask, other=0.0).to(tl.float32)
            inv_variance = tl.rsqrt(
                tl.sum(value * value, axis=0) / NORMALIZED_ELEMENTS + EPSILON
            )
            tl.store(inv_variance_ptr + row, inv_variance)
            result = value * inv_variance * scale + bias
            tl.store(
                y_ptr + linear_index,
                result.to(y_ptr.dtype.element_ty),
                mask=mask,
            )
        else:
            square_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            normalized_base = tl.zeros((), dtype=tl.int64)
            while normalized_base < NORMALIZED_ELEMENTS:
                normalized_index = lane + normalized_base
                linear_index = row * NORMALIZED_ELEMENTS + normalized_index
                mask = (normalized_index < NORMALIZED_ELEMENTS) & (
                    linear_index < n_elements
                )
                value = tl.load(x_ptr + linear_index, mask=mask, other=0.0).to(
                    tl.float32
                )
                square_sum += value * value
                normalized_base += BLOCK_SIZE
            inv_variance = tl.rsqrt(
                tl.sum(square_sum, axis=0) / NORMALIZED_ELEMENTS + EPSILON
            )
            tl.store(inv_variance_ptr + row, inv_variance)

            normalized_base = tl.zeros((), dtype=tl.int64)
            while normalized_base < NORMALIZED_ELEMENTS:
                normalized_index = lane + normalized_base
                linear_index = row * NORMALIZED_ELEMENTS + normalized_index
                mask = (normalized_index < NORMALIZED_ELEMENTS) & (
                    linear_index < n_elements
                )
                value = tl.load(x_ptr + linear_index, mask=mask, other=0.0).to(
                    tl.float32
                )
                scale = tl.load(scale_ptr + normalized_index, mask=mask, other=0.0).to(
                    tl.float32
                )
                bias = tl.load(bias_ptr + normalized_index, mask=mask, other=0.0).to(
                    tl.float32
                )
                result = value * inv_variance * scale + bias
                tl.store(
                    y_ptr + linear_index,
                    result.to(y_ptr.dtype.element_ty),
                    mask=mask,
                )
                normalized_base += BLOCK_SIZE
        row += 1


@triton.jit
def batchnorm_inference_nchw_persistent_kernel(
    x_ptr,
    mean_ptr,
    inv_variance_ptr,
    scale_ptr,
    bias_ptr,
    y_ptr,
    n_elements,
    RANK: tl.constexpr,
    CHANNELS: tl.constexpr,
    SPATIAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WORKER_COUNT: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    task_stride = tl.num_programs(0)
    if SPATIAL == 1:
        unit_block_batch: tl.constexpr = 32
        unit_block_channels: tl.constexpr = BLOCK_SIZE // unit_block_batch
        unit_channel_blocks: tl.constexpr = (
            CHANNELS + unit_block_channels - 1
        ) // unit_block_channels
        unit_batch_count = n_elements // CHANNELS
        unit_batch_blocks = tl.cdiv(unit_batch_count, unit_block_batch)
        unit_task_count = unit_batch_blocks * unit_channel_blocks
        unit_batch_lane = tl.arange(0, unit_block_batch)[:, None].to(tl.int64)
        unit_channel_lane = tl.arange(0, unit_block_channels)[None, :].to(tl.int64)
        unit_task = program_id
        while unit_task < unit_task_count:
            unit_channel_block = unit_task % unit_channel_blocks
            unit_batch_block = unit_task // unit_channel_blocks
            unit_batches = unit_batch_block * unit_block_batch + unit_batch_lane
            unit_channels = unit_channel_block * unit_block_channels + unit_channel_lane
            unit_channel_mask = unit_channels < CHANNELS
            unit_mask = (unit_batches < unit_batch_count) & unit_channel_mask
            unit_offset = unit_batches * CHANNELS + unit_channels
            unit_value = tl.load(x_ptr + unit_offset, mask=unit_mask, other=0.0).to(
                tl.float32
            )
            unit_mean = tl.load(
                mean_ptr + unit_channels,
                mask=unit_channel_mask,
                other=0.0,
            ).to(tl.float32)
            unit_inv_variance = tl.load(
                inv_variance_ptr + unit_channels,
                mask=unit_channel_mask,
                other=0.0,
            ).to(tl.float32)
            unit_scale = tl.load(
                scale_ptr + unit_channels,
                mask=unit_channel_mask,
                other=0.0,
            ).to(tl.float32)
            unit_bias = tl.load(
                bias_ptr + unit_channels,
                mask=unit_channel_mask,
                other=0.0,
            ).to(tl.float32)
            unit_result = (
                unit_value - unit_mean
            ) * unit_inv_variance * unit_scale + unit_bias
            tl.store(
                y_ptr + unit_offset,
                unit_result.to(y_ptr.dtype.element_ty),
                mask=unit_mask,
            )
            unit_task += task_stride
        return

    block_spatial: tl.constexpr = (
        BLOCK_SIZE if SPATIAL > BLOCK_SIZE else triton.next_power_of_2(SPATIAL)
    )
    block_channels: tl.constexpr = BLOCK_SIZE // block_spatial
    spatial_blocks: tl.constexpr = (SPATIAL + block_spatial - 1) // block_spatial
    channel_blocks: tl.constexpr = (CHANNELS + block_channels - 1) // block_channels
    batch_count = n_elements // (CHANNELS * SPATIAL)
    task_count = batch_count * channel_blocks * spatial_blocks
    channel_lane = tl.arange(0, block_channels)[:, None].to(tl.int64)
    spatial_lane = tl.arange(0, block_spatial)[None, :].to(tl.int64)
    task = program_id
    while task < task_count:
        spatial_block = task % spatial_blocks
        remaining = task // spatial_blocks
        channel_block = remaining % channel_blocks
        batch = remaining // channel_blocks
        channels = channel_block * block_channels + channel_lane
        spatial = spatial_block * block_spatial + spatial_lane
        channel_mask = channels < CHANNELS
        mask = channel_mask & (spatial < SPATIAL)
        offset = (batch * CHANNELS + channels) * SPATIAL + spatial
        value = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
        inv_variance = tl.load(
            inv_variance_ptr + channels, mask=channel_mask, other=0.0
        ).to(tl.float32)
        scale = tl.load(scale_ptr + channels, mask=channel_mask, other=0.0).to(
            tl.float32
        )
        bias = tl.load(bias_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
        result = (value - mean) * inv_variance * scale + bias
        tl.store(
            y_ptr + offset,
            result.to(y_ptr.dtype.element_ty),
            mask=mask,
        )
        task += task_stride


@triton.jit
def batchnorm_inference_strided_persistent_kernel(
    x_ptr,
    mean_ptr,
    inv_variance_ptr,
    scale_ptr,
    bias_ptr,
    y_ptr,
    n_elements,
    RANK: tl.constexpr,
    CHANNELS: tl.constexpr,
    SPATIAL: tl.constexpr,
    DIM_0: tl.constexpr,
    DIM_1: tl.constexpr,
    DIM_2: tl.constexpr,
    DIM_3: tl.constexpr,
    DIM_4: tl.constexpr,
    DIM_5: tl.constexpr,
    DIM_6: tl.constexpr,
    DIM_7: tl.constexpr,
    X_STRIDE_0: tl.constexpr,
    X_STRIDE_1: tl.constexpr,
    X_STRIDE_2: tl.constexpr,
    X_STRIDE_3: tl.constexpr,
    X_STRIDE_4: tl.constexpr,
    X_STRIDE_5: tl.constexpr,
    X_STRIDE_6: tl.constexpr,
    X_STRIDE_7: tl.constexpr,
    Y_STRIDE_0: tl.constexpr,
    Y_STRIDE_1: tl.constexpr,
    Y_STRIDE_2: tl.constexpr,
    Y_STRIDE_3: tl.constexpr,
    Y_STRIDE_4: tl.constexpr,
    Y_STRIDE_5: tl.constexpr,
    Y_STRIDE_6: tl.constexpr,
    Y_STRIDE_7: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WORKER_COUNT: tl.constexpr,
):
    if RANK == 5:
        rank5_block_spatial: tl.constexpr = BLOCK_SIZE // 128
        rank5_block_channels: tl.constexpr = 2
        rank5_spatial_blocks: tl.constexpr = (
            SPATIAL + rank5_block_spatial - 1
        ) // rank5_block_spatial
        rank5_channel_blocks: tl.constexpr = (
            CHANNELS + rank5_block_channels - 1
        ) // rank5_block_channels
        rank5_batch_count = n_elements // (CHANNELS * SPATIAL)
        rank5_task_count = (
            rank5_batch_count * rank5_channel_blocks * rank5_spatial_blocks
        )
        rank5_channel_lane = tl.arange(0, rank5_block_channels)[:, None].to(tl.int64)
        rank5_spatial_lane = tl.arange(0, rank5_block_spatial)[None, :].to(tl.int64)
        rank5_task = tl.program_id(0).to(tl.int64)
        rank5_task_stride = tl.num_programs(0)
        while rank5_task < rank5_task_count:
            rank5_spatial_block = rank5_task % rank5_spatial_blocks
            rank5_remaining_task = rank5_task // rank5_spatial_blocks
            rank5_channel_block = rank5_remaining_task % rank5_channel_blocks
            rank5_batch = rank5_remaining_task // rank5_channel_blocks
            rank5_channels = (
                rank5_channel_block * rank5_block_channels + rank5_channel_lane
            )
            rank5_spatial = (
                rank5_spatial_block * rank5_block_spatial + rank5_spatial_lane
            )
            rank5_channel_mask = rank5_channels < CHANNELS
            rank5_mask = rank5_channel_mask & (rank5_spatial < SPATIAL)

            rank5_remaining_spatial = rank5_spatial
            rank5_coordinate_7 = rank5_remaining_spatial % DIM_7
            rank5_remaining_spatial //= DIM_7
            rank5_coordinate_6 = rank5_remaining_spatial % DIM_6
            rank5_remaining_spatial //= DIM_6
            rank5_coordinate_5 = rank5_remaining_spatial
            rank5_x_offset = (
                rank5_batch * X_STRIDE_3
                + rank5_channels * X_STRIDE_4
                + rank5_coordinate_5 * X_STRIDE_5
                + rank5_coordinate_6 * X_STRIDE_6
                + rank5_coordinate_7 * X_STRIDE_7
            )
            rank5_y_offset = (
                rank5_batch * Y_STRIDE_3
                + rank5_channels * Y_STRIDE_4
                + rank5_coordinate_5 * Y_STRIDE_5
                + rank5_coordinate_6 * Y_STRIDE_6
                + rank5_coordinate_7 * Y_STRIDE_7
            )
            rank5_value = tl.load(
                x_ptr + rank5_x_offset, mask=rank5_mask, other=0.0
            ).to(tl.float32)
            rank5_mean = tl.load(
                mean_ptr + rank5_channels,
                mask=rank5_channel_mask,
                other=0.0,
            ).to(tl.float32)
            rank5_inv_variance = tl.load(
                inv_variance_ptr + rank5_channels,
                mask=rank5_channel_mask,
                other=0.0,
            ).to(tl.float32)
            rank5_scale = tl.load(
                scale_ptr + rank5_channels,
                mask=rank5_channel_mask,
                other=0.0,
            ).to(tl.float32)
            rank5_bias = tl.load(
                bias_ptr + rank5_channels,
                mask=rank5_channel_mask,
                other=0.0,
            ).to(tl.float32)
            rank5_result = (
                rank5_value - rank5_mean
            ) * rank5_inv_variance * rank5_scale + rank5_bias
            tl.store(
                y_ptr + rank5_y_offset,
                rank5_result.to(y_ptr.dtype.element_ty),
                mask=rank5_mask,
            )
            rank5_task += rank5_task_stride
        return

    strided_block: tl.constexpr = BLOCK_SIZE // 16
    block_inner: tl.constexpr = (
        strided_block if DIM_7 > strided_block else triton.next_power_of_2(DIM_7)
    )
    block_outer: tl.constexpr = strided_block // block_inner
    inner_blocks: tl.constexpr = (DIM_7 + block_inner - 1) // block_inner
    outer_elements = n_elements // DIM_7
    outer_blocks = tl.cdiv(outer_elements, block_outer)
    task_count = outer_blocks * inner_blocks
    outer_lane = tl.arange(0, block_outer)[:, None].to(tl.int64)
    inner_lane = tl.arange(0, block_inner)[None, :].to(tl.int64)
    task = tl.program_id(0).to(tl.int64)
    task_stride = tl.num_programs(0)
    while task < task_count:
        inner_block = task % inner_blocks
        outer_block = task // inner_blocks
        outer = outer_block * block_outer + outer_lane
        inner = inner_block * block_inner + inner_lane
        logical_index = outer * DIM_7 + inner
        mask = (outer < outer_elements) & (inner < DIM_7)
        remaining = outer
        x_offset = inner * X_STRIDE_7
        y_offset = inner * Y_STRIDE_7
        coordinate = remaining % DIM_6
        remaining //= DIM_6
        x_offset += coordinate * X_STRIDE_6
        y_offset += coordinate * Y_STRIDE_6
        coordinate = remaining % DIM_5
        remaining //= DIM_5
        x_offset += coordinate * X_STRIDE_5
        y_offset += coordinate * Y_STRIDE_5
        coordinate = remaining % DIM_4
        remaining //= DIM_4
        x_offset += coordinate * X_STRIDE_4
        y_offset += coordinate * Y_STRIDE_4
        coordinate = remaining % DIM_3
        remaining //= DIM_3
        x_offset += coordinate * X_STRIDE_3
        y_offset += coordinate * Y_STRIDE_3
        coordinate = remaining % DIM_2
        remaining //= DIM_2
        x_offset += coordinate * X_STRIDE_2
        y_offset += coordinate * Y_STRIDE_2
        coordinate = remaining % DIM_1
        remaining //= DIM_1
        x_offset += coordinate * X_STRIDE_1
        y_offset += coordinate * Y_STRIDE_1
        coordinate = remaining % DIM_0
        x_offset += coordinate * X_STRIDE_0
        y_offset += coordinate * Y_STRIDE_0

        if RANK > 2:
            channel = (outer // (SPATIAL // DIM_7)) % CHANNELS
        else:
            channel = logical_index % CHANNELS
        value = tl.load(x_ptr + x_offset, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        inv_variance = tl.load(inv_variance_ptr + channel, mask=mask, other=0.0).to(
            tl.float32
        )
        scale = tl.load(scale_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        result = (value - mean) * inv_variance * scale + bias
        tl.store(y_ptr + y_offset, result.to(y_ptr.dtype.element_ty), mask=mask)
        task += task_stride


# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0


@triton.jit
def _coordinates(group, reduced, dims: tl.constexpr, mask: tl.constexpr):
    reduced = reduced.to(tl.int64)
    # Keep block-shaped addresses even when the reduction contains one element
    # and all logical coordinates originate from the scalar group index.
    group = group.to(tl.int64) + tl.zeros(reduced.shape, tl.int64)
    if mask & 128:
        c7 = reduced % dims[7]
        reduced //= dims[7]
    else:
        c7 = group % dims[7]
        group //= dims[7]
    if mask & 64:
        c6 = reduced % dims[6]
        reduced //= dims[6]
    else:
        c6 = group % dims[6]
        group //= dims[6]
    if mask & 32:
        c5 = reduced % dims[5]
        reduced //= dims[5]
    else:
        c5 = group % dims[5]
        group //= dims[5]
    if mask & 16:
        c4 = reduced % dims[4]
        reduced //= dims[4]
    else:
        c4 = group % dims[4]
        group //= dims[4]
    if mask & 8:
        c3 = reduced % dims[3]
        reduced //= dims[3]
    else:
        c3 = group % dims[3]
        group //= dims[3]
    if mask & 4:
        c2 = reduced % dims[2]
        reduced //= dims[2]
    else:
        c2 = group % dims[2]
        group //= dims[2]
    if mask & 2:
        c1 = reduced % dims[1]
        reduced //= dims[1]
    else:
        c1 = group % dims[1]
        group //= dims[1]
    if mask & 1:
        c0 = reduced % dims[0]
        reduced //= dims[0]
    else:
        c0 = group % dims[0]
        group //= dims[0]
    return (c0, c1, c2, c3, c4, c5, c6, c7)


@triton.jit
def _offset(coords, strides: tl.constexpr):
    result = coords[0] * strides[0]
    for axis in tl.static_range(1, 8):
        result += coords[axis] * strides[axis]
    return result


@triton.jit
def extended_normalization_forward(
    x_ptr,
    scale_ptr,
    bias_ptr,
    y_ptr,
    mean_ptr,
    inv_ptr,
    GROUPS: tl.constexpr,
    REDUCTION: tl.constexpr,
    AXES: tl.constexpr,
    PARAM_AXES: tl.constexpr,
    PARAMETERS: tl.constexpr,
    AFFINE_REDUCTION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPSILON: tl.constexpr,
    DIM_0: tl.constexpr,
    DIM_1: tl.constexpr,
    DIM_2: tl.constexpr,
    DIM_3: tl.constexpr,
    DIM_4: tl.constexpr,
    DIM_5: tl.constexpr,
    DIM_6: tl.constexpr,
    DIM_7: tl.constexpr,
    X_STRIDE_0: tl.constexpr,
    X_STRIDE_1: tl.constexpr,
    X_STRIDE_2: tl.constexpr,
    X_STRIDE_3: tl.constexpr,
    X_STRIDE_4: tl.constexpr,
    X_STRIDE_5: tl.constexpr,
    X_STRIDE_6: tl.constexpr,
    X_STRIDE_7: tl.constexpr,
    SCALE_STRIDE_0: tl.constexpr,
    SCALE_STRIDE_1: tl.constexpr,
    SCALE_STRIDE_2: tl.constexpr,
    SCALE_STRIDE_3: tl.constexpr,
    SCALE_STRIDE_4: tl.constexpr,
    SCALE_STRIDE_5: tl.constexpr,
    SCALE_STRIDE_6: tl.constexpr,
    SCALE_STRIDE_7: tl.constexpr,
    BIAS_STRIDE_0: tl.constexpr,
    BIAS_STRIDE_1: tl.constexpr,
    BIAS_STRIDE_2: tl.constexpr,
    BIAS_STRIDE_3: tl.constexpr,
    BIAS_STRIDE_4: tl.constexpr,
    BIAS_STRIDE_5: tl.constexpr,
    BIAS_STRIDE_6: tl.constexpr,
    BIAS_STRIDE_7: tl.constexpr,
    Y_STRIDE_0: tl.constexpr,
    Y_STRIDE_1: tl.constexpr,
    Y_STRIDE_2: tl.constexpr,
    Y_STRIDE_3: tl.constexpr,
    Y_STRIDE_4: tl.constexpr,
    Y_STRIDE_5: tl.constexpr,
    Y_STRIDE_6: tl.constexpr,
    Y_STRIDE_7: tl.constexpr,
    MEAN_STRIDE_0: tl.constexpr,
    MEAN_STRIDE_1: tl.constexpr,
    MEAN_STRIDE_2: tl.constexpr,
    MEAN_STRIDE_3: tl.constexpr,
    MEAN_STRIDE_4: tl.constexpr,
    MEAN_STRIDE_5: tl.constexpr,
    MEAN_STRIDE_6: tl.constexpr,
    MEAN_STRIDE_7: tl.constexpr,
    INV_STRIDE_0: tl.constexpr,
    INV_STRIDE_1: tl.constexpr,
    INV_STRIDE_2: tl.constexpr,
    INV_STRIDE_3: tl.constexpr,
    INV_STRIDE_4: tl.constexpr,
    INV_STRIDE_5: tl.constexpr,
    INV_STRIDE_6: tl.constexpr,
    INV_STRIDE_7: tl.constexpr,
):
    dims: tl.constexpr = (
        DIM_0,
        DIM_1,
        DIM_2,
        DIM_3,
        DIM_4,
        DIM_5,
        DIM_6,
        DIM_7,
    )
    x_strides: tl.constexpr = (
        X_STRIDE_0,
        X_STRIDE_1,
        X_STRIDE_2,
        X_STRIDE_3,
        X_STRIDE_4,
        X_STRIDE_5,
        X_STRIDE_6,
        X_STRIDE_7,
    )
    scale_strides: tl.constexpr = (
        SCALE_STRIDE_0,
        SCALE_STRIDE_1,
        SCALE_STRIDE_2,
        SCALE_STRIDE_3,
        SCALE_STRIDE_4,
        SCALE_STRIDE_5,
        SCALE_STRIDE_6,
        SCALE_STRIDE_7,
    )
    bias_strides: tl.constexpr = (
        BIAS_STRIDE_0,
        BIAS_STRIDE_1,
        BIAS_STRIDE_2,
        BIAS_STRIDE_3,
        BIAS_STRIDE_4,
        BIAS_STRIDE_5,
        BIAS_STRIDE_6,
        BIAS_STRIDE_7,
    )
    y_strides: tl.constexpr = (
        Y_STRIDE_0,
        Y_STRIDE_1,
        Y_STRIDE_2,
        Y_STRIDE_3,
        Y_STRIDE_4,
        Y_STRIDE_5,
        Y_STRIDE_6,
        Y_STRIDE_7,
    )
    mean_strides: tl.constexpr = (
        MEAN_STRIDE_0,
        MEAN_STRIDE_1,
        MEAN_STRIDE_2,
        MEAN_STRIDE_3,
        MEAN_STRIDE_4,
        MEAN_STRIDE_5,
        MEAN_STRIDE_6,
        MEAN_STRIDE_7,
    )
    inv_strides: tl.constexpr = (
        INV_STRIDE_0,
        INV_STRIDE_1,
        INV_STRIDE_2,
        INV_STRIDE_3,
        INV_STRIDE_4,
        INV_STRIDE_5,
        INV_STRIDE_6,
        INV_STRIDE_7,
    )
    group = tl.program_id(0).to(tl.int64)
    lanes = tl.arange(0, BLOCK_SIZE)
    partial = tl.zeros((BLOCK_SIZE,), tl.float32)
    for start in range(0, REDUCTION, BLOCK_SIZE):
        reduced = start + lanes
        coords = _coordinates(group, reduced, dims, AXES)
        x = tl.load(
            x_ptr + _offset(coords, x_strides), reduced < REDUCTION, other=0
        ).to(tl.float32)
        partial += x
    mean = tl.sum(partial, 0) / REDUCTION
    variance = tl.zeros((BLOCK_SIZE,), tl.float32)
    for start in range(0, REDUCTION, BLOCK_SIZE):
        reduced = start + lanes
        coords = _coordinates(group, reduced, dims, AXES)
        x = tl.load(
            x_ptr + _offset(coords, x_strides), reduced < REDUCTION, other=0
        ).to(tl.float32)
        centered = x - mean
        variance += tl.where(reduced < REDUCTION, centered * centered, 0.0)
    inverse = tl.rsqrt(tl.sum(variance, 0) / REDUCTION + EPSILON)
    statistic_coords = _coordinates(group, tl.full((), 0, tl.int64), dims, AXES)
    tl.store(mean_ptr + _offset(statistic_coords, mean_strides), mean)
    tl.store(inv_ptr + _offset(statistic_coords, inv_strides), inverse)
    for start in range(0, REDUCTION, BLOCK_SIZE):
        reduced = start + lanes
        coords = _coordinates(group, reduced, dims, AXES)
        valid = reduced < REDUCTION
        x = tl.load(x_ptr + _offset(coords, x_strides), valid, other=0).to(tl.float32)
        scale = tl.load(scale_ptr + _offset(coords, scale_strides), valid, other=0).to(
            tl.float32
        )
        bias = tl.load(bias_ptr + _offset(coords, bias_strides), valid, other=0).to(
            tl.float32
        )
        y = (x - mean) * inverse * scale + bias
        tl.store(y_ptr + _offset(coords, y_strides), y, valid)


@triton.jit
def extended_normalization_backward(
    dy_ptr,
    x_ptr,
    scale_ptr,
    mean_ptr,
    inv_ptr,
    dx_ptr,
    dscale_ptr,
    dbias_ptr,
    GROUPS: tl.constexpr,
    REDUCTION: tl.constexpr,
    AXES: tl.constexpr,
    PARAM_AXES: tl.constexpr,
    PARAMETERS: tl.constexpr,
    AFFINE_REDUCTION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RMS: tl.constexpr,
    DIM_0: tl.constexpr,
    DIM_1: tl.constexpr,
    DIM_2: tl.constexpr,
    DIM_3: tl.constexpr,
    DIM_4: tl.constexpr,
    DIM_5: tl.constexpr,
    DIM_6: tl.constexpr,
    DIM_7: tl.constexpr,
    DY_STRIDE_0: tl.constexpr,
    DY_STRIDE_1: tl.constexpr,
    DY_STRIDE_2: tl.constexpr,
    DY_STRIDE_3: tl.constexpr,
    DY_STRIDE_4: tl.constexpr,
    DY_STRIDE_5: tl.constexpr,
    DY_STRIDE_6: tl.constexpr,
    DY_STRIDE_7: tl.constexpr,
    X_STRIDE_0: tl.constexpr,
    X_STRIDE_1: tl.constexpr,
    X_STRIDE_2: tl.constexpr,
    X_STRIDE_3: tl.constexpr,
    X_STRIDE_4: tl.constexpr,
    X_STRIDE_5: tl.constexpr,
    X_STRIDE_6: tl.constexpr,
    X_STRIDE_7: tl.constexpr,
    SCALE_STRIDE_0: tl.constexpr,
    SCALE_STRIDE_1: tl.constexpr,
    SCALE_STRIDE_2: tl.constexpr,
    SCALE_STRIDE_3: tl.constexpr,
    SCALE_STRIDE_4: tl.constexpr,
    SCALE_STRIDE_5: tl.constexpr,
    SCALE_STRIDE_6: tl.constexpr,
    SCALE_STRIDE_7: tl.constexpr,
    MEAN_STRIDE_0: tl.constexpr,
    MEAN_STRIDE_1: tl.constexpr,
    MEAN_STRIDE_2: tl.constexpr,
    MEAN_STRIDE_3: tl.constexpr,
    MEAN_STRIDE_4: tl.constexpr,
    MEAN_STRIDE_5: tl.constexpr,
    MEAN_STRIDE_6: tl.constexpr,
    MEAN_STRIDE_7: tl.constexpr,
    INV_STRIDE_0: tl.constexpr,
    INV_STRIDE_1: tl.constexpr,
    INV_STRIDE_2: tl.constexpr,
    INV_STRIDE_3: tl.constexpr,
    INV_STRIDE_4: tl.constexpr,
    INV_STRIDE_5: tl.constexpr,
    INV_STRIDE_6: tl.constexpr,
    INV_STRIDE_7: tl.constexpr,
    DX_STRIDE_0: tl.constexpr,
    DX_STRIDE_1: tl.constexpr,
    DX_STRIDE_2: tl.constexpr,
    DX_STRIDE_3: tl.constexpr,
    DX_STRIDE_4: tl.constexpr,
    DX_STRIDE_5: tl.constexpr,
    DX_STRIDE_6: tl.constexpr,
    DX_STRIDE_7: tl.constexpr,
    DSCALE_STRIDE_0: tl.constexpr,
    DSCALE_STRIDE_1: tl.constexpr,
    DSCALE_STRIDE_2: tl.constexpr,
    DSCALE_STRIDE_3: tl.constexpr,
    DSCALE_STRIDE_4: tl.constexpr,
    DSCALE_STRIDE_5: tl.constexpr,
    DSCALE_STRIDE_6: tl.constexpr,
    DSCALE_STRIDE_7: tl.constexpr,
    DBIAS_STRIDE_0: tl.constexpr,
    DBIAS_STRIDE_1: tl.constexpr,
    DBIAS_STRIDE_2: tl.constexpr,
    DBIAS_STRIDE_3: tl.constexpr,
    DBIAS_STRIDE_4: tl.constexpr,
    DBIAS_STRIDE_5: tl.constexpr,
    DBIAS_STRIDE_6: tl.constexpr,
    DBIAS_STRIDE_7: tl.constexpr,
):
    dims: tl.constexpr = (
        DIM_0,
        DIM_1,
        DIM_2,
        DIM_3,
        DIM_4,
        DIM_5,
        DIM_6,
        DIM_7,
    )
    dy_strides: tl.constexpr = (
        DY_STRIDE_0,
        DY_STRIDE_1,
        DY_STRIDE_2,
        DY_STRIDE_3,
        DY_STRIDE_4,
        DY_STRIDE_5,
        DY_STRIDE_6,
        DY_STRIDE_7,
    )
    x_strides: tl.constexpr = (
        X_STRIDE_0,
        X_STRIDE_1,
        X_STRIDE_2,
        X_STRIDE_3,
        X_STRIDE_4,
        X_STRIDE_5,
        X_STRIDE_6,
        X_STRIDE_7,
    )
    scale_strides: tl.constexpr = (
        SCALE_STRIDE_0,
        SCALE_STRIDE_1,
        SCALE_STRIDE_2,
        SCALE_STRIDE_3,
        SCALE_STRIDE_4,
        SCALE_STRIDE_5,
        SCALE_STRIDE_6,
        SCALE_STRIDE_7,
    )
    mean_strides: tl.constexpr = (
        MEAN_STRIDE_0,
        MEAN_STRIDE_1,
        MEAN_STRIDE_2,
        MEAN_STRIDE_3,
        MEAN_STRIDE_4,
        MEAN_STRIDE_5,
        MEAN_STRIDE_6,
        MEAN_STRIDE_7,
    )
    inv_strides: tl.constexpr = (
        INV_STRIDE_0,
        INV_STRIDE_1,
        INV_STRIDE_2,
        INV_STRIDE_3,
        INV_STRIDE_4,
        INV_STRIDE_5,
        INV_STRIDE_6,
        INV_STRIDE_7,
    )
    dx_strides: tl.constexpr = (
        DX_STRIDE_0,
        DX_STRIDE_1,
        DX_STRIDE_2,
        DX_STRIDE_3,
        DX_STRIDE_4,
        DX_STRIDE_5,
        DX_STRIDE_6,
        DX_STRIDE_7,
    )
    dscale_strides: tl.constexpr = (
        DSCALE_STRIDE_0,
        DSCALE_STRIDE_1,
        DSCALE_STRIDE_2,
        DSCALE_STRIDE_3,
        DSCALE_STRIDE_4,
        DSCALE_STRIDE_5,
        DSCALE_STRIDE_6,
        DSCALE_STRIDE_7,
    )
    dbias_strides: tl.constexpr = (
        DBIAS_STRIDE_0,
        DBIAS_STRIDE_1,
        DBIAS_STRIDE_2,
        DBIAS_STRIDE_3,
        DBIAS_STRIDE_4,
        DBIAS_STRIDE_5,
        DBIAS_STRIDE_6,
        DBIAS_STRIDE_7,
    )
    program = tl.program_id(0).to(tl.int64)
    lanes = tl.arange(0, BLOCK_SIZE)
    if program < GROUPS:
        statistic_coords = _coordinates(program, tl.full((), 0, tl.int64), dims, AXES)
        if RMS:
            mean = 0.0
        else:
            mean = tl.load(mean_ptr + _offset(statistic_coords, mean_strides)).to(
                tl.float32
            )
        inverse = tl.load(inv_ptr + _offset(statistic_coords, inv_strides)).to(
            tl.float32
        )
        sum_gradient = tl.zeros((BLOCK_SIZE,), tl.float32)
        sum_gradient_normalized = tl.zeros((BLOCK_SIZE,), tl.float32)
        for start in range(0, REDUCTION, BLOCK_SIZE):
            reduced = start + lanes
            coords = _coordinates(program, reduced, dims, AXES)
            valid = reduced < REDUCTION
            x = tl.load(x_ptr + _offset(coords, x_strides), valid, other=0).to(
                tl.float32
            )
            dy = tl.load(dy_ptr + _offset(coords, dy_strides), valid, other=0).to(
                tl.float32
            )
            scale = tl.load(
                scale_ptr + _offset(coords, scale_strides), valid, other=0
            ).to(tl.float32)
            gradient = dy * scale
            normalized = (x - mean) * inverse
            sum_gradient += gradient
            sum_gradient_normalized += gradient * normalized
        gradient_mean = tl.sum(sum_gradient, 0) / REDUCTION
        gradient_normalized_mean = tl.sum(sum_gradient_normalized, 0) / REDUCTION
        for start in range(0, REDUCTION, BLOCK_SIZE):
            reduced = start + lanes
            coords = _coordinates(program, reduced, dims, AXES)
            valid = reduced < REDUCTION
            x = tl.load(x_ptr + _offset(coords, x_strides), valid, other=0).to(
                tl.float32
            )
            dy = tl.load(dy_ptr + _offset(coords, dy_strides), valid, other=0).to(
                tl.float32
            )
            scale = tl.load(
                scale_ptr + _offset(coords, scale_strides), valid, other=0
            ).to(tl.float32)
            gradient = dy * scale - (x - mean) * inverse * gradient_normalized_mean
            if not RMS:
                gradient -= gradient_mean
            tl.store(dx_ptr + _offset(coords, dx_strides), gradient * inverse, valid)
    else:
        parameter = program - GROUPS
        sum_scale = tl.zeros((BLOCK_SIZE,), tl.float32)
        sum_bias = tl.zeros((BLOCK_SIZE,), tl.float32)
        for start in range(0, AFFINE_REDUCTION, BLOCK_SIZE):
            reduced = start + lanes
            coords = _coordinates(parameter, reduced, dims, PARAM_AXES)
            valid = reduced < AFFINE_REDUCTION
            x = tl.load(x_ptr + _offset(coords, x_strides), valid, other=0).to(
                tl.float32
            )
            dy = tl.load(dy_ptr + _offset(coords, dy_strides), valid, other=0).to(
                tl.float32
            )
            if RMS:
                mean = 0.0
            else:
                mean = tl.load(
                    mean_ptr + _offset(coords, mean_strides), valid, other=0
                ).to(tl.float32)
            inverse = tl.load(
                inv_ptr + _offset(coords, inv_strides), valid, other=0
            ).to(tl.float32)
            sum_scale += dy * (x - mean) * inverse
            sum_bias += dy
        coords = _coordinates(parameter, tl.full((), 0, tl.int64), dims, PARAM_AXES)
        tl.store(dscale_ptr + _offset(coords, dscale_strides), tl.sum(sum_scale, 0))
        tl.store(dbias_ptr + _offset(coords, dbias_strides), tl.sum(sum_bias, 0))


@triton.jit
def compact_normalization_forward(
    x_ptr,
    scale_ptr,
    bias_ptr,
    y_ptr,
    mean_ptr,
    inv_ptr,
    REDUCTION: tl.constexpr,
    CHANNELS: tl.constexpr,
    ROWS_PER_BATCH: tl.constexpr,
    AFFINE_MODE: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK_SIZE)
    valid = columns < REDUCTION
    x = tl.load(x_ptr + row * REDUCTION + columns, valid, other=0).to(tl.float32)
    mean = tl.sum(x, 0) / REDUCTION
    centered = tl.where(valid, x - mean, 0.0)
    inverse = tl.rsqrt(tl.sum(centered * centered, 0) / REDUCTION + EPSILON)
    if AFFINE_MODE == 0:
        scale = tl.load(scale_ptr + row % CHANNELS).to(tl.float32)
        bias = tl.load(bias_ptr + row % CHANNELS).to(tl.float32)
    else:
        affine = columns
        if AFFINE_MODE == 2:
            affine += row // ROWS_PER_BATCH * REDUCTION
        scale = tl.load(scale_ptr + affine, valid, other=0).to(tl.float32)
        bias = tl.load(bias_ptr + affine, valid, other=0).to(tl.float32)
    tl.store(
        y_ptr + row * REDUCTION + columns,
        centered * inverse * scale + bias,
        valid,
    )
    tl.store(mean_ptr + row, mean)
    tl.store(inv_ptr + row, inverse)


@triton.jit
def compact_batchnorm_backward(
    dy_ptr,
    x_ptr,
    scale_ptr,
    mean_ptr,
    inv_ptr,
    dx_ptr,
    dscale_ptr,
    dbias_ptr,
    CHANNELS: tl.constexpr,
    SPATIAL: tl.constexpr,
    REDUCTION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    channel = tl.program_id(0).to(tl.int64)
    reduced = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    valid = reduced < REDUCTION
    offset = (
        reduced // SPATIAL * (CHANNELS * SPATIAL)
        + channel * SPATIAL
        + reduced % SPATIAL
    )
    mean = tl.load(mean_ptr + channel)
    inverse = tl.load(inv_ptr + channel)
    scale = tl.load(scale_ptr + channel).to(tl.float32)
    x = tl.load(x_ptr + offset, valid, other=0).to(tl.float32)
    dy = tl.load(dy_ptr + offset, valid, other=0).to(tl.float32)
    normalized = (x - mean) * inverse
    sum_dy = tl.sum(dy, 0)
    sum_normalized = tl.sum(dy * normalized, 0)
    dx = (dy - sum_dy / REDUCTION - normalized * (sum_normalized / REDUCTION)) * (
        scale * inverse
    )
    tl.store(dx_ptr + offset, dx, valid)
    tl.store(dscale_ptr + channel, sum_normalized)
    tl.store(dbias_ptr + channel, sum_dy)
