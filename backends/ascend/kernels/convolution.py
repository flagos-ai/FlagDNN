# Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0

"""Ascend-owned strided 1D/2D/3D convolution FProp kernel."""

import triton
import triton.language as tl


@triton.jit
def _convolution_fprop_im2col_stride2_width(
    input_ptr,
    columns_ptr,
    batch,
    filter_h,
    output_h,
    channel_start,
    output_w_start,
    safe_input_h,
    valid_input_h,
    CHANNELS_PER_GROUP: tl.constexpr,
    INPUT_DIM_4: tl.constexpr,
    INPUT_STRIDE_0: tl.constexpr,
    INPUT_STRIDE_1: tl.constexpr,
    INPUT_STRIDE_3: tl.constexpr,
    INPUT_STRIDE_4: tl.constexpr,
    FILTER_DIM_3: tl.constexpr,
    FILTER_DIM_4: tl.constexpr,
    OUTPUT_DIM_3: tl.constexpr,
    OUTPUT_DIM_4: tl.constexpr,
    PRE_PADDING_2: tl.constexpr,
    DILATION_2: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
    BLOCK_OUTPUT_WIDTH: tl.constexpr,
):
    block_channels: tl.constexpr = BLOCK_CHANNELS
    block_output_width: tl.constexpr = BLOCK_OUTPUT_WIDTH
    use_deinterleave: tl.constexpr = (
        (FILTER_DIM_4 == 3 or FILTER_DIM_4 == 5) and DILATION_2 == 1
    )
    valid_output_width: tl.constexpr = (
        OUTPUT_DIM_4 if OUTPUT_DIM_4 < block_output_width
        else block_output_width
    )
    input_window_span: tl.constexpr = (
        (valid_output_width - 1) * 2
        + (FILTER_DIM_4 - 1) * DILATION_2
        + 1
    )
    block_input_width: tl.constexpr = (
        block_output_width * 2
        if use_deinterleave
        else (64 if input_window_span <= 64 else 128)
    )
    filter_area: tl.constexpr = FILTER_DIM_3 * FILTER_DIM_4
    output_spatial: tl.constexpr = OUTPUT_DIM_3 * OUTPUT_DIM_4
    input_window_origin = output_w_start * 2 - PRE_PADDING_2
    safe_window_origin = tl.maximum(input_window_origin, 0)
    channel_offset = channel_start.to(tl.int32)
    input_window_offset = input_window_origin.to(tl.int32)
    safe_window_offset = safe_window_origin.to(tl.int32)
    output_w_offset = output_w_start.to(tl.int32)
    if use_deinterleave:
        left_window = tl.make_block_ptr(
            base=(
                input_ptr
                + batch * INPUT_STRIDE_0
                + safe_input_h * INPUT_STRIDE_3
            ),
            shape=(CHANNELS_PER_GROUP, INPUT_DIM_4),
            strides=(INPUT_STRIDE_1, INPUT_STRIDE_4),
            offsets=(channel_offset, input_window_offset),
            block_shape=(block_channels, block_input_width),
            order=(1, 0),
        )
        left_values = tl.load(
            left_window,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        left_pairs = tl.reshape(
            left_values, (block_channels, block_output_width, 2)
        )
        filter_w0_values, filter_w1_values = tl.split(left_pairs)
        right_window = tl.make_block_ptr(
            base=(
                input_ptr
                + batch * INPUT_STRIDE_0
                + safe_input_h * INPUT_STRIDE_3
            ),
            shape=(CHANNELS_PER_GROUP, INPUT_DIM_4),
            strides=(INPUT_STRIDE_1, INPUT_STRIDE_4),
            offsets=(channel_offset, input_window_offset + 2),
            block_shape=(block_channels, block_input_width),
            order=(1, 0),
        )
        right_values = tl.load(
            right_window,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        right_pairs = tl.reshape(
            right_values, (block_channels, block_output_width, 2)
        )
        filter_w2_values, filter_w3_values = tl.split(right_pairs)
        if FILTER_DIM_4 == 5:
            far_window = tl.make_block_ptr(
                base=(
                    input_ptr
                    + batch * INPUT_STRIDE_0
                    + safe_input_h * INPUT_STRIDE_3
                ),
                shape=(CHANNELS_PER_GROUP, INPUT_DIM_4),
                strides=(INPUT_STRIDE_1, INPUT_STRIDE_4),
                offsets=(channel_offset, input_window_offset + 4),
                block_shape=(block_channels, block_input_width),
                order=(1, 0),
            )
            far_values = tl.load(
                far_window,
                boundary_check=(0, 1),
                padding_option="zero",
            )
            far_pairs = tl.reshape(
                far_values, (block_channels, block_output_width, 2)
            )
            filter_w4_values, _ = tl.split(far_pairs)
    else:
        input_window = tl.make_block_ptr(
            base=(
                input_ptr
                + batch * INPUT_STRIDE_0
                + safe_input_h * INPUT_STRIDE_3
            ),
            shape=(CHANNELS_PER_GROUP, INPUT_DIM_4),
            strides=(INPUT_STRIDE_1, INPUT_STRIDE_4),
            offsets=(
                channel_offset,
                safe_window_offset,
            ),
            block_shape=(block_channels, block_input_width),
            order=(1, 0),
        )
        window_values = tl.load(
            input_window,
            boundary_check=(0, 1),
            padding_option="zero",
        )
    width_lane = tl.arange(0, block_output_width)
    output_w = output_w_start + width_lane
    output_lane_valid = output_w < OUTPUT_DIM_4
    for filter_w in tl.static_range(0, FILTER_DIM_4):
        input_w = (
            output_w * 2 - PRE_PADDING_2 + filter_w * DILATION_2
        )
        input_lane_valid = (
            output_lane_valid & (input_w >= 0) & (input_w < INPUT_DIM_4)
        )
        if use_deinterleave:
            if filter_w == 0:
                values = filter_w0_values
            elif filter_w == 1:
                values = filter_w1_values
            elif filter_w == 2:
                values = filter_w2_values
            elif filter_w == 3:
                values = filter_w3_values
            else:
                values = filter_w4_values
        else:
            gather_lane = tl.where(
                input_lane_valid,
                input_w - safe_window_origin,
                0,
            )
            gather_index = tl.broadcast_to(
                gather_lane[None, :],
                (block_channels, block_output_width),
            )
            values = tl.gather(window_values, gather_index, axis=1)
        values = tl.where(
            valid_input_h & input_lane_valid[None, :], values, 0.0
        )
        columns_block = tl.make_block_ptr(
            base=(
                columns_ptr
                + batch * output_spatial * CHANNELS_PER_GROUP * filter_area
                + (filter_h * FILTER_DIM_4 + filter_w) * output_spatial
                + output_h * OUTPUT_DIM_4
            ),
            shape=(CHANNELS_PER_GROUP, OUTPUT_DIM_4),
            strides=(filter_area * output_spatial, 1),
            offsets=(channel_offset, output_w_offset),
            block_shape=(block_channels, block_output_width),
            order=(1, 0),
        )
        tl.store(
            columns_block,
            values,
            boundary_check=(0, 1),
        )


@triton.jit
def _convolution_fprop_im2col_rgb_stride2_rows(
    input_ptr,
    columns_ptr,
    batch,
    filter_h,
    output_h_start,
    output_w_start,
    input_h_start,
    CHANNELS_PER_GROUP: tl.constexpr,
    INPUT_DIM_4: tl.constexpr,
    INPUT_STRIDE_0: tl.constexpr,
    INPUT_STRIDE_1: tl.constexpr,
    INPUT_STRIDE_3: tl.constexpr,
    INPUT_STRIDE_4: tl.constexpr,
    FILTER_DIM_3: tl.constexpr,
    FILTER_DIM_4: tl.constexpr,
    OUTPUT_DIM_3: tl.constexpr,
    OUTPUT_DIM_4: tl.constexpr,
    PRE_PADDING_2: tl.constexpr,
):
    block_channels: tl.constexpr = 4
    block_rows: tl.constexpr = 4
    block_output_width: tl.constexpr = 64
    block_input_width: tl.constexpr = block_output_width * 2
    filter_area: tl.constexpr = FILTER_DIM_3 * FILTER_DIM_4
    output_spatial: tl.constexpr = OUTPUT_DIM_3 * OUTPUT_DIM_4
    input_window_offset = (
        output_w_start * 2 - PRE_PADDING_2
    ).to(tl.int32)
    output_h_offset = output_h_start.to(tl.int32)
    output_w_offset = output_w_start.to(tl.int32)
    input_base = (
        input_ptr
        + batch * INPUT_STRIDE_0
        + input_h_start * INPUT_STRIDE_3
    )
    left_window = tl.make_block_ptr(
        base=input_base,
        shape=(CHANNELS_PER_GROUP, block_rows, INPUT_DIM_4),
        strides=(
            INPUT_STRIDE_1,
            2 * INPUT_STRIDE_3,
            INPUT_STRIDE_4,
        ),
        offsets=(0, 0, input_window_offset),
        block_shape=(block_channels, block_rows, block_input_width),
        order=(2, 1, 0),
    )
    left_values = tl.load(
        left_window,
        boundary_check=(0, 2),
        padding_option="zero",
    )
    left_pairs = tl.reshape(
        left_values,
        (block_channels, block_rows, block_output_width, 2),
    )
    filter_w0_values, filter_w1_values = tl.split(left_pairs)
    right_window = tl.make_block_ptr(
        base=input_base,
        shape=(CHANNELS_PER_GROUP, block_rows, INPUT_DIM_4),
        strides=(
            INPUT_STRIDE_1,
            2 * INPUT_STRIDE_3,
            INPUT_STRIDE_4,
        ),
        offsets=(0, 0, input_window_offset + 2),
        block_shape=(block_channels, block_rows, block_input_width),
        order=(2, 1, 0),
    )
    right_values = tl.load(
        right_window,
        boundary_check=(0, 2),
        padding_option="zero",
    )
    right_pairs = tl.reshape(
        right_values,
        (block_channels, block_rows, block_output_width, 2),
    )
    filter_w2_values, _ = tl.split(right_pairs)

    for filter_w in tl.static_range(0, FILTER_DIM_4):
        if filter_w == 0:
            values = filter_w0_values
        elif filter_w == 1:
            values = filter_w1_values
        else:
            values = filter_w2_values
        columns_block = tl.make_block_ptr(
            base=(
                columns_ptr
                + batch
                * output_spatial
                * CHANNELS_PER_GROUP
                * filter_area
                + (filter_h * FILTER_DIM_4 + filter_w) * output_spatial
            ),
            shape=(CHANNELS_PER_GROUP, OUTPUT_DIM_3, OUTPUT_DIM_4),
            strides=(filter_area * output_spatial, OUTPUT_DIM_4, 1),
            offsets=(0, output_h_offset, output_w_offset),
            block_shape=(block_channels, block_rows, block_output_width),
            order=(2, 1, 0),
        )
        tl.store(
            columns_block,
            values,
            boundary_check=(0, 1, 2),
        )


@triton.jit
def _convolution_fprop_im2col_1d(
    input_ptr,
    columns_ptr,
    CHANNELS_PER_GROUP: tl.constexpr,
    INPUT_DIM_0: tl.constexpr,
    INPUT_DIM_4: tl.constexpr,
    INPUT_STRIDE_0: tl.constexpr,
    INPUT_STRIDE_1: tl.constexpr,
    INPUT_STRIDE_4: tl.constexpr,
    FILTER_DIM_4: tl.constexpr,
    OUTPUT_DIM_4: tl.constexpr,
    PRE_PADDING_2: tl.constexpr,
    CONV_STRIDE_2: tl.constexpr,
    DILATION_2: tl.constexpr,
):
    tile_channels: tl.constexpr = 32
    tile_length: tl.constexpr = 128
    reduction_extent: tl.constexpr = (
        CHANNELS_PER_GROUP * FILTER_DIM_4
    )
    task_count: tl.constexpr = INPUT_DIM_0 * FILTER_DIM_4
    channel_lane = tl.arange(0, tile_channels)
    length_lane = tl.arange(0, tile_length)
    task_stride = tl.num_programs(0).to(tl.int64)
    task = tl.program_id(0).to(tl.int64)
    while task < task_count:
        batch = task // FILTER_DIM_4
        filter_w = task - batch * FILTER_DIM_4
        channel_start = tl.zeros((), dtype=tl.int64)
        while channel_start < CHANNELS_PER_GROUP:
            input_channel = channel_start + channel_lane
            channel_mask = input_channel < CHANNELS_PER_GROUP
            channel_offset = channel_start.to(tl.int32)
            length_start = tl.zeros((), dtype=tl.int64)
            while length_start < OUTPUT_DIM_4:
                if CONV_STRIDE_2 == 1:
                    input_block = tl.make_block_ptr(
                        base=input_ptr + batch * INPUT_STRIDE_0,
                        shape=(CHANNELS_PER_GROUP, INPUT_DIM_4),
                        strides=(INPUT_STRIDE_1, INPUT_STRIDE_4),
                        offsets=(
                            channel_offset,
                            (
                                length_start
                                - PRE_PADDING_2
                                + filter_w * DILATION_2
                            ).to(tl.int32),
                        ),
                        block_shape=(tile_channels, tile_length),
                        order=(1, 0),
                    )
                    input_values = tl.load(
                        input_block,
                        boundary_check=(0, 1),
                        padding_option="zero",
                    )
                elif CONV_STRIDE_2 == 2:
                    input_window = tl.make_block_ptr(
                        base=input_ptr + batch * INPUT_STRIDE_0,
                        shape=(CHANNELS_PER_GROUP, INPUT_DIM_4),
                        strides=(INPUT_STRIDE_1, INPUT_STRIDE_4),
                        offsets=(
                            channel_offset,
                            (
                                length_start * 2
                                - PRE_PADDING_2
                                + filter_w * DILATION_2
                            ).to(tl.int32),
                        ),
                        block_shape=(tile_channels, tile_length * 2),
                        order=(1, 0),
                    )
                    input_window_values = tl.load(
                        input_window,
                        boundary_check=(0, 1),
                        padding_option="zero",
                    )
                    gather_index = tl.broadcast_to(
                        (length_lane * 2)[None, :],
                        (tile_channels, tile_length),
                    )
                    input_values = tl.gather(
                        input_window_values, gather_index, axis=1
                    )
                else:
                    output_l = length_start + length_lane
                    length_mask = output_l < OUTPUT_DIM_4
                    input_l = (
                        output_l * CONV_STRIDE_2
                        - PRE_PADDING_2
                        + filter_w * DILATION_2
                    )
                    valid_input_l = (
                        (input_l >= 0) & (input_l < INPUT_DIM_4)
                    )
                    safe_input_l = tl.where(valid_input_l, input_l, 0)
                    input_values = tl.load(
                        input_ptr
                        + batch * INPUT_STRIDE_0
                        + input_channel[:, None] * INPUT_STRIDE_1
                        + safe_input_l[None, :] * INPUT_STRIDE_4,
                        mask=(
                            channel_mask[:, None]
                            & length_mask[None, :]
                            & valid_input_l[None, :]
                        ),
                        other=0.0,
                    )
                columns_block = tl.make_block_ptr(
                    base=(
                        columns_ptr
                        + batch * OUTPUT_DIM_4 * reduction_extent
                        + filter_w * OUTPUT_DIM_4
                    ),
                    shape=(CHANNELS_PER_GROUP, OUTPUT_DIM_4),
                    strides=(FILTER_DIM_4 * OUTPUT_DIM_4, 1),
                    offsets=(
                        channel_offset,
                        length_start.to(tl.int32),
                    ),
                    block_shape=(tile_channels, tile_length),
                    order=(1, 0),
                )
                tl.store(
                    columns_block,
                    input_values,
                    boundary_check=(0, 1),
                )
                length_start += tile_length
            channel_start += tile_channels
        task += task_stride


@triton.jit
def convolution_fprop_im2col_kernel(
    input_ptr,
    filter_ptr,
    columns_ptr,
    n_elements,
    SPATIAL_RANK: tl.constexpr,
    GROUPS: tl.constexpr,
    INPUT_CHANNELS: tl.constexpr,
    OUTPUT_CHANNELS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    INPUT_DIM_0: tl.constexpr,
    INPUT_DIM_1: tl.constexpr,
    INPUT_DIM_2: tl.constexpr,
    INPUT_DIM_3: tl.constexpr,
    INPUT_DIM_4: tl.constexpr,
    INPUT_STRIDE_0: tl.constexpr,
    INPUT_STRIDE_1: tl.constexpr,
    INPUT_STRIDE_2: tl.constexpr,
    INPUT_STRIDE_3: tl.constexpr,
    INPUT_STRIDE_4: tl.constexpr,
    FILTER_DIM_0: tl.constexpr,
    FILTER_DIM_1: tl.constexpr,
    FILTER_DIM_2: tl.constexpr,
    FILTER_DIM_3: tl.constexpr,
    FILTER_DIM_4: tl.constexpr,
    FILTER_STRIDE_0: tl.constexpr,
    FILTER_STRIDE_1: tl.constexpr,
    FILTER_STRIDE_2: tl.constexpr,
    FILTER_STRIDE_3: tl.constexpr,
    FILTER_STRIDE_4: tl.constexpr,
    OUTPUT_DIM_0: tl.constexpr,
    OUTPUT_DIM_1: tl.constexpr,
    OUTPUT_DIM_2: tl.constexpr,
    OUTPUT_DIM_3: tl.constexpr,
    OUTPUT_DIM_4: tl.constexpr,
    OUTPUT_STRIDE_0: tl.constexpr,
    OUTPUT_STRIDE_1: tl.constexpr,
    OUTPUT_STRIDE_2: tl.constexpr,
    OUTPUT_STRIDE_3: tl.constexpr,
    OUTPUT_STRIDE_4: tl.constexpr,
    PRE_PADDING_0: tl.constexpr,
    PRE_PADDING_1: tl.constexpr,
    PRE_PADDING_2: tl.constexpr,
    POST_PADDING_0: tl.constexpr,
    POST_PADDING_1: tl.constexpr,
    POST_PADDING_2: tl.constexpr,
    CONV_STRIDE_0: tl.constexpr,
    CONV_STRIDE_1: tl.constexpr,
    CONV_STRIDE_2: tl.constexpr,
    DILATION_0: tl.constexpr,
    DILATION_1: tl.constexpr,
    DILATION_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WORKER_COUNT: tl.constexpr,
):
    if SPATIAL_RANK == 1:
        _convolution_fprop_im2col_1d(
            input_ptr,
            columns_ptr,
            CHANNELS_PER_GROUP,
            INPUT_DIM_0,
            INPUT_DIM_4,
            INPUT_STRIDE_0,
            INPUT_STRIDE_1,
            INPUT_STRIDE_4,
            FILTER_DIM_4,
            OUTPUT_DIM_4,
            PRE_PADDING_2,
            CONV_STRIDE_2,
            DILATION_2,
        )
        return

    task_stride = tl.num_programs(0).to(tl.int64)
    compact_3d: tl.constexpr = (
        SPATIAL_RANK == 3
        and GROUPS == 1
        and INPUT_DIM_3 == 16
        and INPUT_DIM_4 == 16
        and OUTPUT_DIM_3 == 16
        and OUTPUT_DIM_4 == 16
        and CONV_STRIDE_0 == 1
        and CONV_STRIDE_1 == 1
        and CONV_STRIDE_2 == 1
    )
    if compact_3d:
        d3c_rows: tl.constexpr = 16
        d3c_width: tl.constexpr = 16
        d3c_output_plane: tl.constexpr = OUTPUT_DIM_3 * OUTPUT_DIM_4
        d3c_output_spatial: tl.constexpr = OUTPUT_DIM_2 * d3c_output_plane
        d3c_filter_volume: tl.constexpr = (
            FILTER_DIM_2 * FILTER_DIM_3 * FILTER_DIM_4
        )
        d3c_reduction_extent: tl.constexpr = (
            CHANNELS_PER_GROUP * d3c_filter_volume
        )
        d3c_planes_per_batch: tl.constexpr = (
            CHANNELS_PER_GROUP * FILTER_DIM_2 * FILTER_DIM_3
        )
        d3c_task_count: tl.constexpr = INPUT_DIM_0 * d3c_planes_per_batch
        d3c_task = tl.program_id(0).to(tl.int64)
        while d3c_task < d3c_task_count:
            d3c_batch = d3c_task // d3c_planes_per_batch
            d3c_plane = d3c_task - d3c_batch * d3c_planes_per_batch
            d3c_filter_h = d3c_plane % FILTER_DIM_3
            d3c_channel_filter_d = d3c_plane // FILTER_DIM_3
            d3c_filter_d = d3c_channel_filter_d % FILTER_DIM_2
            d3c_input_channel = d3c_channel_filter_d // FILTER_DIM_2

            for d3c_filter_w in range(0, FILTER_DIM_4):
                d3c_input_block = tl.make_block_ptr(
                    base=(
                        input_ptr
                        + d3c_batch * INPUT_STRIDE_0
                        + d3c_input_channel * INPUT_STRIDE_1
                    ),
                    shape=(INPUT_DIM_2, INPUT_DIM_3, INPUT_DIM_4),
                    strides=(
                        INPUT_STRIDE_2,
                        INPUT_STRIDE_3,
                        INPUT_STRIDE_4,
                    ),
                    offsets=(
                        (
                            -PRE_PADDING_0
                            + d3c_filter_d * DILATION_0
                        ).to(tl.int32),
                        (
                            -PRE_PADDING_1
                            + d3c_filter_h * DILATION_1
                        ).to(tl.int32),
                        (
                            -PRE_PADDING_2
                            + d3c_filter_w * DILATION_2
                        ).to(tl.int32),
                    ),
                    block_shape=(OUTPUT_DIM_2, d3c_rows, d3c_width),
                    order=(2, 1, 0),
                )
                d3c_values = tl.load(
                    d3c_input_block,
                    boundary_check=(0, 1, 2),
                    padding_option="zero",
                )
                d3c_columns_block = tl.make_block_ptr(
                    base=(
                        columns_ptr
                        + d3c_batch
                        * d3c_output_spatial
                        * d3c_reduction_extent
                        + (d3c_plane * FILTER_DIM_4 + d3c_filter_w)
                        * d3c_output_spatial
                    ),
                    shape=(OUTPUT_DIM_2, OUTPUT_DIM_3, OUTPUT_DIM_4),
                    strides=(d3c_output_plane, OUTPUT_DIM_4, 1),
                    offsets=(0, 0, 0),
                    block_shape=(OUTPUT_DIM_2, d3c_rows, d3c_width),
                    order=(2, 1, 0),
                )
                tl.store(
                    d3c_columns_block,
                    d3c_values,
                    boundary_check=(0, 1, 2),
                )
            d3c_task += task_stride
        return

    if SPATIAL_RANK == 3:
        d3_block_spatial: tl.constexpr = 128
        d3_output_spatial: tl.constexpr = (
            OUTPUT_DIM_2 * OUTPUT_DIM_3 * OUTPUT_DIM_4
        )
        d3_spatial_tiles: tl.constexpr = tl.cdiv(
            d3_output_spatial, d3_block_spatial
        )
        d3_filter_volume: tl.constexpr = (
            FILTER_DIM_2 * FILTER_DIM_3 * FILTER_DIM_4
        )
        d3_reduction_extent: tl.constexpr = (
            CHANNELS_PER_GROUP * d3_filter_volume
        )
        d3_planes_per_batch: tl.constexpr = (
            CHANNELS_PER_GROUP * FILTER_DIM_2 * FILTER_DIM_3
        )
        d3_total_tasks: tl.constexpr = (
            INPUT_DIM_0 * d3_planes_per_batch * d3_spatial_tiles
        )
        d3_spatial_lane = tl.arange(0, d3_block_spatial)
        d3_task = tl.program_id(0).to(tl.int64)
        while d3_task < d3_total_tasks:
            d3_batch_plane = d3_task // d3_spatial_tiles
            d3_spatial_tile = (
                d3_task - d3_batch_plane * d3_spatial_tiles
            )
            d3_batch = d3_batch_plane // d3_planes_per_batch
            d3_plane = (
                d3_batch_plane - d3_batch * d3_planes_per_batch
            )
            d3_filter_h = d3_plane % FILTER_DIM_3
            d3_channel_filter_d = d3_plane // FILTER_DIM_3
            d3_filter_d = d3_channel_filter_d % FILTER_DIM_2
            d3_input_channel = d3_channel_filter_d // FILTER_DIM_2

            d3_output_linear = (
                d3_spatial_tile * d3_block_spatial + d3_spatial_lane
            )
            d3_output_mask = d3_output_linear < d3_output_spatial
            d3_safe_output_linear = tl.where(
                d3_output_mask, d3_output_linear, 0
            )
            d3_output_w = d3_safe_output_linear % OUTPUT_DIM_4
            d3_output_plane = d3_safe_output_linear // OUTPUT_DIM_4
            d3_output_h = d3_output_plane % OUTPUT_DIM_3
            d3_output_d = d3_output_plane // OUTPUT_DIM_3
            d3_input_d = (
                d3_output_d * CONV_STRIDE_0
                - PRE_PADDING_0
                + d3_filter_d * DILATION_0
            )
            d3_input_h = (
                d3_output_h * CONV_STRIDE_1
                - PRE_PADDING_1
                + d3_filter_h * DILATION_1
            )
            d3_valid_dh = (
                d3_output_mask
                & (d3_input_d >= 0)
                & (d3_input_d < INPUT_DIM_2)
                & (d3_input_h >= 0)
                & (d3_input_h < INPUT_DIM_3)
            )
            d3_safe_input_d = tl.where(d3_valid_dh, d3_input_d, 0)
            d3_safe_input_h = tl.where(d3_valid_dh, d3_input_h, 0)
            d3_column_plane = d3_plane * FILTER_DIM_4
            for d3_filter_w in range(0, FILTER_DIM_4):
                d3_input_w = (
                    d3_output_w * CONV_STRIDE_2
                    - PRE_PADDING_2
                    + d3_filter_w * DILATION_2
                )
                d3_valid_input = (
                    d3_valid_dh
                    & (d3_input_w >= 0)
                    & (d3_input_w < INPUT_DIM_4)
                )
                d3_safe_input_w = tl.where(
                    d3_valid_input, d3_input_w, 0
                )
                d3_values = tl.load(
                    input_ptr
                    + d3_batch * INPUT_STRIDE_0
                    + d3_input_channel * INPUT_STRIDE_1
                    + d3_safe_input_d * INPUT_STRIDE_2
                    + d3_safe_input_h * INPUT_STRIDE_3
                    + d3_safe_input_w * INPUT_STRIDE_4,
                    mask=d3_valid_input,
                    other=0.0,
                )
                tl.store(
                    columns_ptr
                    + d3_batch * d3_output_spatial * d3_reduction_extent
                    + (d3_column_plane + d3_filter_w)
                    * d3_output_spatial
                    + d3_output_linear,
                    d3_values,
                    mask=d3_output_mask,
                )
            d3_task += task_stride
        return

    compact_2d_rgb_stride2: tl.constexpr = (
        SPATIAL_RANK == 2
        and GROUPS == 1
        and CHANNELS_PER_GROUP <= 4
        and FILTER_DIM_3 == 3
        and FILTER_DIM_4 == 3
        and CONV_STRIDE_1 == 2
        and CONV_STRIDE_2 == 2
        and DILATION_1 == 1
        and DILATION_2 == 1
        and OUTPUT_DIM_3 >= 16
        and OUTPUT_DIM_4 >= 32
    )
    if compact_2d_rgb_stride2:
        rgb_block_channels: tl.constexpr = 4
        rgb_block_width: tl.constexpr = 64
        rgb_block_rows: tl.constexpr = 4
        # Use one row group per persistent worker. With three filter rows this
        # gives three balanced row tasks per worker per batch when the output
        # height covers all workers, while keeping that row range hot as the
        # worker advances filter_h.
        rgb_row_groups: tl.constexpr = (
            WORKER_COUNT
            if OUTPUT_DIM_3 >= WORKER_COUNT
            else OUTPUT_DIM_3
        )
        rgb_tasks_per_batch: tl.constexpr = (
            FILTER_DIM_3 * rgb_row_groups
        )
        rgb_task_count: tl.constexpr = (
            INPUT_DIM_0 * rgb_tasks_per_batch
        )
        rgb_task = tl.program_id(0).to(tl.int64)
        while rgb_task < rgb_task_count:
            rgb_batch = rgb_task // rgb_tasks_per_batch
            rgb_batch_task = rgb_task - rgb_batch * rgb_tasks_per_batch
            rgb_filter_h = rgb_batch_task // rgb_row_groups
            rgb_row_group = rgb_batch_task - rgb_filter_h * rgb_row_groups
            rgb_output_h = (
                rgb_row_group * OUTPUT_DIM_3 // rgb_row_groups
            )
            rgb_channel_start = tl.zeros((), dtype=tl.int64)
            rgb_output_h_end = (
                (rgb_row_group + 1) * OUTPUT_DIM_3 // rgb_row_groups
            )
            while rgb_output_h < rgb_output_h_end:
                rgb_input_h = (
                    rgb_output_h * CONV_STRIDE_1
                    - PRE_PADDING_1
                    + rgb_filter_h * DILATION_1
                )
                rgb_valid_h = (
                    (rgb_input_h >= 0) & (rgb_input_h < INPUT_DIM_3)
                )
                rgb_safe_input_h = tl.where(rgb_valid_h, rgb_input_h, 0)
                rgb_full_row_block = (
                    rgb_output_h + rgb_block_rows <= rgb_output_h_end
                ) & (
                    rgb_input_h
                    + (rgb_block_rows - 1) * CONV_STRIDE_1
                    < INPUT_DIM_3
                ) & rgb_valid_h
                if rgb_full_row_block:
                    rgb_output_w_start = tl.zeros((), dtype=tl.int64)
                    while rgb_output_w_start < OUTPUT_DIM_4:
                        _convolution_fprop_im2col_rgb_stride2_rows(
                            input_ptr,
                            columns_ptr,
                            rgb_batch,
                            rgb_filter_h,
                            rgb_output_h,
                            rgb_output_w_start,
                            rgb_input_h,
                            CHANNELS_PER_GROUP,
                            INPUT_DIM_4,
                            INPUT_STRIDE_0,
                            INPUT_STRIDE_1,
                            INPUT_STRIDE_3,
                            INPUT_STRIDE_4,
                            FILTER_DIM_3,
                            FILTER_DIM_4,
                            OUTPUT_DIM_3,
                            OUTPUT_DIM_4,
                            PRE_PADDING_2,
                        )
                        rgb_output_w_start += rgb_block_width
                    rgb_output_h += rgb_block_rows
                else:
                    rgb_output_w_start = tl.zeros((), dtype=tl.int64)
                    while rgb_output_w_start < OUTPUT_DIM_4:
                        _convolution_fprop_im2col_stride2_width(
                            input_ptr,
                            columns_ptr,
                            rgb_batch,
                            rgb_filter_h,
                            rgb_output_h,
                            rgb_channel_start,
                            rgb_output_w_start,
                            rgb_safe_input_h,
                            rgb_valid_h,
                            CHANNELS_PER_GROUP,
                            INPUT_DIM_4,
                            INPUT_STRIDE_0,
                            INPUT_STRIDE_1,
                            INPUT_STRIDE_3,
                            INPUT_STRIDE_4,
                            FILTER_DIM_3,
                            FILTER_DIM_4,
                            OUTPUT_DIM_3,
                            OUTPUT_DIM_4,
                            PRE_PADDING_2,
                            DILATION_2,
                            rgb_block_channels,
                            rgb_block_width,
                        )
                        rgb_output_w_start += rgb_block_width
                    rgb_output_h += 1
            rgb_task += task_stride
        return

    compact_2d_3x3: tl.constexpr = (
        SPATIAL_RANK == 2
        and GROUPS == 1
        and CHANNELS_PER_GROUP == 32
        and FILTER_DIM_3 == 3
        and FILTER_DIM_4 == 3
        and OUTPUT_DIM_4 == 32
        and CONV_STRIDE_1 == 1
        and CONV_STRIDE_2 == 1
    )
    if compact_2d_3x3:
        compact_channels: tl.constexpr = 32
        compact_width: tl.constexpr = 32
        compact_rows_per_task: tl.constexpr = 16
        compact_filter_area: tl.constexpr = 9
        compact_output_spatial: tl.constexpr = OUTPUT_DIM_3 * OUTPUT_DIM_4
        compact_reduction: tl.constexpr = (
            CHANNELS_PER_GROUP * compact_filter_area
        )
        compact_row_groups: tl.constexpr = tl.cdiv(
            OUTPUT_DIM_3, compact_rows_per_task
        )
        compact_task_count: tl.constexpr = (
            INPUT_DIM_0 * compact_filter_area * compact_row_groups
        )

        compact_task = tl.program_id(0).to(tl.int64)
        while compact_task < compact_task_count:
            compact_batch_filter = compact_task // compact_row_groups
            compact_row_group = (
                compact_task
                - compact_batch_filter * compact_row_groups
            )
            compact_batch = compact_batch_filter // compact_filter_area
            compact_filter_position = (
                compact_batch_filter
                - compact_batch * compact_filter_area
            )
            compact_filter_h = compact_filter_position // FILTER_DIM_4
            compact_filter_w = (
                compact_filter_position
                - compact_filter_h * FILTER_DIM_4
            )
            compact_output_h = (
                compact_row_group * compact_rows_per_task
            )
            compact_output_h_end = tl.minimum(
                compact_output_h + compact_rows_per_task, OUTPUT_DIM_3
            )
            while compact_output_h < compact_output_h_end:
                compact_input_h = (
                    compact_output_h
                    - PRE_PADDING_1
                    + compact_filter_h * DILATION_1
                )
                compact_valid_h = (
                    (compact_input_h >= 0)
                    & (compact_input_h < INPUT_DIM_3)
                )
                compact_safe_h = tl.where(
                    compact_valid_h, compact_input_h, 0
                )
                compact_input_block = tl.make_block_ptr(
                    base=(
                        input_ptr
                        + compact_batch * INPUT_STRIDE_0
                        + compact_safe_h * INPUT_STRIDE_3
                    ),
                    shape=(CHANNELS_PER_GROUP, INPUT_DIM_4),
                    strides=(INPUT_STRIDE_1, INPUT_STRIDE_4),
                    offsets=(
                        0,
                        (
                            -PRE_PADDING_2
                            + compact_filter_w * DILATION_2
                        ).to(tl.int32),
                    ),
                    block_shape=(compact_channels, compact_width),
                    order=(1, 0),
                )
                compact_values = tl.load(
                    compact_input_block,
                    boundary_check=(1,),
                    padding_option="zero",
                )
                compact_values = tl.where(
                    compact_valid_h, compact_values, 0.0
                )
                compact_columns_block = tl.make_block_ptr(
                    base=(
                        columns_ptr
                        + compact_batch
                        * compact_output_spatial
                        * compact_reduction
                        + compact_filter_position
                        * compact_output_spatial
                    ),
                    shape=(CHANNELS_PER_GROUP, compact_output_spatial),
                    strides=(
                        compact_filter_area * compact_output_spatial,
                        1,
                    ),
                    offsets=(
                        0,
                        (compact_output_h * OUTPUT_DIM_4).to(tl.int32),
                    ),
                    block_shape=(compact_channels, compact_width),
                    order=(1, 0),
                )
                tl.store(compact_columns_block, compact_values)
                compact_output_h += 1
            compact_task += task_stride
        return

    flattened_2d: tl.constexpr = (
        SPATIAL_RANK == 2
        and (CONV_STRIDE_2 != 1 or OUTPUT_DIM_4 % 32 != 0)
    )
    if flattened_2d:
        blocked_output_width: tl.constexpr = 32
        blocked_valid_output_width: tl.constexpr = (
            OUTPUT_DIM_4
            if OUTPUT_DIM_4 < blocked_output_width
            else blocked_output_width
        )
        blocked_input_window_span: tl.constexpr = (
            (blocked_valid_output_width - 1) * CONV_STRIDE_2
            + (FILTER_DIM_4 - 1) * DILATION_2
            + 1
        )
        blocked_stride2: tl.constexpr = (
            CONV_STRIDE_2 == 2 and blocked_input_window_span <= 128
        )
        if blocked_stride2:
            blocked_channels: tl.constexpr = (
                64
                if CHANNELS_PER_GROUP >= 64
                else (
                    32
                    if CHANNELS_PER_GROUP >= 32
                    else (
                        16
                        if CHANNELS_PER_GROUP >= 16
                        else (8 if CHANNELS_PER_GROUP >= 8 else 4)
                    )
                )
            )
            blocked_channel_tiles: tl.constexpr = tl.cdiv(
                CHANNELS_PER_GROUP, blocked_channels
            )
            blocked_width_tiles: tl.constexpr = tl.cdiv(
                OUTPUT_DIM_4, blocked_output_width
            )
            blocked_tasks_per_row: tl.constexpr = (
                blocked_channel_tiles * blocked_width_tiles
            )
            blocked_tasks_per_filter_h: tl.constexpr = (
                OUTPUT_DIM_3 * blocked_tasks_per_row
            )
            blocked_tasks_per_batch: tl.constexpr = (
                FILTER_DIM_3 * blocked_tasks_per_filter_h
            )
            blocked_task_count: tl.constexpr = (
                INPUT_DIM_0 * blocked_tasks_per_batch
            )
            blocked_task = tl.program_id(0).to(tl.int64)
            while blocked_task < blocked_task_count:
                blocked_batch = blocked_task // blocked_tasks_per_batch
                blocked_batch_task = (
                    blocked_task
                    - blocked_batch * blocked_tasks_per_batch
                )
                blocked_filter_h = (
                    blocked_batch_task // blocked_tasks_per_filter_h
                )
                blocked_filter_task = (
                    blocked_batch_task
                    - blocked_filter_h * blocked_tasks_per_filter_h
                )
                blocked_output_h = (
                    blocked_filter_task // blocked_tasks_per_row
                )
                blocked_row_task = (
                    blocked_filter_task
                    - blocked_output_h * blocked_tasks_per_row
                )
                blocked_channel_tile = (
                    blocked_row_task // blocked_width_tiles
                )
                blocked_width_tile = (
                    blocked_row_task
                    - blocked_channel_tile * blocked_width_tiles
                )
                blocked_channel_start = (
                    blocked_channel_tile * blocked_channels
                )
                blocked_output_w_start = (
                    blocked_width_tile * blocked_output_width
                )
                blocked_input_h = (
                    blocked_output_h * CONV_STRIDE_1
                    - PRE_PADDING_1
                    + blocked_filter_h * DILATION_1
                )
                blocked_valid_h = (
                    (blocked_input_h >= 0)
                    & (blocked_input_h < INPUT_DIM_3)
                )
                blocked_safe_h = tl.where(
                    blocked_valid_h, blocked_input_h, 0
                )
                _convolution_fprop_im2col_stride2_width(
                    input_ptr,
                    columns_ptr,
                    blocked_batch,
                    blocked_filter_h,
                    blocked_output_h,
                    blocked_channel_start,
                    blocked_output_w_start,
                    blocked_safe_h,
                    blocked_valid_h,
                    CHANNELS_PER_GROUP,
                    INPUT_DIM_4,
                    INPUT_STRIDE_0,
                    INPUT_STRIDE_1,
                    INPUT_STRIDE_3,
                    INPUT_STRIDE_4,
                    FILTER_DIM_3,
                    FILTER_DIM_4,
                    OUTPUT_DIM_3,
                    OUTPUT_DIM_4,
                    PRE_PADDING_2,
                    DILATION_2,
                    blocked_channels,
                    blocked_output_width,
                )
                blocked_task += task_stride
            return

        flat_block: tl.constexpr = 128
        flat_filter_area: tl.constexpr = FILTER_DIM_3 * FILTER_DIM_4
        flat_output_spatial: tl.constexpr = OUTPUT_DIM_3 * OUTPUT_DIM_4
        flat_output_tiles: tl.constexpr = tl.cdiv(
            flat_output_spatial, flat_block
        )
        flat_planes_per_batch: tl.constexpr = (
            CHANNELS_PER_GROUP * FILTER_DIM_3
        )
        flat_task_count: tl.constexpr = (
            INPUT_DIM_0 * flat_planes_per_batch * flat_output_tiles
        )
        flat_lane = tl.arange(0, flat_block)
        flat_task = tl.program_id(0).to(tl.int64)
        while flat_task < flat_task_count:
            flat_plane = flat_task // flat_output_tiles
            flat_output_tile = (
                flat_task - flat_plane * flat_output_tiles
            )
            flat_batch = flat_plane // flat_planes_per_batch
            flat_plane_in_batch = (
                flat_plane - flat_batch * flat_planes_per_batch
            )
            flat_input_channel = flat_plane_in_batch // FILTER_DIM_3
            flat_filter_h = (
                flat_plane_in_batch
                - flat_input_channel * FILTER_DIM_3
            )
            flat_output_hw = flat_output_tile * flat_block + flat_lane
            flat_output_mask = flat_output_hw < flat_output_spatial
            flat_output_h = flat_output_hw // OUTPUT_DIM_4
            flat_output_w = (
                flat_output_hw - flat_output_h * OUTPUT_DIM_4
            )
            flat_input_h = (
                flat_output_h * CONV_STRIDE_1
                - PRE_PADDING_1
                + flat_filter_h * DILATION_1
            )
            flat_valid_h = (
                flat_output_mask
                & (flat_input_h >= 0)
                & (flat_input_h < INPUT_DIM_3)
            )
            flat_safe_h = tl.where(flat_valid_h, flat_input_h, 0)
            flat_column_base = (
                flat_batch * flat_output_spatial * CHANNELS_PER_GROUP
                * flat_filter_area
                + (
                    flat_input_channel * flat_filter_area
                    + flat_filter_h * FILTER_DIM_4
                )
                * flat_output_spatial
            )
            for flat_filter_w in range(0, FILTER_DIM_4):
                flat_input_w = (
                    flat_output_w * CONV_STRIDE_2
                    - PRE_PADDING_2
                    + flat_filter_w * DILATION_2
                )
                flat_valid = (
                    flat_valid_h
                    & (flat_input_w >= 0)
                    & (flat_input_w < INPUT_DIM_4)
                )
                flat_safe_w = tl.where(flat_valid, flat_input_w, 0)
                flat_values = tl.load(
                    input_ptr
                    + flat_batch * INPUT_STRIDE_0
                    + flat_input_channel * INPUT_STRIDE_1
                    + flat_safe_h * INPUT_STRIDE_3
                    + flat_safe_w * INPUT_STRIDE_4,
                    mask=flat_valid,
                    other=0.0,
                )
                tl.store(
                    columns_ptr
                    + flat_column_base
                    + flat_filter_w * flat_output_spatial
                    + flat_output_hw,
                    flat_values,
                    mask=flat_output_mask,
                )
            flat_task += task_stride
        return

    filter_area: tl.constexpr = FILTER_DIM_3 * FILTER_DIM_4
    reduction_extent: tl.constexpr = CHANNELS_PER_GROUP * filter_area
    tile_channels: tl.constexpr = (
        64
        if SPATIAL_RANK == 2 and CHANNELS_PER_GROUP >= 64
        else 32
    )
    tile_length: tl.constexpr = 128
    tile_width: tl.constexpr = 32
    rows_per_task: tl.constexpr = 16
    batch_filter_count: tl.constexpr = INPUT_DIM_0 * filter_area
    row_groups: tl.constexpr = tl.cdiv(OUTPUT_DIM_3, rows_per_task)
    column_task_count: tl.constexpr = (
        batch_filter_count
        if SPATIAL_RANK == 1
        else batch_filter_count * row_groups
    )
    output_spatial: tl.constexpr = OUTPUT_DIM_3 * OUTPUT_DIM_4

    # Store columns physically as [batch, K, output_spatial].  This lets the
    # following MatMul consume a strided [batch, output_spatial, K] view
    # without an explicit tile transpose.
    channel_lane = tl.arange(0, tile_channels)
    length_lane = tl.arange(0, tile_length)
    # The 1D path maps workers to (batch, filter_w).  The 2D path also splits
    # output rows into fixed groups so every worker receives comparable work.
    column_task = tl.program_id(0).to(tl.int64)
    while column_task < column_task_count:
        if SPATIAL_RANK == 1:
            batch_filter = column_task
            output_h_start = tl.zeros((), dtype=tl.int64)
            output_h_end = OUTPUT_DIM_3
        else:
            batch_filter = column_task // row_groups
            row_group = column_task - batch_filter * row_groups
            output_h_start = row_group * rows_per_task
            output_h_end = tl.minimum(
                output_h_start + rows_per_task, OUTPUT_DIM_3
            )
        batch = batch_filter // filter_area
        filter_position = batch_filter - batch * filter_area
        filter_h = filter_position // FILTER_DIM_4
        filter_w = filter_position - filter_h * FILTER_DIM_4
        channel_start = tl.zeros((), dtype=tl.int64)
        while channel_start < CHANNELS_PER_GROUP:
            input_channel = channel_start + channel_lane
            channel_mask = input_channel < CHANNELS_PER_GROUP
            channel_offset = channel_start.to(tl.int32)
            if SPATIAL_RANK == 1:
                length_start = tl.zeros((), dtype=tl.int64)
                while length_start < OUTPUT_DIM_4:
                    if CONV_STRIDE_2 == 1:
                        input_block = tl.make_block_ptr(
                            base=input_ptr + batch * INPUT_STRIDE_0,
                            shape=(CHANNELS_PER_GROUP, INPUT_DIM_4),
                            strides=(INPUT_STRIDE_1, INPUT_STRIDE_4),
                            offsets=(
                                channel_offset,
                                (
                                    length_start
                                    - PRE_PADDING_2
                                    + filter_w * DILATION_2
                                ).to(tl.int32),
                            ),
                            block_shape=(tile_channels, tile_length),
                            order=(1, 0),
                        )
                        input_values = tl.load(
                            input_block,
                            boundary_check=(0, 1),
                            padding_option="zero",
                        )
                    elif CONV_STRIDE_2 == 2:
                        input_window = tl.make_block_ptr(
                            base=input_ptr + batch * INPUT_STRIDE_0,
                            shape=(CHANNELS_PER_GROUP, INPUT_DIM_4),
                            strides=(INPUT_STRIDE_1, INPUT_STRIDE_4),
                            offsets=(
                                channel_offset,
                                (
                                    length_start * 2
                                    - PRE_PADDING_2
                                    + filter_w * DILATION_2
                                ).to(tl.int32),
                            ),
                            block_shape=(tile_channels, tile_length * 2),
                            order=(1, 0),
                        )
                        input_window_values = tl.load(
                            input_window,
                            boundary_check=(0, 1),
                            padding_option="zero",
                        )
                        gather_lane = length_lane * 2
                        gather_index = tl.broadcast_to(
                            gather_lane[None, :],
                            (tile_channels, tile_length),
                        )
                        input_values = tl.gather(
                            input_window_values, gather_index, axis=1
                        )
                    else:
                        output_l = length_start + length_lane
                        length_mask = output_l < OUTPUT_DIM_4
                        input_l = (
                            output_l * CONV_STRIDE_2
                            - PRE_PADDING_2
                            + filter_w * DILATION_2
                        )
                        valid_input_l = (
                            (input_l >= 0) & (input_l < INPUT_DIM_4)
                        )
                        safe_input_l = tl.where(valid_input_l, input_l, 0)
                        input_values = tl.load(
                            input_ptr
                            + batch * INPUT_STRIDE_0
                            + input_channel[:, None] * INPUT_STRIDE_1
                            + safe_input_l[None, :] * INPUT_STRIDE_4,
                            mask=(
                                channel_mask[:, None]
                                & length_mask[None, :]
                                & valid_input_l[None, :]
                            ),
                            other=0.0,
                        )
                    columns_block = tl.make_block_ptr(
                        base=(
                            columns_ptr
                            + batch * OUTPUT_DIM_4 * reduction_extent
                            + filter_w * OUTPUT_DIM_4
                        ),
                        shape=(CHANNELS_PER_GROUP, OUTPUT_DIM_4),
                        strides=(FILTER_DIM_4 * OUTPUT_DIM_4, 1),
                        offsets=(
                            channel_offset,
                            length_start.to(tl.int32),
                        ),
                        block_shape=(tile_channels, tile_length),
                        order=(1, 0),
                    )
                    tl.store(
                        columns_block,
                        input_values,
                        boundary_check=(0, 1),
                    )
                    length_start += tile_length
            else:
                output_h = output_h_start
                while output_h < output_h_end:
                    input_h = (
                        output_h * CONV_STRIDE_1
                        - PRE_PADDING_1
                        + filter_h * DILATION_1
                    )
                    valid_input_h = (
                        (input_h >= 0) & (input_h < INPUT_DIM_3)
                    )
                    safe_input_h = tl.where(valid_input_h, input_h, 0)
                    output_w_start = tl.zeros((), dtype=tl.int64)
                    while output_w_start < OUTPUT_DIM_4:
                        input_block = tl.make_block_ptr(
                            base=(
                                input_ptr
                                + batch * INPUT_STRIDE_0
                                + safe_input_h * INPUT_STRIDE_3
                            ),
                            shape=(CHANNELS_PER_GROUP, INPUT_DIM_4),
                            strides=(INPUT_STRIDE_1, INPUT_STRIDE_4),
                            offsets=(
                                channel_offset,
                                (
                                    output_w_start
                                    - PRE_PADDING_2
                                    + filter_w * DILATION_2
                                ).to(tl.int32),
                            ),
                            block_shape=(tile_channels, tile_width),
                            order=(1, 0),
                        )
                        input_values = tl.load(
                            input_block,
                            boundary_check=(0, 1),
                            padding_option="zero",
                        )
                        input_values = tl.where(
                            valid_input_h, input_values, 0.0
                        )
                        columns_block = tl.make_block_ptr(
                            base=(
                                columns_ptr
                                + batch * output_spatial * reduction_extent
                                + filter_position * output_spatial
                            ),
                            shape=(CHANNELS_PER_GROUP, output_spatial),
                            strides=(filter_area * output_spatial, 1),
                            offsets=(
                                channel_offset,
                                (
                                    output_h * OUTPUT_DIM_4
                                    + output_w_start
                                ).to(tl.int32),
                            ),
                            block_shape=(tile_channels, tile_width),
                            order=(1, 0),
                        )
                        tl.store(
                            columns_block,
                            input_values,
                            boundary_check=(0, 1),
                        )
                        output_w_start += tile_width
                    output_h += 1
            channel_start += tile_channels
        column_task += task_stride


@triton.jit
def _convolution_fprop_1d_gemm(
    input_ptr,
    filter_ptr,
    output_ptr,
    GROUPS: tl.constexpr,
    OUTPUT_CHANNELS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    INPUT_DIM_0: tl.constexpr,
    INPUT_DIM_4: tl.constexpr,
    INPUT_STRIDE_0: tl.constexpr,
    INPUT_STRIDE_1: tl.constexpr,
    INPUT_STRIDE_4: tl.constexpr,
    FILTER_DIM_4: tl.constexpr,
    FILTER_STRIDE_0: tl.constexpr,
    FILTER_STRIDE_1: tl.constexpr,
    FILTER_STRIDE_4: tl.constexpr,
    OUTPUT_DIM_4: tl.constexpr,
    OUTPUT_STRIDE_0: tl.constexpr,
    OUTPUT_STRIDE_1: tl.constexpr,
    OUTPUT_STRIDE_4: tl.constexpr,
    PRE_PADDING_2: tl.constexpr,
    CONV_STRIDE_2: tl.constexpr,
    DILATION_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_output_channels: tl.constexpr = 64
    block_output_length: tl.constexpr = BLOCK_SIZE // 4
    block_reduction: tl.constexpr = 64
    output_channels_per_group: tl.constexpr = OUTPUT_CHANNELS // GROUPS
    channel_tiles: tl.constexpr = tl.cdiv(
        output_channels_per_group, block_output_channels
    )
    length_tiles: tl.constexpr = tl.cdiv(
        OUTPUT_DIM_4, block_output_length
    )
    tiles_per_group: tl.constexpr = channel_tiles * length_tiles
    tiles_per_batch: tl.constexpr = GROUPS * tiles_per_group
    total_tasks: tl.constexpr = INPUT_DIM_0 * tiles_per_batch

    task = tl.program_id(0).to(tl.int64)
    task_stride = tl.num_programs(0)
    while task < total_tasks:
        batch = task // tiles_per_batch
        batch_tile = task % tiles_per_batch
        group = batch_tile // tiles_per_group
        group_tile = batch_tile % tiles_per_group
        channel_tile = group_tile // length_tiles
        length_tile = group_tile % length_tiles

        group_output_channels = (
            channel_tile * block_output_channels
            + tl.arange(0, block_output_channels)
        )
        channel_mask = group_output_channels < output_channels_per_group
        safe_group_output_channels = tl.where(
            channel_mask, group_output_channels, 0
        )
        output_channels = (
            group * output_channels_per_group
            + safe_group_output_channels
        )
        output_l = (
            length_tile * block_output_length
            + tl.arange(0, block_output_length)
        )
        length_mask = output_l < OUTPUT_DIM_4
        safe_output_l = tl.where(length_mask, output_l, 0)
        reduction_base = tl.arange(0, block_reduction)
        accumulator = tl.zeros(
            (block_output_channels, block_output_length), dtype=tl.float32
        )

        for filter_w in range(0, FILTER_DIM_4):
            input_l = (
                safe_output_l * CONV_STRIDE_2
                - PRE_PADDING_2
                + filter_w * DILATION_2
            )
            valid_input_l = (input_l >= 0) & (input_l < INPUT_DIM_4)
            safe_input_l = tl.where(valid_input_l, input_l, 0)
            for reduction_start in range(
                0, CHANNELS_PER_GROUP, block_reduction
            ):
                input_channel = reduction_start + reduction_base
                reduction_mask = input_channel < CHANNELS_PER_GROUP
                safe_input_channel = tl.where(
                    reduction_mask, input_channel, 0
                )
                weights = tl.load(
                    filter_ptr
                    + output_channels[:, None] * FILTER_STRIDE_0
                    + safe_input_channel[None, :] * FILTER_STRIDE_1
                    + filter_w * FILTER_STRIDE_4,
                    mask=(
                        channel_mask[:, None]
                        & reduction_mask[None, :]
                    ),
                    other=0.0,
                )
                input_values = tl.load(
                    input_ptr
                    + batch * INPUT_STRIDE_0
                    + (
                        group * CHANNELS_PER_GROUP
                        + safe_input_channel[:, None]
                    )
                    * INPUT_STRIDE_1
                    + safe_input_l[None, :] * INPUT_STRIDE_4,
                    mask=(
                        reduction_mask[:, None]
                        & length_mask[None, :]
                        & valid_input_l[None, :]
                    ),
                    other=0.0,
                )
                accumulator += tl.dot(
                    weights,
                    input_values,
                    input_precision="ieee",
                )

        if OUTPUT_STRIDE_1 == 1:
            tl.store(
                output_ptr
                + batch * OUTPUT_STRIDE_0
                + safe_output_l[:, None] * OUTPUT_STRIDE_4
                + output_channels[None, :] * OUTPUT_STRIDE_1,
                tl.trans(accumulator).to(output_ptr.dtype.element_ty),
                mask=length_mask[:, None] & channel_mask[None, :],
            )
        else:
            tl.store(
                output_ptr
                + batch * OUTPUT_STRIDE_0
                + output_channels[:, None] * OUTPUT_STRIDE_1
                + safe_output_l[None, :] * OUTPUT_STRIDE_4,
                accumulator.to(output_ptr.dtype.element_ty),
                mask=channel_mask[:, None] & length_mask[None, :],
            )
        task += task_stride


@triton.jit
def _convolution_fprop_2d_nchw_1x1(
    input_ptr,
    filter_ptr,
    output_ptr,
    GROUPS: tl.constexpr,
    OUTPUT_CHANNELS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    INPUT_DIM_0: tl.constexpr,
    INPUT_DIM_3: tl.constexpr,
    INPUT_DIM_4: tl.constexpr,
    INPUT_STRIDE_0: tl.constexpr,
    INPUT_STRIDE_1: tl.constexpr,
    FILTER_STRIDE_0: tl.constexpr,
    FILTER_STRIDE_1: tl.constexpr,
    OUTPUT_STRIDE_0: tl.constexpr,
    OUTPUT_STRIDE_1: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_channels: tl.constexpr = BLOCK_SIZE // 2
    block_spatial: tl.constexpr = BLOCK_SIZE // 2
    block_reduction: tl.constexpr = 64
    spatial: tl.constexpr = INPUT_DIM_3 * INPUT_DIM_4
    output_channels_per_group: tl.constexpr = OUTPUT_CHANNELS // GROUPS
    channel_tiles: tl.constexpr = tl.cdiv(
        output_channels_per_group, block_channels
    )
    spatial_tiles: tl.constexpr = tl.cdiv(spatial, block_spatial)
    tiles_per_group: tl.constexpr = channel_tiles * spatial_tiles
    tiles_per_batch: tl.constexpr = GROUPS * tiles_per_group
    total_tasks: tl.constexpr = INPUT_DIM_0 * tiles_per_batch

    task = tl.program_id(0).to(tl.int64)
    task_stride = tl.num_programs(0)
    while task < total_tasks:
        batch = task // tiles_per_batch
        batch_tile = task % tiles_per_batch
        group = batch_tile // tiles_per_group
        group_tile = batch_tile % tiles_per_group
        channel_tile = group_tile // spatial_tiles
        spatial_tile = group_tile % spatial_tiles

        group_output_channels = (
            channel_tile * block_channels + tl.arange(0, block_channels)
        )
        channel_mask = group_output_channels < output_channels_per_group
        output_channels = (
            group * output_channels_per_group + group_output_channels
        )
        spatial_offsets = (
            spatial_tile * block_spatial + tl.arange(0, block_spatial)
        )
        spatial_mask = spatial_offsets < spatial
        reduction = tl.arange(0, block_reduction)
        accumulator = tl.zeros(
            (block_channels, block_spatial), dtype=tl.float32
        )

        for reduction_start in range(
            0, CHANNELS_PER_GROUP, block_reduction
        ):
            reduction_offsets = reduction_start + reduction
            reduction_mask = reduction_offsets < CHANNELS_PER_GROUP
            weights = tl.load(
                filter_ptr
                + output_channels[:, None] * FILTER_STRIDE_0
                + reduction_offsets[None, :] * FILTER_STRIDE_1,
                mask=channel_mask[:, None] & reduction_mask[None, :],
                other=0.0,
            )
            input_values = tl.load(
                input_ptr
                + batch * INPUT_STRIDE_0
                + (
                    group * CHANNELS_PER_GROUP
                    + reduction_offsets[:, None]
                )
                * INPUT_STRIDE_1
                + spatial_offsets[None, :],
                mask=reduction_mask[:, None] & spatial_mask[None, :],
                other=0.0,
            )
            accumulator += tl.dot(
                weights,
                input_values,
                input_precision="ieee",
            )

        tl.store(
            output_ptr
            + batch * OUTPUT_STRIDE_0
            + output_channels[:, None] * OUTPUT_STRIDE_1
            + spatial_offsets[None, :],
            accumulator.to(output_ptr.dtype.element_ty),
            mask=channel_mask[:, None] & spatial_mask[None, :],
        )
        task += task_stride


@triton.jit
def _convolution_fprop_2d_nchw_tiled(
    input_ptr,
    filter_ptr,
    output_ptr,
    GROUPS: tl.constexpr,
    OUTPUT_CHANNELS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    INPUT_DIM_0: tl.constexpr,
    INPUT_DIM_3: tl.constexpr,
    INPUT_DIM_4: tl.constexpr,
    INPUT_STRIDE_0: tl.constexpr,
    INPUT_STRIDE_1: tl.constexpr,
    INPUT_STRIDE_3: tl.constexpr,
    INPUT_STRIDE_4: tl.constexpr,
    FILTER_DIM_3: tl.constexpr,
    FILTER_DIM_4: tl.constexpr,
    FILTER_STRIDE_0: tl.constexpr,
    FILTER_STRIDE_1: tl.constexpr,
    FILTER_STRIDE_3: tl.constexpr,
    FILTER_STRIDE_4: tl.constexpr,
    OUTPUT_DIM_3: tl.constexpr,
    OUTPUT_DIM_4: tl.constexpr,
    OUTPUT_STRIDE_0: tl.constexpr,
    OUTPUT_STRIDE_1: tl.constexpr,
    OUTPUT_STRIDE_3: tl.constexpr,
    OUTPUT_STRIDE_4: tl.constexpr,
    PRE_PADDING_1: tl.constexpr,
    PRE_PADDING_2: tl.constexpr,
    CONV_STRIDE_1: tl.constexpr,
    CONV_STRIDE_2: tl.constexpr,
    DILATION_1: tl.constexpr,
    DILATION_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WORKER_COUNT: tl.constexpr,
):
    block_channels: tl.constexpr = BLOCK_SIZE // 2
    block_spatial: tl.constexpr = BLOCK_SIZE // 2
    block_reduction: tl.constexpr = 64
    output_area: tl.constexpr = OUTPUT_DIM_3 * OUTPUT_DIM_4
    output_channels_per_group: tl.constexpr = OUTPUT_CHANNELS // GROUPS
    channel_tiles: tl.constexpr = tl.cdiv(
        output_channels_per_group, block_channels
    )
    spatial_tiles: tl.constexpr = tl.cdiv(output_area, block_spatial)
    tiles_per_group: tl.constexpr = channel_tiles * spatial_tiles
    tiles_per_batch: tl.constexpr = GROUPS * tiles_per_group
    total_tasks: tl.constexpr = INPUT_DIM_0 * tiles_per_batch

    task = tl.program_id(0).to(tl.int64)
    task_stride = tl.num_programs(0)
    while task < total_tasks:
        batch = task // tiles_per_batch
        batch_tile = task % tiles_per_batch
        group = batch_tile // tiles_per_group
        group_tile = batch_tile % tiles_per_group
        channel_tile = group_tile // spatial_tiles
        spatial_tile = group_tile % spatial_tiles

        group_output_channels = (
            channel_tile * block_channels + tl.arange(0, block_channels)
        )
        channel_mask = group_output_channels < output_channels_per_group
        safe_group_output_channels = tl.where(
            channel_mask, group_output_channels, 0
        )
        output_channels = (
            group * output_channels_per_group + safe_group_output_channels
        )
        spatial_offsets = (
            spatial_tile * block_spatial + tl.arange(0, block_spatial)
        )
        spatial_mask = spatial_offsets < output_area
        safe_spatial_offsets = tl.where(spatial_mask, spatial_offsets, 0)
        output_h = safe_spatial_offsets // OUTPUT_DIM_4
        output_w = safe_spatial_offsets % OUTPUT_DIM_4
        reduction = tl.arange(0, block_reduction)
        accumulator = tl.zeros(
            (block_channels, block_spatial), dtype=tl.float32
        )

        for filter_h in range(0, FILTER_DIM_3):
            input_h = (
                output_h * CONV_STRIDE_1
                - PRE_PADDING_1
                + filter_h * DILATION_1
            )
            valid_h = (input_h >= 0) & (input_h < INPUT_DIM_3)
            safe_input_h = tl.where(valid_h, input_h, 0)
            for filter_w in range(0, FILTER_DIM_4):
                input_w = (
                    output_w * CONV_STRIDE_2
                    - PRE_PADDING_2
                    + filter_w * DILATION_2
                )
                valid_w = (input_w >= 0) & (input_w < INPUT_DIM_4)
                safe_input_w = tl.where(valid_w, input_w, 0)
                valid_spatial = spatial_mask & valid_h & valid_w
                for reduction_start in range(
                    0, CHANNELS_PER_GROUP, block_reduction
                ):
                    reduction_offsets = reduction_start + reduction
                    reduction_mask = reduction_offsets < CHANNELS_PER_GROUP
                    weights = tl.load(
                        filter_ptr
                        + output_channels[:, None] * FILTER_STRIDE_0
                        + reduction_offsets[None, :] * FILTER_STRIDE_1
                        + filter_h * FILTER_STRIDE_3
                        + filter_w * FILTER_STRIDE_4,
                        mask=(
                            channel_mask[:, None]
                            & reduction_mask[None, :]
                        ),
                        other=0.0,
                    )
                    input_values = tl.load(
                        input_ptr
                        + batch * INPUT_STRIDE_0
                        + (
                            group * CHANNELS_PER_GROUP
                            + reduction_offsets[:, None]
                        )
                        * INPUT_STRIDE_1
                        + safe_input_h[None, :] * INPUT_STRIDE_3
                        + safe_input_w[None, :] * INPUT_STRIDE_4,
                        mask=(
                            reduction_mask[:, None]
                            & valid_spatial[None, :]
                        ),
                        other=0.0,
                    )
                    accumulator += tl.dot(
                        weights,
                        input_values,
                        input_precision="ieee",
                    )

        tl.store(
            output_ptr
            + batch * OUTPUT_STRIDE_0
            + output_channels[:, None] * OUTPUT_STRIDE_1
            + output_h[None, :] * OUTPUT_STRIDE_3
            + output_w[None, :] * OUTPUT_STRIDE_4,
            accumulator.to(output_ptr.dtype.element_ty),
            mask=channel_mask[:, None] & spatial_mask[None, :],
        )
        task += task_stride


@triton.jit
def _convolution_fprop_2d_nchw_3x3_blocked(
    input_ptr,
    filter_ptr,
    output_ptr,
    OUTPUT_CHANNELS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    INPUT_DIM_0: tl.constexpr,
    INPUT_DIM_3: tl.constexpr,
    INPUT_DIM_4: tl.constexpr,
    INPUT_STRIDE_0: tl.constexpr,
    INPUT_STRIDE_1: tl.constexpr,
    INPUT_STRIDE_3: tl.constexpr,
    INPUT_STRIDE_4: tl.constexpr,
    FILTER_STRIDE_0: tl.constexpr,
    FILTER_STRIDE_1: tl.constexpr,
    FILTER_STRIDE_3: tl.constexpr,
    FILTER_STRIDE_4: tl.constexpr,
    OUTPUT_DIM_3: tl.constexpr,
    OUTPUT_DIM_4: tl.constexpr,
    OUTPUT_STRIDE_0: tl.constexpr,
    OUTPUT_STRIDE_1: tl.constexpr,
    OUTPUT_STRIDE_3: tl.constexpr,
    OUTPUT_STRIDE_4: tl.constexpr,
    PRE_PADDING_1: tl.constexpr,
    PRE_PADDING_2: tl.constexpr,
    CONV_STRIDE_1: tl.constexpr,
    DILATION_1: tl.constexpr,
):
    block_output_channels: tl.constexpr = 64
    block_output_width: tl.constexpr = 32
    block_reduction: tl.constexpr = 32
    channel_tiles: tl.constexpr = tl.cdiv(
        OUTPUT_CHANNELS, block_output_channels
    )
    width_tiles: tl.constexpr = tl.cdiv(
        OUTPUT_DIM_4, block_output_width
    )
    tiles_per_row: tl.constexpr = channel_tiles * width_tiles
    tiles_per_batch: tl.constexpr = OUTPUT_DIM_3 * tiles_per_row
    total_tasks: tl.constexpr = INPUT_DIM_0 * tiles_per_batch

    task = tl.program_id(0).to(tl.int64)
    task_stride = tl.num_programs(0)
    while task < total_tasks:
        batch = task // tiles_per_batch
        batch_tile = task % tiles_per_batch
        output_h = batch_tile // tiles_per_row
        row_tile = batch_tile % tiles_per_row
        channel_tile = row_tile // width_tiles
        width_tile = row_tile % width_tiles
        output_channel_start = (
            channel_tile * block_output_channels
        ).to(tl.int32)
        output_width_start = (
            width_tile * block_output_width
        ).to(tl.int32)
        accumulator = tl.zeros(
            (block_output_channels, block_output_width), dtype=tl.float32
        )

        for filter_h in range(0, 3):
            input_h = (
                output_h * CONV_STRIDE_1
                - PRE_PADDING_1
                + filter_h * DILATION_1
            )
            valid_input_h = (input_h >= 0) & (input_h < INPUT_DIM_3)
            safe_input_h = tl.where(valid_input_h, input_h, 0)
            for filter_w in range(0, 3):
                input_width_start = (
                    output_width_start - PRE_PADDING_2 + filter_w
                )
                for reduction_start in range(
                    0, CHANNELS_PER_GROUP, block_reduction
                ):
                    input_block = tl.make_block_ptr(
                        base=(
                            input_ptr
                            + batch * INPUT_STRIDE_0
                            + safe_input_h * INPUT_STRIDE_3
                        ),
                        shape=(CHANNELS_PER_GROUP, INPUT_DIM_4),
                        strides=(INPUT_STRIDE_1, INPUT_STRIDE_4),
                        offsets=(reduction_start, input_width_start),
                        block_shape=(block_reduction, block_output_width),
                        order=(1, 0),
                    )
                    input_values = tl.load(
                        input_block,
                        boundary_check=(0, 1),
                        padding_option="zero",
                    )
                    input_values = tl.where(
                        valid_input_h, input_values, 0.0
                    )
                    weight_block = tl.make_block_ptr(
                        base=(
                            filter_ptr
                            + filter_h * FILTER_STRIDE_3
                            + filter_w * FILTER_STRIDE_4
                        ),
                        shape=(OUTPUT_CHANNELS, CHANNELS_PER_GROUP),
                        strides=(FILTER_STRIDE_0, FILTER_STRIDE_1),
                        offsets=(output_channel_start, reduction_start),
                        block_shape=(
                            block_output_channels,
                            block_reduction,
                        ),
                        order=(1, 0),
                    )
                    weights = tl.load(
                        weight_block,
                        boundary_check=(0, 1),
                        padding_option="zero",
                    )
                    accumulator += tl.dot(
                        weights,
                        input_values,
                        input_precision="ieee",
                    )

        output_block = tl.make_block_ptr(
            base=(
                output_ptr
                + batch * OUTPUT_STRIDE_0
                + output_h * OUTPUT_STRIDE_3
            ),
            shape=(OUTPUT_CHANNELS, OUTPUT_DIM_4),
            strides=(OUTPUT_STRIDE_1, OUTPUT_STRIDE_4),
            offsets=(output_channel_start, output_width_start),
            block_shape=(block_output_channels, block_output_width),
            order=(1, 0),
        )
        tl.store(
            output_block,
            accumulator.to(output_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )
        task += task_stride


@triton.jit
def convolution_fprop_persistent_kernel(
    input_ptr,
    filter_ptr,
    output_ptr,
    n_elements,
    SPATIAL_RANK: tl.constexpr,
    GROUPS: tl.constexpr,
    INPUT_CHANNELS: tl.constexpr,
    OUTPUT_CHANNELS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    INPUT_DIM_0: tl.constexpr,
    INPUT_DIM_1: tl.constexpr,
    INPUT_DIM_2: tl.constexpr,
    INPUT_DIM_3: tl.constexpr,
    INPUT_DIM_4: tl.constexpr,
    INPUT_STRIDE_0: tl.constexpr,
    INPUT_STRIDE_1: tl.constexpr,
    INPUT_STRIDE_2: tl.constexpr,
    INPUT_STRIDE_3: tl.constexpr,
    INPUT_STRIDE_4: tl.constexpr,
    FILTER_DIM_0: tl.constexpr,
    FILTER_DIM_1: tl.constexpr,
    FILTER_DIM_2: tl.constexpr,
    FILTER_DIM_3: tl.constexpr,
    FILTER_DIM_4: tl.constexpr,
    FILTER_STRIDE_0: tl.constexpr,
    FILTER_STRIDE_1: tl.constexpr,
    FILTER_STRIDE_2: tl.constexpr,
    FILTER_STRIDE_3: tl.constexpr,
    FILTER_STRIDE_4: tl.constexpr,
    OUTPUT_DIM_0: tl.constexpr,
    OUTPUT_DIM_1: tl.constexpr,
    OUTPUT_DIM_2: tl.constexpr,
    OUTPUT_DIM_3: tl.constexpr,
    OUTPUT_DIM_4: tl.constexpr,
    OUTPUT_STRIDE_0: tl.constexpr,
    OUTPUT_STRIDE_1: tl.constexpr,
    OUTPUT_STRIDE_2: tl.constexpr,
    OUTPUT_STRIDE_3: tl.constexpr,
    OUTPUT_STRIDE_4: tl.constexpr,
    PRE_PADDING_0: tl.constexpr,
    PRE_PADDING_1: tl.constexpr,
    PRE_PADDING_2: tl.constexpr,
    POST_PADDING_0: tl.constexpr,
    POST_PADDING_1: tl.constexpr,
    POST_PADDING_2: tl.constexpr,
    CONV_STRIDE_0: tl.constexpr,
    CONV_STRIDE_1: tl.constexpr,
    CONV_STRIDE_2: tl.constexpr,
    DILATION_0: tl.constexpr,
    DILATION_1: tl.constexpr,
    DILATION_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WORKER_COUNT: tl.constexpr,
):
    if SPATIAL_RANK == 1:
        _convolution_fprop_1d_gemm(
            input_ptr,
            filter_ptr,
            output_ptr,
            GROUPS,
            OUTPUT_CHANNELS,
            CHANNELS_PER_GROUP,
            INPUT_DIM_0,
            INPUT_DIM_4,
            INPUT_STRIDE_0,
            INPUT_STRIDE_1,
            INPUT_STRIDE_4,
            FILTER_DIM_4,
            FILTER_STRIDE_0,
            FILTER_STRIDE_1,
            FILTER_STRIDE_4,
            OUTPUT_DIM_4,
            OUTPUT_STRIDE_0,
            OUTPUT_STRIDE_1,
            OUTPUT_STRIDE_4,
            PRE_PADDING_2,
            CONV_STRIDE_2,
            DILATION_2,
            BLOCK_SIZE,
        )
        return

    dense_nchw_2d: tl.constexpr = (
        SPATIAL_RANK == 2
        and INPUT_STRIDE_4 == 1
        and INPUT_STRIDE_3 == INPUT_DIM_4
        and INPUT_STRIDE_1 == INPUT_DIM_3 * INPUT_STRIDE_3
        and INPUT_STRIDE_0 == INPUT_CHANNELS * INPUT_STRIDE_1
        and FILTER_STRIDE_4 == 1
        and FILTER_STRIDE_3 == FILTER_DIM_4
        and FILTER_STRIDE_1 == FILTER_DIM_3 * FILTER_STRIDE_3
        and FILTER_STRIDE_0 == CHANNELS_PER_GROUP * FILTER_STRIDE_1
        and OUTPUT_STRIDE_4 == 1
        and OUTPUT_STRIDE_3 == OUTPUT_DIM_4
        and OUTPUT_STRIDE_1 == OUTPUT_DIM_3 * OUTPUT_STRIDE_3
        and OUTPUT_STRIDE_0 == OUTPUT_CHANNELS * OUTPUT_STRIDE_1
    )
    if dense_nchw_2d:
        if (
            FILTER_DIM_3 == 1
            and FILTER_DIM_4 == 1
            and PRE_PADDING_1 == 0
            and PRE_PADDING_2 == 0
            and POST_PADDING_1 == 0
            and POST_PADDING_2 == 0
            and CONV_STRIDE_1 == 1
            and CONV_STRIDE_2 == 1
            and OUTPUT_DIM_3 == INPUT_DIM_3
            and OUTPUT_DIM_4 == INPUT_DIM_4
        ):
            _convolution_fprop_2d_nchw_1x1(
                input_ptr,
                filter_ptr,
                output_ptr,
                GROUPS,
                OUTPUT_CHANNELS,
                CHANNELS_PER_GROUP,
                INPUT_DIM_0,
                INPUT_DIM_3,
                INPUT_DIM_4,
                INPUT_STRIDE_0,
                INPUT_STRIDE_1,
                FILTER_STRIDE_0,
                FILTER_STRIDE_1,
                OUTPUT_STRIDE_0,
                OUTPUT_STRIDE_1,
                BLOCK_SIZE,
            )
            return
        if (
            GROUPS == 1
            and FILTER_DIM_3 == 3
            and FILTER_DIM_4 == 3
            and CHANNELS_PER_GROUP <= 32
            and CONV_STRIDE_2 == 1
            and DILATION_2 == 1
            # The blocked implementation uses signed-i32 block-pointer
            # coordinates. Padded output lanes may extend beyond the logical
            # input, so include the two filter positions and 31-lane block
            # tail before selecting it; the tiled fallback keeps i64 pointer
            # arithmetic for extreme shapes.
            and OUTPUT_DIM_4 - PRE_PADDING_2 + 32 <= 2147483647
        ):
            _convolution_fprop_2d_nchw_3x3_blocked(
                input_ptr,
                filter_ptr,
                output_ptr,
                OUTPUT_CHANNELS,
                CHANNELS_PER_GROUP,
                INPUT_DIM_0,
                INPUT_DIM_3,
                INPUT_DIM_4,
                INPUT_STRIDE_0,
                INPUT_STRIDE_1,
                INPUT_STRIDE_3,
                INPUT_STRIDE_4,
                FILTER_STRIDE_0,
                FILTER_STRIDE_1,
                FILTER_STRIDE_3,
                FILTER_STRIDE_4,
                OUTPUT_DIM_3,
                OUTPUT_DIM_4,
                OUTPUT_STRIDE_0,
                OUTPUT_STRIDE_1,
                OUTPUT_STRIDE_3,
                OUTPUT_STRIDE_4,
                PRE_PADDING_1,
                PRE_PADDING_2,
                CONV_STRIDE_1,
                DILATION_1,
            )
            return
        _convolution_fprop_2d_nchw_tiled(
            input_ptr,
            filter_ptr,
            output_ptr,
            GROUPS,
            OUTPUT_CHANNELS,
            CHANNELS_PER_GROUP,
            INPUT_DIM_0,
            INPUT_DIM_3,
            INPUT_DIM_4,
            INPUT_STRIDE_0,
            INPUT_STRIDE_1,
            INPUT_STRIDE_3,
            INPUT_STRIDE_4,
            FILTER_DIM_3,
            FILTER_DIM_4,
            FILTER_STRIDE_0,
            FILTER_STRIDE_1,
            FILTER_STRIDE_3,
            FILTER_STRIDE_4,
            OUTPUT_DIM_3,
            OUTPUT_DIM_4,
            OUTPUT_STRIDE_0,
            OUTPUT_STRIDE_1,
            OUTPUT_STRIDE_3,
            OUTPUT_STRIDE_4,
            PRE_PADDING_1,
            PRE_PADDING_2,
            CONV_STRIDE_1,
            CONV_STRIDE_2,
            DILATION_1,
            DILATION_2,
            BLOCK_SIZE,
            WORKER_COUNT,
        )
        return

    program = tl.program_id(0).to(tl.int64)
    tile_count = tl.cdiv(n_elements.to(tl.int64), BLOCK_SIZE)
    tile = program
    while tile < tile_count:
        logical = (
            tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        )
        logical_mask = logical < n_elements
        remaining = logical.to(tl.int64)
        output_w = remaining % OUTPUT_DIM_4
        remaining //= OUTPUT_DIM_4
        output_h = remaining % OUTPUT_DIM_3
        remaining //= OUTPUT_DIM_3
        output_d = remaining % OUTPUT_DIM_2
        remaining //= OUTPUT_DIM_2
        output_channel = remaining % OUTPUT_DIM_1
        output_batch = remaining // OUTPUT_DIM_1
        safe_output_batch = tl.where(logical_mask, output_batch, 0)

        output_channels_per_group = OUTPUT_CHANNELS // GROUPS
        group = output_channel // output_channels_per_group
        input_channel_base = group * CHANNELS_PER_GROUP
        accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for channel_offset in range(0, CHANNELS_PER_GROUP):
            input_channel = input_channel_base + channel_offset
            for filter_d in range(0, FILTER_DIM_2):
                input_d = (
                    output_d * CONV_STRIDE_0
                    - PRE_PADDING_0
                    + filter_d * DILATION_0
                )
                for filter_h in range(0, FILTER_DIM_3):
                    input_h = (
                        output_h * CONV_STRIDE_1
                        - PRE_PADDING_1
                        + filter_h * DILATION_1
                    )
                    for filter_w in range(0, FILTER_DIM_4):
                        input_w = (
                            output_w * CONV_STRIDE_2
                            - PRE_PADDING_2
                            + filter_w * DILATION_2
                        )
                        valid_d = (input_d >= 0) & (input_d < INPUT_DIM_2)
                        valid_h = (input_h >= 0) & (input_h < INPUT_DIM_3)
                        valid_w = (input_w >= 0) & (input_w < INPUT_DIM_4)
                        input_mask = (
                            logical_mask & valid_d & valid_h & valid_w
                        )
                        # Ascend may still evaluate masked pointer arithmetic.
                        # Keep every lane's address within its allocation and
                        # use input_mask solely to provide padding zeros.
                        safe_input_d = tl.where(valid_d, input_d, 0)
                        safe_input_h = tl.where(valid_h, input_h, 0)
                        safe_input_w = tl.where(valid_w, input_w, 0)
                        input_offset = (
                            safe_output_batch * INPUT_STRIDE_0
                            + input_channel * INPUT_STRIDE_1
                            + safe_input_d * INPUT_STRIDE_2
                            + safe_input_h * INPUT_STRIDE_3
                            + safe_input_w * INPUT_STRIDE_4
                        )
                        filter_offset = (
                            output_channel * FILTER_STRIDE_0
                            + channel_offset * FILTER_STRIDE_1
                            + filter_d * FILTER_STRIDE_2
                            + filter_h * FILTER_STRIDE_3
                            + filter_w * FILTER_STRIDE_4
                        )
                        input_value = tl.load(
                            input_ptr + input_offset,
                            mask=input_mask,
                            other=0.0,
                        ).to(tl.float32)
                        filter_value = tl.load(
                            filter_ptr + filter_offset,
                            mask=logical_mask,
                            other=0.0,
                        ).to(tl.float32)
                        accumulator += input_value * filter_value

        output_offset = (
            safe_output_batch * OUTPUT_STRIDE_0
            + output_channel * OUTPUT_STRIDE_1
            + output_d * OUTPUT_STRIDE_2
            + output_h * OUTPUT_STRIDE_3
            + output_w * OUTPUT_STRIDE_4
        )
        tl.store(
            output_ptr + output_offset,
            accumulator.to(output_ptr.dtype.element_ty),
            mask=logical_mask,
        )
        tile += WORKER_COUNT
