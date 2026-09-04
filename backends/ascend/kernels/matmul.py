# Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0

"""Ascend-owned strided and broadcasted MatMul kernel."""

import triton
import triton.language as tl


@triton.jit
def matmul_strided_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    n_elements,
    BATCH: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    DIM_0: tl.constexpr,
    DIM_1: tl.constexpr,
    DIM_2: tl.constexpr,
    DIM_3: tl.constexpr,
    DIM_4: tl.constexpr,
    DIM_5: tl.constexpr,
    A_BATCH_STRIDE_0: tl.constexpr,
    A_BATCH_STRIDE_1: tl.constexpr,
    A_BATCH_STRIDE_2: tl.constexpr,
    A_BATCH_STRIDE_3: tl.constexpr,
    A_BATCH_STRIDE_4: tl.constexpr,
    A_BATCH_STRIDE_5: tl.constexpr,
    B_BATCH_STRIDE_0: tl.constexpr,
    B_BATCH_STRIDE_1: tl.constexpr,
    B_BATCH_STRIDE_2: tl.constexpr,
    B_BATCH_STRIDE_3: tl.constexpr,
    B_BATCH_STRIDE_4: tl.constexpr,
    B_BATCH_STRIDE_5: tl.constexpr,
    C_BATCH_STRIDE_0: tl.constexpr,
    C_BATCH_STRIDE_1: tl.constexpr,
    C_BATCH_STRIDE_2: tl.constexpr,
    C_BATCH_STRIDE_3: tl.constexpr,
    C_BATCH_STRIDE_4: tl.constexpr,
    C_BATCH_STRIDE_5: tl.constexpr,
    A_STRIDE_M: tl.constexpr,
    A_STRIDE_K: tl.constexpr,
    B_STRIDE_K: tl.constexpr,
    B_STRIDE_N: tl.constexpr,
    C_STRIDE_M: tl.constexpr,
    C_STRIDE_N: tl.constexpr,
    INPUT_IS_FLOAT32: tl.constexpr,
    GROUP_M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WORKER_COUNT: tl.constexpr,
):
    wide_tiles: tl.constexpr = (
        M >= BLOCK_SIZE
        and N >= BLOCK_SIZE * 2
        and K >= 512
    )
    small_k_wide_n: tl.constexpr = (
        K <= 64 and M <= BLOCK_SIZE and N >= BLOCK_SIZE * 2
    )
    # Keep the externally attested tuning configuration unchanged while
    # avoiding mostly masked 128-wide tiles for genuinely tiny matrices.
    tiny_matrix: tl.constexpr = M <= 64 and N <= 64 and K <= 128
    # RGB stem im2col produces K=27 and a very wide spatial dimension. A wider
    # N tile halves the persistent task waves without enlarging M past 64.
    very_small_k_very_wide_n: tl.constexpr = (
        K <= 32 and M <= 64 and N >= BLOCK_SIZE * 8
    )
    narrow_n_tiles: tl.constexpr = BLOCK_SIZE == 128 and N <= 64
    narrow_m_tiles: tl.constexpr = (
        BLOCK_SIZE == 128 and M <= 64 and N >= 128 and K >= 128
    )
    tiny_m_wide_n_tiles: tl.constexpr = (
        BLOCK_SIZE == 128
        and BATCH >= 2
        and M <= 16
        and N >= 2048
        and K >= 128
        and K < 512
    )
    medium_n_tiles: tl.constexpr = (
        BLOCK_SIZE == 128
        and M >= 512
        and N >= 256
        and N < 512
        and K >= 512
    )
    batched_medium_n_tiles: tl.constexpr = (
        BLOCK_SIZE == 128
        and BATCH >= 8
        and M <= 128
        and N >= 512
        and N < 1024
        and K >= 512
        and K < 1024
    )
    fp32_mid_tiles: tl.constexpr = (
        INPUT_IS_FLOAT32
        and M >= 512
        and M <= 1024
        and N >= 512
        and N <= 1024
        and K >= 512
        and K <= 1024
    )
    fp32_tall_residual: tl.constexpr = (
        INPUT_IS_FLOAT32
        and BATCH == 16
        and M == 1024
        and N == 1024
        and K == 1024
    )
    transposed_small_k: tl.constexpr = (
        K < 512 and A_STRIDE_M == 1 and B_STRIDE_K == 1
    )
    block_m: tl.constexpr = (
        (16 if M <= 16 else (32 if M <= 32 else 64))
        if tiny_matrix
        else (
            BLOCK_SIZE * 2
            if fp32_tall_residual
            else (
                (
                    16
                    if M <= 16
                    else (
                        32
                        if M <= 32
                        else (64 if M <= 64 else BLOCK_SIZE)
                    )
                )
                if small_k_wide_n
                else (
                    BLOCK_SIZE
                    if medium_n_tiles
                    else (
                        (
                            16
                            if M <= 16
                            else (32 if M <= 32 else BLOCK_SIZE // 2)
                        )
                        if narrow_m_tiles
                        else BLOCK_SIZE
                    )
                )
            )
        )
    )
    block_n: tl.constexpr = (
        (16 if N <= 16 else (32 if N <= 32 else 64))
        if tiny_matrix
        else (
            BLOCK_SIZE * 4
            if very_small_k_very_wide_n
            else (
                BLOCK_SIZE
                if (
                    fp32_mid_tiles
                    or medium_n_tiles
                    or batched_medium_n_tiles
                )
                else (
                    BLOCK_SIZE * 2
                    if (
                        wide_tiles
                        or small_k_wide_n
                        or tiny_m_wide_n_tiles
                    )
                    else (
                        BLOCK_SIZE // 2 if narrow_n_tiles else BLOCK_SIZE
                    )
                )
            )
        )
    )
    reduction_block: tl.constexpr = (
        (32 if K <= 32 else (64 if K <= 64 else 128))
        if tiny_matrix
        else (
            (32 if K <= 32 else 64)
            if small_k_wide_n
            else (
                (64 if K % 64 == 0 else 32)
                if transposed_small_k
                else (
                    128
                    if fp32_tall_residual
                    else (
                        256
                        if fp32_mid_tiles
                        else (
                            (
                                128
                                if (
                                    K >= 512
                                    or narrow_n_tiles
                                    or narrow_m_tiles
                                )
                                else 64
                            )
                            if INPUT_IS_FLOAT32
                            else (
                                128
                                if medium_n_tiles and K % 512 != 0
                                else 256
                            )
                        )
                    )
                )
            )
        )
    )
    tiles_m: tl.constexpr = (M + block_m - 1) // block_m
    tiles_n: tl.constexpr = (N + block_n - 1) // block_n
    reuse_m_tiles: tl.constexpr = (
        2
        if (
            not INPUT_IS_FLOAT32
            and K == 512
            and M % (block_m * 2) == 0
            and N % block_n == 0
            and A_STRIDE_K == 1
            and B_STRIDE_N == 1
            and C_STRIDE_N == 1
        )
        else 1
    )
    scheduled_tiles_m: tl.constexpr = (
        tiles_m // reuse_m_tiles
    )
    tiles_per_batch: tl.constexpr = scheduled_tiles_m * tiles_n
    total_tasks: tl.constexpr = BATCH * tiles_per_batch
    convolution_wide_n_group: tl.constexpr = (
        INPUT_IS_FLOAT32
        and BATCH == 1
        and M >= 256
        and M <= 512
        and N >= 1024
        and N <= 2048
        and K >= 512
        and K <= 2304
    )
    fp32_residual_group: tl.constexpr = (
        INPUT_IS_FLOAT32
        and BATCH == 16
        and M == 1024
        and N == 1024
        and K == 1024
    )
    schedule_group_m: tl.constexpr = (
        4
        if fp32_residual_group
        else (
            2
            if GROUP_M == 8 or convolution_wide_n_group
            else GROUP_M
        )
    )
    full_n_tiles: tl.constexpr = N // block_n
    tail_last_schedule: tl.constexpr = (
        scheduled_tiles_m == 1
        and N % block_n != 0
        and BATCH * full_n_tiles == WORKER_COUNT
    )
    task = tl.program_id(0).to(tl.int64)
    program_count = tl.num_programs(0).to(tl.int64)
    while task < total_tasks:
        if tail_last_schedule:
            tail_task = task - WORKER_COUNT
            full_batch = task // full_n_tiles
            is_tail = task >= WORKER_COUNT
            batch = tl.where(is_tail, tail_task, full_batch)
            tile_m = tl.zeros((), dtype=tl.int64)
            tile_n = tl.where(
                is_tail,
                full_n_tiles,
                task - full_batch * full_n_tiles,
            )
        else:
            batch = task // tiles_per_batch
            tile = task % tiles_per_batch
            if schedule_group_m == 1:
                tile_m = tile // tiles_n
                tile_n = tile % tiles_n
            elif schedule_group_m >= scheduled_tiles_m:
                tile_m = tile % scheduled_tiles_m
                tile_n = tile // scheduled_tiles_m
            elif scheduled_tiles_m % schedule_group_m == 0:
                tiles_per_group = schedule_group_m * tiles_n
                group = tile // tiles_per_group
                tile_in_group = tile % tiles_per_group
                tile_m = (
                    group * schedule_group_m
                    + tile_in_group % schedule_group_m
                )
                tile_n = tile_in_group // schedule_group_m
            else:
                tiles_per_group = schedule_group_m * tiles_n
                group = tile // tiles_per_group
                first_tile_m = group * schedule_group_m
                group_m = tl.minimum(
                    scheduled_tiles_m - first_tile_m, schedule_group_m
                )
                tile_in_group = tile % tiles_per_group
                tile_m = first_tile_m + tile_in_group % group_m
                tile_n = tile_in_group // group_m

        physical_tile_m = tile_m * reuse_m_tiles

        simple_batch: tl.constexpr = (
            DIM_0 == 1
            and DIM_1 == 1
            and DIM_2 == 1
            and DIM_3 == 1
            and DIM_4 == 1
            and DIM_5 == BATCH
        )
        if simple_batch:
            batch_i64 = batch.to(tl.int64)
            a_batch_offset = batch_i64 * A_BATCH_STRIDE_5
            b_batch_offset = batch_i64 * B_BATCH_STRIDE_5
            c_batch_offset = batch_i64 * C_BATCH_STRIDE_5
        else:
            remaining = batch
            a_batch_offset = tl.zeros((), dtype=tl.int64)
            b_batch_offset = tl.zeros((), dtype=tl.int64)
            c_batch_offset = tl.zeros((), dtype=tl.int64)
            coordinate = remaining % DIM_5
            remaining //= DIM_5
            a_batch_offset += coordinate * A_BATCH_STRIDE_5
            b_batch_offset += coordinate * B_BATCH_STRIDE_5
            c_batch_offset += coordinate * C_BATCH_STRIDE_5
            coordinate = remaining % DIM_4
            remaining //= DIM_4
            a_batch_offset += coordinate * A_BATCH_STRIDE_4
            b_batch_offset += coordinate * B_BATCH_STRIDE_4
            c_batch_offset += coordinate * C_BATCH_STRIDE_4
            coordinate = remaining % DIM_3
            remaining //= DIM_3
            a_batch_offset += coordinate * A_BATCH_STRIDE_3
            b_batch_offset += coordinate * B_BATCH_STRIDE_3
            c_batch_offset += coordinate * C_BATCH_STRIDE_3
            coordinate = remaining % DIM_2
            remaining //= DIM_2
            a_batch_offset += coordinate * A_BATCH_STRIDE_2
            b_batch_offset += coordinate * B_BATCH_STRIDE_2
            c_batch_offset += coordinate * C_BATCH_STRIDE_2
            coordinate = remaining % DIM_1
            remaining //= DIM_1
            a_batch_offset += coordinate * A_BATCH_STRIDE_1
            b_batch_offset += coordinate * B_BATCH_STRIDE_1
            c_batch_offset += coordinate * C_BATCH_STRIDE_1
            coordinate = remaining % DIM_0
            a_batch_offset += coordinate * A_BATCH_STRIDE_0
            b_batch_offset += coordinate * B_BATCH_STRIDE_0
            c_batch_offset += coordinate * C_BATCH_STRIDE_0

        a_tile_ptr = tl.make_block_ptr(
            base=a_ptr + a_batch_offset,
            shape=(M, K),
            strides=(A_STRIDE_M, A_STRIDE_K),
            offsets=(physical_tile_m.to(tl.int32) * block_m, 0),
            block_shape=(block_m, reduction_block),
            order=(0, 1) if A_STRIDE_M == 1 else (1, 0),
        )
        b_tile_ptr = tl.make_block_ptr(
            base=b_ptr + b_batch_offset,
            shape=(K, N),
            strides=(B_STRIDE_K, B_STRIDE_N),
            offsets=(0, tile_n.to(tl.int32) * block_n),
            block_shape=(reduction_block, block_n),
            order=(0, 1) if B_STRIDE_K == 1 else (1, 0),
        )
        accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
        reduction_tiles: tl.constexpr = (K + reduction_block - 1) // reduction_block
        full_input_tiles: tl.constexpr = (
            M % block_m == 0
            and N % block_n == 0
            and K % reduction_block == 0
        )
        output_tile_ptr = tl.make_block_ptr(
            base=output_ptr + c_batch_offset,
            shape=(M, N),
            strides=(C_STRIDE_M, C_STRIDE_N),
            offsets=(
                physical_tile_m.to(tl.int32) * block_m,
                tile_n.to(tl.int32) * block_n,
            ),
            block_shape=(block_m, block_n),
            order=(0, 1) if C_STRIDE_M == 1 else (1, 0),
        )
        if reuse_m_tiles > 1:
            next_a_tile_ptr = tl.advance(
                a_tile_ptr, (0, reduction_block)
            )
            next_b_tile_ptr = tl.advance(
                b_tile_ptr, (reduction_block, 0)
            )
            a = tl.load(a_tile_ptr)
            b = tl.load(b_tile_ptr)
            next_a = tl.load(next_a_tile_ptr)
            next_b = tl.load(next_b_tile_ptr)
            accumulator = tl.dot(a, b, accumulator)
            accumulator = tl.dot(next_a, next_b, accumulator)
            tl.store(
                output_tile_ptr,
                accumulator.to(output_ptr.dtype.element_ty),
            )

            for reused_tile in range(1, reuse_m_tiles):
                reused_a_tile_ptr = tl.make_block_ptr(
                    base=a_ptr + a_batch_offset,
                    shape=(M, K),
                    strides=(A_STRIDE_M, A_STRIDE_K),
                    offsets=(
                        (
                            physical_tile_m.to(tl.int32)
                            + reused_tile
                        )
                        * block_m,
                        0,
                    ),
                    block_shape=(block_m, reduction_block),
                    order=(1, 0),
                )
                reused_next_a_tile_ptr = tl.advance(
                    reused_a_tile_ptr, (0, reduction_block)
                )
                reused_a = tl.load(reused_a_tile_ptr)
                reused_next_a = tl.load(reused_next_a_tile_ptr)
                reused_accumulator = tl.zeros(
                    (block_m, block_n), dtype=tl.float32
                )
                reused_accumulator = tl.dot(
                    reused_a, b, reused_accumulator
                )
                reused_accumulator = tl.dot(
                    reused_next_a, next_b, reused_accumulator
                )
                reused_output_tile_ptr = tl.make_block_ptr(
                    base=output_ptr + c_batch_offset,
                    shape=(M, N),
                    strides=(C_STRIDE_M, C_STRIDE_N),
                    offsets=(
                        (
                            physical_tile_m.to(tl.int32)
                            + reused_tile
                        )
                        * block_m,
                        tile_n.to(tl.int32) * block_n,
                    ),
                    block_shape=(block_m, block_n),
                    order=(1, 0),
                )
                tl.store(
                    reused_output_tile_ptr,
                    reused_accumulator.to(
                        output_ptr.dtype.element_ty
                    ),
                )
        else:
            if reduction_tiles % 2 == 0:
                for _ in range(0, K, reduction_block * 2):
                    next_a_tile_ptr = tl.advance(
                        a_tile_ptr, (0, reduction_block)
                    )
                    next_b_tile_ptr = tl.advance(
                        b_tile_ptr, (reduction_block, 0)
                    )
                    if full_input_tiles:
                        a = tl.load(a_tile_ptr)
                        b = tl.load(b_tile_ptr)
                        next_a = tl.load(next_a_tile_ptr)
                        next_b = tl.load(next_b_tile_ptr)
                    else:
                        a = tl.load(
                            a_tile_ptr,
                            boundary_check=(0, 1),
                            padding_option="zero",
                        )
                        b = tl.load(
                            b_tile_ptr,
                            boundary_check=(0, 1),
                            padding_option="zero",
                        )
                        next_a = tl.load(
                            next_a_tile_ptr,
                            boundary_check=(0, 1),
                            padding_option="zero",
                        )
                        next_b = tl.load(
                            next_b_tile_ptr,
                            boundary_check=(0, 1),
                            padding_option="zero",
                        )
                    if INPUT_IS_FLOAT32:
                        accumulator = tl.dot(
                            a,
                            b,
                            accumulator,
                            input_precision="hf32",
                        )
                        accumulator = tl.dot(
                            next_a,
                            next_b,
                            accumulator,
                            input_precision="hf32",
                        )
                    else:
                        accumulator = tl.dot(a, b, accumulator)
                        accumulator = tl.dot(
                            next_a, next_b, accumulator
                        )
                    a_tile_ptr = tl.advance(
                        a_tile_ptr, (0, reduction_block * 2)
                    )
                    b_tile_ptr = tl.advance(
                        b_tile_ptr, (reduction_block * 2, 0)
                    )
            else:
                for _ in range(0, K, reduction_block):
                    if full_input_tiles:
                        a = tl.load(a_tile_ptr)
                        b = tl.load(b_tile_ptr)
                    else:
                        a = tl.load(
                            a_tile_ptr,
                            boundary_check=(0, 1),
                            padding_option="zero",
                        )
                        b = tl.load(
                            b_tile_ptr,
                            boundary_check=(0, 1),
                            padding_option="zero",
                        )
                    if INPUT_IS_FLOAT32:
                        accumulator = tl.dot(
                            a,
                            b,
                            accumulator,
                            input_precision="hf32",
                        )
                    else:
                        accumulator = tl.dot(a, b, accumulator)
                    a_tile_ptr = tl.advance(
                        a_tile_ptr, (0, reduction_block)
                    )
                    b_tile_ptr = tl.advance(
                        b_tile_ptr, (reduction_block, 0)
                    )

            if M % block_m == 0 and N % block_n == 0:
                tl.store(
                    output_tile_ptr,
                    accumulator.to(output_ptr.dtype.element_ty),
                )
            else:
                tl.store(
                    output_tile_ptr,
                    accumulator.to(output_ptr.dtype.element_ty),
                    boundary_check=(0, 1),
                )

        task += program_count
