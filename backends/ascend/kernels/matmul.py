"""Ascend kernels for matmul."""

import triton
import triton.language as tl
import triton as tr

# Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0


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
    wide_tiles: tl.constexpr = M >= BLOCK_SIZE and N >= BLOCK_SIZE * 2 and K >= 512
    small_k_wide_n: tl.constexpr = K <= 64 and M <= BLOCK_SIZE and N >= BLOCK_SIZE * 2
    # Keep the externally attested tuning configuration unchanged while
    # avoiding mostly masked 128-wide tiles for genuinely tiny matrices.
    tiny_matrix: tl.constexpr = M <= 64 and N <= 64 and K <= 128
    # RGB stem im2col produces K=27 and a very wide spatial dimension. A wider
    # N tile halves the persistent task waves without enlarging M past 64.
    very_small_k_very_wide_n: tl.constexpr = K <= 32 and M <= 64 and N >= BLOCK_SIZE * 8
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
        BLOCK_SIZE == 128 and M >= 512 and N >= 256 and N < 512 and K >= 512
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
        INPUT_IS_FLOAT32 and BATCH == 16 and M == 1024 and N == 1024 and K == 1024
    )
    transposed_small_k: tl.constexpr = K < 512 and A_STRIDE_M == 1 and B_STRIDE_K == 1
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
                    else (32 if M <= 32 else (64 if M <= 64 else BLOCK_SIZE))
                )
                if small_k_wide_n
                else (
                    BLOCK_SIZE
                    if medium_n_tiles
                    else (
                        (16 if M <= 16 else (32 if M <= 32 else BLOCK_SIZE // 2))
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
                if (fp32_mid_tiles or medium_n_tiles or batched_medium_n_tiles)
                else (
                    BLOCK_SIZE * 2
                    if (wide_tiles or small_k_wide_n or tiny_m_wide_n_tiles)
                    else (BLOCK_SIZE // 2 if narrow_n_tiles else BLOCK_SIZE)
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
                                if (K >= 512 or narrow_n_tiles or narrow_m_tiles)
                                else 64
                            )
                            if INPUT_IS_FLOAT32
                            else (128 if medium_n_tiles and K % 512 != 0 else 256)
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
    scheduled_tiles_m: tl.constexpr = tiles_m // reuse_m_tiles
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
        INPUT_IS_FLOAT32 and BATCH == 16 and M == 1024 and N == 1024 and K == 1024
    )
    schedule_group_m: tl.constexpr = (
        4
        if fp32_residual_group
        else (2 if GROUP_M == 8 or convolution_wide_n_group else GROUP_M)
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
                tile_m = group * schedule_group_m + tile_in_group % schedule_group_m
                tile_n = tile_in_group // schedule_group_m
            else:
                tiles_per_group = schedule_group_m * tiles_n
                group = tile // tiles_per_group
                first_tile_m = group * schedule_group_m
                group_m = tl.minimum(scheduled_tiles_m - first_tile_m, schedule_group_m)
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
            M % block_m == 0 and N % block_n == 0 and K % reduction_block == 0
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
            next_a_tile_ptr = tl.advance(a_tile_ptr, (0, reduction_block))
            next_b_tile_ptr = tl.advance(b_tile_ptr, (reduction_block, 0))
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
                        (physical_tile_m.to(tl.int32) + reused_tile) * block_m,
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
                reused_accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
                reused_accumulator = tl.dot(reused_a, b, reused_accumulator)
                reused_accumulator = tl.dot(reused_next_a, next_b, reused_accumulator)
                reused_output_tile_ptr = tl.make_block_ptr(
                    base=output_ptr + c_batch_offset,
                    shape=(M, N),
                    strides=(C_STRIDE_M, C_STRIDE_N),
                    offsets=(
                        (physical_tile_m.to(tl.int32) + reused_tile) * block_m,
                        tile_n.to(tl.int32) * block_n,
                    ),
                    block_shape=(block_m, block_n),
                    order=(1, 0),
                )
                tl.store(
                    reused_output_tile_ptr,
                    reused_accumulator.to(output_ptr.dtype.element_ty),
                )
        else:
            if reduction_tiles % 2 == 0:
                for _ in range(0, K, reduction_block * 2):
                    next_a_tile_ptr = tl.advance(a_tile_ptr, (0, reduction_block))
                    next_b_tile_ptr = tl.advance(b_tile_ptr, (reduction_block, 0))
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
                        accumulator = tl.dot(next_a, next_b, accumulator)
                    a_tile_ptr = tl.advance(a_tile_ptr, (0, reduction_block * 2))
                    b_tile_ptr = tl.advance(b_tile_ptr, (reduction_block * 2, 0))
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
                    a_tile_ptr = tl.advance(a_tile_ptr, (0, reduction_block))
                    b_tile_ptr = tl.advance(b_tile_ptr, (reduction_block, 0))

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


# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0


@triton.jit
def _tf32_rne(x, NATIVE: tl.constexpr = False):
    """Software rounding helper retained by the shared launch signature.

    Ascend dispatch selects IEEE input precision and rejects explicit TF32.
    """
    bits = x.to(tl.uint32, bitcast=True)
    rounded = (bits + 0xFFF + ((bits >> 13) & 1)) & 0xFFFFE000
    result = tl.where((bits & 0x7F800000) == 0x7F800000, bits, rounded)
    return result.to(tl.float32, bitcast=True)


@triton.jit
def matmul_tiled_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
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
    USE_TF32: tl.constexpr,
    NATIVE_TF32_RNE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    tile = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    tiles_m = tl.cdiv(M, BLOCK_M)
    tiles_n = tl.cdiv(N, BLOCK_N)
    if GROUP_M == 1:
        tile_m = tile // tiles_n
        tile_n = tile % tiles_n
    elif GROUP_M >= tiles_m:
        tile_m = tile % tiles_m
        tile_n = tile // tiles_m
    else:
        tiles_per_group = GROUP_M * tiles_n
        group = tile // tiles_per_group
        first_tile_m = group * GROUP_M
        group_m = tl.minimum(tiles_m - first_tile_m, GROUP_M)
        tile_in_group = tile % tiles_per_group
        tile_m = first_tile_m + tile_in_group % group_m
        tile_n = tile_in_group // group_m

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

    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    reduction = tl.arange(0, BLOCK_K)
    a_tile_ptrs = (
        a_ptr
        + a_batch_offset
        + rows[:, None] * A_STRIDE_M
        + reduction[None, :] * A_STRIDE_K
    )
    b_tile_ptrs = (
        b_ptr
        + b_batch_offset
        + reduction[:, None] * B_STRIDE_K
        + columns[None, :] * B_STRIDE_N
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for reduction_start in range(0, K, BLOCK_K):
        reduction_offsets = reduction_start + reduction
        a = tl.load(
            a_tile_ptrs,
            mask=(rows[:, None] < M) & (reduction_offsets[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_tile_ptrs,
            mask=(reduction_offsets[:, None] < K) & (columns[None, :] < N),
            other=0.0,
        )
        if INPUT_IS_FLOAT32 and USE_TF32:
            a, b = _tf32_rne(a, NATIVE_TF32_RNE), _tf32_rne(b, NATIVE_TF32_RNE)
            accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
        elif INPUT_IS_FLOAT32:
            accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
        else:
            accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
        a_tile_ptrs += BLOCK_K * A_STRIDE_K
        b_tile_ptrs += BLOCK_K * B_STRIDE_K

    tl.store(
        c_ptr
        + c_batch_offset
        + rows[:, None] * C_STRIDE_M
        + columns[None, :] * C_STRIDE_N,
        accumulator.to(c_ptr.dtype.element_ty),
        mask=(rows[:, None] < M) & (columns[None, :] < N),
    )


# Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0


@tr.jit
def moe_matmul_kernel(
    token_ptr,
    matrix_ptr,
    offsets_ptr,
    index_ptr,
    ks_ptr,
    output_ptr,
    EXPERTS: tl.constexpr,
    TOKENS: tl.constexpr,
    ROUTED: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    MODE: tl.constexpr,
    TOP_K: tl.constexpr,
    TM: tl.constexpr,
    TK: tl.constexpr,
    ME: tl.constexpr,
    MK: tl.constexpr,
    MN: tl.constexpr,
    OE: tl.constexpr,
    OM: tl.constexpr,
    ON: tl.constexpr,
    OS: tl.constexpr,
    IS: tl.constexpr,
    KS: tl.constexpr,
    BACKWARD: tl.constexpr,
):
    tile = tl.program_id(0).to(tl.int32)
    tiles_n: tl.constexpr = tr.cdiv(N, 32)
    tiles_m: tl.constexpr = tr.cdiv(K if BACKWARD else ROUTED, 16)
    expert = tile // (tiles_m * tiles_n)
    row = tile // tiles_n % tiles_m * 16 + tl.arange(0, 16)
    col = tile % tiles_n * 32 + tl.arange(0, 32)
    # Guard every address even for invalid routing metadata. Valid
    # callers supply
    # monotone offsets beginning at zero and a permutation for scatter slots.
    begin = tl.minimum(
        tl.maximum(tl.load(offsets_ptr + expert * OS).to(tl.int32), 0), ROUTED
    )
    end = tl.minimum(
        tl.maximum(
            tl.load(offsets_ptr + (expert + 1) * OS, expert + 1 < EXPERTS, ROUTED).to(
                tl.int32
            ),
            begin,
        ),
        ROUTED,
    )
    if BACKWARD:
        accumulator = tl.zeros((16, 32), tl.float32)
        # Explicitly guard empty experts: the Ascend dot pipeline must not
        # enter a zero-trip dynamic loop with an uninitialized accumulator.
        if end > begin:
            for start in range(begin, end, 32):
                reduction = start + tl.arange(0, 32)
                x = tl.load(
                    token_ptr + reduction[None, :] * TM + row[:, None] * TK,
                    (reduction[None, :] < end) & (row[:, None] < K),
                    0.0,
                )
                dy = tl.load(
                    matrix_ptr + reduction[:, None] * MK + col[None, :] * MN,
                    (reduction[:, None] < end) & (col[None, :] < N),
                    0.0,
                )
                accumulator = tl.dot(x, dy, accumulator)
        tl.store(
            output_ptr + expert * OE + row[:, None] * OM + col[None, :] * ON,
            accumulator,
            (row[:, None] < K) & (col[None, :] < N),
        )
    else:
        # Every program follows the same Cube pipeline; inactive rows are
        # masked at loads/stores instead of bypassing the dot operation.
        slot = begin + row
        valid = slot < end
        if MODE == 1:
            source = tl.load(index_ptr + slot * IS, valid, -1).to(tl.int32)
            valid = valid & (source >= 0) & (source < TOKENS)
        else:
            source = slot
        accumulator = tl.zeros((16, 32), tl.float32)
        for start in range(tr.cdiv(K, 32)):
            reduction = start * 32 + tl.arange(0, 32)
            offsets = source[:, None] * TM + reduction[None, :] * TK
            mask = valid[:, None] & (reduction[None, :] < K)
            x = tl.reshape(
                tl.load(
                    token_ptr + tl.reshape(offsets, (16 * 32,)),
                    tl.reshape(mask, (16 * 32,)),
                    0.0,
                ),
                (16, 32),
            )
            w = tl.load(
                matrix_ptr + expert * ME + reduction[:, None] * MK + col[None, :] * MN,
                (reduction[:, None] < K) & (col[None, :] < N),
                0.0,
            )
            accumulator = tl.dot(x, w, accumulator)
        if MODE == 2:
            token = tl.load(index_ptr + slot * IS, valid, -1).to(tl.int32)
            rank = tl.load(ks_ptr + slot * KS, valid, -1).to(tl.int32)
            destination = token * TOP_K + rank
            valid = valid & (
                (token >= 0) & (token < ROUTED // TOP_K) & (rank >= 0) & (rank < TOP_K)
            )
        else:
            destination = slot
        tl.store(
            output_ptr + destination[:, None] * OM + col[None, :] * ON,
            accumulator,
            valid[:, None] & (col[None, :] < N),
        )


@tr.jit
def _compensated_add(a, ae, b, be):
    total = a + b
    recovered = total - a
    error = (a - (total - recovered)) + (b - recovered)
    return total, error + ae + be


@tr.jit
def moe_matmul_forward_kernel(
    token_ptr,
    matrix_ptr,
    offsets_ptr,
    index_ptr,
    ks_ptr,
    output_ptr,
    EXPERTS: tl.constexpr,
    TOKENS: tl.constexpr,
    ROUTED: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    MODE: tl.constexpr,
    TOP_K: tl.constexpr,
    TM: tl.constexpr,
    TK: tl.constexpr,
    ME: tl.constexpr,
    MK: tl.constexpr,
    MN: tl.constexpr,
    OE: tl.constexpr,
    OM: tl.constexpr,
    ON: tl.constexpr,
    OS: tl.constexpr,
    IS: tl.constexpr,
    KS: tl.constexpr,
    BACKWARD: tl.constexpr,
):
    # A row owns its complete output. Scalar routing avoids irregular Cube
    # gathers, which lose rows with the current Ascend compiler.
    program = tl.program_id(0)
    blocks_n: tl.constexpr = tr.cdiv(N, 16)
    slot = program // blocks_n
    col = program % blocks_n * 16 + tl.arange(0, 16)
    expert = tl.full((), 0, tl.int32)
    for e in range(1, EXPERTS):
        boundary = tl.load(offsets_ptr + e * OS)
        expert = tl.where(slot >= boundary, e, expert)
    if MODE == 1:
        source = tl.load(index_ptr + slot * IS)
    else:
        source = slot
    valid = (source >= 0) & (source < TOKENS)
    acc = tl.full((16,), 0.0, tl.float32)
    correction = tl.full((16,), 0.0, tl.float32)
    for k in range(K):
        x = tl.load(token_ptr + source * TM + k * TK, valid, 0).to(tl.float32)
        w = tl.load(matrix_ptr + expert * ME + k * MK + col * MN, col < N, 0).to(
            tl.float32
        )
        # Preserve exact FP32 products of FP16/BF16 inputs and the residual
        # from each addition, including values close to output rounding ties.
        acc, correction = _compensated_add(acc, correction, x * w, 0.0)
    if MODE == 2:
        token = tl.load(index_ptr + slot * IS)
        rank = tl.load(ks_ptr + slot * KS)
        destination = token * TOP_K + rank
        valid = (
            valid
            & (token >= 0)
            & (token < ROUTED // TOP_K)
            & (rank >= 0)
            & (rank < TOP_K)
        )
    else:
        destination = slot
    tl.store(
        output_ptr + destination * OM + col * ON, acc + correction, valid & (col < N)
    )
