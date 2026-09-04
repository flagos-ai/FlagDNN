# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""MTGPU-specialized descriptor and strided batched MatMul kernels."""

import triton
import triton.experimental.tle.language as tle
import triton.language as tl


@triton.jit
def matmul_descriptor_kernel(
    a_desc,
    b_desc,
    c_desc,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BATCH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Dense, non-broadcast MatMul lowered through MTGPU TME and SQMMA."""

    tile = tl.program_id(0)
    batch = tl.program_id(1) % BATCH
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

    a_row = (batch * M + tile_m * BLOCK_M).to(tl.int32)
    b_row = (batch * K).to(tl.int32)
    c_row = (batch * M + tile_m * BLOCK_M).to(tl.int32)
    c_column = (tile_n * BLOCK_N).to(tl.int32)
    a_reduction = tl.zeros((), dtype=tl.int32)
    b_reduction = b_row
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load_tensor_descriptor(
            a_desc,
            [a_row, a_reduction],
        )
        b = tl.load_tensor_descriptor(
            b_desc,
            [b_reduction, c_column],
        )
        accumulator = tl.dot(a, b, acc=accumulator)
        a_reduction += BLOCK_K
        b_reduction += BLOCK_K
    tl.store_tensor_descriptor(
        c_desc,
        [c_row, c_column],
        accumulator.to(c_desc.dtype),
    )


@triton.jit
def _matmul_tle_consumer(
    a_reader,
    b_reader,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    batch,
    tile_m,
    tile_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    reduction_tiles: tl.constexpr = tl.cdiv(K, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for reduction_tile in tl.range(
        0,
        reduction_tiles,
        num_stages=1,
        loop_unroll_factor=reduction_tiles,
    ):
        a_wait = a_reader.wait(reduction_tile)
        b_wait = b_reader.wait(reduction_tile)
        accumulator = tle.gpu.wgmma(
            a_wait.slot.a,
            b_wait.slot.b,
            accumulator,
        )
        accumulator = tle.gpu.wgmma_wait(0, accumulator)
        a_reader.release(reduction_tile)
        b_reader.release(reduction_tile)

    # TLE plans are admitted only for complete output tiles, so this store
    # cannot cross either matrix boundary.
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = (
        (batch * M + rows[:, None]) * N + columns[None, :]
    )
    tl.store(
        c_ptr + offsets,
        accumulator.to(c_ptr.dtype.element_ty),
    )


@triton.jit
def _matmul_tle_producer(
    a_writer,
    b_writer,
    a_desc,
    b_desc,
    M: tl.constexpr,
    K: tl.constexpr,
    batch,
    tile_m,
    tile_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    reduction_tiles: tl.constexpr = tl.cdiv(K, BLOCK_K)
    for reduction_tile in tl.range(
        0,
        reduction_tiles,
        num_stages=1,
        loop_unroll_factor=reduction_tiles,
    ):
        reduction = reduction_tile * BLOCK_K
        a_slot = a_writer.acquire(reduction_tile)
        tle.gpu.copy(
            a_desc,
            a_slot.a,
            (BLOCK_M, BLOCK_K),
            (batch * M + tile_m * BLOCK_M, reduction),
        )
        a_writer.commit(reduction_tile)

        b_slot = b_writer.acquire(reduction_tile)
        tle.gpu.copy(
            b_desc,
            b_slot.b,
            (BLOCK_K, BLOCK_N),
            (batch * K + reduction, tile_n * BLOCK_N),
        )
        b_writer.commit(reduction_tile)


@triton.jit
def matmul_tle_kernel(
    a_desc,
    b_desc,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BATCH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    INPUT_KIND: tl.constexpr,
    PANEL_WIDTH: tl.constexpr,
):
    if INPUT_KIND == 1:
        a_smem = tle.gpu.alloc(
            (PIPELINE_STAGES, BLOCK_M, BLOCK_K),
            dtype=tl.bfloat16,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=True,
        )
        b_smem = tle.gpu.alloc(
            (PIPELINE_STAGES, BLOCK_K, BLOCK_N),
            dtype=tl.bfloat16,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=True,
        )
    else:
        a_smem = tle.gpu.alloc(
            (PIPELINE_STAGES, BLOCK_M, BLOCK_K),
            dtype=tl.float16,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=True,
        )
        b_smem = tle.gpu.alloc(
            (PIPELINE_STAGES, BLOCK_K, BLOCK_N),
            dtype=tl.float16,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=True,
        )
    a_pipe = tle.pipe(
        capacity=PIPELINE_STAGES,
        scope="cta",
        name="matmul_a",
        a=a_smem,
    )
    b_pipe = tle.pipe(
        capacity=PIPELINE_STAGES,
        scope="cta",
        name="matmul_b",
        b=b_smem,
    )

    tiles_m: tl.constexpr = tl.cdiv(M, BLOCK_M)
    tiles_n: tl.constexpr = tl.cdiv(N, BLOCK_N)
    tiles_per_batch: tl.constexpr = tiles_m * tiles_n
    linear_tile = tl.program_id(0)
    batch = linear_tile // tiles_per_batch
    tile = linear_tile % tiles_per_batch
    panel_size: tl.constexpr = PANEL_WIDTH * tiles_m
    panel = tile // panel_size
    panel_offset = tile % panel_size
    tile_m = panel_offset // PANEL_WIDTH
    tile_n_in_panel = panel_offset % PANEL_WIDTH
    tile_n_in_panel = tl.where(
        tile_m % 2 == 1,
        PANEL_WIDTH - 1 - tile_n_in_panel,
        tile_n_in_panel,
    )
    tile_m = tl.where(panel % 2 == 1, tiles_m - 1 - tile_m, tile_m)
    tile_n = panel * PANEL_WIDTH + tile_n_in_panel

    tle.gpu.warp_specialize(
        [
            (
                _matmul_tle_consumer,
                (
                    a_pipe.reader(),
                    b_pipe.reader(),
                    c_ptr,
                    M,
                    N,
                    K,
                    batch,
                    tile_m,
                    tile_n,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                ),
            ),
            (
                _matmul_tle_producer,
                (
                    a_pipe.writer(),
                    b_pipe.writer(),
                    a_desc,
                    b_desc,
                    M,
                    K,
                    batch,
                    tile_m,
                    tile_n,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                ),
            ),
        ],
        worker_num_warps=[4],
        worker_num_regs=[32],
    )


@triton.jit
def matmul_strided_kernel(
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

    # Dense row-major matrices use block pointers so the MTGPU lowering can
    # preserve matrix-tile structure through global loads and tl.dot. MTGPU
    # block-pointer zero padding is unreliable for partial reduction tiles.
    if (
        A_STRIDE_K == 1 and B_STRIDE_N == 1 and C_STRIDE_N == 1
        and K % BLOCK_K == 0
    ):
        a_block = tl.make_block_ptr(
            base=a_ptr + a_batch_offset,
            shape=(M, K),
            strides=(A_STRIDE_M, A_STRIDE_K),
            offsets=(tile_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        b_block = tl.make_block_ptr(
            base=b_ptr + b_batch_offset,
            shape=(K, N),
            strides=(B_STRIDE_K, B_STRIDE_N),
            offsets=(0, tile_n * BLOCK_N),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in tl.range(0, K, BLOCK_K):
            if M % BLOCK_M == 0 and K % BLOCK_K == 0:
                a = tl.load(a_block)
            else:
                a = tl.load(
                    a_block,
                    boundary_check=(0, 1),
                    padding_option="zero",
                )
            if K % BLOCK_K == 0 and N % BLOCK_N == 0:
                b = tl.load(b_block)
            else:
                b = tl.load(
                    b_block,
                    boundary_check=(0, 1),
                    padding_option="zero",
                )
            if INPUT_IS_FLOAT32 and USE_TF32:
                accumulator += tl.dot(a, b, input_precision="tf32")
            elif INPUT_IS_FLOAT32:
                accumulator += tl.dot(a, b, input_precision="ieee")
            else:
                accumulator += tl.dot(a, b)
            a_block = tl.advance(a_block, (0, BLOCK_K))
            b_block = tl.advance(b_block, (BLOCK_K, 0))
        c_block = tl.make_block_ptr(
            base=c_ptr + c_batch_offset,
            shape=(M, N),
            strides=(C_STRIDE_M, C_STRIDE_N),
            offsets=(tile_m * BLOCK_M, tile_n * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        output = accumulator.to(c_ptr.dtype.element_ty)
        if M % BLOCK_M == 0 and N % BLOCK_N == 0:
            tl.store(c_block, output)
        else:
            tl.store(c_block, output, boundary_check=(0, 1))
    else:
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
                mask=(rows[:, None] < M)
                & (reduction_offsets[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_tile_ptrs,
                mask=(reduction_offsets[:, None] < K)
                & (columns[None, :] < N),
                other=0.0,
            )
            if INPUT_IS_FLOAT32 and USE_TF32:
                accumulator += tl.dot(a, b, input_precision="tf32")
            elif INPUT_IS_FLOAT32:
                accumulator += tl.dot(a, b, input_precision="ieee")
            else:
                accumulator += tl.dot(a, b)
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
