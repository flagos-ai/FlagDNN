# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Platform-neutral strided/batched MatMul kernel from ``flag_dnn.ops.mm``."""

import triton
import triton.language as tl


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
            accumulator += tl.dot(a, b, input_precision="tf32")
        else:
            accumulator += tl.dot(a, b, input_precision="ieee")
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
