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


# -----------------------------------------------------------------------------
# Kernel algorithm variants. Native dispatch selects only registry-declared
# entry points; auxiliary kernels remain available to explicit compiler policy.
# -----------------------------------------------------------------------------


@triton.jit
def _round_fp32_to_tf32(x):
    bits = x.to(tl.uint32, bitcast=True)
    rounded = bits + 0xFFF + ((bits >> 13) & 1)
    rounded &= 0xFFFFE000
    is_special = (bits & 0x7F800000) == 0x7F800000
    rounded = tl.where(is_special, bits, rounded)
    return rounded.to(tl.float32, bitcast=True)


@triton.jit
def _batched_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ROUND_F32_TO_TF32: tl.constexpr,
):
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    if GROUP_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    elif GROUP_M >= (M + BLOCK_M - 1) // BLOCK_M:
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m
    else:
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_in_group = pid % num_pid_in_group
        pid_m = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m

    a_base = a_ptr + bid * M * K
    b_base = b_ptr + bid * K * N
    c_base = c_ptr + bid * M * N
    a_block = tl.make_block_ptr(
        base=a_base,
        shape=(M, K),
        strides=(K, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block = tl.make_block_ptr(
        base=b_base,
        shape=(K, N),
        strides=(N, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in tl.range(0, K, BLOCK_K):
        if M % BLOCK_M == 0 and K % BLOCK_K == 0:
            a = tl.load(a_block, boundary_check=())
        else:
            a = tl.load(
                a_block,
                boundary_check=(0, 1),
                padding_option="zero",
            )
        if K % BLOCK_K == 0 and N % BLOCK_N == 0:
            b = tl.load(b_block, boundary_check=())
        else:
            b = tl.load(
                b_block,
                boundary_check=(0, 1),
                padding_option="zero",
            )
        if ROUND_F32_TO_TF32:
            a = _round_fp32_to_tf32(a)
            b = _round_fp32_to_tf32(b)
            acc += tl.dot(a, b, input_precision="tf32")
        else:
            acc += tl.dot(a, b)
        a_block = tl.advance(a_block, (0, BLOCK_K))
        b_block = tl.advance(b_block, (BLOCK_K, 0))

    c_block = tl.make_block_ptr(
        base=c_base,
        shape=(M, N),
        strides=(N, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    c = acc.to(c_ptr.dtype.element_ty)
    if M % BLOCK_M == 0 and N % BLOCK_N == 0:
        tl.store(c_block, c, boundary_check=())
    else:
        tl.store(c_block, c, boundary_check=(0, 1))


@triton.jit
def _batched_matmul_fp32_ieee_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    if GROUP_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid - pid_m * num_pid_n
    elif GROUP_M >= num_pid_m:
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m
    else:
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_in_group = pid - group_id * num_pid_in_group
        pid_m = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + bid * M * K + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + bid * K * N + offs_k[:, None] * N + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in tl.range(0, K, BLOCK_K):
        if M % BLOCK_M == 0 and K % BLOCK_K == 0:
            a = tl.load(a_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                other=0.0,
            )
        if K % BLOCK_K == 0 and N % BLOCK_N == 0:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
        acc += tl.dot(a, b, input_precision="ieee")
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
        offs_k += BLOCK_K

    c_ptrs = c_ptr + bid * M * N + offs_m[:, None] * N + offs_n[None, :]
    c = acc.to(c_ptr.dtype.element_ty)
    if M % BLOCK_M == 0 and N % BLOCK_N == 0:
        tl.store(c_ptrs, c)
    else:
        tl.store(
            c_ptrs,
            c,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


@triton.jit
def _batched_matmul_persistent_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BATCH: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    tiles_per_batch = num_pid_m * num_pid_n
    total_tiles = tiles_per_batch * BATCH
    tile_id = start_pid

    while tile_id < total_tiles:
        bid = tile_id // tiles_per_batch
        pid = tile_id - bid * tiles_per_batch
        if GROUP_M == 1:
            pid_m = pid // num_pid_n
            pid_n = pid - pid_m * num_pid_n
        elif GROUP_M >= num_pid_m:
            pid_m = pid % num_pid_m
            pid_n = pid // num_pid_m
        else:
            num_pid_in_group = GROUP_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
            pid_in_group = pid - group_id * num_pid_in_group
            pid_m = first_pid_m + (pid_in_group % group_size_m)
            pid_n = pid_in_group // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + bid * M * K + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = b_ptr + bid * K * N + offs_k[:, None] * N + offs_n[None, :]

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in tl.range(0, K, BLOCK_K):
            if M % BLOCK_M == 0 and K % BLOCK_K == 0:
                a = tl.load(a_ptrs)
            else:
                a = tl.load(
                    a_ptrs,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                    other=0.0,
                )
            if K % BLOCK_K == 0 and N % BLOCK_N == 0:
                b = tl.load(b_ptrs)
            else:
                b = tl.load(
                    b_ptrs,
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0,
                )
            acc += tl.dot(a, b, input_precision="tf32")
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * N
            offs_k += BLOCK_K

        c_ptrs = c_ptr + bid * M * N + offs_m[:, None] * N + offs_n[None, :]
        c = acc.to(c_ptr.dtype.element_ty)
        if M % BLOCK_M == 0 and N % BLOCK_N == 0:
            tl.store(c_ptrs, c)
        else:
            tl.store(
                c_ptrs,
                c,
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
            )
        tile_id += NUM_SMS
