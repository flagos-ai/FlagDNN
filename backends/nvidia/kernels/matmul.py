# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA MatMul kernels with cuDNN-aligned TF32 multiply / FP32 accumulation.

FP32 operands retain their storage range; TF32 affects multiplication only.
"""

import triton
import triton.language as tl


@triton.jit
def _tf32_rne(x, NATIVE: tl.constexpr = False):
    """Explicit nearest-even conversion; TF32 dot alone truncates on Hopper.

    SM90 has a native RNE conversion. The integer fallback also works on SM80,
    where cvt.rn.tf32 is unavailable, and preserves nonfinite encodings.
    """
    if NATIVE:
        return tl.inline_asm_elementwise(
            "{ .reg .b32 t; cvt.rn.tf32.f32 t, $1; mov.b32 $0, t; }",
            constraints="=f,f",
            args=[x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    bits = x.to(tl.uint32, bitcast=True)
    rounded = (bits + 0xFFF + ((bits >> 13) & 1)) & 0xFFFFE000
    result = tl.where((bits & 0x7F800000) == 0x7F800000, bits, rounded)
    return result.to(tl.float32, bitcast=True)


@triton.jit
def matmul_tf32_pack_b_kernel(
    b_ptr,
    packed_ptr,
    BATCH: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NATIVE_TF32_RNE: tl.constexpr,
):
    """Transpose/round B once per execution into FP32 workspace."""
    tile = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    tiles_n: tl.constexpr = triton.cdiv(N, BLOCK_SIZE)
    k = tile // tiles_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n = tile % tiles_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(
        b_ptr + batch * K * N + k[:, None].to(tl.int64) * N + n[None, :],
        (k[:, None] < K) & (n[None, :] < N),
        other=0.0,
    )
    tl.store(
        packed_ptr + batch * K * N + n[None, :].to(tl.int64) * K + k[:, None],
        _tf32_rne(values, NATIVE_TF32_RNE),
        (k[:, None] < K) & (n[None, :] < N),
    )


@triton.jit
def matmul_tf32_tma_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    BATCH: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PERSISTENT_GRID: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Hopper TF32 GEMM over FP32, K-contiguous, already-rounded B batches."""
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[BATCH * M, K],
        strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[BATCH * N, K],
        strides=[K, 1],
        block_shape=[BLOCK_N, BLOCK_K],
    )
    tiles_m: tl.constexpr = triton.cdiv(M, BLOCK_M)
    tiles_n: tl.constexpr = triton.cdiv(N, BLOCK_N)
    for tile in tl.range(
        tl.program_id(0), BATCH * tiles_m * tiles_n, tl.num_programs(0)
    ):
        batch = tile // (tiles_m * tiles_n)
        local = tile % (tiles_m * tiles_n)
        group = local // (GROUP_M * tiles_n)
        first_m = group * GROUP_M
        group_m = tl.minimum(tiles_m - first_m, GROUP_M)
        row = (first_m + local % (GROUP_M * tiles_n) % group_m) * BLOCK_M
        col = (local % (GROUP_M * tiles_n) // group_m) * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for kt in tl.range(triton.cdiv(K, BLOCK_K)):
            a = _tf32_rne(a_desc.load([batch * M + row, kt * BLOCK_K]), True)
            b = tl.trans(b_desc.load([batch * N + col, kt * BLOCK_K]))
            acc = tl.dot(a, b, acc, input_precision="tf32")
        rows = row + tl.arange(0, BLOCK_M)
        columns = col + tl.arange(0, BLOCK_N)
        tl.store(
            c_ptr
            + batch.to(tl.int64) * M * N
            + rows[:, None].to(tl.int64) * N
            + columns[None, :],
            acc,
            (rows[:, None] < M) & (columns[None, :] < N),
            cache_modifier=".cs",
        )


@triton.jit
def matmul_tf32_tensor_map_kernel(
    a_desc,
    b_desc,
    c_ptr,
    BATCH: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    PERSISTENT_GRID: tl.constexpr,
):
    # Host TensorMaps perform TF32-RNE while all global storage stays FP32.
    # Compute C.T = B.T @ A.T to keep the right WGMMA operand K-contiguous.
    tiles_m: tl.constexpr = M // BLOCK_M
    tiles_n: tl.constexpr = N // BLOCK_N
    for tile in range(
        tl.program_id(0), BATCH * tiles_m * tiles_n, tl.num_programs(0)
    ):
        batch = tile // (tiles_m * tiles_n)
        local = tile % (tiles_m * tiles_n)
        first = local // (GROUP_M * tiles_m) * GROUP_M
        size = tl.minimum(tiles_n - first, GROUP_M)
        col = (first + local % (GROUP_M * tiles_m) % size) * BLOCK_N
        row = local % (GROUP_M * tiles_m) // size * BLOCK_M
        acc = tl.zeros((BLOCK_N, BLOCK_M), tl.float32)
        for kt in range(K // BLOCK_K):
            a = a_desc.load([batch * M + row, kt * BLOCK_K])
            b = b_desc.load([batch * K + kt * BLOCK_K, col])
            # Force register-left WGMMA. ptxas removes this identity move;
            # it does not add rounding or alter already-rounded operands.
            b = tl.inline_asm_elementwise(
                "mov.b32 $0,$1;",
                "=f,f",
                [b],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
            acc = tl.dot(tl.trans(b), tl.trans(a), acc, input_precision="tf32")
        rows = row + tl.arange(0, BLOCK_M)
        cols = col + tl.arange(0, BLOCK_N)
        tl.store(
            c_ptr
            + batch.to(tl.int64) * M * N
            + rows[:, None].to(tl.int64) * N
            + cols[None, :],
            tl.trans(acc),
            cache_modifier=".cs",
        )


@triton.jit
def matmul_tf32_tma_direct_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    BATCH: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PERSISTENT_GRID: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Compute C.T = B.T @ A.T, avoiding a global B packing stage.

    Both inputs remain FP32 and are rounded to TF32 RNE in the tile. The right
    MMA operand is now the original K-contiguous A, simplifying shared layout.
    Dispatch admits complete tiles only; flattened TMA loads stay in batch.
    """
    a_desc = tl.make_tensor_descriptor(
        a_ptr, [BATCH * M, K], [K, 1], [BLOCK_M, BLOCK_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, [BATCH * K, N], [N, 1], [BLOCK_K, BLOCK_N]
    )
    tiles_m: tl.constexpr = M // BLOCK_M
    tiles_n: tl.constexpr = N // BLOCK_N
    for tile in tl.range(
        tl.program_id(0), BATCH * tiles_m * tiles_n, tl.num_programs(0)
    ):
        batch = tile // (tiles_m * tiles_n)
        local = tile % (tiles_m * tiles_n)
        # Group on the transposed output's row axis (the original N axis).
        group = local // (GROUP_M * tiles_m)
        first_n = group * GROUP_M
        group_n = tl.minimum(tiles_n - first_n, GROUP_M)
        col = (first_n + local % (GROUP_M * tiles_m) % group_n) * BLOCK_N
        row = (local % (GROUP_M * tiles_m) // group_n) * BLOCK_M
        acc = tl.zeros((BLOCK_N, BLOCK_M), tl.float32)
        for kt in range(K // BLOCK_K):
            a = _tf32_rne(a_desc.load([batch * M + row, kt * BLOCK_K]), True)
            b = _tf32_rne(b_desc.load([batch * K + kt * BLOCK_K, col]), True)
            acc = tl.dot(tl.trans(b), tl.trans(a), acc, input_precision="tf32")
        rows = row + tl.arange(0, BLOCK_M)
        columns = col + tl.arange(0, BLOCK_N)
        tl.store(
            c_ptr
            + batch.to(tl.int64) * M * N
            + rows[:, None].to(tl.int64) * N
            + columns[None, :],
            tl.trans(acc),
            cache_modifier=".cs",
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
            accumulator = tl.dot(a, b, accumulator, input_precision="tf32")
        elif INPUT_IS_FLOAT32:
            accumulator = tl.dot(a, b, accumulator, input_precision="tf32x3")
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


@triton.jit
def matmul_batched_contiguous_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    BATCH: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    INPUT_IS_FLOAT32: tl.constexpr,
    USE_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NATIVE_TF32_RNE: tl.constexpr = False,
):
    """Fast N×M×K contiguous GEMM using compiler-visible block pointers."""
    tile = tl.program_id(0)
    batch = tl.program_id(1)
    tiles_m = tl.cdiv(M, BLOCK_M)
    tiles_n = tl.cdiv(N, BLOCK_N)
    if GROUP_M == 1:
        tile_m = tile // tiles_n
        tile_n = tile - tile_m * tiles_n
    elif GROUP_M >= tiles_m:
        tile_m = tile % tiles_m
        tile_n = tile // tiles_m
    else:
        tiles_per_group = GROUP_M * tiles_n
        group = tile // tiles_per_group
        first_tile_m = group * GROUP_M
        group_m = tl.minimum(tiles_m - first_tile_m, GROUP_M)
        tile_in_group = tile - group * tiles_per_group
        tile_m = first_tile_m + tile_in_group % group_m
        tile_n = tile_in_group // group_m

    a_block = tl.make_block_ptr(
        base=a_ptr + batch * M * K,
        shape=(M, K),
        strides=(K, 1),
        offsets=(tile_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block = tl.make_block_ptr(
        base=b_ptr + batch * K * N,
        shape=(K, N),
        strides=(N, 1),
        offsets=(0, tile_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
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
        if INPUT_IS_FLOAT32 and USE_TF32:
            a, b = _tf32_rne(a, NATIVE_TF32_RNE), _tf32_rne(b, NATIVE_TF32_RNE)
            accumulator = tl.dot(a, b, accumulator, input_precision="tf32")
        elif INPUT_IS_FLOAT32:
            accumulator = tl.dot(a, b, accumulator, input_precision="tf32x3")
        else:
            accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
        a_block = tl.advance(a_block, (0, BLOCK_K))
        b_block = tl.advance(b_block, (BLOCK_K, 0))

    c_block = tl.make_block_ptr(
        base=c_ptr + batch * M * N,
        shape=(M, N),
        strides=(N, 1),
        offsets=(tile_m * BLOCK_M, tile_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    output = accumulator.to(c_ptr.dtype.element_ty)
    if M % BLOCK_M == 0 and N % BLOCK_N == 0:
        tl.store(c_block, output, boundary_check=())
    else:
        tl.store(c_block, output, boundary_check=(0, 1))


@triton.jit
def matmul_batched_broadcast_a_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    BATCH: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    INPUT_IS_FLOAT32: tl.constexpr,
    USE_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Batched contiguous GEMM with one A matrix broadcast to every batch."""
    tile = tl.program_id(0)
    batch = tl.program_id(1)
    tiles_m = tl.cdiv(M, BLOCK_M)
    tiles_n = tl.cdiv(N, BLOCK_N)
    if GROUP_M == 1:
        tile_m = tile // tiles_n
        tile_n = tile - tile_m * tiles_n
    elif GROUP_M >= tiles_m:
        tile_m = tile % tiles_m
        tile_n = tile // tiles_m
    else:
        tiles_per_group = GROUP_M * tiles_n
        group = tile // tiles_per_group
        first_tile_m = group * GROUP_M
        group_m = tl.minimum(tiles_m - first_tile_m, GROUP_M)
        tile_in_group = tile - group * tiles_per_group
        tile_m = first_tile_m + tile_in_group % group_m
        tile_n = tile_in_group // group_m

    a_block = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(K, 1),
        offsets=(tile_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block = tl.make_block_ptr(
        base=b_ptr + batch * K * N,
        shape=(K, N),
        strides=(N, 1),
        offsets=(0, tile_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
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
        if INPUT_IS_FLOAT32 and USE_TF32:
            a, b = _tf32_rne(a), _tf32_rne(b)
            accumulator = tl.dot(a, b, accumulator, input_precision="tf32")
        elif INPUT_IS_FLOAT32:
            accumulator = tl.dot(a, b, accumulator, input_precision="tf32x3")
        else:
            accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
        a_block = tl.advance(a_block, (0, BLOCK_K))
        b_block = tl.advance(b_block, (BLOCK_K, 0))

    c_block = tl.make_block_ptr(
        base=c_ptr + batch * M * N,
        shape=(M, N),
        strides=(N, 1),
        offsets=(tile_m * BLOCK_M, tile_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    output = accumulator.to(c_ptr.dtype.element_ty)
    if M % BLOCK_M == 0 and N % BLOCK_N == 0:
        tl.store(c_block, output, boundary_check=())
    else:
        tl.store(c_block, output, boundary_check=(0, 1))


@triton.jit
def matmul_p5_split_k_kernel(
    a_ptr,
    b_ptr,
    partial_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    B_STRIDE_K: tl.constexpr,
    B_STRIDE_N: tl.constexpr,
    SPLITS: tl.constexpr,
    INPUT_IS_FLOAT32: tl.constexpr,
    USE_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Increase P5 parallelism by assigning independent K ranges to CTAs."""
    tile = tl.program_id(0)
    split = tl.program_id(1)
    tiles_m = tl.cdiv(M, BLOCK_M)
    tiles_n = tl.cdiv(N, BLOCK_N)
    if GROUP_M == 1:
        tile_m = tile // tiles_n
        tile_n = tile - tile_m * tiles_n
    elif GROUP_M >= tiles_m:
        tile_m = tile % tiles_m
        tile_n = tile // tiles_m
    else:
        tiles_per_group = GROUP_M * tiles_n
        group = tile // tiles_per_group
        first_tile_m = group * GROUP_M
        group_m = tl.minimum(tiles_m - first_tile_m, GROUP_M)
        tile_in_group = tile - group * tiles_per_group
        tile_m = first_tile_m + tile_in_group % group_m
        tile_n = tile_in_group // group_m

    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    reduction = tl.arange(0, BLOCK_K)
    reduction_blocks = tl.cdiv(K, BLOCK_K)
    blocks_per_split = tl.cdiv(reduction_blocks, SPLITS)
    reduction_begin = split * blocks_per_split * BLOCK_K
    reduction_end = tl.minimum(reduction_begin + blocks_per_split * BLOCK_K, K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for reduction_start in tl.range(reduction_begin, reduction_end, BLOCK_K):
        reduction_offsets = reduction_start + reduction
        a = tl.load(
            a_ptr + rows[:, None] * K + reduction_offsets[None, :],
            mask=(rows[:, None] < M)
            & (reduction_offsets[None, :] < reduction_end),
            other=0.0,
        )
        b = tl.load(
            b_ptr
            + reduction_offsets[:, None] * B_STRIDE_K
            + columns[None, :] * B_STRIDE_N,
            mask=(reduction_offsets[:, None] < reduction_end)
            & (columns[None, :] < N),
            other=0.0,
        )
        if INPUT_IS_FLOAT32 and USE_TF32:
            a, b = _tf32_rne(a), _tf32_rne(b)
            accumulator = tl.dot(a, b, accumulator, input_precision="tf32")
        elif INPUT_IS_FLOAT32:
            accumulator = tl.dot(a, b, accumulator, input_precision="tf32x3")
        else:
            accumulator += tl.dot(a, b, input_precision="ieee")

    tl.store(
        partial_ptr + split * M * N + rows[:, None] * N + columns[None, :],
        accumulator,
        mask=(rows[:, None] < M) & (columns[None, :] < N),
    )


@triton.jit
def matmul_p5_split_k_reduce_kernel(
    partial_ptr,
    output_ptr,
    TOTAL: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOTAL
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for split in tl.static_range(0, SPLITS):
        accumulator += tl.load(
            partial_ptr + split * TOTAL + offsets,
            mask=mask,
            other=0.0,
        )
    tl.store(output_ptr + offsets, accumulator, mask=mask)


# -----------------------------------------------------------------------------
# Kernel algorithm variants. Native dispatch selects only registry-declared
# entry points; auxiliary kernels remain available to explicit compiler policy.
# -----------------------------------------------------------------------------


@triton.jit
def _transpose_b_kernel(
    source,
    destination,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    offs_k = pid_k * BLOCK + tl.arange(0, BLOCK)
    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)
    source_offsets = pid_b * K * N + offs_k[:, None] * N + offs_n[None, :]
    values = tl.load(source + source_offsets)
    destination_offsets = pid_b * N * K + offs_n[:, None] * K + offs_k[None, :]
    tl.store(destination + destination_offsets, tl.trans(values))


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, group_m):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * group_m
    group_size_m = tl.minimum(num_pid_m - first_pid_m, group_m)
    pid_in_group = tile_id - group_id * num_pid_in_group
    pid_m = first_pid_m + pid_in_group % group_size_m
    pid_n = pid_in_group // group_size_m
    return pid_m, pid_n


@triton.jit
def matmul_batched_tma_short_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    BATCH: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    PERSISTENT_GRID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    INPUT_IS_FLOAT32: tl.constexpr,
):
    """Short-K GEMM with overlapping reduction tiles and a direct epilogue."""
    tl.static_assert(not INPUT_IS_FLOAT32)
    tl.static_assert(M % BLOCK_M == 0 and K % BLOCK_K == 0)
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[BATCH * M, K],
        strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[BATCH * K, N],
        strides=[N, 1],
        block_shape=[BLOCK_K, BLOCK_N],
    )
    tiles_m: tl.constexpr = triton.cdiv(M, BLOCK_M)
    tiles_n: tl.constexpr = triton.cdiv(N, BLOCK_N)
    for tile in tl.range(
        tl.program_id(0),
        BATCH * tiles_m * tiles_n,
        tl.num_programs(0),
        flatten=True,
    ):
        batch = tile // (tiles_m * tiles_n)
        local_tile = tile % (tiles_m * tiles_n)
        tile_m, tile_n = _compute_pid(
            local_tile, GROUP_M * tiles_n, tiles_m, GROUP_M
        )
        off_m, off_n = tile_m * BLOCK_M, tile_n * BLOCK_N
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for tile_k in tl.range(tl.cdiv(K, BLOCK_K), loop_unroll_factor=2):
            a = a_desc.load([batch * M + off_m, tile_k * BLOCK_K])
            b = b_desc.load([batch * K + tile_k * BLOCK_K, off_n])
            accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
        rows = off_m + tl.arange(0, BLOCK_M)
        columns = off_n + tl.arange(0, BLOCK_N)
        tl.store(
            c_ptr
            + batch.to(tl.int64) * M * N
            + rows[:, None] * N
            + columns[None, :],
            accumulator.to(c_ptr.dtype.element_ty),
            mask=(rows[:, None] < M) & (columns[None, :] < N),
            # Inputs are reused by later persistent tiles; output is streamed.
            cache_modifier=".cs",
        )


@triton.jit
def matmul_batched_tma_persistent_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    BATCH: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    PERSISTENT_GRID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    INPUT_IS_FLOAT32: tl.constexpr,
):
    """Hopper persistent GEMM using device-side TMA descriptors."""
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[BATCH * M, K],
        strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[BATCH * K, N],
        strides=[N, 1],
        block_shape=[BLOCK_K, BLOCK_N],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[BATCH * M, N],
        strides=[N, 1],
        block_shape=[
            BLOCK_M,
            BLOCK_N // 2 if BLOCK_N >= 128 else BLOCK_N,
        ],
    )
    start_pid = tl.program_id(0)
    num_sms = tl.num_programs(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    tiles_per_batch = num_pid_m * num_pid_n
    total_tiles = BATCH * tiles_per_batch
    num_pid_in_group = GROUP_M * num_pid_n
    reduction_tiles = tl.cdiv(K, BLOCK_K)
    tiles_per_sm = total_tiles // num_sms
    if start_pid < total_tiles % num_sms:
        tiles_per_sm += 1

    tile_id = start_pid - num_sms
    reduction_tile = -1
    batch_id = 0
    off_m = 0
    off_n = 0
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, reduction_tiles * tiles_per_sm):
        reduction_tile = tl.where(
            reduction_tile == reduction_tiles - 1,
            0,
            reduction_tile + 1,
        )
        if reduction_tile == 0:
            tile_id += num_sms
            batch_id = tile_id // tiles_per_batch
            local_tile = tile_id - batch_id * tiles_per_batch
            pid_m, pid_n = _compute_pid(
                local_tile, num_pid_in_group, num_pid_m, GROUP_M
            )
            off_m = pid_m * BLOCK_M
            off_n = pid_n * BLOCK_N

        off_k = reduction_tile * BLOCK_K
        a = a_desc.load([batch_id * M + off_m, off_k])
        b = b_desc.load([batch_id * K + off_k, off_n])
        if INPUT_IS_FLOAT32:
            a, b = _tf32_rne(a), _tf32_rne(b)
            accumulator = tl.dot(
                a,
                b,
                accumulator,
                input_precision="tf32",
            )
        else:
            accumulator = tl.dot(
                a,
                b,
                accumulator,
                input_precision="ieee",
            )

        if reduction_tile == reduction_tiles - 1:
            if BLOCK_N >= 128:
                reshaped = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
                permuted = tl.permute(reshaped, (0, 2, 1))
                output0, output1 = tl.split(permuted)
                c_desc.store(
                    [batch_id * M + off_m, off_n],
                    output0.to(c_ptr.dtype.element_ty),
                )
                c_desc.store(
                    [batch_id * M + off_m, off_n + BLOCK_N // 2],
                    output1.to(c_ptr.dtype.element_ty),
                )
            else:
                c_desc.store(
                    [batch_id * M + off_m, off_n],
                    accumulator.to(c_ptr.dtype.element_ty),
                )
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
