# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Hopper short-K TF32 GEMM: one TMA producer and two math warp groups.

The consumers own adjacent N tiles and share each A tile. Four input slots
overlap loads and math; an empty slot requires both consumers to release it.
TensorMaps perform TF32-RNE on FP32 storage. All launches use libtriton_jit.
"""

import triton
import triton.experimental.gluon as g
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language import (
    BlockedLayout,
    DotOperandLayout,
    NVMMADistributedLayout,
    NVMMASharedLayout,
    SliceLayout,
)
from triton.experimental.gluon.language.nvidia.hopper import (
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_wait,
)


@g.jit
def _produce(
    a_desc,
    b_desc,
    a_shared,
    b_shared,
    empty,
    full,
    BATCH: gl.constexpr,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    SLOTS: gl.constexpr,
):
    tiles_m: gl.constexpr = M // BLOCK_M
    tiles_n: gl.constexpr = N // BLOCK_N
    sequence = 0
    for base in range(
        gl.program_id(0) * 2, BATCH * tiles_m * tiles_n, gl.num_programs(0) * 2
    ):
        batch = base // (tiles_m * tiles_n)
        row = base % (tiles_m * tiles_n) // tiles_n * BLOCK_M
        col = base % tiles_n * BLOCK_N
        for kt in range(K // BLOCK_K):
            slot = (sequence + kt) % SLOTS
            phase = ((sequence + kt) // SLOTS) % 2
            mbarrier.wait(empty.index(slot), phase ^ 1)
            mbarrier.expect(
                full.index(slot), (BLOCK_M + 2 * BLOCK_N) * BLOCK_K * 4
            )
            tma.async_copy_global_to_shared(
                a_desc,
                [batch * M + row, kt * BLOCK_K],
                full.index(slot),
                a_shared.index(slot),
            )
            for cid in gl.static_range(2):
                tma.async_copy_global_to_shared(
                    b_desc,
                    [batch * K + kt * BLOCK_K, col + cid * BLOCK_N],
                    full.index(slot),
                    b_shared.index(cid * SLOTS + slot),
                )
        sequence += K // BLOCK_K


@g.jit
def _consume(
    c_ptr,
    a_shared,
    b_shared,
    empty,
    full,
    BATCH: gl.constexpr,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    SLOTS: gl.constexpr,
    CID: gl.constexpr,
):
    mma: gl.constexpr = NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BLOCK_M, 8]
    )
    output: gl.constexpr = BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    tiles_m: gl.constexpr = M // BLOCK_M
    tiles_n: gl.constexpr = N // BLOCK_N
    sequence = 0
    for tile in range(
        gl.program_id(0) * 2 + CID,
        BATCH * tiles_m * tiles_n,
        gl.num_programs(0) * 2,
    ):
        batch = tile // (tiles_m * tiles_n)
        row = tile % (tiles_m * tiles_n) // tiles_n * BLOCK_M
        col = tile % tiles_n * BLOCK_N
        acc = gl.full((BLOCK_N, BLOCK_M), 0, gl.float32, mma)
        for kt in range(K // BLOCK_K):
            slot = (sequence + kt) % SLOTS
            mbarrier.wait(full.index(slot), ((sequence + kt) // SLOTS) % 2)
            b = (
                b_shared.index(CID * SLOTS + slot)
                .permute([1, 0])
                .load(DotOperandLayout(0, mma, 1))
            )
            # Precision is pinned in the cached compilation options; this
            # Gluon version does not unwrap an explicit constexpr string here.
            acc = warpgroup_mma(
                b, a_shared.index(slot).permute([1, 0]), acc, is_async=True
            )
            # No register operand crosses a loop boundary while still in flight.
            acc = warpgroup_mma_wait(0, (acc,))
            mbarrier.arrive(empty.index(slot))
        values = gl.convert_layout(acc.T, output)
        rr = row + gl.arange(0, BLOCK_M, layout=SliceLayout(1, output))
        cc = col + gl.arange(0, BLOCK_N, layout=SliceLayout(0, output))
        gl.store(
            c_ptr
            + batch.to(gl.int64) * M * N
            + rr[:, None].to(gl.int64) * N
            + cc[None, :],
            values,
            cache_modifier=".cs",
        )
        sequence += K // BLOCK_K


@g.jit
def matmul_tf32_short_kernel(
    a_desc,
    b_desc,
    c_ptr,
    BATCH: gl.constexpr,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    SLOTS: gl.constexpr,
    PERSISTENT_GRID: gl.constexpr,
):
    gl.static_assert(K == 512 and BLOCK_K == 32)
    gl.static_assert(BLOCK_M == 256 and BLOCK_N == 64)
    gl.static_assert(M % BLOCK_M == 0 and N % (2 * BLOCK_N) == 0)
    gl.static_assert(SLOTS == 3 or SLOTS == 4)
    layout: gl.constexpr = NVMMASharedLayout(128, 32)
    a_shared = gl.allocate_shared_memory(
        gl.float32, [SLOTS, BLOCK_M, BLOCK_K], layout
    )
    b_shared = gl.allocate_shared_memory(
        gl.float32, [2 * SLOTS, BLOCK_K, BLOCK_N], layout
    )
    empty = gl.allocate_shared_memory(
        gl.int64, [SLOTS, 1], mbarrier.MBarrierLayout()
    )
    full = gl.allocate_shared_memory(
        gl.int64, [SLOTS, 1], mbarrier.MBarrierLayout()
    )
    for slot in gl.static_range(SLOTS):
        mbarrier.init(empty.index(slot), count=2)
        mbarrier.init(full.index(slot), count=1)
    gl.warp_specialize(
        [
            (
                _produce,
                (
                    a_desc,
                    b_desc,
                    a_shared,
                    b_shared,
                    empty,
                    full,
                    BATCH,
                    M,
                    N,
                    K,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    SLOTS,
                ),
            ),
            (
                _consume,
                (
                    c_ptr,
                    a_shared,
                    b_shared,
                    empty,
                    full,
                    BATCH,
                    M,
                    N,
                    K,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    SLOTS,
                    0,
                ),
            ),
            (
                _consume,
                (
                    c_ptr,
                    a_shared,
                    b_shared,
                    empty,
                    full,
                    BATCH,
                    M,
                    N,
                    K,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    SLOTS,
                    1,
                ),
            ),
        ],
        [4, 4],
        [224, 224],
    )
    for slot in gl.static_range(SLOTS):
        mbarrier.invalidate(empty.index(slot))
        mbarrier.invalidate(full.index(slot))
