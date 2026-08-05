# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compiler-safe NVIDIA SM90 Gluon MatMul kernels."""

from typing import Union

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_accumulator,
    warpgroup_mma_wait,
)
from triton.language.core import _aggregate as aggregate

# -----------------------------------------------------------------------------
# Kernel algorithm variants. Native dispatch selects only registry-declared
# entry points; auxiliary kernels remain available to explicit compiler policy.
# -----------------------------------------------------------------------------


@gluon.constexpr_function
def _get_warps_per_cta(block_m, block_n, num_warps):
    warps_per_cta = [4, 1]
    instruction_m = 16
    while warps_per_cta[0] * warps_per_cta[1] != num_warps:
        if block_m > instruction_m * warps_per_cta[0]:
            warps_per_cta[0] *= 2
        else:
            warps_per_cta[1] *= 2
    return warps_per_cta


@gluon.constexpr_function
def _get_instruction_n(block_m, block_n, num_warps):
    instruction_m = 16
    m_repetitions = triton.cdiv(block_m, instruction_m)
    n_repetitions = triton.cdiv(num_warps, m_repetitions)
    max_n = max(block_n // n_repetitions, 8)
    instruction_n = 256
    while instruction_n > max_n or block_n % instruction_n != 0:
        instruction_n -= 8
    assert instruction_n >= 8, "expected a valid WGMMA N instruction shape"
    return instruction_n


@gluon.constexpr_function
def _pick_wgmma_layout(dtype, block_m, block_n, num_warps):
    return gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=_get_warps_per_cta(block_m, block_n, num_warps),
        instr_shape=[
            16,
            _get_instruction_n(block_m, block_n, num_warps),
            256 // dtype.primitive_bitwidth,
        ],
    )


@aggregate
class _WGMMA:
    acc: Union[warpgroup_mma_accumulator, gl.tensor]
    use_acc: gl.tensor

    @gluon.constexpr_function
    def __init__(self, acc, use_acc):
        self.acc = acc
        self.use_acc = use_acc

    @gluon.jit
    def initialize(
        dtype: gl.constexpr,
        block_m: gl.constexpr,
        block_n: gl.constexpr,
        num_warps: gl.constexpr,
    ):
        layout: gl.constexpr = _pick_wgmma_layout(
            dtype, block_m, block_n, num_warps
        )
        acc = gl.zeros((block_m, block_n), dtype=gl.float32, layout=layout)
        return _WGMMA(acc, gl.to_tensor(False))

    @gluon.jit
    def issue_async_mma(self, a, b, precision: gl.constexpr):
        acc = warpgroup_mma(
            a,
            b,
            self.acc,
            is_async=True,
            use_acc=self.use_acc,
            precision=precision.value,
            max_num_imprecise_acc=32,
        )
        return _WGMMA(acc, gl.to_tensor(True))

    @gluon.jit
    def wait_num_outstanding(self, num_outstanding: gl.constexpr):
        acc = warpgroup_mma_wait(num_outstanding, (self.acc,))
        return _WGMMA(acc, self.use_acc)

    @gluon.jit
    def take_result(self):
        return self.acc


@aggregate
class _BarrierCounter:
    index: gl.tensor
    phase: gl.tensor
    num_barriers: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, index, phase, num_barriers):
        self.index = index
        self.phase = phase
        self.num_barriers = gl.constexpr(num_barriers)

    @gluon.jit
    def create(phase, num_barriers: gl.constexpr):
        return _BarrierCounter(
            gl.to_tensor(0), gl.to_tensor(phase), num_barriers
        )

    @gluon.must_use_result
    @gluon.jit
    def next(self):
        incremented = self.index + 1
        rollover = incremented == self.num_barriers
        index = gl.where(rollover, 0, incremented)
        phase = gl.where(rollover, self.phase ^ 1, self.phase)
        return _BarrierCounter(index, phase, self.num_barriers)


@aggregate
class _BatchedTileScheduler:
    start_tile: gl.tensor
    total_tiles: gl.tensor
    tiles_per_batch: gl.tensor
    num_pid_m: gl.tensor

    @gluon.constexpr_function
    def __init__(self, start_tile, total_tiles, tiles_per_batch, num_pid_m):
        self.start_tile = start_tile
        self.total_tiles = total_tiles
        self.tiles_per_batch = tiles_per_batch
        self.num_pid_m = num_pid_m

    @gluon.jit
    def initialize(
        batch,
        m,
        n,
        block_m: gl.constexpr,
        block_n: gl.constexpr,
    ):
        start_tile = gl.program_id(axis=0)
        num_pid_m = gl.cdiv(m, block_m)
        num_pid_n = gl.cdiv(n, block_n)
        tiles_per_batch = num_pid_m * num_pid_n
        total_tiles = batch * tiles_per_batch
        return _BatchedTileScheduler(
            start_tile, total_tiles, tiles_per_batch, num_pid_m
        )

    @gluon.jit
    def get_num_tiles(self):
        return gl.cdiv(
            self.total_tiles - self.start_tile, gl.num_programs(axis=0)
        )

    @gluon.jit
    def get_tile(self, index):
        tile_id = self.start_tile + index * gl.num_programs(axis=0)
        batch_id = tile_id // self.tiles_per_batch
        local_id = tile_id % self.tiles_per_batch
        pid_m = local_id % self.num_pid_m
        pid_n = local_id // self.num_pid_m
        return batch_id, pid_m, pid_n


@gluon.jit
def _issue_loads(
    producer,
    a_desc,
    b_desc,
    a_off_m,
    b_off_k,
    off_n,
    k_offset,
    bars,
    a_buffers,
    b_buffers,
    num_buffers: gl.constexpr,
):
    index = producer % num_buffers
    producer += 1
    barrier = bars.index(index)
    mbarrier.expect(
        barrier,
        a_desc.block_type.nbytes + b_desc.block_type.nbytes,
    )
    tma.async_copy_global_to_shared(
        a_desc, [a_off_m, k_offset], barrier, a_buffers.index(index)
    )
    tma.async_copy_global_to_shared(
        b_desc,
        [b_off_k + k_offset, off_n],
        barrier,
        b_buffers.index(index),
    )
    return producer


@gluon.jit
def _issue_mma(
    consumer,
    mma,
    bars,
    a_buffers,
    b_buffers,
    b_wgmma,
    b_copy_layout: gl.constexpr,
    transpose_b: gl.constexpr,
    num_buffers: gl.constexpr,
    precision: gl.constexpr,
):
    index = consumer % num_buffers
    phase = consumer // num_buffers & 1
    consumer += 1
    mbarrier.wait(bars.index(index), phase)
    mma = mma.wait_num_outstanding(0)
    b = b_buffers.index(index)
    if transpose_b:
        gl.thread_barrier()
        b_wgmma.store(b.load(b_copy_layout))
        gl.thread_barrier()
        fence_async_shared()
        b = b_wgmma
    mma = mma.issue_async_mma(a_buffers.index(index), b, precision)
    return consumer, mma


@gluon.jit
def _issue_counter_loads(
    producer,
    a_desc,
    b_desc,
    a_off_m,
    b_off_k,
    off_n,
    k_offset,
    bars,
    a_buffers,
    b_buffers,
):
    barrier = bars.index(producer.index)
    mbarrier.expect(
        barrier,
        a_desc.block_type.nbytes + b_desc.block_type.nbytes,
    )
    tma.async_copy_global_to_shared(
        a_desc,
        [a_off_m, k_offset],
        barrier,
        a_buffers.index(producer.index),
    )
    tma.async_copy_global_to_shared(
        b_desc,
        [b_off_k + k_offset, off_n],
        barrier,
        b_buffers.index(producer.index),
    )
    return producer.next()


@gluon.jit
def _issue_counter_mma(
    consumer,
    mma,
    bars,
    a_buffers,
    b_buffers,
    precision: gl.constexpr,
):
    mbarrier.wait(bars.index(consumer.index), consumer.phase)
    mma = mma.wait_num_outstanding(0)
    mma = mma.issue_async_mma(
        a_buffers.index(consumer.index),
        b_buffers.index(consumer.index),
        precision,
    )
    return consumer.next(), mma


@gluon.jit
def _matmul_sm90_ws_load(
    a_desc,
    b_desc,
    c_desc,
    batch,
    m,
    n,
    k,
    ready_bars,
    empty_bars,
    a_buffers,
    b_buffers,
    c_ready_bar,
    c_empty_bar,
    c_buffer,
):
    block_m: gl.constexpr = a_desc.block_type.shape[0]
    block_n: gl.constexpr = b_desc.block_type.shape[1]
    block_k: gl.constexpr = a_desc.block_type.shape[1]
    num_buffers: gl.constexpr = ready_bars.shape[0]
    scheduler = _BatchedTileScheduler.initialize(
        16, 2048, 2048, block_m, block_n
    )
    producer = _BarrierCounter.create(0, num_buffers)
    for tile_index in range(scheduler.get_num_tiles()):
        batch_id, pid_m, pid_n = scheduler.get_tile(tile_index)
        a_off_m = batch_id * 2048 + pid_m * block_m
        b_off_k = batch_id * 512
        off_n = pid_n * block_n
        for k_offset in gl.static_range(0, 512, block_k):
            mbarrier.wait(empty_bars.index(producer.index), producer.phase)
            ready_bar = ready_bars.index(producer.index)
            mbarrier.expect(
                ready_bar,
                a_desc.block_type.nbytes + b_desc.block_type.nbytes,
            )
            tma.async_copy_global_to_shared(
                a_desc,
                [a_off_m, k_offset],
                ready_bar,
                a_buffers.index(producer.index),
            )
            tma.async_copy_global_to_shared(
                b_desc,
                [b_off_k + k_offset, off_n],
                ready_bar,
                b_buffers.index(producer.index),
            )
            producer = producer.next()


@gluon.jit
def _matmul_sm90_ws_compute(
    a_desc,
    b_desc,
    c_desc,
    batch,
    m,
    n,
    k,
    ready_bars,
    empty_bars,
    a_buffers,
    b_buffers,
    c_ready_bar,
    c_empty_bar,
    c_buffer,
):
    block_m: gl.constexpr = c_desc.block_type.shape[0]
    block_n: gl.constexpr = c_desc.block_type.shape[1]
    block_k: gl.constexpr = a_desc.block_type.shape[1]
    input_dtype: gl.constexpr = a_desc.dtype
    output_dtype: gl.constexpr = c_desc.dtype
    num_buffers: gl.constexpr = ready_bars.shape[0]
    mma_warps: gl.constexpr = 4 if block_m == 64 else 8
    scheduler = _BatchedTileScheduler.initialize(
        16, 2048, 2048, block_m, block_n
    )
    consumer = _BarrierCounter.create(0, num_buffers)
    previous_index = gl.to_tensor(0)
    for tile_index in range(scheduler.get_num_tiles()):
        mma = _WGMMA.initialize(input_dtype, block_m, block_n, mma_warps)
        for k_offset in gl.static_range(0, 512, block_k):
            mbarrier.wait(ready_bars.index(consumer.index), consumer.phase)
            mma = mma.wait_num_outstanding(0)
            if k_offset != 0:
                mbarrier.arrive(empty_bars.index(previous_index), count=1)
            mma = mma.issue_async_mma(
                a_buffers.index(consumer.index),
                b_buffers.index(consumer.index),
                "ieee",
            )
            previous_index = consumer.index
            consumer = consumer.next()

        mma = mma.wait_num_outstanding(0)
        mbarrier.arrive(empty_bars.index(previous_index), count=1)
        result = mma.take_result()
        output_phase = tile_index & 1
        mbarrier.wait(c_empty_bar, output_phase)
        c_buffer.store(result.to(output_dtype))
        fence_async_shared()
        mbarrier.arrive(c_ready_bar, count=1)


@gluon.jit
def _matmul_sm90_ws_store(
    a_desc,
    b_desc,
    c_desc,
    batch,
    m,
    n,
    k,
    ready_bars,
    empty_bars,
    a_buffers,
    b_buffers,
    c_ready_bar,
    c_empty_bar,
    c_buffer,
):
    block_m: gl.constexpr = c_desc.block_type.shape[0]
    block_n: gl.constexpr = c_desc.block_type.shape[1]
    scheduler = _BatchedTileScheduler.initialize(
        16, 2048, 2048, block_m, block_n
    )
    for tile_index in range(scheduler.get_num_tiles()):
        batch_id, pid_m, pid_n = scheduler.get_tile(tile_index)
        c_off_m = batch_id * 2048 + pid_m * block_m
        off_n = pid_n * block_n
        output_phase = tile_index & 1
        mbarrier.wait(c_ready_bar, output_phase)
        tma.async_copy_shared_to_global(c_desc, [c_off_m, off_n], c_buffer)
        tma.store_wait(pendings=0)
        mbarrier.arrive(c_empty_bar, count=1)


@gluon.jit
def _matmul_sm90_ws_cube_load(
    a_desc,
    b_desc,
    c_desc,
    batch,
    m,
    n,
    k,
    ready_bars,
    empty_bars,
    a_buffers,
    b_buffers,
    c_ready_bar,
    c_empty_bar,
    c_buffer,
):
    block_m: gl.constexpr = a_desc.block_type.shape[0]
    block_n: gl.constexpr = b_desc.block_type.shape[1]
    block_k: gl.constexpr = a_desc.block_type.shape[1]
    num_buffers: gl.constexpr = ready_bars.shape[0]
    scheduler = _BatchedTileScheduler.initialize(
        16, 1024, 1024, block_m, block_n
    )
    producer = _BarrierCounter.create(0, num_buffers)
    for tile_index in range(scheduler.get_num_tiles()):
        batch_id, pid_m, pid_n = scheduler.get_tile(tile_index)
        a_off_m = batch_id * 1024 + pid_m * block_m
        b_off_k = batch_id * 1024
        off_n = pid_n * block_n
        for k_offset in gl.static_range(0, 1024, block_k):
            mbarrier.wait(empty_bars.index(producer.index), producer.phase)
            ready_bar = ready_bars.index(producer.index)
            mbarrier.expect(
                ready_bar,
                a_desc.block_type.nbytes + b_desc.block_type.nbytes,
            )
            tma.async_copy_global_to_shared(
                a_desc,
                [a_off_m, k_offset],
                ready_bar,
                a_buffers.index(producer.index),
            )
            tma.async_copy_global_to_shared(
                b_desc,
                [b_off_k + k_offset, off_n],
                ready_bar,
                b_buffers.index(producer.index),
            )
            producer = producer.next()


@gluon.jit
def _matmul_sm90_ws_cube_compute(
    a_desc,
    b_desc,
    c_desc,
    batch,
    m,
    n,
    k,
    ready_bars,
    empty_bars,
    a_buffers,
    b_buffers,
    c_ready_bar,
    c_empty_bar,
    c_buffer,
):
    block_m: gl.constexpr = c_desc.block_type.shape[0]
    block_n: gl.constexpr = c_desc.block_type.shape[1]
    block_k: gl.constexpr = a_desc.block_type.shape[1]
    input_dtype: gl.constexpr = a_desc.dtype
    output_dtype: gl.constexpr = c_desc.dtype
    num_buffers: gl.constexpr = ready_bars.shape[0]
    mma_warps: gl.constexpr = 4 if block_m == 64 else 8
    scheduler = _BatchedTileScheduler.initialize(
        16, 1024, 1024, block_m, block_n
    )
    consumer = _BarrierCounter.create(0, num_buffers)
    previous_index = gl.to_tensor(0)
    for tile_index in range(scheduler.get_num_tiles()):
        mma = _WGMMA.initialize(input_dtype, block_m, block_n, mma_warps)
        for k_offset in gl.static_range(0, 1024, block_k):
            mbarrier.wait(ready_bars.index(consumer.index), consumer.phase)
            mma = mma.issue_async_mma(
                a_buffers.index(consumer.index),
                b_buffers.index(consumer.index),
                "ieee",
            )
            if k_offset != 0:
                mma = mma.wait_num_outstanding(1)
                mbarrier.arrive(empty_bars.index(previous_index), count=1)
            previous_index = consumer.index
            consumer = consumer.next()

        mma = mma.wait_num_outstanding(0)
        mbarrier.arrive(empty_bars.index(previous_index), count=1)
        result = mma.take_result()
        output_phase = tile_index & 1
        mbarrier.wait(c_empty_bar, output_phase)
        c_buffer.store(result.to(output_dtype))
        fence_async_shared()
        mbarrier.arrive(c_ready_bar, count=1)


@gluon.jit
def _matmul_sm90_ws_cube_store(
    a_desc,
    b_desc,
    c_desc,
    batch,
    m,
    n,
    k,
    ready_bars,
    empty_bars,
    a_buffers,
    b_buffers,
    c_ready_bar,
    c_empty_bar,
    c_buffer,
):
    block_m: gl.constexpr = c_desc.block_type.shape[0]
    block_n: gl.constexpr = c_desc.block_type.shape[1]
    scheduler = _BatchedTileScheduler.initialize(
        16, 1024, 1024, block_m, block_n
    )
    for tile_index in range(scheduler.get_num_tiles()):
        batch_id, pid_m, pid_n = scheduler.get_tile(tile_index)
        c_off_m = batch_id * 1024 + pid_m * block_m
        off_n = pid_n * block_n
        output_phase = tile_index & 1
        mbarrier.wait(c_ready_bar, output_phase)
        tma.async_copy_shared_to_global(c_desc, [c_off_m, off_n], c_buffer)
        tma.store_wait(pendings=0)
        mbarrier.arrive(c_empty_bar, count=1)


@gluon.jit
def _matmul_sm90_fp8_ws_load(
    a_desc,
    b_desc,
    c_desc,
    batch,
    m,
    n,
    k,
    tma_bars,
    ready_bars,
    empty_bars,
    a_buffers,
    b_staging,
    b_wgmma_buffers,
    c_ready_bar,
    c_empty_bar,
    c_buffer,
):
    block_m: gl.constexpr = a_desc.block_type.shape[0]
    block_n: gl.constexpr = b_desc.block_type.shape[1]
    block_k: gl.constexpr = a_desc.block_type.shape[1]
    num_buffers: gl.constexpr = ready_bars.shape[0]
    copy_k: gl.constexpr = 8
    copy_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [1, 1], [1, 0]
    )
    gl.static_assert(
        block_k % copy_k == 0,
        "FP8 WS B proxy copy requires BLOCK_K divisible by 8",
    )
    gl.static_assert(
        block_n % 32 == 0,
        "FP8 WS B proxy copy requires BLOCK_N divisible by 32",
    )

    scheduler = _BatchedTileScheduler.initialize(batch, m, n, block_m, block_n)
    producer = _BarrierCounter.create(0, num_buffers)
    for tile_index in range(scheduler.get_num_tiles()):
        batch_id, pid_m, pid_n = scheduler.get_tile(tile_index)
        a_off_m = batch_id * m + pid_m * block_m
        b_off_k = batch_id * k
        off_n = pid_n * block_n
        for k_offset in range(0, k, block_k):
            mbarrier.wait(empty_bars.index(producer.index), producer.phase)
            tma_bar = tma_bars.index(producer.index)
            mbarrier.expect(
                tma_bar,
                a_desc.block_type.nbytes + b_desc.block_type.nbytes,
            )
            tma.async_copy_global_to_shared(
                a_desc,
                [a_off_m, k_offset],
                tma_bar,
                a_buffers.index(producer.index),
            )
            tma.async_copy_global_to_shared(
                b_desc,
                [b_off_k + k_offset, off_n],
                tma_bar,
                b_staging,
            )
            mbarrier.wait(tma_bar, producer.phase)
            b_wgmma = b_wgmma_buffers.index(producer.index)
            for copy_offset in gl.static_range(0, block_k, copy_k):
                values = b_staging.slice(copy_offset, copy_k, 0).load(
                    copy_layout
                )
                b_wgmma.slice(copy_offset, copy_k, 0).store(values)
            fence_async_shared()
            mbarrier.arrive(ready_bars.index(producer.index), count=1)
            producer = producer.next()


@gluon.jit
def _matmul_sm90_fp8_ws_compute(
    a_desc,
    b_desc,
    c_desc,
    batch,
    m,
    n,
    k,
    tma_bars,
    ready_bars,
    empty_bars,
    a_buffers,
    b_staging,
    b_wgmma_buffers,
    c_ready_bar,
    c_empty_bar,
    c_buffer,
):
    block_m: gl.constexpr = c_desc.block_type.shape[0]
    block_n: gl.constexpr = c_desc.block_type.shape[1]
    block_k: gl.constexpr = a_desc.block_type.shape[1]
    input_dtype: gl.constexpr = a_desc.dtype
    output_dtype: gl.constexpr = c_desc.dtype
    num_buffers: gl.constexpr = ready_bars.shape[0]
    mma_warps: gl.constexpr = 4 if block_m == 64 else 8
    scheduler = _BatchedTileScheduler.initialize(batch, m, n, block_m, block_n)
    consumer = _BarrierCounter.create(0, num_buffers)
    previous_index = gl.to_tensor(0)
    for tile_index in range(scheduler.get_num_tiles()):
        mma = _WGMMA.initialize(input_dtype, block_m, block_n, mma_warps)
        mbarrier.wait(ready_bars.index(consumer.index), consumer.phase)
        mma = mma.issue_async_mma(
            a_buffers.index(consumer.index),
            b_wgmma_buffers.index(consumer.index),
            "ieee",
        )
        previous_index = consumer.index
        consumer = consumer.next()
        for k_offset in range(block_k, k, block_k):
            mbarrier.wait(ready_bars.index(consumer.index), consumer.phase)
            mma = mma.issue_async_mma(
                a_buffers.index(consumer.index),
                b_wgmma_buffers.index(consumer.index),
                "ieee",
            )
            mma = mma.wait_num_outstanding(1)
            fence_async_shared()
            mbarrier.arrive(empty_bars.index(previous_index), count=1)
            previous_index = consumer.index
            consumer = consumer.next()

        mma = mma.wait_num_outstanding(0)
        fence_async_shared()
        mbarrier.arrive(empty_bars.index(previous_index), count=1)
        result = mma.take_result()
        output_phase = tile_index & 1
        mbarrier.wait(c_empty_bar, output_phase)
        c_buffer.store(result.to(output_dtype))
        fence_async_shared()
        mbarrier.arrive(c_ready_bar, count=1)


@gluon.jit
def _matmul_sm90_fp8_ws_store(
    a_desc,
    b_desc,
    c_desc,
    batch,
    m,
    n,
    k,
    tma_bars,
    ready_bars,
    empty_bars,
    a_buffers,
    b_staging,
    b_wgmma_buffers,
    c_ready_bar,
    c_empty_bar,
    c_buffer,
):
    block_m: gl.constexpr = c_desc.block_type.shape[0]
    block_n: gl.constexpr = c_desc.block_type.shape[1]
    scheduler = _BatchedTileScheduler.initialize(batch, m, n, block_m, block_n)
    for tile_index in range(scheduler.get_num_tiles()):
        batch_id, pid_m, pid_n = scheduler.get_tile(tile_index)
        c_off_m = batch_id * m + pid_m * block_m
        off_n = pid_n * block_n
        output_phase = tile_index & 1
        mbarrier.wait(c_ready_bar, output_phase)
        tma.async_copy_shared_to_global(c_desc, [c_off_m, off_n], c_buffer)
        tma.store_wait(pendings=0)
        mbarrier.arrive(c_empty_bar, count=1)


@gluon.jit
def _matmul_sm90_fp8_ws_kernel(
    a_desc,
    b_desc,
    c_desc,
    batch,
    m,
    n,
    k,
    num_buffers: gl.constexpr,
    num_warps: gl.constexpr,
):
    input_dtype: gl.constexpr = a_desc.dtype
    block_m: gl.constexpr = c_desc.block_type.shape[0]
    block_n: gl.constexpr = c_desc.block_type.shape[1]
    block_k: gl.constexpr = a_desc.block_type.shape[1]
    expected_warps: gl.constexpr = 4 if block_m == 64 else 8
    gl.static_assert(
        input_dtype.primitive_bitwidth == 8,
        "FP8 WS kernel requires 8-bit WGMMA inputs",
    )
    gl.static_assert(block_k == 64, "FP8 WS kernel requires BLOCK_K=64")
    gl.static_assert(num_buffers >= 2, "expected at least two TMA buffers")
    gl.static_assert(
        num_warps == expected_warps,
        "FP8 WS expects 4 warps for BLOCK_M=64 and 8 otherwise",
    )

    tma_bars = gl.allocate_shared_memory(
        gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout()
    )
    ready_bars = gl.allocate_shared_memory(
        gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout()
    )
    empty_bars = gl.allocate_shared_memory(
        gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout()
    )
    for index in gl.static_range(num_buffers):
        mbarrier.init(tma_bars.index(index), count=1)
        mbarrier.init(ready_bars.index(index), count=1)
        mbarrier.init(empty_bars.index(index), count=1)
        mbarrier.arrive(empty_bars.index(index), count=1)

    a_buffers = gl.allocate_shared_memory(
        input_dtype,
        [num_buffers] + a_desc.block_type.shape,
        a_desc.layout,
    )
    b_staging = gl.allocate_shared_memory(
        input_dtype, b_desc.block_type.shape, b_desc.layout
    )
    b_wgmma_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [block_k, block_n], input_dtype, transposed=True
    )
    b_wgmma_buffers = gl.allocate_shared_memory(
        input_dtype,
        [num_buffers, block_k, block_n],
        b_wgmma_layout,
    )
    c_ready_bar = gl.allocate_shared_memory(
        gl.int64, [1], mbarrier.MBarrierLayout()
    )
    c_empty_bar = gl.allocate_shared_memory(
        gl.int64, [1], mbarrier.MBarrierLayout()
    )
    mbarrier.init(c_ready_bar, count=1)
    mbarrier.init(c_empty_bar, count=1)
    mbarrier.arrive(c_empty_bar, count=1)
    c_buffer = gl.allocate_shared_memory(
        c_desc.dtype, c_desc.block_type.shape, c_desc.layout
    )
    args = (
        a_desc,
        b_desc,
        c_desc,
        batch,
        m,
        n,
        k,
        tma_bars,
        ready_bars,
        empty_bars,
        a_buffers,
        b_staging,
        b_wgmma_buffers,
        c_ready_bar,
        c_empty_bar,
        c_buffer,
    )
    gl.warp_specialize(
        [
            (_matmul_sm90_fp8_ws_compute, args),
            (_matmul_sm90_fp8_ws_load, args),
            (_matmul_sm90_fp8_ws_store, args),
        ],
        [1, 1],
        [24, 24],
    )


@gluon.jit
def _matmul_sm90_ws_kernel(
    a_desc,
    b_desc,
    c_desc,
    batch,
    m,
    n,
    k,
    num_buffers: gl.constexpr,
    precision: gl.constexpr,
    num_warps: gl.constexpr,
    cube_variant: gl.constexpr,
):
    input_dtype: gl.constexpr = a_desc.dtype
    block_m: gl.constexpr = c_desc.block_type.shape[0]
    expected_warps: gl.constexpr = 4 if block_m == 64 else 8
    gl.static_assert(num_buffers >= 2, "expected at least two TMA buffers")
    gl.static_assert(
        num_warps == expected_warps,
        "warp specialization expects one warpgroup per 64 M rows",
    )
    ready_bars = gl.allocate_shared_memory(
        gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout()
    )
    empty_bars = gl.allocate_shared_memory(
        gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout()
    )
    for index in gl.static_range(num_buffers):
        mbarrier.init(ready_bars.index(index), count=1)
        mbarrier.init(empty_bars.index(index), count=1)
        mbarrier.arrive(empty_bars.index(index), count=1)

    a_buffers = gl.allocate_shared_memory(
        input_dtype,
        [num_buffers] + a_desc.block_type.shape,
        a_desc.layout,
    )
    b_buffers = gl.allocate_shared_memory(
        input_dtype,
        [num_buffers] + b_desc.block_type.shape,
        b_desc.layout,
    )
    c_ready_bar = gl.allocate_shared_memory(
        gl.int64, [1], mbarrier.MBarrierLayout()
    )
    c_empty_bar = gl.allocate_shared_memory(
        gl.int64, [1], mbarrier.MBarrierLayout()
    )
    mbarrier.init(c_ready_bar, count=1)
    mbarrier.init(c_empty_bar, count=1)
    mbarrier.arrive(c_empty_bar, count=1)
    c_buffer = gl.allocate_shared_memory(
        c_desc.dtype, c_desc.block_type.shape, c_desc.layout
    )
    args = (
        a_desc,
        b_desc,
        c_desc,
        batch,
        m,
        n,
        k,
        ready_bars,
        empty_bars,
        a_buffers,
        b_buffers,
        c_ready_bar,
        c_empty_bar,
        c_buffer,
    )
    if cube_variant:
        gl.warp_specialize(
            [
                (_matmul_sm90_ws_cube_compute, args),
                (_matmul_sm90_ws_cube_load, args),
                (_matmul_sm90_ws_cube_store, args),
            ],
            [1, 1],
            [24, 24],
        )
    else:
        gl.warp_specialize(
            [
                (_matmul_sm90_ws_compute, args),
                (_matmul_sm90_ws_load, args),
                (_matmul_sm90_ws_store, args),
            ],
            [1, 1],
            [24, 24],
        )


@gluon.jit
def _matmul_sm90_kernel(
    a_desc,
    b_desc,
    c_desc,
    batch,
    m,
    n,
    k,
    num_buffers: gl.constexpr,
    precision: gl.constexpr,
    num_warps: gl.constexpr,
):
    block_m: gl.constexpr = c_desc.block_type.shape[0]
    block_n: gl.constexpr = c_desc.block_type.shape[1]
    block_k: gl.constexpr = a_desc.block_type.shape[1]
    input_dtype: gl.constexpr = a_desc.dtype
    output_dtype: gl.constexpr = c_desc.dtype

    gl.static_assert(num_buffers >= 2, "expected at least two TMA buffers")
    barriers = gl.allocate_shared_memory(
        gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout()
    )
    for index in gl.static_range(num_buffers):
        mbarrier.init(barriers.index(index), count=1)

    producer = 0
    consumer = 0
    scheduler = _BatchedTileScheduler.initialize(batch, m, n, block_m, block_n)
    for tile_index in range(scheduler.get_num_tiles()):
        batch_id, pid_m, pid_n = scheduler.get_tile(tile_index)
        a_off_m = batch_id * m + pid_m * block_m
        b_off_k = batch_id * k
        c_off_m = batch_id * m + pid_m * block_m
        off_n = pid_n * block_n

        mma = _WGMMA.initialize(input_dtype, block_m, block_n, num_warps)
        a_buffers = gl.allocate_shared_memory(
            input_dtype,
            [num_buffers] + a_desc.block_type.shape,
            a_desc.layout,
        )
        b_buffers = gl.allocate_shared_memory(
            input_dtype,
            [num_buffers] + b_desc.block_type.shape,
            b_desc.layout,
        )
        transpose_b: gl.constexpr = input_dtype.primitive_bitwidth == 8
        b_copy_layout: gl.constexpr = gl.BlockedLayout(
            [1, 1], [1, 32], [num_warps, 1], [1, 0]
        )
        if transpose_b:
            b_wgmma_layout: gl.constexpr = (
                gl.NVMMASharedLayout.get_default_for(
                    b_desc.block_type.shape,
                    input_dtype,
                    transposed=True,
                )
            )
            b_wgmma = gl.allocate_shared_memory(
                input_dtype,
                b_desc.block_type.shape,
                b_wgmma_layout,
            )
        else:
            b_wgmma = b_buffers.index(0)

        for k_offset in gl.static_range(
            0, block_k * (num_buffers - 2), block_k
        ):
            producer = _issue_loads(
                producer,
                a_desc,
                b_desc,
                a_off_m,
                b_off_k,
                off_n,
                k_offset,
                barriers,
                a_buffers,
                b_buffers,
                num_buffers,
            )

        for k_offset in range(block_k * (num_buffers - 2), k, block_k):
            producer = _issue_loads(
                producer,
                a_desc,
                b_desc,
                a_off_m,
                b_off_k,
                off_n,
                k_offset,
                barriers,
                a_buffers,
                b_buffers,
                num_buffers,
            )
            consumer, mma = _issue_mma(
                consumer,
                mma,
                barriers,
                a_buffers,
                b_buffers,
                b_wgmma,
                b_copy_layout,
                transpose_b,
                num_buffers,
                precision,
            )

        for _ in gl.static_range(num_buffers - 2):
            consumer, mma = _issue_mma(
                consumer,
                mma,
                barriers,
                a_buffers,
                b_buffers,
                b_wgmma,
                b_copy_layout,
                transpose_b,
                num_buffers,
                precision,
            )

        mma = mma.wait_num_outstanding(0)
        result = mma.take_result()
        c_shared = gl.allocate_shared_memory(
            output_dtype, c_desc.block_type.shape, c_desc.layout
        )
        c_shared.store(result.to(output_dtype))
        fence_async_shared()
        tma.async_copy_shared_to_global(c_desc, [c_off_m, off_n], c_shared)
        tma.store_wait(pendings=0)


@gluon.jit
def _matmul_sm90_id7_serial_kernel(
    a_desc,
    b_desc,
    c_desc,
    batch,
    m,
    n,
    k,
    num_buffers: gl.constexpr,
    precision: gl.constexpr,
    num_warps: gl.constexpr,
):
    block_m: gl.constexpr = c_desc.block_type.shape[0]
    block_n: gl.constexpr = c_desc.block_type.shape[1]
    block_k: gl.constexpr = a_desc.block_type.shape[1]
    input_dtype: gl.constexpr = a_desc.dtype
    output_dtype: gl.constexpr = c_desc.dtype

    gl.static_assert(block_m == 128, "ID7 serial kernel requires BLOCK_M=128")
    gl.static_assert(block_n == 256, "ID7 serial kernel requires BLOCK_N=256")
    gl.static_assert(block_k == 64, "ID7 serial kernel requires BLOCK_K=64")
    gl.static_assert(num_buffers == 3, "ID7 serial kernel requires 3 buffers")
    gl.static_assert(num_warps == 8, "ID7 serial kernel requires 8 warps")

    barriers = gl.allocate_shared_memory(
        gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout()
    )
    for index in gl.static_range(num_buffers):
        mbarrier.init(barriers.index(index), count=1)

    producer = _BarrierCounter.create(0, num_buffers)
    consumer = _BarrierCounter.create(0, num_buffers)
    scheduler = _BatchedTileScheduler.initialize(
        32, 1024, 1024, block_m, block_n
    )
    for tile_index in range(scheduler.get_num_tiles()):
        batch_id, pid_m, pid_n = scheduler.get_tile(tile_index)
        a_off_m = batch_id * 1024 + pid_m * block_m
        b_off_k = batch_id * 4096
        c_off_m = batch_id * 1024 + pid_m * block_m
        off_n = pid_n * block_n

        mma = _WGMMA.initialize(input_dtype, block_m, block_n, num_warps)
        a_buffers = gl.allocate_shared_memory(
            input_dtype,
            [num_buffers] + a_desc.block_type.shape,
            a_desc.layout,
        )
        b_buffers = gl.allocate_shared_memory(
            input_dtype,
            [num_buffers] + b_desc.block_type.shape,
            b_desc.layout,
        )

        producer = _issue_counter_loads(
            producer,
            a_desc,
            b_desc,
            a_off_m,
            b_off_k,
            off_n,
            0,
            barriers,
            a_buffers,
            b_buffers,
        )
        for k_offset in range(64, 4096, 64):
            producer = _issue_counter_loads(
                producer,
                a_desc,
                b_desc,
                a_off_m,
                b_off_k,
                off_n,
                k_offset,
                barriers,
                a_buffers,
                b_buffers,
            )
            consumer, mma = _issue_counter_mma(
                consumer,
                mma,
                barriers,
                a_buffers,
                b_buffers,
                precision,
            )

        consumer, mma = _issue_counter_mma(
            consumer,
            mma,
            barriers,
            a_buffers,
            b_buffers,
            precision,
        )
        mma = mma.wait_num_outstanding(0)
        result = mma.take_result()
        c_shared = gl.allocate_shared_memory(
            output_dtype, c_desc.block_type.shape, c_desc.layout
        )
        c_shared.store(result.to(output_dtype))
        fence_async_shared()
        tma.async_copy_shared_to_global(c_desc, [c_off_m, off_n], c_shared)
        tma.store_wait(pendings=0)


@gluon.jit
def _matmul_sm90_dynamic_output_kernel(
    a_desc,
    b_desc,
    c_ptr,
    batch,
    m,
    n,
    k,
    num_buffers: gl.constexpr,
    block_m: gl.constexpr,
    block_n: gl.constexpr,
    block_k: gl.constexpr,
    precision: gl.constexpr,
    num_warps: gl.constexpr,
    warp_specialized: gl.constexpr,
    cube_variant: gl.constexpr,
    use_id7_serial: gl.constexpr,
):
    """Run a bound-input SM90 GEMM with a functional output pointer.

    A and B descriptors are stable for a compiled graph and remain host-side.
    Only C changes for ``compiled.run`` calls, so create its descriptor on the
    device instead of rebuilding all three host descriptors on every replay.
    """
    output_dtype: gl.constexpr = c_ptr.dtype.element_ty
    c_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [block_m, block_n], output_dtype
    )
    c_desc = tma.make_tensor_descriptor(
        c_ptr,
        [batch * m, n],
        [n, 1],
        [block_m, block_n],
        c_layout,
    )
    if warp_specialized:
        _matmul_sm90_ws_kernel(
            a_desc,
            b_desc,
            c_desc,
            batch,
            m,
            n,
            k,
            num_buffers,
            precision,
            num_warps,
            cube_variant,
        )
    elif use_id7_serial:
        _matmul_sm90_id7_serial_kernel(
            a_desc,
            b_desc,
            c_desc,
            batch,
            m,
            n,
            k,
            num_buffers,
            precision,
            num_warps,
        )
    else:
        _matmul_sm90_kernel(
            a_desc,
            b_desc,
            c_desc,
            batch,
            m,
            n,
            k,
            num_buffers,
            precision,
            num_warps,
        )
