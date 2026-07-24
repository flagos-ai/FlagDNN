# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils.device_info import get_sm_count_for

if TYPE_CHECKING:
    from flag_dnn.runtime.backend._nvidia.ops.matmul_sm90 import (
        Sm90MatmulConfig,
    )

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_FP8_SM90_CONFIGS = runtime.get_tuned_config("matmul_fp8_sm90")
_FP8_TRANSPOSE_SM90_CONFIGS = runtime.get_tuned_config(
    "matmul_fp8_transpose_sm90"
)
_TMA_ALLOCATOR_SET = False
_TRANSPOSED_B_SHAPES = {
    (16, 1024, 1024, 1024),
    (8, 2048, 2048, 2048),
    (4, 4096, 4096, 4096),
    (32, 1024, 1024, 4096),
}


def _triton_tma_alloc(size: int, alignment: int, stream: Optional[int]):
    del alignment, stream
    return torch.empty(size, device="cuda", dtype=torch.int8)


def _ensure_triton_tma_allocator() -> None:
    global _TMA_ALLOCATOR_SET
    if not _TMA_ALLOCATOR_SET:
        triton.set_allocator(_triton_tma_alloc)
        _TMA_ALLOCATOR_SET = True


def _uses_transposed_b(
    dimensions: tuple[int, int, int, int],
) -> bool:
    return dimensions in _TRANSPOSED_B_SHAPES


@libentry()
@libtuner(
    configs=_FP8_TRANSPOSE_SM90_CONFIGS,
    key=["N", "K"],
    warmup=5,
    rep=10,
)
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


@libentry()
@libtuner(
    configs=_FP8_SM90_CONFIGS,
    key=["BATCH", "M", "N", "K"],
    warmup=5,
    rep=10,
)
@triton.jit
def _batched_fp8_tma_persistent_kernel(
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
    GROUP_M: tl.constexpr,
    GRID_MULTIPLIER: tl.constexpr,
    TRANSPOSED_B: tl.constexpr,
):
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[BATCH * M, K],
        strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    if TRANSPOSED_B:
        b_desc = tl.make_tensor_descriptor(
            b_ptr,
            shape=[BATCH * N, K],
            strides=[K, 1],
            block_shape=[BLOCK_N, BLOCK_K],
        )
    else:
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
        block_shape=[BLOCK_M, BLOCK_N],
    )
    start_pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    tiles_per_batch = num_pid_m * num_pid_n
    total_tiles = BATCH * tiles_per_batch
    num_pid_in_group = GROUP_M * num_pid_n

    for flat_tile in tl.range(start_pid, total_tiles, num_ctas, flatten=True):
        batch_id = flat_tile // tiles_per_batch
        tile_id = flat_tile - batch_id * tiles_per_batch
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_M
        )
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for off_k in range(0, K, BLOCK_K):
            a = a_desc.load([batch_id * M + off_m, off_k])
            if TRANSPOSED_B:
                b = tl.trans(b_desc.load([batch_id * N + off_n, off_k]))
            else:
                b = b_desc.load([batch_id * K + off_k, off_n])
            accumulator = tl.dot(
                a,
                b,
                accumulator,
                input_precision="ieee",
            )
        c_desc.store(
            [batch_id * M + off_m, off_n],
            accumulator.to(c_ptr.dtype.element_ty),
        )


def _validate_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    output_dtype: torch.dtype,
    config: "Sm90MatmulConfig",
) -> tuple[int, int, int, int]:
    if config.family != "fp8_tma":
        raise ValueError("expected an FP8 TMA SM90 matmul configuration")
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("FP8 TMA SM90 matmul requires rank-3 inputs")
    batch, m, k = map(int, a.shape)
    b_batch, b_k, n = map(int, b.shape)
    if batch != b_batch or k != b_k:
        raise ValueError("FP8 TMA SM90 matmul input shapes do not match")
    if (
        a.dtype not in _FP8_DTYPES
        or b.dtype != a.dtype
        or output_dtype != torch.float32
    ):
        raise TypeError(
            "FP8 TMA SM90 matmul requires FP8 inputs and FP32 output"
        )
    if (
        a.device.type != "cuda"
        or b.device != a.device
        or not a.is_contiguous()
        or not b.is_contiguous()
    ):
        raise ValueError("FP8 TMA SM90 matmul requires contiguous CUDA inputs")
    if m % config.block_m or n % config.block_n or k % config.block_k:
        raise ValueError(
            "FP8 TMA SM90 dimensions must align to its tile sizes"
        )
    return batch, m, n, k


def _launch(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    dimensions: tuple[int, int, int, int],
    config: "Sm90MatmulConfig",
    transposed_b: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch, m, n, k = dimensions
    if (
        tuple(c.shape) != (batch, m, n)
        or c.dtype != torch.float32
        or c.device != a.device
        or not c.is_contiguous()
    ):
        raise ValueError("FP8 TMA SM90 matmul received an invalid output")
    use_transposed_b = _uses_transposed_b(dimensions)
    if use_transposed_b:
        if transposed_b is None:
            transposed_b = torch.empty(
                (batch, n, k), device=b.device, dtype=b.dtype
            )
        if (
            tuple(transposed_b.shape) != (batch, n, k)
            or transposed_b.dtype != b.dtype
            or transposed_b.device != b.device
            or not transposed_b.is_contiguous()
        ):
            raise ValueError(
                "FP8 TMA SM90 matmul received an invalid B transpose"
            )
    sm_count = get_sm_count_for(a.device)

    def grid(meta):
        tiles = (
            batch
            * triton.cdiv(m, meta["BLOCK_M"])
            * triton.cdiv(n, meta["BLOCK_N"])
        )
        return (min(tiles, sm_count * meta["GRID_MULTIPLIER"]),)

    _ensure_triton_tma_allocator()
    with torch_device_fn.device(a.device):
        if use_transposed_b:
            assert transposed_b is not None

            def transpose_grid(meta):
                return (
                    triton.cdiv(k, meta["BLOCK"]),
                    triton.cdiv(n, meta["BLOCK"]),
                    batch,
                )

            _transpose_b_kernel[transpose_grid](b, transposed_b, n, k)
        _batched_fp8_tma_persistent_kernel[grid](
            a,
            transposed_b if use_transposed_b else b,
            c,
            batch,
            m,
            n,
            k,
            TRANSPOSED_B=use_transposed_b,
        )
    return c


def run_sm90_fp8_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    config: "Sm90MatmulConfig",
) -> torch.Tensor:
    dimensions = _validate_inputs(a, b, c.dtype, config)
    return _launch(a, b, c, dimensions, config)


def prepare_sm90_fp8_matmul_dynamic_output(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    output_dtype: torch.dtype,
    config: "Sm90MatmulConfig",
) -> Callable[[torch.Tensor], torch.Tensor]:
    dimensions = _validate_inputs(a, b, output_dtype, config)
    batch, _m, n, k = dimensions
    transposed_b = (
        torch.empty((batch, n, k), device=b.device, dtype=b.dtype)
        if _uses_transposed_b(dimensions)
        else None
    )

    def launch(output: torch.Tensor) -> torch.Tensor:
        return _launch(
            a,
            b,
            output,
            dimensions,
            config,
            transposed_b=transposed_b,
        )

    return launch


__all__ = (
    "prepare_sm90_fp8_matmul_dynamic_output",
    "run_sm90_fp8_matmul",
)
