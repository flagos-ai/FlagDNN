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

"""Ascend-only Triton kernels for dense batched matrix multiplication."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import triton
import triton.language as tl

from flag_dnn.graph.prepared import (
    RunFn,
    get_prepared_output,
    runtime_tensor_checks_from_specs,
    runtime_tensor_checks_pass,
)
from flag_dnn.graph.prepared.common import (
    _is_runtime_device_spec,
    _static_shape,
)
from flag_dnn.graph.tensor import TensorSpec, torch_dtype
from flag_dnn.ops.matmul import (
    _resolve_matmul_compute_mode,
    _resolve_matmul_out_dtype,
)
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils.libentry import libentry


@libentry()
@triton.jit
def _batched_matmul_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    TF32: tl.constexpr,
):
    program_id = tl.program_id(0)
    programs_m = tl.cdiv(M, BLOCK_M)
    programs_n = tl.cdiv(N, BLOCK_N)
    programs_per_batch = programs_m * programs_n
    batch_id = program_id // programs_per_batch
    tile_id = program_id - batch_id * programs_per_batch

    programs_per_group = GROUP_M * programs_n
    group_id = tile_id // programs_per_group
    first_program_m = group_id * GROUP_M
    group_size_m = tl.minimum(programs_m - first_program_m, GROUP_M)
    tile_in_group = tile_id - group_id * programs_per_group
    program_m = first_program_m + tile_in_group % group_size_m
    program_n = tile_in_group // group_size_m

    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    a_ptrs = (
        a_ptr + batch_id * M * K + offsets_m[:, None] * K + offsets_k[None, :]
    )
    b_ptrs = (
        b_ptr + batch_id * K * N + offsets_k[:, None] * N + offsets_n[None, :]
    )

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in tl.range(0, K, BLOCK_K):
        if M % BLOCK_M == 0 and K % BLOCK_K == 0:
            a = tl.load(a_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=(offsets_m[:, None] < M)
                & (k_start + offsets_k[None, :] < K),
                other=0.0,
            )
        if K % BLOCK_K == 0 and N % BLOCK_N == 0:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(
                b_ptrs,
                mask=(k_start + offsets_k[:, None] < K)
                & (offsets_n[None, :] < N),
                other=0.0,
            )
        if TF32:
            accumulator = tl.dot(
                a,
                b,
                accumulator,
                input_precision="tf32",
            )
        else:
            accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    output_ptrs = (
        output_ptr
        + batch_id * M * N
        + offsets_m[:, None] * N
        + offsets_n[None, :]
    )
    output = accumulator.to(output_ptr.dtype.element_ty)
    if M % BLOCK_M == 0 and N % BLOCK_N == 0:
        tl.store(output_ptrs, output)
    else:
        tl.store(
            output_ptrs,
            output,
            mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
        )


@triton.jit
def _compute_batched_matmul_tile(
    a_ptr,
    b_ptr,
    output_ptr,
    batch_id,
    tile_id,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    TF32: tl.constexpr,
):
    programs_m: tl.constexpr = M // BLOCK_M
    programs_n: tl.constexpr = N // BLOCK_N
    programs_per_group: tl.constexpr = GROUP_M * programs_n
    group_id = tile_id // programs_per_group
    first_program_m = group_id * GROUP_M
    group_size_m = tl.minimum(
        programs_m - first_program_m,
        GROUP_M,
    )
    tile_in_group = tile_id - group_id * programs_per_group
    program_m = first_program_m + tile_in_group % group_size_m
    program_n = tile_in_group // group_size_m

    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_m = tl.max_contiguous(
        tl.multiple_of(offsets_m, BLOCK_M),
        BLOCK_M,
    )
    offsets_n = tl.max_contiguous(
        tl.multiple_of(offsets_n, BLOCK_N),
        BLOCK_N,
    )
    offsets_k = tl.arange(0, BLOCK_K)
    a_ptrs = (
        a_ptr + batch_id * M * K + offsets_m[:, None] * K + offsets_k[None, :]
    )
    b_ptrs = (
        b_ptr + batch_id * K * N + offsets_k[:, None] * N + offsets_n[None, :]
    )
    accumulator = tl.zeros(
        (BLOCK_M, BLOCK_N),
        dtype=tl.float32,
    )
    for _ in tl.range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        if TF32:
            accumulator = tl.dot(
                a,
                b,
                accumulator,
                input_precision="tf32",
            )
        else:
            accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    output_ptrs = (
        output_ptr
        + batch_id * M * N
        + offsets_m[:, None] * N
        + offsets_n[None, :]
    )
    tl.store(
        output_ptrs,
        accumulator.to(output_ptr.dtype.element_ty),
    )


@libentry()
@triton.jit
def _multitile_batched_matmul_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
    TF32: tl.constexpr,
):
    program_id = tl.program_id(0)
    programs_m: tl.constexpr = M // BLOCK_M
    programs_n: tl.constexpr = N // BLOCK_N
    tiles_per_batch: tl.constexpr = programs_m * programs_n
    programs_per_batch: tl.constexpr = tiles_per_batch // TILES_PER_PROGRAM
    batch_id = program_id // programs_per_batch
    batch_program = program_id - batch_id * programs_per_batch
    first_tile = batch_program * TILES_PER_PROGRAM

    if TILES_PER_PROGRAM == 2:
        for tile_offset in range(0, TILES_PER_PROGRAM):
            _compute_batched_matmul_tile(
                a_ptr,
                b_ptr,
                output_ptr,
                batch_id,
                first_tile + tile_offset,
                M,
                N,
                K,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                GROUP_M,
                TF32,
            )
    else:
        for tile_offset in tl.range(
            0,
            TILES_PER_PROGRAM,
            disallow_acc_multi_buffer=True,
        ):
            _compute_batched_matmul_tile(
                a_ptr,
                b_ptr,
                output_ptr,
                batch_id,
                first_tile + tile_offset,
                M,
                N,
                K,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                GROUP_M,
                TF32,
            )


_MULTITILE_LOW_PRECISION_SHAPES = {
    (16, 1024, 1024, 1024): 2,
    (16, 2048, 512, 2048): 8,
    (32, 1024, 4096, 1024): 2,
    (32, 512, 512, 512): 4,
    (4, 4096, 4096, 4096): 2,
    (8, 2048, 2048, 2048): 2,
}

_MULTITILE_TF32_SHAPES = {
    (32, 512, 512, 512): 4,
    (16, 2048, 512, 2048): 8,
    (4, 4096, 4096, 4096): 2,
}

_BLOCK_K_256_LOW_PRECISION_SHAPES = {
    (16, 1024, 1024, 1024),
    (16, 2048, 512, 2048),
    (32, 1024, 4096, 1024),
    (32, 512, 512, 512),
    (4, 4096, 4096, 4096),
    (8, 2048, 2048, 2048),
}

_LEGACY_SYNC_SOLVER_LOW_PRECISION_SHAPES = {
    (16, 2048, 512, 2048),
    (4, 4096, 4096, 4096),
}


def _select_blocks(
    m: int,
    n: int,
    *,
    tf32: bool,
) -> tuple[int, int, int, int]:
    if m <= 32 or n <= 32:
        return 32, 32, 32, 1
    if tf32:
        return 256, 128, 128, 4
    return 128, 256, 128, 4


def prepare_matmul(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    """Prepare the dense Ascend path without changing other backends."""
    if len(input_specs) != 2 or not all(
        _is_runtime_device_spec(spec) for spec in input_specs
    ):
        return None
    a_spec, b_spec = input_specs
    a_shape = _static_shape(a_spec)
    b_shape = _static_shape(b_spec)
    if (
        a_shape is None
        or b_shape is None
        or len(a_shape) != 3
        or len(b_shape) != 3
        or a_shape[0] != b_shape[0]
        or a_shape[2] != b_shape[1]
        or not bool(a_spec.contiguous)
        or not bool(b_spec.contiguous)
        or a_spec.dtype != b_spec.dtype
        or a_spec.dtype not in ("float16", "bfloat16", "float32")
    ):
        return None

    checks = runtime_tensor_checks_from_specs(
        input_specs,
        (0, 1),
        require_shape=True,
        require_stride=True,
        require_dtype=True,
    )
    if checks is None:
        return None

    input_dtype = torch_dtype(a_spec.dtype)
    requested_output_dtype = attrs.get("out_dtype")
    output_dtype = _resolve_matmul_out_dtype(
        input_dtype,
        (
            torch_dtype(requested_output_dtype)
            if requested_output_dtype is not None
            else None
        ),
    )
    compute_mode = _resolve_matmul_compute_mode(
        input_dtype,
        attrs.get("compute_data_type"),
    )
    if (
        output_dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or (input_dtype == torch.float32 and compute_mode != "tf32")
        or (input_dtype != torch.float32 and compute_mode != "float32")
    ):
        return None

    output_shape = (a_shape[0], a_shape[1], b_shape[2])
    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def run(inputs: Sequence[Any], run_attrs: dict[str, Any]) -> Any:
        if (
            not runtime_tensor_checks_pass(inputs, checks)
            or len(inputs) != 2
            or not isinstance(inputs[0], torch.Tensor)
            or not isinstance(inputs[1], torch.Tensor)
        ):
            return default_run_fn(inputs, run_attrs)
        a, b = inputs
        cache_key = (
            a.device.type,
            a.device.index,
            output_dtype,
            output_shape,
        )
        output = get_prepared_output(
            output_cache,
            cache_key,
            lambda: torch.empty(
                output_shape,
                device=a.device,
                dtype=output_dtype,
            ),
        )
        with torch_device_fn.device(a.device):
            if not matmul_3d_out(
                a,
                b,
                output,
                compute_mode=compute_mode,
            ):
                return default_run_fn(inputs, run_attrs)
        return output

    return run


def matmul_3d_out(
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    *,
    compute_mode: str,
) -> bool:
    """Launch the Ascend kernel when the dense rank-3 contract is supported."""
    if (
        a.device.type != "npu"
        or b.device != a.device
        or output.device != a.device
        or a.dim() != 3
        or b.dim() != 3
        or output.dim() != 3
        or not a.is_contiguous()
        or not b.is_contiguous()
        or not output.is_contiguous()
        or a.dtype != b.dtype
        or a.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or output.dtype
        not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        )
    ):
        return False

    batch, m, k = (int(value) for value in a.shape)
    b_batch, b_k, n = (int(value) for value in b.shape)
    if (
        batch != b_batch
        or k != b_k
        or tuple(output.shape) != (batch, m, n)
        or compute_mode not in ("float32", "ieee", "tf32")
        or (a.dtype == torch.float32 and compute_mode != "tf32")
        or (a.dtype != torch.float32 and compute_mode != "float32")
    ):
        return False
    if output.numel() == 0:
        return True

    block_m, block_n, block_k, group_m = _select_blocks(
        m,
        n,
        tf32=a.dtype == torch.float32,
    )
    shape_key = (batch, m, k, n)
    if (
        a.dtype in (torch.float16, torch.bfloat16)
        and output.dtype == a.dtype
        and shape_key in _BLOCK_K_256_LOW_PRECISION_SHAPES
    ):
        block_k = 256
    if a.dtype in (torch.float16, torch.bfloat16) and output.dtype == a.dtype:
        tiles_per_program = _MULTITILE_LOW_PRECISION_SHAPES.get(shape_key, 1)
    elif a.dtype == torch.float32 and output.dtype == torch.float32:
        tiles_per_program = _MULTITILE_TF32_SHAPES.get(
            shape_key,
            1,
        )
    else:
        tiles_per_program = 1
    total_tiles = batch * triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
    if tiles_per_program > 1:
        tiles_per_batch = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
        if tiles_per_batch % tiles_per_program != 0:
            return False
        launch_options: dict[str, bool] = {}
        if (
            a.dtype in (torch.float16, torch.bfloat16)
            and output.dtype == a.dtype
            and shape_key in _LEGACY_SYNC_SOLVER_LOW_PRECISION_SHAPES
        ):
            launch_options["sync_solver"] = False
        _multitile_batched_matmul_kernel[(total_tiles // tiles_per_program,)](
            a,
            b,
            output,
            m,
            n,
            k,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=group_m,
            TILES_PER_PROGRAM=tiles_per_program,
            TF32=a.dtype == torch.float32,
            num_warps=8,
            num_stages=2,
            **launch_options,
        )
        return True

    _batched_matmul_kernel[(total_tiles,)](
        a,
        b,
        output,
        m,
        n,
        k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        TF32=a.dtype == torch.float32,
        num_warps=4 if m <= 32 or n <= 32 else 8,
        num_stages=2,
    )
    return True


__all__ = ("matmul_3d_out", "prepare_matmul")
