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

from typing import Callable

import torch
import triton

from flag_dnn.ops.matmul import _batched_matmul_kernel
from flag_dnn.runtime import torch_device_fn

_CONFIGS = {
    (4, 16, 24, 32): (16, 32, 32, 8, 4, 3),
    (8, 32, 32, 64): (32, 32, 32, 8, 4, 3),
}


def _metadata(
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[
    tuple[int, int, int, int],
    tuple[int, int, int, int, int, int],
]:
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("small SM90 TF32 matmul requires rank-3 inputs")
    batch, m, k = map(int, a.shape)
    b_batch, b_k, n = map(int, b.shape)
    dimensions = (batch, m, n, k)
    config = _CONFIGS.get(dimensions)
    if config is None or b_batch != batch or b_k != k:
        raise ValueError(
            "small SM90 TF32 matmul received an unsupported shape"
        )
    if (
        a.dtype != torch.float32
        or b.dtype != torch.float32
        or a.device.type != "cuda"
        or b.device != a.device
        or not a.is_contiguous()
        or not b.is_contiguous()
    ):
        raise ValueError(
            "small SM90 TF32 matmul requires contiguous FP32 CUDA inputs"
        )
    return dimensions, config


def _validate_output(
    output: torch.Tensor,
    a: torch.Tensor,
    dimensions: tuple[int, int, int, int],
) -> None:
    batch, m, n, _k = dimensions
    if (
        tuple(output.shape) != (batch, m, n)
        or output.dtype != torch.float32
        or output.device != a.device
        or not output.is_contiguous()
    ):
        raise ValueError("small SM90 TF32 matmul received an invalid output")


def _launch_arguments(
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    dimensions: tuple[int, int, int, int],
    config: tuple[int, int, int, int, int, int],
) -> tuple[object, ...]:
    _batch, m, n, k = dimensions
    block_m, block_n, block_k, group_m, _warps, _stages = config
    return (
        a,
        b,
        output,
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        group_m,
        True,
    )


def prepare_small_tf32_matmul_dynamic_output(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if output_dtype != torch.float32:
        raise TypeError("small SM90 TF32 matmul requires FP32 output")
    dimensions, config = _metadata(a, b)
    batch, m, n, _k = dimensions
    block_m, block_n, block_k, group_m, warps, stages = config
    grid = (
        triton.cdiv(m, block_m) * triton.cdiv(n, block_n),
        batch,
    )
    static_grid = (grid[0], grid[1], 1)
    kernel = _batched_matmul_kernel.fn.fn
    cached_runner = None

    def launch(output: torch.Tensor) -> torch.Tensor:
        nonlocal cached_runner
        _validate_output(output, a, dimensions)
        launch_args = _launch_arguments(a, b, output, dimensions, config)
        with torch_device_fn.device(a.device):
            runner = cached_runner
            if runner is None:
                compiled = kernel[grid](
                    a,
                    b,
                    output,
                    m,
                    n,
                    dimensions[3],
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    GROUP_M=group_m,
                    ROUND_F32_TO_TF32=True,
                    num_warps=warps,
                    num_stages=stages,
                )
                cached_runner = compiled[static_grid]
            else:
                runner(*launch_args)
        return output

    return launch


def run_small_tf32_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    launcher = prepare_small_tf32_matmul_dynamic_output(
        a,
        b,
        output_dtype=output.dtype,
    )
    return launcher(output)


__all__ = (
    "prepare_small_tf32_matmul_dynamic_output",
    "run_small_tf32_matmul",
)
