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

from typing import TYPE_CHECKING, Callable

import torch

from flag_dnn.ops.matmul import _batched_matmul_persistent_kernel
from flag_dnn.runtime import torch_device_fn

if TYPE_CHECKING:
    from flag_dnn.runtime.backend._nvidia.ops.matmul_sm90 import (
        Sm90MatmulConfig,
    )


_SHAPE = (32, 512, 512, 512)
_ACTIVE_CTAS = 128


def _validate_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    output_dtype: torch.dtype,
    config: "Sm90MatmulConfig",
) -> None:
    batch, m, k = map(int, a.shape)
    b_batch, b_k, n = map(int, b.shape)
    if (batch, m, n, k) != _SHAPE or (b_batch, b_k) != (batch, k):
        raise ValueError(
            "SM90 persistent matmul received an unsupported shape"
        )
    if (
        a.dtype not in (torch.float16, torch.bfloat16)
        or b.dtype != a.dtype
        or output_dtype != a.dtype
        or a.device.type != "cuda"
        or b.device != a.device
        or not a.is_contiguous()
        or not b.is_contiguous()
    ):
        raise ValueError(
            "SM90 persistent matmul requires contiguous FP16 or BF16 CUDA "
            "tensors"
        )
    expected_config = (128, 256, 64, 4, 8, 232)
    actual_config = (
        config.block_m,
        config.block_n,
        config.block_k,
        config.num_buffers,
        config.num_warps,
        config.maxnreg,
    )
    if config.family != "lowp_persistent" or actual_config != expected_config:
        raise ValueError("SM90 persistent matmul received an invalid config")


def _validate_output(
    output: torch.Tensor,
    a: torch.Tensor,
) -> None:
    if (
        tuple(output.shape) != _SHAPE[:3]
        or output.dtype != a.dtype
        or output.device != a.device
        or not output.is_contiguous()
    ):
        raise ValueError("SM90 persistent matmul received an invalid output")


def prepare_persistent_sm90_matmul_dynamic_output(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    output_dtype: torch.dtype,
    config: "Sm90MatmulConfig",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Bind inputs to the tuned two-wave H100 persistent GEMM."""
    _validate_inputs(a, b, output_dtype=output_dtype, config=config)
    batch, m, n, k = _SHAPE
    grid = (_ACTIVE_CTAS,)
    static_grid = (_ACTIVE_CTAS, 1, 1)
    kernel = _batched_matmul_persistent_kernel.fn.fn
    cached_runner = None

    def launch(output: torch.Tensor) -> torch.Tensor:
        nonlocal cached_runner
        _validate_output(output, a)
        launch_args = (
            a,
            b,
            output,
            m,
            n,
            k,
            batch,
            _ACTIVE_CTAS,
            config.block_m,
            config.block_n,
            config.block_k,
            4,
        )
        with torch_device_fn.device(a.device):
            runner = cached_runner
            if runner is None:
                compiled = kernel[grid](
                    a,
                    b,
                    output,
                    m,
                    n,
                    k,
                    batch,
                    _ACTIVE_CTAS,
                    BLOCK_M=config.block_m,
                    BLOCK_N=config.block_n,
                    BLOCK_K=config.block_k,
                    GROUP_M=4,
                    num_warps=config.num_warps,
                    num_stages=config.num_buffers,
                    maxnreg=config.maxnreg,
                )
                cached_runner = compiled[static_grid]
            else:
                runner(*launch_args)
        return output

    return launch


def run_persistent_sm90_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    *,
    config: "Sm90MatmulConfig",
) -> torch.Tensor:
    launcher = prepare_persistent_sm90_matmul_dynamic_output(
        a,
        b,
        output_dtype=output.dtype,
        config=config,
    )
    return launcher(output)


__all__ = (
    "prepare_persistent_sm90_matmul_dynamic_output",
    "run_persistent_sm90_matmul",
)
