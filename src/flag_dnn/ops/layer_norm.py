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

import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("layer_norm"),
    key=["N"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def layer_norm_kernel(
    x_ptr,
    y_ptr,
    mean_ptr,
    rstd_ptr,
    weight_ptr,
    bias_ptr,
    M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RETURN_STATS: tl.constexpr,
):
    row_idx = tle.program_id(0)

    x_row_ptr = x_ptr + row_idx * N
    y_row_ptr = y_ptr + row_idx * N

    sum_x = 0.0
    sum_x2 = 0.0
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    # 均值和方差
    mean = sum_x / N
    var = (sum_x2 / N) - (mean * mean)
    # 防浮点精度越界产生负数
    var = tl.maximum(var, 0.0)
    rstd = 1.0 / tl.sqrt(var + eps)
    if RETURN_STATS:
        tl.store(mean_ptr + row_idx, mean)
        tl.store(rstd_ptr + row_idx, rstd)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd

        # 仿射变换
        if HAS_WEIGHT:
            weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(
                tl.float32
            )
            x_hat = x_hat * weight

        if HAS_BIAS:
            bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(
                tl.float32
            )
            x_hat = x_hat + bias

        y = x_hat.to(x_ptr.dtype.element_ty)
        tl.store(y_row_ptr + cols, y, mask=mask)


def _layer_norm_impl(
    input: torch.Tensor,
    normalized_shape: Tuple[int, ...],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-05,
    *,
    return_stats: bool = False,
):
    logger.debug(f"FLAG_DNN LAYER_NORM (eps={eps})")

    stat_shape = input.shape[: input.ndim - len(normalized_shape)] + (
        1,
    ) * len(normalized_shape)
    if input.numel() == 0:
        y = torch.empty_like(input)
        if return_stats:
            stats = torch.empty(
                stat_shape, dtype=torch.float32, device=input.device
            )
            return y, stats, torch.empty_like(stats)
        return y

    assert input.ndim >= len(
        normalized_shape
    ), "Input dimensions must be >= normalized_shape length"

    if not input.is_contiguous():
        assert False, "input must be contiguous."
        input = input.contiguous()

    y = torch.empty_like(input)

    N = 1
    tail_shape = input.shape[-len(normalized_shape) :]
    if tuple(normalized_shape) != tuple(tail_shape):
        raise ValueError(
            "The normalized_shape must match"
            " the last few dimensions of"
            " the input tensor."
        )

    for dim in normalized_shape:
        N *= dim

    M = input.numel() // N
    mean = (
        torch.empty((M,), dtype=torch.float32, device=input.device)
        if return_stats
        else y
    )
    rstd = (
        torch.empty((M,), dtype=torch.float32, device=input.device)
        if return_stats
        else y
    )

    grid = (M,)

    with torch_device_fn.device(input.device):
        layer_norm_kernel[grid](
            input,
            y,
            mean,
            rstd,
            weight,
            bias,
            M,
            N,
            eps,
            HAS_WEIGHT=(weight is not None),
            HAS_BIAS=(bias is not None),
            RETURN_STATS=return_stats,
        )

    if return_stats:
        return y, mean.reshape(stat_shape), rstd.reshape(stat_shape)
    return y


def layer_norm(
    input: torch.Tensor,
    normalized_shape: Tuple[int, ...],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-05,
) -> torch.Tensor:
    return _layer_norm_impl(
        input, normalized_shape, weight=weight, bias=bias, eps=eps
    )


def layer_norm_forward(
    input: torch.Tensor,
    normalized_shape: Tuple[int, ...],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-05,
):
    return _layer_norm_impl(
        input,
        normalized_shape,
        weight=weight,
        bias=bias,
        eps=eps,
        return_stats=True,
    )
