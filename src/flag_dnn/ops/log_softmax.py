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
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


@triton.jit
def fast_log_softmax_kernel(
    x_ptr,
    y_ptr,
    N,
    stride_x_row,
    stride_y_row,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    row_x_ptr = x_ptr + pid * stride_x_row
    row_y_ptr = y_ptr + pid * stride_y_row

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(row_x_ptr + offsets, mask=mask, other=-float("inf"))
    x_fp32 = x.to(tl.float32)

    row_max = tl.max(x_fp32, axis=0)
    shifted = x_fp32 - row_max
    exp_shifted = tl.exp(shifted)
    log_denom = tl.log(tl.sum(exp_shifted, axis=0))
    log_softmax_val = shifted - log_denom

    y = log_softmax_val.to(x.dtype)
    tl.store(row_y_ptr + offsets, y, mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("softmax"),
    key=["N"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def online_log_softmax_kernel(
    x_ptr,
    y_ptr,
    N,
    stride_x_row,
    stride_y_row,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    row_x_ptr = x_ptr + pid * stride_x_row
    row_y_ptr = y_ptr + pid * stride_y_row

    m_i = -float("inf")
    d_i = 0.0

    ptrs = row_x_ptr + tl.arange(0, BLOCK_SIZE)

    for offset in range(0, N, BLOCK_SIZE):
        mask = (offset + tl.arange(0, BLOCK_SIZE)) < N
        x = tl.load(ptrs, mask=mask, other=-float("inf"))
        x_fp32 = x.to(tl.float32)
        m_block = tl.max(x_fp32, axis=0)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        exp_vals = tl.where(mask, tl.exp(x_fp32 - m_new), 0.0)
        d_block = tl.sum(exp_vals, axis=0)
        d_i = d_i * alpha + d_block
        m_i = m_new
        ptrs += BLOCK_SIZE

    log_denom = tl.log(d_i)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(row_x_ptr + cols, mask=mask, other=-float("inf"))
        x_fp32 = x.to(tl.float32)
        log_softmax_val = (x_fp32 - m_i) - log_denom
        out = log_softmax_val.to(x_ptr.dtype.element_ty)
        tl.store(row_y_ptr + cols, out, mask=mask)


# Dtype-specific tuning for float16/bfloat16: include BLOCK_SIZE=32768
# to reduce loop iterations for medium-N rows (65536, 128256)
_FP16_CONFIGS = [
    triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=ns)
    for bs in [2048, 4096, 8192, 16384, 32768]
    for nw in [8, 16, 32]
    for ns in [2, 3]
]


@libentry()
@libtuner(
    configs=_FP16_CONFIGS,
    key=["N"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def online_log_softmax_fp16_kernel(
    x_ptr,
    y_ptr,
    N,
    stride_x_row,
    stride_y_row,
    BLOCK_SIZE: tl.constexpr,
):
    """Same algorithm as online_log_softmax_kernel but with fp16-specific
    autotuner configs including larger BLOCK_SIZE options for better
    SM utilization at medium N (65536, 128256)."""
    pid = tle.program_id(0)
    row_x_ptr = x_ptr + pid * stride_x_row
    row_y_ptr = y_ptr + pid * stride_y_row

    m_i = -float("inf")
    d_i = 0.0

    ptrs = row_x_ptr + tl.arange(0, BLOCK_SIZE)

    for offset in range(0, N, BLOCK_SIZE):
        mask = (offset + tl.arange(0, BLOCK_SIZE)) < N
        x = tl.load(ptrs, mask=mask, other=-float("inf"))
        x_fp32 = x.to(tl.float32)
        m_block = tl.max(x_fp32, axis=0)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        exp_vals = tl.where(mask, tl.exp(x_fp32 - m_new), 0.0)
        d_block = tl.sum(exp_vals, axis=0)
        d_i = d_i * alpha + d_block
        m_i = m_new
        ptrs += BLOCK_SIZE

    log_denom = tl.log(d_i)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(row_x_ptr + cols, mask=mask, other=-float("inf"))
        x_fp32 = x.to(tl.float32)
        log_softmax_val = (x_fp32 - m_i) - log_denom
        out = log_softmax_val.to(x_ptr.dtype.element_ty)
        tl.store(row_y_ptr + cols, out, mask=mask)


def log_softmax(
    input: torch.Tensor,
    dim: Optional[int] = None,
    _stacklevel: int = 3,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    logger.debug(f"FLAG_DNN LOG_SOFTMAX (dim={dim}, dtype={dtype})")

    x = input
    if dtype is not None:
        x = x.to(dtype)

    if x.numel() == 0:
        return torch.empty_like(x)

    if dim is None:
        dim = -1
    if dim < 0:
        dim = x.ndim + dim

    need_transpose = dim != x.ndim - 1
    if need_transpose:
        x = x.transpose(dim, -1)

    if not x.is_contiguous():
        x = x.contiguous()

    N = x.shape[-1]
    M = x.numel() // N

    y = torch.empty_like(x)
    grid = (M,)
    MAX_FAST_BLOCK = 1024

    is_half_precision = x.dtype in (torch.float16, torch.bfloat16)

    with torch_device_fn.device(x.device):
        if N <= MAX_FAST_BLOCK:
            BLOCK_SIZE = triton.next_power_of_2(N)
            fast_log_softmax_kernel[grid](x, y, N, N, N, BLOCK_SIZE=BLOCK_SIZE)
        elif is_half_precision:
            # Use dtype-specific kernel with larger block size configs
            online_log_softmax_fp16_kernel[grid](x, y, N, N, N)
        else:
            online_log_softmax_kernel[grid](x, y, N, N, N)

    if need_transpose:
        y = y.transpose(dim, -1).contiguous()

    return y
