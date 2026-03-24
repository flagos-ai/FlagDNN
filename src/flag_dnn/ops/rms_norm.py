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


@triton.jit
def rms_norm_kernel(
    x_ptr, y_ptr,
    weight_ptr,
    M, N, eps,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    # 获取当前处理的行索引
    row_idx = tle.program_id(0)

    # 计算当前行的起始内存指针
    x_row_ptr = x_ptr + row_idx * N
    y_row_ptr = y_ptr + row_idx * N

    sum_squares = 0.0
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        
        # 加载数据并上采样到 fp32
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        
        # 累加平方和
        sum_squares += tl.sum(x * x, axis=0)

    # 计算 RMS 倒数 (使用 Triton 原生底层 rsqrt 指令提速)
    rrms = tl.math.rsqrt((sum_squares / N) + eps)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        
        # 核心 RMSNorm 计算
        x_hat = x * rrms

        # 权重缩放 (如果传入了 weight)
        if HAS_WEIGHT:
            weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            x_hat = x_hat * weight

        y = x_hat.to(x_ptr.dtype.element_ty)
        tl.store(y_row_ptr + cols, y, mask=mask)


def rms_norm(
    input: torch.Tensor,
    normalized_shape: Tuple[int, ...],
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-05,
) -> torch.Tensor:
    logger.debug(f"FLAG_DNN RMS_NORM (eps={eps})")

    # 拦截空张量
    if input.numel() == 0:
        return torch.empty_like(input)

    assert input.ndim >= len(normalized_shape), "Input dimensions must be >= normalized_shape length"

    x = input.contiguous()
    y = torch.empty_like(x)

    # 计算 N (归一化维度的乘积) 和 M (前面所有独立维度的乘积)
    N = 1
    for dim in normalized_shape:
        N *= dim
    M = x.numel() // N

    if weight is None:
        weight_ptr = x  # 传入一个 Dummy Pointer (内核里有 HAS_WEIGHT 拦截，不会真的去读)
        has_weight = False
    else:
        weight_ptr = weight
        has_weight = True

    grid = (M,)
    
    BLOCK_SIZE = min(triton.next_power_of_2(N), 2048)
    
    num_warps = 4 if BLOCK_SIZE <= 1024 else 8

    with torch_device_fn.device(x.device):
        rms_norm_kernel[grid](
            x, y,
            weight_ptr,
            M, N, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_WEIGHT=has_weight,
            num_warps=num_warps,
        )

    return y