import logging
import math
from typing import Tuple, Union, Optional

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@triton.jit
def adaptive_avg_pool1d_kernel(
    x_ptr, y_ptr,
    N, C, W,
    OW,
    MAX_K_W: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    num_elements = N * C * OW
    valid_mask = offsets < num_elements

    # 反推 1D 输出坐标 (n, c, ow)
    ow = offsets % OW
    c = (offsets // OW) % C
    n = offsets // (C * OW)

    x_base_idx = n * (C * W) + c * W

    # Adaptive Pooling 核心数学公式：动态推导起点和终点
    # start = floor(ow * W / OW), end = ceil((ow+1) * W / OW)
    start_w = (ow * W) // OW
    end_w = ((ow + 1) * W + OW - 1) // OW
    
    # 防止极少数由于精度异常导致的越界
    start_w = tl.minimum(start_w, W)
    end_w = tl.minimum(end_w, W)
    
    pool_size = end_w - start_w
    # 防止极端情况下除数为 0
    pool_size = tl.where(pool_size == 0, 1, pool_size)

    # 使用高精度初始化累加器
    sum_val = tl.zeros([BLOCK_SIZE], dtype=ACC_DTYPE)

    for kw in range(MAX_K_W):
        iw = start_w + kw
        # 动态判定当前循环是否在当前输出像素的有效窗口内
        in_window = iw < end_w
        
        load_idx = x_base_idx + iw
        
        # 安全加载：必须同时满足 线程有效(valid_mask) 和 窗口有效(in_window)
        val = tl.load(x_ptr + load_idx, mask=valid_mask & in_window, other=0.0).to(ACC_DTYPE)
        sum_val += val

    # 计算平均值并转回原数据类型
    res = sum_val / pool_size
    tl.store(y_ptr + offsets, res.to(x_ptr.dtype.element_ty), mask=valid_mask)


def adaptive_avg_pool1d(
    input: torch.Tensor,
    output_size: Union[int, Tuple[int]],
) -> torch.Tensor:
    logger.debug(f"FLAG_DNN ADAPTIVE_AVG_POOL1D (output_size={output_size})")

    if isinstance(output_size, int):
        OW = output_size
    else:
        OW = output_size[0]

    assert input.ndim in [2, 3], "Input must be 2D or 3D"
    is_2d = input.ndim == 2
    if is_2d:
        input = input.unsqueeze(0)

    N, C, W = input.shape

    x = input.contiguous()
    y = torch.empty((N, C, OW), dtype=x.dtype, device=x.device)

    M = N * C * OW
    if M == 0:
        return y.squeeze(0) if is_2d else y

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(M, BLOCK_SIZE),)

    # 计算当前输入与输出比例下，可能出现的最大窗口尺寸
    max_k_w = math.ceil(W / OW) + 1

    # 动态分发累加精度：防溢出和防掉精度
    acc_dtype = tl.float64 if x.dtype == torch.float64 else tl.float32

    with torch_device_fn.device(x.device):
        adaptive_avg_pool1d_kernel[grid](
            x, y,
            N, C, W, OW,
            MAX_K_W=max_k_w,
            ACC_DTYPE=acc_dtype,
            BLOCK_SIZE=BLOCK_SIZE
        )

    return y.squeeze(0) if is_2d else y