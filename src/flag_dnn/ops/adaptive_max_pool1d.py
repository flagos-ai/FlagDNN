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
def adaptive_max_pool1d_kernel(
    x_ptr, y_ptr, idx_ptr,
    N, C, W,
    OW,
    MAX_K_W: tl.constexpr,
    RETURN_INDICES: tl.constexpr,
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

    # 动态推导 1D 起点和终点
    start_w = (ow * W) // OW
    end_w = ((ow + 1) * W + OW - 1) // OW
    
    # 防止边界溢出
    start_w = tl.minimum(start_w, W)
    end_w = tl.minimum(end_w, W)

    input_dtype = x_ptr.dtype.element_ty
    max_val = tl.full([BLOCK_SIZE], -float('inf'), dtype=input_dtype)
    
    max_idx = tl.full([BLOCK_SIZE], -1, dtype=tl.int64)

    # 1D 动态窗口寻找最大值
    for kw in range(MAX_K_W):
        iw = start_w + kw
        in_window = iw < end_w
        
        load_idx = x_base_idx + iw
        
        # 只有当 valid_mask 和 in_window 均为真时，才会去比较更新
        val = tl.load(x_ptr + load_idx, mask=valid_mask & in_window, other=0.0)
        
        is_new_max = val > max_val
        update_mask = is_new_max & in_window & valid_mask
        
        max_val = tl.where(update_mask, val, max_val)
        if RETURN_INDICES:
            max_idx = tl.where(update_mask, iw, max_idx)

    # 存储结果
    tl.store(y_ptr + offsets, max_val, mask=valid_mask)
    if RETURN_INDICES:
        tl.store(idx_ptr + offsets, max_idx.to(tl.int64), mask=valid_mask)


def adaptive_max_pool1d(
    input: torch.Tensor,
    output_size: Union[int, Tuple[int]],
    return_indices: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    logger.debug(f"FLAG_DNN ADAPTIVE_MAX_POOL1D (output_size={output_size}, return_indices={return_indices})")

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
    
    indices = None
    idx_ptr = x # 若不需要返回索引，传入任意合法张量当占位符即可
    if return_indices:
        indices = torch.empty((N, C, OW), dtype=torch.int64, device=x.device)
        idx_ptr = indices

    M = N * C * OW
    if M == 0:
        y_out = y.squeeze(0) if is_2d else y
        if return_indices:
            idx_out = indices.squeeze(0) if is_2d else indices
            return y_out, idx_out
        return y_out

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(M, BLOCK_SIZE),)

    # 计算 1D 维度最大可能窗口，给 Triton 的 range() 提供静态上限
    max_k_w = math.ceil(W / OW) + 1

    with torch_device_fn.device(x.device):
        adaptive_max_pool1d_kernel[grid](
            x, y, idx_ptr,
            N, C, W,
            OW,
            MAX_K_W=max_k_w,
            RETURN_INDICES=return_indices,
            BLOCK_SIZE=BLOCK_SIZE
        )

    y_out = y.squeeze(0) if is_2d else y
    if return_indices:
        idx_out = indices.squeeze(0) if is_2d else indices
        return y_out, idx_out
    
    return y_out