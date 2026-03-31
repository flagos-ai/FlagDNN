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
def adaptive_max_pool3d_kernel(
    x_ptr, y_ptr, idx_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    MAX_K_D: tl.constexpr, MAX_K_H: tl.constexpr, MAX_K_W: tl.constexpr,
    RETURN_INDICES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    num_elements = N * C * OD * OH * OW
    valid_mask = offsets < num_elements

    # 反推 3D 输出坐标 (n, c, od, oh, ow)
    ow = offsets % OW
    oh = (offsets // OW) % OH
    od = (offsets // (OW * OH)) % OD
    c = (offsets // (OW * OH * OD)) % C
    n = offsets // (C * OW * OH * OD)

    x_base_idx = n * (C * D * H * W) + c * (D * H * W)

    # 动态推导 3D 起点和终点
    start_d = (od * D) // OD
    end_d = ((od + 1) * D + OD - 1) // OD
    
    start_h = (oh * H) // OH
    end_h = ((oh + 1) * H + OH - 1) // OH
    
    start_w = (ow * W) // OW
    end_w = ((ow + 1) * W + OW - 1) // OW
    
    # 防止边界溢出
    start_d = tl.minimum(start_d, D)
    end_d = tl.minimum(end_d, D)
    start_h = tl.minimum(start_h, H)
    end_h = tl.minimum(end_h, H)
    start_w = tl.minimum(start_w, W)
    end_w = tl.minimum(end_w, W)

    input_dtype = x_ptr.dtype.element_ty
    max_val = tl.full([BLOCK_SIZE], -float('inf'), dtype=input_dtype)
    # 索引记录的是该元素在当前 (D, H, W) 空间内的局部展平索引
    max_idx = tl.full([BLOCK_SIZE], -1, dtype=tl.int64)

    # 3D 动态窗口寻找最大值
    for kd in range(MAX_K_D):
        id_ = start_d + kd
        in_window_d = id_ < end_d
        
        for kh in range(MAX_K_H):
            ih = start_h + kh
            in_window_h = in_window_d & (ih < end_h)
            
            for kw in range(MAX_K_W):
                iw = start_w + kw
                in_window = in_window_h & (iw < end_w)
                
                spatial_idx = id_ * (H * W) + ih * W + iw
                load_idx = x_base_idx + spatial_idx
                
                # 只有当 valid_mask 和 in_window 均为真时，才会去比较更新
                val = tl.load(x_ptr + load_idx, mask=valid_mask & in_window, other=0.0)
                
                # 是否为新的最大值？（对于 MaxPool，原生 PyTorch 遇到相等时通常保留较前的索引）
                is_new_max = val > max_val
                update_mask = is_new_max & in_window & valid_mask
                
                max_val = tl.where(update_mask, val, max_val)
                if RETURN_INDICES:
                    max_idx = tl.where(update_mask, spatial_idx, max_idx)

    # 存储结果
    tl.store(y_ptr + offsets, max_val, mask=valid_mask)
    if RETURN_INDICES:
        # PyTorch 要求 indices 是 int64 (long) 类型
        tl.store(idx_ptr + offsets, max_idx.to(tl.int64), mask=valid_mask)


def adaptive_max_pool3d(
    input: torch.Tensor,
    output_size: Union[int, Tuple[int, int, int]],
    return_indices: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    logger.debug(f"FLAG_DNN ADAPTIVE_MAX_POOL3D (output_size={output_size}, return_indices={return_indices})")

    if isinstance(output_size, int):
        OD = OH = OW = output_size
    else:
        assert len(output_size) == 3, "output_size must be an int or a tuple of 3 ints"
        OD, OH, OW = output_size

    assert input.ndim in [4, 5], "Input must be 4D (C, D, H, W) or 5D (N, C, D, H, W)"
    is_4d = input.ndim == 4
    if is_4d:
        input = input.unsqueeze(0)

    N, C, D, H, W = input.shape

    x = input.contiguous()
    y = torch.empty((N, C, OD, OH, OW), dtype=x.dtype, device=x.device)
    
    indices = None
    idx_ptr = x # 若不需要返回索引，传入任意合法张量当占位符即可
    if return_indices:
        indices = torch.empty((N, C, OD, OH, OW), dtype=torch.int64, device=x.device)
        idx_ptr = indices

    M = N * C * OD * OH * OW
    if M == 0:
        y_out = y.squeeze(0) if is_4d else y
        if return_indices:
            idx_out = indices.squeeze(0) if is_4d else indices
            return y_out, idx_out
        return y_out

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(M, BLOCK_SIZE),)

    # 计算 3D 各维度最大可能窗口，给 Triton 的 range() 提供静态上限
    max_k_d = math.ceil(D / OD) + 1
    max_k_h = math.ceil(H / OH) + 1
    max_k_w = math.ceil(W / OW) + 1

    with torch_device_fn.device(x.device):
        adaptive_max_pool3d_kernel[grid](
            x, y, idx_ptr,
            N, C, D, H, W,
            OD, OH, OW,
            MAX_K_D=max_k_d, MAX_K_H=max_k_h, MAX_K_W=max_k_w,
            RETURN_INDICES=return_indices,
            BLOCK_SIZE=BLOCK_SIZE
        )

    y_out = y.squeeze(0) if is_4d else y
    if return_indices:
        idx_out = indices.squeeze(0) if is_4d else indices
        return y_out, idx_out
    
    return y_out