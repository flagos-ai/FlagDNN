import logging
import math
from typing import Tuple, Union

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@triton.jit
def adaptive_avg_pool3d_kernel(
    x_ptr, y_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    MAX_K_D: tl.constexpr, MAX_K_H: tl.constexpr, MAX_K_W: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
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

    # Adaptive Pooling 核心数学公式：动态推导 3D 起点和终点
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
    
    pool_size = (end_d - start_d) * (end_h - start_h) * (end_w - start_w)
    
    # 防止极端情况下除数为 0
    pool_size = tl.where(pool_size == 0, 1, pool_size)

    # 使用高精度初始化累加器
    sum_val = tl.zeros([BLOCK_SIZE], dtype=ACC_DTYPE)

    # 3D 动态窗口循环
    for kd in range(MAX_K_D):
        id_ = start_d + kd
        in_window_d = id_ < end_d
        
        for kh in range(MAX_K_H):
            ih = start_h + kh
            in_window_h = in_window_d & (ih < end_h)
            
            for kw in range(MAX_K_W):
                iw = start_w + kw
                # 必须同时满足三个维度的窗口内条件
                in_window = in_window_h & (iw < end_w)
                
                load_idx = x_base_idx + id_ * (H * W) + ih * W + iw
                
                # 安全加载：必须同时满足 线程有效(valid_mask) 和 3D窗口有效(in_window)
                val = tl.load(x_ptr + load_idx, mask=valid_mask & in_window, other=0.0).to(ACC_DTYPE)
                sum_val += val

    # 计算平均值并转回原数据类型
    res = sum_val / pool_size
    tl.store(y_ptr + offsets, res.to(x_ptr.dtype.element_ty), mask=valid_mask)


def adaptive_avg_pool3d(
    input: torch.Tensor,
    output_size: Union[int, Tuple[int, int, int]],
) -> torch.Tensor:
    logger.debug(f"FLAG_DNN ADAPTIVE_AVG_POOL3D (output_size={output_size})")

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

    M = N * C * OD * OH * OW
    if M == 0:
        return y.squeeze(0) if is_4d else y

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(M, BLOCK_SIZE),)

    # 计算 3D 各维度最大可能窗口，给 Triton 的 range() 提供静态上限
    max_k_d = math.ceil(D / OD) + 1
    max_k_h = math.ceil(H / OH) + 1
    max_k_w = math.ceil(W / OW) + 1

    # 动态分发累加精度
    acc_dtype = tl.float64 if x.dtype == torch.float64 else tl.float32

    with torch_device_fn.device(x.device):
        adaptive_avg_pool3d_kernel[grid](
            x, y,
            N, C, D, H, W,
            OD, OH, OW,
            MAX_K_D=max_k_d, MAX_K_H=max_k_h, MAX_K_W=max_k_w,
            ACC_DTYPE=acc_dtype,
            BLOCK_SIZE=BLOCK_SIZE
        )

    return y.squeeze(0) if is_4d else y