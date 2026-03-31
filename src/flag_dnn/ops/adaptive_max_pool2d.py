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


# @libentry()
# @libtuner(
#     configs=runtime.get_tuned_config("rms_norm"),
#     key=["M", "N"],
#     warmup=5,
#     rep=10,
# )
@triton.jit
def adaptive_max_pool2d_kernel(
    x_ptr, y_ptr, indices_ptr,
    N, C, H, W,
    OH, OW,
    MAX_KH: tl.constexpr,
    MAX_KW: tl.constexpr,
    RETURN_INDICES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    num_elements = N * C * OH * OW
    mask = offsets < num_elements

    # 反推输出坐标 (n, c, oh, ow)
    ow = offsets % OW
    oh = (offsets // OW) % OH
    c = (offsets // (OW * OH)) % C
    n = offsets // (C * OW * OH)

    x_base_idx = n * (C * H * W) + c * (H * W)

    # 动态计算自适应滑动窗口在原图上的起始和结束边界
    ih_start = (oh * H) // OH
    ih_end = ((oh + 1) * H + OH - 1) // OH
    
    iw_start = (ow * W) // OW
    iw_end = ((ow + 1) * W + OW - 1) // OW

    dtype = x_ptr.dtype.element_ty

    # 初始化最大值为负无穷大，索引为 0
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=dtype)
    
    if RETURN_INDICES:
        max_idx = tl.full([BLOCK_SIZE], 0, dtype=tl.int64)

    # 遍历最大可能的自适应窗口区域 (使用 tl.constexpr 避免编译崩溃)
    for kh in range(MAX_KH):
        for kw in range(MAX_KW):
            ih = ih_start + kh
            iw = iw_start + kw
            
            # 判断当前 (ih, iw) 是否在动态窗口的有效范围内
            valid = (ih < ih_end) & (iw < iw_end)
            
            load_idx = x_base_idx + ih * W + iw
            
            # 加载时超出边界的填充负无穷大，确保它们不会成为最大值
            val = tl.load(x_ptr + load_idx, mask=mask & valid, other=float('-inf'))
            
            # 严格大于 (>) 时才更新，这样碰到相同最大值时，会保留第一次遇到的索引，对齐 PyTorch 逻辑
            update_mask = valid & (val > max_val)
            max_val = tl.where(update_mask, val, max_val)
            
            if RETURN_INDICES:
                # 记录局部索引 (相对于单张 HW 特征图)
                local_idx = ih * W + iw
                max_idx = tl.where(update_mask, local_idx, max_idx)

    # 写回输出结果
    tl.store(y_ptr + offsets, max_val, mask=mask)
    
    if RETURN_INDICES:
        tl.store(indices_ptr + offsets, max_idx, mask=mask)


def adaptive_max_pool2d(
    input: torch.Tensor,
    output_size: Union[int, Tuple[Optional[int], Optional[int]]],
    return_indices: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    logger.debug(f"FLAG_DNN ADAPTIVE_MAX_POOL2D (output_size={output_size}, return_indices={return_indices})")

    assert input.ndim in [3, 4], "Input must be 3D or 4D"
    is_3d = input.ndim == 3
    if is_3d:
        input = input.unsqueeze(0)

    N, C, H, W = input.shape

    # 解析 output_size 逻辑
    if isinstance(output_size, int):
        OH = OW = output_size
    else:
        OH = output_size[0] if output_size[0] is not None else H
        OW = output_size[1] if output_size[1] is not None else W

    x = input.contiguous()
    y = torch.empty((N, C, OH, OW), dtype=x.dtype, device=x.device)
    
    indices = None
    if return_indices:
        indices = torch.empty((N, C, OH, OW), dtype=torch.int64, device=x.device)

    M = N * C * OH * OW
    
    # 拦截特例：Batch 为 0 的空张量
    if M == 0:
        out_y = y.squeeze(0) if is_3d else y
        if return_indices:
            out_idx = indices.squeeze(0) if is_3d else indices
            return out_y, out_idx
        return out_y

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(M, BLOCK_SIZE),)

    # 计算最大可能的池化核大小，作为常量传入 Triton
    max_k_h = math.ceil(H / OH) + 1
    max_k_w = math.ceil(W / OW) + 1
    
    # 将 None 转换为底层可处理的空指针占位符
    indices_ptr = indices if return_indices else x

    with torch_device_fn.device(x.device):
        adaptive_max_pool2d_kernel[grid](
            x, y, indices_ptr,
            N, C, H, W,
            OH, OW,
            MAX_KH=max_k_h,
            MAX_KW=max_k_w,
            RETURN_INDICES=return_indices,
            BLOCK_SIZE=BLOCK_SIZE
        )

    out_y = y.squeeze(0) if is_3d else y
    if return_indices:
        out_idx = indices.squeeze(0) if is_3d else indices
        return out_y, out_idx
    
    return out_y