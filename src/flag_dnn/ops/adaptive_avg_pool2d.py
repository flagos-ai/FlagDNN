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
def adaptive_avg_pool2d_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
    OH, OW,
    MAX_KH: tl.constexpr,
    MAX_KW: tl.constexpr,
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

    # 升至 float32 防止累加溢出
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # 遍历最大可能的自适应窗口区域 (使用 tl.constexpr 标量控制循环边界)
    for kh in range(MAX_KH):
        for kw in range(MAX_KW):
            ih = ih_start + kh
            iw = iw_start + kw
            
            # 判断当前 (ih, iw) 是否在动态窗口的有效范围内
            valid = (ih < ih_end) & (iw < iw_end)
            
            load_idx = x_base_idx + ih * W + iw
            
            # 只有在 block mask 和 valid 双重掩码下才加载并累加
            val = tl.load(x_ptr + load_idx, mask=mask & valid, other=0.0).to(tl.float32)
            sum_val += val

    # 计算该动态窗口的实际有效面积 (Divisor)
    pool_h = ih_end - ih_start
    pool_w = iw_end - iw_start
    divisor = pool_h * pool_w
    
    # 防止除零错误
    divisor = tl.where(divisor <= 0, 1, divisor)

    # 计算均值并写回
    avg_val = sum_val / divisor
    tl.store(y_ptr + offsets, avg_val.to(x_ptr.dtype.element_ty), mask=mask)


def adaptive_avg_pool2d(
    input: torch.Tensor,
    output_size: Union[int, Tuple[Optional[int], Optional[int]]],
) -> torch.Tensor:
    logger.debug(f"FLAG_DNN ADAPTIVE_AVG_POOL2D (output_size={output_size})")

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

    M = N * C * OH * OW
    if M == 0:
        return y.squeeze(0) if is_3d else y

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(M, BLOCK_SIZE),)

    # 计算最大可能的池化核大小，作为常量传入 Triton，避免底层 MLIR 循环编译崩溃
    max_k_h = math.ceil(H / OH) + 1
    max_k_w = math.ceil(W / OW) + 1

    with torch_device_fn.device(x.device):
        adaptive_avg_pool2d_kernel[grid](
            x, y,
            N, C, H, W,
            OH, OW,
            MAX_KH=max_k_h,
            MAX_KW=max_k_w,
            BLOCK_SIZE=BLOCK_SIZE
        )

    return y.squeeze(0) if is_3d else y