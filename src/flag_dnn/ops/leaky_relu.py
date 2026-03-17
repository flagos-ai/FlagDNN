import logging
from typing import Union

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
    configs=runtime.get_tuned_config("leaky_relu"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def leaky_relu_kernel(
    x_ptr,            # 输入张量指针
    y_ptr,            # 输出张量指针
    n_elements,       # 张量总元素个数
    negative_slope,   # 负半轴斜率 (标量，不需要设为 constexpr，因为可以动态传参)
    BLOCK_SIZE: tl.constexpr,  # 编译期常量：线程块大小
):
    # 计算当前线程处理的全局索引
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 创建掩码，防止越界访问
    mask = offsets < n_elements
    
    # 从显存中加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # 计算 LeakyReLU
    # 在 GPU 底层，这通常会被编译为高效的条件选择指令，避免了真正的分支跳转
    y = tl.where(x >= 0.0, x, x * negative_slope)
        
    # 将计算结果写回显存
    tl.store(y_ptr + offsets, y, mask=mask)


def leaky_relu(x: torch.Tensor, negative_slope: float = 0.01, inplace: bool = False) -> torch.Tensor:
    logger.debug(f"FLAG_DNN LEAKY_RELU (negative_slope={negative_slope}, inplace={inplace})")

    assert x.is_contiguous(), "x must be contiguous"
    
    # 根据 inplace 参数决定是否复用输入张量的显存
    if inplace:
        y = x
    else:
        y = torch.empty_like(x)
        
    n_elements = x.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    
    # 启动 Triton Kernel
    with torch_device_fn.device(x.device):
        leaky_relu_kernel[grid](
            x, y, n_elements, 
            negative_slope, # 传入浮点数参数
        )
    
    return y