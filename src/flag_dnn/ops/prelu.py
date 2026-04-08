import logging
from typing import Union

import torch
import triton
import triton.language as tl
import math

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("prelu"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def prelu_kernel(
    x_ptr,  # 输入张量指针
    weight_ptr,  # 可学习参数 a (weight) 的指针
    y_ptr,  # 输出张量指针
    n_elements,  # 张量总元素个数
    inner_size,  # 通道维度之后的内部维度大小 (H * W * ...)
    num_channels,  # 通道总数
    MULTI_CHANNEL: tl.constexpr,  # 编译期常量：判断是否为多通道模式
    BLOCK_SIZE: tl.constexpr,  # 编译期常量：线程块大小
):
    # 计算当前线程处理的全局索引
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 创建掩码，防止越界访问
    mask = offsets < n_elements

    # 从显存中加载输入数据 x
    x = tl.load(x_ptr + offsets, mask=mask)

    # 根据模式加载对应的斜率 a (weight)
    if MULTI_CHANNEL:
        # 逐通道模式：计算当前元素属于哪个通道
        # 假设形状为 (N, C, H, W)，内层大小为 H*W
        # 则通道索引 c = (offset // (H * W)) % C
        c_idx = (offsets // inner_size) % num_channels
        a = tl.load(weight_ptr + c_idx, mask=mask)
    else:
        # 全局共享模式：只加载第 0 个元素，Triton 会自动广播给当前 block 的所有线程
        a = tl.load(weight_ptr)

    # 计算 PReLU
    y = tl.where(x >= 0.0, x, x * a)

    # 将计算结果写回显存
    tl.store(y_ptr + offsets, y, mask=mask)


def prelu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    logger.debug("FLAG_DNN PRELU")

    assert x.is_contiguous(), "x must be contiguous"
    assert weight.is_contiguous(), "weight must be contiguous"

    # 解析 weight 的参数量来判断模式
    num_parameters = weight.numel()
    multi_channel = num_parameters > 1

    if multi_channel:
        # 如果是多通道模式，PyTorch 规定 dim=1 为通道维度
        assert x.dim() >= 2, "当 num_parameters > 1 时，输入张量必须至少有 2 个维度"
        assert (
            num_parameters == x.shape[1]
        ), f"权重数量 ({num_parameters}) 必须等于通道数 ({x.shape[1]})"

        # 计算 inner_size (例如 H * W * ...)
        # 如果张量只有 2 维 (N, C)，则 inner_size = 1
        inner_size = math.prod(x.shape[2:]) if x.dim() > 2 else 1
    else:
        # 全局共享模式，inner_size 用不到，随意赋一个合法值即可
        inner_size = 1

    n_elements = x.numel()
    y = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # 启动 Triton Kernel
    with torch_device_fn.device(x.device):
        prelu_kernel[grid](
            x_ptr=x,
            weight_ptr=weight,
            y_ptr=y,
            n_elements=n_elements,
            inner_size=inner_size,
            num_channels=num_parameters,
            MULTI_CHANNEL=multi_channel,  # 传入编译期常量
        )

    return y
