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


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sqrt"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def sqrt_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 加载数据
    x = tl.load(x_ptr + offsets, mask=mask)

    # 向上转型为 float32 以保证 tl.sqrt 在所有硬件后端的稳定性
    x_f32 = x.to(tl.float32)

    # 核心运算：计算平方根
    res = tl.sqrt(x_f32)

    # 向下转型并存回目标类型
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def sqrt(
    input: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    logger.debug("FLAG_DNN SQRT")

    if not input.is_contiguous():
        assert False, "input must be contiguous."
        input = input.contiguous()

    # 类型推导 (Type Promotion)
    # 利用 PyTorch 原生逻辑处理整数输入提升为浮点的情况
    dummy_input = input.new_empty((0,))
    out_dtype = torch.sqrt(dummy_input).dtype

    out_shape = input.shape

    # 输出内存分配
    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=input.device)
    else:
        assert (
            out.shape == out_shape
        ), f"out shape {out.shape} mismatch with input shape {out_shape}"
        out_dtype = out.dtype

    n_elements = out.numel()
    if n_elements == 0:
        return out

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # 启动 Kernel
    with torch_device_fn.device(input.device):
        sqrt_kernel[grid](input, out, n_elements)

    return out
