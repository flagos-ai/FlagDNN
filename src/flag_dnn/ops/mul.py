import logging
from typing import Union, Optional

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@triton.jit
def mul_tensor_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 核心运算：逐元素相乘
    res = x * y

    # 强制转换回目标类型
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def mul_scalar_kernel(
    x_ptr, out_ptr,
    n_elements,
    other_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    
    # 标量乘法
    res = x * other_val

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def mul(
    input: torch.Tensor,
    other: Union[torch.Tensor, int, float],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    logger.debug("FLAG_DNN MUL")

    is_other_tensor = isinstance(other, torch.Tensor)

    # 广播形状 (Broadcasting)
    if is_other_tensor:
        out_shape = torch.broadcast_shapes(input.shape, other.shape)
    else:
        out_shape = input.shape

    # 类型推导与提升 (Type Promotion)
    dummy_input = input.new_empty((0,))
    dummy_other = other.new_empty((0,)) if is_other_tensor else other
    # 使用乘法推导目标 dtype
    out_dtype = (dummy_input * dummy_other).dtype

    # 输出内存分配
    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=input.device)
    else:
        assert out.shape == out_shape, f"out shape {out.shape} mismatch with broadcast shape {out_shape}"
        out_dtype = out.dtype

    n_elements = out.numel()
    if n_elements == 0:
        return out

    # 内存连续化处理
    input_c = input.expand(out_shape).contiguous()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    with torch_device_fn.device(input.device):
        if is_other_tensor:
            other_c = other.expand(out_shape).contiguous()
            mul_tensor_kernel[grid](
                input_c, other_c, out,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE
            )
        else:
            mul_scalar_kernel[grid](
                input_c, out,
                n_elements,
                float(other),
                BLOCK_SIZE=BLOCK_SIZE
            )

    return out