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
def add_tensor_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    alpha_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 自动进行计算
    res = x + alpha_val * y

    # 结果强制转换回输出张量的目标类型，防止隐式提升导致的错误
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def add_scalar_kernel(
    x_ptr, out_ptr,
    n_elements,
    other_val, alpha_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    
    res = x + alpha_val * other_val

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def add(
    input: torch.Tensor,
    other: Union[torch.Tensor, int, float],
    *,
    alpha: float = 1.0,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    logger.debug(f"FLAG_DNN ADD (alpha={alpha})")

    is_other_tensor = isinstance(other, torch.Tensor)

    # 广播形状 (Broadcasting)
    if is_other_tensor:
        out_shape = torch.broadcast_shapes(input.shape, other.shape)
    else:
        out_shape = input.shape

    # 类型推导与提升 (Type Promotion)
    # 利用 PyTorch 底层的类型推导逻辑，构建一个极小的 dummy tensor 探测目标 dtype
    dummy_input = input.new_empty((0,))
    dummy_other = other.new_empty((0,)) if is_other_tensor else other
    out_dtype = (dummy_input + alpha * dummy_other).dtype

    # 输出内存分配
    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=input.device)
    else:
        assert out.shape == out_shape, f"out shape {out.shape} mismatch with broadcast shape {out_shape}"
        out_dtype = out.dtype  # 如果用户指定了 out，强制使用 out 的数据类型

    n_elements = out.numel()
    if n_elements == 0:
        return out

    # 内存连续化处理
    # Triton 1D block 依赖连续内存。通过 expand 广播逻辑形状后，调用 contiguous 在底层展开
    input_c = input.expand(out_shape).contiguous()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    with torch_device_fn.device(input.device):
        if is_other_tensor:
            other_c = other.expand(out_shape).contiguous()
            add_tensor_kernel[grid](
                input_c, other_c, out,
                n_elements,
                float(alpha),
                BLOCK_SIZE=BLOCK_SIZE
            )
        else:
            add_scalar_kernel[grid](
                input_c, out,
                n_elements,
                float(other), float(alpha),
                BLOCK_SIZE=BLOCK_SIZE
            )

    return out