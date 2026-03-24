import logging
from typing import Union, Optional

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
# if error try :
# res = tl.math.exp(y_f32 * tl.math.log(x_f32))

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@triton.jit
def pow_tensor_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 向上转型到 float32，防止底层 libdevice 找不到 fp16/bf16 的 pow 签名
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    res = libdevice.pow(x_f32, y_f32)

    # 写回时向下转型回目标数据类型
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def pow_scalar_exponent_kernel(
    x_ptr, out_ptr,
    n_elements,
    exponent_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    
    x_f32 = x.to(tl.float32)

    exp_f32 = tl.cast(exponent_val, tl.float32)
    res = libdevice.pow(x_f32, exp_f32)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def pow_scalar_base_kernel(
    y_ptr, out_ptr,
    n_elements,
    base_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    y = tl.load(y_ptr + offsets, mask=mask)
    
    y_f32 = y.to(tl.float32)

    base_f32 = tl.cast(base_val, tl.float32)
    res = libdevice.pow(base_f32, y_f32)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def pow(
    input: Union[torch.Tensor, int, float],
    exponent: Union[torch.Tensor, int, float],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    logger.debug("FLAG_DNN POW")

    input_is_tensor = isinstance(input, torch.Tensor)
    exp_is_tensor = isinstance(exponent, torch.Tensor)

    if not (input_is_tensor or exp_is_tensor):
        raise TypeError("At least one of input or exponent must be a Tensor")

    # 确定输出形状与广播
    if input_is_tensor and exp_is_tensor:
        out_shape = torch.broadcast_shapes(input.shape, exponent.shape)
        device = input.device
    elif input_is_tensor:
        out_shape = input.shape
        device = input.device
    else:
        out_shape = exponent.shape
        device = exponent.device

    # 类型推导与提升 (Type Promotion)
    dummy_input = input.new_empty((0,)) if input_is_tensor else input
    dummy_exponent = exponent.new_empty((0,)) if exp_is_tensor else exponent
    out_dtype = torch.pow(dummy_input, dummy_exponent).dtype

    # 输出内存分配
    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=device)
    else:
        assert out.shape == out_shape, f"out shape {out.shape} mismatch with broadcast shape {out_shape}"
        out_dtype = out.dtype

    n_elements = out.numel()
    if n_elements == 0:
        return out

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # 4. 路由到对应的 Kernel
    with torch_device_fn.device(device):
        if input_is_tensor and exp_is_tensor:
            input_c = input.expand(out_shape).contiguous()
            exponent_c = exponent.expand(out_shape).contiguous()
            pow_tensor_kernel[grid](
                input_c, exponent_c, out,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE
            )
        elif input_is_tensor:
            input_c = input.expand(out_shape).contiguous()
            pow_scalar_exponent_kernel[grid](
                input_c, out,
                n_elements,
                float(exponent),
                BLOCK_SIZE=BLOCK_SIZE
            )
        else:
            exponent_c = exponent.expand(out_shape).contiguous()
            pow_scalar_base_kernel[grid](
                exponent_c, out,
                n_elements,
                float(input),
                BLOCK_SIZE=BLOCK_SIZE
            )

    return out