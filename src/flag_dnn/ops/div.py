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
def div_tensor_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    ROUND_MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 核心运算：逐元素相除
    res = x / y

    # 处理舍入模式 (0: None, 1: trunc, 2: floor)
    if ROUND_MODE == 1:
        # trunc (向零取整): 正数向下取整，负数向上取整
        res = tl.where(res >= 0, tl.math.floor(res), tl.math.ceil(res))
    elif ROUND_MODE == 2:
        # floor (向下取整)
        res = tl.math.floor(res)

    # 强制转换回目标类型
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def div_scalar_kernel(
    x_ptr, out_ptr,
    n_elements,
    other_val,
    ROUND_MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    
    res = x / other_val

    if ROUND_MODE == 1:
        res = tl.where(res >= 0, tl.math.floor(res), tl.math.ceil(res))
    elif ROUND_MODE == 2:
        res = tl.math.floor(res)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def div(
    input: torch.Tensor,
    other: Union[torch.Tensor, int, float],
    *,
    rounding_mode: Optional[str] = None,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    logger.debug(f"FLAG_DNN DIV (rounding_mode={rounding_mode})")

    # 映射 rounding_mode 到整型常量
    mode_map = {None: 0, 'trunc': 1, 'floor': 2}
    if rounding_mode not in mode_map:
        raise RuntimeError(f"div expected rounding_mode to be one of None, 'trunc', 'floor' but found {rounding_mode}")
    mode_idx = mode_map[rounding_mode]

    is_other_tensor = isinstance(other, torch.Tensor)

    # 广播形状 (Broadcasting)
    if is_other_tensor:
        out_shape = torch.broadcast_shapes(input.shape, other.shape)
    else:
        out_shape = input.shape

    # 类型推导与提升 (Type Promotion)
    dummy_input = input.new_empty((0,))
    dummy_other = other.new_empty((0,)) if is_other_tensor else other
    # 利用 PyTorch 原生 div 获取目标数据类型（特别是在 rounding_mode=None 时，整数也会提升为浮点）
    out_dtype = torch.div(dummy_input, dummy_other, rounding_mode=rounding_mode).dtype

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
            div_tensor_kernel[grid](
                input_c, other_c, out,
                n_elements,
                ROUND_MODE=mode_idx,
                BLOCK_SIZE=BLOCK_SIZE
            )
        else:
            div_scalar_kernel[grid](
                input_c, out,
                n_elements,
                float(other),
                ROUND_MODE=mode_idx,
                BLOCK_SIZE=BLOCK_SIZE
            )

    return out