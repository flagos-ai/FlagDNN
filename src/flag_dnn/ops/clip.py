import logging
from typing import Optional, Union

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@triton.jit
def clamp_kernel(
    x_ptr, out_ptr,
    min_ptr, max_ptr,       # Tensor 指针
    min_val, max_val,       # Scalar 数值
    n_elements,
    HAS_MIN: tl.constexpr,
    HAS_MAX: tl.constexpr,
    IS_MIN_TENSOR: tl.constexpr,
    IS_MAX_TENSOR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    # 统一转换到 float32 进行数值比较，防止低精度截断
    res = x.to(tl.float32)

    # 处理下界 (min)
    if HAS_MIN:
        if IS_MIN_TENSOR:
            min_t = tl.load(min_ptr + offsets, mask=mask).to(tl.float32)
            res = tl.maximum(res, min_t)
        else:
            res = tl.maximum(res, min_val)

    # 处理上界 (max)
    if HAS_MAX:
        if IS_MAX_TENSOR:
            max_t = tl.load(max_ptr + offsets, mask=mask).to(tl.float32)
            res = tl.minimum(res, max_t)
        else:
            res = tl.minimum(res, max_val)

    # 存回目标类型
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def clamp(
    input: torch.Tensor,
    min: Optional[Union[float, int, torch.Tensor]] = None,
    max: Optional[Union[float, int, torch.Tensor]] = None,
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    logger.debug("FLAG_DNN CLAMP")

    if min is None and max is None:
        raise RuntimeError("torch.clamp: At least one of 'min' or 'max' must not be None")

    has_min = min is not None
    has_max = max is not None
    is_min_tensor = isinstance(min, torch.Tensor)
    is_max_tensor = isinstance(max, torch.Tensor)

    # 形状推导 (Broadcasting)
    # 动态计算 input, min, max 广播后的最终全局形状
    out_shape = input.shape
    if has_min and is_min_tensor:
        out_shape = torch.broadcast_shapes(out_shape, min.shape)
    if has_max and is_max_tensor:
        out_shape = torch.broadcast_shapes(out_shape, max.shape)

    # 类型推导 (Type Promotion)
    dummy_input = input.new_empty((0,))
    dummy_min = min if not is_min_tensor else min.new_empty((0,))
    dummy_max = max if not is_max_tensor else max.new_empty((0,))
    out_dtype = torch.clamp(dummy_input, min=dummy_min, max=dummy_max).dtype

    # 输出内存分配
    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=input.device)
    else:
        assert out.shape == out_shape, f"out shape {out.shape} mismatch with broadcasted shape {out_shape}"
        out_dtype = out.dtype

    n_elements = out.numel()
    if n_elements == 0:
        return out

    # 内存连续化与全局广播铺平
    # 将所有的输入 Tensor 都强行扩展对齐到最终的 out_shape，然后连续化
    input_c = input.expand(out_shape).contiguous()

    # 处理 min 指针与常量
    min_ptr = input_c  # 占位废指针，防止 Triton 报错
    min_val = 0.0
    if has_min:
        if is_min_tensor:
            min_ptr = min.expand(out_shape).contiguous()
        else:
            min_val = float(min)

    # 处理 max 指针与常量
    max_ptr = input_c  # 占位废指针
    max_val = 0.0
    if has_max:
        if is_max_tensor:
            max_ptr = max.expand(out_shape).contiguous()
        else:
            max_val = float(max)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # 启动 Kernel
    with torch_device_fn.device(input.device):
        clamp_kernel[grid](
            input_c, out,
            min_ptr, max_ptr,
            min_val, max_val,
            n_elements,
            HAS_MIN=has_min, HAS_MAX=has_max,
            IS_MIN_TENSOR=is_min_tensor, IS_MAX_TENSOR=is_max_tensor,
            BLOCK_SIZE=BLOCK_SIZE
        )

    return out