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
def ne_1d_kernel(
    in_ptr1, in_ptr2, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # 读取数据
    x = tl.load(in_ptr1 + offsets, mask=mask)
    y = tl.load(in_ptr2 + offsets, mask=mask)
    
    # 执行不相等比较 (结果为 tl.int1)
    res = x != y
    
    # 强制转型为 out_ptr 的元素类型 (通常是 int8/bool)，防止对齐问题
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def ne(
    input: torch.Tensor,
    other: Union[torch.Tensor, float, int, bool],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    logger.debug("FLAG_DNN NE")

    # 标量处理：正确处理 Python 标量，不强制指定 dtype，交由 PyTorch 自动推导
    if not isinstance(other, torch.Tensor):
        other_tensor = torch.tensor(other, device=input.device)
    else:
        other_tensor = other

    # 数据类型提升 (Type Promotion)
    compute_dtype = torch.promote_types(input.dtype, other_tensor.dtype)

    # 形状广播 (Broadcasting)
    try:
        input_b, other_b = torch.broadcast_tensors(input, other_tensor)
    except RuntimeError as e:
        raise RuntimeError(f"The size of tensor a ({input.shape}) must match the size of tensor b ({other_tensor.shape}) at non-singleton dimension") from e

    out_shape = input_b.shape
    N = input_b.numel()

    # out 参数及显存分配 (强制类型为 bool)
    if out is None:
        out = torch.empty(out_shape, dtype=torch.bool, device=input.device)
    else:
        assert out.shape == out_shape, f"out shape {out.shape} mismatch with broadcasted shape {out_shape}"

    # 空张量边界处理
    if N == 0:
        return out

    # 数据排布连续化，并打平为 1D
    input_c = input_b.to(compute_dtype).contiguous().view(-1)
    other_c = other_b.to(compute_dtype).contiguous().view(-1)
    out_c = out.view(-1)

    # 启动 Kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    with torch_device_fn.device(input.device):
        ne_1d_kernel[grid](
            input_c, other_c, out_c,
            N,
            BLOCK_SIZE=BLOCK_SIZE
        )

    return out