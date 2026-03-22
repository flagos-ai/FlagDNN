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
def eq_1d_kernel(
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
    
    # 执行相等比较 (结果为 tl.int1)
    res = x == y
    
    # PyTorch 的 torch.bool 在显存中以 1-byte (int8) 的形式存储
    # 必须将其转型后存储，否则会发生段错误或对齐问题
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def eq(
    input: torch.Tensor,
    other: Union[torch.Tensor, float, int, bool],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    logger.debug("FLAG_DNN EQ")

    # 标量处理：如果 other 是数字，把它变成 Tensor
    if not isinstance(other, torch.Tensor):
        # 切忌强制指定 dtype=input.dtype，否则会导致 0.5 变成 0
        # 让 PyTorch 自动推导该标量应有的 dtype (比如 float 会被推导为 float32 或 float64)
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
        # 原地写入时如果 out 不是 bool 也没关系，PyTorch 原生逻辑允许 cast，但正常场景通常为 bool

    # 空张量边界处理
    if N == 0:
        return out

    # 数据排布连续化，并打平为 1D
    # 这一步非常重要，因为 broadcast 出来的 tensor 其 stride 可能是 0
    input_c = input_b.to(compute_dtype).contiguous().view(-1)
    other_c = other_b.to(compute_dtype).contiguous().view(-1)
    out_c = out.view(-1)

    # 启动 Kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    with torch_device_fn.device(input.device):
        eq_1d_kernel[grid](
            input_c, other_c, out_c,
            N,
            BLOCK_SIZE=BLOCK_SIZE
        )

    return out