import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


# 自定义乘法组合函数，供 tl.reduce 使用 (Triton 没有内置 tl.prod)
@triton.jit
def _prod_combine(a, b):
    return a * b


@triton.jit
def prod_2d_kernel(
    in_ptr, out_ptr,
    M, N,
    in_stride_m, in_stride_n,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    row_start_ptr = in_ptr + pid_m * in_stride_m
    offsets_n = tl.arange(0, BLOCK_SIZE_N)
    
    # 累加器：初始值必须是 1.0 (使用 float32 防止乘积快速溢出)
    acc = tl.full((BLOCK_SIZE_N,), 1.0, dtype=tl.float32)
    
    for n_offset in range(0, N, BLOCK_SIZE_N):
        cols = n_offset + offsets_n
        mask = cols < N
        
        val = tl.load(row_start_ptr + cols * in_stride_n, mask=mask, other=1)
        acc *= val.to(tl.float32)
        
    # Block 内求积，使用自定义的 _prod_combine
    row_prod = tl.reduce(acc, axis=0, combine_fn=_prod_combine)
    tl.store(out_ptr + pid_m, row_prod.to(out_ptr.dtype.element_ty))


def prod(
    input: torch.Tensor,
    dim: Optional[int] = None,  # 【严格对齐 API】prod 的 dim 只能是 int 或 None
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    logger.debug("FLAG_DNN PROD")

    # 计算输出形状
    out_shape = []
    if dim is None:
        if keepdim:
            out_shape = [1] * input.ndim
    else:
        # 负数维度对齐
        d = dim if dim >= 0 else dim + input.ndim
        
        for i in range(input.ndim):
            if i == d:
                if keepdim:
                    out_shape.append(1)
            else:
                out_shape.append(input.shape[i])
                
    out_shape = tuple(out_shape)

    # 类型推导
    dummy_input = input.new_empty((0,))
    out_dtype = torch.prod(dummy_input, dtype=dtype).dtype

    # 分配输出张量
    out = torch.empty(out_shape, dtype=out_dtype, device=input.device)

    # 如果是空张量，直接填 1 并返回 (乘积的空集定义为 1)
    if input.numel() == 0:
        out.fill_(1)
        return out

    # 高维规约转 2D 矩阵规约
    if dim is None:
        M = 1
        N = input.numel()
        input_c = input.contiguous().view(M, N)
    else:
        keep_dims = [i for i in range(input.ndim) if i != d]
        permuted_dims = keep_dims + [d]
        
        input_permuted = input.permute(*permuted_dims).contiguous()
        
        M = 1
        for i in keep_dims:
            M *= input.shape[i]
        N = input.shape[d]
            
        input_c = input_permuted.view(M, N)

    in_stride_m = input_c.stride(0)
    in_stride_n = input_c.stride(1)

    BLOCK_SIZE_N = 1024
    grid = (M,)

    # 启动 Kernel
    with torch_device_fn.device(input.device):
        prod_2d_kernel[grid](
            input_c, 
            out.view(-1),
            M, N,
            in_stride_m, in_stride_n,
            BLOCK_SIZE_N=BLOCK_SIZE_N
        )

    return out