import logging
from typing import Optional, Union, Tuple

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@triton.jit
def sum_2d_kernel(
    in_ptr, out_ptr,
    M, N,
    in_stride_m, in_stride_n,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    row_start_ptr = in_ptr + pid_m * in_stride_m
    offsets_n = tl.arange(0, BLOCK_SIZE_N)
    
    # 使用累加器
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    for n_offset in range(0, N, BLOCK_SIZE_N):
        cols = n_offset + offsets_n
        mask = cols < N
        
        # 使用 other=0 防止对 int32 或 bfloat16 的指针产生类型污染导致越界崩溃
        val = tl.load(row_start_ptr + cols * in_stride_n, mask=mask, other=0)
        acc += val.to(tl.float32)
        
    row_sum = tl.sum(acc, axis=0)
    tl.store(out_ptr + pid_m, row_sum.to(out_ptr.dtype.element_ty))


def sum(
    input: torch.Tensor,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    logger.debug("FLAG_DNN SUM")

    out_shape = []
    if dim is None:
        if keepdim:
            out_shape = [1] * input.ndim
    else:
        dims = (dim,) if isinstance(dim, int) else dim
        # 负数维度对齐
        dims = tuple(d if d >= 0 else d + input.ndim for d in dims)
        
        # 遇到重复的 dim 报错
        if len(set(dims)) != len(dims):
            raise RuntimeError("dim appears multiple times in the list of dims")
            
        for i in range(input.ndim):
            if i in dims:
                if keepdim:
                    out_shape.append(1)
            else:
                out_shape.append(input.shape[i])
                
    out_shape = tuple(out_shape)

    # 单独提取类型推导
    dummy_input = input.new_empty((0,))
    out_dtype = torch.sum(dummy_input, dtype=dtype).dtype

    # 分配输出张量
    out = torch.empty(out_shape, dtype=out_dtype, device=input.device)

    # 如果是空张量，直接填 0 并返回
    if input.numel() == 0:
        out.fill_(0)
        return out

    # 高维规约转 2D 矩阵规约
    if dim is None:
        M = 1
        N = input.numel()
        input_c = input.contiguous().view(M, N)
    else:
        keep_dims = [i for i in range(input.ndim) if i not in dims]
        permuted_dims = keep_dims + list(dims)
        
        input_permuted = input.permute(*permuted_dims).contiguous()
        
        M = 1
        for d in keep_dims:
            M *= input.shape[d]
        N = 1
        for d in dims:
            N *= input.shape[d]
            
        input_c = input_permuted.view(M, N)

    in_stride_m = input_c.stride(0)
    in_stride_n = input_c.stride(1)

    BLOCK_SIZE_N = 1024
    grid = (M,)

    # 启动 Kernel
    with torch_device_fn.device(input.device):
        sum_2d_kernel[grid](
            input_c,
            out.view(-1),
            M, N,
            in_stride_m, in_stride_n,
            BLOCK_SIZE_N=BLOCK_SIZE_N
        )

    return out