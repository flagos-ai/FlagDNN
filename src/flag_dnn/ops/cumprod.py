import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


# 自定义 Scan 的 Combine 函数 (乘法)
@triton.jit
def _prod_combine(a, b):
    return a * b

@triton.jit
def cumprod_2d_kernel(
    in_ptr, out_ptr,
    M, N,
    in_stride_m, in_stride_n,
    out_stride_m, out_stride_n,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    in_row_start_ptr = in_ptr + pid_m * in_stride_m
    out_row_start_ptr = out_ptr + pid_m * out_stride_m

    offsets_n = tl.arange(0, BLOCK_SIZE_N)
    
    # 循环剥离
    cols = offsets_n
    mask = cols < N
    
    val = tl.load(in_row_start_ptr + cols * in_stride_n, mask=mask, other=1)
    val = val.to(out_ptr.dtype.element_ty)
    
    # 使用 associative_scan 实现累乘
    chunk_cumprod = tl.associative_scan(val, axis=0, combine_fn=_prod_combine)
    tl.store(out_row_start_ptr + cols * out_stride_n, chunk_cumprod, mask=mask)
    
    # 记录当前 Chunk 的总乘积，传给下一个 Chunk
    running_prod = tl.reduce(val, axis=0, combine_fn=_prod_combine)
    
    for n_offset in range(BLOCK_SIZE_N, N, BLOCK_SIZE_N):
        cols = n_offset + offsets_n
        mask = cols < N
        
        val = tl.load(in_row_start_ptr + cols * in_stride_n, mask=mask, other=1)
        val = val.to(out_ptr.dtype.element_ty)
        
        # 算当前块内部的累乘
        chunk_cumprod = tl.associative_scan(val, axis=0, combine_fn=_prod_combine)
        
        # 加上（乘以）前面所有块的累计乘积
        out_val = chunk_cumprod * running_prod
        
        tl.store(out_row_start_ptr + cols * out_stride_n, out_val, mask=mask)
        
        # 更新 running_prod
        chunk_prod = tl.reduce(val, axis=0, combine_fn=_prod_combine)
        running_prod = running_prod * chunk_prod


def cumprod(
    input: torch.Tensor,
    dim: int,
    *,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    logger.debug("FLAG_DNN CUMPROD")

    # 推导输出类型
    dummy = torch.empty((0,), dtype=input.dtype, device=input.device)
    out_dtype = torch.cumprod(dummy, dim=0, dtype=dtype).dtype

    # 空张量边界处理
    if input.numel() == 0:
        res = torch.empty_like(input, dtype=out_dtype)
        if out is not None:
            if out.shape != res.shape:
                out.resize_(res.shape)
            out.copy_(res)
            return out
        return res

    # 维度处理与降维为 2D
    d = dim if dim >= 0 else dim + input.ndim
    if not (0 <= d < input.ndim):
        raise IndexError(f"Dimension out of range (expected to be in range of [-{input.ndim}, {input.ndim-1}], but got {dim})")

    keep_dims = [i for i in range(input.ndim) if i != d]
    permuted_dims = keep_dims + [d]
    
    # 将归约维度放到最后，并保证连续性
    input_permuted = input.permute(*permuted_dims).contiguous()
    
    M = 1
    for i in keep_dims:
        M *= input.shape[i]
    N = input.shape[d]
    
    input_c = input_permuted.view(M, N)
    
    # cumprod 保持原有形状
    out_c = torch.empty_like(input_c, dtype=out_dtype)

    in_stride_m = input_c.stride(0)
    in_stride_n = input_c.stride(1)
    out_stride_m = out_c.stride(0)
    out_stride_n = out_c.stride(1)

    BLOCK_SIZE_N = triton.next_power_of_2(N)
    if BLOCK_SIZE_N > 4096:
        BLOCK_SIZE_N = 4096

    grid = (M,)

    # 启动 Kernel
    with torch_device_fn.device(input.device):
        cumprod_2d_kernel[grid](
            input_c, out_c,
            M, N,
            in_stride_m, in_stride_n,
            out_stride_m, out_stride_n,
            BLOCK_SIZE_N=BLOCK_SIZE_N
        )

    # 恢复目标形状
    final_out = out_c.view(input_permuted.shape)
    inverse_perm = [0] * input.ndim
    for i, p in enumerate(permuted_dims):
        inverse_perm[p] = i
    final_out = final_out.permute(*inverse_perm)

    # 处理 out 参数
    if out is not None:
        if out.shape != input.shape:
            out.resize_(input.shape)
        out.copy_(final_out)
        return out

    return final_out