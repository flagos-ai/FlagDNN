import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@triton.jit
def cumsum_2d_kernel(
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
    
    # 循环剥离,初始化 running_sum，让 Triton 自动且正确地推导出数据类型
    cols = offsets_n
    mask = cols < N
    
    # other=0：防止越界访问带来的 NaN 或垃圾值污染累加结果
    val = tl.load(in_row_start_ptr + cols * in_stride_n, mask=mask, other=0)
    val = val.to(out_ptr.dtype.element_ty)
    
    chunk_cumsum = tl.cumsum(val, axis=0)
    tl.store(out_row_start_ptr + cols * out_stride_n, chunk_cumsum, mask=mask)
    
    # 初始化供下一个 Chunk 使用的 running_sum (降维成了一个标量)
    running_sum = tl.sum(val, axis=0)

    # 当 N 大于 BLOCK_SIZE_N 时触发
    for n_offset in range(BLOCK_SIZE_N, N, BLOCK_SIZE_N):
        cols = n_offset + offsets_n
        mask = cols < N
        
        val = tl.load(in_row_start_ptr + cols * in_stride_n, mask=mask, other=0)
        val = val.to(out_ptr.dtype.element_ty)
        
        # 计算当前 Chunk 的前缀和，并加上前面所有 Chunk 的总和
        chunk_cumsum = tl.cumsum(val, axis=0)
        out_val = chunk_cumsum + running_sum
        
        tl.store(out_row_start_ptr + cols * out_stride_n, out_val, mask=mask)
        
        # 将当前 Chunk 的总和累积到 running_sum 中
        running_sum += tl.sum(val, axis=0)


def cumsum(
    input: torch.Tensor,
    dim: int,
    *,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    logger.debug("FLAG_DNN CUMSUM")

    # 推导输出类型
    dummy = torch.empty((0,), dtype=input.dtype, device=input.device)
    out_dtype = torch.cumsum(dummy, dim=0, dtype=dtype).dtype

    # 空张量边界处理
    if input.numel() == 0:
        if out is not None:
            if out.shape != input.shape:
                out.resize_(input.shape)
            return out
        return torch.empty(input.shape, dtype=out_dtype, device=input.device)

    # 负数维度对齐与越界检查
    d = dim if dim >= 0 else dim + input.ndim
    if not (0 <= d < input.ndim):
        raise IndexError(f"Dimension out of range (expected to be in range of [-{input.ndim}, {input.ndim-1}], but got {dim})")

    # 将要计算的维度置换到最后一维，确保内存连续性
    keep_dims = [i for i in range(input.ndim) if i != d]
    permuted_dims = keep_dims + [d]
    
    input_permuted = input.permute(*permuted_dims).contiguous()
    
    # 拍扁为 2D 矩阵 (M 行 N 列)
    M = 1
    for i in keep_dims:
        M *= input.shape[i]
    N = input.shape[d]
    
    input_c = input_permuted.view(M, N)
    
    # 分配一段连续的中间内存，避免在 Kernel 里做高维度的步长反推
    out_c = torch.empty((M, N), dtype=out_dtype, device=input.device)

    in_stride_m = input_c.stride(0)
    in_stride_n = input_c.stride(1)
    out_stride_m = out_c.stride(0)
    out_stride_n = out_c.stride(1)

    # 跨 Chunk 的 running_sum 传递
    BLOCK_SIZE_N = 1024
    grid = (M,)

    # 6. 启动 Kernel
    with torch_device_fn.device(input.device):
        cumsum_2d_kernel[grid](
            input_c, out_c,
            M, N,
            in_stride_m, in_stride_n,
            out_stride_m, out_stride_n,
            BLOCK_SIZE_N=BLOCK_SIZE_N
        )

    # 施加“逆排列”，恢复到原来的内存形状和维度布局
    inverse_perm = [0] * input.ndim
    for i, p in enumerate(permuted_dims):
        inverse_perm[p] = i
        
    final_out = out_c.view(input_permuted.shape).permute(*inverse_perm)

    # 处理 out 参数
    if out is not None:
        if out.shape != input.shape:
            out.resize_(input.shape)
        out.copy_(final_out)
        return out

    return final_out