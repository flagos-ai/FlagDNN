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
def mean_2d_kernel(
    in_ptr, out_ptr,
    M, N,
    in_stride_m, in_stride_n,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    row_start_ptr = in_ptr + pid_m * in_stride_m
    offsets_n = tl.arange(0, BLOCK_SIZE_N)
    
    # 累加器，统一使用 float32 防止中间计算溢出
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    # 循环读取并累加
    for n_offset in range(0, N, BLOCK_SIZE_N):
        cols = n_offset + offsets_n
        mask = cols < N
        
        # other=0 安全兜底，防止类型污染
        val = tl.load(row_start_ptr + cols * in_stride_n, mask=mask, other=0)
        acc += val.to(tl.float32)
        
    # Block 内求和
    row_sum = tl.sum(acc, axis=0)
    
    # 求均值，N 会隐式提升为 float32 与 row_sum 进行计算
    row_mean = row_sum / N
    
    # 存入目标地址
    tl.store(out_ptr + pid_m, row_mean.to(out_ptr.dtype.element_ty))


def mean(
    input: torch.Tensor,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    logger.debug("FLAG_DNN MEAN")

    # 计算出正确的输出形状
    out_shape = []
    if dim is None:
        if keepdim:
            out_shape = [1] * input.ndim
    else:
        dims = (dim,) if isinstance(dim, int) else dim
        dims = tuple(d if d >= 0 else d + input.ndim for d in dims)
        
        # 重复的 dim 报错
        if len(set(dims)) != len(dims):
            raise RuntimeError("dim appears multiple times in the list of dims")

        for i in range(input.ndim):
            if i in dims:
                if keepdim:
                    out_shape.append(1)
            else:
                out_shape.append(input.shape[i])
    out_shape = tuple(out_shape)

    # 类型推导与合法性校验
    dummy_input = input.new_empty((0,))
    out_dtype = torch.mean(dummy_input, dtype=dtype).dtype

    # 处理 out 参数及显存分配
    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=input.device)
    else:
        assert out.shape == out_shape, f"out shape {out.shape} mismatch with expected shape {out_shape}"
        # 原地操作时，类型以传入的 out 张量为准
        out_dtype = out.dtype

    # 空张量边界处理 (必须返回 NaN)
    if input.numel() == 0:
        out.fill_(float('nan'))
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
        mean_2d_kernel[grid](
            input_c,
            out.view(-1),
            M, N,
            in_stride_m, in_stride_n,
            BLOCK_SIZE_N=BLOCK_SIZE_N
        )

    return out