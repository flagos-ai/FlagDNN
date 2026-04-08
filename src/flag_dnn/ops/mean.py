import logging
from typing import Optional, Union, Tuple

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 8192}, num_warps=8),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 4096}, num_warps=8),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 2048}, num_warps=8),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 512, "BLOCK_N": 16}, num_warps=4),
        triton.Config({"BLOCK_M": 1024, "BLOCK_N": 8}, num_warps=8),
        triton.Config({"BLOCK_M": 2048, "BLOCK_N": 4}, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4),
    ],
    key=["M", "N"],
)
@triton.jit
def _mean_kernel_2d_loop(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        mask = m_mask[:, None] & n_mask[None, :]
        x_ptrs = (
            x_ptr
            + m_offsets[:, None] * stride_xm
            + n_offsets[None, :] * stride_xn
        )

        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x = x.to(tl.float32)

        acc += tl.sum(x, axis=1)

    out_ptrs = out_ptr + m_offsets
    # 写入前计算平均值
    tl.store(out_ptrs, acc / N, mask=m_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 8192}, num_warps=8),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 4096}, num_warps=8),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 2048}, num_warps=8),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 512, "BLOCK_N": 16}, num_warps=4),
        triton.Config({"BLOCK_M": 1024, "BLOCK_N": 8}, num_warps=8),
        triton.Config({"BLOCK_M": 2048, "BLOCK_N": 4}, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4),
    ],
    key=["M", "N"],
)
@triton.jit
def _mean_kernel_3d_loop(
    x_ptr,
    out_ptr,
    M,
    N,
    I,
    stride_xo,
    stride_xr,
    stride_xi,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    o_idx = m_offsets // I
    i_idx = m_offsets % I
    base_ptrs = x_ptr + (o_idx * stride_xo + i_idx * stride_xi)

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        mask = m_mask[:, None] & n_mask[None, :]
        x_ptrs = base_ptrs[:, None] + n_offsets[None, :] * stride_xr

        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x = x.to(tl.float32)

        acc += tl.sum(x, axis=1)

    out_ptrs = out_ptr + m_offsets
    # 写入前计算平均值
    tl.store(out_ptrs, acc / N, mask=m_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 8192}, num_warps=8),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 4096}, num_warps=8),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 2048}, num_warps=8),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 512, "BLOCK_N": 16}, num_warps=4),
        triton.Config({"BLOCK_M": 1024, "BLOCK_N": 8}, num_warps=8),
        triton.Config({"BLOCK_M": 2048, "BLOCK_N": 4}, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4),
    ],
    key=["M", "N"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _mean_kernel_2d_atomic(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N

    mask = m_mask[:, None] & n_mask[None, :]
    x_ptrs = (
        x_ptr + m_offsets[:, None] * stride_xm + n_offsets[None, :] * stride_xn
    )

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    x = x.to(tl.float32)

    # 累加前直接除以 N，根据分配律等价于总体求均值
    sum_vals = tl.sum(x, axis=1) / N

    out_ptrs = out_ptr + m_offsets
    tl.atomic_add(out_ptrs, sum_vals, mask=m_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 8192}, num_warps=8),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 4096}, num_warps=8),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 2048}, num_warps=8),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 512, "BLOCK_N": 16}, num_warps=4),
        triton.Config({"BLOCK_M": 1024, "BLOCK_N": 8}, num_warps=8),
        triton.Config({"BLOCK_M": 2048, "BLOCK_N": 4}, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4),
    ],
    key=["M", "N"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _mean_kernel_3d_atomic(
    x_ptr,
    out_ptr,
    M,
    N,
    I,
    stride_xo,
    stride_xr,
    stride_xi,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N

    o_idx = m_offsets // I
    i_idx = m_offsets % I

    mask = m_mask[:, None] & n_mask[None, :]
    x_ptrs = x_ptr + (
        o_idx[:, None] * stride_xo
        + n_offsets[None, :] * stride_xr
        + i_idx[:, None] * stride_xi
    )

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    x = x.to(tl.float32)

    # 累加前直接除以 N
    sum_vals = tl.sum(x, axis=1) / N

    out_ptrs = out_ptr + m_offsets
    tl.atomic_add(out_ptrs, sum_vals, mask=m_mask)


def mean(
    input: torch.Tensor,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logger.debug("FLAG_DNN MEAN")
    target_dtype = dtype if dtype is not None else input.dtype

    if target_dtype in (torch.int8, torch.int16, torch.bool):
        target_dtype = torch.float32

    ndim = input.ndim
    if dim is None:
        dims = list(range(ndim))
    elif isinstance(dim, int):
        dims = [dim]
    else:
        dims = list(dim)

    dims = [d if d >= 0 else d + ndim for d in dims]
    dims = sorted(dims)

    out_shape = []
    for i in range(ndim):
        if i in dims:
            if keepdim:
                out_shape.append(1)
        else:
            out_shape.append(input.shape[i])

    if not out_shape and not keepdim:
        out_shape = []

    acc_dtype = torch.float32

    is_reduce_at_end = dims == list(range(ndim - len(dims), ndim))

    is_consecutive = True
    for j in range(len(dims) - 1):
        if dims[j] + 1 != dims[j + 1]:
            is_consecutive = False
            break

    def _launch_kernel(M, N, input_view, is_3d=False, I_dim=1):
        if M == 0 or N == 0:
            val = float("nan") if N == 0 else 0.0
            return torch.full(
                out_shape, val, dtype=target_dtype, device=input.device
            )

        out_buffer = torch.zeros((M,), dtype=acc_dtype, device=input.device)

        use_atomic = (M <= 32) and (N > 65536)

        if not is_3d:
            if use_atomic:

                def grid_2d_atomic(meta):
                    return (
                        triton.cdiv(M, meta["BLOCK_M"])
                        * triton.cdiv(N, meta["BLOCK_N"]),
                    )

                _mean_kernel_2d_atomic[grid_2d_atomic](
                    input_view,
                    out_buffer,
                    M,
                    N,
                    input_view.stride(0),
                    input_view.stride(1),
                )
            else:

                def grid_2d_loop(meta):
                    return (triton.cdiv(M, meta["BLOCK_M"]),)

                _mean_kernel_2d_loop[grid_2d_loop](
                    input_view,
                    out_buffer,
                    M,
                    N,
                    input_view.stride(0),
                    input_view.stride(1),
                )
        else:
            if use_atomic:

                def grid_3d_atomic(meta):
                    return (
                        triton.cdiv(M, meta["BLOCK_M"])
                        * triton.cdiv(N, meta["BLOCK_N"]),
                    )

                _mean_kernel_3d_atomic[grid_3d_atomic](
                    input_view,
                    out_buffer,
                    M,
                    N,
                    I_dim,
                    input_view.stride(0),
                    input_view.stride(1),
                    input_view.stride(2),
                )
            else:

                def grid_3d_loop(meta):
                    return (triton.cdiv(M, meta["BLOCK_M"]),)

                _mean_kernel_3d_loop[grid_3d_loop](
                    input_view,
                    out_buffer,
                    M,
                    N,
                    I_dim,
                    input_view.stride(0),
                    input_view.stride(1),
                    input_view.stride(2),
                )

        return out_buffer

    with torch_device_fn.device(input.device):
        if is_reduce_at_end:
            M, N = 1, 1
            for i in range(ndim - len(dims)):
                M *= input.shape[i]
            for i in range(ndim - len(dims), ndim):
                N *= input.shape[i]
            out_buffer = _launch_kernel(M, N, input.reshape(M, N), is_3d=False)
        elif is_consecutive:
            dim_min, dim_max = dims[0], dims[-1]
            O, R, I_dim = 1, 1, 1
            for j in range(dim_min):
                O *= input.shape[j]
            for j in range(dim_min, dim_max + 1):
                R *= input.shape[j]
            for j in range(dim_max + 1, ndim):
                I_dim *= input.shape[j]
            M, N = O * I_dim, R
            out_buffer = _launch_kernel(
                M, N, input.reshape(O, R, I_dim), is_3d=True, I_dim=I_dim
            )
        else:
            kept_dims = [i for i in range(ndim) if i not in dims]
            M, N = 1, 1
            for d in kept_dims:
                M *= input.shape[d]
            for d in dims:
                N *= input.shape[d]
            input_view = input.permute(*kept_dims, *dims).reshape(M, N)
            out_buffer = _launch_kernel(M, N, input_view, is_3d=False)

    out_buffer = out_buffer.to(target_dtype).reshape(out_shape)

    if out is not None:
        out.resize_(out_shape)
        out.copy_(out_buffer)
        return out

    return out_buffer
