import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


@triton.jit
def _prod_combine(a, b):
    return a * b


@triton.autotune(
    configs=[
        # 1. 针对 N 极大，M 极小的极端场景 (Reduce at end)
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 8192}, num_warps=8),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 4096}, num_warps=8),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 2048}, num_warps=8),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=8),
        # 2. 针对 M 极大，N 极小的极端场景 (例如 dim=0 的情况)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 512, "BLOCK_N": 16}, num_warps=4),
        triton.Config({"BLOCK_M": 1024, "BLOCK_N": 8}, num_warps=8),
        triton.Config({"BLOCK_M": 2048, "BLOCK_N": 4}, num_warps=8),
        # 3. 增加高 Warp 数（8 warps）加强版，提升带宽高负载时的吞吐
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8),
        # 4. 常规均衡态
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4),
    ],
    key=["M", "N"],
)
@triton.jit
def _prod_kernel_2d_loop(
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

    acc = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        mask = m_mask[:, None] & n_mask[None, :]
        x_ptrs = (
            x_ptr
            + m_offsets[:, None] * stride_xm
            + n_offsets[None, :] * stride_xn
        )

        x = tl.load(x_ptrs, mask=mask, other=1.0)
        x = x.to(tl.float32)

        acc *= tl.reduce(x, axis=1, combine_fn=_prod_combine)

    out_ptrs = out_ptr + m_offsets
    tl.store(out_ptrs, acc, mask=m_mask)


@triton.autotune(
    configs=[
        # 1. 针对 N 极大，M 极小的极端场景 (Reduce at end)
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 8192}, num_warps=8),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 4096}, num_warps=8),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 2048}, num_warps=8),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=8),
        # 2. 针对 M 极大，N 极小的极端场景 (例如 dim=0 的情况)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 512, "BLOCK_N": 16}, num_warps=4),
        triton.Config({"BLOCK_M": 1024, "BLOCK_N": 8}, num_warps=8),
        triton.Config({"BLOCK_M": 2048, "BLOCK_N": 4}, num_warps=8),
        # 3. 增加高 Warp 数（8 warps）加强版，提升带宽高负载时的吞吐
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8),
        # 4. 常规均衡态
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4),
    ],
    key=["M", "N"],
)
@triton.jit
def _prod_kernel_3d_loop(
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

    acc = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        mask = m_mask[:, None] & n_mask[None, :]
        x_ptrs = base_ptrs[:, None] + n_offsets[None, :] * stride_xr

        x = tl.load(x_ptrs, mask=mask, other=1.0)
        x = x.to(tl.float32)

        acc *= tl.reduce(x, axis=1, combine_fn=_prod_combine)

    out_ptrs = out_ptr + m_offsets
    tl.store(out_ptrs, acc, mask=m_mask)


@triton.jit
def _prod_kernel_2d_split_stage1(
    x_ptr,
    partial_ptr,
    M,
    N,
    PARTIAL_N,
    stride_xm,
    stride_xn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    pid_m = pid // PARTIAL_N
    pid_n = pid % PARTIAL_N

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N
    mask = m_mask[:, None] & n_mask[None, :]

    x_ptrs = (
        x_ptr + m_offsets[:, None] * stride_xm + n_offsets[None, :] * stride_xn
    )
    x = tl.load(x_ptrs, mask=mask, other=1.0)
    x = x.to(tl.float32)

    part_vals = tl.reduce(x, axis=1, combine_fn=_prod_combine)

    # partial buffer layout: [M, PARTIAL_N]
    partial_ptrs = partial_ptr + m_offsets * PARTIAL_N + pid_n
    tl.store(partial_ptrs, part_vals, mask=m_mask)


@triton.jit
def _prod_kernel_3d_split_stage1(
    x_ptr,
    partial_ptr,
    M,
    N,
    I,
    PARTIAL_N,
    stride_xo,
    stride_xr,
    stride_xi,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    pid_m = pid // PARTIAL_N
    pid_n = pid % PARTIAL_N

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N
    mask = m_mask[:, None] & n_mask[None, :]

    o_idx = m_offsets // I
    i_idx = m_offsets % I

    x_ptrs = x_ptr + (
        o_idx[:, None] * stride_xo
        + n_offsets[None, :] * stride_xr
        + i_idx[:, None] * stride_xi
    )

    x = tl.load(x_ptrs, mask=mask, other=1.0)
    x = x.to(tl.float32)

    part_vals = tl.reduce(x, axis=1, combine_fn=_prod_combine)

    # partial buffer layout: [M, PARTIAL_N]
    partial_ptrs = partial_ptr + m_offsets * PARTIAL_N + pid_n
    tl.store(partial_ptrs, part_vals, mask=m_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_I": 128}, num_warps=4),
        triton.Config({"BLOCK_I": 256}, num_warps=4),
        triton.Config({"BLOCK_I": 512}, num_warps=4),
        triton.Config({"BLOCK_I": 1024}, num_warps=8),
        triton.Config({"BLOCK_I": 2048}, num_warps=8),
    ],
    key=["O", "R", "I"],
)
@triton.jit
def _prod_kernel_3d_inner_parallel(
    x_ptr,
    out_ptr,
    O,
    R,
    I,
    stride_xo,
    stride_xr,
    stride_xi,
    BLOCK_I: tl.constexpr,
):
    pid_o = tle.program_id(0)  # 对应 O 维
    pid_i = tle.program_id(1)  # 对应 I 维分块

    i_offsets = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    i_mask = i_offsets < I

    # 给编译器一点连续性提示
    i_offsets = tl.max_contiguous(i_offsets, BLOCK_I)

    acc = tl.full((BLOCK_I,), 1.0, dtype=tl.float32)

    base_ptrs = x_ptr + pid_o * stride_xo + i_offsets * stride_xi

    # 用 tl.range 做 reduction loop pipeline
    for r in range(0, R):
        x = tl.load(base_ptrs + r * stride_xr, mask=i_mask, other=1.0)
        x = x.to(tl.float32)
        acc *= x

    # 写回到展平后的一维输出: [O, I] -> [O*I]
    out_ptrs = out_ptr + pid_o * I + i_offsets
    tl.store(out_ptrs, acc, mask=i_mask)


def prod(
    input: torch.Tensor,
    dim: Optional[int] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    logger.debug("FLAG_DNN SUM")
    target_dtype = dtype if dtype is not None else input.dtype

    if target_dtype in (torch.int8, torch.int16, torch.bool):
        target_dtype = torch.int64

    ndim = input.ndim
    if dim is None:
        dims = list(range(ndim))
    elif isinstance(dim, int):
        if dim < -ndim or dim >= ndim:
            raise IndexError(
                f"Dimension out of range "
                f"(expected to be in range "
                f"of [{-ndim}, {ndim - 1}], "
                f"but got {dim})"
            )
        dims = [dim]
    else:
        raise AssertionError("Not Support dim is tuple")

    dims = [d if d >= 0 else d + ndim for d in dims]

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

    def _launch_kernel_split_n(M, N, input_view, is_3d=False, I_dim=1):
        if M == 0:
            return torch.empty(
                out_shape, dtype=target_dtype, device=input.device
            )

        if input.dtype in (torch.float16, torch.bfloat16):
            BLOCK_N_SPLIT = 16384
            BLOCK_M_SPLIT = 8
        else:
            BLOCK_N_SPLIT = 8192
            BLOCK_M_SPLIT = 4 if input.dtype == torch.float64 else 8

        num_tiles = triton.cdiv(N, BLOCK_N_SPLIT)
        partial_buffer = torch.empty(
            (M, num_tiles), dtype=acc_dtype, device=input_view.device
        )

        if not is_3d:
            grid = (triton.cdiv(M, BLOCK_M_SPLIT) * num_tiles,)
            _prod_kernel_2d_split_stage1[grid](
                input_view,
                partial_buffer,
                M,
                N,
                num_tiles,
                input_view.stride(0),
                input_view.stride(1),
                BLOCK_M=BLOCK_M_SPLIT,
                BLOCK_N=BLOCK_N_SPLIT,
                num_warps=8,
            )
        else:
            grid = (triton.cdiv(M, BLOCK_M_SPLIT) * num_tiles,)
            _prod_kernel_3d_split_stage1[grid](
                input_view,
                partial_buffer,
                M,
                N,
                I_dim,
                num_tiles,
                input_view.stride(0),
                input_view.stride(1),
                input_view.stride(2),
                BLOCK_M=BLOCK_M_SPLIT,
                BLOCK_N=BLOCK_N_SPLIT,
                num_warps=8,
            )

        # stage2: 再把 partial 做一次普通 reduce
        out_buffer = torch.empty(
            (M,), dtype=acc_dtype, device=input_view.device
        )

        def grid_2d_loop(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]),)

        _prod_kernel_2d_loop[grid_2d_loop](
            partial_buffer,
            out_buffer,
            M,
            num_tiles,
            partial_buffer.stride(0),
            partial_buffer.stride(1),
        )

        return out_buffer

    def _launch_kernel(M, N, input_view, is_3d=False, I_dim=1, O_dim=1):
        if M == 0:
            return torch.empty(
                out_shape, dtype=target_dtype, device=input.device
            )

        # full-reduce: M很小、N很大，走 split-N
        use_split_n = (M <= 4 and N >= (1 << 18)) or (
            M <= 32 and N >= (1 << 20)
        )

        # 非尾维归约专用：I 连续且比较大，R 适中
        # use_inner_parallel = (
        #     is_3d and
        #     input_view.stride(2) == 1 and
        #     O_dim >= 8 and
        #     I_dim >= 16384 and
        #     N <= 64
        # )
        use_inner_parallel = False

        if use_split_n:
            return _launch_kernel_split_n(
                M, N, input_view, is_3d=is_3d, I_dim=I_dim
            )

        out_buffer = torch.ones((M,), dtype=acc_dtype, device=input.device)

        if not is_3d:

            def grid_2d_loop(meta):
                return (triton.cdiv(M, meta["BLOCK_M"]),)

            _prod_kernel_2d_loop[grid_2d_loop](
                input_view,
                out_buffer,
                M,
                N,
                input_view.stride(0),
                input_view.stride(1),
            )
        else:
            if use_inner_parallel:

                def grid_3d_inner(meta):
                    return (O_dim, triton.cdiv(I_dim, meta["BLOCK_I"]))

                _prod_kernel_3d_inner_parallel[grid_3d_inner](
                    input_view,
                    out_buffer,
                    O_dim,
                    N,
                    I_dim,
                    input_view.stride(0),
                    input_view.stride(1),
                    input_view.stride(2),
                )
            else:

                def grid_3d_loop(meta):
                    return (triton.cdiv(M, meta["BLOCK_M"]),)

                _prod_kernel_3d_loop[grid_3d_loop](
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
        else:
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
                M,
                N,
                input.reshape(O, R, I_dim),
                is_3d=True,
                I_dim=I_dim,
                O_dim=O,
            )

    out = out_buffer.to(target_dtype).reshape(out_shape)
    return out
