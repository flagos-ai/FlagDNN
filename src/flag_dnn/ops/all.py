import logging
from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


@triton.jit
def _fill_bool_kernel(
    out_ptr, n_elements, VALUE: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.full([BLOCK_SIZE], VALUE, dtype=tl.int1)
    tl.store(out_ptr + offsets, vals, mask=mask)


@triton.jit
def _fill_i32_kernel(
    out_ptr, n_elements, VALUE: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.full([BLOCK_SIZE], VALUE, dtype=tl.int32)
    tl.store(out_ptr + offsets, vals, mask=mask)


@triton.jit
def _i32_to_bool_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(in_ptr + offsets, mask=mask, other=0) > 0
    tl.store(out_ptr + offsets, vals, mask=mask)


def _filled_bool(shape, value: bool, device) -> torch.Tensor:
    out = torch.empty(shape, dtype=torch.bool, device=device)
    n_elements = out.numel()
    if n_elements > 0:
        block = 1024
        with torch_device_fn.device(device):
            _fill_bool_kernel[(triton.cdiv(n_elements, block),)](
                out, n_elements, VALUE=value, BLOCK_SIZE=block
            )
    return out


def _filled_i32(shape, value: int, device) -> torch.Tensor:
    out = torch.empty(shape, dtype=torch.int32, device=device)
    n_elements = out.numel()
    if n_elements > 0:
        block = 1024
        with torch_device_fn.device(device):
            _fill_i32_kernel[(triton.cdiv(n_elements, block),)](
                out, n_elements, VALUE=value, BLOCK_SIZE=block
            )
    return out


def _i32_to_bool(inp: torch.Tensor) -> torch.Tensor:
    out = torch.empty(inp.shape, dtype=torch.bool, device=inp.device)
    n_elements = inp.numel()
    if n_elements > 0:
        block = 1024
        with torch_device_fn.device(inp.device):
            _i32_to_bool_kernel[(triton.cdiv(n_elements, block),)](
                inp, out, n_elements, BLOCK_SIZE=block
            )
    return out


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.jit
def _all_direct_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    row_ptr = x_ptr + pid_m * stride_xm
    n_offsets = tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N
    x = (
        tl.load(row_ptr + n_offsets * stride_xn, mask=n_mask, other=1) != 0
    ).to(tl.int32)
    result = tl.min(x, axis=0)
    tl.store(out_ptr + pid_m, result.to(tl.int8))


@triton.jit
def _all_direct_multi_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Process BLOCK_M rows per CTA to reduce scheduling overhead."""
    pid = tle.program_id(0)
    row_start = pid * BLOCK_M

    m_offsets = row_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    m_mask = m_offsets < M
    n_offsets = tl.arange(0, BLOCK_N)  # [BLOCK_N]
    n_mask = n_offsets < N

    ptrs = (
        x_ptr + m_offsets[:, None] * stride_xm + n_offsets[None, :] * stride_xn
    )
    x = tl.load(ptrs, mask=m_mask[:, None] & n_mask[None, :], other=1)
    nonzero = (x != 0).to(tl.int32)  # [BLOCK_M, BLOCK_N]
    result = tl.min(nonzero, axis=1)  # [BLOCK_M] per-row min
    tl.store(out_ptr + m_offsets, result.to(tl.int8), mask=m_mask)


@triton.jit
def _all_stage1_kernel(
    x_ptr,
    partial_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

    row_ptr = x_ptr + pid_m * stride_xm
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    x = (
        tl.load(row_ptr + n_offsets * stride_xn, mask=n_mask, other=1) != 0
    ).to(tl.int32)
    local_all = tl.min(x, axis=0)

    num_blocks_n = tl.num_programs(1)
    tl.store(partial_ptr + pid_m * num_blocks_n + pid_n, local_all)


@triton.jit
def _all_finalize_kernel(
    partial_ptr,
    out_ptr,
    M,
    K,
    BLOCK_K: tl.constexpr,
):
    pid_m = tle.program_id(0)

    k_offsets = tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K
    part = (
        tl.load(partial_ptr + pid_m * K + k_offsets, mask=k_mask, other=1) != 0
    ).to(tl.int32)
    result = tl.min(part, axis=0)
    tl.store(out_ptr + pid_m, result)


@triton.jit
def _all_col2d_s1_kernel(
    x_ptr,
    partial_ptr,
    M_red,
    M_out,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Stage 1: partial column-ALL on [BLOCK_M, BLOCK_N] tiles."""
    pid_n = tle.program_id(0)
    pid_m = tle.program_id(1)

    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = col_offsets < M_out
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < M_red

    ptrs = x_ptr + row_offsets[:, None] * M_out + col_offsets[None, :]
    mask_2d = row_mask[:, None] & col_mask[None, :]
    x = (tl.load(ptrs, mask=mask_2d, other=1) != 0).to(tl.int32)
    tile_all = tl.min(x, axis=0)

    partial_ptrs = partial_ptr + pid_m * M_out + col_offsets
    tl.store(partial_ptrs, tile_all, mask=col_mask)


@triton.jit
def _all_col2d_s2_kernel(
    partial_ptr,
    out_ptr,
    n_groups,
    M_out,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Stage 2: finalize partial results."""
    pid_n = tle.program_id(0)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = col_offsets < M_out

    acc = tl.full([BLOCK_N], 1, dtype=tl.int8)

    for k_start in range(0, n_groups, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < n_groups
        ptrs = partial_ptr + k_offsets[:, None] * M_out + col_offsets[None, :]
        mask_2d = k_mask[:, None] & col_mask[None, :]
        partial = (tl.load(ptrs, mask=mask_2d, other=1) != 0).to(tl.int32)
        row_all = tl.min(partial, axis=0)
        acc = tl.minimum(acc, row_all)

    tl.store(out_ptr + col_offsets, acc, mask=col_mask)


def _all_reduce_2d(x: torch.Tensor, M: int, N: int) -> torch.Tensor:
    if N == 0:
        return _filled_bool((M,), True, x.device)

    BLOCK_N = min(triton.next_power_of_2(N), 8192)
    num_blocks_n = triton.cdiv(N, BLOCK_N)

    if x.dim() >= 2:
        stride_m = x.stride(0)
        stride_n = x.stride(-1)
    else:
        stride_m = 0
        stride_n = x.stride(0)

    out = torch.empty(M, dtype=torch.int8, device=x.device)
    if num_blocks_n == 1:
        if M <= 4096:
            BLOCK_M_MR = 8
            num_warps_mr = 8
            grid_mr = (triton.cdiv(M, BLOCK_M_MR),)
            with torch_device_fn.device(x.device):
                _all_direct_multi_kernel[grid_mr](
                    x,
                    out,
                    M,
                    N,
                    stride_m,
                    stride_n,
                    BLOCK_M=BLOCK_M_MR,
                    BLOCK_N=BLOCK_N,
                    num_warps=num_warps_mr,
                )
        else:
            with torch_device_fn.device(x.device):
                _all_direct_kernel[(M,)](
                    x, out, M, N, stride_m, stride_n, BLOCK_N=BLOCK_N
                )
    else:
        if not x.is_contiguous():
            x = x.contiguous()
        stride_m = x.stride(0)
        stride_n = x.stride(-1)

        partial = torch.empty(
            M * num_blocks_n, dtype=torch.int32, device=x.device
        )
        grid1 = (M, num_blocks_n)
        with torch_device_fn.device(x.device):
            _all_stage1_kernel[grid1](
                x,
                partial,
                M,
                N,
                stride_m,
                stride_n,
                BLOCK_N=BLOCK_N,
            )

        BLOCK_K = min(triton.next_power_of_2(num_blocks_n), 1024)
        grid2 = (M,)
        with torch_device_fn.device(x.device):
            _all_finalize_kernel[grid2](
                partial, out, M, num_blocks_n, BLOCK_K=BLOCK_K
            )

    return out.view(torch.bool)


@triton.jit
def _all_col_atomic_kernel(
    x_ptr,
    out_ptr,
    M_red,
    M_out,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Single-kernel column ALL-reduction using atomic_min.
    Grid: (ceil(M_out/BLOCK_N), ceil(M_red/BLOCK_M)).
    Initialized with 1; any False → atomic_min sets output to 0."""
    pid_n = tle.program_id(0)
    pid_m = tle.program_id(1)

    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = col_offsets < M_out
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < M_red

    ptrs = x_ptr + row_offsets[:, None] * M_out + col_offsets[None, :]
    x = (
        tl.load(ptrs, mask=row_mask[:, None] & col_mask[None, :], other=1) != 0
    ).to(tl.int32)
    tile_all = tl.min(x, axis=0)  # [BLOCK_N]

    tl.atomic_min(out_ptr + col_offsets, tile_all, mask=col_mask)


def _all_col2d_reduce(
    x_flat: torch.Tensor, M_red: int, M_out: int
) -> torch.Tensor:
    """Single-kernel atomic column ALL-reduction. High-occupancy."""
    if M_out == 0 or M_red == 0:
        return _filled_bool((M_out,), True, x_flat.device)

    BLOCK_M = 16
    BLOCK_N = 32
    grid = (triton.cdiv(M_out, BLOCK_N), triton.cdiv(M_red, BLOCK_M))

    # Initialize with 1 (all True; any False will reduce to 0)
    out = _filled_i32((M_out,), 1, x_flat.device)
    with torch_device_fn.device(x_flat.device):
        _all_col_atomic_kernel[grid](
            x_flat,
            out,
            M_red,
            M_out,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
    return _i32_to_bool(out)


def all(
    input: torch.Tensor,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logger.debug(f"FLAG_DNN ALL (dim={dim})")

    ndim = input.ndim

    if dim is None:
        N = input.numel()
        x_flat = input.contiguous().view(-1)
        result = _all_reduce_2d(x_flat.unsqueeze(0), 1, N)
        result = result.view(())
    elif isinstance(dim, int) and ndim == 2 and input.is_contiguous():
        # Fast path: bypass Python overhead for common 2D contiguous case
        d = dim if dim >= 0 else dim + 2
        M_rows, N_cols = input.shape[0], input.shape[1]
        if d == 1:
            result = _all_reduce_2d(input, M_rows, N_cols)
            if keepdim:
                result = result.unsqueeze(1)
        else:  # d == 0
            result = _all_col2d_reduce(input, M_rows, N_cols)
            if keepdim:
                result = result.unsqueeze(0)
        if out is not None:
            out.copy_(result)
            return out
        return result
    else:
        if isinstance(dim, int):
            dims = [dim]
        else:
            dims = list(dim)
        dims = sorted([d if d >= 0 else d + ndim for d in dims])

        kept_dims = [i for i in range(ndim) if i not in dims]
        M_out = 1
        for d in kept_dims:
            M_out *= input.shape[d]
        N_red = 1
        for d in dims:
            N_red *= input.shape[d]

        perm_view = input.permute(*kept_dims, *dims)

        BLOCK_N = min(triton.next_power_of_2(N_red), 8192) if N_red > 0 else 1
        num_blocks_n = triton.cdiv(N_red, BLOCK_N) if N_red > 0 else 1

        dims_are_leading = dims == list(range(len(dims)))

        if perm_view.is_contiguous():
            input_view = perm_view.view(M_out, N_red)
            result_flat = _all_reduce_2d(input_view, M_out, N_red)
        elif input.is_contiguous() and dims_are_leading:
            result_flat = _all_col2d_reduce(
                input.view(N_red, M_out), N_red, M_out
            )
        elif (
            num_blocks_n == 1
            and perm_view.ndim == 2
            and perm_view.stride(-1) == 1
        ):
            result_flat = _all_reduce_2d(perm_view, M_out, N_red)
        else:
            input_view = perm_view.contiguous().view(M_out, N_red)
            result_flat = _all_reduce_2d(input_view, M_out, N_red)

        out_shape = []
        for i in range(ndim):
            if i in dims:
                if keepdim:
                    out_shape.append(1)
            else:
                out_shape.append(input.shape[i])
        if not out_shape:
            out_shape = []
        result = result_flat.reshape(out_shape if out_shape else (1,))
        if not out_shape:
            result = result.squeeze()

    if out is not None:
        out.copy_(result)
        return out
    return result
