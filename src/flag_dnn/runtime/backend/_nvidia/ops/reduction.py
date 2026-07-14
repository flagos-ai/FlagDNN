import torch
import triton
import triton.language as tl

from flag_dnn.ops.abs import abs as flag_abs
from flag_dnn.ops.mean import mean
from flag_dnn.ops.mul import mul
from flag_dnn.ops.prod import prod
from flag_dnn.ops.sqrt import sqrt
from flag_dnn.ops.sum import sum
from flag_dnn.runtime import torch_device_fn

_REDUCE_CONFIGS = [
    triton.Config({"BLOCK_M": 1, "BLOCK_N": 1024}, num_warps=4),
    triton.Config({"BLOCK_M": 2, "BLOCK_N": 512}, num_warps=4),
    triton.Config({"BLOCK_M": 4, "BLOCK_N": 256}, num_warps=4),
    triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}, num_warps=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4),
]


@triton.autotune(configs=_REDUCE_CONFIGS, key=["M", "N"])
@triton.jit
def _reduction_2d_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    OP: tl.constexpr,
    IS_FP64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    if OP == "MIN":
        acc = tl.full(
            (BLOCK_M,),
            float("inf"),
            dtype=tl.float64 if IS_FP64 else tl.float32,
        )
    elif OP == "MUL_NO_ZEROS":
        acc = tl.full(
            (BLOCK_M,),
            1.0,
            dtype=tl.float64 if IS_FP64 else tl.float32,
        )
    else:
        acc = tl.full(
            (BLOCK_M,),
            -float("inf"),
            dtype=tl.float64 if IS_FP64 else tl.float32,
        )

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N
        ptrs = (
            x_ptr
            + m_offsets[:, None] * stride_xm
            + n_offsets[None, :] * stride_xn
        )
        mask = m_mask[:, None] & n_mask[None, :]
        if OP == "MIN":
            vals = tl.load(ptrs, mask=mask, other=float("inf"))
            vals = vals.to(tl.float64 if IS_FP64 else tl.float32)
            acc = tl.minimum(acc, tl.min(vals, axis=1))
        elif OP == "MUL_NO_ZEROS":
            vals = tl.load(ptrs, mask=mask, other=1.0)
            vals = vals.to(tl.float64 if IS_FP64 else tl.float32)
            vals = tl.where(vals == 0.0, 1.0, vals)
            products = tl.cumprod(vals, axis=1)
            last = tl.arange(0, BLOCK_N) == (BLOCK_N - 1)
            acc *= tl.sum(tl.where(last[None, :], products, 0.0), axis=1)
        else:
            other = 0.0 if OP == "AMAX" else -float("inf")
            vals = tl.load(ptrs, mask=mask, other=other)
            vals = vals.to(tl.float64 if IS_FP64 else tl.float32)
            if OP == "AMAX":
                vals = tl.abs(vals)
            acc = tl.maximum(acc, tl.max(vals, axis=1))

    tl.store(out_ptr + m_offsets, acc, mask=m_mask)


@triton.autotune(configs=_REDUCE_CONFIGS, key=["M", "N"])
@triton.jit
def _reduction_3d_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    I,
    stride_xo,
    stride_xr,
    stride_xi,
    OP: tl.constexpr,
    IS_FP64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M
    o_idx = m_offsets // I
    i_idx = m_offsets % I
    base_ptrs = x_ptr + o_idx * stride_xo + i_idx * stride_xi

    if OP == "MIN":
        acc = tl.full(
            (BLOCK_M,),
            float("inf"),
            dtype=tl.float64 if IS_FP64 else tl.float32,
        )
    elif OP == "MUL_NO_ZEROS":
        acc = tl.full(
            (BLOCK_M,),
            1.0,
            dtype=tl.float64 if IS_FP64 else tl.float32,
        )
    else:
        acc = tl.full(
            (BLOCK_M,),
            -float("inf"),
            dtype=tl.float64 if IS_FP64 else tl.float32,
        )

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N
        ptrs = base_ptrs[:, None] + n_offsets[None, :] * stride_xr
        mask = m_mask[:, None] & n_mask[None, :]
        if OP == "MIN":
            vals = tl.load(ptrs, mask=mask, other=float("inf"))
            vals = vals.to(tl.float64 if IS_FP64 else tl.float32)
            acc = tl.minimum(acc, tl.min(vals, axis=1))
        elif OP == "MUL_NO_ZEROS":
            vals = tl.load(ptrs, mask=mask, other=1.0)
            vals = vals.to(tl.float64 if IS_FP64 else tl.float32)
            vals = tl.where(vals == 0.0, 1.0, vals)
            products = tl.cumprod(vals, axis=1)
            last = tl.arange(0, BLOCK_N) == (BLOCK_N - 1)
            acc *= tl.sum(tl.where(last[None, :], products, 0.0), axis=1)
        else:
            other = 0.0 if OP == "AMAX" else -float("inf")
            vals = tl.load(ptrs, mask=mask, other=other)
            vals = vals.to(tl.float64 if IS_FP64 else tl.float32)
            if OP == "AMAX":
                vals = tl.abs(vals)
            acc = tl.maximum(acc, tl.max(vals, axis=1))

    tl.store(out_ptr + m_offsets, acc, mask=m_mask)


def _mode_name(mode) -> str:
    name = getattr(mode, "name", None)
    if name is None:
        name = str(mode).rsplit(".", 1)[-1]
    return str(name).upper()


def _dims(input: torch.Tensor, dim):
    rank = input.dim()
    if dim is None:
        return tuple(range(rank))
    if isinstance(dim, int):
        return dim if dim >= 0 else dim + rank
    return tuple(item if item >= 0 else item + rank for item in dim)


def _prod_reduce(
    input: torch.Tensor,
    dims,
    keepdim: bool,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if isinstance(dims, int):
        return prod(input, dim=dims, keepdim=keepdim, dtype=dtype)
    dim_tuple = tuple(dims)
    result = input
    ordered_dims = dim_tuple if keepdim else sorted(dim_tuple, reverse=True)
    for index, dim in enumerate(ordered_dims):
        result = prod(
            result,
            dim=dim,
            keepdim=keepdim,
            dtype=dtype if index == 0 else None,
        )
    return result


def _out_shape(
    input: torch.Tensor, dims: list[int], keepdim: bool
) -> list[int]:
    out_shape = []
    for axis in range(input.ndim):
        if axis in dims:
            if keepdim:
                out_shape.append(1)
        else:
            out_shape.append(input.shape[axis])
    return out_shape


def _native_reduction(
    input: torch.Tensor,
    mode_name: str,
    dims: list[int],
    keepdim: bool,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    target_dtype = dtype if dtype is not None else input.dtype
    dims = sorted(dims)
    if not dims:
        return input if dtype is None else input.to(dtype)
    if input.numel() == 0:
        if mode_name == "MUL_NO_ZEROS":
            return torch.ones(
                _out_shape(input, dims, keepdim),
                dtype=target_dtype,
                device=input.device,
            )
        raise RuntimeError(
            f"reduction mode {mode_name} cannot reduce an empty tensor"
        )

    ndim = input.ndim
    out_shape = _out_shape(input, dims, keepdim)
    is_reduce_at_end = dims == list(range(ndim - len(dims), ndim))
    is_consecutive = all(
        dims[index] + 1 == dims[index + 1] for index in range(len(dims) - 1)
    )

    def launch_2d(M: int, N: int, input_view: torch.Tensor) -> torch.Tensor:
        if N == 0:
            if mode_name == "MUL_NO_ZEROS":
                return torch.ones(
                    (M,), dtype=target_dtype, device=input.device
                )
            raise RuntimeError(
                f"reduction mode {mode_name} cannot reduce an empty dimension"
            )
        out_buffer = torch.empty((M,), dtype=target_dtype, device=input.device)
        is_fp64 = (
            input_view.dtype == torch.float64 or target_dtype == torch.float64
        )

        def grid(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]),)

        _reduction_2d_kernel[grid](
            input_view,
            out_buffer,
            M,
            N,
            input_view.stride(0),
            input_view.stride(1),
            mode_name,
            is_fp64,
        )
        return out_buffer

    def launch_3d(
        M: int, N: int, I_dim: int, input_view: torch.Tensor
    ) -> torch.Tensor:
        if N == 0:
            if mode_name == "MUL_NO_ZEROS":
                return torch.ones(
                    (M,), dtype=target_dtype, device=input.device
                )
            raise RuntimeError(
                f"reduction mode {mode_name} cannot reduce an empty dimension"
            )
        out_buffer = torch.empty((M,), dtype=target_dtype, device=input.device)
        is_fp64 = (
            input_view.dtype == torch.float64 or target_dtype == torch.float64
        )

        def grid(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]),)

        _reduction_3d_kernel[grid](
            input_view,
            out_buffer,
            M,
            N,
            I_dim,
            input_view.stride(0),
            input_view.stride(1),
            input_view.stride(2),
            mode_name,
            is_fp64,
        )
        return out_buffer

    with torch_device_fn.device(input.device):
        if is_reduce_at_end:
            M, N = 1, 1
            for axis in range(ndim - len(dims)):
                M *= input.shape[axis]
            for axis in range(ndim - len(dims), ndim):
                N *= input.shape[axis]
            out_buffer = launch_2d(M, N, input.reshape(M, N))
        elif is_consecutive:
            dim_min, dim_max = dims[0], dims[-1]
            O, R, I_dim = 1, 1, 1
            for axis in range(dim_min):
                O *= input.shape[axis]
            for axis in range(dim_min, dim_max + 1):
                R *= input.shape[axis]
            for axis in range(dim_max + 1, ndim):
                I_dim *= input.shape[axis]
            out_buffer = launch_3d(
                O * I_dim,
                R,
                I_dim,
                input.reshape(O, R, I_dim),
            )
        else:
            kept_dims = [axis for axis in range(ndim) if axis not in dims]
            M, N = 1, 1
            for axis in kept_dims:
                M *= input.shape[axis]
            for axis in dims:
                N *= input.shape[axis]
            input_view = input.permute(*kept_dims, *dims).reshape(M, N)
            out_buffer = launch_2d(M, N, input_view)

    return out_buffer.reshape(out_shape)


def reduction(
    input: torch.Tensor,
    mode,
    *,
    dim=None,
    keepdim: bool = True,
    dtype: torch.dtype | None = None,
    compute_data_type=None,
    name: str = "",
) -> torch.Tensor:
    del compute_data_type, name
    mode_name = _mode_name(mode)
    dims = _dims(input, dim)

    if mode_name in ("ADD", "SUM"):
        return sum(input, dim=dim, keepdim=keepdim, dtype=dtype)
    if mode_name in ("AVG", "MEAN"):
        return mean(input, dim=dim, keepdim=keepdim, dtype=dtype)
    if mode_name in ("MUL", "PROD"):
        return _prod_reduce(input, dims, keepdim, dtype=dtype)
    if mode_name == "NORM1":
        return sum(flag_abs(input), dim=dim, keepdim=keepdim, dtype=dtype)
    if mode_name == "NORM2":
        squared = mul(input, input)
        return sqrt(sum(squared, dim=dim, keepdim=keepdim, dtype=dtype))

    if mode_name == "MIN":
        return _native_reduction(input, mode_name, dims, keepdim, dtype)
    if mode_name == "MAX":
        return _native_reduction(input, mode_name, dims, keepdim, dtype)
    if mode_name == "AMAX":
        return _native_reduction(input, mode_name, dims, keepdim, dtype)
    if mode_name == "MUL_NO_ZEROS":
        return _native_reduction(input, mode_name, dims, keepdim, dtype)
    raise NotImplementedError(
        f"flag_dnn reduction does not support mode={mode}"
    )
