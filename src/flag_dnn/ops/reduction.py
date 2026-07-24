# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner

_REDUCE_CONFIGS = runtime.get_tuned_config("reduction")
_PORTABLE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MODE_ALIASES = {"SUM": "ADD", "MEAN": "AVG", "PROD": "MUL"}


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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    if OP == "MIN":
        acc = tl.full((BLOCK_M,), float("inf"), dtype=tl.float32)
    elif OP == "MUL" or OP == "MUL_NO_ZEROS":
        acc = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
    elif OP == "AMAX" or (
        OP == "ADD" or OP == "AVG" or OP == "NORM1" or OP == "NORM2"
    ):
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    else:
        acc = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N
        ptrs = (
            x_ptr
            + m_offsets[:, None] * stride_xm
            + n_offsets[None, :] * stride_xn
        )
        mask = m_mask[:, None] & n_mask[None, :]
        if OP == "ADD" or OP == "AVG" or OP == "NORM1" or OP == "NORM2":
            vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
            if OP == "NORM1":
                vals = tl.abs(vals)
            elif OP == "NORM2":
                vals *= vals
            acc += tl.sum(vals, axis=1)
        elif OP == "MUL" or OP == "MUL_NO_ZEROS":
            vals = tl.load(ptrs, mask=mask, other=1.0).to(tl.float32)
            if OP == "MUL_NO_ZEROS":
                vals = tl.where(vals == 0.0, 1.0, vals)
            products = tl.cumprod(vals, axis=1)
            last = tl.arange(0, BLOCK_N) == (BLOCK_N - 1)
            chunk = tl.sum(tl.where(last[None, :], products, 0.0), axis=1)
            acc *= chunk
        elif OP == "MIN":
            vals = tl.load(ptrs, mask=mask, other=float("inf"))
            vals = vals.to(tl.float32)
            acc = tl.minimum(acc, tl.min(vals, axis=1))
        else:
            other = 0.0 if OP == "AMAX" else -float("inf")
            vals = tl.load(ptrs, mask=mask, other=other)
            vals = vals.to(tl.float32)
            if OP == "AMAX":
                vals = tl.abs(vals)
            acc = tl.maximum(acc, tl.max(vals, axis=1))

    if OP == "AVG":
        acc /= N
    elif OP == "NORM2":
        acc = tl.sqrt(acc)
    tl.store(out_ptr + m_offsets, acc, mask=m_mask)


@libentry()
@libtuner(
    configs=_REDUCE_CONFIGS,
    key=["M", "N"],
    strategy=["align32", "align32"],
    warmup=5,
    rep=10,
)
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
        acc = tl.full((BLOCK_M,), float("inf"), dtype=tl.float32)
    elif OP == "MUL" or OP == "MUL_NO_ZEROS":
        acc = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
    elif OP == "AMAX" or (
        OP == "ADD" or OP == "AVG" or OP == "NORM1" or OP == "NORM2"
    ):
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    else:
        acc = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N
        ptrs = base_ptrs[:, None] + n_offsets[None, :] * stride_xr
        mask = m_mask[:, None] & n_mask[None, :]
        if OP == "ADD" or OP == "AVG" or OP == "NORM1" or OP == "NORM2":
            vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
            if OP == "NORM1":
                vals = tl.abs(vals)
            elif OP == "NORM2":
                vals *= vals
            acc += tl.sum(vals, axis=1)
        elif OP == "MUL" or OP == "MUL_NO_ZEROS":
            vals = tl.load(ptrs, mask=mask, other=1.0).to(tl.float32)
            if OP == "MUL_NO_ZEROS":
                vals = tl.where(vals == 0.0, 1.0, vals)
            products = tl.cumprod(vals, axis=1)
            last = tl.arange(0, BLOCK_N) == (BLOCK_N - 1)
            chunk = tl.sum(tl.where(last[None, :], products, 0.0), axis=1)
            acc *= chunk
        elif OP == "MIN":
            vals = tl.load(ptrs, mask=mask, other=float("inf"))
            vals = vals.to(tl.float32)
            acc = tl.minimum(acc, tl.min(vals, axis=1))
        else:
            other = 0.0 if OP == "AMAX" else -float("inf")
            vals = tl.load(ptrs, mask=mask, other=other)
            vals = vals.to(tl.float32)
            if OP == "AMAX":
                vals = tl.abs(vals)
            acc = tl.maximum(acc, tl.max(vals, axis=1))

    if OP == "AVG":
        acc /= N
    elif OP == "NORM2":
        acc = tl.sqrt(acc)
    tl.store(out_ptr + m_offsets, acc, mask=m_mask)


def _mode_name(mode) -> str:
    name = getattr(mode, "name", None)
    if name is None:
        name = str(mode).rsplit(".", 1)[-1]
    return str(name).upper()


def _validate_portable_input(
    input: torch.Tensor, dtype: torch.dtype | None
) -> None:
    if input.dtype not in _PORTABLE_DTYPES:
        raise NotImplementedError(
            f"flag_dnn reduction does not support input dtype={input.dtype} "
            f"on device={runtime.device.name}"
        )
    if dtype is not None and dtype not in _PORTABLE_DTYPES:
        raise NotImplementedError(
            f"flag_dnn reduction does not support output dtype={dtype} "
            f"on device={runtime.device.name}"
        )
    if input.device.type != runtime.device.name:
        raise RuntimeError(
            f"flag_dnn reduction expected a {runtime.device.name} tensor, "
            f"got device={input.device}"
        )


def _dims(input: torch.Tensor, dim) -> list[int]:
    rank = input.dim()
    if dim is None:
        raw_dims = list(range(rank))
    elif isinstance(dim, int):
        raw_dims = [dim]
    else:
        raw_dims = list(dim)

    normalized = []
    for axis in raw_dims:
        if not isinstance(axis, int):
            raise TypeError(
                f"reduction dimensions must be integers, got {axis!r}"
            )
        if axis < -rank or axis >= rank:
            raise IndexError(
                "Dimension out of range (expected to be in range of "
                f"[{-rank}, {rank - 1}], but got {axis})"
            )
        normalized.append(axis if axis >= 0 else axis + rank)
    return sorted(set(normalized))


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


def _shape_numel(shape) -> int:
    result = 1
    for size in shape:
        result *= size
    return result


def _empty_reduction_result(
    input: torch.Tensor,
    mode_name: str,
    dims: list[int],
    keepdim: bool,
    target_dtype: torch.dtype,
) -> torch.Tensor | None:
    out_shape = _out_shape(input, dims, keepdim)
    if _shape_numel(out_shape) == 0:
        return torch.empty(out_shape, dtype=target_dtype, device=input.device)

    reduced_extent = 1
    for axis in dims:
        reduced_extent *= input.shape[axis]
    if reduced_extent != 0:
        return None

    if mode_name in ("MIN", "MAX", "AMAX"):
        raise RuntimeError(
            f"reduction mode {mode_name} cannot reduce an empty dimension"
        )
    if mode_name == "AVG":
        value = float("nan")
    elif mode_name in ("MUL", "MUL_NO_ZEROS"):
        value = 1.0
    else:
        value = 0.0
    return torch.full(
        out_shape,
        value,
        dtype=target_dtype,
        device=input.device,
    )


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
    empty_result = _empty_reduction_result(
        input, mode_name, dims, keepdim, target_dtype
    )
    if empty_result is not None:
        return empty_result

    ndim = input.ndim
    out_shape = _out_shape(input, dims, keepdim)
    is_reduce_at_end = dims == list(range(ndim - len(dims), ndim))
    is_consecutive = all(
        dims[index] + 1 == dims[index + 1] for index in range(len(dims) - 1)
    )

    def launch_2d(M: int, N: int, input_view: torch.Tensor) -> torch.Tensor:
        out_buffer = torch.empty((M,), dtype=target_dtype, device=input.device)

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
        )
        return out_buffer

    def launch_3d(
        M: int, N: int, I_dim: int, input_view: torch.Tensor
    ) -> torch.Tensor:
        out_buffer = torch.empty((M,), dtype=target_dtype, device=input.device)

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
    _validate_portable_input(input, dtype)
    mode_name = _mode_name(mode)
    mode_name = _MODE_ALIASES.get(mode_name, mode_name)
    dims = _dims(input, dim)

    supported_modes = (
        "ADD",
        "AVG",
        "MUL",
        "NORM1",
        "NORM2",
        "MIN",
        "MAX",
        "AMAX",
        "MUL_NO_ZEROS",
    )
    if mode_name in supported_modes:
        return _native_reduction(input, mode_name, dims, keepdim, dtype)
    raise NotImplementedError(
        f"flag_dnn reduction does not support mode={mode}"
    )
