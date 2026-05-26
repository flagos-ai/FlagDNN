import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("binary"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def fmin_tensor_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    x_nan = x != x
    y_nan = y != y
    res = tl.where(x_nan, y, tl.where(y_nan, x, tl.minimum(x, y)))
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("binary"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def fmin_int_tensor_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    res = tl.minimum(x, y)
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def fmin_broadcast2d_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    N,
    sx0,
    sx1,
    sy0,
    sy1,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    idx1 = offsets % N
    idx0 = offsets // N
    x = tl.load(x_ptr + idx0 * sx0 + idx1 * sx1, mask=mask)
    y = tl.load(y_ptr + idx0 * sy0 + idx1 * sy1, mask=mask)
    x_nan = x != x
    y_nan = y != y
    res = tl.where(x_nan, y, tl.where(y_nan, x, tl.minimum(x, y)))
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def fmin_int_broadcast2d_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    N,
    sx0,
    sx1,
    sy0,
    sy1,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    idx1 = offsets % N
    idx0 = offsets // N
    x = tl.load(x_ptr + idx0 * sx0 + idx1 * sx1, mask=mask)
    y = tl.load(y_ptr + idx0 * sy0 + idx1 * sy1, mask=mask)
    res = tl.minimum(x, y)
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def fmin(
    input: torch.Tensor,
    other: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logger.debug("FLAG_DNN FMIN")

    if input.dtype.is_complex:
        raise RuntimeError(f"fmin not implemented for '{input.dtype}'")

    if not input.is_contiguous():
        input = input.contiguous()
    if not other.is_contiguous():
        other = other.contiguous()

    out_shape = torch.broadcast_shapes(input.shape, other.shape)

    if out is None:
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)

    n_elements = out.numel()
    if n_elements == 0:
        return out

    if input.shape == other.shape and input.shape == out_shape:
        def grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        with torch_device_fn.device(input.device):
            if input.dtype.is_floating_point:
                fmin_tensor_kernel[grid](input, other, out, n_elements)
            else:
                fmin_int_tensor_kernel[grid](input, other, out, n_elements)
        return out

    from flag_dnn.ops.binary import collapse_dims, binary

    in_exp = input.expand(out_shape)
    oth_exp = other.expand(out_shape)

    c_shape, c_sx, c_sy = collapse_dims(out_shape, in_exp.stride(), oth_exp.stride())
    n_collapsed_dims = len(c_shape)

    if n_collapsed_dims <= 2:
        if n_collapsed_dims == 1:
            N = c_shape[0]
            sx0_v, sx1_v = 0, c_sx[0]
            sy0_v, sy1_v = 0, c_sy[0]
        else:
            N = c_shape[1]
            sx0_v, sx1_v = c_sx[0], c_sx[1]
            sy0_v, sy1_v = c_sy[0], c_sy[1]

        grid_fn = (triton.cdiv(n_elements, 1024),)
        with torch_device_fn.device(input.device):
            if input.dtype.is_floating_point:
                fmin_broadcast2d_kernel[grid_fn](
                    input, other, out, n_elements, N,
                    sx0_v, sx1_v, sy0_v, sy1_v,
                    BLOCK_SIZE=1024, num_warps=8,
                )
            else:
                fmin_int_broadcast2d_kernel[grid_fn](
                    input, other, out, n_elements, N,
                    sx0_v, sx1_v, sy0_v, sy1_v,
                    BLOCK_SIZE=1024, num_warps=8,
                )
        return out
    else:
        return binary(input, other, op_type="fmin", out=out)
