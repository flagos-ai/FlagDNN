import logging
from typing import Optional, Tuple

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
    configs=runtime.get_tuned_config("rms_norm"),
    key=["N"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def rms_norm_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    rstd_ptr,
    M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RETURN_STATS: tl.constexpr,
):
    row_idx = tle.program_id(0)

    x_row_ptr = x_ptr + row_idx * N
    y_row_ptr = y_ptr + row_idx * N

    sum_squares = 0.0
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sum_squares += tl.sum(x * x, axis=0)

    rrms = tl.math.rsqrt((sum_squares / N) + eps)
    if RETURN_STATS:
        tl.store(rstd_ptr + row_idx, rrms)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = x * rrms

        if HAS_WEIGHT:
            weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(
                tl.float32
            )
            x_hat = x_hat * weight

        if HAS_BIAS:
            bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(
                tl.float32
            )
            x_hat = x_hat + bias

        y = x_hat.to(x_ptr.dtype.element_ty)
        tl.store(y_row_ptr + cols, y, mask=mask)


def _rms_norm_impl(
    input: torch.Tensor,
    normalized_shape: Tuple[int, ...],
    weight: Optional[torch.Tensor] = None,
    *,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-05,
    return_stats: bool = False,
):
    logger.debug(f"FLAG_DNN RMS_NORM (eps={eps})")

    stat_shape = input.shape[: input.ndim - len(normalized_shape)] + (
        1,
    ) * len(normalized_shape)
    if input.numel() == 0:
        y = torch.empty_like(input)
        if return_stats:
            stats = torch.empty(
                stat_shape, dtype=torch.float32, device=input.device
            )
            return y, stats
        return y

    assert input.ndim >= len(
        normalized_shape
    ), "Input dimensions must be >= normalized_shape length"

    if not input.is_contiguous():
        assert False, "input must be contiguous."
        input = input.contiguous()
    y = torch.empty_like(input)

    N = 1
    tail_shape = input.shape[-len(normalized_shape) :]
    if tuple(normalized_shape) != tuple(tail_shape):
        raise ValueError(
            "The normalized_shape must match"
            " the last few dimensions of"
            " the input tensor."
        )

    for dim in normalized_shape:
        N *= dim

    M = input.numel() // N

    weight_ptr = weight if weight is not None else input
    bias_ptr = bias if bias is not None else input
    rstd = (
        torch.empty((M,), dtype=torch.float32, device=input.device)
        if return_stats
        else y
    )

    grid = (M,)

    with torch_device_fn.device(input.device):
        rms_norm_kernel[grid](
            input,
            y,
            weight_ptr,
            bias_ptr,
            rstd,
            M,
            N,
            eps,
            HAS_WEIGHT=(weight is not None),
            HAS_BIAS=(bias is not None),
            RETURN_STATS=return_stats,
        )

    if return_stats:
        return y, rstd.reshape(stat_shape)
    return y


def rms_norm(
    input: torch.Tensor,
    normalized_shape: Tuple[int, ...],
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-05,
) -> torch.Tensor:
    return _rms_norm_impl(input, normalized_shape, weight=weight, eps=eps)


def rms_norm_forward(
    input: torch.Tensor,
    normalized_shape: Tuple[int, ...],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-05,
):
    return _rms_norm_impl(
        input,
        normalized_shape,
        weight=weight,
        bias=bias,
        eps=eps,
        return_stats=True,
    )
