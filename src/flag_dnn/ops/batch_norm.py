import logging
from typing import Any, Optional

import torch
import triton
import triton.language as tl

from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


def _scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _channel_param(
    param: torch.Tensor, channels: int, name: str
) -> torch.Tensor:
    if param.numel() != channels:
        raise RuntimeError(
            f"{name} must contain {channels} channel values, got {param.numel()}"
        )
    if not param.is_contiguous():
        raise NotImplementedError(
            f"flag_dnn batch_norm currently requires contiguous {name}"
        )
    return param.reshape(channels)


def _empty_stat_like(reference: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        tuple(reference.shape), dtype=torch.float32, device=reference.device
    )


@triton.jit
def batch_norm_inference_kernel(
    x_ptr,
    y_ptr,
    mean_ptr,
    stat_ptr,
    weight_ptr,
    bias_ptr,
    total_elements,
    C,
    S,
    eps,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    STAT_IS_INV_VARIANCE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    c_idx = (offsets // S) % C

    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    mean = tl.load(mean_ptr + c_idx, mask=mask).to(tl.float32)
    stat = tl.load(stat_ptr + c_idx, mask=mask).to(tl.float32)
    weight = (
        tl.load(weight_ptr + c_idx, mask=mask).to(tl.float32)
        if HAS_WEIGHT
        else 1.0
    )
    bias = (
        tl.load(bias_ptr + c_idx, mask=mask).to(tl.float32)
        if HAS_BIAS
        else 0.0
    )

    rstd = stat if STAT_IS_INV_VARIANCE else 1.0 / tl.sqrt(stat + eps)
    y = (x - mean) * rstd * weight + bias
    tl.store(y_ptr + offsets, y.to(y_ptr.dtype.element_ty), mask=mask)


def get_autotune_configs():
    return [
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=2),
    ]


@triton.autotune(
    configs=get_autotune_configs(),
    key=["N", "C", "S"],
    restore_value=["mean_ptr", "var_ptr"],
)
@triton.jit
def batch_norm_fused_kernel_optimized_(
    x_ptr,
    y_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    saved_mean_ptr,
    saved_inv_var_ptr,
    next_running_mean_ptr,
    next_running_var_ptr,
    N,
    C,
    S,
    eps,
    momentum,
    BLOCK_SIZE: tl.constexpr,
    IS_TRAINING: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RUNNING_STATS: tl.constexpr,
    RETURN_STATS: tl.constexpr,
):
    c = tle.program_id(0)
    M = N * S

    stride_gap = S * (C - 1)
    base_x_ptr = x_ptr + c * S
    base_y_ptr = y_ptr + c * S

    if IS_TRAINING:
        sum_x = 0.0
        sum_x2 = 0.0

        for i_offset in range(0, M, BLOCK_SIZE):
            i = i_offset + tl.arange(0, BLOCK_SIZE)
            mask = i < M
            mem_ptrs = base_x_ptr + i + (i // S) * stride_gap
            x = tl.load(mem_ptrs, mask=mask, other=0.0).to(tl.float32)
            sum_x += tl.sum(x, axis=0)
            sum_x2 += tl.sum(x * x, axis=0)

        mean = sum_x / M
        var = (sum_x2 / M) - (mean * mean)
        var = tl.maximum(var, 0.0)
        rstd = 1.0 / tl.sqrt(var + eps)

        if RETURN_STATS:
            tl.store(saved_mean_ptr + c, mean)
            tl.store(saved_inv_var_ptr + c, rstd)

        if HAS_RUNNING_STATS:
            rm = tl.load(mean_ptr + c).to(tl.float32)
            rv = tl.load(var_ptr + c).to(tl.float32)
            unbiased_var = var * (M / (M - 1)) if M > 1 else var
            new_rm = rm * (1.0 - momentum) + mean * momentum
            new_rv = rv * (1.0 - momentum) + unbiased_var * momentum
            if RETURN_STATS:
                tl.store(next_running_mean_ptr + c, new_rm)
                tl.store(next_running_var_ptr + c, new_rv)
            else:
                tl.store(mean_ptr + c, new_rm.to(mean_ptr.dtype.element_ty))
                tl.store(var_ptr + c, new_rv.to(var_ptr.dtype.element_ty))
    else:
        mean = tl.load(mean_ptr + c).to(tl.float32)
        var = tl.load(var_ptr + c).to(tl.float32)
        rstd = 1.0 / tl.sqrt(var + eps)

    weight = tl.load(weight_ptr + c).to(tl.float32) if HAS_WEIGHT else 1.0
    bias = tl.load(bias_ptr + c).to(tl.float32) if HAS_BIAS else 0.0

    for i_offset in range(0, M, BLOCK_SIZE):
        i = i_offset + tl.arange(0, BLOCK_SIZE)
        mask = i < M
        mem_ptrs = base_x_ptr + i + (i // S) * stride_gap
        x = tl.load(mem_ptrs, mask=mask).to(tl.float32)
        y = (x - mean) * rstd * weight + bias
        out_ptrs = base_y_ptr + i + (i // S) * stride_gap
        tl.store(out_ptrs, y.to(y_ptr.dtype.element_ty), mask=mask)


def batchnorm_inference_forward(
    input: torch.Tensor,
    mean: torch.Tensor,
    inv_variance: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    logger.debug("FLAG_DNN BATCHNORM_INFERENCE")

    if input.numel() == 0:
        return torch.empty_like(input)
    if input.ndim < 2:
        raise RuntimeError("batchnorm_inference expects input rank >= 2")
    if not input.is_contiguous():
        raise NotImplementedError(
            "flag_dnn batchnorm_inference currently requires contiguous NCHW input"
        )

    channels = int(input.shape[1])
    mean = _channel_param(mean, channels, "mean")
    inv_variance = _channel_param(inv_variance, channels, "inv_variance")
    scale = _channel_param(scale, channels, "scale")
    bias = _channel_param(bias, channels, "bias")

    y = torch.empty_like(input)
    total_elements = input.numel()
    spatial = total_elements // (int(input.shape[0]) * channels)

    def grid(meta):
        return (triton.cdiv(total_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        batch_norm_inference_kernel[grid](
            input,
            y,
            mean,
            inv_variance,
            scale,
            bias,
            total_elements,
            channels,
            spatial,
            0.0,
            BLOCK_SIZE=1024,
            HAS_WEIGHT=True,
            HAS_BIAS=True,
            STAT_IS_INV_VARIANCE=True,
        )
    return y


def batchnorm_forward(
    input: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    in_running_mean: torch.Tensor,
    in_running_var: torch.Tensor,
    epsilon,
    momentum,
    peer_stats=None,
):
    if peer_stats:
        raise NotImplementedError(
            "flag_dnn batchnorm does not support peer_stats"
        )
    if input.dim() < 2:
        raise RuntimeError("batchnorm expects input rank >= 2")
    if not input.is_contiguous():
        raise NotImplementedError(
            "flag_dnn batchnorm currently requires contiguous NCHW input"
        )

    channels = int(input.shape[1])
    scale = _channel_param(scale, channels, "scale")
    bias = _channel_param(bias, channels, "bias")
    running_mean = _channel_param(
        in_running_mean, channels, "in_running_mean"
    )
    running_var = _channel_param(in_running_var, channels, "in_running_var")

    y = torch.empty_like(input)
    saved_mean = _empty_stat_like(in_running_mean)
    saved_inv_var = _empty_stat_like(in_running_var)
    next_running_mean = _empty_stat_like(in_running_mean)
    next_running_var = _empty_stat_like(in_running_var)
    if input.numel() == 0:
        return y, saved_mean, saved_inv_var, next_running_mean, next_running_var

    batch = int(input.shape[0])
    spatial = input.numel() // (batch * channels)
    grid = (channels,)

    with torch_device_fn.device(input.device):
        batch_norm_fused_kernel_optimized_[grid](
            input,
            y,
            running_mean,
            running_var,
            scale,
            bias,
            saved_mean,
            saved_inv_var,
            next_running_mean,
            next_running_var,
            batch,
            channels,
            spatial,
            _scalar(epsilon),
            _scalar(momentum),
            IS_TRAINING=True,
            HAS_WEIGHT=True,
            HAS_BIAS=True,
            HAS_RUNNING_STATS=True,
            RETURN_STATS=True,
        )

    return y, saved_mean, saved_inv_var, next_running_mean, next_running_var


def batch_norm_aten(
    input: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    running_mean: Optional[torch.Tensor],
    running_var: Optional[torch.Tensor],
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
    cudnn_enabled: bool = True,
) -> torch.Tensor:
    logger.debug(f"FLAG_DNN FUSED BATCH_NORM (training={training}, eps={eps})")
    del cudnn_enabled

    if input.numel() == 0:
        return torch.empty_like(input)

    assert input.ndim >= 2, "BatchNorm requires at least 2D input (N, C, ...)"

    if not training:
        assert (
            running_mean is not None and running_var is not None
        ), "running stats must be provided in eval mode"

    if not input.is_contiguous():
        input = input.contiguous()

    y = torch.empty_like(input)

    N = input.shape[0]
    C = input.shape[1]
    S = input.numel() // (N * C)
    total_elements = input.numel()

    dummy_ptr = torch.empty(0, device=input.device)
    mean_ptr = running_mean if running_mean is not None else dummy_ptr
    var_ptr = running_var if running_var is not None else dummy_ptr
    weight_ptr = weight if weight is not None else dummy_ptr
    bias_ptr = bias if bias is not None else dummy_ptr
    has_running_stats = running_mean is not None and running_var is not None

    with torch_device_fn.device(input.device):
        if not training:
            def grid(meta):
                return (
                    triton.cdiv(total_elements, meta["BLOCK_SIZE"]),
                )

            batch_norm_inference_kernel[grid](
                input,
                y,
                mean_ptr,
                var_ptr,
                weight_ptr,
                bias_ptr,
                total_elements,
                C,
                S,
                eps,
                BLOCK_SIZE=1024,
                HAS_WEIGHT=(weight is not None),
                HAS_BIAS=(bias is not None),
                STAT_IS_INV_VARIANCE=False,
            )
        else:
            grid = (C,)
            batch_norm_fused_kernel_optimized_[grid](
                input,
                y,
                mean_ptr,
                var_ptr,
                weight_ptr,
                bias_ptr,
                dummy_ptr,
                dummy_ptr,
                dummy_ptr,
                dummy_ptr,
                N,
                C,
                S,
                eps,
                momentum,
                IS_TRAINING=training,
                HAS_WEIGHT=(weight is not None),
                HAS_BIAS=(bias is not None),
                HAS_RUNNING_STATS=has_running_stats,
                RETURN_STATS=False,
            )

    return y


def batch_norm(
    input: torch.Tensor,
    running_mean: Optional[torch.Tensor],
    running_var: Optional[torch.Tensor],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> torch.Tensor:
    return batch_norm_aten(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
        torch.backends.cudnn.enabled,
    )
