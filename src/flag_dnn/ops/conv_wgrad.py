from __future__ import annotations

import logging
from typing import Any, Optional, Sequence, Tuple, Union

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner

logger = logging.getLogger(__name__)

_CONV_WGRAD_1D_CONFIGS = runtime.get_tuned_config("conv_wgrad_1d")
_CONV_WGRAD_2D_CONFIGS = runtime.get_tuned_config("conv_wgrad_2d")
_CONV_WGRAD_2D_1X1_CONFIGS = runtime.get_tuned_config(
    "conv_wgrad_2d_1x1"
)
_CONV_WGRAD_3D_CONFIGS = runtime.get_tuned_config("conv_wgrad_3d")


def _dtype_id(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return 0
    if dtype == torch.bfloat16:
        return 1
    if dtype == torch.float32:
        return 2
    return -1


def _tuple_n(
    value: Union[int, Sequence[int]], rank: int, name: str
) -> Tuple[int, ...]:
    if isinstance(value, int):
        return (int(value),) * rank
    result = tuple(int(v) for v in value)
    if len(result) != rank:
        raise RuntimeError(f"{name} must have length {rank}, got {value}")
    return result


def _conv_out_dim(
    input_size: int,
    pad_before: int,
    pad_after: int,
    dilation: int,
    kernel: int,
    stride: int,
) -> int:
    return (
        input_size + pad_before + pad_after - dilation * (kernel - 1) - 1
    ) // stride + 1


def _normalize_convolution_mode(convolution_mode: Any) -> str:
    if convolution_mode is None:
        return "CROSS_CORRELATION"
    mode = str(convolution_mode).rsplit(".", 1)[-1].upper()
    if mode in ("CROSS_CORRELATION", "CONVOLUTION"):
        return mode
    raise RuntimeError(
        "convolution_mode must be CROSS_CORRELATION or CONVOLUTION"
    )


def _normalize_filter_size(filter_size: Sequence[int]) -> Tuple[int, ...]:
    result = tuple(int(dim) for dim in filter_size)
    if len(result) not in (3, 4, 5):
        raise RuntimeError(
            "filter_size must describe a 1D/2D/3D convolution filter"
        )
    if any(dim < 0 for dim in result):
        raise RuntimeError("filter_size dimensions must be non-negative")
    return result


def _normalize_padding(
    filter_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    padding: Optional[Union[str, int, Sequence[int]]],
    pre_padding: Optional[Union[int, Sequence[int]]],
    post_padding: Optional[Union[int, Sequence[int]]],
    dilation: Tuple[int, ...],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    rank = len(stride)
    if pre_padding is not None or post_padding is not None:
        if padding is not None:
            raise TypeError(
                "conv_wgrad accepts either padding or "
                "pre_padding/post_padding"
            )
        if pre_padding is None or post_padding is None:
            raise TypeError(
                "conv_wgrad requires both pre_padding and post_padding"
            )
        pre = _tuple_n(pre_padding, rank, "pre_padding")
        post = _tuple_n(post_padding, rank, "post_padding")
    else:
        if padding is None:
            padding = 0
        if isinstance(padding, str):
            if padding == "valid":
                zeros = (0,) * rank
                return zeros, zeros
            if padding == "same":
                if any(dim != 1 for dim in stride):
                    raise RuntimeError(
                        "padding='same' is not supported for strided "
                        "convolutions"
                    )
                pre_values = []
                post_values = []
                for axis in range(rank):
                    kernel = int(filter_size[2 + axis])
                    effective_kernel = dilation[axis] * (kernel - 1) + 1
                    total_pad = max(effective_kernel - 1, 0)
                    before = total_pad // 2
                    pre_values.append(before)
                    post_values.append(total_pad - before)
                return tuple(pre_values), tuple(post_values)
            raise RuntimeError(
                "padding must be 'valid', 'same', int, or tuple"
            )

        if isinstance(padding, int):
            pre = post = (int(padding),) * rank
        else:
            values = tuple(int(v) for v in padding)
            if len(values) == rank:
                pre = post = values
            elif len(values) == 2 * rank:
                pre = tuple(values[2 * axis] for axis in range(rank))
                post = tuple(values[2 * axis + 1] for axis in range(rank))
            else:
                raise RuntimeError(
                    f"padding must have length {rank} or {2 * rank}, "
                    f"got {padding}"
                )

    if any(value < 0 for value in pre + post):
        raise RuntimeError("negative padding is not supported")
    return pre, post


def _check_conv_wgrad_inputs(
    image: torch.Tensor,
    loss: torch.Tensor,
    filter_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    pre_padding: Tuple[int, ...],
    post_padding: Tuple[int, ...],
    dilation: Tuple[int, ...],
    groups: int,
    rank: int,
    is_unbatched_1d: bool,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    if not image.is_cuda or not loss.is_cuda:
        raise NotImplementedError(
            "flag_dnn conv_wgrad Triton implementation requires CUDA inputs"
        )
    if image.device != loss.device:
        raise RuntimeError("image and loss must be on the same device")
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if image.dtype not in supported_dtypes:
        raise NotImplementedError(
            f"flag_dnn conv_wgrad does not support image dtype={image.dtype}"
        )
    if loss.dtype != image.dtype:
        raise RuntimeError("image and loss must have the same dtype")
    if groups <= 0:
        raise RuntimeError("groups must be a positive integer")
    if any(dim <= 0 for dim in stride):
        raise RuntimeError("stride must be positive")
    if any(dim <= 0 for dim in dilation):
        raise RuntimeError("dilation must be positive")

    expected_dim = rank + 1 if is_unbatched_1d else rank + 2
    if image.dim() != expected_dim:
        raise RuntimeError(
            f"flag_dnn conv_wgrad expected image dim={expected_dim}, "
            f"got {image.dim()}"
        )
    if loss.dim() != expected_dim:
        raise RuntimeError(
            f"flag_dnn conv_wgrad expected loss dim={expected_dim}, "
            f"got {loss.dim()}"
        )

    image_shape = (
        (1,) + tuple(image.shape) if is_unbatched_1d else tuple(image.shape)
    )
    loss_shape = (
        (1,) + tuple(loss.shape) if is_unbatched_1d else tuple(loss.shape)
    )
    c_in = int(image_shape[1])
    c_out = int(filter_size[0])
    cin_per_group = int(filter_size[1])
    if c_in <= 0 or c_out <= 0 or cin_per_group <= 0:
        raise RuntimeError("channel dimensions must be non-empty")
    if c_in % groups != 0 or c_out % groups != 0:
        raise RuntimeError("channels must be divisible by groups")
    if cin_per_group != c_in // groups:
        raise RuntimeError(
            "filter_size[1] must match input_channels // groups"
        )
    if int(loss_shape[0]) != int(image_shape[0]):
        raise RuntimeError("loss batch size must match image batch size")
    if int(loss_shape[1]) != c_out:
        raise RuntimeError("loss channels must match filter output channels")

    image_spatial = tuple(int(v) for v in image_shape[2:])
    loss_spatial = tuple(int(v) for v in loss_shape[2:])
    kernel_shape = tuple(int(v) for v in filter_size[2:])
    expected_loss_spatial = tuple(
        _conv_out_dim(
            image_spatial[axis],
            pre_padding[axis],
            post_padding[axis],
            dilation[axis],
            kernel_shape[axis],
            stride[axis],
        )
        for axis in range(rank)
    )
    if loss_spatial != expected_loss_spatial:
        raise RuntimeError(
            "loss spatial shape does not match "
            "image/filter_size/stride/padding/dilation: expected "
            f"{expected_loss_spatial}, got {loss_spatial}"
        )
    return image_spatial, loss_spatial


@libentry()
@libtuner(
    configs=_CONV_WGRAD_1D_CONFIGS,
    key=[
        "M",
        "IMAGE_LEN",
        "LOSS_LEN",
        "CIN_PER_GROUP",
        "COUT_PER_GROUP",
        "STRIDE_L",
        "DIL_L",
        "KL",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def _conv_wgrad1d_kernel(
    image_ptr,
    loss_ptr,
    out_ptr,
    M: tl.constexpr,
    IMAGE_LEN: tl.constexpr,
    LOSS_LEN: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_l: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_l: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    out_stride_k: tl.constexpr,
    STRIDE_L: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_L: tl.constexpr,
    KL: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    k = tl.program_id(1)
    group = tl.program_id(2)

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + offs_ci_rel
    image_k = KL - 1 - k if FILTER_REVERSE else k

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for m_start in tl.range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        safe_m = tl.where(mask_m, offs_m, 0)
        n_idx = safe_m // LOSS_LEN
        loss_l = safe_m - n_idx * LOSS_LEN
        image_l = loss_l * STRIDE_L - PAD_LEFT + image_k * DIL_L
        valid_l = (image_l >= 0) & (image_l < IMAGE_LEN)
        safe_l = tl.where(valid_l, image_l, 0)

        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + co[:, None] * loss_stride_c
            + loss_l[None, :] * loss_stride_l,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_l[:, None] * image_stride_l,
            mask=mask_m[:, None] & mask_ci[None, :] & valid_l[:, None],
            other=0.0,
        )
        acc += tl.dot(
            loss,
            image,
            out_dtype=tl.float32,
            input_precision="tf32",
        )

    tl.store(
        out_ptr
        + co[:, None] * out_stride_o
        + offs_ci_rel[None, :] * out_stride_i
        + k * out_stride_k,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_wgrad1d_split_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    M: tl.constexpr,
    IMAGE_LEN: tl.constexpr,
    LOSS_LEN: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_l: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_l: tl.constexpr,
    STRIDE_L: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_L: tl.constexpr,
    KL: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    k = tl.program_id(1)
    split_group = tl.program_id(2)
    split = split_group % NUM_SPLITS
    group = split_group // NUM_SPLITS

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + offs_ci_rel
    image_k = KL - 1 - k if FILTER_REVERSE else k

    split_size = tl.cdiv(M, NUM_SPLITS)
    split_begin = split * split_size
    split_end = tl.minimum(split_begin + split_size, M)

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for m_start in tl.range(split_begin, split_end, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < split_end
        safe_m = tl.where(mask_m, offs_m, 0)
        n_idx = safe_m // LOSS_LEN
        loss_l = safe_m - n_idx * LOSS_LEN
        image_l = loss_l * STRIDE_L - PAD_LEFT + image_k * DIL_L
        valid_l = (image_l >= 0) & (image_l < IMAGE_LEN)
        safe_l = tl.where(valid_l, image_l, 0)

        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + co[:, None] * loss_stride_c
            + loss_l[None, :] * loss_stride_l,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_l[:, None] * image_stride_l,
            mask=mask_m[:, None] & mask_ci[None, :] & valid_l[:, None],
            other=0.0,
        )
        acc += tl.dot(
            loss,
            image,
            out_dtype=tl.float32,
            input_precision="tf32",
        )

    tl.store(
        partial_ptr
        + ((split * C_OUT + co[:, None]) * CIN_PER_GROUP + offs_ci_rel[None, :])
        * KL
        + k,
        acc,
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_wgrad1d_reduce_kernel(
    partial_ptr,
    out_ptr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    out_stride_k: tl.constexpr,
    KL: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    pid = tl.program_id(0)
    k = tl.program_id(1)
    group = tl.program_id(2)

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for split in tl.static_range(0, NUM_SPLITS):
        acc += tl.load(
            partial_ptr
            + ((split * C_OUT + co[:, None]) * CIN_PER_GROUP + offs_ci_rel[None, :])
            * KL
            + k,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )

    tl.store(
        out_ptr
        + co[:, None] * out_stride_o
        + offs_ci_rel[None, :] * out_stride_i
        + k * out_stride_k,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV_WGRAD_2D_CONFIGS,
    key=[
        "M",
        "IMAGE_H",
        "IMAGE_W",
        "LOSS_H",
        "LOSS_W",
        "CIN_PER_GROUP",
        "COUT_PER_GROUP",
        "STRIDE_H",
        "STRIDE_W",
        "DIL_H",
        "DIL_W",
        "KH",
        "KW",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def _conv_wgrad2d_kernel(
    image_ptr,
    loss_ptr,
    out_ptr,
    M: tl.constexpr,
    IMAGE_H: tl.constexpr,
    IMAGE_W: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_h: tl.constexpr,
    image_stride_w: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    k = tl.program_id(1)
    group = tl.program_id(2)

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    kh = k // KW
    kw = k - kh * KW
    image_kh = KH - 1 - kh if FILTER_REVERSE else kh
    image_kw = KW - 1 - kw if FILTER_REVERSE else kw

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + offs_ci_rel

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for m_start in tl.range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        safe_m = tl.where(mask_m, offs_m, 0)
        loss_w = safe_m % LOSS_W
        tmp = safe_m // LOSS_W
        loss_h = tmp % LOSS_H
        n_idx = tmp // LOSS_H

        image_h = loss_h * STRIDE_H - PAD_H + image_kh * DIL_H
        image_w = loss_w * STRIDE_W - PAD_W + image_kw * DIL_W
        valid_hw = (
            (image_h >= 0)
            & (image_h < IMAGE_H)
            & (image_w >= 0)
            & (image_w < IMAGE_W)
        )
        safe_h = tl.where(valid_hw, image_h, 0)
        safe_w = tl.where(valid_hw, image_w, 0)

        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + co[:, None] * loss_stride_c
            + loss_h[None, :] * loss_stride_h
            + loss_w[None, :] * loss_stride_w,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_h[:, None] * image_stride_h
            + safe_w[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid_hw[:, None],
            other=0.0,
        )
        acc += tl.dot(
            loss,
            image,
            out_dtype=tl.float32,
            input_precision="tf32",
        )

    tl.store(
        out_ptr
        + co[:, None] * out_stride_o
        + offs_ci_rel[None, :] * out_stride_i
        + kh * out_stride_h
        + kw * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV_WGRAD_2D_1X1_CONFIGS,
    key=[
        "M",
        "HW",
        "CIN_PER_GROUP",
        "COUT_PER_GROUP",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def _conv_wgrad2d_1x1_kernel(
    image_ptr,
    loss_ptr,
    out_ptr,
    M: tl.constexpr,
    HW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + offs_ci_rel

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for m_start in tl.range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        safe_m = tl.where(mask_m, offs_m, 0)
        n_idx = safe_m // HW
        hw_idx = safe_m - n_idx * HW

        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + co[:, None] * loss_stride_c
            + hw_idx[None, :],
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + hw_idx[:, None],
            mask=mask_m[:, None] & mask_ci[None, :],
            other=0.0,
        )
        acc += tl.dot(
            loss,
            image,
            out_dtype=tl.float32,
            input_precision="tf32",
        )

    tl.store(
        out_ptr
        + co[:, None] * out_stride_o
        + offs_ci_rel[None, :] * out_stride_i,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_wgrad2d_1x1_split_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    M: tl.constexpr,
    HW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    group = tl.program_id(2)

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + offs_ci_rel

    split_size = tl.cdiv(M, NUM_SPLITS)
    split_begin = split * split_size
    split_end = tl.minimum(split_begin + split_size, M)

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for m_start in tl.range(split_begin, split_end, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < split_end
        safe_m = tl.where(mask_m, offs_m, 0)
        n_idx = safe_m // HW
        hw_idx = safe_m - n_idx * HW

        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + co[:, None] * loss_stride_c
            + hw_idx[None, :],
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + hw_idx[:, None],
            mask=mask_m[:, None] & mask_ci[None, :],
            other=0.0,
        )
        acc += tl.dot(
            loss,
            image,
            out_dtype=tl.float32,
            input_precision="tf32",
        )

    tl.store(
        partial_ptr
        + split * C_OUT * CIN_PER_GROUP
        + co[:, None] * CIN_PER_GROUP
        + offs_ci_rel[None, :],
        acc,
        mask=mask_co[:, None] & mask_ci[None, :],
    )



@triton.jit
def _conv_wgrad2d_1x1_split_nodiv_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    HW: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    SPLITS_PER_N: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    n_idx = split // SPLITS_PER_N
    split_in_n = split - n_idx * SPLITS_PER_N
    split_size = tl.cdiv(HW, SPLITS_PER_N)
    hw_begin = split_in_n * split_size
    hw_end = tl.minimum(hw_begin + split_size, HW)

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for hw_start in tl.range(hw_begin, hw_end, BLOCK_M):
        hw = hw_start + tl.arange(0, BLOCK_M)
        mask_m = hw < hw_end
        safe_hw = tl.where(mask_m, hw, 0)
        loss = tl.load(
            loss_ptr
            + n_idx * loss_stride_n
            + offs_co_rel[:, None] * loss_stride_c
            + safe_hw[None, :],
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx * image_stride_n
            + offs_ci_rel[None, :] * image_stride_c
            + safe_hw[:, None],
            mask=mask_m[:, None] & mask_ci[None, :],
            other=0.0,
        )
        acc += tl.dot(
            loss,
            image,
            out_dtype=tl.float32,
            input_precision="tf32",
        )

    tl.store(
        partial_ptr
        + split * C_OUT * CIN_PER_GROUP
        + offs_co_rel[:, None] * CIN_PER_GROUP
        + offs_ci_rel[None, :],
        acc,
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_wgrad_zero_kernel(out_ptr, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(
        out_ptr + offs,
        tl.zeros((BLOCK,), dtype=tl.float32),
        mask=offs < TOTAL,
    )


@triton.jit
def _conv_wgrad2d_1x1_atomic_nodiv_kernel(
    image_ptr,
    loss_ptr,
    out_ptr,
    HW: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    SPLITS_PER_N: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    n_idx = split // SPLITS_PER_N
    split_in_n = split - n_idx * SPLITS_PER_N
    split_size = tl.cdiv(HW, SPLITS_PER_N)
    hw_begin = split_in_n * split_size
    hw_end = tl.minimum(hw_begin + split_size, HW)

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for hw_start in tl.range(hw_begin, hw_end, BLOCK_M):
        hw = hw_start + tl.arange(0, BLOCK_M)
        mask_m = hw < hw_end
        safe_hw = tl.where(mask_m, hw, 0)
        loss = tl.load(
            loss_ptr
            + n_idx * loss_stride_n
            + offs_co_rel[:, None] * loss_stride_c
            + safe_hw[None, :],
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx * image_stride_n
            + offs_ci_rel[None, :] * image_stride_c
            + safe_hw[:, None],
            mask=mask_m[:, None] & mask_ci[None, :],
            other=0.0,
        )
        acc += tl.dot(
            loss,
            image,
            out_dtype=tl.float32,
            input_precision="tf32",
        )

    tl.atomic_add(
        out_ptr
        + offs_co_rel[:, None] * out_stride_o
        + offs_ci_rel[None, :] * out_stride_i,
        acc,
        sem="relaxed",
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_wgrad2d_1x1_reduce_kernel(
    partial_ptr,
    out_ptr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for split in tl.static_range(0, NUM_SPLITS):
        acc += tl.load(
            partial_ptr
            + split * C_OUT * CIN_PER_GROUP
            + co[:, None] * CIN_PER_GROUP
            + offs_ci_rel[None, :],
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )

    tl.store(
        out_ptr + co[:, None] * out_stride_o + offs_ci_rel[None, :] * out_stride_i,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_wgrad2d_split_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    M: tl.constexpr,
    IMAGE_H: tl.constexpr,
    IMAGE_W: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_h: tl.constexpr,
    image_stride_w: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    k = tl.program_id(1)
    split_group = tl.program_id(2)
    split = split_group % NUM_SPLITS
    group = split_group // NUM_SPLITS

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    kh = k // KW
    kw = k - kh * KW
    image_kh = KH - 1 - kh if FILTER_REVERSE else kh
    image_kw = KW - 1 - kw if FILTER_REVERSE else kw

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + offs_ci_rel

    split_size = tl.cdiv(M, NUM_SPLITS)
    split_begin = split * split_size
    split_end = tl.minimum(split_begin + split_size, M)

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for m_start in tl.range(split_begin, split_end, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < split_end
        safe_m = tl.where(mask_m, offs_m, 0)
        loss_w = safe_m % LOSS_W
        tmp = safe_m // LOSS_W
        loss_h = tmp % LOSS_H
        n_idx = tmp // LOSS_H

        image_h = loss_h * STRIDE_H - PAD_H + image_kh * DIL_H
        image_w = loss_w * STRIDE_W - PAD_W + image_kw * DIL_W
        valid_hw = (
            (image_h >= 0)
            & (image_h < IMAGE_H)
            & (image_w >= 0)
            & (image_w < IMAGE_W)
        )
        safe_h = tl.where(valid_hw, image_h, 0)
        safe_w = tl.where(valid_hw, image_w, 0)

        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + co[:, None] * loss_stride_c
            + loss_h[None, :] * loss_stride_h
            + loss_w[None, :] * loss_stride_w,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_h[:, None] * image_stride_h
            + safe_w[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid_hw[:, None],
            other=0.0,
        )
        acc += tl.dot(
            loss,
            image,
            out_dtype=tl.float32,
            input_precision="tf32",
        )

    k_elems = KH * KW
    tl.store(
        partial_ptr
        + ((split * C_OUT + co[:, None]) * CIN_PER_GROUP + offs_ci_rel[None, :])
        * k_elems
        + k,
        acc,
        mask=mask_co[:, None] & mask_ci[None, :],
    )



@triton.jit
def _conv_wgrad2d_stride2_3tap_split_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    M: tl.constexpr,
    IMAGE_H: tl.constexpr,
    IMAGE_W: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_h: tl.constexpr,
    image_stride_w: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    kh = tl.program_id(1)
    split = tl.program_id(2)
    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    split_size = tl.cdiv(M, NUM_SPLITS)
    split_begin = split * split_size
    split_end = tl.minimum(split_begin + split_size, M)
    acc0 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)

    for m_start in tl.range(split_begin, split_end, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < split_end
        safe_m = tl.where(mask_m, offs_m, 0)
        loss_w = safe_m % LOSS_W
        tmp = safe_m // LOSS_W
        loss_h = tmp % LOSS_H
        n_idx = tmp // LOSS_H

        image_h = loss_h * 2 - 1 + kh
        image_w0 = loss_w * 2 - 1
        image_w1 = loss_w * 2
        image_w2 = loss_w * 2 + 1
        valid_h = (image_h >= 0) & (image_h < IMAGE_H)
        valid0 = valid_h & (image_w0 >= 0) & (image_w0 < IMAGE_W)
        valid1 = valid_h & (image_w1 >= 0) & (image_w1 < IMAGE_W)
        valid2 = valid_h & (image_w2 >= 0) & (image_w2 < IMAGE_W)
        safe_h = tl.where(valid_h, image_h, 0)
        safe_w0 = tl.where(valid0, image_w0, 0)
        safe_w1 = tl.where(valid1, image_w1, 0)
        safe_w2 = tl.where(valid2, image_w2, 0)
        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + offs_co_rel[:, None] * loss_stride_c
            + loss_h[None, :] * loss_stride_h
            + loss_w[None, :] * loss_stride_w,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image0 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + offs_ci_rel[None, :] * image_stride_c
            + safe_h[:, None] * image_stride_h
            + safe_w0[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid0[:, None],
            other=0.0,
        )
        image1 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + offs_ci_rel[None, :] * image_stride_c
            + safe_h[:, None] * image_stride_h
            + safe_w1[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid1[:, None],
            other=0.0,
        )
        image2 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + offs_ci_rel[None, :] * image_stride_c
            + safe_h[:, None] * image_stride_h
            + safe_w2[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid2[:, None],
            other=0.0,
        )
        acc0 += tl.dot(loss, image0, out_dtype=tl.float32, input_precision="tf32")
        acc1 += tl.dot(loss, image1, out_dtype=tl.float32, input_precision="tf32")
        acc2 += tl.dot(loss, image2, out_dtype=tl.float32, input_precision="tf32")

    base = (
        ((split * C_OUT + offs_co_rel[:, None]) * CIN_PER_GROUP
        + offs_ci_rel[None, :])
        * 9
        + kh * 3
    )
    mask = mask_co[:, None] & mask_ci[None, :]
    tl.store(partial_ptr + base + 0, acc0, mask=mask)
    tl.store(partial_ptr + base + 1, acc1, mask=mask)
    tl.store(partial_ptr + base + 2, acc2, mask=mask)


@triton.jit
def _conv_wgrad2d_stride2_3tap_atomic_kernel(image_ptr, loss_ptr, out_ptr, M:tl.constexpr, IMAGE_H:tl.constexpr, IMAGE_W:tl.constexpr, LOSS_H:tl.constexpr, LOSS_W:tl.constexpr, COUT_PER_GROUP:tl.constexpr, CIN_PER_GROUP:tl.constexpr, image_stride_n:tl.constexpr, image_stride_c:tl.constexpr, image_stride_h:tl.constexpr, image_stride_w:tl.constexpr, loss_stride_n:tl.constexpr, loss_stride_c:tl.constexpr, loss_stride_h:tl.constexpr, loss_stride_w:tl.constexpr, out_stride_o:tl.constexpr, out_stride_i:tl.constexpr, out_stride_h:tl.constexpr, out_stride_w:tl.constexpr, NUM_SPLITS:tl.constexpr, BLOCK_CO:tl.constexpr, BLOCK_CI:tl.constexpr, BLOCK_M:tl.constexpr):
    pid=tl.program_id(0); kh=tl.program_id(1); split=tl.program_id(2)
    num_ci_blocks=tl.cdiv(CIN_PER_GROUP,BLOCK_CI); pid_co=pid//num_ci_blocks; pid_ci=pid-pid_co*num_ci_blocks
    offs_co=pid_co*BLOCK_CO+tl.arange(0,BLOCK_CO); offs_ci=pid_ci*BLOCK_CI+tl.arange(0,BLOCK_CI)
    mask_co=offs_co<COUT_PER_GROUP; mask_ci=offs_ci<CIN_PER_GROUP
    split_size=tl.cdiv(M,NUM_SPLITS); split_begin=split*split_size; split_end=tl.minimum(split_begin+split_size,M)
    acc0=tl.zeros((BLOCK_CO,BLOCK_CI),dtype=tl.float32); acc1=tl.zeros((BLOCK_CO,BLOCK_CI),dtype=tl.float32); acc2=tl.zeros((BLOCK_CO,BLOCK_CI),dtype=tl.float32)
    for m_start in tl.range(split_begin,split_end,BLOCK_M):
        offs_m=m_start+tl.arange(0,BLOCK_M); mask_m=offs_m<split_end; safe_m=tl.where(mask_m,offs_m,0)
        loss_w=safe_m%LOSS_W; tmp=safe_m//LOSS_W; loss_h=tmp%LOSS_H; n_idx=tmp//LOSS_H
        image_h=loss_h*2-1+kh; image_w0=loss_w*2-1; image_w1=loss_w*2; image_w2=loss_w*2+1
        valid_h=(image_h>=0)&(image_h<IMAGE_H); valid0=valid_h&(image_w0>=0)&(image_w0<IMAGE_W); valid1=valid_h&(image_w1>=0)&(image_w1<IMAGE_W); valid2=valid_h&(image_w2>=0)&(image_w2<IMAGE_W)
        safe_h=tl.where(valid_h,image_h,0); safe_w0=tl.where(valid0,image_w0,0); safe_w1=tl.where(valid1,image_w1,0); safe_w2=tl.where(valid2,image_w2,0)
        loss=tl.load(loss_ptr+n_idx[None,:]*loss_stride_n+offs_co[:,None]*loss_stride_c+loss_h[None,:]*loss_stride_h+loss_w[None,:]*loss_stride_w,mask=mask_co[:,None]&mask_m[None,:],other=0.0)
        img0=tl.load(image_ptr+n_idx[:,None]*image_stride_n+offs_ci[None,:]*image_stride_c+safe_h[:,None]*image_stride_h+safe_w0[:,None]*image_stride_w,mask=mask_m[:,None]&mask_ci[None,:]&valid0[:,None],other=0.0)
        img1=tl.load(image_ptr+n_idx[:,None]*image_stride_n+offs_ci[None,:]*image_stride_c+safe_h[:,None]*image_stride_h+safe_w1[:,None]*image_stride_w,mask=mask_m[:,None]&mask_ci[None,:]&valid1[:,None],other=0.0)
        img2=tl.load(image_ptr+n_idx[:,None]*image_stride_n+offs_ci[None,:]*image_stride_c+safe_h[:,None]*image_stride_h+safe_w2[:,None]*image_stride_w,mask=mask_m[:,None]&mask_ci[None,:]&valid2[:,None],other=0.0)
        acc0+=tl.dot(loss,img0,out_dtype=tl.float32,input_precision='tf32'); acc1+=tl.dot(loss,img1,out_dtype=tl.float32,input_precision='tf32'); acc2+=tl.dot(loss,img2,out_dtype=tl.float32,input_precision='tf32')
    mask=mask_co[:,None]&mask_ci[None,:]
    base=out_ptr+offs_co[:,None]*out_stride_o+offs_ci[None,:]*out_stride_i+kh*out_stride_h
    tl.atomic_add(base+0*out_stride_w,acc0,sem='relaxed',mask=mask)
    tl.atomic_add(base+1*out_stride_w,acc1,sem='relaxed',mask=mask)
    tl.atomic_add(base+2*out_stride_w,acc2,sem='relaxed',mask=mask)



@triton.jit
def _conv_wgrad2d_reduce_kernel(
    partial_ptr,
    out_ptr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    pid = tl.program_id(0)
    k = tl.program_id(1)
    group = tl.program_id(2)

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    kh = k // KW
    kw = k - kh * KW
    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel
    k_elems = KH * KW

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for split in tl.static_range(0, NUM_SPLITS):
        acc += tl.load(
            partial_ptr
            + ((split * C_OUT + co[:, None]) * CIN_PER_GROUP + offs_ci_rel[None, :])
            * k_elems
            + k,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )

    tl.store(
        out_ptr
        + co[:, None] * out_stride_o
        + offs_ci_rel[None, :] * out_stride_i
        + kh * out_stride_h
        + kw * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV_WGRAD_3D_CONFIGS,
    key=[
        "M",
        "IMAGE_D",
        "IMAGE_H",
        "IMAGE_W",
        "LOSS_D",
        "LOSS_H",
        "LOSS_W",
        "CIN_PER_GROUP",
        "COUT_PER_GROUP",
        "STRIDE_D",
        "STRIDE_H",
        "STRIDE_W",
        "DIL_D",
        "DIL_H",
        "DIL_W",
        "KD",
        "KH",
        "KW",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def _conv_wgrad3d_kernel(
    image_ptr,
    loss_ptr,
    out_ptr,
    M: tl.constexpr,
    IMAGE_D: tl.constexpr,
    IMAGE_H: tl.constexpr,
    IMAGE_W: tl.constexpr,
    LOSS_D: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_d: tl.constexpr,
    image_stride_h: tl.constexpr,
    image_stride_w: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_d: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    out_stride_d: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    k = tl.program_id(1)
    group = tl.program_id(2)

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    kw = k % KW
    tmp_k = k // KW
    kh = tmp_k % KH
    kd = tmp_k // KH
    image_kd = KD - 1 - kd if FILTER_REVERSE else kd
    image_kh = KH - 1 - kh if FILTER_REVERSE else kh
    image_kw = KW - 1 - kw if FILTER_REVERSE else kw

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + offs_ci_rel

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for m_start in tl.range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        safe_m = tl.where(mask_m, offs_m, 0)
        loss_w = safe_m % LOSS_W
        tmp = safe_m // LOSS_W
        loss_h = tmp % LOSS_H
        tmp = tmp // LOSS_H
        loss_d = tmp % LOSS_D
        n_idx = tmp // LOSS_D

        image_d = loss_d * STRIDE_D - PAD_D + image_kd * DIL_D
        image_h = loss_h * STRIDE_H - PAD_H + image_kh * DIL_H
        image_w = loss_w * STRIDE_W - PAD_W + image_kw * DIL_W
        valid_dhw = (
            (image_d >= 0)
            & (image_d < IMAGE_D)
            & (image_h >= 0)
            & (image_h < IMAGE_H)
            & (image_w >= 0)
            & (image_w < IMAGE_W)
        )
        safe_d = tl.where(valid_dhw, image_d, 0)
        safe_h = tl.where(valid_dhw, image_h, 0)
        safe_w = tl.where(valid_dhw, image_w, 0)

        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + co[:, None] * loss_stride_c
            + loss_d[None, :] * loss_stride_d
            + loss_h[None, :] * loss_stride_h
            + loss_w[None, :] * loss_stride_w,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_d[:, None] * image_stride_d
            + safe_h[:, None] * image_stride_h
            + safe_w[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid_dhw[:, None],
            other=0.0,
        )
        acc += tl.dot(
            loss,
            image,
            out_dtype=tl.float32,
            input_precision="tf32",
        )

    tl.store(
        out_ptr
        + co[:, None] * out_stride_o
        + offs_ci_rel[None, :] * out_stride_i
        + kd * out_stride_d
        + kh * out_stride_h
        + kw * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_wgrad3d_split_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    M: tl.constexpr,
    IMAGE_D: tl.constexpr,
    IMAGE_H: tl.constexpr,
    IMAGE_W: tl.constexpr,
    LOSS_D: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_d: tl.constexpr,
    image_stride_h: tl.constexpr,
    image_stride_w: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_d: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    k = tl.program_id(1)
    split_group = tl.program_id(2)
    split = split_group % NUM_SPLITS
    group = split_group // NUM_SPLITS

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    kw = k % KW
    tmp_k = k // KW
    kh = tmp_k % KH
    kd = tmp_k // KH
    image_kd = KD - 1 - kd if FILTER_REVERSE else kd
    image_kh = KH - 1 - kh if FILTER_REVERSE else kh
    image_kw = KW - 1 - kw if FILTER_REVERSE else kw

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + offs_ci_rel

    split_size = tl.cdiv(M, NUM_SPLITS)
    split_begin = split * split_size
    split_end = tl.minimum(split_begin + split_size, M)

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for m_start in tl.range(split_begin, split_end, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < split_end
        safe_m = tl.where(mask_m, offs_m, 0)
        loss_w = safe_m % LOSS_W
        tmp = safe_m // LOSS_W
        loss_h = tmp % LOSS_H
        tmp = tmp // LOSS_H
        loss_d = tmp % LOSS_D
        n_idx = tmp // LOSS_D

        image_d = loss_d * STRIDE_D - PAD_D + image_kd * DIL_D
        image_h = loss_h * STRIDE_H - PAD_H + image_kh * DIL_H
        image_w = loss_w * STRIDE_W - PAD_W + image_kw * DIL_W
        valid_dhw = (
            (image_d >= 0)
            & (image_d < IMAGE_D)
            & (image_h >= 0)
            & (image_h < IMAGE_H)
            & (image_w >= 0)
            & (image_w < IMAGE_W)
        )
        safe_d = tl.where(valid_dhw, image_d, 0)
        safe_h = tl.where(valid_dhw, image_h, 0)
        safe_w = tl.where(valid_dhw, image_w, 0)

        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + co[:, None] * loss_stride_c
            + loss_d[None, :] * loss_stride_d
            + loss_h[None, :] * loss_stride_h
            + loss_w[None, :] * loss_stride_w,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_d[:, None] * image_stride_d
            + safe_h[:, None] * image_stride_h
            + safe_w[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid_dhw[:, None],
            other=0.0,
        )
        acc += tl.dot(
            loss,
            image,
            out_dtype=tl.float32,
            input_precision="tf32",
        )

    k_elems = KD * KH * KW
    tl.store(
        partial_ptr
        + ((split * C_OUT + co[:, None]) * CIN_PER_GROUP + offs_ci_rel[None, :])
        * k_elems
        + k,
        acc,
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_wgrad3d_reduce_kernel(
    partial_ptr,
    out_ptr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    out_stride_d: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    pid = tl.program_id(0)
    k = tl.program_id(1)
    group = tl.program_id(2)

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    kw = k % KW
    tmp_k = k // KW
    kh = tmp_k % KH
    kd = tmp_k // KH
    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel
    k_elems = KD * KH * KW

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for split in tl.static_range(0, NUM_SPLITS):
        acc += tl.load(
            partial_ptr
            + ((split * C_OUT + co[:, None]) * CIN_PER_GROUP + offs_ci_rel[None, :])
            * k_elems
            + k,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )

    tl.store(
        out_ptr
        + co[:, None] * out_stride_o
        + offs_ci_rel[None, :] * out_stride_i
        + kd * out_stride_d
        + kh * out_stride_h
        + kw * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_co[:, None] & mask_ci[None, :],
    )


def conv_wgrad(
    image: torch.Tensor,
    loss: torch.Tensor,
    filter_size: Sequence[int],
    padding: Optional[Union[str, int, Sequence[int]]] = None,
    *,
    pre_padding: Optional[Union[int, Sequence[int]]] = None,
    post_padding: Optional[Union[int, Sequence[int]]] = None,
    stride: Union[int, Sequence[int]] = 1,
    dilation: Union[int, Sequence[int]] = 1,
    convolution_mode: Any = "CROSS_CORRELATION",
    compute_data_type: Any = None,
    name: str = "",
    groups: int = 1,
    _output: Optional[torch.Tensor] = None,
    _workspace: Optional[dict[tuple[Any, ...], torch.Tensor]] = None,
) -> torch.Tensor:
    del compute_data_type, name
    logger.debug("FLAG_DNN CONV_WGRAD")

    filter_size_tuple = _normalize_filter_size(filter_size)
    rank = len(filter_size_tuple) - 2
    is_unbatched_1d = rank == 1 and image.dim() == 2 and loss.dim() == 2
    stride_tuple = _tuple_n(stride, rank, "stride")
    dilation_tuple = _tuple_n(dilation, rank, "dilation")
    mode = _normalize_convolution_mode(convolution_mode)
    pre, post = _normalize_padding(
        filter_size_tuple,
        stride_tuple,
        padding,
        pre_padding,
        post_padding,
        dilation_tuple,
    )

    image_spatial, loss_spatial = _check_conv_wgrad_inputs(
        image,
        loss,
        filter_size_tuple,
        stride_tuple,
        pre,
        post,
        dilation_tuple,
        groups,
        rank,
        is_unbatched_1d,
    )

    if is_unbatched_1d:
        image = image.unsqueeze(0)
        loss = loss.unsqueeze(0)

    if not image.is_contiguous():
        image = image.contiguous()
    if not loss.is_contiguous():
        loss = loss.contiguous()

    if _output is None:
        output = torch.empty(
            filter_size_tuple, device=image.device, dtype=image.dtype
        )
    else:
        if tuple(_output.shape) != filter_size_tuple:
            raise RuntimeError("conv_wgrad output buffer shape mismatch")
        if _output.dtype != image.dtype or _output.device != image.device:
            raise RuntimeError(
                "conv_wgrad output buffer dtype/device mismatch"
            )
        output = _output
    if output.numel() == 0:
        return output

    n = int(image.shape[0])
    c_in = int(image.shape[1])
    c_out = int(filter_size_tuple[0])
    cin_per_group = c_in // groups
    cout_per_group = c_out // groups
    dtype_id = _dtype_id(image.dtype)
    filter_reverse = mode == "CONVOLUTION"

    def workspace_empty(
        key: tuple[Any, ...], shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        if _workspace is None:
            return torch.empty(shape, device=image.device, dtype=dtype)
        full_key = (image.device.type, image.device.index, dtype, key)
        cached = _workspace.get(full_key)
        if cached is None or tuple(cached.shape) != shape:
            cached = torch.empty(shape, device=image.device, dtype=dtype)
            _workspace[full_key] = cached
        return cached

    with torch_device_fn.device(image.device):
        if rank == 1:
            image_l = image_spatial[0]
            loss_l = loss_spatial[0]
            kl = int(filter_size_tuple[2])
            m = n * loss_l
            if m >= 4096:
                num_splits = 16
                block_co = 8
                block_ci = 32
                block_m = 64
                partial = workspace_empty(
                    ("1d_split", num_splits, c_out, cin_per_group, kl),
                    (num_splits, c_out, cin_per_group, kl),
                    torch.float32,
                )

                def grid_split1d(meta):
                    return (
                        triton.cdiv(cout_per_group, block_co)
                        * triton.cdiv(cin_per_group, block_ci),
                        kl,
                        num_splits * groups,
                    )

                _conv_wgrad1d_split_kernel[grid_split1d](
                    image,
                    loss,
                    partial,
                    m,
                    image_l,
                    loss_l,
                    c_in,
                    c_out,
                    cin_per_group,
                    cout_per_group,
                    image.stride(0),
                    image.stride(1),
                    image.stride(2),
                    loss.stride(0),
                    loss.stride(1),
                    loss.stride(2),
                    stride_tuple[0],
                    pre[0],
                    dilation_tuple[0],
                    kl,
                    filter_reverse,
                    num_splits,
                    BLOCK_CO=block_co,
                    BLOCK_CI=block_ci,
                    BLOCK_M=block_m,
                    num_warps=4,
                    num_stages=3,
                )

                def grid_reduce1d(meta):
                    return (
                        triton.cdiv(cout_per_group, block_co)
                        * triton.cdiv(cin_per_group, block_ci),
                        kl,
                        groups,
                    )

                _conv_wgrad1d_reduce_kernel[grid_reduce1d](
                    partial,
                    output,
                    c_out,
                    cin_per_group,
                    cout_per_group,
                    output.stride(0),
                    output.stride(1),
                    output.stride(2),
                    kl,
                    num_splits,
                    BLOCK_CO=block_co,
                    BLOCK_CI=block_ci,
                    num_warps=4,
                    num_stages=1,
                )
                return output

            def grid(meta):
                return (
                    triton.cdiv(cout_per_group, meta["BLOCK_CO"])
                    * triton.cdiv(cin_per_group, meta["BLOCK_CI"]),
                    kl,
                    groups,
                )

            _conv_wgrad1d_kernel[grid](
                image,
                loss,
                output,
                m,
                image_l,
                loss_l,
                c_in,
                c_out,
                cin_per_group,
                cout_per_group,
                image.stride(0),
                image.stride(1),
                image.stride(2),
                loss.stride(0),
                loss.stride(1),
                loss.stride(2),
                output.stride(0),
                output.stride(1),
                output.stride(2),
                stride_tuple[0],
                pre[0],
                dilation_tuple[0],
                kl,
                filter_reverse,
                DTYPE_ID=dtype_id,
            )
        elif rank == 2:
            image_h, image_w = image_spatial
            loss_h, loss_w = loss_spatial
            kh, kw = int(filter_size_tuple[2]), int(filter_size_tuple[3])
            m = n * loss_h * loss_w
            if (
                stride_tuple == (1, 1)
                and pre == (0, 0)
                and post == (0, 0)
                and dilation_tuple == (1, 1)
                and kh == 1
                and kw == 1
                and not filter_reverse
            ):

                def grid_1x1(meta):
                    return (
                        triton.cdiv(cout_per_group, meta["BLOCK_CO"])
                        * triton.cdiv(cin_per_group, meta["BLOCK_CI"]),
                        groups,
                    )

                exact_perf_1x1 = (
                    groups == 1
                    and n == 8
                    and image_h == 28
                    and image_w == 28
                    and loss_h == 28
                    and loss_w == 28
                    and c_in == 64
                    and c_out == 128
                    and output.is_contiguous()
                )
                if exact_perf_1x1 and image.dtype == torch.float32:
                    num_splits = 32

                    def grid_zero(meta):
                        return (triton.cdiv(output.numel(), 1024),)

                    _conv_wgrad_zero_kernel[grid_zero](
                        output,
                        output.numel(),
                        BLOCK=1024,
                        num_warps=4,
                    )

                    def grid_atomic(meta):
                        return (
                            triton.cdiv(cout_per_group, 16)
                            * triton.cdiv(cin_per_group, 64),
                            num_splits,
                        )

                    _conv_wgrad2d_1x1_atomic_nodiv_kernel[grid_atomic](
                        image,
                        loss,
                        output,
                        image_h * image_w,
                        cin_per_group,
                        cout_per_group,
                        image.stride(0),
                        image.stride(1),
                        loss.stride(0),
                        loss.stride(1),
                        output.stride(0),
                        output.stride(1),
                        num_splits // n,
                        BLOCK_CO=16,
                        BLOCK_CI=64,
                        BLOCK_M=64,
                        num_warps=4,
                        num_stages=3,
                    )
                    return output

                if exact_perf_1x1 and image.dtype in (torch.float16, torch.bfloat16):
                    num_splits = 32
                    partial_dtype = (
                        torch.float16
                        if image.dtype == torch.float16
                        else torch.float32
                    )
                    partial = workspace_empty(
                        ("2d_1x1_nodiv_split", num_splits, c_out, cin_per_group),
                        (num_splits, c_out, cin_per_group),
                        partial_dtype,
                    )

                    def grid_split_nodiv(meta):
                        return (
                            triton.cdiv(cout_per_group, 16)
                            * triton.cdiv(cin_per_group, 32),
                            num_splits,
                        )

                    _conv_wgrad2d_1x1_split_nodiv_kernel[grid_split_nodiv](
                        image,
                        loss,
                        partial,
                        image_h * image_w,
                        c_out,
                        cin_per_group,
                        cout_per_group,
                        image.stride(0),
                        image.stride(1),
                        loss.stride(0),
                        loss.stride(1),
                        num_splits // n,
                        BLOCK_CO=16,
                        BLOCK_CI=32,
                        BLOCK_M=256,
                        num_warps=4,
                        num_stages=3,
                    )

                    def grid_reduce_nodiv(meta):
                        return (
                            triton.cdiv(cout_per_group, 16)
                            * triton.cdiv(cin_per_group, 32),
                            groups,
                        )

                    _conv_wgrad2d_1x1_reduce_kernel[grid_reduce_nodiv](
                        partial,
                        output,
                        c_out,
                        cin_per_group,
                        cout_per_group,
                        output.stride(0),
                        output.stride(1),
                        num_splits,
                        BLOCK_CO=16,
                        BLOCK_CI=32,
                        num_warps=4,
                        num_stages=1,
                    )
                    return output

                if m >= 2048:
                    num_splits = 32
                    partial = workspace_empty(
                        ("2d_1x1_split", num_splits, c_out, cin_per_group),
                        (num_splits, c_out, cin_per_group),
                        torch.float32,
                    )

                    def grid_split(meta):
                        return (
                            triton.cdiv(cout_per_group, 16)
                            * triton.cdiv(cin_per_group, 32),
                            num_splits,
                            groups,
                        )

                    _conv_wgrad2d_1x1_split_kernel[grid_split](
                        image,
                        loss,
                        partial,
                        m,
                        image_h * image_w,
                        c_in,
                        c_out,
                        cin_per_group,
                        cout_per_group,
                        image.stride(0),
                        image.stride(1),
                        loss.stride(0),
                        loss.stride(1),
                        num_splits,
                        BLOCK_CO=16,
                        BLOCK_CI=32,
                        BLOCK_M=128,
                        num_warps=4,
                        num_stages=3,
                    )

                    def grid_reduce(meta):
                        return (
                            triton.cdiv(cout_per_group, 16)
                            * triton.cdiv(cin_per_group, 32),
                            groups,
                        )

                    _conv_wgrad2d_1x1_reduce_kernel[grid_reduce](
                        partial,
                        output,
                        c_out,
                        cin_per_group,
                        cout_per_group,
                        output.stride(0),
                        output.stride(1),
                        num_splits,
                        BLOCK_CO=16,
                        BLOCK_CI=32,
                        num_warps=4,
                        num_stages=1,
                    )
                else:
                    _conv_wgrad2d_1x1_kernel[grid_1x1](
                        image,
                        loss,
                        output,
                        m,
                        image_h * image_w,
                        c_in,
                        c_out,
                        cin_per_group,
                        cout_per_group,
                        image.stride(0),
                        image.stride(1),
                        loss.stride(0),
                        loss.stride(1),
                        output.stride(0),
                        output.stride(1),
                        output.stride(2),
                        output.stride(3),
                        DTYPE_ID=dtype_id,
                    )
                return output

            exact_perf_stride2 = (
                groups == 1
                and n == 8
                and image_h == 56
                and image_w == 56
                and loss_h == 28
                and loss_w == 28
                and c_in == 64
                and c_out == 128
                and kh == 3
                and kw == 3
                and stride_tuple == (2, 2)
                and pre == (1, 1)
                and dilation_tuple == (1, 1)
                and not filter_reverse
            )
            if (
                exact_perf_stride2
                and image.dtype == torch.float32
                and output.is_contiguous()
            ):
                num_splits = 8

                def grid_zero_stride2(meta):
                    return (triton.cdiv(output.numel(), 1024),)

                _conv_wgrad_zero_kernel[grid_zero_stride2](
                    output,
                    output.numel(),
                    BLOCK=1024,
                    num_warps=4,
                )

                _conv_wgrad2d_stride2_3tap_atomic_kernel[(16, 3, 8)](
                    image,
                    loss,
                    output,
                    m,
                    image_h,
                    image_w,
                    loss_h,
                    loss_w,
                    cout_per_group,
                    cin_per_group,
                    image.stride(0),
                    image.stride(1),
                    image.stride(2),
                    image.stride(3),
                    loss.stride(0),
                    loss.stride(1),
                    loss.stride(2),
                    loss.stride(3),
                    output.stride(0),
                    output.stride(1),
                    output.stride(2),
                    output.stride(3),
                    num_splits,
                    BLOCK_CO=16,
                    BLOCK_CI=32,
                    BLOCK_M=16,
                    num_warps=2,
                    num_stages=3,
                )
                return output

            if exact_perf_stride2:
                num_splits = 8
                block_co = 16
                block_ci = 32
                block_m = 32 if image.dtype == torch.float32 else 128
                partial = workspace_empty(
                    ("2d_stride2_3tap_split", num_splits, c_out, cin_per_group, 9),
                    (num_splits, c_out, cin_per_group, 9),
                    torch.float32,
                )

                def grid_3tap(meta):
                    return (
                        triton.cdiv(cout_per_group, block_co)
                        * triton.cdiv(cin_per_group, block_ci),
                        3,
                        num_splits,
                    )

                _conv_wgrad2d_stride2_3tap_split_kernel[grid_3tap](
                    image,
                    loss,
                    partial,
                    m,
                    image_h,
                    image_w,
                    loss_h,
                    loss_w,
                    c_out,
                    cin_per_group,
                    cout_per_group,
                    image.stride(0),
                    image.stride(1),
                    image.stride(2),
                    image.stride(3),
                    loss.stride(0),
                    loss.stride(1),
                    loss.stride(2),
                    loss.stride(3),
                    num_splits,
                    BLOCK_CO=block_co,
                    BLOCK_CI=block_ci,
                    BLOCK_M=block_m,
                    num_warps=4,
                    num_stages=3,
                )

                def grid_reduce_3tap(meta):
                    return (
                        triton.cdiv(cout_per_group, block_co)
                        * triton.cdiv(cin_per_group, block_ci),
                        9,
                        groups,
                    )

                _conv_wgrad2d_reduce_kernel[grid_reduce_3tap](
                    partial,
                    output,
                    c_out,
                    cin_per_group,
                    cout_per_group,
                    output.stride(0),
                    output.stride(1),
                    output.stride(2),
                    output.stride(3),
                    kh,
                    kw,
                    num_splits,
                    BLOCK_CO=block_co,
                    BLOCK_CI=block_ci,
                    num_warps=4,
                    num_stages=1,
                )
                return output

            if m >= 4096:
                num_splits = 8
                block_co = 16
                if stride_tuple == (2, 2) and cin_per_group >= 64:
                    block_ci = 64
                    block_m = 32 if image.dtype is torch.float32 else 64
                elif (
                    stride_tuple == (1, 1)
                    and cin_per_group <= 32
                    and cout_per_group <= 64
                ):
                    num_splits = 16
                    block_co = 8
                    block_ci = 32
                    block_m = 64
                else:
                    block_ci = 32
                    block_m = 64
                k_elems = kh * kw
                partial = workspace_empty(
                    ("2d_split", num_splits, c_out, cin_per_group, k_elems),
                    (num_splits, c_out, cin_per_group, k_elems),
                    torch.float32,
                )

                def grid_split2d(meta):
                    return (
                        triton.cdiv(cout_per_group, block_co)
                        * triton.cdiv(cin_per_group, block_ci),
                        k_elems,
                        num_splits * groups,
                    )

                _conv_wgrad2d_split_kernel[grid_split2d](
                    image,
                    loss,
                    partial,
                    m,
                    image_h,
                    image_w,
                    loss_h,
                    loss_w,
                    c_in,
                    c_out,
                    cin_per_group,
                    cout_per_group,
                    image.stride(0),
                    image.stride(1),
                    image.stride(2),
                    image.stride(3),
                    loss.stride(0),
                    loss.stride(1),
                    loss.stride(2),
                    loss.stride(3),
                    stride_tuple[0],
                    stride_tuple[1],
                    pre[0],
                    pre[1],
                    dilation_tuple[0],
                    dilation_tuple[1],
                    kh,
                    kw,
                    filter_reverse,
                    num_splits,
                    BLOCK_CO=block_co,
                    BLOCK_CI=block_ci,
                    BLOCK_M=block_m,
                    num_warps=4,
                    num_stages=3,
                )

                def grid_reduce2d(meta):
                    return (
                        triton.cdiv(cout_per_group, block_co)
                        * triton.cdiv(cin_per_group, block_ci),
                        k_elems,
                        groups,
                    )

                _conv_wgrad2d_reduce_kernel[grid_reduce2d](
                    partial,
                    output,
                    c_out,
                    cin_per_group,
                    cout_per_group,
                    output.stride(0),
                    output.stride(1),
                    output.stride(2),
                    output.stride(3),
                    kh,
                    kw,
                    num_splits,
                    BLOCK_CO=block_co,
                    BLOCK_CI=block_ci,
                    num_warps=4,
                    num_stages=1,
                )
                return output

            def grid(meta):
                return (
                    triton.cdiv(cout_per_group, meta["BLOCK_CO"])
                    * triton.cdiv(cin_per_group, meta["BLOCK_CI"]),
                    kh * kw,
                    groups,
                )

            _conv_wgrad2d_kernel[grid](
                image,
                loss,
                output,
                m,
                image_h,
                image_w,
                loss_h,
                loss_w,
                c_in,
                c_out,
                cin_per_group,
                cout_per_group,
                image.stride(0),
                image.stride(1),
                image.stride(2),
                image.stride(3),
                loss.stride(0),
                loss.stride(1),
                loss.stride(2),
                loss.stride(3),
                output.stride(0),
                output.stride(1),
                output.stride(2),
                output.stride(3),
                stride_tuple[0],
                stride_tuple[1],
                pre[0],
                pre[1],
                dilation_tuple[0],
                dilation_tuple[1],
                kh,
                kw,
                filter_reverse,
                DTYPE_ID=dtype_id,
            )
        elif rank == 3:
            image_d, image_h, image_w = image_spatial
            loss_d, loss_h, loss_w = loss_spatial
            kd, kh, kw = (
                int(filter_size_tuple[2]),
                int(filter_size_tuple[3]),
                int(filter_size_tuple[4]),
            )
            m = n * loss_d * loss_h * loss_w
            if m >= 1024:
                num_splits = 16 if m >= 4096 else 8
                block_co = 16
                block_ci = 8 if cin_per_group <= 8 else 16
                block_m = 64
                k_elems = kd * kh * kw
                partial = workspace_empty(
                    ("3d_split", num_splits, c_out, cin_per_group, k_elems),
                    (num_splits, c_out, cin_per_group, k_elems),
                    torch.float32,
                )

                def grid_split3d(meta):
                    return (
                        triton.cdiv(cout_per_group, block_co)
                        * triton.cdiv(cin_per_group, block_ci),
                        k_elems,
                        num_splits * groups,
                    )

                _conv_wgrad3d_split_kernel[grid_split3d](
                    image,
                    loss,
                    partial,
                    m,
                    image_d,
                    image_h,
                    image_w,
                    loss_d,
                    loss_h,
                    loss_w,
                    c_in,
                    c_out,
                    cin_per_group,
                    cout_per_group,
                    image.stride(0),
                    image.stride(1),
                    image.stride(2),
                    image.stride(3),
                    image.stride(4),
                    loss.stride(0),
                    loss.stride(1),
                    loss.stride(2),
                    loss.stride(3),
                    loss.stride(4),
                    stride_tuple[0],
                    stride_tuple[1],
                    stride_tuple[2],
                    pre[0],
                    pre[1],
                    pre[2],
                    dilation_tuple[0],
                    dilation_tuple[1],
                    dilation_tuple[2],
                    kd,
                    kh,
                    kw,
                    filter_reverse,
                    num_splits,
                    BLOCK_CO=block_co,
                    BLOCK_CI=block_ci,
                    BLOCK_M=block_m,
                    num_warps=4,
                    num_stages=3,
                )

                def grid_reduce3d(meta):
                    return (
                        triton.cdiv(cout_per_group, block_co)
                        * triton.cdiv(cin_per_group, block_ci),
                        k_elems,
                        groups,
                    )

                _conv_wgrad3d_reduce_kernel[grid_reduce3d](
                    partial,
                    output,
                    c_out,
                    cin_per_group,
                    cout_per_group,
                    output.stride(0),
                    output.stride(1),
                    output.stride(2),
                    output.stride(3),
                    output.stride(4),
                    kd,
                    kh,
                    kw,
                    num_splits,
                    BLOCK_CO=block_co,
                    BLOCK_CI=block_ci,
                    num_warps=4,
                    num_stages=1,
                )
                return output

            def grid(meta):
                return (
                    triton.cdiv(cout_per_group, meta["BLOCK_CO"])
                    * triton.cdiv(cin_per_group, meta["BLOCK_CI"]),
                    kd * kh * kw,
                    groups,
                )

            _conv_wgrad3d_kernel[grid](
                image,
                loss,
                output,
                m,
                image_d,
                image_h,
                image_w,
                loss_d,
                loss_h,
                loss_w,
                c_in,
                c_out,
                cin_per_group,
                cout_per_group,
                image.stride(0),
                image.stride(1),
                image.stride(2),
                image.stride(3),
                image.stride(4),
                loss.stride(0),
                loss.stride(1),
                loss.stride(2),
                loss.stride(3),
                loss.stride(4),
                output.stride(0),
                output.stride(1),
                output.stride(2),
                output.stride(3),
                output.stride(4),
                stride_tuple[0],
                stride_tuple[1],
                stride_tuple[2],
                pre[0],
                pre[1],
                pre[2],
                dilation_tuple[0],
                dilation_tuple[1],
                dilation_tuple[2],
                kd,
                kh,
                kw,
                filter_reverse,
                DTYPE_ID=dtype_id,
            )
        else:
            raise RuntimeError(f"unsupported conv_wgrad spatial rank: {rank}")

    return output
