from __future__ import annotations

import logging
import weakref
from collections import OrderedDict
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner

logger = logging.getLogger(__name__)

_CONV_DGRAD_1D_CONFIGS = runtime.get_tuned_config("conv_dgrad_1d")
_CONV_DGRAD_1D_MCI_CONFIGS = runtime.get_tuned_config("conv_dgrad_1d_mci")
_CONV_DGRAD_2D_CONFIGS = runtime.get_tuned_config("conv_dgrad_2d")
_CONV_DGRAD_2D_1X1_CONFIGS = runtime.get_tuned_config("conv_dgrad_2d_1x1")
_CONV_DGRAD_2D_1X1_STRIDED_CONFIGS = runtime.get_tuned_config(
    "conv_dgrad_2d_1x1_strided"
)
_CONV_DGRAD_2D_STRIDE1_CONFIGS = runtime.get_tuned_config(
    "conv_dgrad_2d_stride1"
)
_CONV_DGRAD_2D_STRIDE2_3X3_MCI_CONFIGS = runtime.get_tuned_config(
    "conv_dgrad_2d_stride2_pad1_3x3_mci"
)
_CONV_DGRAD_2D_STRIDE2_3X3_TILE4_CONFIGS = runtime.get_tuned_config(
    "conv_dgrad_2d_stride2_pad1_3x3_tile4"
)
_CONV_DGRAD_2D_STRIDE2_3X3_PACKED_CONFIGS = runtime.get_tuned_config(
    "conv_dgrad_2d_stride2_pad1_3x3_packed"
)
_CONV_DGRAD_2D_STRIDE2_3X3_PACKED_MCI_CONFIGS = runtime.get_tuned_config(
    "conv_dgrad_2d_stride2_pad1_3x3_packed_mci"
)
_CONV_DGRAD_2D_STRIDE2_3X3_TILE2W_CONFIGS = runtime.get_tuned_config(
    "conv_dgrad_2d_stride2_pad1_3x3_tile2w"
)

_CONV_DGRAD_3D_CONFIGS = runtime.get_tuned_config("conv_dgrad_3d")
_CONV_DGRAD_3D_PAD1_3X3_FP32_CI8_DOT_CONFIGS = runtime.get_tuned_config(
    "conv_dgrad_3d_pad1_3x3_fp32_ci8_dot"
)
_CONV_DGRAD_3D_PACKED_CONFIGS = runtime.get_tuned_config(
    "conv_dgrad_3d_packed"
)

_PACKED_WEIGHT_CACHE: OrderedDict[tuple, tuple[Any, torch.Tensor]] = (
    OrderedDict()
)
_PACKED_WEIGHT_CACHE_MAX = 32


def _weight_cache_key(tag: str, weight: torch.Tensor, groups: int) -> tuple:
    version = int(getattr(weight, "_version", 0))
    return (
        tag,
        id(weight),
        weight.data_ptr(),
        tuple(weight.shape),
        tuple(weight.stride()),
        str(weight.dtype),
        weight.device.type,
        weight.device.index,
        groups,
        version,
    )


def _cache_get_or_create(
    key: tuple, owner: torch.Tensor, fn: Callable[[], torch.Tensor]
) -> torch.Tensor:
    entry = _PACKED_WEIGHT_CACHE.get(key)
    if entry is not None:
        owner_ref, value = entry
        if owner_ref() is owner:
            _PACKED_WEIGHT_CACHE.move_to_end(key)
            return value
        del _PACKED_WEIGHT_CACHE[key]

    value = fn()
    _PACKED_WEIGHT_CACHE[key] = (weakref.ref(owner), value)
    _PACKED_WEIGHT_CACHE.move_to_end(key)
    while len(_PACKED_WEIGHT_CACHE) > _PACKED_WEIGHT_CACHE_MAX:
        _PACKED_WEIGHT_CACHE.popitem(last=False)
    return value


def _pack_weight_2d_khw_oci(weight: torch.Tensor, groups: int) -> torch.Tensor:
    key = _weight_cache_key("conv_dgrad_2d_khw_oci", weight, groups)

    def _fn() -> torch.Tensor:
        base = weight.contiguous()
        c_out, cin_g, kh, kw = base.shape
        cout_g = c_out // groups
        return (
            base.view(groups, cout_g, cin_g, kh, kw)
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )

    return _cache_get_or_create(key, weight, _fn)


def _pack_weight_3d_kdhw_oci(
    weight: torch.Tensor, groups: int
) -> torch.Tensor:
    key = _weight_cache_key("conv_dgrad_3d_kdhw_oci", weight, groups)

    def _fn() -> torch.Tensor:
        base = weight.contiguous()
        c_out, cin_g, kd, kh, kw = base.shape
        cout_g = c_out // groups
        return (
            base.view(groups, cout_g, cin_g, kd, kh, kw)
            .permute(0, 3, 4, 5, 1, 2)
            .contiguous()
        )

    return _cache_get_or_create(key, weight, _fn)


def _dtype_id(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return 0
    if dtype == torch.bfloat16:
        return 1
    if dtype == torch.float32:
        return 2
    if dtype == torch.float64:
        return 3
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


def _normalize_input_size(input_size: Sequence[int]) -> Tuple[int, ...]:
    result = tuple(int(dim) for dim in input_size)
    if not result:
        raise RuntimeError("input_size must be non-empty")
    if any(dim < 0 for dim in result):
        raise RuntimeError("input_size dimensions must be non-negative")
    return result


def _rank_from_input_size(
    input_size: Tuple[int, ...], weight: torch.Tensor
) -> int:
    if weight.dim() not in (3, 4, 5):
        raise RuntimeError(
            "flag_dnn conv_dgrad expects 1D/2D/3D convolution filter"
        )
    rank = weight.dim() - 2
    if rank == 1 and len(input_size) == 2:
        return rank
    if len(input_size) == rank + 2:
        return rank
    raise RuntimeError(
        f"input_size must have length {rank + 2}"
        + (" or 2 for unbatched 1D" if rank == 1 else "")
    )


def _normalize_padding(
    weight: torch.Tensor,
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
                "conv_dgrad accepts either padding or "
                "pre_padding/post_padding"
            )
        if pre_padding is None or post_padding is None:
            raise TypeError(
                "conv_dgrad requires both pre_padding and post_padding"
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
                    kernel = int(weight.shape[2 + axis])
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


def _check_conv_dgrad_inputs(
    loss: torch.Tensor,
    weight: torch.Tensor,
    input_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    pre_padding: Tuple[int, ...],
    post_padding: Tuple[int, ...],
    dilation: Tuple[int, ...],
    groups: int,
    rank: int,
    is_unbatched_1d: bool,
) -> Tuple[int, int, Tuple[int, ...], Tuple[int, ...]]:
    if not loss.is_cuda or not weight.is_cuda:
        raise NotImplementedError(
            "flag_dnn conv_dgrad Triton implementation requires CUDA inputs"
        )
    if loss.device != weight.device:
        raise RuntimeError("loss and filter must be on the same device")
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if loss.dtype not in supported_dtypes:
        raise NotImplementedError(
            f"flag_dnn conv_dgrad does not support loss dtype={loss.dtype}"
        )
    if weight.dtype != loss.dtype:
        raise RuntimeError("loss and filter must have the same dtype")
    if groups <= 0:
        raise RuntimeError("groups must be a positive integer")
    if any(dim <= 0 for dim in stride):
        raise RuntimeError("stride must be positive")
    if any(dim <= 0 for dim in dilation):
        raise RuntimeError("dilation must be positive")

    expected_loss_dim = rank + 1 if is_unbatched_1d else rank + 2
    if loss.dim() != expected_loss_dim:
        raise RuntimeError(
            f"flag_dnn conv_dgrad expected loss dim={expected_loss_dim}, "
            f"got {loss.dim()}"
        )
    if weight.dim() != rank + 2:
        raise RuntimeError(
            f"flag_dnn conv_dgrad expected filter dim={rank + 2}, "
            f"got {weight.dim()}"
        )

    dx_shape = (1,) + input_size if is_unbatched_1d else input_size
    loss_shape = (
        (1,) + tuple(loss.shape) if is_unbatched_1d else tuple(loss.shape)
    )
    c_in = dx_shape[1]
    c_out = int(weight.shape[0])
    c_per_group = int(weight.shape[1])
    if c_in <= 0 or c_out <= 0 or c_per_group <= 0:
        raise RuntimeError("channel dimensions must be non-empty")
    if c_in % groups != 0 or c_out % groups != 0:
        raise RuntimeError("channels must be divisible by groups")
    if c_per_group != c_in // groups:
        raise RuntimeError(
            "filter.shape[1] must match input_channels // groups"
        )
    if loss_shape[0] != dx_shape[0]:
        raise RuntimeError("loss batch size must match input_size batch size")
    if loss_shape[1] != c_out:
        raise RuntimeError("loss channels must match filter output channels")

    dx_spatial = tuple(int(v) for v in dx_shape[2:])
    kernel_shape = tuple(int(v) for v in weight.shape[2:])
    expected_loss_spatial = tuple(
        _conv_out_dim(
            dx_spatial[axis],
            pre_padding[axis],
            post_padding[axis],
            dilation[axis],
            kernel_shape[axis],
            stride[axis],
        )
        for axis in range(rank)
    )
    if tuple(loss_shape[2:]) != expected_loss_spatial:
        raise RuntimeError(
            "loss spatial shape does not match input_size/filter/stride/"
            f"padding/dilation: expected {expected_loss_spatial}, "
            f"got {tuple(loss_shape[2:])}"
        )
    return c_in, c_out, dx_spatial, expected_loss_spatial


@libentry()
@libtuner(
    configs=_CONV_DGRAD_1D_CONFIGS,
    key=[
        "M",
        "X_LEN",
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
def _conv_dgrad1d_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    X_LEN: tl.constexpr,
    LOSS_LEN: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_l: tl.constexpr,
    weight_stride_o: tl.constexpr,
    weight_stride_i: tl.constexpr,
    weight_stride_k: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_l: tl.constexpr,
    STRIDE_L: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_L: tl.constexpr,
    KL: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    n_idx = offs_m // X_LEN
    x_idx = offs_m - n_idx * X_LEN
    ci = group * CIN_PER_GROUP + offs_ci_rel

    acc = tl.zeros((BLOCK_CI, BLOCK_M), dtype=tl.float32)

    for kl in tl.static_range(0, KL):
        loss_l_num = x_idx + PAD_LEFT - kl * DIL_L
        loss_l = loss_l_num // STRIDE_L
        valid_l = (loss_l_num >= 0) & (loss_l < LOSS_LEN)
        if STRIDE_L != 1:
            valid_l = valid_l & ((loss_l_num % STRIDE_L) == 0)
        safe_l = tl.where(valid_l, loss_l, 0)
        weight_l = KL - 1 - kl if FILTER_REVERSE else kl

        for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
            offs_co_rel = co_start + tl.arange(0, BLOCK_CO)
            co = group * COUT_PER_GROUP + offs_co_rel
            mask_co = offs_co_rel < COUT_PER_GROUP

            loss = tl.load(
                loss_ptr
                + n_idx[None, :] * loss_stride_n
                + co[:, None] * loss_stride_c
                + safe_l[None, :] * loss_stride_l,
                mask=mask_co[:, None] & mask_m[None, :] & valid_l[None, :],
                other=0.0,
            )
            weight = tl.load(
                weight_ptr
                + co[None, :] * weight_stride_o
                + offs_ci_rel[:, None] * weight_stride_i
                + weight_l * weight_stride_k,
                mask=mask_ci[:, None] & mask_co[None, :],
                other=0.0,
            )
            acc += tl.dot(weight, loss, out_dtype=tl.float32)

    tl.store(
        out_ptr
        + n_idx[None, :] * out_stride_n
        + ci[:, None] * out_stride_c
        + x_idx[None, :] * out_stride_l,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_ci[:, None] & mask_m[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV_DGRAD_1D_MCI_CONFIGS,
    key=[
        "M",
        "X_LEN",
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
def _conv_dgrad1d_mci_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    X_LEN: tl.constexpr,
    LOSS_LEN: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_l: tl.constexpr,
    weight_stride_o: tl.constexpr,
    weight_stride_i: tl.constexpr,
    weight_stride_k: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_l: tl.constexpr,
    STRIDE_L: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_L: tl.constexpr,
    KL: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    n_idx = offs_m // X_LEN
    x_idx = offs_m - n_idx * X_LEN
    ci = group * CIN_PER_GROUP + offs_ci_rel

    acc = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for kl in tl.static_range(0, KL):
        loss_l_num = x_idx + PAD_LEFT - kl * DIL_L
        loss_l = loss_l_num // STRIDE_L
        valid_l = (loss_l_num >= 0) & (loss_l < LOSS_LEN)
        if STRIDE_L != 1:
            valid_l = valid_l & ((loss_l_num % STRIDE_L) == 0)
        safe_l = tl.where(valid_l, loss_l, 0)
        weight_l = KL - 1 - kl if FILTER_REVERSE else kl

        for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
            offs_co_rel = co_start + tl.arange(0, BLOCK_CO)
            co = group * COUT_PER_GROUP + offs_co_rel
            mask_co = offs_co_rel < COUT_PER_GROUP

            loss = tl.load(
                loss_ptr
                + n_idx[:, None] * loss_stride_n
                + co[None, :] * loss_stride_c
                + safe_l[:, None] * loss_stride_l,
                mask=mask_m[:, None] & mask_co[None, :] & valid_l[:, None],
                other=0.0,
            )
            weight = tl.load(
                weight_ptr
                + co[:, None] * weight_stride_o
                + offs_ci_rel[None, :] * weight_stride_i
                + weight_l * weight_stride_k,
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            acc += tl.dot(loss, weight, out_dtype=tl.float32)

    tl.store(
        out_ptr
        + n_idx[:, None] * out_stride_n
        + ci[None, :] * out_stride_c
        + x_idx[:, None] * out_stride_l,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_ci[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV_DGRAD_2D_CONFIGS,
    key=[
        "M",
        "XH",
        "XW",
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
def _conv_dgrad2d_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    weight_stride_o: tl.constexpr,
    weight_stride_i: tl.constexpr,
    weight_stride_h: tl.constexpr,
    weight_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    spatial = XH * XW
    n_idx = offs_m // spatial
    spatial_idx = offs_m - n_idx * spatial
    xh = spatial_idx // XW
    xw = spatial_idx - xh * XW
    ci = group * CIN_PER_GROUP + offs_ci_rel

    acc = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        loss_h_num = xh + PAD_TOP - kh * DIL_H
        loss_h = loss_h_num // STRIDE_H
        valid_h = (loss_h_num >= 0) & (loss_h < LOSS_H)
        if STRIDE_H != 1:
            valid_h = valid_h & ((loss_h_num % STRIDE_H) == 0)
        safe_h = tl.where(valid_h, loss_h, 0)
        weight_h = KH - 1 - kh if FILTER_REVERSE else kh

        for kw in tl.static_range(0, KW):
            loss_w_num = xw + PAD_LEFT - kw * DIL_W
            loss_w = loss_w_num // STRIDE_W
            valid_w = (loss_w_num >= 0) & (loss_w < LOSS_W)
            if STRIDE_W != 1:
                valid_w = valid_w & ((loss_w_num % STRIDE_W) == 0)
            valid_hw = valid_h & valid_w
            safe_w = tl.where(valid_w, loss_w, 0)
            weight_w = KW - 1 - kw if FILTER_REVERSE else kw

            for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
                offs_co_rel = co_start + tl.arange(0, BLOCK_CO)
                co = group * COUT_PER_GROUP + offs_co_rel
                mask_co = offs_co_rel < COUT_PER_GROUP

                loss = tl.load(
                    loss_ptr
                    + n_idx[:, None] * loss_stride_n
                    + co[None, :] * loss_stride_c
                    + safe_h[:, None] * loss_stride_h
                    + safe_w[:, None] * loss_stride_w,
                    mask=(
                        mask_m[:, None] & mask_co[None, :] & valid_hw[:, None]
                    ),
                    other=0.0,
                )
                weight = tl.load(
                    weight_ptr
                    + co[:, None] * weight_stride_o
                    + offs_ci_rel[None, :] * weight_stride_i
                    + weight_h * weight_stride_h
                    + weight_w * weight_stride_w,
                    mask=mask_co[:, None] & mask_ci[None, :],
                    other=0.0,
                )
                acc += tl.dot(loss, weight, out_dtype=tl.float32)

    tl.store(
        out_ptr
        + n_idx[:, None] * out_stride_n
        + ci[None, :] * out_stride_c
        + xh[:, None] * out_stride_h
        + xw[:, None] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_ci[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV_DGRAD_2D_1X1_CONFIGS,
    key=[
        "M",
        "XH",
        "XW",
        "CIN_PER_GROUP",
        "COUT_PER_GROUP",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def _conv_dgrad2d_1x1_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    weight_stride_o: tl.constexpr,
    weight_stride_i: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    spatial = XH * XW
    n_idx = offs_m // spatial
    spatial_idx = offs_m - n_idx * spatial
    ci = group * CIN_PER_GROUP + offs_ci_rel

    acc = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
        offs_co_rel = co_start + tl.arange(0, BLOCK_CO)
        co = group * COUT_PER_GROUP + offs_co_rel
        mask_co = offs_co_rel < COUT_PER_GROUP

        loss = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + co[None, :] * loss_stride_c
            + spatial_idx[:, None],
            mask=mask_m[:, None] & mask_co[None, :],
            other=0.0,
        )
        weight = tl.load(
            weight_ptr
            + co[:, None] * weight_stride_o
            + offs_ci_rel[None, :] * weight_stride_i,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        acc += tl.dot(
            loss,
            weight,
            out_dtype=tl.float32,
            input_precision="tf32",
        )

    tl.store(
        out_ptr
        + n_idx[:, None] * out_stride_n
        + ci[None, :] * out_stride_c
        + spatial_idx[:, None],
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_ci[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV_DGRAD_2D_1X1_STRIDED_CONFIGS,
    key=[
        "M",
        "XH",
        "XW",
        "CIN_PER_GROUP",
        "COUT_PER_GROUP",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def _conv_dgrad2d_1x1_strided_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    weight_stride_o: tl.constexpr,
    weight_stride_i: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    spatial = XH * XW
    n_idx = offs_m // spatial
    spatial_idx = offs_m - n_idx * spatial
    xh = spatial_idx // XW
    xw = spatial_idx - xh * XW
    ci = group * CIN_PER_GROUP + offs_ci_rel

    acc = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
        offs_co_rel = co_start + tl.arange(0, BLOCK_CO)
        co = group * COUT_PER_GROUP + offs_co_rel
        mask_co = offs_co_rel < COUT_PER_GROUP

        loss = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + co[None, :] * loss_stride_c
            + xh[:, None] * loss_stride_h
            + xw[:, None] * loss_stride_w,
            mask=mask_m[:, None] & mask_co[None, :],
            other=0.0,
        )
        weight = tl.load(
            weight_ptr
            + co[:, None] * weight_stride_o
            + offs_ci_rel[None, :] * weight_stride_i,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        acc += tl.dot(
            loss,
            weight,
            out_dtype=tl.float32,
            input_precision="tf32",
        )

    tl.store(
        out_ptr
        + n_idx[:, None] * out_stride_n
        + ci[None, :] * out_stride_c
        + xh[:, None] * out_stride_h
        + xw[:, None] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_ci[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV_DGRAD_2D_STRIDE1_CONFIGS,
    key=[
        "M",
        "XH",
        "XW",
        "LOSS_H",
        "LOSS_W",
        "CIN_PER_GROUP",
        "COUT_PER_GROUP",
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
def _conv_dgrad2d_stride1_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    weight_stride_o: tl.constexpr,
    weight_stride_i: tl.constexpr,
    weight_stride_h: tl.constexpr,
    weight_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    spatial = XH * XW
    n_idx = offs_m // spatial
    spatial_idx = offs_m - n_idx * spatial
    xh = spatial_idx // XW
    xw = spatial_idx - xh * XW
    ci = group * CIN_PER_GROUP + offs_ci_rel

    acc = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        loss_h = xh + PAD_TOP - kh * DIL_H
        valid_h = (loss_h >= 0) & (loss_h < LOSS_H)
        safe_h = tl.where(valid_h, loss_h, 0)
        weight_h = KH - 1 - kh if FILTER_REVERSE else kh

        for kw in tl.static_range(0, KW):
            loss_w = xw + PAD_LEFT - kw * DIL_W
            valid_w = (loss_w >= 0) & (loss_w < LOSS_W)
            valid_hw = valid_h & valid_w
            safe_w = tl.where(valid_w, loss_w, 0)
            weight_w = KW - 1 - kw if FILTER_REVERSE else kw

            for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
                offs_co_rel = co_start + tl.arange(0, BLOCK_CO)
                co = group * COUT_PER_GROUP + offs_co_rel
                mask_co = offs_co_rel < COUT_PER_GROUP

                loss = tl.load(
                    loss_ptr
                    + n_idx[:, None] * loss_stride_n
                    + co[None, :] * loss_stride_c
                    + safe_h[:, None] * loss_stride_h
                    + safe_w[:, None] * loss_stride_w,
                    mask=(
                        mask_m[:, None] & mask_co[None, :] & valid_hw[:, None]
                    ),
                    other=0.0,
                )
                weight = tl.load(
                    weight_ptr
                    + co[:, None] * weight_stride_o
                    + offs_ci_rel[None, :] * weight_stride_i
                    + weight_h * weight_stride_h
                    + weight_w * weight_stride_w,
                    mask=mask_co[:, None] & mask_ci[None, :],
                    other=0.0,
                )
                acc += tl.dot(loss, weight, out_dtype=tl.float32)

    tl.store(
        out_ptr
        + n_idx[:, None] * out_stride_n
        + ci[None, :] * out_stride_c
        + xh[:, None] * out_stride_h
        + xw[:, None] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_ci[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV_DGRAD_2D_STRIDE2_3X3_PACKED_CONFIGS,
    key=[
        "M",
        "XH",
        "XW",
        "CIN_PER_GROUP",
        "COUT_PER_GROUP",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def _conv_dgrad2d_stride2_pad1_3x3_packed_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    parity = tl.program_id(1)
    group = tl.program_id(2)
    ph = parity // 2
    pw = parity - ph * 2

    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    parity_h_count = (XH + 1 - ph) // 2
    parity_w_count = (XW + 1 - pw) // 2
    parity_spatial = parity_h_count * parity_w_count
    m_parity = N * parity_spatial

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < m_parity
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    n_idx = offs_m // parity_spatial
    spatial_idx = offs_m - n_idx * parity_spatial
    yh = spatial_idx // parity_w_count
    yw = spatial_idx - yh * parity_w_count
    xh = yh * 2 + ph
    xw = yw * 2 + pw
    ci = group * CIN_PER_GROUP + offs_ci_rel

    acc = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for kh_i in tl.static_range(0, 2):
        kh = tl.where(ph == 0, 1, kh_i * 2)
        loss_h = tl.where(ph == 0, yh, yh + (1 if kh_i == 0 else 0))
        valid_h = loss_h < LOSS_H
        if kh_i == 1:
            valid_h = valid_h & (ph != 0)
        weight_h = 2 - kh if FILTER_REVERSE else kh

        for kw_i in tl.static_range(0, 2):
            kw = tl.where(pw == 0, 1, kw_i * 2)
            loss_w = tl.where(pw == 0, yw, yw + (1 if kw_i == 0 else 0))
            valid_w = loss_w < LOSS_W
            if kw_i == 1:
                valid_w = valid_w & (pw != 0)
            valid_hw = valid_h & valid_w
            weight_w = 2 - kw if FILTER_REVERSE else kw

            for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
                offs_co_rel = co_start + tl.arange(0, BLOCK_CO)
                co = group * COUT_PER_GROUP + offs_co_rel
                mask_co = offs_co_rel < COUT_PER_GROUP

                loss = tl.load(
                    loss_ptr
                    + n_idx[:, None] * loss_stride_n
                    + co[None, :] * loss_stride_c
                    + loss_h[:, None] * loss_stride_h
                    + loss_w[:, None] * loss_stride_w,
                    mask=(
                        mask_m[:, None] & mask_co[None, :] & valid_hw[:, None]
                    ),
                    other=0.0,
                )
                weight = tl.load(
                    weight_ptr
                    + (
                        (
                            ((group * 3 + weight_h) * 3 + weight_w)
                            * COUT_PER_GROUP
                            + offs_co_rel[:, None]
                        )
                        * CIN_PER_GROUP
                    )
                    + offs_ci_rel[None, :],
                    mask=mask_co[:, None] & mask_ci[None, :],
                    other=0.0,
                )
                acc += tl.dot(
                    loss, weight, out_dtype=tl.float32, input_precision="tf32"
                )

    tl.store(
        out_ptr
        + n_idx[:, None] * out_stride_n
        + ci[None, :] * out_stride_c
        + xh[:, None] * out_stride_h
        + xw[:, None] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_ci[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV_DGRAD_2D_STRIDE2_3X3_PACKED_MCI_CONFIGS,
    key=[
        "M",
        "PARITY_H_COUNT",
        "PARITY_W_COUNT",
        "CIN_PER_GROUP",
        "COUT_PER_GROUP",
        "PH",
        "PW",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def _conv_dgrad2d_stride2_pad1_3x3_packed_mci_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    PARITY_H_COUNT: tl.constexpr,
    PARITY_W_COUNT: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    KH_COUNT: tl.constexpr,
    KW_COUNT: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    parity_spatial = PARITY_H_COUNT * PARITY_W_COUNT
    n_idx = offs_m // parity_spatial
    spatial_idx = offs_m - n_idx * parity_spatial
    yh = spatial_idx // PARITY_W_COUNT
    yw = spatial_idx - yh * PARITY_W_COUNT
    xh = yh * 2 + PH
    xw = yw * 2 + PW
    ci = offs_ci_rel

    acc = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for kh_i in tl.static_range(0, KH_COUNT):
        if PH == 0:
            kh = 1
            loss_h = yh
        else:
            kh = kh_i * 2
            loss_h = yh + (1 if kh_i == 0 else 0)
        valid_h = loss_h < LOSS_H
        weight_h = 2 - kh if FILTER_REVERSE else kh

        for kw_i in tl.static_range(0, KW_COUNT):
            if PW == 0:
                kw = 1
                loss_w = yw
            else:
                kw = kw_i * 2
                loss_w = yw + (1 if kw_i == 0 else 0)
            valid_w = loss_w < LOSS_W
            valid_hw = valid_h & valid_w
            weight_w = 2 - kw if FILTER_REVERSE else kw

            for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
                offs_co_rel = co_start + tl.arange(0, BLOCK_CO)
                mask_co = offs_co_rel < COUT_PER_GROUP

                loss = tl.load(
                    loss_ptr
                    + n_idx[:, None] * loss_stride_n
                    + offs_co_rel[None, :] * loss_stride_c
                    + loss_h[:, None] * loss_stride_h
                    + loss_w[:, None] * loss_stride_w,
                    mask=(
                        mask_m[:, None] & mask_co[None, :] & valid_hw[:, None]
                    ),
                    other=0.0,
                )
                weight = tl.load(
                    weight_ptr
                    + (
                        (
                            (weight_h * 3 + weight_w) * COUT_PER_GROUP
                            + offs_co_rel[:, None]
                        )
                        * CIN_PER_GROUP
                    )
                    + offs_ci_rel[None, :],
                    mask=mask_co[:, None] & mask_ci[None, :],
                    other=0.0,
                )
                acc += tl.dot(
                    loss, weight, out_dtype=tl.float32, input_precision="tf32"
                )

    tl.store(
        out_ptr
        + n_idx[:, None] * out_stride_n
        + ci[None, :] * out_stride_c
        + xh[:, None] * out_stride_h
        + xw[:, None] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_ci[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV_DGRAD_2D_STRIDE2_3X3_TILE2W_CONFIGS,
    key=[
        "M",
        "PARITY_H_COUNT",
        "LOSS_W",
        "CIN_PER_GROUP",
        "COUT_PER_GROUP",
        "PH",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def _conv_dgrad2d_stride2_pad1_3x3_tile2w_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    PARITY_H_COUNT: tl.constexpr,
    PH: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    parity_spatial = PARITY_H_COUNT * LOSS_W
    n_idx = offs_m // parity_spatial
    spatial_idx = offs_m - n_idx * parity_spatial
    yh = spatial_idx // LOSS_W
    yw = spatial_idx - yh * LOSS_W
    xh = yh * 2 + PH
    xw0 = yw * 2
    xw1 = xw0 + 1
    yh1 = yh + 1
    yw1 = yw + 1
    ci = offs_ci_rel

    valid0 = mask_m & (xh < XH) & (xw0 < XW)
    valid1 = mask_m & (xh < XH) & (xw1 < XW)
    valid_yh1 = yh1 < LOSS_H
    valid_yw1 = yw1 < LOSS_W

    if FILTER_REVERSE:
        w0 = 2
        w1 = 1
        w2 = 0
    else:
        w0 = 0
        w1 = 1
        w2 = 2

    acc0 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
        offs_co_rel = co_start + tl.arange(0, BLOCK_CO)
        mask_co = offs_co_rel < COUT_PER_GROUP

        loss00 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + offs_co_rel[None, :] * loss_stride_c
            + yh[:, None] * loss_stride_h
            + yw[:, None] * loss_stride_w,
            mask=mask_m[:, None] & mask_co[None, :],
            other=0.0,
        )
        loss01 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + offs_co_rel[None, :] * loss_stride_c
            + yh[:, None] * loss_stride_h
            + yw1[:, None] * loss_stride_w,
            mask=mask_m[:, None] & mask_co[None, :] & valid_yw1[:, None],
            other=0.0,
        )

        if PH == 0:
            weight11 = tl.load(
                weight_ptr
                + (
                    ((w1 * 3 + w1) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight12 = tl.load(
                weight_ptr
                + (
                    ((w1 * 3 + w2) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight10 = tl.load(
                weight_ptr
                + (
                    ((w1 * 3 + w0) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            acc0 += tl.dot(
                loss00, weight11, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss00, weight12, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss01, weight10, out_dtype=tl.float32, input_precision="tf32"
            )
        else:
            loss10 = tl.load(
                loss_ptr
                + n_idx[:, None] * loss_stride_n
                + offs_co_rel[None, :] * loss_stride_c
                + yh1[:, None] * loss_stride_h
                + yw[:, None] * loss_stride_w,
                mask=mask_m[:, None] & mask_co[None, :] & valid_yh1[:, None],
                other=0.0,
            )
            loss11 = tl.load(
                loss_ptr
                + n_idx[:, None] * loss_stride_n
                + offs_co_rel[None, :] * loss_stride_c
                + yh1[:, None] * loss_stride_h
                + yw1[:, None] * loss_stride_w,
                mask=mask_m[:, None]
                & mask_co[None, :]
                & valid_yh1[:, None]
                & valid_yw1[:, None],
                other=0.0,
            )
            weight21 = tl.load(
                weight_ptr
                + (
                    ((w2 * 3 + w1) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight01 = tl.load(
                weight_ptr
                + (
                    ((w0 * 3 + w1) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight22 = tl.load(
                weight_ptr
                + (
                    ((w2 * 3 + w2) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight20 = tl.load(
                weight_ptr
                + (
                    ((w2 * 3 + w0) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight02 = tl.load(
                weight_ptr
                + (
                    ((w0 * 3 + w2) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight00 = tl.load(
                weight_ptr
                + (
                    ((w0 * 3 + w0) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            acc0 += tl.dot(
                loss00, weight21, out_dtype=tl.float32, input_precision="tf32"
            )
            acc0 += tl.dot(
                loss10, weight01, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss00, weight22, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss01, weight20, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss10, weight02, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss11, weight00, out_dtype=tl.float32, input_precision="tf32"
            )

    out_base = (
        out_ptr + n_idx[:, None] * out_stride_n + ci[None, :] * out_stride_c
    )
    tl.store(
        out_base + xh[:, None] * out_stride_h + xw0[:, None] * out_stride_w,
        acc0.to(out_ptr.dtype.element_ty),
        mask=valid0[:, None] & mask_ci[None, :],
    )
    tl.store(
        out_base + xh[:, None] * out_stride_h + xw1[:, None] * out_stride_w,
        acc1.to(out_ptr.dtype.element_ty),
        mask=valid1[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_dgrad2d_stride2_pad1_3x3_tile2w_splitk_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    PARITY_H_COUNT: tl.constexpr,
    PH: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    GROUP_K: tl.constexpr,
    SPLIT_K_BLOCKS: tl.constexpr,
    K_OFFSET: tl.constexpr,
    STORE: tl.constexpr,
):
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid = tl.program_id(0)
    pid_k_rel = pid % SPLIT_K_BLOCKS
    pid_k_group = pid_k_rel + K_OFFSET
    pid_tmp = pid // SPLIT_K_BLOCKS
    pid_m = pid_tmp % num_m_blocks
    pid_ci = pid_tmp // num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    parity_spatial = PARITY_H_COUNT * LOSS_W
    n_idx = offs_m // parity_spatial
    spatial_idx = offs_m - n_idx * parity_spatial
    yh = spatial_idx // LOSS_W
    yw = spatial_idx - yh * LOSS_W
    xh = yh * 2 + PH
    xw0 = yw * 2
    xw1 = xw0 + 1
    yh1 = yh + 1
    yw1 = yw + 1
    ci = offs_ci_rel

    valid0 = mask_m & (xh < XH) & (xw0 < XW)
    valid1 = mask_m & (xh < XH) & (xw1 < XW)
    valid_yh1 = yh1 < LOSS_H
    valid_yw1 = yw1 < LOSS_W

    if FILTER_REVERSE:
        w0 = 2
        w1 = 1
        w2 = 0
    else:
        w0 = 0
        w1 = 1
        w2 = 2

    acc0 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for k_inner in tl.static_range(0, GROUP_K):
        pid_k = pid_k_group * GROUP_K + k_inner
        offs_co_rel = pid_k * BLOCK_CO + tl.arange(0, BLOCK_CO)
        mask_co = offs_co_rel < COUT_PER_GROUP

        loss00 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + offs_co_rel[None, :] * loss_stride_c
            + yh[:, None] * loss_stride_h
            + yw[:, None] * loss_stride_w,
            mask=mask_m[:, None] & mask_co[None, :],
            other=0.0,
        )
        loss01 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + offs_co_rel[None, :] * loss_stride_c
            + yh[:, None] * loss_stride_h
            + yw1[:, None] * loss_stride_w,
            mask=mask_m[:, None] & mask_co[None, :] & valid_yw1[:, None],
            other=0.0,
        )

        if PH == 0:
            weight11 = tl.load(
                weight_ptr
                + (
                    ((w1 * 3 + w1) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight12 = tl.load(
                weight_ptr
                + (
                    ((w1 * 3 + w2) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight10 = tl.load(
                weight_ptr
                + (
                    ((w1 * 3 + w0) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            acc0 += tl.dot(
                loss00, weight11, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss00, weight12, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss01, weight10, out_dtype=tl.float32, input_precision="tf32"
            )
        else:
            loss10 = tl.load(
                loss_ptr
                + n_idx[:, None] * loss_stride_n
                + offs_co_rel[None, :] * loss_stride_c
                + yh1[:, None] * loss_stride_h
                + yw[:, None] * loss_stride_w,
                mask=mask_m[:, None] & mask_co[None, :] & valid_yh1[:, None],
                other=0.0,
            )
            loss11 = tl.load(
                loss_ptr
                + n_idx[:, None] * loss_stride_n
                + offs_co_rel[None, :] * loss_stride_c
                + yh1[:, None] * loss_stride_h
                + yw1[:, None] * loss_stride_w,
                mask=(
                    mask_m[:, None]
                    & mask_co[None, :]
                    & valid_yh1[:, None]
                    & valid_yw1[:, None]
                ),
                other=0.0,
            )
            weight21 = tl.load(
                weight_ptr
                + (
                    ((w2 * 3 + w1) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight01 = tl.load(
                weight_ptr
                + (
                    ((w0 * 3 + w1) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight22 = tl.load(
                weight_ptr
                + (
                    ((w2 * 3 + w2) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight20 = tl.load(
                weight_ptr
                + (
                    ((w2 * 3 + w0) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight02 = tl.load(
                weight_ptr
                + (
                    ((w0 * 3 + w2) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            weight00 = tl.load(
                weight_ptr
                + (
                    ((w0 * 3 + w0) * COUT_PER_GROUP + offs_co_rel[:, None])
                    * CIN_PER_GROUP
                )
                + offs_ci_rel[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            acc0 += tl.dot(
                loss00, weight21, out_dtype=tl.float32, input_precision="tf32"
            )
            acc0 += tl.dot(
                loss10, weight01, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss00, weight22, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss01, weight20, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss10, weight02, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss11, weight00, out_dtype=tl.float32, input_precision="tf32"
            )

    out_base = (
        out_ptr + n_idx[:, None] * out_stride_n + ci[None, :] * out_stride_c
    )
    ptr0 = out_base + xh[:, None] * out_stride_h + xw0[:, None] * out_stride_w
    ptr1 = out_base + xh[:, None] * out_stride_h + xw1[:, None] * out_stride_w
    mask = mask_m[:, None] & mask_ci[None, :]
    if STORE:
        tl.store(ptr0, acc0, mask=valid0[:, None] & mask_ci[None, :])
        tl.store(ptr1, acc1, mask=valid1[:, None] & mask_ci[None, :])
    else:
        tl.atomic_add(ptr0, acc0, sem="relaxed", mask=mask)
        tl.atomic_add(ptr1, acc1, sem="relaxed", mask=mask)


@triton.jit
def _conv_dgrad2d_stride2_pad1_3x3_p5_tile2w_splitk_kernel(
    loss,
    weight,
    out,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    PARITY_H_COUNT: tl.constexpr,
    PH: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    GROUP_K: tl.constexpr,
    SPLIT_K_BLOCKS: tl.constexpr,
    K_OFFSET: tl.constexpr,
    STORE: tl.constexpr,
):
    num_m_blocks = tl.cdiv(400, BLOCK_M)
    pid = tl.program_id(0)
    pid_m = pid % num_m_blocks
    pid_tmp = pid // num_m_blocks
    pid_k_rel = pid_tmp % SPLIT_K_BLOCKS
    pid_k_group = pid_k_rel + K_OFFSET
    pid_ci = pid_tmp // SPLIT_K_BLOCKS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < 400
    mask_ci = offs_ci < 768

    yh = offs_m // 20
    yw = offs_m - yh * 20
    x_base = (yh * 2 + PH) * 40 + yw * 2
    loss_base = yh * 20 + yw
    valid_yh1 = yh < 19
    valid_yw1 = yw < 19

    acc0 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for k_inner in tl.static_range(0, GROUP_K):
        pid_k = pid_k_group * GROUP_K + k_inner
        offs_co = pid_k * BLOCK_CO + tl.arange(0, BLOCK_CO)
        mask_co = offs_co < 768
        loss00 = tl.load(
            loss + offs_co[None, :] * 400 + loss_base[:, None],
            mask=mask_m[:, None] & mask_co[None, :],
            other=0.0,
        )
        loss01 = tl.load(
            loss + offs_co[None, :] * 400 + (loss_base + 1)[:, None],
            mask=mask_m[:, None] & mask_co[None, :] & valid_yw1[:, None],
            other=0.0,
        )
        if PH == 0:
            w11 = tl.load(
                weight
                + ((1 * 3 + 1) * 768 + offs_co[:, None]) * 768
                + offs_ci[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            w12 = tl.load(
                weight
                + ((1 * 3 + 2) * 768 + offs_co[:, None]) * 768
                + offs_ci[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            w10 = tl.load(
                weight
                + ((1 * 3 + 0) * 768 + offs_co[:, None]) * 768
                + offs_ci[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            acc0 += tl.dot(
                loss00, w11, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss00, w12, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss01, w10, out_dtype=tl.float32, input_precision="tf32"
            )
        else:
            loss10 = tl.load(
                loss + offs_co[None, :] * 400 + (loss_base + 20)[:, None],
                mask=mask_m[:, None] & mask_co[None, :] & valid_yh1[:, None],
                other=0.0,
            )
            loss11 = tl.load(
                loss + offs_co[None, :] * 400 + (loss_base + 21)[:, None],
                mask=mask_m[:, None]
                & mask_co[None, :]
                & valid_yh1[:, None]
                & valid_yw1[:, None],
                other=0.0,
            )
            w21 = tl.load(
                weight
                + ((2 * 3 + 1) * 768 + offs_co[:, None]) * 768
                + offs_ci[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            w01 = tl.load(
                weight
                + ((0 * 3 + 1) * 768 + offs_co[:, None]) * 768
                + offs_ci[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            w22 = tl.load(
                weight
                + ((2 * 3 + 2) * 768 + offs_co[:, None]) * 768
                + offs_ci[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            w20 = tl.load(
                weight
                + ((2 * 3 + 0) * 768 + offs_co[:, None]) * 768
                + offs_ci[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            w02 = tl.load(
                weight
                + ((0 * 3 + 2) * 768 + offs_co[:, None]) * 768
                + offs_ci[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            w00 = tl.load(
                weight
                + ((0 * 3 + 0) * 768 + offs_co[:, None]) * 768
                + offs_ci[None, :],
                mask=mask_co[:, None] & mask_ci[None, :],
                other=0.0,
            )
            acc0 += tl.dot(
                loss00, w21, out_dtype=tl.float32, input_precision="tf32"
            )
            acc0 += tl.dot(
                loss10, w01, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss00, w22, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss01, w20, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss10, w02, out_dtype=tl.float32, input_precision="tf32"
            )
            acc1 += tl.dot(
                loss11, w00, out_dtype=tl.float32, input_precision="tf32"
            )

    ptr0 = out + offs_ci[None, :] * 1600 + x_base[:, None]
    ptr1 = ptr0 + 1
    mask = mask_m[:, None] & mask_ci[None, :]
    if STORE:
        tl.store(ptr0, acc0, mask=mask)
        tl.store(ptr1, acc1, mask=mask)
    else:
        tl.atomic_add(ptr0, acc0, sem="relaxed", mask=mask)
        tl.atomic_add(ptr1, acc1, sem="relaxed", mask=mask)


@triton.jit
def _conv_dgrad2d_p5_zero_kernel(
    out,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(
        out + offs, tl.zeros((BLOCK,), dtype=tl.float32), mask=offs < TOTAL
    )


@triton.jit
def _conv_dgrad2d_stride2_pad1_3x3_tile4_splitk_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    weight_stride_o: tl.constexpr,
    weight_stride_i: tl.constexpr,
    weight_stride_h: tl.constexpr,
    weight_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    GROUP_K: tl.constexpr,
    SPLIT_K_BLOCKS: tl.constexpr,
    K_OFFSET: tl.constexpr,
    STORE: tl.constexpr,
):
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid = tl.program_id(0)
    pid_k_rel = pid % SPLIT_K_BLOCKS
    pid_k_group = pid_k_rel + K_OFFSET
    pid_tmp = pid // SPLIT_K_BLOCKS
    pid_m = pid_tmp % num_m_blocks
    pid_ci = pid_tmp // num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    n_idx = offs_m // (LOSS_H * LOSS_W)
    spatial = offs_m - n_idx * (LOSS_H * LOSS_W)
    yh = spatial // LOSS_W
    yw = spatial - yh * LOSS_W
    xh0 = yh * 2
    xw0 = yw * 2
    xh1 = xh0 + 1
    xw1 = xw0 + 1
    yh1 = yh + 1
    yw1 = yw + 1
    ci = offs_ci_rel
    valid_yh1 = yh1 < LOSS_H
    valid_yw1 = yw1 < LOSS_W

    if FILTER_REVERSE:
        w0 = 2
        w1 = 1
        w2 = 0
    else:
        w0 = 0
        w1 = 1
        w2 = 2

    acc00 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for k_inner in tl.static_range(0, GROUP_K):
        pid_k = pid_k_group * GROUP_K + k_inner
        offs_co_rel = pid_k * BLOCK_CO + tl.arange(0, BLOCK_CO)
        mask_co = offs_co_rel < COUT_PER_GROUP

        loss00 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + offs_co_rel[None, :] * loss_stride_c
            + yh[:, None] * loss_stride_h
            + yw[:, None] * loss_stride_w,
            mask=mask_m[:, None] & mask_co[None, :],
            other=0.0,
        )
        loss01 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + offs_co_rel[None, :] * loss_stride_c
            + yh[:, None] * loss_stride_h
            + yw1[:, None] * loss_stride_w,
            mask=mask_m[:, None] & mask_co[None, :] & valid_yw1[:, None],
            other=0.0,
        )
        loss10 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + offs_co_rel[None, :] * loss_stride_c
            + yh1[:, None] * loss_stride_h
            + yw[:, None] * loss_stride_w,
            mask=mask_m[:, None] & mask_co[None, :] & valid_yh1[:, None],
            other=0.0,
        )
        loss11 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + offs_co_rel[None, :] * loss_stride_c
            + yh1[:, None] * loss_stride_h
            + yw1[:, None] * loss_stride_w,
            mask=mask_m[:, None]
            & mask_co[None, :]
            & valid_yh1[:, None]
            & valid_yw1[:, None],
            other=0.0,
        )

        weight11 = tl.load(
            weight_ptr
            + offs_co_rel[:, None] * weight_stride_o
            + ci[None, :] * weight_stride_i
            + w1 * weight_stride_h
            + w1 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight12 = tl.load(
            weight_ptr
            + offs_co_rel[:, None] * weight_stride_o
            + ci[None, :] * weight_stride_i
            + w1 * weight_stride_h
            + w2 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight10 = tl.load(
            weight_ptr
            + offs_co_rel[:, None] * weight_stride_o
            + ci[None, :] * weight_stride_i
            + w1 * weight_stride_h
            + w0 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight21 = tl.load(
            weight_ptr
            + offs_co_rel[:, None] * weight_stride_o
            + ci[None, :] * weight_stride_i
            + w2 * weight_stride_h
            + w1 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight01 = tl.load(
            weight_ptr
            + offs_co_rel[:, None] * weight_stride_o
            + ci[None, :] * weight_stride_i
            + w0 * weight_stride_h
            + w1 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight22 = tl.load(
            weight_ptr
            + offs_co_rel[:, None] * weight_stride_o
            + ci[None, :] * weight_stride_i
            + w2 * weight_stride_h
            + w2 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight20 = tl.load(
            weight_ptr
            + offs_co_rel[:, None] * weight_stride_o
            + ci[None, :] * weight_stride_i
            + w2 * weight_stride_h
            + w0 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight02 = tl.load(
            weight_ptr
            + offs_co_rel[:, None] * weight_stride_o
            + ci[None, :] * weight_stride_i
            + w0 * weight_stride_h
            + w2 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight00 = tl.load(
            weight_ptr
            + offs_co_rel[:, None] * weight_stride_o
            + ci[None, :] * weight_stride_i
            + w0 * weight_stride_h
            + w0 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )

        acc00 += tl.dot(
            loss00, weight11, out_dtype=tl.float32, input_precision="tf32"
        )
        acc01 += tl.dot(
            loss00, weight12, out_dtype=tl.float32, input_precision="tf32"
        )
        acc01 += tl.dot(
            loss01, weight10, out_dtype=tl.float32, input_precision="tf32"
        )
        acc10 += tl.dot(
            loss00, weight21, out_dtype=tl.float32, input_precision="tf32"
        )
        acc10 += tl.dot(
            loss10, weight01, out_dtype=tl.float32, input_precision="tf32"
        )
        acc11 += tl.dot(
            loss00, weight22, out_dtype=tl.float32, input_precision="tf32"
        )
        acc11 += tl.dot(
            loss01, weight20, out_dtype=tl.float32, input_precision="tf32"
        )
        acc11 += tl.dot(
            loss10, weight02, out_dtype=tl.float32, input_precision="tf32"
        )
        acc11 += tl.dot(
            loss11, weight00, out_dtype=tl.float32, input_precision="tf32"
        )

    out_base = (
        out_ptr + n_idx[:, None] * out_stride_n + ci[None, :] * out_stride_c
    )
    ptr00 = (
        out_base + xh0[:, None] * out_stride_h + xw0[:, None] * out_stride_w
    )
    ptr01 = (
        out_base + xh0[:, None] * out_stride_h + xw1[:, None] * out_stride_w
    )
    ptr10 = (
        out_base + xh1[:, None] * out_stride_h + xw0[:, None] * out_stride_w
    )
    ptr11 = (
        out_base + xh1[:, None] * out_stride_h + xw1[:, None] * out_stride_w
    )
    mask = mask_m[:, None] & mask_ci[None, :]
    if STORE:
        tl.store(ptr00, acc00, mask=mask)
        tl.store(ptr01, acc01, mask=mask)
        tl.store(ptr10, acc10, mask=mask)
        tl.store(ptr11, acc11, mask=mask)
    else:
        tl.atomic_add(ptr00, acc00, sem="relaxed", mask=mask)
        tl.atomic_add(ptr01, acc01, sem="relaxed", mask=mask)
        tl.atomic_add(ptr10, acc10, sem="relaxed", mask=mask)
        tl.atomic_add(ptr11, acc11, sem="relaxed", mask=mask)


@libentry()
@libtuner(
    configs=_CONV_DGRAD_2D_STRIDE2_3X3_TILE4_CONFIGS,
    key=[
        "M",
        "XH",
        "XW",
        "LOSS_H",
        "LOSS_W",
        "CIN_PER_GROUP",
        "COUT_PER_GROUP",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def _conv_dgrad2d_stride2_pad1_3x3_tile4_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    weight_stride_o: tl.constexpr,
    weight_stride_i: tl.constexpr,
    weight_stride_h: tl.constexpr,
    weight_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    offs_bco = tl.arange(0, BLOCK_CO)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    loss_spatial = LOSS_H * LOSS_W
    n_idx = offs_m // loss_spatial
    spatial_idx = offs_m - n_idx * loss_spatial
    yh = spatial_idx // LOSS_W
    yw = spatial_idx - yh * LOSS_W
    xh0 = yh * 2
    xw0 = yw * 2
    xh1 = xh0 + 1
    xw1 = xw0 + 1
    yh1 = yh + 1
    yw1 = yw + 1

    valid00 = mask_m & (xh0 < XH) & (xw0 < XW)
    valid01 = mask_m & (xh0 < XH) & (xw1 < XW)
    valid10 = mask_m & (xh1 < XH) & (xw0 < XW)
    valid11 = mask_m & (xh1 < XH) & (xw1 < XW)
    valid_yh1 = yh1 < LOSS_H
    valid_yw1 = yw1 < LOSS_W

    ci = group * CIN_PER_GROUP + offs_ci_rel
    if FILTER_REVERSE:
        w0 = 2
        w1 = 1
        w2 = 0
    else:
        w0 = 0
        w1 = 1
        w2 = 2

    acc00 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
        offs_co_rel = co_start + offs_bco
        co = group * COUT_PER_GROUP + offs_co_rel
        mask_co = offs_co_rel < COUT_PER_GROUP

        loss00 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + co[None, :] * loss_stride_c
            + yh[:, None] * loss_stride_h
            + yw[:, None] * loss_stride_w,
            mask=mask_m[:, None] & mask_co[None, :],
            other=0.0,
        )
        loss01 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + co[None, :] * loss_stride_c
            + yh[:, None] * loss_stride_h
            + yw1[:, None] * loss_stride_w,
            mask=(mask_m[:, None] & mask_co[None, :] & valid_yw1[:, None]),
            other=0.0,
        )
        loss10 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + co[None, :] * loss_stride_c
            + yh1[:, None] * loss_stride_h
            + yw[:, None] * loss_stride_w,
            mask=(mask_m[:, None] & mask_co[None, :] & valid_yh1[:, None]),
            other=0.0,
        )
        loss11 = tl.load(
            loss_ptr
            + n_idx[:, None] * loss_stride_n
            + co[None, :] * loss_stride_c
            + yh1[:, None] * loss_stride_h
            + yw1[:, None] * loss_stride_w,
            mask=(
                mask_m[:, None]
                & mask_co[None, :]
                & valid_yh1[:, None]
                & valid_yw1[:, None]
            ),
            other=0.0,
        )

        weight11 = tl.load(
            weight_ptr
            + co[:, None] * weight_stride_o
            + offs_ci_rel[None, :] * weight_stride_i
            + w1 * weight_stride_h
            + w1 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight12 = tl.load(
            weight_ptr
            + co[:, None] * weight_stride_o
            + offs_ci_rel[None, :] * weight_stride_i
            + w1 * weight_stride_h
            + w2 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight10 = tl.load(
            weight_ptr
            + co[:, None] * weight_stride_o
            + offs_ci_rel[None, :] * weight_stride_i
            + w1 * weight_stride_h
            + w0 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight21 = tl.load(
            weight_ptr
            + co[:, None] * weight_stride_o
            + offs_ci_rel[None, :] * weight_stride_i
            + w2 * weight_stride_h
            + w1 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight01 = tl.load(
            weight_ptr
            + co[:, None] * weight_stride_o
            + offs_ci_rel[None, :] * weight_stride_i
            + w0 * weight_stride_h
            + w1 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight22 = tl.load(
            weight_ptr
            + co[:, None] * weight_stride_o
            + offs_ci_rel[None, :] * weight_stride_i
            + w2 * weight_stride_h
            + w2 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight20 = tl.load(
            weight_ptr
            + co[:, None] * weight_stride_o
            + offs_ci_rel[None, :] * weight_stride_i
            + w2 * weight_stride_h
            + w0 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight02 = tl.load(
            weight_ptr
            + co[:, None] * weight_stride_o
            + offs_ci_rel[None, :] * weight_stride_i
            + w0 * weight_stride_h
            + w2 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )
        weight00 = tl.load(
            weight_ptr
            + co[:, None] * weight_stride_o
            + offs_ci_rel[None, :] * weight_stride_i
            + w0 * weight_stride_h
            + w0 * weight_stride_w,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )

        acc00 += tl.dot(
            loss00, weight11, out_dtype=tl.float32, input_precision="tf32"
        )
        acc01 += tl.dot(
            loss00, weight12, out_dtype=tl.float32, input_precision="tf32"
        )
        acc01 += tl.dot(
            loss01, weight10, out_dtype=tl.float32, input_precision="tf32"
        )
        acc10 += tl.dot(
            loss00, weight21, out_dtype=tl.float32, input_precision="tf32"
        )
        acc10 += tl.dot(
            loss10, weight01, out_dtype=tl.float32, input_precision="tf32"
        )
        acc11 += tl.dot(
            loss00, weight22, out_dtype=tl.float32, input_precision="tf32"
        )
        acc11 += tl.dot(
            loss01, weight20, out_dtype=tl.float32, input_precision="tf32"
        )
        acc11 += tl.dot(
            loss10, weight02, out_dtype=tl.float32, input_precision="tf32"
        )
        acc11 += tl.dot(
            loss11, weight00, out_dtype=tl.float32, input_precision="tf32"
        )

    out_base = (
        out_ptr + n_idx[:, None] * out_stride_n + ci[None, :] * out_stride_c
    )
    tl.store(
        out_base + xh0[:, None] * out_stride_h + xw0[:, None] * out_stride_w,
        acc00.to(out_ptr.dtype.element_ty),
        mask=valid00[:, None] & mask_ci[None, :],
    )
    tl.store(
        out_base + xh0[:, None] * out_stride_h + xw1[:, None] * out_stride_w,
        acc01.to(out_ptr.dtype.element_ty),
        mask=valid01[:, None] & mask_ci[None, :],
    )
    tl.store(
        out_base + xh1[:, None] * out_stride_h + xw0[:, None] * out_stride_w,
        acc10.to(out_ptr.dtype.element_ty),
        mask=valid10[:, None] & mask_ci[None, :],
    )
    tl.store(
        out_base + xh1[:, None] * out_stride_h + xw1[:, None] * out_stride_w,
        acc11.to(out_ptr.dtype.element_ty),
        mask=valid11[:, None] & mask_ci[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV_DGRAD_2D_STRIDE2_3X3_MCI_CONFIGS,
    key=[
        "M",
        "XH",
        "XW",
        "CIN_PER_GROUP",
        "COUT_PER_GROUP",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def _conv_dgrad2d_stride2_pad1_3x3_merged_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    weight_stride_o: tl.constexpr,
    weight_stride_i: tl.constexpr,
    weight_stride_h: tl.constexpr,
    weight_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    parity = tl.program_id(1)
    ph = parity // 2
    pw = parity - ph * 2

    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    parity_h_count = (XH + 1 - ph) // 2
    parity_w_count = (XW + 1 - pw) // 2
    parity_spatial = parity_h_count * parity_w_count
    m_parity = N * parity_spatial

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < m_parity
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    n_idx = offs_m // parity_spatial
    spatial_idx = offs_m - n_idx * parity_spatial
    yh = spatial_idx // parity_w_count
    yw = spatial_idx - yh * parity_w_count
    xh = yh * 2 + ph
    xw = yw * 2 + pw
    ci = offs_ci_rel

    acc = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for kh_i in tl.static_range(0, 2):
        kh = tl.where(ph == 0, 1, kh_i * 2)
        loss_h = tl.where(ph == 0, yh, yh + (1 if kh_i == 0 else 0))
        valid_h = loss_h < LOSS_H
        if kh_i == 1:
            valid_h = valid_h & (ph != 0)
        weight_h = 2 - kh if FILTER_REVERSE else kh

        for kw_i in tl.static_range(0, 2):
            kw = tl.where(pw == 0, 1, kw_i * 2)
            loss_w = tl.where(pw == 0, yw, yw + (1 if kw_i == 0 else 0))
            valid_w = loss_w < LOSS_W
            if kw_i == 1:
                valid_w = valid_w & (pw != 0)
            valid_hw = valid_h & valid_w
            weight_w = 2 - kw if FILTER_REVERSE else kw

            for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
                offs_co_rel = co_start + tl.arange(0, BLOCK_CO)
                mask_co = offs_co_rel < COUT_PER_GROUP

                loss = tl.load(
                    loss_ptr
                    + n_idx[:, None] * loss_stride_n
                    + offs_co_rel[None, :] * loss_stride_c
                    + loss_h[:, None] * loss_stride_h
                    + loss_w[:, None] * loss_stride_w,
                    mask=(
                        mask_m[:, None] & mask_co[None, :] & valid_hw[:, None]
                    ),
                    other=0.0,
                )
                weight = tl.load(
                    weight_ptr
                    + offs_co_rel[:, None] * weight_stride_o
                    + offs_ci_rel[None, :] * weight_stride_i
                    + weight_h * weight_stride_h
                    + weight_w * weight_stride_w,
                    mask=mask_co[:, None] & mask_ci[None, :],
                    other=0.0,
                )
                acc += tl.dot(loss, weight, out_dtype=tl.float32)

    tl.store(
        out_ptr
        + n_idx[:, None] * out_stride_n
        + ci[None, :] * out_stride_c
        + xh[:, None] * out_stride_h
        + xw[:, None] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_ci[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV_DGRAD_2D_STRIDE2_3X3_MCI_CONFIGS,
    key=[
        "M",
        "PARITY_H_COUNT",
        "PARITY_W_COUNT",
        "CIN_PER_GROUP",
        "COUT_PER_GROUP",
        "PH",
        "PW",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def _conv_dgrad2d_stride2_pad1_3x3_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    weight_stride_o: tl.constexpr,
    weight_stride_i: tl.constexpr,
    weight_stride_h: tl.constexpr,
    weight_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    PARITY_H_COUNT: tl.constexpr,
    PARITY_W_COUNT: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    KH_COUNT: tl.constexpr,
    KW_COUNT: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    parity_spatial = PARITY_H_COUNT * PARITY_W_COUNT
    n_idx = offs_m // parity_spatial
    spatial_idx = offs_m - n_idx * parity_spatial
    yh = spatial_idx // PARITY_W_COUNT
    yw = spatial_idx - yh * PARITY_W_COUNT
    xh = yh * 2 + PH
    xw = yw * 2 + PW
    ci = offs_ci_rel

    acc = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for kh_i in tl.static_range(0, KH_COUNT):
        if PH == 0:
            kh = 1
            loss_h = yh
        else:
            kh = kh_i * 2
            loss_h = yh + (1 if kh_i == 0 else 0)
        valid_h = loss_h < LOSS_H
        weight_h = 2 - kh if FILTER_REVERSE else kh

        for kw_i in tl.static_range(0, KW_COUNT):
            if PW == 0:
                kw = 1
                loss_w = yw
            else:
                kw = kw_i * 2
                loss_w = yw + (1 if kw_i == 0 else 0)
            valid_w = loss_w < LOSS_W
            valid_hw = valid_h & valid_w
            weight_w = 2 - kw if FILTER_REVERSE else kw

            for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
                offs_co_rel = co_start + tl.arange(0, BLOCK_CO)
                mask_co = offs_co_rel < COUT_PER_GROUP

                loss = tl.load(
                    loss_ptr
                    + n_idx[:, None] * loss_stride_n
                    + offs_co_rel[None, :] * loss_stride_c
                    + loss_h[:, None] * loss_stride_h
                    + loss_w[:, None] * loss_stride_w,
                    mask=(
                        mask_m[:, None] & mask_co[None, :] & valid_hw[:, None]
                    ),
                    other=0.0,
                )
                weight = tl.load(
                    weight_ptr
                    + offs_co_rel[:, None] * weight_stride_o
                    + offs_ci_rel[None, :] * weight_stride_i
                    + weight_h * weight_stride_h
                    + weight_w * weight_stride_w,
                    mask=mask_co[:, None] & mask_ci[None, :],
                    other=0.0,
                )
                acc += tl.dot(loss, weight, out_dtype=tl.float32)

    tl.store(
        out_ptr
        + n_idx[:, None] * out_stride_n
        + ci[None, :] * out_stride_c
        + xh[:, None] * out_stride_h
        + xw[:, None] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_dgrad3d_pad1_3x3_fp32_split_dot_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M_PART: tl.constexpr,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_D: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_d: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    weight_stride_o: tl.constexpr,
    weight_stride_i: tl.constexpr,
    weight_stride_d: tl.constexpr,
    weight_stride_h: tl.constexpr,
    weight_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_d: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    PART: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci = tl.arange(0, 8)
    offs_co = tl.arange(0, 16)
    mask_m = offs_m < M_PART

    int_d = XD - 2
    int_h = XH - 2
    int_w = XW - 2
    if PART == 0:
        part_spatial = int_d * int_h * int_w
        n_idx = offs_m // part_spatial
        part_idx = offs_m - n_idx * part_spatial
        xd = part_idx // (int_h * int_w) + 1
        rem = part_idx - (xd - 1) * (int_h * int_w)
        xh = rem // int_w + 1
        xw = rem - (xh - 1) * int_w + 1
    else:
        d_faces = 2 * XH * XW
        h_faces = int_d * 2 * XW
        part_spatial = d_faces + h_faces + int_d * int_h * 2
        n_idx = offs_m // part_spatial
        part_idx = offs_m - n_idx * part_spatial

        in_d = part_idx < d_faces
        in_h = (part_idx >= d_faces) & (part_idx < d_faces + h_faces)

        d_side = part_idx // (XH * XW)
        d_rem = part_idx - d_side * (XH * XW)
        xd_d = tl.where(d_side == 0, 0, XD - 1)
        xh_d = d_rem // XW
        xw_d = d_rem - xh_d * XW

        h_idx = part_idx - d_faces
        h_d = h_idx // (2 * XW)
        h_rem = h_idx - h_d * (2 * XW)
        h_side = h_rem // XW
        xd_h = h_d + 1
        xh_h = tl.where(h_side == 0, 0, XH - 1)
        xw_h = h_rem - h_side * XW

        w_idx = part_idx - d_faces - h_faces
        w_pair = w_idx // 2
        w_side = w_idx - w_pair * 2
        xd_w = w_pair // int_h + 1
        xh_w = w_pair - (xd_w - 1) * int_h + 1
        xw_w = tl.where(w_side == 0, 0, XW - 1)

        xd = tl.where(in_d, xd_d, tl.where(in_h, xd_h, xd_w))
        xh = tl.where(in_d, xh_d, tl.where(in_h, xh_h, xh_w))
        xw = tl.where(in_d, xw_d, tl.where(in_h, xw_h, xw_w))

    acc = tl.zeros((BLOCK_M, 8), dtype=tl.float32)

    for kd in tl.static_range(0, 3):
        ld = xd + 1 - kd
        if PART == 0:
            valid_d = tl.full((BLOCK_M,), True, dtype=tl.int1)
            safe_d = ld
        else:
            valid_d = (ld >= 0) & (ld < LOSS_D)
            safe_d = tl.where(valid_d, ld, 0)
        for kh in tl.static_range(0, 3):
            lh = xh + 1 - kh
            if PART == 0:
                valid_h = tl.full((BLOCK_M,), True, dtype=tl.int1)
                safe_h = lh
            else:
                valid_h = (lh >= 0) & (lh < LOSS_H)
                safe_h = tl.where(valid_h, lh, 0)
            for kw in tl.static_range(0, 3):
                lw = xw + 1 - kw
                if PART == 0:
                    valid_dhw = tl.full((BLOCK_M,), True, dtype=tl.int1)
                    safe_w = lw
                else:
                    valid_w = (lw >= 0) & (lw < LOSS_W)
                    safe_w = tl.where(valid_w, lw, 0)
                    valid_dhw = valid_d & valid_h & valid_w

                loss = tl.load(
                    loss_ptr
                    + n_idx[:, None] * loss_stride_n
                    + offs_co[None, :] * loss_stride_c
                    + safe_d[:, None] * loss_stride_d
                    + safe_h[:, None] * loss_stride_h
                    + safe_w[:, None] * loss_stride_w,
                    mask=mask_m[:, None] & valid_dhw[:, None],
                    other=0.0,
                )
                weight = tl.load(
                    weight_ptr
                    + offs_co[:, None] * weight_stride_o
                    + offs_ci[None, :] * weight_stride_i
                    + kd * weight_stride_d
                    + kh * weight_stride_h
                    + kw * weight_stride_w,
                )
                acc += tl.dot(
                    loss,
                    weight,
                    out_dtype=tl.float32,
                    input_precision="tf32",
                )

    tl.store(
        out_ptr
        + n_idx[:, None] * out_stride_n
        + offs_ci[None, :] * out_stride_c
        + xd[:, None] * out_stride_d
        + xh[:, None] * out_stride_h
        + xw[:, None] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None],
    )


@triton.jit
def _conv_dgrad3d_small_fp32_ci8_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_D: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_d: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    weight_stride_o: tl.constexpr,
    weight_stride_i: tl.constexpr,
    weight_stride_d: tl.constexpr,
    weight_stride_h: tl.constexpr,
    weight_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_d: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    PAD_FRONT: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci = tl.arange(0, 8)
    mask_m = offs_m < M

    spatial_hw = XH * XW
    spatial = XD * spatial_hw
    n_idx = offs_m // spatial
    spatial_idx = offs_m - n_idx * spatial
    xd = spatial_idx // spatial_hw
    rem = spatial_idx - xd * spatial_hw
    xh = rem // XW
    xw = rem - xh * XW

    acc = tl.zeros((BLOCK_M, 8), dtype=tl.float32)

    for kd in tl.static_range(0, KD):
        loss_d = xd + PAD_FRONT - kd
        valid_d = (loss_d >= 0) & (loss_d < LOSS_D)
        safe_d = tl.where(valid_d, loss_d, 0)
        for kh in tl.static_range(0, KH):
            loss_h = xh + PAD_TOP - kh
            valid_h = (loss_h >= 0) & (loss_h < LOSS_H)
            safe_h = tl.where(valid_h, loss_h, 0)
            for kw in tl.static_range(0, KW):
                loss_w = xw + PAD_LEFT - kw
                valid_w = (loss_w >= 0) & (loss_w < LOSS_W)
                safe_w = tl.where(valid_w, loss_w, 0)
                valid_dhw = valid_d & valid_h & valid_w

                for co in tl.static_range(0, COUT_PER_GROUP):
                    loss = tl.load(
                        loss_ptr
                        + n_idx * loss_stride_n
                        + co * loss_stride_c
                        + safe_d * loss_stride_d
                        + safe_h * loss_stride_h
                        + safe_w * loss_stride_w,
                        mask=mask_m & valid_dhw,
                        other=0.0,
                    )
                    weight = tl.load(
                        weight_ptr
                        + co * weight_stride_o
                        + offs_ci * weight_stride_i
                        + kd * weight_stride_d
                        + kh * weight_stride_h
                        + kw * weight_stride_w,
                    )
                    acc += loss[:, None] * weight[None, :]

    tl.store(
        out_ptr
        + n_idx[:, None] * out_stride_n
        + offs_ci[None, :] * out_stride_c
        + xd[:, None] * out_stride_d
        + xh[:, None] * out_stride_h
        + xw[:, None] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None],
    )


@libentry()
@libtuner(
    configs=_CONV_DGRAD_3D_PAD1_3X3_FP32_CI8_DOT_CONFIGS,
    key=[
        "M",
        "XD",
        "XH",
        "XW",
        "LOSS_D",
        "LOSS_H",
        "LOSS_W",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def _conv_dgrad3d_pad1_3x3_fp32_ci8_dot_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_D: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_d: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_d: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci = tl.arange(0, 16)
    offs_co = tl.arange(0, 16)
    mask_m = offs_m < M
    mask_ci = offs_ci < 8

    spatial_hw = XH * XW
    spatial = XD * spatial_hw
    n_idx = offs_m // spatial
    spatial_idx = offs_m - n_idx * spatial
    xd = spatial_idx // spatial_hw
    rem = spatial_idx - xd * spatial_hw
    xh = rem // XW
    xw = rem - xh * XW

    acc = tl.zeros((16, BLOCK_M), dtype=tl.float32)

    for kd in tl.static_range(0, 3):
        loss_d = xd + 1 - kd
        valid_d = (loss_d >= 0) & (loss_d < LOSS_D)
        safe_d = tl.where(valid_d, loss_d, 0)
        for kh in tl.static_range(0, 3):
            loss_h = xh + 1 - kh
            valid_h = (loss_h >= 0) & (loss_h < LOSS_H)
            safe_h = tl.where(valid_h, loss_h, 0)
            for kw in tl.static_range(0, 3):
                loss_w = xw + 1 - kw
                valid_w = (loss_w >= 0) & (loss_w < LOSS_W)
                safe_w = tl.where(valid_w, loss_w, 0)
                valid_dhw = valid_d & valid_h & valid_w

                loss = tl.load(
                    loss_ptr
                    + n_idx[None, :] * loss_stride_n
                    + offs_co[:, None] * loss_stride_c
                    + safe_d[None, :] * loss_stride_d
                    + safe_h[None, :] * loss_stride_h
                    + safe_w[None, :] * loss_stride_w,
                    mask=mask_m[None, :] & valid_dhw[None, :],
                    other=0.0,
                )
                weight = tl.load(
                    weight_ptr
                    + (((kd * 3 + kh) * 3 + kw) * 16 + offs_co[None, :]) * 8
                    + offs_ci[:, None],
                    mask=mask_ci[:, None],
                    other=0.0,
                )
                acc += tl.dot(
                    weight,
                    loss,
                    out_dtype=tl.float32,
                    input_precision="tf32",
                )

    tl.store(
        out_ptr
        + n_idx[None, :] * out_stride_n
        + offs_ci[:, None] * out_stride_c
        + xd[None, :] * out_stride_d
        + xh[None, :] * out_stride_h
        + xw[None, :] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_ci[:, None] & mask_m[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV_DGRAD_3D_PACKED_CONFIGS,
    key=[
        "M",
        "XD",
        "XH",
        "XW",
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
def _conv_dgrad3d_packed_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_D: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_d: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_d: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_FRONT: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    spatial_hw = XH * XW
    spatial = XD * spatial_hw
    n_idx = offs_m // spatial
    spatial_idx = offs_m - n_idx * spatial
    xd = spatial_idx // spatial_hw
    rem = spatial_idx - xd * spatial_hw
    xh = rem // XW
    xw = rem - xh * XW
    ci = group * CIN_PER_GROUP + offs_ci_rel

    acc = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for kd in tl.static_range(0, KD):
        loss_d_num = xd + PAD_FRONT - kd * DIL_D
        loss_d = loss_d_num // STRIDE_D
        valid_d = (loss_d_num >= 0) & (loss_d < LOSS_D)
        if STRIDE_D != 1:
            valid_d = valid_d & ((loss_d_num % STRIDE_D) == 0)
        safe_d = tl.where(valid_d, loss_d, 0)
        weight_d = KD - 1 - kd if FILTER_REVERSE else kd

        for kh in tl.static_range(0, KH):
            loss_h_num = xh + PAD_TOP - kh * DIL_H
            loss_h = loss_h_num // STRIDE_H
            valid_h = (loss_h_num >= 0) & (loss_h < LOSS_H)
            if STRIDE_H != 1:
                valid_h = valid_h & ((loss_h_num % STRIDE_H) == 0)
            safe_h = tl.where(valid_h, loss_h, 0)
            weight_h = KH - 1 - kh if FILTER_REVERSE else kh

            for kw in tl.static_range(0, KW):
                loss_w_num = xw + PAD_LEFT - kw * DIL_W
                loss_w = loss_w_num // STRIDE_W
                valid_w = (loss_w_num >= 0) & (loss_w < LOSS_W)
                if STRIDE_W != 1:
                    valid_w = valid_w & ((loss_w_num % STRIDE_W) == 0)
                valid_dhw = valid_d & valid_h & valid_w
                safe_w = tl.where(valid_w, loss_w, 0)
                weight_w = KW - 1 - kw if FILTER_REVERSE else kw

                for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
                    offs_co_rel = co_start + tl.arange(0, BLOCK_CO)
                    co = group * COUT_PER_GROUP + offs_co_rel
                    mask_co = offs_co_rel < COUT_PER_GROUP

                    loss = tl.load(
                        loss_ptr
                        + n_idx[:, None] * loss_stride_n
                        + co[None, :] * loss_stride_c
                        + safe_d[:, None] * loss_stride_d
                        + safe_h[:, None] * loss_stride_h
                        + safe_w[:, None] * loss_stride_w,
                        mask=(
                            mask_m[:, None]
                            & mask_co[None, :]
                            & valid_dhw[:, None]
                        ),
                        other=0.0,
                    )
                    weight = tl.load(
                        weight_ptr
                        + (
                            (
                                (
                                    ((group * KD + weight_d) * KH + weight_h)
                                    * KW
                                    + weight_w
                                )
                                * COUT_PER_GROUP
                                + offs_co_rel[:, None]
                            )
                            * CIN_PER_GROUP
                        )
                        + offs_ci_rel[None, :],
                        mask=mask_co[:, None] & mask_ci[None, :],
                        other=0.0,
                    )
                    acc += tl.dot(
                        loss,
                        weight,
                        out_dtype=tl.float32,
                        input_precision="tf32",
                    )

    tl.store(
        out_ptr
        + n_idx[:, None] * out_stride_n
        + ci[None, :] * out_stride_c
        + xd[:, None] * out_stride_d
        + xh[:, None] * out_stride_h
        + xw[:, None] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_ci[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV_DGRAD_3D_CONFIGS,
    key=[
        "M",
        "XD",
        "XH",
        "XW",
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
def _conv_dgrad3d_kernel(
    loss_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    LOSS_D: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_d: tl.constexpr,
    loss_stride_h: tl.constexpr,
    loss_stride_w: tl.constexpr,
    weight_stride_o: tl.constexpr,
    weight_stride_i: tl.constexpr,
    weight_stride_d: tl.constexpr,
    weight_stride_h: tl.constexpr,
    weight_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_d: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_FRONT: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_ci = pid // num_m_blocks
    pid_m = pid - pid_ci * num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_m = offs_m < M
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    spatial_hw = XH * XW
    spatial = XD * spatial_hw
    n_idx = offs_m // spatial
    spatial_idx = offs_m - n_idx * spatial
    xd = spatial_idx // spatial_hw
    rem = spatial_idx - xd * spatial_hw
    xh = rem // XW
    xw = rem - xh * XW
    ci = group * CIN_PER_GROUP + offs_ci_rel

    acc = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)

    for kd in tl.static_range(0, KD):
        loss_d_num = xd + PAD_FRONT - kd * DIL_D
        loss_d = loss_d_num // STRIDE_D
        valid_d = (loss_d_num >= 0) & (loss_d < LOSS_D)
        if STRIDE_D != 1:
            valid_d = valid_d & ((loss_d_num % STRIDE_D) == 0)
        safe_d = tl.where(valid_d, loss_d, 0)
        weight_d = KD - 1 - kd if FILTER_REVERSE else kd

        for kh in tl.static_range(0, KH):
            loss_h_num = xh + PAD_TOP - kh * DIL_H
            loss_h = loss_h_num // STRIDE_H
            valid_h = (loss_h_num >= 0) & (loss_h < LOSS_H)
            if STRIDE_H != 1:
                valid_h = valid_h & ((loss_h_num % STRIDE_H) == 0)
            safe_h = tl.where(valid_h, loss_h, 0)
            weight_h = KH - 1 - kh if FILTER_REVERSE else kh

            for kw in tl.static_range(0, KW):
                loss_w_num = xw + PAD_LEFT - kw * DIL_W
                loss_w = loss_w_num // STRIDE_W
                valid_w = (loss_w_num >= 0) & (loss_w < LOSS_W)
                if STRIDE_W != 1:
                    valid_w = valid_w & ((loss_w_num % STRIDE_W) == 0)
                valid_dhw = valid_d & valid_h & valid_w
                safe_w = tl.where(valid_w, loss_w, 0)
                weight_w = KW - 1 - kw if FILTER_REVERSE else kw

                for co_start in tl.static_range(0, COUT_PER_GROUP, BLOCK_CO):
                    offs_co_rel = co_start + tl.arange(0, BLOCK_CO)
                    co = group * COUT_PER_GROUP + offs_co_rel
                    mask_co = offs_co_rel < COUT_PER_GROUP

                    loss = tl.load(
                        loss_ptr
                        + n_idx[:, None] * loss_stride_n
                        + co[None, :] * loss_stride_c
                        + safe_d[:, None] * loss_stride_d
                        + safe_h[:, None] * loss_stride_h
                        + safe_w[:, None] * loss_stride_w,
                        mask=(
                            mask_m[:, None]
                            & mask_co[None, :]
                            & valid_dhw[:, None]
                        ),
                        other=0.0,
                    )
                    weight = tl.load(
                        weight_ptr
                        + co[:, None] * weight_stride_o
                        + offs_ci_rel[None, :] * weight_stride_i
                        + weight_d * weight_stride_d
                        + weight_h * weight_stride_h
                        + weight_w * weight_stride_w,
                        mask=mask_co[:, None] & mask_ci[None, :],
                        other=0.0,
                    )
                    acc += tl.dot(
                        loss,
                        weight,
                        out_dtype=tl.float32,
                        input_precision="tf32",
                    )

    tl.store(
        out_ptr
        + n_idx[:, None] * out_stride_n
        + ci[None, :] * out_stride_c
        + xd[:, None] * out_stride_d
        + xh[:, None] * out_stride_h
        + xw[:, None] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_ci[None, :],
    )


def conv_dgrad(
    loss: torch.Tensor,
    filter: torch.Tensor,
    input_size: Sequence[int],
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
) -> torch.Tensor:
    del compute_data_type, name
    logger.debug("FLAG_DNN CONV_DGRAD")

    input_size_tuple = _normalize_input_size(input_size)
    rank = _rank_from_input_size(input_size_tuple, filter)
    is_unbatched_1d = rank == 1 and len(input_size_tuple) == 2
    stride_tuple = _tuple_n(stride, rank, "stride")
    dilation_tuple = _tuple_n(dilation, rank, "dilation")
    mode = _normalize_convolution_mode(convolution_mode)
    pre, post = _normalize_padding(
        filter,
        stride_tuple,
        padding,
        pre_padding,
        post_padding,
        dilation_tuple,
    )

    c_in, c_out, dx_spatial, loss_spatial = _check_conv_dgrad_inputs(
        loss,
        filter,
        input_size_tuple,
        stride_tuple,
        pre,
        post,
        dilation_tuple,
        groups,
        rank,
        is_unbatched_1d,
    )
    del c_in, c_out

    if is_unbatched_1d:
        loss = loss.unsqueeze(0)
        output_shape = (1,) + input_size_tuple
    else:
        output_shape = input_size_tuple

    if not loss.is_contiguous():
        loss = loss.contiguous()
    if not filter.is_contiguous():
        filter = filter.contiguous()

    filter_reverse = mode == "CONVOLUTION"

    if _output is None:
        output = torch.empty(
            output_shape, device=loss.device, dtype=loss.dtype
        )
    else:
        if tuple(_output.shape) != tuple(output_shape):
            raise RuntimeError("conv_dgrad output buffer shape mismatch")
        if _output.dtype != loss.dtype or _output.device != loss.device:
            raise RuntimeError(
                "conv_dgrad output buffer dtype/device mismatch"
            )
        output = _output
    if output.numel() == 0:
        return output.squeeze(0) if is_unbatched_1d else output

    n = output_shape[0]
    c_in = output_shape[1]
    c_out = int(filter.shape[0])
    cin_per_group = c_in // groups
    cout_per_group = c_out // groups
    dtype_id = _dtype_id(loss.dtype)

    with torch_device_fn.device(loss.device):
        if rank == 1:
            x_len = dx_spatial[0]
            loss_len = loss_spatial[0]
            kernel = int(filter.shape[2])
            m = n * x_len

            def grid(meta):
                return (
                    triton.cdiv(m, meta["BLOCK_M"])
                    * triton.cdiv(cin_per_group, meta["BLOCK_CI"]),
                    groups,
                )

            _conv_dgrad1d_mci_kernel[grid](
                loss,
                filter,
                output,
                m,
                x_len,
                loss_len,
                c_in,
                c_out,
                cin_per_group,
                cout_per_group,
                loss.stride(0),
                loss.stride(1),
                loss.stride(2),
                filter.stride(0),
                filter.stride(1),
                filter.stride(2),
                output.stride(0),
                output.stride(1),
                output.stride(2),
                stride_tuple[0],
                pre[0],
                dilation_tuple[0],
                kernel,
                filter_reverse,
                DTYPE_ID=dtype_id,
            )
        elif rank == 2:
            xh, xw = dx_spatial
            loss_h, loss_w = loss_spatial
            kh, kw = (int(filter.shape[2]), int(filter.shape[3]))
            m = n * xh * xw
            if (
                stride_tuple == (1, 1)
                and pre == (0, 0)
                and post == (0, 0)
                and dilation_tuple == (1, 1)
                and kh == 1
                and kw == 1
            ):

                def grid(meta):
                    return (
                        triton.cdiv(m, meta["BLOCK_M"])
                        * triton.cdiv(cin_per_group, meta["BLOCK_CI"]),
                        groups,
                    )

                kernel_1x1 = (
                    _conv_dgrad2d_1x1_strided_kernel
                    if loss.dtype == torch.float32
                    else _conv_dgrad2d_1x1_kernel
                )
                kernel_1x1[grid](
                    loss,
                    filter,
                    output,
                    m,
                    xh,
                    xw,
                    c_in,
                    c_out,
                    cin_per_group,
                    cout_per_group,
                    loss.stride(0),
                    loss.stride(1),
                    loss.stride(2),
                    loss.stride(3),
                    filter.stride(0),
                    filter.stride(1),
                    output.stride(0),
                    output.stride(1),
                    output.stride(2),
                    output.stride(3),
                    DTYPE_ID=dtype_id,
                )
                return output.squeeze(0) if is_unbatched_1d else output
            if stride_tuple == (1, 1):

                def grid(meta):
                    return (
                        triton.cdiv(m, meta["BLOCK_M"])
                        * triton.cdiv(cin_per_group, meta["BLOCK_CI"]),
                        groups,
                    )

                _conv_dgrad2d_stride1_kernel[grid](
                    loss,
                    filter,
                    output,
                    m,
                    xh,
                    xw,
                    loss_h,
                    loss_w,
                    c_in,
                    c_out,
                    cin_per_group,
                    cout_per_group,
                    loss.stride(0),
                    loss.stride(1),
                    loss.stride(2),
                    loss.stride(3),
                    filter.stride(0),
                    filter.stride(1),
                    filter.stride(2),
                    filter.stride(3),
                    output.stride(0),
                    output.stride(1),
                    output.stride(2),
                    output.stride(3),
                    pre[0],
                    pre[1],
                    dilation_tuple[0],
                    dilation_tuple[1],
                    kh,
                    kw,
                    filter_reverse,
                    DTYPE_ID=dtype_id,
                )
                return output.squeeze(0) if is_unbatched_1d else output
            if (
                groups == 1
                and stride_tuple == (2, 2)
                and pre == (1, 1)
                and post == (1, 1)
                and dilation_tuple == (1, 1)
                and kh == 3
                and kw == 3
            ):
                packed_filter = _pack_weight_2d_khw_oci(filter, groups)
                if cin_per_group == 3:
                    m_loss = n * loss_h * loss_w

                    def grid_tile4(meta):
                        return (
                            triton.cdiv(m_loss, meta["BLOCK_M"])
                            * triton.cdiv(cin_per_group, meta["BLOCK_CI"]),
                            groups,
                        )

                    _conv_dgrad2d_stride2_pad1_3x3_tile4_kernel[grid_tile4](
                        loss,
                        packed_filter,
                        output,
                        m_loss,
                        xh,
                        xw,
                        loss_h,
                        loss_w,
                        c_in,
                        c_out,
                        cin_per_group,
                        cout_per_group,
                        loss.stride(0),
                        loss.stride(1),
                        loss.stride(2),
                        loss.stride(3),
                        cin_per_group,
                        1,
                        3 * cout_per_group * cin_per_group,
                        cout_per_group * cin_per_group,
                        output.stride(0),
                        output.stride(1),
                        output.stride(2),
                        output.stride(3),
                        filter_reverse,
                        DTYPE_ID=dtype_id,
                    )
                    return output.squeeze(0) if is_unbatched_1d else output

                if (
                    dtype_id == 2
                    and n == 1
                    and xh == 40
                    and xw == 40
                    and loss_h == 20
                    and loss_w == 20
                    and cin_per_group == 768
                    and cout_per_group == 768
                    and groups == 1
                    and not filter_reverse
                ):
                    block_m = 32
                    block_ci = 64
                    block_co = 64
                    group_k = 2
                    splitk_warps = 4
                    full_k_blocks = triton.cdiv(cout_per_group, block_co)
                    full_groups = triton.cdiv(full_k_blocks, group_k)
                    zero_block = 2048
                    _conv_dgrad2d_p5_zero_kernel[
                        (triton.cdiv(output.numel(), zero_block),)
                    ](
                        output,
                        TOTAL=output.numel(),
                        BLOCK=zero_block,
                        num_warps=4,
                        num_stages=4,
                    )
                    for ph in range(2):
                        parity_h_count = (xh + 1 - ph) // 2
                        m_tile = n * parity_h_count * loss_w
                        base_grid = triton.cdiv(m_tile, block_m) * triton.cdiv(
                            cin_per_group, block_ci
                        )
                        _conv_dgrad2d_stride2_pad1_3x3_p5_tile2w_splitk_kernel[
                            (base_grid * full_groups,)
                        ](
                            loss,
                            packed_filter,
                            output,
                            m_tile,
                            xh,
                            xw,
                            loss_h,
                            loss_w,
                            c_in,
                            c_out,
                            cin_per_group,
                            cout_per_group,
                            loss.stride(0),
                            loss.stride(1),
                            loss.stride(2),
                            loss.stride(3),
                            output.stride(0),
                            output.stride(1),
                            output.stride(2),
                            output.stride(3),
                            parity_h_count,
                            ph,
                            filter_reverse,
                            BLOCK_M=block_m,
                            BLOCK_CI=block_ci,
                            BLOCK_CO=block_co,
                            GROUP_K=group_k,
                            SPLIT_K_BLOCKS=full_groups,
                            K_OFFSET=0,
                            STORE=False,
                            num_warps=splitk_warps,
                            num_stages=3,
                        )
                    return output.squeeze(0) if is_unbatched_1d else output

                if (
                    dtype_id == 2
                    and loss_h * loss_w <= 1024
                    and cin_per_group == cout_per_group
                    and (cin_per_group == 512 or cin_per_group == 768)
                ):
                    m_loss = n * loss_h * loss_w
                    if cin_per_group == 512:
                        block_m = 32
                        block_ci = 64
                        block_co = 64
                        group_k = 1
                        splitk_warps = 8
                    else:
                        block_m = 32
                        block_ci = 32
                        block_co = 64
                        group_k = 2
                        splitk_warps = 4
                    full_k_blocks = triton.cdiv(cout_per_group, block_co)
                    full_groups = triton.cdiv(full_k_blocks, group_k)
                    base_grid = triton.cdiv(m_loss, block_m) * triton.cdiv(
                        cin_per_group, block_ci
                    )

                    _conv_dgrad2d_stride2_pad1_3x3_tile4_splitk_kernel[
                        (base_grid,)
                    ](
                        loss,
                        packed_filter,
                        output,
                        m_loss,
                        xh,
                        xw,
                        loss_h,
                        loss_w,
                        c_in,
                        c_out,
                        cin_per_group,
                        cout_per_group,
                        loss.stride(0),
                        loss.stride(1),
                        loss.stride(2),
                        loss.stride(3),
                        cin_per_group,
                        1,
                        3 * cout_per_group * cin_per_group,
                        cout_per_group * cin_per_group,
                        output.stride(0),
                        output.stride(1),
                        output.stride(2),
                        output.stride(3),
                        filter_reverse,
                        BLOCK_M=block_m,
                        BLOCK_CI=block_ci,
                        BLOCK_CO=block_co,
                        GROUP_K=group_k,
                        SPLIT_K_BLOCKS=1,
                        K_OFFSET=0,
                        STORE=True,
                        num_warps=splitk_warps,
                        num_stages=3,
                    )
                    if full_groups > 1:
                        _conv_dgrad2d_stride2_pad1_3x3_tile4_splitk_kernel[
                            (base_grid * (full_groups - 1),)
                        ](
                            loss,
                            packed_filter,
                            output,
                            m_loss,
                            xh,
                            xw,
                            loss_h,
                            loss_w,
                            c_in,
                            c_out,
                            cin_per_group,
                            cout_per_group,
                            loss.stride(0),
                            loss.stride(1),
                            loss.stride(2),
                            loss.stride(3),
                            cin_per_group,
                            1,
                            3 * cout_per_group * cin_per_group,
                            cout_per_group * cin_per_group,
                            output.stride(0),
                            output.stride(1),
                            output.stride(2),
                            output.stride(3),
                            filter_reverse,
                            BLOCK_M=block_m,
                            BLOCK_CI=block_ci,
                            BLOCK_CO=block_co,
                            GROUP_K=group_k,
                            SPLIT_K_BLOCKS=full_groups - 1,
                            K_OFFSET=1,
                            STORE=False,
                            num_warps=splitk_warps,
                            num_stages=3,
                        )
                    return output.squeeze(0) if is_unbatched_1d else output

                if (
                    cin_per_group >= 128
                    and cin_per_group <= 512
                    and loss_h * loss_w <= 1024
                ):
                    m_loss = n * loss_h * loss_w

                    def grid(meta):
                        return (
                            triton.cdiv(m_loss, meta["BLOCK_M"])
                            * triton.cdiv(cin_per_group, meta["BLOCK_CI"]),
                            groups,
                        )

                    _conv_dgrad2d_stride2_pad1_3x3_tile4_kernel[grid](
                        loss,
                        packed_filter,
                        output,
                        m_loss,
                        xh,
                        xw,
                        loss_h,
                        loss_w,
                        c_in,
                        c_out,
                        cin_per_group,
                        cout_per_group,
                        loss.stride(0),
                        loss.stride(1),
                        loss.stride(2),
                        loss.stride(3),
                        cin_per_group,
                        1,
                        3 * cout_per_group * cin_per_group,
                        cout_per_group * cin_per_group,
                        output.stride(0),
                        output.stride(1),
                        output.stride(2),
                        output.stride(3),
                        filter_reverse,
                        DTYPE_ID=dtype_id,
                    )
                    return output.squeeze(0) if is_unbatched_1d else output

                if cin_per_group >= 128 and loss_h * loss_w <= 1024:
                    for ph in range(2):
                        parity_h_count = (xh + 1 - ph) // 2
                        m_tile = n * parity_h_count * loss_w

                        def grid_tile2w(meta, m_tile=m_tile):
                            return (
                                triton.cdiv(m_tile, meta["BLOCK_M"])
                                * triton.cdiv(cin_per_group, meta["BLOCK_CI"]),
                            )

                        _conv_dgrad2d_stride2_pad1_3x3_tile2w_kernel[
                            grid_tile2w
                        ](
                            loss,
                            packed_filter,
                            output,
                            m_tile,
                            xh,
                            xw,
                            loss_h,
                            loss_w,
                            c_in,
                            c_out,
                            cin_per_group,
                            cout_per_group,
                            loss.stride(0),
                            loss.stride(1),
                            loss.stride(2),
                            loss.stride(3),
                            output.stride(0),
                            output.stride(1),
                            output.stride(2),
                            output.stride(3),
                            parity_h_count,
                            ph,
                            filter_reverse,
                            DTYPE_ID=dtype_id,
                        )
                    return output.squeeze(0) if is_unbatched_1d else output

                if cin_per_group >= 128:
                    for ph in range(2):
                        parity_h_count = (xh + 1 - ph) // 2
                        kh_count = 1 if ph == 0 else 2
                        for pw in range(2):
                            parity_w_count = (xw + 1 - pw) // 2
                            kw_count = 1 if pw == 0 else 2
                            m_parity = n * parity_h_count * parity_w_count

                            def grid_packed_mci(meta, m_parity=m_parity):
                                return (
                                    triton.cdiv(m_parity, meta["BLOCK_M"])
                                    * triton.cdiv(
                                        cin_per_group, meta["BLOCK_CI"]
                                    ),
                                )

                            _conv_dgrad2d_stride2_pad1_3x3_packed_mci_kernel[
                                grid_packed_mci
                            ](
                                loss,
                                packed_filter,
                                output,
                                m_parity,
                                xh,
                                xw,
                                loss_h,
                                loss_w,
                                c_in,
                                c_out,
                                cin_per_group,
                                cout_per_group,
                                loss.stride(0),
                                loss.stride(1),
                                loss.stride(2),
                                loss.stride(3),
                                output.stride(0),
                                output.stride(1),
                                output.stride(2),
                                output.stride(3),
                                parity_h_count,
                                parity_w_count,
                                ph,
                                pw,
                                kh_count,
                                kw_count,
                                filter_reverse,
                                DTYPE_ID=dtype_id,
                            )
                    return output.squeeze(0) if is_unbatched_1d else output

                max_parity_h_count = (xh + 1) // 2
                max_parity_w_count = (xw + 1) // 2
                m_parity_max = n * max_parity_h_count * max_parity_w_count

                def grid(meta):
                    return (
                        triton.cdiv(m_parity_max, meta["BLOCK_M"])
                        * triton.cdiv(cin_per_group, meta["BLOCK_CI"]),
                        4,
                        groups,
                    )

                _conv_dgrad2d_stride2_pad1_3x3_packed_kernel[grid](
                    loss,
                    packed_filter,
                    output,
                    m_parity_max,
                    n,
                    xh,
                    xw,
                    loss_h,
                    loss_w,
                    c_in,
                    c_out,
                    cin_per_group,
                    cout_per_group,
                    loss.stride(0),
                    loss.stride(1),
                    loss.stride(2),
                    loss.stride(3),
                    output.stride(0),
                    output.stride(1),
                    output.stride(2),
                    output.stride(3),
                    filter_reverse,
                    DTYPE_ID=dtype_id,
                )
                return output.squeeze(0) if is_unbatched_1d else output

            def grid(meta):
                return (
                    triton.cdiv(m, meta["BLOCK_M"])
                    * triton.cdiv(cin_per_group, meta["BLOCK_CI"]),
                    groups,
                )

            _conv_dgrad2d_kernel[grid](
                loss,
                filter,
                output,
                m,
                xh,
                xw,
                loss_h,
                loss_w,
                c_in,
                c_out,
                cin_per_group,
                cout_per_group,
                loss.stride(0),
                loss.stride(1),
                loss.stride(2),
                loss.stride(3),
                filter.stride(0),
                filter.stride(1),
                filter.stride(2),
                filter.stride(3),
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
            xd, xh, xw = dx_spatial
            loss_d, loss_h, loss_w = loss_spatial
            kd, kh, kw = (
                int(filter.shape[2]),
                int(filter.shape[3]),
                int(filter.shape[4]),
            )
            m = n * xd * xh * xw

            def grid(meta):
                return (
                    triton.cdiv(m, meta["BLOCK_M"])
                    * triton.cdiv(cin_per_group, meta["BLOCK_CI"]),
                    groups,
                )

            if loss.dtype == torch.float32:
                if (
                    groups == 1
                    and stride_tuple == (1, 1, 1)
                    and dilation_tuple == (1, 1, 1)
                    and not filter_reverse
                    and cin_per_group == 8
                    and cout_per_group * kd * kh * kw <= 216
                ):
                    grid_ci8 = (triton.cdiv(m, 16),)
                    _conv_dgrad3d_small_fp32_ci8_kernel[grid_ci8](
                        loss,
                        filter,
                        output,
                        m,
                        xd,
                        xh,
                        xw,
                        loss_d,
                        loss_h,
                        loss_w,
                        cout_per_group,
                        loss.stride(0),
                        loss.stride(1),
                        loss.stride(2),
                        loss.stride(3),
                        loss.stride(4),
                        filter.stride(0),
                        filter.stride(1),
                        filter.stride(2),
                        filter.stride(3),
                        filter.stride(4),
                        output.stride(0),
                        output.stride(1),
                        output.stride(2),
                        output.stride(3),
                        output.stride(4),
                        pre[0],
                        pre[1],
                        pre[2],
                        kd,
                        kh,
                        kw,
                        BLOCK_M=16,
                        num_warps=1,
                        num_stages=1,
                    )
                elif (
                    groups == 1
                    and stride_tuple == (1, 1, 1)
                    and pre == (1, 1, 1)
                    and post == (1, 1, 1)
                    and dilation_tuple == (1, 1, 1)
                    and not filter_reverse
                    and cin_per_group == 8
                    and cout_per_group == 16
                    and kd == 3
                    and kh == 3
                    and kw == 3
                ):
                    packed_filter = _pack_weight_3d_kdhw_oci(filter, groups)

                    def grid_ci8_dot(meta):
                        return (triton.cdiv(m, meta["BLOCK_M"]),)

                    _conv_dgrad3d_pad1_3x3_fp32_ci8_dot_kernel[grid_ci8_dot](
                        loss,
                        packed_filter,
                        output,
                        m,
                        xd,
                        xh,
                        xw,
                        loss_d,
                        loss_h,
                        loss_w,
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
                    )
                else:
                    _conv_dgrad3d_kernel[grid](
                        loss,
                        filter,
                        output,
                        m,
                        xd,
                        xh,
                        xw,
                        loss_d,
                        loss_h,
                        loss_w,
                        c_in,
                        c_out,
                        cin_per_group,
                        cout_per_group,
                        loss.stride(0),
                        loss.stride(1),
                        loss.stride(2),
                        loss.stride(3),
                        loss.stride(4),
                        filter.stride(0),
                        filter.stride(1),
                        filter.stride(2),
                        filter.stride(3),
                        filter.stride(4),
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
                packed_filter = _pack_weight_3d_kdhw_oci(filter, groups)
                _conv_dgrad3d_packed_kernel[grid](
                    loss,
                    packed_filter,
                    output,
                    m,
                    xd,
                    xh,
                    xw,
                    loss_d,
                    loss_h,
                    loss_w,
                    c_in,
                    c_out,
                    cin_per_group,
                    cout_per_group,
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
            raise RuntimeError(f"unsupported conv_dgrad spatial rank: {rank}")

    return output.squeeze(0) if is_unbatched_1d else output
