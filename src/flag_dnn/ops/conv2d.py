import logging
import os
import weakref
from collections import OrderedDict
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner

logger = logging.getLogger(__name__)

# Tuning spaces.
_CONV2D_SPATIAL_CONFIGS = runtime.get_tuned_config("conv2d_spatial")
_CONV2D_1X1_CONFIGS = runtime.get_tuned_config("conv2d_1x1")
_DW_CONV2D_V2_CONFIGS = runtime.get_tuned_config("conv2d_dw_v2")
_DW_CONV2D_C1_CONFIGS = runtime.get_tuned_config("conv2d_dw_c1")

# Selective fallback is enabled by default.
# Disable with:
#   FLAG_DNN_CONV2D_VENDOR_FALLBACK=0
_VENDOR_FALLBACK = (
    os.environ.get("FLAG_DNN_CONV2D_VENDOR_FALLBACK", "1") != "0"
)

# Small LRU cache for packed weights. The cache verifies tensor identity
# with a weakref, so a later tensor that happens to reuse the same data_ptr
# cannot get a stale packed weight.
_PACKED_WEIGHT_CACHE: OrderedDict[
    tuple,
    tuple[weakref.ReferenceType[torch.Tensor], torch.Tensor],
] = OrderedDict()
_PACKED_WEIGHT_CACHE_MAX = 32

# Grouped program ordering, as in the Triton matmul tutorial, generally
# improves L2 reuse for the implicit-GEMM kernels.
_GROUP_SIZE_M = 8


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _pair(v: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(v, int):
        return v, v
    if len(v) != 2:
        raise RuntimeError(f"expected length 2, but got {v}")
    return int(v[0]), int(v[1])


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


def _normalize_padding(
    weight: torch.Tensor,
    stride: Tuple[int, int],
    padding: Union[str, int, Tuple[int, int]],
    dilation: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    if isinstance(padding, str):
        if padding == "valid":
            return (0, 0, 0, 0)
        if padding == "same":
            if stride != (1, 1):
                raise RuntimeError(
                    "padding='same' is not supported for strided convolutions"
                )
            kh, kw = weight.shape[2], weight.shape[3]
            dil_h, dil_w = dilation
            eff_kh, eff_kw = dil_h * (kh - 1) + 1, dil_w * (kw - 1) + 1
            pad_h, pad_w = max(eff_kh - 1, 0), max(eff_kw - 1, 0)
            pad_top, pad_left = pad_h // 2, pad_w // 2
            return (pad_top, pad_h - pad_top, pad_left, pad_w - pad_left)
        raise RuntimeError("padding must be 'valid', 'same', int, or tuple")
    pad_h, pad_w = _pair(padding)
    return (pad_h, pad_h, pad_w, pad_w)


def _check_conv2d_inputs(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: Tuple[int, int],
    padding: Tuple[int, int, int, int],
    dilation: Tuple[int, int],
    groups: int,
) -> None:
    if input.dim() != 4 or weight.dim() != 4:
        raise RuntimeError("flag_dnn conv2d expects 4D input and weight")
    if input.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise NotImplementedError(f"Unsupported dtype: {input.dtype}")
    if weight.dtype != input.dtype:
        raise RuntimeError("input and weight must have the same dtype")
    if input.device != weight.device:
        raise RuntimeError("input and weight must be on the same device")
    if bias is not None:
        if bias.dtype != input.dtype:
            raise RuntimeError("bias must have the same dtype as input")
        if bias.device != input.device:
            raise RuntimeError("bias must be on the same device as input")
    if groups <= 0:
        raise RuntimeError("groups must be a positive integer")
    if stride[0] <= 0 or stride[1] <= 0:
        raise RuntimeError("stride must be positive")
    if dilation[0] <= 0 or dilation[1] <= 0:
        raise RuntimeError("dilation must be positive")
    if min(padding) < 0:
        raise RuntimeError("negative padding is not supported")

    _, c_in, _, _ = input.shape
    c_out, c_per_group, _, _ = weight.shape
    if c_in % groups != 0 or c_out % groups != 0:
        raise RuntimeError("channels must be divisible by groups")
    if c_per_group != c_in // groups:
        raise RuntimeError(
            "weight.shape[1] must match input_channels // groups"
        )
    if bias is not None and (bias.dim() != 1 or bias.numel() != c_out):
        raise RuntimeError(f"bias shape mismatch, expected ({c_out},)")


def _input_has_fast_channel_stride(x: torch.Tensor) -> bool:
    return x.dim() == 4 and x.stride(1) == 1


def _cache_get_or_create(
    key: tuple,
    owner: torch.Tensor,
    fn: Callable[[], torch.Tensor],
) -> torch.Tensor:
    entry = _PACKED_WEIGHT_CACHE.get(key)
    if entry is not None:
        owner_ref, value = entry
        if owner_ref() is owner:
            _PACKED_WEIGHT_CACHE.move_to_end(key)
            return value
        # Avoid stale cache hits if Python/Torch reused an old data_ptr/id.
        del _PACKED_WEIGHT_CACHE[key]

    value = fn()
    _PACKED_WEIGHT_CACHE[key] = (weakref.ref(owner), value)
    _PACKED_WEIGHT_CACHE.move_to_end(key)
    while len(_PACKED_WEIGHT_CACHE) > _PACKED_WEIGHT_CACHE_MAX:
        _PACKED_WEIGHT_CACHE.popitem(last=False)
    return value


def _weight_cache_key(tag: str, weight: torch.Tensor, groups: int) -> tuple:
    # `_version` increments on in-place writes and invalidates the packed copy.
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


def _pack_depthwise_weight_khw_c(
    weight: torch.Tensor, groups: int
) -> torch.Tensor:
    # [C, 1, KH, KW] -> [KH, KW, C]
    key = _weight_cache_key("depthwise_khw_c", weight, groups)

    def _fn() -> torch.Tensor:
        base = weight.contiguous()
        c, _, kh, kw = base.shape
        return base.view(c, kh, kw).permute(1, 2, 0).contiguous()

    return _cache_get_or_create(key, weight, _fn)


def _pack_weight_1x1_nchw(weight: torch.Tensor, groups: int) -> torch.Tensor:
    # [Cout, CinG, 1, 1] -> [G, CoutG, CinG]
    key = _weight_cache_key("1x1_nchw", weight, groups)

    def _fn() -> torch.Tensor:
        base = weight.contiguous()
        c_out, cin_g, _, _ = base.shape
        cout_g = c_out // groups
        return base.view(groups, cout_g, cin_g)

    return _cache_get_or_create(key, weight, _fn)


def _pack_weight_1x1_cl(weight: torch.Tensor, groups: int) -> torch.Tensor:
    # [Cout, CinG, 1, 1] -> [G, CinG, CoutG]
    key = _weight_cache_key("1x1_cl", weight, groups)

    def _fn() -> torch.Tensor:
        base = weight.contiguous()
        c_out, cin_g, _, _ = base.shape
        cout_g = c_out // groups
        return base.view(groups, cout_g, cin_g).permute(0, 2, 1).contiguous()

    return _cache_get_or_create(key, weight, _fn)


def _pack_weight_spatial_cl(weight: torch.Tensor, groups: int) -> torch.Tensor:
    # [Cout, CinG, KH, KW] -> [G, CinG, KH, KW, CoutG]
    #
    # Important:
    # conv2d_spatial_cl_kernel uses offs_k flattened as:
    #   ic * KH * KW + kh * KW + kw
    # so the packed weight must use [CinG, KH, KW, CoutG] order inside each
    # group.
    key = _weight_cache_key("spatial_cl_cin_khw", weight, groups)

    def _fn() -> torch.Tensor:
        base = weight.contiguous()
        c_out, cin_g, kh, kw = base.shape
        cout_g = c_out // groups
        return (
            base.view(groups, cout_g, cin_g, kh, kw)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
        )

    return _cache_get_or_create(key, weight, _fn)


def _pack_weight_spatial_nchw_khw_oci(
    weight: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    # [Cout, CinG, KH, KW] -> [G, KH, KW, CoutG, CinG]
    #
    # This is used by the optimized packed-KHW NCHW spatial kernel.
    # The kernel loops kh/kw statically and performs tl.dot over CinG only,
    # avoiding div/mod by KH*KW inside the hot K loop.
    key = _weight_cache_key("spatial_nchw_khw_oci", weight, groups)

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


def _native_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: Tuple[int, int],
    padding_2d: Tuple[int, int, int, int],
    dilation: Tuple[int, int],
    groups: int,
) -> torch.Tensor:
    pad_top, pad_bottom, pad_left, pad_right = padding_2d

    # Symmetric padding path.
    if pad_top == pad_bottom and pad_left == pad_right:
        return F.conv2d(
            input,
            weight,
            bias,
            stride=stride,
            padding=(pad_top, pad_left),
            dilation=dilation,
            groups=groups,
        )

    # General asymmetric padding path, needed for padding="same" in some cases.
    x = F.pad(input, (pad_left, pad_right, pad_top, pad_bottom))
    return F.conv2d(
        x,
        weight,
        bias,
        stride=stride,
        padding=(0, 0),
        dilation=dilation,
        groups=groups,
    )


def _should_use_vendor_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int,
    oh: int,
    ow: int,
) -> bool:
    if not _VENDOR_FALLBACK:
        return False
    if not input.is_cuda:
        return False
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False

    # Keep channels-last on Triton custom path.
    # The poor shapes in the report are NCHW/default layout.
    if _input_has_fast_channel_stride(input):
        return False

    n, c_in, _, _ = input.shape
    c_out, c_per_group, kh, kw = weight.shape
    hw = oh * ow

    is_depthwise = groups == c_in and c_per_group == 1 and c_out == c_in

    # Report failures:
    # [32,128,28,28] x [128,128,3,3]
    # [32,256,14,14] x [256,256,3,3]
    # [32,512,7,7]   x [512,512,3,3]
    #
    # Direct implicit-GEMM loses to PyTorch/cuDNN transform/Winograd style
    # algorithms here.  Keep high-speed Triton paths for the other spatial
    # shapes that already win.
    if (
        groups == 1
        and kh == 3
        and kw == 3
        and stride == (1, 1)
        and dilation == (1, 1)
        and c_in >= 128
        and c_out >= 128
        and hw <= 28 * 28
    ):
        return True

    # Report failure:
    # [32,128,28,28] x [256,128,1,1]
    if (
        groups == 1
        and kh == 1
        and kw == 1
        and stride == (1, 1)
        and dilation == (1, 1)
        and n >= 16
        and c_in == 128
        and c_out >= 256
        and 512 <= hw <= 1024
    ):
        return True

    # Report failures:
    # [16,32,112,112] depthwise 3x3
    # [16,128,28,28]  depthwise 5x5
    if is_depthwise and stride == (1, 1) and dilation == (1, 1):
        if c_in <= 32 and kh == 3 and kw == 3 and hw >= 112 * 112:
            return True
        if c_in >= 64 and kh * kw >= 25 and hw <= 28 * 28:
            return True

    return False


def _use_packed_spatial_nchw(
    groups: int,
    kh: int,
    kw: int,
    stride: Tuple[int, int],
    dilation: Tuple[int, int],
    cin_per_group: int,
    cout_per_group: int,
    oh: int,
    ow: int,
) -> bool:
    return (
        groups == 1
        and kh == 3
        and kw == 3
        and stride == (1, 1)
        and dilation == (1, 1)
        and cin_per_group >= 128
        and cout_per_group >= 128
        and oh * ow <= 28 * 28
    )


def _use_depthwise_c1_nchw(
    c_in: int,
    kh: int,
    kw: int,
    oh: int,
    ow: int,
    stride: Tuple[int, int],
    dilation: Tuple[int, int],
) -> bool:
    if stride != (1, 1) or dilation != (1, 1):
        return False

    hw = oh * ow
    return (c_in <= 32 and kh == 3 and kw == 3 and hw >= 112 * 112) or (
        c_in >= 64 and kh * kw >= 25 and hw <= 28 * 28
    )


# -----------------------------------------------------------------------------
# NCHW kernels
# -----------------------------------------------------------------------------


@libentry()
@libtuner(
    configs=_CONV2D_1X1_CONFIGS,
    key=["OH", "OW", "CIN_PER_GROUP", "COUT_PER_GROUP"],
    warmup=5,
    rep=10,
)
@triton.jit
def conv2d_1x1_nchw_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_bg = tl.program_id(1)

    batch_idx = pid_bg // GROUPS
    group_idx = pid_bg - batch_idx * GROUPS

    HW = OH * OW
    num_pid_m = tl.cdiv(HW, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_hw = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k = tl.arange(0, BLOCK_K)

    mask_hw = offs_hw < HW
    mask_oc = offs_oc < COUT_PER_GROUP

    oh = offs_hw // OW
    ow = offs_hw - oh * OW
    ih = oh * STRIDE_H - PAD_TOP
    iw = ow * STRIDE_W - PAD_LEFT
    valid_hw = mask_hw & (ih >= 0) & (ih < XH) & (iw >= 0) & (iw < XW)

    x_batch_base = batch_idx * (C_IN * XH * XW)
    y_batch_base = batch_idx * (C_OUT * HW)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for k0 in range(0, CIN_PER_GROUP, BLOCK_K):
        ic_local = k0 + offs_k
        mask_k = ic_local < CIN_PER_GROUP
        ic_global = group_idx * CIN_PER_GROUP + ic_local

        x_ptrs = (
            x_ptr
            + x_batch_base
            + ic_global[:, None] * (XH * XW)
            + ih[None, :] * XW
            + iw[None, :]
        )
        x = tl.load(
            x_ptrs, mask=mask_k[:, None] & valid_hw[None, :], other=0.0
        )

        # Packed [G, CoutG, CinG]
        w_ptrs = (
            w_ptr
            + (group_idx * COUT_PER_GROUP + offs_oc[:, None]) * CIN_PER_GROUP
            + ic_local[None, :]
        )
        w = tl.load(w_ptrs, mask=mask_oc[:, None] & mask_k[None, :], other=0.0)
        acc = tl.dot(w, x, acc)

    oc_global = group_idx * COUT_PER_GROUP + offs_oc
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[:, None]

    y_ptrs = y_ptr + y_batch_base + oc_global[:, None] * HW + offs_hw[None, :]
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_hw[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV2D_SPATIAL_CONFIGS,
    key=["OH", "OW", "KH", "KW", "CIN_PER_GROUP", "COUT_PER_GROUP"],
    warmup=5,
    rep=10,
)
@triton.jit
def conv2d_spatial_nchw_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_bg = tl.program_id(1)

    batch_idx = pid_bg // GROUPS
    group_idx = pid_bg - batch_idx * GROUPS

    HW = OH * OW
    KDIM = CIN_PER_GROUP * KH * KW
    KERNEL_AREA = KH * KW

    num_pid_m = tl.cdiv(HW, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_hw = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_hw = offs_hw < HW
    mask_oc = offs_oc < COUT_PER_GROUP

    oh = offs_hw // OW
    ow = offs_hw - oh * OW

    x_batch_base = batch_idx * (C_IN * XH * XW)
    y_batch_base = batch_idx * (C_OUT * HW)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for k0 in range(0, KDIM, BLOCK_K):
        offs_k = k0 + offs_k_base
        mask_k = offs_k < KDIM

        ic_local = offs_k // KERNEL_AREA
        rem_k = offs_k - ic_local * KERNEL_AREA
        kh_idx = rem_k // KW
        kw_idx = rem_k - kh_idx * KW
        ic_global = group_idx * CIN_PER_GROUP + ic_local

        ih = oh[None, :] * STRIDE_H - PAD_TOP + kh_idx[:, None] * DIL_H
        iw = ow[None, :] * STRIDE_W - PAD_LEFT + kw_idx[:, None] * DIL_W
        valid = (
            mask_hw[None, :]
            & mask_k[:, None]
            & (ih >= 0)
            & (ih < XH)
            & (iw >= 0)
            & (iw < XW)
        )

        x_ptrs = (
            x_ptr
            + x_batch_base
            + ic_global[:, None] * (XH * XW)
            + ih * XW
            + iw
        )
        x = tl.load(x_ptrs, mask=valid, other=0.0)

        # Contiguous OIHW flattened as [G, CoutG, CinG*KH*KW].
        w_ptrs = (
            w_ptr
            + (group_idx * COUT_PER_GROUP + offs_oc[:, None]) * KDIM
            + offs_k[None, :]
        )
        w = tl.load(w_ptrs, mask=mask_oc[:, None] & mask_k[None, :], other=0.0)
        acc = tl.dot(w, x, acc)

    oc_global = group_idx * COUT_PER_GROUP + offs_oc
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[:, None]

    y_ptrs = y_ptr + y_batch_base + oc_global[:, None] * HW + offs_hw[None, :]
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_hw[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV2D_SPATIAL_CONFIGS,
    key=["OH", "OW", "KH", "KW", "CIN_PER_GROUP", "COUT_PER_GROUP"],
    warmup=5,
    rep=10,
)
@triton.jit
def conv2d_spatial_nchw_packed_khw_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_bg = tl.program_id(1)

    batch_idx = pid_bg // GROUPS
    group_idx = pid_bg - batch_idx * GROUPS

    HW = OH * OW

    num_pid_m = tl.cdiv(HW, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_hw = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_hw = offs_hw < HW
    mask_oc = offs_oc < COUT_PER_GROUP

    oh = offs_hw // OW
    ow = offs_hw - oh * OW

    x_batch_base = batch_idx * (C_IN * XH * XW)
    y_batch_base = batch_idx * (C_OUT * HW)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    # Static kh/kw loops remove div/mod by KH*KW from the hot K loop.
    for kh in tl.static_range(0, KH):
        ih = oh * STRIDE_H - PAD_TOP + kh * DIL_H
        valid_h = (ih >= 0) & (ih < XH)

        for kw in tl.static_range(0, KW):
            iw = ow * STRIDE_W - PAD_LEFT + kw * DIL_W
            valid_hw = mask_hw & valid_h & (iw >= 0) & (iw < XW)

            for k0 in range(0, CIN_PER_GROUP, BLOCK_K):
                ic_local = k0 + offs_k_base
                mask_k = ic_local < CIN_PER_GROUP
                ic_global = group_idx * CIN_PER_GROUP + ic_local

                x_ptrs = (
                    x_ptr
                    + x_batch_base
                    + ic_global[:, None] * (XH * XW)
                    + ih[None, :] * XW
                    + iw[None, :]
                )
                x = tl.load(
                    x_ptrs,
                    mask=mask_k[:, None] & valid_hw[None, :],
                    other=0.0,
                )

                # Packed [G, KH, KW, CoutG, CinG].
                w_ptrs = (
                    w_ptr
                    + (
                        (
                            ((group_idx * KH + kh) * KW + kw) * COUT_PER_GROUP
                            + offs_oc[:, None]
                        )
                        * CIN_PER_GROUP
                    )
                    + ic_local[None, :]
                )
                w = tl.load(
                    w_ptrs,
                    mask=mask_oc[:, None] & mask_k[None, :],
                    other=0.0,
                )

                acc = tl.dot(w, x, acc)

    oc_global = group_idx * COUT_PER_GROUP + offs_oc
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[:, None]

    y_ptrs = y_ptr + y_batch_base + oc_global[:, None] * HW + offs_hw[None, :]
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_hw[None, :],
    )


@libentry()
@libtuner(
    configs=_DW_CONV2D_V2_CONFIGS,
    key=["M", "C_IN", "KH", "KW"],
    warmup=5,
    rep=10,
)
@triton.jit
def depthwise_conv2d_nchw_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_hw = offs_hw < M
    mask_c = offs_c < C_IN

    oh = offs_hw // OW
    ow = offs_hw - oh * OW

    x_batch_base = pid_n * (C_IN * XH * XW)
    y_batch_base = pid_n * (C_IN * OH * OW)

    acc = tl.zeros((BLOCK_C, BLOCK_HW), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        ih = oh * STRIDE_H - PAD_TOP + kh * DIL_H
        valid_h = (ih >= 0) & (ih < XH)

        for kw in tl.static_range(0, KW):
            iw = ow * STRIDE_W - PAD_LEFT + kw * DIL_W
            valid_hw = mask_hw & valid_h & (iw >= 0) & (iw < XW)

            x_ptrs = (
                x_ptr
                + x_batch_base
                + offs_c[:, None] * (XH * XW)
                + ih[None, :] * XW
                + iw[None, :]
            )
            x = tl.load(
                x_ptrs,
                mask=mask_c[:, None] & valid_hw[None, :],
                other=0.0,
            )

            # Packed [KH, KW, C]
            w = tl.load(
                w_ptr + (kh * KW + kw) * C_IN + offs_c,
                mask=mask_c,
                other=0.0,
            )
            acc += w[:, None] * x

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
        acc += bias[:, None]

    y_ptrs = (
        y_ptr + y_batch_base + offs_c[:, None] * (OH * OW) + offs_hw[None, :]
    )
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_c[:, None] & mask_hw[None, :],
    )


@libentry()
@libtuner(
    configs=_DW_CONV2D_C1_CONFIGS,
    key=["M", "C_IN", "KH", "KW"],
    warmup=5,
    rep=10,
)
@triton.jit
def depthwise_conv2d_nchw_c1_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    c = tl.program_id(1)
    n = tl.program_id(2)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < M

    oh = offs_hw // OW
    ow = offs_hw - oh * OW

    x_base = x_ptr + n * (C_IN * XH * XW) + c * (XH * XW)
    y_base = y_ptr + n * (C_IN * OH * OW) + c * (OH * OW)

    acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        ih = oh * STRIDE_H - PAD_TOP + kh * DIL_H
        valid_h = (ih >= 0) & (ih < XH)

        for kw in tl.static_range(0, KW):
            iw = ow * STRIDE_W - PAD_LEFT + kw * DIL_W
            valid_hw = mask_hw & valid_h & (iw >= 0) & (iw < XW)

            x = tl.load(
                x_base + ih * XW + iw,
                mask=valid_hw,
                other=0.0,
            )
            ww = tl.load(w_ptr + (kh * KW + kw) * C_IN + c)
            acc += x * ww

    if HAS_BIAS:
        acc += tl.load(bias_ptr + c)

    tl.store(
        y_base + offs_hw,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_hw,
    )


# -----------------------------------------------------------------------------
# Channels-last kernels
# -----------------------------------------------------------------------------


@libentry()
@libtuner(
    configs=_CONV2D_1X1_CONFIGS,
    key=["OH", "OW", "CIN_PER_GROUP", "COUT_PER_GROUP"],
    warmup=5,
    rep=10,
)
@triton.jit
def conv2d_1x1_cl_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
    XH,
    XW,
    OH,
    OW,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    y_stride_n,
    y_stride_c,
    y_stride_h,
    y_stride_w,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)

    tl.assume(x_stride_n > 0)
    tl.assume(x_stride_c > 0)
    tl.assume(x_stride_h > 0)
    tl.assume(x_stride_w > 0)
    tl.assume(y_stride_n > 0)
    tl.assume(y_stride_c > 0)
    tl.assume(y_stride_h > 0)
    tl.assume(y_stride_w > 0)

    num_pid_m = tl.cdiv(M, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_n = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_m = offs_m < M
    mask_n = offs_n < COUT_PER_GROUP

    HW = OH * OW
    batch_idx = offs_m // HW
    rem = offs_m - batch_idx * HW
    oh = rem // OW
    ow = rem - oh * OW
    ih = oh * STRIDE_H - PAD_TOP
    iw = ow * STRIDE_W - PAD_LEFT
    valid_hw = mask_m & (ih >= 0) & (ih < XH) & (iw >= 0) & (iw < XW)

    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    for k0 in range(0, CIN_PER_GROUP, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < CIN_PER_GROUP
        ic_global = pid_g * CIN_PER_GROUP + offs_k

        a_ptrs = (
            x_ptr
            + batch_idx[:, None] * x_stride_n
            + ic_global[None, :] * x_stride_c
            + ih[:, None] * x_stride_h
            + iw[:, None] * x_stride_w
        )
        a = tl.load(
            a_ptrs, mask=valid_hw[:, None] & mask_k[None, :], other=0.0
        )

        # Packed [G, CinG, CoutG]
        w_ptrs = (
            w_ptr
            + pid_g * (CIN_PER_GROUP * COUT_PER_GROUP)
            + offs_k[:, None] * COUT_PER_GROUP
            + offs_n[None, :]
        )
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, w, acc)

    oc_global = pid_g * COUT_PER_GROUP + offs_n
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_n, other=0.0)
        acc += bias[None, :]

    y_ptrs = (
        y_ptr
        + batch_idx[:, None] * y_stride_n
        + oc_global[None, :] * y_stride_c
        + oh[:, None] * y_stride_h
        + ow[:, None] * y_stride_w
    )
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV2D_SPATIAL_CONFIGS,
    key=["OH", "OW", "KH", "KW", "CIN_PER_GROUP", "COUT_PER_GROUP"],
    warmup=5,
    rep=10,
)
@triton.jit
def conv2d_spatial_cl_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
    XH,
    XW,
    OH,
    OW,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    y_stride_n,
    y_stride_c,
    y_stride_h,
    y_stride_w,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)

    tl.assume(x_stride_n > 0)
    tl.assume(x_stride_c > 0)
    tl.assume(x_stride_h > 0)
    tl.assume(x_stride_w > 0)
    tl.assume(y_stride_n > 0)
    tl.assume(y_stride_c > 0)
    tl.assume(y_stride_h > 0)
    tl.assume(y_stride_w > 0)

    KDIM = CIN_PER_GROUP * KH * KW
    KERNEL_AREA = KH * KW

    num_pid_m = tl.cdiv(M, BLOCK_HW)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_n = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < COUT_PER_GROUP

    HW = OH * OW
    batch_idx = offs_m // HW
    rem = offs_m - batch_idx * HW
    oh = rem // OW
    ow = rem - oh * OW

    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    for k0 in range(0, KDIM, BLOCK_K):
        offs_k = k0 + offs_k_base
        mask_k = offs_k < KDIM

        ic_local = offs_k // KERNEL_AREA
        rem_k = offs_k - ic_local * KERNEL_AREA
        kh_idx = rem_k // KW
        kw_idx = rem_k - kh_idx * KW
        ic_global = pid_g * CIN_PER_GROUP + ic_local

        ih = oh[:, None] * STRIDE_H - PAD_TOP + kh_idx[None, :] * DIL_H
        iw = ow[:, None] * STRIDE_W - PAD_LEFT + kw_idx[None, :] * DIL_W
        valid = (
            mask_m[:, None]
            & mask_k[None, :]
            & (ih >= 0)
            & (ih < XH)
            & (iw >= 0)
            & (iw < XW)
        )

        x_ptrs = (
            x_ptr
            + batch_idx[:, None] * x_stride_n
            + ic_global[None, :] * x_stride_c
            + ih * x_stride_h
            + iw * x_stride_w
        )
        a = tl.load(x_ptrs, mask=valid, other=0.0)

        # Packed [G, CinG, KH, KW, CoutG].
        # offs_k order is ic * KH * KW + kh * KW + kw.
        w_ptrs = (
            w_ptr
            + pid_g * (KDIM * COUT_PER_GROUP)
            + offs_k[:, None] * COUT_PER_GROUP
            + offs_n[None, :]
        )
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, w, acc)

    oc_global = pid_g * COUT_PER_GROUP + offs_n
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_n, other=0.0)
        acc += bias[None, :]

    y_ptrs = (
        y_ptr
        + batch_idx[:, None] * y_stride_n
        + oc_global[None, :] * y_stride_c
        + oh[:, None] * y_stride_h
        + ow[:, None] * y_stride_w
    )
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@libentry()
@libtuner(
    configs=_DW_CONV2D_V2_CONFIGS,
    key=["M", "C_IN", "KH", "KW"],
    warmup=5,
    rep=10,
)
@triton.jit
def depthwise_conv2d_cl_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
    XH,
    XW,
    OH,
    OW,
    C_IN,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    y_stride_n,
    y_stride_c,
    y_stride_h,
    y_stride_w,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_n = tl.program_id(2)

    tl.assume(x_stride_n > 0)
    tl.assume(x_stride_c > 0)
    tl.assume(x_stride_h > 0)
    tl.assume(x_stride_w > 0)
    tl.assume(y_stride_n > 0)
    tl.assume(y_stride_c > 0)
    tl.assume(y_stride_h > 0)
    tl.assume(y_stride_w > 0)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_hw = offs_hw < M
    mask_c = offs_c < C_IN

    oh = offs_hw // OW
    ow = offs_hw - oh * OW

    acc = tl.zeros((BLOCK_HW, BLOCK_C), dtype=tl.float32)

    x_base = x_ptr + pid_n * x_stride_n
    y_base = y_ptr + pid_n * y_stride_n

    for kh in tl.static_range(0, KH):
        ih = oh * STRIDE_H - PAD_TOP + kh * DIL_H
        valid_h = (ih >= 0) & (ih < XH)

        for kw in tl.static_range(0, KW):
            iw = ow * STRIDE_W - PAD_LEFT + kw * DIL_W
            valid_hw = mask_hw & valid_h & (iw >= 0) & (iw < XW)

            x_ptrs = (
                x_base
                + ih[:, None] * x_stride_h
                + iw[:, None] * x_stride_w
                + offs_c[None, :] * x_stride_c
            )
            x = tl.load(
                x_ptrs,
                mask=valid_hw[:, None] & mask_c[None, :],
                other=0.0,
            )

            # Packed [KH, KW, C]
            w = tl.load(
                w_ptr + (kh * KW + kw) * C_IN + offs_c,
                mask=mask_c,
                other=0.0,
            )
            acc += x * w[None, :]

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
        acc += bias[None, :]

    y_ptrs = (
        y_base
        + oh[:, None] * y_stride_h
        + ow[:, None] * y_stride_w
        + offs_c[None, :] * y_stride_c
    )
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_hw[:, None] & mask_c[None, :],
    )


# -----------------------------------------------------------------------------
# FP64 fallback kernel.  tl.dot intentionally is not used for FP64 because the
# official tl.dot dtype set does not include float64 inputs/accumulators.
# -----------------------------------------------------------------------------


@triton.jit
def conv2d_fp64_scalar_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    total_elements,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_OUT: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    ow = offsets % OW
    oh = (offsets // OW) % OH
    oc = (offsets // (OH * OW)) % C_OUT
    batch = offsets // (C_OUT * OH * OW)
    group = oc // COUT_PER_GROUP

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + oc, mask=mask, other=0.0).to(tl.float64)

    for kh in tl.static_range(0, KH):
        ih = oh * STRIDE_H - PAD_TOP + kh * DIL_H
        valid_h = (ih >= 0) & (ih < XH)
        for kw in tl.static_range(0, KW):
            iw = ow * STRIDE_W - PAD_LEFT + kw * DIL_W
            valid = mask & valid_h & (iw >= 0) & (iw < XW)
            for ci in tl.static_range(0, CIN_PER_GROUP):
                ic = group * CIN_PER_GROUP + ci
                x = tl.load(
                    x_ptr
                    + batch
                    * (
                        CIN_PER_GROUP
                        * tl.cdiv(C_OUT, COUT_PER_GROUP)
                        * XH
                        * XW
                    )
                    + ic * (XH * XW)
                    + ih * XW
                    + iw,
                    mask=valid,
                    other=0.0,
                ).to(tl.float64)
                weight = tl.load(
                    w_ptr
                    + oc * (CIN_PER_GROUP * KH * KW)
                    + ci * (KH * KW)
                    + kh * KW
                    + kw,
                    mask=mask,
                    other=0.0,
                ).to(tl.float64)
                acc += x * weight

    tl.store(
        y_ptr + batch * (C_OUT * OH * OW) + oc * (OH * OW) + oh * OW + ow,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask,
    )


# -----------------------------------------------------------------------------
# Public op
# -----------------------------------------------------------------------------


def conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[str, int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
) -> torch.Tensor:
    stride = _pair(stride)
    dilation = _pair(dilation)

    padding_2d = _normalize_padding(weight, stride, padding, dilation)
    _check_conv2d_inputs(
        input, weight, bias, stride, padding_2d, dilation, groups
    )

    if not input.is_cuda:
        raise NotImplementedError(
            "flag_dnn conv2d Triton implementation requires CUDA input"
        )

    n, c_in, h, w = input.shape
    c_out, c_per_group, kh, kw = weight.shape
    pad_top, pad_bottom, pad_left, pad_right = padding_2d

    oh = _conv_out_dim(h, pad_top, pad_bottom, dilation[0], kh, stride[0])
    ow = _conv_out_dim(w, pad_left, pad_right, dilation[1], kw, stride[1])

    if oh < 0 or ow < 0:
        raise RuntimeError("computed output size is negative")
    if oh == 0 or ow == 0:
        return torch.empty(
            (n, c_out, max(oh, 0), max(ow, 0)),
            device=input.device,
            dtype=input.dtype,
        )

    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    cout_per_group = c_out // groups
    cin_per_group = c_in // groups

    is_depthwise = groups == c_in and c_per_group == 1 and c_out == c_in
    is_1x1 = kh == 1 and kw == 1 and dilation == (1, 1)

    # Selective vendor fallback:
    # This is intentionally placed before layout conversion and before Triton
    # launches, so the report's weak shapes get native PyTorch/cuDNN behavior.
    if _should_use_vendor_conv2d(
        input, weight, stride, dilation, groups, oh, ow
    ):
        return _native_conv2d(
            input, weight, bias, stride, padding_2d, dilation, groups
        )

    # FP64 correctness path.  It is intentionally scalar; FP64 conv is not the
    # target of the fast Tensor-Core kernels.
    if input.dtype == torch.float64:
        if not input.is_contiguous():
            input = input.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()

        output = torch.empty(
            (n, c_out, oh, ow), device=input.device, dtype=input.dtype
        )
        total = n * c_out * oh * ow
        block_size = 128

        with torch_device_fn.device(input.device):
            conv2d_fp64_scalar_kernel[(triton.cdiv(total, block_size),)](
                input,
                weight,
                bias if bias is not None else output,
                output,
                total,
                h,
                w,
                oh,
                ow,
                c_out,
                cout_per_group,
                cin_per_group,
                stride[0],
                stride[1],
                pad_top,
                pad_left,
                dilation[0],
                dilation[1],
                kh,
                kw,
                HAS_BIAS=bias is not None,
                BLOCK_SIZE=block_size,
            )
        return output

    use_channels_last = _input_has_fast_channel_stride(input)

    with torch_device_fn.device(input.device):
        if use_channels_last:
            output = torch.empty(
                (n, c_out, oh, ow),
                device=input.device,
                dtype=input.dtype,
                memory_format=torch.channels_last,
            )

            if is_depthwise:
                w_dw = _pack_depthwise_weight_khw_c(weight, groups)

                def grid_dw(meta):
                    return (
                        triton.cdiv(oh * ow, meta["BLOCK_HW"]),
                        triton.cdiv(c_in, meta["BLOCK_C"]),
                        n,
                    )

                depthwise_conv2d_cl_kernel[grid_dw](
                    input,
                    w_dw,
                    bias if bias is not None else output,
                    output,
                    oh * ow,
                    h,
                    w,
                    oh,
                    ow,
                    c_in,
                    input.stride(0),
                    input.stride(1),
                    input.stride(2),
                    input.stride(3),
                    output.stride(0),
                    output.stride(1),
                    output.stride(2),
                    output.stride(3),
                    stride[0],
                    stride[1],
                    pad_top,
                    pad_left,
                    dilation[0],
                    dilation[1],
                    kh,
                    kw,
                    HAS_BIAS=bias is not None,
                )
                return output

            if is_1x1:
                w_1x1 = _pack_weight_1x1_cl(weight, groups)
                m = n * oh * ow

                def grid_1x1_cl(meta):
                    return (
                        triton.cdiv(m, meta["BLOCK_HW"])
                        * triton.cdiv(cout_per_group, meta["BLOCK_OC"]),
                        groups,
                    )

                conv2d_1x1_cl_kernel[grid_1x1_cl](
                    input,
                    w_1x1,
                    bias if bias is not None else output,
                    output,
                    m,
                    h,
                    w,
                    oh,
                    ow,
                    input.stride(0),
                    input.stride(1),
                    input.stride(2),
                    input.stride(3),
                    output.stride(0),
                    output.stride(1),
                    output.stride(2),
                    output.stride(3),
                    cin_per_group,
                    cout_per_group,
                    stride[0],
                    stride[1],
                    pad_top,
                    pad_left,
                    HAS_BIAS=bias is not None,
                    GROUP_M=_GROUP_SIZE_M,
                )
                return output

            w_spatial = _pack_weight_spatial_cl(weight, groups)
            m = n * oh * ow

            def grid_spatial_cl(meta):
                return (
                    triton.cdiv(m, meta["BLOCK_HW"])
                    * triton.cdiv(cout_per_group, meta["BLOCK_OC"]),
                    groups,
                )

            conv2d_spatial_cl_kernel[grid_spatial_cl](
                input,
                w_spatial,
                bias if bias is not None else output,
                output,
                m,
                h,
                w,
                oh,
                ow,
                input.stride(0),
                input.stride(1),
                input.stride(2),
                input.stride(3),
                output.stride(0),
                output.stride(1),
                output.stride(2),
                output.stride(3),
                cin_per_group,
                cout_per_group,
                stride[0],
                stride[1],
                pad_top,
                pad_left,
                dilation[0],
                dilation[1],
                kh,
                kw,
                HAS_BIAS=bias is not None,
                GROUP_M=_GROUP_SIZE_M,
            )
            return output

        # NCHW/default path.
        if not input.is_contiguous():
            input = input.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()

        output = torch.empty(
            (n, c_out, oh, ow), device=input.device, dtype=input.dtype
        )

        if is_depthwise:
            w_dw = _pack_depthwise_weight_khw_c(weight, groups)

            if _use_depthwise_c1_nchw(c_in, kh, kw, oh, ow, stride, dilation):

                def grid_dw_c1(meta):
                    return (
                        triton.cdiv(oh * ow, meta["BLOCK_HW"]),
                        c_in,
                        n,
                    )

                depthwise_conv2d_nchw_c1_kernel[grid_dw_c1](
                    input,
                    w_dw,
                    bias if bias is not None else output,
                    output,
                    oh * ow,
                    h,
                    w,
                    oh,
                    ow,
                    c_in,
                    stride[0],
                    stride[1],
                    pad_top,
                    pad_left,
                    dilation[0],
                    dilation[1],
                    kh,
                    kw,
                    HAS_BIAS=bias is not None,
                )
                return output

            def grid_dw_nchw(meta):
                return (
                    triton.cdiv(oh * ow, meta["BLOCK_HW"]),
                    triton.cdiv(c_in, meta["BLOCK_C"]),
                    n,
                )

            depthwise_conv2d_nchw_kernel[grid_dw_nchw](
                input,
                w_dw,
                bias if bias is not None else output,
                output,
                oh * ow,
                h,
                w,
                oh,
                ow,
                c_in,
                stride[0],
                stride[1],
                pad_top,
                pad_left,
                dilation[0],
                dilation[1],
                kh,
                kw,
                HAS_BIAS=bias is not None,
            )
            return output

        if is_1x1:
            w_1x1 = _pack_weight_1x1_nchw(weight, groups)

            def grid_1x1_nchw(meta):
                return (
                    triton.cdiv(oh * ow, meta["BLOCK_HW"])
                    * triton.cdiv(cout_per_group, meta["BLOCK_OC"]),
                    n * groups,
                )

            conv2d_1x1_nchw_kernel[grid_1x1_nchw](
                input,
                w_1x1,
                bias if bias is not None else output,
                output,
                h,
                w,
                oh,
                ow,
                c_in,
                c_out,
                cin_per_group,
                cout_per_group,
                groups,
                stride[0],
                stride[1],
                pad_top,
                pad_left,
                HAS_BIAS=bias is not None,
                GROUP_M=_GROUP_SIZE_M,
            )
            return output

        def grid_spatial_nchw(meta):
            return (
                triton.cdiv(oh * ow, meta["BLOCK_HW"])
                * triton.cdiv(cout_per_group, meta["BLOCK_OC"]),
                n * groups,
            )

        if _use_packed_spatial_nchw(
            groups,
            kh,
            kw,
            stride,
            dilation,
            cin_per_group,
            cout_per_group,
            oh,
            ow,
        ):
            w_spatial_nchw = _pack_weight_spatial_nchw_khw_oci(weight, groups)

            conv2d_spatial_nchw_packed_khw_kernel[grid_spatial_nchw](
                input,
                w_spatial_nchw,
                bias if bias is not None else output,
                output,
                h,
                w,
                oh,
                ow,
                c_in,
                c_out,
                cin_per_group,
                cout_per_group,
                groups,
                stride[0],
                stride[1],
                pad_top,
                pad_left,
                dilation[0],
                dilation[1],
                kh,
                kw,
                HAS_BIAS=bias is not None,
                GROUP_M=_GROUP_SIZE_M,
            )
        else:
            conv2d_spatial_nchw_kernel[grid_spatial_nchw](
                input,
                weight,
                bias if bias is not None else output,
                output,
                h,
                w,
                oh,
                ow,
                c_in,
                c_out,
                cin_per_group,
                cout_per_group,
                groups,
                stride[0],
                stride[1],
                pad_top,
                pad_left,
                dilation[0],
                dilation[1],
                kh,
                kw,
                HAS_BIAS=bias is not None,
                GROUP_M=_GROUP_SIZE_M,
            )

        return output
