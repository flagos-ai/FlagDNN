import logging
from collections import OrderedDict
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner

logger = logging.getLogger(__name__)

# Triton is only kept for the two cases where a custom kernel can still be
# competitive in practice: 1x1 and true depthwise (multiplier=1).
_CONV2D_1X1_CONFIGS = runtime.get_tuned_config("conv2d_1x1")
_DW_CONV2D_V2_CONFIGS = runtime.get_tuned_config("conv2d_dw_v2")

# Small LRU cache for packed weights. This matters a lot for inference /
# benchmarking because weight packing is otherwise paid on every call.
_PACKED_WEIGHT_CACHE: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
_PACKED_WEIGHT_CACHE_MAX = 32

# Reuse the grouped-ordering idea from Triton's GEMM tutorial.
_GROUP_SIZE_M = 8


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _pair(v: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(v, int):
        return v, v
    if len(v) != 2:
        raise RuntimeError(f"expected length 2, but got {v}")
    return int(v[0]), int(v[1])


def _conv_out_dim(
    input_size: int, pad: int, dilation: int, kernel: int, stride: int
) -> int:
    return (input_size + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1


def _normalize_padding(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride: Tuple[int, int],
    padding: Union[str, int, Tuple[int, int]],
    dilation: Tuple[int, int],
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    if isinstance(padding, str):
        if padding == "valid":
            return input, (0, 0)
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
            input = F.pad(
                input, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top)
            )
            return input, (0, 0)
        raise RuntimeError("padding must be 'valid', 'same', int, or tuple")
    return input, _pair(padding)


def _check_conv2d_inputs(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
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
    if groups <= 0:
        raise RuntimeError("groups must be a positive integer")
    if stride[0] <= 0 or stride[1] <= 0:
        raise RuntimeError("stride must be positive")
    if dilation[0] <= 0 or dilation[1] <= 0:
        raise RuntimeError("dilation must be positive")

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


def _cache_get_or_create(key: tuple, fn):
    hit = _PACKED_WEIGHT_CACHE.get(key)
    if hit is not None:
        _PACKED_WEIGHT_CACHE.move_to_end(key)
        return hit
    value = fn()
    _PACKED_WEIGHT_CACHE[key] = value
    _PACKED_WEIGHT_CACHE.move_to_end(key)
    while len(_PACKED_WEIGHT_CACHE) > _PACKED_WEIGHT_CACHE_MAX:
        _PACKED_WEIGHT_CACHE.popitem(last=False)
    return value


def _weight_cache_key(tag: str, weight: torch.Tensor, groups: int) -> tuple:
    # `_version` increments on in-place modifications,
    # which is exactly the cache
    # invalidation signal we want for packed weights.
    version = int(getattr(weight, "_version", 0))
    return (
        tag,
        weight.data_ptr(),
        tuple(weight.shape),
        tuple(weight.stride()),
        str(weight.dtype),
        weight.device.type,
        weight.device.index,
        groups,
        version,
    )


def _pack_weight_native_channels_last(
    weight: torch.Tensor, groups: int
) -> torch.Tensor:
    key = _weight_cache_key("native_channels_last", weight, groups)
    return _cache_get_or_create(
        key,
        lambda: weight.to(memory_format=torch.channels_last),
    )


def _pack_depthwise_weight_cl(
    weight: torch.Tensor, groups: int
) -> torch.Tensor:
    # [C, 1, KH, KW] -> [KH, KW, C]
    key = _weight_cache_key("depthwise_cl", weight, groups)

    def _fn():
        base = weight.contiguous()
        c, _, kh, kw = base.shape
        return base.view(c, kh, kw).permute(1, 2, 0).contiguous()

    return _cache_get_or_create(key, _fn)


def _pack_weight_1x1_cl(weight: torch.Tensor, groups: int) -> torch.Tensor:
    # [Cout, CinG, 1, 1] -> [G, CinG, CoutG]
    key = _weight_cache_key("1x1_cl", weight, groups)

    def _fn():
        base = weight.contiguous()
        c_out, cin_g, _, _ = base.shape
        cout_g = c_out // groups
        return base.view(groups, cout_g, cin_g).permute(0, 2, 1).contiguous()

    return _cache_get_or_create(key, _fn)


def _input_has_fast_channel_stride(x: torch.Tensor) -> bool:
    # For a channels-last-friendly 4D tensor, channel dimension should be the
    # densest dimension. This is the property the Triton kernels rely on for
    # coalesced channel loads.
    return x.dim() == 4 and x.stride(1) == 1


def _any_requires_grad(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> bool:
    return (
        input.requires_grad
        or weight.requires_grad
        or (bias is not None and bias.requires_grad)
    )


def _should_fallback_to_native(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int,
) -> bool:
    # Triton fast paths here are forward-only and only intended for CUDA-ish
    # backends with Tensor-Core-friendly dtypes.
    if _any_requires_grad(input, weight, bias):
        return True
    if not input.is_cuda:
        return True
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return True

    c_out, c_per_group, kh, kw = weight.shape
    c_in = input.shape[1]
    is_depthwise = groups == c_in and c_per_group == 1 and c_out == c_in
    is_1x1 = kh == 1 and kw == 1 and dilation == (1, 1)

    if not (is_depthwise or is_1x1):
        return True

    # Per-op format conversion usually loses against native conv. The custom
    # kernels are only enabled when the caller already keeps tensors in a
    # channels-last-compatible layout.
    if not _input_has_fast_channel_stride(input):
        return True

    return False


# -----------------------------------------------------------------------------
# Triton kernels
# -----------------------------------------------------------------------------


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
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
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
    ow = offs_hw % OW

    acc = tl.zeros((BLOCK_HW, BLOCK_C), dtype=tl.float32)

    x_base = x_ptr + pid_n * x_stride_n
    y_base = y_ptr + pid_n * y_stride_n

    for kh in tl.static_range(KH):
        ih = oh * STRIDE_H - PAD_H + kh * DIL_H
        valid_h = (ih >= 0) & (ih < XH)

        for kw in tl.static_range(KW):
            iw = ow * STRIDE_W - PAD_W + kw * DIL_W
            valid_w = (iw >= 0) & (iw < XW)
            valid_hw = mask_hw & valid_h & valid_w

            x_ptrs = (
                x_base
                + ih[:, None] * x_stride_h
                + iw[:, None] * x_stride_w
                + offs_c[None, :] * x_stride_c
            )
            x_mask = valid_hw[:, None] & mask_c[None, :]
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)

            w_ptrs = w_ptr + (kh * KW + kw) * C_IN + offs_c
            w = tl.load(w_ptrs, mask=mask_c, other=0.0)

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
    y_mask = mask_hw[:, None] & mask_c[None, :]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


@libentry()
@libtuner(
    configs=_CONV2D_1X1_CONFIGS,
    key=["M", "CIN_PER_GROUP", "COUT_PER_GROUP"],
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
    CIN_PER_GROUP,
    COUT_PER_GROUP,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
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

    hw = OH * OW
    batch_idx = offs_m // hw
    rem = offs_m % hw
    oh = rem // OW
    ow = rem % OW

    ih = oh * STRIDE_H - PAD_H
    iw = ow * STRIDE_W - PAD_W
    valid_hw = mask_m & (ih >= 0) & (ih < XH) & (iw >= 0) & (iw < XW)

    oc_global = pid_g * COUT_PER_GROUP + offs_n

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
        a_mask = valid_hw[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        w_ptrs = (
            w_ptr
            + pid_g * (CIN_PER_GROUP * COUT_PER_GROUP)
            + offs_k[:, None] * COUT_PER_GROUP
            + offs_n[None, :]
        )
        w_mask = mask_k[:, None] & mask_n[None, :]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(a, w, acc)

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
    y_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


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

    input, padding = _normalize_padding(
        input, weight, stride, padding, dilation
    )
    _check_conv2d_inputs(
        input, weight, bias, stride, padding, dilation, groups
    )

    n, c_in, h, w = input.shape
    c_out, c_per_group, kh, kw = weight.shape
    oh = _conv_out_dim(h, padding[0], dilation[0], kh, stride[0])
    ow = _conv_out_dim(w, padding[1], dilation[1], kw, stride[1])

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

    # Default path: native conv. This is the correct high-performance choice
    # for all regular spatial convolutions and for training.
    if _should_fallback_to_native(
        input, weight, bias, stride, padding, dilation, groups
    ):
        if (
            not _any_requires_grad(input, weight, bias)
            and _input_has_fast_channel_stride(input)
            and input.dtype in (torch.float16, torch.bfloat16, torch.float32)
        ):
            # If the caller already keeps tensors in channels-last layout, feed
            # the native backend a matching weight layout too.
            weight_native = _pack_weight_native_channels_last(weight, groups)
            return F.conv2d(
                input, weight_native, bias, stride, padding, dilation, groups
            )
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    cout_per_group = c_out // groups
    cin_per_group = c_in // groups

    is_depthwise = groups == c_in and c_per_group == 1 and c_out == c_in
    is_1x1 = kh == 1 and kw == 1 and dilation == (1, 1)

    # We only reach here when the input is already channels-last-friendly.
    output = torch.empty(
        (n, c_out, oh, ow),
        device=input.device,
        dtype=input.dtype,
        memory_format=torch.channels_last,
    )

    with torch_device_fn.device(input.device):
        if is_depthwise:
            w_dw = _pack_depthwise_weight_cl(weight, groups)

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
                padding[0],
                padding[1],
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

            def grid_1x1(meta):
                return (
                    triton.cdiv(m, meta["BLOCK_HW"])
                    * triton.cdiv(cout_per_group, meta["BLOCK_OC"]),
                    groups,
                )

            conv2d_1x1_cl_kernel[grid_1x1](
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
                padding[0],
                padding[1],
                HAS_BIAS=bias is not None,
                GROUP_M=_GROUP_SIZE_M,
            )
            return output

    # Safety net. Stay correct and fast
    # if a new shape classification sneaks in.
    weight_native = _pack_weight_native_channels_last(weight, groups)
    return F.conv2d(
        input, weight_native, bias, stride, padding, dilation, groups
    )
