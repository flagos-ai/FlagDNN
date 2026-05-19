from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import torch
import triton
import triton.language as tl

from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import triton_lang_extension as tle


def _pair(value: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise RuntimeError(f"expected length 2, got {value}")
    return int(value[0]), int(value[1])


def _normalize_padding(
    weight: torch.Tensor,
    stride: Tuple[int, int],
    padding: Union[str, int, Tuple[int, int]],
    dilation: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    if isinstance(padding, str):
        if padding == "valid":
            return 0, 0, 0, 0
        if padding == "same":
            if stride != (1, 1):
                raise RuntimeError(
                    "padding='same' is not supported for strided convolutions"
                )
            kh, kw = weight.shape[2], weight.shape[3]
            eff_kh, eff_kw = (
                dilation[0] * (kh - 1) + 1,
                dilation[1] * (kw - 1) + 1,
            )
            pad_h, pad_w = max(eff_kh - 1, 0), max(eff_kw - 1, 0)
            pad_top, pad_left = pad_h // 2, pad_w // 2
            return pad_top, pad_h - pad_top, pad_left, pad_w - pad_left
        raise RuntimeError("padding must be 'valid', 'same', int, or tuple")
    pad_h, pad_w = _pair(padding)
    return pad_h, pad_h, pad_w, pad_w


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


def fused_conv2d_bias_relu(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[str, int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    config: Optional[dict[str, Any]] = None,
) -> torch.Tensor:
    if not input.is_cuda:
        raise NotImplementedError("fused_conv2d_bias_relu requires CUDA input")
    if groups != 1:
        raise NotImplementedError(
            "fused_conv2d_bias_relu currently supports groups=1"
        )
    if input.dim() != 4 or weight.dim() != 4 or bias.dim() != 1:
        raise RuntimeError(
            "fused_conv2d_bias_relu expects 4D input/weight and 1D bias"
        )
    if input.dtype != weight.dtype or input.dtype != bias.dtype:
        raise RuntimeError("input, weight, and bias must have the same dtype")
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError(
            f"fused_conv2d_bias_relu does not support dtype={input.dtype}"
        )
    if input.device != weight.device or input.device != bias.device:
        raise RuntimeError(
            "input, weight, and bias must be on the same device"
        )

    stride = _pair(stride)
    dilation = _pair(dilation)
    padding_2d = _normalize_padding(weight, stride, padding, dilation)
    n, c_in, h, w = input.shape
    c_out, c_per_group, kh, kw = weight.shape
    if c_per_group != c_in:
        raise RuntimeError(
            "weight.shape[1] must match input channels for groups=1"
        )
    if bias.numel() != c_out:
        raise RuntimeError(f"bias shape mismatch, expected ({c_out},)")

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

    use_channels_last = input.dim() == 4 and input.stride(1) == 1
    output = torch.empty(
        (n, c_out, oh, ow),
        device=input.device,
        dtype=input.dtype,
        memory_format=(
            torch.channels_last
            if use_channels_last
            else torch.contiguous_format
        ),
    )

    cfg = {
        "BLOCK_M": 64,
        "BLOCK_N": 32,
        "BLOCK_K": 32,
        "GROUP_M": 8,
        "num_warps": 4,
        "num_stages": 3,
    }
    if config:
        cfg.update(config)

    total_m = n * oh * ow

    def grid(meta):
        return (
            triton.cdiv(total_m, meta["BLOCK_M"])
            * triton.cdiv(c_out, meta["BLOCK_N"]),
        )

    with torch_device_fn.device(input.device):
        _fused_conv2d_bias_relu_kernel[grid](
            input,
            weight,
            bias,
            output,
            total_m,
            c_out,
            c_in * kh * kw,
            h,
            w,
            oh,
            ow,
            c_in,
            kh,
            kw,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            input.stride(3),
            weight.stride(0),
            weight.stride(1),
            weight.stride(2),
            weight.stride(3),
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
            BLOCK_M=cfg["BLOCK_M"],
            BLOCK_N=cfg["BLOCK_N"],
            BLOCK_K=cfg["BLOCK_K"],
            GROUP_M=cfg["GROUP_M"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )
    return output


@triton.jit
def _fused_conv2d_bias_relu_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    M,
    COUT,
    K_TOTAL,
    H,
    W_IN,
    OH,
    OW,
    CIN,
    KH: tl.constexpr,
    KW: tl.constexpr,
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_wo,
    stride_wi,
    stride_wkh,
    stride_wkw,
    stride_on,
    stride_oc,
    stride_oh,
    stride_ow,
    conv_stride_h: tl.constexpr,
    conv_stride_w: tl.constexpr,
    pad_top: tl.constexpr,
    pad_left: tl.constexpr,
    dil_h: tl.constexpr,
    dil_w: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tle.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(COUT, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_oc = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    n_idx = offs_m // (OH * OW)
    rem = offs_m - n_idx * OH * OW
    oh_idx = rem // OW
    ow_idx = rem - oh_idx * OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in tl.range(0, K_TOTAL, BLOCK_K):
        k = k_start + offs_k
        valid_k = k < K_TOTAL
        c_idx = k // (KH * KW)
        k_rem = k - c_idx * KH * KW
        kh_idx = k_rem // KW
        kw_idx = k_rem - kh_idx * KW

        ih = (
            oh_idx[:, None] * conv_stride_h + kh_idx[None, :] * dil_h - pad_top
        )
        iw = (
            ow_idx[:, None] * conv_stride_w
            + kw_idx[None, :] * dil_w
            - pad_left
        )

        x_offsets = (
            n_idx[:, None] * stride_xn
            + c_idx[None, :] * stride_xc
            + ih * stride_xh
            + iw * stride_xw
        )
        w_offsets = (
            offs_oc[None, :] * stride_wo
            + c_idx[:, None] * stride_wi
            + kh_idx[:, None] * stride_wkh
            + kw_idx[:, None] * stride_wkw
        )
        x_mask = (
            (offs_m[:, None] < M)
            & valid_k[None, :]
            & (ih >= 0)
            & (ih < H)
            & (iw >= 0)
            & (iw < W_IN)
        )
        w_mask = valid_k[:, None] & (offs_oc[None, :] < COUT)
        x = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
        w = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)
        acc += tl.dot(x, w, input_precision="tf32")

    b = tl.load(bias_ptr + offs_oc, mask=offs_oc < COUT, other=0.0)
    acc += b[None, :]
    acc = tl.maximum(acc, 0.0)

    out_offsets = (
        n_idx[:, None] * stride_on
        + offs_oc[None, :] * stride_oc
        + oh_idx[:, None] * stride_oh
        + ow_idx[:, None] * stride_ow
    )
    out_mask = (offs_m[:, None] < M) & (offs_oc[None, :] < COUT)
    tl.store(
        out_ptr + out_offsets, acc.to(out_ptr.dtype.element_ty), mask=out_mask
    )
