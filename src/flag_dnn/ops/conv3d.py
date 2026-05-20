from typing import Optional, Sequence, Tuple, Union

import torch
import triton
import triton.language as tl

from flag_dnn.runtime import torch_device_fn

_GROUP_SIZE_M = 8


def _triple(v: Union[int, Sequence[int]]) -> Tuple[int, int, int]:
    if isinstance(v, int):
        return v, v, v
    if len(v) != 3:
        raise RuntimeError(f"expected length 3, but got {v}")
    return int(v[0]), int(v[1]), int(v[2])


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
    stride: Tuple[int, int, int],
    padding: Union[str, int, Tuple[int, ...]],
    dilation: Tuple[int, int, int],
) -> Tuple[int, int, int, int, int, int]:
    if isinstance(padding, str):
        if padding == "valid":
            return (0, 0, 0, 0, 0, 0)
        if padding == "same":
            if stride != (1, 1, 1):
                raise RuntimeError(
                    "padding='same' is not supported for strided convolutions"
                )
            kd, kh, kw = weight.shape[2], weight.shape[3], weight.shape[4]
            dil_d, dil_h, dil_w = dilation
            eff_kd = dil_d * (kd - 1) + 1
            eff_kh = dil_h * (kh - 1) + 1
            eff_kw = dil_w * (kw - 1) + 1
            pad_d = max(eff_kd - 1, 0)
            pad_h = max(eff_kh - 1, 0)
            pad_w = max(eff_kw - 1, 0)
            pad_front = pad_d // 2
            pad_top = pad_h // 2
            pad_left = pad_w // 2
            return (
                pad_front,
                pad_d - pad_front,
                pad_top,
                pad_h - pad_top,
                pad_left,
                pad_w - pad_left,
            )
        raise RuntimeError("padding must be 'valid', 'same', int, or tuple")
    if isinstance(padding, int):
        pad_d, pad_h, pad_w = padding, padding, padding
        return (pad_d, pad_d, pad_h, pad_h, pad_w, pad_w)
    if len(padding) == 3:
        pad_d, pad_h, pad_w = int(padding[0]), int(padding[1]), int(padding[2])
        return (pad_d, pad_d, pad_h, pad_h, pad_w, pad_w)
    if len(padding) == 6:
        return (
            int(padding[0]),
            int(padding[1]),
            int(padding[2]),
            int(padding[3]),
            int(padding[4]),
            int(padding[5]),
        )
    raise RuntimeError(
        "padding must be an int, length-3 tuple, or length-6 tuple"
    )


def _check_conv3d_inputs(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int, int, int, int],
    dilation: Tuple[int, int, int],
    groups: int,
) -> None:
    if input.dim() not in (4, 5) or weight.dim() != 5:
        raise RuntimeError("flag_dnn conv3d expects 4D/5D input and 5D weight")
    if not input.is_cuda:
        raise NotImplementedError("flag_dnn conv3d only supports CUDA input")
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
    if stride[0] <= 0 or stride[1] <= 0 or stride[2] <= 0:
        raise RuntimeError("stride must be positive")
    if dilation[0] <= 0 or dilation[1] <= 0 or dilation[2] <= 0:
        raise RuntimeError("dilation must be positive")
    if min(padding) < 0:
        raise RuntimeError("negative padding is not supported")

    c_in = input.shape[-4]
    c_out, c_per_group, kd, kh, kw = weight.shape
    if min(c_in, c_out, c_per_group, kd, kh, kw) <= 0:
        raise RuntimeError("input and weight dimensions must be non-empty")
    if c_in % groups != 0 or c_out % groups != 0:
        raise RuntimeError("channels must be divisible by groups")
    if c_per_group != c_in // groups:
        raise RuntimeError(
            "weight.shape[1] must match input_channels // groups"
        )
    if bias is not None and (bias.dim() != 1 or bias.numel() != c_out):
        raise RuntimeError(f"bias shape mismatch, expected ({c_out},)")


def _kernel_meta(
    dtype: torch.dtype,
    cin_per_group: int,
    cout_per_group: int,
    kd: int,
    kh: int,
    kw: int,
) -> tuple[int, int, int, int, int]:
    kernel_volume = kd * kh * kw
    if dtype == torch.float64:
        block_oc = (
            8
            if cout_per_group >= 8
            else triton.next_power_of_2(cout_per_group)
        )
        block_m = 16
        block_k = 8
        return block_oc, block_m, block_k, 4, 1
    if kernel_volume == 1:
        block_oc = (
            32
            if cout_per_group >= 32
            else triton.next_power_of_2(cout_per_group)
        )
        block_m = 64
        block_k = (
            32
            if cin_per_group >= 32
            else max(16, triton.next_power_of_2(cin_per_group))
        )
    else:
        block_oc = 16 if cout_per_group <= 16 else 32
        block_m = 32
        block_k = 32
    return block_oc, block_m, block_k, 4, 3


@triton.jit
def conv3d_spatial_ncdhw_m_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
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
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)

    DHW = OD * OH * OW
    XHW = XH * XW
    XCDHW = C_IN * XD * XHW
    YCDHW = C_OUT * DHW
    KERNEL_VOLUME = KD * KH * KW
    KDIM = CIN_PER_GROUP * KERNEL_VOLUME

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_oc = offs_oc < COUT_PER_GROUP

    batch_idx = offs_m // DHW
    spatial = offs_m - batch_idx * DHW
    od = spatial // (OH * OW)
    rem_ohw = spatial - od * (OH * OW)
    oh = rem_ohw // OW
    ow = rem_ohw - oh * OW

    acc = tl.zeros((BLOCK_OC, BLOCK_M), dtype=tl.float32)

    for k0 in range(0, KDIM, BLOCK_K):
        offs_k = k0 + offs_k_base
        mask_k = offs_k < KDIM

        ic_local = offs_k // KERNEL_VOLUME
        rem_kernel = offs_k - ic_local * KERNEL_VOLUME
        kd_idx = rem_kernel // (KH * KW)
        rem_hw = rem_kernel - kd_idx * (KH * KW)
        kh_idx = rem_hw // KW
        kw_idx = rem_hw - kh_idx * KW
        ic_global = pid_g * CIN_PER_GROUP + ic_local

        id_in = od[None, :] * STRIDE_D - PAD_FRONT + kd_idx[:, None] * DIL_D
        ih = oh[None, :] * STRIDE_H - PAD_TOP + kh_idx[:, None] * DIL_H
        iw = ow[None, :] * STRIDE_W - PAD_LEFT + kw_idx[:, None] * DIL_W
        valid = (
            mask_k[:, None]
            & mask_m[None, :]
            & (id_in >= 0)
            & (id_in < XD)
            & (ih >= 0)
            & (ih < XH)
            & (iw >= 0)
            & (iw < XW)
        )

        x_ptrs = (
            x_ptr
            + batch_idx[None, :] * XCDHW
            + ic_global[:, None] * (XD * XHW)
            + id_in * XHW
            + ih * XW
            + iw
        )
        x = tl.load(x_ptrs, mask=valid, other=0.0)

        w_ptrs = (
            w_ptr
            + (pid_g * COUT_PER_GROUP + offs_oc[:, None]) * KDIM
            + offs_k[None, :]
        )
        w = tl.load(w_ptrs, mask=mask_oc[:, None] & mask_k[None, :], other=0.0)
        acc = tl.dot(w, x, acc, input_precision="tf32")

    oc_global = pid_g * COUT_PER_GROUP + offs_oc
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[:, None]

    y_ptrs = (
        y_ptr
        + batch_idx[None, :] * YCDHW
        + oc_global[:, None] * DHW
        + spatial[None, :]
    )
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_m[None, :],
    )


@triton.jit
def conv3d_fp64_ncdhw_m_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
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
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)

    DHW = OD * OH * OW
    XHW = XH * XW
    XCDHW = C_IN * XD * XHW
    YCDHW = C_OUT * DHW

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)

    mask_m = offs_m < M
    mask_oc = offs_oc < COUT_PER_GROUP

    batch_idx = offs_m // DHW
    spatial = offs_m - batch_idx * DHW
    od = spatial // (OH * OW)
    rem_ohw = spatial - od * (OH * OW)
    oh = rem_ohw // OW
    ow = rem_ohw - oh * OW

    oc_global = pid_g * COUT_PER_GROUP + offs_oc
    acc = tl.zeros((BLOCK_OC, BLOCK_M), dtype=tl.float64)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0).to(
            tl.float64
        )
        acc += bias[:, None]

    for kd in tl.static_range(0, KD):
        id_in = od * STRIDE_D - PAD_FRONT + kd * DIL_D
        valid_d = (id_in >= 0) & (id_in < XD)
        for kh in tl.static_range(0, KH):
            ih = oh * STRIDE_H - PAD_TOP + kh * DIL_H
            valid_h = valid_d & (ih >= 0) & (ih < XH)
            for kw in tl.static_range(0, KW):
                iw = ow * STRIDE_W - PAD_LEFT + kw * DIL_W
                valid_spatial = mask_m & valid_h & (iw >= 0) & (iw < XW)
                for c0 in range(0, CIN_PER_GROUP, BLOCK_K):
                    for kk in tl.static_range(0, BLOCK_K):
                        ic_local = c0 + kk
                        mask_k = ic_local < CIN_PER_GROUP
                        ic_global = pid_g * CIN_PER_GROUP + ic_local

                        x = tl.load(
                            x_ptr
                            + batch_idx * XCDHW
                            + ic_global * (XD * XHW)
                            + id_in * XHW
                            + ih * XW
                            + iw,
                            mask=valid_spatial & mask_k,
                            other=0.0,
                        ).to(tl.float64)
                        w = tl.load(
                            w_ptr
                            + oc_global * (CIN_PER_GROUP * KD * KH * KW)
                            + ((ic_local * KD + kd) * KH + kh) * KW
                            + kw,
                            mask=mask_oc & mask_k,
                            other=0.0,
                        ).to(tl.float64)
                        acc += w[:, None] * x[None, :]

    y_ptrs = (
        y_ptr
        + batch_idx[None, :] * YCDHW
        + oc_global[:, None] * DHW
        + spatial[None, :]
    )
    tl.store(y_ptrs, acc, mask=mask_oc[:, None] & mask_m[None, :])


def conv3d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int, int, int]] = 1,
    padding: Union[str, int, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, int, int]] = 1,
    groups: int = 1,
) -> torch.Tensor:
    stride = _triple(stride)
    dilation = _triple(dilation)
    padding_3d = _normalize_padding(weight, stride, padding, dilation)
    _check_conv3d_inputs(
        input, weight, bias, stride, padding_3d, dilation, groups
    )

    is_batched = input.dim() == 5
    conv_input = input if is_batched else input.unsqueeze(0)

    if not conv_input.is_contiguous():
        conv_input = conv_input.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    n, c_in, d, h, w = conv_input.shape
    c_out, _, kd, kh, kw = weight.shape
    pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right = padding_3d

    od = _conv_out_dim(d, pad_front, pad_back, dilation[0], kd, stride[0])
    oh = _conv_out_dim(h, pad_top, pad_bottom, dilation[1], kh, stride[1])
    ow = _conv_out_dim(w, pad_left, pad_right, dilation[2], kw, stride[2])

    if od < 0 or oh < 0 or ow < 0:
        raise RuntimeError("computed output size is negative")
    if od == 0 or oh == 0 or ow == 0:
        shape = (n, c_out, max(od, 0), max(oh, 0), max(ow, 0))
        output = torch.empty(shape, device=input.device, dtype=input.dtype)
        return output if is_batched else output.squeeze(0)

    output = torch.empty(
        (n, c_out, od, oh, ow), device=input.device, dtype=input.dtype
    )

    cout_per_group = c_out // groups
    cin_per_group = c_in // groups
    m = n * od * oh * ow
    block_oc, block_m, block_k, num_warps, num_stages = _kernel_meta(
        input.dtype, cin_per_group, cout_per_group, kd, kh, kw
    )
    grid = (
        triton.cdiv(m, block_m) * triton.cdiv(cout_per_group, block_oc),
        groups,
    )

    with torch_device_fn.device(input.device):
        if input.dtype == torch.float64:
            conv3d_fp64_ncdhw_m_kernel[grid](
                conv_input,
                weight,
                bias if bias is not None else output,
                output,
                m,
                d,
                h,
                w,
                od,
                oh,
                ow,
                c_in,
                c_out,
                cin_per_group,
                cout_per_group,
                stride[0],
                stride[1],
                stride[2],
                pad_front,
                pad_top,
                pad_left,
                dilation[0],
                dilation[1],
                dilation[2],
                kd,
                kh,
                kw,
                HAS_BIAS=bias is not None,
                BLOCK_OC=block_oc,
                BLOCK_M=block_m,
                BLOCK_K=block_k,
                GROUP_M=_GROUP_SIZE_M,
                num_warps=num_warps,
                num_stages=1,
            )
        else:
            conv3d_spatial_ncdhw_m_kernel[grid](
                conv_input,
                weight,
                bias if bias is not None else output,
                output,
                m,
                d,
                h,
                w,
                od,
                oh,
                ow,
                c_in,
                c_out,
                cin_per_group,
                cout_per_group,
                stride[0],
                stride[1],
                stride[2],
                pad_front,
                pad_top,
                pad_left,
                dilation[0],
                dilation[1],
                dilation[2],
                kd,
                kh,
                kw,
                HAS_BIAS=bias is not None,
                BLOCK_OC=block_oc,
                BLOCK_M=block_m,
                BLOCK_K=block_k,
                GROUP_M=_GROUP_SIZE_M,
                num_warps=num_warps,
                num_stages=num_stages,
            )

    return output if is_batched else output.squeeze(0)
