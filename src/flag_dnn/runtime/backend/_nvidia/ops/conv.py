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

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.graph.prepared import (
    PreparedKernelPipelineSpec,
    PreparedPipelineStepSpec,
    PreparedSingleKernelRunSpec,
    PreparedSingleKernelSpec,
    RunFn,
    get_prepared_output,
    make_kernel_pipeline_launcher,
    make_single_kernel_run_fn,
    runtime_tensor_checks_from_specs,
    runtime_tensor_checks_pass,
)
from flag_dnn.graph.prepared.common import (
    _is_runtime_device_spec,
    _static_shape,
)
from flag_dnn.graph.tensor import TensorSpec, torch_dtype
from flag_dnn.ops.conv1d import conv1d_gemm_kernel
from flag_dnn.ops.conv2d import (
    _pack_weight_spatial_nchw_khw_oci,
    conv2d_1x1_nchw_pad0_kernel,
    conv2d_spatial_nchw_3x3_stride2_pad1_im2col_kernel,
    conv2d_spatial_nchw_kernel,
)
from flag_dnn.ops.conv_dgrad import (
    _pack_weight_2d_khw_oci,
    _pack_weight_3d_kdhw_oci,
    _conv_dgrad1d_mci_kernel,
    _conv_dgrad2d_1x1_kernel,
    _conv_dgrad2d_1x1_strided_kernel,
    _conv_dgrad2d_stride2_pad1_3x3_packed_kernel,
    _conv_dgrad2d_stride2_pad1_3x3_tile2w_kernel,
    _conv_dgrad2d_stride2_pad1_3x3_tile4_kernel,
    _conv_dgrad2d_stride1_kernel,
    _conv_dgrad3d_packed_kernel,
    _conv_dgrad3d_pad1_3x3_fp32_ci8_dot_kernel,
)
from flag_dnn.ops.matmul import _batched_matmul_kernel
from flag_dnn.utils import libentry, libtuner


_FPROP_DTYPES = ("float16", "bfloat16", "float32")
_GROUP_M = 8
_CONV2D_SPATIAL_NCHW_PACKED_CONFIGS = runtime.get_tuned_config(
    "conv2d_spatial_nchw_packed"
)
_CONV_FPROP3D_CONFIGS = runtime.get_tuned_config("conv_fprop_3d")
_CONV_WGRAD_P5_MM_CONFIGS = runtime.get_tuned_config("mm")


@libentry()
@libtuner(
    configs=_CONV2D_SPATIAL_NCHW_PACKED_CONFIGS,
    key=[
        "OH",
        "OW",
        "KH",
        "KW",
        "CIN_PER_GROUP",
        "COUT_PER_GROUP",
        "STRIDE_H",
        "STRIDE_W",
        "DIL_H",
        "DIL_W",
        "HAS_BIAS",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def _conv2d_spatial_nchw_packed_khw_kernel(
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
    DTYPE_ID: tl.constexpr,
):
    """NVIDIA TF32 implicit-GEMM with pre-packed [G, KH, KW, OC, IC]."""
    pid = tl.program_id(0)
    pid_bg = tl.program_id(1)

    batch_idx = pid_bg // GROUPS
    group_idx = pid_bg - batch_idx * GROUPS
    output_hw = OH * OW

    num_pid_m = tl.cdiv(output_hw, BLOCK_HW)
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
    mask_hw = offs_hw < output_hw
    mask_oc = offs_oc < COUT_PER_GROUP

    oh = offs_hw // OW
    ow = offs_hw - oh * OW
    x_batch_base = batch_idx * (C_IN * XH * XW)
    y_batch_base = batch_idx * (C_OUT * output_hw)
    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

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
                weight = tl.load(
                    w_ptrs,
                    mask=mask_oc[:, None] & mask_k[None, :],
                    other=0.0,
                )
                acc = tl.dot(weight, x, acc, input_precision="tf32")

    oc_global = group_idx * COUT_PER_GROUP + offs_oc
    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_global, mask=mask_oc, other=0.0)
        acc += bias[:, None]
    y_ptrs = (
        y_ptr
        + y_batch_base
        + oc_global[:, None] * output_hw
        + offs_hw[None, :]
    )
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_hw[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV_FPROP3D_CONFIGS,
    key=[
        "M",
        "XD",
        "XH",
        "XW",
        "OD",
        "OH",
        "OW",
        "C_IN",
        "C_OUT",
        "KD",
        "KH",
        "KW",
        "DTYPE_ID",
    ],
    warmup=5,
    rep=10,
)
@triton.jit
def _conv_fprop3d_ncdhw_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    M: tl.constexpr,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_FRONT: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Hopper NCDHW implicit GEMM for the benchmark 3D FProp family."""
    pid = tl.program_id(0)
    output_dhw = OD * OH * OW
    input_hw = XH * XW
    input_cdhw = C_IN * XD * input_hw
    output_cdhw = C_OUT * output_dhw
    kernel_volume = KD * KH * KW
    reduction_size = C_IN * kernel_volume

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(C_OUT, BLOCK_OC)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid - group_id * num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_oc = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k_base = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_oc = offs_oc < C_OUT

    batch = offs_m // output_dhw
    spatial = offs_m - batch * output_dhw
    od = spatial // (OH * OW)
    rem_hw = spatial - od * (OH * OW)
    oh = rem_hw // OW
    ow = rem_hw - oh * OW
    acc = tl.zeros((BLOCK_OC, BLOCK_M), dtype=tl.float32)

    for k_start in range(0, reduction_size, BLOCK_K):
        offs_k = k_start + offs_k_base
        mask_k = offs_k < reduction_size
        ic = offs_k // kernel_volume
        rem_kernel = offs_k - ic * kernel_volume
        kd = rem_kernel // (KH * KW)
        rem_kernel_hw = rem_kernel - kd * (KH * KW)
        kh = rem_kernel_hw // KW
        kw = rem_kernel_hw - kh * KW

        input_d = od[None, :] * STRIDE_D - PAD_FRONT + kd[:, None] * DIL_D
        input_h = oh[None, :] * STRIDE_H - PAD_TOP + kh[:, None] * DIL_H
        input_w = ow[None, :] * STRIDE_W - PAD_LEFT + kw[:, None] * DIL_W
        valid = (
            mask_k[:, None]
            & mask_m[None, :]
            & (input_d >= 0)
            & (input_d < XD)
            & (input_h >= 0)
            & (input_h < XH)
            & (input_w >= 0)
            & (input_w < XW)
        )
        x = tl.load(
            x_ptr
            + batch[None, :] * input_cdhw
            + ic[:, None] * (XD * input_hw)
            + input_d * input_hw
            + input_h * XW
            + input_w,
            mask=valid,
            other=0.0,
        )
        weight = tl.load(
            w_ptr + offs_oc[:, None] * reduction_size + offs_k[None, :],
            mask=mask_oc[:, None] & mask_k[None, :],
            other=0.0,
        )
        acc = tl.dot(weight, x, acc, input_precision="tf32")

    tl.store(
        y_ptr
        + batch[None, :] * output_cdhw
        + offs_oc[:, None] * output_dhw
        + spatial[None, :],
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_m[None, :],
    )


@triton.jit
def _conv_wgrad1d_3tap_nodiv_split_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    LOSS_LEN: tl.constexpr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    image_stride_n: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_l: tl.constexpr,
    loss_stride_n: tl.constexpr,
    loss_stride_c: tl.constexpr,
    loss_stride_l: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    KL: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    SPLITS_PER_N: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """H100 fixed-shape 3-tap split kernel migrated from the generic op."""
    pid = tl.program_id(0)
    split = tl.program_id(1)
    group = 0
    n_idx = split // SPLITS_PER_N
    split_in_n = split - n_idx * SPLITS_PER_N
    split_size = tl.cdiv(LOSS_LEN, SPLITS_PER_N)
    l_begin = split_in_n * split_size
    l_end = tl.minimum(l_begin + split_size, LOSS_LEN)

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks
    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + offs_ci_rel

    image_k0 = KL - 1 if FILTER_REVERSE else 0
    image_k1 = KL - 2 if FILTER_REVERSE else 1
    image_k2 = KL - 3 if FILTER_REVERSE else 2
    acc0 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for l_start in tl.range(l_begin, l_end, BLOCK_M):
        loss_l = l_start + tl.arange(0, BLOCK_M)
        mask_m = loss_l < l_end
        image_l0 = loss_l - PAD_LEFT + image_k0
        image_l1 = loss_l - PAD_LEFT + image_k1
        image_l2 = loss_l - PAD_LEFT + image_k2
        valid0 = (image_l0 >= 0) & (image_l0 < LOSS_LEN)
        valid1 = (image_l1 >= 0) & (image_l1 < LOSS_LEN)
        valid2 = (image_l2 >= 0) & (image_l2 < LOSS_LEN)
        safe_l0 = tl.where(valid0, image_l0, 0)
        safe_l1 = tl.where(valid1, image_l1, 0)
        safe_l2 = tl.where(valid2, image_l2, 0)

        loss = tl.load(
            loss_ptr
            + n_idx * loss_stride_n
            + co[:, None] * loss_stride_c
            + loss_l[None, :] * loss_stride_l,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image0 = tl.load(
            image_ptr
            + n_idx * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_l0[:, None] * image_stride_l,
            mask=mask_m[:, None] & mask_ci[None, :] & valid0[:, None],
            other=0.0,
        )
        image1 = tl.load(
            image_ptr
            + n_idx * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_l1[:, None] * image_stride_l,
            mask=mask_m[:, None] & mask_ci[None, :] & valid1[:, None],
            other=0.0,
        )
        image2 = tl.load(
            image_ptr
            + n_idx * image_stride_n
            + ci[None, :] * image_stride_c
            + safe_l2[:, None] * image_stride_l,
            mask=mask_m[:, None] & mask_ci[None, :] & valid2[:, None],
            other=0.0,
        )
        acc0 += tl.dot(
            loss, image0, out_dtype=tl.float32, input_precision="tf32"
        )
        acc1 += tl.dot(
            loss, image1, out_dtype=tl.float32, input_precision="tf32"
        )
        acc2 += tl.dot(
            loss, image2, out_dtype=tl.float32, input_precision="tf32"
        )

    base = (
        (split * C_OUT + co[:, None]) * CIN_PER_GROUP + offs_ci_rel[None, :]
    ) * KL
    mask = mask_co[:, None] & mask_ci[None, :]
    tl.store(
        partial_ptr + base,
        acc0.to(partial_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        partial_ptr + base + 1,
        acc1.to(partial_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        partial_ptr + base + 2,
        acc2.to(partial_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _conv_wgrad1d_col_direct_nodiv_kernel(
    image_ptr,
    loss_ptr,
    out_ptr,
    IMAGE_LEN: tl.constexpr,
    LOSS_LEN: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
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
    BATCH_N: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """H100 fixed-shape one-launch CIK kernel migrated from the generic op."""
    pid = tl.program_id(0)
    group = tl.program_id(1)
    cik = CIN_PER_GROUP * KL
    num_n_blocks = tl.cdiv(cik, BLOCK_N)
    pid_co = pid // num_n_blocks
    pid_n = pid - pid_co * num_n_blocks
    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ci_rel = offs_n // KL
    k = offs_n - ci_rel * KL
    image_k = KL - 1 - k if FILTER_REVERSE else k
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_n = offs_n < cik
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + ci_rel
    acc = tl.zeros((BLOCK_CO, BLOCK_N), dtype=tl.float32)
    for n_idx in tl.static_range(0, BATCH_N):
        for l_start in tl.range(0, LOSS_LEN, BLOCK_M):
            loss_l = l_start + tl.arange(0, BLOCK_M)
            mask_l = loss_l < LOSS_LEN
            image_l = (
                loss_l[:, None] * STRIDE_L
                - PAD_LEFT
                + image_k[None, :] * DIL_L
            )
            valid_l = (image_l >= 0) & (image_l < IMAGE_LEN)
            safe_l = tl.where(valid_l, image_l, 0)
            loss = tl.load(
                loss_ptr
                + n_idx * loss_stride_n
                + co[:, None] * loss_stride_c
                + loss_l[None, :] * loss_stride_l,
                mask=mask_co[:, None] & mask_l[None, :],
                other=0.0,
            )
            image = tl.load(
                image_ptr
                + n_idx * image_stride_n
                + ci[None, :] * image_stride_c
                + safe_l * image_stride_l,
                mask=mask_l[:, None] & mask_n[None, :] & valid_l,
                other=0.0,
            )
            acc += tl.dot(
                loss, image, out_dtype=tl.float32, input_precision="tf32"
            )
    tl.store(
        out_ptr
        + co[:, None] * out_stride_o
        + ci_rel[None, :] * out_stride_i
        + k[None, :] * out_stride_k,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_co[:, None] & mask_n[None, :],
    )


@triton.jit
def _conv_wgrad1d_reduce3_kernel(
    partial_ptr,
    out_ptr,
    C_OUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    out_stride_o: tl.constexpr,
    out_stride_i: tl.constexpr,
    out_stride_k: tl.constexpr,
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
    mask = mask_co[:, None] & mask_ci[None, :]
    acc0 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for split in tl.static_range(0, NUM_SPLITS):
        base = (
            (split * C_OUT + co[:, None]) * CIN_PER_GROUP
            + offs_ci_rel[None, :]
        ) * 3
        acc0 += tl.load(partial_ptr + base, mask=mask, other=0.0).to(
            tl.float32
        )
        acc1 += tl.load(partial_ptr + base + 1, mask=mask, other=0.0).to(
            tl.float32
        )
        acc2 += tl.load(partial_ptr + base + 2, mask=mask, other=0.0).to(
            tl.float32
        )
    out_base = (
        out_ptr
        + co[:, None] * out_stride_o
        + offs_ci_rel[None, :] * out_stride_i
    )
    tl.store(
        out_base,
        acc0.to(out_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        out_base + out_stride_k,
        acc1.to(out_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        out_base + 2 * out_stride_k,
        acc2.to(out_ptr.dtype.element_ty),
        mask=mask,
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
            input_precision="tf32x3",
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
        out_ptr
        + co[:, None] * out_stride_o
        + offs_ci_rel[None, :] * out_stride_i,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_wgrad2d_stride2_row4_split_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
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
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    kh = tl.program_id(1)
    n_idx = tl.program_id(2)
    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks
    co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    ci = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    offs = tl.arange(0, BLOCK_HW)
    row_off = offs // 28
    loss_w = offs - row_off * 28
    valid_base = (row_off < 4) & (loss_w < 28)
    mask_co = co < COUT_PER_GROUP
    mask_ci = ci < CIN_PER_GROUP

    acc0 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for loss_h_base in tl.static_range(0, 28, 4):
        loss_h = loss_h_base + row_off
        valid_loss = valid_base & (loss_h < 28)
        image_h = loss_h * 2 - 1 + kh
        valid_h = (image_h >= 0) & (image_h < 56) & valid_loss
        image_w0 = loss_w * 2 - 1
        image_w1 = loss_w * 2
        image_w2 = loss_w * 2 + 1
        valid0 = valid_h & (image_w0 >= 0) & (image_w0 < 56)
        valid1 = valid_h & (image_w1 >= 0) & (image_w1 < 56)
        valid2 = valid_h & (image_w2 >= 0) & (image_w2 < 56)
        safe_w0 = tl.where(valid0, image_w0, 0)
        safe_w1 = tl.where(valid1, image_w1, 0)
        safe_w2 = tl.where(valid2, image_w2, 0)

        loss = tl.load(
            loss_ptr
            + n_idx * loss_stride_n
            + co[:, None] * loss_stride_c
            + loss_h[None, :] * loss_stride_h
            + loss_w[None, :] * loss_stride_w,
            mask=mask_co[:, None] & valid_loss[None, :],
            other=0.0,
        )
        image0 = tl.load(
            image_ptr
            + n_idx * image_stride_n
            + ci[None, :] * image_stride_c
            + image_h[:, None] * image_stride_h
            + safe_w0[:, None] * image_stride_w,
            mask=mask_ci[None, :] & valid0[:, None],
            other=0.0,
        )
        image1 = tl.load(
            image_ptr
            + n_idx * image_stride_n
            + ci[None, :] * image_stride_c
            + image_h[:, None] * image_stride_h
            + safe_w1[:, None] * image_stride_w,
            mask=mask_ci[None, :] & valid1[:, None],
            other=0.0,
        )
        image2 = tl.load(
            image_ptr
            + n_idx * image_stride_n
            + ci[None, :] * image_stride_c
            + image_h[:, None] * image_stride_h
            + safe_w2[:, None] * image_stride_w,
            mask=mask_ci[None, :] & valid2[:, None],
            other=0.0,
        )
        acc0 += tl.dot(
            loss, image0, out_dtype=tl.float32, input_precision="tf32"
        )
        acc1 += tl.dot(
            loss, image1, out_dtype=tl.float32, input_precision="tf32"
        )
        acc2 += tl.dot(
            loss, image2, out_dtype=tl.float32, input_precision="tf32"
        )

    base = (
        (n_idx * C_OUT + co[:, None]) * CIN_PER_GROUP + ci[None, :]
    ) * 9 + kh * 3
    mask = mask_co[:, None] & mask_ci[None, :]
    tl.store(partial_ptr + base, acc0, mask=mask)
    tl.store(partial_ptr + base + 1, acc1, mask=mask)
    tl.store(partial_ptr + base + 2, acc2, mask=mask)


@triton.jit
def _conv_wgrad2d_stride2_3tap_atomic_kernel(
    image_ptr,
    loss_ptr,
    out_ptr,
    M: tl.constexpr,
    IMAGE_H: tl.constexpr,
    IMAGE_W: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
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
    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co < COUT_PER_GROUP
    mask_ci = offs_ci < CIN_PER_GROUP
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
            + offs_co[:, None] * loss_stride_c
            + loss_h[None, :] * loss_stride_h
            + loss_w[None, :] * loss_stride_w,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image0 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + offs_ci[None, :] * image_stride_c
            + safe_h[:, None] * image_stride_h
            + safe_w0[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid0[:, None],
            other=0.0,
        )
        image1 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + offs_ci[None, :] * image_stride_c
            + safe_h[:, None] * image_stride_h
            + safe_w1[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid1[:, None],
            other=0.0,
        )
        image2 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + offs_ci[None, :] * image_stride_c
            + safe_h[:, None] * image_stride_h
            + safe_w2[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid2[:, None],
            other=0.0,
        )
        acc0 += tl.dot(
            loss, image0, out_dtype=tl.float32, input_precision="tf32"
        )
        acc1 += tl.dot(
            loss, image1, out_dtype=tl.float32, input_precision="tf32"
        )
        acc2 += tl.dot(
            loss, image2, out_dtype=tl.float32, input_precision="tf32"
        )
    mask = mask_co[:, None] & mask_ci[None, :]
    base = (
        out_ptr
        + offs_co[:, None] * out_stride_o
        + offs_ci[None, :] * out_stride_i
        + kh * out_stride_h
    )
    tl.atomic_add(base, acc0, sem="relaxed", mask=mask)
    tl.atomic_add(base + out_stride_w, acc1, sem="relaxed", mask=mask)
    tl.atomic_add(base + 2 * out_stride_w, acc2, sem="relaxed", mask=mask)


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
            + (
                (split * C_OUT + co[:, None]) * CIN_PER_GROUP
                + offs_ci_rel[None, :]
            )
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


@triton.jit
def _conv_wgrad2d_col_split_kernel(
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
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    split_group = tl.program_id(1)
    split = split_group % NUM_SPLITS
    group = split_group // NUM_SPLITS
    k_elems = KH * KW
    cik = CIN_PER_GROUP * k_elems
    num_n_blocks = tl.cdiv(cik, BLOCK_N)
    pid_co = pid // num_n_blocks
    pid_n = pid - pid_co * num_n_blocks
    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_ci_rel = offs_n // k_elems
    rem = offs_n - offs_ci_rel * k_elems
    kh = rem // KW
    kw = rem - kh * KW
    image_kh = KH - 1 - kh if FILTER_REVERSE else kh
    image_kw = KW - 1 - kw if FILTER_REVERSE else kw
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_n = offs_n < cik
    co = group * COUT_PER_GROUP + offs_co_rel
    ci = group * CIN_PER_GROUP + offs_ci_rel
    split_size = tl.cdiv(M, NUM_SPLITS)
    split_begin = split * split_size
    split_end = tl.minimum(split_begin + split_size, M)

    acc = tl.zeros((BLOCK_CO, BLOCK_N), dtype=tl.float32)
    for m_start in tl.range(split_begin, split_end, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < split_end
        safe_m = tl.where(mask_m, offs_m, 0)
        loss_w = safe_m % LOSS_W
        tmp = safe_m // LOSS_W
        loss_h = tmp % LOSS_H
        n_idx = tmp // LOSS_H
        image_h = (
            loss_h[:, None] * STRIDE_H - PAD_H + image_kh[None, :] * DIL_H
        )
        image_w = (
            loss_w[:, None] * STRIDE_W - PAD_W + image_kw[None, :] * DIL_W
        )
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
            + safe_h * image_stride_h
            + safe_w * image_stride_w,
            mask=mask_m[:, None] & mask_n[None, :] & valid_hw,
            other=0.0,
        )
        acc += tl.dot(
            loss, image, out_dtype=tl.float32, input_precision="tf32"
        )
    tl.store(
        partial_ptr + (split * C_OUT + co[:, None]) * cik + offs_n[None, :],
        acc,
        mask=mask_co[:, None] & mask_n[None, :],
    )


@triton.jit
def _conv_wgrad2d_col_reduce_kernel(
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
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)
    k_elems = KH * KW
    cik = CIN_PER_GROUP * k_elems
    num_n_blocks = tl.cdiv(cik, BLOCK_N)
    pid_co = pid // num_n_blocks
    pid_n = pid - pid_co * num_n_blocks
    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_ci_rel = offs_n // k_elems
    rem = offs_n - offs_ci_rel * k_elems
    kh = rem // KW
    kw = rem - kh * KW
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_n = offs_n < cik
    co = group * COUT_PER_GROUP + offs_co_rel

    acc = tl.zeros((BLOCK_CO, BLOCK_N), dtype=tl.float32)
    for split in tl.static_range(0, NUM_SPLITS):
        acc += tl.load(
            partial_ptr
            + (split * C_OUT + co[:, None]) * cik
            + offs_n[None, :],
            mask=mask_co[:, None] & mask_n[None, :],
            other=0.0,
        )
    tl.store(
        out_ptr
        + co[:, None] * out_stride_o
        + offs_ci_rel[None, :] * out_stride_i
        + kh[None, :] * out_stride_h
        + kw[None, :] * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_co[:, None] & mask_n[None, :],
    )


@triton.jit
def _conv_wgrad2d_p5_pack_image_kernel(
    image_ptr,
    packed_ptr,
    CIN_PER_GROUP: tl.constexpr,
    image_stride_c: tl.constexpr,
    image_stride_h: tl.constexpr,
    image_stride_w: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    cik = CIN_PER_GROUP * 9

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < 400
    mask_n = offs_n < cik

    loss_h = offs_m // 20
    loss_w = offs_m - loss_h * 20
    ci = offs_n // 9
    kpos = offs_n - ci * 9
    kh = kpos // 3
    kw = kpos - kh * 3
    safe_ci = tl.where(mask_n, ci, 0)

    image_h = loss_h[:, None] * 2 - 1 + kh[None, :]
    image_w = loss_w[:, None] * 2 - 1 + kw[None, :]
    valid = (
        mask_m[:, None]
        & mask_n[None, :]
        & (image_h >= 0)
        & (image_h < 40)
        & (image_w >= 0)
        & (image_w < 40)
    )
    safe_h = tl.where(valid, image_h, 0)
    safe_w = tl.where(valid, image_w, 0)
    values = tl.load(
        image_ptr
        + safe_ci[None, :] * image_stride_c
        + safe_h * image_stride_h
        + safe_w * image_stride_w,
        mask=valid,
        other=0.0,
    )
    tl.store(
        packed_ptr + offs_m[:, None] * cik + offs_n[None, :],
        values,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@libentry()
@libtuner(
    configs=_CONV_WGRAD_P5_MM_CONFIGS,
    key=["M", "N", "K", "DTYPE_ID"],
    strategy=["align32", "align32", "align32", "default"],
    warmup=5,
    rep=10,
)
@triton.jit
def _conv_wgrad2d_p5_mm_kernel(
    loss_ptr,
    packed_ptr,
    out_ptr,
    M,
    N,
    K,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid - group_id * num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    loss_ptrs = loss_ptr + offs_m[:, None] * K + offs_k[None, :]
    packed_ptrs = packed_ptr + offs_k[:, None] * N + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in tl.range(0, K, BLOCK_K):
        k_offsets = k_start + offs_k
        loss = tl.load(
            loss_ptrs,
            mask=(offs_m[:, None] < M) & (k_offsets[None, :] < K),
            other=0.0,
        )
        packed = tl.load(
            packed_ptrs,
            mask=(k_offsets[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(loss, packed, input_precision="tf32")
        loss_ptrs += BLOCK_K
        packed_ptrs += BLOCK_K * N

    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
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
            + (
                (split * C_OUT + co[:, None]) * CIN_PER_GROUP
                + offs_ci_rel[None, :]
            )
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


@triton.jit
def _conv_wgrad3d_valid_nsplit_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    IMAGE_D: tl.constexpr,
    IMAGE_H: tl.constexpr,
    IMAGE_W: tl.constexpr,
    LOSS_D: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
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
    SPLITS_PER_N: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    k = tl.program_id(1)
    split = tl.program_id(2)

    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks

    kw = k % KW
    tmp_k = k // KW
    kh = tmp_k % KH
    kd = tmp_k // KH
    image_kd = kd
    image_kh = kh
    image_kw = kw
    n_idx = split // SPLITS_PER_N
    split_in_n = split - n_idx * SPLITS_PER_N

    d_begin = (PAD_D - image_kd * DIL_D + STRIDE_D - 1) // STRIDE_D
    d_begin = tl.maximum(d_begin, 0)
    d_end = (IMAGE_D - 1 + PAD_D - image_kd * DIL_D) // STRIDE_D + 1
    d_end = tl.minimum(d_end, LOSS_D)
    h_begin = (PAD_H - image_kh * DIL_H + STRIDE_H - 1) // STRIDE_H
    h_begin = tl.maximum(h_begin, 0)
    h_end = (IMAGE_H - 1 + PAD_H - image_kh * DIL_H) // STRIDE_H + 1
    h_end = tl.minimum(h_end, LOSS_H)
    w_begin = (PAD_W - image_kw * DIL_W + STRIDE_W - 1) // STRIDE_W
    w_begin = tl.maximum(w_begin, 0)
    w_end = (IMAGE_W - 1 + PAD_W - image_kw * DIL_W) // STRIDE_W + 1
    w_end = tl.minimum(w_end, LOSS_W)
    valid_d = tl.maximum(d_end - d_begin, 0)
    valid_h = tl.maximum(h_end - h_begin, 0)
    valid_w = tl.maximum(w_end - w_begin, 0)
    valid_hw = valid_h * valid_w
    valid_vol = valid_d * valid_hw
    split_size = tl.cdiv(valid_vol, SPLITS_PER_N)
    vol_begin = split_in_n * split_size
    vol_end = tl.minimum(vol_begin + split_size, valid_vol)

    offs_co_rel = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci_rel = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co_rel < COUT_PER_GROUP
    mask_ci = offs_ci_rel < CIN_PER_GROUP

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for vol_start in tl.range(vol_begin, vol_end, BLOCK_M):
        vol = vol_start + tl.arange(0, BLOCK_M)
        mask_m = vol < vol_end
        safe_vol = tl.where(mask_m, vol, 0)
        rel_d = safe_vol // valid_hw
        rem = safe_vol - rel_d * valid_hw
        rel_h = rem // valid_w
        rel_w = rem - rel_h * valid_w
        loss_d = d_begin + rel_d
        loss_h = h_begin + rel_h
        loss_w = w_begin + rel_w
        image_d = loss_d * STRIDE_D - PAD_D + image_kd * DIL_D
        image_h = loss_h * STRIDE_H - PAD_H + image_kh * DIL_H
        image_w = loss_w * STRIDE_W - PAD_W + image_kw * DIL_W
        loss = tl.load(
            loss_ptr
            + n_idx * loss_stride_n
            + offs_co_rel[:, None] * loss_stride_c
            + loss_d[None, :] * loss_stride_d
            + loss_h[None, :] * loss_stride_h
            + loss_w[None, :] * loss_stride_w,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + n_idx * image_stride_n
            + offs_ci_rel[None, :] * image_stride_c
            + image_d[:, None] * image_stride_d
            + image_h[:, None] * image_stride_h
            + image_w[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :],
            other=0.0,
        )
        acc += tl.dot(
            loss, image, out_dtype=tl.float32, input_precision="tf32"
        )

    k_elems = KD * KH * KW
    tl.store(
        partial_ptr
        + (
            (split * C_OUT + offs_co_rel[:, None]) * CIN_PER_GROUP
            + offs_ci_rel[None, :]
        )
        * k_elems
        + k,
        acc,
        mask=mask_co[:, None] & mask_ci[None, :],
    )


@triton.jit
def _conv_wgrad3d_valid_col_direct_kernel(
    image_ptr,
    loss_ptr,
    out_ptr,
    IMAGE_D: tl.constexpr,
    IMAGE_H: tl.constexpr,
    IMAGE_W: tl.constexpr,
    LOSS_D: tl.constexpr,
    LOSS_H: tl.constexpr,
    LOSS_W: tl.constexpr,
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
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    co_rel = tl.program_id(0)
    pid_n = tl.program_id(1)
    group = tl.program_id(2)

    k_elems = KD * KH * KW
    cik = CIN_PER_GROUP * k_elems
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ci_rel = offs_n // k_elems
    rem = offs_n - ci_rel * k_elems
    kw = rem % KW
    tmp_k = rem // KW
    kh = tmp_k % KH
    kd = tmp_k // KH
    mask_n = offs_n < cik

    d_begin = (PAD_D - kd * DIL_D + STRIDE_D - 1) // STRIDE_D
    d_begin = tl.maximum(d_begin, 0)
    d_end = (IMAGE_D - 1 + PAD_D - kd * DIL_D) // STRIDE_D + 1
    d_end = tl.minimum(d_end, LOSS_D)
    h_begin = (PAD_H - kh * DIL_H + STRIDE_H - 1) // STRIDE_H
    h_begin = tl.maximum(h_begin, 0)
    h_end = (IMAGE_H - 1 + PAD_H - kh * DIL_H) // STRIDE_H + 1
    h_end = tl.minimum(h_end, LOSS_H)
    w_begin = (PAD_W - kw * DIL_W + STRIDE_W - 1) // STRIDE_W
    w_begin = tl.maximum(w_begin, 0)
    w_end = (IMAGE_W - 1 + PAD_W - kw * DIL_W) // STRIDE_W + 1
    w_end = tl.minimum(w_end, LOSS_W)
    valid_d = tl.maximum(d_end - d_begin, 0)
    valid_h = tl.maximum(h_end - h_begin, 0)
    valid_w = tl.maximum(w_end - w_begin, 0)
    valid_hw = valid_h * valid_w
    valid_vol = valid_d * valid_hw

    co = group * COUT_PER_GROUP + co_rel
    ci = group * CIN_PER_GROUP + ci_rel
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    max_vol = LOSS_D * LOSS_H * LOSS_W
    for vol_start in tl.range(0, max_vol, BLOCK_M):
        vol = vol_start + tl.arange(0, BLOCK_M)
        mask_m = vol < max_vol
        valid_mn = (
            mask_m[:, None]
            & mask_n[None, :]
            & (vol[:, None] < valid_vol[None, :])
        )
        safe_vol = tl.where(valid_mn, vol[:, None], 0)
        rel_d = safe_vol // valid_hw[None, :]
        rem_vol = safe_vol - rel_d * valid_hw[None, :]
        rel_h = rem_vol // valid_w[None, :]
        rel_w = rem_vol - rel_h * valid_w[None, :]
        loss_d = d_begin[None, :] + rel_d
        loss_h = h_begin[None, :] + rel_h
        loss_w = w_begin[None, :] + rel_w
        image_d = loss_d * STRIDE_D - PAD_D + kd[None, :] * DIL_D
        image_h = loss_h * STRIDE_H - PAD_H + kh[None, :] * DIL_H
        image_w = loss_w * STRIDE_W - PAD_W + kw[None, :] * DIL_W

        loss = tl.load(
            loss_ptr
            + co * loss_stride_c
            + loss_d * loss_stride_d
            + loss_h * loss_stride_h
            + loss_w * loss_stride_w,
            mask=valid_mn,
            other=0.0,
        )
        image = tl.load(
            image_ptr
            + ci[None, :] * image_stride_c
            + image_d * image_stride_d
            + image_h * image_stride_h
            + image_w * image_stride_w,
            mask=valid_mn,
            other=0.0,
        )
        acc += tl.sum(loss.to(tl.float32) * image.to(tl.float32), axis=0)

    tl.store(
        out_ptr
        + co * out_stride_o
        + ci_rel * out_stride_i
        + kd * out_stride_d
        + kh * out_stride_h
        + kw * out_stride_w,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_n,
    )


@triton.jit
def _conv_wgrad3d_kw3_atomic_kernel(
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
    COUT_PER_GROUP: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
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
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    plane = tl.program_id(1)
    split = tl.program_id(2)
    num_ci_blocks = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    pid_co = pid // num_ci_blocks
    pid_ci = pid - pid_co * num_ci_blocks
    kd = plane // KH
    kh = plane - kd * KH
    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    mask_co = offs_co < COUT_PER_GROUP
    mask_ci = offs_ci < CIN_PER_GROUP
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
        tmp = tmp // LOSS_H
        loss_d = tmp % LOSS_D
        n_idx = tmp // LOSS_D
        image_d = loss_d * STRIDE_D - PAD_D + kd * DIL_D
        image_h = loss_h * STRIDE_H - PAD_H + kh * DIL_H
        image_w0 = loss_w * STRIDE_W - PAD_W + 0 * DIL_W
        image_w1 = loss_w * STRIDE_W - PAD_W + 1 * DIL_W
        image_w2 = loss_w * STRIDE_W - PAD_W + 2 * DIL_W
        valid_dh = (
            (image_d >= 0)
            & (image_d < IMAGE_D)
            & (image_h >= 0)
            & (image_h < IMAGE_H)
        )
        valid0 = valid_dh & (image_w0 >= 0) & (image_w0 < IMAGE_W)
        valid1 = valid_dh & (image_w1 >= 0) & (image_w1 < IMAGE_W)
        valid2 = valid_dh & (image_w2 >= 0) & (image_w2 < IMAGE_W)
        safe_d = tl.where(valid_dh, image_d, 0)
        safe_h = tl.where(valid_dh, image_h, 0)
        safe_w0 = tl.where(valid0, image_w0, 0)
        safe_w1 = tl.where(valid1, image_w1, 0)
        safe_w2 = tl.where(valid2, image_w2, 0)
        loss = tl.load(
            loss_ptr
            + n_idx[None, :] * loss_stride_n
            + offs_co[:, None] * loss_stride_c
            + loss_d[None, :] * loss_stride_d
            + loss_h[None, :] * loss_stride_h
            + loss_w[None, :] * loss_stride_w,
            mask=mask_co[:, None] & mask_m[None, :],
            other=0.0,
        )
        img0 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + offs_ci[None, :] * image_stride_c
            + safe_d[:, None] * image_stride_d
            + safe_h[:, None] * image_stride_h
            + safe_w0[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid0[:, None],
            other=0.0,
        )
        img1 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + offs_ci[None, :] * image_stride_c
            + safe_d[:, None] * image_stride_d
            + safe_h[:, None] * image_stride_h
            + safe_w1[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid1[:, None],
            other=0.0,
        )
        img2 = tl.load(
            image_ptr
            + n_idx[:, None] * image_stride_n
            + offs_ci[None, :] * image_stride_c
            + safe_d[:, None] * image_stride_d
            + safe_h[:, None] * image_stride_h
            + safe_w2[:, None] * image_stride_w,
            mask=mask_m[:, None] & mask_ci[None, :] & valid2[:, None],
            other=0.0,
        )
        acc0 += tl.dot(
            loss, img0, out_dtype=tl.float32, input_precision="tf32"
        )
        acc1 += tl.dot(
            loss, img1, out_dtype=tl.float32, input_precision="tf32"
        )
        acc2 += tl.dot(
            loss, img2, out_dtype=tl.float32, input_precision="tf32"
        )
    mask = mask_co[:, None] & mask_ci[None, :]
    base = (
        out_ptr
        + offs_co[:, None] * out_stride_o
        + offs_ci[None, :] * out_stride_i
        + kd * out_stride_d
        + kh * out_stride_h
    )
    tl.atomic_add(base + 0 * out_stride_w, acc0, sem="relaxed", mask=mask)
    tl.atomic_add(base + 1 * out_stride_w, acc1, sem="relaxed", mask=mask)
    tl.atomic_add(base + 2 * out_stride_w, acc2, sem="relaxed", mask=mask)


def _single(value: Any, name: str) -> int:
    if isinstance(value, int):
        return int(value)
    values = tuple(int(item) for item in value)
    if len(values) != 1:
        raise RuntimeError(f"{name} must have length 1, got {value}")
    return values[0]


def _pair(value: Any, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return int(value), int(value)
    values = tuple(int(item) for item in value)
    if len(values) != 2:
        raise RuntimeError(f"{name} must have length 2, got {value}")
    return values


def _triple(value: Any, name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        return int(value), int(value), int(value)
    values = tuple(int(item) for item in value)
    if len(values) != 3:
        raise RuntimeError(f"{name} must have length 3, got {value}")
    return values


def _padding_1d(attrs: dict[str, Any]) -> tuple[int, int]:
    pre_padding = attrs.get("pre_padding")
    post_padding = attrs.get("post_padding")
    if pre_padding is not None or post_padding is not None:
        if pre_padding is None or post_padding is None:
            raise RuntimeError(
                "both pre_padding and post_padding are required"
            )
        return _single(pre_padding, "pre_padding"), _single(
            post_padding, "post_padding"
        )
    padding = attrs.get("padding", 0)
    if padding is None:
        padding = 0
    if isinstance(padding, str):
        raise RuntimeError("string padding is not a prepared NVIDIA fast path")
    if isinstance(padding, int):
        return int(padding), int(padding)
    values = tuple(int(item) for item in padding)
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values
    raise RuntimeError(f"padding must have length 1 or 2, got {padding}")


def _is_cross_correlation(value: Any) -> bool:
    if value is None:
        return True
    return str(value).rsplit(".", 1)[-1].upper() == "CROSS_CORRELATION"


def _padding_2d(attrs: dict[str, Any]) -> tuple[int, int, int, int]:
    pre_padding = attrs.get("pre_padding")
    post_padding = attrs.get("post_padding")
    if pre_padding is not None or post_padding is not None:
        if pre_padding is None or post_padding is None:
            raise RuntimeError(
                "both pre_padding and post_padding are required"
            )
        pre = _pair(pre_padding, "pre_padding")
        post = _pair(post_padding, "post_padding")
        return pre[0], post[0], pre[1], post[1]
    padding = attrs.get("padding", 0)
    if padding is None:
        padding = 0
    if isinstance(padding, str):
        raise RuntimeError("string padding is not a prepared NVIDIA fast path")
    if isinstance(padding, int):
        value = int(padding)
        return value, value, value, value
    values = tuple(int(item) for item in padding)
    if len(values) == 2:
        return values[0], values[0], values[1], values[1]
    if len(values) == 4:
        return values
    raise RuntimeError(f"padding must have length 2 or 4, got {padding}")


def _padding_3d(
    attrs: dict[str, Any],
) -> tuple[int, int, int, int, int, int]:
    pre_padding = attrs.get("pre_padding")
    post_padding = attrs.get("post_padding")
    if pre_padding is not None or post_padding is not None:
        if pre_padding is None or post_padding is None:
            raise RuntimeError(
                "both pre_padding and post_padding are required"
            )
        pre = _triple(pre_padding, "pre_padding")
        post = _triple(post_padding, "post_padding")
        return pre[0], post[0], pre[1], post[1], pre[2], post[2]
    padding = attrs.get("padding", 0)
    if padding is None:
        padding = 0
    if isinstance(padding, str):
        raise RuntimeError("string padding is not a prepared NVIDIA fast path")
    if isinstance(padding, int):
        value = int(padding)
        return value, value, value, value, value, value
    values = tuple(int(item) for item in padding)
    if len(values) == 3:
        return (
            values[0],
            values[0],
            values[1],
            values[1],
            values[2],
            values[2],
        )
    if len(values) == 6:
        return values
    raise RuntimeError(f"padding must have length 3 or 6, got {padding}")


def _make_bound_pipeline_run_fn(
    spec: PreparedKernelPipelineSpec,
    default_run_fn: RunFn,
    *,
    validate_inputs: bool,
) -> RunFn:
    """NVIDIA-local pipeline wrapper that hoists context creation into bind."""
    launch = make_kernel_pipeline_launcher(spec)

    def can_run(inputs: Sequence[Any]) -> bool:
        if validate_inputs:
            if not runtime_tensor_checks_pass(inputs, spec.input_checks):
                return False
            if spec.extra_check is not None and not spec.extra_check(inputs):
                return False
        index = spec.device_input_index
        return index < len(inputs) and isinstance(inputs[index], torch.Tensor)

    def bind(inputs: Sequence[Any], run_attrs: dict[str, Any]) -> Any:
        if not can_run(inputs):
            return lambda: default_run_fn(inputs, run_attrs)
        source = inputs[spec.device_input_index]
        assert isinstance(source, torch.Tensor)
        context = spec.context_factory(inputs)

        def run_bound() -> Any:
            if spec.pre_launch is not None:
                spec.pre_launch()
            launch(source.device, inputs, context)
            return spec.result(context)

        return run_bound

    def run(inputs: Sequence[Any], run_attrs: dict[str, Any]) -> Any:
        if not can_run(inputs):
            return default_run_fn(inputs, run_attrs)
        source = inputs[spec.device_input_index]
        assert isinstance(source, torch.Tensor)
        if spec.pre_launch is not None:
            spec.pre_launch()
        context = spec.context_factory(inputs)
        launch(source.device, inputs, context)
        return spec.result(context)

    setattr(run, "bind", bind)
    setattr(run, "_flagdnn_functional_output_safe", True)
    return run


def _prepare_fprop_2d_im2col(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
    checks: tuple[Any, ...],
    image_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    output_h: int,
    output_w: int,
) -> RunFn:
    n, c_in, input_h, input_w = image_shape
    c_out, cin_per_group, kernel_h, kernel_w = weight_shape
    output_shape = (n, c_out, output_h, output_w)
    output_hw = output_h * output_w
    kdim = cin_per_group * kernel_h * kernel_w
    total = kdim * output_hw
    output_dtype = torch_dtype(input_specs[0].dtype)
    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    workspace_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def context_factory(
        inputs: Sequence[Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = inputs[0]
        assert isinstance(image, torch.Tensor)
        device_key = (image.device.type, image.device.index, output_dtype)
        output = get_prepared_output(
            output_cache,
            device_key + (output_shape,),
            lambda: torch.empty(
                output_shape, device=image.device, dtype=output_dtype
            ),
        )
        cols_shape = (kdim, output_hw)
        cols_key = device_key + (cols_shape,)
        cols = workspace_cache.get(cols_key)
        if cols is None:
            cols = torch.empty(
                cols_shape, device=image.device, dtype=output_dtype
            )
            workspace_cache[cols_key] = cols
        return output, cols

    block_size = 512 if input_specs[0].dtype == "float32" else 1024
    im2col_grid = ((total + block_size - 1) // block_size, 1, 1)
    im2col_cached_args = (
        total,
        input_h,
        input_w,
        output_h,
        output_w,
        cin_per_group,
        block_size,
    )

    def im2col_args(
        inputs: Sequence[Any], context: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[Any, ...]:
        return inputs[0], context[1]

    def build_im2col_cached_call(
        _metadata: dict[str, Any],
    ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        return im2col_grid, im2col_cached_args

    if input_specs[0].dtype == "float32":
        block_m = 32
        block_n = 128 if cin_per_group >= 768 else 64
    else:
        block_m = 64
        block_n = 64
    block_k = 128
    group_m = 8
    matmul_grid = (
        ((c_out + block_m - 1) // block_m)
        * ((output_hw + block_n - 1) // block_n),
        1,
        1,
    )
    matmul_cached_args = (
        c_out,
        output_hw,
        kdim,
        block_m,
        block_n,
        block_k,
        group_m,
        False,
    )

    def matmul_args(
        inputs: Sequence[Any], context: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[Any, ...]:
        return inputs[1], context[1], context[0]

    def build_matmul_cached_call(
        _metadata: dict[str, Any],
    ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        return matmul_grid, matmul_cached_args

    def extra_check(inputs: Sequence[Any]) -> bool:
        image, weight = inputs
        return (
            isinstance(image, torch.Tensor)
            and isinstance(weight, torch.Tensor)
            and image.device == weight.device
        )

    pipeline = PreparedKernelPipelineSpec(
        steps=(
            PreparedPipelineStepSpec(
                kernel=conv2d_spatial_nchw_3x3_stride2_pad1_im2col_kernel,
                grid=im2col_grid,
                runtime_args=im2col_args,
                static_args=im2col_cached_args[:-1],
                constexpr_kwargs={
                    "BLOCK_SIZE": block_size,
                    "num_warps": 4,
                },
                build_cached_call=build_im2col_cached_call,
            ),
            PreparedPipelineStepSpec(
                kernel=_batched_matmul_kernel.fn.fn,
                grid=matmul_grid,
                runtime_args=matmul_args,
                static_args=matmul_cached_args[:3],
                constexpr_kwargs={
                    "BLOCK_M": block_m,
                    "BLOCK_N": block_n,
                    "BLOCK_K": block_k,
                    "GROUP_M": group_m,
                    "ROUND_F32_TO_TF32": False,
                    "num_warps": 4,
                    "num_stages": 3,
                },
                build_cached_call=build_matmul_cached_call,
            ),
        ),
        input_checks=checks,
        context_factory=context_factory,
        result=lambda context: context[0],
        extra_check=extra_check,
    )
    return _make_bound_pipeline_run_fn(
        pipeline,
        default_run_fn,
        validate_inputs=bool(attrs.get("_validate_inputs", True)),
    )


def _prepare_fprop_1d(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if len(input_specs) != 2 or not all(
        _is_runtime_device_spec(spec) for spec in input_specs
    ):
        return None
    image_spec, weight_spec = input_specs
    image_shape = _static_shape(image_spec)
    weight_shape = _static_shape(weight_spec)
    if (
        image_shape is None
        or weight_shape is None
        or len(image_shape) != 3
        or len(weight_shape) != 3
        or image_spec.contiguous is not True
        or weight_spec.contiguous is not True
        or image_spec.stride is None
        or weight_spec.stride is None
        or image_spec.dtype not in _FPROP_DTYPES
        or weight_spec.dtype != image_spec.dtype
        or not _is_cross_correlation(attrs.get("convolution_mode"))
    ):
        return None

    batch, c_in, input_l = image_shape
    c_out, weight_c, kernel_w = weight_shape
    groups = int(attrs.get("groups", 1))
    stride_w = _single(attrs.get("stride", 1), "stride")
    dilation_w = _single(attrs.get("dilation", 1), "dilation")
    pad_left, pad_right = _padding_1d(attrs)
    if (
        min(batch, c_in, input_l, c_out, weight_c, kernel_w) <= 0
        or groups <= 0
        or c_in % groups != 0
        or c_out % groups != 0
        or weight_c != c_in // groups
        or stride_w <= 0
        or dilation_w <= 0
    ):
        return None
    out_l = (
        input_l + pad_left + pad_right - dilation_w * (kernel_w - 1) - 1
    ) // stride_w + 1
    if out_l <= 0:
        return None

    checks = runtime_tensor_checks_from_specs(
        input_specs,
        (0, 1),
        require_shape=True,
        require_stride=True,
        require_dtype=True,
    )
    if checks is None:
        return None

    output_shape = (batch, c_out, out_l)
    output_dtype = torch_dtype(image_spec.dtype)
    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    dtype_id = {"float16": 0, "bfloat16": 1, "float32": 2}[image_spec.dtype]
    cin_per_group = c_in // groups
    cout_per_group = c_out // groups
    m = batch * out_l

    def output_factory(inputs: Sequence[Any]) -> torch.Tensor:
        image = inputs[0]
        assert isinstance(image, torch.Tensor)
        key = (
            image.device.type,
            image.device.index,
            output_dtype,
            output_shape,
        )
        return get_prepared_output(
            output_cache,
            key,
            lambda: torch.empty(
                output_shape, device=image.device, dtype=output_dtype
            ),
        )

    def runtime_args(
        inputs: Sequence[Any], output: torch.Tensor
    ) -> tuple[Any, ...]:
        image, weight = inputs
        assert isinstance(image, torch.Tensor)
        assert isinstance(weight, torch.Tensor)
        return (
            image,
            weight,
            output,
            output,
            m,
            input_l,
            out_l,
            dtype_id,
            image.stride(0),
            image.stride(1),
            image.stride(2),
            weight.stride(0),
            weight.stride(1),
            weight.stride(2),
            0,
            output.stride(0),
            output.stride(1),
            output.stride(2),
        )

    def grid(meta: dict[str, Any]) -> tuple[int, int]:
        return (
            (m + int(meta["BLOCK_M"]) - 1)
            // int(meta["BLOCK_M"])
            * (
                (cout_per_group + int(meta["BLOCK_OC"]) - 1)
                // int(meta["BLOCK_OC"])
            ),
            groups,
        )

    def build_cached_call(
        constexprs: dict[str, Any],
    ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
        block_m = int(constexprs["BLOCK_M"])
        block_oc = int(constexprs["BLOCK_OC"])
        block_k = int(constexprs["BLOCK_K"])
        static_grid = (
            (m + block_m - 1)
            // block_m
            * ((cout_per_group + block_oc - 1) // block_oc),
            groups,
            1,
        )
        return static_grid, (
            cin_per_group,
            cout_per_group,
            kernel_w,
            stride_w,
            pad_left,
            dilation_w,
            False,
            block_m,
            block_oc,
            block_k,
            _GROUP_M,
        )

    def extra_check(inputs: Sequence[Any]) -> bool:
        image, weight = inputs
        return (
            isinstance(image, torch.Tensor)
            and isinstance(weight, torch.Tensor)
            and image.device == weight.device
        )

    return make_single_kernel_run_fn(
        PreparedSingleKernelRunSpec(
            kernel=PreparedSingleKernelSpec(
                kernel=conv1d_gemm_kernel,
                grid=grid,
                static_args=(
                    cin_per_group,
                    cout_per_group,
                    kernel_w,
                    stride_w,
                    pad_left,
                    dilation_w,
                ),
                constexpr_kwargs={"HAS_BIAS": False, "GROUP_M": _GROUP_M},
                build_cached_call=build_cached_call,
            ),
            input_checks=checks,
            output_factory=output_factory,
            runtime_args=runtime_args,
            extra_check=extra_check,
            validate_inputs=bool(attrs.get("_validate_inputs", True)),
        ),
        default_run_fn,
    )


def _prepare_dgrad_1d(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if len(input_specs) != 2 or not all(
        _is_runtime_device_spec(spec) for spec in input_specs
    ):
        return None
    loss_spec, weight_spec = input_specs
    loss_shape = _static_shape(loss_spec)
    weight_shape = _static_shape(weight_spec)
    input_size = tuple(int(dim) for dim in attrs.get("input_size", ()))
    if (
        loss_shape is None
        or weight_shape is None
        or len(input_size) != 3
        or len(loss_shape) != 3
        or len(weight_shape) != 3
        or loss_spec.contiguous is not True
        or weight_spec.contiguous is not True
        or loss_spec.stride is None
        or weight_spec.stride is None
        or loss_spec.dtype not in _FPROP_DTYPES
        or weight_spec.dtype != loss_spec.dtype
        or not _is_cross_correlation(attrs.get("convolution_mode"))
    ):
        return None

    n, c_in, x_len = input_size
    loss_n, c_out, loss_len = loss_shape
    weight_c_out, weight_c_in, kernel_l = weight_shape
    groups = int(attrs.get("groups", 1))
    stride_l = _single(attrs.get("stride", 1), "stride")
    dilation_l = _single(attrs.get("dilation", 1), "dilation")
    pad_left, pad_right = _padding_1d(attrs)
    if (
        min(n, c_in, x_len, loss_n, c_out, loss_len, kernel_l) <= 0
        or loss_n != n
        or weight_c_out != c_out
        or groups <= 0
        or c_in % groups != 0
        or c_out % groups != 0
        or weight_c_in != c_in // groups
        or stride_l <= 0
        or dilation_l <= 0
    ):
        return None
    expected_loss_len = (
        x_len + pad_left + pad_right - dilation_l * (kernel_l - 1) - 1
    ) // stride_l + 1
    if expected_loss_len != loss_len:
        return None

    checks = runtime_tensor_checks_from_specs(
        input_specs,
        (0, 1),
        require_shape=True,
        require_stride=True,
        require_dtype=True,
    )
    if checks is None:
        return None

    output_shape = input_size
    output_dtype = torch_dtype(loss_spec.dtype)
    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    dtype_id = {"float16": 0, "bfloat16": 1, "float32": 2}[loss_spec.dtype]
    cin_per_group = c_in // groups
    cout_per_group = c_out // groups
    m = n * x_len
    output_stride = (c_in * x_len, x_len, 1)

    def output_factory(inputs: Sequence[Any]) -> torch.Tensor:
        loss = inputs[0]
        assert isinstance(loss, torch.Tensor)
        key = (
            loss.device.type,
            loss.device.index,
            output_dtype,
            output_shape,
        )
        return get_prepared_output(
            output_cache,
            key,
            lambda: torch.empty(
                output_shape, device=loss.device, dtype=output_dtype
            ),
        )

    def runtime_args(
        inputs: Sequence[Any], output: torch.Tensor
    ) -> tuple[Any, ...]:
        return inputs[0], inputs[1], output

    def grid(meta: dict[str, Any]) -> tuple[int, int]:
        return (
            ((m + int(meta["BLOCK_M"]) - 1) // int(meta["BLOCK_M"]))
            * (
                (cin_per_group + int(meta["BLOCK_CI"]) - 1)
                // int(meta["BLOCK_CI"])
            ),
            groups,
        )

    static_args = (
        m,
        x_len,
        loss_len,
        c_in,
        c_out,
        cin_per_group,
        cout_per_group,
        *loss_spec.stride,
        *weight_spec.stride,
        *output_stride,
        stride_l,
        pad_left,
        dilation_l,
        kernel_l,
        False,
    )

    def build_cached_call(
        constexprs: dict[str, Any],
    ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
        block_m = int(constexprs["BLOCK_M"])
        block_ci = int(constexprs["BLOCK_CI"])
        block_co = int(constexprs["BLOCK_CO"])
        return (
            ((m + block_m - 1) // block_m)
            * ((cin_per_group + block_ci - 1) // block_ci),
            groups,
            1,
        ), static_args + (dtype_id, block_m, block_ci, block_co)

    def extra_check(inputs: Sequence[Any]) -> bool:
        loss, weight = inputs
        return (
            isinstance(loss, torch.Tensor)
            and isinstance(weight, torch.Tensor)
            and loss.device == weight.device
        )

    return make_single_kernel_run_fn(
        PreparedSingleKernelRunSpec(
            kernel=PreparedSingleKernelSpec(
                kernel=_conv_dgrad1d_mci_kernel,
                grid=grid,
                static_args=static_args,
                constexpr_kwargs={"DTYPE_ID": dtype_id},
                build_cached_call=build_cached_call,
            ),
            input_checks=checks,
            output_factory=output_factory,
            runtime_args=runtime_args,
            extra_check=extra_check,
            validate_inputs=bool(attrs.get("_validate_inputs", True)),
        ),
        default_run_fn,
    )


def _prepare_dgrad_2d_stride1(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if len(input_specs) != 2 or not all(
        _is_runtime_device_spec(spec) for spec in input_specs
    ):
        return None
    loss_spec, weight_spec = input_specs
    loss_shape = _static_shape(loss_spec)
    weight_shape = _static_shape(weight_spec)
    input_size = tuple(int(dim) for dim in attrs.get("input_size", ()))
    if (
        loss_shape is None
        or weight_shape is None
        or len(input_size) != 4
        or len(loss_shape) != 4
        or len(weight_shape) != 4
        or loss_spec.contiguous is not True
        or weight_spec.contiguous is not True
        or loss_spec.stride is None
        or weight_spec.stride is None
        or loss_spec.dtype not in _FPROP_DTYPES
        or weight_spec.dtype != loss_spec.dtype
        or not _is_cross_correlation(attrs.get("convolution_mode"))
    ):
        return None

    n, c_in, xh, xw = input_size
    loss_n, c_out, loss_h, loss_w = loss_shape
    weight_c_out, weight_c_in, kernel_h, kernel_w = weight_shape
    groups = int(attrs.get("groups", 1))
    stride_h, stride_w = _pair(attrs.get("stride", 1), "stride")
    if (stride_h, stride_w) != (1, 1):
        return None
    dilation_h, dilation_w = _pair(attrs.get("dilation", 1), "dilation")
    pad_top, pad_bottom, pad_left, pad_right = _padding_2d(attrs)
    if (
        min(n, c_in, xh, xw, loss_n, c_out, loss_h, loss_w) <= 0
        or min(kernel_h, kernel_w) <= 0
        or loss_n != n
        or weight_c_out != c_out
        or groups <= 0
        or c_in % groups != 0
        or c_out % groups != 0
        or weight_c_in != c_in // groups
        or dilation_h <= 0
        or dilation_w <= 0
    ):
        return None
    expected_loss_h = (
        xh + pad_top + pad_bottom - dilation_h * (kernel_h - 1) - 1
    ) // stride_h + 1
    expected_loss_w = (
        xw + pad_left + pad_right - dilation_w * (kernel_w - 1) - 1
    ) // stride_w + 1
    if (expected_loss_h, expected_loss_w) != (loss_h, loss_w):
        return None

    checks = runtime_tensor_checks_from_specs(
        input_specs,
        (0, 1),
        require_shape=True,
        require_stride=True,
        require_dtype=True,
    )
    if checks is None:
        return None

    output_shape = input_size
    output_dtype = torch_dtype(loss_spec.dtype)
    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    dtype_id = {"float16": 0, "bfloat16": 1, "float32": 2}[loss_spec.dtype]
    cin_per_group = c_in // groups
    cout_per_group = c_out // groups
    m = n * xh * xw
    output_stride = (c_in * xh * xw, xh * xw, xw, 1)

    def output_factory(inputs: Sequence[Any]) -> torch.Tensor:
        loss = inputs[0]
        assert isinstance(loss, torch.Tensor)
        key = (
            loss.device.type,
            loss.device.index,
            output_dtype,
            output_shape,
        )
        return get_prepared_output(
            output_cache,
            key,
            lambda: torch.empty(
                output_shape, device=loss.device, dtype=output_dtype
            ),
        )

    def runtime_args(
        inputs: Sequence[Any], output: torch.Tensor
    ) -> tuple[Any, ...]:
        return inputs[0], inputs[1], output

    def grid(meta: dict[str, Any]) -> tuple[int, int]:
        return (
            ((m + int(meta["BLOCK_M"]) - 1) // int(meta["BLOCK_M"]))
            * (
                (cin_per_group + int(meta["BLOCK_CI"]) - 1)
                // int(meta["BLOCK_CI"])
            ),
            groups,
        )

    is_1x1 = (
        kernel_h == 1
        and kernel_w == 1
        and (pad_top, pad_bottom, pad_left, pad_right) == (0, 0, 0, 0)
        and (dilation_h, dilation_w) == (1, 1)
    )
    if is_1x1:
        kernel = (
            _conv_dgrad2d_1x1_strided_kernel
            if loss_spec.dtype == "float32"
            else _conv_dgrad2d_1x1_kernel
        )
        static_args = (
            m,
            xh,
            xw,
            c_in,
            c_out,
            cin_per_group,
            cout_per_group,
            *loss_spec.stride,
            weight_spec.stride[0],
            weight_spec.stride[1],
            *output_stride,
        )
    else:
        kernel = _conv_dgrad2d_stride1_kernel
        static_args = (
            m,
            xh,
            xw,
            loss_h,
            loss_w,
            c_in,
            c_out,
            cin_per_group,
            cout_per_group,
            *loss_spec.stride,
            *weight_spec.stride,
            *output_stride,
            pad_top,
            pad_left,
            dilation_h,
            dilation_w,
            kernel_h,
            kernel_w,
            False,
        )

    def build_cached_call(
        constexprs: dict[str, Any],
    ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
        block_m = int(constexprs["BLOCK_M"])
        block_ci = int(constexprs["BLOCK_CI"])
        block_co = int(constexprs["BLOCK_CO"])
        return (
            ((m + block_m - 1) // block_m)
            * ((cin_per_group + block_ci - 1) // block_ci),
            groups,
            1,
        ), static_args + (dtype_id, block_m, block_ci, block_co)

    def extra_check(inputs: Sequence[Any]) -> bool:
        loss, weight = inputs
        return (
            isinstance(loss, torch.Tensor)
            and isinstance(weight, torch.Tensor)
            and loss.device == weight.device
        )

    return make_single_kernel_run_fn(
        PreparedSingleKernelRunSpec(
            kernel=PreparedSingleKernelSpec(
                kernel=kernel,
                grid=grid,
                static_args=static_args,
                constexpr_kwargs={"DTYPE_ID": dtype_id},
                build_cached_call=build_cached_call,
            ),
            input_checks=checks,
            output_factory=output_factory,
            runtime_args=runtime_args,
            extra_check=extra_check,
            validate_inputs=bool(attrs.get("_validate_inputs", True)),
        ),
        default_run_fn,
    )


def _prepare_dgrad_2d_tile2w_pipeline(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
    checks: tuple[Any, ...],
    input_size: tuple[int, ...],
    loss_shape: tuple[int, ...],
) -> RunFn:
    n, c_in, xh, xw = input_size
    _, c_out, loss_h, loss_w = loss_shape
    groups = int(attrs.get("groups", 1))
    cin_per_group = c_in // groups
    cout_per_group = c_out // groups
    output_dtype = torch_dtype(input_specs[0].dtype)
    dtype_id = {"float16": 0, "bfloat16": 1, "float32": 2}[
        input_specs[0].dtype
    ]
    output_stride = (c_in * xh * xw, xh * xw, xw, 1)
    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def context_factory(
        inputs: Sequence[Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss, weight = inputs
        assert isinstance(loss, torch.Tensor)
        assert isinstance(weight, torch.Tensor)
        key = (
            loss.device.type,
            loss.device.index,
            output_dtype,
            input_size,
        )
        output = get_prepared_output(
            output_cache,
            key,
            lambda: torch.empty(
                input_size, device=loss.device, dtype=output_dtype
            ),
        )
        packed_weight = _pack_weight_2d_khw_oci(weight, groups)
        return output, packed_weight

    def make_step(ph: int) -> PreparedPipelineStepSpec:
        parity_h_count = (xh + 1 - ph) // 2
        m = n * parity_h_count * loss_w

        def grid(meta: dict[str, Any]) -> tuple[int, ...]:
            return (
                ((m + int(meta["BLOCK_M"]) - 1) // int(meta["BLOCK_M"]))
                * (
                    (cin_per_group + int(meta["BLOCK_CI"]) - 1)
                    // int(meta["BLOCK_CI"])
                ),
            )

        static_args = (
            m,
            xh,
            xw,
            loss_h,
            loss_w,
            c_in,
            c_out,
            cin_per_group,
            cout_per_group,
            *input_specs[0].stride,
            *output_stride,
            parity_h_count,
            ph,
            False,
        )

        def runtime_args(
            inputs: Sequence[Any], context: tuple[torch.Tensor, torch.Tensor]
        ) -> tuple[Any, ...]:
            return inputs[0], context[1], context[0]

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
            block_m = int(constexprs["BLOCK_M"])
            block_ci = int(constexprs["BLOCK_CI"])
            block_co = int(constexprs["BLOCK_CO"])
            static_grid = (
                ((m + block_m - 1) // block_m)
                * ((cin_per_group + block_ci - 1) // block_ci),
                1,
                1,
            )
            return static_grid, static_args + (
                dtype_id,
                block_m,
                block_ci,
                block_co,
            )

        return PreparedPipelineStepSpec(
            kernel=_conv_dgrad2d_stride2_pad1_3x3_tile2w_kernel,
            grid=grid,
            runtime_args=runtime_args,
            static_args=static_args,
            constexpr_kwargs={"DTYPE_ID": dtype_id},
            build_cached_call=build_cached_call,
            first_launch_returns_metadata=True,
        )

    def extra_check(inputs: Sequence[Any]) -> bool:
        loss, weight = inputs
        return (
            isinstance(loss, torch.Tensor)
            and isinstance(weight, torch.Tensor)
            and loss.device == weight.device
        )

    pipeline = PreparedKernelPipelineSpec(
        steps=(make_step(0), make_step(1)),
        input_checks=checks,
        context_factory=context_factory,
        result=lambda context: context[0],
        extra_check=extra_check,
    )
    return _make_bound_pipeline_run_fn(
        pipeline,
        default_run_fn,
        validate_inputs=bool(attrs.get("_validate_inputs", True)),
    )


def _prepare_dgrad_2d_stride2(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if len(input_specs) != 2 or not all(
        _is_runtime_device_spec(spec) for spec in input_specs
    ):
        return None
    loss_spec, weight_spec = input_specs
    loss_shape = _static_shape(loss_spec)
    weight_shape = _static_shape(weight_spec)
    input_size = tuple(int(dim) for dim in attrs.get("input_size", ()))
    if (
        loss_shape is None
        or weight_shape is None
        or len(input_size) != 4
        or len(loss_shape) != 4
        or len(weight_shape) != 4
        or loss_spec.contiguous is not True
        or weight_spec.contiguous is not True
        or loss_spec.stride is None
        or weight_spec.stride is None
        or loss_spec.dtype not in _FPROP_DTYPES
        or weight_spec.dtype != loss_spec.dtype
        or not _is_cross_correlation(attrs.get("convolution_mode"))
    ):
        return None

    n, c_in, xh, xw = input_size
    loss_n, c_out, loss_h, loss_w = loss_shape
    weight_c_out, weight_c_in, kernel_h, kernel_w = weight_shape
    groups = int(attrs.get("groups", 1))
    stride_h, stride_w = _pair(attrs.get("stride", 1), "stride")
    dilation_h, dilation_w = _pair(attrs.get("dilation", 1), "dilation")
    pad_top, pad_bottom, pad_left, pad_right = _padding_2d(attrs)
    cin_per_group = c_in // groups if groups > 0 else 0
    cout_per_group = c_out // groups if groups > 0 else 0
    if (
        min(n, c_in, xh, xw, loss_n, c_out, loss_h, loss_w) <= 0
        or loss_n != n
        or weight_c_out != c_out
        or groups != 1
        or weight_c_in != cin_per_group
        or (stride_h, stride_w) != (2, 2)
        or (pad_top, pad_bottom, pad_left, pad_right) != (1, 1, 1, 1)
        or (dilation_h, dilation_w) != (1, 1)
        or (kernel_h, kernel_w) != (3, 3)
    ):
        return None
    expected_loss_h = (xh + 2 - 3) // 2 + 1
    expected_loss_w = (xw + 2 - 3) // 2 + 1
    if (expected_loss_h, expected_loss_w) != (loss_h, loss_w):
        return None

    checks = runtime_tensor_checks_from_specs(
        input_specs,
        (0, 1),
        require_shape=True,
        require_stride=True,
        require_dtype=True,
    )
    if checks is None:
        return None

    loss_hw = loss_h * loss_w
    if cin_per_group > 512:
        if (
            loss_spec.dtype in ("float16", "bfloat16")
            and n == 1
            and (xh, xw) == (40, 40)
            and (loss_h, loss_w) == (20, 20)
            and cin_per_group == 768
            and cout_per_group == 768
        ):
            return _prepare_dgrad_2d_tile2w_pipeline(
                attrs,
                input_specs,
                default_run_fn,
                checks,
                input_size,
                loss_shape,
            )
        return None
    if (
        loss_spec.dtype == "float32"
        and cin_per_group == cout_per_group
        and cin_per_group == 512
        and loss_hw <= 1024
    ):
        return None
    use_tile4 = cin_per_group == 3 or (
        128 <= cin_per_group <= 512 and loss_hw <= 1024
    )

    output_shape = input_size
    output_dtype = torch_dtype(loss_spec.dtype)
    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    dtype_id = {"float16": 0, "bfloat16": 1, "float32": 2}[loss_spec.dtype]
    output_stride = (c_in * xh * xw, xh * xw, xw, 1)

    def output_factory(inputs: Sequence[Any]) -> torch.Tensor:
        loss = inputs[0]
        assert isinstance(loss, torch.Tensor)
        key = (
            loss.device.type,
            loss.device.index,
            output_dtype,
            output_shape,
        )
        return get_prepared_output(
            output_cache,
            key,
            lambda: torch.empty(
                output_shape, device=loss.device, dtype=output_dtype
            ),
        )

    def runtime_args(
        inputs: Sequence[Any], output: torch.Tensor
    ) -> tuple[Any, ...]:
        weight = inputs[1]
        assert isinstance(weight, torch.Tensor)
        packed_weight = _pack_weight_2d_khw_oci(weight, groups)
        return inputs[0], packed_weight, output

    if use_tile4:
        m = n * loss_hw

        def grid(meta: dict[str, Any]) -> tuple[int, ...]:
            return (
                ((m + int(meta["BLOCK_M"]) - 1) // int(meta["BLOCK_M"]))
                * (
                    (cin_per_group + int(meta["BLOCK_CI"]) - 1)
                    // int(meta["BLOCK_CI"])
                ),
                groups,
            )

        kernel = _conv_dgrad2d_stride2_pad1_3x3_tile4_kernel
        static_args = (
            m,
            xh,
            xw,
            loss_h,
            loss_w,
            c_in,
            c_out,
            cin_per_group,
            cout_per_group,
            *loss_spec.stride,
            cin_per_group,
            1,
            3 * cout_per_group * cin_per_group,
            cout_per_group * cin_per_group,
            *output_stride,
            False,
        )

        def static_grid(block_m: int, block_ci: int) -> tuple[int, int, int]:
            return (
                ((m + block_m - 1) // block_m)
                * ((cin_per_group + block_ci - 1) // block_ci),
                groups,
                1,
            )

    else:
        max_parity_h = (xh + 1) // 2
        max_parity_w = (xw + 1) // 2
        m = n * max_parity_h * max_parity_w

        def grid(meta: dict[str, Any]) -> tuple[int, ...]:
            return (
                ((m + int(meta["BLOCK_M"]) - 1) // int(meta["BLOCK_M"]))
                * (
                    (cin_per_group + int(meta["BLOCK_CI"]) - 1)
                    // int(meta["BLOCK_CI"])
                ),
                4,
                groups,
            )

        kernel = _conv_dgrad2d_stride2_pad1_3x3_packed_kernel
        static_args = (
            m,
            n,
            xh,
            xw,
            loss_h,
            loss_w,
            c_in,
            c_out,
            cin_per_group,
            cout_per_group,
            *loss_spec.stride,
            *output_stride,
            False,
        )

        def static_grid(block_m: int, block_ci: int) -> tuple[int, int, int]:
            return (
                ((m + block_m - 1) // block_m)
                * ((cin_per_group + block_ci - 1) // block_ci),
                4,
                groups,
            )

    def build_cached_call(
        constexprs: dict[str, Any],
    ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
        block_m = int(constexprs["BLOCK_M"])
        block_ci = int(constexprs["BLOCK_CI"])
        block_co = int(constexprs["BLOCK_CO"])
        return static_grid(block_m, block_ci), static_args + (
            dtype_id,
            block_m,
            block_ci,
            block_co,
        )

    def extra_check(inputs: Sequence[Any]) -> bool:
        loss, weight = inputs
        return (
            isinstance(loss, torch.Tensor)
            and isinstance(weight, torch.Tensor)
            and loss.device == weight.device
        )

    return make_single_kernel_run_fn(
        PreparedSingleKernelRunSpec(
            kernel=PreparedSingleKernelSpec(
                kernel=kernel,
                grid=grid,
                static_args=static_args,
                constexpr_kwargs={"DTYPE_ID": dtype_id},
                build_cached_call=build_cached_call,
            ),
            input_checks=checks,
            output_factory=output_factory,
            runtime_args=runtime_args,
            extra_check=extra_check,
            validate_inputs=bool(attrs.get("_validate_inputs", True)),
        ),
        default_run_fn,
    )


def _prepare_dgrad_3d(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if len(input_specs) != 2 or not all(
        _is_runtime_device_spec(spec) for spec in input_specs
    ):
        return None
    loss_spec, weight_spec = input_specs
    loss_shape = _static_shape(loss_spec)
    weight_shape = _static_shape(weight_spec)
    input_size = tuple(int(dim) for dim in attrs.get("input_size", ()))
    if (
        loss_shape is None
        or weight_shape is None
        or len(input_size) != 5
        or len(loss_shape) != 5
        or len(weight_shape) != 5
        or loss_spec.contiguous is not True
        or weight_spec.contiguous is not True
        or loss_spec.stride is None
        or weight_spec.stride is None
        or loss_spec.dtype not in _FPROP_DTYPES
        or weight_spec.dtype != loss_spec.dtype
        or not _is_cross_correlation(attrs.get("convolution_mode"))
    ):
        return None

    n, c_in, xd, xh, xw = input_size
    loss_n, c_out, loss_d, loss_h, loss_w = loss_shape
    weight_c_out, weight_c_in, kernel_d, kernel_h, kernel_w = weight_shape
    groups = int(attrs.get("groups", 1))
    stride_d, stride_h, stride_w = _triple(attrs.get("stride", 1), "stride")
    dilation_d, dilation_h, dilation_w = _triple(
        attrs.get("dilation", 1), "dilation"
    )
    (
        pad_front,
        pad_back,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
    ) = _padding_3d(attrs)
    cin_per_group = c_in // groups if groups > 0 else 0
    cout_per_group = c_out // groups if groups > 0 else 0
    if (
        min(n, c_in, xd, xh, xw, loss_n, c_out, loss_d, loss_h, loss_w) <= 0
        or min(kernel_d, kernel_h, kernel_w) <= 0
        or loss_n != n
        or weight_c_out != c_out
        or groups <= 0
        or c_in % groups != 0
        or c_out % groups != 0
        or weight_c_in != cin_per_group
        or min(stride_d, stride_h, stride_w) <= 0
        or min(dilation_d, dilation_h, dilation_w) <= 0
    ):
        return None
    expected_loss = (
        (xd + pad_front + pad_back - dilation_d * (kernel_d - 1) - 1)
        // stride_d
        + 1,
        (xh + pad_top + pad_bottom - dilation_h * (kernel_h - 1) - 1)
        // stride_h
        + 1,
        (xw + pad_left + pad_right - dilation_w * (kernel_w - 1) - 1)
        // stride_w
        + 1,
    )
    if expected_loss != (loss_d, loss_h, loss_w):
        return None

    use_fp32_ci8_dot = (
        loss_spec.dtype == "float32"
        and groups == 1
        and (stride_d, stride_h, stride_w) == (1, 1, 1)
        and (
            pad_front,
            pad_back,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
        )
        == (1, 1, 1, 1, 1, 1)
        and (dilation_d, dilation_h, dilation_w) == (1, 1, 1)
        and cin_per_group == 8
        and cout_per_group == 16
        and (kernel_d, kernel_h, kernel_w) == (3, 3, 3)
    )
    if loss_spec.dtype == "float32" and not use_fp32_ci8_dot:
        return None

    checks = runtime_tensor_checks_from_specs(
        input_specs,
        (0, 1),
        require_shape=True,
        require_stride=True,
        require_dtype=True,
    )
    if checks is None:
        return None

    output_shape = input_size
    output_dtype = torch_dtype(loss_spec.dtype)
    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    dtype_id = {"float16": 0, "bfloat16": 1, "float32": 2}[loss_spec.dtype]
    spatial = xd * xh * xw
    m = n * spatial
    output_stride = (c_in * spatial, spatial, xh * xw, xw, 1)

    def output_factory(inputs: Sequence[Any]) -> torch.Tensor:
        loss = inputs[0]
        assert isinstance(loss, torch.Tensor)
        key = (
            loss.device.type,
            loss.device.index,
            output_dtype,
            output_shape,
        )
        return get_prepared_output(
            output_cache,
            key,
            lambda: torch.empty(
                output_shape, device=loss.device, dtype=output_dtype
            ),
        )

    def runtime_args(
        inputs: Sequence[Any], output: torch.Tensor
    ) -> tuple[Any, ...]:
        weight = inputs[1]
        assert isinstance(weight, torch.Tensor)
        packed_weight = _pack_weight_3d_kdhw_oci(weight, groups)
        return inputs[0], packed_weight, output

    if use_fp32_ci8_dot:

        def grid(meta: dict[str, Any]) -> tuple[int, ...]:
            block_m = int(meta["BLOCK_M"])
            return ((m + block_m - 1) // block_m,)

        kernel = _conv_dgrad3d_pad1_3x3_fp32_ci8_dot_kernel
        static_args = (
            m,
            xd,
            xh,
            xw,
            loss_d,
            loss_h,
            loss_w,
            *loss_spec.stride,
            *output_stride,
        )
        constexpr_kwargs: dict[str, Any] = {}

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
            block_m = int(constexprs["BLOCK_M"])
            return ((m + block_m - 1) // block_m, 1, 1), static_args + (
                block_m,
            )

    else:

        def grid(meta: dict[str, Any]) -> tuple[int, ...]:
            return (
                ((m + int(meta["BLOCK_M"]) - 1) // int(meta["BLOCK_M"]))
                * (
                    (cin_per_group + int(meta["BLOCK_CI"]) - 1)
                    // int(meta["BLOCK_CI"])
                ),
                groups,
            )

        kernel = _conv_dgrad3d_packed_kernel
        static_args = (
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
            *loss_spec.stride,
            *output_stride,
            stride_d,
            stride_h,
            stride_w,
            pad_front,
            pad_top,
            pad_left,
            dilation_d,
            dilation_h,
            dilation_w,
            kernel_d,
            kernel_h,
            kernel_w,
            False,
        )
        constexpr_kwargs = {"DTYPE_ID": dtype_id}

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
            block_m = int(constexprs["BLOCK_M"])
            block_ci = int(constexprs["BLOCK_CI"])
            block_co = int(constexprs["BLOCK_CO"])
            static_grid = (
                ((m + block_m - 1) // block_m)
                * ((cin_per_group + block_ci - 1) // block_ci),
                groups,
                1,
            )
            return static_grid, static_args + (
                dtype_id,
                block_m,
                block_ci,
                block_co,
            )

    def extra_check(inputs: Sequence[Any]) -> bool:
        loss, weight = inputs
        return (
            isinstance(loss, torch.Tensor)
            and isinstance(weight, torch.Tensor)
            and loss.device == weight.device
        )

    return make_single_kernel_run_fn(
        PreparedSingleKernelRunSpec(
            kernel=PreparedSingleKernelSpec(
                kernel=kernel,
                grid=grid,
                static_args=static_args,
                constexpr_kwargs=constexpr_kwargs,
                build_cached_call=build_cached_call,
            ),
            input_checks=checks,
            output_factory=output_factory,
            runtime_args=runtime_args,
            extra_check=extra_check,
            validate_inputs=bool(attrs.get("_validate_inputs", True)),
        ),
        default_run_fn,
    )


def _prepare_fprop_3d(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if len(input_specs) != 2 or not all(
        _is_runtime_device_spec(spec) for spec in input_specs
    ):
        return None
    image_spec, weight_spec = input_specs
    image_shape = _static_shape(image_spec)
    weight_shape = _static_shape(weight_spec)
    if (
        image_shape is None
        or weight_shape is None
        or len(image_shape) != 5
        or len(weight_shape) != 5
        or image_spec.contiguous is not True
        or weight_spec.contiguous is not True
        or image_spec.stride is None
        or weight_spec.stride is None
        or image_spec.dtype not in _FPROP_DTYPES
        or weight_spec.dtype != image_spec.dtype
        or not _is_cross_correlation(attrs.get("convolution_mode"))
    ):
        return None

    n, c_in, input_d, input_h, input_w = image_shape
    c_out, weight_c, kernel_d, kernel_h, kernel_w = weight_shape
    groups = int(attrs.get("groups", 1))
    stride_d, stride_h, stride_w = _triple(attrs.get("stride", 1), "stride")
    dilation_d, dilation_h, dilation_w = _triple(
        attrs.get("dilation", 1), "dilation"
    )
    (
        pad_front,
        pad_back,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
    ) = _padding_3d(attrs)
    if (
        min(
            n,
            c_in,
            input_d,
            input_h,
            input_w,
            c_out,
            weight_c,
            kernel_d,
            kernel_h,
            kernel_w,
        )
        <= 0
        or groups != 1
        or weight_c != c_in
        or min(
            stride_d,
            stride_h,
            stride_w,
            dilation_d,
            dilation_h,
            dilation_w,
        )
        <= 0
    ):
        return None

    output_d = (
        input_d + pad_front + pad_back - dilation_d * (kernel_d - 1) - 1
    ) // stride_d + 1
    output_h = (
        input_h + pad_top + pad_bottom - dilation_h * (kernel_h - 1) - 1
    ) // stride_h + 1
    output_w = (
        input_w + pad_left + pad_right - dilation_w * (kernel_w - 1) - 1
    ) // stride_w + 1
    if min(output_d, output_h, output_w) <= 0:
        return None

    exact_symmetric = (
        image_shape == (2, 8, 8, 16, 16)
        and weight_shape == (16, 8, 3, 3, 3)
        and (output_d, output_h, output_w) == (8, 16, 16)
        and (stride_d, stride_h, stride_w) == (1, 1, 1)
        and (
            pad_front,
            pad_back,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
        )
        == (1, 1, 1, 1, 1, 1)
        and (dilation_d, dilation_h, dilation_w) == (1, 1, 1)
    )
    exact_asymmetric = (
        image_shape == (1, 8, 10, 12, 14)
        and weight_shape == (12, 8, 2, 3, 3)
        and (output_d, output_h, output_w) == (10, 11, 15)
        and (stride_d, stride_h, stride_w) == (1, 1, 1)
        and (
            pad_front,
            pad_back,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
        )
        == (1, 0, 0, 1, 1, 2)
        and (dilation_d, dilation_h, dilation_w) == (1, 1, 1)
    )
    if not exact_symmetric and not exact_asymmetric:
        return None

    checks = runtime_tensor_checks_from_specs(
        input_specs,
        (0, 1),
        require_shape=True,
        require_stride=True,
        require_dtype=True,
    )
    if checks is None:
        return None

    output_shape = (n, c_out, output_d, output_h, output_w)
    output_dtype = torch_dtype(image_spec.dtype)
    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    dtype_id = {"float16": 0, "bfloat16": 1, "float32": 2}[image_spec.dtype]
    m = n * output_d * output_h * output_w

    def output_factory(inputs: Sequence[Any]) -> torch.Tensor:
        image = inputs[0]
        assert isinstance(image, torch.Tensor)
        key = (
            image.device.type,
            image.device.index,
            output_dtype,
            output_shape,
        )
        return get_prepared_output(
            output_cache,
            key,
            lambda: torch.empty(
                output_shape, device=image.device, dtype=output_dtype
            ),
        )

    def runtime_args(
        inputs: Sequence[Any], output: torch.Tensor
    ) -> tuple[Any, ...]:
        image, weight = inputs
        assert isinstance(image, torch.Tensor)
        assert isinstance(weight, torch.Tensor)
        return image, weight, output

    static_args = (
        m,
        input_d,
        input_h,
        input_w,
        output_d,
        output_h,
        output_w,
        c_in,
        c_out,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        pad_front,
        pad_top,
        pad_left,
        dilation_d,
        dilation_h,
        dilation_w,
    )

    def grid(meta: dict[str, Any]) -> tuple[int, ...]:
        block_m = int(meta["BLOCK_M"])
        block_oc = int(meta["BLOCK_OC"])
        return (
            ((m + block_m - 1) // block_m)
            * ((c_out + block_oc - 1) // block_oc),
        )

    def build_cached_call(
        constexprs: dict[str, Any],
    ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
        block_oc = int(constexprs["BLOCK_OC"])
        block_m = int(constexprs["BLOCK_M"])
        block_k = int(constexprs["BLOCK_K"])
        static_grid = (
            ((m + block_m - 1) // block_m)
            * ((c_out + block_oc - 1) // block_oc),
            1,
            1,
        )
        return static_grid, static_args + (
            dtype_id,
            block_oc,
            block_m,
            block_k,
            _GROUP_M,
        )

    def extra_check(inputs: Sequence[Any]) -> bool:
        image, weight = inputs
        return (
            isinstance(image, torch.Tensor)
            and isinstance(weight, torch.Tensor)
            and image.device == weight.device
        )

    return make_single_kernel_run_fn(
        PreparedSingleKernelRunSpec(
            kernel=PreparedSingleKernelSpec(
                kernel=_conv_fprop3d_ncdhw_kernel,
                grid=grid,
                static_args=static_args,
                constexpr_kwargs={
                    "DTYPE_ID": dtype_id,
                    "GROUP_M": _GROUP_M,
                },
                build_cached_call=build_cached_call,
            ),
            input_checks=checks,
            output_factory=output_factory,
            runtime_args=runtime_args,
            extra_check=extra_check,
            validate_inputs=bool(attrs.get("_validate_inputs", True)),
        ),
        default_run_fn,
    )


def _prepare_fprop_2d(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if len(input_specs) != 2 or not all(
        _is_runtime_device_spec(spec) for spec in input_specs
    ):
        return None
    image_spec, weight_spec = input_specs
    image_shape = _static_shape(image_spec)
    weight_shape = _static_shape(weight_spec)
    if (
        image_shape is None
        or weight_shape is None
        or len(image_shape) != 4
        or len(weight_shape) != 4
        or image_spec.contiguous is not True
        or weight_spec.contiguous is not True
        or image_spec.stride is None
        or weight_spec.stride is None
        or image_spec.dtype not in _FPROP_DTYPES
        or weight_spec.dtype != image_spec.dtype
        or not _is_cross_correlation(attrs.get("convolution_mode"))
    ):
        return None

    n, c_in, input_h, input_w = image_shape
    c_out, weight_c, kernel_h, kernel_w = weight_shape
    groups = int(attrs.get("groups", 1))
    stride_h, stride_w = _pair(attrs.get("stride", 1), "stride")
    dilation_h, dilation_w = _pair(attrs.get("dilation", 1), "dilation")
    pad_top, pad_bottom, pad_left, pad_right = _padding_2d(attrs)
    if (
        min(
            n,
            c_in,
            input_h,
            input_w,
            c_out,
            weight_c,
            kernel_h,
            kernel_w,
        )
        <= 0
        or groups <= 0
        or c_in % groups != 0
        or c_out % groups != 0
        or weight_c != c_in // groups
        or stride_h <= 0
        or stride_w <= 0
        or dilation_h <= 0
        or dilation_w <= 0
    ):
        return None
    output_h = (
        input_h + pad_top + pad_bottom - dilation_h * (kernel_h - 1) - 1
    ) // stride_h + 1
    output_w = (
        input_w + pad_left + pad_right - dilation_w * (kernel_w - 1) - 1
    ) // stride_w + 1
    if output_h <= 0 or output_w <= 0:
        return None

    cin_per_group = c_in // groups
    cout_per_group = c_out // groups
    is_depthwise = groups == c_in and weight_c == 1 and c_out == c_in
    if is_depthwise:
        return None

    use_im2col_pipeline = (
        n == 1
        and input_h == 40
        and input_w == 40
        and output_h == 20
        and output_w == 20
        and kernel_h == 3
        and kernel_w == 3
        and (stride_h, stride_w) == (2, 2)
        and (pad_top, pad_bottom, pad_left, pad_right) == (1, 1, 1, 1)
        and (dilation_h, dilation_w) == (1, 1)
        and cin_per_group >= 128
        and cout_per_group >= 256
    )

    use_packed_fp32 = (
        image_spec.dtype == "float32"
        and groups == 1
        and kernel_h == 3
        and kernel_w == 3
        and (stride_h, stride_w) == (1, 1)
        and cin_per_group >= 64
        and cout_per_group >= 64
    )

    checks = runtime_tensor_checks_from_specs(
        input_specs,
        (0, 1),
        require_shape=True,
        require_stride=True,
        require_dtype=True,
    )
    if checks is None:
        return None

    if use_im2col_pipeline:
        return _prepare_fprop_2d_im2col(
            attrs,
            input_specs,
            default_run_fn,
            checks,
            image_shape,
            weight_shape,
            output_h,
            output_w,
        )

    output_shape = (n, c_out, output_h, output_w)
    output_dtype = torch_dtype(image_spec.dtype)
    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    dtype_id = {"float16": 0, "bfloat16": 1, "float32": 2}[image_spec.dtype]
    output_hw = output_h * output_w

    def output_factory(inputs: Sequence[Any]) -> torch.Tensor:
        image = inputs[0]
        assert isinstance(image, torch.Tensor)
        key = (
            image.device.type,
            image.device.index,
            output_dtype,
            output_shape,
        )
        return get_prepared_output(
            output_cache,
            key,
            lambda: torch.empty(
                output_shape, device=image.device, dtype=output_dtype
            ),
        )

    if use_packed_fp32:

        def runtime_args(
            inputs: Sequence[Any], output: torch.Tensor
        ) -> tuple[Any, ...]:
            image, weight = inputs
            assert isinstance(image, torch.Tensor)
            assert isinstance(weight, torch.Tensor)
            packed_weight = _pack_weight_spatial_nchw_khw_oci(weight, groups)
            return image, packed_weight, output, output

    else:

        def runtime_args(
            inputs: Sequence[Any], output: torch.Tensor
        ) -> tuple[Any, ...]:
            image, weight = inputs
            assert isinstance(image, torch.Tensor)
            assert isinstance(weight, torch.Tensor)
            return image, weight, output, output

    def extra_check(inputs: Sequence[Any]) -> bool:
        image, weight = inputs
        return (
            isinstance(image, torch.Tensor)
            and isinstance(weight, torch.Tensor)
            and image.device == weight.device
        )

    is_1x1_pad0 = (
        kernel_h == 1
        and kernel_w == 1
        and (stride_h, stride_w) == (1, 1)
        and (pad_top, pad_bottom, pad_left, pad_right) == (0, 0, 0, 0)
        and (dilation_h, dilation_w) == (1, 1)
    )
    if is_1x1_pad0:

        def grid_1x1(meta: dict[str, Any]) -> tuple[int, int]:
            return (
                (
                    (output_hw + int(meta["BLOCK_HW"]) - 1)
                    // int(meta["BLOCK_HW"])
                )
                * (
                    (cout_per_group + int(meta["BLOCK_OC"]) - 1)
                    // int(meta["BLOCK_OC"])
                ),
                n * groups,
            )

        def build_1x1_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
            block_oc = int(constexprs["BLOCK_OC"])
            block_hw = int(constexprs["BLOCK_HW"])
            block_k = int(constexprs["BLOCK_K"])
            static_grid = (
                ((output_hw + block_hw - 1) // block_hw)
                * ((cout_per_group + block_oc - 1) // block_oc),
                n * groups,
                1,
            )
            return static_grid, (
                output_hw,
                c_in,
                c_out,
                cin_per_group,
                cout_per_group,
                groups,
                False,
                block_oc,
                block_hw,
                block_k,
                _GROUP_M,
                dtype_id,
            )

        kernel_spec = PreparedSingleKernelSpec(
            kernel=conv2d_1x1_nchw_pad0_kernel,
            grid=grid_1x1,
            static_args=(
                output_hw,
                c_in,
                c_out,
                cin_per_group,
                cout_per_group,
                groups,
            ),
            constexpr_kwargs={
                "HAS_BIAS": False,
                "GROUP_M": _GROUP_M,
                "DTYPE_ID": dtype_id,
            },
            build_cached_call=build_1x1_cached_call,
        )
    else:

        def grid_spatial(meta: dict[str, Any]) -> tuple[int, int]:
            return (
                (
                    (output_hw + int(meta["BLOCK_HW"]) - 1)
                    // int(meta["BLOCK_HW"])
                )
                * (
                    (cout_per_group + int(meta["BLOCK_OC"]) - 1)
                    // int(meta["BLOCK_OC"])
                ),
                n * groups,
            )

        def build_spatial_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
            block_oc = int(constexprs["BLOCK_OC"])
            block_hw = int(constexprs["BLOCK_HW"])
            block_k = int(constexprs["BLOCK_K"])
            static_grid = (
                ((output_hw + block_hw - 1) // block_hw)
                * ((cout_per_group + block_oc - 1) // block_oc),
                n * groups,
                1,
            )
            return static_grid, (
                input_h,
                input_w,
                output_h,
                output_w,
                c_in,
                c_out,
                cin_per_group,
                cout_per_group,
                groups,
                stride_h,
                stride_w,
                pad_top,
                pad_left,
                dilation_h,
                dilation_w,
                kernel_h,
                kernel_w,
                False,
                block_oc,
                block_hw,
                block_k,
                _GROUP_M,
                dtype_id,
            )

        kernel_spec = PreparedSingleKernelSpec(
            kernel=(
                _conv2d_spatial_nchw_packed_khw_kernel
                if use_packed_fp32
                else conv2d_spatial_nchw_kernel
            ),
            grid=grid_spatial,
            static_args=(
                input_h,
                input_w,
                output_h,
                output_w,
                c_in,
                c_out,
                cin_per_group,
                cout_per_group,
                groups,
                stride_h,
                stride_w,
                pad_top,
                pad_left,
                dilation_h,
                dilation_w,
                kernel_h,
                kernel_w,
            ),
            constexpr_kwargs={
                "HAS_BIAS": False,
                "GROUP_M": _GROUP_M,
                "DTYPE_ID": dtype_id,
            },
            build_cached_call=build_spatial_cached_call,
        )

    return make_single_kernel_run_fn(
        PreparedSingleKernelRunSpec(
            kernel=kernel_spec,
            input_checks=checks,
            output_factory=output_factory,
            runtime_args=runtime_args,
            extra_check=extra_check,
            validate_inputs=bool(attrs.get("_validate_inputs", True)),
        ),
        default_run_fn,
    )


def _prepare_wgrad_1d(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if len(input_specs) != 2 or not all(
        _is_runtime_device_spec(spec) for spec in input_specs
    ):
        return None
    image_spec, loss_spec = input_specs
    image_shape = _static_shape(image_spec)
    loss_shape = _static_shape(loss_spec)
    filter_size = tuple(int(dim) for dim in attrs.get("filter_size", ()))
    if (
        image_shape is None
        or loss_shape is None
        or len(image_shape) != 3
        or len(loss_shape) != 3
        or len(filter_size) != 3
        or image_spec.contiguous is not True
        or loss_spec.contiguous is not True
        or image_spec.stride is None
        or loss_spec.stride is None
        or image_spec.dtype not in _FPROP_DTYPES
        or loss_spec.dtype != image_spec.dtype
        or not _is_cross_correlation(attrs.get("convolution_mode"))
    ):
        return None

    n, c_in, image_l = image_shape
    loss_n, c_out, loss_l = loss_shape
    filter_c_out, filter_c_in, kernel_l = filter_size
    groups = int(attrs.get("groups", 1))
    stride_l = _single(attrs.get("stride", 1), "stride")
    dilation_l = _single(attrs.get("dilation", 1), "dilation")
    pad_left, pad_right = _padding_1d(attrs)
    cin_per_group = c_in // groups if groups > 0 else 0
    cout_per_group = c_out // groups if groups > 0 else 0
    expected_loss_l = (
        image_l + pad_left + pad_right - dilation_l * (kernel_l - 1) - 1
    ) // stride_l + 1
    if (
        min(n, c_in, image_l, loss_n, c_out, loss_l, kernel_l) <= 0
        or loss_n != n
        or filter_c_out != c_out
        or groups != 1
        or filter_c_in != cin_per_group
        or expected_loss_l != loss_l
        or stride_l <= 0
        or dilation_l <= 0
    ):
        return None

    exact_stride1 = (
        image_shape == (16, 32, 256)
        and loss_shape == (16, 64, 256)
        and filter_size == (64, 32, 3)
        and stride_l == 1
        and (pad_left, pad_right) == (1, 1)
        and dilation_l == 1
    )
    exact_stride2 = (
        image_shape == (8, 64, 255)
        and loss_shape == (8, 96, 127)
        and filter_size == (96, 64, 5)
        and stride_l == 2
        and (pad_left, pad_right) == (2, 1)
        and dilation_l == 1
    )
    if not exact_stride1 and not exact_stride2:
        return None

    checks = runtime_tensor_checks_from_specs(
        input_specs,
        (0, 1),
        require_shape=True,
        require_stride=True,
        require_dtype=True,
    )
    if checks is None:
        return None

    output_dtype = torch_dtype(image_spec.dtype)
    output_stride = (cin_per_group * kernel_l, kernel_l, 1)
    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    workspace_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def get_output(image: torch.Tensor) -> torch.Tensor:
        key = (
            image.device.type,
            image.device.index,
            output_dtype,
            filter_size,
        )
        return get_prepared_output(
            output_cache,
            key,
            lambda: torch.empty(
                filter_size, device=image.device, dtype=output_dtype
            ),
        )

    def extra_check(inputs: Sequence[Any]) -> bool:
        image, loss = inputs
        return (
            isinstance(image, torch.Tensor)
            and isinstance(loss, torch.Tensor)
            and image.device == loss.device
        )

    if exact_stride2:
        block_co = 16
        block_n = 32
        block_m = 128
        cik = cin_per_group * kernel_l
        static_grid = (
            ((cout_per_group + block_co - 1) // block_co)
            * ((cik + block_n - 1) // block_n),
            groups,
            1,
        )
        static_args = (
            image_l,
            loss_l,
            cout_per_group,
            cin_per_group,
            *image_spec.stride,
            *loss_spec.stride,
            *output_stride,
            stride_l,
            pad_left,
            dilation_l,
            kernel_l,
            False,
            n,
        )

        def context_factory(inputs: Sequence[Any]) -> torch.Tensor:
            image = inputs[0]
            assert isinstance(image, torch.Tensor)
            return get_output(image)

        def direct_args(
            inputs: Sequence[Any], output: torch.Tensor
        ) -> tuple[Any, ...]:
            return inputs[0], inputs[1], output

        def build_direct_cached_call(
            _metadata: dict[str, Any],
        ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
            return static_grid, static_args + (block_co, block_n, block_m)

        pipeline = PreparedKernelPipelineSpec(
            steps=(
                PreparedPipelineStepSpec(
                    kernel=_conv_wgrad1d_col_direct_nodiv_kernel,
                    grid=static_grid,
                    runtime_args=direct_args,
                    static_args=static_args,
                    constexpr_kwargs={
                        "BLOCK_CO": block_co,
                        "BLOCK_N": block_n,
                        "BLOCK_M": block_m,
                        "num_warps": 4,
                        "num_stages": 3,
                    },
                    build_cached_call=build_direct_cached_call,
                ),
            ),
            input_checks=checks,
            context_factory=context_factory,
            result=lambda output: output,
            extra_check=extra_check,
        )
        return _make_bound_pipeline_run_fn(
            pipeline,
            default_run_fn,
            validate_inputs=bool(attrs.get("_validate_inputs", True)),
        )

    num_splits = n
    splits_per_n = 1
    block_co = 16
    if image_spec.dtype == "float32":
        block_ci = 32
        block_m = 64
        partial_dtype = torch.float32
    elif image_spec.dtype == "float16":
        block_ci = 16
        block_m = 256
        partial_dtype = torch.float16
    else:
        block_ci = 16
        block_m = 256
        partial_dtype = torch.float32
    cta_x = ((cout_per_group + block_co - 1) // block_co) * (
        (cin_per_group + block_ci - 1) // block_ci
    )
    split_grid = (cta_x, num_splits * groups, 1)
    reduce_grid = (cta_x, groups, 1)
    split_static_args = (
        loss_l,
        c_out,
        cin_per_group,
        cout_per_group,
        *image_spec.stride,
        *loss_spec.stride,
        pad_left,
        kernel_l,
        False,
        splits_per_n,
    )
    reduce_static_args = (
        c_out,
        cin_per_group,
        cout_per_group,
        *output_stride,
        num_splits,
    )

    def split_context_factory(
        inputs: Sequence[Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = inputs[0]
        assert isinstance(image, torch.Tensor)
        output = get_output(image)
        partial_shape = (num_splits, c_out, cin_per_group, kernel_l)
        partial_key = (
            image.device.type,
            image.device.index,
            partial_dtype,
            partial_shape,
        )
        partial = workspace_cache.get(partial_key)
        if partial is None:
            partial = torch.empty(
                partial_shape, device=image.device, dtype=partial_dtype
            )
            workspace_cache[partial_key] = partial
        return output, partial

    def split_args(
        inputs: Sequence[Any], context: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[Any, ...]:
        return inputs[0], inputs[1], context[1]

    def reduce_args(
        _inputs: Sequence[Any], context: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[Any, ...]:
        return context[1], context[0]

    def build_split_cached_call(
        _metadata: dict[str, Any],
    ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
        return split_grid, split_static_args + (
            block_co,
            block_ci,
            block_m,
        )

    def build_reduce_cached_call(
        _metadata: dict[str, Any],
    ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
        return reduce_grid, reduce_static_args + (block_co, block_ci)

    pipeline = PreparedKernelPipelineSpec(
        steps=(
            PreparedPipelineStepSpec(
                kernel=_conv_wgrad1d_3tap_nodiv_split_kernel,
                grid=split_grid,
                runtime_args=split_args,
                static_args=split_static_args,
                constexpr_kwargs={
                    "BLOCK_CO": block_co,
                    "BLOCK_CI": block_ci,
                    "BLOCK_M": block_m,
                    "num_warps": 4,
                    "num_stages": 3,
                },
                build_cached_call=build_split_cached_call,
            ),
            PreparedPipelineStepSpec(
                kernel=_conv_wgrad1d_reduce3_kernel,
                grid=reduce_grid,
                runtime_args=reduce_args,
                static_args=reduce_static_args,
                constexpr_kwargs={
                    "BLOCK_CO": block_co,
                    "BLOCK_CI": block_ci,
                    "num_warps": 4,
                    "num_stages": 1,
                },
                build_cached_call=build_reduce_cached_call,
            ),
        ),
        input_checks=checks,
        context_factory=split_context_factory,
        result=lambda context: context[0],
        extra_check=extra_check,
    )
    return _make_bound_pipeline_run_fn(
        pipeline,
        default_run_fn,
        validate_inputs=bool(attrs.get("_validate_inputs", True)),
    )


def _prepare_wgrad_2d(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if len(input_specs) != 2 or not all(
        _is_runtime_device_spec(spec) for spec in input_specs
    ):
        return None
    image_spec, loss_spec = input_specs
    image_shape = _static_shape(image_spec)
    loss_shape = _static_shape(loss_spec)
    filter_size = tuple(int(dim) for dim in attrs.get("filter_size", ()))
    if (
        image_shape is None
        or loss_shape is None
        or len(image_shape) != 4
        or len(loss_shape) != 4
        or len(filter_size) != 4
        or image_spec.contiguous is not True
        or loss_spec.contiguous is not True
        or image_spec.stride is None
        or loss_spec.stride is None
        or image_spec.dtype not in _FPROP_DTYPES
        or loss_spec.dtype != image_spec.dtype
        or not _is_cross_correlation(attrs.get("convolution_mode"))
    ):
        return None

    n, c_in, image_h, image_w = image_shape
    loss_n, c_out, loss_h, loss_w = loss_shape
    filter_c_out, filter_c_in, kernel_h, kernel_w = filter_size
    groups = int(attrs.get("groups", 1))
    stride_h, stride_w = _pair(attrs.get("stride", 1), "stride")
    dilation_h, dilation_w = _pair(attrs.get("dilation", 1), "dilation")
    pad_top, pad_bottom, pad_left, pad_right = _padding_2d(attrs)
    cin_per_group = c_in // groups if groups > 0 else 0
    cout_per_group = c_out // groups if groups > 0 else 0
    expected_loss_h = (
        image_h + pad_top + pad_bottom - dilation_h * (kernel_h - 1) - 1
    ) // stride_h + 1
    expected_loss_w = (
        image_w + pad_left + pad_right - dilation_w * (kernel_w - 1) - 1
    ) // stride_w + 1
    if (
        min(
            n,
            c_in,
            image_h,
            image_w,
            loss_n,
            c_out,
            loss_h,
            loss_w,
            kernel_h,
            kernel_w,
        )
        <= 0
        or loss_n != n
        or filter_c_out != c_out
        or groups != 1
        or filter_c_in != cin_per_group
        or expected_loss_h != loss_h
        or expected_loss_w != loss_w
        or min(stride_h, stride_w, dilation_h, dilation_w) <= 0
    ):
        return None

    exact_stride1 = (
        image_shape == (8, 32, 32, 32)
        and loss_shape == (8, 64, 32, 32)
        and filter_size == (64, 32, 3, 3)
        and (stride_h, stride_w) == (1, 1)
        and (pad_top, pad_bottom, pad_left, pad_right) == (1, 1, 1, 1)
        and (dilation_h, dilation_w) == (1, 1)
    )
    exact_1x1 = (
        image_shape == (8, 64, 28, 28)
        and loss_shape == (8, 128, 28, 28)
        and filter_size == (128, 64, 1, 1)
        and (stride_h, stride_w) == (1, 1)
        and (pad_top, pad_bottom, pad_left, pad_right) == (0, 0, 0, 0)
        and (dilation_h, dilation_w) == (1, 1)
    )
    exact_stride2 = (
        image_shape == (8, 64, 56, 56)
        and loss_shape == (8, 128, 28, 28)
        and filter_size == (128, 64, 3, 3)
        and (stride_h, stride_w) == (2, 2)
        and (pad_top, pad_bottom, pad_left, pad_right) == (1, 1, 1, 1)
        and (dilation_h, dilation_w) == (1, 1)
    )
    exact_p5_stride2 = (
        (image_shape, loss_shape, filter_size)
        in (
            (
                (1, 128, 40, 40),
                (1, 256, 20, 20),
                (256, 128, 3, 3),
            ),
            (
                (1, 256, 40, 40),
                (1, 512, 20, 20),
                (512, 256, 3, 3),
            ),
            (
                (1, 512, 40, 40),
                (1, 512, 20, 20),
                (512, 512, 3, 3),
            ),
            (
                (1, 768, 40, 40),
                (1, 768, 20, 20),
                (768, 768, 3, 3),
            ),
        )
        and (stride_h, stride_w) == (2, 2)
        and (pad_top, pad_bottom, pad_left, pad_right) == (1, 1, 1, 1)
        and (dilation_h, dilation_w) == (1, 1)
    )
    exact_stem_stride2 = (
        image_shape == (1, 3, 640, 640)
        and loss_shape == (1, c_out, 320, 320)
        and filter_size == (c_out, 3, 3, 3)
        and c_out in (16, 32, 64, 96)
        and (stride_h, stride_w) == (2, 2)
        and (pad_top, pad_bottom, pad_left, pad_right) == (1, 1, 1, 1)
        and (dilation_h, dilation_w) == (1, 1)
    )
    if (
        not exact_stride1
        and not exact_1x1
        and not exact_stride2
        and not exact_p5_stride2
        and not exact_stem_stride2
    ):
        return None

    checks = runtime_tensor_checks_from_specs(
        input_specs,
        (0, 1),
        require_shape=True,
        require_stride=True,
        require_dtype=True,
    )
    if checks is None:
        return None

    output_dtype = torch_dtype(image_spec.dtype)
    output_stride = (
        cin_per_group * kernel_h * kernel_w,
        kernel_h * kernel_w,
        kernel_w,
        1,
    )
    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    workspace_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def get_output(image: torch.Tensor) -> torch.Tensor:
        key = (
            image.device.type,
            image.device.index,
            output_dtype,
            filter_size,
        )
        return get_prepared_output(
            output_cache,
            key,
            lambda: torch.empty(
                filter_size, device=image.device, dtype=output_dtype
            ),
        )

    def extra_check(inputs: Sequence[Any]) -> bool:
        image, loss = inputs
        return (
            isinstance(image, torch.Tensor)
            and isinstance(loss, torch.Tensor)
            and image.device == loss.device
        )

    def zero_args(_inputs: Sequence[Any], context: Any) -> tuple[torch.Tensor]:
        output = context if isinstance(context, torch.Tensor) else context[0]
        return (output,)

    total = c_out * cin_per_group * kernel_h * kernel_w
    zero_block = 1024
    zero_grid = ((total + zero_block - 1) // zero_block, 1, 1)
    zero_static_args = (total,)

    def build_zero_cached_call(
        _metadata: dict[str, Any],
    ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
        return zero_grid, (total, zero_block)

    zero_step = PreparedPipelineStepSpec(
        kernel=_conv_wgrad_zero_kernel,
        grid=zero_grid,
        runtime_args=zero_args,
        static_args=zero_static_args,
        constexpr_kwargs={"BLOCK": zero_block, "num_warps": 4},
        build_cached_call=build_zero_cached_call,
    )

    if exact_stem_stride2:
        num_splits = 64
        block_co = 16
        block_n = 32
        block_m = 128
        cik = cin_per_group * kernel_h * kernel_w
        cta_x = ((cout_per_group + block_co - 1) // block_co) * (
            (cik + block_n - 1) // block_n
        )
        split_grid = (cta_x, num_splits * groups, 1)
        reduce_grid = (cta_x, groups, 1)
        split_static_args = (
            n * loss_h * loss_w,
            image_h,
            image_w,
            loss_h,
            loss_w,
            c_out,
            cin_per_group,
            cout_per_group,
            *image_spec.stride,
            *loss_spec.stride,
            stride_h,
            stride_w,
            pad_top,
            pad_left,
            dilation_h,
            dilation_w,
            kernel_h,
            kernel_w,
            False,
            num_splits,
        )
        reduce_static_args = (
            c_out,
            cin_per_group,
            cout_per_group,
            *output_stride,
            kernel_h,
            kernel_w,
            num_splits,
        )

        def context_factory(
            inputs: Sequence[Any],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            image = inputs[0]
            assert isinstance(image, torch.Tensor)
            output = get_output(image)
            partial_shape = (num_splits, c_out, cik)
            key = (
                image.device.type,
                image.device.index,
                torch.float32,
                partial_shape,
            )
            partial = workspace_cache.get(key)
            if partial is None:
                partial = torch.empty(
                    partial_shape,
                    device=image.device,
                    dtype=torch.float32,
                )
                workspace_cache[key] = partial
            return output, partial

        def split_args(
            inputs: Sequence[Any], context: tuple[torch.Tensor, torch.Tensor]
        ) -> tuple[Any, ...]:
            return inputs[0], inputs[1], context[1]

        def reduce_args(
            _inputs: Sequence[Any], context: tuple[torch.Tensor, torch.Tensor]
        ) -> tuple[Any, ...]:
            return context[1], context[0]

        def build_split_cached_call(
            _metadata: dict[str, Any],
        ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
            return split_grid, split_static_args + (
                block_co,
                block_n,
                block_m,
            )

        def build_reduce_cached_call(
            _metadata: dict[str, Any],
        ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
            return reduce_grid, reduce_static_args + (block_co, block_n)

        pipeline = PreparedKernelPipelineSpec(
            steps=(
                PreparedPipelineStepSpec(
                    kernel=_conv_wgrad2d_col_split_kernel,
                    grid=split_grid,
                    runtime_args=split_args,
                    static_args=split_static_args,
                    constexpr_kwargs={
                        "BLOCK_CO": block_co,
                        "BLOCK_N": block_n,
                        "BLOCK_M": block_m,
                        "num_warps": 4,
                        "num_stages": 3,
                    },
                    build_cached_call=build_split_cached_call,
                ),
                PreparedPipelineStepSpec(
                    kernel=_conv_wgrad2d_col_reduce_kernel,
                    grid=reduce_grid,
                    runtime_args=reduce_args,
                    static_args=reduce_static_args,
                    constexpr_kwargs={
                        "BLOCK_CO": block_co,
                        "BLOCK_N": block_n,
                        "num_warps": 4,
                        "num_stages": 1,
                    },
                    build_cached_call=build_reduce_cached_call,
                ),
            ),
            input_checks=checks,
            context_factory=context_factory,
            result=lambda context: context[0],
            extra_check=extra_check,
        )
        return _make_bound_pipeline_run_fn(
            pipeline,
            default_run_fn,
            validate_inputs=bool(attrs.get("_validate_inputs", True)),
        )

    if exact_p5_stride2:
        dtype_id = {"float16": 0, "bfloat16": 1, "float32": 2}[
            image_spec.dtype
        ]
        cik = cin_per_group * kernel_h * kernel_w
        pack_block_m = 16
        pack_block_n = 256
        pack_grid = (
            (loss_h * loss_w + pack_block_m - 1) // pack_block_m,
            (cik + pack_block_n - 1) // pack_block_n,
            1,
        )
        pack_static_args = (
            cin_per_group,
            image_spec.stride[1],
            image_spec.stride[2],
            image_spec.stride[3],
        )
        mm_static_args = (
            cout_per_group,
            cik,
            loss_h * loss_w,
            dtype_id,
        )

        def context_factory(
            inputs: Sequence[Any],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            image = inputs[0]
            assert isinstance(image, torch.Tensor)
            output = get_output(image)
            packed_shape = (loss_h * loss_w, cik)
            key = (
                image.device.type,
                image.device.index,
                output_dtype,
                packed_shape,
            )
            packed = workspace_cache.get(key)
            if packed is None:
                packed = torch.empty(
                    packed_shape,
                    device=image.device,
                    dtype=output_dtype,
                )
                workspace_cache[key] = packed
            return output, packed

        def pack_args(
            inputs: Sequence[Any], context: tuple[torch.Tensor, torch.Tensor]
        ) -> tuple[Any, ...]:
            return inputs[0], context[1]

        def mm_args(
            inputs: Sequence[Any], context: tuple[torch.Tensor, torch.Tensor]
        ) -> tuple[Any, ...]:
            return inputs[1], context[1], context[0]

        def mm_grid(meta: dict[str, Any]) -> tuple[int, ...]:
            block_m = int(meta["BLOCK_M"])
            block_n = int(meta["BLOCK_N"])
            return (
                ((cout_per_group + block_m - 1) // block_m)
                * ((cik + block_n - 1) // block_n),
            )

        def build_pack_cached_call(
            _metadata: dict[str, Any],
        ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
            return pack_grid, pack_static_args + (
                pack_block_m,
                pack_block_n,
            )

        def build_mm_cached_call(
            metadata: dict[str, Any],
        ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
            block_m = int(metadata["BLOCK_M"])
            block_n = int(metadata["BLOCK_N"])
            block_k = int(metadata["BLOCK_K"])
            group_m = int(metadata["GROUP_M"])
            static_grid = (
                ((cout_per_group + block_m - 1) // block_m)
                * ((cik + block_n - 1) // block_n),
                1,
                1,
            )
            return static_grid, mm_static_args + (
                block_m,
                block_n,
                block_k,
                group_m,
            )

        pipeline = PreparedKernelPipelineSpec(
            steps=(
                PreparedPipelineStepSpec(
                    kernel=_conv_wgrad2d_p5_pack_image_kernel,
                    grid=pack_grid,
                    runtime_args=pack_args,
                    static_args=pack_static_args,
                    constexpr_kwargs={
                        "BLOCK_M": pack_block_m,
                        "BLOCK_N": pack_block_n,
                        "num_warps": 4,
                        "num_stages": 3,
                    },
                    build_cached_call=build_pack_cached_call,
                ),
                PreparedPipelineStepSpec(
                    kernel=_conv_wgrad2d_p5_mm_kernel,
                    grid=mm_grid,
                    runtime_args=mm_args,
                    static_args=mm_static_args,
                    build_cached_call=build_mm_cached_call,
                    first_launch_returns_metadata=True,
                ),
            ),
            input_checks=checks,
            context_factory=context_factory,
            result=lambda context: context[0],
            extra_check=extra_check,
        )
        return _make_bound_pipeline_run_fn(
            pipeline,
            default_run_fn,
            validate_inputs=bool(attrs.get("_validate_inputs", True)),
        )

    if exact_stride1:
        num_splits = 16
        block_co = 16
        block_n = 32
        block_m = 32 if image_spec.dtype == "float32" else 64
        k_elems = kernel_h * kernel_w
        cik = cin_per_group * k_elems
        cta_x = ((cout_per_group + block_co - 1) // block_co) * (
            (cik + block_n - 1) // block_n
        )
        split_grid = (cta_x, num_splits * groups, 1)
        reduce_grid = (cta_x, groups, 1)
        split_static_args = (
            n * loss_h * loss_w,
            image_h,
            image_w,
            loss_h,
            loss_w,
            c_out,
            cin_per_group,
            cout_per_group,
            *image_spec.stride,
            *loss_spec.stride,
            stride_h,
            stride_w,
            pad_top,
            pad_left,
            dilation_h,
            dilation_w,
            kernel_h,
            kernel_w,
            False,
            num_splits,
        )
        reduce_static_args = (
            c_out,
            cin_per_group,
            cout_per_group,
            *output_stride,
            kernel_h,
            kernel_w,
            num_splits,
        )

        def context_factory(
            inputs: Sequence[Any],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            image = inputs[0]
            assert isinstance(image, torch.Tensor)
            output = get_output(image)
            partial_shape = (num_splits, c_out, cik)
            key = (
                image.device.type,
                image.device.index,
                torch.float32,
                partial_shape,
            )
            partial = workspace_cache.get(key)
            if partial is None:
                partial = torch.empty(
                    partial_shape, device=image.device, dtype=torch.float32
                )
                workspace_cache[key] = partial
            return output, partial

        def split_args(
            inputs: Sequence[Any], context: tuple[torch.Tensor, torch.Tensor]
        ) -> tuple[Any, ...]:
            return inputs[0], inputs[1], context[1]

        def reduce_args(
            _inputs: Sequence[Any], context: tuple[torch.Tensor, torch.Tensor]
        ) -> tuple[Any, ...]:
            return context[1], context[0]

        def build_split_cached_call(
            _metadata: dict[str, Any],
        ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
            return split_grid, split_static_args + (
                block_co,
                block_n,
                block_m,
            )

        def build_reduce_cached_call(
            _metadata: dict[str, Any],
        ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
            return reduce_grid, reduce_static_args + (block_co, block_n)

        pipeline = PreparedKernelPipelineSpec(
            steps=(
                PreparedPipelineStepSpec(
                    kernel=_conv_wgrad2d_col_split_kernel,
                    grid=split_grid,
                    runtime_args=split_args,
                    static_args=split_static_args,
                    constexpr_kwargs={
                        "BLOCK_CO": block_co,
                        "BLOCK_N": block_n,
                        "BLOCK_M": block_m,
                        "num_warps": 4,
                        "num_stages": 3,
                        **(
                            {"maxnreg": 128}
                            if image_spec.dtype == "float32"
                            else {}
                        ),
                    },
                    build_cached_call=build_split_cached_call,
                ),
                PreparedPipelineStepSpec(
                    kernel=_conv_wgrad2d_col_reduce_kernel,
                    grid=reduce_grid,
                    runtime_args=reduce_args,
                    static_args=reduce_static_args,
                    constexpr_kwargs={
                        "BLOCK_CO": block_co,
                        "BLOCK_N": block_n,
                        "num_warps": 4,
                        "num_stages": 1,
                    },
                    build_cached_call=build_reduce_cached_call,
                ),
            ),
            input_checks=checks,
            context_factory=context_factory,
            result=lambda context: context[0],
            extra_check=extra_check,
        )
        return _make_bound_pipeline_run_fn(
            pipeline,
            default_run_fn,
            validate_inputs=bool(attrs.get("_validate_inputs", True)),
        )

    if exact_1x1 and image_spec.dtype == "float32":
        num_splits = 32
        block_co = 16
        block_ci = 64
        block_m = 64
        atomic_grid = (8, num_splits, 1)
        atomic_static_args = (
            image_h * image_w,
            cin_per_group,
            cout_per_group,
            image_spec.stride[0],
            image_spec.stride[1],
            loss_spec.stride[0],
            loss_spec.stride[1],
            output_stride[0],
            output_stride[1],
            num_splits // n,
        )

        def atomic_1x1_context(inputs: Sequence[Any]) -> torch.Tensor:
            image = inputs[0]
            assert isinstance(image, torch.Tensor)
            return get_output(image)

        def atomic_args(
            inputs: Sequence[Any], output: torch.Tensor
        ) -> tuple[Any, ...]:
            return inputs[0], inputs[1], output

        def build_atomic_cached_call(
            _metadata: dict[str, Any],
        ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
            return atomic_grid, atomic_static_args + (
                block_co,
                block_ci,
                block_m,
            )

        pipeline = PreparedKernelPipelineSpec(
            steps=(
                zero_step,
                PreparedPipelineStepSpec(
                    kernel=_conv_wgrad2d_1x1_atomic_nodiv_kernel,
                    grid=atomic_grid,
                    runtime_args=atomic_args,
                    static_args=atomic_static_args,
                    constexpr_kwargs={
                        "BLOCK_CO": block_co,
                        "BLOCK_CI": block_ci,
                        "BLOCK_M": block_m,
                        "num_warps": 4,
                        "num_stages": 3,
                    },
                    build_cached_call=build_atomic_cached_call,
                ),
            ),
            input_checks=checks,
            context_factory=atomic_1x1_context,
            result=lambda output: output,
            extra_check=extra_check,
        )
        return _make_bound_pipeline_run_fn(
            pipeline,
            default_run_fn,
            validate_inputs=bool(attrs.get("_validate_inputs", True)),
        )

    if exact_1x1:
        num_splits = 32
        split_block_co = 16
        split_block_ci = 64
        split_block_m = 256
        partial_dtype = (
            torch.float16 if image_spec.dtype == "float16" else torch.float32
        )
        split_grid = (8, num_splits, 1)
        split_static_args = (
            image_h * image_w,
            c_out,
            cin_per_group,
            cout_per_group,
            image_spec.stride[0],
            image_spec.stride[1],
            loss_spec.stride[0],
            loss_spec.stride[1],
            num_splits // n,
        )
        reduce_block_co = 8
        reduce_block_ci = 16 if image_spec.dtype == "bfloat16" else 32
        reduce_grid = (
            ((cout_per_group + reduce_block_co - 1) // reduce_block_co)
            * ((cin_per_group + reduce_block_ci - 1) // reduce_block_ci),
            groups,
            1,
        )
        reduce_1x1_static_args = (
            c_out,
            cin_per_group,
            cout_per_group,
            output_stride[0],
            output_stride[1],
            num_splits,
        )

        def context_factory(
            inputs: Sequence[Any],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            image = inputs[0]
            assert isinstance(image, torch.Tensor)
            output = get_output(image)
            partial_shape = (num_splits, c_out, cin_per_group)
            key = (
                image.device.type,
                image.device.index,
                partial_dtype,
                partial_shape,
            )
            partial = workspace_cache.get(key)
            if partial is None:
                partial = torch.empty(
                    partial_shape, device=image.device, dtype=partial_dtype
                )
                workspace_cache[key] = partial
            return output, partial

        def split_args(
            inputs: Sequence[Any], context: tuple[torch.Tensor, torch.Tensor]
        ) -> tuple[Any, ...]:
            return inputs[0], inputs[1], context[1]

        def reduce_args(
            _inputs: Sequence[Any],
            context: tuple[torch.Tensor, torch.Tensor],
        ) -> tuple[Any, ...]:
            return context[1], context[0]

        def build_split_cached_call(
            _metadata: dict[str, Any],
        ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
            return split_grid, split_static_args + (
                split_block_co,
                split_block_ci,
                split_block_m,
            )

        def build_reduce_cached_call(
            _metadata: dict[str, Any],
        ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
            return reduce_grid, reduce_1x1_static_args + (
                reduce_block_co,
                reduce_block_ci,
            )

        pipeline = PreparedKernelPipelineSpec(
            steps=(
                PreparedPipelineStepSpec(
                    kernel=_conv_wgrad2d_1x1_split_nodiv_kernel,
                    grid=split_grid,
                    runtime_args=split_args,
                    static_args=split_static_args,
                    constexpr_kwargs={
                        "BLOCK_CO": split_block_co,
                        "BLOCK_CI": split_block_ci,
                        "BLOCK_M": split_block_m,
                        "num_warps": 8,
                        "num_stages": 3,
                    },
                    build_cached_call=build_split_cached_call,
                ),
                PreparedPipelineStepSpec(
                    kernel=_conv_wgrad2d_1x1_reduce_kernel,
                    grid=reduce_grid,
                    runtime_args=reduce_args,
                    static_args=reduce_1x1_static_args,
                    constexpr_kwargs={
                        "BLOCK_CO": reduce_block_co,
                        "BLOCK_CI": reduce_block_ci,
                        "num_warps": 4,
                        "num_stages": 1,
                    },
                    build_cached_call=build_reduce_cached_call,
                ),
            ),
            input_checks=checks,
            context_factory=context_factory,
            result=lambda context: context[0],
            extra_check=extra_check,
        )
        return _make_bound_pipeline_run_fn(
            pipeline,
            default_run_fn,
            validate_inputs=bool(attrs.get("_validate_inputs", True)),
        )

    if image_spec.dtype == "float32":
        num_splits = 8
        block_co = 16
        block_ci = 32
        block_m = 16
        atomic_grid = (16, 3, num_splits)
        atomic_stride2_static_args = (
            n * loss_h * loss_w,
            image_h,
            image_w,
            loss_h,
            loss_w,
            cout_per_group,
            cin_per_group,
            *image_spec.stride,
            *loss_spec.stride,
            *output_stride,
            num_splits,
        )

        def atomic_stride2_context(inputs: Sequence[Any]) -> torch.Tensor:
            image = inputs[0]
            assert isinstance(image, torch.Tensor)
            return get_output(image)

        def atomic_args(
            inputs: Sequence[Any], output: torch.Tensor
        ) -> tuple[Any, ...]:
            return inputs[0], inputs[1], output

        def build_atomic_cached_call(
            _metadata: dict[str, Any],
        ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
            return atomic_grid, atomic_stride2_static_args + (
                block_co,
                block_ci,
                block_m,
            )

        pipeline = PreparedKernelPipelineSpec(
            steps=(
                zero_step,
                PreparedPipelineStepSpec(
                    kernel=_conv_wgrad2d_stride2_3tap_atomic_kernel,
                    grid=atomic_grid,
                    runtime_args=atomic_args,
                    static_args=atomic_stride2_static_args,
                    constexpr_kwargs={
                        "BLOCK_CO": block_co,
                        "BLOCK_CI": block_ci,
                        "BLOCK_M": block_m,
                        "num_warps": 2,
                        "num_stages": 3,
                    },
                    build_cached_call=build_atomic_cached_call,
                ),
            ),
            input_checks=checks,
            context_factory=atomic_stride2_context,
            result=lambda output: output,
            extra_check=extra_check,
        )
        return _make_bound_pipeline_run_fn(
            pipeline,
            default_run_fn,
            validate_inputs=bool(attrs.get("_validate_inputs", True)),
        )

    num_splits = n
    block_co = 16
    block_ci = 32
    block_hw = 128
    split_grid = (16, 3, num_splits)
    split_static_args = (
        c_out,
        cin_per_group,
        cout_per_group,
        *image_spec.stride,
        *loss_spec.stride,
    )
    reduce_grid = (16, kernel_h * kernel_w, groups)
    reduce_static_args = (
        c_out,
        cin_per_group,
        cout_per_group,
        *output_stride,
        kernel_h,
        kernel_w,
        num_splits,
    )

    def split_stride2_context(
        inputs: Sequence[Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = inputs[0]
        assert isinstance(image, torch.Tensor)
        output = get_output(image)
        partial_shape = (
            num_splits,
            c_out,
            cin_per_group,
            kernel_h * kernel_w,
        )
        key = (
            image.device.type,
            image.device.index,
            output_dtype,
            partial_shape,
        )
        partial = workspace_cache.get(key)
        if partial is None:
            partial = torch.empty(
                partial_shape, device=image.device, dtype=output_dtype
            )
            workspace_cache[key] = partial
        return output, partial

    def split_stride2_args(
        inputs: Sequence[Any], context: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[Any, ...]:
        return inputs[0], inputs[1], context[1]

    def reduce_stride2_args(
        _inputs: Sequence[Any], context: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[Any, ...]:
        return context[1], context[0]

    def build_split_stride2_call(
        _metadata: dict[str, Any],
    ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
        return split_grid, split_static_args + (
            block_co,
            block_ci,
            block_hw,
        )

    def build_reduce_stride2_call(
        _metadata: dict[str, Any],
    ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
        return reduce_grid, reduce_static_args + (block_co, block_ci)

    pipeline = PreparedKernelPipelineSpec(
        steps=(
            PreparedPipelineStepSpec(
                kernel=_conv_wgrad2d_stride2_row4_split_kernel,
                grid=split_grid,
                runtime_args=split_stride2_args,
                static_args=split_static_args,
                constexpr_kwargs={
                    "BLOCK_CO": block_co,
                    "BLOCK_CI": block_ci,
                    "BLOCK_HW": block_hw,
                    "num_warps": 4,
                    "num_stages": 2,
                },
                build_cached_call=build_split_stride2_call,
            ),
            PreparedPipelineStepSpec(
                kernel=_conv_wgrad2d_reduce_kernel,
                grid=reduce_grid,
                runtime_args=reduce_stride2_args,
                static_args=reduce_static_args,
                constexpr_kwargs={
                    "BLOCK_CO": block_co,
                    "BLOCK_CI": block_ci,
                    "num_warps": 4,
                    "num_stages": 1,
                },
                build_cached_call=build_reduce_stride2_call,
            ),
        ),
        input_checks=checks,
        context_factory=split_stride2_context,
        result=lambda context: context[0],
        extra_check=extra_check,
    )
    return _make_bound_pipeline_run_fn(
        pipeline,
        default_run_fn,
        validate_inputs=bool(attrs.get("_validate_inputs", True)),
    )


def _prepare_wgrad_3d(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if len(input_specs) != 2 or not all(
        _is_runtime_device_spec(spec) for spec in input_specs
    ):
        return None
    image_spec, loss_spec = input_specs
    image_shape = _static_shape(image_spec)
    loss_shape = _static_shape(loss_spec)
    filter_size = tuple(int(dim) for dim in attrs.get("filter_size", ()))
    if (
        image_shape is None
        or loss_shape is None
        or len(image_shape) != 5
        or len(loss_shape) != 5
        or len(filter_size) != 5
        or image_spec.contiguous is not True
        or loss_spec.contiguous is not True
        or image_spec.stride is None
        or loss_spec.stride is None
        or image_spec.dtype not in _FPROP_DTYPES
        or loss_spec.dtype != image_spec.dtype
        or not _is_cross_correlation(attrs.get("convolution_mode"))
    ):
        return None

    n, c_in, image_d, image_h, image_w = image_shape
    loss_n, c_out, loss_d, loss_h, loss_w = loss_shape
    filter_c_out, filter_c_in, kernel_d, kernel_h, kernel_w = filter_size
    groups = int(attrs.get("groups", 1))
    stride_d, stride_h, stride_w = _triple(attrs.get("stride", 1), "stride")
    dilation_d, dilation_h, dilation_w = _triple(
        attrs.get("dilation", 1), "dilation"
    )
    (
        pad_front,
        pad_back,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
    ) = _padding_3d(attrs)
    cin_per_group = c_in // groups if groups > 0 else 0
    cout_per_group = c_out // groups if groups > 0 else 0
    expected_loss_d = (
        image_d + pad_front + pad_back - dilation_d * (kernel_d - 1) - 1
    ) // stride_d + 1
    expected_loss_h = (
        image_h + pad_top + pad_bottom - dilation_h * (kernel_h - 1) - 1
    ) // stride_h + 1
    expected_loss_w = (
        image_w + pad_left + pad_right - dilation_w * (kernel_w - 1) - 1
    ) // stride_w + 1
    if (
        min(
            n,
            c_in,
            image_d,
            image_h,
            image_w,
            loss_n,
            c_out,
            loss_d,
            loss_h,
            loss_w,
            kernel_d,
            kernel_h,
            kernel_w,
        )
        <= 0
        or loss_n != n
        or filter_c_out != c_out
        or groups != 1
        or filter_c_in != cin_per_group
        or expected_loss_d != loss_d
        or expected_loss_h != loss_h
        or expected_loss_w != loss_w
        or min(
            stride_d,
            stride_h,
            stride_w,
            dilation_d,
            dilation_h,
            dilation_w,
        )
        <= 0
    ):
        return None

    exact_symmetric = (
        image_shape == (2, 8, 8, 16, 16)
        and loss_shape == (2, 16, 8, 16, 16)
        and filter_size == (16, 8, 3, 3, 3)
        and (stride_d, stride_h, stride_w) == (1, 1, 1)
        and (
            pad_front,
            pad_back,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
        )
        == (1, 1, 1, 1, 1, 1)
        and (dilation_d, dilation_h, dilation_w) == (1, 1, 1)
    )
    exact_asymmetric = (
        image_shape == (1, 8, 10, 12, 14)
        and loss_shape == (1, 12, 10, 11, 15)
        and filter_size == (12, 8, 2, 3, 3)
        and (stride_d, stride_h, stride_w) == (1, 1, 1)
        and (
            pad_front,
            pad_back,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
        )
        == (1, 0, 0, 1, 1, 2)
        and (dilation_d, dilation_h, dilation_w) == (1, 1, 1)
    )
    if not exact_symmetric and not exact_asymmetric:
        return None

    checks = runtime_tensor_checks_from_specs(
        input_specs,
        (0, 1),
        require_shape=True,
        require_stride=True,
        require_dtype=True,
    )
    if checks is None:
        return None

    output_dtype = torch_dtype(image_spec.dtype)
    kernel_elems = kernel_d * kernel_h * kernel_w
    output_stride = (
        cin_per_group * kernel_elems,
        kernel_elems,
        kernel_h * kernel_w,
        kernel_w,
        1,
    )
    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    workspace_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def get_output(image: torch.Tensor) -> torch.Tensor:
        key = (
            image.device.type,
            image.device.index,
            output_dtype,
            filter_size,
        )
        return get_prepared_output(
            output_cache,
            key,
            lambda: torch.empty(
                filter_size, device=image.device, dtype=output_dtype
            ),
        )

    def extra_check(inputs: Sequence[Any]) -> bool:
        image, loss = inputs
        return (
            isinstance(image, torch.Tensor)
            and isinstance(loss, torch.Tensor)
            and image.device == loss.device
        )

    if exact_asymmetric:
        block_n = 1
        block_m = 256 if image_spec.dtype == "float32" else 128
        num_warps = 4 if image_spec.dtype == "float32" else 2
        cik = cin_per_group * kernel_elems
        direct_grid = (
            cout_per_group,
            (cik + block_n - 1) // block_n,
            groups,
        )
        direct_static_args = (
            image_d,
            image_h,
            image_w,
            loss_d,
            loss_h,
            loss_w,
            cin_per_group,
            cout_per_group,
            *image_spec.stride,
            *loss_spec.stride,
            *output_stride,
            stride_d,
            stride_h,
            stride_w,
            pad_front,
            pad_top,
            pad_left,
            dilation_d,
            dilation_h,
            dilation_w,
            kernel_d,
            kernel_h,
            kernel_w,
        )

        def context_factory(inputs: Sequence[Any]) -> torch.Tensor:
            image = inputs[0]
            assert isinstance(image, torch.Tensor)
            return get_output(image)

        def direct_args(
            inputs: Sequence[Any], output: torch.Tensor
        ) -> tuple[Any, ...]:
            return inputs[0], inputs[1], output

        def build_direct_cached_call(
            _metadata: dict[str, Any],
        ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
            return direct_grid, direct_static_args + (block_n, block_m)

        pipeline = PreparedKernelPipelineSpec(
            steps=(
                PreparedPipelineStepSpec(
                    kernel=_conv_wgrad3d_valid_col_direct_kernel,
                    grid=direct_grid,
                    runtime_args=direct_args,
                    static_args=direct_static_args,
                    constexpr_kwargs={
                        "BLOCK_N": block_n,
                        "BLOCK_M": block_m,
                        "num_warps": num_warps,
                        "num_stages": 2,
                    },
                    build_cached_call=build_direct_cached_call,
                ),
            ),
            input_checks=checks,
            context_factory=context_factory,
            result=lambda output: output,
            extra_check=extra_check,
        )
        return _make_bound_pipeline_run_fn(
            pipeline,
            default_run_fn,
            validate_inputs=bool(attrs.get("_validate_inputs", True)),
        )

    if image_spec.dtype == "float32":
        num_splits = 8
        block_co = 16
        block_ci = 8
        block_m = 64
        m = n * loss_d * loss_h * loss_w
        total = c_out * cin_per_group * kernel_elems
        zero_block = 1024
        zero_grid = ((total + zero_block - 1) // zero_block, 1, 1)
        atomic_grid = (
            ((cout_per_group + block_co - 1) // block_co)
            * ((cin_per_group + block_ci - 1) // block_ci),
            kernel_d * kernel_h,
            num_splits,
        )
        atomic_static_args = (
            m,
            image_d,
            image_h,
            image_w,
            loss_d,
            loss_h,
            loss_w,
            cout_per_group,
            cin_per_group,
            *image_spec.stride,
            *loss_spec.stride,
            *output_stride,
            stride_d,
            stride_h,
            stride_w,
            pad_front,
            pad_top,
            pad_left,
            dilation_d,
            dilation_h,
            dilation_w,
            kernel_d,
            kernel_h,
            kernel_w,
            num_splits,
        )

        def context_factory(inputs: Sequence[Any]) -> torch.Tensor:
            image = inputs[0]
            assert isinstance(image, torch.Tensor)
            return get_output(image)

        def zero_args(
            _inputs: Sequence[Any], output: torch.Tensor
        ) -> tuple[torch.Tensor]:
            return (output,)

        def atomic_args(
            inputs: Sequence[Any], output: torch.Tensor
        ) -> tuple[Any, ...]:
            return inputs[0], inputs[1], output

        def build_zero_cached_call(
            _metadata: dict[str, Any],
        ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
            return zero_grid, (total, zero_block)

        def build_atomic_cached_call(
            _metadata: dict[str, Any],
        ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
            return atomic_grid, atomic_static_args + (
                block_co,
                block_ci,
                block_m,
            )

        pipeline = PreparedKernelPipelineSpec(
            steps=(
                PreparedPipelineStepSpec(
                    kernel=_conv_wgrad_zero_kernel,
                    grid=zero_grid,
                    runtime_args=zero_args,
                    static_args=(total,),
                    constexpr_kwargs={
                        "BLOCK": zero_block,
                        "num_warps": 4,
                    },
                    build_cached_call=build_zero_cached_call,
                ),
                PreparedPipelineStepSpec(
                    kernel=_conv_wgrad3d_kw3_atomic_kernel,
                    grid=atomic_grid,
                    runtime_args=atomic_args,
                    static_args=atomic_static_args,
                    constexpr_kwargs={
                        "BLOCK_CO": block_co,
                        "BLOCK_CI": block_ci,
                        "BLOCK_M": block_m,
                        "num_warps": 4,
                        "num_stages": 3,
                    },
                    build_cached_call=build_atomic_cached_call,
                ),
            ),
            input_checks=checks,
            context_factory=context_factory,
            result=lambda output: output,
            extra_check=extra_check,
        )
        return _make_bound_pipeline_run_fn(
            pipeline,
            default_run_fn,
            validate_inputs=bool(attrs.get("_validate_inputs", True)),
        )

    splits_per_n = 8
    num_splits = n * splits_per_n
    block_co = 16
    block_ci = 8
    block_m = 64
    cta_x = ((cout_per_group + block_co - 1) // block_co) * (
        (cin_per_group + block_ci - 1) // block_ci
    )
    split_grid = (cta_x, kernel_elems, num_splits)
    reduce_grid = (cta_x, kernel_elems, groups)
    split_static_args = (
        image_d,
        image_h,
        image_w,
        loss_d,
        loss_h,
        loss_w,
        c_out,
        cin_per_group,
        cout_per_group,
        *image_spec.stride,
        *loss_spec.stride,
        stride_d,
        stride_h,
        stride_w,
        pad_front,
        pad_top,
        pad_left,
        dilation_d,
        dilation_h,
        dilation_w,
        kernel_d,
        kernel_h,
        kernel_w,
        splits_per_n,
    )
    reduce_static_args = (
        c_out,
        cin_per_group,
        cout_per_group,
        *output_stride,
        kernel_d,
        kernel_h,
        kernel_w,
        num_splits,
    )

    def nsplit_context(
        inputs: Sequence[Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = inputs[0]
        assert isinstance(image, torch.Tensor)
        output = get_output(image)
        partial_shape = (
            num_splits,
            c_out,
            cin_per_group,
            kernel_elems,
        )
        key = (
            image.device.type,
            image.device.index,
            torch.float32,
            partial_shape,
        )
        partial = workspace_cache.get(key)
        if partial is None:
            partial = torch.empty(
                partial_shape, device=image.device, dtype=torch.float32
            )
            workspace_cache[key] = partial
        return output, partial

    def split_args(
        inputs: Sequence[Any], context: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[Any, ...]:
        return inputs[0], inputs[1], context[1]

    def reduce_args(
        _inputs: Sequence[Any], context: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[Any, ...]:
        return context[1], context[0]

    def build_split_cached_call(
        _metadata: dict[str, Any],
    ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
        return split_grid, split_static_args + (
            block_co,
            block_ci,
            block_m,
        )

    def build_reduce_cached_call(
        _metadata: dict[str, Any],
    ) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
        return reduce_grid, reduce_static_args + (block_co, block_ci)

    pipeline = PreparedKernelPipelineSpec(
        steps=(
            PreparedPipelineStepSpec(
                kernel=_conv_wgrad3d_valid_nsplit_kernel,
                grid=split_grid,
                runtime_args=split_args,
                static_args=split_static_args,
                constexpr_kwargs={
                    "BLOCK_CO": block_co,
                    "BLOCK_CI": block_ci,
                    "BLOCK_M": block_m,
                    "num_warps": 4,
                    "num_stages": 3,
                },
                build_cached_call=build_split_cached_call,
            ),
            PreparedPipelineStepSpec(
                kernel=_conv_wgrad3d_reduce_kernel,
                grid=reduce_grid,
                runtime_args=reduce_args,
                static_args=reduce_static_args,
                constexpr_kwargs={
                    "BLOCK_CO": block_co,
                    "BLOCK_CI": block_ci,
                    "num_warps": 4,
                    "num_stages": 1,
                },
                build_cached_call=build_reduce_cached_call,
            ),
        ),
        input_checks=checks,
        context_factory=nsplit_context,
        result=lambda context: context[0],
        extra_check=extra_check,
    )
    return _make_bound_pipeline_run_fn(
        pipeline,
        default_run_fn,
        validate_inputs=bool(attrs.get("_validate_inputs", True)),
    )


def prepare_conv(
    op_type: str,
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if op_type == "conv_fprop":
        if len(input_specs) >= 2:
            image_shape = _static_shape(input_specs[0])
            weight_shape = _static_shape(input_specs[1])
            if image_shape is not None and weight_shape is not None:
                if len(image_shape) == 3 and len(weight_shape) == 3:
                    return _prepare_fprop_1d(
                        attrs, input_specs, default_run_fn
                    )
                if len(image_shape) == 4 and len(weight_shape) == 4:
                    return _prepare_fprop_2d(
                        attrs, input_specs, default_run_fn
                    )
                if len(image_shape) == 5 and len(weight_shape) == 5:
                    return _prepare_fprop_3d(
                        attrs, input_specs, default_run_fn
                    )
    if op_type == "conv_dgrad":
        input_size = tuple(int(dim) for dim in attrs.get("input_size", ()))
        if len(input_size) == 3:
            return _prepare_dgrad_1d(attrs, input_specs, default_run_fn)
        if len(input_size) == 4:
            prepared = _prepare_dgrad_2d_stride1(
                attrs, input_specs, default_run_fn
            )
            if prepared is not None:
                return prepared
            return _prepare_dgrad_2d_stride2(
                attrs, input_specs, default_run_fn
            )
        if len(input_size) == 5:
            return _prepare_dgrad_3d(attrs, input_specs, default_run_fn)
    if op_type == "conv_wgrad":
        filter_size = tuple(int(dim) for dim in attrs.get("filter_size", ()))
        if len(filter_size) == 3:
            return _prepare_wgrad_1d(attrs, input_specs, default_run_fn)
        if len(filter_size) == 4:
            return _prepare_wgrad_2d(attrs, input_specs, default_run_fn)
        if len(filter_size) == 5:
            return _prepare_wgrad_3d(attrs, input_specs, default_run_fn)
    return None


__all__ = ("prepare_conv",)
