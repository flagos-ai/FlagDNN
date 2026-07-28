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

"""Ascend-only Triton convolution kernels and prepared graph dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import torch
import triton
import triton.language as tl

from flag_dnn.graph.prepared import (
    RunFn,
    runtime_tensor_checks_from_specs,
    runtime_tensor_checks_pass,
)
from flag_dnn.graph.prepared.common import (
    _is_runtime_device_spec,
    _static_shape,
)
from flag_dnn.graph.tensor import TensorSpec, torch_dtype
from flag_dnn.runtime import torch_device_fn
from flag_dnn.runtime.backend._ascend.ops.matmul import matmul_3d_out
from flag_dnn.utils.libentry import libentry


_SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")


@libentry()
@triton.jit
def _conv_fprop_small_reduction_kernel(
    image_ptr,
    weight_ptr,
    output_ptr,
    N: tl.constexpr,
    GROUPS: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    IMAGE_STRIDE_N: tl.constexpr,
    IMAGE_STRIDE_C: tl.constexpr,
    IMAGE_STRIDE_D: tl.constexpr,
    IMAGE_STRIDE_H: tl.constexpr,
    IMAGE_STRIDE_W: tl.constexpr,
    WEIGHT_STRIDE_O: tl.constexpr,
    WEIGHT_STRIDE_I: tl.constexpr,
    WEIGHT_STRIDE_D: tl.constexpr,
    WEIGHT_STRIDE_H: tl.constexpr,
    WEIGHT_STRIDE_W: tl.constexpr,
    OUTPUT_STRIDE_N: tl.constexpr,
    OUTPUT_STRIDE_C: tl.constexpr,
    OUTPUT_STRIDE_D: tl.constexpr,
    OUTPUT_STRIDE_H: tl.constexpr,
    OUTPUT_STRIDE_W: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    output_spatial = OD * OH * OW
    tiles_m = tl.cdiv(output_spatial, BLOCK_M)
    tiles_oc = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    tiles_per_matrix = tiles_m * tiles_oc
    program_id = tl.program_id(0)
    matrix_id = program_id // tiles_per_matrix
    tile_id = program_id - matrix_id * tiles_per_matrix
    tile_oc = tile_id // tiles_m
    tile_m = tile_id - tile_oc * tiles_m
    conv_group = matrix_id % GROUPS
    batch = matrix_id // GROUPS

    offsets_oc = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offsets_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_oc = offsets_oc < COUT_PER_GROUP
    mask_m = offsets_m < output_spatial
    oc = conv_group * COUT_PER_GROUP + offsets_oc
    ow = offsets_m % OW
    oh = (offsets_m // OW) % OH
    od = offsets_m // (OH * OW)
    accumulator = tl.zeros((BLOCK_OC, BLOCK_M), dtype=tl.float32)

    for kd in tl.static_range(0, KD):
        input_d = od * STRIDE_D - PAD_D + kd * DIL_D
        valid_d = (input_d >= 0) & (input_d < XD)
        safe_d = tl.where(valid_d, input_d, 0)
        weight_d = KD - 1 - kd if FILTER_REVERSE else kd
        for kh in tl.static_range(0, KH):
            input_h = oh * STRIDE_H - PAD_H + kh * DIL_H
            valid_h = (input_h >= 0) & (input_h < XH)
            safe_h = tl.where(valid_h, input_h, 0)
            weight_h = KH - 1 - kh if FILTER_REVERSE else kh
            for kw in tl.static_range(0, KW):
                input_w = ow * STRIDE_W - PAD_W + kw * DIL_W
                valid_w = (input_w >= 0) & (input_w < XW)
                safe_w = tl.where(valid_w, input_w, 0)
                valid_spatial = mask_m & valid_d & valid_h & valid_w
                weight_w = KW - 1 - kw if FILTER_REVERSE else kw
                for ci_local in tl.static_range(0, CIN_PER_GROUP):
                    ci = conv_group * CIN_PER_GROUP + ci_local
                    image = tl.load(
                        image_ptr
                        + batch * IMAGE_STRIDE_N
                        + ci * IMAGE_STRIDE_C
                        + safe_d * IMAGE_STRIDE_D
                        + safe_h * IMAGE_STRIDE_H
                        + safe_w * IMAGE_STRIDE_W,
                        mask=valid_spatial,
                        other=0.0,
                    ).to(tl.float32)
                    weight = tl.load(
                        weight_ptr
                        + oc * WEIGHT_STRIDE_O
                        + ci_local * WEIGHT_STRIDE_I
                        + weight_d * WEIGHT_STRIDE_D
                        + weight_h * WEIGHT_STRIDE_H
                        + weight_w * WEIGHT_STRIDE_W,
                        mask=mask_oc,
                        other=0.0,
                    ).to(tl.float32)
                    accumulator += weight[:, None] * image[None, :]

    tl.store(
        output_ptr
        + batch * OUTPUT_STRIDE_N
        + oc[:, None] * OUTPUT_STRIDE_C
        + od[None, :] * OUTPUT_STRIDE_D
        + oh[None, :] * OUTPUT_STRIDE_H
        + ow[None, :] * OUTPUT_STRIDE_W,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_m[None, :],
    )


@libentry()
@triton.jit
def _conv_fprop_im2col_kernel(
    image_ptr,
    column_ptr,
    TOTAL: tl.constexpr,
    GROUPS: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    IMAGE_STRIDE_N: tl.constexpr,
    IMAGE_STRIDE_C: tl.constexpr,
    IMAGE_STRIDE_D: tl.constexpr,
    IMAGE_STRIDE_H: tl.constexpr,
    IMAGE_STRIDE_W: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    BLOCK_K_PACK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    output_spatial = OD * OH * OW
    kernel_volume = KD * KH * KW
    reduction = CIN_PER_GROUP * kernel_volume
    spatial = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    packed_k = tl.program_id(1) * BLOCK_K_PACK + tl.arange(0, BLOCK_K_PACK)
    batch_group = tl.program_id(2)
    mask_spatial = spatial < output_spatial
    mask_k = packed_k < reduction
    safe_packed_k = tl.where(mask_k, packed_k, 0)
    conv_group = batch_group % GROUPS
    batch = batch_group // GROUPS

    ci_local = safe_packed_k // kernel_volume
    kernel_offset = safe_packed_k - ci_local * kernel_volume
    weight_w = kernel_offset % KW
    weight_h = (kernel_offset // KW) % KH
    weight_d = kernel_offset // (KH * KW)
    sample_d = KD - 1 - weight_d if FILTER_REVERSE else weight_d
    sample_h = KH - 1 - weight_h if FILTER_REVERSE else weight_h
    sample_w = KW - 1 - weight_w if FILTER_REVERSE else weight_w

    ow = spatial % OW
    oh = (spatial // OW) % OH
    od = spatial // (OH * OW)
    input_d = od[None, :] * STRIDE_D - PAD_D + sample_d[:, None] * DIL_D
    input_h = oh[None, :] * STRIDE_H - PAD_H + sample_h[:, None] * DIL_H
    input_w = ow[None, :] * STRIDE_W - PAD_W + sample_w[:, None] * DIL_W
    valid = (
        mask_k[:, None]
        & mask_spatial[None, :]
        & (input_d >= 0)
        & (input_d < XD)
        & (input_h >= 0)
        & (input_h < XH)
        & (input_w >= 0)
        & (input_w < XW)
    )
    safe_d = tl.where(valid, input_d, 0)
    safe_h = tl.where(valid, input_h, 0)
    safe_w = tl.where(valid, input_w, 0)
    ci = conv_group * CIN_PER_GROUP + ci_local
    value = tl.load(
        image_ptr
        + batch * IMAGE_STRIDE_N
        + ci[:, None] * IMAGE_STRIDE_C
        + safe_d * IMAGE_STRIDE_D
        + safe_h * IMAGE_STRIDE_H
        + safe_w * IMAGE_STRIDE_W,
        mask=valid,
        other=0.0,
    )
    output_offsets = (
        batch_group * reduction + safe_packed_k[:, None]
    ) * output_spatial + spatial[None, :]
    tl.store(
        column_ptr + output_offsets,
        value,
        mask=mask_k[:, None] & mask_spatial[None, :],
    )


@libentry()
@triton.jit
def _conv_fprop_pack_nchw_to_nhwc_kernel(
    image_ptr,
    packed_image_ptr,
    SPATIAL: tl.constexpr,
    CHANNELS: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    """Stage one contiguous NCHW image as spatial-major channel blocks."""
    spatial = tl.program_id(0) * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
    channel = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    valid = (channel[:, None] < CHANNELS) & (spatial[None, :] < SPATIAL)
    value = tl.load(
        image_ptr + channel[:, None] * SPATIAL + spatial[None, :],
        mask=valid,
        other=0.0,
    )
    tl.store(
        packed_image_ptr + spatial[None, :] * CHANNELS + channel[:, None],
        value,
        mask=valid,
    )


@libentry()
@triton.jit
def _conv_fprop_pack_nhwc_im2col_kernel(
    packed_image_ptr,
    column_ptr,
    C_IN: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    """Pack a measured 3x3/stride-two image using contiguous channel DMA."""
    kernel_offset = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    spatial = tl.program_id(2) * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
    valid_channel = channel < C_IN
    valid_spatial = spatial < OH * OW
    safe_spatial = tl.where(valid_spatial, spatial, 0)
    output_h = safe_spatial // OW
    output_w = safe_spatial - output_h * OW
    kernel_h = kernel_offset // 3
    kernel_w = kernel_offset % 3
    input_h = output_h * 2 - 1 + kernel_h
    input_w = output_w * 2 - 1 + kernel_w
    valid_input = (
        valid_spatial
        & (input_h >= 0)
        & (input_h < XH)
        & (input_w >= 0)
        & (input_w < XW)
    )
    safe_h = tl.where(valid_input, input_h, 0)
    safe_w = tl.where(valid_input, input_w, 0)
    input_spatial = safe_h * XW + safe_w
    value = tl.load(
        packed_image_ptr + input_spatial[:, None] * C_IN + channel[None, :],
        mask=valid_input[:, None] & valid_channel[None, :],
        other=0.0,
    )
    output_spatial = OH * OW
    packed_k = channel * 9 + kernel_offset
    tl.store(
        column_ptr + packed_k[None, :] * output_spatial + spatial[:, None],
        value,
        mask=valid_spatial[:, None] & valid_channel[None, :],
    )


@libentry()
@triton.jit
def _conv_fprop_pack_nchw_to_nhwc_batched_kernel(
    image_ptr,
    packed_image_ptr,
    SPATIAL: tl.constexpr,
    CHANNELS: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    """Stage contiguous batched NCHW images as spatial-major channels."""
    spatial = tl.program_id(0) * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
    channel = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    batch = tl.program_id(2)
    valid = (channel[:, None] < CHANNELS) & (spatial[None, :] < SPATIAL)
    value = tl.load(
        image_ptr
        + batch * CHANNELS * SPATIAL
        + channel[:, None] * SPATIAL
        + spatial[None, :],
        mask=valid,
        other=0.0,
    )
    tl.store(
        packed_image_ptr
        + batch * SPATIAL * CHANNELS
        + spatial[None, :] * CHANNELS
        + channel[:, None],
        value,
        mask=valid,
    )


@libentry()
@triton.jit
def _conv_fprop_pack_nhwc_im2col_2d_persistent_kernel(
    packed_image_ptr,
    column_ptr,
    C_IN: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    """Pack arbitrary measured 2D windows from staged NHWC storage."""
    kernel_volume = KH * KW
    matrix_id = tl.program_id(0)
    batch = matrix_id // kernel_volume
    kernel_offset = matrix_id - batch * kernel_volume
    channel = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    valid_channel = channel < C_IN
    weight_h = kernel_offset // KW
    weight_w = kernel_offset - weight_h * KW
    sample_h = KH - 1 - weight_h if FILTER_REVERSE else weight_h
    sample_w = KW - 1 - weight_w if FILTER_REVERSE else weight_w
    output_spatial = OH * OW
    reduction = C_IN * kernel_volume
    packed_k = channel * kernel_volume + kernel_offset
    program_spatial = tl.program_id(2) * TILES_PER_PROGRAM * BLOCK_SPATIAL

    for tile in range(0, TILES_PER_PROGRAM):
        spatial = (
            program_spatial
            + tile * BLOCK_SPATIAL
            + tl.arange(0, BLOCK_SPATIAL)
        )
        valid_spatial = spatial < output_spatial
        safe_spatial = tl.where(valid_spatial, spatial, 0)
        output_h = safe_spatial // OW
        output_w = safe_spatial - output_h * OW
        input_h = output_h * STRIDE_H - PAD_H + sample_h * DIL_H
        input_w = output_w * STRIDE_W - PAD_W + sample_w * DIL_W
        valid_input = (
            valid_spatial
            & (input_h >= 0)
            & (input_h < XH)
            & (input_w >= 0)
            & (input_w < XW)
        )
        safe_h = tl.where(valid_input, input_h, 0)
        safe_w = tl.where(valid_input, input_w, 0)
        input_spatial = safe_h * XW + safe_w
        value = tl.load(
            packed_image_ptr
            + batch * XH * XW * C_IN
            + input_spatial[:, None] * C_IN
            + channel[None, :],
            mask=valid_input[:, None] & valid_channel[None, :],
            other=0.0,
        )
        tl.store(
            column_ptr
            + (batch * reduction + packed_k[None, :]) * output_spatial
            + spatial[:, None],
            value,
            mask=valid_spatial[:, None] & valid_channel[None, :],
        )


@libentry()
@triton.jit
def _conv_fprop_im2col_1d_block_kernel(
    image_ptr,
    column_ptr,
    GROUPS: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    XW: tl.constexpr,
    OW: tl.constexpr,
    KW: tl.constexpr,
    REDUCTION: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_W: tl.constexpr,
    IMAGE_STRIDE_N: tl.constexpr,
    IMAGE_STRIDE_C: tl.constexpr,
    IMAGE_STRIDE_W: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    block_w_start = tl.program_id(0) * BLOCK_W
    block_ci_start = tl.program_id(1) * BLOCK_CI
    batch_group = tl.program_id(2)
    conv_group = batch_group % GROUPS
    batch = batch_group // GROUPS
    image_base = (
        image_ptr
        + batch * IMAGE_STRIDE_N
        + conv_group * CIN_PER_GROUP * IMAGE_STRIDE_C
    )
    column_base = column_ptr + batch_group * REDUCTION * OW

    for kernel_w in tl.static_range(0, KW):
        sample_w = KW - 1 - kernel_w if FILTER_REVERSE else kernel_w
        image_block = tl.make_block_ptr(
            base=image_base,
            shape=(CIN_PER_GROUP, XW),
            strides=(IMAGE_STRIDE_C, IMAGE_STRIDE_W),
            offsets=(
                block_ci_start,
                block_w_start - PAD_W + sample_w * DIL_W,
            ),
            block_shape=(BLOCK_CI, BLOCK_W),
            order=(1, 0),
        )
        value = tl.load(
            image_block,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        column_block = tl.make_block_ptr(
            base=column_base + kernel_w * OW,
            shape=(CIN_PER_GROUP, OW),
            strides=(KW * OW, 1),
            offsets=(block_ci_start, block_w_start),
            block_shape=(BLOCK_CI, BLOCK_W),
            order=(1, 0),
        )
        tl.store(
            column_block,
            value,
            boundary_check=(0, 1),
        )


@libentry()
@triton.jit
def _conv_fprop_im2col_1d_gather_kernel(
    image_ptr,
    column_ptr,
    GROUPS: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    XW: tl.constexpr,
    OW: tl.constexpr,
    KW: tl.constexpr,
    REDUCTION: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_W: tl.constexpr,
    IMAGE_STRIDE_N: tl.constexpr,
    IMAGE_STRIDE_C: tl.constexpr,
    IMAGE_STRIDE_W: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    LOAD_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    block_w_start = tl.program_id(0) * BLOCK_W
    block_ci_start = tl.program_id(1) * BLOCK_CI
    batch_group = tl.program_id(2)
    conv_group = batch_group % GROUPS
    batch = batch_group // GROUPS
    image_base = (
        image_ptr
        + batch * IMAGE_STRIDE_N
        + conv_group * CIN_PER_GROUP * IMAGE_STRIDE_C
    )
    image_block = tl.make_block_ptr(
        base=image_base,
        shape=(CIN_PER_GROUP, XW),
        strides=(IMAGE_STRIDE_C, IMAGE_STRIDE_W),
        offsets=(block_ci_start, 0),
        block_shape=(BLOCK_CI, LOAD_W),
        order=(1, 0),
    )
    image = tl.load(
        image_block,
        boundary_check=(0, 1),
        padding_option="zero",
    )
    ow = block_w_start + tl.arange(0, BLOCK_W)
    column_base = column_ptr + batch_group * REDUCTION * OW

    for kernel_w in tl.static_range(0, KW):
        sample_w = KW - 1 - kernel_w if FILTER_REVERSE else kernel_w
        input_w = ow * STRIDE_W - PAD_W + sample_w * DIL_W
        valid = (input_w >= 0) & (input_w < XW) & (ow < OW)
        safe_input_w = tl.where(valid, input_w, 0)
        index = tl.broadcast_to(
            safe_input_w[None, :],
            (BLOCK_CI, BLOCK_W),
        )
        value = tl.gather(image, index, axis=1)
        value = tl.where(valid[None, :], value, 0.0)
        column_block = tl.make_block_ptr(
            base=column_base + kernel_w * OW,
            shape=(CIN_PER_GROUP, OW),
            strides=(KW * OW, 1),
            offsets=(block_ci_start, block_w_start),
            block_shape=(BLOCK_CI, BLOCK_W),
            order=(1, 0),
        )
        tl.store(
            column_block,
            value,
            boundary_check=(0, 1),
        )


@libentry()
@triton.jit
def _conv_fprop_im2col_2d_gather_kernel(
    image_ptr,
    column_ptr,
    GROUPS: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    IMAGE_STRIDE_N: tl.constexpr,
    IMAGE_STRIDE_C: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    LOAD_X: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
):
    block_ci_start = tl.program_id(0) * BLOCK_CI
    batch_group = tl.program_id(1)
    conv_group = batch_group % GROUPS
    batch = batch_group // GROUPS
    input_spatial = XH * XW
    output_spatial = OH * OW
    kernel_volume = KH * KW
    image_base = (
        image_ptr
        + batch * IMAGE_STRIDE_N
        + conv_group * CIN_PER_GROUP * IMAGE_STRIDE_C
    )
    image_block = tl.make_block_ptr(
        base=image_base,
        shape=(CIN_PER_GROUP, input_spatial),
        strides=(IMAGE_STRIDE_C, 1),
        offsets=(block_ci_start, 0),
        block_shape=(BLOCK_CI, LOAD_X),
        order=(1, 0),
    )
    image = tl.load(
        image_block,
        boundary_check=(0, 1),
        padding_option="zero",
    )
    column_base = (
        column_ptr
        + batch_group * CIN_PER_GROUP * kernel_volume * output_spatial
    )

    for kernel_h in tl.static_range(0, KH):
        sample_h = KH - 1 - kernel_h if FILTER_REVERSE else kernel_h
        for kernel_w in tl.static_range(0, KW):
            sample_w = KW - 1 - kernel_w if FILTER_REVERSE else kernel_w
            kernel_offset = kernel_h * KW + kernel_w
            for spatial_start in tl.range(
                0,
                output_spatial,
                BLOCK_SPATIAL,
                disallow_acc_multi_buffer=True,
            ):
                spatial = spatial_start + tl.arange(0, BLOCK_SPATIAL)
                oh = spatial // OW
                ow = spatial - oh * OW
                input_h = oh * STRIDE_H - PAD_H + sample_h * DIL_H
                input_w = ow * STRIDE_W - PAD_W + sample_w * DIL_W
                valid = (
                    (spatial < output_spatial)
                    & (input_h >= 0)
                    & (input_h < XH)
                    & (input_w >= 0)
                    & (input_w < XW)
                )
                safe_input = tl.where(
                    valid,
                    input_h * XW + input_w,
                    0,
                )
                index = tl.broadcast_to(
                    safe_input[None, :],
                    (BLOCK_CI, BLOCK_SPATIAL),
                )
                value = tl.gather(image, index, axis=1)
                value = tl.where(
                    valid[None, :],
                    value,
                    0.0,
                )
                column_block = tl.make_block_ptr(
                    base=(column_base + kernel_offset * output_spatial),
                    shape=(CIN_PER_GROUP, output_spatial),
                    strides=(
                        kernel_volume * output_spatial,
                        1,
                    ),
                    offsets=(block_ci_start, spatial_start),
                    block_shape=(BLOCK_CI, BLOCK_SPATIAL),
                    order=(1, 0),
                )
                tl.store(
                    column_block,
                    value,
                    boundary_check=(0, 1),
                )


@libentry()
@triton.jit
def _conv_fprop_im2col_2d_stem_row_reuse_kernel(
    image_ptr,
    column_ptr,
    CIN_PER_GROUP: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    IMAGE_STRIDE_N: tl.constexpr,
    IMAGE_STRIDE_C: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    LOAD_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    channel_kernel_h = tl.program_id(0)
    channel = channel_kernel_h // KH
    kernel_h = channel_kernel_h - channel * KH
    sample_h = KH - 1 - kernel_h if FILTER_REVERSE else kernel_h
    output_h = tl.program_id(1) * ROWS_PER_PROGRAM + tl.arange(
        0, ROWS_PER_PROGRAM
    )
    output_w = tl.arange(0, BLOCK_W)
    load_w = tl.arange(0, LOAD_W)
    batch = tl.program_id(2)
    input_h = output_h * STRIDE_H - PAD_H + sample_h * DIL_H
    valid_h = (output_h < OH) & (input_h >= 0) & (input_h < XH)
    safe_h = tl.where(valid_h, input_h, 0)
    image = tl.load(
        image_ptr
        + batch * IMAGE_STRIDE_N
        + channel * IMAGE_STRIDE_C
        + safe_h[:, None] * XW
        + load_w[None, :],
        mask=valid_h[:, None] & (load_w[None, :] < XW),
        other=0.0,
    )
    output_spatial = OH * OW
    kernel_volume = KH * KW
    column_base = (
        column_ptr + batch * CIN_PER_GROUP * kernel_volume * output_spatial
    )
    for kernel_w in tl.static_range(0, KW):
        sample_w = KW - 1 - kernel_w if FILTER_REVERSE else kernel_w
        input_w = output_w * STRIDE_W - PAD_W + sample_w * DIL_W
        valid_w = (output_w < OW) & (input_w >= 0) & (input_w < XW)
        safe_w = tl.where(valid_w, input_w, 0)
        index = tl.broadcast_to(
            safe_w[None, :],
            (ROWS_PER_PROGRAM, BLOCK_W),
        )
        value = tl.gather(image, index, axis=1)
        valid = valid_h[:, None] & valid_w[None, :]
        value = tl.where(valid, value, 0.0)
        packed_k = channel * kernel_volume + kernel_h * KW + kernel_w
        tl.store(
            column_base
            + packed_k * output_spatial
            + output_h[:, None] * OW
            + output_w[None, :],
            value,
            mask=((output_h[:, None] < OH) & (output_w[None, :] < OW)),
        )


@libentry()
@triton.jit
def _conv_fprop_im2col_3d_full_volume_kernel(
    image_ptr,
    column_ptr,
    CIN_PER_GROUP: tl.constexpr,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    IMAGE_STRIDE_N: tl.constexpr,
    IMAGE_STRIDE_C: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    LOAD_X: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
):
    block_ci_start = tl.program_id(0) * BLOCK_CI
    batch = tl.program_id(1)
    input_spatial = XD * XH * XW
    output_spatial = OD * OH * OW
    kernel_volume = KD * KH * KW
    image_block = tl.make_block_ptr(
        base=image_ptr + batch * IMAGE_STRIDE_N,
        shape=(CIN_PER_GROUP, input_spatial),
        strides=(IMAGE_STRIDE_C, 1),
        offsets=(block_ci_start, 0),
        block_shape=(BLOCK_CI, LOAD_X),
        order=(1, 0),
    )
    image = tl.load(
        image_block,
        boundary_check=(0, 1),
        padding_option="zero",
    )
    column_base = (
        column_ptr + batch * CIN_PER_GROUP * kernel_volume * output_spatial
    )

    for kernel_d in tl.static_range(0, KD):
        sample_d = KD - 1 - kernel_d if FILTER_REVERSE else kernel_d
        for kernel_h in tl.static_range(0, KH):
            sample_h = KH - 1 - kernel_h if FILTER_REVERSE else kernel_h
            for kernel_w in tl.static_range(0, KW):
                sample_w = KW - 1 - kernel_w if FILTER_REVERSE else kernel_w
                kernel_offset = kernel_d * KH * KW + kernel_h * KW + kernel_w
                for spatial_start in tl.range(
                    0,
                    output_spatial,
                    BLOCK_SPATIAL,
                    disallow_acc_multi_buffer=True,
                ):
                    spatial = spatial_start + tl.arange(0, BLOCK_SPATIAL)
                    output_d = spatial // (OH * OW)
                    output_hw = spatial - output_d * OH * OW
                    output_h = output_hw // OW
                    output_w = output_hw - output_h * OW
                    input_d = output_d * STRIDE_D - PAD_D + sample_d * DIL_D
                    input_h = output_h * STRIDE_H - PAD_H + sample_h * DIL_H
                    input_w = output_w * STRIDE_W - PAD_W + sample_w * DIL_W
                    valid = (
                        (spatial < output_spatial)
                        & (input_d >= 0)
                        & (input_d < XD)
                        & (input_h >= 0)
                        & (input_h < XH)
                        & (input_w >= 0)
                        & (input_w < XW)
                    )
                    safe_input = tl.where(
                        valid,
                        (input_d * XH * XW + input_h * XW + input_w),
                        0,
                    )
                    index = tl.broadcast_to(
                        safe_input[None, :],
                        (BLOCK_CI, BLOCK_SPATIAL),
                    )
                    value = tl.gather(image, index, axis=1)
                    value = tl.where(
                        valid[None, :],
                        value,
                        0.0,
                    )
                    column_block = tl.make_block_ptr(
                        base=(column_base + kernel_offset * output_spatial),
                        shape=(
                            CIN_PER_GROUP,
                            output_spatial,
                        ),
                        strides=(
                            kernel_volume * output_spatial,
                            1,
                        ),
                        offsets=(
                            block_ci_start,
                            spatial_start,
                        ),
                        block_shape=(
                            BLOCK_CI,
                            BLOCK_SPATIAL,
                        ),
                        order=(1, 0),
                    )
                    tl.store(
                        column_block,
                        value,
                        boundary_check=(0, 1),
                    )


@libentry()
@triton.jit
def _conv_fprop_im2col_2d_row_tile_gather_kernel(
    image_ptr,
    column_ptr,
    GROUPS: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    IMAGE_STRIDE_N: tl.constexpr,
    IMAGE_STRIDE_C: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    ROW_TILES: tl.constexpr,
    ROWS_PER_TILE: tl.constexpr,
    LOAD_X: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
):
    block_ci_start = tl.program_id(0) * BLOCK_CI
    batch_group = tl.program_id(1)
    conv_group = batch_group % GROUPS
    batch = batch_group // GROUPS
    input_spatial = XH * XW
    output_spatial = OH * OW
    kernel_volume = KH * KW
    image_base = (
        image_ptr
        + batch * IMAGE_STRIDE_N
        + conv_group * CIN_PER_GROUP * IMAGE_STRIDE_C
    )
    column_base = (
        column_ptr
        + batch_group * CIN_PER_GROUP * kernel_volume * output_spatial
    )

    for row_tile in tl.static_range(0, ROW_TILES):
        output_row_start = row_tile * ROWS_PER_TILE
        output_start = output_row_start * OW
        input_row_start = tl.maximum(
            output_row_start * STRIDE_H - PAD_H,
            0,
        )
        input_start = input_row_start * XW
        image_block = tl.make_block_ptr(
            base=image_base,
            shape=(CIN_PER_GROUP, input_spatial),
            strides=(IMAGE_STRIDE_C, 1),
            offsets=(block_ci_start, input_start),
            block_shape=(BLOCK_CI, LOAD_X),
            order=(1, 0),
        )
        image = tl.load(
            image_block,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        local_spatial = tl.arange(0, BLOCK_SPATIAL)
        spatial = output_start + local_spatial
        oh = spatial // OW
        ow = spatial - oh * OW
        mask_spatial = (local_spatial < ROWS_PER_TILE * OW) & (
            spatial < output_spatial
        )

        for kernel_h in tl.static_range(0, KH):
            sample_h = KH - 1 - kernel_h if FILTER_REVERSE else kernel_h
            for kernel_w in tl.static_range(0, KW):
                sample_w = KW - 1 - kernel_w if FILTER_REVERSE else kernel_w
                input_h = oh * STRIDE_H - PAD_H + sample_h * DIL_H
                input_w = ow * STRIDE_W - PAD_W + sample_w * DIL_W
                valid = (
                    mask_spatial
                    & (input_h >= 0)
                    & (input_h < XH)
                    & (input_w >= 0)
                    & (input_w < XW)
                )
                safe_input = tl.where(
                    valid,
                    input_h * XW + input_w - input_start,
                    0,
                )
                index = tl.broadcast_to(
                    safe_input[None, :],
                    (BLOCK_CI, BLOCK_SPATIAL),
                )
                value = tl.gather(image, index, axis=1)
                value = tl.where(
                    valid[None, :],
                    value,
                    0.0,
                )
                kernel_offset = kernel_h * KW + kernel_w
                column_block = tl.make_block_ptr(
                    base=(column_base + kernel_offset * output_spatial),
                    shape=(CIN_PER_GROUP, output_spatial),
                    strides=(
                        kernel_volume * output_spatial,
                        1,
                    ),
                    offsets=(block_ci_start, output_start),
                    block_shape=(BLOCK_CI, BLOCK_SPATIAL),
                    order=(1, 0),
                )
                tl.store(
                    column_block,
                    value,
                    boundary_check=(0, 1),
                )


@libentry()
@triton.jit
def _conv_fprop_gemm_kernel(
    column_ptr,
    weight_ptr,
    output_ptr,
    N: tl.constexpr,
    GROUPS: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    WEIGHT_STRIDE_O: tl.constexpr,
    WEIGHT_STRIDE_I: tl.constexpr,
    WEIGHT_STRIDE_D: tl.constexpr,
    WEIGHT_STRIDE_H: tl.constexpr,
    WEIGHT_STRIDE_W: tl.constexpr,
    OUTPUT_STRIDE_N: tl.constexpr,
    OUTPUT_STRIDE_C: tl.constexpr,
    OUTPUT_STRIDE_D: tl.constexpr,
    OUTPUT_STRIDE_H: tl.constexpr,
    OUTPUT_STRIDE_W: tl.constexpr,
    FP32_INPUT: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    output_spatial = OD * OH * OW
    kernel_volume = KD * KH * KW
    reduction = CIN_PER_GROUP * kernel_volume
    tiles_m = tl.cdiv(output_spatial, BLOCK_M)
    tiles_oc = tl.cdiv(COUT_PER_GROUP, BLOCK_OC)
    tiles_per_matrix = tiles_m * tiles_oc
    program_id = tl.program_id(0)
    matrix_id = program_id // tiles_per_matrix
    tile_id = program_id - matrix_id * tiles_per_matrix
    tile_oc = tile_id // tiles_m
    tile_m = tile_id - tile_oc * tiles_m
    conv_group = matrix_id % GROUPS
    batch = matrix_id // GROUPS

    offsets_oc = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offsets_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_oc = offsets_oc < COUT_PER_GROUP
    mask_m = offsets_m < output_spatial
    oc = conv_group * COUT_PER_GROUP + offsets_oc
    accumulator = tl.zeros((BLOCK_OC, BLOCK_M), dtype=tl.float32)
    column_base = matrix_id * reduction * output_spatial

    for reduction_start in range(0, reduction, BLOCK_K):
        offsets_k = reduction_start + tl.arange(0, BLOCK_K)
        mask_k = offsets_k < reduction
        safe_k = tl.where(mask_k, offsets_k, 0)
        ci_local = safe_k // kernel_volume
        kernel_offset = safe_k - ci_local * kernel_volume
        kw = kernel_offset % KW
        kh = (kernel_offset // KW) % KH
        kd = kernel_offset // (KH * KW)
        weight = tl.load(
            weight_ptr
            + oc[:, None] * WEIGHT_STRIDE_O
            + ci_local[None, :] * WEIGHT_STRIDE_I
            + kd[None, :] * WEIGHT_STRIDE_D
            + kh[None, :] * WEIGHT_STRIDE_H
            + kw[None, :] * WEIGHT_STRIDE_W,
            mask=mask_oc[:, None] & mask_k[None, :],
            other=0.0,
        )
        column = tl.load(
            column_ptr
            + column_base
            + offsets_k[:, None] * output_spatial
            + offsets_m[None, :],
            mask=mask_k[:, None] & mask_m[None, :],
            other=0.0,
        )
        if FP32_INPUT:
            accumulator = tl.dot(
                weight,
                column,
                accumulator,
                input_precision="tf32",
            )
        else:
            accumulator = tl.dot(weight, column, accumulator)

    ow = offsets_m % OW
    oh = (offsets_m // OW) % OH
    od = offsets_m // (OH * OW)
    tl.store(
        output_ptr
        + batch * OUTPUT_STRIDE_N
        + oc[:, None] * OUTPUT_STRIDE_C
        + od[None, :] * OUTPUT_STRIDE_D
        + oh[None, :] * OUTPUT_STRIDE_H
        + ow[None, :] * OUTPUT_STRIDE_W,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=mask_oc[:, None] & mask_m[None, :],
    )


@libentry()
@triton.jit
def _conv_dgrad_pack_weight_kernel(
    weight_ptr,
    packed_ptr,
    TOTAL_WEIGHT: tl.constexpr,
    GROUPS: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    WEIGHT_STRIDE_O: tl.constexpr,
    WEIGHT_STRIDE_I: tl.constexpr,
    WEIGHT_STRIDE_D: tl.constexpr,
    WEIGHT_STRIDE_H: tl.constexpr,
    WEIGHT_STRIDE_W: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    kernel_volume = KD * KH * KW
    chunk_start = tl.program_id(0) * CHUNK_SIZE
    for block_start in tl.range(0, CHUNK_SIZE, BLOCK):
        offsets = chunk_start + block_start + tl.arange(0, BLOCK)
        mask = offsets < TOTAL_WEIGHT
        safe_offsets = tl.where(mask, offsets, 0)
        co_local = safe_offsets % COUT_PER_GROUP
        remainder = safe_offsets // COUT_PER_GROUP
        ci_local = remainder % CIN_PER_GROUP
        remainder = remainder // CIN_PER_GROUP
        kernel_offset = remainder % kernel_volume
        conv_group = (remainder // kernel_volume) % GROUPS
        sample_w = kernel_offset % KW
        sample_h = (kernel_offset // KW) % KH
        sample_d = kernel_offset // (KH * KW)
        weight_d = KD - 1 - sample_d if FILTER_REVERSE else sample_d
        weight_h = KH - 1 - sample_h if FILTER_REVERSE else sample_h
        weight_w = KW - 1 - sample_w if FILTER_REVERSE else sample_w
        co = conv_group * COUT_PER_GROUP + co_local
        value = tl.load(
            weight_ptr
            + co * WEIGHT_STRIDE_O
            + ci_local * WEIGHT_STRIDE_I
            + weight_d * WEIGHT_STRIDE_D
            + weight_h * WEIGHT_STRIDE_H
            + weight_w * WEIGHT_STRIDE_W,
            mask=mask,
            other=0.0,
        )
        tl.store(packed_ptr + safe_offsets, value, mask=mask)


@libentry()
@triton.jit
def _conv_dgrad_pack_weight_tiled_kernel(
    weight_ptr,
    packed_ptr,
    GROUPS: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    WEIGHT_STRIDE_O: tl.constexpr,
    WEIGHT_STRIDE_I: tl.constexpr,
    WEIGHT_STRIDE_D: tl.constexpr,
    WEIGHT_STRIDE_H: tl.constexpr,
    WEIGHT_STRIDE_W: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    kernel_volume = KD * KH * KW
    tiles_ci = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    tiles_co = tl.cdiv(COUT_PER_GROUP, BLOCK_CO)
    tiles_per_matrix = tiles_ci * tiles_co
    program_id = tl.program_id(0)
    matrix_id = program_id // tiles_per_matrix
    tile_id = program_id - matrix_id * tiles_per_matrix
    tile_ci = tile_id // tiles_co
    tile_co = tile_id - tile_ci * tiles_co
    conv_group = matrix_id // kernel_volume
    kernel_offset = matrix_id - conv_group * kernel_volume

    offsets_ci = tile_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    offsets_co = tile_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    mask_ci = offsets_ci < CIN_PER_GROUP
    mask_co = offsets_co < COUT_PER_GROUP
    safe_ci = tl.where(mask_ci, offsets_ci, 0)
    safe_co = tl.where(mask_co, offsets_co, 0)
    sample_w = kernel_offset % KW
    sample_h = (kernel_offset // KW) % KH
    sample_d = kernel_offset // (KH * KW)
    weight_d = KD - 1 - sample_d if FILTER_REVERSE else sample_d
    weight_h = KH - 1 - sample_h if FILTER_REVERSE else sample_h
    weight_w = KW - 1 - sample_w if FILTER_REVERSE else sample_w
    co = conv_group * COUT_PER_GROUP + safe_co
    value = tl.load(
        weight_ptr
        + co[None, :] * WEIGHT_STRIDE_O
        + safe_ci[:, None] * WEIGHT_STRIDE_I
        + weight_d * WEIGHT_STRIDE_D
        + weight_h * WEIGHT_STRIDE_H
        + weight_w * WEIGHT_STRIDE_W,
        mask=mask_ci[:, None] & mask_co[None, :],
        other=0.0,
    )
    packed_base = matrix_id * CIN_PER_GROUP * COUT_PER_GROUP
    tl.store(
        packed_ptr
        + packed_base
        + safe_ci[:, None] * COUT_PER_GROUP
        + safe_co[None, :],
        value,
        mask=mask_ci[:, None] & mask_co[None, :],
    )


@libentry()
@triton.jit
def _conv_dgrad_pack_transposed_3d_weight_kernel(
    weight_ptr,
    packed_ptr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    WEIGHT_STRIDE_O: tl.constexpr,
    WEIGHT_STRIDE_I: tl.constexpr,
    WEIGHT_STRIDE_D: tl.constexpr,
    WEIGHT_STRIDE_H: tl.constexpr,
    WEIGHT_STRIDE_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    kernel_volume = KD * KH * KW
    total = C_IN * C_OUT * kernel_volume
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < total
    safe_offset = tl.where(valid, offset, 0)
    new_kernel = safe_offset % kernel_volume
    channel_out = (safe_offset // kernel_volume) % C_OUT
    channel_in = safe_offset // (C_OUT * kernel_volume)
    new_kernel_w = new_kernel % KW
    new_kernel_h = (new_kernel // KW) % KH
    new_kernel_d = new_kernel // (KH * KW)
    old_kernel_d = KD - 1 - new_kernel_d
    old_kernel_h = KH - 1 - new_kernel_h
    old_kernel_w = KW - 1 - new_kernel_w
    value = tl.load(
        weight_ptr
        + channel_out * WEIGHT_STRIDE_O
        + channel_in * WEIGHT_STRIDE_I
        + old_kernel_d * WEIGHT_STRIDE_D
        + old_kernel_h * WEIGHT_STRIDE_H
        + old_kernel_w * WEIGHT_STRIDE_W,
        mask=valid,
        other=0.0,
    )
    tl.store(
        packed_ptr + offset,
        value,
        mask=valid,
    )


@libentry()
@triton.jit
def _conv_dgrad_pack_pointwise_weight_kernel(
    weight_ptr,
    packed_ptr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    WEIGHT_STRIDE_O: tl.constexpr,
    WEIGHT_STRIDE_I: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    conv_group = tl.program_id(0)
    weight_block = tl.make_block_ptr(
        base=(weight_ptr + conv_group * COUT_PER_GROUP * WEIGHT_STRIDE_O),
        shape=(COUT_PER_GROUP, CIN_PER_GROUP),
        strides=(WEIGHT_STRIDE_O, WEIGHT_STRIDE_I),
        offsets=(0, 0),
        block_shape=(BLOCK_CO, BLOCK_CI),
        order=(1, 0),
    )
    weight = tl.load(
        weight_block,
        boundary_check=(0, 1),
        padding_option="zero",
    )
    packed_block = tl.make_block_ptr(
        base=(packed_ptr + conv_group * CIN_PER_GROUP * COUT_PER_GROUP),
        shape=(CIN_PER_GROUP, COUT_PER_GROUP),
        strides=(COUT_PER_GROUP, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_CI, BLOCK_CO),
        order=(1, 0),
    )
    tl.store(
        packed_block,
        tl.trans(weight),
        boundary_check=(0, 1),
    )


@libentry()
@triton.jit
def _conv_dgrad_pack_weight_1d_gather_kernel(
    weight_ptr,
    packed_ptr,
    GROUPS: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    KW: tl.constexpr,
    WEIGHT_STRIDE_O: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    LOAD_K: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    block_co_start = tl.program_id(0) * BLOCK_CO
    conv_group = tl.program_id(1)
    weight_base = weight_ptr + conv_group * COUT_PER_GROUP * WEIGHT_STRIDE_O
    weight_block = tl.make_block_ptr(
        base=weight_base,
        shape=(COUT_PER_GROUP, CIN_PER_GROUP * KW),
        strides=(WEIGHT_STRIDE_O, 1),
        offsets=(block_co_start, 0),
        block_shape=(BLOCK_CO, LOAD_K),
        order=(1, 0),
    )
    weight = tl.load(
        weight_block,
        boundary_check=(0, 1),
        padding_option="zero",
    )
    mask_co = block_co_start + tl.arange(0, BLOCK_CO) < COUT_PER_GROUP

    for kernel_w in tl.static_range(0, KW):
        weight_w = KW - 1 - kernel_w if FILTER_REVERSE else kernel_w
        for ci_start in tl.range(
            0,
            CIN_PER_GROUP,
            BLOCK_CI,
            disallow_acc_multi_buffer=True,
        ):
            ci = ci_start + tl.arange(0, BLOCK_CI)
            mask_ci = ci < CIN_PER_GROUP
            safe_ci = tl.where(mask_ci, ci, 0)
            index = tl.broadcast_to(
                (safe_ci * KW + weight_w)[None, :],
                (BLOCK_CO, BLOCK_CI),
            )
            value = tl.gather(weight, index, axis=1)
            value = tl.where(
                mask_co[:, None] & mask_ci[None, :],
                value,
                0.0,
            )
            packed_base = (
                packed_ptr
                + (conv_group * KW + kernel_w) * CIN_PER_GROUP * COUT_PER_GROUP
            )
            packed_block = tl.make_block_ptr(
                base=packed_base,
                shape=(CIN_PER_GROUP, COUT_PER_GROUP),
                strides=(COUT_PER_GROUP, 1),
                offsets=(ci_start, block_co_start),
                block_shape=(BLOCK_CI, BLOCK_CO),
                order=(1, 0),
            )
            tl.store(
                packed_block,
                tl.trans(value),
                boundary_check=(0, 1),
            )


@libentry()
@triton.jit
def _conv_dgrad_pack_weight_2d_gather_kernel(
    weight_ptr,
    packed_ptr,
    GROUPS: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    WEIGHT_STRIDE_O: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    LOAD_K: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    block_co_start = tl.program_id(0) * BLOCK_CO
    conv_group = tl.program_id(1)
    kernel_volume = KH * KW
    weight_base = weight_ptr + conv_group * COUT_PER_GROUP * WEIGHT_STRIDE_O
    weight_block = tl.make_block_ptr(
        base=weight_base,
        shape=(
            COUT_PER_GROUP,
            CIN_PER_GROUP * kernel_volume,
        ),
        strides=(WEIGHT_STRIDE_O, 1),
        offsets=(block_co_start, 0),
        block_shape=(BLOCK_CO, LOAD_K),
        order=(1, 0),
    )
    weight = tl.load(
        weight_block,
        boundary_check=(0, 1),
        padding_option="zero",
    )
    mask_co = block_co_start + tl.arange(0, BLOCK_CO) < COUT_PER_GROUP

    for kernel_h in tl.static_range(0, KH):
        weight_h = KH - 1 - kernel_h if FILTER_REVERSE else kernel_h
        for kernel_w in tl.static_range(0, KW):
            weight_w = KW - 1 - kernel_w if FILTER_REVERSE else kernel_w
            kernel_offset = kernel_h * KW + kernel_w
            weight_offset = weight_h * KW + weight_w
            for ci_start in tl.range(
                0,
                CIN_PER_GROUP,
                BLOCK_CI,
                disallow_acc_multi_buffer=True,
            ):
                ci = ci_start + tl.arange(0, BLOCK_CI)
                mask_ci = ci < CIN_PER_GROUP
                safe_ci = tl.where(mask_ci, ci, 0)
                index = tl.broadcast_to(
                    (safe_ci * kernel_volume + weight_offset)[None, :],
                    (BLOCK_CO, BLOCK_CI),
                )
                value = tl.gather(weight, index, axis=1)
                value = tl.where(
                    mask_co[:, None] & mask_ci[None, :],
                    value,
                    0.0,
                )
                packed_base = (
                    packed_ptr
                    + (conv_group * kernel_volume + kernel_offset)
                    * CIN_PER_GROUP
                    * COUT_PER_GROUP
                )
                packed_block = tl.make_block_ptr(
                    base=packed_base,
                    shape=(CIN_PER_GROUP, COUT_PER_GROUP),
                    strides=(COUT_PER_GROUP, 1),
                    offsets=(ci_start, block_co_start),
                    block_shape=(BLOCK_CI, BLOCK_CO),
                    order=(1, 0),
                )
                tl.store(
                    packed_block,
                    tl.trans(value),
                    boundary_check=(0, 1),
                )


@libentry()
@triton.jit
def _conv_dgrad_pack_weight_2d_tiled_gather_kernel(
    weight_ptr,
    packed_ptr,
    GROUPS: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    WEIGHT_STRIDE_O: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    LOAD_K: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    block_co_start = tl.program_id(0) * BLOCK_CO
    block_ci_start = tl.program_id(1) * BLOCK_CI
    conv_group = tl.program_id(2)
    kernel_volume = KH * KW
    weight_base = weight_ptr + conv_group * COUT_PER_GROUP * WEIGHT_STRIDE_O
    weight_block = tl.make_block_ptr(
        base=weight_base,
        shape=(COUT_PER_GROUP, CIN_PER_GROUP * kernel_volume),
        strides=(WEIGHT_STRIDE_O, 1),
        offsets=(
            block_co_start,
            block_ci_start * kernel_volume,
        ),
        block_shape=(BLOCK_CO, LOAD_K),
        order=(1, 0),
    )
    weight = tl.load(
        weight_block,
        boundary_check=(0, 1),
        padding_option="zero",
    )
    local_ci = tl.arange(0, BLOCK_CI)
    ci = block_ci_start + local_ci
    co = block_co_start + tl.arange(0, BLOCK_CO)
    valid_ci = ci < CIN_PER_GROUP
    valid_co = co < COUT_PER_GROUP

    # Read each [CI, KH, KW] slice contiguously once, then split all
    # nine kernel planes in UB.  This avoids the stride-nine GM reads
    # of the generic plane-by-plane pack on high-channel filters.
    for kernel_h in tl.static_range(0, KH):
        weight_h = KH - 1 - kernel_h if FILTER_REVERSE else kernel_h
        for kernel_w in tl.static_range(0, KW):
            weight_w = KW - 1 - kernel_w if FILTER_REVERSE else kernel_w
            kernel_offset = kernel_h * KW + kernel_w
            weight_offset = weight_h * KW + weight_w
            gather_index = tl.broadcast_to(
                (local_ci * kernel_volume + weight_offset)[None, :],
                (BLOCK_CO, BLOCK_CI),
            )
            value = tl.gather(weight, gather_index, axis=1)
            value = tl.where(
                valid_co[:, None] & valid_ci[None, :],
                value,
                0.0,
            )
            packed_base = (
                packed_ptr
                + (conv_group * kernel_volume + kernel_offset)
                * CIN_PER_GROUP
                * COUT_PER_GROUP
            )
            packed_block = tl.make_block_ptr(
                base=packed_base,
                shape=(CIN_PER_GROUP, COUT_PER_GROUP),
                strides=(COUT_PER_GROUP, 1),
                offsets=(block_ci_start, block_co_start),
                block_shape=(BLOCK_CI, BLOCK_CO),
                order=(1, 0),
            )
            tl.store(
                packed_block,
                tl.trans(value),
                boundary_check=(0, 1),
            )


@libentry()
@triton.jit
def _conv_dgrad_pack_loss_kernel(
    loss_ptr,
    packed_ptr,
    TOTAL_PACKED: tl.constexpr,
    GROUPS: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    OUTPUT_SPATIAL: tl.constexpr,
    KERNEL_VOLUME: tl.constexpr,
    LOSS_STRIDE_N: tl.constexpr,
    LOSS_STRIDE_C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL_PACKED
    safe_offsets = tl.where(mask, offsets, 0)
    spatial = safe_offsets % OUTPUT_SPATIAL
    remainder = safe_offsets // OUTPUT_SPATIAL
    co_local = remainder % COUT_PER_GROUP
    matrix_id = remainder // COUT_PER_GROUP
    batch_group = matrix_id // KERNEL_VOLUME
    conv_group = batch_group % GROUPS
    batch = batch_group // GROUPS
    co = conv_group * COUT_PER_GROUP + co_local
    value = tl.load(
        loss_ptr + batch * LOSS_STRIDE_N + co * LOSS_STRIDE_C + spatial,
        mask=mask,
        other=0.0,
    )
    tl.store(packed_ptr + safe_offsets, value, mask=mask)


@libentry()
@triton.jit
def _conv_dgrad_partial_kernel(
    loss_ptr,
    packed_weight_ptr,
    partial_ptr,
    GROUPS: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    OUTPUT_SPATIAL: tl.constexpr,
    KERNEL_VOLUME: tl.constexpr,
    LOSS_STRIDE_N: tl.constexpr,
    LOSS_STRIDE_C: tl.constexpr,
    FP32_INPUT: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_CI: tl.constexpr,
):
    tiles_m = tl.cdiv(OUTPUT_SPATIAL, BLOCK_M)
    tiles_ci = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    tiles_per_matrix = tiles_ci * tiles_m
    program_id = tl.program_id(0)
    matrix_id = program_id // tiles_per_matrix
    tile_id = program_id - matrix_id * tiles_per_matrix
    programs_per_group = GROUP_CI * tiles_m
    tile_group = tile_id // programs_per_group
    first_tile_ci = tile_group * GROUP_CI
    group_size_ci = tl.minimum(
        tiles_ci - first_tile_ci,
        GROUP_CI,
    )
    tile_in_group = tile_id - tile_group * programs_per_group
    tile_ci = first_tile_ci + tile_in_group % group_size_ci
    tile_m = tile_in_group // group_size_ci

    kernel_offset = matrix_id % KERNEL_VOLUME
    batch_group = matrix_id // KERNEL_VOLUME
    conv_group = batch_group % GROUPS
    batch = batch_group // GROUPS

    offsets_ci = tile_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    offsets_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_ci = offsets_ci < CIN_PER_GROUP
    mask_m = offsets_m < OUTPUT_SPATIAL
    safe_ci = tl.where(mask_ci, offsets_ci, 0)
    safe_m = tl.where(mask_m, offsets_m, 0)
    accumulator = tl.zeros((BLOCK_CI, BLOCK_M), dtype=tl.float32)

    for co_start in range(0, COUT_PER_GROUP, BLOCK_K):
        co_local = co_start + tl.arange(0, BLOCK_K)
        mask_k = co_local < COUT_PER_GROUP
        safe_co_local = tl.where(mask_k, co_local, 0)
        co = conv_group * COUT_PER_GROUP + safe_co_local
        packed_base = (
            (conv_group * KERNEL_VOLUME + kernel_offset)
            * CIN_PER_GROUP
            * COUT_PER_GROUP
        )
        weight = tl.load(
            packed_weight_ptr
            + packed_base
            + safe_ci[:, None] * COUT_PER_GROUP
            + safe_co_local[None, :],
            mask=mask_ci[:, None] & mask_k[None, :],
            other=0.0,
        )
        loss = tl.load(
            loss_ptr
            + batch * LOSS_STRIDE_N
            + co[:, None] * LOSS_STRIDE_C
            + safe_m[None, :],
            mask=mask_k[:, None] & mask_m[None, :],
            other=0.0,
        )
        if FP32_INPUT:
            accumulator = tl.dot(
                weight,
                loss,
                accumulator,
                input_precision="tf32",
            )
        else:
            accumulator = tl.dot(weight, loss, accumulator)

    partial_base = matrix_id * CIN_PER_GROUP * OUTPUT_SPATIAL
    tl.store(
        partial_ptr
        + partial_base
        + safe_ci[:, None] * OUTPUT_SPATIAL
        + safe_m[None, :],
        accumulator,
        mask=mask_ci[:, None] & mask_m[None, :],
    )


@libentry()
@triton.jit
def _conv_dgrad_broadcast_matmul_kernel(
    packed_weight_ptr,
    loss_ptr,
    partial_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUPS: tl.constexpr,
    KERNEL_VOLUME: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    TF32: tl.constexpr,
):
    program_id = tl.program_id(0)
    programs_m = tl.cdiv(M, BLOCK_M)
    programs_n = tl.cdiv(N, BLOCK_N)
    programs_per_matrix = programs_m * programs_n
    matrix_id = program_id // programs_per_matrix
    tile_id = program_id - matrix_id * programs_per_matrix

    programs_per_group = GROUP_M * programs_n
    group_id = tile_id // programs_per_group
    first_program_m = group_id * GROUP_M
    group_size_m = tl.minimum(programs_m - first_program_m, GROUP_M)
    tile_in_group = tile_id - group_id * programs_per_group
    program_m = first_program_m + tile_in_group % group_size_m
    program_n = tile_in_group // group_size_m

    batch_group = matrix_id // KERNEL_VOLUME
    conv_group = batch_group % GROUPS
    kernel_offset = matrix_id % KERNEL_VOLUME
    weight_batch = conv_group * KERNEL_VOLUME + kernel_offset
    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    weight_ptrs = (
        packed_weight_ptr
        + weight_batch * M * K
        + offsets_m[:, None] * K
        + offsets_k[None, :]
    )
    loss_ptrs = (
        loss_ptr
        + batch_group * K * N
        + offsets_k[:, None] * N
        + offsets_n[None, :]
    )

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in tl.range(0, K, BLOCK_K):
        if M % BLOCK_M == 0 and K % BLOCK_K == 0:
            weight = tl.load(weight_ptrs)
        else:
            weight = tl.load(
                weight_ptrs,
                mask=(offsets_m[:, None] < M)
                & (k_start + offsets_k[None, :] < K),
                other=0.0,
            )
        if K % BLOCK_K == 0 and N % BLOCK_N == 0:
            loss = tl.load(loss_ptrs)
        else:
            loss = tl.load(
                loss_ptrs,
                mask=(k_start + offsets_k[:, None] < K)
                & (offsets_n[None, :] < N),
                other=0.0,
            )
        if TF32:
            accumulator = tl.dot(
                weight,
                loss,
                accumulator,
                input_precision="tf32",
            )
        else:
            accumulator = tl.dot(weight, loss, accumulator)
        weight_ptrs += BLOCK_K
        loss_ptrs += BLOCK_K * N

    output_ptrs = (
        partial_ptr
        + matrix_id * M * N
        + offsets_m[:, None] * N
        + offsets_n[None, :]
    )
    output = accumulator.to(partial_ptr.dtype.element_ty)
    if M % BLOCK_M == 0 and N % BLOCK_N == 0:
        tl.store(output_ptrs, output)
    else:
        tl.store(
            output_ptrs,
            output,
            mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
        )


@libentry()
@triton.jit
def _conv_dgrad_direct_weight_matmul_kernel(
    weight_ptr,
    loss_ptr,
    partial_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUPS: tl.constexpr,
    KERNEL_VOLUME: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    WEIGHT_STRIDE_O: tl.constexpr,
    WEIGHT_STRIDE_I: tl.constexpr,
    WEIGHT_STRIDE_D: tl.constexpr,
    WEIGHT_STRIDE_H: tl.constexpr,
    WEIGHT_STRIDE_W: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    TF32: tl.constexpr,
):
    program_id = tl.program_id(0)
    programs_m = tl.cdiv(M, BLOCK_M)
    programs_n = tl.cdiv(N, BLOCK_N)
    programs_per_matrix = programs_m * programs_n
    matrix_id = program_id // programs_per_matrix
    tile_id = program_id - matrix_id * programs_per_matrix

    programs_per_group = GROUP_M * programs_n
    group_id = tile_id // programs_per_group
    first_program_m = group_id * GROUP_M
    group_size_m = tl.minimum(programs_m - first_program_m, GROUP_M)
    tile_in_group = tile_id - group_id * programs_per_group
    program_m = first_program_m + tile_in_group % group_size_m
    program_n = tile_in_group // group_size_m

    batch_group = matrix_id // KERNEL_VOLUME
    conv_group = batch_group % GROUPS
    kernel_offset = matrix_id % KERNEL_VOLUME
    sample_w = kernel_offset % KW
    sample_h = (kernel_offset // KW) % KH
    sample_d = kernel_offset // (KH * KW)
    weight_d = KD - 1 - sample_d if FILTER_REVERSE else sample_d
    weight_h = KH - 1 - sample_h if FILTER_REVERSE else sample_h
    weight_w = KW - 1 - sample_w if FILTER_REVERSE else sample_w
    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    loss_ptrs = (
        loss_ptr
        + batch_group * K * N
        + offsets_k[:, None] * N
        + offsets_n[None, :]
    )

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in tl.range(0, K, BLOCK_K):
        co_local = k_start + offsets_k
        mask_m = offsets_m < M
        mask_k = co_local < K
        safe_m = tl.where(mask_m, offsets_m, 0)
        safe_k = tl.where(mask_k, co_local, 0)
        co = conv_group * K + safe_k
        weight = tl.load(
            weight_ptr
            + co[None, :] * WEIGHT_STRIDE_O
            + safe_m[:, None] * WEIGHT_STRIDE_I
            + weight_d * WEIGHT_STRIDE_D
            + weight_h * WEIGHT_STRIDE_H
            + weight_w * WEIGHT_STRIDE_W,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        if K % BLOCK_K == 0 and N % BLOCK_N == 0:
            loss = tl.load(loss_ptrs)
        else:
            loss = tl.load(
                loss_ptrs,
                mask=(co_local[:, None] < K) & (offsets_n[None, :] < N),
                other=0.0,
            )
        if TF32:
            accumulator = tl.dot(
                weight,
                loss,
                accumulator,
                input_precision="tf32",
            )
        else:
            accumulator = tl.dot(weight, loss, accumulator)
        loss_ptrs += BLOCK_K * N

    output_ptrs = (
        partial_ptr
        + matrix_id * M * N
        + offsets_m[:, None] * N
        + offsets_n[None, :]
    )
    output = accumulator.to(partial_ptr.dtype.element_ty)
    tl.store(
        output_ptrs,
        output,
        mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
    )


@libentry()
@triton.jit
def _conv_dgrad_pointwise_kernel(
    weight_ptr,
    loss_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUPS: tl.constexpr,
    WEIGHT_STRIDE_O: tl.constexpr,
    WEIGHT_STRIDE_I: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    TF32: tl.constexpr,
):
    program_id = tl.program_id(0)
    programs_m = tl.cdiv(M, BLOCK_M)
    programs_n = tl.cdiv(N, BLOCK_N)
    programs_per_matrix = programs_m * programs_n
    matrix_id = program_id // programs_per_matrix
    tile_id = program_id - matrix_id * programs_per_matrix

    programs_per_group = GROUP_M * programs_n
    tile_group = tile_id // programs_per_group
    first_program_m = tile_group * GROUP_M
    group_size_m = tl.minimum(
        programs_m - first_program_m,
        GROUP_M,
    )
    tile_in_group = tile_id - tile_group * programs_per_group
    program_m = first_program_m + tile_in_group % group_size_m
    program_n = tile_in_group // group_size_m

    conv_group = matrix_id % GROUPS
    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    mask_m = offsets_m < M
    safe_m = tl.where(mask_m, offsets_m, 0)
    weight_ptrs = (
        weight_ptr
        + (conv_group * K + offsets_k[:, None]) * WEIGHT_STRIDE_O
        + safe_m[None, :] * WEIGHT_STRIDE_I
    )
    loss_ptrs = (
        loss_ptr
        + matrix_id * K * N
        + offsets_k[:, None] * N
        + offsets_n[None, :]
    )
    accumulator = tl.zeros(
        (BLOCK_M, BLOCK_N),
        dtype=tl.float32,
    )
    for k_start in tl.range(0, K, BLOCK_K):
        mask_k = k_start + offsets_k < K
        weight = tl.load(
            weight_ptrs,
            mask=mask_k[:, None] & mask_m[None, :],
            other=0.0,
        )
        loss = tl.load(
            loss_ptrs,
            mask=(mask_k[:, None] & (offsets_n[None, :] < N)),
            other=0.0,
        )
        if TF32:
            accumulator = tl.dot(
                tl.trans(weight),
                loss,
                accumulator,
                input_precision="tf32",
            )
        else:
            accumulator = tl.dot(
                tl.trans(weight),
                loss,
                accumulator,
            )
        weight_ptrs += BLOCK_K * WEIGHT_STRIDE_O
        loss_ptrs += BLOCK_K * N

    output_ptrs = (
        output_ptr
        + matrix_id * M * N
        + offsets_m[:, None] * N
        + offsets_n[None, :]
    )
    tl.store(
        output_ptrs,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=((offsets_m[:, None] < M) & (offsets_n[None, :] < N)),
    )


@libentry()
@triton.jit
def _conv_wgrad_pointwise_batch_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    N_BATCH: tl.constexpr,
    GROUPS: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    SPATIAL: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    IMAGE_STRIDE_N: tl.constexpr,
    IMAGE_STRIDE_C: tl.constexpr,
    LOSS_STRIDE_N: tl.constexpr,
    LOSS_STRIDE_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    TF32: tl.constexpr,
):
    programs_m = tl.cdiv(COUT_PER_GROUP, BLOCK_M)
    programs_n = tl.cdiv(CIN_PER_GROUP, BLOCK_N)
    programs_per_matrix = programs_m * programs_n
    program_id = tl.program_id(0)
    matrix_id = program_id // programs_per_matrix
    tile_id = program_id - matrix_id * programs_per_matrix

    programs_per_group = GROUP_M * programs_n
    tile_group = tile_id // programs_per_group
    first_program_m = tile_group * GROUP_M
    group_size_m = tl.minimum(
        programs_m - first_program_m,
        GROUP_M,
    )
    tile_in_group = tile_id - tile_group * programs_per_group
    program_m = first_program_m + tile_in_group % group_size_m
    program_n = tile_in_group // group_size_m

    batch = matrix_id // GROUPS
    conv_group = matrix_id - batch * GROUPS
    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    mask_m = offsets_m < COUT_PER_GROUP
    mask_n = offsets_n < CIN_PER_GROUP
    safe_m = tl.where(mask_m, offsets_m, 0)
    safe_n = tl.where(mask_n, offsets_n, 0)
    co = conv_group * COUT_PER_GROUP + safe_m
    ci = conv_group * CIN_PER_GROUP + safe_n
    loss_ptrs = (
        loss_ptr
        + batch * LOSS_STRIDE_N
        + co[:, None] * LOSS_STRIDE_C
        + offsets_k[None, :]
    )
    image_ptrs = (
        image_ptr
        + batch * IMAGE_STRIDE_N
        + ci[:, None] * IMAGE_STRIDE_C
        + offsets_k[None, :]
    )
    accumulator = tl.zeros(
        (BLOCK_M, BLOCK_N),
        dtype=tl.float32,
    )
    for k_start in tl.range(0, SPATIAL, BLOCK_K):
        mask_k = k_start + offsets_k < SPATIAL
        loss = tl.load(
            loss_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        image = tl.load(
            image_ptrs,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        )
        if TF32:
            accumulator = tl.dot(
                loss,
                tl.trans(image),
                accumulator,
                input_precision="tf32",
            )
        else:
            accumulator = tl.dot(
                loss,
                tl.trans(image),
                accumulator,
            )
        loss_ptrs += BLOCK_K
        image_ptrs += BLOCK_K

    output_ptrs = (
        partial_ptr
        + matrix_id * COUT_PER_GROUP * CIN_PER_GROUP
        + offsets_m[:, None] * CIN_PER_GROUP
        + offsets_n[None, :]
    )
    tl.store(
        output_ptrs,
        accumulator.to(partial_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@libentry()
@triton.jit
def _conv_dgrad_gather_kernel(
    partial_ptr,
    output_ptr,
    TOTAL_OUTPUT: tl.constexpr,
    GROUPS: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    OUTPUT_STRIDE_N: tl.constexpr,
    OUTPUT_STRIDE_C: tl.constexpr,
    OUTPUT_STRIDE_D: tl.constexpr,
    OUTPUT_STRIDE_H: tl.constexpr,
    OUTPUT_STRIDE_W: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    input_spatial = XD * XH * XW
    output_spatial = OD * OH * OW
    kernel_volume = KD * KH * KW
    chunk_start = tl.program_id(0) * CHUNK_SIZE
    for block_start in tl.range(0, CHUNK_SIZE, BLOCK):
        offsets = chunk_start + block_start + tl.arange(0, BLOCK)
        mask = offsets < TOTAL_OUTPUT
        safe_offsets = tl.where(mask, offsets, 0)
        spatial = safe_offsets % input_spatial
        channel_batch = safe_offsets // input_spatial
        ci = channel_batch % (GROUPS * CIN_PER_GROUP)
        batch = channel_batch // (GROUPS * CIN_PER_GROUP)
        conv_group = ci // CIN_PER_GROUP
        ci_local = ci - conv_group * CIN_PER_GROUP
        xw = spatial % XW
        xh = (spatial // XW) % XH
        xd = spatial // (XH * XW)
        accumulator = tl.zeros((BLOCK,), dtype=tl.float32)

        for kd in tl.static_range(0, KD):
            numerator_d = xd + PAD_D - kd * DIL_D
            od = numerator_d // STRIDE_D
            valid_d = (
                (numerator_d >= 0) & (numerator_d % STRIDE_D == 0) & (od < OD)
            )
            for kh in tl.static_range(0, KH):
                numerator_h = xh + PAD_H - kh * DIL_H
                oh = numerator_h // STRIDE_H
                valid_h = (
                    (numerator_h >= 0)
                    & (numerator_h % STRIDE_H == 0)
                    & (oh < OH)
                )
                for kw in tl.static_range(0, KW):
                    numerator_w = xw + PAD_W - kw * DIL_W
                    ow = numerator_w // STRIDE_W
                    valid_w = (
                        (numerator_w >= 0)
                        & (numerator_w % STRIDE_W == 0)
                        & (ow < OW)
                    )
                    valid = mask & valid_d & valid_h & valid_w
                    safe_od = tl.where(valid, od, 0)
                    safe_oh = tl.where(valid, oh, 0)
                    safe_ow = tl.where(valid, ow, 0)
                    kernel_offset = (kd * KH + kh) * KW + kw
                    matrix_id = (
                        batch * GROUPS + conv_group
                    ) * kernel_volume + kernel_offset
                    partial_offset = (
                        (matrix_id * CIN_PER_GROUP + ci_local) * output_spatial
                        + (safe_od * OH + safe_oh) * OW
                        + safe_ow
                    )
                    accumulator += tl.load(
                        partial_ptr + partial_offset,
                        mask=valid,
                        other=0.0,
                    )

        tl.store(
            output_ptr
            + batch * OUTPUT_STRIDE_N
            + ci * OUTPUT_STRIDE_C
            + xd * OUTPUT_STRIDE_D
            + xh * OUTPUT_STRIDE_H
            + xw * OUTPUT_STRIDE_W,
            accumulator.to(output_ptr.dtype.element_ty),
            mask=mask,
        )


@libentry()
@triton.jit
def _conv_dgrad_gather_tiled_kernel(
    partial_ptr,
    output_ptr,
    GROUPS: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    OUTPUT_STRIDE_N: tl.constexpr,
    OUTPUT_STRIDE_C: tl.constexpr,
    OUTPUT_STRIDE_D: tl.constexpr,
    OUTPUT_STRIDE_H: tl.constexpr,
    OUTPUT_STRIDE_W: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    input_spatial = XD * XH * XW
    output_spatial = OD * OH * OW
    kernel_volume = KD * KH * KW
    tiles_ci = tl.cdiv(CIN_PER_GROUP, BLOCK_CI)
    tiles_m = tl.cdiv(input_spatial, BLOCK_M)
    tiles_per_matrix = tiles_ci * tiles_m
    program_id = tl.program_id(0)
    batch_group = program_id // tiles_per_matrix
    tile_id = program_id - batch_group * tiles_per_matrix
    tile_ci = tile_id // tiles_m
    tile_m = tile_id - tile_ci * tiles_m
    conv_group = batch_group % GROUPS
    batch = batch_group // GROUPS

    offsets_ci = tile_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    spatial = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_ci = offsets_ci < CIN_PER_GROUP
    mask_m = spatial < input_spatial
    safe_ci = tl.where(mask_ci, offsets_ci, 0)
    safe_spatial = tl.where(mask_m, spatial, 0)
    xw = safe_spatial % XW
    xh = (safe_spatial // XW) % XH
    xd = safe_spatial // (XH * XW)
    accumulator = tl.zeros((BLOCK_CI, BLOCK_M), dtype=tl.float32)

    for kd in tl.static_range(0, KD):
        numerator_d = xd + PAD_D - kd * DIL_D
        od = numerator_d // STRIDE_D
        valid_d = (
            (numerator_d >= 0) & (numerator_d % STRIDE_D == 0) & (od < OD)
        )
        for kh in tl.static_range(0, KH):
            numerator_h = xh + PAD_H - kh * DIL_H
            oh = numerator_h // STRIDE_H
            valid_h = (
                (numerator_h >= 0) & (numerator_h % STRIDE_H == 0) & (oh < OH)
            )
            for kw in tl.static_range(0, KW):
                numerator_w = xw + PAD_W - kw * DIL_W
                ow = numerator_w // STRIDE_W
                valid_w = (
                    (numerator_w >= 0)
                    & (numerator_w % STRIDE_W == 0)
                    & (ow < OW)
                )
                valid_m = mask_m & valid_d & valid_h & valid_w
                safe_od = tl.where(valid_m, od, 0)
                safe_oh = tl.where(valid_m, oh, 0)
                safe_ow = tl.where(valid_m, ow, 0)
                kernel_offset = (kd * KH + kh) * KW + kw
                matrix_id = batch_group * kernel_volume + kernel_offset
                partial_offset = (
                    (matrix_id * CIN_PER_GROUP + safe_ci[:, None])
                    * output_spatial
                    + (safe_od[None, :] * OH + safe_oh[None, :]) * OW
                    + safe_ow[None, :]
                )
                accumulator += tl.load(
                    partial_ptr + partial_offset,
                    mask=mask_ci[:, None] & valid_m[None, :],
                    other=0.0,
                )

    ci = conv_group * CIN_PER_GROUP + safe_ci
    tl.store(
        output_ptr
        + batch * OUTPUT_STRIDE_N
        + ci[:, None] * OUTPUT_STRIDE_C
        + xd[None, :] * OUTPUT_STRIDE_D
        + xh[None, :] * OUTPUT_STRIDE_H
        + xw[None, :] * OUTPUT_STRIDE_W,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=mask_ci[:, None] & mask_m[None, :],
    )


@libentry()
@triton.jit
def _conv_dgrad_gather_2d_full_plane_kernel(
    partial_ptr,
    output_ptr,
    GROUPS: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    OUTPUT_STRIDE_N: tl.constexpr,
    OUTPUT_STRIDE_C: tl.constexpr,
    LOAD_O: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
):
    block_ci_start = tl.program_id(0) * BLOCK_CI
    batch_group = tl.program_id(1)
    conv_group = batch_group % GROUPS
    batch = batch_group // GROUPS
    input_spatial = XH * XW
    output_spatial = OH * OW
    kernel_volume = KH * KW

    for spatial_start in tl.range(
        0,
        input_spatial,
        BLOCK_SPATIAL,
        disallow_acc_multi_buffer=True,
    ):
        spatial = spatial_start + tl.arange(0, BLOCK_SPATIAL)
        xh = spatial // XW
        xw = spatial - xh * XW
        mask_spatial = spatial < input_spatial
        accumulator = tl.zeros(
            (BLOCK_CI, BLOCK_SPATIAL),
            dtype=tl.float32,
        )
        for kernel_h in tl.static_range(0, KH):
            numerator_h = xh + PAD_H - kernel_h * DIL_H
            loss_h = numerator_h // STRIDE_H
            valid_h = (
                (numerator_h >= 0)
                & (numerator_h % STRIDE_H == 0)
                & (loss_h < OH)
            )
            for kernel_w in tl.static_range(0, KW):
                numerator_w = xw + PAD_W - kernel_w * DIL_W
                loss_w = numerator_w // STRIDE_W
                valid = (
                    mask_spatial
                    & valid_h
                    & (numerator_w >= 0)
                    & (numerator_w % STRIDE_W == 0)
                    & (loss_w < OW)
                )
                safe_loss = tl.where(
                    valid,
                    loss_h * OW + loss_w,
                    0,
                )
                kernel_offset = kernel_h * KW + kernel_w
                partial_base = (
                    partial_ptr
                    + (batch_group * kernel_volume + kernel_offset)
                    * CIN_PER_GROUP
                    * output_spatial
                )
                partial_block = tl.make_block_ptr(
                    base=partial_base,
                    shape=(CIN_PER_GROUP, output_spatial),
                    strides=(output_spatial, 1),
                    offsets=(block_ci_start, 0),
                    block_shape=(BLOCK_CI, LOAD_O),
                    order=(1, 0),
                )
                partial = tl.load(
                    partial_block,
                    boundary_check=(0, 1),
                    padding_option="zero",
                )
                index = tl.broadcast_to(
                    safe_loss[None, :],
                    (BLOCK_CI, BLOCK_SPATIAL),
                )
                value = tl.gather(partial, index, axis=1)
                accumulator += tl.where(
                    valid[None, :],
                    value,
                    0.0,
                )

        output_base = (
            output_ptr
            + batch * OUTPUT_STRIDE_N
            + conv_group * CIN_PER_GROUP * OUTPUT_STRIDE_C
        )
        output_block = tl.make_block_ptr(
            base=output_base,
            shape=(CIN_PER_GROUP, input_spatial),
            strides=(OUTPUT_STRIDE_C, 1),
            offsets=(block_ci_start, spatial_start),
            block_shape=(BLOCK_CI, BLOCK_SPATIAL),
            order=(1, 0),
        )
        tl.store(
            output_block,
            accumulator.to(output_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )


@libentry()
@triton.jit
def _conv_dgrad_stride2_partial_row_interleave_kernel(
    partial_ptr,
    output_ptr,
    GROUPS: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    OUTPUT_STRIDE_N: tl.constexpr,
    OUTPUT_STRIDE_C: tl.constexpr,
    PARITY_H: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    LOAD_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    block_ci_start = tl.program_id(0) * BLOCK_CI
    row_start = tl.program_id(1) * BLOCK_ROWS
    batch_group = tl.program_id(2)
    conv_group = batch_group % GROUPS
    batch = batch_group // GROUPS
    output_spatial = OH * OW
    partial_base = (
        partial_ptr + batch_group * 9 * CIN_PER_GROUP * output_spatial
    )
    output_w = tl.arange(0, BLOCK_W)
    compact_w = output_w // 2
    gather_index = tl.broadcast_to(
        compact_w[None, None, :],
        (BLOCK_CI, BLOCK_ROWS, BLOCK_W),
    )

    # A 3x3, padding-one, stride-two dgrad only uses kernel row 1
    # for even input rows and kernel rows 0/2 for odd input rows.
    # Splitting that parity at compile time keeps all GM transfers as
    # block-pointer row copies; the width interleave happens inside UB.
    if PARITY_H == 0:
        left_block = tl.make_block_ptr(
            base=partial_base + 3 * CIN_PER_GROUP * output_spatial,
            shape=(CIN_PER_GROUP, OH, OW),
            strides=(output_spatial, OW, 1),
            offsets=(block_ci_start, row_start, 0),
            block_shape=(BLOCK_CI, BLOCK_ROWS, LOAD_W),
            order=(2, 1, 0),
        )
        center_block = tl.make_block_ptr(
            base=partial_base + 4 * CIN_PER_GROUP * output_spatial,
            shape=(CIN_PER_GROUP, OH, OW),
            strides=(output_spatial, OW, 1),
            offsets=(block_ci_start, row_start, 0),
            block_shape=(BLOCK_CI, BLOCK_ROWS, LOAD_W),
            order=(2, 1, 0),
        )
        right_block = tl.make_block_ptr(
            base=partial_base + 5 * CIN_PER_GROUP * output_spatial,
            shape=(CIN_PER_GROUP, OH, OW),
            strides=(output_spatial, OW, 1),
            offsets=(block_ci_start, row_start, 0),
            block_shape=(BLOCK_CI, BLOCK_ROWS, LOAD_W),
            order=(2, 1, 0),
        )
        left = tl.gather(
            tl.load(
                left_block,
                boundary_check=(0, 1, 2),
                padding_option="zero",
            ),
            gather_index + 1,
            axis=2,
        ).to(tl.float32)
        center = tl.gather(
            tl.load(
                center_block,
                boundary_check=(0, 1, 2),
                padding_option="zero",
            ),
            gather_index,
            axis=2,
        ).to(tl.float32)
        right = tl.gather(
            tl.load(
                right_block,
                boundary_check=(0, 1, 2),
                padding_option="zero",
            ),
            gather_index,
            axis=2,
        ).to(tl.float32)
        value = tl.where(
            (output_w % 2)[None, None, :] == 0,
            center,
            left + right,
        )
    else:
        top_left_block = tl.make_block_ptr(
            base=partial_base,
            shape=(CIN_PER_GROUP, OH, OW),
            strides=(output_spatial, OW, 1),
            offsets=(block_ci_start, row_start + 1, 0),
            block_shape=(BLOCK_CI, BLOCK_ROWS, LOAD_W),
            order=(2, 1, 0),
        )
        top_center_block = tl.make_block_ptr(
            base=partial_base + CIN_PER_GROUP * output_spatial,
            shape=(CIN_PER_GROUP, OH, OW),
            strides=(output_spatial, OW, 1),
            offsets=(block_ci_start, row_start + 1, 0),
            block_shape=(BLOCK_CI, BLOCK_ROWS, LOAD_W),
            order=(2, 1, 0),
        )
        top_right_block = tl.make_block_ptr(
            base=partial_base + 2 * CIN_PER_GROUP * output_spatial,
            shape=(CIN_PER_GROUP, OH, OW),
            strides=(output_spatial, OW, 1),
            offsets=(block_ci_start, row_start + 1, 0),
            block_shape=(BLOCK_CI, BLOCK_ROWS, LOAD_W),
            order=(2, 1, 0),
        )
        bottom_left_block = tl.make_block_ptr(
            base=partial_base + 6 * CIN_PER_GROUP * output_spatial,
            shape=(CIN_PER_GROUP, OH, OW),
            strides=(output_spatial, OW, 1),
            offsets=(block_ci_start, row_start, 0),
            block_shape=(BLOCK_CI, BLOCK_ROWS, LOAD_W),
            order=(2, 1, 0),
        )
        bottom_center_block = tl.make_block_ptr(
            base=partial_base + 7 * CIN_PER_GROUP * output_spatial,
            shape=(CIN_PER_GROUP, OH, OW),
            strides=(output_spatial, OW, 1),
            offsets=(block_ci_start, row_start, 0),
            block_shape=(BLOCK_CI, BLOCK_ROWS, LOAD_W),
            order=(2, 1, 0),
        )
        bottom_right_block = tl.make_block_ptr(
            base=partial_base + 8 * CIN_PER_GROUP * output_spatial,
            shape=(CIN_PER_GROUP, OH, OW),
            strides=(output_spatial, OW, 1),
            offsets=(block_ci_start, row_start, 0),
            block_shape=(BLOCK_CI, BLOCK_ROWS, LOAD_W),
            order=(2, 1, 0),
        )
        top_left = tl.gather(
            tl.load(
                top_left_block,
                boundary_check=(0, 1, 2),
                padding_option="zero",
            ),
            gather_index + 1,
            axis=2,
        ).to(tl.float32)
        top_center = tl.gather(
            tl.load(
                top_center_block,
                boundary_check=(0, 1, 2),
                padding_option="zero",
            ),
            gather_index,
            axis=2,
        ).to(tl.float32)
        top_right = tl.gather(
            tl.load(
                top_right_block,
                boundary_check=(0, 1, 2),
                padding_option="zero",
            ),
            gather_index,
            axis=2,
        ).to(tl.float32)
        bottom_left = tl.gather(
            tl.load(
                bottom_left_block,
                boundary_check=(0, 1, 2),
                padding_option="zero",
            ),
            gather_index + 1,
            axis=2,
        ).to(tl.float32)
        bottom_center = tl.gather(
            tl.load(
                bottom_center_block,
                boundary_check=(0, 1, 2),
                padding_option="zero",
            ),
            gather_index,
            axis=2,
        ).to(tl.float32)
        bottom_right = tl.gather(
            tl.load(
                bottom_right_block,
                boundary_check=(0, 1, 2),
                padding_option="zero",
            ),
            gather_index,
            axis=2,
        ).to(tl.float32)
        center = top_center + bottom_center
        sides = top_left + top_right + bottom_left + bottom_right
        value = tl.where(
            (output_w % 2)[None, None, :] == 0,
            center,
            sides,
        )

    output_base = (
        output_ptr
        + batch * OUTPUT_STRIDE_N
        + conv_group * CIN_PER_GROUP * OUTPUT_STRIDE_C
        + PARITY_H * XW
    )
    output_block = tl.make_block_ptr(
        base=output_base,
        shape=(CIN_PER_GROUP, OH, XW),
        strides=(OUTPUT_STRIDE_C, 2 * XW, 1),
        offsets=(block_ci_start, row_start, 0),
        block_shape=(BLOCK_CI, BLOCK_ROWS, BLOCK_W),
        order=(2, 1, 0),
    )
    tl.store(
        output_block,
        value.to(output_ptr.dtype.element_ty),
        boundary_check=(0, 1, 2),
    )


@libentry()
@triton.jit
def _conv_dgrad_stride2_partial_tiled_interleave_kernel(
    partial_ptr,
    output_ptr,
    GROUPS: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    OUTPUT_STRIDE_N: tl.constexpr,
    OUTPUT_STRIDE_C: tl.constexpr,
    PARITY_H: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    LOAD_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    block_ci_start = tl.program_id(0) * BLOCK_CI
    row_width_tile = tl.program_id(1)
    width_tiles = tl.cdiv(XW, BLOCK_W)
    row_start = (row_width_tile // width_tiles) * BLOCK_ROWS
    output_w_start = (row_width_tile % width_tiles) * BLOCK_W
    compact_w_start = output_w_start // 2
    batch_group = tl.program_id(2)
    conv_group = batch_group % GROUPS
    batch = batch_group // GROUPS
    output_spatial = OH * OW
    partial_base = (
        partial_ptr + batch_group * 9 * CIN_PER_GROUP * output_spatial
    )
    local_output_w = tl.arange(0, BLOCK_W)
    output_w = output_w_start + local_output_w
    local_compact_w = local_output_w // 2
    compact_w = compact_w_start + local_compact_w
    compact_row = row_start + tl.arange(0, BLOCK_ROWS)
    output_valid = output_w < XW
    left_valid = output_valid & (compact_w + 1 < OW)
    top_valid = compact_row + 1 < OH
    gather_index = tl.broadcast_to(
        local_compact_w[None, None, :],
        (BLOCK_CI, BLOCK_ROWS, BLOCK_W),
    )

    if PARITY_H == 0:
        left_block = tl.make_block_ptr(
            base=partial_base + 3 * CIN_PER_GROUP * output_spatial,
            shape=(CIN_PER_GROUP, OH, OW),
            strides=(output_spatial, OW, 1),
            offsets=(
                block_ci_start,
                row_start,
                compact_w_start + 1,
            ),
            block_shape=(BLOCK_CI, BLOCK_ROWS, LOAD_W),
            order=(2, 1, 0),
        )
        center_block = tl.make_block_ptr(
            base=partial_base + 4 * CIN_PER_GROUP * output_spatial,
            shape=(CIN_PER_GROUP, OH, OW),
            strides=(output_spatial, OW, 1),
            offsets=(block_ci_start, row_start, compact_w_start),
            block_shape=(BLOCK_CI, BLOCK_ROWS, LOAD_W),
            order=(2, 1, 0),
        )
        right_block = tl.make_block_ptr(
            base=partial_base + 5 * CIN_PER_GROUP * output_spatial,
            shape=(CIN_PER_GROUP, OH, OW),
            strides=(output_spatial, OW, 1),
            offsets=(block_ci_start, row_start, compact_w_start),
            block_shape=(BLOCK_CI, BLOCK_ROWS, LOAD_W),
            order=(2, 1, 0),
        )
        left = tl.gather(
            tl.load(
                left_block,
                boundary_check=(0, 1, 2),
                padding_option="zero",
            ),
            gather_index,
            axis=2,
        ).to(tl.float32)
        center = tl.gather(
            tl.load(
                center_block,
                boundary_check=(0, 1, 2),
                padding_option="zero",
            ),
            gather_index,
            axis=2,
        ).to(tl.float32)
        right = tl.gather(
            tl.load(
                right_block,
                boundary_check=(0, 1, 2),
                padding_option="zero",
            ),
            gather_index,
            axis=2,
        ).to(tl.float32)
        left = tl.where(
            left_valid[None, None, :],
            left,
            0.0,
        )
        value = tl.where(
            (output_w % 2)[None, None, :] == 0,
            center,
            left + right,
        )
    else:
        top_left_block = tl.make_block_ptr(
            base=partial_base,
            shape=(CIN_PER_GROUP, OH, OW),
            strides=(output_spatial, OW, 1),
            offsets=(
                block_ci_start,
                row_start + 1,
                compact_w_start + 1,
            ),
            block_shape=(BLOCK_CI, BLOCK_ROWS, LOAD_W),
            order=(2, 1, 0),
        )
        top_center_block = tl.make_block_ptr(
            base=partial_base + CIN_PER_GROUP * output_spatial,
            shape=(CIN_PER_GROUP, OH, OW),
            strides=(output_spatial, OW, 1),
            offsets=(
                block_ci_start,
                row_start + 1,
                compact_w_start,
            ),
            block_shape=(BLOCK_CI, BLOCK_ROWS, LOAD_W),
            order=(2, 1, 0),
        )
        top_right_block = tl.make_block_ptr(
            base=partial_base + 2 * CIN_PER_GROUP * output_spatial,
            shape=(CIN_PER_GROUP, OH, OW),
            strides=(output_spatial, OW, 1),
            offsets=(
                block_ci_start,
                row_start + 1,
                compact_w_start,
            ),
            block_shape=(BLOCK_CI, BLOCK_ROWS, LOAD_W),
            order=(2, 1, 0),
        )
        bottom_left_block = tl.make_block_ptr(
            base=partial_base + 6 * CIN_PER_GROUP * output_spatial,
            shape=(CIN_PER_GROUP, OH, OW),
            strides=(output_spatial, OW, 1),
            offsets=(
                block_ci_start,
                row_start,
                compact_w_start + 1,
            ),
            block_shape=(BLOCK_CI, BLOCK_ROWS, LOAD_W),
            order=(2, 1, 0),
        )
        bottom_center_block = tl.make_block_ptr(
            base=partial_base + 7 * CIN_PER_GROUP * output_spatial,
            shape=(CIN_PER_GROUP, OH, OW),
            strides=(output_spatial, OW, 1),
            offsets=(block_ci_start, row_start, compact_w_start),
            block_shape=(BLOCK_CI, BLOCK_ROWS, LOAD_W),
            order=(2, 1, 0),
        )
        bottom_right_block = tl.make_block_ptr(
            base=partial_base + 8 * CIN_PER_GROUP * output_spatial,
            shape=(CIN_PER_GROUP, OH, OW),
            strides=(output_spatial, OW, 1),
            offsets=(block_ci_start, row_start, compact_w_start),
            block_shape=(BLOCK_CI, BLOCK_ROWS, LOAD_W),
            order=(2, 1, 0),
        )
        top_left = tl.gather(
            tl.load(
                top_left_block,
                boundary_check=(0, 1, 2),
                padding_option="zero",
            ),
            gather_index,
            axis=2,
        ).to(tl.float32)
        top_center = tl.gather(
            tl.load(
                top_center_block,
                boundary_check=(0, 1, 2),
                padding_option="zero",
            ),
            gather_index,
            axis=2,
        ).to(tl.float32)
        top_right = tl.gather(
            tl.load(
                top_right_block,
                boundary_check=(0, 1, 2),
                padding_option="zero",
            ),
            gather_index,
            axis=2,
        ).to(tl.float32)
        bottom_left = tl.gather(
            tl.load(
                bottom_left_block,
                boundary_check=(0, 1, 2),
                padding_option="zero",
            ),
            gather_index,
            axis=2,
        ).to(tl.float32)
        bottom_center = tl.gather(
            tl.load(
                bottom_center_block,
                boundary_check=(0, 1, 2),
                padding_option="zero",
            ),
            gather_index,
            axis=2,
        ).to(tl.float32)
        bottom_right = tl.gather(
            tl.load(
                bottom_right_block,
                boundary_check=(0, 1, 2),
                padding_option="zero",
            ),
            gather_index,
            axis=2,
        ).to(tl.float32)
        top_left = tl.where(
            (top_valid[None, :, None] & left_valid[None, None, :]),
            top_left,
            0.0,
        )
        top_center = tl.where(
            top_valid[None, :, None],
            top_center,
            0.0,
        )
        top_right = tl.where(
            top_valid[None, :, None],
            top_right,
            0.0,
        )
        bottom_left = tl.where(
            left_valid[None, None, :],
            bottom_left,
            0.0,
        )
        center = top_center + bottom_center
        sides = top_left + top_right + bottom_left + bottom_right
        value = tl.where(
            (output_w % 2)[None, None, :] == 0,
            center,
            sides,
        )

    output_base = (
        output_ptr
        + batch * OUTPUT_STRIDE_N
        + conv_group * CIN_PER_GROUP * OUTPUT_STRIDE_C
        + PARITY_H * XW
    )
    output_block = tl.make_block_ptr(
        base=output_base,
        shape=(CIN_PER_GROUP, OH, XW),
        strides=(OUTPUT_STRIDE_C, 2 * XW, 1),
        offsets=(block_ci_start, row_start, output_w_start),
        block_shape=(BLOCK_CI, BLOCK_ROWS, BLOCK_W),
        order=(2, 1, 0),
    )
    tl.store(
        output_block,
        value.to(output_ptr.dtype.element_ty),
        boundary_check=(0, 1, 2),
    )


@libentry()
@triton.jit
def _conv_dgrad_gather_1d_block_kernel(
    partial_ptr,
    output_ptr,
    GROUPS: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    XW: tl.constexpr,
    OW: tl.constexpr,
    KW: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_W: tl.constexpr,
    OUTPUT_STRIDE_N: tl.constexpr,
    OUTPUT_STRIDE_C: tl.constexpr,
    OUTPUT_STRIDE_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    block_w_start = tl.program_id(0) * BLOCK_W
    block_ci_start = tl.program_id(1) * BLOCK_CI
    batch_group = tl.program_id(2)
    conv_group = batch_group % GROUPS
    batch = batch_group // GROUPS
    accumulator = tl.zeros(
        (BLOCK_CI, BLOCK_W),
        dtype=tl.float32,
    )

    for kernel_w in tl.static_range(0, KW):
        partial_base = (
            partial_ptr + (batch_group * KW + kernel_w) * CIN_PER_GROUP * OW
        )
        partial_block = tl.make_block_ptr(
            base=partial_base,
            shape=(CIN_PER_GROUP, OW),
            strides=(OW, 1),
            offsets=(
                block_ci_start,
                block_w_start + PAD_W - kernel_w * DIL_W,
            ),
            block_shape=(BLOCK_CI, BLOCK_W),
            order=(1, 0),
        )
        accumulator += tl.load(
            partial_block,
            boundary_check=(0, 1),
            padding_option="zero",
        )

    output_base = (
        output_ptr
        + batch * OUTPUT_STRIDE_N
        + conv_group * CIN_PER_GROUP * OUTPUT_STRIDE_C
    )
    output_block = tl.make_block_ptr(
        base=output_base,
        shape=(CIN_PER_GROUP, XW),
        strides=(OUTPUT_STRIDE_C, OUTPUT_STRIDE_W),
        offsets=(block_ci_start, block_w_start),
        block_shape=(BLOCK_CI, BLOCK_W),
        order=(1, 0),
    )
    tl.store(
        output_block,
        accumulator.to(output_ptr.dtype.element_ty),
        boundary_check=(0, 1),
    )


@libentry()
@triton.jit
def _conv_dgrad_gather_1d_strided_kernel(
    partial_ptr,
    output_ptr,
    GROUPS: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    XW: tl.constexpr,
    OW: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_W: tl.constexpr,
    OUTPUT_STRIDE_N: tl.constexpr,
    OUTPUT_STRIDE_C: tl.constexpr,
    OUTPUT_STRIDE_W: tl.constexpr,
    LOAD_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    block_w_start = tl.program_id(0) * BLOCK_W
    block_ci_start = tl.program_id(1) * BLOCK_CI
    batch_group = tl.program_id(2)
    conv_group = batch_group % GROUPS
    batch = batch_group // GROUPS
    xw = block_w_start + tl.arange(0, BLOCK_W)
    accumulator = tl.zeros(
        (BLOCK_CI, BLOCK_W),
        dtype=tl.float32,
    )

    for kernel_w in tl.static_range(0, KW):
        partial_base = (
            partial_ptr + (batch_group * KW + kernel_w) * CIN_PER_GROUP * OW
        )
        partial_block = tl.make_block_ptr(
            base=partial_base,
            shape=(CIN_PER_GROUP, OW),
            strides=(OW, 1),
            offsets=(block_ci_start, 0),
            block_shape=(BLOCK_CI, LOAD_W),
            order=(1, 0),
        )
        partial = tl.load(
            partial_block,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        numerator = xw + PAD_W - kernel_w * DIL_W
        loss_w = numerator // STRIDE_W
        valid = (
            (xw < XW)
            & (numerator >= 0)
            & (numerator % STRIDE_W == 0)
            & (loss_w >= 0)
            & (loss_w < OW)
        )
        safe_loss_w = tl.where(valid, loss_w, 0)
        index = tl.broadcast_to(
            safe_loss_w[None, :],
            (BLOCK_CI, BLOCK_W),
        )
        value = tl.gather(partial, index, axis=1)
        accumulator += tl.where(valid[None, :], value, 0.0)

    output_base = (
        output_ptr
        + batch * OUTPUT_STRIDE_N
        + conv_group * CIN_PER_GROUP * OUTPUT_STRIDE_C
    )
    output_block = tl.make_block_ptr(
        base=output_base,
        shape=(CIN_PER_GROUP, XW),
        strides=(OUTPUT_STRIDE_C, OUTPUT_STRIDE_W),
        offsets=(block_ci_start, block_w_start),
        block_shape=(BLOCK_CI, BLOCK_W),
        order=(1, 0),
    )
    tl.store(
        output_block,
        accumulator.to(output_ptr.dtype.element_ty),
        boundary_check=(0, 1),
    )


@libentry()
@triton.jit
def _conv_wgrad_pack_loss_kernel(
    loss_ptr,
    packed_ptr,
    REDUCTION: tl.constexpr,
    GROUPS: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    OUTPUT_SPATIAL: tl.constexpr,
    LOSS_STRIDE_N: tl.constexpr,
    LOSS_STRIDE_C: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    offsets_co = tl.program_id(0) * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offsets_r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    conv_group = tl.program_id(2)
    mask_co = offsets_co < COUT_PER_GROUP
    mask_r = offsets_r < REDUCTION
    safe_co = tl.where(mask_co, offsets_co, 0)
    safe_r = tl.where(mask_r, offsets_r, 0)
    batch = safe_r // OUTPUT_SPATIAL
    spatial = safe_r - batch * OUTPUT_SPATIAL
    co = conv_group * COUT_PER_GROUP + safe_co
    value = tl.load(
        loss_ptr
        + batch[None, :] * LOSS_STRIDE_N
        + co[:, None] * LOSS_STRIDE_C
        + spatial[None, :],
        mask=mask_co[:, None] & mask_r[None, :],
        other=0.0,
    )
    tl.store(
        packed_ptr
        + (conv_group * COUT_PER_GROUP + safe_co[:, None]) * REDUCTION
        + safe_r[None, :],
        value,
        mask=mask_co[:, None] & mask_r[None, :],
    )


@libentry()
@triton.jit
def _conv_wgrad_pack_loss_1d_kernel(
    loss_ptr,
    packed_ptr,
    GROUPS: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    OW: tl.constexpr,
    REDUCTION: tl.constexpr,
    LOSS_STRIDE_N: tl.constexpr,
    LOSS_STRIDE_C: tl.constexpr,
    LOSS_STRIDE_W: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    co_local = tl.program_id(0) * BLOCK_CO + tl.arange(0, BLOCK_CO)
    ow = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    batch_group = tl.program_id(2)
    batch = batch_group // GROUPS
    conv_group = batch_group - batch * GROUPS
    co = conv_group * COUT_PER_GROUP + co_local
    mask_co = co_local < COUT_PER_GROUP
    mask_w = ow < OW
    value = tl.load(
        loss_ptr
        + batch * LOSS_STRIDE_N
        + co[:, None] * LOSS_STRIDE_C
        + ow[None, :] * LOSS_STRIDE_W,
        mask=mask_co[:, None] & mask_w[None, :],
        other=0.0,
    )
    tl.store(
        packed_ptr
        + (conv_group * COUT_PER_GROUP + co_local[:, None]) * REDUCTION
        + batch * OW
        + ow[None, :],
        value,
        mask=mask_co[:, None] & mask_w[None, :],
    )


@libentry()
@triton.jit
def _conv_wgrad_pack_image_kernel(
    image_ptr,
    packed_ptr,
    REDUCTION: tl.constexpr,
    GROUPS: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    IMAGE_STRIDE_N: tl.constexpr,
    IMAGE_STRIDE_C: tl.constexpr,
    IMAGE_STRIDE_D: tl.constexpr,
    IMAGE_STRIDE_H: tl.constexpr,
    IMAGE_STRIDE_W: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_COL: tl.constexpr,
):
    output_spatial = OD * OH * OW
    kernel_volume = KD * KH * KW
    columns = CIN_PER_GROUP * kernel_volume
    offsets_r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    offsets_col = tl.program_id(1) * BLOCK_COL + tl.arange(0, BLOCK_COL)
    conv_group = tl.program_id(2)
    mask_r = offsets_r < REDUCTION
    mask_col = offsets_col < columns
    safe_r = tl.where(mask_r, offsets_r, 0)
    safe_col = tl.where(mask_col, offsets_col, 0)
    batch = safe_r // output_spatial
    spatial = safe_r - batch * output_spatial
    ow = spatial % OW
    oh = (spatial // OW) % OH
    od = spatial // (OH * OW)
    ci_local = safe_col // kernel_volume
    kernel_offset = safe_col - ci_local * kernel_volume
    weight_w = kernel_offset % KW
    weight_h = (kernel_offset // KW) % KH
    weight_d = kernel_offset // (KH * KW)
    sample_d = KD - 1 - weight_d if FILTER_REVERSE else weight_d
    sample_h = KH - 1 - weight_h if FILTER_REVERSE else weight_h
    sample_w = KW - 1 - weight_w if FILTER_REVERSE else weight_w
    input_d = od[:, None] * STRIDE_D - PAD_D + sample_d[None, :] * DIL_D
    input_h = oh[:, None] * STRIDE_H - PAD_H + sample_h[None, :] * DIL_H
    input_w = ow[:, None] * STRIDE_W - PAD_W + sample_w[None, :] * DIL_W
    valid = (
        mask_r[:, None]
        & mask_col[None, :]
        & (input_d >= 0)
        & (input_d < XD)
        & (input_h >= 0)
        & (input_h < XH)
        & (input_w >= 0)
        & (input_w < XW)
    )
    safe_d = tl.where(valid, input_d, 0)
    safe_h = tl.where(valid, input_h, 0)
    safe_w = tl.where(valid, input_w, 0)
    ci = conv_group * CIN_PER_GROUP + ci_local
    value = tl.load(
        image_ptr
        + batch[:, None] * IMAGE_STRIDE_N
        + ci[None, :] * IMAGE_STRIDE_C
        + safe_d * IMAGE_STRIDE_D
        + safe_h * IMAGE_STRIDE_H
        + safe_w * IMAGE_STRIDE_W,
        mask=valid,
        other=0.0,
    )
    tl.store(
        packed_ptr
        + (conv_group * REDUCTION + safe_r[:, None]) * columns
        + safe_col[None, :],
        value,
        mask=mask_r[:, None] & mask_col[None, :],
    )


@libentry()
@triton.jit
def _conv_wgrad_pack_image_1d_kernel(
    image_ptr,
    packed_ptr,
    GROUPS: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    XW: tl.constexpr,
    OW: tl.constexpr,
    KW: tl.constexpr,
    REDUCTION: tl.constexpr,
    COLUMNS: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_W: tl.constexpr,
    IMAGE_STRIDE_N: tl.constexpr,
    IMAGE_STRIDE_C: tl.constexpr,
    IMAGE_STRIDE_W: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    ow = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    ci_local = tl.program_id(1) * BLOCK_CI + tl.arange(0, BLOCK_CI)
    batch_group = tl.program_id(2)
    batch = batch_group // GROUPS
    conv_group = batch_group - batch * GROUPS
    ci = conv_group * CIN_PER_GROUP + ci_local
    mask_ci = ci_local < CIN_PER_GROUP

    for kernel_w in tl.static_range(0, KW):
        sample_w = KW - 1 - kernel_w if FILTER_REVERSE else kernel_w
        input_w = ow * STRIDE_W - PAD_W + sample_w * DIL_W
        mask_r = (ow < OW) & (input_w >= 0) & (input_w < XW)
        safe_w = tl.where(mask_r, input_w, 0)
        value = tl.load(
            image_ptr
            + batch * IMAGE_STRIDE_N
            + ci[:, None] * IMAGE_STRIDE_C
            + safe_w[None, :] * IMAGE_STRIDE_W,
            mask=mask_ci[:, None] & mask_r[None, :],
            other=0.0,
        )
        tl.store(
            packed_ptr
            + (conv_group * REDUCTION + batch * OW + ow[:, None]) * COLUMNS
            + kernel_w * CIN_PER_GROUP
            + ci_local[None, :],
            tl.trans(value),
            mask=(ow[:, None] < OW) & mask_ci[None, :],
        )


@libentry()
@triton.jit
def _conv_wgrad_pack_image_1d_block_kernel(
    image_ptr,
    packed_ptr,
    GROUPS: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    XW: tl.constexpr,
    OW: tl.constexpr,
    KW: tl.constexpr,
    REDUCTION: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_W: tl.constexpr,
    IMAGE_STRIDE_N: tl.constexpr,
    IMAGE_STRIDE_C: tl.constexpr,
    IMAGE_STRIDE_W: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    block_r_start = tl.program_id(0) * BLOCK_R
    block_ci_start = tl.program_id(1) * BLOCK_CI
    batch_group = tl.program_id(2)
    batch = batch_group // GROUPS
    conv_group = batch_group - batch * GROUPS
    image_base = (
        image_ptr
        + batch * IMAGE_STRIDE_N
        + conv_group * CIN_PER_GROUP * IMAGE_STRIDE_C
    )
    packed_base = (
        packed_ptr + conv_group * KW * CIN_PER_GROUP * REDUCTION + batch * OW
    )

    for kernel_w in tl.static_range(0, KW):
        sample_w = KW - 1 - kernel_w if FILTER_REVERSE else kernel_w
        image_block = tl.make_block_ptr(
            base=image_base,
            shape=(CIN_PER_GROUP, XW),
            strides=(IMAGE_STRIDE_C, IMAGE_STRIDE_W),
            offsets=(
                block_ci_start,
                block_r_start - PAD_W + sample_w * DIL_W,
            ),
            block_shape=(BLOCK_CI, BLOCK_R),
            order=(1, 0),
        )
        value = tl.load(
            image_block,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        packed_block = tl.make_block_ptr(
            base=packed_base,
            shape=(KW * CIN_PER_GROUP, OW),
            strides=(REDUCTION, 1),
            offsets=(
                kernel_w * CIN_PER_GROUP + block_ci_start,
                block_r_start,
            ),
            block_shape=(BLOCK_CI, BLOCK_R),
            order=(1, 0),
        )
        tl.store(
            packed_block,
            value,
            boundary_check=(0, 1),
        )


@libentry()
@triton.jit
def _conv_wgrad_pack_image_1d_gather_kernel(
    image_ptr,
    packed_ptr,
    GROUPS: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    XW: tl.constexpr,
    OW: tl.constexpr,
    KW: tl.constexpr,
    REDUCTION: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_W: tl.constexpr,
    IMAGE_STRIDE_N: tl.constexpr,
    IMAGE_STRIDE_C: tl.constexpr,
    IMAGE_STRIDE_W: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    LOAD_W: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    block_r_start = tl.program_id(0) * BLOCK_R
    block_ci_start = tl.program_id(1) * BLOCK_CI
    batch_group = tl.program_id(2)
    batch = batch_group // GROUPS
    conv_group = batch_group - batch * GROUPS
    image_base = (
        image_ptr
        + batch * IMAGE_STRIDE_N
        + conv_group * CIN_PER_GROUP * IMAGE_STRIDE_C
    )
    image_block = tl.make_block_ptr(
        base=image_base,
        shape=(CIN_PER_GROUP, XW),
        strides=(IMAGE_STRIDE_C, IMAGE_STRIDE_W),
        offsets=(block_ci_start, 0),
        block_shape=(BLOCK_CI, LOAD_W),
        order=(1, 0),
    )
    image = tl.load(
        image_block,
        boundary_check=(0, 1),
        padding_option="zero",
    )
    ow = block_r_start + tl.arange(0, BLOCK_R)
    packed_base = (
        packed_ptr + conv_group * KW * CIN_PER_GROUP * REDUCTION + batch * OW
    )

    for kernel_w in tl.static_range(0, KW):
        sample_w = KW - 1 - kernel_w if FILTER_REVERSE else kernel_w
        input_w = ow * STRIDE_W - PAD_W + sample_w * DIL_W
        valid = (input_w >= 0) & (input_w < XW) & (ow < OW)
        safe_input_w = tl.where(valid, input_w, 0)
        index = tl.broadcast_to(
            safe_input_w[None, :],
            (BLOCK_CI, BLOCK_R),
        )
        value = tl.gather(image, index, axis=1)
        value = tl.where(valid[None, :], value, 0.0)
        packed_block = tl.make_block_ptr(
            base=packed_base,
            shape=(KW * CIN_PER_GROUP, OW),
            strides=(REDUCTION, 1),
            offsets=(
                kernel_w * CIN_PER_GROUP + block_ci_start,
                block_r_start,
            ),
            block_shape=(BLOCK_CI, BLOCK_R),
            order=(1, 0),
        )
        tl.store(
            packed_block,
            value,
            boundary_check=(0, 1),
        )


@libentry()
@triton.jit
def _conv_wgrad_pack_image_3d_full_volume_kernel(
    image_ptr,
    packed_ptr,
    GROUPS: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    REDUCTION: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    IMAGE_STRIDE_N: tl.constexpr,
    IMAGE_STRIDE_C: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    LOAD_X: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
):
    block_ci_start = tl.program_id(0) * BLOCK_CI
    batch_group = tl.program_id(1)
    batch = batch_group // GROUPS
    conv_group = batch_group - batch * GROUPS
    input_spatial = XD * XH * XW
    output_spatial = OD * OH * OW
    kernel_volume = KD * KH * KW
    columns = CIN_PER_GROUP * kernel_volume
    image_base = (
        image_ptr
        + batch * IMAGE_STRIDE_N
        + conv_group * CIN_PER_GROUP * IMAGE_STRIDE_C
    )
    image_block = tl.make_block_ptr(
        base=image_base,
        shape=(CIN_PER_GROUP, input_spatial),
        strides=(IMAGE_STRIDE_C, 1),
        offsets=(block_ci_start, 0),
        block_shape=(BLOCK_CI, LOAD_X),
        order=(1, 0),
    )
    image = tl.load(
        image_block,
        boundary_check=(0, 1),
        padding_option="zero",
    )
    packed_base = (
        packed_ptr + conv_group * columns * REDUCTION + batch * output_spatial
    )

    for kernel_d in tl.static_range(0, KD):
        sample_d = KD - 1 - kernel_d if FILTER_REVERSE else kernel_d
        for kernel_h in tl.static_range(0, KH):
            sample_h = KH - 1 - kernel_h if FILTER_REVERSE else kernel_h
            for kernel_w in tl.static_range(0, KW):
                sample_w = KW - 1 - kernel_w if FILTER_REVERSE else kernel_w
                kernel_offset = kernel_d * KH * KW + kernel_h * KW + kernel_w
                for spatial_start in tl.range(
                    0,
                    output_spatial,
                    BLOCK_SPATIAL,
                    disallow_acc_multi_buffer=True,
                ):
                    spatial = spatial_start + tl.arange(0, BLOCK_SPATIAL)
                    output_d = spatial // (OH * OW)
                    output_hw = spatial - output_d * OH * OW
                    output_h = output_hw // OW
                    output_w = output_hw - output_h * OW
                    input_d = output_d * STRIDE_D - PAD_D + sample_d * DIL_D
                    input_h = output_h * STRIDE_H - PAD_H + sample_h * DIL_H
                    input_w = output_w * STRIDE_W - PAD_W + sample_w * DIL_W
                    valid = (
                        (spatial < output_spatial)
                        & (input_d >= 0)
                        & (input_d < XD)
                        & (input_h >= 0)
                        & (input_h < XH)
                        & (input_w >= 0)
                        & (input_w < XW)
                    )
                    safe_input = tl.where(
                        valid,
                        (input_d * XH * XW + input_h * XW + input_w),
                        0,
                    )
                    index = tl.broadcast_to(
                        safe_input[None, :],
                        (BLOCK_CI, BLOCK_SPATIAL),
                    )
                    value = tl.gather(image, index, axis=1)
                    value = tl.where(
                        valid[None, :],
                        value,
                        0.0,
                    )
                    packed_block = tl.make_block_ptr(
                        base=(packed_base + kernel_offset * REDUCTION),
                        shape=(CIN_PER_GROUP, output_spatial),
                        strides=(kernel_volume * REDUCTION, 1),
                        offsets=(block_ci_start, spatial_start),
                        block_shape=(BLOCK_CI, BLOCK_SPATIAL),
                        order=(1, 0),
                    )
                    tl.store(
                        packed_block,
                        value,
                        boundary_check=(0, 1),
                    )


@libentry()
@triton.jit
def _conv_wgrad_pack_image_2d_full_plane_kernel(
    image_ptr,
    packed_ptr,
    GROUPS: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    REDUCTION: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    IMAGE_STRIDE_N: tl.constexpr,
    IMAGE_STRIDE_C: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    LOAD_X: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
):
    block_ci_start = tl.program_id(0) * BLOCK_CI
    batch_group = tl.program_id(1)
    batch = batch_group // GROUPS
    conv_group = batch_group - batch * GROUPS
    input_spatial = XH * XW
    output_spatial = OH * OW
    kernel_volume = KH * KW
    columns = CIN_PER_GROUP * kernel_volume
    image_base = (
        image_ptr
        + batch * IMAGE_STRIDE_N
        + conv_group * CIN_PER_GROUP * IMAGE_STRIDE_C
    )
    image_block = tl.make_block_ptr(
        base=image_base,
        shape=(CIN_PER_GROUP, input_spatial),
        strides=(IMAGE_STRIDE_C, 1),
        offsets=(block_ci_start, 0),
        block_shape=(BLOCK_CI, LOAD_X),
        order=(1, 0),
    )
    image = tl.load(
        image_block,
        boundary_check=(0, 1),
        padding_option="zero",
    )
    packed_base = (
        packed_ptr + conv_group * columns * REDUCTION + batch * output_spatial
    )

    for kernel_h in tl.static_range(0, KH):
        sample_h = KH - 1 - kernel_h if FILTER_REVERSE else kernel_h
        for kernel_w in tl.static_range(0, KW):
            sample_w = KW - 1 - kernel_w if FILTER_REVERSE else kernel_w
            kernel_offset = kernel_h * KW + kernel_w
            for spatial_start in tl.range(
                0,
                output_spatial,
                BLOCK_SPATIAL,
                disallow_acc_multi_buffer=True,
            ):
                spatial = spatial_start + tl.arange(0, BLOCK_SPATIAL)
                oh = spatial // OW
                ow = spatial - oh * OW
                input_h = oh * STRIDE_H - PAD_H + sample_h * DIL_H
                input_w = ow * STRIDE_W - PAD_W + sample_w * DIL_W
                valid = (
                    (spatial < output_spatial)
                    & (input_h >= 0)
                    & (input_h < XH)
                    & (input_w >= 0)
                    & (input_w < XW)
                )
                safe_input = tl.where(
                    valid,
                    input_h * XW + input_w,
                    0,
                )
                index = tl.broadcast_to(
                    safe_input[None, :],
                    (BLOCK_CI, BLOCK_SPATIAL),
                )
                value = tl.gather(image, index, axis=1)
                value = tl.where(
                    valid[None, :],
                    value,
                    0.0,
                )
                packed_block = tl.make_block_ptr(
                    base=(packed_base + kernel_offset * REDUCTION),
                    shape=(CIN_PER_GROUP, output_spatial),
                    strides=(kernel_volume * REDUCTION, 1),
                    offsets=(block_ci_start, spatial_start),
                    block_shape=(BLOCK_CI, BLOCK_SPATIAL),
                    order=(1, 0),
                )
                tl.store(
                    packed_block,
                    value,
                    boundary_check=(0, 1),
                )


@libentry()
@triton.jit
def _conv_wgrad_pack_image_2d_row_tile_kernel(
    image_ptr,
    packed_ptr,
    GROUPS: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    REDUCTION: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    IMAGE_STRIDE_N: tl.constexpr,
    IMAGE_STRIDE_C: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    ROW_TILES: tl.constexpr,
    ROWS_PER_TILE: tl.constexpr,
    LOAD_X: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
):
    block_ci_start = tl.program_id(0) * BLOCK_CI
    batch_group = tl.program_id(1)
    batch = batch_group // GROUPS
    conv_group = batch_group - batch * GROUPS
    input_spatial = XH * XW
    output_spatial = OH * OW
    kernel_volume = KH * KW
    columns = CIN_PER_GROUP * kernel_volume
    image_base = (
        image_ptr
        + batch * IMAGE_STRIDE_N
        + conv_group * CIN_PER_GROUP * IMAGE_STRIDE_C
    )
    packed_base = (
        packed_ptr + conv_group * columns * REDUCTION + batch * output_spatial
    )

    for row_tile in tl.static_range(0, ROW_TILES):
        output_row_start = row_tile * ROWS_PER_TILE
        output_start = output_row_start * OW
        input_row_start = tl.maximum(
            output_row_start * STRIDE_H - PAD_H,
            0,
        )
        input_start = input_row_start * XW
        image_block = tl.make_block_ptr(
            base=image_base,
            shape=(CIN_PER_GROUP, input_spatial),
            strides=(IMAGE_STRIDE_C, 1),
            offsets=(block_ci_start, input_start),
            block_shape=(BLOCK_CI, LOAD_X),
            order=(1, 0),
        )
        image = tl.load(
            image_block,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        local_spatial = tl.arange(0, BLOCK_SPATIAL)
        spatial = output_start + local_spatial
        oh = spatial // OW
        ow = spatial - oh * OW
        mask_spatial = (local_spatial < ROWS_PER_TILE * OW) & (
            spatial < output_spatial
        )

        for kernel_h in tl.static_range(0, KH):
            sample_h = KH - 1 - kernel_h if FILTER_REVERSE else kernel_h
            for kernel_w in tl.static_range(0, KW):
                sample_w = KW - 1 - kernel_w if FILTER_REVERSE else kernel_w
                input_h = oh * STRIDE_H - PAD_H + sample_h * DIL_H
                input_w = ow * STRIDE_W - PAD_W + sample_w * DIL_W
                valid = (
                    mask_spatial
                    & (input_h >= 0)
                    & (input_h < XH)
                    & (input_w >= 0)
                    & (input_w < XW)
                )
                safe_input = tl.where(
                    valid,
                    input_h * XW + input_w - input_start,
                    0,
                )
                index = tl.broadcast_to(
                    safe_input[None, :],
                    (BLOCK_CI, BLOCK_SPATIAL),
                )
                value = tl.gather(image, index, axis=1)
                value = tl.where(
                    valid[None, :],
                    value,
                    0.0,
                )
                kernel_offset = kernel_h * KW + kernel_w
                packed_block = tl.make_block_ptr(
                    base=(packed_base + kernel_offset * REDUCTION),
                    shape=(CIN_PER_GROUP, output_spatial),
                    strides=(kernel_volume * REDUCTION, 1),
                    offsets=(block_ci_start, output_start),
                    block_shape=(BLOCK_CI, BLOCK_SPATIAL),
                    order=(1, 0),
                )
                tl.store(
                    packed_block,
                    value,
                    boundary_check=(0, 1),
                )


@libentry()
@triton.jit
def _conv_wgrad_matmul_transposed_image_kernel(
    loss_ptr,
    image_t_ptr,
    weight_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    TF32: tl.constexpr,
):
    program_id = tl.program_id(0)
    programs_m = tl.cdiv(M, BLOCK_M)
    programs_n = tl.cdiv(N, BLOCK_N)
    programs_per_batch = programs_m * programs_n
    batch_id = program_id // programs_per_batch
    tile_id = program_id - batch_id * programs_per_batch

    programs_per_group = GROUP_M * programs_n
    group_id = tile_id // programs_per_group
    first_program_m = group_id * GROUP_M
    group_size_m = tl.minimum(
        programs_m - first_program_m,
        GROUP_M,
    )
    tile_in_group = tile_id - group_id * programs_per_group
    program_m = first_program_m + tile_in_group % group_size_m
    program_n = tile_in_group // group_size_m

    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    loss_ptrs = (
        loss_ptr
        + batch_id * M * K
        + offsets_m[:, None] * K
        + offsets_k[None, :]
    )
    image_t_ptrs = (
        image_t_ptr
        + batch_id * N * K
        + offsets_n[:, None] * K
        + offsets_k[None, :]
    )
    accumulator = tl.zeros(
        (BLOCK_M, BLOCK_N),
        dtype=tl.float32,
    )
    for k_start in tl.range(0, K, BLOCK_K):
        packed_loss = tl.load(
            loss_ptrs,
            mask=(offsets_m[:, None] < M) & (k_start + offsets_k[None, :] < K),
            other=0.0,
        )
        packed_image_t = tl.load(
            image_t_ptrs,
            mask=(offsets_n[:, None] < N) & (k_start + offsets_k[None, :] < K),
            other=0.0,
        )
        if TF32:
            accumulator = tl.dot(
                packed_loss,
                tl.trans(packed_image_t),
                accumulator,
                input_precision="tf32",
            )
        else:
            accumulator = tl.dot(
                packed_loss,
                tl.trans(packed_image_t),
                accumulator,
            )
        loss_ptrs += BLOCK_K
        image_t_ptrs += BLOCK_K

    tl.store(
        weight_ptr
        + batch_id * M * N
        + offsets_m[:, None] * N
        + offsets_n[None, :],
        accumulator.to(weight_ptr.dtype.element_ty),
        mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
    )


@libentry()
@triton.jit
def _conv_wgrad_matmul_transposed_image_splitk_kernel(
    loss_ptr,
    image_t_ptr,
    partial_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TF32: tl.constexpr,
):
    programs_m = tl.cdiv(M, BLOCK_M)
    programs_n = tl.cdiv(N, BLOCK_N)
    programs_per_split = programs_m * programs_n
    program_id = tl.program_id(0)
    split = program_id // programs_per_split
    tile = program_id - split * programs_per_split
    program_m = tile // programs_n
    program_n = tile - program_m * programs_n
    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    split_start = split * SPLIT_K
    loss_ptrs = (
        loss_ptr + offsets_m[:, None] * K + split_start + offsets_k[None, :]
    )
    image_ptrs = (
        image_t_ptr + offsets_n[:, None] * K + split_start + offsets_k[None, :]
    )
    accumulator = tl.zeros(
        (BLOCK_M, BLOCK_N),
        dtype=tl.float32,
    )
    for k_offset in tl.range(0, SPLIT_K, BLOCK_K):
        global_k = split_start + k_offset + offsets_k
        packed_loss = tl.load(
            loss_ptrs,
            mask=(offsets_m[:, None] < M) & (global_k[None, :] < K),
            other=0.0,
        )
        packed_image = tl.load(
            image_ptrs,
            mask=(offsets_n[:, None] < N) & (global_k[None, :] < K),
            other=0.0,
        )
        if TF32:
            accumulator = tl.dot(
                packed_loss,
                tl.trans(packed_image),
                accumulator,
                input_precision="tf32",
            )
        else:
            accumulator = tl.dot(
                packed_loss,
                tl.trans(packed_image),
                accumulator,
            )
        loss_ptrs += BLOCK_K
        image_ptrs += BLOCK_K
    output_offset = split * M * N + offsets_m[:, None] * N + offsets_n[None, :]
    tl.store(
        partial_ptr + output_offset,
        accumulator,
        mask=(split < NUM_SPLITS)
        & (offsets_m[:, None] < M)
        & (offsets_n[None, :] < N),
    )


@libentry()
@triton.jit
def _conv_wgrad_reorder_1d_kernel(
    packed_ptr,
    output_ptr,
    TOTAL: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    KW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    safe_offsets = tl.where(mask, offsets, 0)
    kernel_w = safe_offsets % KW
    channel_offset = safe_offsets // KW
    ci_local = channel_offset % CIN_PER_GROUP
    co = channel_offset // CIN_PER_GROUP
    source_offsets = (
        co * CIN_PER_GROUP * KW + kernel_w * CIN_PER_GROUP + ci_local
    )
    value = tl.load(
        packed_ptr + source_offsets,
        mask=mask,
        other=0.0,
    )
    tl.store(output_ptr + offsets, value, mask=mask)


@libentry()
@triton.jit
def _conv_fprop_kernel(
    image_ptr,
    weight_ptr,
    output_ptr,
    N: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    IMAGE_STRIDE_N: tl.constexpr,
    IMAGE_STRIDE_C: tl.constexpr,
    IMAGE_STRIDE_D: tl.constexpr,
    IMAGE_STRIDE_H: tl.constexpr,
    IMAGE_STRIDE_W: tl.constexpr,
    WEIGHT_STRIDE_O: tl.constexpr,
    WEIGHT_STRIDE_I: tl.constexpr,
    WEIGHT_STRIDE_D: tl.constexpr,
    WEIGHT_STRIDE_H: tl.constexpr,
    WEIGHT_STRIDE_W: tl.constexpr,
    OUTPUT_STRIDE_N: tl.constexpr,
    OUTPUT_STRIDE_C: tl.constexpr,
    OUTPUT_STRIDE_D: tl.constexpr,
    OUTPUT_STRIDE_H: tl.constexpr,
    OUTPUT_STRIDE_W: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    FP32_INPUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    output_spatial = OD * OH * OW
    matrix_m = N * output_spatial
    tiles_m = tl.cdiv(matrix_m, BLOCK_M)
    tiles_n = tl.cdiv(COUT_PER_GROUP, BLOCK_N)
    programs_per_group = tiles_m * tiles_n
    program_id = tl.program_id(0)
    conv_group = program_id // programs_per_group
    tile_id = program_id - conv_group * programs_per_group

    programs_per_m_group = GROUP_M * tiles_n
    m_group = tile_id // programs_per_m_group
    first_tile_m = m_group * GROUP_M
    group_size_m = tl.minimum(tiles_m - first_tile_m, GROUP_M)
    tile_in_group = tile_id - m_group * programs_per_m_group
    tile_m = first_tile_m + tile_in_group % group_size_m
    tile_n = tile_in_group // group_size_m

    offsets_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_oc = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offsets_m < matrix_m
    mask_oc = offsets_oc < COUT_PER_GROUP
    batch = offsets_m // output_spatial
    spatial = offsets_m - batch * output_spatial
    ow = spatial % OW
    oh = (spatial // OW) % OH
    od = spatial // (OH * OW)
    oc = conv_group * COUT_PER_GROUP + offsets_oc

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kd in tl.static_range(0, KD):
        input_d = od * STRIDE_D - PAD_D + kd * DIL_D
        valid_d = (input_d >= 0) & (input_d < XD)
        safe_d = tl.where(valid_d, input_d, 0)
        weight_d = KD - 1 - kd if FILTER_REVERSE else kd
        for kh in tl.static_range(0, KH):
            input_h = oh * STRIDE_H - PAD_H + kh * DIL_H
            valid_h = (input_h >= 0) & (input_h < XH)
            safe_h = tl.where(valid_h, input_h, 0)
            weight_h = KH - 1 - kh if FILTER_REVERSE else kh
            for kw in tl.static_range(0, KW):
                input_w = ow * STRIDE_W - PAD_W + kw * DIL_W
                valid_w = (input_w >= 0) & (input_w < XW)
                safe_w = tl.where(valid_w, input_w, 0)
                valid_spatial = mask_m & valid_d & valid_h & valid_w
                weight_w = KW - 1 - kw if FILTER_REVERSE else kw
                for ci_start in tl.range(
                    0,
                    CIN_PER_GROUP,
                    BLOCK_K,
                ):
                    ci_local = ci_start + tl.arange(0, BLOCK_K)
                    mask_k = ci_local < CIN_PER_GROUP
                    ci = conv_group * CIN_PER_GROUP + ci_local
                    image = tl.load(
                        image_ptr
                        + batch[:, None] * IMAGE_STRIDE_N
                        + ci[None, :] * IMAGE_STRIDE_C
                        + safe_d[:, None] * IMAGE_STRIDE_D
                        + safe_h[:, None] * IMAGE_STRIDE_H
                        + safe_w[:, None] * IMAGE_STRIDE_W,
                        mask=valid_spatial[:, None] & mask_k[None, :],
                        other=0.0,
                    )
                    weight = tl.load(
                        weight_ptr
                        + oc[None, :] * WEIGHT_STRIDE_O
                        + ci_local[:, None] * WEIGHT_STRIDE_I
                        + weight_d * WEIGHT_STRIDE_D
                        + weight_h * WEIGHT_STRIDE_H
                        + weight_w * WEIGHT_STRIDE_W,
                        mask=mask_k[:, None] & mask_oc[None, :],
                        other=0.0,
                    )
                    if FP32_INPUT:
                        accumulator = tl.dot(
                            image,
                            weight,
                            accumulator,
                            input_precision="tf32",
                        )
                    else:
                        accumulator = tl.dot(
                            image,
                            weight,
                            accumulator,
                        )

    tl.store(
        output_ptr
        + batch[:, None] * OUTPUT_STRIDE_N
        + oc[None, :] * OUTPUT_STRIDE_C
        + od[:, None] * OUTPUT_STRIDE_D
        + oh[:, None] * OUTPUT_STRIDE_H
        + ow[:, None] * OUTPUT_STRIDE_W,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_oc[None, :],
    )


@libentry()
@triton.jit
def _conv_dgrad_kernel(
    loss_ptr,
    weight_ptr,
    output_ptr,
    N: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    LOSS_STRIDE_N: tl.constexpr,
    LOSS_STRIDE_C: tl.constexpr,
    LOSS_STRIDE_D: tl.constexpr,
    LOSS_STRIDE_H: tl.constexpr,
    LOSS_STRIDE_W: tl.constexpr,
    WEIGHT_STRIDE_O: tl.constexpr,
    WEIGHT_STRIDE_I: tl.constexpr,
    WEIGHT_STRIDE_D: tl.constexpr,
    WEIGHT_STRIDE_H: tl.constexpr,
    WEIGHT_STRIDE_W: tl.constexpr,
    OUTPUT_STRIDE_N: tl.constexpr,
    OUTPUT_STRIDE_C: tl.constexpr,
    OUTPUT_STRIDE_D: tl.constexpr,
    OUTPUT_STRIDE_H: tl.constexpr,
    OUTPUT_STRIDE_W: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    FP32_INPUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    input_spatial = XD * XH * XW
    matrix_m = N * input_spatial
    tiles_m = tl.cdiv(matrix_m, BLOCK_M)
    tiles_n = tl.cdiv(CIN_PER_GROUP, BLOCK_N)
    programs_per_group = tiles_m * tiles_n
    program_id = tl.program_id(0)
    conv_group = program_id // programs_per_group
    tile_id = program_id - conv_group * programs_per_group

    programs_per_m_group = GROUP_M * tiles_n
    m_group = tile_id // programs_per_m_group
    first_tile_m = m_group * GROUP_M
    group_size_m = tl.minimum(tiles_m - first_tile_m, GROUP_M)
    tile_in_group = tile_id - m_group * programs_per_m_group
    tile_m = first_tile_m + tile_in_group % group_size_m
    tile_n = tile_in_group // group_size_m

    offsets_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_ci = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offsets_m < matrix_m
    mask_ci = offsets_ci < CIN_PER_GROUP
    batch = offsets_m // input_spatial
    spatial = offsets_m - batch * input_spatial
    xw = spatial % XW
    xh = (spatial // XW) % XH
    xd = spatial // (XH * XW)
    ci = conv_group * CIN_PER_GROUP + offsets_ci

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kd in tl.static_range(0, KD):
        numerator_d = xd + PAD_D - kd * DIL_D
        od = numerator_d // STRIDE_D
        valid_d = (
            (numerator_d >= 0) & (numerator_d % STRIDE_D == 0) & (od < OD)
        )
        safe_od = tl.where(valid_d, od, 0)
        weight_d = KD - 1 - kd if FILTER_REVERSE else kd
        for kh in tl.static_range(0, KH):
            numerator_h = xh + PAD_H - kh * DIL_H
            oh = numerator_h // STRIDE_H
            valid_h = (
                (numerator_h >= 0) & (numerator_h % STRIDE_H == 0) & (oh < OH)
            )
            safe_oh = tl.where(valid_h, oh, 0)
            weight_h = KH - 1 - kh if FILTER_REVERSE else kh
            for kw in tl.static_range(0, KW):
                numerator_w = xw + PAD_W - kw * DIL_W
                ow = numerator_w // STRIDE_W
                valid_w = (
                    (numerator_w >= 0)
                    & (numerator_w % STRIDE_W == 0)
                    & (ow < OW)
                )
                safe_ow = tl.where(valid_w, ow, 0)
                valid_spatial = mask_m & valid_d & valid_h & valid_w
                weight_w = KW - 1 - kw if FILTER_REVERSE else kw
                for co_start in tl.range(
                    0,
                    COUT_PER_GROUP,
                    BLOCK_K,
                ):
                    co_local = co_start + tl.arange(0, BLOCK_K)
                    mask_k = co_local < COUT_PER_GROUP
                    co = conv_group * COUT_PER_GROUP + co_local
                    loss = tl.load(
                        loss_ptr
                        + batch[:, None] * LOSS_STRIDE_N
                        + co[None, :] * LOSS_STRIDE_C
                        + safe_od[:, None] * LOSS_STRIDE_D
                        + safe_oh[:, None] * LOSS_STRIDE_H
                        + safe_ow[:, None] * LOSS_STRIDE_W,
                        mask=valid_spatial[:, None] & mask_k[None, :],
                        other=0.0,
                    )
                    weight = tl.load(
                        weight_ptr
                        + co_local[:, None] * WEIGHT_STRIDE_O
                        + conv_group * COUT_PER_GROUP * WEIGHT_STRIDE_O
                        + offsets_ci[None, :] * WEIGHT_STRIDE_I
                        + weight_d * WEIGHT_STRIDE_D
                        + weight_h * WEIGHT_STRIDE_H
                        + weight_w * WEIGHT_STRIDE_W,
                        mask=mask_k[:, None] & mask_ci[None, :],
                        other=0.0,
                    )
                    if FP32_INPUT:
                        accumulator = tl.dot(
                            loss,
                            weight,
                            accumulator,
                            input_precision="tf32",
                        )
                    else:
                        accumulator = tl.dot(
                            loss,
                            weight,
                            accumulator,
                        )

    tl.store(
        output_ptr
        + batch[:, None] * OUTPUT_STRIDE_N
        + ci[None, :] * OUTPUT_STRIDE_C
        + xd[:, None] * OUTPUT_STRIDE_D
        + xh[:, None] * OUTPUT_STRIDE_H
        + xw[:, None] * OUTPUT_STRIDE_W,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_ci[None, :],
    )


@libentry()
@triton.jit
def _conv_wgrad_kernel(
    image_ptr,
    loss_ptr,
    partial_ptr,
    output_ptr,
    N: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    XD: tl.constexpr,
    XH: tl.constexpr,
    XW: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    IMAGE_STRIDE_N: tl.constexpr,
    IMAGE_STRIDE_C: tl.constexpr,
    IMAGE_STRIDE_D: tl.constexpr,
    IMAGE_STRIDE_H: tl.constexpr,
    IMAGE_STRIDE_W: tl.constexpr,
    LOSS_STRIDE_N: tl.constexpr,
    LOSS_STRIDE_C: tl.constexpr,
    LOSS_STRIDE_D: tl.constexpr,
    LOSS_STRIDE_H: tl.constexpr,
    LOSS_STRIDE_W: tl.constexpr,
    FILTER_REVERSE: tl.constexpr,
    FP32_INPUT: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    kernel_volume = KD * KH * KW
    columns = CIN_PER_GROUP * kernel_volume
    total_weight = C_OUT * CIN_PER_GROUP * kernel_volume
    tiles_n = tl.cdiv(columns, BLOCK_N)
    tiles_co = tl.cdiv(COUT_PER_GROUP, BLOCK_CO)
    tiles_channels = tiles_co * tiles_n
    program_id = tl.program_id(0)
    channel_tile = program_id % tiles_channels
    remainder = program_id // tiles_channels
    split = remainder % NUM_SPLITS
    conv_group = remainder // NUM_SPLITS

    tile_co = channel_tile // tiles_n
    tile_n = channel_tile - tile_co * tiles_n
    co_local = tile_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offsets_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ci_local = offsets_n // kernel_volume
    kernel_offset = offsets_n - ci_local * kernel_volume
    mask_co = co_local < COUT_PER_GROUP
    mask_n = offsets_n < columns
    co = conv_group * COUT_PER_GROUP + co_local
    ci = conv_group * CIN_PER_GROUP + ci_local

    kw = kernel_offset % KW
    kh = (kernel_offset // KW) % KH
    kd = kernel_offset // (KH * KW)
    sample_d = KD - 1 - kd if FILTER_REVERSE else kd
    sample_h = KH - 1 - kh if FILTER_REVERSE else kh
    sample_w = KW - 1 - kw if FILTER_REVERSE else kw

    output_spatial = OD * OH * OW
    reduction = N * output_spatial
    split_size = tl.cdiv(reduction, NUM_SPLITS)
    split_begin = split * split_size
    split_end = tl.minimum(split_begin + split_size, reduction)
    accumulator = tl.zeros((BLOCK_CO, BLOCK_N), dtype=tl.float32)
    for reduction_start in tl.range(split_begin, split_end, BLOCK_R):
        offsets_r = reduction_start + tl.arange(0, BLOCK_R)
        mask_r = offsets_r < split_end
        safe_r = tl.where(mask_r, offsets_r, 0)
        spatial = safe_r % output_spatial
        batch = safe_r // output_spatial
        ow = spatial % OW
        oh = (spatial // OW) % OH
        od = spatial // (OH * OW)
        input_d = od[:, None] * STRIDE_D - PAD_D + sample_d[None, :] * DIL_D
        input_h = oh[:, None] * STRIDE_H - PAD_H + sample_h[None, :] * DIL_H
        input_w = ow[:, None] * STRIDE_W - PAD_W + sample_w[None, :] * DIL_W
        valid_image = (
            mask_r[:, None]
            & mask_n[None, :]
            & (input_d >= 0)
            & (input_d < XD)
            & (input_h >= 0)
            & (input_h < XH)
            & (input_w >= 0)
            & (input_w < XW)
        )
        safe_d = tl.where(valid_image, input_d, 0)
        safe_h = tl.where(valid_image, input_h, 0)
        safe_w = tl.where(valid_image, input_w, 0)
        image = tl.load(
            image_ptr
            + batch[:, None] * IMAGE_STRIDE_N
            + ci[None, :] * IMAGE_STRIDE_C
            + safe_d * IMAGE_STRIDE_D
            + safe_h * IMAGE_STRIDE_H
            + safe_w * IMAGE_STRIDE_W,
            mask=valid_image,
            other=0.0,
        )
        loss = tl.load(
            loss_ptr
            + batch[None, :] * LOSS_STRIDE_N
            + co[:, None] * LOSS_STRIDE_C
            + od[None, :] * LOSS_STRIDE_D
            + oh[None, :] * LOSS_STRIDE_H
            + ow[None, :] * LOSS_STRIDE_W,
            mask=mask_co[:, None] & mask_r[None, :],
            other=0.0,
        )
        if FP32_INPUT:
            accumulator = tl.dot(
                loss,
                image,
                accumulator,
                input_precision="tf32",
            )
        else:
            accumulator = tl.dot(loss, image, accumulator)

    output_offsets = co[:, None] * columns + offsets_n[None, :]
    output_mask = mask_co[:, None] & mask_n[None, :]
    if NUM_SPLITS == 1:
        tl.store(
            output_ptr + output_offsets,
            accumulator.to(output_ptr.dtype.element_ty),
            mask=output_mask,
        )
    else:
        tl.store(
            partial_ptr + split * total_weight + output_offsets,
            accumulator,
            mask=output_mask,
        )


@libentry()
@triton.jit
def _conv_wgrad_reduce_kernel(
    partial_ptr,
    output_ptr,
    TOTAL_WEIGHT: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    BLOCK_WEIGHT: tl.constexpr,
):
    offsets_weight = tl.program_id(0) * BLOCK_WEIGHT + tl.arange(
        0, BLOCK_WEIGHT
    )
    offsets_split = tl.arange(0, BLOCK_SPLITS)
    values = tl.load(
        partial_ptr
        + offsets_split[None, :] * TOTAL_WEIGHT
        + offsets_weight[:, None],
        mask=(offsets_weight[:, None] < TOTAL_WEIGHT)
        & (offsets_split[None, :] < NUM_SPLITS),
        other=0.0,
    )
    result = tl.sum(values, axis=1)
    tl.store(
        output_ptr + offsets_weight,
        result.to(output_ptr.dtype.element_ty),
        mask=offsets_weight < TOTAL_WEIGHT,
    )


def _tuple_n(value: Any, rank: int, name: str) -> tuple[int, ...]:
    if isinstance(value, int):
        return (int(value),) * rank
    result = tuple(int(item) for item in value)
    if len(result) != rank:
        raise RuntimeError(f"{name} must have length {rank}, got {value}")
    return result


def _normalize_padding(
    attrs: dict[str, Any],
    rank: int,
    kernel: tuple[int, ...],
    stride: tuple[int, ...],
    dilation: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    pre = attrs.get("pre_padding")
    post = attrs.get("post_padding")
    if pre is not None or post is not None:
        if pre is None or post is None:
            raise RuntimeError(
                "pre_padding and post_padding must be specified together"
            )
        return (
            _tuple_n(pre, rank, "pre_padding"),
            _tuple_n(post, rank, "post_padding"),
        )

    padding = attrs.get("padding")
    if padding is None:
        padding = 0
    if isinstance(padding, str):
        normalized = padding.lower()
        if normalized == "valid":
            zeros = (0,) * rank
            return zeros, zeros
        if normalized != "same":
            raise RuntimeError(f"unsupported padding mode: {padding}")
        if any(value != 1 for value in stride):
            raise RuntimeError("padding='same' requires stride=1")
        totals = tuple(
            dilation[axis] * (kernel[axis] - 1) for axis in range(rank)
        )
        before = tuple(value // 2 for value in totals)
        return before, tuple(
            totals[axis] - before[axis] for axis in range(rank)
        )
    if isinstance(padding, int):
        values = (int(padding),) * rank
        return values, values
    values = tuple(int(value) for value in padding)
    if len(values) == rank:
        return values, values
    if len(values) == 2 * rank:
        return (
            tuple(values[2 * axis] for axis in range(rank)),
            tuple(values[2 * axis + 1] for axis in range(rank)),
        )
    raise RuntimeError(
        f"padding must have length {rank} or {2 * rank}, got {padding}"
    )


def _output_spatial(
    input_spatial: tuple[int, ...],
    kernel: tuple[int, ...],
    stride: tuple[int, ...],
    pre: tuple[int, ...],
    post: tuple[int, ...],
    dilation: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        (
            input_spatial[axis]
            + pre[axis]
            + post[axis]
            - dilation[axis] * (kernel[axis] - 1)
            - 1
        )
        // stride[axis]
        + 1
        for axis in range(len(input_spatial))
    )


def _normalized_spatial(
    values: Sequence[int],
    fill: int,
) -> tuple[int, int, int]:
    result = tuple(int(value) for value in values)
    if len(result) == 1:
        return fill, fill, result[0]
    if len(result) == 2:
        return fill, result[0], result[1]
    if len(result) == 3:
        return result
    raise RuntimeError(f"expected 1D/2D/3D values, got {values}")


def _normalized_tensor(
    tensor: torch.Tensor,
    rank: int,
) -> tuple[
    tuple[int, int, int, int, int],
    tuple[int, int, int, int, int],
]:
    shape = tuple(int(value) for value in tensor.shape)
    strides = tuple(int(value) for value in tensor.stride())
    if rank == 1 and tensor.dim() == 2:
        c, width = shape
        stride_c, stride_w = strides
        return (1, c, 1, 1, width), (0, stride_c, 0, 0, stride_w)
    if rank == 1:
        n, c, width = shape
        stride_n, stride_c, stride_w = strides
        return (
            n,
            c,
            1,
            1,
            width,
        ), (
            stride_n,
            stride_c,
            0,
            0,
            stride_w,
        )
    if rank == 2:
        n, c, height, width = shape
        stride_n, stride_c, stride_h, stride_w = strides
        return (
            n,
            c,
            1,
            height,
            width,
        ), (
            stride_n,
            stride_c,
            0,
            stride_h,
            stride_w,
        )
    n, c, depth, height, width = shape
    return (
        n,
        c,
        depth,
        height,
        width,
    ), (
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        strides[4],
    )


def _normalized_weight(
    tensor: torch.Tensor,
    rank: int,
) -> tuple[
    tuple[int, int, int, int, int],
    tuple[int, int, int, int, int],
]:
    shape = tuple(int(value) for value in tensor.shape)
    strides = tuple(int(value) for value in tensor.stride())
    if rank == 1:
        c_out, c_in, width = shape
        stride_o, stride_i, stride_w = strides
        return (
            c_out,
            c_in,
            1,
            1,
            width,
        ), (
            stride_o,
            stride_i,
            0,
            0,
            stride_w,
        )
    if rank == 2:
        c_out, c_in, height, width = shape
        stride_o, stride_i, stride_h, stride_w = strides
        return (
            c_out,
            c_in,
            1,
            height,
            width,
        ), (
            stride_o,
            stride_i,
            0,
            stride_h,
            stride_w,
        )
    return (
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        shape[4],
    ), (
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        strides[4],
    )


@dataclass(frozen=True)
class _ConvPlan:
    op_type: str
    rank: int
    output_shape: tuple[int, ...]
    stride: tuple[int, ...]
    pre: tuple[int, ...]
    post: tuple[int, ...]
    dilation: tuple[int, ...]
    groups: int
    filter_reverse: bool


def _make_plan(
    op_type: str,
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
) -> Optional[_ConvPlan]:
    if len(input_specs) < 2:
        return None
    first_shape = _static_shape(input_specs[0])
    second_shape = _static_shape(input_specs[1])
    if first_shape is None or second_shape is None:
        return None

    if op_type == "conv_fprop":
        rank = len(second_shape) - 2
        image_shape = first_shape
        weight_shape = second_shape
    elif op_type == "conv_dgrad":
        output_shape = tuple(int(value) for value in attrs["input_size"])
        rank = len(second_shape) - 2
        image_shape = output_shape
        weight_shape = second_shape
    elif op_type == "conv_wgrad":
        output_shape = tuple(int(value) for value in attrs["filter_size"])
        rank = len(output_shape) - 2
        image_shape = first_shape
        weight_shape = output_shape
    else:
        return None

    if rank not in (1, 2, 3):
        return None
    groups = int(attrs.get("groups", 1))
    input_channels = int(image_shape[-rank - 1])
    output_channels = int(weight_shape[0])
    if (
        groups <= 0
        or input_channels % groups != 0
        or output_channels % groups != 0
        or int(weight_shape[1]) != input_channels // groups
    ):
        return None
    unbatched_1d = rank == 1 and len(image_shape) == 2
    input_spatial = tuple(int(value) for value in image_shape[-rank:])
    kernel = tuple(int(value) for value in weight_shape[-rank:])
    stride = _tuple_n(attrs.get("stride", 1), rank, "stride")
    dilation = _tuple_n(attrs.get("dilation", 1), rank, "dilation")
    pre, post = _normalize_padding(attrs, rank, kernel, stride, dilation)
    loss_spatial = _output_spatial(
        input_spatial, kernel, stride, pre, post, dilation
    )
    if any(value <= 0 for value in loss_spatial):
        return None

    if op_type == "conv_fprop":
        if unbatched_1d:
            output_shape = (int(weight_shape[0]), *loss_spatial)
        else:
            output_shape = (
                int(image_shape[0]),
                int(weight_shape[0]),
                *loss_spatial,
            )
    elif op_type == "conv_dgrad":
        output_shape = tuple(int(value) for value in attrs["input_size"])
    else:
        output_shape = tuple(int(value) for value in attrs["filter_size"])

    mode = (
        str(attrs.get("convolution_mode", "CROSS_CORRELATION"))
        .rsplit(".", 1)[-1]
        .upper()
    )
    if mode not in ("CROSS_CORRELATION", "CONVOLUTION"):
        return None
    return _ConvPlan(
        op_type=op_type,
        rank=rank,
        output_shape=output_shape,
        stride=stride,
        pre=pre,
        post=post,
        dilation=dilation,
        groups=groups,
        filter_reverse=mode == "CONVOLUTION",
    )


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _is_pointwise_conv(
    plan: _ConvPlan,
    weight_shape: tuple[int, int, int, int, int],
) -> bool:
    return (
        weight_shape[2:] == (1, 1, 1)
        and plan.stride == (1,) * plan.rank
        and all(value == 0 for value in plan.pre + plan.post)
    )


def _supports_fast_conv_1d_pack(
    plan: _ConvPlan,
    input_width: int,
) -> bool:
    if plan.rank != 1:
        return False
    # Unit-stride windows can be expressed directly as block pointers.
    # For strided windows, stage one complete input row in UB and gather
    # there; cap that path so its power-of-two tile stays within 256 values.
    return plan.stride[0] == 1 or input_width <= 256


def _supports_full_plane_2d_fprop_pack(
    plan: _ConvPlan,
    image_shape: tuple[int, int, int, int, int],
    weight_shape: tuple[int, int, int, int, int],
    loss_shape: tuple[int, int, int, int, int],
) -> bool:
    """Whether the measured Ascend full-plane UB gather is applicable."""
    return (
        plan.rank == 2
        and plan.groups == 1
        and image_shape == (8, 32, 1, 32, 32)
        and weight_shape == (64, 32, 1, 3, 3)
        and loss_shape == (8, 64, 1, 32, 32)
    )


def _supports_row_tile_2d_fprop_pack(
    plan: _ConvPlan,
    image_shape: tuple[int, int, int, int, int],
    weight_shape: tuple[int, int, int, int, int],
    loss_shape: tuple[int, int, int, int, int],
) -> bool:
    """Whether the measured Ascend row-tiled UB gather is applicable."""
    return (
        plan.rank == 2
        and plan.groups == 1
        and image_shape == (8, 64, 1, 56, 56)
        and weight_shape == (128, 64, 1, 3, 3)
        and loss_shape == (8, 128, 1, 28, 28)
    )


def _supports_staged_nhwc_2d_fprop(
    plan: _ConvPlan,
    image_shape: tuple[int, int, int, int, int],
    weight_shape: tuple[int, int, int, int, int],
) -> bool:
    """Whether the measured Ascend NHWC-staged fprop path is applicable."""
    channel_pair = (image_shape[1], weight_shape[0])
    return (
        plan.rank == 2
        and plan.groups == 1
        and not plan.filter_reverse
        and plan.stride == (2, 2)
        and plan.pre == (1, 1)
        and plan.post == (1, 1)
        and plan.dilation == (1, 1)
        and image_shape[0] == 1
        and image_shape[2:] == (1, 40, 40)
        and weight_shape[2:] == (1, 3, 3)
        and weight_shape[1] == image_shape[1]
        and channel_pair
        in {
            (128, 256),
            (256, 512),
            (512, 512),
            (768, 768),
        }
    )


def _supports_staged_nhwc_batched_2d_fprop(
    plan: _ConvPlan,
    image_shape: tuple[int, int, int, int, int],
    weight_shape: tuple[int, int, int, int, int],
) -> bool:
    """Whether an exact measured batched NHWC staging path applies."""
    if plan.rank != 2 or plan.groups != 1 or plan.filter_reverse:
        return False
    dilation_case = (
        image_shape == (4, 64, 1, 32, 32)
        and weight_shape == (64, 64, 1, 3, 3)
        and plan.stride == (1, 1)
        and plan.pre == (2, 2)
        and plan.post == (2, 2)
        and plan.dilation == (2, 2)
    )
    asymmetric_case = (
        image_shape == (4, 32, 1, 35, 37)
        and weight_shape == (48, 32, 1, 3, 5)
        and plan.stride == (1, 2)
        and plan.pre == (1, 0)
        and plan.post == (1, 2)
        and plan.dilation == (1, 1)
    )
    return dilation_case or asymmetric_case


def _supports_stem_2d_fprop(
    plan: _ConvPlan,
    image_shape: tuple[int, int, int, int, int],
    weight_shape: tuple[int, int, int, int, int],
    loss_shape: tuple[int, int, int, int, int],
) -> bool:
    """Whether the measured Ascend low-channel stem path applies."""
    return (
        plan.rank == 2
        and plan.groups == 1
        and not plan.filter_reverse
        and plan.stride == (2, 2)
        and plan.pre == (1, 1)
        and plan.post == (1, 1)
        and plan.dilation == (1, 1)
        and image_shape == (1, 3, 1, 640, 640)
        and weight_shape[0] in (16, 32, 64, 96)
        and weight_shape[1:] == (3, 1, 3, 3)
        and loss_shape == (1, weight_shape[0], 1, 320, 320)
    )


def _supports_full_volume_3d_fprop(
    plan: _ConvPlan,
    image_shape: tuple[int, int, int, int, int],
    weight_shape: tuple[int, int, int, int, int],
    loss_shape: tuple[int, int, int, int, int],
) -> bool:
    """Whether an exact measured Ascend 3D full-volume pack applies."""
    if (
        plan.rank != 3
        or plan.groups != 1
        or plan.filter_reverse
        or plan.stride != (1, 1, 1)
        or plan.dilation != (1, 1, 1)
    ):
        return False
    symmetric_case = (
        image_shape == (2, 8, 8, 16, 16)
        and weight_shape == (16, 8, 3, 3, 3)
        and loss_shape == (2, 16, 8, 16, 16)
        and plan.pre == (1, 1, 1)
        and plan.post == (1, 1, 1)
    )
    asymmetric_case = (
        image_shape == (1, 8, 10, 12, 14)
        and weight_shape == (12, 8, 2, 3, 3)
        and loss_shape == (1, 12, 10, 11, 15)
        and plan.pre == (1, 0, 1)
        and plan.post == (0, 1, 2)
    )
    return symmetric_case or asymmetric_case


def _supports_fprop_style_3d_dgrad(
    plan: _ConvPlan,
    image_shape: tuple[int, int, int, int, int],
    loss_shape: tuple[int, int, int, int, int],
    weight_shape: tuple[int, int, int, int, int],
) -> bool:
    """Whether an exact 3D dgrad can use one fprop-style GEMM."""
    if (
        plan.rank != 3
        or plan.groups != 1
        or plan.filter_reverse
        or plan.stride != (1, 1, 1)
        or plan.dilation != (1, 1, 1)
    ):
        return False
    symmetric_case = (
        image_shape == (2, 8, 8, 16, 16)
        and loss_shape == (2, 16, 8, 16, 16)
        and weight_shape == (16, 8, 3, 3, 3)
        and plan.pre == (1, 1, 1)
        and plan.post == (1, 1, 1)
    )
    asymmetric_case = (
        image_shape == (1, 8, 10, 12, 14)
        and loss_shape == (1, 12, 10, 11, 15)
        and weight_shape == (12, 8, 2, 3, 3)
        and plan.pre == (1, 0, 1)
        and plan.post == (0, 1, 2)
    )
    return symmetric_case or asymmetric_case


def _supports_full_plane_2d_dgrad(
    plan: _ConvPlan,
    image_shape: tuple[int, int, int, int, int],
    loss_shape: tuple[int, int, int, int, int],
    weight_shape: tuple[int, int, int, int, int],
) -> bool:
    """Whether the measured Ascend 2D dgrad replay is applicable."""
    return (
        plan.rank == 2
        and plan.groups == 1
        and image_shape == (8, 32, 1, 32, 32)
        and loss_shape == (8, 64, 1, 32, 32)
        and weight_shape == (64, 32, 1, 3, 3)
    )


def _supports_stem_stride2_2d_dgrad(
    plan: _ConvPlan,
    image_shape: tuple[int, int, int, int, int],
    loss_shape: tuple[int, int, int, int, int],
    weight_shape: tuple[int, int, int, int, int],
) -> bool:
    """Whether the measured low-channel Ascend dgrad path applies."""
    c_out = loss_shape[1]
    return (
        plan.rank == 2
        and plan.groups == 1
        and not plan.filter_reverse
        and plan.stride == (2, 2)
        and plan.pre == (1, 1)
        and plan.post == (1, 1)
        and plan.dilation == (1, 1)
        and image_shape == (1, 3, 1, 640, 640)
        and loss_shape == (1, c_out, 1, 320, 320)
        and weight_shape == (c_out, 3, 1, 3, 3)
        and c_out in (16, 32, 64, 96)
    )


def _supports_deep_stride2_2d_dgrad(
    plan: _ConvPlan,
    image_shape: tuple[int, int, int, int, int],
    loss_shape: tuple[int, int, int, int, int],
    weight_shape: tuple[int, int, int, int, int],
) -> bool:
    """Whether the measured high-channel Ascend dgrad path applies."""
    channel_pair = (image_shape[1], loss_shape[1])
    return (
        plan.rank == 2
        and plan.groups == 1
        and not plan.filter_reverse
        and plan.stride == (2, 2)
        and plan.pre == (1, 1)
        and plan.post == (1, 1)
        and plan.dilation == (1, 1)
        and image_shape[0] == 1
        and image_shape[2:] == (1, 40, 40)
        and loss_shape[0] == 1
        and loss_shape[2:] == (1, 20, 20)
        and weight_shape[1] == image_shape[1]
        and weight_shape[0] == loss_shape[1]
        and weight_shape[2:] == (1, 3, 3)
        and channel_pair
        in {
            (128, 256),
            (256, 512),
            (512, 512),
            (768, 768),
        }
    )


def _supports_stride2_2d_dgrad(
    plan: _ConvPlan,
    image_shape: tuple[int, int, int, int, int],
    loss_shape: tuple[int, int, int, int, int],
    weight_shape: tuple[int, int, int, int, int],
) -> bool:
    """Whether the measured Ascend stride-two row replay is applicable."""
    return (
        plan.rank == 2
        and plan.groups == 1
        and plan.stride == (2, 2)
        and plan.pre == (1, 1)
        and plan.post == (1, 1)
        and plan.dilation == (1, 1)
        and image_shape == (8, 64, 1, 56, 56)
        and loss_shape == (8, 128, 1, 28, 28)
        and weight_shape == (128, 64, 1, 3, 3)
    )


def _supports_full_volume_3d_wgrad(
    plan: _ConvPlan,
    image_shape: tuple[int, int, int, int, int],
    loss_shape: tuple[int, int, int, int, int],
    weight_shape: tuple[int, int, int, int, int],
) -> bool:
    """Whether an exact measured Ascend 3D split-K wgrad applies."""
    if (
        plan.rank != 3
        or plan.groups != 1
        or plan.filter_reverse
        or plan.stride != (1, 1, 1)
        or plan.dilation != (1, 1, 1)
    ):
        return False
    symmetric_case = (
        image_shape == (2, 8, 8, 16, 16)
        and loss_shape == (2, 16, 8, 16, 16)
        and weight_shape == (16, 8, 3, 3, 3)
        and plan.pre == (1, 1, 1)
        and plan.post == (1, 1, 1)
    )
    asymmetric_case = (
        image_shape == (1, 8, 10, 12, 14)
        and loss_shape == (1, 12, 10, 11, 15)
        and weight_shape == (12, 8, 2, 3, 3)
        and plan.pre == (1, 0, 1)
        and plan.post == (0, 1, 2)
    )
    return symmetric_case or asymmetric_case


def _supports_full_plane_2d_wgrad(
    plan: _ConvPlan,
    image_shape: tuple[int, int, int, int, int],
    loss_shape: tuple[int, int, int, int, int],
    weight_shape: tuple[int, int, int, int, int],
) -> bool:
    """Whether the measured Ascend 2D wgrad replay is applicable."""
    return (
        plan.rank == 2
        and plan.groups == 1
        and image_shape == (8, 32, 1, 32, 32)
        and loss_shape == (8, 64, 1, 32, 32)
        and weight_shape == (64, 32, 1, 3, 3)
    )


def _supports_row_tile_2d_wgrad(
    plan: _ConvPlan,
    image_shape: tuple[int, int, int, int, int],
    loss_shape: tuple[int, int, int, int, int],
    weight_shape: tuple[int, int, int, int, int],
) -> bool:
    """Whether the measured Ascend row-tiled wgrad replay is applicable."""
    return (
        plan.rank == 2
        and plan.groups == 1
        and plan.stride == (2, 2)
        and plan.pre == (1, 1)
        and plan.post == (1, 1)
        and plan.dilation == (1, 1)
        and image_shape == (8, 64, 1, 56, 56)
        and loss_shape == (8, 128, 1, 28, 28)
        and weight_shape == (128, 64, 1, 3, 3)
    )


def _supports_deep_stride2_2d_wgrad(
    plan: _ConvPlan,
    image_shape: tuple[int, int, int, int, int],
    loss_shape: tuple[int, int, int, int, int],
    weight_shape: tuple[int, int, int, int, int],
) -> bool:
    """Whether the measured high-channel Ascend wgrad path applies."""
    channel_pair = (image_shape[1], loss_shape[1])
    return (
        plan.rank == 2
        and plan.groups == 1
        and not plan.filter_reverse
        and plan.stride == (2, 2)
        and plan.pre == (1, 1)
        and plan.post == (1, 1)
        and plan.dilation == (1, 1)
        and image_shape[0] == 1
        and image_shape[2:] == (1, 40, 40)
        and loss_shape[0] == 1
        and loss_shape[2:] == (1, 20, 20)
        and weight_shape[1] == image_shape[1]
        and weight_shape[0] == loss_shape[1]
        and weight_shape[2:] == (1, 3, 3)
        and channel_pair
        in {
            (128, 256),
            (256, 512),
            (512, 512),
            (768, 768),
        }
    )


def _supports_stem_stride2_2d_wgrad(
    plan: _ConvPlan,
    image_shape: tuple[int, int, int, int, int],
    loss_shape: tuple[int, int, int, int, int],
    weight_shape: tuple[int, int, int, int, int],
) -> bool:
    """Whether the measured low-channel split-K wgrad path applies."""
    c_out = loss_shape[1]
    return (
        plan.rank == 2
        and plan.groups == 1
        and not plan.filter_reverse
        and plan.stride == (2, 2)
        and plan.pre == (1, 1)
        and plan.post == (1, 1)
        and plan.dilation == (1, 1)
        and image_shape == (1, 3, 1, 640, 640)
        and loss_shape == (1, c_out, 1, 320, 320)
        and weight_shape == (c_out, 3, 1, 3, 3)
        and c_out in (16, 32, 64, 96)
    )


def _supports_fast_dgrad_1d_weight_pack(
    plan: _ConvPlan,
    cin_per_group: int,
    kernel_width: int,
) -> bool:
    return plan.rank == 1 and cin_per_group * kernel_width <= 512


def _supports_packed_pointwise_dgrad(
    plan: _ConvPlan,
    loss_shape: tuple[int, int, int, int, int],
    weight_shape: tuple[int, int, int, int, int],
) -> bool:
    """Whether the measured Ascend packed-weight replay is applicable."""
    n, _, od, oh, ow = loss_shape
    c_out, cin_per_group, _, _, _ = weight_shape
    return (
        _is_pointwise_conv(plan, weight_shape)
        and plan.groups == 1
        and n == 8
        and cin_per_group == 64
        and c_out == 128
        and od * oh * ow == 784
    )


def _wgrad_launch_config(
    *,
    n: int,
    c_out: int,
    cin_per_group: int,
    cout_per_group: int,
    kd: int,
    kh: int,
    kw: int,
    od: int,
    oh: int,
    ow: int,
    groups: int,
    fp32: bool,
) -> tuple[int, int, int, int, int]:
    block_co = 64 if cout_per_group >= 64 else 32
    kernel_volume = kd * kh * kw
    block_r = 64
    columns = cin_per_group * kernel_volume
    block_n = 64 if columns >= 64 else 32
    channel_programs = (
        groups
        * _ceil_div(cout_per_group, block_co)
        * _ceil_div(columns, block_n)
    )
    reduction = n * od * oh * ow
    target_splits = min(
        64,
        _ceil_div(128, max(channel_programs, 1)),
        max(_ceil_div(reduction, block_r), 1),
    )
    num_splits = 1
    while num_splits < target_splits:
        num_splits *= 2
    return block_co, block_n, block_r, num_splits, kernel_volume


def _launch(
    plan: _ConvPlan,
    inputs: Sequence[Any],
    output: torch.Tensor,
    workspace: Optional[Any] = None,
) -> torch.Tensor:
    rank = plan.rank
    if plan.op_type == "conv_fprop":
        image, weight = inputs[:2]
        loss = output
    elif plan.op_type == "conv_dgrad":
        loss, weight = inputs[:2]
        image = output
    else:
        image, loss = inputs[:2]
        weight = output

    image_shape, image_strides = _normalized_tensor(image, rank)
    loss_shape, loss_strides = _normalized_tensor(loss, rank)
    weight_shape, weight_strides = _normalized_weight(weight, rank)
    n, c_in, xd, xh, xw = image_shape
    _, c_out, od, oh, ow = loss_shape
    _, cin_per_group, kd, kh, kw = weight_shape
    cout_per_group = c_out // plan.groups
    stride_d, stride_h, stride_w = _normalized_spatial(plan.stride, 1)
    pad_d, pad_h, pad_w = _normalized_spatial(plan.pre, 0)
    dil_d, dil_h, dil_w = _normalized_spatial(plan.dilation, 1)

    common = {
        "N": n,
        "C_IN": c_in,
        "C_OUT": c_out,
        "XD": xd,
        "XH": xh,
        "XW": xw,
        "OD": od,
        "OH": oh,
        "OW": ow,
        "KD": kd,
        "KH": kh,
        "KW": kw,
        "CIN_PER_GROUP": cin_per_group,
        "COUT_PER_GROUP": cout_per_group,
        "STRIDE_D": stride_d,
        "STRIDE_H": stride_h,
        "STRIDE_W": stride_w,
        "PAD_D": pad_d,
        "PAD_H": pad_h,
        "PAD_W": pad_w,
        "DIL_D": dil_d,
        "DIL_H": dil_h,
        "DIL_W": dil_w,
        "FILTER_REVERSE": plan.filter_reverse,
        "FP32_INPUT": image.dtype == torch.float32,
    }
    with torch_device_fn.device(image.device):
        if plan.op_type == "conv_fprop":
            output_spatial = od * oh * ow
            kernel_volume = kd * kh * kw
            reduction = cin_per_group * kernel_volume
            if (
                _is_pointwise_conv(plan, weight_shape)
                and image.is_contiguous()
                and weight.is_contiguous()
                and output.is_contiguous()
            ):
                block_oc = 128
                block_spatial = 128
                block_k = 64
                gemm_grid = (
                    n
                    * plan.groups
                    * _ceil_div(cout_per_group, block_oc)
                    * _ceil_div(output_spatial, block_spatial),
                )
                _conv_dgrad_broadcast_matmul_kernel[gemm_grid](
                    weight,
                    image,
                    output,
                    M=cout_per_group,
                    N=output_spatial,
                    K=cin_per_group,
                    GROUPS=plan.groups,
                    KERNEL_VOLUME=1,
                    BLOCK_M=block_oc,
                    BLOCK_N=block_spatial,
                    BLOCK_K=block_k,
                    GROUP_M=4,
                    TF32=image.dtype == torch.float32,
                    num_warps=4,
                    num_stages=2,
                    sync_solver=True,
                )
                return output
            if (
                _supports_staged_nhwc_2d_fprop(
                    plan,
                    image_shape,
                    weight_shape,
                )
                and image.is_contiguous()
                and weight.is_contiguous()
                and output.is_contiguous()
            ):
                if not isinstance(workspace, tuple) or len(workspace) != 2:
                    raise RuntimeError(
                        "Ascend staged-NHWC convolution workspaces are missing"
                    )
                packed_image, column_workspace = workspace
                image_pack_block_spatial = 128
                image_pack_block_channels = 128
                _conv_fprop_pack_nchw_to_nhwc_kernel[
                    (
                        _ceil_div(
                            xh * xw,
                            image_pack_block_spatial,
                        ),
                        _ceil_div(
                            c_in,
                            image_pack_block_channels,
                        ),
                    )
                ](
                    image,
                    packed_image,
                    SPATIAL=xh * xw,
                    CHANNELS=c_in,
                    BLOCK_SPATIAL=image_pack_block_spatial,
                    BLOCK_CHANNELS=image_pack_block_channels,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                column_block_spatial = 32
                column_block_channels = 128 if c_in == 128 else 256
                _conv_fprop_pack_nhwc_im2col_kernel[
                    (
                        9,
                        _ceil_div(c_in, column_block_channels),
                        _ceil_div(
                            output_spatial,
                            column_block_spatial,
                        ),
                    )
                ](
                    packed_image,
                    column_workspace,
                    C_IN=c_in,
                    XH=xh,
                    XW=xw,
                    OH=oh,
                    OW=ow,
                    BLOCK_SPATIAL=column_block_spatial,
                    BLOCK_CHANNELS=column_block_channels,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                block_oc = 128
                block_spatial = 128
                block_k = 128
                gemm_grid = (
                    _ceil_div(cout_per_group, block_oc)
                    * _ceil_div(output_spatial, block_spatial),
                )
                _conv_dgrad_broadcast_matmul_kernel[gemm_grid](
                    weight,
                    column_workspace,
                    output,
                    M=cout_per_group,
                    N=output_spatial,
                    K=reduction,
                    GROUPS=1,
                    KERNEL_VOLUME=1,
                    BLOCK_M=block_oc,
                    BLOCK_N=block_spatial,
                    BLOCK_K=block_k,
                    GROUP_M=4,
                    TF32=image.dtype == torch.float32,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                return output
            if (
                _supports_staged_nhwc_batched_2d_fprop(
                    plan,
                    image_shape,
                    weight_shape,
                )
                and image.is_contiguous()
                and weight.is_contiguous()
                and output.is_contiguous()
            ):
                if not isinstance(workspace, tuple) or len(workspace) != 2:
                    raise RuntimeError(
                        "Ascend batched staged-NHWC workspaces are missing"
                    )
                packed_image, column_workspace = workspace
                if c_in == 32:
                    full_plane_block_channels = (
                        2 if image.dtype == torch.float32 else 4
                    )
                    full_plane_block_spatial = 1024
                    _conv_fprop_im2col_2d_gather_kernel[
                        (
                            _ceil_div(
                                c_in,
                                full_plane_block_channels,
                            ),
                            n,
                        )
                    ](
                        image,
                        column_workspace,
                        GROUPS=1,
                        CIN_PER_GROUP=c_in,
                        XH=xh,
                        XW=xw,
                        OH=oh,
                        OW=ow,
                        KH=kh,
                        KW=kw,
                        STRIDE_H=stride_h,
                        STRIDE_W=stride_w,
                        PAD_H=pad_h,
                        PAD_W=pad_w,
                        DIL_H=dil_h,
                        DIL_W=dil_w,
                        IMAGE_STRIDE_N=image_strides[0],
                        IMAGE_STRIDE_C=image_strides[1],
                        FILTER_REVERSE=plan.filter_reverse,
                        LOAD_X=triton.next_power_of_2(xh * xw),
                        BLOCK_CI=full_plane_block_channels,
                        BLOCK_SPATIAL=full_plane_block_spatial,
                        num_warps=4,
                        num_stages=1,
                        multibuffer=False,
                        sync_solver=True,
                    )
                    block_k = 128
                else:
                    image_pack_block_spatial = 128
                    image_pack_block_channels = 64
                    column_block_spatial = (
                        128 if image.dtype == torch.float32 else 256
                    )
                    column_block_channels = 64
                    column_tiles_per_program = (
                        8 if image.dtype == torch.float32 else 4
                    )
                    block_k = 64
                    _conv_fprop_pack_nchw_to_nhwc_batched_kernel[
                        (
                            _ceil_div(
                                xh * xw,
                                image_pack_block_spatial,
                            ),
                            _ceil_div(
                                c_in,
                                image_pack_block_channels,
                            ),
                            n,
                        )
                    ](
                        image,
                        packed_image,
                        SPATIAL=xh * xw,
                        CHANNELS=c_in,
                        BLOCK_SPATIAL=image_pack_block_spatial,
                        BLOCK_CHANNELS=image_pack_block_channels,
                        num_warps=4,
                        num_stages=1,
                        multibuffer=False,
                        sync_solver=True,
                    )
                    _conv_fprop_pack_nhwc_im2col_2d_persistent_kernel[
                        (
                            n * kernel_volume,
                            _ceil_div(
                                c_in,
                                column_block_channels,
                            ),
                            _ceil_div(
                                output_spatial,
                                column_block_spatial
                                * column_tiles_per_program,
                            ),
                        )
                    ](
                        packed_image,
                        column_workspace,
                        C_IN=c_in,
                        XH=xh,
                        XW=xw,
                        OH=oh,
                        OW=ow,
                        KH=kh,
                        KW=kw,
                        STRIDE_H=stride_h,
                        STRIDE_W=stride_w,
                        PAD_H=pad_h,
                        PAD_W=pad_w,
                        DIL_H=dil_h,
                        DIL_W=dil_w,
                        FILTER_REVERSE=plan.filter_reverse,
                        TILES_PER_PROGRAM=column_tiles_per_program,
                        BLOCK_SPATIAL=column_block_spatial,
                        BLOCK_CHANNELS=column_block_channels,
                        num_warps=4,
                        num_stages=1,
                        multibuffer=False,
                        sync_solver=True,
                    )
                block_oc = 64
                block_spatial = 256
                gemm_grid = (
                    n
                    * _ceil_div(cout_per_group, block_oc)
                    * _ceil_div(output_spatial, block_spatial),
                )
                _conv_dgrad_broadcast_matmul_kernel[gemm_grid](
                    weight,
                    column_workspace,
                    output,
                    M=cout_per_group,
                    N=output_spatial,
                    K=reduction,
                    GROUPS=1,
                    KERNEL_VOLUME=1,
                    BLOCK_M=block_oc,
                    BLOCK_N=block_spatial,
                    BLOCK_K=block_k,
                    GROUP_M=4,
                    TF32=image.dtype == torch.float32,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                return output
            if workspace is None:
                raise RuntimeError(
                    "Ascend convolution fprop im2col workspace is missing"
                )
            total_columns = n * plan.groups * reduction * output_spatial
            if (
                _supports_stem_2d_fprop(
                    plan,
                    image_shape,
                    weight_shape,
                    loss_shape,
                )
                and image.is_contiguous()
                and weight.is_contiguous()
                and output.is_contiguous()
            ):
                rows_per_program = 16
                _conv_fprop_im2col_2d_stem_row_reuse_kernel[
                    (
                        cin_per_group * kh,
                        _ceil_div(oh, rows_per_program),
                        n,
                    )
                ](
                    image,
                    workspace,
                    CIN_PER_GROUP=cin_per_group,
                    XH=xh,
                    XW=xw,
                    OH=oh,
                    OW=ow,
                    KH=kh,
                    KW=kw,
                    STRIDE_H=stride_h,
                    STRIDE_W=stride_w,
                    PAD_H=pad_h,
                    PAD_W=pad_w,
                    DIL_H=dil_h,
                    DIL_W=dil_w,
                    IMAGE_STRIDE_N=image_strides[0],
                    IMAGE_STRIDE_C=image_strides[1],
                    FILTER_REVERSE=plan.filter_reverse,
                    ROWS_PER_PROGRAM=rows_per_program,
                    LOAD_W=triton.next_power_of_2(xw),
                    BLOCK_W=triton.next_power_of_2(ow),
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                fp32 = image.dtype == torch.float32
                if fp32:
                    block_oc = 16 if c_out == 16 else 32
                    block_spatial = 1024
                elif c_out == 16:
                    block_oc = 16
                    block_spatial = 2048
                elif c_out == 32:
                    block_oc = 32
                    block_spatial = 1024
                else:
                    block_oc = 16
                    block_spatial = 2048
                if c_out <= 32:
                    group_oc = 1
                elif c_out == 64:
                    group_oc = 4
                else:
                    group_oc = 8
                gemm_grid = (
                    n
                    * _ceil_div(c_out, block_oc)
                    * _ceil_div(output_spatial, block_spatial),
                )
                _conv_dgrad_broadcast_matmul_kernel[gemm_grid](
                    weight,
                    workspace,
                    output,
                    M=c_out,
                    N=output_spatial,
                    K=reduction,
                    GROUPS=1,
                    KERNEL_VOLUME=1,
                    BLOCK_M=block_oc,
                    BLOCK_N=block_spatial,
                    BLOCK_K=32,
                    GROUP_M=group_oc,
                    TF32=fp32,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                return output
            if (
                _supports_full_volume_3d_fprop(
                    plan,
                    image_shape,
                    weight_shape,
                    loss_shape,
                )
                and image.is_contiguous()
                and weight.is_contiguous()
                and output.is_contiguous()
            ):
                pack_block_ci = 1
                pack_block_spatial = 1024
                _conv_fprop_im2col_3d_full_volume_kernel[
                    (
                        _ceil_div(
                            cin_per_group,
                            pack_block_ci,
                        ),
                        n,
                    )
                ](
                    image,
                    workspace,
                    CIN_PER_GROUP=cin_per_group,
                    XD=xd,
                    XH=xh,
                    XW=xw,
                    OD=od,
                    OH=oh,
                    OW=ow,
                    KD=kd,
                    KH=kh,
                    KW=kw,
                    STRIDE_D=stride_d,
                    STRIDE_H=stride_h,
                    STRIDE_W=stride_w,
                    PAD_D=pad_d,
                    PAD_H=pad_h,
                    PAD_W=pad_w,
                    DIL_D=dil_d,
                    DIL_H=dil_h,
                    DIL_W=dil_w,
                    IMAGE_STRIDE_N=image_strides[0],
                    IMAGE_STRIDE_C=image_strides[1],
                    FILTER_REVERSE=plan.filter_reverse,
                    LOAD_X=triton.next_power_of_2(xd * xh * xw),
                    BLOCK_CI=pack_block_ci,
                    BLOCK_SPATIAL=pack_block_spatial,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                block_oc = 16
                block_spatial = 512
                block_k = 32
                gemm_grid = (
                    n
                    * _ceil_div(cout_per_group, block_oc)
                    * _ceil_div(output_spatial, block_spatial),
                )
                _conv_dgrad_broadcast_matmul_kernel[gemm_grid](
                    weight,
                    workspace,
                    output,
                    M=cout_per_group,
                    N=output_spatial,
                    K=reduction,
                    GROUPS=1,
                    KERNEL_VOLUME=1,
                    BLOCK_M=block_oc,
                    BLOCK_N=block_spatial,
                    BLOCK_K=block_k,
                    GROUP_M=1,
                    TF32=image.dtype == torch.float32,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                return output
            if (
                _supports_full_plane_2d_fprop_pack(
                    plan,
                    image_shape,
                    weight_shape,
                    loss_shape,
                )
                and image.is_contiguous()
                and weight.is_contiguous()
                and output.is_contiguous()
            ):
                pack_block_ci = 8
                pack_block_spatial = 512
                _conv_fprop_im2col_2d_gather_kernel[
                    (
                        _ceil_div(
                            cin_per_group,
                            pack_block_ci,
                        ),
                        n * plan.groups,
                    )
                ](
                    image,
                    workspace,
                    GROUPS=plan.groups,
                    CIN_PER_GROUP=cin_per_group,
                    XH=xh,
                    XW=xw,
                    OH=oh,
                    OW=ow,
                    KH=kh,
                    KW=kw,
                    STRIDE_H=stride_h,
                    STRIDE_W=stride_w,
                    PAD_H=pad_h,
                    PAD_W=pad_w,
                    DIL_H=dil_h,
                    DIL_W=dil_w,
                    IMAGE_STRIDE_N=image_strides[0],
                    IMAGE_STRIDE_C=image_strides[1],
                    FILTER_REVERSE=plan.filter_reverse,
                    LOAD_X=triton.next_power_of_2(xh * xw),
                    BLOCK_CI=pack_block_ci,
                    BLOCK_SPATIAL=pack_block_spatial,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                block_oc = 64
                block_spatial = 512
                block_k = 32
                gemm_grid = (
                    n
                    * plan.groups
                    * _ceil_div(cout_per_group, block_oc)
                    * _ceil_div(output_spatial, block_spatial),
                )
                _conv_dgrad_broadcast_matmul_kernel[gemm_grid](
                    weight,
                    workspace,
                    output,
                    M=cout_per_group,
                    N=output_spatial,
                    K=reduction,
                    GROUPS=plan.groups,
                    KERNEL_VOLUME=1,
                    BLOCK_M=block_oc,
                    BLOCK_N=block_spatial,
                    BLOCK_K=block_k,
                    GROUP_M=1,
                    TF32=image.dtype == torch.float32,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                return output
            if (
                _supports_row_tile_2d_fprop_pack(
                    plan,
                    image_shape,
                    weight_shape,
                    loss_shape,
                )
                and image.is_contiguous()
                and weight.is_contiguous()
                and output.is_contiguous()
            ):
                if image.dtype == torch.bfloat16:
                    pack_block_ci = 8
                elif image.dtype == torch.float16:
                    pack_block_ci = 16
                else:
                    pack_block_ci = 4
                pack_block_spatial = 512
                _conv_fprop_im2col_2d_row_tile_gather_kernel[
                    (
                        _ceil_div(
                            cin_per_group,
                            pack_block_ci,
                        ),
                        n * plan.groups,
                    )
                ](
                    image,
                    workspace,
                    GROUPS=plan.groups,
                    CIN_PER_GROUP=cin_per_group,
                    XH=xh,
                    XW=xw,
                    OH=oh,
                    OW=ow,
                    KH=kh,
                    KW=kw,
                    STRIDE_H=stride_h,
                    STRIDE_W=stride_w,
                    PAD_H=pad_h,
                    PAD_W=pad_w,
                    DIL_H=dil_h,
                    DIL_W=dil_w,
                    IMAGE_STRIDE_N=image_strides[0],
                    IMAGE_STRIDE_C=image_strides[1],
                    FILTER_REVERSE=plan.filter_reverse,
                    ROW_TILES=2,
                    ROWS_PER_TILE=14,
                    LOAD_X=2048,
                    BLOCK_CI=pack_block_ci,
                    BLOCK_SPATIAL=pack_block_spatial,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                block_oc = 128
                block_spatial = 128 if image.dtype == torch.float32 else 256
                block_k = 64
                gemm_grid = (
                    n
                    * plan.groups
                    * _ceil_div(cout_per_group, block_oc)
                    * _ceil_div(output_spatial, block_spatial),
                )
                _conv_dgrad_broadcast_matmul_kernel[gemm_grid](
                    weight,
                    workspace,
                    output,
                    M=cout_per_group,
                    N=output_spatial,
                    K=reduction,
                    GROUPS=plan.groups,
                    KERNEL_VOLUME=1,
                    BLOCK_M=block_oc,
                    BLOCK_N=block_spatial,
                    BLOCK_K=block_k,
                    GROUP_M=1,
                    TF32=image.dtype == torch.float32,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                return output
            if (
                plan.rank == 1
                and _supports_fast_conv_1d_pack(plan, xw)
                and image.is_contiguous()
                and weight.is_contiguous()
                and output.is_contiguous()
            ):
                if stride_w == 1:
                    pack_block = 256
                    pack_block_ci = 8
                    pack_kernel = _conv_fprop_im2col_1d_block_kernel
                    pack_kwargs = {}
                else:
                    pack_block = 128
                    pack_block_ci = 16
                    pack_kernel = _conv_fprop_im2col_1d_gather_kernel
                    pack_kwargs = {
                        "STRIDE_W": stride_w,
                        "LOAD_W": triton.next_power_of_2(xw),
                    }
                pack_kernel[
                    (
                        _ceil_div(ow, pack_block),
                        _ceil_div(cin_per_group, pack_block_ci),
                        n * plan.groups,
                    )
                ](
                    image,
                    workspace,
                    GROUPS=plan.groups,
                    CIN_PER_GROUP=cin_per_group,
                    XW=xw,
                    OW=ow,
                    KW=kw,
                    REDUCTION=reduction,
                    PAD_W=pad_w,
                    DIL_W=dil_w,
                    IMAGE_STRIDE_N=image_strides[0],
                    IMAGE_STRIDE_C=image_strides[1],
                    IMAGE_STRIDE_W=image_strides[4],
                    FILTER_REVERSE=plan.filter_reverse,
                    BLOCK_W=pack_block,
                    BLOCK_CI=pack_block_ci,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                    **pack_kwargs,
                )
            else:
                pack_block = 256
                pack_block_k = 16
                _conv_fprop_im2col_kernel[
                    (
                        _ceil_div(output_spatial, pack_block),
                        _ceil_div(reduction, pack_block_k),
                        n * plan.groups,
                    )
                ](
                    image,
                    workspace,
                    TOTAL=total_columns,
                    GROUPS=plan.groups,
                    CIN_PER_GROUP=cin_per_group,
                    XD=xd,
                    XH=xh,
                    XW=xw,
                    OD=od,
                    OH=oh,
                    OW=ow,
                    KD=kd,
                    KH=kh,
                    KW=kw,
                    STRIDE_D=stride_d,
                    STRIDE_H=stride_h,
                    STRIDE_W=stride_w,
                    PAD_D=pad_d,
                    PAD_H=pad_h,
                    PAD_W=pad_w,
                    DIL_D=dil_d,
                    DIL_H=dil_h,
                    DIL_W=dil_w,
                    IMAGE_STRIDE_N=image_strides[0],
                    IMAGE_STRIDE_C=image_strides[1],
                    IMAGE_STRIDE_D=image_strides[2],
                    IMAGE_STRIDE_H=image_strides[3],
                    IMAGE_STRIDE_W=image_strides[4],
                    FILTER_REVERSE=plan.filter_reverse,
                    BLOCK_K_PACK=pack_block_k,
                    BLOCK=pack_block,
                    num_warps=4,
                    num_stages=1,
                )
            fp32 = image.dtype == torch.float32
            if cout_per_group <= 32 or output_spatial <= 32:
                block_oc, block_m, block_k, group_oc = (
                    32,
                    32,
                    32,
                    1,
                )
            elif fp32:
                block_oc, block_m, block_k, group_oc = (
                    256,
                    128,
                    128,
                    4,
                )
            else:
                block_oc, block_m, block_k, group_oc = (
                    128,
                    256,
                    128,
                    4,
                )
            gemm_grid = (
                n
                * plan.groups
                * _ceil_div(cout_per_group, block_oc)
                * _ceil_div(output_spatial, block_m),
            )
            _conv_dgrad_broadcast_matmul_kernel[gemm_grid](
                weight,
                workspace,
                output,
                M=cout_per_group,
                N=output_spatial,
                K=reduction,
                GROUPS=plan.groups,
                KERNEL_VOLUME=1,
                BLOCK_M=block_oc,
                BLOCK_N=block_m,
                BLOCK_K=block_k,
                GROUP_M=group_oc,
                TF32=fp32,
                num_warps=(
                    4 if cout_per_group <= 32 or output_spatial <= 32 else 8
                ),
                num_stages=2,
            )
        elif plan.op_type == "conv_dgrad":
            if (
                _is_pointwise_conv(plan, weight_shape)
                and loss.is_contiguous()
                and weight.is_contiguous()
                and output.is_contiguous()
            ):
                output_spatial = od * oh * ow
                fp32 = loss.dtype == torch.float32
                if _supports_packed_pointwise_dgrad(
                    plan,
                    loss_shape,
                    weight_shape,
                ) and isinstance(workspace, torch.Tensor):
                    pack_block_ci = triton.next_power_of_2(cin_per_group)
                    pack_block_co = triton.next_power_of_2(cout_per_group)
                    _conv_dgrad_pack_pointwise_weight_kernel[(plan.groups,)](
                        weight,
                        workspace,
                        CIN_PER_GROUP=cin_per_group,
                        COUT_PER_GROUP=cout_per_group,
                        WEIGHT_STRIDE_O=weight_strides[0],
                        WEIGHT_STRIDE_I=weight_strides[1],
                        BLOCK_CI=pack_block_ci,
                        BLOCK_CO=pack_block_co,
                        num_warps=4,
                        num_stages=1,
                        multibuffer=False,
                        sync_solver=True,
                    )
                    if fp32:
                        block_ci = 64
                        block_spatial = 128
                        block_co = 64
                    else:
                        block_ci = 64
                        block_spatial = 512
                        block_co = 128
                    grid = (
                        n
                        * plan.groups
                        * _ceil_div(cin_per_group, block_ci)
                        * _ceil_div(
                            output_spatial,
                            block_spatial,
                        ),
                    )
                    _conv_dgrad_broadcast_matmul_kernel[grid](
                        workspace,
                        loss,
                        output,
                        M=cin_per_group,
                        N=output_spatial,
                        K=cout_per_group,
                        GROUPS=plan.groups,
                        KERNEL_VOLUME=1,
                        BLOCK_M=block_ci,
                        BLOCK_N=block_spatial,
                        BLOCK_K=block_co,
                        GROUP_M=1,
                        TF32=fp32,
                        num_warps=4,
                        num_stages=1,
                        multibuffer=False,
                        sync_solver=True,
                    )
                    return output
                block_ci = 32
                block_spatial = 1024
                block_co = 64
                grid = (
                    n
                    * plan.groups
                    * _ceil_div(cin_per_group, block_ci)
                    * _ceil_div(output_spatial, block_spatial),
                )
                _conv_dgrad_pointwise_kernel[grid](
                    weight,
                    loss,
                    output,
                    M=cin_per_group,
                    N=output_spatial,
                    K=cout_per_group,
                    GROUPS=plan.groups,
                    WEIGHT_STRIDE_O=weight_strides[0],
                    WEIGHT_STRIDE_I=weight_strides[1],
                    BLOCK_M=block_ci,
                    BLOCK_N=block_spatial,
                    BLOCK_K=block_co,
                    GROUP_M=4,
                    TF32=fp32,
                    num_warps=4,
                    num_stages=1 if fp32 else 2,
                    multibuffer=False if fp32 else True,
                    sync_solver=True,
                )
                return output
            if not loss.is_contiguous() or not weight.is_contiguous():
                block_m = 64 if plan.rank == 3 else 128
                block_n = 32 if cin_per_group <= 32 else 64
                block_k = 32
                grid = (
                    plan.groups
                    * _ceil_div(n * xd * xh * xw, block_m)
                    * _ceil_div(cin_per_group, block_n),
                )
                _conv_dgrad_kernel[grid](
                    loss,
                    weight,
                    output,
                    **common,
                    LOSS_STRIDE_N=loss_strides[0],
                    LOSS_STRIDE_C=loss_strides[1],
                    LOSS_STRIDE_D=loss_strides[2],
                    LOSS_STRIDE_H=loss_strides[3],
                    LOSS_STRIDE_W=loss_strides[4],
                    WEIGHT_STRIDE_O=weight_strides[0],
                    WEIGHT_STRIDE_I=weight_strides[1],
                    WEIGHT_STRIDE_D=weight_strides[2],
                    WEIGHT_STRIDE_H=weight_strides[3],
                    WEIGHT_STRIDE_W=weight_strides[4],
                    OUTPUT_STRIDE_N=image_strides[0],
                    OUTPUT_STRIDE_C=image_strides[1],
                    OUTPUT_STRIDE_D=image_strides[2],
                    OUTPUT_STRIDE_H=image_strides[3],
                    OUTPUT_STRIDE_W=image_strides[4],
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    GROUP_M=4,
                    num_warps=4,
                    num_stages=1,
                )
                return output
            if not isinstance(workspace, tuple) or len(workspace) != 2:
                raise RuntimeError(
                    "Ascend convolution dgrad workspaces are missing"
                )
            packed_weight, partial_workspace = workspace
            output_spatial = od * oh * ow
            kernel_volume = kd * kh * kw
            matrix_count = n * plan.groups * kernel_volume
            if (
                _supports_fprop_style_3d_dgrad(
                    plan,
                    image_shape,
                    loss_shape,
                    weight_shape,
                )
                and loss.is_contiguous()
                and weight.is_contiguous()
                and image.is_contiguous()
            ):
                reduction = c_out * kernel_volume
                input_spatial = xd * xh * xw
                weight_block = 256
                _conv_dgrad_pack_transposed_3d_weight_kernel[
                    (
                        _ceil_div(
                            c_in * reduction,
                            weight_block,
                        ),
                    )
                ](
                    weight,
                    packed_weight,
                    C_IN=c_in,
                    C_OUT=c_out,
                    KD=kd,
                    KH=kh,
                    KW=kw,
                    WEIGHT_STRIDE_O=weight_strides[0],
                    WEIGHT_STRIDE_I=weight_strides[1],
                    WEIGHT_STRIDE_D=weight_strides[2],
                    WEIGHT_STRIDE_H=weight_strides[3],
                    WEIGHT_STRIDE_W=weight_strides[4],
                    BLOCK=weight_block,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                column_block_co = 1
                column_block_spatial = 1024
                _conv_fprop_im2col_3d_full_volume_kernel[
                    (
                        _ceil_div(c_out, column_block_co),
                        n,
                    )
                ](
                    loss,
                    partial_workspace,
                    CIN_PER_GROUP=c_out,
                    XD=od,
                    XH=oh,
                    XW=ow,
                    OD=xd,
                    OH=xh,
                    OW=xw,
                    KD=kd,
                    KH=kh,
                    KW=kw,
                    STRIDE_D=1,
                    STRIDE_H=1,
                    STRIDE_W=1,
                    PAD_D=kd - 1 - pad_d,
                    PAD_H=kh - 1 - pad_h,
                    PAD_W=kw - 1 - pad_w,
                    DIL_D=1,
                    DIL_H=1,
                    DIL_W=1,
                    IMAGE_STRIDE_N=loss_strides[0],
                    IMAGE_STRIDE_C=loss_strides[1],
                    FILTER_REVERSE=False,
                    LOAD_X=triton.next_power_of_2(output_spatial),
                    BLOCK_CI=column_block_co,
                    BLOCK_SPATIAL=column_block_spatial,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                block_ci = 16
                block_spatial = 512
                block_k = 64
                gemm_grid = (
                    n
                    * _ceil_div(c_in, block_ci)
                    * _ceil_div(input_spatial, block_spatial),
                )
                _conv_dgrad_broadcast_matmul_kernel[gemm_grid](
                    packed_weight,
                    partial_workspace,
                    image,
                    M=c_in,
                    N=input_spatial,
                    K=reduction,
                    GROUPS=1,
                    KERNEL_VOLUME=1,
                    BLOCK_M=block_ci,
                    BLOCK_N=block_spatial,
                    BLOCK_K=block_k,
                    GROUP_M=1,
                    TF32=loss.dtype == torch.float32,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                return output
            if (
                _supports_stem_stride2_2d_dgrad(
                    plan,
                    image_shape,
                    loss_shape,
                    weight_shape,
                )
                and loss.is_contiguous()
                and weight.is_contiguous()
                and image.is_contiguous()
            ):
                pack_ci = 16
                pack_co = 8
                _conv_dgrad_pack_weight_2d_gather_kernel[
                    (
                        _ceil_div(cout_per_group, pack_co),
                        plan.groups,
                    )
                ](
                    weight,
                    packed_weight,
                    GROUPS=plan.groups,
                    CIN_PER_GROUP=cin_per_group,
                    COUT_PER_GROUP=cout_per_group,
                    KH=kh,
                    KW=kw,
                    WEIGHT_STRIDE_O=weight_strides[0],
                    FILTER_REVERSE=plan.filter_reverse,
                    LOAD_K=triton.next_power_of_2(
                        cin_per_group * kernel_volume
                    ),
                    BLOCK_CO=pack_co,
                    BLOCK_CI=pack_ci,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                block_m = 16
                block_n = 1024 if loss.dtype == torch.float32 else 2048
                block_k = 32
                matmul_grid = (
                    matrix_count
                    * _ceil_div(cin_per_group, block_m)
                    * _ceil_div(output_spatial, block_n),
                )
                _conv_dgrad_broadcast_matmul_kernel[matmul_grid](
                    packed_weight,
                    loss,
                    partial_workspace,
                    M=cin_per_group,
                    N=output_spatial,
                    K=cout_per_group,
                    GROUPS=plan.groups,
                    KERNEL_VOLUME=kernel_volume,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    GROUP_M=1,
                    TF32=loss.dtype == torch.float32,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                gather_ci = 4
                gather_rows = 8
                gather_width = 128
                gather_grid = (
                    _ceil_div(cin_per_group, gather_ci),
                    _ceil_div(oh, gather_rows) * _ceil_div(xw, gather_width),
                    n * plan.groups,
                )
                for parity_h in (0, 1):
                    _conv_dgrad_stride2_partial_tiled_interleave_kernel[
                        gather_grid
                    ](
                        partial_workspace,
                        output,
                        GROUPS=plan.groups,
                        CIN_PER_GROUP=cin_per_group,
                        XW=xw,
                        OH=oh,
                        OW=ow,
                        OUTPUT_STRIDE_N=image_strides[0],
                        OUTPUT_STRIDE_C=image_strides[1],
                        PARITY_H=parity_h,
                        BLOCK_CI=gather_ci,
                        BLOCK_ROWS=gather_rows,
                        LOAD_W=gather_width // 2,
                        BLOCK_W=gather_width,
                        num_warps=4,
                        num_stages=1,
                        multibuffer=False,
                        sync_solver=True,
                    )
                return output
            if (
                _supports_deep_stride2_2d_dgrad(
                    plan,
                    image_shape,
                    loss_shape,
                    weight_shape,
                )
                and loss.is_contiguous()
                and weight.is_contiguous()
                and image.is_contiguous()
            ):
                if cin_per_group == 128:
                    pack_ci = 128
                    pack_co = 8
                    block_m = 128
                else:
                    pack_ci = 64
                    pack_co = 16 if cin_per_group == 768 else 8
                    block_m = 256
                pack_grid = (
                    _ceil_div(cout_per_group, pack_co),
                    _ceil_div(cin_per_group, pack_ci),
                    plan.groups,
                )
                _conv_dgrad_pack_weight_2d_tiled_gather_kernel[pack_grid](
                    weight,
                    packed_weight,
                    GROUPS=plan.groups,
                    CIN_PER_GROUP=cin_per_group,
                    COUT_PER_GROUP=cout_per_group,
                    KH=kh,
                    KW=kw,
                    WEIGHT_STRIDE_O=weight_strides[0],
                    FILTER_REVERSE=plan.filter_reverse,
                    LOAD_K=triton.next_power_of_2(pack_ci * kernel_volume),
                    BLOCK_CO=pack_co,
                    BLOCK_CI=pack_ci,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                block_n = 128
                block_k = 128
                matmul_grid = (
                    matrix_count
                    * _ceil_div(cin_per_group, block_m)
                    * _ceil_div(output_spatial, block_n),
                )
                _conv_dgrad_broadcast_matmul_kernel[matmul_grid](
                    packed_weight,
                    loss,
                    partial_workspace,
                    M=cin_per_group,
                    N=output_spatial,
                    K=cout_per_group,
                    GROUPS=plan.groups,
                    KERNEL_VOLUME=kernel_volume,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    GROUP_M=1,
                    TF32=loss.dtype == torch.float32,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                gather_ci = 32
                gather_rows = 2
                gather_width = 64
                gather_grid = (
                    _ceil_div(cin_per_group, gather_ci),
                    _ceil_div(oh, gather_rows) * _ceil_div(xw, gather_width),
                    n * plan.groups,
                )
                for parity_h in (0, 1):
                    _conv_dgrad_stride2_partial_tiled_interleave_kernel[
                        gather_grid
                    ](
                        partial_workspace,
                        output,
                        GROUPS=plan.groups,
                        CIN_PER_GROUP=cin_per_group,
                        XW=xw,
                        OH=oh,
                        OW=ow,
                        OUTPUT_STRIDE_N=image_strides[0],
                        OUTPUT_STRIDE_C=image_strides[1],
                        PARITY_H=parity_h,
                        BLOCK_CI=gather_ci,
                        BLOCK_ROWS=gather_rows,
                        LOAD_W=gather_width // 2,
                        BLOCK_W=gather_width,
                        num_warps=4,
                        num_stages=1,
                        multibuffer=False,
                        sync_solver=True,
                    )
                return output
            if (
                _supports_stride2_2d_dgrad(
                    plan,
                    image_shape,
                    loss_shape,
                    weight_shape,
                )
                and loss.is_contiguous()
                and weight.is_contiguous()
                and image.is_contiguous()
            ):
                pack_ci = 16
                pack_co = 8
                _conv_dgrad_pack_weight_2d_gather_kernel[
                    (
                        _ceil_div(cout_per_group, pack_co),
                        plan.groups,
                    )
                ](
                    weight,
                    packed_weight,
                    GROUPS=plan.groups,
                    CIN_PER_GROUP=cin_per_group,
                    COUT_PER_GROUP=cout_per_group,
                    KH=kh,
                    KW=kw,
                    WEIGHT_STRIDE_O=weight_strides[0],
                    FILTER_REVERSE=plan.filter_reverse,
                    LOAD_K=triton.next_power_of_2(
                        cin_per_group * kernel_volume
                    ),
                    BLOCK_CO=pack_co,
                    BLOCK_CI=pack_ci,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                if loss.dtype == torch.float32:
                    block_m = 64
                    block_n = 256
                    block_k = 64
                else:
                    block_m = 64
                    block_n = 512
                    block_k = 128
                matmul_grid = (
                    matrix_count
                    * _ceil_div(cin_per_group, block_m)
                    * _ceil_div(output_spatial, block_n),
                )
                _conv_dgrad_broadcast_matmul_kernel[matmul_grid](
                    packed_weight,
                    loss,
                    partial_workspace,
                    M=cin_per_group,
                    N=output_spatial,
                    K=cout_per_group,
                    GROUPS=plan.groups,
                    KERNEL_VOLUME=kernel_volume,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    GROUP_M=1,
                    TF32=loss.dtype == torch.float32,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                gather_ci = 32
                gather_rows = 2
                gather_grid = (
                    _ceil_div(cin_per_group, gather_ci),
                    _ceil_div(oh, gather_rows),
                    n * plan.groups,
                )
                for parity_h in (0, 1):
                    _conv_dgrad_stride2_partial_row_interleave_kernel[
                        gather_grid
                    ](
                        partial_workspace,
                        output,
                        GROUPS=plan.groups,
                        CIN_PER_GROUP=cin_per_group,
                        XW=xw,
                        OH=oh,
                        OW=ow,
                        OUTPUT_STRIDE_N=image_strides[0],
                        OUTPUT_STRIDE_C=image_strides[1],
                        PARITY_H=parity_h,
                        BLOCK_CI=gather_ci,
                        BLOCK_ROWS=gather_rows,
                        LOAD_W=32,
                        BLOCK_W=64,
                        num_warps=4,
                        num_stages=1,
                        multibuffer=False,
                        sync_solver=True,
                    )
                return output
            if (
                _supports_full_plane_2d_dgrad(
                    plan,
                    image_shape,
                    loss_shape,
                    weight_shape,
                )
                and loss.is_contiguous()
                and weight.is_contiguous()
                and image.is_contiguous()
            ):
                pack_ci = 16
                pack_co = 8
                _conv_dgrad_pack_weight_2d_gather_kernel[
                    (
                        _ceil_div(cout_per_group, pack_co),
                        plan.groups,
                    )
                ](
                    weight,
                    packed_weight,
                    GROUPS=plan.groups,
                    CIN_PER_GROUP=cin_per_group,
                    COUT_PER_GROUP=cout_per_group,
                    KH=kh,
                    KW=kw,
                    WEIGHT_STRIDE_O=weight_strides[0],
                    FILTER_REVERSE=plan.filter_reverse,
                    LOAD_K=triton.next_power_of_2(
                        cin_per_group * kernel_volume
                    ),
                    BLOCK_CO=pack_co,
                    BLOCK_CI=pack_ci,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                block_m = 32
                block_n = 512 if loss.dtype == torch.float32 else 1024
                block_k = 64
                matmul_grid = (
                    matrix_count
                    * _ceil_div(cin_per_group, block_m)
                    * _ceil_div(output_spatial, block_n),
                )
                _conv_dgrad_broadcast_matmul_kernel[matmul_grid](
                    packed_weight,
                    loss,
                    partial_workspace,
                    M=cin_per_group,
                    N=output_spatial,
                    K=cout_per_group,
                    GROUPS=plan.groups,
                    KERNEL_VOLUME=kernel_volume,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    GROUP_M=1,
                    TF32=loss.dtype == torch.float32,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                gather_ci = 8
                gather_spatial = 512
                _conv_dgrad_gather_2d_full_plane_kernel[
                    (
                        _ceil_div(cin_per_group, gather_ci),
                        n * plan.groups,
                    )
                ](
                    partial_workspace,
                    output,
                    GROUPS=plan.groups,
                    CIN_PER_GROUP=cin_per_group,
                    XH=xh,
                    XW=xw,
                    OH=oh,
                    OW=ow,
                    KH=kh,
                    KW=kw,
                    STRIDE_H=stride_h,
                    STRIDE_W=stride_w,
                    PAD_H=pad_h,
                    PAD_W=pad_w,
                    DIL_H=dil_h,
                    DIL_W=dil_w,
                    OUTPUT_STRIDE_N=image_strides[0],
                    OUTPUT_STRIDE_C=image_strides[1],
                    LOAD_O=triton.next_power_of_2(output_spatial),
                    BLOCK_CI=gather_ci,
                    BLOCK_SPATIAL=gather_spatial,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                return output
            fast_dgrad_weight_pack = _supports_fast_dgrad_1d_weight_pack(
                plan,
                cin_per_group,
                kw,
            )
            if fast_dgrad_weight_pack:
                pack_ci = 16
                pack_co = 8
                _conv_dgrad_pack_weight_1d_gather_kernel[
                    (
                        _ceil_div(cout_per_group, pack_co),
                        plan.groups,
                    )
                ](
                    weight,
                    packed_weight,
                    GROUPS=plan.groups,
                    CIN_PER_GROUP=cin_per_group,
                    COUT_PER_GROUP=cout_per_group,
                    KW=kw,
                    WEIGHT_STRIDE_O=weight_strides[0],
                    FILTER_REVERSE=plan.filter_reverse,
                    LOAD_K=triton.next_power_of_2(cin_per_group * kw),
                    BLOCK_CO=pack_co,
                    BLOCK_CI=pack_ci,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
            else:
                pack_ci = 64
                pack_co = 128
                tiled_pack_grid = (
                    plan.groups
                    * kernel_volume
                    * _ceil_div(cin_per_group, pack_ci)
                    * _ceil_div(cout_per_group, pack_co),
                )
                _conv_dgrad_pack_weight_tiled_kernel[tiled_pack_grid](
                    weight,
                    packed_weight,
                    GROUPS=plan.groups,
                    CIN_PER_GROUP=cin_per_group,
                    COUT_PER_GROUP=cout_per_group,
                    KD=kd,
                    KH=kh,
                    KW=kw,
                    WEIGHT_STRIDE_O=weight_strides[0],
                    WEIGHT_STRIDE_I=weight_strides[1],
                    WEIGHT_STRIDE_D=weight_strides[2],
                    WEIGHT_STRIDE_H=weight_strides[3],
                    WEIGHT_STRIDE_W=weight_strides[4],
                    FILTER_REVERSE=plan.filter_reverse,
                    BLOCK_CI=pack_ci,
                    BLOCK_CO=pack_co,
                    num_warps=4,
                    num_stages=1,
                )
            if fast_dgrad_weight_pack:
                if loss.dtype == torch.float32:
                    if cin_per_group <= 32:
                        block_m, block_n, block_k, group_m = (
                            32,
                            256,
                            32,
                            1,
                        )
                    else:
                        block_m, block_n, block_k, group_m = (
                            64,
                            256,
                            64,
                            1,
                        )
                    matmul_warps = 4
                elif cin_per_group <= 32:
                    block_m, block_n, block_k, group_m = (
                        64,
                        256,
                        64,
                        1,
                    )
                    matmul_warps = 4
                else:
                    block_m, block_n, block_k, group_m = (
                        128,
                        128,
                        128,
                        4,
                    )
                    matmul_warps = 8
            elif cin_per_group <= 32 or output_spatial <= 32:
                block_m, block_n, block_k, group_m = 32, 32, 32, 1
                matmul_warps = 4
            elif loss.dtype == torch.float32:
                block_m, block_n, block_k, group_m = (
                    256,
                    128,
                    128,
                    4,
                )
                matmul_warps = 8
            else:
                block_m, block_n, block_k, group_m = (
                    128,
                    256,
                    128,
                    4,
                )
                matmul_warps = 8
            matmul_grid = (
                matrix_count
                * _ceil_div(cin_per_group, block_m)
                * _ceil_div(output_spatial, block_n),
            )
            _conv_dgrad_broadcast_matmul_kernel[matmul_grid](
                packed_weight,
                loss,
                partial_workspace,
                M=cin_per_group,
                N=output_spatial,
                K=cout_per_group,
                GROUPS=plan.groups,
                KERNEL_VOLUME=kernel_volume,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                GROUP_M=group_m,
                TF32=loss.dtype == torch.float32,
                num_warps=matmul_warps,
                num_stages=2,
            )
            if (
                plan.rank == 1
                and _supports_fast_conv_1d_pack(plan, xw)
                and image.is_contiguous()
            ):
                gather_w = 256
                gather_ci = 16
                gather_grid = (
                    _ceil_div(xw, gather_w),
                    _ceil_div(cin_per_group, gather_ci),
                    n * plan.groups,
                )
                if stride_w == 1:
                    _conv_dgrad_gather_1d_block_kernel[gather_grid](
                        partial_workspace,
                        output,
                        GROUPS=plan.groups,
                        CIN_PER_GROUP=cin_per_group,
                        XW=xw,
                        OW=ow,
                        KW=kw,
                        PAD_W=pad_w,
                        DIL_W=dil_w,
                        OUTPUT_STRIDE_N=image_strides[0],
                        OUTPUT_STRIDE_C=image_strides[1],
                        OUTPUT_STRIDE_W=image_strides[4],
                        BLOCK_W=gather_w,
                        BLOCK_CI=gather_ci,
                        num_warps=4,
                        num_stages=1,
                        multibuffer=False,
                        sync_solver=True,
                    )
                else:
                    _conv_dgrad_gather_1d_strided_kernel[gather_grid](
                        partial_workspace,
                        output,
                        GROUPS=plan.groups,
                        CIN_PER_GROUP=cin_per_group,
                        XW=xw,
                        OW=ow,
                        KW=kw,
                        STRIDE_W=stride_w,
                        PAD_W=pad_w,
                        DIL_W=dil_w,
                        OUTPUT_STRIDE_N=image_strides[0],
                        OUTPUT_STRIDE_C=image_strides[1],
                        OUTPUT_STRIDE_W=image_strides[4],
                        LOAD_W=triton.next_power_of_2(ow),
                        BLOCK_W=gather_w,
                        BLOCK_CI=gather_ci,
                        num_warps=4,
                        num_stages=1,
                        multibuffer=False,
                        sync_solver=True,
                    )
                return output
            if kernel_volume > 9:
                gather_ci, gather_m = 2, 128
            else:
                gather_ci, gather_m = 16, 256
            tiled_gather_grid = (
                n
                * plan.groups
                * _ceil_div(cin_per_group, gather_ci)
                * _ceil_div(xd * xh * xw, gather_m),
            )
            _conv_dgrad_gather_tiled_kernel[tiled_gather_grid](
                partial_workspace,
                output,
                GROUPS=plan.groups,
                CIN_PER_GROUP=cin_per_group,
                XD=xd,
                XH=xh,
                XW=xw,
                OD=od,
                OH=oh,
                OW=ow,
                KD=kd,
                KH=kh,
                KW=kw,
                STRIDE_D=stride_d,
                STRIDE_H=stride_h,
                STRIDE_W=stride_w,
                PAD_D=pad_d,
                PAD_H=pad_h,
                PAD_W=pad_w,
                DIL_D=dil_d,
                DIL_H=dil_h,
                DIL_W=dil_w,
                OUTPUT_STRIDE_N=image_strides[0],
                OUTPUT_STRIDE_C=image_strides[1],
                OUTPUT_STRIDE_D=image_strides[2],
                OUTPUT_STRIDE_H=image_strides[3],
                OUTPUT_STRIDE_W=image_strides[4],
                BLOCK_CI=gather_ci,
                BLOCK_M=gather_m,
                num_warps=4,
                num_stages=1,
            )
        else:
            kernel_volume = kd * kh * kw
            if (
                _is_pointwise_conv(plan, weight_shape)
                and image.is_contiguous()
                and loss.is_contiguous()
                and output.is_contiguous()
            ):
                output_spatial = od * oh * ow
                total_weight = c_out * cin_per_group
                if n == 1:
                    partial = output
                else:
                    if (
                        not isinstance(workspace, torch.Tensor)
                        or workspace.dtype != torch.float32
                        or workspace.numel() < n * total_weight
                    ):
                        raise RuntimeError(
                            "Ascend pointwise convolution wgrad "
                            "workspace is missing"
                        )
                    partial = workspace
                block_co = 128
                block_ci = 64
                block_spatial = 64
                grid = (
                    n
                    * plan.groups
                    * _ceil_div(cout_per_group, block_co)
                    * _ceil_div(cin_per_group, block_ci),
                )
                _conv_wgrad_pointwise_batch_kernel[grid](
                    image,
                    loss,
                    partial,
                    N_BATCH=n,
                    GROUPS=plan.groups,
                    CIN_PER_GROUP=cin_per_group,
                    COUT_PER_GROUP=cout_per_group,
                    SPATIAL=output_spatial,
                    C_IN=c_in,
                    C_OUT=c_out,
                    IMAGE_STRIDE_N=image_strides[0],
                    IMAGE_STRIDE_C=image_strides[1],
                    LOSS_STRIDE_N=loss_strides[0],
                    LOSS_STRIDE_C=loss_strides[1],
                    BLOCK_M=block_co,
                    BLOCK_N=block_ci,
                    BLOCK_K=block_spatial,
                    GROUP_M=4,
                    TF32=image.dtype == torch.float32,
                    num_warps=4,
                    num_stages=2,
                    sync_solver=True,
                )
                if n > 1:
                    block_weight = 64
                    block_batches = triton.next_power_of_2(n)
                    _conv_wgrad_reduce_kernel[
                        (_ceil_div(total_weight, block_weight),)
                    ](
                        workspace,
                        output,
                        TOTAL_WEIGHT=total_weight,
                        NUM_SPLITS=n,
                        BLOCK_SPLITS=block_batches,
                        BLOCK_WEIGHT=block_weight,
                        num_warps=4,
                        num_stages=1,
                    )
                return output
            if (
                _supports_full_volume_3d_wgrad(
                    plan,
                    image_shape,
                    loss_shape,
                    weight_shape,
                )
                and image.is_contiguous()
                and loss.is_contiguous()
                and output.is_contiguous()
            ):
                if not isinstance(workspace, tuple) or len(workspace) != 3:
                    raise RuntimeError(
                        "Ascend 3D convolution wgrad workspaces are missing"
                    )
                packed_loss, packed_image, partial_weight = workspace
                output_spatial = od * oh * ow
                reduction = n * output_spatial
                columns = cin_per_group * kernel_volume
                if n == 1:
                    loss_matrix = loss
                else:
                    loss_block_co = 16
                    loss_block_spatial = 512
                    _conv_wgrad_pack_loss_1d_kernel[
                        (
                            _ceil_div(
                                cout_per_group,
                                loss_block_co,
                            ),
                            _ceil_div(
                                output_spatial,
                                loss_block_spatial,
                            ),
                            n * plan.groups,
                        )
                    ](
                        loss,
                        packed_loss,
                        GROUPS=plan.groups,
                        COUT_PER_GROUP=cout_per_group,
                        OW=output_spatial,
                        REDUCTION=reduction,
                        LOSS_STRIDE_N=loss_strides[0],
                        LOSS_STRIDE_C=loss_strides[1],
                        LOSS_STRIDE_W=1,
                        BLOCK_CO=loss_block_co,
                        BLOCK_W=loss_block_spatial,
                        num_warps=4,
                        num_stages=1,
                        multibuffer=False,
                        sync_solver=True,
                    )
                    loss_matrix = packed_loss
                image_block_ci = 1
                image_block_spatial = 1024
                _conv_wgrad_pack_image_3d_full_volume_kernel[
                    (
                        _ceil_div(
                            cin_per_group,
                            image_block_ci,
                        ),
                        n * plan.groups,
                    )
                ](
                    image,
                    packed_image,
                    GROUPS=plan.groups,
                    CIN_PER_GROUP=cin_per_group,
                    XD=xd,
                    XH=xh,
                    XW=xw,
                    OD=od,
                    OH=oh,
                    OW=ow,
                    KD=kd,
                    KH=kh,
                    KW=kw,
                    REDUCTION=reduction,
                    STRIDE_D=stride_d,
                    STRIDE_H=stride_h,
                    STRIDE_W=stride_w,
                    PAD_D=pad_d,
                    PAD_H=pad_h,
                    PAD_W=pad_w,
                    DIL_D=dil_d,
                    DIL_H=dil_h,
                    DIL_W=dil_w,
                    IMAGE_STRIDE_N=image_strides[0],
                    IMAGE_STRIDE_C=image_strides[1],
                    FILTER_REVERSE=plan.filter_reverse,
                    LOAD_X=triton.next_power_of_2(xd * xh * xw),
                    BLOCK_CI=image_block_ci,
                    BLOCK_SPATIAL=image_block_spatial,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                split_k = 512
                num_splits = _ceil_div(reduction, split_k)
                block_m = 16
                block_n = 128 if columns == 216 else 32
                block_k = 128
                matmul_grid = (
                    num_splits
                    * _ceil_div(cout_per_group, block_m)
                    * _ceil_div(columns, block_n),
                )
                _conv_wgrad_matmul_transposed_image_splitk_kernel[matmul_grid](
                    loss_matrix,
                    packed_image,
                    partial_weight,
                    M=cout_per_group,
                    N=columns,
                    K=reduction,
                    NUM_SPLITS=num_splits,
                    SPLIT_K=split_k,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    TF32=image.dtype == torch.float32,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                total_weight = c_out * columns
                reduce_block = 64
                _conv_wgrad_reduce_kernel[
                    (_ceil_div(total_weight, reduce_block),)
                ](
                    partial_weight,
                    output,
                    TOTAL_WEIGHT=total_weight,
                    NUM_SPLITS=num_splits,
                    BLOCK_SPLITS=triton.next_power_of_2(num_splits),
                    BLOCK_WEIGHT=reduce_block,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                return output
            if (
                _supports_stem_stride2_2d_wgrad(
                    plan,
                    image_shape,
                    loss_shape,
                    weight_shape,
                )
                and image.is_contiguous()
                and loss.is_contiguous()
                and output.is_contiguous()
            ):
                if not isinstance(workspace, tuple) or len(workspace) != 3:
                    raise RuntimeError(
                        "Ascend stem 2D convolution wgrad "
                        "workspaces are missing"
                    )
                _, packed_image, partial_weight = workspace
                output_spatial = od * oh * ow
                reduction = n * output_spatial
                columns = cin_per_group * kernel_volume
                rows_per_program = 16
                _conv_fprop_im2col_2d_stem_row_reuse_kernel[
                    (
                        cin_per_group * kh,
                        _ceil_div(oh, rows_per_program),
                        n,
                    )
                ](
                    image,
                    packed_image,
                    CIN_PER_GROUP=cin_per_group,
                    XH=xh,
                    XW=xw,
                    OH=oh,
                    OW=ow,
                    KH=kh,
                    KW=kw,
                    STRIDE_H=stride_h,
                    STRIDE_W=stride_w,
                    PAD_H=pad_h,
                    PAD_W=pad_w,
                    DIL_H=dil_h,
                    DIL_W=dil_w,
                    IMAGE_STRIDE_N=image_strides[0],
                    IMAGE_STRIDE_C=image_strides[1],
                    FILTER_REVERSE=plan.filter_reverse,
                    ROWS_PER_PROGRAM=rows_per_program,
                    LOAD_W=triton.next_power_of_2(xw),
                    BLOCK_W=triton.next_power_of_2(ow),
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                split_k = 2048
                num_splits = _ceil_div(reduction, split_k)
                block_m = (
                    64
                    if image.dtype == torch.float32
                    else triton.next_power_of_2(cout_per_group)
                )
                block_n = 32
                block_k = 1024
                matmul_grid = (
                    num_splits
                    * _ceil_div(cout_per_group, block_m)
                    * _ceil_div(columns, block_n),
                )
                _conv_wgrad_matmul_transposed_image_splitk_kernel[matmul_grid](
                    loss,
                    packed_image,
                    partial_weight,
                    M=cout_per_group,
                    N=columns,
                    K=reduction,
                    NUM_SPLITS=num_splits,
                    SPLIT_K=split_k,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    TF32=image.dtype == torch.float32,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                total_weight = c_out * columns
                reduce_block = 64
                _conv_wgrad_reduce_kernel[
                    (_ceil_div(total_weight, reduce_block),)
                ](
                    partial_weight,
                    output,
                    TOTAL_WEIGHT=total_weight,
                    NUM_SPLITS=num_splits,
                    BLOCK_SPLITS=triton.next_power_of_2(num_splits),
                    BLOCK_WEIGHT=reduce_block,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                return output
            if (
                _supports_deep_stride2_2d_wgrad(
                    plan,
                    image_shape,
                    loss_shape,
                    weight_shape,
                )
                and image.is_contiguous()
                and loss.is_contiguous()
                and output.is_contiguous()
            ):
                if not isinstance(workspace, tuple) or len(workspace) != 2:
                    raise RuntimeError(
                        "Ascend high-channel 2D convolution wgrad "
                        "workspaces are missing"
                    )
                _, packed_image = workspace
                output_spatial = od * oh * ow
                reduction = n * output_spatial
                columns = cin_per_group * kernel_volume
                image_block_ci = 4 if cin_per_group == 128 else 8
                image_block_spatial = 256
                _conv_wgrad_pack_image_2d_full_plane_kernel[
                    (
                        _ceil_div(
                            cin_per_group,
                            image_block_ci,
                        ),
                        n * plan.groups,
                    )
                ](
                    image,
                    packed_image,
                    GROUPS=plan.groups,
                    CIN_PER_GROUP=cin_per_group,
                    XH=xh,
                    XW=xw,
                    OH=oh,
                    OW=ow,
                    KH=kh,
                    KW=kw,
                    REDUCTION=reduction,
                    STRIDE_H=stride_h,
                    STRIDE_W=stride_w,
                    PAD_H=pad_h,
                    PAD_W=pad_w,
                    DIL_H=dil_h,
                    DIL_W=dil_w,
                    IMAGE_STRIDE_N=image_strides[0],
                    IMAGE_STRIDE_C=image_strides[1],
                    FILTER_REVERSE=plan.filter_reverse,
                    LOAD_X=triton.next_power_of_2(xh * xw),
                    BLOCK_CI=image_block_ci,
                    BLOCK_SPATIAL=image_block_spatial,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                block_m = 128
                block_n = 128 if cin_per_group == 128 else 256
                block_k = 128
                matmul_grid = (
                    plan.groups
                    * _ceil_div(cout_per_group, block_m)
                    * _ceil_div(columns, block_n),
                )
                _conv_wgrad_matmul_transposed_image_kernel[matmul_grid](
                    loss,
                    packed_image,
                    output,
                    M=cout_per_group,
                    N=columns,
                    K=reduction,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    GROUP_M=1,
                    TF32=image.dtype == torch.float32,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                return output
            if (
                _supports_row_tile_2d_wgrad(
                    plan,
                    image_shape,
                    loss_shape,
                    weight_shape,
                )
                and image.is_contiguous()
                and loss.is_contiguous()
                and output.is_contiguous()
            ):
                if not isinstance(workspace, tuple) or len(workspace) != 2:
                    raise RuntimeError(
                        "Ascend row-tiled 2D convolution wgrad "
                        "workspaces are missing"
                    )
                packed_loss, packed_image = workspace
                output_spatial = od * oh * ow
                reduction = n * output_spatial
                columns = cin_per_group * kernel_volume
                fp32 = image.dtype == torch.float32
                loss_block_co = 64
                loss_block_spatial = 512
                _conv_wgrad_pack_loss_1d_kernel[
                    (
                        _ceil_div(
                            cout_per_group,
                            loss_block_co,
                        ),
                        _ceil_div(
                            output_spatial,
                            loss_block_spatial,
                        ),
                        n * plan.groups,
                    )
                ](
                    loss,
                    packed_loss,
                    GROUPS=plan.groups,
                    COUT_PER_GROUP=cout_per_group,
                    OW=output_spatial,
                    REDUCTION=reduction,
                    LOSS_STRIDE_N=loss_strides[0],
                    LOSS_STRIDE_C=loss_strides[1],
                    LOSS_STRIDE_W=1,
                    BLOCK_CO=loss_block_co,
                    BLOCK_W=loss_block_spatial,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                if image.dtype == torch.float16:
                    image_block_ci = 8
                else:
                    image_block_ci = 4
                image_block_spatial = 512
                _conv_wgrad_pack_image_2d_row_tile_kernel[
                    (
                        _ceil_div(
                            cin_per_group,
                            image_block_ci,
                        ),
                        n * plan.groups,
                    )
                ](
                    image,
                    packed_image,
                    GROUPS=plan.groups,
                    CIN_PER_GROUP=cin_per_group,
                    XH=xh,
                    XW=xw,
                    OH=oh,
                    OW=ow,
                    KH=kh,
                    KW=kw,
                    REDUCTION=reduction,
                    STRIDE_H=stride_h,
                    STRIDE_W=stride_w,
                    PAD_H=pad_h,
                    PAD_W=pad_w,
                    DIL_H=dil_h,
                    DIL_W=dil_w,
                    IMAGE_STRIDE_N=image_strides[0],
                    IMAGE_STRIDE_C=image_strides[1],
                    FILTER_REVERSE=plan.filter_reverse,
                    ROW_TILES=2,
                    ROWS_PER_TILE=14,
                    LOAD_X=2048,
                    BLOCK_CI=image_block_ci,
                    BLOCK_SPATIAL=image_block_spatial,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                block_m = 64
                block_n = 64
                block_k = 128 if fp32 else 256
                matmul_grid = (
                    plan.groups
                    * _ceil_div(cout_per_group, block_m)
                    * _ceil_div(columns, block_n),
                )
                _conv_wgrad_matmul_transposed_image_kernel[matmul_grid](
                    packed_loss,
                    packed_image,
                    output,
                    M=cout_per_group,
                    N=columns,
                    K=reduction,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    GROUP_M=1,
                    TF32=fp32,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                return output
            if (
                _supports_full_plane_2d_wgrad(
                    plan,
                    image_shape,
                    loss_shape,
                    weight_shape,
                )
                and image.is_contiguous()
                and loss.is_contiguous()
                and output.is_contiguous()
            ):
                if not isinstance(workspace, tuple) or len(workspace) != 2:
                    raise RuntimeError(
                        "Ascend 2D convolution wgrad workspaces are missing"
                    )
                packed_loss, packed_image = workspace
                output_spatial = od * oh * ow
                reduction = n * output_spatial
                columns = cin_per_group * kernel_volume
                fp32 = image.dtype == torch.float32
                loss_block_co = 32 if fp32 else 64
                loss_block_spatial = 512
                _conv_wgrad_pack_loss_1d_kernel[
                    (
                        _ceil_div(
                            cout_per_group,
                            loss_block_co,
                        ),
                        _ceil_div(
                            output_spatial,
                            loss_block_spatial,
                        ),
                        n * plan.groups,
                    )
                ](
                    loss,
                    packed_loss,
                    GROUPS=plan.groups,
                    COUT_PER_GROUP=cout_per_group,
                    OW=output_spatial,
                    REDUCTION=reduction,
                    LOSS_STRIDE_N=loss_strides[0],
                    LOSS_STRIDE_C=loss_strides[1],
                    LOSS_STRIDE_W=1,
                    BLOCK_CO=loss_block_co,
                    BLOCK_W=loss_block_spatial,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                image_block_ci = 8
                image_block_spatial = 512
                _conv_wgrad_pack_image_2d_full_plane_kernel[
                    (
                        _ceil_div(
                            cin_per_group,
                            image_block_ci,
                        ),
                        n * plan.groups,
                    )
                ](
                    image,
                    packed_image,
                    GROUPS=plan.groups,
                    CIN_PER_GROUP=cin_per_group,
                    XH=xh,
                    XW=xw,
                    OH=oh,
                    OW=ow,
                    KH=kh,
                    KW=kw,
                    REDUCTION=reduction,
                    STRIDE_H=stride_h,
                    STRIDE_W=stride_w,
                    PAD_H=pad_h,
                    PAD_W=pad_w,
                    DIL_H=dil_h,
                    DIL_W=dil_w,
                    IMAGE_STRIDE_N=image_strides[0],
                    IMAGE_STRIDE_C=image_strides[1],
                    FILTER_REVERSE=plan.filter_reverse,
                    LOAD_X=triton.next_power_of_2(xh * xw),
                    BLOCK_CI=image_block_ci,
                    BLOCK_SPATIAL=image_block_spatial,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                block_m = 64
                if fp32:
                    block_n = 32
                    block_k = 128
                else:
                    block_n = 64
                    block_k = 256
                matmul_grid = (
                    plan.groups
                    * _ceil_div(cout_per_group, block_m)
                    * _ceil_div(columns, block_n),
                )
                _conv_wgrad_matmul_transposed_image_kernel[matmul_grid](
                    packed_loss,
                    packed_image,
                    output,
                    M=cout_per_group,
                    N=columns,
                    K=reduction,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    GROUP_M=1,
                    TF32=fp32,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                return output
            if (
                plan.rank == 1
                and _supports_fast_conv_1d_pack(plan, xw)
                and image.is_contiguous()
                and loss.is_contiguous()
                and output.is_contiguous()
            ):
                if not isinstance(workspace, tuple) or len(workspace) != 3:
                    raise RuntimeError(
                        "Ascend 1D convolution wgrad workspaces are missing"
                    )
                packed_loss, packed_image, packed_weight = workspace
                reduction = n * ow
                columns = cin_per_group * kw
                loss_block_co = 32
                loss_block_w = 256
                _conv_wgrad_pack_loss_1d_kernel[
                    (
                        _ceil_div(cout_per_group, loss_block_co),
                        _ceil_div(ow, loss_block_w),
                        n * plan.groups,
                    )
                ](
                    loss,
                    packed_loss,
                    GROUPS=plan.groups,
                    COUT_PER_GROUP=cout_per_group,
                    OW=ow,
                    REDUCTION=reduction,
                    LOSS_STRIDE_N=loss_strides[0],
                    LOSS_STRIDE_C=loss_strides[1],
                    LOSS_STRIDE_W=loss_strides[4],
                    BLOCK_CO=loss_block_co,
                    BLOCK_W=loss_block_w,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                )
                if stride_w == 1:
                    image_block_r = 256
                    image_block_ci = 8
                    image_pack_kernel = _conv_wgrad_pack_image_1d_block_kernel
                    image_pack_kwargs = {}
                else:
                    image_block_r = 128
                    image_block_ci = 16
                    image_pack_kernel = _conv_wgrad_pack_image_1d_gather_kernel
                    image_pack_kwargs = {
                        "STRIDE_W": stride_w,
                        "LOAD_W": triton.next_power_of_2(xw),
                    }
                image_pack_kernel[
                    (
                        _ceil_div(ow, image_block_r),
                        _ceil_div(cin_per_group, image_block_ci),
                        n * plan.groups,
                    )
                ](
                    image,
                    packed_image,
                    GROUPS=plan.groups,
                    CIN_PER_GROUP=cin_per_group,
                    XW=xw,
                    OW=ow,
                    KW=kw,
                    REDUCTION=reduction,
                    PAD_W=pad_w,
                    DIL_W=dil_w,
                    IMAGE_STRIDE_N=image_strides[0],
                    IMAGE_STRIDE_C=image_strides[1],
                    IMAGE_STRIDE_W=image_strides[4],
                    FILTER_REVERSE=plan.filter_reverse,
                    BLOCK_R=image_block_r,
                    BLOCK_CI=image_block_ci,
                    num_warps=4,
                    num_stages=1,
                    multibuffer=False,
                    sync_solver=True,
                    **image_pack_kwargs,
                )
                if columns <= 128:
                    block_m, block_n, block_k, matmul_warps = (
                        64,
                        128,
                        128,
                        8,
                    )
                else:
                    block_m, block_n, block_k, matmul_warps = (
                        64,
                        64,
                        64,
                        4,
                    )
                matmul_grid = (
                    plan.groups
                    * _ceil_div(cout_per_group, block_m)
                    * _ceil_div(columns, block_n),
                )
                _conv_wgrad_matmul_transposed_image_kernel[matmul_grid](
                    packed_loss,
                    packed_image,
                    packed_weight,
                    M=cout_per_group,
                    N=columns,
                    K=reduction,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    GROUP_M=4,
                    TF32=image.dtype == torch.float32,
                    num_warps=matmul_warps,
                    num_stages=2,
                    multibuffer=image.dtype != torch.float32,
                    sync_solver=True,
                )
                total_weight = c_out * cin_per_group * kw
                reorder_block = 256
                _conv_wgrad_reorder_1d_kernel[
                    (_ceil_div(total_weight, reorder_block),)
                ](
                    packed_weight,
                    output,
                    TOTAL=total_weight,
                    CIN_PER_GROUP=cin_per_group,
                    KW=kw,
                    BLOCK=reorder_block,
                    num_warps=4,
                    num_stages=1,
                )
                return output
            if image.is_contiguous() and loss.is_contiguous():
                if not isinstance(workspace, tuple) or len(workspace) != 2:
                    raise RuntimeError(
                        "Ascend convolution wgrad pack workspaces are missing"
                    )
                packed_loss, packed_image = workspace
                output_spatial = od * oh * ow
                reduction = n * output_spatial
                columns = cin_per_group * kernel_volume
                if n == 1:
                    loss_matrix = loss.view(
                        plan.groups,
                        cout_per_group,
                        output_spatial,
                    )
                else:
                    loss_block_co = 16
                    loss_block_r = 256
                    _conv_wgrad_pack_loss_kernel[
                        (
                            _ceil_div(
                                cout_per_group,
                                loss_block_co,
                            ),
                            _ceil_div(reduction, loss_block_r),
                            plan.groups,
                        )
                    ](
                        loss,
                        packed_loss,
                        REDUCTION=reduction,
                        GROUPS=plan.groups,
                        COUT_PER_GROUP=cout_per_group,
                        OUTPUT_SPATIAL=output_spatial,
                        LOSS_STRIDE_N=loss_strides[0],
                        LOSS_STRIDE_C=loss_strides[1],
                        BLOCK_CO=loss_block_co,
                        BLOCK_R=loss_block_r,
                        num_warps=4,
                        num_stages=1,
                    )
                    loss_matrix = packed_loss
                image_block_r = 256
                image_block_col = 16
                _conv_wgrad_pack_image_kernel[
                    (
                        _ceil_div(reduction, image_block_r),
                        _ceil_div(columns, image_block_col),
                        plan.groups,
                    )
                ](
                    image,
                    packed_image,
                    REDUCTION=reduction,
                    GROUPS=plan.groups,
                    CIN_PER_GROUP=cin_per_group,
                    XD=xd,
                    XH=xh,
                    XW=xw,
                    OD=od,
                    OH=oh,
                    OW=ow,
                    KD=kd,
                    KH=kh,
                    KW=kw,
                    STRIDE_D=stride_d,
                    STRIDE_H=stride_h,
                    STRIDE_W=stride_w,
                    PAD_D=pad_d,
                    PAD_H=pad_h,
                    PAD_W=pad_w,
                    DIL_D=dil_d,
                    DIL_H=dil_h,
                    DIL_W=dil_w,
                    IMAGE_STRIDE_N=image_strides[0],
                    IMAGE_STRIDE_C=image_strides[1],
                    IMAGE_STRIDE_D=image_strides[2],
                    IMAGE_STRIDE_H=image_strides[3],
                    IMAGE_STRIDE_W=image_strides[4],
                    FILTER_REVERSE=plan.filter_reverse,
                    BLOCK_R=image_block_r,
                    BLOCK_COL=image_block_col,
                    num_warps=4,
                    num_stages=1,
                )
                weight_matrix = output.view(
                    plan.groups,
                    cout_per_group,
                    columns,
                )
                if not matmul_3d_out(
                    loss_matrix,
                    packed_image,
                    weight_matrix,
                    compute_mode=(
                        "tf32" if image.dtype == torch.float32 else "float32"
                    ),
                ):
                    raise RuntimeError(
                        "Ascend convolution wgrad packed matmul is unsupported"
                    )
                return output
            (
                block_co,
                block_n,
                block_r,
                num_splits,
                _,
            ) = _wgrad_launch_config(
                n=n,
                c_out=c_out,
                cin_per_group=cin_per_group,
                cout_per_group=cout_per_group,
                kd=kd,
                kh=kh,
                kw=kw,
                od=od,
                oh=oh,
                ow=ow,
                groups=plan.groups,
                fp32=image.dtype == torch.float32,
            )
            total_weight = c_out * cin_per_group * kernel_volume
            grid = (
                plan.groups
                * num_splits
                * _ceil_div(cout_per_group, block_co)
                * _ceil_div(
                    cin_per_group * kernel_volume,
                    block_n,
                ),
            )
            if num_splits > 1 and workspace is None:
                raise RuntimeError(
                    "Ascend convolution wgrad split workspace is missing"
                )
            partial = output if workspace is None else workspace
            _conv_wgrad_kernel[grid](
                image,
                loss,
                partial,
                output,
                **common,
                IMAGE_STRIDE_N=image_strides[0],
                IMAGE_STRIDE_C=image_strides[1],
                IMAGE_STRIDE_D=image_strides[2],
                IMAGE_STRIDE_H=image_strides[3],
                IMAGE_STRIDE_W=image_strides[4],
                LOSS_STRIDE_N=loss_strides[0],
                LOSS_STRIDE_C=loss_strides[1],
                LOSS_STRIDE_D=loss_strides[2],
                LOSS_STRIDE_H=loss_strides[3],
                LOSS_STRIDE_W=loss_strides[4],
                NUM_SPLITS=num_splits,
                BLOCK_CO=block_co,
                BLOCK_N=block_n,
                BLOCK_R=block_r,
                num_warps=4,
                num_stages=1,
            )
            if num_splits > 1:
                block_weight = 64
                block_splits = triton.next_power_of_2(num_splits)
                _conv_wgrad_reduce_kernel[
                    (_ceil_div(total_weight, block_weight),)
                ](
                    workspace,
                    output,
                    TOTAL_WEIGHT=total_weight,
                    NUM_SPLITS=num_splits,
                    BLOCK_SPLITS=block_splits,
                    BLOCK_WEIGHT=block_weight,
                    num_warps=4,
                    num_stages=1,
                )
    return output


def prepare_conv(
    op_type: str,
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if (
        len(input_specs) < 2
        or not all(_is_runtime_device_spec(spec) for spec in input_specs[:2])
        or input_specs[0].dtype not in _SUPPORTED_DTYPES
        or input_specs[1].dtype != input_specs[0].dtype
    ):
        return None
    plan = _make_plan(op_type, attrs, input_specs)
    if plan is None:
        return None
    checks = runtime_tensor_checks_from_specs(
        input_specs,
        tuple(range(len(input_specs))),
        require_shape=True,
        require_stride=True,
        require_dtype=True,
    )
    if checks is None:
        return None
    validate_inputs = bool(attrs.get("_validate_inputs", True))
    output_dtype = torch_dtype(input_specs[0].dtype)

    def can_run(inputs: Sequence[Any]) -> bool:
        if validate_inputs and not runtime_tensor_checks_pass(inputs, checks):
            return False
        return all(
            isinstance(value, torch.Tensor) and value.device.type == "npu"
            for value in inputs
        )

    def make_output(inputs: Sequence[Any]) -> torch.Tensor:
        return torch.empty(
            plan.output_shape,
            device=inputs[0].device,
            dtype=output_dtype,
        )

    def make_workspace(
        inputs: Sequence[Any],
    ) -> Optional[Any]:
        if plan.op_type == "conv_fprop":
            image_shape, _ = _normalized_tensor(inputs[0], plan.rank)
            weight_shape, _ = _normalized_weight(inputs[1], plan.rank)
            if (
                _is_pointwise_conv(plan, weight_shape)
                and inputs[0].is_contiguous()
                and inputs[1].is_contiguous()
            ):
                return None
            n = image_shape[0]
            cin_per_group = weight_shape[1]
            kernel_volume = weight_shape[2] * weight_shape[3] * weight_shape[4]
            output_spatial = _normalized_spatial(
                plan.output_shape[-plan.rank :],
                1,
            )
            column_elements = (
                n
                * plan.groups
                * cin_per_group
                * kernel_volume
                * output_spatial[0]
                * output_spatial[1]
                * output_spatial[2]
            )
            if (
                (
                    _supports_staged_nhwc_2d_fprop(
                        plan,
                        image_shape,
                        weight_shape,
                    )
                    or _supports_staged_nhwc_batched_2d_fprop(
                        plan,
                        image_shape,
                        weight_shape,
                    )
                )
                and inputs[0].is_contiguous()
                and inputs[1].is_contiguous()
            ):
                packed_image_elements = (
                    image_shape[0]
                    * image_shape[1]
                    * image_shape[2]
                    * image_shape[3]
                    * image_shape[4]
                )
                return (
                    torch.empty(
                        (packed_image_elements,),
                        device=inputs[0].device,
                        dtype=output_dtype,
                    ),
                    torch.empty(
                        (column_elements,),
                        device=inputs[0].device,
                        dtype=output_dtype,
                    ),
                )
            return torch.empty(
                (column_elements,),
                device=inputs[0].device,
                dtype=output_dtype,
            )
        if plan.op_type == "conv_dgrad":
            loss_shape, _ = _normalized_tensor(inputs[0], plan.rank)
            weight_shape, _ = _normalized_weight(inputs[1], plan.rank)
            if (
                _is_pointwise_conv(plan, weight_shape)
                and inputs[0].is_contiguous()
                and inputs[1].is_contiguous()
            ):
                if _supports_packed_pointwise_dgrad(
                    plan,
                    loss_shape,
                    weight_shape,
                ):
                    cin_per_group = weight_shape[1]
                    cout_per_group = weight_shape[0] // plan.groups
                    return torch.empty(
                        (
                            plan.groups,
                            cin_per_group,
                            cout_per_group,
                        ),
                        device=inputs[0].device,
                        dtype=output_dtype,
                    )
                return None
            n = loss_shape[0]
            cin_per_group = weight_shape[1]
            kernel_volume = weight_shape[2] * weight_shape[3] * weight_shape[4]
            loss_spatial = loss_shape[2] * loss_shape[3] * loss_shape[4]
            cout_per_group = weight_shape[0] // plan.groups
            matrix_count = n * plan.groups * kernel_volume
            dgrad_image_shape = (
                int(plan.output_shape[0]),
                int(plan.output_shape[1]),
                int(plan.output_shape[2]),
                int(plan.output_shape[3]),
                int(plan.output_shape[4]),
            )
            if (
                _supports_fprop_style_3d_dgrad(
                    plan,
                    dgrad_image_shape,
                    loss_shape,
                    weight_shape,
                )
                and inputs[0].is_contiguous()
                and inputs[1].is_contiguous()
            ):
                input_spatial = (
                    dgrad_image_shape[2]
                    * dgrad_image_shape[3]
                    * dgrad_image_shape[4]
                )
                return (
                    torch.empty(
                        (
                            plan.groups * kernel_volume,
                            cin_per_group,
                            cout_per_group,
                        ),
                        device=inputs[0].device,
                        dtype=output_dtype,
                    ),
                    torch.empty(
                        (
                            n,
                            cout_per_group * kernel_volume,
                            input_spatial,
                        ),
                        device=inputs[0].device,
                        dtype=output_dtype,
                    ),
                )
            return (
                torch.empty(
                    (
                        plan.groups * kernel_volume,
                        cin_per_group,
                        cout_per_group,
                    ),
                    device=inputs[0].device,
                    dtype=output_dtype,
                ),
                torch.empty(
                    (matrix_count, cin_per_group, loss_spatial),
                    device=inputs[0].device,
                    dtype=output_dtype,
                ),
            )
        if plan.op_type != "conv_wgrad":
            return None
        image_shape, _ = _normalized_tensor(inputs[0], plan.rank)
        loss_shape, _ = _normalized_tensor(inputs[1], plan.rank)
        n = image_shape[0]
        c_out = int(plan.output_shape[0])
        cin_per_group = int(plan.output_shape[1])
        kernel = _normalized_spatial(
            plan.output_shape[-plan.rank :],
            1,
        )
        _, _, od, oh, ow = loss_shape
        cout_per_group = c_out // plan.groups
        kernel_volume = kernel[0] * kernel[1] * kernel[2]
        weight_shape = (
            c_out,
            cin_per_group,
            kernel[0],
            kernel[1],
            kernel[2],
        )
        if (
            _is_pointwise_conv(plan, weight_shape)
            and inputs[0].is_contiguous()
            and inputs[1].is_contiguous()
        ):
            if n == 1:
                return None
            return torch.empty(
                (n * c_out * cin_per_group,),
                device=inputs[0].device,
                dtype=torch.float32,
            )
        if inputs[0].is_contiguous() and inputs[1].is_contiguous():
            reduction = n * od * oh * ow
            columns = cin_per_group * kernel_volume
            packed_loss = torch.empty(
                (plan.groups, cout_per_group, reduction),
                device=inputs[0].device,
                dtype=output_dtype,
            )
            if _supports_full_volume_3d_wgrad(
                plan,
                image_shape,
                loss_shape,
                weight_shape,
            ):
                num_splits = _ceil_div(reduction, 512)
                return (
                    packed_loss,
                    torch.empty(
                        (plan.groups, columns, reduction),
                        device=inputs[0].device,
                        dtype=output_dtype,
                    ),
                    torch.empty(
                        (
                            num_splits,
                            c_out,
                            columns,
                        ),
                        device=inputs[0].device,
                        dtype=torch.float32,
                    ),
                )
            if _supports_stem_stride2_2d_wgrad(
                plan,
                image_shape,
                loss_shape,
                weight_shape,
            ):
                num_splits = _ceil_div(reduction, 2048)
                return (
                    packed_loss,
                    torch.empty(
                        (plan.groups, columns, reduction),
                        device=inputs[0].device,
                        dtype=output_dtype,
                    ),
                    torch.empty(
                        (
                            num_splits,
                            c_out,
                            columns,
                        ),
                        device=inputs[0].device,
                        dtype=torch.float32,
                    ),
                )
            if (
                _supports_full_plane_2d_wgrad(
                    plan,
                    image_shape,
                    loss_shape,
                    weight_shape,
                )
                or _supports_row_tile_2d_wgrad(
                    plan,
                    image_shape,
                    loss_shape,
                    weight_shape,
                )
                or _supports_deep_stride2_2d_wgrad(
                    plan,
                    image_shape,
                    loss_shape,
                    weight_shape,
                )
            ):
                return (
                    packed_loss,
                    torch.empty(
                        (plan.groups, columns, reduction),
                        device=inputs[0].device,
                        dtype=output_dtype,
                    ),
                )
            if _supports_fast_conv_1d_pack(
                plan,
                image_shape[4],
            ):
                return (
                    packed_loss,
                    torch.empty(
                        (plan.groups, columns, reduction),
                        device=inputs[0].device,
                        dtype=output_dtype,
                    ),
                    torch.empty(
                        (plan.groups, cout_per_group, columns),
                        device=inputs[0].device,
                        dtype=output_dtype,
                    ),
                )
            return (
                packed_loss,
                torch.empty(
                    (plan.groups, reduction, columns),
                    device=inputs[0].device,
                    dtype=output_dtype,
                ),
            )
        _, _, _, num_splits, kernel_volume = _wgrad_launch_config(
            n=n,
            c_out=c_out,
            cin_per_group=cin_per_group,
            cout_per_group=cout_per_group,
            kd=kernel[0],
            kh=kernel[1],
            kw=kernel[2],
            od=od,
            oh=oh,
            ow=ow,
            groups=plan.groups,
            fp32=output_dtype == torch.float32,
        )
        if num_splits == 1:
            return None
        return torch.empty(
            (num_splits * c_out * cin_per_group * kernel_volume,),
            device=inputs[0].device,
            dtype=torch.float32,
        )

    def execute(
        inputs: Sequence[Any],
        output: Optional[torch.Tensor] = None,
        workspace: Optional[Any] = None,
    ) -> torch.Tensor:
        if output is None:
            output = make_output(inputs)
        if (
            plan.op_type in ("conv_fprop", "conv_dgrad", "conv_wgrad")
            and workspace is None
        ):
            workspace = make_workspace(inputs)
        return _launch(
            plan,
            inputs,
            output,
            workspace,
        )

    def run(inputs: Sequence[Any], run_attrs: dict[str, Any]) -> Any:
        if not can_run(inputs):
            return default_run_fn(inputs, run_attrs)
        return execute(inputs)

    def bind(
        inputs: Sequence[Any], run_attrs: dict[str, Any]
    ) -> Callable[[], Any]:
        if not can_run(inputs):
            return lambda: default_run_fn(inputs, run_attrs)
        output = make_output(inputs)
        workspace = make_workspace(inputs)
        return lambda: execute(inputs, output, workspace)

    # ``run`` allocates a fresh result for every functional invocation, so the
    # prepared-graph ownership wrapper must not route it back to the portable
    # default implementation (which intentionally remains CUDA-only).
    setattr(run, "_flagdnn_functional_output_safe", True)
    setattr(run, "bind", bind)
    return run


__all__ = ("prepare_conv",)
