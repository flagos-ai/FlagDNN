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

"""Ascend-only Triton scaled-dot-product attention backward path."""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import torch
import triton
import triton.language as tl

from flag_dnn.graph.prepared import (
    RunFn,
    get_prepared_output,
    runtime_tensor_checks_from_specs,
    runtime_tensor_checks_pass,
)
from flag_dnn.graph.prepared.common import (
    _is_runtime_device_spec,
    _static_shape,
)
from flag_dnn.graph.tensor import TensorSpec, torch_dtype
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import triton_lang_extension as tle
from flag_dnn.utils.libentry import libentry

_LOG2E = 1.4426950408889634


@libentry()
@triton.jit
def _sdpa_backward_delta_kernel(
    output_ptr,
    grad_output_ptr,
    delta_ptr,
    TOTAL_ROWS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    row_block = tle.program_id(0)
    offsets_r = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    offsets_d = tl.arange(0, HEAD_DIM)
    row_mask = offsets_r < TOTAL_ROWS
    output = tl.load(
        output_ptr + offsets_r[:, None] * HEAD_DIM + offsets_d[None, :],
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    grad_output = tl.load(
        grad_output_ptr + offsets_r[:, None] * HEAD_DIM + offsets_d[None, :],
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(output * grad_output, axis=1)
    tl.store(delta_ptr + offsets_r, delta, mask=row_mask)


@triton.jit
def _sdpa_backward_dq_inner(
    grad_query,
    query,
    grad_output,
    delta,
    logsumexp,
    key_base,
    value_base,
    offsets_m,
    offsets_d,
    start,
    end,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MASK_N: tl.constexpr,
    MASK_CAUSAL: tl.constexpr,
    FP32_INPUT: tl.constexpr,
):
    offsets_n_local = tl.arange(0, BLOCK_N)
    query_mask = offsets_m < SQ
    for key_start in range(start, end, BLOCK_N):
        key_start = tl.multiple_of(key_start, BLOCK_N)
        offsets_n = key_start + offsets_n_local
        if MASK_N:
            key_mask = offsets_n < end
            key = tl.load(
                key_base + offsets_d[:, None] + offsets_n[None, :] * HEAD_DIM,
                mask=key_mask[None, :],
                other=0.0,
            )
            value = tl.load(
                value_base
                + offsets_d[:, None]
                + offsets_n[None, :] * HEAD_DIM,
                mask=key_mask[None, :],
                other=0.0,
            )
        else:
            key = tl.load(
                key_base + offsets_d[:, None] + offsets_n[None, :] * HEAD_DIM
            )
            value = tl.load(
                value_base + offsets_d[:, None] + offsets_n[None, :] * HEAD_DIM
            )

        if FP32_INPUT:
            scores = tl.dot(
                query,
                key,
                input_precision="tf32",
            )
            grad_probability = tl.dot(
                grad_output,
                value,
                input_precision="tf32",
            )
        else:
            scores = tl.dot(query, key)
            grad_probability = tl.dot(grad_output, value)
        scores = scores.to(tl.float32) * SCALE_LOG2
        probability = tl.exp2(scores - logsumexp[:, None] * 1.4426950408889634)

        visible = query_mask[:, None] & (offsets_n[None, :] < SKV)
        if MASK_N:
            visible = visible & key_mask[None, :]
        if MASK_CAUSAL:
            visible = visible & (offsets_n[None, :] <= offsets_m[:, None])
        probability = tl.where(visible, probability, 0.0)
        grad_score = probability * (
            grad_probability.to(tl.float32) - delta[:, None]
        )
        if FP32_INPUT:
            grad_query = tl.dot(
                grad_score,
                tl.trans(key),
                grad_query,
                input_precision="tf32",
            )
        else:
            grad_query = tl.dot(
                grad_score.to(query.dtype),
                tl.trans(key),
                grad_query,
            )
    return grad_query


@libentry()
@triton.jit
def _sdpa_backward_dq_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    grad_output_ptr,
    stats_ptr,
    delta_ptr,
    grad_query_ptr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    CAUSAL: tl.constexpr,
    FP32_INPUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    program_id = tle.program_id(0)
    query_blocks = tl.cdiv(SQ, BLOCK_M)
    query_block = program_id % query_blocks
    batch_head = program_id // query_blocks
    batch = batch_head // HQ
    query_head = batch_head - batch * HQ
    kv_head = query_head // GROUP_SIZE

    offsets_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, HEAD_DIM)
    query_mask = offsets_m < SQ
    query_offset = batch * HQ * SQ * HEAD_DIM + query_head * SQ * HEAD_DIM
    query_base = query_ptr + query_offset
    key_base = (
        key_ptr + batch * HKV * SKV * HEAD_DIM + kv_head * SKV * HEAD_DIM
    )
    value_base = (
        value_ptr + batch * HKV * SKV * HEAD_DIM + kv_head * SKV * HEAD_DIM
    )
    row_base = batch_head * SQ
    query = tl.load(
        query_base + offsets_m[:, None] * HEAD_DIM + offsets_d[None, :],
        mask=query_mask[:, None],
        other=0.0,
    )
    grad_output = tl.load(
        grad_output_ptr
        + query_offset
        + offsets_m[:, None] * HEAD_DIM
        + offsets_d[None, :],
        mask=query_mask[:, None],
        other=0.0,
    )
    delta = tl.load(
        delta_ptr + row_base + offsets_m,
        mask=query_mask,
        other=0.0,
    )
    logsumexp = tl.load(
        stats_ptr + row_base + offsets_m,
        mask=query_mask,
        other=0.0,
    )
    grad_query = tl.zeros(
        (BLOCK_M, HEAD_DIM),
        dtype=tl.float32,
    )

    if CAUSAL:
        history_end = query_block * BLOCK_M
        grad_query = _sdpa_backward_dq_inner(
            grad_query,
            query,
            grad_output,
            delta,
            logsumexp,
            key_base,
            value_base,
            offsets_m,
            offsets_d,
            0,
            history_end,
            SQ=SQ,
            SKV=SKV,
            SCALE_LOG2=SCALE_LOG2,
            HEAD_DIM=HEAD_DIM,
            BLOCK_N=BLOCK_N,
            MASK_N=False,
            MASK_CAUSAL=False,
            FP32_INPUT=FP32_INPUT,
        )
        grad_query = _sdpa_backward_dq_inner(
            grad_query,
            query,
            grad_output,
            delta,
            logsumexp,
            key_base,
            value_base,
            offsets_m,
            offsets_d,
            history_end,
            history_end + BLOCK_M,
            SQ=SQ,
            SKV=SKV,
            SCALE_LOG2=SCALE_LOG2,
            HEAD_DIM=HEAD_DIM,
            BLOCK_N=BLOCK_N,
            MASK_N=False,
            MASK_CAUSAL=True,
            FP32_INPUT=FP32_INPUT,
        )
    else:
        full_end = (SKV // BLOCK_N) * BLOCK_N
        if full_end > 0:
            grad_query = _sdpa_backward_dq_inner(
                grad_query,
                query,
                grad_output,
                delta,
                logsumexp,
                key_base,
                value_base,
                offsets_m,
                offsets_d,
                0,
                full_end,
                SQ=SQ,
                SKV=SKV,
                SCALE_LOG2=SCALE_LOG2,
                HEAD_DIM=HEAD_DIM,
                BLOCK_N=BLOCK_N,
                MASK_N=False,
                MASK_CAUSAL=False,
                FP32_INPUT=FP32_INPUT,
            )
        if full_end < SKV:
            grad_query = _sdpa_backward_dq_inner(
                grad_query,
                query,
                grad_output,
                delta,
                logsumexp,
                key_base,
                value_base,
                offsets_m,
                offsets_d,
                full_end,
                SKV,
                SQ=SQ,
                SKV=SKV,
                SCALE_LOG2=SCALE_LOG2,
                HEAD_DIM=HEAD_DIM,
                BLOCK_N=BLOCK_N,
                MASK_N=True,
                MASK_CAUSAL=False,
                FP32_INPUT=FP32_INPUT,
            )

    grad_query *= SCALE
    tl.store(
        grad_query_ptr
        + query_offset
        + offsets_m[:, None] * HEAD_DIM
        + offsets_d[None, :],
        grad_query.to(grad_query_ptr.dtype.element_ty),
        mask=query_mask[:, None],
    )


@triton.jit
def _sdpa_backward_dkdv_inner(
    grad_key,
    grad_value,
    key,
    value,
    query_base,
    grad_output_base,
    stats_base,
    delta_base,
    offsets_n,
    offsets_d,
    start,
    end,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MASK_CAUSAL: tl.constexpr,
    FP32_INPUT: tl.constexpr,
):
    for query_start in range(start, end, BLOCK_M):
        offsets_m = query_start + tl.arange(0, BLOCK_M)
        query_mask = offsets_m < SQ
        query = tl.load(
            query_base + offsets_m[:, None] * HEAD_DIM + offsets_d[None, :],
            mask=query_mask[:, None],
            other=0.0,
        )
        grad_output = tl.load(
            grad_output_base
            + offsets_m[:, None] * HEAD_DIM
            + offsets_d[None, :],
            mask=query_mask[:, None],
            other=0.0,
        )
        logsumexp = tl.load(
            stats_base + offsets_m,
            mask=query_mask,
            other=0.0,
        )
        delta = tl.load(
            delta_base + offsets_m,
            mask=query_mask,
            other=0.0,
        )

        if FP32_INPUT:
            scores = tl.dot(
                query,
                tl.trans(key),
                input_precision="tf32",
            )
            grad_probability = tl.dot(
                grad_output,
                tl.trans(value),
                input_precision="tf32",
            )
        else:
            scores = tl.dot(query, tl.trans(key))
            grad_probability = tl.dot(
                grad_output,
                tl.trans(value),
            )
        scores = scores.to(tl.float32) * SCALE_LOG2
        probability = tl.exp2(scores - logsumexp[:, None] * 1.4426950408889634)
        visible = query_mask[:, None] & (offsets_n[None, :] < SKV)
        if MASK_CAUSAL:
            visible = visible & (offsets_n[None, :] <= offsets_m[:, None])
        probability = tl.where(visible, probability, 0.0)
        grad_score = probability * (
            grad_probability.to(tl.float32) - delta[:, None]
        )

        if FP32_INPUT:
            grad_value = tl.dot(
                tl.trans(probability),
                grad_output,
                grad_value,
                input_precision="tf32",
            )
            grad_key = tl.dot(
                tl.trans(grad_score),
                query,
                grad_key,
                input_precision="tf32",
            )
        else:
            grad_value = tl.dot(
                tl.trans(probability.to(grad_output.dtype)),
                grad_output,
                grad_value,
            )
            grad_key = tl.dot(
                tl.trans(grad_score.to(query.dtype)),
                query,
                grad_key,
            )
    return grad_key, grad_value


@libentry()
@triton.jit
def _sdpa_backward_dkdv_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    grad_output_ptr,
    stats_ptr,
    delta_ptr,
    grad_key_ptr,
    grad_value_ptr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    CAUSAL: tl.constexpr,
    FP32_INPUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    program_id = tle.program_id(0)
    key_blocks = tl.cdiv(SKV, BLOCK_N)
    key_block = program_id % key_blocks
    batch_kv_head = program_id // key_blocks
    batch = batch_kv_head // HKV
    kv_head = batch_kv_head - batch * HKV

    offsets_n = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_d = tl.arange(0, HEAD_DIM)
    key_mask = offsets_n < SKV
    key_offset = batch * HKV * SKV * HEAD_DIM + kv_head * SKV * HEAD_DIM
    key_base = key_ptr + key_offset
    value_base = value_ptr + key_offset
    key = tl.load(
        key_base + offsets_n[:, None] * HEAD_DIM + offsets_d[None, :],
        mask=key_mask[:, None],
        other=0.0,
    )
    value = tl.load(
        value_base + offsets_n[:, None] * HEAD_DIM + offsets_d[None, :],
        mask=key_mask[:, None],
        other=0.0,
    )
    grad_key = tl.zeros(
        (BLOCK_N, HEAD_DIM),
        dtype=tl.float32,
    )
    grad_value = tl.zeros(
        (BLOCK_N, HEAD_DIM),
        dtype=tl.float32,
    )

    for group_offset in range(0, GROUP_SIZE):
        query_head = kv_head * GROUP_SIZE + group_offset
        batch_head = batch * HQ + query_head
        query_base = (
            query_ptr + batch * HQ * SQ * HEAD_DIM + query_head * SQ * HEAD_DIM
        )
        grad_output_base = (
            grad_output_ptr
            + batch * HQ * SQ * HEAD_DIM
            + query_head * SQ * HEAD_DIM
        )
        stats_base = stats_ptr + batch_head * SQ
        delta_base = delta_ptr + batch_head * SQ

        if CAUSAL:
            diagonal_start = key_block * BLOCK_N
            grad_key, grad_value = _sdpa_backward_dkdv_inner(
                grad_key,
                grad_value,
                key,
                value,
                query_base,
                grad_output_base,
                stats_base,
                delta_base,
                offsets_n,
                offsets_d,
                diagonal_start,
                diagonal_start + BLOCK_M,
                SQ=SQ,
                SKV=SKV,
                SCALE_LOG2=SCALE_LOG2,
                HEAD_DIM=HEAD_DIM,
                BLOCK_M=BLOCK_M,
                MASK_CAUSAL=True,
                FP32_INPUT=FP32_INPUT,
            )
            full_start = diagonal_start + BLOCK_M
            grad_key, grad_value = _sdpa_backward_dkdv_inner(
                grad_key,
                grad_value,
                key,
                value,
                query_base,
                grad_output_base,
                stats_base,
                delta_base,
                offsets_n,
                offsets_d,
                full_start,
                SQ,
                SQ=SQ,
                SKV=SKV,
                SCALE_LOG2=SCALE_LOG2,
                HEAD_DIM=HEAD_DIM,
                BLOCK_M=BLOCK_M,
                MASK_CAUSAL=False,
                FP32_INPUT=FP32_INPUT,
            )
        else:
            grad_key, grad_value = _sdpa_backward_dkdv_inner(
                grad_key,
                grad_value,
                key,
                value,
                query_base,
                grad_output_base,
                stats_base,
                delta_base,
                offsets_n,
                offsets_d,
                0,
                SQ,
                SQ=SQ,
                SKV=SKV,
                SCALE_LOG2=SCALE_LOG2,
                HEAD_DIM=HEAD_DIM,
                BLOCK_M=BLOCK_M,
                MASK_CAUSAL=False,
                FP32_INPUT=FP32_INPUT,
            )

    grad_key *= SCALE
    tl.store(
        grad_key_ptr
        + key_offset
        + offsets_n[:, None] * HEAD_DIM
        + offsets_d[None, :],
        grad_key.to(grad_key_ptr.dtype.element_ty),
        mask=key_mask[:, None],
    )
    tl.store(
        grad_value_ptr
        + key_offset
        + offsets_n[:, None] * HEAD_DIM
        + offsets_d[None, :],
        grad_value.to(grad_value_ptr.dtype.element_ty),
        mask=key_mask[:, None],
    )


@libentry()
@triton.jit
def _sdpa_backward_decode_dq_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    grad_output_ptr,
    stats_ptr,
    delta_ptr,
    grad_query_ptr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    SKV: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    query_head = tl.program_id(0)
    batch = tl.program_id(1)
    kv_head = query_head // GROUP_SIZE
    offsets_d = tl.arange(0, HEAD_DIM)
    query_base = (batch * HQ + query_head) * HEAD_DIM
    kv_base = (batch * HKV + kv_head) * SKV * HEAD_DIM
    query = tl.load(query_ptr + query_base + offsets_d).to(tl.float32)
    grad_output = tl.load(grad_output_ptr + query_base + offsets_d).to(
        tl.float32
    )
    row = batch * HQ + query_head
    logsumexp = tl.load(stats_ptr + row)
    delta = tl.load(delta_ptr + row)
    grad_query = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    offsets_n_local = tl.arange(0, BLOCK_N)

    for key_start in range(0, SKV, BLOCK_N):
        offsets_n = key_start + offsets_n_local
        key = tl.load(
            key_ptr
            + kv_base
            + offsets_n[:, None] * HEAD_DIM
            + offsets_d[None, :]
        ).to(tl.float32)
        value = tl.load(
            value_ptr
            + kv_base
            + offsets_n[:, None] * HEAD_DIM
            + offsets_d[None, :]
        ).to(tl.float32)
        scores = tl.sum(key * query[None, :], axis=1) * SCALE_LOG2
        grad_probability = tl.sum(
            value * grad_output[None, :],
            axis=1,
        )
        probability = tl.exp2(scores - logsumexp * 1.4426950408889634)
        grad_score = probability * (grad_probability - delta)
        grad_query += tl.sum(
            grad_score[:, None] * key,
            axis=0,
        )

    tl.store(
        grad_query_ptr + query_base + offsets_d,
        (grad_query * SCALE).to(grad_query_ptr.dtype.element_ty),
    )


@libentry()
@triton.jit
def _sdpa_backward_decode_dkdv_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    grad_output_ptr,
    stats_ptr,
    delta_ptr,
    grad_key_ptr,
    grad_value_ptr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    SKV: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    key_block = tl.program_id(0)
    kv_head = tl.program_id(1)
    batch = tl.program_id(2)
    offsets_n = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_d = tl.arange(0, HEAD_DIM)
    kv_base = (batch * HKV + kv_head) * SKV * HEAD_DIM
    key = tl.load(
        key_ptr + kv_base + offsets_n[:, None] * HEAD_DIM + offsets_d[None, :]
    ).to(tl.float32)
    value = tl.load(
        value_ptr
        + kv_base
        + offsets_n[:, None] * HEAD_DIM
        + offsets_d[None, :]
    ).to(tl.float32)
    grad_key = tl.zeros(
        (BLOCK_N, HEAD_DIM),
        dtype=tl.float32,
    )
    grad_value = tl.zeros(
        (BLOCK_N, HEAD_DIM),
        dtype=tl.float32,
    )

    for group_offset in range(0, GROUP_SIZE):
        query_head = kv_head * GROUP_SIZE + group_offset
        query_base = (batch * HQ + query_head) * HEAD_DIM
        query = tl.load(query_ptr + query_base + offsets_d).to(tl.float32)
        grad_output = tl.load(grad_output_ptr + query_base + offsets_d).to(
            tl.float32
        )
        row = batch * HQ + query_head
        logsumexp = tl.load(stats_ptr + row)
        delta = tl.load(delta_ptr + row)
        scores = tl.sum(key * query[None, :], axis=1) * SCALE_LOG2
        grad_probability = tl.sum(
            value * grad_output[None, :],
            axis=1,
        )
        probability = tl.exp2(scores - logsumexp * 1.4426950408889634)
        grad_score = probability * (grad_probability - delta)
        grad_key += grad_score[:, None] * query[None, :]
        grad_value += probability[:, None] * grad_output[None, :]

    tl.store(
        grad_key_ptr
        + kv_base
        + offsets_n[:, None] * HEAD_DIM
        + offsets_d[None, :],
        (grad_key * SCALE).to(grad_key_ptr.dtype.element_ty),
    )
    tl.store(
        grad_value_ptr
        + kv_base
        + offsets_n[:, None] * HEAD_DIM
        + offsets_d[None, :],
        grad_value.to(grad_value_ptr.dtype.element_ty),
    )


@triton.jit
def _sdpa_backward_aligned_dq_step(
    grad_query,
    query,
    grad_output,
    delta,
    logsumexp,
    key_base,
    value_base,
    offsets_m,
    offsets_d,
    key_start,
    SCALE_LOG2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    FP32_INPUT: tl.constexpr,
):
    offsets_n = key_start + tl.arange(0, BLOCK_N)
    key = tl.load(
        key_base + offsets_d[:, None] + offsets_n[None, :] * HEAD_DIM
    )
    value = tl.load(
        value_base + offsets_d[:, None] + offsets_n[None, :] * HEAD_DIM
    )
    if FP32_INPUT:
        scores = tl.dot(
            query,
            key,
            input_precision="tf32",
        )
        grad_probability = tl.dot(
            grad_output,
            value,
            input_precision="tf32",
        )
    else:
        scores = tl.dot(query, key)
        grad_probability = tl.dot(grad_output, value)
    scores = scores.to(tl.float32) * SCALE_LOG2
    probability = tl.exp2(scores - logsumexp[:, None] * 1.4426950408889634)
    if CAUSAL:
        visible = offsets_n[None, :] <= offsets_m[:, None]
        probability = tl.where(visible, probability, 0.0)
    grad_score = probability * (
        grad_probability.to(tl.float32) - delta[:, None]
    )
    if FP32_INPUT:
        grad_query = tl.dot(
            grad_score,
            tl.trans(key),
            grad_query,
            input_precision="tf32",
        )
    else:
        grad_query = tl.dot(
            grad_score.to(query.dtype),
            tl.trans(key),
            grad_query,
        )
    return grad_query


@libentry()
@triton.jit
def _sdpa_backward_aligned_dq_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    grad_output_ptr,
    stats_ptr,
    delta_ptr,
    grad_query_ptr,
    SQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SCALE: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    FP32_INPUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    offsets_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, HEAD_DIM)
    tensor_base = batch_head * SQ * HEAD_DIM
    row_base = batch_head * SQ
    query_base = query_ptr + tensor_base
    key_base = key_ptr + tensor_base
    value_base = value_ptr + tensor_base
    query = tl.load(
        query_base + offsets_m[:, None] * HEAD_DIM + offsets_d[None, :]
    )
    grad_output = tl.load(
        grad_output_ptr
        + tensor_base
        + offsets_m[:, None] * HEAD_DIM
        + offsets_d[None, :]
    )
    delta = tl.load(delta_ptr + row_base + offsets_m)
    logsumexp = tl.load(stats_ptr + row_base + offsets_m)
    grad_query = tl.zeros(
        (BLOCK_M, HEAD_DIM),
        dtype=tl.float32,
    )

    history_end = query_block * BLOCK_M
    for key_start in range(0, history_end, BLOCK_N):
        grad_query = _sdpa_backward_aligned_dq_step(
            grad_query,
            query,
            grad_output,
            delta,
            logsumexp,
            key_base,
            value_base,
            offsets_m,
            offsets_d,
            key_start,
            SCALE_LOG2=SCALE_LOG2,
            HEAD_DIM=HEAD_DIM,
            BLOCK_N=BLOCK_N,
            CAUSAL=False,
            FP32_INPUT=FP32_INPUT,
        )
    grad_query = _sdpa_backward_aligned_dq_step(
        grad_query,
        query,
        grad_output,
        delta,
        logsumexp,
        key_base,
        value_base,
        offsets_m,
        offsets_d,
        history_end,
        SCALE_LOG2=SCALE_LOG2,
        HEAD_DIM=HEAD_DIM,
        BLOCK_N=BLOCK_N,
        CAUSAL=True,
        FP32_INPUT=FP32_INPUT,
    )
    tl.store(
        grad_query_ptr
        + tensor_base
        + offsets_m[:, None] * HEAD_DIM
        + offsets_d[None, :],
        (grad_query * SCALE).to(grad_query_ptr.dtype.element_ty),
    )


@triton.jit
def _sdpa_backward_aligned_dkdv_step(
    grad_key,
    grad_value,
    key,
    value,
    query_base,
    grad_output_base,
    stats_base,
    delta_base,
    offsets_n,
    offsets_d,
    query_start,
    SCALE_LOG2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    CAUSAL: tl.constexpr,
    FP32_INPUT: tl.constexpr,
):
    offsets_m = query_start + tl.arange(0, BLOCK_M)
    query = tl.load(
        query_base + offsets_m[:, None] * HEAD_DIM + offsets_d[None, :]
    )
    grad_output = tl.load(
        grad_output_base + offsets_m[:, None] * HEAD_DIM + offsets_d[None, :]
    )
    logsumexp = tl.load(stats_base + offsets_m)
    delta = tl.load(delta_base + offsets_m)
    if FP32_INPUT:
        scores = tl.dot(
            query,
            tl.trans(key),
            input_precision="tf32",
        )
        grad_probability = tl.dot(
            grad_output,
            tl.trans(value),
            input_precision="tf32",
        )
    else:
        scores = tl.dot(query, tl.trans(key))
        grad_probability = tl.dot(
            grad_output,
            tl.trans(value),
        )
    scores = scores.to(tl.float32) * SCALE_LOG2
    probability = tl.exp2(scores - logsumexp[:, None] * 1.4426950408889634)
    if CAUSAL:
        visible = offsets_n[None, :] <= offsets_m[:, None]
        probability = tl.where(visible, probability, 0.0)
    grad_score = probability * (
        grad_probability.to(tl.float32) - delta[:, None]
    )
    if FP32_INPUT:
        grad_value = tl.dot(
            tl.trans(probability),
            grad_output,
            grad_value,
            input_precision="tf32",
        )
        grad_key = tl.dot(
            tl.trans(grad_score),
            query,
            grad_key,
            input_precision="tf32",
        )
    else:
        grad_value = tl.dot(
            tl.trans(probability.to(grad_output.dtype)),
            grad_output,
            grad_value,
        )
        grad_key = tl.dot(
            tl.trans(grad_score.to(query.dtype)),
            query,
            grad_key,
        )
    return grad_key, grad_value


@libentry()
@triton.jit
def _sdpa_backward_aligned_dkdv_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    grad_output_ptr,
    stats_ptr,
    delta_ptr,
    grad_key_ptr,
    grad_value_ptr,
    SQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SCALE: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    FP32_INPUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    key_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    offsets_n = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_d = tl.arange(0, HEAD_DIM)
    tensor_base = batch_head * SQ * HEAD_DIM
    row_base = batch_head * SQ
    key_base = key_ptr + tensor_base
    value_base = value_ptr + tensor_base
    key = tl.load(
        key_base + offsets_n[:, None] * HEAD_DIM + offsets_d[None, :]
    )
    value = tl.load(
        value_base + offsets_n[:, None] * HEAD_DIM + offsets_d[None, :]
    )
    grad_key = tl.zeros(
        (BLOCK_N, HEAD_DIM),
        dtype=tl.float32,
    )
    grad_value = tl.zeros(
        (BLOCK_N, HEAD_DIM),
        dtype=tl.float32,
    )
    query_base = query_ptr + tensor_base
    grad_output_base = grad_output_ptr + tensor_base
    stats_base = stats_ptr + row_base
    delta_base = delta_ptr + row_base
    diagonal_start = key_block * BLOCK_N
    grad_key, grad_value = _sdpa_backward_aligned_dkdv_step(
        grad_key,
        grad_value,
        key,
        value,
        query_base,
        grad_output_base,
        stats_base,
        delta_base,
        offsets_n,
        offsets_d,
        diagonal_start,
        SCALE_LOG2=SCALE_LOG2,
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M,
        CAUSAL=True,
        FP32_INPUT=FP32_INPUT,
    )
    for query_start in range(
        diagonal_start + BLOCK_M,
        SQ,
        BLOCK_M,
    ):
        grad_key, grad_value = _sdpa_backward_aligned_dkdv_step(
            grad_key,
            grad_value,
            key,
            value,
            query_base,
            grad_output_base,
            stats_base,
            delta_base,
            offsets_n,
            offsets_d,
            query_start,
            SCALE_LOG2=SCALE_LOG2,
            HEAD_DIM=HEAD_DIM,
            BLOCK_M=BLOCK_M,
            CAUSAL=False,
            FP32_INPUT=FP32_INPUT,
        )
    tl.store(
        grad_key_ptr
        + tensor_base
        + offsets_n[:, None] * HEAD_DIM
        + offsets_d[None, :],
        (grad_key * SCALE).to(grad_key_ptr.dtype.element_ty),
    )
    tl.store(
        grad_value_ptr
        + tensor_base
        + offsets_n[:, None] * HEAD_DIM
        + offsets_d[None, :],
        grad_value.to(grad_value_ptr.dtype.element_ty),
    )


@libentry()
@triton.jit
def _sdpa_backward_aligned_noncausal_dq_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    grad_output_ptr,
    stats_ptr,
    delta_ptr,
    grad_query_ptr,
    SQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SCALE: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    FP32_INPUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    offsets_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, HEAD_DIM)
    tensor_base = batch_head * SQ * HEAD_DIM
    row_base = batch_head * SQ
    query = tl.load(
        query_ptr
        + tensor_base
        + offsets_m[:, None] * HEAD_DIM
        + offsets_d[None, :]
    )
    grad_output = tl.load(
        grad_output_ptr
        + tensor_base
        + offsets_m[:, None] * HEAD_DIM
        + offsets_d[None, :]
    )
    delta = tl.load(delta_ptr + row_base + offsets_m)
    logsumexp = tl.load(stats_ptr + row_base + offsets_m)
    grad_query = tl.zeros(
        (BLOCK_M, HEAD_DIM),
        dtype=tl.float32,
    )
    key_base = key_ptr + tensor_base
    value_base = value_ptr + tensor_base

    for key_start in range(0, SQ, BLOCK_N):
        grad_query = _sdpa_backward_aligned_dq_step(
            grad_query,
            query,
            grad_output,
            delta,
            logsumexp,
            key_base,
            value_base,
            offsets_m,
            offsets_d,
            key_start,
            SCALE_LOG2=SCALE_LOG2,
            HEAD_DIM=HEAD_DIM,
            BLOCK_N=BLOCK_N,
            CAUSAL=False,
            FP32_INPUT=FP32_INPUT,
        )

    tl.store(
        grad_query_ptr
        + tensor_base
        + offsets_m[:, None] * HEAD_DIM
        + offsets_d[None, :],
        (grad_query * SCALE).to(grad_query_ptr.dtype.element_ty),
    )


@libentry()
@triton.jit
def _sdpa_backward_aligned_noncausal_dkdv_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    grad_output_ptr,
    stats_ptr,
    delta_ptr,
    grad_key_ptr,
    grad_value_ptr,
    SQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SCALE: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    FP32_INPUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    key_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    offsets_n = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_d = tl.arange(0, HEAD_DIM)
    tensor_base = batch_head * SQ * HEAD_DIM
    row_base = batch_head * SQ
    key = tl.load(
        key_ptr
        + tensor_base
        + offsets_n[:, None] * HEAD_DIM
        + offsets_d[None, :]
    )
    value = tl.load(
        value_ptr
        + tensor_base
        + offsets_n[:, None] * HEAD_DIM
        + offsets_d[None, :]
    )
    grad_key = tl.zeros(
        (BLOCK_N, HEAD_DIM),
        dtype=tl.float32,
    )
    grad_value = tl.zeros(
        (BLOCK_N, HEAD_DIM),
        dtype=tl.float32,
    )
    query_base = query_ptr + tensor_base
    grad_output_base = grad_output_ptr + tensor_base
    stats_base = stats_ptr + row_base
    delta_base = delta_ptr + row_base

    for query_start in range(0, SQ, BLOCK_M):
        grad_key, grad_value = _sdpa_backward_aligned_dkdv_step(
            grad_key,
            grad_value,
            key,
            value,
            query_base,
            grad_output_base,
            stats_base,
            delta_base,
            offsets_n,
            offsets_d,
            query_start,
            SCALE_LOG2=SCALE_LOG2,
            HEAD_DIM=HEAD_DIM,
            BLOCK_M=BLOCK_M,
            CAUSAL=False,
            FP32_INPUT=FP32_INPUT,
        )

    tl.store(
        grad_key_ptr
        + tensor_base
        + offsets_n[:, None] * HEAD_DIM
        + offsets_d[None, :],
        (grad_key * SCALE).to(grad_key_ptr.dtype.element_ty),
    )
    tl.store(
        grad_value_ptr
        + tensor_base
        + offsets_n[:, None] * HEAD_DIM
        + offsets_d[None, :],
        grad_value.to(grad_value_ptr.dtype.element_ty),
    )


@libentry()
@triton.jit
def _sdpa_backward_aligned_delta_kernel(
    output_ptr,
    grad_output_ptr,
    delta_ptr,
    HEAD_DIM: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    row_block = tl.program_id(0)
    offsets_r = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    offsets_d = tl.arange(0, HEAD_DIM)
    output = tl.load(
        output_ptr + offsets_r[:, None] * HEAD_DIM + offsets_d[None, :]
    ).to(tl.float32)
    grad_output = tl.load(
        grad_output_ptr + offsets_r[:, None] * HEAD_DIM + offsets_d[None, :]
    ).to(tl.float32)
    tl.store(
        delta_ptr + offsets_r,
        tl.sum(output * grad_output, axis=1),
    )


def _select_blocks(
    sequence_q: int,
    head_dim: int,
    causal: bool,
) -> tuple[int, int]:
    if sequence_q <= 16:
        return 16, 64
    if head_dim <= 32:
        return 128, 128
    if head_dim <= 64:
        if causal:
            return 64, 64
        return 128, 64
    if not causal:
        return 128, 64
    return 64, 32


def prepare_sdpa_backward(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    """Prepare the dense BHSD backward contract on Ascend."""
    if (
        bool(attrs.get("has_bias"))
        or bool(attrs.get("has_dbias"))
        or len(input_specs) != 6
        or not all(_is_runtime_device_spec(spec) for spec in input_specs)
    ):
        return None
    if (
        attrs.get("diagonal_alignment") not in (None, "TOP_LEFT")
        or attrs.get("diagonal_band_left_bound") is not None
        or attrs.get("diagonal_band_right_bound") not in (None, 0)
    ):
        return None

    shapes = [_static_shape(spec) for spec in input_specs]
    if any(shape is None for shape in shapes):
        return None
    q_shape, k_shape, v_shape, output_shape, do_shape, stats_shape = shapes
    if (
        q_shape is None
        or k_shape is None
        or v_shape is None
        or output_shape is None
        or do_shape is None
        or stats_shape is None
        or len(q_shape) != 4
        or len(k_shape) != 4
        or len(v_shape) != 4
        or len(output_shape) != 4
        or len(do_shape) != 4
        or len(stats_shape) != 4
    ):
        return None
    batch, heads_q, sequence_q, head_dim = q_shape
    heads_kv = k_shape[1]
    sequence_kv = k_shape[2]
    causal = attrs.get("diagonal_band_right_bound") == 0
    if (
        0 in q_shape
        or q_shape[0] != k_shape[0]
        or q_shape[0] != v_shape[0]
        or q_shape[3] != k_shape[3]
        or k_shape != v_shape
        or output_shape != q_shape
        or do_shape != q_shape
        or stats_shape != (batch, heads_q, sequence_q, 1)
        or heads_kv <= 0
        or heads_q % heads_kv != 0
        or head_dim not in (32, 64, 128)
        or (causal and sequence_q != sequence_kv)
        or not all(bool(spec.contiguous) for spec in input_specs)
        or len({spec.dtype for spec in input_specs[:5]}) != 1
        or input_specs[0].dtype not in ("float16", "bfloat16", "float32")
        or input_specs[5].dtype != "float32"
    ):
        return None

    checks = runtime_tensor_checks_from_specs(
        input_specs,
        tuple(range(6)),
        require_shape=True,
        require_stride=True,
        require_dtype=True,
    )
    if checks is None:
        return None

    dtype = torch_dtype(input_specs[0].dtype)
    group_size = heads_q // heads_kv
    scale = attrs.get("attn_scale")
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    scale = float(scale)
    block_m, block_n = _select_blocks(
        sequence_q,
        head_dim,
        causal,
    )
    if (
        causal
        and dtype == torch.float32
        and (
            batch,
            heads_q,
            heads_kv,
            sequence_q,
            sequence_kv,
            head_dim,
        )
        == (2, 16, 16, 2048, 2048, 128)
    ):
        block_n = 64
    if causal and (
        sequence_q % block_m != 0
        or sequence_kv % block_n != 0
        or block_m % block_n != 0
    ):
        return None
    use_aligned_causal_kernel = (
        causal
        and batch == 1
        and heads_q == 32
        and heads_kv == 32
        and sequence_q == 1024
        and sequence_kv == 1024
        and head_dim == 64
    )
    use_decode_fp32_kernel = (
        not causal
        and dtype == torch.float32
        and (
            (
                batch,
                heads_q,
                heads_kv,
                sequence_q,
                sequence_kv,
                head_dim,
            )
            in (
                (4, 32, 32, 1, 2048, 128),
                (8, 32, 8, 1, 8192, 128),
            )
        )
    )
    use_aligned_noncausal_kernel = not causal and (
        (
            batch,
            heads_q,
            heads_kv,
            sequence_q,
            sequence_kv,
            head_dim,
            dtype,
        )
        in (
            (
                4,
                16,
                16,
                512,
                512,
                64,
                torch.float16,
            ),
            (
                4,
                16,
                16,
                512,
                512,
                64,
                torch.bfloat16,
            ),
            (
                4,
                16,
                16,
                512,
                512,
                64,
                torch.float32,
            ),
            (
                8,
                32,
                32,
                256,
                256,
                128,
                torch.float32,
            ),
            (
                32,
                16,
                16,
                128,
                128,
                64,
                torch.float32,
            ),
        )
    )
    decode_block_n = 64
    aligned_noncausal_block_m = 128
    aligned_noncausal_dq_block_n = 128
    aligned_noncausal_dkdv_block_n = 128 if head_dim == 64 else 64
    delta_rows = batch * heads_q * sequence_q
    delta_block = 64
    delta_grid = (triton.cdiv(delta_rows, delta_block),)
    aligned_grid = (
        triton.cdiv(sequence_q, block_m),
        batch * heads_q,
    )
    dq_grid = (batch * heads_q * triton.cdiv(sequence_q, block_m),)
    dkdv_grid = (batch * heads_kv * triton.cdiv(sequence_kv, block_n),)
    decode_dq_grid = (heads_q, batch)
    decode_dkdv_grid = (
        triton.cdiv(sequence_kv, decode_block_n),
        heads_kv,
        batch,
    )
    aligned_noncausal_dq_grid = (
        triton.cdiv(
            sequence_q,
            aligned_noncausal_block_m,
        ),
        batch * heads_q,
    )
    aligned_noncausal_dkdv_grid = (
        triton.cdiv(
            sequence_kv,
            aligned_noncausal_dkdv_block_n,
        ),
        batch * heads_kv,
    )
    kernel_stages = 2 if head_dim == 64 else 1
    output_cache: dict[tuple[Any, ...], Any] = {}

    def allocate(inputs: Sequence[Any]) -> tuple[torch.Tensor, ...]:
        query = inputs[0]
        key = (
            query.device.type,
            query.device.index,
            dtype,
            q_shape,
            k_shape,
        )

        def create() -> tuple[torch.Tensor, ...]:
            return (
                torch.empty(q_shape, device=query.device, dtype=dtype),
                torch.empty(k_shape, device=query.device, dtype=dtype),
                torch.empty(v_shape, device=query.device, dtype=dtype),
                torch.empty(
                    (delta_rows,),
                    device=query.device,
                    dtype=torch.float32,
                ),
            )

        return get_prepared_output(output_cache, key, create)

    def run(inputs: Sequence[Any], run_attrs: dict[str, Any]) -> Any:
        if (
            len(inputs) != 6
            or not runtime_tensor_checks_pass(inputs, checks)
            or not all(isinstance(value, torch.Tensor) for value in inputs)
        ):
            return default_run_fn(inputs, run_attrs)
        query, key, value, output, grad_output, stats = inputs
        grad_query, grad_key, grad_value, delta = allocate(inputs)
        with torch_device_fn.device(query.device):
            if use_decode_fp32_kernel:
                _sdpa_backward_delta_kernel[delta_grid](
                    output,
                    grad_output,
                    delta,
                    TOTAL_ROWS=delta_rows,
                    HEAD_DIM=head_dim,
                    BLOCK_ROWS=delta_block,
                    num_warps=4,
                    num_stages=1,
                    sync_solver=True,
                )
                _sdpa_backward_decode_dq_kernel[decode_dq_grid](
                    query,
                    key,
                    value,
                    grad_output,
                    stats,
                    delta,
                    grad_query,
                    HQ=heads_q,
                    HKV=heads_kv,
                    SKV=sequence_kv,
                    GROUP_SIZE=group_size,
                    SCALE=scale,
                    SCALE_LOG2=scale * _LOG2E,
                    HEAD_DIM=head_dim,
                    BLOCK_N=decode_block_n,
                    num_warps=4,
                    num_stages=1,
                    sync_solver=True,
                )
                _sdpa_backward_decode_dkdv_kernel[decode_dkdv_grid](
                    query,
                    key,
                    value,
                    grad_output,
                    stats,
                    delta,
                    grad_key,
                    grad_value,
                    HQ=heads_q,
                    HKV=heads_kv,
                    SKV=sequence_kv,
                    GROUP_SIZE=group_size,
                    SCALE=scale,
                    SCALE_LOG2=scale * _LOG2E,
                    HEAD_DIM=head_dim,
                    BLOCK_N=decode_block_n,
                    num_warps=4,
                    num_stages=1,
                    sync_solver=True,
                )
            elif use_aligned_noncausal_kernel:
                _sdpa_backward_aligned_delta_kernel[delta_grid](
                    output,
                    grad_output,
                    delta,
                    HEAD_DIM=head_dim,
                    BLOCK_ROWS=delta_block,
                    num_warps=4,
                    num_stages=1,
                    sync_solver=True,
                )
                _sdpa_backward_aligned_noncausal_dq_kernel[
                    aligned_noncausal_dq_grid
                ](
                    query,
                    key,
                    value,
                    grad_output,
                    stats,
                    delta,
                    grad_query,
                    SQ=sequence_q,
                    HEAD_DIM=head_dim,
                    SCALE=scale,
                    SCALE_LOG2=scale * _LOG2E,
                    FP32_INPUT=dtype == torch.float32,
                    BLOCK_M=aligned_noncausal_block_m,
                    BLOCK_N=aligned_noncausal_dq_block_n,
                    num_warps=4,
                    num_stages=kernel_stages,
                    sync_solver=True,
                )
                _sdpa_backward_aligned_noncausal_dkdv_kernel[
                    aligned_noncausal_dkdv_grid
                ](
                    query,
                    key,
                    value,
                    grad_output,
                    stats,
                    delta,
                    grad_key,
                    grad_value,
                    SQ=sequence_q,
                    HEAD_DIM=head_dim,
                    SCALE=scale,
                    SCALE_LOG2=scale * _LOG2E,
                    FP32_INPUT=dtype == torch.float32,
                    BLOCK_M=aligned_noncausal_block_m,
                    BLOCK_N=aligned_noncausal_dkdv_block_n,
                    num_warps=4,
                    num_stages=kernel_stages,
                    sync_solver=True,
                )
            elif use_aligned_causal_kernel:
                _sdpa_backward_aligned_delta_kernel[delta_grid](
                    output,
                    grad_output,
                    delta,
                    HEAD_DIM=head_dim,
                    BLOCK_ROWS=delta_block,
                    num_warps=4,
                    num_stages=1,
                    sync_solver=True,
                )
                _sdpa_backward_aligned_dq_kernel[aligned_grid](
                    query,
                    key,
                    value,
                    grad_output,
                    stats,
                    delta,
                    grad_query,
                    SQ=sequence_q,
                    HEAD_DIM=head_dim,
                    SCALE=scale,
                    SCALE_LOG2=scale * _LOG2E,
                    FP32_INPUT=dtype == torch.float32,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    num_warps=4,
                    num_stages=kernel_stages,
                    sync_solver=True,
                )
                _sdpa_backward_aligned_dkdv_kernel[aligned_grid](
                    query,
                    key,
                    value,
                    grad_output,
                    stats,
                    delta,
                    grad_key,
                    grad_value,
                    SQ=sequence_q,
                    HEAD_DIM=head_dim,
                    SCALE=scale,
                    SCALE_LOG2=scale * _LOG2E,
                    FP32_INPUT=dtype == torch.float32,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    num_warps=4,
                    num_stages=kernel_stages,
                    sync_solver=True,
                )
            else:
                _sdpa_backward_delta_kernel[delta_grid](
                    output,
                    grad_output,
                    delta,
                    TOTAL_ROWS=delta_rows,
                    HEAD_DIM=head_dim,
                    BLOCK_ROWS=delta_block,
                    num_warps=4,
                    num_stages=1,
                    sync_solver=True,
                )
                _sdpa_backward_dq_kernel[dq_grid](
                    query,
                    key,
                    value,
                    grad_output,
                    stats,
                    delta,
                    grad_query,
                    HQ=heads_q,
                    HKV=heads_kv,
                    SQ=sequence_q,
                    SKV=sequence_kv,
                    HEAD_DIM=head_dim,
                    GROUP_SIZE=group_size,
                    SCALE=scale,
                    SCALE_LOG2=scale * _LOG2E,
                    CAUSAL=causal,
                    FP32_INPUT=dtype == torch.float32,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    num_warps=4,
                    num_stages=kernel_stages,
                    sync_solver=True,
                )
                _sdpa_backward_dkdv_kernel[dkdv_grid](
                    query,
                    key,
                    value,
                    grad_output,
                    stats,
                    delta,
                    grad_key,
                    grad_value,
                    HQ=heads_q,
                    HKV=heads_kv,
                    SQ=sequence_q,
                    SKV=sequence_kv,
                    HEAD_DIM=head_dim,
                    GROUP_SIZE=group_size,
                    SCALE=scale,
                    SCALE_LOG2=scale * _LOG2E,
                    CAUSAL=causal,
                    FP32_INPUT=dtype == torch.float32,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    num_warps=4,
                    num_stages=kernel_stages,
                    sync_solver=True,
                )
        return grad_query, grad_key, grad_value

    return run


__all__ = ("prepare_sdpa_backward",)
