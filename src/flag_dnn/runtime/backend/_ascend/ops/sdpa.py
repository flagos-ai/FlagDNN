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

"""Ascend-only Triton scaled-dot-product attention forward path."""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import torch
import triton
import triton.language as tl

from flag_dnn.graph.device import is_runtime_device_tensor
from flag_dnn.graph.prepared import (
    PreparedSingleKernelRunSpec,
    PreparedSingleKernelSpec,
    RunFn,
    get_prepared_output,
    make_single_kernel_run_fn,
    runtime_tensor_checks_from_specs,
)
from flag_dnn.graph.prepared.common import (
    _is_runtime_device_spec,
    _static_shape,
)
from flag_dnn.graph.tensor import TensorSpec, torch_dtype
from flag_dnn.utils import triton_lang_extension as tle
from flag_dnn.utils.libentry import libentry

_LOG2E = 1.4426950408889634


@triton.jit
def _sdpa_forward_inner(
    accumulator,
    row_sum,
    row_max,
    query,
    key_base,
    value_base,
    offsets_m,
    offsets_d,
    start,
    end,
    SCALE_LOG2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MASK_LOAD: tl.constexpr,
    MASK_CAUSAL: tl.constexpr,
    FP32_INPUT: tl.constexpr,
):
    offsets_n_local = tl.arange(0, BLOCK_N)
    for key_start in range(start, end, BLOCK_N):
        key_start = tl.multiple_of(key_start, BLOCK_N)
        offsets_n = key_start + offsets_n_local
        if MASK_LOAD:
            kv_mask = offsets_n < end
            key = tl.load(
                key_base + offsets_d[:, None] + offsets_n[None, :] * HEAD_DIM,
                mask=kv_mask[None, :],
                other=0.0,
            )
        else:
            key = tl.load(
                key_base + offsets_d[:, None] + offsets_n[None, :] * HEAD_DIM
            )

        if FP32_INPUT:
            scores = tl.dot(
                query,
                key,
                input_precision="tf32",
            )
        else:
            scores = tl.dot(query, key)
        scores = scores.to(tl.float32) * SCALE_LOG2

        if MASK_LOAD:
            visible = kv_mask[None, :]
            if MASK_CAUSAL:
                visible = visible & (offsets_n[None, :] <= offsets_m[:, None])
            scores = tl.where(visible, scores, float("-inf"))
        elif MASK_CAUSAL:
            visible = offsets_n[None, :] <= offsets_m[:, None]
            scores = tl.where(visible, scores, float("-inf"))

        new_max = tl.maximum(row_max, tl.max(scores, axis=1))
        if MASK_LOAD or MASK_CAUSAL:
            safe_max = tl.where(
                new_max == float("-inf"),
                0.0,
                new_max,
            )
        else:
            safe_max = new_max
        probabilities = tl.exp2(scores - safe_max[:, None])
        alpha = tl.exp2(row_max - safe_max)
        row_sum = row_sum * alpha + tl.sum(
            probabilities,
            axis=1,
        )
        accumulator *= alpha[:, None]

        if MASK_LOAD:
            value = tl.load(
                value_base
                + offsets_n[:, None] * HEAD_DIM
                + offsets_d[None, :],
                mask=kv_mask[:, None],
                other=0.0,
            )
        else:
            value = tl.load(
                value_base + offsets_n[:, None] * HEAD_DIM + offsets_d[None, :]
            )
        accumulator = tl.dot(
            probabilities.to(value.dtype),
            value,
            accumulator,
        )
        row_max = new_max
    return accumulator, row_sum, row_max


@libentry()
@triton.jit
def _sdpa_forward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    stats_ptr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    CAUSAL: tl.constexpr,
    GENERATE_STATS: tl.constexpr,
    FP32_INPUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    query_block = tle.program_id(0)
    batch_head = tle.program_id(1)
    batch = batch_head // HQ
    query_head = batch_head - batch * HQ
    kv_head = query_head // GROUP_SIZE

    offsets_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, HEAD_DIM)
    query_mask = offsets_m < SQ

    query_base = (
        q_ptr + batch * HQ * SQ * HEAD_DIM + query_head * SQ * HEAD_DIM
    )
    key_base = k_ptr + batch * HKV * SKV * HEAD_DIM + kv_head * SKV * HEAD_DIM
    value_base = (
        v_ptr + batch * HKV * SKV * HEAD_DIM + kv_head * SKV * HEAD_DIM
    )
    query = tl.load(
        query_base + offsets_m[:, None] * HEAD_DIM + offsets_d[None, :],
        mask=query_mask[:, None],
        other=0.0,
    )

    accumulator = tl.zeros(
        (BLOCK_M, HEAD_DIM),
        dtype=tl.float32,
    )
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    row_max = tl.full(
        (BLOCK_M,),
        float("-inf"),
        dtype=tl.float32,
    )

    if CAUSAL:
        history_end = tl.minimum(query_block * BLOCK_M, SKV)
        full_history_end = tl.minimum(
            history_end,
            (SKV // BLOCK_N) * BLOCK_N,
        )
        if full_history_end > 0:
            accumulator, row_sum, row_max = _sdpa_forward_inner(
                accumulator,
                row_sum,
                row_max,
                query,
                key_base,
                value_base,
                offsets_m,
                offsets_d,
                0,
                full_history_end,
                SCALE_LOG2=SCALE_LOG2,
                HEAD_DIM=HEAD_DIM,
                BLOCK_N=BLOCK_N,
                MASK_LOAD=False,
                MASK_CAUSAL=False,
                FP32_INPUT=FP32_INPUT,
            )
        if full_history_end < history_end:
            accumulator, row_sum, row_max = _sdpa_forward_inner(
                accumulator,
                row_sum,
                row_max,
                query,
                key_base,
                value_base,
                offsets_m,
                offsets_d,
                full_history_end,
                history_end,
                SCALE_LOG2=SCALE_LOG2,
                HEAD_DIM=HEAD_DIM,
                BLOCK_N=BLOCK_N,
                MASK_LOAD=True,
                MASK_CAUSAL=False,
                FP32_INPUT=FP32_INPUT,
            )
        diagonal_end = tl.minimum(
            (query_block + 1) * BLOCK_M,
            SKV,
        )
        if history_end < diagonal_end:
            if HEAD_DIM == 64 and diagonal_end <= (SKV // BLOCK_N) * BLOCK_N:
                accumulator, row_sum, row_max = _sdpa_forward_inner(
                    accumulator,
                    row_sum,
                    row_max,
                    query,
                    key_base,
                    value_base,
                    offsets_m,
                    offsets_d,
                    history_end,
                    diagonal_end,
                    SCALE_LOG2=SCALE_LOG2,
                    HEAD_DIM=HEAD_DIM,
                    BLOCK_N=BLOCK_N,
                    MASK_LOAD=False,
                    MASK_CAUSAL=True,
                    FP32_INPUT=FP32_INPUT,
                )
            else:
                accumulator, row_sum, row_max = _sdpa_forward_inner(
                    accumulator,
                    row_sum,
                    row_max,
                    query,
                    key_base,
                    value_base,
                    offsets_m,
                    offsets_d,
                    history_end,
                    diagonal_end,
                    SCALE_LOG2=SCALE_LOG2,
                    HEAD_DIM=HEAD_DIM,
                    BLOCK_N=BLOCK_N,
                    MASK_LOAD=True,
                    MASK_CAUSAL=True,
                    FP32_INPUT=FP32_INPUT,
                )
    else:
        full_end = (SKV // BLOCK_N) * BLOCK_N
        if full_end > 0:
            accumulator, row_sum, row_max = _sdpa_forward_inner(
                accumulator,
                row_sum,
                row_max,
                query,
                key_base,
                value_base,
                offsets_m,
                offsets_d,
                0,
                full_end,
                SCALE_LOG2=SCALE_LOG2,
                HEAD_DIM=HEAD_DIM,
                BLOCK_N=BLOCK_N,
                MASK_LOAD=False,
                MASK_CAUSAL=False,
                FP32_INPUT=FP32_INPUT,
            )
        if full_end < SKV:
            accumulator, row_sum, row_max = _sdpa_forward_inner(
                accumulator,
                row_sum,
                row_max,
                query,
                key_base,
                value_base,
                offsets_m,
                offsets_d,
                full_end,
                SKV,
                SCALE_LOG2=SCALE_LOG2,
                HEAD_DIM=HEAD_DIM,
                BLOCK_N=BLOCK_N,
                MASK_LOAD=True,
                MASK_CAUSAL=False,
                FP32_INPUT=FP32_INPUT,
            )

    safe_sum = tl.where(row_sum == 0.0, 1.0, row_sum)
    output = accumulator / safe_sum[:, None]
    output_base = (
        output_ptr + batch * HQ * SQ * HEAD_DIM + query_head * SQ * HEAD_DIM
    )
    tl.store(
        output_base + offsets_m[:, None] * HEAD_DIM + offsets_d[None, :],
        output.to(output_ptr.dtype.element_ty),
        mask=query_mask[:, None],
    )

    if GENERATE_STATS:
        stats = row_max / 1.4426950408889634 + tl.log(safe_sum)
        tl.store(
            stats_ptr + batch_head * SQ + offsets_m,
            stats,
            mask=query_mask,
        )


@triton.jit
def _sdpa_forward_aligned_causal_step(
    accumulator,
    row_sum,
    row_max,
    query,
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
    if FP32_INPUT:
        scores = tl.dot(
            query,
            key,
            input_precision="tf32",
        )
    else:
        scores = tl.dot(query, key)
    scores = scores.to(tl.float32) * SCALE_LOG2
    if CAUSAL:
        visible = offsets_n[None, :] <= offsets_m[:, None]
        scores = tl.where(visible, scores, float("-inf"))

    new_max = tl.maximum(row_max, tl.max(scores, axis=1))
    probabilities = tl.exp2(scores - new_max[:, None])
    alpha = tl.exp2(row_max - new_max)
    row_sum = row_sum * alpha + tl.sum(probabilities, axis=1)
    accumulator *= alpha[:, None]
    value = tl.load(
        value_base + offsets_n[:, None] * HEAD_DIM + offsets_d[None, :]
    )
    accumulator = tl.dot(
        probabilities.to(value.dtype),
        value,
        accumulator,
    )
    return accumulator, row_sum, new_max


@libentry()
@triton.jit
def _sdpa_forward_aligned_causal_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    stats_ptr,
    SQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    FP32_INPUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # This kernel is selected only for the bounded aligned contract in
    # prepare_sdpa, so 32-bit indices cannot overflow.  Keeping program IDs
    # in their native type also avoids scalar i64 address arithmetic.
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    offsets_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, HEAD_DIM)
    tensor_base = batch_head * SQ * HEAD_DIM
    query_base = q_ptr + tensor_base
    key_base = k_ptr + tensor_base
    value_base = v_ptr + tensor_base
    query = tl.load(
        query_base + offsets_m[:, None] * HEAD_DIM + offsets_d[None, :]
    )
    accumulator = tl.zeros(
        (BLOCK_M, HEAD_DIM),
        dtype=tl.float32,
    )
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    row_max = tl.full(
        (BLOCK_M,),
        float("-inf"),
        dtype=tl.float32,
    )

    history_end = query_block * BLOCK_M
    for key_start in range(0, history_end, BLOCK_N):
        (
            accumulator,
            row_sum,
            row_max,
        ) = _sdpa_forward_aligned_causal_step(
            accumulator,
            row_sum,
            row_max,
            query,
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
    (
        accumulator,
        row_sum,
        row_max,
    ) = _sdpa_forward_aligned_causal_step(
        accumulator,
        row_sum,
        row_max,
        query,
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

    output = accumulator / row_sum[:, None]
    tl.store(
        output_ptr
        + tensor_base
        + offsets_m[:, None] * HEAD_DIM
        + offsets_d[None, :],
        output.to(output_ptr.dtype.element_ty),
    )
    stats = row_max / 1.4426950408889634 + tl.log(row_sum)
    tl.store(
        stats_ptr + batch_head * SQ + offsets_m,
        stats,
    )


def _select_blocks(
    sequence_q: int,
    head_dim: int,
    causal: bool,
) -> tuple[int, int]:
    if causal:
        return 64, 64
    if sequence_q <= 16:
        if head_dim == 128:
            return 16, 128
        return 16, min(64, head_dim)
    if head_dim == 32:
        if sequence_q < 256:
            return 128, 32
        return 256, 32
    if head_dim == 64:
        return 128, 64
    return 128, 64


def prepare_sdpa(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    """Prepare the supported dense Ascend attention contract."""
    if bool(attrs.get("has_bias")) or len(input_specs) != 3:
        return None
    if (
        attrs.get("diagonal_alignment") not in (None, "TOP_LEFT")
        or attrs.get("diagonal_band_left_bound") is not None
        or attrs.get("diagonal_band_right_bound") not in (None, 0)
    ):
        return None
    if not all(_is_runtime_device_spec(spec) for spec in input_specs):
        return None

    q_spec, k_spec, v_spec = input_specs
    q_shape = _static_shape(q_spec)
    k_shape = _static_shape(k_spec)
    v_shape = _static_shape(v_spec)
    if (
        q_shape is None
        or k_shape is None
        or v_shape is None
        or len(q_shape) != 4
        or len(k_shape) != 4
        or len(v_shape) != 4
        or q_shape[0] != k_shape[0]
        or q_shape[0] != v_shape[0]
        or q_shape[3] != k_shape[3]
        or k_shape[1] != v_shape[1]
        or k_shape[2] != v_shape[2]
        or q_shape[1] % k_shape[1] != 0
        or v_shape[3] != q_shape[3]
        or q_shape[3] not in (32, 64, 128)
        or not all(bool(spec.contiguous) for spec in input_specs)
        or q_spec.dtype != k_spec.dtype
        or q_spec.dtype != v_spec.dtype
        or q_spec.dtype not in ("float16", "bfloat16", "float32")
    ):
        return None

    checks = runtime_tensor_checks_from_specs(
        input_specs,
        (0, 1, 2),
        require_shape=True,
        require_stride=True,
        require_dtype=True,
    )
    if checks is None:
        return None

    batch, heads_q, sequence_q, head_dim = q_shape
    heads_kv = k_shape[1]
    sequence_kv = k_shape[2]
    group_size = heads_q // heads_kv
    generate_stats = bool(attrs.get("generate_stats"))
    causal = attrs.get("diagonal_band_right_bound") == 0
    scale = attrs.get("attn_scale")
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    use_aligned_causal_kernel = (
        causal
        and generate_stats
        and batch == 1
        and heads_q == 32
        and heads_kv == 32
        and sequence_q == 1024
        and sequence_kv == 1024
        and head_dim == 64
    )
    block_m, block_n = _select_blocks(
        sequence_q,
        head_dim,
        causal,
    )
    static_grid = (
        triton.cdiv(sequence_q, block_m),
        batch * heads_q,
    )
    output_shape = (batch, heads_q, sequence_q, head_dim)
    stats_shape = (batch, heads_q, sequence_q, 1)
    output_dtype = torch_dtype(q_spec.dtype)
    output_cache: dict[tuple[Any, ...], Any] = {}

    def allocate_output(inputs: Sequence[Any]) -> Any:
        q = inputs[0]
        key = (
            q.device.type,
            q.device.index,
            output_dtype,
            output_shape,
            generate_stats,
        )

        def allocate() -> tuple[torch.Tensor, torch.Tensor]:
            output = torch.empty(
                output_shape,
                device=q.device,
                dtype=output_dtype,
            )
            stats = (
                torch.empty(
                    stats_shape,
                    device=q.device,
                    dtype=torch.float32,
                )
                if generate_stats
                else output
            )
            return output, stats

        return get_prepared_output(output_cache, key, allocate)

    def runtime_args(
        inputs: Sequence[Any],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[Any, ...]:
        return (
            inputs[0],
            inputs[1],
            inputs[2],
            output[0],
            output[1],
        )

    def result(output: tuple[torch.Tensor, torch.Tensor]) -> Any:
        return output if generate_stats else output[0]

    def extra_check(inputs: Sequence[Any]) -> bool:
        return all(
            isinstance(value, torch.Tensor) and is_runtime_device_tensor(value)
            for value in inputs
        )

    def grid(_meta: dict[str, Any]) -> tuple[int, ...]:
        return static_grid

    constexpr_kwargs: dict[str, Any]
    cached_static_args: tuple[Any, ...]
    if use_aligned_causal_kernel:
        prepared_kernel = _sdpa_forward_aligned_causal_kernel
        constexpr_kwargs = {
            "SQ": sequence_q,
            "HEAD_DIM": head_dim,
            "SCALE_LOG2": float(scale) * _LOG2E,
            "FP32_INPUT": output_dtype == torch.float32,
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "num_warps": 4,
            "num_stages": 1,
            "sync_solver": True,
        }
        cached_static_args = (
            sequence_q,
            head_dim,
            float(scale) * _LOG2E,
            output_dtype == torch.float32,
            block_m,
            block_n,
        )
    else:
        prepared_kernel = _sdpa_forward_kernel
        constexpr_kwargs = {
            "HQ": heads_q,
            "HKV": heads_kv,
            "SQ": sequence_q,
            "SKV": sequence_kv,
            "HEAD_DIM": head_dim,
            "GROUP_SIZE": group_size,
            "SCALE_LOG2": float(scale) * _LOG2E,
            "CAUSAL": causal,
            "GENERATE_STATS": generate_stats,
            "FP32_INPUT": output_dtype == torch.float32,
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "num_warps": 4,
            "num_stages": 1 if causal and head_dim == 64 else 2,
            "sync_solver": True,
        }
        cached_static_args = (
            heads_q,
            heads_kv,
            sequence_q,
            sequence_kv,
            head_dim,
            group_size,
            float(scale) * _LOG2E,
            causal,
            generate_stats,
            output_dtype == torch.float32,
            block_m,
            block_n,
        )

    def build_cached_call(
        _constexprs: dict[str, Any],
    ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        return (*static_grid, 1), cached_static_args

    return make_single_kernel_run_fn(
        PreparedSingleKernelRunSpec(
            kernel=PreparedSingleKernelSpec(
                kernel=prepared_kernel,
                grid=grid,
                static_args=(),
                constexpr_kwargs=constexpr_kwargs,
                build_cached_call=build_cached_call,
            ),
            input_checks=checks,
            output_factory=allocate_output,
            runtime_args=runtime_args,
            result=result,
            extra_check=extra_check,
            validate_inputs=bool(attrs.get("_validate_inputs", True)),
        ),
        default_run_fn,
    )


__all__ = ("prepare_sdpa",)
