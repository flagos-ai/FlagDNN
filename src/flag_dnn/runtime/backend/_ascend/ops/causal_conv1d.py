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

"""Ascend-only prepared Triton kernel for causal depthwise Conv1d."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import triton
import triton.language as tl

from flag_dnn.graph.device import is_runtime_device_tensor
from flag_dnn.utils import triton_lang_extension as tle
from flag_dnn.utils.libentry import libentry


@libentry()
@triton.jit
def causal_conv1d_row_loop_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    ROWS: tl.constexpr,
    CHANNELS: tl.constexpr,
    SEQUENCE: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_SEQUENCE: tl.constexpr,
):
    row_offsets = (
        tle.program_id(0) * ROWS_PER_PROGRAM
        + tl.arange(0, ROWS_PER_PROGRAM)[:, None]
    )
    row_mask = row_offsets < ROWS
    channel_offsets = row_offsets % CHANNELS
    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + channel_offsets,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
    else:
        bias = 0.0

    for sequence_start in range(0, SEQUENCE, BLOCK_SEQUENCE):
        sequence_offsets = (
            sequence_start + tl.arange(0, BLOCK_SEQUENCE)[None, :]
        )
        output_mask = row_mask & (sequence_offsets < SEQUENCE)
        accumulator = tl.zeros(
            (ROWS_PER_PROGRAM, BLOCK_SEQUENCE),
            dtype=tl.float32,
        )
        for tap in range(0, KERNEL_SIZE):
            input_sequence = sequence_offsets - (KERNEL_SIZE - 1) + tap
            input_mask = (
                output_mask
                & (input_sequence >= 0)
                & (input_sequence < SEQUENCE)
            )
            value = tl.load(
                input_ptr + row_offsets * SEQUENCE + input_sequence,
                mask=input_mask,
                other=0.0,
            ).to(tl.float32)
            weight = tl.load(
                weight_ptr + channel_offsets * KERNEL_SIZE + tap,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            accumulator += value * weight

        accumulator += bias
        if ACTIVATION == "silu":
            accumulator *= tl.sigmoid(accumulator)
        output_offsets = row_offsets * SEQUENCE + sequence_offsets
        tl.store(
            output_ptr + output_offsets,
            accumulator.to(output_ptr.dtype.element_ty),
            mask=output_mask,
        )


def prepare_causal_conv1d(
    *,
    attrs: dict[str, Any],
    input_specs: Sequence[Any],
    default_run_fn: Any,
    input_checks: Any,
    x_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    has_bias: bool,
    activation: str,
) -> Optional[Any]:
    expected_inputs = 3 if has_bias else 2
    if len(input_specs) != expected_inputs:
        return None
    x_spec, weight_spec = input_specs[:2]
    if (
        x_spec.dtype not in ("float16", "bfloat16", "float32")
        or weight_spec.dtype != x_spec.dtype
        or not bool(x_spec.contiguous)
        or not bool(weight_spec.contiguous)
        or x_spec.stride is None
        or weight_spec.stride is None
    ):
        return None
    if has_bias:
        bias_spec = input_specs[2]
        if (
            bias_spec.dtype != x_spec.dtype
            or not bool(bias_spec.contiguous)
            or bias_spec.stride is None
        ):
            return None

    from flag_dnn.graph.prepared import (
        PreparedSingleKernelRunSpec,
        PreparedSingleKernelSpec,
        get_prepared_output,
        make_single_kernel_run_fn,
    )
    from flag_dnn.graph.tensor import torch_dtype

    batch, channels, sequence = (int(item) for item in x_shape)
    kernel_size = int(weight_shape[1])
    rows = batch * channels
    block_sequence = min(256, triton.next_power_of_2(sequence))
    rows_per_program = min(
        16,
        max(1, 4096 // block_sequence),
    )
    static_grid = (triton.cdiv(rows, rows_per_program),)
    output_dtype = torch_dtype(x_spec.dtype)
    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def output_factory(inputs: Sequence[Any]) -> torch.Tensor:
        source = inputs[0]
        key = (
            source.device.type,
            source.device.index,
            output_dtype,
            x_shape,
        )
        return get_prepared_output(
            output_cache,
            key,
            lambda: torch.empty(
                x_shape,
                device=source.device,
                dtype=output_dtype,
            ),
        )

    def runtime_args(
        inputs: Sequence[Any], output: torch.Tensor
    ) -> tuple[Any, ...]:
        bias = inputs[2] if has_bias else output
        return inputs[0], inputs[1], bias, output

    def extra_check(inputs: Sequence[Any]) -> bool:
        tensors = inputs[:expected_inputs]
        if not all(
            isinstance(value, torch.Tensor) and is_runtime_device_tensor(value)
            for value in tensors
        ):
            return False
        first = tensors[0]
        return all(value.device == first.device for value in tensors[1:])

    constexpr_kwargs = {
        "ROWS": rows,
        "CHANNELS": channels,
        "SEQUENCE": sequence,
        "KERNEL_SIZE": kernel_size,
        "HAS_BIAS": has_bias,
        "ACTIVATION": activation,
        "ROWS_PER_PROGRAM": rows_per_program,
        "BLOCK_SEQUENCE": block_sequence,
        "num_warps": 4,
        "num_stages": 1,
    }
    cached_args = (
        rows,
        channels,
        sequence,
        kernel_size,
        has_bias,
        activation,
        rows_per_program,
        block_sequence,
    )

    def grid(_meta: dict[str, Any]) -> tuple[int, ...]:
        return static_grid

    def build_cached_call(
        constexprs: dict[str, Any],
    ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        del constexprs
        return (*static_grid, 1, 1), cached_args

    return make_single_kernel_run_fn(
        PreparedSingleKernelRunSpec(
            kernel=PreparedSingleKernelSpec(
                kernel=causal_conv1d_row_loop_kernel,
                grid=grid,
                static_args=(),
                constexpr_kwargs=constexpr_kwargs,
                build_cached_call=build_cached_call,
            ),
            input_checks=input_checks,
            output_factory=output_factory,
            runtime_args=runtime_args,
            extra_check=extra_check,
            validate_inputs=bool(attrs.get("_validate_inputs", True)),
        ),
        default_run_fn,
    )


__all__ = (
    "causal_conv1d_row_loop_kernel",
    "prepare_causal_conv1d",
)
