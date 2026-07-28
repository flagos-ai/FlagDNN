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

"""Ascend-only full-row prepared normalization kernels."""

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
def layernorm_full_row_kernel(
    input_ptr,
    output_ptr,
    mean_ptr,
    rstd_ptr,
    scale_ptr,
    bias_ptr,
    epsilon,
    ROWS: tl.constexpr,
    COLUMNS: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tle.program_id(0) * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    columns = tl.arange(0, BLOCK_SIZE)
    row_mask = row < ROWS
    column_mask = columns < COLUMNS
    mask = row_mask[:, None] & column_mask[None, :]
    offsets = row[:, None] * COLUMNS + columns[None, :]
    value = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(value, axis=1) / COLUMNS
    mean_square = tl.sum(value * value, axis=1) / COLUMNS
    variance = tl.maximum(mean_square - mean * mean, 0.0)
    rstd = tl.rsqrt(variance + epsilon)
    tl.store(mean_ptr + row, mean, mask=row_mask)
    tl.store(rstd_ptr + row, rstd, mask=row_mask)
    if ROWS_PER_PROGRAM * BLOCK_SIZE > 8192:
        value = tl.load(
            input_ptr + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
    centered = tl.where(mask, value - mean[:, None], 0.0)
    scale = tl.load(scale_ptr + columns, mask=column_mask, other=0.0).to(
        tl.float32
    )
    bias = tl.load(bias_ptr + columns, mask=column_mask, other=0.0).to(
        tl.float32
    )
    output = centered * rstd[:, None] * scale[None, :] + bias[None, :]
    tl.store(
        output_ptr + offsets,
        output.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


@libentry()
@triton.jit
def rmsnorm_full_row_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    bias_ptr,
    rstd_ptr,
    epsilon,
    ROWS: tl.constexpr,
    COLUMNS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tle.program_id(0) * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    columns = tl.arange(0, BLOCK_SIZE)
    row_mask = row < ROWS
    column_mask = columns < COLUMNS
    mask = row_mask[:, None] & column_mask[None, :]
    offsets = row[:, None] * COLUMNS + columns[None, :]
    value = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(value * value, axis=1) / COLUMNS
    rstd = tl.rsqrt(mean_square + epsilon)
    tl.store(rstd_ptr + row, rstd, mask=row_mask)
    scale = tl.load(scale_ptr + columns, mask=column_mask, other=0.0).to(
        tl.float32
    )
    output = value * rstd[:, None] * scale[None, :]
    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + columns,
            mask=column_mask,
            other=0.0,
        ).to(tl.float32)
        output += bias[None, :]
    tl.store(
        output_ptr + offsets,
        output.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


def _extra_check(
    inputs: Sequence[Any],
    tensor_indices: tuple[int, ...],
) -> bool:
    tensors = tuple(inputs[index] for index in tensor_indices)
    if not all(
        isinstance(value, torch.Tensor) and is_runtime_device_tensor(value)
        for value in tensors
    ):
        return False
    first = tensors[0]
    return all(value.device == first.device for value in tensors[1:])


def prepare_dense_layernorm(
    *,
    attrs: dict[str, Any],
    input_specs: Sequence[Any],
    default_run_fn: Any,
    input_checks: Any,
    input_shape: tuple[int, ...],
    stat_shape: tuple[int, ...],
    rows: int,
    columns: int,
) -> Optional[Any]:
    if len(input_specs) != 4 or columns <= 0 or columns > 4096 or rows <= 0:
        return None

    from flag_dnn.graph.prepared import (
        PreparedSingleKernelRunSpec,
        PreparedSingleKernelSpec,
        get_prepared_output,
        make_single_kernel_run_fn,
    )

    input_spec = input_specs[0]
    input_stride = tuple(int(item) for item in input_spec.stride)
    block_size = triton.next_power_of_2(columns)
    if block_size == 4096:
        rows_per_program = min(triton.next_power_of_2(rows), 4)
    else:
        rows_per_program = min(
            triton.next_power_of_2(rows),
            max(1, 8192 // block_size),
        )
    static_grid = (triton.cdiv(rows, rows_per_program),)
    output_cache: dict[tuple[Any, ...], tuple[torch.Tensor, ...]] = {}

    def output_factory(
        inputs: Sequence[Any],
    ) -> tuple[torch.Tensor, ...]:
        source = inputs[0]
        key = (
            source.device.type,
            source.device.index,
            source.dtype,
            input_shape,
            input_stride,
            stat_shape,
        )

        def allocate() -> tuple[torch.Tensor, ...]:
            output = torch.empty_strided(
                input_shape,
                input_stride,
                device=source.device,
                dtype=source.dtype,
            )
            mean = torch.empty(
                stat_shape,
                device=source.device,
                dtype=torch.float32,
            )
            rstd = torch.empty_like(mean)
            return output, mean, rstd

        return get_prepared_output(output_cache, key, allocate)

    def runtime_args(
        inputs: Sequence[Any],
        outputs: tuple[torch.Tensor, ...],
    ) -> tuple[Any, ...]:
        output, mean, rstd = outputs
        return (
            inputs[0],
            output,
            mean,
            rstd,
            inputs[1],
            inputs[2],
            float(inputs[3]),
        )

    def grid(_meta: dict[str, Any]) -> tuple[int, ...]:
        return static_grid

    def build_cached_call(
        constexprs: dict[str, Any],
    ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        del constexprs
        return (*static_grid, 1, 1), (
            rows,
            columns,
            rows_per_program,
            block_size,
        )

    return make_single_kernel_run_fn(
        PreparedSingleKernelRunSpec(
            kernel=PreparedSingleKernelSpec(
                kernel=layernorm_full_row_kernel,
                grid=grid,
                static_args=(),
                constexpr_kwargs={
                    "ROWS": rows,
                    "COLUMNS": columns,
                    "ROWS_PER_PROGRAM": rows_per_program,
                    "BLOCK_SIZE": block_size,
                    "num_warps": 4,
                    "num_stages": 1,
                },
                build_cached_call=build_cached_call,
            ),
            input_checks=input_checks,
            output_factory=output_factory,
            runtime_args=runtime_args,
            extra_check=lambda inputs: _extra_check(inputs, (0, 1, 2)),
            validate_inputs=bool(attrs.get("_validate_inputs", True)),
        ),
        default_run_fn,
    )


def prepare_dense_rmsnorm(
    *,
    attrs: dict[str, Any],
    input_specs: Sequence[Any],
    default_run_fn: Any,
    input_checks: Any,
    input_shape: tuple[int, ...],
    stat_shape: tuple[int, ...],
    rows: int,
    columns: int,
    has_bias: bool,
    bias_index: Optional[int],
    epsilon_index: int,
    tensor_indices: tuple[int, ...],
) -> Optional[Any]:
    expected_inputs = 4 if has_bias else 3
    if (
        len(input_specs) != expected_inputs
        or columns <= 0
        or columns > 4096
        or rows <= 0
    ):
        return None

    from flag_dnn.graph.prepared import (
        PreparedSingleKernelRunSpec,
        PreparedSingleKernelSpec,
        get_prepared_output,
        make_single_kernel_run_fn,
    )

    input_spec = input_specs[0]
    input_stride = tuple(int(item) for item in input_spec.stride)
    block_size = triton.next_power_of_2(columns)
    rows_per_program = min(
        triton.next_power_of_2(rows),
        max(1, 8192 // block_size),
    )
    static_grid = (triton.cdiv(rows, rows_per_program),)
    output_cache: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] = {}

    def output_factory(
        inputs: Sequence[Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = inputs[0]
        key = (
            source.device.type,
            source.device.index,
            source.dtype,
            input_shape,
            input_stride,
            stat_shape,
        )

        def allocate() -> tuple[torch.Tensor, torch.Tensor]:
            output = torch.empty_strided(
                input_shape,
                input_stride,
                device=source.device,
                dtype=source.dtype,
            )
            rstd = torch.empty(
                stat_shape,
                device=source.device,
                dtype=torch.float32,
            )
            return output, rstd

        return get_prepared_output(output_cache, key, allocate)

    def runtime_args(
        inputs: Sequence[Any],
        outputs: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[Any, ...]:
        output, rstd = outputs
        bias = inputs[bias_index] if bias_index is not None else inputs[0]
        return (
            inputs[0],
            output,
            inputs[1],
            bias,
            rstd,
            float(inputs[epsilon_index]),
        )

    def grid(_meta: dict[str, Any]) -> tuple[int, ...]:
        return static_grid

    def build_cached_call(
        constexprs: dict[str, Any],
    ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        del constexprs
        return (*static_grid, 1, 1), (
            rows,
            columns,
            has_bias,
            rows_per_program,
            block_size,
        )

    return make_single_kernel_run_fn(
        PreparedSingleKernelRunSpec(
            kernel=PreparedSingleKernelSpec(
                kernel=rmsnorm_full_row_kernel,
                grid=grid,
                static_args=(),
                constexpr_kwargs={
                    "ROWS": rows,
                    "COLUMNS": columns,
                    "HAS_BIAS": has_bias,
                    "ROWS_PER_PROGRAM": rows_per_program,
                    "BLOCK_SIZE": block_size,
                    "num_warps": 4,
                    "num_stages": 1,
                },
                build_cached_call=build_cached_call,
            ),
            input_checks=input_checks,
            output_factory=output_factory,
            runtime_args=runtime_args,
            extra_check=lambda inputs: _extra_check(inputs, tensor_indices),
            validate_inputs=bool(attrs.get("_validate_inputs", True)),
        ),
        default_run_fn,
    )


__all__ = (
    "layernorm_full_row_kernel",
    "prepare_dense_layernorm",
    "prepare_dense_rmsnorm",
    "rmsnorm_full_row_kernel",
)
