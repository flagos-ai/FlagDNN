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

"""Ascend-only prepared kernels for batch normalization."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import triton
import triton.language as tl

from flag_dnn.graph.device import is_runtime_device_tensor
from flag_dnn.utils import triton_lang_extension as tle
from flag_dnn.utils.libentry import libentry


def get_batchnorm_inference_block_size(total_elements: int) -> int:
    return 2048 if total_elements <= 4096 else 4096


@libentry()
@triton.jit
def batchnorm_inference_core_loop_kernel(
    input_ptr,
    output_ptr,
    mean_ptr,
    inv_variance_ptr,
    scale_ptr,
    bias_ptr,
    N_ELEMENTS: tl.constexpr,
    CHANNELS: tl.constexpr,
    SPATIAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_programs = tle.num_programs(0)
    elements_per_program = tl.cdiv(N_ELEMENTS, num_programs)
    chunk_size = tl.cdiv(elements_per_program, 256) * 256
    chunk_start = pid * chunk_size
    chunk_end = tl.minimum(chunk_start + chunk_size, N_ELEMENTS)
    num_blocks = tl.cdiv(chunk_size, BLOCK_SIZE)

    for block_index in range(0, num_blocks):
        offsets = (
            chunk_start + block_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        )
        mask = offsets < chunk_end
        channel_index = (offsets // SPATIAL) % CHANNELS
        value = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
        mean = tl.load(mean_ptr + channel_index, mask=mask)
        inv_variance = tl.load(inv_variance_ptr + channel_index, mask=mask)
        scale = tl.load(scale_ptr + channel_index, mask=mask)
        bias = tl.load(bias_ptr + channel_index, mask=mask)
        output = (value - mean) * inv_variance * scale + bias
        tl.store(
            output_ptr + offsets,
            output.to(output_ptr.dtype.element_ty),
            mask=mask,
        )


@libentry()
@triton.jit
def batchnorm_inference_rows_kernel(
    input_ptr,
    output_ptr,
    mean_ptr,
    inv_variance_ptr,
    scale_ptr,
    bias_ptr,
    ROWS: tl.constexpr,
    CHANNELS: tl.constexpr,
    SPATIAL: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_COLUMNS: tl.constexpr,
):
    column_offsets = (
        tle.program_id(0) * BLOCK_COLUMNS
        + tl.arange(0, BLOCK_COLUMNS)[None, :]
    )
    row_offsets = (
        tle.program_id(1) * ROWS_PER_PROGRAM
        + tl.arange(0, ROWS_PER_PROGRAM)[:, None]
    )
    mask = (row_offsets < ROWS) & (column_offsets < SPATIAL)
    channel_index = row_offsets % CHANNELS
    mean = tl.load(mean_ptr + channel_index, mask=row_offsets < ROWS)
    inv_variance = tl.load(
        inv_variance_ptr + channel_index, mask=row_offsets < ROWS
    )
    scale = tl.load(scale_ptr + channel_index, mask=row_offsets < ROWS)
    bias = tl.load(bias_ptr + channel_index, mask=row_offsets < ROWS)
    offsets = row_offsets * SPATIAL + column_offsets
    value = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
    output = (value - mean) * inv_variance * scale + bias
    tl.store(
        output_ptr + offsets,
        output.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


@libentry()
@triton.jit
def batchnorm_inference_row_loop_kernel(
    input_ptr,
    output_ptr,
    mean_ptr,
    inv_variance_ptr,
    scale_ptr,
    bias_ptr,
    ROWS: tl.constexpr,
    CHANNELS: tl.constexpr,
    SPATIAL: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_COLUMNS: tl.constexpr,
):
    row_offsets = (
        tle.program_id(0) * ROWS_PER_PROGRAM
        + tl.arange(0, ROWS_PER_PROGRAM)[:, None]
    )
    row_mask = row_offsets < ROWS
    channel_index = row_offsets % CHANNELS
    mean = tl.load(mean_ptr + channel_index, mask=row_mask)
    inv_variance = tl.load(inv_variance_ptr + channel_index, mask=row_mask)
    scale = tl.load(scale_ptr + channel_index, mask=row_mask)
    bias = tl.load(bias_ptr + channel_index, mask=row_mask)

    for column_start in range(0, SPATIAL, BLOCK_COLUMNS):
        column_offsets = column_start + tl.arange(0, BLOCK_COLUMNS)[None, :]
        mask = row_mask & (column_offsets < SPATIAL)
        offsets = row_offsets * SPATIAL + column_offsets
        value = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
        output = (value - mean) * inv_variance * scale + bias
        tl.store(
            output_ptr + offsets,
            output.to(output_ptr.dtype.element_ty),
            mask=mask,
        )


@libentry()
@triton.jit
def batchnorm_inference_nc_kernel(
    input_ptr,
    output_ptr,
    mean_ptr,
    inv_variance_ptr,
    scale_ptr,
    bias_ptr,
    BATCH: tl.constexpr,
    CHANNELS: tl.constexpr,
    BATCH_PER_PROGRAM: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    channel_offsets = (
        tle.program_id(0) * BLOCK_CHANNELS
        + tl.arange(0, BLOCK_CHANNELS)[None, :]
    )
    batch_offsets = (
        tle.program_id(1) * BATCH_PER_PROGRAM
        + tl.arange(0, BATCH_PER_PROGRAM)[:, None]
    )
    mask = (batch_offsets < BATCH) & (channel_offsets < CHANNELS)
    param_mask = channel_offsets < CHANNELS
    mean = tl.load(mean_ptr + channel_offsets, mask=param_mask)
    inv_variance = tl.load(inv_variance_ptr + channel_offsets, mask=param_mask)
    scale = tl.load(scale_ptr + channel_offsets, mask=param_mask)
    bias = tl.load(bias_ptr + channel_offsets, mask=param_mask)
    offsets = batch_offsets * CHANNELS + channel_offsets
    value = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
    output = (value - mean) * inv_variance * scale + bias
    tl.store(
        output_ptr + offsets,
        output.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


@libentry()
@triton.jit
def batchnorm_training_channel_kernel(
    input_ptr,
    output_ptr,
    running_mean_ptr,
    running_var_ptr,
    scale_ptr,
    bias_ptr,
    saved_mean_ptr,
    saved_inv_var_ptr,
    next_running_mean_ptr,
    next_running_var_ptr,
    epsilon,
    momentum,
    BATCH: tl.constexpr,
    CHANNELS: tl.constexpr,
    SPATIAL: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
):
    channel = tle.program_id(0)
    batch_offsets = tl.arange(0, BLOCK_BATCH)[:, None]
    batch_mask = batch_offsets < BATCH
    sum_x = 0.0
    sum_x2 = 0.0

    for spatial_start in range(0, SPATIAL, BLOCK_SPATIAL):
        spatial_offsets = spatial_start + tl.arange(0, BLOCK_SPATIAL)[None, :]
        mask = batch_mask & (spatial_offsets < SPATIAL)
        offsets = (
            batch_offsets * CHANNELS + channel
        ) * SPATIAL + spatial_offsets
        value = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        sum_x += tl.sum(tl.sum(value, axis=1), axis=0)
        sum_x2 += tl.sum(tl.sum(value * value, axis=1), axis=0)

    sample_count: tl.constexpr = BATCH * SPATIAL
    mean = sum_x / sample_count
    variance = sum_x2 / sample_count - mean * mean
    variance = tl.maximum(variance, 0.0)
    inv_variance = 1.0 / tl.sqrt(variance + epsilon)
    tl.store(saved_mean_ptr + channel, mean)
    tl.store(saved_inv_var_ptr + channel, inv_variance)

    old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
    old_variance = tl.load(running_var_ptr + channel).to(tl.float32)
    if sample_count > 1:
        unbiased_variance = variance * (sample_count / (sample_count - 1))
    else:
        unbiased_variance = variance
    next_mean = old_mean * (1.0 - momentum) + mean * momentum
    next_variance = (
        old_variance * (1.0 - momentum) + unbiased_variance * momentum
    )
    tl.store(next_running_mean_ptr + channel, next_mean)
    tl.store(next_running_var_ptr + channel, next_variance)

    scale = tl.load(scale_ptr + channel).to(tl.float32)
    bias = tl.load(bias_ptr + channel).to(tl.float32)
    for spatial_start in range(0, SPATIAL, BLOCK_SPATIAL):
        spatial_offsets = spatial_start + tl.arange(0, BLOCK_SPATIAL)[None, :]
        mask = batch_mask & (spatial_offsets < SPATIAL)
        offsets = (
            batch_offsets * CHANNELS + channel
        ) * SPATIAL + spatial_offsets
        value = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
        output = (value - mean) * inv_variance * scale + bias
        tl.store(
            output_ptr + offsets,
            output.to(output_ptr.dtype.element_ty),
            mask=mask,
        )


@libentry()
@triton.jit
def batchnorm_training_nc_kernel(
    input_ptr,
    output_ptr,
    running_mean_ptr,
    running_var_ptr,
    scale_ptr,
    bias_ptr,
    saved_mean_ptr,
    saved_inv_var_ptr,
    next_running_mean_ptr,
    next_running_var_ptr,
    epsilon,
    momentum,
    BATCH: tl.constexpr,
    CHANNELS: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    channel_offsets = tle.program_id(0) * BLOCK_CHANNELS + tl.arange(
        0, BLOCK_CHANNELS
    )
    batch_offsets = tl.arange(0, BLOCK_BATCH)[:, None]
    channel_mask = channel_offsets < CHANNELS
    mask = (batch_offsets < BATCH) & channel_mask[None, :]
    offsets = batch_offsets * CHANNELS + channel_offsets[None, :]
    value = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_x = tl.sum(value, axis=0)
    sum_x2 = tl.sum(value * value, axis=0)
    mean = sum_x / BATCH
    variance = sum_x2 / BATCH - mean * mean
    variance = tl.maximum(variance, 0.0)
    inv_variance = 1.0 / tl.sqrt(variance + epsilon)
    tl.store(saved_mean_ptr + channel_offsets, mean, mask=channel_mask)
    tl.store(
        saved_inv_var_ptr + channel_offsets,
        inv_variance,
        mask=channel_mask,
    )

    old_mean = tl.load(
        running_mean_ptr + channel_offsets,
        mask=channel_mask,
    ).to(tl.float32)
    old_variance = tl.load(
        running_var_ptr + channel_offsets,
        mask=channel_mask,
    ).to(tl.float32)
    if BATCH > 1:
        unbiased_variance = variance * (BATCH / (BATCH - 1))
    else:
        unbiased_variance = variance
    next_mean = old_mean * (1.0 - momentum) + mean * momentum
    next_variance = (
        old_variance * (1.0 - momentum) + unbiased_variance * momentum
    )
    tl.store(
        next_running_mean_ptr + channel_offsets,
        next_mean,
        mask=channel_mask,
    )
    tl.store(
        next_running_var_ptr + channel_offsets,
        next_variance,
        mask=channel_mask,
    )

    scale = tl.load(
        scale_ptr + channel_offsets,
        mask=channel_mask,
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + channel_offsets,
        mask=channel_mask,
    ).to(tl.float32)
    output = (value - mean[None, :]) * inv_variance[None, :] * scale[
        None, :
    ] + bias[None, :]
    tl.store(
        output_ptr + offsets,
        output.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


def prepare_dense_batchnorm_training(
    *,
    attrs: dict[str, Any],
    input_specs: Sequence[Any],
    default_run_fn: Any,
    shape: tuple[int, ...],
    input_checks: Any,
    batch: int,
    channels: int,
    spatial: int,
    stat_shape: tuple[int, ...],
) -> Optional[Any]:
    if (
        len(input_specs) != 7
        or input_specs[0].dtype not in ("float16", "bfloat16", "float32")
        or batch <= 0
        or channels <= 0
        or spatial <= 0
    ):
        return None

    from flag_dnn.graph.prepared import (
        PreparedSingleKernelRunSpec,
        PreparedSingleKernelSpec,
        get_prepared_output,
        make_single_kernel_run_fn,
    )
    from flag_dnn.graph.tensor import torch_dtype

    input_spec = input_specs[0]
    static_stride = tuple(int(item) for item in input_spec.stride)
    output_dtype = torch_dtype(input_spec.dtype)
    static_grid: tuple[Any, ...]
    constexpr_kwargs: dict[str, Any]
    cached_args: tuple[Any, ...]
    if spatial == 1:
        block_batch = triton.next_power_of_2(batch)
        max_block_channels = max(1, 4096 // block_batch)
        block_channels = min(
            128,
            max_block_channels,
            triton.next_power_of_2(channels),
        )
        static_grid = (triton.cdiv(channels, block_channels),)
        kernel = batchnorm_training_nc_kernel
        constexpr_kwargs = {
            "BATCH": batch,
            "CHANNELS": channels,
            "BLOCK_BATCH": block_batch,
            "BLOCK_CHANNELS": block_channels,
            "num_warps": 4,
            "num_stages": 1,
        }
        cached_args = (
            batch,
            channels,
            block_batch,
            block_channels,
        )
    else:
        block_batch = triton.next_power_of_2(batch)
        max_block_spatial = max(1, 4096 // block_batch)
        block_spatial = min(
            512,
            max_block_spatial,
            triton.next_power_of_2(spatial),
        )
        static_grid = (channels,)
        kernel = batchnorm_training_channel_kernel
        constexpr_kwargs = {
            "BATCH": batch,
            "CHANNELS": channels,
            "SPATIAL": spatial,
            "BLOCK_BATCH": block_batch,
            "BLOCK_SPATIAL": block_spatial,
            "num_warps": 4,
            "num_stages": 1,
        }
        cached_args = (
            batch,
            channels,
            spatial,
            block_batch,
            block_spatial,
        )

    output_cache: dict[tuple[Any, ...], tuple[torch.Tensor, ...]] = {}

    def output_factory(
        inputs: Sequence[Any],
    ) -> tuple[torch.Tensor, ...]:
        source = inputs[0]
        key = (
            source.device.type,
            source.device.index,
            output_dtype,
            shape,
            static_stride,
            stat_shape,
        )

        def allocate() -> tuple[torch.Tensor, ...]:
            output = torch.empty_strided(
                shape,
                static_stride,
                device=source.device,
                dtype=output_dtype,
            )
            stats = tuple(
                torch.empty(
                    stat_shape,
                    device=source.device,
                    dtype=torch.float32,
                )
                for _ in range(4)
            )
            return (output, *stats)

        return get_prepared_output(output_cache, key, allocate)

    def runtime_args(
        inputs: Sequence[Any],
        outputs: tuple[torch.Tensor, ...],
    ) -> tuple[Any, ...]:
        output, saved_mean, saved_inv_var, next_mean, next_var = outputs
        return (
            inputs[0],
            output,
            inputs[3],
            inputs[4],
            inputs[1],
            inputs[2],
            saved_mean,
            saved_inv_var,
            next_mean,
            next_var,
            float(inputs[5]),
            float(inputs[6]),
        )

    def extra_check(inputs: Sequence[Any]) -> bool:
        if not all(
            isinstance(value, torch.Tensor) and is_runtime_device_tensor(value)
            for value in inputs[:5]
        ):
            return False
        first = inputs[0]
        return all(value.device == first.device for value in inputs[1:5])

    def build_cached_call(
        constexprs: dict[str, Any],
    ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        del constexprs
        return (*static_grid, 1, 1), cached_args

    def grid(_meta: dict[str, Any]) -> tuple[int, ...]:
        return static_grid

    return make_single_kernel_run_fn(
        PreparedSingleKernelRunSpec(
            kernel=PreparedSingleKernelSpec(
                kernel=kernel,
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


def prepare_dense_batchnorm_inference(
    *,
    attrs: dict[str, Any],
    input_specs: Sequence[Any],
    default_run_fn: Any,
    shape: tuple[int, ...],
    input_checks: Any,
    total_elements: int,
    channels: int,
    spatial: int,
) -> Optional[Any]:
    if (
        len(input_specs) != 5
        or input_specs[0].dtype not in ("float16", "bfloat16", "float32")
        or total_elements <= 0
        or channels <= 0
        or spatial <= 0
    ):
        return None

    from flag_dnn.graph.prepared import (
        PreparedSingleKernelRunSpec,
        PreparedSingleKernelSpec,
        get_prepared_output,
        make_single_kernel_run_fn,
    )
    from flag_dnn.graph.tensor import torch_dtype

    input_spec = input_specs[0]
    static_stride = tuple(int(item) for item in input_spec.stride)
    output_dtype = torch_dtype(input_spec.dtype)
    batch = int(shape[0])
    rows = total_elements // spatial
    static_grid: tuple[Any, ...]
    constexpr_kwargs: dict[str, Any]
    cached_args: tuple[Any, ...]
    if spatial == 1:
        block_channels = min(256, triton.next_power_of_2(channels))
        batch_per_program = min(16, triton.next_power_of_2(batch))
        static_grid = (
            triton.cdiv(channels, block_channels),
            triton.cdiv(batch, batch_per_program),
        )
        kernel = batchnorm_inference_nc_kernel
        constexpr_kwargs = {
            "BATCH": batch,
            "CHANNELS": channels,
            "BATCH_PER_PROGRAM": batch_per_program,
            "BLOCK_CHANNELS": block_channels,
            "num_warps": 4,
            "num_stages": 1,
        }
        cached_args = (
            batch,
            channels,
            batch_per_program,
            block_channels,
        )
    else:
        if spatial < 256:
            block_columns = min(128, triton.next_power_of_2(spatial))
        else:
            block_columns = 256
        rows_per_program = min(64, max(1, 4096 // block_columns))
        static_grid = (triton.cdiv(rows, rows_per_program),)
        kernel = batchnorm_inference_row_loop_kernel
        constexpr_kwargs = {
            "ROWS": rows,
            "CHANNELS": channels,
            "SPATIAL": spatial,
            "ROWS_PER_PROGRAM": rows_per_program,
            "BLOCK_COLUMNS": block_columns,
            "num_warps": 4,
            "num_stages": 1,
        }
        cached_args = (
            rows,
            channels,
            spatial,
            rows_per_program,
            block_columns,
        )
    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def output_factory(inputs: Sequence[Any]) -> torch.Tensor:
        source = inputs[0]
        key = (
            source.device.type,
            source.device.index,
            output_dtype,
            shape,
            static_stride,
        )
        return get_prepared_output(
            output_cache,
            key,
            lambda: torch.empty_strided(
                shape,
                static_stride,
                device=source.device,
                dtype=output_dtype,
            ),
        )

    def runtime_args(
        inputs: Sequence[Any], output: torch.Tensor
    ) -> tuple[Any, ...]:
        return (
            inputs[0],
            output,
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
        )

    def extra_check(inputs: Sequence[Any]) -> bool:
        if not all(
            isinstance(value, torch.Tensor) and is_runtime_device_tensor(value)
            for value in inputs
        ):
            return False
        first = inputs[0]
        return all(value.device == first.device for value in inputs[1:])

    def build_cached_call(
        constexprs: dict[str, Any],
    ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        del constexprs
        return (*static_grid, *(1,) * (3 - len(static_grid))), cached_args

    def grid(_meta: dict[str, Any]) -> tuple[int, ...]:
        return static_grid

    return make_single_kernel_run_fn(
        PreparedSingleKernelRunSpec(
            kernel=PreparedSingleKernelSpec(
                kernel=kernel,
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
    "batchnorm_inference_core_loop_kernel",
    "batchnorm_inference_rows_kernel",
    "batchnorm_inference_row_loop_kernel",
    "batchnorm_inference_nc_kernel",
    "batchnorm_training_channel_kernel",
    "batchnorm_training_nc_kernel",
    "get_batchnorm_inference_block_size",
    "prepare_dense_batchnorm_inference",
    "prepare_dense_batchnorm_training",
)
