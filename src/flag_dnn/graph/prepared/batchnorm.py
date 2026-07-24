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

from flag_dnn.graph.prepared import (
    PreparedSingleKernelRunSpec,
    PreparedSingleKernelSpec,
    RunFn,
    get_prepared_output,
    make_single_kernel_run_fn,
    register_prepared_run_fn,
    runtime_tensor_checks_from_specs,
    runtime_tensor_checks_pass,
)
from flag_dnn.graph.prepared.common import (
    _is_runtime_device_spec,
    _require_runtime_backend,
    _static_shape,
)
from flag_dnn.graph.tensor import TensorSpec


def _static_numel(shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


def _valid_channel_param(spec: TensorSpec, channels: int) -> bool:
    shape = _static_shape(spec)
    if shape is None or _static_numel(shape) != channels:
        return False
    return (
        spec.dtype == "float32"
        and spec.stride is not None
        and bool(spec.contiguous)
    )


def _valid_training_param(spec: TensorSpec, channels: int, dtype: str) -> bool:
    shape = _static_shape(spec)
    return bool(
        shape is not None
        and _static_numel(shape) == channels
        and spec.dtype == dtype
        and spec.stride is not None
        and spec.contiguous
    )


@register_prepared_run_fn("batchnorm")
def _prepare_batchnorm_training(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if int(attrs.get("peer_stats_count", 0)) != 0 or len(input_specs) != 7:
        return None
    if not all(_is_runtime_device_spec(spec) for spec in input_specs[:5]):
        return None

    input_spec = input_specs[0]
    shape = _static_shape(input_spec)
    if (
        shape is None
        or len(shape) < 2
        or input_spec.stride is None
        or not bool(input_spec.contiguous)
        or input_spec.dtype not in ("float16", "bfloat16", "float32")
    ):
        return None

    channels = int(shape[1])
    if not _valid_training_param(
        input_specs[1], channels, input_spec.dtype
    ) or not _valid_training_param(input_specs[2], channels, input_spec.dtype):
        return None
    if not _valid_training_param(
        input_specs[3], channels, "float32"
    ) or not _valid_training_param(input_specs[4], channels, "float32"):
        return None

    checks = runtime_tensor_checks_from_specs(
        input_specs,
        (0, 1, 2, 3, 4),
        require_shape=True,
        require_stride=True,
        require_dtype=True,
    )
    if checks is None:
        return None

    total_elements = _static_numel(shape)
    batch = int(shape[0])
    if total_elements == 0 or batch <= 0 or channels <= 0:
        return None
    spatial = total_elements // (batch * channels)
    stat_shape = _static_shape(input_specs[3])
    if stat_shape is None:
        return None
    input_stride = tuple(int(item) for item in input_spec.stride)
    output_cache: dict[tuple[Any, ...], tuple[torch.Tensor, ...]] = {}

    def output_factory(inputs: Sequence[Any]) -> tuple[torch.Tensor, ...]:
        source = inputs[0]
        assert isinstance(source, torch.Tensor)
        key = (
            source.device.type,
            source.device.index,
            source.dtype,
            shape,
            input_stride,
            stat_shape,
        )

        def allocate() -> tuple[torch.Tensor, ...]:
            output = torch.empty_strided(
                shape,
                input_stride,
                device=source.device,
                dtype=source.dtype,
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
        inputs: Sequence[Any], outputs: tuple[torch.Tensor, ...]
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
            batch,
            channels,
            spatial,
            float(inputs[5]),
            float(inputs[6]),
        )

    def grid(_meta: dict[str, Any]) -> tuple[int, ...]:
        return (channels,)

    def build_cached_call(
        constexprs: dict[str, Any],
    ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        block_size = int(constexprs["BLOCK_SIZE"])
        return (channels, 1, 1), (
            block_size,
            True,
            True,
            True,
            True,
            True,
        )

    def extra_check(inputs: Sequence[Any]) -> bool:
        tensors = inputs[:5]
        return all(
            isinstance(value, torch.Tensor)
            and value.device == tensors[0].device
            for value in tensors
        )

    from flag_dnn.ops.batch_norm import batch_norm_fused_kernel_optimized_

    return make_single_kernel_run_fn(
        PreparedSingleKernelRunSpec(
            kernel=PreparedSingleKernelSpec(
                kernel=batch_norm_fused_kernel_optimized_,
                grid=grid,
                static_args=(),
                constexpr_kwargs={
                    "IS_TRAINING": True,
                    "HAS_WEIGHT": True,
                    "HAS_BIAS": True,
                    "HAS_RUNNING_STATS": True,
                    "RETURN_STATS": True,
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


@register_prepared_run_fn("batchnorm_inference")
def _prepare_batchnorm_inference(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    del attrs
    if len(input_specs) != 5:
        return None
    if not all(_is_runtime_device_spec(spec) for spec in input_specs):
        return None

    input_spec = input_specs[0]
    shape = _static_shape(input_spec)
    if (
        shape is None
        or len(shape) < 2
        or input_spec.stride is None
        or not bool(input_spec.contiguous)
        or input_spec.dtype not in ("float16", "bfloat16", "float32")
    ):
        return None

    channels = int(shape[1])
    if not all(
        _valid_channel_param(spec, channels) for spec in input_specs[1:]
    ):
        return None

    checks = runtime_tensor_checks_from_specs(
        input_specs,
        (0, 1, 2, 3, 4),
        require_shape=True,
        require_stride=True,
        require_dtype=True,
    )
    if checks is None:
        return None

    from flag_dnn.ops.batch_norm import batch_norm_inference_kernel

    total_elements = _static_numel(shape)
    if total_elements == 0:
        return None
    spatial = total_elements // (int(shape[0]) * channels)
    block_size = 1024
    grid = ((total_elements + block_size - 1) // block_size,)
    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def output_for(input_tensor: torch.Tensor) -> torch.Tensor:
        key = (
            input_tensor.device.type,
            input_tensor.device.index,
            input_tensor.dtype,
            shape,
            input_spec.stride,
        )
        output = output_cache.get(key)
        if output is None:
            output = torch.empty_strided(
                shape,
                tuple(int(item) for item in input_spec.stride or ()),
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )
            output_cache[key] = output
        return output

    def run(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
        _require_runtime_backend(inputs, "batchnorm_inference")
        if not runtime_tensor_checks_pass(inputs, checks):
            return default_run_fn(inputs, _attrs)

        input_tensor = inputs[0]
        if not isinstance(input_tensor, torch.Tensor):
            return default_run_fn(inputs, _attrs)
        output = output_for(input_tensor)
        batch_norm_inference_kernel[grid](
            input_tensor,
            output,
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            total_elements,
            channels,
            spatial,
            0.0,
            BLOCK_SIZE=block_size,
            HAS_WEIGHT=True,
            HAS_BIAS=True,
            STAT_IS_INV_VARIANCE=True,
        )
        return output

    return run
