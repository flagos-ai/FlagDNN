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

from flag_dnn import runtime
from flag_dnn.graph.prepared import (
    PreparedSingleKernelRunSpec,
    PreparedSingleKernelSpec,
    RunFn,
    get_prepared_output,
    make_single_kernel_run_fn,
    register_prepared_run_fn,
    runtime_tensor_checks_from_specs,
)
from flag_dnn.graph.prepared.common import (
    _is_runtime_device_spec,
    _static_shape,
)
from flag_dnn.graph.tensor import TensorSpec


def _numel(shape: Sequence[int]) -> int:
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def _norm_layout(
    input_spec: TensorSpec,
    scale_spec: TensorSpec,
    bias_spec: Optional[TensorSpec],
) -> Optional[tuple[tuple[int, ...], tuple[int, ...], int, int]]:
    input_shape = _static_shape(input_spec)
    scale_shape = _static_shape(scale_spec)
    bias_shape = None if bias_spec is None else _static_shape(bias_spec)
    if (
        input_shape is None
        or scale_shape is None
        or (bias_spec is not None and bias_shape is None)
    ):
        return None
    rank = len(input_shape)
    if not rank or len(scale_shape) > rank:
        return None
    aligned_scale = (1,) * (rank - len(scale_shape)) + scale_shape
    axes = tuple(
        index for index, size in enumerate(aligned_scale) if size != 1
    )
    if not axes:
        axes = (rank - 1,)
    if axes != tuple(range(rank - len(axes), rank)):
        return None
    normalized_shape = tuple(input_shape[index] for index in axes)
    normalized_elements = _numel(normalized_shape)
    if _numel(scale_shape) != normalized_elements or (
        bias_shape is not None and _numel(bias_shape) != normalized_elements
    ):
        return None
    total_elements = _numel(input_shape)
    if normalized_elements <= 0 or total_elements <= 0:
        return None
    stat_shape = tuple(
        1 if index in axes else size for index, size in enumerate(input_shape)
    )
    return (
        input_shape,
        stat_shape,
        total_elements // normalized_elements,
        normalized_elements,
    )


@register_prepared_run_fn("layernorm")
def _prepare_layernorm(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if len(input_specs) != 4:
        return None
    if str(attrs.get("norm_forward_phase", "")).upper() != "TRAINING":
        return None
    if not all(_is_runtime_device_spec(spec) for spec in input_specs[:3]):
        return None

    input_spec, scale_spec, bias_spec = input_specs[:3]
    if (
        input_spec.dtype not in ("float16", "bfloat16", "float32")
        or input_spec.dtype != scale_spec.dtype
        or input_spec.dtype != bias_spec.dtype
        or input_spec.stride is None
        or not input_spec.contiguous
        or scale_spec.stride is None
        or not scale_spec.contiguous
        or bias_spec.stride is None
        or not bias_spec.contiguous
    ):
        return None
    layout = _norm_layout(input_spec, scale_spec, bias_spec)
    if layout is None:
        return None
    input_shape, stat_shape, rows, columns = layout

    checks = runtime_tensor_checks_from_specs(
        input_specs,
        (0, 1, 2),
        require_shape=True,
        require_stride=True,
        require_dtype=True,
    )
    if checks is None:
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
                stat_shape, device=source.device, dtype=torch.float32
            )
            rstd = torch.empty_like(mean)
            return output, mean, rstd

        return get_prepared_output(output_cache, key, allocate)

    def runtime_args(
        inputs: Sequence[Any], outputs: tuple[torch.Tensor, ...]
    ) -> tuple[Any, ...]:
        output, mean, rstd = outputs
        return (
            inputs[0],
            output,
            mean,
            rstd,
            inputs[1],
            inputs[2],
            rows,
            float(inputs[3]),
        )

    def grid(_meta: dict[str, Any]) -> tuple[int, ...]:
        return (rows,)

    def build_cached_call(
        constexprs: dict[str, Any],
    ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        return (rows, 1, 1), (
            columns,
            int(constexprs["BLOCK_SIZE"]),
            True,
            True,
            True,
        )

    def extra_check(inputs: Sequence[Any]) -> bool:
        tensors = inputs[:3]
        return all(
            isinstance(value, torch.Tensor)
            and value.device == tensors[0].device
            for value in tensors
        )

    from flag_dnn.ops.layer_norm import layer_norm_kernel

    return make_single_kernel_run_fn(
        PreparedSingleKernelRunSpec(
            kernel=PreparedSingleKernelSpec(
                kernel=layer_norm_kernel,
                grid=grid,
                static_args=(columns,),
                constexpr_kwargs={
                    "HAS_WEIGHT": True,
                    "HAS_BIAS": True,
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


@register_prepared_run_fn("rmsnorm")
def _prepare_rmsnorm(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    backend_prepare = runtime.get_backend_hook("prepare_rmsnorm")
    if backend_prepare is not None:
        backend_run = backend_prepare(attrs, input_specs, default_run_fn)
        if backend_run is not None:
            return backend_run

    has_bias = bool(attrs.get("has_bias"))
    expected_inputs = 4 if has_bias else 3
    if len(input_specs) != expected_inputs:
        return None
    if str(attrs.get("norm_forward_phase", "")).upper() != "TRAINING":
        return None

    bias_index = 2 if has_bias else None
    epsilon_index = 3 if has_bias else 2
    tensor_indices = (0, 1, 2) if has_bias else (0, 1)
    if not all(
        _is_runtime_device_spec(input_specs[index]) for index in tensor_indices
    ):
        return None

    input_spec = input_specs[0]
    scale_spec = input_specs[1]
    bias_spec = input_specs[bias_index] if bias_index is not None else None
    if (
        input_spec.dtype not in ("float16", "bfloat16", "float32")
        or input_spec.dtype != scale_spec.dtype
        or input_spec.stride is None
        or not input_spec.contiguous
        or scale_spec.stride is None
        or not scale_spec.contiguous
    ):
        return None
    if bias_spec is not None and (
        input_spec.dtype != bias_spec.dtype
        or bias_spec.stride is None
        or not bias_spec.contiguous
    ):
        return None
    layout = _norm_layout(input_spec, scale_spec, bias_spec)
    if layout is None:
        return None
    input_shape, stat_shape, rows, columns = layout

    checks = runtime_tensor_checks_from_specs(
        input_specs,
        tensor_indices,
        require_shape=True,
        require_stride=True,
        require_dtype=True,
    )
    if checks is None:
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
            rstd = torch.empty(
                stat_shape, device=source.device, dtype=torch.float32
            )
            return output, rstd

        return get_prepared_output(output_cache, key, allocate)

    def runtime_args(
        inputs: Sequence[Any], outputs: tuple[torch.Tensor, ...]
    ) -> tuple[Any, ...]:
        output, rstd = outputs
        bias = inputs[bias_index] if bias_index is not None else inputs[0]
        return (
            inputs[0],
            output,
            inputs[1],
            bias,
            rstd,
            rows,
            columns,
            float(inputs[epsilon_index]),
        )

    def grid(_meta: dict[str, Any]) -> tuple[int, ...]:
        return (rows,)

    def build_cached_call(
        constexprs: dict[str, Any],
    ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        return (rows, 1, 1), (
            int(constexprs["BLOCK_SIZE"]),
            True,
            has_bias,
            True,
        )

    def extra_check(inputs: Sequence[Any]) -> bool:
        tensors = tuple(inputs[index] for index in tensor_indices)
        return all(
            isinstance(value, torch.Tensor)
            and value.device == tensors[0].device
            for value in tensors
        )

    from flag_dnn.ops.rms_norm import rms_norm_kernel

    return make_single_kernel_run_fn(
        PreparedSingleKernelRunSpec(
            kernel=PreparedSingleKernelSpec(
                kernel=rms_norm_kernel,
                grid=grid,
                static_args=(),
                constexpr_kwargs={
                    "HAS_WEIGHT": True,
                    "HAS_BIAS": has_bias,
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
