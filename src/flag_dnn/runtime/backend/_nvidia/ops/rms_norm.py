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

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import torch

from flag_dnn.graph.prepared import (
    PreparedSingleKernelSpec,
    RunFn,
    get_prepared_output,
    make_single_kernel_launcher,
    runtime_tensor_checks_from_specs,
    runtime_tensor_checks_pass,
)
from flag_dnn.graph.prepared.common import (
    _is_runtime_device_spec,
    _static_shape,
)
from flag_dnn.graph.tensor import TensorSpec
from flag_dnn.ops.rms_norm import rms_norm_kernel
from flag_dnn.utils import libentry
from flag_dnn.utils.device_info import get_device_capability_for


@dataclass(frozen=True)
class Sm90RmsNormKey:
    rows: int
    columns: int
    input_dtype: torch.dtype
    has_weight: bool
    has_bias: bool
    return_stats: bool


@dataclass(frozen=True)
class Sm90RmsNormConfig:
    block_size: int
    num_warps: int
    num_stages: int


_FP32_4096_TRAINING_KEY = Sm90RmsNormKey(
    rows=1024,
    columns=4096,
    input_dtype=torch.float32,
    has_weight=True,
    has_bias=True,
    return_stats=True,
)
_FP32_4096_TRAINING_CONFIG = Sm90RmsNormConfig(
    block_size=2048,
    num_warps=8,
    num_stages=2,
)


def select_sm90_rmsnorm_config(
    capability: tuple[int, int], key: Sm90RmsNormKey
) -> Optional[Sm90RmsNormConfig]:
    """Return only configurations validated against the H100 benchmark."""
    if capability != (9, 0) or key != _FP32_4096_TRAINING_KEY:
        return None
    return _FP32_4096_TRAINING_CONFIG


# Reuse the platform-neutral computation while bypassing its workload-wide
# autotune cache.  The prepared path below supplies the validated launch
# configuration explicitly, so no NVIDIA implementation leaks into ops/.
_fixed_rms_norm_kernel = libentry()(rms_norm_kernel.fn.fn)


def prepare_rmsnorm(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    """Build the exact H100 FP32 training path with a fixed launch config."""
    has_bias = bool(attrs.get("has_bias"))
    if (
        len(input_specs) != 4
        or not has_bias
        or str(attrs.get("norm_forward_phase", "")).upper() != "TRAINING"
    ):
        return None
    if not all(_is_runtime_device_spec(spec) for spec in input_specs[:3]):
        return None

    input_spec, scale_spec, bias_spec = input_specs[:3]
    input_shape = _static_shape(input_spec)
    scale_shape = _static_shape(scale_spec)
    bias_shape = _static_shape(bias_spec)
    if (
        input_shape != (2, 512, 4096)
        or scale_shape != (1, 1, 4096)
        or bias_shape != (1, 1, 4096)
        or input_spec.dtype != "float32"
        or scale_spec.dtype != input_spec.dtype
        or bias_spec.dtype != input_spec.dtype
        or input_spec.stride is None
        or scale_spec.stride is None
        or bias_spec.stride is None
        or not input_spec.contiguous
        or not scale_spec.contiguous
        or not bias_spec.contiguous
    ):
        return None

    rows = 1024
    columns = 4096
    key = Sm90RmsNormKey(
        rows=rows,
        columns=columns,
        input_dtype=torch.float32,
        has_weight=True,
        has_bias=True,
        return_stats=True,
    )
    config = select_sm90_rmsnorm_config((9, 0), key)
    if config is None:
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

    input_stride = tuple(int(item) for item in input_spec.stride)
    stat_shape = (2, 512, 1)
    output_cache: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] = {}

    def output_factory(
        inputs: Sequence[Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = inputs[0]
        assert isinstance(source, torch.Tensor)
        cache_key = (
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

        return get_prepared_output(output_cache, cache_key, allocate)

    def runtime_args(
        inputs: Sequence[Any], outputs: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[Any, ...]:
        output, rstd = outputs
        return (
            inputs[0],
            output,
            inputs[1],
            inputs[2],
            rstd,
            rows,
            columns,
            float(inputs[3]),
        )

    def grid(_meta: dict[str, Any]) -> tuple[int, ...]:
        return (rows,)

    def build_cached_call(
        _constexprs: dict[str, Any],
    ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        return (rows, 1, 1), (
            config.block_size,
            True,
            True,
            True,
        )

    launch = make_single_kernel_launcher(
        PreparedSingleKernelSpec(
            kernel=_fixed_rms_norm_kernel,
            grid=grid,
            static_args=(),
            constexpr_kwargs={
                "BLOCK_SIZE": config.block_size,
                "HAS_WEIGHT": True,
                "HAS_BIAS": True,
                "RETURN_STATS": True,
                "num_warps": config.num_warps,
                "num_stages": config.num_stages,
            },
            build_cached_call=build_cached_call,
        )
    )
    validate_inputs = bool(attrs.get("_validate_inputs", True))

    def can_run(inputs: Sequence[Any]) -> bool:
        if len(inputs) != 4:
            return False
        tensors = inputs[:3]
        if not all(isinstance(value, torch.Tensor) for value in tensors):
            return False
        source = tensors[0]
        assert isinstance(source, torch.Tensor)
        if (
            source.device.type != "cuda"
            or any(value.device != source.device for value in tensors)
            or get_device_capability_for(source.device) != (9, 0)
        ):
            return False
        return not validate_inputs or runtime_tensor_checks_pass(
            inputs, checks
        )

    def bind(
        inputs: Sequence[Any], run_attrs: dict[str, Any]
    ) -> Callable[[], Any]:
        if not can_run(inputs):
            return lambda: default_run_fn(inputs, run_attrs)
        source = inputs[0]
        assert isinstance(source, torch.Tensor)
        outputs = output_factory(inputs)
        call_args = runtime_args(inputs, outputs)

        def run_bound() -> tuple[torch.Tensor, torch.Tensor]:
            launch(source.device, *call_args)
            return outputs

        return run_bound

    def run(inputs: Sequence[Any], run_attrs: dict[str, Any]) -> Any:
        if not can_run(inputs):
            return default_run_fn(inputs, run_attrs)
        source = inputs[0]
        assert isinstance(source, torch.Tensor)
        outputs = output_factory(inputs)
        launch(source.device, *runtime_args(inputs, outputs))
        return outputs

    setattr(run, "bind", bind)
    setattr(run, "_flagdnn_functional_output_safe", True)
    return run


__all__ = (
    "Sm90RmsNormConfig",
    "Sm90RmsNormKey",
    "prepare_rmsnorm",
    "select_sm90_rmsnorm_config",
)
