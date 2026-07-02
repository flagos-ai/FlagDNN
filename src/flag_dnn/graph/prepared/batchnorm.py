from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from flag_dnn.graph.prepared import (
    RunFn,
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
