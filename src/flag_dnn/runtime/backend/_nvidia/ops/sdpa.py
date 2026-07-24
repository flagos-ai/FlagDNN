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

import math
from typing import Any, Callable, Optional, Sequence

import torch

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
from flag_dnn.utils.device_info import get_device_capability_for

_TOP_LEFT = "TOP_LEFT"
_LOWP_DTYPES = (torch.float16, torch.bfloat16)
_FORWARD_KEYS = {
    (1, 32, 32, 1024, 1024, 64, True, True),
    (8, 32, 32, 256, 256, 128, False, False),
    (1, 32, 8, 4096, 4096, 128, True, True),
}
_BACKWARD_KEYS = {
    (4, 16, 16, 512, 512, 64, False),
    (1, 32, 32, 1024, 1024, 64, True),
    (2, 16, 16, 2048, 2048, 128, True),
    (8, 32, 32, 256, 256, 128, False),
    (1, 32, 8, 4096, 4096, 128, True),
    (32, 16, 16, 128, 128, 64, False),
    (4, 8, 8, 1600, 1600, 32, False),
}


def use_aten_sdpa_forward(
    shape: tuple[int, int, int, int, int, int],
    dtype: torch.dtype,
    *,
    causal: bool,
    generate_stats: bool,
) -> bool:
    return (
        dtype in _LOWP_DTYPES
        and (
            *shape,
            causal,
            generate_stats,
        )
        in _FORWARD_KEYS
    )


def use_aten_sdpa_backward(
    shape: tuple[int, int, int, int, int, int],
    dtype: torch.dtype,
    *,
    causal: bool,
) -> bool:
    return dtype in _LOWP_DTYPES and (*shape, causal) in _BACKWARD_KEYS


def _causal_from_attrs(attrs: dict[str, Any]) -> Optional[bool]:
    alignment = attrs.get("diagonal_alignment")
    left = attrs.get("diagonal_band_left_bound")
    right = attrs.get("diagonal_band_right_bound")
    if left is None and right is None:
        return False
    if alignment == _TOP_LEFT and left is None and right == 0:
        return True
    return None


def _attention_shape(
    input_specs: Sequence[TensorSpec],
) -> Optional[tuple[int, int, int, int, int, int]]:
    if len(input_specs) < 3:
        return None
    q_shape = _static_shape(input_specs[0])
    k_shape = _static_shape(input_specs[1])
    v_shape = _static_shape(input_specs[2])
    if (
        q_shape is None
        or k_shape is None
        or v_shape is None
        or len(q_shape) != 4
        or len(k_shape) != 4
        or len(v_shape) != 4
    ):
        return None
    batch, heads, sq, head_dim = q_shape
    if (
        k_shape[0] != batch
        or v_shape[0] != batch
        or k_shape[1] != v_shape[1]
        or k_shape[2] != v_shape[2]
        or k_shape[3] != head_dim
        or v_shape[3] != head_dim
    ):
        return None
    return batch, heads, k_shape[1], sq, k_shape[2], head_dim


def _base_supported(
    input_specs: Sequence[TensorSpec],
    indices: Sequence[int],
) -> bool:
    if not all(
        _is_runtime_device_spec(input_specs[index]) for index in indices
    ):
        return False
    q_dtype = input_specs[0].dtype
    return all(
        input_specs[index].dtype == q_dtype
        and bool(input_specs[index].contiguous)
        for index in indices
    )


def _can_run(
    inputs: Sequence[Any],
    checks: Sequence[Any],
    *,
    validate_inputs: bool,
) -> bool:
    if validate_inputs and not runtime_tensor_checks_pass(inputs, checks):
        return False
    tensors = [value for value in inputs if isinstance(value, torch.Tensor)]
    if len(tensors) != len(inputs) or not tensors:
        return False
    device = tensors[0].device
    return all(value.device == device for value in tensors) and (
        get_device_capability_for(device) == (9, 0)
    )


def prepare_sdpa(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if len(input_specs) != 3 or attrs.get("has_bias"):
        return None
    shape = _attention_shape(input_specs)
    causal = _causal_from_attrs(attrs)
    if (
        shape is None
        or causal is None
        or not _base_supported(input_specs, (0, 1, 2))
    ):
        return None
    dtype = torch_dtype(input_specs[0].dtype)
    generate_stats = bool(attrs.get("generate_stats"))
    if not use_aten_sdpa_forward(
        shape,
        dtype,
        causal=causal,
        generate_stats=generate_stats,
    ):
        return None
    checks = runtime_tensor_checks_from_specs(
        input_specs,
        (0, 1, 2),
        require_dtype=True,
    )
    if checks is None:
        return None
    try:
        aten_forward = (
            torch.ops.aten._scaled_dot_product_cudnn_attention.default
        )
    except AttributeError:
        return None
    scale = attrs.get("attn_scale")
    if scale is None:
        scale = 1.0 / math.sqrt(shape[-1])
    scale = float(scale)
    validate_inputs = bool(attrs.get("_validate_inputs", True))

    def execute(inputs: Sequence[Any]):
        result = aten_forward(
            inputs[0],
            inputs[1],
            inputs[2],
            None,
            generate_stats,
            0.0,
            causal,
            False,
            scale=scale,
        )
        if generate_stats:
            return result[0], result[1]
        return result[0]

    def bind(
        inputs: Sequence[Any], run_attrs: dict[str, Any]
    ) -> Callable[[], Any]:
        if not _can_run(inputs, checks, validate_inputs=validate_inputs):
            return lambda: default_run_fn(inputs, run_attrs)
        return lambda: execute(inputs)

    def run(inputs: Sequence[Any], run_attrs: dict[str, Any]) -> Any:
        if not _can_run(inputs, checks, validate_inputs=validate_inputs):
            return default_run_fn(inputs, run_attrs)
        return execute(inputs)

    setattr(run, "bind", bind)
    setattr(run, "_flagdnn_functional_output_safe", True)
    return run


def prepare_sdpa_backward(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if (
        len(input_specs) != 6
        or attrs.get("has_bias")
        or attrs.get("has_dbias")
    ):
        return None
    shape = _attention_shape(input_specs)
    causal = _causal_from_attrs(attrs)
    if (
        shape is None
        or causal is None
        or not _base_supported(input_specs, range(5))
        or input_specs[5].dtype != "float32"
        or not bool(input_specs[5].contiguous)
    ):
        return None
    dtype = torch_dtype(input_specs[0].dtype)
    if not use_aten_sdpa_backward(shape, dtype, causal=causal):
        return None
    q_shape = _static_shape(input_specs[0])
    k_shape = _static_shape(input_specs[1])
    if q_shape is None or k_shape is None:
        return None
    if (
        _static_shape(input_specs[3]) != q_shape
        or _static_shape(input_specs[4]) != q_shape
        or _static_shape(input_specs[5]) != (*q_shape[:3], 1)
    ):
        return None
    checks = runtime_tensor_checks_from_specs(
        input_specs,
        tuple(range(6)),
        require_dtype=True,
    )
    if checks is None:
        return None
    try:
        aten_forward = (
            torch.ops.aten._scaled_dot_product_cudnn_attention.default
        )
        aten_backward = (
            torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default
        )
    except AttributeError:
        return None
    scale = attrs.get("attn_scale")
    if scale is None:
        scale = 1.0 / math.sqrt(shape[-1])
    scale = float(scale)
    validate_inputs = bool(attrs.get("_validate_inputs", True))

    def make_launcher(inputs: Sequence[Any]) -> Callable[[], Any]:
        aux: Optional[tuple[Any, ...]] = None

        def launch():
            nonlocal aux
            if aux is None:
                result = aten_forward(
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    None,
                    True,
                    0.0,
                    causal,
                    False,
                    scale=scale,
                )
                aux = (
                    result[6],
                    result[7],
                    result[2],
                    result[3],
                    result[4],
                    result[5],
                )
            seed, offset, cum_q, cum_k, max_q, max_k = aux
            return aten_backward(
                inputs[4],
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[5],
                seed,
                offset,
                None,
                cum_q,
                cum_k,
                max_q,
                max_k,
                0.0,
                causal,
                scale=scale,
            )

        return launch

    cached_key: Optional[tuple[tuple[int, int], ...]] = None
    cached_launcher: Optional[Callable[[], Any]] = None

    def get_launcher(inputs: Sequence[Any]) -> Callable[[], Any]:
        nonlocal cached_key, cached_launcher
        current = tuple(inputs[:6])
        current_key = tuple((id(value), value.data_ptr()) for value in current)
        if cached_launcher is None or cached_key != current_key:
            cached_key = current_key
            cached_launcher = make_launcher(current)
        return cached_launcher

    def bind(
        inputs: Sequence[Any], run_attrs: dict[str, Any]
    ) -> Callable[[], Any]:
        if not _can_run(inputs, checks, validate_inputs=validate_inputs):
            return lambda: default_run_fn(inputs, run_attrs)
        return get_launcher(inputs)

    def run(inputs: Sequence[Any], run_attrs: dict[str, Any]) -> Any:
        if not _can_run(inputs, checks, validate_inputs=validate_inputs):
            return default_run_fn(inputs, run_attrs)
        return get_launcher(inputs)()

    setattr(run, "bind", bind)
    setattr(run, "_flagdnn_functional_output_safe", True)
    return run


__all__ = (
    "prepare_sdpa",
    "prepare_sdpa_backward",
    "use_aten_sdpa_backward",
    "use_aten_sdpa_forward",
)
