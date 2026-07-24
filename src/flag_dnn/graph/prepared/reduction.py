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
)
from flag_dnn.graph.prepared.common import (
    _is_runtime_device_spec,
    _static_shape,
)
from flag_dnn.graph.tensor import TensorSpec, torch_dtype

_MODE_ALIASES = {"SUM": "ADD", "MEAN": "AVG", "PROD": "MUL"}
_SUPPORTED_MODES = {"ADD", "AVG", "MUL"}


def _mode_name(mode: Any) -> str:
    name = getattr(mode, "name", None)
    if name is None:
        name = str(mode).rsplit(".", 1)[-1]
    return _MODE_ALIASES.get(str(name).upper(), str(name).upper())


def _numel(shape: Sequence[int]) -> int:
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


@register_prepared_run_fn("reduction")
def _prepare_reduction(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if len(input_specs) != 1 or not _is_runtime_device_spec(input_specs[0]):
        return None
    input_spec = input_specs[0]
    shape = _static_shape(input_spec)
    if (
        shape is None
        or not shape
        or input_spec.stride is None
        or input_spec.dtype not in ("float16", "bfloat16", "float32")
    ):
        return None

    dim = attrs.get("dim")
    if not isinstance(dim, int):
        return None
    axis = dim if dim >= 0 else dim + len(shape)
    if axis < 0 or axis >= len(shape):
        return None
    channels_last = (
        input_spec.layout == "nhwc" and len(shape) == 4 and axis == 1
    )
    if not input_spec.contiguous and not channels_last:
        return None
    mode = _mode_name(attrs.get("mode"))
    if mode not in _SUPPORTED_MODES:
        return None
    requested_dtype = attrs.get("dtype")
    if requested_dtype is not None:
        try:
            if torch_dtype(requested_dtype) != torch_dtype(input_spec.dtype):
                return None
        except ValueError:
            return None

    outer = _numel(shape[:axis])
    reduced = int(shape[axis])
    inner = _numel(shape[axis + 1 :])
    rows = outer * inner
    if rows <= 0 or reduced <= 0:
        return None
    keepdim = bool(attrs.get("keepdim", True))
    output_shape = tuple(
        1 if index == axis and keepdim else size
        for index, size in enumerate(shape)
        if keepdim or index != axis
    )
    output_dtype = torch_dtype(input_spec.dtype)
    if channels_last:
        stride_outer = int(input_spec.stride[0])
        stride_reduced = int(input_spec.stride[1])
        stride_inner = int(input_spec.stride[-1])
    else:
        stride_outer = reduced * inner
        stride_reduced = inner
        stride_inner = 1
    checks = runtime_tensor_checks_from_specs(
        input_specs,
        (0,),
        require_shape=True,
        require_stride=True,
        require_dtype=True,
    )
    if checks is None:
        return None

    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def output_factory(inputs: Sequence[Any]) -> torch.Tensor:
        source = inputs[0]
        assert isinstance(source, torch.Tensor)
        key = (
            source.device.type,
            source.device.index,
            output_dtype,
            output_shape,
        )
        return get_prepared_output(
            output_cache,
            key,
            lambda: torch.empty(
                output_shape, device=source.device, dtype=output_dtype
            ),
        )

    def runtime_args(
        inputs: Sequence[Any], output: torch.Tensor
    ) -> tuple[Any, ...]:
        return (
            inputs[0],
            output,
            rows,
            reduced,
            inner,
            stride_outer,
            stride_reduced,
            stride_inner,
        )

    def grid(meta: dict[str, Any]) -> tuple[int, ...]:
        block_m = int(meta["BLOCK_M"])
        return ((rows + block_m - 1) // block_m,)

    def build_cached_call(
        constexprs: dict[str, Any],
    ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        block_m = int(constexprs["BLOCK_M"])
        block_n = int(constexprs["BLOCK_N"])
        return ((rows + block_m - 1) // block_m, 1, 1), (
            mode,
            block_m,
            block_n,
        )

    from flag_dnn.ops.reduction import _reduction_3d_kernel

    return make_single_kernel_run_fn(
        PreparedSingleKernelRunSpec(
            kernel=PreparedSingleKernelSpec(
                kernel=_reduction_3d_kernel,
                grid=grid,
                static_args=(),
                constexpr_kwargs={"OP": mode},
                build_cached_call=build_cached_call,
            ),
            input_checks=checks,
            output_factory=output_factory,
            runtime_args=runtime_args,
            validate_inputs=bool(attrs.get("_validate_inputs", True)),
        ),
        default_run_fn,
    )
