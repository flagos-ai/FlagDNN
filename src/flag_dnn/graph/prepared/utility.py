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


def _numel(shape: Sequence[int]) -> int:
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


@register_prepared_run_fn("gen_index")
def _prepare_gen_index(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if len(input_specs) != 1 or not _is_runtime_device_spec(input_specs[0]):
        return None
    input_spec = input_specs[0]
    shape = _static_shape(input_spec)
    if shape is None or not shape:
        return None
    axis = attrs.get("axis")
    if not isinstance(axis, int) or axis < 0 or axis >= len(shape):
        return None
    try:
        output_dtype = torch_dtype(attrs.get("compute_data_type") or "int32")
    except ValueError:
        return None
    elements = _numel(shape)
    axis_size = int(shape[axis])
    inner_size = _numel(shape[axis + 1 :])
    if elements <= 0 or axis_size <= 0:
        return None

    checks = runtime_tensor_checks_from_specs(
        input_specs,
        (0,),
        require_shape=True,
        require_stride=False,
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
            shape,
        )
        return get_prepared_output(
            output_cache,
            key,
            lambda: torch.empty(
                shape, device=source.device, dtype=output_dtype
            ),
        )

    def runtime_args(
        _inputs: Sequence[Any], output: torch.Tensor
    ) -> tuple[Any, ...]:
        return output, elements

    def grid(meta: dict[str, Any]) -> tuple[int, ...]:
        block_size = int(meta["BLOCK_SIZE"])
        return ((elements + block_size - 1) // block_size,)

    def build_cached_call(
        constexprs: dict[str, Any],
    ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        block_size = int(constexprs["BLOCK_SIZE"])
        return (
            (elements + block_size - 1) // block_size,
            1,
            1,
        ), (axis_size, inner_size, block_size)

    from flag_dnn.ops.gen_index import _gen_index_kernel

    return make_single_kernel_run_fn(
        PreparedSingleKernelRunSpec(
            kernel=PreparedSingleKernelSpec(
                kernel=_gen_index_kernel,
                grid=grid,
                static_args=(axis_size, inner_size),
                constexpr_kwargs={},
                build_cached_call=build_cached_call,
            ),
            input_checks=checks,
            output_factory=output_factory,
            runtime_args=runtime_args,
            validate_inputs=bool(attrs.get("_validate_inputs", True)),
        ),
        default_run_fn,
    )
