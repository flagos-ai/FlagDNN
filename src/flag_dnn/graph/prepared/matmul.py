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

from typing import Any, Callable, Optional, Sequence

import torch

from flag_dnn.runtime import torch_device_fn
from flag_dnn.graph.prepared import (
    PreparedTensorCache,
    RunFn,
    get_cached_empty_tensor,
    register_prepared_run_fn,
    runtime_tensor_checks_from_specs,
    runtime_tensor_checks_pass,
)
from flag_dnn.graph.prepared.common import (
    _is_runtime_device_spec,
    _require_runtime_backend,
    _static_shape,
)
from flag_dnn.graph.tensor import TensorSpec, torch_dtype
from flag_dnn.utils.device_info import get_device_capability_for


_FAST_DTYPES = {
    "float16",
    "bfloat16",
    "float32",
    "float8_e4m3fn",
    "float8_e5m2",
}


@register_prepared_run_fn("matmul")
def _prepare_matmul(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if len(input_specs) != 2:
        return None
    if not all(_is_runtime_device_spec(spec) for spec in input_specs):
        return None

    a_spec, b_spec = input_specs
    a_shape = _static_shape(a_spec)
    b_shape = _static_shape(b_spec)
    if (
        a_shape is None
        or b_shape is None
        or len(a_shape) != 3
        or len(b_shape) != 3
        or a_shape[0] != b_shape[0]
        or a_shape[2] != b_shape[1]
        or not bool(a_spec.contiguous)
        or not bool(b_spec.contiguous)
        or a_spec.dtype != b_spec.dtype
        or a_spec.dtype not in _FAST_DTYPES
    ):
        return None

    checks = runtime_tensor_checks_from_specs(
        input_specs,
        (0, 1),
        require_shape=True,
        require_stride=True,
        require_dtype=True,
    )
    if checks is None:
        return None

    from flag_dnn.ops.matmul import (
        _batched_matmul_3d_out,
        _resolve_matmul_compute_mode,
        _resolve_matmul_out_dtype,
    )
    from flag_dnn.ops.matmul_sm90 import (
        Sm90MatmulKey,
        prepare_sm90_matmul_if_supported,
        select_sm90_matmul_config,
    )

    input_dtype = torch_dtype(a_spec.dtype)
    out_dtype_attr = attrs.get("out_dtype")
    requested_out_dtype = (
        torch_dtype(out_dtype_attr) if out_dtype_attr is not None else None
    )
    out_dtype = _resolve_matmul_out_dtype(input_dtype, requested_out_dtype)
    compute_mode = _resolve_matmul_compute_mode(
        input_dtype, attrs.get("compute_data_type")
    )
    output_shape = (a_shape[0], a_shape[1], b_shape[2])
    output_cache: PreparedTensorCache = {}
    sm90_key = Sm90MatmulKey(
        a_shape[0],
        a_shape[1],
        b_shape[2],
        a_shape[2],
        input_dtype,
        out_dtype,
        compute_mode,
    )
    prepare_sm90 = select_sm90_matmul_config((9, 0), sm90_key) is not None
    cached_a: Optional[torch.Tensor] = None
    cached_b: Optional[torch.Tensor] = None
    cached_output: Optional[torch.Tensor] = None
    cached_a_data_ptr: Optional[int] = None
    cached_b_data_ptr: Optional[int] = None
    cached_output_data_ptr: Optional[int] = None
    cached_sm90_runner: Optional[Callable[[], torch.Tensor]] = None

    def run(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
        nonlocal cached_a, cached_b, cached_output, cached_sm90_runner
        nonlocal cached_a_data_ptr, cached_b_data_ptr
        nonlocal cached_output_data_ptr
        _require_runtime_backend(inputs, "matmul")
        if not runtime_tensor_checks_pass(inputs, checks):
            return default_run_fn(inputs, _attrs)
        a = inputs[0]
        b = inputs[1]
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            return default_run_fn(inputs, _attrs)
        key = (
            a.device.type,
            a.device.index,
            out_dtype,
            output_shape,
        )
        output = get_cached_empty_tensor(
            output_cache,
            key,
            output_shape,
            device=a.device,
            dtype=out_dtype,
        )
        if prepare_sm90:
            a_data_ptr = a.data_ptr()
            b_data_ptr = b.data_ptr()
            output_data_ptr = output.data_ptr()
            if (
                a is not cached_a
                or b is not cached_b
                or output is not cached_output
                or a_data_ptr != cached_a_data_ptr
                or b_data_ptr != cached_b_data_ptr
                or output_data_ptr != cached_output_data_ptr
            ):
                with torch_device_fn.device(a.device):
                    cached_sm90_runner = prepare_sm90_matmul_if_supported(
                        a,
                        b,
                        output,
                        compute_mode=compute_mode,
                        capability=get_device_capability_for(a.device),
                    )
                cached_a = a
                cached_b = b
                cached_output = output
                cached_a_data_ptr = a_data_ptr
                cached_b_data_ptr = b_data_ptr
                cached_output_data_ptr = output_data_ptr
            if cached_sm90_runner is not None:
                return cached_sm90_runner()
        return _batched_matmul_3d_out(a, b, output, compute_mode=compute_mode)

    return run
