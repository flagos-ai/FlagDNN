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

from flag_dnn.graph.prepared import (
    RunFn,
    get_prepared_output,
    runtime_tensor_checks_from_specs,
    runtime_tensor_checks_pass,
)
from flag_dnn.graph.prepared.common import (
    _is_runtime_device_spec,
    _static_shape,
)
from flag_dnn.graph.tensor import TensorSpec, torch_dtype
from flag_dnn.ops.matmul import (
    _resolve_matmul_compute_mode,
    _resolve_matmul_out_dtype,
)
from flag_dnn.runtime.backend._nvidia.ops.matmul_sm90 import (
    Sm90MatmulKey,
    launch_sm90_matmul_if_supported,
    select_sm90_matmul_config,
)
from flag_dnn.runtime.backend._nvidia.ops.matmul_sm90_gluon import (
    prepare_sm90_matmul_dynamic_output,
)
from flag_dnn.utils.device_info import get_device_capability_for

_SUPPORTED_SPEC_DTYPES = {
    "float16",
    "bfloat16",
    "float32",
    "float8_e4m3fn",
    "float8_e5m2",
}


def matmul_3d_out(
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    *,
    compute_mode: str,
) -> bool:
    """Try an exact-shape NVIDIA kernel without exposing it to common ops."""
    if (
        a.device.type != "cuda"
        or b.device != a.device
        or output.device != a.device
    ):
        return False
    return launch_sm90_matmul_if_supported(
        a,
        b,
        output,
        compute_mode=compute_mode,
        capability=get_device_capability_for(a.device),
    )


def _matmul_key_from_specs(
    attrs: dict[str, Any],
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    input_dtype: torch.dtype,
) -> tuple[Sm90MatmulKey, torch.dtype, str]:
    out_dtype_attr = attrs.get("out_dtype")
    requested_out_dtype = (
        torch_dtype(out_dtype_attr) if out_dtype_attr is not None else None
    )
    output_dtype = _resolve_matmul_out_dtype(input_dtype, requested_out_dtype)
    compute_mode = _resolve_matmul_compute_mode(
        input_dtype, attrs.get("compute_data_type")
    )
    return (
        Sm90MatmulKey(
            a_shape[0],
            a_shape[1],
            b_shape[2],
            a_shape[2],
            input_dtype,
            output_dtype,
            compute_mode,
        ),
        output_dtype,
        compute_mode,
    )


def prepare_matmul(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    """Build the H100 graph path while preserving functional outputs."""
    if len(input_specs) != 2 or not all(
        _is_runtime_device_spec(spec) for spec in input_specs
    ):
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
        or a_spec.dtype not in _SUPPORTED_SPEC_DTYPES
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
    input_dtype = torch_dtype(a_spec.dtype)
    key, output_dtype, _compute_mode = _matmul_key_from_specs(
        attrs, a_shape, b_shape, input_dtype
    )
    config = select_sm90_matmul_config((9, 0), key)
    if config is None or config.family not in (
        "lowp",
        "lowp_cublaslt",
        "lowp_persistent",
        "fp8_tma",
        "tf32_cublaslt",
        "tf32_small",
    ):
        return None

    output_shape = (a_shape[0], a_shape[1], b_shape[2])
    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    validate_inputs = bool(attrs.get("_validate_inputs", True))
    cached_a: Optional[torch.Tensor] = None
    cached_b: Optional[torch.Tensor] = None
    cached_a_ptr: Optional[int] = None
    cached_b_ptr: Optional[int] = None
    cached_launcher: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    def can_run(inputs: Sequence[Any]) -> bool:
        if validate_inputs and not runtime_tensor_checks_pass(inputs, checks):
            return False
        if len(inputs) != 2:
            return False
        a, b = inputs
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            return False
        return b.device == a.device and get_device_capability_for(
            a.device
        ) == (9, 0)

    def output_factory(inputs: Sequence[Any]) -> torch.Tensor:
        source = inputs[0]
        assert isinstance(source, torch.Tensor)
        cache_key = (
            source.device.type,
            source.device.index,
            output_dtype,
            output_shape,
        )
        return get_prepared_output(
            output_cache,
            cache_key,
            lambda: torch.empty(
                output_shape, device=source.device, dtype=output_dtype
            ),
        )

    def get_launcher(
        a: torch.Tensor, b: torch.Tensor
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        nonlocal cached_a, cached_b, cached_a_ptr, cached_b_ptr
        nonlocal cached_launcher
        a_ptr = a.data_ptr()
        b_ptr = b.data_ptr()
        if (
            cached_launcher is None
            or a is not cached_a
            or b is not cached_b
            or a_ptr != cached_a_ptr
            or b_ptr != cached_b_ptr
        ):
            if config.family == "lowp_persistent":
                from .matmul_persistent_sm90 import (
                    prepare_persistent_sm90_matmul_dynamic_output,
                )

                cached_launcher = (
                    prepare_persistent_sm90_matmul_dynamic_output(
                        a,
                        b,
                        output_dtype=output_dtype,
                        config=config,
                    )
                )
            elif config.family == "lowp_cublaslt":
                from .matmul_cublaslt import (
                    prepare_cublaslt_bf16_matmul_dynamic_output,
                )

                cached_launcher = prepare_cublaslt_bf16_matmul_dynamic_output(
                    a,
                    b,
                    output_dtype=output_dtype,
                )
            elif config.family == "tf32_small":
                from .matmul_small_sm90 import (
                    prepare_small_tf32_matmul_dynamic_output,
                )

                cached_launcher = prepare_small_tf32_matmul_dynamic_output(
                    a,
                    b,
                    output_dtype=output_dtype,
                )
            elif config.family == "tf32_cublaslt":
                from .matmul_cublaslt import (
                    prepare_cublaslt_tf32_matmul_dynamic_output,
                )

                cached_launcher = prepare_cublaslt_tf32_matmul_dynamic_output(
                    a,
                    b,
                    output_dtype=output_dtype,
                )
            elif config.family == "fp8_tma":
                from .matmul_fp8_sm90 import (
                    prepare_sm90_fp8_matmul_dynamic_output,
                )

                cached_launcher = prepare_sm90_fp8_matmul_dynamic_output(
                    a,
                    b,
                    output_dtype=output_dtype,
                    config=config,
                )
            else:
                cached_launcher = prepare_sm90_matmul_dynamic_output(
                    a,
                    b,
                    output_dtype=output_dtype,
                    config=config,
                )
            cached_a = a
            cached_b = b
            cached_a_ptr = a_ptr
            cached_b_ptr = b_ptr
        return cached_launcher

    def bind(
        inputs: Sequence[Any], run_attrs: dict[str, Any]
    ) -> Callable[[], Any]:
        if not can_run(inputs):
            return lambda: default_run_fn(inputs, run_attrs)
        a, b = inputs
        assert isinstance(a, torch.Tensor)
        assert isinstance(b, torch.Tensor)
        output = output_factory(inputs)
        launcher = get_launcher(a, b)

        def run_bound() -> torch.Tensor:
            return launcher(output)

        return run_bound

    def run(inputs: Sequence[Any], run_attrs: dict[str, Any]) -> Any:
        if not can_run(inputs):
            return default_run_fn(inputs, run_attrs)
        a, b = inputs
        assert isinstance(a, torch.Tensor)
        assert isinstance(b, torch.Tensor)
        output = output_factory(inputs)
        return get_launcher(a, b)(output)

    setattr(run, "bind", bind)
    setattr(run, "_flagdnn_functional_output_safe", True)
    return run


__all__ = ("matmul_3d_out", "prepare_matmul")
