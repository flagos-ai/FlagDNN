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

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional, Sequence

import torch

from flag_dnn import runtime
from flag_dnn.graph.tensor import TensorSpec, torch_dtype

__all__ = (
    "RunFn",
    "PrepareRunFn",
    "GenericPrepareRunFn",
    "PreparedTensorCache",
    "RuntimeTensorCheck",
    "PreparedPipelineStepSpec",
    "PreparedKernelPipelineSpec",
    "PreparedSingleKernelSpec",
    "PreparedSingleKernelRunSpec",
    "make_kernel_pipeline_launcher",
    "make_kernel_pipeline_run_fn",
    "make_static_cached_call",
    "get_cached_empty_tensor",
    "get_prepared_output",
    "make_single_kernel_launcher",
    "make_single_kernel_run_fn",
    "prepared_output_reuse",
    "prepare_run_fn",
    "register_generic_prepared_run_fn",
    "register_prepared_run_fn",
    "runtime_tensor_checks_from_specs",
    "runtime_tensor_checks_pass",
)

RunFn = Callable[[Sequence[Any], dict[str, Any]], Any]
PrepareRunFn = Callable[
    [dict[str, Any], Sequence[TensorSpec], RunFn], Optional[RunFn]
]
GenericPrepareRunFn = Callable[
    [str, dict[str, Any], Sequence[TensorSpec], RunFn], Optional[RunFn]
]
KernelGrid = Callable[[dict[str, Any]], tuple[int, ...]]
CachedKernelCallBuilder = Callable[
    [dict[str, Any]], tuple[tuple[int, ...], tuple[Any, ...]]
]
OutputFactory = Callable[[Sequence[Any]], Any]
RuntimeArgsBuilder = Callable[[Sequence[Any], Any], tuple[Any, ...]]
ResultBuilder = Callable[[Any], Any]
RuntimeGuard = Callable[[Sequence[Any]], bool]
PreLaunchHook = Callable[[], None]
PipelineContextFactory = Callable[[Sequence[Any]], Any]
PipelineStepArgsBuilder = Callable[[Sequence[Any], Any], tuple[Any, ...]]
PreparedTensorCache = dict[tuple[Any, ...], torch.Tensor]

_REUSE_PREPARED_OUTPUTS: ContextVar[bool] = ContextVar(
    "flagdnn_reuse_prepared_outputs", default=True
)
_FUNCTIONAL_OUTPUT_SAFE = "_flagdnn_functional_output_safe"


@contextmanager
def prepared_output_reuse(enabled: bool) -> Iterator[None]:
    """Select cached replay outputs or independent functional outputs."""
    token = _REUSE_PREPARED_OUTPUTS.set(enabled)
    try:
        yield
    finally:
        _REUSE_PREPARED_OUTPUTS.reset(token)


def _prepared_outputs_are_reused() -> bool:
    return _REUSE_PREPARED_OUTPUTS.get()


def get_prepared_output(
    cache: dict[tuple[Any, ...], Any],
    key: tuple[Any, ...],
    factory: Callable[[], Any],
) -> Any:
    """Allocate per call unless the caller explicitly selected replay mode."""
    if not _prepared_outputs_are_reused():
        return factory()
    output = cache.get(key)
    if output is None:
        output = factory()
        cache[key] = output
    return output


def _mark_functional_output_safe(run_fn: RunFn) -> RunFn:
    setattr(run_fn, _FUNCTIONAL_OUTPUT_SAFE, True)
    return run_fn


def _prepared_run_supports_independent_outputs(run_fn: RunFn) -> bool:
    return bool(getattr(run_fn, _FUNCTIONAL_OUTPUT_SAFE, False))


def _wrap_prepared_output_ownership(
    prepared_run_fn: RunFn, default_run_fn: RunFn
) -> RunFn:
    if _prepared_run_supports_independent_outputs(prepared_run_fn):
        return prepared_run_fn

    def run(inputs: Sequence[Any], run_attrs: dict[str, Any]) -> Any:
        if _prepared_outputs_are_reused():
            return prepared_run_fn(inputs, run_attrs)
        return default_run_fn(inputs, run_attrs)

    binder = getattr(prepared_run_fn, "bind", None)
    if binder is not None:
        setattr(run, "bind", binder)
    return run


def _identity_result(output: Any) -> Any:
    return output


def make_static_cached_call(
    grid: tuple[int, ...], args: tuple[Any, ...]
) -> CachedKernelCallBuilder:
    """Build a cached-call adapter when replay grid and args are fixed."""

    def build(
        metadata: dict[str, Any],
    ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        return grid, args

    return build


def get_cached_empty_tensor(
    cache: PreparedTensorCache,
    key: tuple[Any, ...],
    size: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return get_prepared_output(
        cache,
        key,
        lambda: torch.empty(size, device=device, dtype=dtype),
    )


@dataclass(frozen=True)
class RuntimeTensorCheck:
    index: int
    shape: Optional[tuple[int, ...]] = None
    stride: Optional[tuple[int, ...]] = None
    dtype: Optional[torch.dtype] = None


def runtime_tensor_checks_from_specs(
    input_specs: Sequence[TensorSpec],
    indices: Sequence[int],
    *,
    require_shape: bool = True,
    require_stride: bool = True,
    require_dtype: bool = False,
) -> Optional[tuple[RuntimeTensorCheck, ...]]:
    checks: list[RuntimeTensorCheck] = []
    for index in indices:
        if index < 0 or index >= len(input_specs):
            return None
        spec = input_specs[index]
        shape = None
        if require_shape:
            if not all(isinstance(dim, int) for dim in spec.shape):
                return None
            shape = tuple(int(dim) for dim in spec.shape)
        stride = None
        if require_stride:
            if spec.stride is None:
                return None
            stride = tuple(int(dim) for dim in spec.stride)
        dtype = torch_dtype(spec.dtype) if require_dtype else None
        checks.append(
            RuntimeTensorCheck(index, shape=shape, stride=stride, dtype=dtype)
        )
    return tuple(checks)


@dataclass(frozen=True)
class PreparedSingleKernelSpec:
    kernel: Any
    grid: KernelGrid
    static_args: tuple[Any, ...]
    constexpr_kwargs: dict[str, Any]
    build_cached_call: CachedKernelCallBuilder


@dataclass(frozen=True)
class PreparedSingleKernelRunSpec:
    kernel: PreparedSingleKernelSpec
    input_checks: tuple[RuntimeTensorCheck, ...]
    output_factory: OutputFactory
    runtime_args: RuntimeArgsBuilder
    result: ResultBuilder = _identity_result
    extra_check: Optional[RuntimeGuard] = None
    pre_launch: Optional[PreLaunchHook] = None
    device_input_index: int = 0
    validate_inputs: bool = True


@dataclass(frozen=True)
class PreparedPipelineStepSpec:
    """One kernel launch in a prepared multi-kernel replay pipeline."""

    kernel: Any
    grid: Any
    runtime_args: PipelineStepArgsBuilder
    static_args: tuple[Any, ...] = ()
    constexpr_kwargs: dict[str, Any] = field(default_factory=dict)
    build_cached_call: Optional[CachedKernelCallBuilder] = None
    first_launch_returns_metadata: bool = False


@dataclass(frozen=True)
class PreparedKernelPipelineSpec:
    """Fixed-order kernel pipeline plus replay-time checks and context."""

    steps: tuple[PreparedPipelineStepSpec, ...]
    input_checks: tuple[RuntimeTensorCheck, ...]
    context_factory: PipelineContextFactory
    result: ResultBuilder = _identity_result
    extra_check: Optional[RuntimeGuard] = None
    pre_launch: Optional[PreLaunchHook] = None
    device_input_index: int = 0


_OP_PREPARED_RUN_FN_REGISTRY: dict[str, list[PrepareRunFn]] = {}
_GENERIC_PREPARED_RUN_FNS: list[GenericPrepareRunFn] = []


def register_prepared_run_fn(
    op_type: str,
) -> Callable[[PrepareRunFn], PrepareRunFn]:
    """Register a graph compile-time fast path for one op type."""

    def decorator(prepare_fn: PrepareRunFn) -> PrepareRunFn:
        preparers = _OP_PREPARED_RUN_FN_REGISTRY.setdefault(op_type, [])
        if prepare_fn not in preparers:
            preparers.append(prepare_fn)
        return prepare_fn

    return decorator


def register_generic_prepared_run_fn(
    prepare_fn: GenericPrepareRunFn,
) -> GenericPrepareRunFn:
    """Register a graph compile-time fast path shared by many op types."""
    if prepare_fn not in _GENERIC_PREPARED_RUN_FNS:
        _GENERIC_PREPARED_RUN_FNS.append(prepare_fn)
    return prepare_fn


def make_single_kernel_launcher(
    spec: PreparedSingleKernelSpec,
) -> Callable[..., None]:
    kernel_entry = spec.kernel
    grid = spec.grid
    static_args = spec.static_args
    constexpr_kwargs = spec.constexpr_kwargs
    build_cached_call = spec.build_cached_call
    cached_launcher: Any = None
    cached_static_args: tuple[Any, ...] = ()

    def launch(device: Any, *runtime_args: Any) -> None:
        nonlocal cached_launcher, cached_static_args
        launcher = cached_launcher
        if launcher is None:
            with runtime.torch_device_fn.device(device):
                compiled_kernel, constexprs = kernel_entry[grid](
                    *runtime_args, *static_args, **constexpr_kwargs
                )
            static_grid, cached_static_args = build_cached_call(constexprs)
            cached_launcher = compiled_kernel[static_grid]
        else:
            launcher(*runtime_args, *cached_static_args)

    return launch


def _make_pipeline_step_launcher(
    spec: PreparedPipelineStepSpec,
) -> Callable[[Any, Sequence[Any], Any], None]:
    kernel_entry = spec.kernel
    grid = spec.grid
    static_args = spec.static_args
    constexpr_kwargs = spec.constexpr_kwargs
    runtime_args = spec.runtime_args
    build_cached_call = spec.build_cached_call
    first_launch_returns_metadata = spec.first_launch_returns_metadata
    cached_launcher: Any = None
    cached_static_args: tuple[Any, ...] = ()

    def launch(device: Any, inputs: Sequence[Any], context: Any) -> None:
        nonlocal cached_launcher, cached_static_args
        call_args = runtime_args(inputs, context)
        launcher = cached_launcher
        if launcher is None:
            with runtime.torch_device_fn.device(device):
                first_result = kernel_entry[grid](
                    *call_args, *static_args, **constexpr_kwargs
                )
            if first_launch_returns_metadata:
                compiled_kernel, metadata = first_result
                if build_cached_call is None:
                    raise RuntimeError(
                        "pipeline step metadata requires build_cached_call"
                    )
                static_grid, cached_static_args = build_cached_call(metadata)
            else:
                compiled_kernel = first_result
                if build_cached_call is None:
                    static_grid = grid
                    cached_static_args = static_args
                else:
                    static_grid, cached_static_args = build_cached_call({})
            # The first call already launched through libentry/Triton.
            # Cache a direct launcher for later graph replays.
            cached_launcher = compiled_kernel[static_grid]
        else:
            launcher(*call_args, *cached_static_args)

    return launch


def make_kernel_pipeline_launcher(
    spec: PreparedKernelPipelineSpec,
) -> Callable[[Any, Sequence[Any], Any], None]:
    step_launchers = tuple(
        _make_pipeline_step_launcher(step) for step in spec.steps
    )

    def launch(device: Any, inputs: Sequence[Any], context: Any) -> None:
        for step in step_launchers:
            step(device, inputs, context)

    return launch


def runtime_tensor_checks_pass(
    inputs: Sequence[Any], checks: Sequence[RuntimeTensorCheck]
) -> bool:
    for check in checks:
        if check.index < 0 or check.index >= len(inputs):
            return False
        value = inputs[check.index]
        if not isinstance(value, torch.Tensor):
            return False
        if check.shape is not None and tuple(value.shape) != check.shape:
            return False
        if check.stride is not None and value.stride() != check.stride:
            return False
        if check.dtype is not None and value.dtype != check.dtype:
            return False
    return True


def make_single_kernel_run_fn(
    spec: PreparedSingleKernelRunSpec,
    default_run_fn: RunFn,
) -> RunFn:
    launch = make_single_kernel_launcher(spec.kernel)
    input_checks = spec.input_checks
    output_factory = spec.output_factory
    runtime_args = spec.runtime_args
    result = spec.result
    extra_check = spec.extra_check
    pre_launch = spec.pre_launch
    device_input_index = spec.device_input_index
    validate_inputs = spec.validate_inputs

    def bind(
        inputs: Sequence[Any], run_attrs: dict[str, Any]
    ) -> Callable[[], Any]:
        if validate_inputs:
            if not runtime_tensor_checks_pass(inputs, input_checks):
                return lambda: default_run_fn(inputs, run_attrs)
            if extra_check is not None and not extra_check(inputs):
                return lambda: default_run_fn(inputs, run_attrs)
        device_source = inputs[device_input_index]
        if not isinstance(device_source, torch.Tensor):
            return lambda: default_run_fn(inputs, run_attrs)
        output = output_factory(inputs)
        call_args = runtime_args(inputs, output)

        def run_bound() -> Any:
            if pre_launch is not None:
                pre_launch()
            launch(device_source.device, *call_args)
            return result(output)

        return run_bound

    if not validate_inputs:
        cached_output: Any = None

        def run_unchecked(
            inputs: Sequence[Any], _run_attrs: dict[str, Any]
        ) -> Any:
            nonlocal cached_output
            if _prepared_outputs_are_reused():
                if cached_output is None:
                    cached_output = output_factory(inputs)
                output = cached_output
            else:
                output = output_factory(inputs)
            device_source = inputs[device_input_index]
            launch(
                device_source.device,
                *runtime_args(inputs, output),
            )
            return result(output)

        setattr(run_unchecked, "bind", bind)
        return _mark_functional_output_safe(run_unchecked)

    def run(inputs: Sequence[Any], run_attrs: dict[str, Any]) -> Any:
        if not runtime_tensor_checks_pass(inputs, input_checks):
            return default_run_fn(inputs, run_attrs)
        if extra_check is not None and not extra_check(inputs):
            return default_run_fn(inputs, run_attrs)
        device_source = inputs[device_input_index]
        if not isinstance(device_source, torch.Tensor):
            return default_run_fn(inputs, run_attrs)
        if pre_launch is not None:
            pre_launch()
        output = output_factory(inputs)
        launch(device_source.device, *runtime_args(inputs, output))
        return result(output)

    setattr(run, "bind", bind)
    return _mark_functional_output_safe(run)


def make_kernel_pipeline_run_fn(
    spec: PreparedKernelPipelineSpec,
    default_run_fn: RunFn,
) -> RunFn:
    launch = make_kernel_pipeline_launcher(spec)
    input_checks = spec.input_checks
    context_factory = spec.context_factory
    result = spec.result
    extra_check = spec.extra_check
    pre_launch = spec.pre_launch
    device_input_index = spec.device_input_index

    def run(inputs: Sequence[Any], run_attrs: dict[str, Any]) -> Any:
        if not runtime_tensor_checks_pass(inputs, input_checks):
            return default_run_fn(inputs, run_attrs)
        if extra_check is not None and not extra_check(inputs):
            return default_run_fn(inputs, run_attrs)
        device_source = inputs[device_input_index]
        if not isinstance(device_source, torch.Tensor):
            return default_run_fn(inputs, run_attrs)
        if pre_launch is not None:
            pre_launch()
        context = context_factory(inputs)
        launch(device_source.device, inputs, context)
        return result(context)

    return _mark_functional_output_safe(run)


def prepare_run_fn(
    op_type: str,
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> RunFn:
    for generic_prepare in _GENERIC_PREPARED_RUN_FNS:
        prepared = generic_prepare(op_type, attrs, input_specs, default_run_fn)
        if prepared is not None:
            return _wrap_prepared_output_ownership(prepared, default_run_fn)
    for op_prepare in _OP_PREPARED_RUN_FN_REGISTRY.get(op_type, []):
        prepared = op_prepare(attrs, input_specs, default_run_fn)
        if prepared is not None:
            return _wrap_prepared_output_ownership(prepared, default_run_fn)
    return default_run_fn
