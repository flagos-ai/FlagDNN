from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Sequence

import torch

import flag_dnn.graph.prepared.matmul as prepared_matmul
from flag_dnn import runtime
from flag_dnn.graph import prepared
from flag_dnn.graph.prepared import core as prepared_core
from flag_dnn.graph.prepared import ops as prepared_ops
from flag_dnn.graph.prepared import (
    PreparedKernelPipelineSpec,
    PreparedPipelineStepSpec,
    PreparedSingleKernelRunSpec,
    PreparedSingleKernelSpec,
    RuntimeTensorCheck,
    get_cached_empty_tensor,
    make_kernel_pipeline_run_fn,
    make_single_kernel_run_fn,
    make_static_cached_call,
    register_prepared_run_fn,
    runtime_tensor_checks_from_specs,
    runtime_tensor_checks_pass,
)
from flag_dnn.graph.tensor import TensorSpec


class _CompiledKernel:
    def __init__(self, name: str, calls: list[tuple[Any, ...]]) -> None:
        self.name = name
        self.calls = calls

    def __getitem__(self, grid: Any) -> Any:
        def launch(*args: Any) -> None:
            self.calls.append((self.name, "cached", grid, args))

        return launch


class _KernelEntry:
    def __init__(
        self,
        name: str,
        calls: list[tuple[Any, ...]],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.calls = calls
        self.metadata = metadata

    def __getitem__(self, grid: Any) -> Any:
        def launch(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((self.name, "first", grid, args, kwargs))
            compiled = _CompiledKernel(self.name, self.calls)
            if self.metadata is None:
                return compiled
            return compiled, self.metadata

        return launch


def _fallback(inputs: Sequence[Any], attrs: dict[str, Any]) -> str:
    return "fallback"


def test_runtime_tensor_checks_from_specs_and_pass() -> None:
    spec = TensorSpec("x", (2, 3), "float16", stride=(3, 1))
    checks = runtime_tensor_checks_from_specs(
        (spec,), (0,), require_dtype=True
    )
    assert checks == (
        RuntimeTensorCheck(
            0, shape=(2, 3), stride=(3, 1), dtype=torch.float16
        ),
    )

    tensor = torch.empty((2, 3), dtype=torch.float16)
    assert runtime_tensor_checks_pass((tensor,), checks)
    assert not runtime_tensor_checks_pass((tensor.t(),), checks)

    dynamic = TensorSpec("x", ("n", 3), "float16", stride=(3, 1))
    assert runtime_tensor_checks_from_specs((dynamic,), (0,)) is None


def test_static_cached_call_and_tensor_cache() -> None:
    cached_call = make_static_cached_call((2, 1, 1), ("tail", 4))
    assert cached_call({"ignored": True}) == ((2, 1, 1), ("tail", 4))

    cache: dict[tuple[Any, ...], torch.Tensor] = {}
    key = ("cpu", None, torch.float32, (2, 3))
    first = get_cached_empty_tensor(
        cache, key, (2, 3), device=torch.device("cpu"), dtype=torch.float32
    )
    second = get_cached_empty_tensor(
        cache, key, (2, 3), device=torch.device("cpu"), dtype=torch.float32
    )
    assert first is second


def test_single_kernel_run_fn_caches_launcher(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        prepared_core.runtime.torch_device_fn,
        "device",
        lambda device: nullcontext(),
    )
    calls: list[tuple[Any, ...]] = []
    x = torch.empty((2, 3))

    def grid(meta: dict[str, Any]) -> tuple[int, ...]:
        return (1, 1, 1)

    def build_cached_call(
        metadata: dict[str, Any],
    ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        return (metadata["BLOCK"], 1, 1), ("cached-static",)

    def output_factory(inputs: Sequence[Any]) -> torch.Tensor:
        return torch.empty_like(inputs[0])

    def runtime_args(inputs: Sequence[Any], output: Any) -> tuple[Any, ...]:
        return inputs[0], output

    run = make_single_kernel_run_fn(
        PreparedSingleKernelRunSpec(
            kernel=PreparedSingleKernelSpec(
                kernel=_KernelEntry("single", calls, {"BLOCK": 4}),
                grid=grid,
                static_args=("first-static",),
                constexpr_kwargs={"BLOCK_HINT": 8},
                build_cached_call=build_cached_call,
            ),
            input_checks=(
                RuntimeTensorCheck(0, shape=tuple(x.shape), stride=x.stride()),
            ),
            output_factory=output_factory,
            runtime_args=runtime_args,
        ),
        _fallback,
    )

    assert isinstance(run((x,), {}), torch.Tensor)
    assert isinstance(run((x,), {}), torch.Tensor)
    assert run((x.t(),), {}) == "fallback"
    assert calls[0][0:3] == ("single", "first", grid)
    assert calls[1][0:3] == ("single", "cached", (4, 1, 1))
    assert calls[1][3][-1] == "cached-static"


def test_kernel_pipeline_run_fn_caches_each_step(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        prepared_core.runtime.torch_device_fn,
        "device",
        lambda device: nullcontext(),
    )
    calls: list[tuple[Any, ...]] = []
    x = torch.empty((2, 3))

    def make_context(inputs: Sequence[Any]) -> dict[str, Any]:
        return {"tmp": "tmp", "out": "out"}

    def split_args(inputs: Sequence[Any], context: Any) -> tuple[Any, ...]:
        return inputs[0], context["tmp"]

    def combine_args(inputs: Sequence[Any], context: Any) -> tuple[Any, ...]:
        return context["tmp"], context["out"]

    def split_cached_call(
        metadata: dict[str, Any],
    ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        return (metadata["BLOCK_N"], 1, 1), ("split-static",)

    run = make_kernel_pipeline_run_fn(
        PreparedKernelPipelineSpec(
            steps=(
                PreparedPipelineStepSpec(
                    kernel=_KernelEntry("split", calls, {"BLOCK_N": 8}),
                    grid=(1, 1, 1),
                    runtime_args=split_args,
                    static_args=("first-split",),
                    constexpr_kwargs={"BLOCK": 16},
                    build_cached_call=split_cached_call,
                    first_launch_returns_metadata=True,
                ),
                PreparedPipelineStepSpec(
                    kernel=_KernelEntry("combine", calls),
                    grid=(2, 1, 1),
                    runtime_args=combine_args,
                    static_args=("combine-static",),
                ),
            ),
            input_checks=(
                RuntimeTensorCheck(0, shape=tuple(x.shape), stride=x.stride()),
            ),
            context_factory=make_context,
            result=lambda context: context["out"],
        ),
        _fallback,
    )

    assert run((x,), {}) == "out"
    assert run((x,), {}) == "out"
    assert run((x.t(),), {}) == "fallback"
    assert calls[0][0:3] == ("split", "first", (1, 1, 1))
    assert calls[1][0:3] == ("combine", "first", (2, 1, 1))
    assert calls[2][0:3] == ("split", "cached", (8, 1, 1))
    assert calls[2][3][-1] == "split-static"
    assert calls[3][0:3] == ("combine", "cached", (2, 1, 1))
    assert calls[3][3][-1] == "combine-static"


def test_prepared_ops_import_registers_builtin_preparers() -> None:
    assert prepared_ops.prepare_run_fn is prepared.prepare_run_fn
    assert prepared_core._GENERIC_PREPARED_RUN_FNS
    for op_type in (
        "sdpa",
        "sdpa_backward",
        "conv_dgrad",
        "conv_wgrad",
        "conv_fprop",
    ):
        assert op_type in prepared_core._OP_PREPARED_RUN_FN_REGISTRY


def test_register_prepared_run_fn_uses_first_non_none_preparer() -> None:
    op_type = "__prepared_test_unique__"

    @register_prepared_run_fn(op_type)
    def skip_preparer(
        attrs: dict[str, Any], specs: Sequence[TensorSpec], default: Any
    ) -> None:
        return None

    @register_prepared_run_fn(op_type)
    def use_preparer(
        attrs: dict[str, Any], specs: Sequence[TensorSpec], default: Any
    ) -> Any:
        def run(inputs: Sequence[Any], run_attrs: dict[str, Any]) -> str:
            return "prepared"

        return run

    def default(inputs: Sequence[Any], run_attrs: dict[str, Any]) -> str:
        return "default"

    run = prepared.prepare_run_fn(op_type, {}, (), default)
    assert run((), {}) == "prepared"


def test_prepared_matmul_reuses_sm90_runner_for_identical_inputs(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        prepared_matmul, "_require_runtime_backend", lambda inputs, op: None
    )
    active_devices: list[torch.device] = []
    context_devices: list[torch.device] = []
    capability_devices: list[torch.device] = []

    class DeviceContext:
        def __init__(self, device: torch.device) -> None:
            self.device = device

        def __enter__(self) -> None:
            context_devices.append(self.device)
            active_devices.append(self.device)

        def __exit__(self, *args: Any) -> None:
            assert active_devices.pop() == self.device

    class DeviceFn:
        @staticmethod
        def device(device: torch.device) -> DeviceContext:
            return DeviceContext(device)

    def get_device_capability_for(device: torch.device) -> tuple[int, int]:
        assert active_devices[-1] == device
        capability_devices.append(device)
        return (9, 0)

    monkeypatch.setattr(
        prepared_matmul, "torch_device_fn", DeviceFn(), raising=False
    )
    monkeypatch.setattr(
        prepared_matmul,
        "get_device_capability_for",
        get_device_capability_for,
        raising=False,
    )
    from flag_dnn.ops import matmul_sm90

    prepare_calls: list[tuple[Any, Any, Any]] = []
    launch_calls: list[tuple[Any, Any, Any]] = []

    def prepare_runner(a: Any, b: Any, c: Any, **kwargs: Any) -> Any:
        assert active_devices[-1] == a.device
        prepare_calls.append((a, b, c))

        def launch() -> Any:
            launch_calls.append((a, b, c))
            return c

        return launch

    monkeypatch.setattr(
        matmul_sm90,
        "prepare_sm90_matmul_if_supported",
        prepare_runner,
        raising=False,
    )
    shape = (16, 1024, 1024)
    stride = (1024 * 1024, 1024, 1)
    specs = (
        TensorSpec(
            "a",
            shape,
            "float16",
            stride=stride,
            device=runtime.device.name,
            contiguous=True,
        ),
        TensorSpec(
            "b",
            shape,
            "float16",
            stride=stride,
            device=runtime.device.name,
            contiguous=True,
        ),
    )
    fallback_calls: list[tuple[Any, ...]] = []

    def fallback(inputs: Sequence[Any], attrs: dict[str, Any]) -> Any:
        fallback_calls.append(tuple(inputs))
        return "fallback"

    run = prepared_matmul._prepare_matmul(
        {"compute_data_type": "float32", "out_dtype": torch.float16},
        specs,
        fallback,
    )
    assert run is not None
    a = torch.empty(shape, device="meta", dtype=torch.float16)
    b = torch.empty(shape, device="meta", dtype=torch.float16)

    first = run((a, b), {})
    second = run((a, b), {})
    replacement_a = torch.empty(shape, device="meta", dtype=torch.float16)
    third = run((replacement_a, b), {})
    shape_mismatch = run((a[:, :-1, :], b), {})
    stride_mismatch = run((a, b.transpose(1, 2)), {})
    dtype_mismatch = run((a, b.to(torch.bfloat16)), {})

    assert first is second is third
    assert shape_mismatch == "fallback"
    assert stride_mismatch == "fallback"
    assert dtype_mismatch == "fallback"
    assert len(prepare_calls) == 2
    assert prepare_calls[0][0:2] == (a, b)
    assert prepare_calls[1][0:2] == (replacement_a, b)
    assert len(launch_calls) == 3
    assert len(fallback_calls) == 3
    assert context_devices == [a.device, replacement_a.device]
    assert capability_devices == [a.device, replacement_a.device]


def test_prepared_matmul_rebuilds_sm90_runner_after_storage_rebind(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        prepared_matmul, "_require_runtime_backend", lambda inputs, op: None
    )
    monkeypatch.setattr(
        prepared_matmul,
        "get_device_capability_for",
        lambda device: (9, 0),
    )

    class DeviceFn:
        @staticmethod
        def device(device: torch.device) -> Any:
            return nullcontext()

    monkeypatch.setattr(prepared_matmul, "torch_device_fn", DeviceFn())
    from flag_dnn.ops import matmul_sm90

    monkeypatch.setattr(
        matmul_sm90,
        "select_sm90_matmul_config",
        lambda capability, key: object(),
    )
    prepare_ptrs: list[tuple[int, int, int]] = []

    def descriptor_view(tensor: torch.Tensor) -> torch.Tensor:
        view = torch.empty(0, device=tensor.device, dtype=tensor.dtype)
        return view.set_(
            tensor.untyped_storage(),
            tensor.storage_offset(),
            tensor.size(),
            tensor.stride(),
        )

    def prepare_runner(
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        **kwargs: Any,
    ) -> Any:
        prepare_ptrs.append((a.data_ptr(), b.data_ptr(), c.data_ptr()))
        a_desc = descriptor_view(a)
        b_desc = descriptor_view(b)
        c_desc = descriptor_view(c)

        def launch() -> torch.Tensor:
            c_desc.copy_(a_desc + b_desc)
            return c

        return launch

    monkeypatch.setattr(
        matmul_sm90,
        "prepare_sm90_matmul_if_supported",
        prepare_runner,
    )
    shape = (1, 2, 2)
    stride = (4, 2, 1)
    specs = (
        TensorSpec(
            "a",
            shape,
            "float32",
            stride=stride,
            device=runtime.device.name,
            contiguous=True,
        ),
        TensorSpec(
            "b",
            shape,
            "float32",
            stride=stride,
            device=runtime.device.name,
            contiguous=True,
        ),
    )

    def fallback(inputs: Sequence[Any], attrs: dict[str, Any]) -> Any:
        raise AssertionError("valid storage rebind must not use fallback")

    run = prepared_matmul._prepare_matmul(
        {"compute_data_type": "tf32", "out_dtype": torch.float32},
        specs,
        fallback,
    )
    assert run is not None
    a = torch.arange(4, dtype=torch.float32).reshape(shape)
    b = torch.ones(shape, dtype=torch.float32)
    a_identity = id(a)
    b_identity = id(b)

    result = run((a, b), {})
    torch.testing.assert_close(result, a + b)
    assert len(prepare_ptrs) == 1

    old_a_ptr = a.data_ptr()
    replacement_a = torch.full_like(a, 2)
    a.set_(replacement_a)
    assert id(a) == a_identity
    assert a.data_ptr() != old_a_ptr
    result = run((a, b), {})
    torch.testing.assert_close(result, a + b)
    assert len(prepare_ptrs) == 2

    old_b_ptr = b.data_ptr()
    replacement_b = torch.full_like(b, 3)
    b.set_(replacement_b)
    assert id(b) == b_identity
    assert b.data_ptr() != old_b_ptr
    result = run((a, b), {})
    torch.testing.assert_close(result, a + b)
    assert len(prepare_ptrs) == 3

    old_output_ptr = result.data_ptr()
    replacement_output = torch.full_like(result, -1)
    result.set_(replacement_output)
    assert result.data_ptr() != old_output_ptr
    rebound_result = run((a, b), {})
    assert rebound_result is result
    torch.testing.assert_close(rebound_result, a + b)
    assert len(prepare_ptrs) == 4
