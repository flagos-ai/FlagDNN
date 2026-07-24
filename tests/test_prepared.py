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

from contextlib import nullcontext
from typing import Any, Sequence

import pytest
import torch

import flag_dnn.graph.prepared.conv as prepared_conv
from flag_dnn import runtime
from flag_dnn.graph import prepared
from flag_dnn.graph import executor as graph_executor
from flag_dnn.graph.capture import CompiledGraph
from flag_dnn.graph.prepared import core as prepared_core
from flag_dnn.graph.prepared import ops as prepared_ops
from flag_dnn.graph.prepared import pointwise as prepared_pointwise
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


def test_compiled_bind_validates_once_then_uses_prepared_fast_path(
    monkeypatch: Any,
) -> None:
    calls: list[tuple[Any, ...]] = []

    class FastPath:
        def bind(self, inputs: tuple[Any, ...]) -> Any:
            calls.append(("bind", *inputs))

            def replay() -> str:
                calls.append(("replay", *inputs))
                return "fast"

            return replay

    input_check = object()
    prepared_plan = type(
        "PreparedPlan",
        (),
        {
            "input_count": 1,
            "input_checks": (input_check,),
            "validate_inputs": True,
            "fast_path": FastPath(),
        },
    )()
    monkeypatch.setattr(
        graph_executor, "_get_prepared_plan", lambda plan: prepared_plan
    )
    monkeypatch.setattr(
        graph_executor,
        "_validate_prepared_input",
        lambda check, actual: calls.append(("validate", check, actual)),
    )
    plan = type("Plan", (), {"graph": object()})()

    replay = CompiledGraph(plan).bind("input")

    assert calls == [
        ("validate", input_check, "input"),
        ("bind", "input"),
    ]
    assert replay() == "fast"
    assert replay() == "fast"
    assert calls == [
        ("validate", input_check, "input"),
        ("bind", "input"),
        ("replay", "input"),
        ("replay", "input"),
    ]


@pytest.mark.parametrize("op_type", ("logical_and", "logical_or"))
def test_logical_bool_pair_has_prepared_cached_launcher(op_type: str) -> None:
    shape = (4, 16, 64, 128)
    stride = (131072, 1, 2048, 16)
    specs = (
        TensorSpec(
            "left",
            shape,
            "bool",
            stride=stride,
            layout="nhwc",
            device=runtime.device.name,
        ),
        TensorSpec(
            "right",
            shape,
            "bool",
            stride=stride,
            layout="nhwc",
            device=runtime.device.name,
        ),
    )

    run = prepared_pointwise._prepare_pointwise(op_type, {}, specs, _fallback)

    assert run is not None
    assert getattr(run, "bind", None) is not None


def test_batchnorm_training_has_prepared_cached_launcher() -> None:
    input_shape = (32, 1024, 1, 1)
    input_stride = (1024, 1, 1, 1)
    param_shape = (1, 1024, 1, 1)
    param_stride = (1024, 1, 1, 1)
    device = runtime.device.name
    specs = (
        TensorSpec(
            "x",
            input_shape,
            "float16",
            stride=input_stride,
            layout="contiguous",
            device=device,
            contiguous=True,
        ),
        TensorSpec(
            "scale",
            param_shape,
            "float16",
            stride=param_stride,
            layout="contiguous",
            device=device,
            contiguous=True,
        ),
        TensorSpec(
            "bias",
            param_shape,
            "float16",
            stride=param_stride,
            layout="contiguous",
            device=device,
            contiguous=True,
        ),
        TensorSpec(
            "running_mean",
            param_shape,
            "float32",
            stride=param_stride,
            layout="contiguous",
            device=device,
            contiguous=True,
        ),
        TensorSpec(
            "running_var",
            param_shape,
            "float32",
            stride=param_stride,
            layout="contiguous",
            device=device,
            contiguous=True,
        ),
        TensorSpec("epsilon", (), "float32"),
        TensorSpec("momentum", (), "float32"),
    )

    run = prepared.prepare_run_fn(
        "batchnorm",
        {"peer_stats_count": 0, "_validate_inputs": False},
        specs,
        _fallback,
    )

    assert run is not _fallback
    assert getattr(run, "bind", None) is not None


def test_layernorm_training_has_prepared_cached_launcher() -> None:
    device = runtime.device.name
    specs = (
        TensorSpec(
            "x",
            (1, 128, 768),
            "float16",
            stride=(98304, 768, 1),
            layout="contiguous",
            device=device,
            contiguous=True,
        ),
        TensorSpec(
            "scale",
            (1, 1, 768),
            "float16",
            stride=(768, 768, 1),
            layout="contiguous",
            device=device,
            contiguous=True,
        ),
        TensorSpec(
            "bias",
            (1, 1, 768),
            "float16",
            stride=(768, 768, 1),
            layout="contiguous",
            device=device,
            contiguous=True,
        ),
        TensorSpec("epsilon", (), "float32"),
    )

    run = prepared.prepare_run_fn(
        "layernorm",
        {
            "norm_forward_phase": "TRAINING",
            "_validate_inputs": False,
        },
        specs,
        _fallback,
    )

    assert run is not _fallback
    assert getattr(run, "bind", None) is not None


def test_layer_norm_hidden_width_is_constexpr() -> None:
    from flag_dnn.ops.layer_norm import layer_norm_kernel

    jit_kernel = layer_norm_kernel.fn.fn
    constexpr_names = {
        jit_kernel.arg_names[index] for index in jit_kernel.constexprs
    }
    assert "N" in constexpr_names


def test_rmsnorm_training_has_prepared_cached_launcher() -> None:
    device = runtime.device.name
    specs = (
        TensorSpec(
            "x",
            (3, 257, 513),
            "float16",
            stride=(131841, 513, 1),
            layout="contiguous",
            device=device,
            contiguous=True,
        ),
        TensorSpec(
            "scale",
            (1, 1, 513),
            "float16",
            stride=(513, 513, 1),
            layout="contiguous",
            device=device,
            contiguous=True,
        ),
        TensorSpec(
            "bias",
            (1, 1, 513),
            "float16",
            stride=(513, 513, 1),
            layout="contiguous",
            device=device,
            contiguous=True,
        ),
        TensorSpec("epsilon", (), "float32"),
    )

    run = prepared.prepare_run_fn(
        "rmsnorm",
        {
            "norm_forward_phase": "TRAINING",
            "has_bias": True,
            "_validate_inputs": False,
        },
        specs,
        _fallback,
    )

    assert run is not _fallback
    assert getattr(run, "bind", None) is not None


def test_single_axis_reduction_has_prepared_cached_launcher() -> None:
    spec = TensorSpec(
        "x",
        (8, 8, 32, 32),
        "bfloat16",
        stride=(8192, 1, 256, 8),
        layout="nhwc",
        device=runtime.device.name,
        contiguous=False,
    )

    run = prepared.prepare_run_fn(
        "reduction",
        {
            "mode": "ADD",
            "dim": 1,
            "keepdim": True,
            "dtype": None,
            "_validate_inputs": False,
        },
        (spec,),
        _fallback,
    )

    assert run is not _fallback
    assert getattr(run, "bind", None) is not None


def test_gen_index_has_prepared_cached_launcher() -> None:
    spec = TensorSpec(
        "x",
        (32, 128, 256),
        "float32",
        stride=(32768, 256, 1),
        layout="contiguous",
        device=runtime.device.name,
        contiguous=True,
    )

    run = prepared.prepare_run_fn(
        "gen_index",
        {
            "axis": 2,
            "compute_data_type": torch.float32,
            "_validate_inputs": False,
        },
        (spec,),
        _fallback,
    )

    assert run is not _fallback
    assert getattr(run, "bind", None) is not None


def test_gen_index_tuner_keys_index_geometry() -> None:
    from flag_dnn.ops.gen_index import _gen_index_kernel

    assert _gen_index_kernel.fn.keys == [
        "n_elements",
        "axis_size",
        "inner_size",
    ]


@pytest.mark.parametrize("activation", ("identity", "silu"))
def test_causal_conv1d_has_prepared_cached_launcher(activation: str) -> None:
    device = runtime.device.name
    specs = (
        TensorSpec(
            "x",
            (3, 192, 257),
            "float16",
            stride=(49344, 257, 1),
            layout="contiguous",
            device=device,
            contiguous=True,
        ),
        TensorSpec(
            "weight",
            (192, 4),
            "float16",
            stride=(4, 1),
            layout="contiguous",
            device=device,
            contiguous=True,
        ),
        TensorSpec(
            "bias",
            (192,),
            "float16",
            stride=(1,),
            layout="contiguous",
            device=device,
            contiguous=True,
        ),
    )

    run = prepared.prepare_run_fn(
        "causal_conv1d",
        {
            "activation": activation,
            "has_bias": True,
            "_validate_inputs": False,
        },
        specs,
        _fallback,
    )

    assert run is not _fallback
    assert getattr(run, "bind", None) is not None


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


@pytest.mark.parametrize(
    ("op_type", "prepare"),
    (
        ("conv_fprop", prepared_conv._prepare_conv_fprop),
        ("conv_dgrad", prepared_conv._prepare_conv_dgrad),
        ("conv_wgrad", prepared_conv._prepare_conv_wgrad),
    ),
)
def test_prepared_conv_prefers_backend_hook(
    monkeypatch: Any, op_type: str, prepare: Any
) -> None:
    calls: list[tuple[Any, ...]] = []
    backend_run = object()
    specs = (
        TensorSpec(
            "input",
            (8, 64, 28, 28),
            "float16",
            stride=(50176, 784, 28, 1),
            layout="contiguous",
            device=runtime.device.name,
            contiguous=True,
        ),
        TensorSpec(
            "weight",
            (128, 64, 1, 1),
            "float16",
            stride=(64, 1, 1, 1),
            layout="contiguous",
            device=runtime.device.name,
            contiguous=True,
        ),
    )
    attrs = {
        "input_size": (8, 64, 28, 28),
        "filter_size": (128, 64, 1, 1),
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "convolution_mode": "CROSS_CORRELATION",
        "groups": 1,
    }

    def backend_prepare(
        requested_op: str,
        requested_attrs: dict[str, Any],
        requested_specs: Sequence[TensorSpec],
        default: Any,
    ) -> Any:
        calls.append(
            (requested_op, requested_attrs, tuple(requested_specs), default)
        )
        return backend_run

    monkeypatch.setattr(
        prepared_conv.runtime,
        "get_backend_hook",
        lambda name: backend_prepare if name == "prepare_conv" else None,
    )

    run = prepare(attrs, specs, _fallback)

    assert run is backend_run
    assert calls == [(op_type, attrs, specs, _fallback)]


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
    from flag_dnn.runtime.backend._nvidia.ops import matmul as nvidia_matmul

    monkeypatch.setattr(
        nvidia_matmul,
        "get_device_capability_for",
        lambda device: (9, 0),
    )
    prepare_calls: list[tuple[Any, Any]] = []
    launch_calls: list[tuple[Any, Any, Any]] = []

    def prepare_runner(a: Any, b: Any, **kwargs: Any) -> Any:
        prepare_calls.append((a, b))

        def launch(output: Any) -> Any:
            launch_calls.append((a, b, output))
            return output

        return launch

    monkeypatch.setattr(
        nvidia_matmul,
        "prepare_sm90_matmul_dynamic_output",
        prepare_runner,
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

    run = nvidia_matmul.prepare_matmul(
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
    assert prepare_calls[0] == (a, b)
    assert prepare_calls[1] == (replacement_a, b)
    assert len(launch_calls) == 3
    assert len(fallback_calls) == 3


def test_prepared_matmul_rebuilds_sm90_runner_after_storage_rebind(
    monkeypatch: Any,
) -> None:
    from flag_dnn.runtime.backend._nvidia.ops import matmul as nvidia_matmul
    from flag_dnn.runtime.backend._nvidia.ops.matmul_sm90 import (
        Sm90MatmulConfig,
    )

    monkeypatch.setattr(
        nvidia_matmul,
        "get_device_capability_for",
        lambda device: (9, 0),
    )
    monkeypatch.setattr(
        nvidia_matmul,
        "select_sm90_matmul_config",
        lambda capability, key: Sm90MatmulConfig(
            "lowp", 1, 1, 1, 2, 4, 1, 168, False
        ),
    )
    prepare_ptrs: list[tuple[int, int]] = []

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
        **kwargs: Any,
    ) -> Any:
        prepare_ptrs.append((a.data_ptr(), b.data_ptr()))
        a_desc = descriptor_view(a)
        b_desc = descriptor_view(b)

        def launch(output: torch.Tensor) -> torch.Tensor:
            output.copy_(a_desc + b_desc)
            return output

        return launch

    monkeypatch.setattr(
        nvidia_matmul,
        "prepare_sm90_matmul_dynamic_output",
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

    run = nvidia_matmul.prepare_matmul(
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
    assert len(prepare_ptrs) == 3
