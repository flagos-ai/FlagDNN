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

import pytest
import torch

import flag_dnn
from tests import accuracy_utils as utils

COMPARE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _cudnn_causal_conv1d(dnn_reference, x, weight, bias, activation):
    assert dnn_reference.supports("causal_conv1d", x.dtype)
    return dnn_reference.run(
        "causal_conv1d",
        x,
        weight,
        bias=bias,
        activation=activation,
    )


def _run_flag_dnn_causal_conv1d_graph(x, weight, bias, activation):
    @flag_dnn.graph
    def flag_dnn_causal_conv1d_graph(x, weight, bias):
        return flag_dnn.causal_conv1d(
            x, weight, bias=bias, activation=activation
        )

    compiled = flag_dnn.compile(
        flag_dnn_causal_conv1d_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(x, "x"),
            flag_dnn.TensorSpec.from_tensor(weight, "weight"),
            flag_dnn.TensorSpec.from_tensor(bias, "bias"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["causal_conv1d"]
    return compiled.run(x.clone(), weight.clone(), bias.clone())


@pytest.mark.causal_conv1d
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", COMPARE_DTYPES)
@pytest.mark.parametrize("shape_kernel", [((2, 4, 16), 3), ((3, 8, 33), 5)])
@pytest.mark.parametrize("activation", ["identity", "silu"])
def test_causal_conv1d_matches_cudnn(
    dnn_reference, dtype, shape_kernel, activation
):
    torch.manual_seed(0)
    shape, kernel = shape_kernel
    x = torch.randn(shape, device=flag_dnn.device, dtype=dtype)
    weight = torch.randn(
        (shape[1], kernel), device=flag_dnn.device, dtype=dtype
    )
    bias = torch.randn((shape[1],), device=flag_dnn.device, dtype=dtype)
    cudnn_out = _cudnn_causal_conv1d(
        dnn_reference, x, weight, bias, activation
    )
    actual = _run_flag_dnn_causal_conv1d_graph(x, weight, bias, activation)
    atol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-4
    utils.gems_assert_close(actual, cudnn_out, dtype, atol=atol)


@pytest.mark.causal_conv1d
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", COMPARE_DTYPES)
@pytest.mark.parametrize(
    "shape_kernel_activation",
    [
        ((1, 64, 128), 3, "identity"),
        ((3, 192, 257), 4, "silu"),
    ],
)
def test_causal_conv1d_benchmark_regressions(
    dnn_reference, dtype, shape_kernel_activation
):
    torch.manual_seed(0)
    shape, kernel, activation = shape_kernel_activation
    x = torch.randn(shape, device=flag_dnn.device, dtype=dtype)
    weight = torch.randn(
        (shape[1], kernel), device=flag_dnn.device, dtype=dtype
    )
    bias = torch.randn((shape[1],), device=flag_dnn.device, dtype=dtype)
    expected = _cudnn_causal_conv1d(dnn_reference, x, weight, bias, activation)
    actual = _run_flag_dnn_causal_conv1d_graph(x, weight, bias, activation)
    atol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-4
    utils.gems_assert_close(actual, expected, dtype, atol=atol)


@pytest.mark.causal_conv1d
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_causal_conv1d_h100_reference_is_standard_frontend(dnn_reference):
    if torch.cuda.get_device_capability()[0] >= 10:
        pytest.skip("SM100+ uses the native causal cuDNN route")
    x = torch.randn((1, 2, 8), device=flag_dnn.device, dtype=torch.float16)
    weight = torch.randn((2, 3), device=x.device, dtype=x.dtype)
    bias = torch.randn((2,), device=x.device, dtype=x.dtype)
    prepared = dnn_reference.prepare(
        "causal_conv1d", x, weight, bias=bias, activation="silu"
    )
    try:
        assert prepared.reference_name == "cuDNN standard composite"
        assert prepared.output.shape == x.shape
    finally:
        prepared.close()


@pytest.mark.causal_conv1d
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_causal_conv1d_sm100_native_requires_packed_input(
    dnn_reference, monkeypatch
):
    from devtools.dnn_reference.providers.nvidia_ops import (
        causal_conv1d as provider_module,
    )

    native_called = False

    def native_wrapper(*_args, **_kwargs):
        nonlocal native_called
        native_called = True
        raise AssertionError("non-contiguous input reached native wrapper")

    monkeypatch.setattr(
        provider_module.torch.cuda,
        "get_device_capability",
        lambda _device=None: (10, 0),
    )
    monkeypatch.setattr(
        provider_module.cudnn,
        "causal_conv1d_forward",
        native_wrapper,
        raising=False,
    )
    storage = torch.randn(
        (1, 2, 16), device=flag_dnn.device, dtype=torch.float16
    )
    x = storage[:, :, ::2]
    assert not x.is_contiguous()
    weight = torch.randn((2, 3), device=x.device, dtype=x.dtype)
    bias = torch.randn((2,), device=x.device, dtype=x.dtype)
    prepared = dnn_reference.prepare(
        "causal_conv1d", x, weight, bias=bias, activation="identity"
    )
    try:
        assert not native_called
        assert prepared.reference_name == "cuDNN standard composite"
    finally:
        prepared.close()


@pytest.mark.causal_conv1d
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_causal_conv1d_sm100_runtime_failure_is_not_retried(
    dnn_reference, monkeypatch
):
    from devtools.dnn_reference.providers.nvidia_ops import (
        causal_conv1d as provider_module,
    )

    def fail_native(*_args, **_kwargs):
        raise RuntimeError("native execution failed")

    monkeypatch.setattr(
        provider_module.torch.cuda,
        "get_device_capability",
        lambda _device=None: (10, 0),
    )
    monkeypatch.setattr(
        provider_module.cudnn,
        "causal_conv1d_forward",
        fail_native,
        raising=False,
    )
    x = torch.randn((1, 2, 8), device=flag_dnn.device, dtype=torch.float16)
    weight = torch.randn((2, 3), device=x.device, dtype=x.dtype)
    bias = torch.randn((2,), device=x.device, dtype=x.dtype)

    with pytest.raises(RuntimeError, match="native execution failed"):
        dnn_reference.prepare(
            "causal_conv1d",
            x,
            weight,
            bias=bias,
            activation="identity",
        )
