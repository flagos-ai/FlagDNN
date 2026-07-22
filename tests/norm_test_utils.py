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

from typing import Any, Sequence

import pytest
import torch

import flag_dnn
from flag_dnn.graph import graph as graph_decorator
from tests import accuracy_utils as utils


def _to_device(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().to(flag_dnn.device)


def _assert_outputs_close(
    actual: Sequence[torch.Tensor],
    expected: Sequence[torch.Tensor],
    dtype: torch.dtype,
    atol: float,
) -> None:
    assert len(actual) == len(expected)
    for index, (actual_value, expected_value) in enumerate(
        zip(actual, expected)
    ):
        assert actual_value.shape == expected_value.shape
        assert actual_value.dtype == expected_value.dtype
        assert actual_value.device == expected_value.device
        if index == 0:
            utils.gems_assert_close(
                actual_value, expected_value, dtype, atol=atol
            )
        else:
            torch.testing.assert_close(
                actual_value, expected_value, atol=atol, rtol=atol
            )


def _compile_layernorm(
    x: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    epsilon: float,
):
    @graph_decorator
    def layernorm_graph(x, scale, bias):
        return flag_dnn.layernorm(
            "TRAINING",
            x,
            scale,
            bias,
            epsilon,
            compute_data_type="float32",
            name="layernorm",
        )

    compiled = flag_dnn.compile(
        layernorm_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(x, "x"),
            flag_dnn.TensorSpec.from_tensor(scale, "scale"),
            flag_dnn.TensorSpec.from_tensor(bias, "bias"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["layernorm"]
    return compiled


def run_layernorm_test(dnn_reference: Any, dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    epsilon = 1e-3
    x = _to_device(torch.randn((2, 5, 17), dtype=dtype))
    scale = _to_device(torch.randn((1, 1, 17), dtype=dtype))
    bias = _to_device(torch.randn((17, 1), dtype=dtype))
    assert dnn_reference.supports("layernorm", dtype)
    expected = dnn_reference.run(
        "layernorm", "TRAINING", x, scale, bias, epsilon
    )
    actual = _compile_layernorm(x, scale, bias, epsilon).run(x, scale, bias)
    dnn_reference.synchronize()
    atol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-4
    _assert_outputs_close(actual, expected, dtype, atol)


def _compile_rmsnorm(
    x: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    epsilon: float,
):
    @graph_decorator
    def rmsnorm_graph(x, scale, bias):
        return flag_dnn.rmsnorm(
            "TRAINING",
            x,
            scale,
            bias=bias,
            epsilon=epsilon,
            compute_data_type="float32",
            name="rmsnorm",
        )

    compiled = flag_dnn.compile(
        rmsnorm_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(x, "x"),
            flag_dnn.TensorSpec.from_tensor(scale, "scale"),
            flag_dnn.TensorSpec.from_tensor(bias, "bias"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["rmsnorm"]
    return compiled


def run_rmsnorm_test(dnn_reference: Any, dtype: torch.dtype) -> None:
    torch.manual_seed(1)
    epsilon = 1e-3
    x = _to_device(torch.randn((2, 5, 17), dtype=dtype))
    scale = _to_device(torch.randn((1, 1, 17), dtype=dtype))
    bias = _to_device(torch.randn((17, 1), dtype=dtype))
    assert dnn_reference.supports("rmsnorm", dtype)
    expected = dnn_reference.run(
        "rmsnorm", "TRAINING", x, scale, bias=bias, epsilon=epsilon
    )
    actual = _compile_rmsnorm(x, scale, bias, epsilon).run(x, scale, bias)
    dnn_reference.synchronize()
    atol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-4
    _assert_outputs_close(actual, expected, dtype, atol)


def _batchnorm_parameters(
    channels: int, dtype: torch.dtype
) -> tuple[torch.Tensor, ...]:
    scale = _to_device(torch.randn((1, channels, 1, 1), dtype=dtype))
    bias = _to_device(torch.randn((1, channels, 1, 1), dtype=dtype))
    running_mean = _to_device(
        torch.randn((1, channels, 1, 1), dtype=torch.float32)
    )
    running_var = _to_device(
        torch.rand((1, channels, 1, 1), dtype=torch.float32) + 0.5
    )
    return scale, bias, running_mean, running_var


def _compile_batchnorm(
    inputs: Sequence[torch.Tensor], epsilon: float, momentum: float
):
    x, scale, bias, running_mean, running_var = inputs

    @graph_decorator
    def batchnorm_graph(x, scale, bias, running_mean, running_var):
        return flag_dnn.batchnorm(
            x,
            scale,
            bias,
            running_mean,
            running_var,
            epsilon,
            momentum,
            compute_data_type="float32",
            name="batchnorm",
        )

    compiled = flag_dnn.compile(
        batchnorm_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(x, "x"),
            flag_dnn.TensorSpec.from_tensor(scale, "scale"),
            flag_dnn.TensorSpec.from_tensor(bias, "bias"),
            flag_dnn.TensorSpec.from_tensor(running_mean, "running_mean"),
            flag_dnn.TensorSpec.from_tensor(running_var, "running_var"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["batchnorm"]
    return compiled


def run_batchnorm_test(dnn_reference: Any, dtype: torch.dtype) -> None:
    if not dnn_reference.supports("batchnorm", dtype):
        pytest.skip(
            f"batchnorm does not support {dtype} on the active DNN provider"
        )
    torch.manual_seed(2)
    epsilon = 1e-3
    momentum = 0.1
    x = _to_device(torch.randn((2, 8, 8, 8), dtype=dtype))
    inputs = (x, *_batchnorm_parameters(8, dtype))
    expected = dnn_reference.run("batchnorm", *inputs, epsilon, momentum)
    actual = _compile_batchnorm(inputs, epsilon, momentum).run(
        *(tensor.clone() for tensor in inputs)
    )
    dnn_reference.synchronize()
    atol = 3e-2 if dtype in (torch.float16, torch.bfloat16) else 3e-4
    _assert_outputs_close(actual, expected, dtype, atol)


def _compile_batchnorm_inference(inputs: Sequence[torch.Tensor]):
    x, mean, inv_variance, scale, bias = inputs

    @graph_decorator
    def batchnorm_inference_graph(x, mean, inv_variance, scale, bias):
        return flag_dnn.batchnorm_inference(
            x,
            mean,
            inv_variance,
            scale,
            bias,
            compute_data_type="float32",
            name="batchnorm_inference",
        )

    compiled = flag_dnn.compile(
        batchnorm_inference_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(x, "x"),
            flag_dnn.TensorSpec.from_tensor(mean, "mean"),
            flag_dnn.TensorSpec.from_tensor(inv_variance, "inv_variance"),
            flag_dnn.TensorSpec.from_tensor(scale, "scale"),
            flag_dnn.TensorSpec.from_tensor(bias, "bias"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == [
        "batchnorm_inference"
    ]
    return compiled


def run_batchnorm_inference_test(
    dnn_reference: Any,
    dtype: torch.dtype,
    shape: tuple[int, int, int, int],
) -> None:
    torch.manual_seed(3)
    x = _to_device(torch.randn(shape, dtype=dtype))
    channels = shape[1]
    mean = _to_device(torch.randn((1, channels, 1, 1), dtype=torch.float32))
    inv_variance = _to_device(
        torch.rand((1, channels, 1, 1), dtype=torch.float32) + 0.5
    )
    scale = _to_device(torch.randn((1, channels, 1, 1), dtype=torch.float32))
    bias = _to_device(torch.randn((1, channels, 1, 1), dtype=torch.float32))
    inputs = (x, mean, inv_variance, scale, bias)
    assert dnn_reference.supports("batchnorm_inference", dtype)
    expected = dnn_reference.run("batchnorm_inference", *inputs)
    actual = _compile_batchnorm_inference(inputs).run(*inputs)
    dnn_reference.synchronize()
    assert actual.shape == expected.shape == x.shape
    assert actual.dtype == expected.dtype == dtype
    assert actual.device == expected.device == x.device
    atol = 5e-2 if dtype == torch.bfloat16 else 2e-2
    utils.gems_assert_close(actual, expected, dtype, atol=atol)
