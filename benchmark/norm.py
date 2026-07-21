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

import torch

import flag_dnn
from benchmark import consts
from benchmark.base import DnnCompareBenchmark


_NORM_PHASE = "TRAINING"
_EPSILON = 1e-3
_MOMENTUM = 0.1


def _device_tensor(
    shape: Sequence[int],
    dtype: torch.dtype,
    *,
    positive: bool = False,
) -> torch.Tensor:
    if positive:
        tensor = torch.rand(shape, device=flag_dnn.device, dtype=dtype) + 0.5
    else:
        tensor = torch.randn(shape, device=flag_dnn.device, dtype=dtype)
    return tensor.contiguous()


def _norm_inputs(
    shape: Sequence[int], dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    parameter_shape = (1,) * (len(shape) - 1) + (int(shape[-1]),)
    return (
        _device_tensor(shape, dtype),
        _device_tensor(parameter_shape, dtype),
        _device_tensor(parameter_shape, dtype),
    )


def _compile_bound(
    graph: Any,
    inputs: Sequence[torch.Tensor],
    names: Sequence[str],
    op_name: str,
):
    compiled = flag_dnn.compile(
        graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(tensor, name)
            for tensor, name in zip(inputs, names)
        ],
        options=consts.compile_options(),
    )
    assert [node.op_type for node in compiled.graph.nodes] == [op_name]
    return compiled.bind(*inputs)


def _input_bytes(inputs: Sequence[torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in inputs)


class LayerNormBenchmarkBase(DnnCompareBenchmark):
    op_name = "layernorm"

    def make_inputs(self, shape, dtype):
        return _norm_inputs(shape, dtype)

    def build_baseline_runner(self, inputs):
        x, scale, bias = inputs
        return self.baseline.prepare(
            self.op_name, _NORM_PHASE, x, scale, bias, _EPSILON
        )

    def build_flag_dnn_runner(self, inputs):
        x, scale, bias = inputs

        @flag_dnn.graph
        def layernorm_graph(x, scale, bias):
            return flag_dnn.layernorm(
                _NORM_PHASE,
                x,
                scale,
                bias,
                _EPSILON,
                compute_data_type="float32",
                name=self.op_name,
            )

        return _compile_bound(
            layernorm_graph,
            inputs,
            ("x", "scale", "bias"),
            self.op_name,
        )

    def transfer_bytes(self, inputs):
        x, scale, _ = inputs
        stat_elements = x.numel() // scale.numel()
        return (
            _input_bytes(inputs)
            + x.numel() * x.element_size()
            + 2 * stat_elements * torch.float32.itemsize
        )


class RmsNormBenchmarkBase(DnnCompareBenchmark):
    op_name = "rmsnorm"

    def make_inputs(self, shape, dtype):
        return _norm_inputs(shape, dtype)

    def build_baseline_runner(self, inputs):
        x, scale, bias = inputs
        return self.baseline.prepare(
            self.op_name,
            _NORM_PHASE,
            x,
            scale,
            bias=bias,
            epsilon=_EPSILON,
        )

    def build_flag_dnn_runner(self, inputs):
        x, scale, bias = inputs

        @flag_dnn.graph
        def rmsnorm_graph(x, scale, bias):
            return flag_dnn.rmsnorm(
                _NORM_PHASE,
                x,
                scale,
                bias=bias,
                epsilon=_EPSILON,
                compute_data_type="float32",
                name=self.op_name,
            )

        return _compile_bound(
            rmsnorm_graph,
            inputs,
            ("x", "scale", "bias"),
            self.op_name,
        )

    def transfer_bytes(self, inputs):
        x, scale, _ = inputs
        stat_elements = x.numel() // scale.numel()
        return (
            _input_bytes(inputs)
            + x.numel() * x.element_size()
            + stat_elements * torch.float32.itemsize
        )


def _batchnorm_inputs(
    shape: Sequence[int], dtype: torch.dtype
) -> tuple[torch.Tensor, ...]:
    channels = int(shape[1])
    parameter_shape = (1, channels, 1, 1)
    return (
        _device_tensor(shape, dtype),
        _device_tensor(parameter_shape, dtype),
        _device_tensor(parameter_shape, dtype),
        _device_tensor(parameter_shape, torch.float32),
        _device_tensor(parameter_shape, torch.float32, positive=True),
    )


class BatchNormBenchmarkBase(DnnCompareBenchmark):
    op_name = "batchnorm"

    def make_inputs(self, shape, dtype):
        return _batchnorm_inputs(shape, dtype)

    def build_baseline_runner(self, inputs):
        return self.baseline.prepare(
            self.op_name, *inputs, _EPSILON, _MOMENTUM
        )

    def build_flag_dnn_runner(self, inputs):
        @flag_dnn.graph
        def batchnorm_graph(x, scale, bias, running_mean, running_var):
            return flag_dnn.batchnorm(
                x,
                scale,
                bias,
                running_mean,
                running_var,
                _EPSILON,
                _MOMENTUM,
                compute_data_type="float32",
                name=self.op_name,
            )

        return _compile_bound(
            batchnorm_graph,
            inputs,
            ("x", "scale", "bias", "running_mean", "running_var"),
            self.op_name,
        )

    def transfer_bytes(self, inputs):
        x, _, _, running_mean, _ = inputs
        return (
            _input_bytes(inputs)
            + x.numel() * x.element_size()
            + 4 * running_mean.numel() * torch.float32.itemsize
        )


def _batchnorm_inference_inputs(
    shape: Sequence[int], dtype: torch.dtype
) -> tuple[torch.Tensor, ...]:
    channels = int(shape[1])
    parameter_shape = (1, channels, 1, 1)
    return (
        _device_tensor(shape, dtype),
        _device_tensor(parameter_shape, torch.float32),
        _device_tensor(parameter_shape, torch.float32, positive=True),
        _device_tensor(parameter_shape, torch.float32),
        _device_tensor(parameter_shape, torch.float32),
    )


class BatchNormInferenceBenchmarkBase(DnnCompareBenchmark):
    op_name = "batchnorm_inference"

    def make_inputs(self, shape, dtype):
        return _batchnorm_inference_inputs(shape, dtype)

    def build_baseline_runner(self, inputs):
        return self.baseline.prepare(self.op_name, *inputs)

    def build_flag_dnn_runner(self, inputs):
        @flag_dnn.graph
        def batchnorm_inference_graph(x, mean, inv_variance, scale, bias):
            return flag_dnn.batchnorm_inference(
                x,
                mean,
                inv_variance,
                scale,
                bias,
                compute_data_type="float32",
                name=self.op_name,
            )

        return _compile_bound(
            batchnorm_inference_graph,
            inputs,
            ("x", "mean", "inv_variance", "scale", "bias"),
            self.op_name,
        )

    def transfer_bytes(self, inputs):
        x = inputs[0]
        return _input_bytes(inputs) + x.numel() * x.element_size()
