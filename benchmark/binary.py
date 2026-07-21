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

import math
from typing import Literal

import torch

import flag_dnn
from benchmark import consts
from benchmark.base import DnnCompareBenchmark


BinaryInputDomain = Literal["real", "divisor", "pow", "mod", "logical"]
_BOOL_OUTPUT_OPERATIONS = {
    "cmp_eq",
    "cmp_neq",
    "cmp_gt",
    "cmp_ge",
    "cmp_lt",
    "cmp_le",
    "logical_and",
    "logical_or",
}


def make_binary_inputs(case, dtype, domain: BinaryInputDomain):
    x_shape, y_shape = case
    if domain == "logical":
        return (
            consts.pointwise_bool(x_shape, flag_dnn.device),
            consts.pointwise_bool(y_shape, flag_dnn.device),
        )
    if domain == "pow":
        return (
            consts.pointwise_layout(
                consts.pointwise_rand(x_shape, dtype, flag_dnn.device) + 0.5
            ),
            consts.pointwise_layout(
                consts.pointwise_rand(y_shape, dtype, flag_dnn.device) * 2.0
            ),
        )
    if domain in ("divisor", "mod"):
        return (
            consts.pointwise_randn(x_shape, dtype, flag_dnn.device),
            consts.pointwise_positive(
                y_shape, dtype, flag_dnn.device, offset=0.5
            ),
        )
    if domain == "real":
        return (
            consts.pointwise_randn(x_shape, dtype, flag_dnn.device),
            consts.pointwise_randn(y_shape, dtype, flag_dnn.device),
        )
    raise ValueError(f"unsupported binary input domain: {domain}")


class BinaryBenchmark(DnnCompareBenchmark):
    input_domain: BinaryInputDomain = "real"
    alpha: float = 1

    def make_inputs(self, case, dtype):
        return make_binary_inputs(case, dtype, self.input_domain)

    def build_baseline_runner(self, inputs):
        x, y = inputs
        kwargs = {"alpha": self.alpha} if self.op_name == "sub" else {}
        return self.baseline.prepare(self.op_name, x, y, **kwargs)

    def build_flag_dnn_runner(self, inputs):
        x, y = inputs
        op = getattr(flag_dnn, self.op_name)

        @flag_dnn.graph
        def binary_graph(x, y):
            kwargs = {
                "compute_data_type": "float32",
                "name": self.op_name,
            }
            if self.op_name == "sub":
                kwargs["alpha"] = self.alpha
            return op(x, y, **kwargs)

        compiled = flag_dnn.compile(
            binary_graph,
            inputs=[
                flag_dnn.TensorSpec.from_tensor(x, "x"),
                flag_dnn.TensorSpec.from_tensor(y, "y"),
            ],
            options=consts.compile_options(),
        )
        assert [node.op_type for node in compiled.graph.nodes] == [
            self.op_name
        ]
        return compiled.bind(x, y)

    def transfer_bytes(self, inputs):
        x, y = inputs
        output_shape = torch.broadcast_shapes(x.shape, y.shape)
        output_element_size = (
            torch.empty((), dtype=torch.bool).element_size()
            if self.op_name in _BOOL_OUTPUT_OPERATIONS
            else x.element_size()
        )
        return (
            x.numel() * x.element_size()
            + y.numel() * y.element_size()
            + math.prod(output_shape) * output_element_size
        )
