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

from typing import Literal

import torch

import flag_dnn
from benchmark import consts
from benchmark.base import DnnCompareBenchmark


UnaryInputDomain = Literal["real", "positive", "scaled", "tan", "logical"]


def make_unary_input(
    shape,
    dtype: torch.dtype,
    domain: UnaryInputDomain,
) -> torch.Tensor:
    if domain == "logical":
        return consts.pointwise_bool(shape, flag_dnn.device)
    if domain == "positive":
        return consts.pointwise_positive(
            shape, dtype, flag_dnn.device, offset=0.5
        )
    if domain == "scaled":
        return consts.pointwise_layout(
            consts.pointwise_randn(shape, dtype, flag_dnn.device) * 4.0
        )
    if domain == "tan":
        return consts.pointwise_layout(
            consts.pointwise_rand(shape, dtype, flag_dnn.device) - 0.5
        )
    if domain == "real":
        return consts.pointwise_randn(shape, dtype, flag_dnn.device)
    raise ValueError(f"unsupported unary input domain: {domain}")


class UnaryBenchmark(DnnCompareBenchmark):
    input_domain: UnaryInputDomain = "real"
    operation_kwargs: dict[str, object] = {}

    def make_inputs(self, shape, dtype):
        return (make_unary_input(shape, dtype, self.input_domain),)

    def build_baseline_runner(self, inputs):
        (x,) = inputs
        return self.baseline.prepare(self.op_name, x, **self.operation_kwargs)

    def build_flag_dnn_runner(self, inputs):
        (x,) = inputs
        op = getattr(flag_dnn, self.op_name)
        call_kwargs = dict(self.operation_kwargs)
        if self.op_name != "gelu":
            call_kwargs.update(
                compute_data_type="float32",
                name=self.op_name,
            )

        @flag_dnn.graph
        def unary_graph(x):
            return op(x, **call_kwargs)

        compiled = flag_dnn.compile(
            unary_graph,
            inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
            options=consts.compile_options(),
        )
        assert [node.op_type for node in compiled.graph.nodes] == [
            self.op_name
        ]
        return compiled.bind(x)
