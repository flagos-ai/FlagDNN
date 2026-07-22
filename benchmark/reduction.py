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

import flag_dnn
from benchmark import consts
from benchmark.base import DnnCompareBenchmark


class ReductionBenchmarkBase(DnnCompareBenchmark):
    op_name = "reduction"

    def make_inputs(self, case, dtype):
        self.case = case
        shape, _, mode = case
        if mode == "MUL":
            x = 0.9 + 0.2 * consts.pointwise_rand(
                shape, dtype, flag_dnn.device
            )
        else:
            x = consts.pointwise_randn(shape, dtype, flag_dnn.device)
        return (x,)

    def shape_detail(self, inputs):
        _, dim, mode = self.case
        (x,) = inputs
        return {
            "input": tuple(x.shape),
            "dim": dim,
            "mode": mode,
            "keepdim": True,
        }

    def build_baseline_runner(self, inputs):
        _, dim, mode = self.case
        (x,) = inputs
        return self.baseline.prepare(
            self.op_name, x, mode, dim=dim, keepdim=True
        )

    def build_flag_dnn_runner(self, inputs):
        _, dim, mode = self.case
        (x,) = inputs

        @flag_dnn.graph
        def reduction_graph(x):
            return flag_dnn.reduction(
                x,
                mode=mode,
                dim=dim,
                keepdim=True,
                compute_data_type="float32",
                name=self.op_name,
            )

        compiled = flag_dnn.compile(
            reduction_graph,
            inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
            options=consts.compile_options(),
        )
        assert [node.op_type for node in compiled.graph.nodes] == [
            self.op_name
        ]
        return compiled.bind(x)
