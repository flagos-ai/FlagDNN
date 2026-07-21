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

from typing import Any

import torch

import flag_dnn
from benchmark import consts
from benchmark.base import DnnCompareBenchmark


class SingleInputUtilityBenchmark(DnnCompareBenchmark):
    def make_inputs(self, case, dtype):
        self.case = case
        shape = case if self.op_name == "identity" else case[0]
        if self.op_name == "gen_index":
            x = torch.empty(shape, device=flag_dnn.device, dtype=dtype)
        else:
            x = consts.pointwise_randn(shape, dtype, flag_dnn.device)
        return (x,)

    def _operation_kwargs(self, x: torch.Tensor) -> dict[str, Any]:
        if self.op_name == "identity":
            return {}
        _, parameter = self.case
        if self.op_name == "reshape":
            return {"shape": parameter}
        if self.op_name == "transpose":
            return {"permutation": parameter}
        if self.op_name == "slice":
            return {"slices": parameter}
        if self.op_name == "gen_index":
            return {"axis": parameter, "compute_data_type": x.dtype}
        raise ValueError(f"unsupported utility benchmark: {self.op_name}")

    def build_baseline_runner(self, inputs):
        (x,) = inputs
        return self.baseline.prepare(
            self.op_name, x, **self._operation_kwargs(x)
        )

    def build_flag_dnn_runner(self, inputs):
        (x,) = inputs
        op = getattr(flag_dnn, self.op_name)
        call_kwargs = self._operation_kwargs(x)
        call_kwargs["name"] = self.op_name
        if self.op_name in ("identity", "transpose", "slice"):
            call_kwargs["compute_data_type"] = "float32"

        @flag_dnn.graph
        def utility_graph(x):
            return op(x, **call_kwargs)

        compiled = flag_dnn.compile(
            utility_graph,
            inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
            options=consts.compile_options(),
        )
        assert [node.op_type for node in compiled.graph.nodes] == [
            self.op_name
        ]
        return compiled.bind(x)


class ConcatenateBenchmarkBase(DnnCompareBenchmark):
    op_name = "concatenate"

    def make_inputs(self, case, dtype):
        self.case = case
        shapes, _ = case
        return tuple(
            consts.pointwise_randn(shape, dtype, flag_dnn.device)
            for shape in shapes
        )

    def build_baseline_runner(self, inputs):
        _, axis = self.case
        return self.baseline.prepare(self.op_name, inputs, axis=axis)

    def build_flag_dnn_runner(self, inputs):
        _, axis = self.case

        @flag_dnn.graph
        def concatenate_graph(*values):
            return flag_dnn.concatenate(
                list(values), axis=axis, name=self.op_name
            )

        compiled = flag_dnn.compile(
            concatenate_graph,
            inputs=[
                flag_dnn.TensorSpec.from_tensor(item, f"x{index}")
                for index, item in enumerate(inputs)
            ],
            options=consts.compile_options(),
        )
        assert [node.op_type for node in compiled.graph.nodes] == [
            self.op_name
        ]
        return compiled.bind(*inputs)
