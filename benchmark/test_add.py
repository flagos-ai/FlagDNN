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

import math

import pytest
from benchmark.base import DnnCompareBenchmark
import torch

import flag_dnn
from benchmark import consts


class AddBenchmark(DnnCompareBenchmark):
    op_name = "add"
    shapes = consts.ADD_SHAPES
    shape_ids_env = "FLAGDNN_ADD_PERF_SHAPE_IDS"
    legacy_shape_ids_env = "FLAGDNN_CUDNN_ADD_PERF_SHAPE_IDS"

    def make_inputs(self, case, dtype):
        self.case = case
        x_shape, y_shape = case
        x = consts.pointwise_randn(x_shape, dtype, flag_dnn.device)
        y = consts.pointwise_randn(y_shape, dtype, flag_dnn.device)
        return x, y

    def build_baseline_runner(self, inputs):
        x, y = inputs
        return self.baseline.prepare("add", x, y, alpha=1)

    def build_flag_dnn_runner(self, inputs):
        x, y = inputs

        @flag_dnn.graph
        def flag_dnn_add_graph(x, y):
            return flag_dnn.add(
                x,
                y,
                compute_data_type="float32",
                name="add",
            )

        compiled = flag_dnn.compile(
            flag_dnn_add_graph,
            inputs=[
                flag_dnn.TensorSpec.from_tensor(x, "x"),
                flag_dnn.TensorSpec.from_tensor(y, "y"),
            ],
            options=consts.compile_options(),
        )
        assert [node.op_type for node in compiled.graph.nodes] == ["add"]
        return compiled.bind(x, y)

    def transfer_bytes(self, inputs):
        x, y = inputs
        out_shape = torch.broadcast_shapes(tuple(x.shape), tuple(y.shape))
        return (
            x.numel() * x.element_size()
            + y.numel() * y.element_size()
            + math.prod(out_shape) * x.element_size()
        )


@pytest.mark.add
@pytest.mark.graph
@pytest.mark.perf
@pytest.mark.parametrize("dtype", AddBenchmark.dtypes)
def test_add(dnn_baseline, dtype):
    torch.manual_seed(0)
    AddBenchmark(dnn_baseline).run(dtype)
