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
from benchmark import consts
from benchmark.base import DnnCompareBenchmark


class AbsBenchmark(DnnCompareBenchmark):
    op_name = "abs"
    shapes = consts.ABS_SHAPES
    shape_ids_env = "FLAGDNN_ABS_PERF_SHAPE_IDS"
    legacy_shape_ids_env = "FLAGDNN_CUDNN_ABS_PERF_SHAPE_IDS"

    def make_inputs(self, shape, dtype):
        x = consts.pointwise_randn(shape, dtype, flag_dnn.device)
        return (x,)

    def build_baseline_runner(self, inputs):
        (x,) = inputs
        return self.baseline.prepare("abs", x)

    def build_flag_dnn_runner(self, inputs):
        (x,) = inputs

        @flag_dnn.graph
        def flag_dnn_abs_graph(x):
            return flag_dnn.abs(
                x,
                compute_data_type="float32",
                name="abs",
            )

        compiled = flag_dnn.compile(
            flag_dnn_abs_graph,
            inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
            options=consts.compile_options(),
        )
        assert [node.op_type for node in compiled.graph.nodes] == ["abs"]
        return compiled.bind(x)


@pytest.mark.abs
@pytest.mark.graph
@pytest.mark.perf
@pytest.mark.parametrize("dtype", AbsBenchmark.dtypes)
def test_abs(dnn_baseline, dtype):
    torch.manual_seed(0)
    AbsBenchmark(dnn_baseline).run(dtype)
