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
from benchmark.base import (
    CudnnCompareBenchmark,
    cudnn_data_type,
    get_cudnn,
    skip_unsupported_cudnn_graph,
)
import torch

import flag_dnn
from benchmark import consts


class ScaleBenchmark(CudnnCompareBenchmark):
    op_name = "scale"
    shapes = consts.SCALE_SHAPES
    shape_ids_env = "FLAGDNN_CUDNN_SCALE_PERF_SHAPE_IDS"

    def make_inputs(self, shape_pair, dtype):
        x_shape, scale_shape = shape_pair
        x = consts.pointwise_randn(x_shape, dtype, flag_dnn.device)
        scale = consts.pointwise_randn(scale_shape, dtype, flag_dnn.device)
        return x, scale

    def build_cudnn_runner(self, inputs):
        cudnn = get_cudnn()
        x, scale = inputs
        io_dtype = cudnn_data_type(x.dtype)
        graph = cudnn.pygraph(
            io_data_type=io_dtype,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=self.cudnn_handle,
        )
        x_tensor = graph.tensor_like(x)
        scale_tensor = graph.tensor_like(scale)
        y_tensor = graph.scale(
            input=x_tensor,
            scale=scale_tensor,
            compute_data_type=cudnn.data_type.FLOAT,
            name=self.op_name,
        )
        y_tensor.set_output(True).set_data_type(io_dtype)

        try:
            graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as exc:
            skip_unsupported_cudnn_graph(exc, self.op_name)

        y = torch.empty_like(x)
        workspace = torch.empty(
            graph.get_workspace_size(), device=x.device, dtype=torch.uint8
        )

        def run():
            graph.execute(
                {**{x_tensor: x, scale_tensor: scale}, y_tensor: y},
                workspace,
                handle=self.cudnn_handle,
            )
            return y

        return run

    def build_flag_dnn_runner(self, inputs):
        x, scale = inputs

        @flag_dnn.graph
        def flag_dnn_scale_graph(x, scale):
            return flag_dnn.scale(
                x,
                scale,
                compute_data_type="float32",
                name=self.op_name,
            )

        compiled = flag_dnn.compile(
            flag_dnn_scale_graph,
            inputs=[
                flag_dnn.TensorSpec.from_tensor(x, "x"),
                flag_dnn.TensorSpec.from_tensor(scale, "scale"),
            ],
            options=consts.compile_options(),
        )
        assert [node.op_type for node in compiled.graph.nodes] == ["scale"]

        def run():
            return compiled.run(x, scale)

        return run


@pytest.mark.scale
@pytest.mark.graph
@pytest.mark.perf
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", ScaleBenchmark.dtypes)
def test_scale(cudnn_handle, dtype):
    torch.manual_seed(0)
    ScaleBenchmark(cudnn_handle).run(dtype)
