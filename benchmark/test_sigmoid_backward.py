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
from benchmark.base import (
    CudnnCompareBenchmark,
    cudnn_data_type,
    get_cudnn,
    skip_unsupported_cudnn_graph,
)
import torch

import flag_dnn
from benchmark import consts


class SigmoidBackwardBenchmark(CudnnCompareBenchmark):
    op_name = "sigmoid_backward"
    shapes = consts.SIGMOID_BACKWARD_SHAPES
    shape_ids_env = "FLAGDNN_CUDNN_SIGMOID_BACKWARD_PERF_SHAPE_IDS"

    def make_inputs(self, case, dtype):
        loss_shape, input_shape = case
        loss = consts.pointwise_randn(loss_shape, dtype, flag_dnn.device)
        x = consts.pointwise_randn(input_shape, dtype, flag_dnn.device)
        return loss, x

    def build_cudnn_runner(self, inputs):
        cudnn = get_cudnn()
        loss, x = inputs
        io_dtype = cudnn_data_type(x.dtype)
        graph = cudnn.pygraph(
            io_data_type=io_dtype,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=self.cudnn_handle,
        )

        loss_tensor = graph.tensor_like(loss)
        x_tensor = graph.tensor_like(x)
        dx_tensor = graph.sigmoid_backward(
            loss=loss_tensor,
            input=x_tensor,
            compute_data_type=cudnn.data_type.FLOAT,
            name="sigmoid_backward",
        )
        dx_tensor.set_output(True).set_data_type(io_dtype)

        try:
            graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as exc:
            skip_unsupported_cudnn_graph(exc, self.op_name)

        dx = torch.empty_like(x)
        workspace = torch.empty(
            graph.get_workspace_size(), device=x.device, dtype=torch.uint8
        )

        def run():
            graph.execute(
                {loss_tensor: loss, x_tensor: x, dx_tensor: dx},
                workspace,
                handle=self.cudnn_handle,
            )
            return dx

        return run

    def build_flag_dnn_runner(self, inputs):
        loss, x = inputs

        @flag_dnn.graph
        def flag_dnn_sigmoid_backward_graph(loss, x):
            return flag_dnn.sigmoid_backward(
                loss,
                x,
                compute_data_type="float32",
                name="sigmoid_backward",
            )

        compiled = flag_dnn.compile(
            flag_dnn_sigmoid_backward_graph,
            inputs=[
                flag_dnn.TensorSpec.from_tensor(loss, "loss"),
                flag_dnn.TensorSpec.from_tensor(x, "x"),
            ],
            options=consts.compile_options(),
        )
        assert [node.op_type for node in compiled.graph.nodes] == [
            "sigmoid_backward"
        ]

        def run():
            return compiled.run(loss, x)

        return run

    def transfer_bytes(self, inputs):
        loss, x = inputs
        return (
            loss.numel() * loss.element_size()
            + x.numel() * x.element_size()
            + math.prod(tuple(x.shape)) * x.element_size()
        )


@pytest.mark.sigmoid_backward
@pytest.mark.graph
@pytest.mark.perf
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", SigmoidBackwardBenchmark.dtypes)
def test_sigmoid_backward(cudnn_handle, dtype):
    torch.manual_seed(0)
    SigmoidBackwardBenchmark(cudnn_handle).run(dtype)
