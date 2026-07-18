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


class ModBenchmark(CudnnCompareBenchmark):
    op_name = "mod"
    shapes = consts.MOD_SHAPES
    shape_ids_env = "FLAGDNN_CUDNN_MOD_PERF_SHAPE_IDS"
    enforce_min_speedup = True

    def make_inputs(self, case, dtype):
        x_shape, y_shape = case
        del x_shape
        y = consts.pointwise_positive(
            y_shape, dtype, flag_dnn.device, offset=0.5
        )
        x = consts.pointwise_layout(y * 2.5)
        return x, y

    def build_cudnn_runner(self, inputs):
        cudnn = get_cudnn()
        x, y = inputs
        io_dtype = cudnn_data_type(x.dtype)
        graph = cudnn.pygraph(
            io_data_type=io_dtype,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=self.cudnn_handle,
        )
        if not hasattr(graph, "mod"):
            pytest.skip("cuDNN frontend Python API does not expose mod")

        x_tensor = graph.tensor_like(x)
        y_tensor = graph.tensor_like(y)
        try:
            out_tensor = graph.mod(
                input0=x_tensor,
                input1=y_tensor,
                compute_data_type=cudnn.data_type.FLOAT,
                name="mod",
            )
        except TypeError:
            out_tensor = graph.mod(
                a=x_tensor,
                b=y_tensor,
                compute_data_type=cudnn.data_type.FLOAT,
                name="mod",
            )
        out_tensor.set_output(True).set_data_type(io_dtype)

        try:
            graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as exc:
            skip_unsupported_cudnn_graph(exc, self.op_name)

        out_shape = torch.broadcast_shapes(tuple(x.shape), tuple(y.shape))
        out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        workspace = torch.empty(
            graph.get_workspace_size(), device=x.device, dtype=torch.uint8
        )

        def run():
            graph.execute(
                {x_tensor: x, y_tensor: y, out_tensor: out},
                workspace,
                handle=self.cudnn_handle,
            )
            return out

        return run

    def build_flag_dnn_runner(self, inputs):
        x, y = inputs

        @flag_dnn.graph
        def flag_dnn_mod_graph(x, y):
            return flag_dnn.mod(
                input0=x,
                input1=y,
                compute_data_type="float32",
                name="mod",
            )

        compiled = flag_dnn.compile(
            flag_dnn_mod_graph,
            inputs=[
                flag_dnn.TensorSpec.from_tensor(x, "x"),
                flag_dnn.TensorSpec.from_tensor(y, "y"),
            ],
            options=consts.compile_options(),
        )
        assert [node.op_type for node in compiled.graph.nodes] == ["mod"]

        def run():
            return compiled.run(x, y)

        return run

    def transfer_bytes(self, inputs):
        x, y = inputs
        out_shape = torch.broadcast_shapes(tuple(x.shape), tuple(y.shape))
        return (
            x.numel() * x.element_size()
            + y.numel() * y.element_size()
            + math.prod(out_shape) * x.element_size()
        )


@pytest.mark.mod
@pytest.mark.graph
@pytest.mark.perf
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", ModBenchmark.dtypes)
def test_mod(cudnn_handle, dtype):
    torch.manual_seed(0)
    ModBenchmark(cudnn_handle).run(dtype)
