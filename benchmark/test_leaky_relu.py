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

NEGATIVE_SLOPE = 0.2


class LeakyReluBenchmark(CudnnCompareBenchmark):
    op_name = "leaky_relu"
    shapes = consts.LEAKY_RELU_SHAPES
    shape_ids_env = "FLAGDNN_CUDNN_LEAKY_RELU_PERF_SHAPE_IDS"

    def make_inputs(self, shape, dtype):
        x = consts.pointwise_randn(shape, dtype, flag_dnn.device)
        return (x,)

    def build_cudnn_runner(self, inputs):
        cudnn = get_cudnn()
        (x,) = inputs
        io_dtype = cudnn_data_type(x.dtype)
        graph = cudnn.pygraph(
            io_data_type=io_dtype,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=self.cudnn_handle,
        )
        x_tensor = graph.tensor_like(x)
        if hasattr(graph, "leaky_relu"):
            y_tensor = graph.leaky_relu(
                input=x_tensor,
                negative_slope=NEGATIVE_SLOPE,
                compute_data_type=cudnn.data_type.FLOAT,
                name=self.op_name,
            )
        else:
            y_tensor = graph.relu(
                input=x_tensor,
                negative_slope=NEGATIVE_SLOPE,
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
                {x_tensor: x, y_tensor: y},
                workspace,
                handle=self.cudnn_handle,
            )
            return y

        return run

    def build_flag_dnn_runner(self, inputs):
        (x,) = inputs

        @flag_dnn.graph
        def flag_dnn_leaky_relu_graph(x):
            return flag_dnn.leaky_relu(
                x,
                negative_slope=NEGATIVE_SLOPE,
                compute_data_type="float32",
                name=self.op_name,
            )

        compiled = flag_dnn.compile(
            flag_dnn_leaky_relu_graph,
            inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
            options=consts.compile_options(),
        )
        assert [node.op_type for node in compiled.graph.nodes] == [
            "leaky_relu"
        ]

        def run():
            return compiled.run(x)

        return run


@pytest.mark.leaky_relu
@pytest.mark.graph
@pytest.mark.perf
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", LeakyReluBenchmark.dtypes)
def test_leaky_relu(cudnn_handle, dtype):
    torch.manual_seed(0)
    LeakyReluBenchmark(cudnn_handle).run(dtype)
