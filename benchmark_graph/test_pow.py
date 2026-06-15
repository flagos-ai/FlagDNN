import math

import pytest
from benchmark_graph.base import (
    CudnnCompareBenchmark,
    cudnn_data_type,
    get_cudnn,
    skip_unsupported_cudnn_graph,
)
import torch

import flag_dnn
from benchmark_graph import consts


class PowBenchmark(CudnnCompareBenchmark):
    op_name = "pow"
    shapes = consts.POW_SHAPES
    shape_ids_env = "FLAGDNN_CUDNN_POW_PERF_SHAPE_IDS"

    def make_inputs(self, case, dtype):
        self.case = case
        x_shape, y_shape = case
        x = consts.pointwise_layout(
            consts.pointwise_rand(x_shape, dtype, flag_dnn.device) + 0.5
        )
        y = consts.pointwise_layout(
            consts.pointwise_rand(y_shape, dtype, flag_dnn.device) * 2.0
        )
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

        x_tensor = graph.tensor_like(x)
        y_tensor = graph.tensor_like(y)
        out_tensor = graph.pow(
            input0=x_tensor,
            input1=y_tensor,
            compute_data_type=cudnn.data_type.FLOAT,
            name="pow",
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
        def flag_dnn_pow_graph(x, y):
            return flag_dnn.pow(
                input=x,
                exponent=y,
                compute_data_type="float32",
                name="pow",
            )

        compiled = flag_dnn.compile(
            flag_dnn_pow_graph,
            inputs=[
                flag_dnn.TensorSpec.from_tensor(x, "x"),
                flag_dnn.TensorSpec.from_tensor(y, "y"),
            ],
            options=consts.compile_options(),
        )
        assert [node.op_type for node in compiled.graph.nodes] == ["pow"]

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


@pytest.mark.pow
@pytest.mark.graph
@pytest.mark.perf
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", PowBenchmark.dtypes)
def test_pow(cudnn_handle, dtype):
    torch.manual_seed(0)
    PowBenchmark(cudnn_handle).run(dtype)
