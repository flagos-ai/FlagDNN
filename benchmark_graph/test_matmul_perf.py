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


class MatmulBenchmark(CudnnCompareBenchmark):
    op_name = "matmul"
    shapes = consts.MATMUL_SHAPES
    shape_ids_env = "FLAGDNN_CUDNN_MATMUL_PERF_SHAPE_IDS"

    def make_inputs(self, shape_pair, dtype):
        a_shape, b_shape = shape_pair
        a = torch.randn(a_shape, dtype=dtype, device=flag_dnn.device)
        b = torch.randn(b_shape, dtype=dtype, device=flag_dnn.device)
        return a, b

    def build_cudnn_runner(self, inputs):
        cudnn = get_cudnn()
        a, b = inputs
        io_dtype = cudnn_data_type(a.dtype)
        graph = cudnn.pygraph(
            io_data_type=io_dtype,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=self.cudnn_handle,
        )
        a_tensor = graph.tensor_like(a)
        b_tensor = graph.tensor_like(b)
        y_tensor = graph.matmul(
            A=a_tensor,
            B=b_tensor,
            compute_data_type=cudnn.data_type.FLOAT,
            name=self.op_name,
        )
        y_tensor.set_output(True).set_data_type(io_dtype)

        try:
            graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as exc:
            skip_unsupported_cudnn_graph(exc, self.op_name)

        y = torch.empty(
            (*a.shape[:-1], b.shape[-1]), device=a.device, dtype=a.dtype
        )
        workspace = torch.empty(
            graph.get_workspace_size(), device=a.device, dtype=torch.uint8
        )

        def run():
            graph.execute(
                {a_tensor: a, b_tensor: b, y_tensor: y},
                workspace,
                handle=self.cudnn_handle,
            )
            return y

        return run

    def build_flag_dnn_runner(self, inputs):
        a, b = inputs

        @flag_dnn.graph
        def flag_dnn_matmul_graph(a, b):
            return flag_dnn.matmul(
                a, b, compute_data_type="float32", name=self.op_name
            )

        compiled = flag_dnn.compile(
            flag_dnn_matmul_graph,
            inputs=[
                flag_dnn.TensorSpec.from_tensor(a, "a"),
                flag_dnn.TensorSpec.from_tensor(b, "b"),
            ],
            options=consts.compile_options(),
        )
        assert [node.op_type for node in compiled.graph.nodes] == ["matmul"]

        def run():
            return compiled.run(a, b)

        return run

    def transfer_bytes(self, inputs):
        a, b = inputs
        output_elements = math.prod((*a.shape[:-1], b.shape[-1]))
        return (
            a.numel() * a.element_size()
            + b.numel() * b.element_size()
            + output_elements * a.element_size()
        )


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.perf
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", MatmulBenchmark.dtypes)
def test_matmul(cudnn_handle, dtype):
    torch.manual_seed(0)
    MatmulBenchmark(cudnn_handle).run(dtype)
