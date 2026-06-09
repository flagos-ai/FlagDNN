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


class ReciprocalBenchmark(CudnnCompareBenchmark):
    op_name = "reciprocal"
    shapes = consts.RECIPROCAL_SHAPES
    shape_ids_env = "FLAGDNN_CUDNN_RECIPROCAL_PERF_SHAPE_IDS"
    enforce_min_speedup = True

    def make_inputs(self, shape, dtype):
        x = consts.pointwise_positive(
            shape, dtype, flag_dnn.device, offset=0.5
        )
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
        if not hasattr(graph, "reciprocal"):
            pytest.skip("cuDNN frontend Python API does not expose reciprocal")

        x_tensor = graph.tensor_like(x)
        y_tensor = graph.reciprocal(
            input=x_tensor,
            compute_data_type=cudnn.data_type.FLOAT,
            name="reciprocal",
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
        def flag_dnn_reciprocal_graph(x):
            return flag_dnn.reciprocal(
                x,
                compute_data_type="float32",
                name="reciprocal",
            )

        compiled = flag_dnn.compile(
            flag_dnn_reciprocal_graph,
            inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
            options=consts.compile_options(),
        )
        assert [node.op_type for node in compiled.graph.nodes] == [
            "reciprocal"
        ]

        def run():
            return compiled.run(x)

        return run


@pytest.mark.reciprocal
@pytest.mark.graph
@pytest.mark.perf
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", ReciprocalBenchmark.dtypes)
def test_reciprocal(cudnn_handle, dtype):
    torch.manual_seed(0)
    ReciprocalBenchmark(cudnn_handle).run(dtype)
