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


class SliceBenchmark(CudnnCompareBenchmark):
    op_name = "slice"
    shapes = consts.CUDNN_SLICE_SHAPES
    shape_ids_env = "FLAGDNN_CUDNN_SLICE_PERF_SHAPE_IDS"

    def make_inputs(self, case, dtype):
        self.case = case
        shape, _ = case
        x = torch.randn(shape, device=flag_dnn.device, dtype=dtype)
        return (x,)

    def build_cudnn_runner(self, inputs):
        cudnn = get_cudnn()
        _, slices = self.case
        (x,) = inputs
        io_dtype = cudnn_data_type(x.dtype)
        graph = cudnn.pygraph(
            io_data_type=io_dtype,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=self.cudnn_handle,
        )

        x_tensor = graph.tensor_like(x)
        y_tensor = graph.slice(
            input=x_tensor,
            slices=list(slices),
            compute_data_type=cudnn.data_type.FLOAT,
            name="slice",
        )
        y_tensor.set_output(True).set_data_type(io_dtype)

        try:
            graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as exc:
            skip_unsupported_cudnn_graph(exc, self.op_name)

        y = torch.empty(
            tuple(x[tuple(slices)].shape),
            device=x.device,
            dtype=x.dtype,
        )
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
        _, slices = self.case
        (x,) = inputs

        @flag_dnn.graph
        def flag_dnn_slice_graph(x):
            return flag_dnn.slice(
                x,
                slices,
                compute_data_type="float32",
                name="slice",
            )

        compiled = flag_dnn.compile(
            flag_dnn_slice_graph,
            inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
            options={"cache": None},
        )
        assert [node.op_type for node in compiled.graph.nodes] == ["slice"]

        def run():
            return compiled.run(x)

        return run


@pytest.mark.cudnn_frontend
@pytest.mark.slice
@pytest.mark.graph
@pytest.mark.perf
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", SliceBenchmark.dtypes)
def test_perf_graph_slice_vs_cudnn_frontend(cudnn_handle, dtype):
    torch.manual_seed(0)
    SliceBenchmark(cudnn_handle).run(dtype)
