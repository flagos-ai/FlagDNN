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


class GenIndexBenchmark(CudnnCompareBenchmark):
    op_name = "gen_index"
    shapes = consts.GEN_INDEX_SHAPES
    shape_ids_env = "FLAGDNN_CUDNN_GEN_INDEX_PERF_SHAPE_IDS"

    def make_inputs(self, case, dtype):
        self.case = case
        shape, _ = case
        x = torch.empty(shape, device=flag_dnn.device, dtype=dtype)
        return (x,)

    def build_cudnn_runner(self, inputs):
        cudnn = get_cudnn()
        _, axis = self.case
        (x,) = inputs
        io_dtype = cudnn_data_type(x.dtype)
        graph = cudnn.pygraph(
            io_data_type=io_dtype,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=self.cudnn_handle,
        )

        x_tensor = graph.tensor_like(x)
        y_tensor = graph.gen_index(
            input=x_tensor,
            axis=axis,
            compute_data_type=io_dtype,
            name="gen_index",
        )
        y_tensor.set_dim(list(x.shape)).set_stride(list(x.stride()))
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
        _, axis = self.case
        (x,) = inputs

        @flag_dnn.graph
        def flag_dnn_gen_index_graph(x):
            return flag_dnn.gen_index(
                x,
                axis=axis,
                compute_data_type=x.dtype,
                name="gen_index",
            )

        compiled = flag_dnn.compile(
            flag_dnn_gen_index_graph,
            inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
            options=consts.compile_options(),
        )
        assert [node.op_type for node in compiled.graph.nodes] == ["gen_index"]

        def run():
            return compiled.run(x)

        return run


@pytest.mark.gen_index
@pytest.mark.graph
@pytest.mark.perf
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", GenIndexBenchmark.dtypes)
def test_gen_index(cudnn_handle, dtype):
    torch.manual_seed(0)
    GenIndexBenchmark(cudnn_handle).run(dtype)
