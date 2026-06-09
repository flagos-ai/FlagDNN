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


class ReshapeBenchmark(CudnnCompareBenchmark):
    op_name = "reshape"
    shapes = consts.RESHAPE_SHAPES
    shape_ids_env = "FLAGDNN_CUDNN_RESHAPE_PERF_SHAPE_IDS"

    def make_inputs(self, case, dtype):
        self.case = case
        input_shape, _ = case
        x = torch.randn(input_shape, device=flag_dnn.device, dtype=dtype)
        return (x,)

    def build_cudnn_runner(self, inputs):
        cudnn = get_cudnn()
        input_shape, new_shape = self.case
        (x,) = inputs
        io_dtype = cudnn_data_type(x.dtype)
        graph = cudnn.pygraph(
            io_data_type=io_dtype,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=self.cudnn_handle,
        )

        x_tensor = graph.tensor_like(x)
        y_tensor = graph.reshape(
            input=x_tensor,
            name="reshape",
            reshape_mode=cudnn.reshape_mode.LOGICAL,
        )
        y = torch.empty(new_shape, device=x.device, dtype=x.dtype)
        y_tensor.set_dim(list(y.shape)).set_stride(list(y.stride()))
        y_tensor.set_output(True).set_data_type(io_dtype)

        try:
            graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as exc:
            skip_unsupported_cudnn_graph(exc, self.op_name)

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
        _, new_shape = self.case
        (x,) = inputs

        @flag_dnn.graph
        def flag_dnn_reshape_graph(x):
            return flag_dnn.reshape(
                x,
                new_shape,
                name="reshape",
                reshape_mode="LOGICAL",
            )

        compiled = flag_dnn.compile(
            flag_dnn_reshape_graph,
            inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
            options=consts.compile_options(),
        )
        assert [node.op_type for node in compiled.graph.nodes] == ["reshape"]

        def run():
            return compiled.run(x)

        return run


@pytest.mark.reshape
@pytest.mark.graph
@pytest.mark.perf
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", ReshapeBenchmark.dtypes)
def test_reshape(cudnn_handle, dtype):
    torch.manual_seed(0)
    ReshapeBenchmark(cudnn_handle).run(dtype)
