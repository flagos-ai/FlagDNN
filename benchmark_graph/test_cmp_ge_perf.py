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


class CmpGeBenchmark(CudnnCompareBenchmark):
    op_name = "cmp_ge"
    shapes = consts.CMP_SHAPES
    shape_ids_env = "FLAGDNN_CUDNN_CMP_GE_PERF_SHAPE_IDS"

    def make_inputs(self, case, dtype):
        self.case = case
        x_shape, y_shape = case
        x = consts.pointwise_randn(x_shape, dtype, flag_dnn.device)
        y = consts.pointwise_randn(y_shape, dtype, flag_dnn.device)
        if x.numel() == y.numel():
            y_flat = y.reshape(-1)
            x_flat = x.reshape(-1)
            y_flat[::3] = x_flat[::3]
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
        out_tensor = graph.cmp_ge(
            input=x_tensor,
            comparison=y_tensor,
            compute_data_type=cudnn.data_type.FLOAT,
            name=self.op_name,
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
        def flag_dnn_cmp_ge_graph(x, y):
            return flag_dnn.cmp_ge(
                input=x,
                comparison=y,
                compute_data_type="float32",
                name=self.op_name,
            )

        compiled = flag_dnn.compile(
            flag_dnn_cmp_ge_graph,
            inputs=[
                flag_dnn.TensorSpec.from_tensor(x, "x"),
                flag_dnn.TensorSpec.from_tensor(y, "y"),
            ],
            options=consts.compile_options(),
        )
        assert [node.op_type for node in compiled.graph.nodes] == ["cmp_ge"]

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


@pytest.mark.cudnn_frontend
@pytest.mark.cmp_ge
@pytest.mark.graph
@pytest.mark.perf
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CmpGeBenchmark.dtypes)
def test_perf_graph_cmp_ge_vs_cudnn_frontend(cudnn_handle, dtype):
    torch.manual_seed(0)
    CmpGeBenchmark(cudnn_handle).run(dtype)
