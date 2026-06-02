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


class BinarySelectBenchmark(CudnnCompareBenchmark):
    op_name = "binary_select"
    shapes = consts.BINARY_SELECT_SHAPES
    shape_ids_env = "FLAGDNN_CUDNN_BINARY_SELECT_PERF_SHAPE_IDS"
    enforce_min_speedup = True

    def make_inputs(self, case, dtype):
        x_shape, y_shape, mask_shape = case
        x = consts.pointwise_randn(x_shape, dtype, flag_dnn.device)
        y = consts.pointwise_randn(y_shape, dtype, flag_dnn.device)
        mask = consts.pointwise_bool(mask_shape, flag_dnn.device).to(dtype)
        return x, y, mask

    def build_cudnn_runner(self, inputs):
        cudnn = get_cudnn()
        x, y, mask = inputs
        io_dtype = cudnn_data_type(x.dtype)
        graph = cudnn.pygraph(
            io_data_type=io_dtype,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=self.cudnn_handle,
        )
        if not hasattr(graph, "binary_select"):
            pytest.skip(
                "cuDNN frontend Python API does not expose binary_select"
            )

        x_tensor = graph.tensor_like(x)
        y_tensor = graph.tensor_like(y)
        mask_tensor = graph.tensor_like(mask)
        try:
            out_tensor = graph.binary_select(
                input0=x_tensor,
                input1=y_tensor,
                mask=mask_tensor,
                compute_data_type=cudnn.data_type.FLOAT,
                name="binary_select",
            )
        except TypeError:
            out_tensor = graph.binary_select(
                a=x_tensor,
                b=y_tensor,
                mask=mask_tensor,
                compute_data_type=cudnn.data_type.FLOAT,
                name="binary_select",
            )
        out_tensor.set_output(True).set_data_type(io_dtype)

        try:
            graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as exc:
            skip_unsupported_cudnn_graph(exc, self.op_name)

        out_shape = torch.broadcast_shapes(
            tuple(x.shape), tuple(y.shape), tuple(mask.shape)
        )
        out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        workspace = torch.empty(
            graph.get_workspace_size(), device=x.device, dtype=torch.uint8
        )

        def run():
            graph.execute(
                {
                    x_tensor: x,
                    y_tensor: y,
                    mask_tensor: mask,
                    out_tensor: out,
                },
                workspace,
                handle=self.cudnn_handle,
            )
            return out

        return run

    def build_flag_dnn_runner(self, inputs):
        x, y, mask = inputs

        @flag_dnn.graph
        def flag_dnn_binary_select_graph(x, y, mask):
            return flag_dnn.binary_select(
                input0=x,
                input1=y,
                mask=mask,
                compute_data_type="float32",
                name="binary_select",
            )

        compiled = flag_dnn.compile(
            flag_dnn_binary_select_graph,
            inputs=[
                flag_dnn.TensorSpec.from_tensor(x, "x"),
                flag_dnn.TensorSpec.from_tensor(y, "y"),
                flag_dnn.TensorSpec.from_tensor(mask, "mask"),
            ],
            options=consts.compile_options(),
        )
        assert [node.op_type for node in compiled.graph.nodes] == [
            "binary_select"
        ]

        def run():
            return compiled.run(x, y, mask)

        return run

    def transfer_bytes(self, inputs):
        x, y, mask = inputs
        out_shape = torch.broadcast_shapes(
            tuple(x.shape), tuple(y.shape), tuple(mask.shape)
        )
        return (
            x.numel() * x.element_size()
            + y.numel() * y.element_size()
            + mask.numel() * mask.element_size()
            + math.prod(out_shape) * x.element_size()
        )


@pytest.mark.cudnn_frontend
@pytest.mark.binary_select
@pytest.mark.graph
@pytest.mark.perf
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", BinarySelectBenchmark.dtypes)
def test_perf_graph_binary_select_vs_cudnn_frontend(cudnn_handle, dtype):
    torch.manual_seed(0)
    BinarySelectBenchmark(cudnn_handle).run(dtype)
