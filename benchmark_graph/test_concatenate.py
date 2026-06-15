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


class ConcatenateBenchmark(CudnnCompareBenchmark):
    op_name = "concatenate"
    shapes = consts.CONCATENATE_SHAPES
    shape_ids_env = "FLAGDNN_CUDNN_CONCATENATE_PERF_SHAPE_IDS"

    def make_inputs(self, case, dtype):
        self.case = case
        shapes, _ = case
        return tuple(
            consts.pointwise_randn(shape, dtype, flag_dnn.device)
            for shape in shapes
        )

    def build_cudnn_runner(self, inputs):
        cudnn = get_cudnn()
        _, axis = self.case
        io_dtype = cudnn_data_type(inputs[0].dtype)
        graph = cudnn.pygraph(
            io_data_type=io_dtype,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=self.cudnn_handle,
        )

        input_tensors = [graph.tensor_like(item) for item in inputs]
        y_tensor = graph.concatenate(
            inputs=input_tensors,
            axis=axis,
            name="concatenate",
        )
        y_tensor.set_output(True).set_data_type(io_dtype)

        try:
            graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as exc:
            skip_unsupported_cudnn_graph(exc, self.op_name)

        ref_shape = tuple(torch.cat(tuple(inputs), dim=axis).shape)
        y = torch.empty(
            ref_shape, device=inputs[0].device, dtype=inputs[0].dtype
        )
        workspace = torch.empty(
            graph.get_workspace_size(),
            device=inputs[0].device,
            dtype=torch.uint8,
        )
        exec_tensors = dict(zip(input_tensors, inputs))
        exec_tensors[y_tensor] = y

        def run():
            graph.execute(
                exec_tensors,
                workspace,
                handle=self.cudnn_handle,
            )
            return y

        return run

    def build_flag_dnn_runner(self, inputs):
        _, axis = self.case

        @flag_dnn.graph
        def flag_dnn_concatenate_graph(*values):
            return flag_dnn.concatenate(
                list(values),
                axis=axis,
                name="concatenate",
            )

        compiled = flag_dnn.compile(
            flag_dnn_concatenate_graph,
            inputs=[
                flag_dnn.TensorSpec.from_tensor(item, f"x{index}")
                for index, item in enumerate(inputs)
            ],
            options=consts.compile_options(),
        )
        assert [node.op_type for node in compiled.graph.nodes] == [
            "concatenate"
        ]

        def run():
            return compiled.run(*inputs)

        return run


@pytest.mark.concatenate
@pytest.mark.graph
@pytest.mark.perf
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", ConcatenateBenchmark.dtypes)
def test_concatenate(cudnn_handle, dtype):
    torch.manual_seed(0)
    ConcatenateBenchmark(cudnn_handle).run(dtype)
