import pytest
from tests_graph.base import (
    CUDNN_COMPARE_DTYPES,
    cudnn,
    cudnn_graph,
    execute_cudnn_graph,
)
import torch

import flag_dnn
from tests_graph import consts
from tests import accuracy_utils as utils


def _cudnn_slice(x, slices, cudnn_handle):
    graph = cudnn_graph(x.dtype, cudnn_handle)
    x_tensor = graph.tensor_like(x)
    y_tensor = graph.slice(
        input=x_tensor,
        slices=list(slices),
        compute_data_type=cudnn.data_type.FLOAT,
        name="slice",
    )
    output_template = torch.empty(
        tuple(x[tuple(slices)].shape),
        device=x.device,
        dtype=x.dtype,
    )
    return execute_cudnn_graph(
        graph,
        {x_tensor: x},
        y_tensor,
        output_template,
        cudnn_handle,
        "slice",
    )


def _run_flag_dnn_slice_graph(x, slices):
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
    return compiled.run(x.clone())


@pytest.mark.cudnn_frontend
@pytest.mark.slice
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("case", consts.SLICE_CASES)
def test_graph_slice_matches_cudnn_frontend(cudnn_handle, dtype, case):
    torch.manual_seed(0)
    shape, slices = case
    x = torch.randn(shape, device=flag_dnn.device, dtype=dtype)

    cudnn_out = _cudnn_slice(x, slices, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_slice_graph(x, slices)

    utils.gems_assert_equal(flag_dnn_out, cudnn_out)
