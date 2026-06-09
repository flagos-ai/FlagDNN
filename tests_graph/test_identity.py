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


def _cudnn_identity(x, cudnn_handle):
    graph = cudnn_graph(x.dtype, cudnn_handle)
    x_tensor = graph.tensor_like(x)
    y_tensor = graph.identity(
        input=x_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
        name="identity",
    )
    return execute_cudnn_graph(
        graph,
        {x_tensor: x},
        y_tensor,
        torch.empty_like(x),
        cudnn_handle,
        "identity",
    )


def _run_flag_dnn_identity_graph(x):
    @flag_dnn.graph
    def flag_dnn_identity_graph(x):
        return flag_dnn.identity(
            x,
            compute_data_type="float32",
            name="identity",
        )

    compiled = flag_dnn.compile(
        flag_dnn_identity_graph,
        inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["identity"]
    return compiled.run(x.clone())


@pytest.mark.identity
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("shape", consts.IDENTITY_SHAPES)
def test_identity(cudnn_handle, dtype, shape):
    torch.manual_seed(0)
    x = consts.pointwise_randn(shape, dtype, flag_dnn.device)

    cudnn_out = _cudnn_identity(x, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_identity_graph(x)

    utils.gems_assert_equal(flag_dnn_out, cudnn_out)
