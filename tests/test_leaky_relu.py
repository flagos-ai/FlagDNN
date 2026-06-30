import pytest
from tests.base import (
    CUDNN_COMPARE_DTYPES,
    cudnn,
    cudnn_graph,
    execute_cudnn_graph,
)
import torch

import flag_dnn
from tests import accuracy_utils as utils
from tests import consts

NEGATIVE_SLOPE = 0.2


def _cudnn_leaky_relu(x, cudnn_handle):
    graph = cudnn_graph(x.dtype, cudnn_handle)
    x_tensor = graph.tensor_like(x)
    if hasattr(graph, "leaky_relu"):
        y_tensor = graph.leaky_relu(
            input=x_tensor,
            negative_slope=NEGATIVE_SLOPE,
            compute_data_type=cudnn.data_type.FLOAT,
            name="leaky_relu",
        )
    else:
        y_tensor = graph.relu(
            input=x_tensor,
            negative_slope=NEGATIVE_SLOPE,
            compute_data_type=cudnn.data_type.FLOAT,
            name="leaky_relu",
        )
    return execute_cudnn_graph(
        graph,
        {x_tensor: x},
        y_tensor,
        torch.empty_like(x),
        cudnn_handle,
        "leaky_relu",
    )


def _run_flag_dnn_leaky_relu_graph(x):
    @flag_dnn.graph
    def flag_dnn_leaky_relu_graph(x):
        return flag_dnn.leaky_relu(
            x,
            negative_slope=NEGATIVE_SLOPE,
            compute_data_type="float32",
            name="leaky_relu",
        )

    compiled = flag_dnn.compile(
        flag_dnn_leaky_relu_graph,
        inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["leaky_relu"]
    assert compiled.graph.nodes[0].attrs["negative_slope"] == NEGATIVE_SLOPE
    assert compiled.graph.nodes[0].attrs["compute_data_type"] == "float32"
    assert compiled.graph.nodes[0].attrs["name"] == "leaky_relu"
    return compiled.run(x.clone())


@pytest.mark.leaky_relu
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("shape", consts.LEAKY_RELU_SHAPES)
def test_leaky_relu(cudnn_handle, dtype, shape):
    torch.manual_seed(0)
    x = consts.pointwise_randn(shape, dtype, flag_dnn.device)

    cudnn_out = _cudnn_leaky_relu(x, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_leaky_relu_graph(x)

    atol = 5e-2 if dtype == torch.bfloat16 else 2e-2
    utils.gems_assert_close(flag_dnn_out, cudnn_out, dtype, atol=atol)
