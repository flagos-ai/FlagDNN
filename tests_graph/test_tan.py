import pytest
from tests_graph.base import (
    CUDNN_COMPARE_DTYPES,
    cudnn,
    cudnn_graph,
    execute_cudnn_graph,
)
import torch

import flag_dnn
from tests import accuracy_utils as utils
from tests_graph import consts


def _make_input(shape, dtype):
    return consts.pointwise_layout(
        (consts.pointwise_rand(shape, dtype, flag_dnn.device) - 0.5) * 1.0
    )


def _cudnn_tan(x, cudnn_handle):
    graph = cudnn_graph(x.dtype, cudnn_handle)
    if not hasattr(graph, "tan"):
        pytest.skip("cuDNN frontend Python API does not expose tan")
    x_tensor = graph.tensor_like(x)
    y_tensor = graph.tan(
        input=x_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
        name="tan",
    )
    return execute_cudnn_graph(
        graph,
        {x_tensor: x},
        y_tensor,
        torch.empty_like(x),
        cudnn_handle,
        "tan",
    )


def _run_flag_dnn_tan_graph(x):
    @flag_dnn.graph
    def flag_dnn_tan_graph(x):
        return flag_dnn.tan(
            x,
            compute_data_type="float32",
            name="tan",
        )

    compiled = flag_dnn.compile(
        flag_dnn_tan_graph,
        inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["tan"]
    assert compiled.graph.nodes[0].attrs["compute_data_type"] == "float32"
    assert compiled.graph.nodes[0].attrs["name"] == "tan"
    return compiled.run(x.clone())


@pytest.mark.tan
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("shape", consts.TAN_SHAPES)
def test_tan(cudnn_handle, dtype, shape):
    torch.manual_seed(0)
    x = _make_input(shape, dtype)

    cudnn_out = _cudnn_tan(x, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_tan_graph(x)

    atol = 5e-2 if dtype == torch.bfloat16 else 2e-2
    utils.gems_assert_close(flag_dnn_out, cudnn_out, dtype, atol=atol)
