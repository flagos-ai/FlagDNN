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
        consts.pointwise_randn(shape, dtype, flag_dnn.device) * 4.0
    )


def _cudnn_floor(x, cudnn_handle):
    graph = cudnn_graph(x.dtype, cudnn_handle)
    if not hasattr(graph, "floor"):
        pytest.skip("cuDNN frontend Python API does not expose floor")
    x_tensor = graph.tensor_like(x)
    y_tensor = graph.floor(
        input=x_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
        name="floor",
    )
    return execute_cudnn_graph(
        graph,
        {x_tensor: x},
        y_tensor,
        torch.empty_like(x),
        cudnn_handle,
        "floor",
    )


def _run_flag_dnn_floor_graph(x):
    @flag_dnn.graph
    def flag_dnn_floor_graph(x):
        return flag_dnn.floor(
            x,
            compute_data_type="float32",
            name="floor",
        )

    compiled = flag_dnn.compile(
        flag_dnn_floor_graph,
        inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["floor"]
    assert compiled.graph.nodes[0].attrs["compute_data_type"] == "float32"
    assert compiled.graph.nodes[0].attrs["name"] == "floor"
    return compiled.run(x.clone())


@pytest.mark.cudnn_frontend
@pytest.mark.floor
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("shape", consts.FLOOR_SHAPES)
def test_graph_floor_matches_cudnn_frontend(cudnn_handle, dtype, shape):
    torch.manual_seed(0)
    x = _make_input(shape, dtype)

    cudnn_out = _cudnn_floor(x, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_floor_graph(x)

    atol = 5e-2 if dtype == torch.bfloat16 else 2e-2
    utils.gems_assert_close(flag_dnn_out, cudnn_out, dtype, atol=atol)
