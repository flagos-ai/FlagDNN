import pytest
from tests_graph.base import cudnn, cudnn_graph, execute_cudnn_graph
import torch

import flag_dnn
from tests_graph import consts
from tests import accuracy_utils as utils


def _cudnn_logical_not(x, cudnn_handle):
    graph = cudnn_graph(x.dtype, cudnn_handle)
    x_tensor = graph.tensor_like(x)
    out_tensor = graph.logical_not(
        input=x_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
        name="logical_not",
    )
    return execute_cudnn_graph(
        graph,
        {x_tensor: x},
        out_tensor,
        torch.empty_like(x),
        cudnn_handle,
        "logical_not",
    )


def _run_flag_dnn_logical_not_graph(x):
    @flag_dnn.graph
    def flag_dnn_logical_not_graph(x):
        return flag_dnn.logical_not(
            x,
            compute_data_type="float32",
            name="logical_not",
        )

    compiled = flag_dnn.compile(
        flag_dnn_logical_not_graph,
        inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["logical_not"]
    assert compiled.graph.nodes[0].attrs["compute_data_type"] == "float32"
    assert compiled.graph.nodes[0].attrs["name"] == "logical_not"
    return compiled.run(x.clone())


@pytest.mark.logical_not
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("shape", consts.EXP_SHAPES)
def test_logical_not(cudnn_handle, shape):
    torch.manual_seed(0)
    x = consts.pointwise_bool(shape, flag_dnn.device)

    cudnn_out = _cudnn_logical_not(x, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_logical_not_graph(x)

    utils.gems_assert_equal(flag_dnn_out, cudnn_out)


@pytest.mark.logical_not
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_graph_logical_not_matches_torch_reference():
    torch.manual_seed(0)
    x = consts.pointwise_bool(consts.EXP_SHAPES[0], flag_dnn.device)

    flag_dnn_out = _run_flag_dnn_logical_not_graph(x)
    utils.gems_assert_equal(flag_dnn_out, torch.logical_not(x))
