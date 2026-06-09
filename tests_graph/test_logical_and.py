import pytest
from tests_graph.base import cudnn, cudnn_graph, execute_cudnn_graph
import torch

import flag_dnn
from tests_graph import consts
from tests import accuracy_utils as utils


def _cudnn_logical_and(x, y, cudnn_handle):
    graph = cudnn_graph(x.dtype, cudnn_handle)
    x_tensor = graph.tensor_like(x)
    y_tensor = graph.tensor_like(y)
    out_tensor = graph.logical_and(
        a=x_tensor,
        b=y_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
        name="logical_and",
    )
    output_shape = torch.broadcast_shapes(tuple(x.shape), tuple(y.shape))
    output_template = torch.empty(output_shape, device=x.device, dtype=x.dtype)
    return execute_cudnn_graph(
        graph,
        {x_tensor: x, y_tensor: y},
        out_tensor,
        output_template,
        cudnn_handle,
        "logical_and",
    )


def _run_flag_dnn_logical_and_graph(x, y):
    @flag_dnn.graph
    def flag_dnn_logical_and_graph(x, y):
        return flag_dnn.logical_and(
            a=x,
            b=y,
            compute_data_type="float32",
            name="logical_and",
        )

    compiled = flag_dnn.compile(
        flag_dnn_logical_and_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(x, "x"),
            flag_dnn.TensorSpec.from_tensor(y, "y"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["logical_and"]
    assert compiled.graph.nodes[0].attrs["compute_data_type"] == "float32"
    assert compiled.graph.nodes[0].attrs["name"] == "logical_and"
    return compiled.run(x.clone(), y.clone())


@pytest.mark.logical_and
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("case", consts.LOGICAL_CASES)
def test_logical_and(cudnn_handle, case):
    torch.manual_seed(0)
    x_shape, y_shape = case
    x = consts.pointwise_bool(x_shape, flag_dnn.device)
    y = consts.pointwise_bool(y_shape, flag_dnn.device)

    cudnn_out = _cudnn_logical_and(x, y, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_logical_and_graph(x, y)

    utils.gems_assert_equal(flag_dnn_out, cudnn_out)


@pytest.mark.logical_and
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_graph_logical_and_matches_torch_reference():
    torch.manual_seed(0)
    x_shape, y_shape = consts.LOGICAL_CASES[0]
    x = consts.pointwise_bool(x_shape, flag_dnn.device)
    y = consts.pointwise_bool(y_shape, flag_dnn.device)

    flag_dnn_out = _run_flag_dnn_logical_and_graph(x, y)
    utils.gems_assert_equal(flag_dnn_out, torch.logical_and(x, y))
