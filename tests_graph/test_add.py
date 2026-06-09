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


def _cudnn_add(x, y, cudnn_handle):
    graph = cudnn_graph(x.dtype, cudnn_handle)
    x_tensor = graph.tensor_like(x)
    y_tensor = graph.tensor_like(y)
    out_tensor = graph.add(
        a=x_tensor,
        b=y_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
        name="add",
    )
    output_shape = torch.broadcast_shapes(tuple(x.shape), tuple(y.shape))
    output_template = torch.empty(
        output_shape,
        device=x.device,
        dtype=x.dtype,
    )
    return execute_cudnn_graph(
        graph,
        {x_tensor: x, y_tensor: y},
        out_tensor,
        output_template,
        cudnn_handle,
        "add",
    )


def _run_flag_dnn_add_graph(x, y):
    @flag_dnn.graph
    def flag_dnn_add_graph(x, y):
        return flag_dnn.add(
            x,
            y,
            compute_data_type="float32",
            name="add",
        )

    compiled = flag_dnn.compile(
        flag_dnn_add_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(x, "x"),
            flag_dnn.TensorSpec.from_tensor(y, "y"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["add"]
    assert compiled.graph.nodes[0].attrs["compute_data_type"] == "float32"
    assert compiled.graph.nodes[0].attrs["name"] == "add"
    return compiled.run(x.clone(), y.clone())


@pytest.mark.add
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("case", consts.ADD_CASES)
def test_add(cudnn_handle, dtype, case):
    torch.manual_seed(0)
    x_shape, y_shape = case
    x = consts.pointwise_randn(x_shape, dtype, flag_dnn.device)
    y = consts.pointwise_randn(y_shape, dtype, flag_dnn.device)

    cudnn_out = _cudnn_add(x, y, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_add_graph(x, y)

    atol = 5e-2 if dtype == torch.bfloat16 else 2e-2
    utils.gems_assert_close(flag_dnn_out, cudnn_out, dtype, atol=atol)
