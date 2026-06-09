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


def _make_inputs(case, dtype):
    x_shape, y_shape = case
    x = consts.pointwise_randn(x_shape, dtype, flag_dnn.device)
    y = consts.pointwise_randn(y_shape, dtype, flag_dnn.device).abs()
    y = consts.pointwise_layout(y + 0.5)
    return x, y


def _cudnn_div(x, y, cudnn_handle):
    graph = cudnn_graph(x.dtype, cudnn_handle)
    x_tensor = graph.tensor_like(x)
    y_tensor = graph.tensor_like(y)
    out_tensor = graph.div(
        a=x_tensor,
        b=y_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
        name="div",
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
        "div",
    )


def _run_flag_dnn_div_graph(x, y):
    @flag_dnn.graph
    def flag_dnn_div_graph(x, y):
        return flag_dnn.div(
            x,
            y,
            compute_data_type="float32",
            name="div",
        )

    compiled = flag_dnn.compile(
        flag_dnn_div_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(x, "x"),
            flag_dnn.TensorSpec.from_tensor(y, "y"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["div"]
    assert compiled.graph.nodes[0].attrs["compute_data_type"] == "float32"
    assert compiled.graph.nodes[0].attrs["name"] == "div"
    return compiled.run(x.clone(), y.clone())


@pytest.mark.div
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("case", consts.DIV_CASES)
def test_div(cudnn_handle, dtype, case):
    torch.manual_seed(0)
    x, y = _make_inputs(case, dtype)

    cudnn_out = _cudnn_div(x, y, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_div_graph(x, y)

    atol = 5e-2 if dtype == torch.bfloat16 else 2e-2
    if "div" == "pow" and dtype == torch.float32:
        atol = 1e-4
    utils.gems_assert_close(flag_dnn_out, cudnn_out, dtype, atol=atol)
