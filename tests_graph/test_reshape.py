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


def _cudnn_reshape(x, new_shape, cudnn_handle):
    graph = cudnn_graph(x.dtype, cudnn_handle)
    x_tensor = graph.tensor_like(x)
    y_tensor = graph.reshape(
        input=x_tensor,
        name="reshape",
        reshape_mode=cudnn.reshape_mode.LOGICAL,
    )
    output_template = torch.empty(new_shape, device=x.device, dtype=x.dtype)
    y_tensor.set_dim(list(output_template.shape)).set_stride(
        list(output_template.stride())
    )
    return execute_cudnn_graph(
        graph,
        {x_tensor: x},
        y_tensor,
        output_template,
        cudnn_handle,
        "reshape",
    )


def _run_flag_dnn_reshape_graph(x, new_shape):
    @flag_dnn.graph
    def flag_dnn_reshape_graph(x):
        return flag_dnn.reshape(
            x,
            new_shape,
            name="reshape",
            reshape_mode="LOGICAL",
        )

    compiled = flag_dnn.compile(
        flag_dnn_reshape_graph,
        inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["reshape"]
    return compiled.run(x.clone())


@pytest.mark.cudnn_frontend
@pytest.mark.reshape
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("case", consts.RESHAPE_CASES)
def test_graph_reshape_matches_cudnn_frontend(cudnn_handle, dtype, case):
    torch.manual_seed(0)
    input_shape, new_shape = case
    x = torch.randn(input_shape, device=flag_dnn.device, dtype=dtype)

    cudnn_out = _cudnn_reshape(x, new_shape, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_reshape_graph(x, new_shape)

    utils.gems_assert_equal(flag_dnn_out, cudnn_out)
