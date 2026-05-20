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


def _cudnn_abs(x, cudnn_handle):
    graph = cudnn_graph(x.dtype, cudnn_handle)
    x_tensor = graph.tensor_like(x)
    y_tensor = graph.abs(
        input=x_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
        name="abs",
    )
    return execute_cudnn_graph(
        graph,
        {x_tensor: x},
        y_tensor,
        torch.empty_like(x),
        cudnn_handle,
        "abs",
    )


def _run_flag_dnn_abs_graph(x):
    @flag_dnn.graph
    def flag_dnn_abs_graph(x):
        return flag_dnn.abs(
            x,
            compute_data_type="float32",
            name="abs",
        )

    compiled = flag_dnn.compile(
        flag_dnn_abs_graph,
        inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["abs"]
    assert compiled.graph.nodes[0].attrs["compute_data_type"] == "float32"
    assert compiled.graph.nodes[0].attrs["name"] == "abs"
    return compiled.run(x.clone())


@pytest.mark.cudnn_frontend
@pytest.mark.abs
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("shape", consts.ABS_SHAPES)
def test_graph_abs_matches_cudnn_frontend(cudnn_handle, dtype, shape):
    torch.manual_seed(0)
    x = torch.randn(shape, device=flag_dnn.device, dtype=dtype)

    cudnn_out = _cudnn_abs(x, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_abs_graph(x)

    utils.gems_assert_equal(flag_dnn_out, cudnn_out)
