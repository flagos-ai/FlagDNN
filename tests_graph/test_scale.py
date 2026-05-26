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


def _cudnn_scale(x, scale, cudnn_handle):
    graph = cudnn_graph(x.dtype, cudnn_handle)
    x_tensor = graph.tensor_like(x)
    scale_tensor = graph.tensor_like(scale)
    y_tensor = graph.scale(
        input=x_tensor,
        scale=scale_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
        name="scale",
    )
    return execute_cudnn_graph(
        graph,
        {x_tensor: x, scale_tensor: scale},
        y_tensor,
        torch.empty_like(x),
        cudnn_handle,
        "scale",
    )


def _run_flag_dnn_scale_graph(x, scale):
    @flag_dnn.graph
    def flag_dnn_scale_graph(x, scale):
        return flag_dnn.scale(
            x, scale, compute_data_type="float32", name="scale"
        )

    compiled = flag_dnn.compile(
        flag_dnn_scale_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(x, "x"),
            flag_dnn.TensorSpec.from_tensor(scale, "scale"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["scale"]
    assert compiled.graph.nodes[0].attrs["op_type"] == "mul"
    return compiled.run(x.clone(), scale.clone())


@pytest.mark.cudnn_frontend
@pytest.mark.scale
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("shape_pair", consts.SCALE_CASES)
def test_graph_scale_matches_cudnn_frontend(cudnn_handle, dtype, shape_pair):
    torch.manual_seed(0)
    shape, scale_shape = shape_pair
    x = consts.pointwise_randn(shape, dtype, flag_dnn.device)
    scale = consts.pointwise_randn(scale_shape, dtype, flag_dnn.device)

    cudnn_out = _cudnn_scale(x, scale, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_scale_graph(x, scale)

    atol = 5e-2 if dtype == torch.bfloat16 else 2e-2
    utils.gems_assert_close(flag_dnn_out, cudnn_out, dtype, atol=atol)
