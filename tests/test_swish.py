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

SWISH_BETA = 1.25


def _cudnn_swish(x, cudnn_handle):
    graph = cudnn_graph(x.dtype, cudnn_handle)
    x_tensor = graph.tensor_like(x)
    y_tensor = graph.swish(
        input=x_tensor,
        swish_beta=SWISH_BETA,
        compute_data_type=cudnn.data_type.FLOAT,
        name="swish",
    )
    return execute_cudnn_graph(
        graph,
        {x_tensor: x},
        y_tensor,
        torch.empty_like(x),
        cudnn_handle,
        "swish",
    )


def _run_flag_dnn_swish_graph(x):
    @flag_dnn.graph
    def flag_dnn_swish_graph(x):
        return flag_dnn.swish(
            x,
            swish_beta=SWISH_BETA,
            compute_data_type="float32",
            name="swish",
        )

    compiled = flag_dnn.compile(
        flag_dnn_swish_graph,
        inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["swish"]
    assert compiled.graph.nodes[0].attrs["swish_beta"] == SWISH_BETA
    return compiled.run(x.clone())


@pytest.mark.swish
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("shape", consts.SWISH_SHAPES)
def test_swish(cudnn_handle, dtype, shape):
    torch.manual_seed(0)
    x = consts.pointwise_randn(shape, dtype, flag_dnn.device)

    cudnn_out = _cudnn_swish(x, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_swish_graph(x)

    atol = 5e-2 if dtype == torch.bfloat16 else 2e-2
    utils.gems_assert_close(flag_dnn_out, cudnn_out, dtype, atol=atol)
