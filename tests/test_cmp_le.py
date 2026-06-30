import pytest
from tests.base import (
    CUDNN_COMPARE_DTYPES,
    cudnn,
    cudnn_graph,
    execute_cudnn_graph,
)
import torch

import flag_dnn
from tests import consts
from tests import accuracy_utils as utils


def _make_inputs(case, dtype):
    x_shape, y_shape = case
    x = consts.pointwise_randn(x_shape, dtype, flag_dnn.device)
    y = consts.pointwise_randn(y_shape, dtype, flag_dnn.device)
    if x.numel() == y.numel():
        y_flat = y.reshape(-1)
        x_flat = x.reshape(-1)
        y_flat[::3] = x_flat[::3]
    return x, y


def _cudnn_cmp_le(x, y, cudnn_handle):
    graph = cudnn_graph(x.dtype, cudnn_handle)
    x_tensor = graph.tensor_like(x)
    y_tensor = graph.tensor_like(y)
    out_tensor = graph.cmp_le(
        input=x_tensor,
        comparison=y_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
        name="cmp_le",
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
        "cmp_le",
    )


def _run_flag_dnn_cmp_le_graph(x, y):
    @flag_dnn.graph
    def flag_dnn_cmp_le_graph(x, y):
        return flag_dnn.cmp_le(
            input=x,
            comparison=y,
            compute_data_type="float32",
            name="cmp_le",
        )

    compiled = flag_dnn.compile(
        flag_dnn_cmp_le_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(x, "x"),
            flag_dnn.TensorSpec.from_tensor(y, "y"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["cmp_le"]
    assert compiled.graph.nodes[0].attrs["compute_data_type"] == "float32"
    assert compiled.graph.nodes[0].attrs["name"] == "cmp_le"
    return compiled.run(x.clone(), y.clone())


@pytest.mark.cmp_le
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("case", consts.CMP_CASES)
def test_cmp_le(cudnn_handle, dtype, case):
    torch.manual_seed(0)
    x, y = _make_inputs(case, dtype)

    cudnn_out = _cudnn_cmp_le(x, y, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_cmp_le_graph(x, y)

    utils.gems_assert_equal(flag_dnn_out, cudnn_out != 0)
