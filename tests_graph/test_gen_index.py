import pytest
from tests_graph.base import (
    CUDNN_COMPARE_DTYPES,
    cudnn_data_type,
    cudnn_graph,
    execute_cudnn_graph,
)
import torch

import flag_dnn
from tests_graph import consts
from tests import accuracy_utils as utils


def _cudnn_gen_index(x, axis, cudnn_handle):
    graph = cudnn_graph(x.dtype, cudnn_handle)
    x_tensor = graph.tensor_like(x)
    y_tensor = graph.gen_index(
        input=x_tensor,
        axis=axis,
        compute_data_type=cudnn_data_type(x.dtype),
        name="gen_index",
    )
    y_tensor.set_dim(list(x.shape)).set_stride(list(x.stride()))
    return execute_cudnn_graph(
        graph,
        {x_tensor: x},
        y_tensor,
        torch.empty_like(x),
        cudnn_handle,
        "gen_index",
    )


def _run_flag_dnn_gen_index_graph(x, axis):
    @flag_dnn.graph
    def flag_dnn_gen_index_graph(x):
        return flag_dnn.gen_index(
            x,
            axis=axis,
            compute_data_type=x.dtype,
            name="gen_index",
        )

    compiled = flag_dnn.compile(
        flag_dnn_gen_index_graph,
        inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["gen_index"]
    return compiled.run(x.clone())


@pytest.mark.gen_index
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("case", consts.GEN_INDEX_CASES)
def test_gen_index(cudnn_handle, dtype, case):
    torch.manual_seed(0)
    shape, axis = case
    x = torch.empty(shape, device=flag_dnn.device, dtype=dtype)

    cudnn_out = _cudnn_gen_index(x, axis, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_gen_index_graph(x, axis)

    utils.gems_assert_equal(flag_dnn_out, cudnn_out)
