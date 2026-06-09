import pytest
from tests_graph.base import (
    CUDNN_COMPARE_DTYPES,
    cudnn,
    cudnn_data_type,
    execute_cudnn_graph,
)
import torch

import flag_dnn
from tests import accuracy_utils as utils
from tests_graph import consts


def _cudnn_matmul(a, b, cudnn_handle):
    graph = cudnn.pygraph(
        io_data_type=cudnn_data_type(a.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )
    a_tensor = graph.tensor_like(a)
    b_tensor = graph.tensor_like(b)
    y_tensor = graph.matmul(
        A=a_tensor,
        B=b_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
        name="matmul",
    )
    return execute_cudnn_graph(
        graph,
        {a_tensor: a, b_tensor: b},
        y_tensor,
        torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype),
        cudnn_handle,
        "matmul",
    )


def _run_flag_dnn_matmul_graph(a, b):
    @flag_dnn.graph
    def flag_dnn_matmul_graph(a, b):
        return flag_dnn.matmul(
            a, b, compute_data_type="float32", name="matmul"
        )

    compiled = flag_dnn.compile(
        flag_dnn_matmul_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(a, "a"),
            flag_dnn.TensorSpec.from_tensor(b, "b"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["matmul"]
    return compiled.run(a.clone(), b.clone())


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("shape_pair", consts.MATMUL_CASES)
def test_matmul(cudnn_handle, dtype, shape_pair):
    torch.manual_seed(0)
    a_shape, b_shape = shape_pair
    a = torch.randn(a_shape, dtype=dtype, device=flag_dnn.device)
    b = torch.randn(b_shape, dtype=dtype, device=flag_dnn.device)

    cudnn_out = _cudnn_matmul(a, b, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_matmul_graph(a, b)

    atol = 1e-1 if dtype == torch.bfloat16 else 5e-2
    utils.gems_assert_close(flag_dnn_out, cudnn_out, dtype, atol=atol)
