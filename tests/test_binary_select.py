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
    x_shape, y_shape, mask_shape = case
    x = consts.pointwise_randn(x_shape, dtype, flag_dnn.device)
    y = consts.pointwise_randn(y_shape, dtype, flag_dnn.device)
    mask = consts.pointwise_bool(mask_shape, flag_dnn.device).to(dtype)
    return x, y, mask


def _cudnn_binary_select(x, y, mask, cudnn_handle):
    graph = cudnn_graph(x.dtype, cudnn_handle)
    if not hasattr(graph, "binary_select"):
        pytest.skip("cuDNN frontend Python API does not expose binary_select")
    x_tensor = graph.tensor_like(x)
    y_tensor = graph.tensor_like(y)
    mask_tensor = graph.tensor_like(mask)
    try:
        out_tensor = graph.binary_select(
            input0=x_tensor,
            input1=y_tensor,
            mask=mask_tensor,
            compute_data_type=cudnn.data_type.FLOAT,
            name="binary_select",
        )
    except TypeError:
        out_tensor = graph.binary_select(
            a=x_tensor,
            b=y_tensor,
            mask=mask_tensor,
            compute_data_type=cudnn.data_type.FLOAT,
            name="binary_select",
        )
    output_shape = torch.broadcast_shapes(
        tuple(x.shape), tuple(y.shape), tuple(mask.shape)
    )
    output_template = torch.empty(output_shape, device=x.device, dtype=x.dtype)
    return execute_cudnn_graph(
        graph,
        {x_tensor: x, y_tensor: y, mask_tensor: mask},
        out_tensor,
        output_template,
        cudnn_handle,
        "binary_select",
    )


def _run_flag_dnn_binary_select_graph(x, y, mask):
    @flag_dnn.graph
    def flag_dnn_binary_select_graph(x, y, mask):
        return flag_dnn.binary_select(
            input0=x,
            input1=y,
            mask=mask,
            compute_data_type="float32",
            name="binary_select",
        )

    compiled = flag_dnn.compile(
        flag_dnn_binary_select_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(x, "x"),
            flag_dnn.TensorSpec.from_tensor(y, "y"),
            flag_dnn.TensorSpec.from_tensor(mask, "mask"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["binary_select"]
    assert compiled.graph.nodes[0].attrs["compute_data_type"] == "float32"
    assert compiled.graph.nodes[0].attrs["name"] == "binary_select"
    return compiled.run(x.clone(), y.clone(), mask.clone())


@pytest.mark.binary_select
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("case", consts.BINARY_SELECT_CASES)
def test_binary_select(cudnn_handle, dtype, case):
    torch.manual_seed(0)
    x, y, mask = _make_inputs(case, dtype)

    cudnn_out = _cudnn_binary_select(x, y, mask, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_binary_select_graph(x, y, mask)

    atol = 5e-2 if dtype == torch.bfloat16 else 2e-2
    utils.gems_assert_close(flag_dnn_out, cudnn_out, dtype, atol=atol)


@pytest.mark.binary_select
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_binary_select_single_case_matches_cudnn(cudnn_handle):
    torch.manual_seed(0)
    x, y, mask = _make_inputs(consts.BINARY_SELECT_CASES[0], torch.float32)

    cudnn_out = _cudnn_binary_select(x, y, mask, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_binary_select_graph(x, y, mask)
    utils.gems_assert_close(flag_dnn_out, cudnn_out, torch.float32)
