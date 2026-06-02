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
    del x_shape
    y = consts.pointwise_positive(y_shape, dtype, flag_dnn.device, offset=0.5)
    x = consts.pointwise_layout(y * 2.5)
    return x, y


def _make_signed_inputs(dtype):
    x = torch.tensor(
        [-3.0, -3.0, 3.0, 3.0, -5.5, 5.5],
        device=flag_dnn.device,
        dtype=dtype,
    ).reshape(1, 1, 6)
    y = torch.tensor(
        [2.0, -2.0, 2.0, -2.0, 2.25, -2.25],
        device=flag_dnn.device,
        dtype=dtype,
    ).reshape(1, 1, 6)
    return x, y


def _cudnn_mod(x, y, cudnn_handle):
    graph = cudnn_graph(x.dtype, cudnn_handle)
    if not hasattr(graph, "mod"):
        pytest.skip("cuDNN frontend Python API does not expose mod")
    x_tensor = graph.tensor_like(x)
    y_tensor = graph.tensor_like(y)
    try:
        out_tensor = graph.mod(
            input0=x_tensor,
            input1=y_tensor,
            compute_data_type=cudnn.data_type.FLOAT,
            name="mod",
        )
    except TypeError:
        out_tensor = graph.mod(
            a=x_tensor,
            b=y_tensor,
            compute_data_type=cudnn.data_type.FLOAT,
            name="mod",
        )
    output_shape = torch.broadcast_shapes(tuple(x.shape), tuple(y.shape))
    output_template = torch.empty(output_shape, device=x.device, dtype=x.dtype)
    return execute_cudnn_graph(
        graph,
        {x_tensor: x, y_tensor: y},
        out_tensor,
        output_template,
        cudnn_handle,
        "mod",
    )


def _run_flag_dnn_mod_graph(x, y):
    @flag_dnn.graph
    def flag_dnn_mod_graph(x, y):
        return flag_dnn.mod(
            input0=x,
            input1=y,
            compute_data_type="float32",
            name="mod",
        )

    compiled = flag_dnn.compile(
        flag_dnn_mod_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(x, "x"),
            flag_dnn.TensorSpec.from_tensor(y, "y"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["mod"]
    assert compiled.graph.nodes[0].attrs["compute_data_type"] == "float32"
    assert compiled.graph.nodes[0].attrs["name"] == "mod"
    return compiled.run(x.clone(), y.clone())


@pytest.mark.cudnn_frontend
@pytest.mark.mod
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("case", consts.MOD_CASES)
def test_graph_mod_matches_cudnn_frontend(cudnn_handle, dtype, case):
    torch.manual_seed(0)
    x, y = _make_inputs(case, dtype)

    cudnn_out = _cudnn_mod(x, y, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_mod_graph(x, y)

    atol = 5e-2 if dtype == torch.bfloat16 else 2e-2
    utils.gems_assert_close(flag_dnn_out, cudnn_out, dtype, atol=atol)


@pytest.mark.mod
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_graph_mod_matches_reference():
    torch.manual_seed(0)
    x, y = _make_inputs(consts.MOD_CASES[0], torch.float32)

    flag_dnn_out = _run_flag_dnn_mod_graph(x, y)
    utils.gems_assert_close(flag_dnn_out, torch.fmod(x, y), torch.float32)


@pytest.mark.cudnn_frontend
@pytest.mark.mod
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
def test_graph_mod_signed_matches_cudnn_frontend(cudnn_handle, dtype):
    x, y = _make_signed_inputs(dtype)

    cudnn_out = _cudnn_mod(x, y, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_mod_graph(x, y)

    utils.gems_assert_close(flag_dnn_out, cudnn_out, dtype, atol=2e-2)


@pytest.mark.mod
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_graph_mod_signed_matches_reference():
    x, y = _make_signed_inputs(torch.float32)

    flag_dnn_out = _run_flag_dnn_mod_graph(x, y)
    utils.gems_assert_close(flag_dnn_out, torch.fmod(x, y), torch.float32)
