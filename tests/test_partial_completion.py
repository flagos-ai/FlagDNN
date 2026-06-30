import pytest
import torch

import flag_dnn
from tests import accuracy_utils as utils
from tests.base import (
    CUDNN_COMPARE_DTYPES,
    conv2d_output_template,
    cudnn,
    cudnn_data_type,
    cudnn_graph,
    execute_cudnn_graph,
)

COMPARE_DTYPES = CUDNN_COMPARE_DTYPES


def _cudnn_conv_fprop_convolution(x, weight, cudnn_handle):
    graph = cudnn_graph(x.dtype, cudnn_handle)
    x_tensor = graph.tensor_like(x)
    weight_tensor = graph.tensor_like(weight)
    out_tensor = graph.conv_fprop(
        image=x_tensor,
        weight=weight_tensor,
        padding=[1, 1],
        stride=[1, 1],
        dilation=[1, 1],
        convolution_mode=cudnn._compiled_module.convolution_mode.CONVOLUTION,
        compute_data_type=cudnn.data_type.FLOAT,
        name="conv_fprop",
    )
    output_template = conv2d_output_template(
        tuple(x.shape),
        tuple(weight.shape),
        1,
        1,
        1,
        x.dtype,
        x.device,
    )
    return execute_cudnn_graph(
        graph,
        {x_tensor: x, weight_tensor: weight},
        out_tensor,
        output_template,
        cudnn_handle,
        "conv_fprop",
    )


def _run_conv_fprop_convolution_graph(x, weight):
    @flag_dnn.graph
    def fn(x, weight):
        return flag_dnn.conv_fprop(
            x,
            weight,
            padding=1,
            stride=1,
            dilation=1,
            convolution_mode="CONVOLUTION",
        )

    compiled = flag_dnn.compile(
        fn,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(x, "x"),
            flag_dnn.TensorSpec.from_tensor(weight, "weight"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["conv_fprop"]
    return compiled.run(x.clone(), weight.clone())


def _matmul_output_shape(a, b):
    return tuple(torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])) + (
        a.shape[-2],
        b.shape[-1],
    )


def _cudnn_matmul(a, b, cudnn_handle, padding=0.0):
    graph = cudnn.pygraph(
        io_data_type=cudnn_data_type(a.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )
    a_tensor = graph.tensor_like(a)
    b_tensor = graph.tensor_like(b)
    out_tensor = graph.matmul(
        A=a_tensor,
        B=b_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
        padding=padding,
        name="matmul",
    )
    output_template = torch.empty(
        _matmul_output_shape(a, b), device=a.device, dtype=a.dtype
    )
    return execute_cudnn_graph(
        graph,
        {a_tensor: a, b_tensor: b},
        out_tensor,
        output_template,
        cudnn_handle,
        "matmul",
    )


def _run_matmul_broadcast_graph(a, b):
    @flag_dnn.graph
    def fn(a, b):
        return flag_dnn.matmul(a, b, compute_data_type="float32")

    compiled = flag_dnn.compile(
        fn,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(a, "a"),
            flag_dnn.TensorSpec.from_tensor(b, "b"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["matmul"]
    return compiled.run(a.clone(), b.clone())


def _run_matmul_padding_graph(a, b, padding):
    @flag_dnn.graph
    def fn(a, b):
        return flag_dnn.matmul(
            a, b, compute_data_type="float32", padding=padding
        )

    compiled = flag_dnn.compile(
        fn,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(a, "a"),
            flag_dnn.TensorSpec.from_tensor(b, "b"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["matmul"]
    return compiled.run(a.clone(), b.clone())


def _cudnn_reduction(x, mode, cudnn_handle):
    try:
        reduction_mode = getattr(cudnn.reduction_mode, mode)
    except AttributeError:
        pytest.skip(f"cuDNN frontend Python API does not expose {mode}")

    graph = cudnn_graph(x.dtype, cudnn_handle)
    x_tensor = graph.tensor_like(x)
    y_shape = (x.shape[0], 1, 1, x.shape[3])
    y_tensor = graph.reduction(
        input=x_tensor,
        mode=reduction_mode,
        compute_data_type=cudnn.data_type.FLOAT,
        name="reduction",
    )
    y_tensor.set_dim(y_shape).set_stride(
        (y_shape[3], y_shape[3], y_shape[3], 1)
    )
    return execute_cudnn_graph(
        graph,
        {x_tensor: x},
        y_tensor,
        torch.empty(y_shape, device=x.device, dtype=x.dtype),
        cudnn_handle,
        "reduction",
    )


def _run_reduction_graph(x, mode):
    @flag_dnn.graph
    def fn(x):
        return flag_dnn.reduction(x, mode, dim=(1, 2), keepdim=True)

    compiled = flag_dnn.compile(
        fn,
        inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["reduction"]
    return compiled.run(x.clone())


@pytest.mark.conv_fprop
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", COMPARE_DTYPES)
def test_conv_fprop_convolution_mode_matches_cudnn(cudnn_handle, dtype):
    torch.manual_seed(3)
    x = torch.randn((2, 3, 9, 11), device=flag_dnn.device, dtype=dtype)
    weight = torch.randn((4, 3, 3, 3), device=flag_dnn.device, dtype=dtype)
    cudnn_out = _cudnn_conv_fprop_convolution(x, weight, cudnn_handle)
    actual = _run_conv_fprop_convolution_graph(x, weight)
    utils.gems_assert_close(actual, cudnn_out, dtype, atol=5e-2)


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", COMPARE_DTYPES)
def test_matmul_broadcast_matches_cudnn(cudnn_handle, dtype):
    torch.manual_seed(4)
    a = torch.randn((2, 1, 4, 8), device=flag_dnn.device, dtype=dtype)
    b = torch.randn((3, 8, 5), device=flag_dnn.device, dtype=dtype)
    cudnn_out = _cudnn_matmul(a, b, cudnn_handle)
    actual = _run_matmul_broadcast_graph(a, b)
    atol = 5e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-4
    utils.gems_assert_close(actual, cudnn_out, dtype, atol=atol)


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", COMPARE_DTYPES)
def test_matmul_nonzero_padding_matches_cudnn(cudnn_handle, dtype):
    torch.manual_seed(6)
    a = torch.randn((2, 4, 8), device=flag_dnn.device, dtype=dtype)
    b = torch.randn((2, 8, 5), device=flag_dnn.device, dtype=dtype)
    cudnn_out = _cudnn_matmul(a, b, cudnn_handle, padding=1.0)
    actual = _run_matmul_padding_graph(a, b, padding=1.0)
    atol = 5e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-4
    utils.gems_assert_close(actual, cudnn_out, dtype, atol=atol)


@pytest.mark.reduction
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", COMPARE_DTYPES)
@pytest.mark.parametrize(
    "mode", ["MIN", "MAX", "AMAX", "NORM1", "NORM2", "MUL_NO_ZEROS"]
)
def test_reduction_extra_modes_match_cudnn(cudnn_handle, dtype, mode):
    torch.manual_seed(5)
    x = torch.randn((2, 3, 4, 5), device=flag_dnn.device, dtype=dtype)
    x[:, 1, 2, :] = 0
    cudnn_out = _cudnn_reduction(x, mode, cudnn_handle)
    actual = _run_reduction_graph(x, mode)
    atol = 5e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-4
    utils.gems_assert_close(actual, cudnn_out, dtype, atol=atol)
