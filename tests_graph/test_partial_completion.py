import pytest
import torch
import torch.nn.functional as F
import flag_dnn
from tests import accuracy_utils as utils

COMPARE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


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


def _prod_reduce(input, dims, keepdim):
    result = input
    ordered_dims = dims if keepdim else sorted(dims, reverse=True)
    for dim in ordered_dims:
        result = torch.prod(result, dim=dim, keepdim=keepdim)
    return result


def _reduction_reference(x, mode):
    dims = (1, 2)
    if mode == "MIN":
        return torch.amin(x, dim=dims, keepdim=True)
    if mode == "MAX":
        return torch.amax(x, dim=dims, keepdim=True)
    if mode == "AMAX":
        return torch.amax(torch.abs(x), dim=dims, keepdim=True)
    if mode == "NORM1":
        return torch.sum(torch.abs(x), dim=dims, keepdim=True)
    if mode == "NORM2":
        return torch.sqrt(torch.sum(x * x, dim=dims, keepdim=True))
    if mode == "MUL_NO_ZEROS":
        return _prod_reduce(
            torch.where(x == 0, torch.ones_like(x), x), dims, True
        )
    raise AssertionError(mode)


@pytest.mark.conv_fprop
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", COMPARE_DTYPES)
def test_graph_conv_fprop_convolution_mode_matches_torch(dtype):
    torch.manual_seed(3)
    x = torch.randn((2, 3, 9, 11), device=flag_dnn.device, dtype=dtype)
    weight = torch.randn((4, 3, 3, 3), device=flag_dnn.device, dtype=dtype)
    expected = F.conv2d(x, torch.flip(weight, dims=(2, 3)), padding=1)
    actual = _run_conv_fprop_convolution_graph(x, weight)
    atol = 5e-2
    utils.gems_assert_close(actual, expected, dtype, atol=atol)


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", COMPARE_DTYPES)
def test_graph_matmul_broadcast_matches_torch(dtype):
    torch.manual_seed(4)
    a = torch.randn((2, 1, 4, 8), device=flag_dnn.device, dtype=dtype)
    b = torch.randn((3, 8, 5), device=flag_dnn.device, dtype=dtype)
    expected = torch.matmul(a, b)
    actual = _run_matmul_broadcast_graph(a, b)
    atol = 5e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-4
    utils.gems_assert_close(actual, expected, dtype, atol=atol)


@pytest.mark.matmul
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", COMPARE_DTYPES)
def test_graph_matmul_nonzero_padding_matches_torch(dtype):
    torch.manual_seed(6)
    a = torch.randn((2, 4, 8), device=flag_dnn.device, dtype=dtype)
    b = torch.randn((2, 8, 5), device=flag_dnn.device, dtype=dtype)
    expected = torch.matmul(a, b)
    actual = _run_matmul_padding_graph(a, b, padding=1.0)
    atol = 5e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-4
    utils.gems_assert_close(actual, expected, dtype, atol=atol)


@pytest.mark.reduction
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", COMPARE_DTYPES)
@pytest.mark.parametrize(
    "mode", ["MIN", "MAX", "AMAX", "NORM1", "NORM2", "MUL_NO_ZEROS"]
)
def test_graph_reduction_extra_modes_match_torch(dtype, mode):
    torch.manual_seed(5)
    x = torch.randn((2, 3, 4, 5), device=flag_dnn.device, dtype=dtype)
    x[:, 1, 2, :] = 0
    expected = _reduction_reference(x, mode)
    actual = _run_reduction_graph(x, mode)
    atol = 5e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-4
    utils.gems_assert_close(actual, expected, dtype, atol=atol)
