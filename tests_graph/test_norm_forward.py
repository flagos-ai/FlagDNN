import pytest
import torch
import flag_dnn
from tests import accuracy_utils as utils

COMPARE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _reference_layernorm(x, scale, bias, eps):
    axes = tuple(i for i, size in enumerate(scale.shape) if size != 1)
    if not axes:
        axes = (x.dim() - 1,)
    xf = x.float()
    mean = torch.mean(xf, dim=axes, keepdim=True)
    inv_var = torch.rsqrt(
        torch.var(xf, dim=axes, correction=0, keepdim=True) + eps
    )
    y = (xf - mean) * inv_var * scale.float() + bias.float()
    return y.to(x.dtype), mean, inv_var


def _reference_rmsnorm(x, scale, bias, eps):
    axes = tuple(i for i, size in enumerate(scale.shape) if size != 1)
    if not axes:
        axes = (x.dim() - 1,)
    xf = x.float()
    inv_var = torch.rsqrt(torch.mean(xf * xf, dim=axes, keepdim=True) + eps)
    y = xf * inv_var * scale.float()
    if bias is not None:
        y = y + bias.float()
    return y.to(x.dtype), inv_var


def _reference_batchnorm(
    x, scale, bias, running_mean, running_var, eps, momentum
):
    reduce_dims = (0,) + tuple(range(2, x.dim()))
    xf = x.float()
    mean = torch.mean(xf, dim=reduce_dims, keepdim=True)
    var = torch.var(xf, dim=reduce_dims, correction=0, keepdim=True)
    inv_var = torch.rsqrt(var + eps)
    y = (xf - mean) * inv_var * scale.float() + bias.float()
    count = 1
    for dim in reduce_dims:
        count *= x.shape[dim]
    unbiased = var if count <= 1 else var * (count / (count - 1))
    next_mean = running_mean.float() * (1.0 - momentum) + mean * momentum
    next_var = running_var.float() * (1.0 - momentum) + unbiased * momentum
    return y.to(x.dtype), mean, inv_var, next_mean, next_var


def _run_layernorm_graph(x, scale, bias, eps):
    @flag_dnn.graph
    def fn(x, scale, bias):
        return flag_dnn.layernorm("TRAINING", x, scale, bias, eps)

    compiled = flag_dnn.compile(
        fn,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(x, "x"),
            flag_dnn.TensorSpec.from_tensor(scale, "scale"),
            flag_dnn.TensorSpec.from_tensor(bias, "bias"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["layernorm"]
    return compiled.run(x.clone(), scale.clone(), bias.clone())


def _run_rmsnorm_graph(x, scale, bias, eps):
    @flag_dnn.graph
    def fn(x, scale, bias):
        return flag_dnn.rmsnorm("TRAINING", x, scale, bias=bias, epsilon=eps)

    compiled = flag_dnn.compile(
        fn,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(x, "x"),
            flag_dnn.TensorSpec.from_tensor(scale, "scale"),
            flag_dnn.TensorSpec.from_tensor(bias, "bias"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["rmsnorm"]
    return compiled.run(x.clone(), scale.clone(), bias.clone())


def _run_batchnorm_graph(
    x, scale, bias, running_mean, running_var, eps, momentum
):
    @flag_dnn.graph
    def fn(x, scale, bias, running_mean, running_var):
        return flag_dnn.batchnorm(
            x, scale, bias, running_mean, running_var, eps, momentum
        )

    compiled = flag_dnn.compile(
        fn,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(x, "x"),
            flag_dnn.TensorSpec.from_tensor(scale, "scale"),
            flag_dnn.TensorSpec.from_tensor(bias, "bias"),
            flag_dnn.TensorSpec.from_tensor(running_mean, "running_mean"),
            flag_dnn.TensorSpec.from_tensor(running_var, "running_var"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["batchnorm"]
    return compiled.run(
        x.clone(),
        scale.clone(),
        bias.clone(),
        running_mean.clone(),
        running_var.clone(),
    )


@pytest.mark.layernorm
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", COMPARE_DTYPES)
def test_graph_layernorm_multi_output_matches_torch(dtype):
    torch.manual_seed(0)
    eps = 1e-3
    x = torch.randn((2, 5, 17), device=flag_dnn.device, dtype=dtype)
    scale = torch.randn((1, 1, 17), device=flag_dnn.device, dtype=dtype)
    bias = torch.randn((1, 1, 17), device=flag_dnn.device, dtype=dtype)
    expected = _reference_layernorm(x, scale, bias, eps)
    actual = _run_layernorm_graph(x, scale, bias, eps)
    atol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-4
    utils.gems_assert_close(actual[0], expected[0], dtype, atol=atol)
    torch.testing.assert_close(actual[1], expected[1], atol=atol, rtol=atol)
    torch.testing.assert_close(actual[2], expected[2], atol=atol, rtol=atol)


@pytest.mark.rmsnorm
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", COMPARE_DTYPES)
def test_graph_rmsnorm_multi_output_matches_torch(dtype):
    torch.manual_seed(1)
    eps = 1e-3
    x = torch.randn((2, 5, 17), device=flag_dnn.device, dtype=dtype)
    scale = torch.randn((1, 1, 17), device=flag_dnn.device, dtype=dtype)
    bias = torch.randn((1, 1, 17), device=flag_dnn.device, dtype=dtype)
    expected = _reference_rmsnorm(x, scale, bias, eps)
    actual = _run_rmsnorm_graph(x, scale, bias, eps)
    atol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-4
    utils.gems_assert_close(actual[0], expected[0], dtype, atol=atol)
    torch.testing.assert_close(actual[1], expected[1], atol=atol, rtol=atol)


@pytest.mark.batchnorm
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", COMPARE_DTYPES)
def test_graph_batchnorm_multi_output_matches_torch(dtype):
    torch.manual_seed(2)
    eps = 1e-3
    momentum = 0.1
    x = torch.randn((2, 4, 8, 8), device=flag_dnn.device, dtype=dtype)
    scale = torch.randn((1, 4, 1, 1), device=flag_dnn.device, dtype=dtype)
    bias = torch.randn((1, 4, 1, 1), device=flag_dnn.device, dtype=dtype)
    running_mean = torch.randn(
        (1, 4, 1, 1), device=flag_dnn.device, dtype=torch.float32
    )
    running_var = (
        torch.rand((1, 4, 1, 1), device=flag_dnn.device, dtype=torch.float32)
        + 0.5
    )
    expected = _reference_batchnorm(
        x, scale, bias, running_mean, running_var, eps, momentum
    )
    actual = _run_batchnorm_graph(
        x, scale, bias, running_mean, running_var, eps, momentum
    )
    atol = 3e-2 if dtype in (torch.float16, torch.bfloat16) else 3e-4
    utils.gems_assert_close(actual[0], expected[0], dtype, atol=atol)
    for actual_value, expected_value in zip(actual[1:], expected[1:]):
        torch.testing.assert_close(
            actual_value, expected_value, atol=atol, rtol=atol
        )
