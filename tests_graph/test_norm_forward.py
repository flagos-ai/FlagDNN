import pytest
import torch

import flag_dnn
from tests import accuracy_utils as utils
from tests_graph.base import (
    cudnn,
    cudnn_data_type,
    skip_unsupported_cudnn_graph,
)

COMPARE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for size in reversed(shape):
        stride.append(running)
        running *= size
    return tuple(reversed(stride))


def _norm_stats_shape(x, scale):
    axes = tuple(i for i, size in enumerate(scale.shape) if size != 1)
    if not axes:
        axes = (x.dim() - 1,)
    axes = set(axes)
    return tuple(
        1 if index in axes else size for index, size in enumerate(x.shape)
    )


def _scalar_tensor(graph, rank, name):
    return graph.tensor(
        dim=(1,) * rank,
        stride=(1,) * rank,
        data_type=cudnn.data_type.FLOAT,
        is_pass_by_value=True,
        name=name,
    )


def _scalar_value(value, rank):
    return torch.full((1,) * rank, value, dtype=torch.float32, device="cpu")


def _execute_cudnn_outputs(
    graph, exec_tensors, outputs, device, cudnn_handle, op_name
):
    for tensor, shape, dtype in outputs:
        tensor.set_output(True).set_dim(tuple(shape)).set_stride(
            _contiguous_stride(shape)
        ).set_data_type(cudnn_data_type(dtype))

    try:
        graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as exc:
        skip_unsupported_cudnn_graph(exc, op_name)

    actual = []
    for tensor, shape, dtype in outputs:
        out = torch.empty_strided(
            tuple(shape),
            _contiguous_stride(shape),
            device=device,
            dtype=dtype,
        )
        exec_tensors[tensor] = out
        actual.append(out)

    workspace = torch.empty(
        graph.get_workspace_size(), device=device, dtype=torch.uint8
    )
    graph.execute(exec_tensors, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()
    return tuple(actual)


def _cudnn_layernorm(x, scale, bias, eps, cudnn_handle):
    graph = cudnn.pygraph(
        io_data_type=cudnn_data_type(x.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )
    x_tensor = graph.tensor_like(x)
    scale_tensor = graph.tensor_like(scale)
    bias_tensor = graph.tensor_like(bias)
    eps_tensor = _scalar_tensor(graph, x.dim(), "epsilon")
    y_tensor, mean_tensor, inv_var_tensor = graph.layernorm(
        cudnn.norm_forward_phase.TRAINING,
        x_tensor,
        scale_tensor,
        bias_tensor,
        eps_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
        name="layernorm",
    )
    stat_shape = _norm_stats_shape(x, scale)
    return _execute_cudnn_outputs(
        graph,
        {
            x_tensor: x,
            scale_tensor: scale,
            bias_tensor: bias,
            eps_tensor: _scalar_value(eps, x.dim()),
        },
        (
            (y_tensor, tuple(x.shape), x.dtype),
            (mean_tensor, stat_shape, torch.float32),
            (inv_var_tensor, stat_shape, torch.float32),
        ),
        x.device,
        cudnn_handle,
        "layernorm",
    )


def _cudnn_rmsnorm(x, scale, bias, eps, cudnn_handle):
    graph = cudnn.pygraph(
        io_data_type=cudnn_data_type(x.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )
    x_tensor = graph.tensor_like(x)
    scale_tensor = graph.tensor_like(scale)
    bias_tensor = graph.tensor_like(bias)
    eps_tensor = _scalar_tensor(graph, x.dim(), "epsilon")
    y_tensor, inv_var_tensor = graph.rmsnorm(
        cudnn.norm_forward_phase.TRAINING,
        x_tensor,
        scale_tensor,
        bias=bias_tensor,
        epsilon=eps_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
        name="rmsnorm",
    )
    stat_shape = _norm_stats_shape(x, scale)
    return _execute_cudnn_outputs(
        graph,
        {
            x_tensor: x,
            scale_tensor: scale,
            bias_tensor: bias,
            eps_tensor: _scalar_value(eps, x.dim()),
        },
        (
            (y_tensor, tuple(x.shape), x.dtype),
            (inv_var_tensor, stat_shape, torch.float32),
        ),
        x.device,
        cudnn_handle,
        "rmsnorm",
    )


def _cudnn_batchnorm(
    x,
    scale,
    bias,
    running_mean,
    running_var,
    eps,
    momentum,
    cudnn_handle,
    peer_stats=None,
):
    graph = cudnn.pygraph(
        io_data_type=cudnn_data_type(x.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )
    x_tensor = graph.tensor_like(x)
    scale_tensor = graph.tensor_like(scale)
    bias_tensor = graph.tensor_like(bias)
    running_mean_tensor = graph.tensor_like(running_mean)
    running_var_tensor = graph.tensor_like(running_var)
    eps_tensor = _scalar_tensor(graph, x.dim(), "epsilon")
    momentum_tensor = _scalar_tensor(graph, x.dim(), "momentum")
    peer_tensors = []
    exec_tensors = {
        x_tensor: x,
        scale_tensor: scale,
        bias_tensor: bias,
        running_mean_tensor: running_mean,
        running_var_tensor: running_var,
        eps_tensor: _scalar_value(eps, x.dim()),
        momentum_tensor: _scalar_value(momentum, x.dim()),
    }
    if peer_stats is not None:
        peer_tensor = graph.tensor_like(peer_stats)
        peer_tensors.append(peer_tensor)
        exec_tensors[peer_tensor] = peer_stats

    outputs = graph.batchnorm(
        x_tensor,
        scale_tensor,
        bias_tensor,
        running_mean_tensor,
        running_var_tensor,
        eps_tensor,
        momentum_tensor,
        peer_stats=peer_tensors,
        compute_data_type=cudnn.data_type.FLOAT,
        name="batchnorm",
    )
    (
        y_tensor,
        mean_tensor,
        inv_var_tensor,
        next_mean_tensor,
        next_var_tensor,
    ) = outputs
    stat_shape = tuple(running_mean.shape)
    return _execute_cudnn_outputs(
        graph,
        exec_tensors,
        (
            (y_tensor, tuple(x.shape), x.dtype),
            (mean_tensor, stat_shape, torch.float32),
            (inv_var_tensor, stat_shape, torch.float32),
            (next_mean_tensor, stat_shape, torch.float32),
            (next_var_tensor, stat_shape, torch.float32),
        ),
        x.device,
        cudnn_handle,
        "batchnorm",
    )


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


def _run_batchnorm_peer_stats_graph(
    x, scale, bias, running_mean, running_var, peer_stats, eps, momentum
):
    @flag_dnn.graph
    def fn(x, scale, bias, running_mean, running_var, peer_stats):
        return flag_dnn.batchnorm(
            x,
            scale,
            bias,
            running_mean,
            running_var,
            eps,
            momentum,
            peer_stats=[peer_stats],
        )

    compiled = flag_dnn.compile(
        fn,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(x, "x"),
            flag_dnn.TensorSpec.from_tensor(scale, "scale"),
            flag_dnn.TensorSpec.from_tensor(bias, "bias"),
            flag_dnn.TensorSpec.from_tensor(running_mean, "running_mean"),
            flag_dnn.TensorSpec.from_tensor(running_var, "running_var"),
            flag_dnn.TensorSpec.from_tensor(peer_stats, "peer_stats"),
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
        peer_stats.clone(),
    )


def _assert_outputs_close(actual, cudnn_out, dtype, atol):
    utils.gems_assert_close(actual[0], cudnn_out[0], dtype, atol=atol)
    for actual_value, cudnn_value in zip(actual[1:], cudnn_out[1:]):
        torch.testing.assert_close(
            actual_value, cudnn_value, atol=atol, rtol=atol
        )


@pytest.mark.layernorm
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", COMPARE_DTYPES)
def test_layernorm_multi_output_matches_cudnn(cudnn_handle, dtype):
    torch.manual_seed(0)
    eps = 1e-3
    x = torch.randn((2, 5, 17), device=flag_dnn.device, dtype=dtype)
    scale = torch.randn((1, 1, 17), device=flag_dnn.device, dtype=dtype)
    bias = torch.randn((1, 1, 17), device=flag_dnn.device, dtype=dtype)
    cudnn_out = _cudnn_layernorm(x, scale, bias, eps, cudnn_handle)
    actual = _run_layernorm_graph(x, scale, bias, eps)
    atol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-4
    _assert_outputs_close(actual, cudnn_out, dtype, atol)


@pytest.mark.rmsnorm
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", COMPARE_DTYPES)
def test_rmsnorm_multi_output_matches_cudnn(cudnn_handle, dtype):
    torch.manual_seed(1)
    eps = 1e-3
    x = torch.randn((2, 5, 17), device=flag_dnn.device, dtype=dtype)
    scale = torch.randn((1, 1, 17), device=flag_dnn.device, dtype=dtype)
    bias = torch.randn((1, 1, 17), device=flag_dnn.device, dtype=dtype)
    cudnn_out = _cudnn_rmsnorm(x, scale, bias, eps, cudnn_handle)
    actual = _run_rmsnorm_graph(x, scale, bias, eps)
    atol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-4
    _assert_outputs_close(actual, cudnn_out, dtype, atol)


@pytest.mark.batchnorm
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", COMPARE_DTYPES)
def test_batchnorm_multi_output_matches_cudnn(cudnn_handle, dtype):
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
    cudnn_out = _cudnn_batchnorm(
        x, scale, bias, running_mean, running_var, eps, momentum, cudnn_handle
    )
    actual = _run_batchnorm_graph(
        x, scale, bias, running_mean, running_var, eps, momentum
    )
    atol = 3e-2 if dtype in (torch.float16, torch.bfloat16) else 3e-4
    _assert_outputs_close(actual, cudnn_out, dtype, atol)


@pytest.mark.batchnorm
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", COMPARE_DTYPES)
def test_batchnorm_single_peer_stats_matches_cudnn(cudnn_handle, dtype):
    torch.manual_seed(7)
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
    peer_stats = torch.empty(
        (16,), device=flag_dnn.device, dtype=torch.float32
    )
    cudnn_out = _cudnn_batchnorm(
        x,
        scale,
        bias,
        running_mean,
        running_var,
        eps,
        momentum,
        cudnn_handle,
        peer_stats=peer_stats,
    )
    actual = _run_batchnorm_peer_stats_graph(
        x,
        scale,
        bias,
        running_mean,
        running_var,
        peer_stats,
        eps,
        momentum,
    )
    atol = 3e-2 if dtype in (torch.float16, torch.bfloat16) else 3e-4
    _assert_outputs_close(actual, cudnn_out, dtype, atol)
