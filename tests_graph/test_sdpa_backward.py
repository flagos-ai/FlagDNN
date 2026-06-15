import math

import pytest
from tests_graph.base import (
    CUDNN_COMPARE_DTYPES,
    cudnn,
    cudnn_data_type,
    skip_unsupported_cudnn_graph,
)
import torch

import flag_dnn


SDPA_BACKWARD_CASES = (
    (1, 2, 2, 32, 32, 32),
    (2, 4, 4, 48, 40, 64),
    (1, 4, 2, 32, 32, 64),
    (1, 2, 2, 1, 96, 64),
)

SDPA_BACKWARD_MASKED_CASES = (
    (1, 2, 2, 32, 32, 32),
    (2, 4, 4, 48, 64, 64),
)


def _cudnn_alignment(diagonal_alignment):
    if diagonal_alignment == "BOTTOM_RIGHT":
        return cudnn.diagonal_alignment.BOTTOM_RIGHT
    return cudnn.diagonal_alignment.TOP_LEFT


def _make_qkv(shape, dtype):
    batch, heads_q, heads_kv, seq_q, seq_kv, head_dim = shape
    q = torch.randn(
        (batch, heads_q, seq_q, head_dim),
        dtype=dtype,
        device=flag_dnn.device,
    )
    k = torch.randn(
        (batch, heads_kv, seq_kv, head_dim),
        dtype=dtype,
        device=flag_dnn.device,
    )
    v = torch.randn(
        (batch, heads_kv, seq_kv, head_dim),
        dtype=dtype,
        device=flag_dnn.device,
    )
    return q, k, v


def _cudnn_sdpa_forward(
    q,
    k,
    v,
    cudnn_handle,
    *,
    attn_scale=None,
    bias=None,
    diagonal_alignment="TOP_LEFT",
    left_bound=None,
    right_bound=None,
):
    graph = cudnn.pygraph(
        io_data_type=cudnn_data_type(q.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )
    q_tensor = graph.tensor_like(q)
    k_tensor = graph.tensor_like(k)
    v_tensor = graph.tensor_like(v)
    kwargs = dict(
        attn_scale=(
            attn_scale
            if attn_scale is not None
            else 1.0 / math.sqrt(q.shape[-1])
        ),
        generate_stats=True,
        diagonal_alignment=_cudnn_alignment(diagonal_alignment),
        name="sdpa",
    )
    if left_bound is not None:
        kwargs["diagonal_band_left_bound"] = left_bound
    if right_bound is not None:
        kwargs["diagonal_band_right_bound"] = right_bound
    bias_tensor = None
    if bias is not None:
        bias_tensor = graph.tensor_like(bias)
        kwargs["bias"] = bias_tensor

    o_tensor, stats_tensor = graph.sdpa(q_tensor, k_tensor, v_tensor, **kwargs)
    batch, heads, seq_q = q.shape[0], q.shape[1], q.shape[2]
    v_dim = v.shape[-1]
    o_tensor.set_output(True).set_data_type(cudnn_data_type(q.dtype))
    o_tensor.set_dim([batch, heads, seq_q, v_dim]).set_stride(
        [heads * seq_q * v_dim, seq_q * v_dim, v_dim, 1]
    )
    stats_tensor.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    try:
        graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as exc:
        skip_unsupported_cudnn_graph(exc, "sdpa")

    o = torch.empty(
        (batch, heads, seq_q, v_dim), device=q.device, dtype=q.dtype
    )
    stats = torch.empty_strided(
        tuple(stats_tensor.get_dim()),
        tuple(stats_tensor.get_stride()),
        device=q.device,
        dtype=torch.float32,
    )
    exec_tensors = {
        q_tensor: q,
        k_tensor: k,
        v_tensor: v,
        o_tensor: o,
        stats_tensor: stats,
    }
    if bias is not None:
        exec_tensors[bias_tensor] = bias
    workspace = torch.empty(
        graph.get_workspace_size(), device=q.device, dtype=torch.uint8
    )
    graph.execute(exec_tensors, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()
    return o, stats


def _cudnn_sdpa_backward(
    q,
    k,
    v,
    o,
    dO,
    stats,
    cudnn_handle,
    *,
    attn_scale=None,
    bias=None,
    dBias=None,
    diagonal_alignment="TOP_LEFT",
    left_bound=None,
    right_bound=None,
):
    graph = cudnn.pygraph(
        io_data_type=cudnn_data_type(q.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )
    q_tensor = graph.tensor_like(q)
    k_tensor = graph.tensor_like(k)
    v_tensor = graph.tensor_like(v)
    o_tensor = graph.tensor_like(o)
    do_tensor = graph.tensor_like(dO)
    stats_tensor = graph.tensor_like(stats)
    kwargs = dict(
        attn_scale=(
            attn_scale
            if attn_scale is not None
            else 1.0 / math.sqrt(q.shape[-1])
        ),
        diagonal_alignment=_cudnn_alignment(diagonal_alignment),
        name="sdpa_backward",
    )
    if left_bound is not None:
        kwargs["diagonal_band_left_bound"] = left_bound
    if right_bound is not None:
        kwargs["diagonal_band_right_bound"] = right_bound
    bias_tensor = None
    if bias is not None:
        bias_tensor = graph.tensor_like(bias)
        kwargs["bias"] = bias_tensor
    dbias_tensor = None
    if dBias is not None:
        dbias_tensor = graph.tensor_like(dBias)
        kwargs["dBias"] = dbias_tensor

    dq_tensor, dk_tensor, dv_tensor = graph.sdpa_backward(
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        do_tensor,
        stats_tensor,
        **kwargs,
    )
    for tensor, ref in (
        (dq_tensor, q),
        (dk_tensor, k),
        (dv_tensor, v),
    ):
        tensor.set_output(True).set_data_type(cudnn_data_type(ref.dtype))
        tensor.set_dim(list(ref.shape)).set_stride(list(ref.stride()))

    try:
        graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as exc:
        skip_unsupported_cudnn_graph(exc, "sdpa_backward")

    dQ = torch.empty_like(q)
    dK = torch.empty_like(k)
    dV = torch.empty_like(v)
    exec_tensors = {
        q_tensor: q,
        k_tensor: k,
        v_tensor: v,
        o_tensor: o,
        do_tensor: dO,
        stats_tensor: stats,
        dq_tensor: dQ,
        dk_tensor: dK,
        dv_tensor: dV,
    }
    if bias is not None:
        exec_tensors[bias_tensor] = bias
    if dBias is not None:
        exec_tensors[dbias_tensor] = dBias
    workspace = torch.empty(
        graph.get_workspace_size(), device=q.device, dtype=torch.uint8
    )
    graph.execute(exec_tensors, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()
    return dQ, dK, dV


def _run_flag_dnn_sdpa_backward_graph(
    q,
    k,
    v,
    o,
    dO,
    stats,
    *,
    attn_scale=None,
    bias=None,
    dBias=None,
    diagonal_alignment="TOP_LEFT",
    left_bound=None,
    right_bound=None,
):
    op_kwargs = dict(
        attn_scale=attn_scale,
        diagonal_alignment=diagonal_alignment,
        diagonal_band_left_bound=left_bound,
        diagonal_band_right_bound=right_bound,
        name="sdpa_backward",
    )
    specs = [
        flag_dnn.TensorSpec.from_tensor(q, "q"),
        flag_dnn.TensorSpec.from_tensor(k, "k"),
        flag_dnn.TensorSpec.from_tensor(v, "v"),
        flag_dnn.TensorSpec.from_tensor(o, "o"),
        flag_dnn.TensorSpec.from_tensor(dO, "dO"),
        flag_dnn.TensorSpec.from_tensor(stats, "stats"),
    ]
    run_args = [q.clone(), k.clone(), v.clone(), o.clone(), dO.clone(), stats]
    if bias is None and dBias is None:

        @flag_dnn.graph
        def fn(q, k, v, o, dO, stats):
            return flag_dnn.sdpa_backward(q, k, v, o, dO, stats, **op_kwargs)

    elif dBias is None:

        @flag_dnn.graph
        def fn(q, k, v, o, dO, stats, bias):
            return flag_dnn.sdpa_backward(
                q, k, v, o, dO, stats, bias=bias, **op_kwargs
            )

        specs.append(flag_dnn.TensorSpec.from_tensor(bias, "bias"))
        run_args.append(bias.clone())
    elif bias is None:

        @flag_dnn.graph
        def fn(q, k, v, o, dO, stats, dBias):
            return flag_dnn.sdpa_backward(
                q, k, v, o, dO, stats, dBias=dBias, **op_kwargs
            )

        specs.append(flag_dnn.TensorSpec.from_tensor(dBias, "dBias"))
        run_args.append(dBias)
    else:

        @flag_dnn.graph
        def fn(q, k, v, o, dO, stats, bias, dBias):
            return flag_dnn.sdpa_backward(
                q,
                k,
                v,
                o,
                dO,
                stats,
                bias=bias,
                dBias=dBias,
                **op_kwargs,
            )

        specs.append(flag_dnn.TensorSpec.from_tensor(bias, "bias"))
        specs.append(flag_dnn.TensorSpec.from_tensor(dBias, "dBias"))
        run_args.append(bias.clone())
        run_args.append(dBias)

    compiled = flag_dnn.compile(fn, inputs=specs, options={"cache": None})
    assert [node.op_type for node in compiled.graph.nodes] == ["sdpa_backward"]
    return compiled.run(*run_args)


def _grad_tol(dtype):
    if dtype == torch.bfloat16:
        return 8e-2, 3e-2
    if dtype == torch.float32:
        return 3e-2, 3e-2
    return 4e-2, 2e-2


def _assert_grads_close(flag_out, cudnn_out, dtype):
    atol, rtol = _grad_tol(dtype)
    for actual, expected in zip(flag_out, cudnn_out):
        torch.testing.assert_close(
            actual, expected, atol=atol, rtol=rtol, check_dtype=False
        )


@pytest.mark.sdpa_backward
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("shape", SDPA_BACKWARD_CASES)
def test_sdpa_backward(cudnn_handle, dtype, shape):
    torch.manual_seed(0)
    q, k, v = _make_qkv(shape, dtype)
    dO = torch.randn(
        (shape[0], shape[1], shape[3], shape[5]),
        dtype=dtype,
        device=flag_dnn.device,
    )
    o, stats = _cudnn_sdpa_forward(q, k, v, cudnn_handle)
    cudnn_out = _cudnn_sdpa_backward(q, k, v, o, dO, stats, cudnn_handle)
    flag_out = _run_flag_dnn_sdpa_backward_graph(q, k, v, o, dO, stats)
    _assert_grads_close(flag_out, cudnn_out, dtype)


@pytest.mark.sdpa_backward
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("shape", SDPA_BACKWARD_MASKED_CASES)
def test_sdpa_backward_causal(cudnn_handle, dtype, shape):
    torch.manual_seed(1)
    q, k, v = _make_qkv(shape, dtype)
    dO = torch.randn(
        (shape[0], shape[1], shape[3], shape[5]),
        dtype=dtype,
        device=flag_dnn.device,
    )
    o, stats = _cudnn_sdpa_forward(q, k, v, cudnn_handle, right_bound=0)
    cudnn_out = _cudnn_sdpa_backward(
        q, k, v, o, dO, stats, cudnn_handle, right_bound=0
    )
    flag_out = _run_flag_dnn_sdpa_backward_graph(
        q, k, v, o, dO, stats, right_bound=0
    )
    _assert_grads_close(flag_out, cudnn_out, dtype)


@pytest.mark.sdpa_backward
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_sdpa_backward_bias_dbias(cudnn_handle, dtype):
    torch.manual_seed(2)
    shape = (2, 4, 4, 32, 40, 64)
    q, k, v = _make_qkv(shape, dtype)
    dO = torch.randn(
        (shape[0], shape[1], shape[3], shape[5]),
        dtype=dtype,
        device=flag_dnn.device,
    )
    bias = torch.randn(
        (1, shape[1], shape[3], shape[4]),
        dtype=dtype,
        device=flag_dnn.device,
    )
    cudnn_dbias = torch.empty_like(bias)
    flag_dbias = torch.empty_like(bias)
    o, stats = _cudnn_sdpa_forward(q, k, v, cudnn_handle, bias=bias)
    cudnn_out = _cudnn_sdpa_backward(
        q,
        k,
        v,
        o,
        dO,
        stats,
        cudnn_handle,
        bias=bias,
        dBias=cudnn_dbias,
    )
    flag_out = _run_flag_dnn_sdpa_backward_graph(
        q, k, v, o, dO, stats, bias=bias, dBias=flag_dbias
    )
    _assert_grads_close(flag_out, cudnn_out, dtype)
    atol, rtol = _grad_tol(dtype)
    torch.testing.assert_close(
        flag_dbias, cudnn_dbias, atol=atol, rtol=rtol, check_dtype=False
    )
