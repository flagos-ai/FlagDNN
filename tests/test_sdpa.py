# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import pytest
from tests.base import (
    CUDNN_COMPARE_DTYPES,
    cudnn,
    cudnn_data_type,
    skip_unsupported_cudnn_graph,
)
import torch

import flag_dnn
from tests import accuracy_utils as utils
from tests import consts


def _cudnn_alignment(diagonal_alignment):
    if diagonal_alignment == "BOTTOM_RIGHT":
        return cudnn.diagonal_alignment.BOTTOM_RIGHT
    return cudnn.diagonal_alignment.TOP_LEFT


def _cudnn_sdpa(
    q,
    k,
    v,
    cudnn_handle,
    attn_scale=None,
    bias=None,
    diagonal_alignment="TOP_LEFT",
    left_bound=None,
    right_bound=None,
    generate_stats=True,
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
        generate_stats=generate_stats,
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
    if generate_stats:
        stats_tensor.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    try:
        graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as exc:
        skip_unsupported_cudnn_graph(exc, "sdpa")

    o = torch.empty(
        (batch, heads, seq_q, v_dim), device=q.device, dtype=q.dtype
    )
    exec_tensors = {q_tensor: q, k_tensor: k, v_tensor: v, o_tensor: o}
    stats = None
    if generate_stats:
        stats = torch.empty_strided(
            tuple(stats_tensor.get_dim()),
            tuple(stats_tensor.get_stride()),
            device=q.device,
            dtype=torch.float32,
        )
        exec_tensors[stats_tensor] = stats
    if bias is not None:
        exec_tensors[bias_tensor] = bias
    workspace = torch.empty(
        graph.get_workspace_size(), device=q.device, dtype=torch.uint8
    )
    graph.execute(exec_tensors, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()
    if generate_stats:
        return o, stats
    return o


def _run_flag_dnn_sdpa_graph(q, k, v, bias=None, **op_kwargs):
    if bias is None:

        @flag_dnn.graph
        def fn(q, k, v):
            return flag_dnn.sdpa(q, k, v, name="sdpa", **op_kwargs)

        specs = [
            flag_dnn.TensorSpec.from_tensor(q, "q"),
            flag_dnn.TensorSpec.from_tensor(k, "k"),
            flag_dnn.TensorSpec.from_tensor(v, "v"),
        ]
        run_args = (q.clone(), k.clone(), v.clone())
    else:

        @flag_dnn.graph
        def fn(q, k, v, bias):
            return flag_dnn.sdpa(q, k, v, bias=bias, name="sdpa", **op_kwargs)

        specs = [
            flag_dnn.TensorSpec.from_tensor(q, "q"),
            flag_dnn.TensorSpec.from_tensor(k, "k"),
            flag_dnn.TensorSpec.from_tensor(v, "v"),
            flag_dnn.TensorSpec.from_tensor(bias, "bias"),
        ]
        run_args = (q.clone(), k.clone(), v.clone(), bias.clone())

    compiled = flag_dnn.compile(fn, inputs=specs, options={"cache": None})
    assert [node.op_type for node in compiled.graph.nodes] == ["sdpa"]
    return compiled.run(*run_args)


def _make_qkv(shape, dtype):
    batch, heads_q, heads_kv, seq_q, seq_kv, head_dim = shape
    q = torch.randn(
        (batch, heads_q, seq_q, head_dim), dtype=dtype, device=flag_dnn.device
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


def _output_atol(dtype):
    return 1e-1 if dtype == torch.bfloat16 else 5e-2


def _stats_tol(dtype):
    return 5e-3 if dtype == torch.float32 else 2e-2


def _assert_sdpa_close(flag_out, cudnn_out, dtype):
    flag_o, flag_stats = flag_out
    cudnn_o, cudnn_stats = cudnn_out
    utils.gems_assert_close(flag_o, cudnn_o, dtype, atol=_output_atol(dtype))
    tol = _stats_tol(dtype)
    torch.testing.assert_close(flag_stats, cudnn_stats, atol=tol, rtol=tol)


@pytest.mark.sdpa
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("shape", consts.SDPA_CASES)
def test_sdpa_default_scale_with_stats(cudnn_handle, dtype, shape):
    torch.manual_seed(0)
    q, k, v = _make_qkv(shape, dtype)
    cudnn_out = _cudnn_sdpa(q, k, v, cudnn_handle)
    flag_out = _run_flag_dnn_sdpa_graph(q, k, v, generate_stats=True)
    _assert_sdpa_close(flag_out, cudnn_out, dtype)


@pytest.mark.sdpa
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("shape", consts.SDPA_CASES)
def test_sdpa_inference(cudnn_handle, dtype, shape):
    torch.manual_seed(1)
    q, k, v = _make_qkv(shape, dtype)
    cudnn_o = _cudnn_sdpa(q, k, v, cudnn_handle, generate_stats=False)
    flag_o = _run_flag_dnn_sdpa_graph(q, k, v)
    assert isinstance(flag_o, torch.Tensor)
    utils.gems_assert_close(flag_o, cudnn_o, dtype, atol=_output_atol(dtype))


@pytest.mark.sdpa
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_sdpa_gqa_decode_low_precision(cudnn_handle, dtype):
    torch.manual_seed(8)
    shape = (2, 8, 2, 1, 512, 64)
    q, k, v = _make_qkv(shape, dtype)
    cudnn_out = _cudnn_sdpa(q, k, v, cudnn_handle, generate_stats=True)
    flag_out = _run_flag_dnn_sdpa_graph(q, k, v, generate_stats=True)
    _assert_sdpa_close(flag_out, cudnn_out, dtype)
    cudnn_o = _cudnn_sdpa(q, k, v, cudnn_handle, generate_stats=False)
    flag_o = _run_flag_dnn_sdpa_graph(q, k, v)
    utils.gems_assert_close(flag_o, cudnn_o, dtype, atol=_output_atol(dtype))


@pytest.mark.sdpa
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_sdpa_gqa_causal_d128_low_precision(cudnn_handle, dtype):
    torch.manual_seed(9)
    shape = (1, 8, 2, 128, 128, 128)
    q, k, v = _make_qkv(shape, dtype)
    cudnn_out = _cudnn_sdpa(
        q, k, v, cudnn_handle, right_bound=0, generate_stats=True
    )
    flag_out = _run_flag_dnn_sdpa_graph(
        q, k, v, diagonal_band_right_bound=0, generate_stats=True
    )
    _assert_sdpa_close(flag_out, cudnn_out, dtype)


@pytest.mark.sdpa
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_sdpa_long_causal_host_k_descriptor_rebinds_storage(
    cudnn_handle, monkeypatch, dtype
):
    from triton.tools import tensor_descriptor as descriptor_module

    real_descriptor = descriptor_module.TensorDescriptor
    descriptor_ptrs = []

    def recording_descriptor(base, desc_shape, strides, block_shape):
        descriptor_ptrs.append(base.data_ptr())
        return real_descriptor(base, desc_shape, strides, block_shape)

    monkeypatch.setattr(
        descriptor_module, "TensorDescriptor", recording_descriptor
    )
    torch.manual_seed(17)
    shape = (2, 16, 16, 2048, 2048, 128)
    q_a, k_a, v_a = _make_qkv(shape, dtype)

    @flag_dnn.graph
    def fn(q, k, v):
        return flag_dnn.sdpa(
            q,
            k,
            v,
            diagonal_band_right_bound=0,
            generate_stats=True,
            name="sdpa",
        )

    compiled = flag_dnn.compile(
        fn,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(q_a, "q"),
            flag_dnn.TensorSpec.from_tensor(k_a, "k"),
            flag_dnn.TensorSpec.from_tensor(v_a, "v"),
        ],
        options={"cache": None},
    )

    def check(q, k, v):
        expected = _cudnn_sdpa(
            q, k, v, cudnn_handle, right_bound=0, generate_stats=True
        )
        actual = compiled.run(q, k, v)
        _assert_sdpa_close(actual, expected, dtype)
        return actual

    out_a = check(q_a, k_a, v_a)
    q_b, k_b, v_b = _make_qkv(shape, dtype)
    out_b = check(q_b, k_b, v_b)
    assert descriptor_ptrs == [k_a.data_ptr(), k_b.data_ptr()]
    assert out_a[0].data_ptr() != out_b[0].data_ptr()

    q_b.normal_()
    k_b.normal_()
    v_b.normal_()
    out_c = check(q_b, k_b, v_b)
    assert descriptor_ptrs == [k_a.data_ptr(), k_b.data_ptr()]
    assert out_b[0].data_ptr() != out_c[0].data_ptr()


@pytest.mark.sdpa
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("shape", consts.SDPA_MASKED_CASES)
def test_sdpa_causal(cudnn_handle, dtype, shape):
    torch.manual_seed(2)
    q, k, v = _make_qkv(shape, dtype)
    cudnn_out = _cudnn_sdpa(q, k, v, cudnn_handle, right_bound=0)
    flag_out = _run_flag_dnn_sdpa_graph(
        q, k, v, use_causal_mask=True, generate_stats=True
    )
    _assert_sdpa_close(flag_out, cudnn_out, dtype)


@pytest.mark.sdpa
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("shape", consts.SDPA_MASKED_CASES)
def test_sdpa_causal_bottom_right(cudnn_handle, dtype, shape):
    torch.manual_seed(3)
    batch, heads_q, heads_kv, seq_q, seq_kv, head_dim = shape
    if seq_q > seq_kv:
        pytest.skip("bottom-right causal requires seq_q <= seq_kv")
    q, k, v = _make_qkv(shape, dtype)
    cudnn_out = _cudnn_sdpa(
        q,
        k,
        v,
        cudnn_handle,
        diagonal_alignment="BOTTOM_RIGHT",
        right_bound=0,
    )
    flag_out = _run_flag_dnn_sdpa_graph(
        q,
        k,
        v,
        diagonal_alignment="BOTTOM_RIGHT",
        diagonal_band_right_bound=0,
        generate_stats=True,
    )
    _assert_sdpa_close(flag_out, cudnn_out, dtype)


@pytest.mark.sdpa
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("shape", consts.SDPA_MASKED_CASES)
def test_sdpa_sliding_window(cudnn_handle, dtype, shape):
    torch.manual_seed(4)
    q, k, v = _make_qkv(shape, dtype)
    cudnn_out = _cudnn_sdpa(
        q, k, v, cudnn_handle, left_bound=32, right_bound=0
    )
    flag_out = _run_flag_dnn_sdpa_graph(
        q,
        k,
        v,
        diagonal_band_left_bound=32,
        diagonal_band_right_bound=0,
        generate_stats=True,
    )
    _assert_sdpa_close(flag_out, cudnn_out, dtype)


@pytest.mark.sdpa
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("shape", consts.SDPA_MASKED_CASES)
def test_sdpa_bias(cudnn_handle, dtype, shape):
    if dtype == torch.float32:
        pytest.skip("cuDNN fp32 sdpa with bias produces NaN on this backend")
    torch.manual_seed(5)
    batch, heads_q, heads_kv, seq_q, seq_kv, head_dim = shape
    q, k, v = _make_qkv(shape, dtype)
    bias = torch.randn(
        (batch, 1, seq_q, seq_kv), dtype=dtype, device=flag_dnn.device
    )
    cudnn_out = _cudnn_sdpa(q, k, v, cudnn_handle, bias=bias)
    flag_out = _run_flag_dnn_sdpa_graph(
        q, k, v, bias=bias, generate_stats=True
    )
    _assert_sdpa_close(flag_out, cudnn_out, dtype)


@pytest.mark.sdpa
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("shape", consts.SDPA_MASKED_CASES)
def test_sdpa_bias_float32(shape):
    del shape
    pytest.skip("cuDNN fp32 sdpa with bias produces NaN on this backend")


@pytest.mark.sdpa
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
def test_sdpa_explicit_scale(cudnn_handle, dtype):
    torch.manual_seed(6)
    shape = (2, 8, 8, 128, 128, 64)
    q, k, v = _make_qkv(shape, dtype)
    cudnn_out = _cudnn_sdpa(q, k, v, cudnn_handle, attn_scale=0.5)
    flag_out = _run_flag_dnn_sdpa_graph(
        q, k, v, attn_scale=0.5, generate_stats=True
    )
    _assert_sdpa_close(flag_out, cudnn_out, dtype)


@pytest.mark.sdpa
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", (torch.float16,))
def test_sdpa_causal_shorthand_and_band_match_cudnn(cudnn_handle, dtype):
    torch.manual_seed(7)
    shape = (2, 4, 4, 100, 256, 64)
    q, k, v = _make_qkv(shape, dtype)
    cudnn_out = _cudnn_sdpa(q, k, v, cudnn_handle, right_bound=0)
    shorthand_out = _run_flag_dnn_sdpa_graph(
        q, k, v, use_causal_mask=True, generate_stats=True
    )
    band_out = _run_flag_dnn_sdpa_graph(
        q,
        k,
        v,
        diagonal_alignment="TOP_LEFT",
        diagonal_band_right_bound=0,
        generate_stats=True,
    )
    _assert_sdpa_close(shorthand_out, cudnn_out, dtype)
    _assert_sdpa_close(band_out, cudnn_out, dtype)
