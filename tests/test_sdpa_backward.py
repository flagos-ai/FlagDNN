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
from devtools.dnn_reference.interfaces import DnnReferenceNotSupportedError
from devtools.dnn_reference.providers.nvidia_ops.common import (
    require_cudnn_sdpa_execution_supported,
)

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


SDPA_BACKWARD_LONG_CAUSAL_D128_CASES = (
    (2, 16, 16, 2048, 2048, 128),
    (1, 32, 8, 4096, 4096, 128),
)


@pytest.mark.parametrize(
    "vendor,capability,expected",
    (
        ("nvidia", (8, 0), True),
        ("nvidia", (9, 0), False),
        ("nvidia", (10, 0), True),
        ("nvidia", (0, 0), False),
        ("ascend", (9, 0), True),
        ("ascend", (0, 0), True),
    ),
)
def test_mloop_device_guard_fails_closed(
    monkeypatch, vendor, capability, expected
):
    from flag_dnn.graph.prepared import sdpa_backward as prepared_module

    monkeypatch.setattr(prepared_module.runtime.device, "vendor_name", vendor)
    monkeypatch.setattr(
        prepared_module,
        "get_device_capability_for",
        lambda _device: capability,
    )
    q = torch.empty(1)
    assert prepared_module._mloop_supported_on_device((q,)) is expected
    assert not prepared_module._mloop_supported_on_device(())


@pytest.mark.parametrize(
    "op_name",
    (
        "sdpa_backward_fused_atomic_gqa_causal_d128",
        "sdpa_backward_gqa_dq_delta_d128",
        "sdpa_backward_owner_mha_causal_d128",
    ),
)
@pytest.mark.parametrize(
    "capability,expected_stages",
    (
        ((8, 0), 2),
        ((9, 0), 1),
        ((10, 0), 2),
        ((0, 0), 2),
    ),
)
def test_hopper_sensitive_tuned_config_uses_explicit_input_device(
    monkeypatch, op_name, capability, expected_stages
):
    import importlib

    sdpa_backward_module = importlib.import_module(
        "flag_dnn.ops.sdpa_backward"
    )

    class FakeConfig:
        kwargs = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 128}
        num_warps = 4
        num_stages = 2

    monkeypatch.setattr(
        sdpa_backward_module.runtime,
        "get_tuned_config",
        lambda _op_name: [FakeConfig()],
    )
    monkeypatch.setattr(
        sdpa_backward_module,
        "get_device_capability_for",
        lambda device: capability if str(device) == "cuda:1" else (8, 0),
    )

    config = sdpa_backward_module._single_tuned_config_kwargs(
        op_name, device=torch.device("cuda:1")
    )

    assert config["num_stages"] == expected_stages


@pytest.mark.parametrize(
    "capability,num_stages,expected",
    (
        ((8, 0), 2, True),
        ((9, 0), 1, True),
        ((9, 0), 2, False),
        ((0, 0), 1, False),
    ),
)
def test_hopper_sensitive_tuned_config_guard_fails_closed(
    monkeypatch, capability, num_stages, expected
):
    import importlib

    sdpa_backward_module = importlib.import_module(
        "flag_dnn.ops.sdpa_backward"
    )
    monkeypatch.setattr(
        sdpa_backward_module,
        "get_device_capability_for",
        lambda _device: capability,
    )
    config = {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_D": 128,
        "num_warps": 4,
        "num_stages": num_stages,
    }

    assert (
        sdpa_backward_module._tuned_config_supported_on_device(
            "sdpa_backward_fused_atomic_gqa_causal_d128",
            config,
            torch.device("cuda:1"),
        )
        is expected
    )


def test_hopper_tuned_config_override_does_not_change_other_vendors(
    monkeypatch,
):
    import importlib

    sdpa_backward_module = importlib.import_module(
        "flag_dnn.ops.sdpa_backward"
    )

    class FakeConfig:
        kwargs = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 128}
        num_warps = 4
        num_stages = 2

    monkeypatch.setattr(
        sdpa_backward_module.runtime,
        "get_tuned_config",
        lambda _op_name: [FakeConfig()],
    )
    monkeypatch.setattr(
        sdpa_backward_module.runtime.device, "vendor_name", "ascend"
    )

    config = sdpa_backward_module._single_tuned_config_kwargs(
        "sdpa_backward_fused_atomic_gqa_causal_d128",
        device=torch.device("cuda:1"),
    )

    assert config["num_stages"] == 2


@pytest.mark.parametrize(
    "op_name",
    (
        "sdpa_backward_fused_atomic_gqa_causal_d128",
        "sdpa_backward_gqa_dq_delta_d128",
        "sdpa_backward_owner_mha_causal_d128",
    ),
)
def test_hopper_loaded_config_restores_generic_signature_on_sm80(
    monkeypatch, op_name
):
    import importlib

    sdpa_backward_module = importlib.import_module(
        "flag_dnn.ops.sdpa_backward"
    )
    safe = sdpa_backward_module._SM90_SAFE_TUNED_CONFIGS[op_name]

    class HopperConfig:
        kwargs = {
            key: value
            for key, value in safe.items()
            if not key.startswith("num_")
        }
        num_warps = safe["num_warps"]
        num_stages = safe["num_stages"]
        num_ctas = 1

    monkeypatch.setattr(
        sdpa_backward_module.runtime,
        "get_tuned_config",
        lambda _op_name: [HopperConfig()],
    )
    monkeypatch.setattr(
        sdpa_backward_module,
        "get_device_capability_for",
        lambda _device: (8, 0),
    )

    config = sdpa_backward_module._single_tuned_config_kwargs(
        op_name, device=torch.device("cuda:1")
    )

    assert (
        config == sdpa_backward_module._NVIDIA_DEFAULT_TUNED_CONFIGS[op_name]
    )


def test_owner_compute_causal_d128_eligibility():
    from flag_dnn.graph.prepared.sdpa_backward import (
        _is_owner_compute_causal_d128,
    )

    def contiguous(b, h, s, d):
        return (h * s * d, s * d, d, 1)

    def stats_stride(h, s):
        return (h * s, s, 1, 1)

    for shape in SDPA_BACKWARD_LONG_CAUSAL_D128_CASES:
        b, hq, hkv, sq, skv, d = shape
        q_shape = (b, hq, sq, d)
        kv_shape = (b, hkv, skv, d)
        s_shape = (b, hq, sq, 1)
        owner_eligible = _is_owner_compute_causal_d128(
            q_shape,
            kv_shape,
            kv_shape,
            q_shape,
            q_shape,
            s_shape,
            contiguous(*q_shape),
            contiguous(*kv_shape),
            contiguous(*kv_shape),
            contiguous(*q_shape),
            contiguous(*q_shape),
            stats_stride(hq, sq),
            torch.float16,
            causal_top_left=True,
        )
        assert owner_eligible == (hq == hkv)
        assert not _is_owner_compute_causal_d128(
            q_shape,
            kv_shape,
            kv_shape,
            q_shape,
            q_shape,
            s_shape,
            contiguous(*q_shape),
            contiguous(*kv_shape),
            contiguous(*kv_shape),
            contiguous(*q_shape),
            contiguous(*q_shape),
            stats_stride(hq, sq),
            torch.float32,
            causal_top_left=True,
        )
        assert not _is_owner_compute_causal_d128(
            q_shape,
            kv_shape,
            kv_shape,
            q_shape,
            q_shape,
            s_shape,
            contiguous(*q_shape),
            contiguous(*kv_shape),
            contiguous(*kv_shape),
            contiguous(*q_shape),
            contiguous(*q_shape),
            stats_stride(hq, sq),
            torch.float16,
            causal_top_left=False,
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


def _skip_unsupported_cudnn_sdpa_execution(tensor, op_name):
    try:
        require_cudnn_sdpa_execution_supported(tensor, op_name)
    except DnnReferenceNotSupportedError as exc:
        pytest.skip(str(exc))


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
    _skip_unsupported_cudnn_sdpa_execution(q, "sdpa")
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
    _skip_unsupported_cudnn_sdpa_execution(q, "sdpa_backward")
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


def _compile_flag_dnn_sdpa_backward_graph(
    q, k, v, o, dO, stats, *, right_bound
):
    @flag_dnn.graph
    def fn(q, k, v, o, dO, stats):
        return flag_dnn.sdpa_backward(
            q,
            k,
            v,
            o,
            dO,
            stats,
            diagonal_alignment="TOP_LEFT",
            diagonal_band_right_bound=right_bound,
            name="sdpa_backward",
        )

    compiled = flag_dnn.compile(
        fn,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(q, "q"),
            flag_dnn.TensorSpec.from_tensor(k, "k"),
            flag_dnn.TensorSpec.from_tensor(v, "v"),
            flag_dnn.TensorSpec.from_tensor(o, "o"),
            flag_dnn.TensorSpec.from_tensor(dO, "dO"),
            flag_dnn.TensorSpec.from_tensor(stats, "stats"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["sdpa_backward"]
    return compiled


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
@pytest.mark.parametrize(
    "shape",
    SDPA_BACKWARD_LONG_CAUSAL_D128_CASES,
    ids=("mha_b2_h16_s2048", "gqa_b1_h32_hkv8_s4096"),
)
def test_sdpa_backward_long_causal_d128(cudnn_handle, dtype, shape):
    torch.manual_seed(23)
    q, k, v = _make_qkv(shape, dtype)
    dO = torch.randn_like(q)
    o, stats = _cudnn_sdpa_forward(q, k, v, cudnn_handle, right_bound=0)
    expected = _cudnn_sdpa_backward(
        q, k, v, o, dO, stats, cudnn_handle, right_bound=0
    )
    actual = _run_flag_dnn_sdpa_backward_graph(
        q, k, v, o, dO, stats, right_bound=0
    )
    _assert_grads_close(actual, expected, dtype)


@pytest.mark.sdpa_backward
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize(
    "shape",
    SDPA_BACKWARD_LONG_CAUSAL_D128_CASES,
    ids=("mha_b2_h16_s2048", "gqa_b1_h32_hkv8_s4096"),
)
def test_sdpa_backward_long_causal_rebinds_storage(cudnn_handle, dtype, shape):
    torch.manual_seed(29)
    q_a, k_a, v_a = _make_qkv(shape, dtype)
    dO_a = torch.randn_like(q_a)
    o_a, stats_a = _cudnn_sdpa_forward(
        q_a, k_a, v_a, cudnn_handle, right_bound=0
    )
    compiled = _compile_flag_dnn_sdpa_backward_graph(
        q_a, k_a, v_a, o_a, dO_a, stats_a, right_bound=0
    )

    def check(q, k, v, o, dO, stats):
        expected = _cudnn_sdpa_backward(
            q, k, v, o, dO, stats, cudnn_handle, right_bound=0
        )
        actual = compiled.run(q, k, v, o, dO, stats)
        _assert_grads_close(actual, expected, dtype)
        return actual

    grads_a = check(q_a, k_a, v_a, o_a, dO_a, stats_a)
    q_b, k_b, v_b = _make_qkv(shape, dtype)
    dO_b = torch.randn_like(q_b)
    o_b, stats_b = _cudnn_sdpa_forward(
        q_b, k_b, v_b, cudnn_handle, right_bound=0
    )
    grads_b = check(q_b, k_b, v_b, o_b, dO_b, stats_b)
    assert all(
        left.data_ptr() != right.data_ptr()
        for left, right in zip(grads_a, grads_b)
    )

    q_b.normal_()
    k_b.normal_()
    v_b.normal_()
    dO_b.normal_()
    o_c, stats_c = _cudnn_sdpa_forward(
        q_b, k_b, v_b, cudnn_handle, right_bound=0
    )
    grads_c = check(q_b, k_b, v_b, o_c, dO_b, stats_c)
    assert all(
        left.data_ptr() != right.data_ptr()
        for left, right in zip(grads_b, grads_c)
    )


@pytest.mark.sdpa_backward
@pytest.mark.graph
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90 CUDA device is required",
)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize(
    "shape",
    (
        (4, 16, 16, 512, 512, 64),
        (8, 32, 32, 256, 256, 128),
    ),
    ids=("dense_d64", "dense_d128"),
)
def test_sdpa_backward_dense_sm90_safe_fallback(cudnn_handle, dtype, shape):
    torch.manual_seed(31)
    q, k, v = _make_qkv(shape, dtype)
    dO = torch.randn_like(q)
    o, stats = _cudnn_sdpa_forward(q, k, v, cudnn_handle)
    expected = _cudnn_sdpa_backward(
        q,
        k,
        v,
        o,
        dO,
        stats,
        cudnn_handle,
    )
    actual = _run_flag_dnn_sdpa_backward_graph(q, k, v, o, dO, stats)
    _assert_grads_close(actual, expected, dtype)


@pytest.mark.sdpa_backward
@pytest.mark.graph
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90 CUDA device is required",
)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_sdpa_backward_causal_d128_mloop_sm90_safe_fallback(
    cudnn_handle, dtype
):
    torch.manual_seed(37)
    shape = (1, 1, 1, 2048, 2048, 128)
    q, k, v = _make_qkv(shape, dtype)
    dO = torch.randn_like(q)
    o, stats = _cudnn_sdpa_forward(q, k, v, cudnn_handle, right_bound=0)
    expected = _cudnn_sdpa_backward(
        q, k, v, o, dO, stats, cudnn_handle, right_bound=0
    )
    actual = _run_flag_dnn_sdpa_backward_graph(
        q,
        k,
        v,
        o,
        dO,
        stats,
        right_bound=0,
    )
    _assert_grads_close(actual, expected, dtype)


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
