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
    cudnn,
    skip_unsupported_cudnn_graph,
)
import torch

import flag_dnn
from tests import consts

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_FP8_BWD_CASES = consts.SDPA_FP8_CASES[:4] + (consts.SDPA_FP8_CASES[5],)
_NVIDIA_FP8_BWD_CASE = (4, 16, 16, 512, 512, 128, False)

_ATOL = 0.18
_RTOL = 0.35


def _fp8_largest(dtype):
    return 128.0 if dtype == torch.float8_e4m3fn else 32768.0


def _fp8_scale(amax, dtype, fudge=0.25, eps=0.0625):
    po2 = 2 ** math.ceil(math.log2(max(amax, eps)))
    return _fp8_largest(dtype) / po2 * fudge


def _cudnn_fp8_type(dtype):
    if dtype == torch.float8_e4m3fn:
        return cudnn.data_type.FP8_E4M3
    return cudnn.data_type.FP8_E5M2


def _sparse_int(shape, rng):
    t = torch.empty(shape, device=flag_dnn.device, dtype=torch.float32)
    t.random_(-2, 3, generator=rng)
    mask = torch.empty(shape, device=flag_dnn.device, dtype=torch.float32)
    mask.uniform_(generator=rng)
    t[mask < 0.8] = 0.0
    return t


def _make_inputs(case, dtype):
    batch, hq, hkv, sq, skv, d, causal = case
    rng = torch.Generator(device=flag_dnn.device).manual_seed(17)
    qg = _sparse_int((batch, hq, sq, d), rng)
    kg = _sparse_int((batch, hkv, skv, d), rng)
    vg = _sparse_int((batch, hkv, skv, d), rng)
    dog = _sparse_int((batch, hq, sq, d), rng)
    q_scale = _fp8_scale(qg.abs().max().item(), dtype)
    k_scale = _fp8_scale(kg.abs().max().item(), dtype)
    v_scale = _fp8_scale(vg.abs().max().item(), dtype)
    do_scale = _fp8_scale(dog.abs().max().item(), dtype)
    q = (qg * q_scale).to(dtype)
    k = (kg * k_scale).to(dtype)
    v = (vg * v_scale).to(dtype)
    dO = (dog * do_scale).to(dtype)
    return (
        q,
        k,
        v,
        dO,
        {
            "q": 1.0 / q_scale,
            "k": 1.0 / k_scale,
            "v": 1.0 / v_scale,
            "dO": 1.0 / do_scale,
        },
        causal,
    )


def _backward_scales(dtype):
    dP = _fp8_scale(1.0, dtype)
    return {
        "dP": dP,
        "descale_dP": 1.0 / dP,
        "dQ": 1.0,
        "dK": 1.0,
        "dV": 1.0,
    }


def _cudnn_sdpa_fp8_forward(
    handle, q, k, v, dq, dk, dv, ds, ss, so, attn_scale, causal, dtype
):
    itype = _cudnn_fp8_type(dtype)
    b, h, sq, d = q.shape
    graph = cudnn.pygraph(
        io_data_type=itype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle,
    )

    def fp8_t(t):
        return graph.tensor(
            dim=tuple(t.shape), stride=tuple(t.stride()), data_type=itype
        )

    def scalar_t():
        return graph.tensor(
            dim=(1, 1, 1, 1),
            stride=(1, 1, 1, 1),
            data_type=cudnn.data_type.FLOAT,
        )

    qt, kt, vt = fp8_t(q), fp8_t(k), fp8_t(v)
    dqt, dkt, dvt, dst, sst, sot = (
        scalar_t(),
        scalar_t(),
        scalar_t(),
        scalar_t(),
        scalar_t(),
        scalar_t(),
    )
    o_t, stats_t, amax_s_t, amax_o_t = graph.sdpa_fp8(
        q=qt,
        k=kt,
        v=vt,
        descale_q=dqt,
        descale_k=dkt,
        descale_v=dvt,
        descale_s=dst,
        scale_s=sst,
        scale_o=sot,
        generate_stats=True,
        attn_scale=attn_scale,
        use_causal_mask=causal,
    )
    o_t.set_output(True).set_dim((b, h, sq, d)).set_stride(
        (h * sq * d, sq * d, d, 1)
    ).set_data_type(itype)
    stats_t.set_output(True).set_dim((b, h, sq, 1)).set_stride(
        (h * sq, sq, 1, 1)
    ).set_data_type(cudnn.data_type.FLOAT)
    amax_s_t.set_output(True).set_dim((1, 1, 1, 1)).set_stride(
        (1, 1, 1, 1)
    ).set_data_type(cudnn.data_type.FLOAT)
    amax_o_t.set_output(True).set_dim((1, 1, 1, 1)).set_stride(
        (1, 1, 1, 1)
    ).set_data_type(cudnn.data_type.FLOAT)
    try:
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans(
            [cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]
        )
        graph.check_support()
        graph.build_plans()
    except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as exc:
        skip_unsupported_cudnn_graph(exc, "sdpa_fp8")

    def f1(x):
        return torch.tensor([x], device=q.device, dtype=torch.float32)

    o = torch.empty((b, h, sq, d), device=q.device, dtype=dtype)
    stats = torch.empty((b, h, sq, 1), device=q.device, dtype=torch.float32)
    amax_s = torch.empty((1, 1, 1, 1), device=q.device, dtype=torch.float32)
    amax_o = torch.empty((1, 1, 1, 1), device=q.device, dtype=torch.float32)
    pack = {
        qt: q,
        kt: k,
        vt: v,
        dqt: f1(dq),
        dkt: f1(dk),
        dvt: f1(dv),
        dst: f1(ds),
        sst: f1(ss),
        sot: f1(so),
        o_t: o,
        stats_t: stats,
        amax_s_t: amax_s,
        amax_o_t: amax_o,
    }
    workspace = torch.empty(
        graph.get_workspace_size(), device=q.device, dtype=torch.uint8
    )
    graph.execute(pack, workspace, handle=handle)
    torch.cuda.synchronize()
    return o, stats


def _cudnn_sdpa_fp8_backward(
    handle, q, k, v, o, dO, stats, scales, descale, attn_scale, causal, dtype
):
    itype = _cudnn_fp8_type(dtype)
    graph = cudnn.pygraph(
        io_data_type=itype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle,
    )

    def fp8_t(t):
        return graph.tensor(
            dim=tuple(t.shape), stride=tuple(t.stride()), data_type=itype
        )

    def scalar_t():
        return graph.tensor(
            dim=(1, 1, 1, 1),
            stride=(1, 1, 1, 1),
            data_type=cudnn.data_type.FLOAT,
        )

    qt, kt, vt, ot, dot, st = (
        fp8_t(q),
        fp8_t(k),
        fp8_t(v),
        fp8_t(o),
        fp8_t(dO),
        graph.tensor_like(stats),
    )
    dqt, dkt, dvt, dot_s, ddot, dst, ddpt = (
        scalar_t(),
        scalar_t(),
        scalar_t(),
        scalar_t(),
        scalar_t(),
        scalar_t(),
        scalar_t(),
    )
    sst, sdqt, sdkt, sdvt, sdpt = (
        scalar_t(),
        scalar_t(),
        scalar_t(),
        scalar_t(),
        scalar_t(),
    )
    dq_t, dk_t, dv_t, adq_t, adk_t, adv_t, adp_t = graph.sdpa_fp8_backward(
        q=qt,
        k=kt,
        v=vt,
        o=ot,
        dO=dot,
        stats=st,
        descale_q=dqt,
        descale_k=dkt,
        descale_v=dvt,
        descale_o=dot_s,
        descale_dO=ddot,
        descale_s=dst,
        descale_dP=ddpt,
        scale_s=sst,
        scale_dQ=sdqt,
        scale_dK=sdkt,
        scale_dV=sdvt,
        scale_dP=sdpt,
        attn_scale=attn_scale,
        use_causal_mask=causal,
    )
    for tensor, ref in ((dq_t, q), (dk_t, k), (dv_t, v)):
        tensor.set_output(True).set_dim(tuple(ref.shape)).set_stride(
            tuple(ref.stride())
        ).set_data_type(itype)
    for tensor in (adq_t, adk_t, adv_t, adp_t):
        tensor.set_output(True).set_dim((1, 1, 1, 1)).set_stride(
            (1, 1, 1, 1)
        ).set_data_type(cudnn.data_type.FLOAT)
    try:
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans(
            [cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]
        )
        graph.check_support()
        graph.build_plans()
    except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as exc:
        skip_unsupported_cudnn_graph(exc, "sdpa_fp8_backward")

    def f1(x):
        return torch.tensor([x], device=q.device, dtype=torch.float32)

    out = (torch.empty_like(q), torch.empty_like(k), torch.empty_like(v))
    amax = tuple(
        torch.empty((1, 1, 1, 1), device=q.device, dtype=torch.float32)
        for _ in range(4)
    )
    pack = {
        qt: q,
        kt: k,
        vt: v,
        ot: o,
        dot: dO,
        st: stats,
        dqt: f1(descale["q"]),
        dkt: f1(descale["k"]),
        dvt: f1(descale["v"]),
        dot_s: f1(1.0 / scales["o"]),
        ddot: f1(descale["dO"]),
        dst: f1(1.0 / scales["s"]),
        ddpt: f1(scales["descale_dP"]),
        sst: f1(scales["s"]),
        sdqt: f1(scales["dQ"]),
        sdkt: f1(scales["dK"]),
        sdvt: f1(scales["dV"]),
        sdpt: f1(scales["dP"]),
        dq_t: out[0],
        dk_t: out[1],
        dv_t: out[2],
        adq_t: amax[0],
        adk_t: amax[1],
        adv_t: amax[2],
        adp_t: amax[3],
    }
    workspace = torch.empty(
        graph.get_workspace_size(), device=q.device, dtype=torch.uint8
    )
    graph.execute(pack, workspace, handle=handle)
    torch.cuda.synchronize()
    return out + amax


def _run_flagdnn_graph(
    q, k, v, o, dO, stats, scales, descale, attn_scale, causal
):
    @flag_dnn.graph
    def fn(q, k, v, o, dO, stats):
        return flag_dnn.sdpa_fp8_backward(
            q,
            k,
            v,
            o,
            dO,
            stats,
            descale["q"],
            descale["k"],
            descale["v"],
            1.0 / scales["o"],
            descale["dO"],
            1.0 / scales["s"],
            scales["descale_dP"],
            scales["s"],
            scales["dQ"],
            scales["dK"],
            scales["dV"],
            scales["dP"],
            attn_scale=attn_scale,
            use_causal_mask=causal,
            name="sdpa_fp8_backward",
        )

    specs = [
        flag_dnn.TensorSpec.from_tensor(q, "q"),
        flag_dnn.TensorSpec.from_tensor(k, "k"),
        flag_dnn.TensorSpec.from_tensor(v, "v"),
        flag_dnn.TensorSpec.from_tensor(o, "o"),
        flag_dnn.TensorSpec.from_tensor(dO, "dO"),
        flag_dnn.TensorSpec.from_tensor(stats, "stats"),
    ]
    compiled = flag_dnn.compile(fn, inputs=specs, options={"cache": None})
    assert [node.op_type for node in compiled.graph.nodes] == [
        "sdpa_fp8_backward"
    ]
    return compiled.run(
        q.clone(), k.clone(), v.clone(), o.clone(), dO.clone(), stats.clone()
    )


def _assert_close(flag_out, cudnn_out, scales):
    for idx, scale in enumerate((scales["dQ"], scales["dK"], scales["dV"])):
        torch.testing.assert_close(
            flag_out[idx].float() / scale,
            cudnn_out[idx].float() / scale,
            atol=_ATOL,
            rtol=_RTOL,
            check_dtype=False,
        )
    for actual in flag_out[3:]:
        actual_v = float(actual.reshape(()).item())
        assert math.isfinite(actual_v)
        assert actual_v > 0.0


def _check_sdpa_fp8_backward(cudnn_handle, dtype, case):
    q, k, v, dO, descale, causal = _make_inputs(case, dtype)
    attn_scale = 1.0 / math.sqrt(q.shape[-1])
    ss = _fp8_scale(1.0, dtype)
    ds = 1.0 / ss
    so = 1.0
    o, stats = _cudnn_sdpa_fp8_forward(
        cudnn_handle,
        q,
        k,
        v,
        descale["q"],
        descale["k"],
        descale["v"],
        ds,
        ss,
        so,
        attn_scale,
        causal,
        dtype,
    )
    scales = _backward_scales(dtype)
    scales["s"] = ss
    scales["o"] = so
    cudnn_out = _cudnn_sdpa_fp8_backward(
        cudnn_handle,
        q,
        k,
        v,
        o,
        dO,
        stats,
        scales,
        descale,
        attn_scale,
        causal,
        dtype,
    )
    flag_out = _run_flagdnn_graph(
        q, k, v, o, dO, stats, scales, descale, attn_scale, causal
    )
    _assert_close(flag_out, cudnn_out, scales)


@pytest.mark.sdpa_fp8_backward
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", _FP8_DTYPES)
@pytest.mark.parametrize("case", _FP8_BWD_CASES)
def test_sdpa_fp8_backward_vs_cudnn(cudnn_handle, dtype, case):
    _check_sdpa_fp8_backward(cudnn_handle, dtype, case)


@pytest.mark.sdpa_fp8_backward
@pytest.mark.graph
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90 CUDA device is required",
)
def test_sdpa_fp8_backward_nvidia_sm90_exact_fast_path(cudnn_handle):
    _check_sdpa_fp8_backward(
        cudnn_handle,
        torch.float8_e4m3fn,
        _NVIDIA_FP8_BWD_CASE,
    )
