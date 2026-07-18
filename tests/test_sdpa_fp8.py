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

# fp8 SDPA forward correctness: FlagDNN triton sdpa_fp8 (captured as a graph
# op) compared against cuDNN frontend graph.sdpa_fp8 on the same fp8 inputs and
# scaling factors. Tolerances mirror cuDNN's own fp8 SDPA tests.
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

_ATOL = 0.08
_RTOL = 0.2


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
    rng = torch.Generator(device=flag_dnn.device).manual_seed(0)
    qg = _sparse_int((batch, hq, sq, d), rng)
    kg = _sparse_int((batch, hkv, skv, d), rng)
    vg = _sparse_int((batch, hkv, skv, d), rng)
    q = (qg * _fp8_scale(qg.abs().max().item(), dtype)).to(dtype)
    k = (kg * _fp8_scale(kg.abs().max().item(), dtype)).to(dtype)
    v = (vg * _fp8_scale(vg.abs().max().item(), dtype)).to(dtype)
    descale = {
        "q": 1.0 / _fp8_scale(qg.abs().max().item(), dtype),
        "k": 1.0 / _fp8_scale(kg.abs().max().item(), dtype),
        "v": 1.0 / _fp8_scale(vg.abs().max().item(), dtype),
    }
    return q, k, v, descale, causal


def _cudnn_sdpa_fp8(
    handle, q, k, v, dq, dk, dv, ds, ss, so, attn_scale, causal, stats, dtype
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
        generate_stats=stats,
        attn_scale=attn_scale,
        use_causal_mask=causal,
    )
    o_t.set_output(True).set_dim((b, h, sq, d)).set_stride(
        (h * sq * d, sq * d, d, 1)
    ).set_data_type(itype)
    amax_s_t.set_output(True).set_dim((1, 1, 1, 1)).set_stride(
        (1, 1, 1, 1)
    ).set_data_type(cudnn.data_type.FLOAT)
    amax_o_t.set_output(True).set_dim((1, 1, 1, 1)).set_stride(
        (1, 1, 1, 1)
    ).set_data_type(cudnn.data_type.FLOAT)
    if stats:
        stats_t.set_output(True).set_dim((b, h, sq, 1)).set_stride(
            (h * sq, sq, 1, 1)
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

    o_gpu = torch.empty((b, h, sq, d), device=q.device, dtype=dtype)
    amax_s_gpu = torch.empty(
        (1, 1, 1, 1), device=q.device, dtype=torch.float32
    )
    amax_o_gpu = torch.empty(
        (1, 1, 1, 1), device=q.device, dtype=torch.float32
    )
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
        o_t: o_gpu,
        amax_s_t: amax_s_gpu,
        amax_o_t: amax_o_gpu,
    }
    stats_gpu = None
    if stats:
        stats_gpu = torch.empty(
            (b, h, sq, 1), device=q.device, dtype=torch.float32
        )
        pack[stats_t] = stats_gpu
    workspace = torch.empty(
        graph.get_workspace_size(), device=q.device, dtype=torch.uint8
    )
    graph.execute(pack, workspace, handle=handle)
    torch.cuda.synchronize()
    return o_gpu, stats_gpu, amax_s_gpu, amax_o_gpu


def _run_flagdnn_fp8_graph(
    q, k, v, dq, dk, dv, ds, ss, so, attn_scale, causal, stats
):
    @flag_dnn.graph
    def fn(q, k, v):
        return flag_dnn.sdpa_fp8(
            q,
            k,
            v,
            dq,
            dk,
            dv,
            ds,
            ss,
            so,
            attn_scale=attn_scale,
            use_causal_mask=causal,
            generate_stats=stats,
            name="sdpa_fp8",
        )

    specs = [
        flag_dnn.TensorSpec.from_tensor(q, "q"),
        flag_dnn.TensorSpec.from_tensor(k, "k"),
        flag_dnn.TensorSpec.from_tensor(v, "v"),
    ]
    compiled = flag_dnn.compile(fn, inputs=specs, options={"cache": None})
    assert [node.op_type for node in compiled.graph.nodes] == ["sdpa_fp8"]
    # Run twice: the first launch triggers libtuner autotuning, whose repeated
    # atomic_max into the (per-call) amax buffers can marginally inflate amax;
    # the cached second launch yields the clean per-tensor amax values.
    out = compiled.run(q.clone(), k.clone(), v.clone())
    out = compiled.run(q.clone(), k.clone(), v.clone())
    return out


def _assert_fp8_close(flag_out, cudnn_out, descale_o, stats):
    o_flag = flag_out[0].float() * descale_o
    o_cudnn = cudnn_out[0].float() * descale_o
    torch.testing.assert_close(o_flag, o_cudnn, atol=_ATOL, rtol=_RTOL)

    amax_o_flag = float(flag_out[-1].reshape(()).item())
    amax_o_cudnn = float(cudnn_out[3].reshape(()).item())
    assert amax_o_flag > 0.0
    assert math.isclose(amax_o_flag, amax_o_cudnn, rel_tol=0.1, abs_tol=0.05)

    if stats:
        stats_flag = flag_out[1].squeeze(-1)
        stats_cudnn = cudnn_out[1].squeeze(-1)
        torch.testing.assert_close(
            stats_flag, stats_cudnn, atol=0.05, rtol=0.05
        )


def _scales(descale, dtype):
    dq, dk, dv = descale["q"], descale["k"], descale["v"]
    ss = _fp8_scale(1.0, dtype)
    ds = 1.0 / ss
    so = 1.0
    return dq, dk, dv, ds, ss, so


@pytest.mark.sdpa_fp8
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", _FP8_DTYPES)
@pytest.mark.parametrize("case", consts.SDPA_FP8_CASES)
def test_sdpa_fp8_vs_cudnn_with_stats(cudnn_handle, dtype, case):
    torch.manual_seed(0)
    q, k, v, descale, causal = _make_inputs(case, dtype)
    attn_scale = 1.0 / math.sqrt(q.shape[-1])
    dq, dk, dv, ds, ss, so = _scales(descale, dtype)

    cudnn_out = _cudnn_sdpa_fp8(
        cudnn_handle,
        q,
        k,
        v,
        dq,
        dk,
        dv,
        ds,
        ss,
        so,
        attn_scale,
        causal,
        True,
        dtype,
    )
    flag_out = _run_flagdnn_fp8_graph(
        q, k, v, dq, dk, dv, ds, ss, so, attn_scale, causal, True
    )
    _assert_fp8_close(flag_out, cudnn_out, 1.0 / so, stats=True)


@pytest.mark.sdpa_fp8
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("case", consts.SDPA_FP8_CASES[:4])
def test_sdpa_fp8_inference_no_stats(cudnn_handle, case):
    dtype = torch.float8_e4m3fn
    torch.manual_seed(0)
    q, k, v, descale, causal = _make_inputs(case, dtype)
    attn_scale = 1.0 / math.sqrt(q.shape[-1])
    dq, dk, dv, ds, ss, so = _scales(descale, dtype)

    cudnn_out = _cudnn_sdpa_fp8(
        cudnn_handle,
        q,
        k,
        v,
        dq,
        dk,
        dv,
        ds,
        ss,
        so,
        attn_scale,
        causal,
        False,
        dtype,
    )
    flag_out = _run_flagdnn_fp8_graph(
        q, k, v, dq, dk, dv, ds, ss, so, attn_scale, causal, False
    )
    assert len(flag_out) == 3
    _assert_fp8_close(flag_out, cudnn_out, 1.0 / so, stats=False)
