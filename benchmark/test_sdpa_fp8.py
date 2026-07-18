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
from benchmark.base import (
    CudnnCompareBenchmark,
    get_cudnn,
    skip_unsupported_cudnn_graph,
)
import torch

import flag_dnn
from benchmark import consts

# fp8 SDPA forward performance: FlagDNN compiled graph (sdpa_fp8 node) vs cuDNN
# frontend graph.sdpa_fp8 on identical fp8 inputs / per-tensor scaling factors.


def _fp8_largest(dtype):
    return 128.0 if dtype == torch.float8_e4m3fn else 32768.0


def _fp8_scale(amax, dtype, fudge=0.25, eps=0.0625):
    po2 = 2 ** math.ceil(math.log2(max(amax, eps)))
    return _fp8_largest(dtype) / po2 * fudge


def _cudnn_fp8_type(dtype):
    cudnn = get_cudnn()
    if dtype == torch.float8_e4m3fn:
        return cudnn.data_type.FP8_E4M3
    return cudnn.data_type.FP8_E5M2


class SdpaFp8Benchmark(CudnnCompareBenchmark):
    op_name = "sdpa_fp8"
    dtypes = (torch.float8_e4m3fn,)
    shapes = consts.SDPA_FP8_SHAPES
    shape_ids_env = "FLAGDNN_CUDNN_SDPA_FP8_PERF_SHAPE_IDS"

    def make_inputs(self, shape, dtype):
        batch, hq, hkv, sq, skv, d, causal, stats = shape
        # Quantised normal data is enough for timing (values are not compared).
        qg = torch.randn((batch, hq, sq, d), device=flag_dnn.device)
        kg = torch.randn((batch, hkv, skv, d), device=flag_dnn.device)
        vg = torch.randn((batch, hkv, skv, d), device=flag_dnn.device)
        q = (qg * _fp8_scale(qg.abs().max().item(), dtype)).to(dtype)
        k = (kg * _fp8_scale(kg.abs().max().item(), dtype)).to(dtype)
        v = (vg * _fp8_scale(vg.abs().max().item(), dtype)).to(dtype)
        dq = 1.0 / _fp8_scale(qg.abs().max().item(), dtype)
        dk = 1.0 / _fp8_scale(kg.abs().max().item(), dtype)
        dv = 1.0 / _fp8_scale(vg.abs().max().item(), dtype)
        ss = _fp8_scale(1.0, dtype)
        ds = 1.0 / ss
        so = _fp8_scale(1.0, dtype)
        return (
            q,
            k,
            v,
            dq,
            dk,
            dv,
            ds,
            ss,
            so,
            bool(causal),
            bool(stats),
            dtype,
        )

    def build_cudnn_runner(self, inputs):
        cudnn = get_cudnn()
        q, k, v, dq, dk, dv, ds, ss, so, causal, stats, dtype = inputs
        itype = _cudnn_fp8_type(dtype)
        b, h, sq, d = q.shape
        graph = cudnn.pygraph(
            io_data_type=itype,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=self.cudnn_handle,
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
            attn_scale=1.0 / math.sqrt(d),
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
            skip_unsupported_cudnn_graph(exc, self.op_name)

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
        if stats:
            pack[stats_t] = torch.empty(
                (b, h, sq, 1), device=q.device, dtype=torch.float32
            )
        workspace = torch.empty(
            graph.get_workspace_size(), device=q.device, dtype=torch.uint8
        )

        def run():
            graph.execute(pack, workspace, handle=self.cudnn_handle)
            return o_gpu

        return run

    def build_flag_dnn_runner(self, inputs):
        q, k, v, dq, dk, dv, ds, ss, so, causal, stats, dtype = inputs
        attn_scale = 1.0 / math.sqrt(q.shape[-1])

        @flag_dnn.graph
        def flag_dnn_sdpa_fp8_graph(q, k, v):
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
                name=self.op_name,
            )

        compiled = flag_dnn.compile(
            flag_dnn_sdpa_fp8_graph,
            inputs=[
                flag_dnn.TensorSpec.from_tensor(q, "q"),
                flag_dnn.TensorSpec.from_tensor(k, "k"),
                flag_dnn.TensorSpec.from_tensor(v, "v"),
            ],
            options=consts.compile_options(),
        )
        assert [node.op_type for node in compiled.graph.nodes] == ["sdpa_fp8"]

        def run():
            return compiled.run(q, k, v)

        return run

    def transfer_bytes(self, inputs):
        q, k, v = inputs[0], inputs[1], inputs[2]
        stats = inputs[10]
        total = (
            q.numel() * q.element_size()
            + k.numel() * k.element_size()
            + v.numel() * v.element_size()
            + q.numel() * q.element_size()
        )
        if stats:
            total += q.shape[0] * q.shape[1] * q.shape[2] * 4
        return total

    def shape_detail(self, inputs):
        q, k = inputs[0], inputs[1]
        causal, stats = inputs[9], inputs[10]
        return [
            tuple(q.shape),
            tuple(k.shape),
            f"causal={causal}",
            f"stats={stats}",
        ]


@pytest.mark.sdpa_fp8
@pytest.mark.graph
@pytest.mark.perf
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", SdpaFp8Benchmark.dtypes)
def test_sdpa_fp8(cudnn_handle, dtype):
    torch.manual_seed(0)
    SdpaFp8Benchmark(cudnn_handle).run(dtype)
