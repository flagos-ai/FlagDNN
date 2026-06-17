import math

import pytest
from benchmark_graph.base import (
    CudnnCompareBenchmark,
    get_cudnn,
    skip_unsupported_cudnn_graph,
)
import torch

import flag_dnn
from benchmark_graph import consts


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


class SdpaFp8BackwardBenchmark(CudnnCompareBenchmark):
    op_name = "sdpa_fp8_backward"
    dtypes = (torch.float8_e4m3fn,)
    shapes = consts.SDPA_FP8_BACKWARD_SHAPES
    shape_ids_env = "FLAGDNN_CUDNN_SDPA_FP8_BACKWARD_PERF_SHAPE_IDS"

    def make_inputs(self, shape, dtype):
        batch, hq, hkv, sq, skv, d, causal = shape
        qg = torch.randn((batch, hq, sq, d), device=flag_dnn.device)
        kg = torch.randn((batch, hkv, skv, d), device=flag_dnn.device)
        vg = torch.randn((batch, hkv, skv, d), device=flag_dnn.device)
        dog = torch.randn((batch, hq, sq, d), device=flag_dnn.device)
        q_scale = _fp8_scale(qg.abs().max().item(), dtype)
        k_scale = _fp8_scale(kg.abs().max().item(), dtype)
        v_scale = _fp8_scale(vg.abs().max().item(), dtype)
        do_scale = _fp8_scale(dog.abs().max().item(), dtype)
        q = (qg * q_scale).to(dtype)
        k = (kg * k_scale).to(dtype)
        v = (vg * v_scale).to(dtype)
        dO = (dog * do_scale).to(dtype)
        dq = 1.0 / q_scale
        dk = 1.0 / k_scale
        dv = 1.0 / v_scale
        ddo = 1.0 / do_scale
        ss = _fp8_scale(1.0, dtype)
        ds = 1.0 / ss
        so = _fp8_scale(1.0, dtype)
        sdp = _fp8_scale(1.0, dtype)
        sdq = _fp8_scale(1.0, dtype)
        sdk = _fp8_scale(1.0, dtype)
        sdv = _fp8_scale(1.0, dtype)
        o, stats = flag_dnn.sdpa_fp8(
            q,
            k,
            v,
            dq,
            dk,
            dv,
            ds,
            ss,
            so,
            attn_scale=1.0 / math.sqrt(d),
            use_causal_mask=bool(causal),
            generate_stats=True,
        )[:2]
        torch.cuda.synchronize()
        return (
            q,
            k,
            v,
            o,
            dO,
            stats,
            dq,
            dk,
            dv,
            1.0 / so,
            ddo,
            ds,
            1.0 / sdp,
            ss,
            sdq,
            sdk,
            sdv,
            sdp,
            bool(causal),
            dtype,
        )

    def build_cudnn_runner(self, inputs):
        cudnn = get_cudnn()
        (
            q,
            k,
            v,
            o,
            dO,
            stats,
            dq,
            dk,
            dv,
            do_,
            ddo,
            ds,
            ddp,
            ss,
            sdq,
            sdk,
            sdv,
            sdp,
            causal,
            dtype,
        ) = inputs
        itype = _cudnn_fp8_type(dtype)
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
            attn_scale=1.0 / math.sqrt(q.shape[-1]),
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
            skip_unsupported_cudnn_graph(exc, self.op_name)

        def f1(x):
            return torch.tensor([x], device=q.device, dtype=torch.float32)

        dQ = torch.empty_like(q)
        dK = torch.empty_like(k)
        dV = torch.empty_like(v)
        amax = [
            torch.empty((1, 1, 1, 1), device=q.device, dtype=torch.float32)
            for _ in range(4)
        ]
        pack = {
            qt: q,
            kt: k,
            vt: v,
            ot: o,
            dot: dO,
            st: stats,
            dqt: f1(dq),
            dkt: f1(dk),
            dvt: f1(dv),
            dot_s: f1(do_),
            ddot: f1(ddo),
            dst: f1(ds),
            ddpt: f1(ddp),
            sst: f1(ss),
            sdqt: f1(sdq),
            sdkt: f1(sdk),
            sdvt: f1(sdv),
            sdpt: f1(sdp),
            dq_t: dQ,
            dk_t: dK,
            dv_t: dV,
            adq_t: amax[0],
            adk_t: amax[1],
            adv_t: amax[2],
            adp_t: amax[3],
        }
        workspace = torch.empty(
            graph.get_workspace_size(), device=q.device, dtype=torch.uint8
        )

        def run():
            graph.execute(pack, workspace, handle=self.cudnn_handle)
            return dQ, dK, dV

        return run

    def build_flag_dnn_runner(self, inputs):
        (
            q,
            k,
            v,
            o,
            dO,
            stats,
            dq,
            dk,
            dv,
            do_,
            ddo,
            ds,
            ddp,
            ss,
            sdq,
            sdk,
            sdv,
            sdp,
            causal,
            _,
        ) = inputs

        @flag_dnn.graph
        def flag_dnn_sdpa_fp8_backward_graph(q, k, v, o, dO, stats):
            return flag_dnn.sdpa_fp8_backward(
                q,
                k,
                v,
                o,
                dO,
                stats,
                dq,
                dk,
                dv,
                do_,
                ddo,
                ds,
                ddp,
                ss,
                sdq,
                sdk,
                sdv,
                sdp,
                attn_scale=1.0 / math.sqrt(q.shape[-1]),
                use_causal_mask=causal,
                name=self.op_name,
            )

        compiled = flag_dnn.compile(
            flag_dnn_sdpa_fp8_backward_graph,
            inputs=[
                flag_dnn.TensorSpec.from_tensor(q, "q"),
                flag_dnn.TensorSpec.from_tensor(k, "k"),
                flag_dnn.TensorSpec.from_tensor(v, "v"),
                flag_dnn.TensorSpec.from_tensor(o, "o"),
                flag_dnn.TensorSpec.from_tensor(dO, "dO"),
                flag_dnn.TensorSpec.from_tensor(stats, "stats"),
            ],
            options=consts.compile_options(),
        )
        assert [node.op_type for node in compiled.graph.nodes] == [
            "sdpa_fp8_backward"
        ]

        def run():
            return compiled.run(q, k, v, o, dO, stats)

        return run

    def transfer_bytes(self, inputs):
        q, k, v, o, dO, stats = inputs[:6]
        return (
            q.numel() * q.element_size()
            + k.numel() * k.element_size()
            + v.numel() * v.element_size()
            + o.numel() * o.element_size()
            + dO.numel() * dO.element_size()
            + stats.numel() * stats.element_size()
            + q.numel() * q.element_size()
            + k.numel() * k.element_size()
            + v.numel() * v.element_size()
            + 4 * 4
        )

    def shape_detail(self, inputs):
        q, k = inputs[0], inputs[1]
        causal = inputs[18]
        return [tuple(q.shape), tuple(k.shape), f"causal={causal}"]


@pytest.mark.sdpa_fp8_backward
@pytest.mark.graph
@pytest.mark.perf
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", SdpaFp8BackwardBenchmark.dtypes)
def test_sdpa_fp8_backward(cudnn_handle, dtype):
    torch.manual_seed(0)
    SdpaFp8BackwardBenchmark(cudnn_handle).run(dtype)
