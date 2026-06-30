import math

import pytest
from benchmark.base import (
    CudnnCompareBenchmark,
    cudnn_data_type,
    get_cudnn,
    skip_unsupported_cudnn_graph,
)
import torch

import flag_dnn
from benchmark import consts


class SdpaBackwardBenchmark(CudnnCompareBenchmark):
    op_name = "sdpa_backward"
    shapes = consts.SDPA_BACKWARD_SHAPES
    shape_ids_env = "FLAGDNN_CUDNN_SDPA_BACKWARD_PERF_SHAPE_IDS"

    def make_inputs(self, shape, dtype):
        batch, heads_q, heads_kv, seq_q, seq_kv, head_dim, causal = shape
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
        dO = torch.randn(
            (batch, heads_q, seq_q, head_dim),
            dtype=dtype,
            device=flag_dnn.device,
        )
        o, stats = flag_dnn.sdpa(
            q, k, v, use_causal_mask=bool(causal), generate_stats=True
        )
        torch.cuda.synchronize()
        return q, k, v, o, dO, stats, bool(causal)

    def build_cudnn_runner(self, inputs):
        cudnn = get_cudnn()
        q, k, v, o, dO, stats, causal = inputs
        io_dtype = cudnn_data_type(q.dtype)
        graph = cudnn.pygraph(
            io_data_type=io_dtype,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=self.cudnn_handle,
        )
        q_tensor = graph.tensor_like(q)
        k_tensor = graph.tensor_like(k)
        v_tensor = graph.tensor_like(v)
        o_tensor = graph.tensor_like(o)
        do_tensor = graph.tensor_like(dO)
        stats_tensor = graph.tensor_like(stats)
        kwargs = dict(
            attn_scale=1.0 / math.sqrt(q.shape[-1]),
            name=self.op_name,
        )
        if causal:
            kwargs["diagonal_band_right_bound"] = 0
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
            tensor.set_output(True).set_data_type(io_dtype)
            tensor.set_dim(list(ref.shape)).set_stride(list(ref.stride()))

        try:
            graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as exc:
            skip_unsupported_cudnn_graph(exc, self.op_name)

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
        workspace = torch.empty(
            graph.get_workspace_size(), device=q.device, dtype=torch.uint8
        )

        def run():
            graph.execute(exec_tensors, workspace, handle=self.cudnn_handle)
            return dQ, dK, dV

        return run

    def build_flag_dnn_runner(self, inputs):
        q, k, v, o, dO, stats, causal = inputs

        @flag_dnn.graph
        def flag_dnn_sdpa_backward_graph(q, k, v, o, dO, stats):
            return flag_dnn.sdpa_backward(
                q,
                k,
                v,
                o,
                dO,
                stats,
                use_causal_mask=causal,
                name=self.op_name,
            )

        compiled = flag_dnn.compile(
            flag_dnn_sdpa_backward_graph,
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
            "sdpa_backward"
        ]

        def run():
            return compiled.run(q, k, v, o, dO, stats)

        return run

    def transfer_bytes(self, inputs):
        q, k, v, o, dO, stats, _ = inputs
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
        )

    def shape_detail(self, inputs):
        q, k, _, _, _, _, causal = inputs
        return [tuple(q.shape), tuple(k.shape), f"causal={causal}"]


@pytest.mark.sdpa_backward
@pytest.mark.graph
@pytest.mark.perf
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", SdpaBackwardBenchmark.dtypes)
def test_sdpa_backward(cudnn_handle, dtype):
    torch.manual_seed(0)
    SdpaBackwardBenchmark(cudnn_handle).run(dtype)
