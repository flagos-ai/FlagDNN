import math

import pytest
from benchmark_graph.base import (
    CudnnCompareBenchmark,
    cudnn_data_type,
    get_cudnn,
    skip_unsupported_cudnn_graph,
)
import torch

import flag_dnn
from benchmark_graph import consts


class SdpaBenchmark(CudnnCompareBenchmark):
    op_name = "sdpa"
    shapes = consts.SDPA_SHAPES
    shape_ids_env = "FLAGDNN_CUDNN_SDPA_PERF_SHAPE_IDS"

    def make_inputs(self, shape, dtype):
        batch, heads_q, heads_kv, seq_q, seq_kv, head_dim, causal, stats = (
            shape
        )
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
        return q, k, v, bool(causal), bool(stats)

    def build_cudnn_runner(self, inputs):
        cudnn = get_cudnn()
        q, k, v, causal, stats = inputs
        batch, heads, seq_q, head_dim = q.shape
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
        kwargs = dict(
            attn_scale=1.0 / math.sqrt(head_dim),
            generate_stats=stats,
            name=self.op_name,
        )
        if causal:
            kwargs["diagonal_band_right_bound"] = 0
        o_tensor, stats_tensor = graph.sdpa(
            q_tensor, k_tensor, v_tensor, **kwargs
        )
        o_tensor.set_output(True).set_data_type(io_dtype)
        o_tensor.set_dim([batch, heads, seq_q, head_dim]).set_stride(
            [heads * seq_q * head_dim, seq_q * head_dim, head_dim, 1]
        )
        if stats:
            stats_tensor.set_output(True).set_data_type(cudnn.data_type.FLOAT)

        try:
            graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as exc:
            skip_unsupported_cudnn_graph(exc, self.op_name)

        o = torch.empty(
            (batch, heads, seq_q, head_dim), device=q.device, dtype=q.dtype
        )
        exec_tensors = {q_tensor: q, k_tensor: k, v_tensor: v, o_tensor: o}
        if stats:
            exec_tensors[stats_tensor] = torch.empty_strided(
                tuple(stats_tensor.get_dim()),
                tuple(stats_tensor.get_stride()),
                device=q.device,
                dtype=torch.float32,
            )
        workspace = torch.empty(
            graph.get_workspace_size(), device=q.device, dtype=torch.uint8
        )

        def run():
            graph.execute(exec_tensors, workspace, handle=self.cudnn_handle)
            return o

        return run

    def build_flag_dnn_runner(self, inputs):
        q, k, v, causal, stats = inputs

        @flag_dnn.graph
        def flag_dnn_sdpa_graph(q, k, v):
            return flag_dnn.sdpa(
                q,
                k,
                v,
                use_causal_mask=causal,
                generate_stats=stats,
                name=self.op_name,
            )

        compiled = flag_dnn.compile(
            flag_dnn_sdpa_graph,
            inputs=[
                flag_dnn.TensorSpec.from_tensor(q, "q"),
                flag_dnn.TensorSpec.from_tensor(k, "k"),
                flag_dnn.TensorSpec.from_tensor(v, "v"),
            ],
            options=consts.compile_options(),
        )
        assert [node.op_type for node in compiled.graph.nodes] == ["sdpa"]

        def run():
            return compiled.run(q, k, v)

        return run

    def transfer_bytes(self, inputs):
        q, k, v, _, stats = inputs
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
        q, k, v, causal, stats = inputs
        return [
            tuple(q.shape),
            tuple(k.shape),
            f"causal={causal}",
            f"stats={stats}",
        ]


@pytest.mark.sdpa
@pytest.mark.graph
@pytest.mark.perf
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", SdpaBenchmark.dtypes)
def test_sdpa(cudnn_handle, dtype):
    torch.manual_seed(0)
    SdpaBenchmark(cudnn_handle).run(dtype)
