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

import pytest
import torch

import flag_dnn
from benchmark import consts
from benchmark.base import CudnnCompareBenchmark, get_cudnn

_DTYPE_TO_CUDNN_INT = {
    torch.float32: 0,
    torch.float16: 2,
    torch.bfloat16: 9,
}

_ACTIVATION_TO_CUDNN_INT = {
    "identity": 0,
    "silu": 1,
}


class CausalConv1dBenchmark(CudnnCompareBenchmark):
    op_name = "causal_conv1d"
    shapes = consts.CAUSAL_CONV1D_SHAPES
    shape_ids_env = "FLAGDNN_CUDNN_CAUSAL_CONV1D_PERF_SHAPE_IDS"

    def make_inputs(self, shape, dtype):
        batch, dim, seq_len, kernel_size, activation = shape
        x = torch.randn(
            (batch, dim, seq_len), device=flag_dnn.device, dtype=dtype
        ).contiguous()
        weight = torch.randn(
            (dim, kernel_size), device=flag_dnn.device, dtype=dtype
        ).contiguous()
        bias = torch.randn((dim,), device=flag_dnn.device, dtype=dtype)
        return x, weight, bias, activation

    def build_cudnn_runner(self, inputs):
        cudnn = get_cudnn()
        x, weight, bias, activation = inputs
        if not hasattr(cudnn, "causal_conv1d_forward"):
            pytest.skip(
                "cuDNN frontend was built without causal_conv1d_forward"
            )

        batch, dim, seq_len = x.shape
        kernel_size = weight.shape[1]
        y = torch.empty_like(x)
        dtype = _DTYPE_TO_CUDNN_INT[x.dtype]
        activation_int = _ACTIVATION_TO_CUDNN_INT[activation]

        def run():
            cudnn.causal_conv1d_forward(
                torch.cuda.current_stream().cuda_stream,
                x.data_ptr(),
                weight.data_ptr(),
                bias.data_ptr(),
                y.data_ptr(),
                batch,
                dim,
                seq_len,
                kernel_size,
                dtype,
                activation_int,
            )
            return y

        return run

    def build_flag_dnn_runner(self, inputs):
        x, weight, bias, activation = inputs

        @flag_dnn.graph
        def flag_dnn_causal_conv1d_graph(x, weight, bias):
            return flag_dnn.causal_conv1d(
                x, weight, bias=bias, activation=activation
            )

        compiled = flag_dnn.compile(
            flag_dnn_causal_conv1d_graph,
            inputs=[
                flag_dnn.TensorSpec.from_tensor(x, "x"),
                flag_dnn.TensorSpec.from_tensor(weight, "weight"),
                flag_dnn.TensorSpec.from_tensor(bias, "bias"),
            ],
            options=consts.compile_options(),
        )
        assert [node.op_type for node in compiled.graph.nodes] == [
            "causal_conv1d"
        ]

        def run():
            return compiled.run(x, weight, bias)

        return run

    def transfer_bytes(self, inputs):
        x, weight, bias, _ = inputs
        return (
            x.numel() * x.element_size()
            + weight.numel() * weight.element_size()
            + bias.numel() * bias.element_size()
            + x.numel() * x.element_size()
        )

    def shape_detail(self, inputs):
        x, weight, _, activation = inputs
        return [
            tuple(x.shape),
            tuple(weight.shape),
            f"activation={activation}",
        ]


@pytest.mark.causal_conv1d
@pytest.mark.graph
@pytest.mark.perf
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CausalConv1dBenchmark.dtypes)
def test_causal_conv1d(cudnn_handle, dtype):
    torch.manual_seed(0)
    CausalConv1dBenchmark(cudnn_handle).run(dtype)
