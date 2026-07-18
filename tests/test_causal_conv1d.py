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
from tests import accuracy_utils as utils
from tests.base import get_cudnn

COMPARE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_DTYPE_TO_CUDNN_INT = {
    torch.float32: 0,
    torch.float16: 2,
    torch.bfloat16: 9,
}

_ACTIVATION_TO_CUDNN_INT = {
    "identity": 0,
    "silu": 1,
}


def _cudnn_causal_conv1d(x, weight, bias, activation):
    cudnn = get_cudnn()
    if not hasattr(cudnn, "causal_conv1d_forward"):
        pytest.skip("cuDNN frontend was built without causal_conv1d_forward")

    batch, dim, seq_len = x.shape
    kernel_size = weight.shape[1]
    out = torch.empty_like(x)
    cudnn.causal_conv1d_forward(
        torch.cuda.current_stream().cuda_stream,
        x.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        out.data_ptr(),
        batch,
        dim,
        seq_len,
        kernel_size,
        _DTYPE_TO_CUDNN_INT[x.dtype],
        _ACTIVATION_TO_CUDNN_INT[activation],
    )
    torch.cuda.synchronize()
    return out


def _run_flag_dnn_causal_conv1d_graph(x, weight, bias, activation):
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
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["causal_conv1d"]
    return compiled.run(x.clone(), weight.clone(), bias.clone())


@pytest.mark.causal_conv1d
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", COMPARE_DTYPES)
@pytest.mark.parametrize("shape_kernel", [((2, 4, 16), 3), ((3, 8, 33), 5)])
@pytest.mark.parametrize("activation", ["identity", "silu"])
def test_causal_conv1d_matches_cudnn(dtype, shape_kernel, activation):
    torch.manual_seed(0)
    shape, kernel = shape_kernel
    x = torch.randn(shape, device=flag_dnn.device, dtype=dtype)
    weight = torch.randn(
        (shape[1], kernel), device=flag_dnn.device, dtype=dtype
    )
    bias = torch.randn((shape[1],), device=flag_dnn.device, dtype=dtype)
    cudnn_out = _cudnn_causal_conv1d(x, weight, bias, activation)
    actual = _run_flag_dnn_causal_conv1d_graph(x, weight, bias, activation)
    atol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-4
    utils.gems_assert_close(actual, cudnn_out, dtype, atol=atol)
