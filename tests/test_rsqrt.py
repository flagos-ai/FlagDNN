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
from tests.base import (
    CUDNN_COMPARE_DTYPES,
    cudnn,
    cudnn_graph,
    execute_cudnn_graph,
)
import torch

import flag_dnn
from tests import accuracy_utils as utils
from tests import consts


def _cudnn_rsqrt(x, cudnn_handle):
    graph = cudnn_graph(x.dtype, cudnn_handle)
    x_tensor = graph.tensor_like(x)
    y_tensor = graph.rsqrt(
        input=x_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
        name="rsqrt",
    )
    return execute_cudnn_graph(
        graph,
        {x_tensor: x},
        y_tensor,
        torch.empty_like(x),
        cudnn_handle,
        "rsqrt",
    )


def _run_flag_dnn_rsqrt_graph(x):
    @flag_dnn.graph
    def flag_dnn_rsqrt_graph(x):
        return flag_dnn.rsqrt(
            x,
            compute_data_type="float32",
            name="rsqrt",
        )

    compiled = flag_dnn.compile(
        flag_dnn_rsqrt_graph,
        inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["rsqrt"]
    assert compiled.graph.nodes[0].attrs["compute_data_type"] == "float32"
    assert compiled.graph.nodes[0].attrs["name"] == "rsqrt"
    return compiled.run(x.clone())


@pytest.mark.rsqrt
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("shape", consts.RSQRT_SHAPES)
def test_rsqrt(cudnn_handle, dtype, shape):
    torch.manual_seed(0)
    x = consts.pointwise_positive(shape, dtype, flag_dnn.device)

    cudnn_out = _cudnn_rsqrt(x, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_rsqrt_graph(x)

    atol = 5e-2 if dtype == torch.bfloat16 else 2e-2
    utils.gems_assert_close(flag_dnn_out, cudnn_out, dtype, atol=atol)
