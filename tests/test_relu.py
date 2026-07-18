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

NEGATIVE_SLOPE = 0.2
LOWER_CLIP = -0.25
UPPER_CLIP = 1.0


def _cudnn_relu(x, cudnn_handle):
    graph = cudnn_graph(x.dtype, cudnn_handle)
    x_tensor = graph.tensor_like(x)
    y_tensor = graph.relu(
        input=x_tensor,
        negative_slope=NEGATIVE_SLOPE,
        lower_clip=LOWER_CLIP,
        upper_clip=UPPER_CLIP,
        compute_data_type=cudnn.data_type.FLOAT,
        name="relu",
    )
    return execute_cudnn_graph(
        graph,
        {x_tensor: x},
        y_tensor,
        torch.empty_like(x),
        cudnn_handle,
        "relu",
    )


def _run_flag_dnn_relu_graph(x):
    @flag_dnn.graph
    def flag_dnn_relu_graph(x):
        return flag_dnn.relu(
            x,
            negative_slope=NEGATIVE_SLOPE,
            lower_clip=LOWER_CLIP,
            upper_clip=UPPER_CLIP,
            compute_data_type="float32",
            name="relu",
        )

    compiled = flag_dnn.compile(
        flag_dnn_relu_graph,
        inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["relu"]
    attrs = compiled.graph.nodes[0].attrs
    assert attrs["negative_slope"] == NEGATIVE_SLOPE
    assert attrs["lower_clip"] == LOWER_CLIP
    assert attrs["upper_clip"] == UPPER_CLIP
    return compiled.run(x.clone())


@pytest.mark.relu
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("shape", consts.RELU_SHAPES)
def test_relu_attrs(cudnn_handle, dtype, shape):
    torch.manual_seed(0)
    x = consts.pointwise_randn(shape, dtype, flag_dnn.device)

    cudnn_out = _cudnn_relu(x, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_relu_graph(x)

    atol = 5e-2 if dtype == torch.bfloat16 else 2e-2
    utils.gems_assert_close(flag_dnn_out, cudnn_out, dtype, atol=atol)
