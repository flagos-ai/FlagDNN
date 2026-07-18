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


def _make_inputs(case, dtype):
    loss_shape, input_shape = case
    loss = consts.pointwise_randn(loss_shape, dtype, flag_dnn.device)
    x = consts.pointwise_randn(input_shape, dtype, flag_dnn.device)
    return loss, x


def _cudnn_sigmoid_backward(loss, x, cudnn_handle):
    graph = cudnn_graph(x.dtype, cudnn_handle)
    loss_tensor = graph.tensor_like(loss)
    x_tensor = graph.tensor_like(x)
    dx_tensor = graph.sigmoid_backward(
        loss=loss_tensor,
        input=x_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
        name="sigmoid_backward",
    )
    return execute_cudnn_graph(
        graph,
        {loss_tensor: loss, x_tensor: x},
        dx_tensor,
        torch.empty_like(x),
        cudnn_handle,
        "sigmoid_backward",
    )


def _run_flag_dnn_sigmoid_backward_graph(loss, x):
    @flag_dnn.graph
    def flag_dnn_sigmoid_backward_graph(loss, x):
        return flag_dnn.sigmoid_backward(
            loss,
            x,
            compute_data_type="float32",
            name="sigmoid_backward",
        )

    compiled = flag_dnn.compile(
        flag_dnn_sigmoid_backward_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(loss, "loss"),
            flag_dnn.TensorSpec.from_tensor(x, "x"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == [
        "sigmoid_backward"
    ]
    assert compiled.graph.nodes[0].attrs["compute_data_type"] == "float32"
    assert compiled.graph.nodes[0].attrs["name"] == "sigmoid_backward"
    return compiled.run(loss.clone(), x.clone())


@pytest.mark.sigmoid_backward
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("case", consts.SIGMOID_BACKWARD_CASES)
def test_sigmoid_backward(cudnn_handle, dtype, case):
    torch.manual_seed(0)
    loss, x = _make_inputs(case, dtype)

    cudnn_out = _cudnn_sigmoid_backward(loss, x, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_sigmoid_backward_graph(loss, x)

    atol = 5e-2 if dtype == torch.bfloat16 else 2e-2
    utils.gems_assert_close(flag_dnn_out, cudnn_out, dtype, atol=atol)
