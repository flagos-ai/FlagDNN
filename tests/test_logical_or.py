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
from tests.base import cudnn, cudnn_graph, execute_cudnn_graph
import torch

import flag_dnn
from tests import consts
from tests import accuracy_utils as utils


def _cudnn_logical_or(x, y, cudnn_handle):
    graph = cudnn_graph(x.dtype, cudnn_handle)
    x_tensor = graph.tensor_like(x)
    y_tensor = graph.tensor_like(y)
    out_tensor = graph.logical_or(
        a=x_tensor,
        b=y_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
        name="logical_or",
    )
    output_shape = torch.broadcast_shapes(tuple(x.shape), tuple(y.shape))
    output_template = torch.empty(output_shape, device=x.device, dtype=x.dtype)
    return execute_cudnn_graph(
        graph,
        {x_tensor: x, y_tensor: y},
        out_tensor,
        output_template,
        cudnn_handle,
        "logical_or",
    )


def _run_flag_dnn_logical_or_graph(x, y):
    @flag_dnn.graph
    def flag_dnn_logical_or_graph(x, y):
        return flag_dnn.logical_or(
            a=x,
            b=y,
            compute_data_type="float32",
            name="logical_or",
        )

    compiled = flag_dnn.compile(
        flag_dnn_logical_or_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(x, "x"),
            flag_dnn.TensorSpec.from_tensor(y, "y"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["logical_or"]
    assert compiled.graph.nodes[0].attrs["compute_data_type"] == "float32"
    assert compiled.graph.nodes[0].attrs["name"] == "logical_or"
    return compiled.run(x.clone(), y.clone())


@pytest.mark.logical_or
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("case", consts.LOGICAL_CASES)
def test_logical_or(cudnn_handle, case):
    torch.manual_seed(0)
    x_shape, y_shape = case
    x = consts.pointwise_bool(x_shape, flag_dnn.device)
    y = consts.pointwise_bool(y_shape, flag_dnn.device)

    cudnn_out = _cudnn_logical_or(x, y, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_logical_or_graph(x, y)

    utils.gems_assert_equal(flag_dnn_out, cudnn_out)


@pytest.mark.logical_or
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_logical_or_single_case_matches_cudnn(cudnn_handle):
    torch.manual_seed(0)
    x_shape, y_shape = consts.LOGICAL_CASES[0]
    x = consts.pointwise_bool(x_shape, flag_dnn.device)
    y = consts.pointwise_bool(y_shape, flag_dnn.device)

    cudnn_out = _cudnn_logical_or(x, y, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_logical_or_graph(x, y)
    utils.gems_assert_equal(flag_dnn_out, cudnn_out)
