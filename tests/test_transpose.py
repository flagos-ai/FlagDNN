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
from tests import consts
from tests import accuracy_utils as utils


def _cudnn_transpose(x, permutation, cudnn_handle):
    graph = cudnn_graph(x.dtype, cudnn_handle)
    x_tensor = graph.tensor_like(x)
    y_tensor = graph.transpose(
        input=x_tensor,
        permutation=list(permutation),
        compute_data_type=cudnn.data_type.FLOAT,
        name="transpose",
    )
    output_template = torch.empty(
        tuple(torch.permute(x, permutation).shape),
        device=x.device,
        dtype=x.dtype,
    )
    return execute_cudnn_graph(
        graph,
        {x_tensor: x},
        y_tensor,
        output_template,
        cudnn_handle,
        "transpose",
    )


def _run_flag_dnn_transpose_graph(x, permutation):
    @flag_dnn.graph
    def flag_dnn_transpose_graph(x):
        return flag_dnn.transpose(
            x,
            permutation,
            compute_data_type="float32",
            name="transpose",
        )

    compiled = flag_dnn.compile(
        flag_dnn_transpose_graph,
        inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["transpose"]
    return compiled.run(x.clone())


@pytest.mark.transpose
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("case", consts.TRANSPOSE_CASES)
def test_transpose(cudnn_handle, dtype, case):
    torch.manual_seed(0)
    shape, permutation = case
    x = consts.pointwise_randn(shape, dtype, flag_dnn.device)

    cudnn_out = _cudnn_transpose(x, permutation, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_transpose_graph(x, permutation)

    utils.gems_assert_equal(flag_dnn_out, cudnn_out)
