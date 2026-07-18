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


def _cudnn_slice(x, slices, cudnn_handle):
    graph = cudnn_graph(x.dtype, cudnn_handle)
    x_tensor = graph.tensor_like(x)
    y_tensor = graph.slice(
        input=x_tensor,
        slices=list(slices),
        compute_data_type=cudnn.data_type.FLOAT,
        name="slice",
    )
    output_template = torch.empty(
        tuple(x[tuple(slices)].shape),
        device=x.device,
        dtype=x.dtype,
    )
    return execute_cudnn_graph(
        graph,
        {x_tensor: x},
        y_tensor,
        output_template,
        cudnn_handle,
        "slice",
    )


def _run_flag_dnn_slice_graph(x, slices):
    @flag_dnn.graph
    def flag_dnn_slice_graph(x):
        return flag_dnn.slice(
            x,
            slices,
            compute_data_type="float32",
            name="slice",
        )

    compiled = flag_dnn.compile(
        flag_dnn_slice_graph,
        inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["slice"]
    return compiled.run(x.clone())


@pytest.mark.slice
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("case", consts.SLICE_CASES)
def test_slice(cudnn_handle, dtype, case):
    torch.manual_seed(0)
    shape, slices = case
    x = consts.pointwise_randn(shape, dtype, flag_dnn.device)

    cudnn_out = _cudnn_slice(x, slices, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_slice_graph(x, slices)

    utils.gems_assert_equal(flag_dnn_out, cudnn_out)
