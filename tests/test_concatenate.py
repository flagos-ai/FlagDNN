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
    cudnn_graph,
    execute_cudnn_graph,
)
import torch

import flag_dnn
from tests import consts
from tests import accuracy_utils as utils


def _cudnn_concatenate(inputs, axis, cudnn_handle):
    graph = cudnn_graph(inputs[0].dtype, cudnn_handle)
    input_tensors = [graph.tensor_like(item) for item in inputs]
    y_tensor = graph.concatenate(
        inputs=input_tensors,
        axis=axis,
        name="concatenate",
    )
    ref = torch.cat(tuple(inputs), dim=axis)
    output_template = torch.empty(
        tuple(ref.shape),
        device=inputs[0].device,
        dtype=inputs[0].dtype,
    )
    return execute_cudnn_graph(
        graph,
        dict(zip(input_tensors, inputs)),
        y_tensor,
        output_template,
        cudnn_handle,
        "concatenate",
    )


def _run_flag_dnn_concatenate_graph(inputs, axis):
    @flag_dnn.graph
    def flag_dnn_concatenate_graph(*values):
        return flag_dnn.concatenate(
            list(values),
            axis=axis,
            name="concatenate",
        )

    compiled = flag_dnn.compile(
        flag_dnn_concatenate_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(item, f"x{index}")
            for index, item in enumerate(inputs)
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["concatenate"]
    return compiled.run(*(item.clone() for item in inputs))


@pytest.mark.concatenate
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", CUDNN_COMPARE_DTYPES)
@pytest.mark.parametrize("case", consts.CONCATENATE_CASES)
def test_concatenate(cudnn_handle, dtype, case):
    torch.manual_seed(0)
    shapes, axis = case
    inputs = [
        consts.pointwise_randn(shape, dtype, flag_dnn.device)
        for shape in shapes
    ]

    cudnn_out = _cudnn_concatenate(inputs, axis, cudnn_handle)
    flag_dnn_out = _run_flag_dnn_concatenate_graph(inputs, axis)

    utils.gems_assert_equal(flag_dnn_out, cudnn_out)
