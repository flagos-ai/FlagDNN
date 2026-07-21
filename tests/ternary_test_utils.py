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

from __future__ import annotations

import torch

import flag_dnn
from flag_dnn.graph import graph as graph_decorator
from tests import accuracy_utils as utils
from tests import consts


def make_binary_select_inputs(
    case: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_shape, y_shape, mask_shape = case
    x_cpu = torch.randn(x_shape, device="cpu", dtype=dtype)
    y_cpu = torch.randn(y_shape, device="cpu", dtype=dtype)
    mask_cpu = torch.rand(mask_shape, device="cpu") > 0.5
    return (
        consts.pointwise_layout(x_cpu).to(flag_dnn.device),
        consts.pointwise_layout(y_cpu).to(flag_dnn.device),
        consts.pointwise_layout(mask_cpu).to(flag_dnn.device),
    )


def compile_binary_select_graph(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
):
    @graph_decorator
    def binary_select_graph(x, y, mask):
        return flag_dnn.binary_select(
            input0=x,
            input1=y,
            mask=mask,
            compute_data_type="float32",
            name="binary_select",
        )

    compiled = flag_dnn.compile(
        binary_select_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(x, "x"),
            flag_dnn.TensorSpec.from_tensor(y, "y"),
            flag_dnn.TensorSpec.from_tensor(mask, "mask"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["binary_select"]
    attrs = compiled.graph.nodes[0].attrs
    assert attrs["compute_data_type"] == "float32"
    assert attrs["name"] == "binary_select"
    return compiled


def run_binary_select_test(
    dnn_reference,
    dtype: torch.dtype,
    case: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ],
) -> None:
    x, y, mask = make_binary_select_inputs(case, dtype)
    assert dnn_reference.supports("binary_select", dtype)

    expected = dnn_reference.run("binary_select", x, y, mask)
    actual = compile_binary_select_graph(x, y, mask).run(x, y, mask)
    dnn_reference.synchronize()

    output_shape = tuple(torch.broadcast_shapes(x.shape, y.shape, mask.shape))
    assert tuple(expected.shape) == tuple(actual.shape) == output_shape
    assert expected.dtype == actual.dtype == dtype
    assert expected.device == actual.device == x.device
    atol = 5e-2 if dtype == torch.bfloat16 else 2e-2
    utils.gems_assert_close(actual, expected, dtype, atol=atol)
