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


def run_reduction_test(
    dnn_reference,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    dim: int,
    mode: str,
) -> None:
    if mode == "MUL":
        cpu = 0.9 + 0.2 * torch.rand(shape, dtype=dtype, device="cpu")
    else:
        cpu = torch.randn(shape, dtype=dtype, device="cpu")
    x = cpu.to(flag_dnn.device)
    assert dnn_reference.supports("reduction", dtype)

    expected = dnn_reference.run("reduction", x, mode, dim=dim, keepdim=True)

    @graph_decorator
    def reduction_graph(x):
        return flag_dnn.reduction(
            x,
            mode=mode,
            dim=dim,
            keepdim=True,
            compute_data_type="float32",
            name="reduction",
        )

    compiled = flag_dnn.compile(
        reduction_graph,
        inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["reduction"]
    actual = compiled.run(x)
    dnn_reference.synchronize()

    expected_shape = list(shape)
    expected_shape[dim] = 1
    assert (
        tuple(expected.shape) == tuple(actual.shape) == tuple(expected_shape)
    )
    assert expected.dtype == actual.dtype == dtype
    assert expected.device == actual.device == x.device
    atol = 8e-2 if dtype == torch.bfloat16 else 5e-2
    utils.gems_assert_close(actual, expected, dtype, atol=atol)
