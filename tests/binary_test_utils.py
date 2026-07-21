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

from typing import Literal

import torch

import flag_dnn
from flag_dnn.graph import graph as graph_decorator
from tests import accuracy_utils as utils
from tests import consts


BinaryInputDomain = Literal["real", "divisor", "pow", "mod", "logical"]

_EXACT_OPERATIONS = {
    "cmp_eq",
    "cmp_neq",
    "cmp_gt",
    "cmp_ge",
    "cmp_lt",
    "cmp_le",
    "logical_and",
    "logical_or",
}


def make_binary_inputs(
    case: tuple[tuple[int, ...], tuple[int, ...]],
    dtype: torch.dtype,
    domain: BinaryInputDomain = "real",
) -> tuple[torch.Tensor, torch.Tensor]:
    x_shape, y_shape = case
    if domain == "logical":
        x_cpu = torch.rand(x_shape, device="cpu") > 0.5
        y_cpu = torch.rand(y_shape, device="cpu") > 0.5
    elif domain == "pow":
        x_cpu = torch.rand(x_shape, device="cpu", dtype=dtype) + 0.5
        y_cpu = torch.rand(y_shape, device="cpu", dtype=dtype) * 2.0
    elif domain in ("divisor", "mod"):
        x_cpu = torch.randn(x_shape, device="cpu", dtype=dtype)
        y_cpu = torch.rand(y_shape, device="cpu", dtype=dtype) + 0.5
    elif domain == "real":
        x_cpu = torch.randn(x_shape, device="cpu", dtype=dtype)
        y_cpu = torch.randn(y_shape, device="cpu", dtype=dtype)
    else:
        raise ValueError(f"unsupported binary input domain: {domain}")
    return (
        consts.pointwise_layout(x_cpu).to(flag_dnn.device),
        consts.pointwise_layout(y_cpu).to(flag_dnn.device),
    )


def compile_binary_graph(
    op_name: str,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    alpha: float = 1,
):
    op = getattr(flag_dnn, op_name)

    @graph_decorator
    def binary_graph(x, y):
        kwargs = {
            "compute_data_type": "float32",
            "name": op_name,
        }
        if op_name == "sub":
            kwargs["alpha"] = alpha
        return op(x, y, **kwargs)

    compiled = flag_dnn.compile(
        binary_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(x, "x"),
            flag_dnn.TensorSpec.from_tensor(y, "y"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == [op_name]
    return compiled


def assert_binary_values(
    dnn_reference,
    op_name: str,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    alpha: float = 1,
) -> None:
    provider_kwargs = {"alpha": alpha} if op_name == "sub" else {}
    expected = dnn_reference.run(op_name, x, y, **provider_kwargs)
    actual = compile_binary_graph(op_name, x, y, alpha=alpha).run(x, y)
    dnn_reference.synchronize()

    output_shape = tuple(torch.broadcast_shapes(x.shape, y.shape))
    output_dtype = torch.bool if op_name in _EXACT_OPERATIONS else x.dtype
    assert tuple(expected.shape) == tuple(actual.shape) == output_shape
    assert expected.dtype == actual.dtype == output_dtype
    assert expected.device == actual.device == x.device
    if op_name in _EXACT_OPERATIONS:
        utils.gems_assert_equal(actual, expected)
    else:
        atol = 5e-2 if x.dtype == torch.bfloat16 else 2e-2
        utils.gems_assert_close(actual, expected, x.dtype, atol=atol)


def run_binary_test(
    dnn_reference,
    op_name: str,
    dtype: torch.dtype,
    case: tuple[tuple[int, ...], tuple[int, ...]],
    *,
    domain: BinaryInputDomain = "real",
    alpha: float = 1,
) -> None:
    x, y = make_binary_inputs(case, dtype, domain)
    assert dnn_reference.supports(op_name, dtype)
    assert_binary_values(dnn_reference, op_name, x, y, alpha=alpha)
