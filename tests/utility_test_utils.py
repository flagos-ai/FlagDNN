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

from typing import Any, Sequence

import torch

import flag_dnn
from flag_dnn.graph import graph as graph_decorator
from tests import accuracy_utils as utils
from tests import consts


def make_input(shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
    cpu = consts.pointwise_layout(
        torch.randn(tuple(shape), device="cpu", dtype=dtype)
    )
    return cpu.to(flag_dnn.device)


def _compile_single_input_graph(
    op_name: str,
    x: torch.Tensor,
    operation_kwargs: dict[str, Any],
):
    op = getattr(flag_dnn, op_name)
    call_kwargs = dict(operation_kwargs)
    call_kwargs["name"] = op_name
    if op_name not in ("reshape",):
        call_kwargs["compute_data_type"] = "float32"

    @graph_decorator
    def utility_graph(x):
        return op(x, **call_kwargs)

    compiled = flag_dnn.compile(
        utility_graph,
        inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == [op_name]
    return compiled


def run_single_input_utility_test(
    dnn_reference,
    op_name: str,
    dtype: torch.dtype,
    shape: Sequence[int],
    *,
    operation_kwargs: dict[str, Any] | None = None,
) -> None:
    x = make_input(shape, dtype)
    kwargs = dict(operation_kwargs or {})
    assert dnn_reference.supports(op_name, dtype)

    expected = dnn_reference.run(op_name, x, **kwargs)
    actual = _compile_single_input_graph(op_name, x, kwargs).run(x)
    dnn_reference.synchronize()

    assert tuple(expected.shape) == tuple(actual.shape)
    assert expected.dtype == actual.dtype == dtype
    assert expected.device == actual.device == x.device
    utils.gems_assert_equal(actual, expected)


def run_concatenate_test(
    dnn_reference,
    dtype: torch.dtype,
    shapes: Sequence[Sequence[int]],
    axis: int,
) -> None:
    inputs = tuple(make_input(shape, dtype) for shape in shapes)
    assert dnn_reference.supports("concatenate", dtype)

    expected = dnn_reference.run("concatenate", inputs, axis=axis)

    @graph_decorator
    def concatenate_graph(*values):
        return flag_dnn.concatenate(
            list(values), axis=axis, name="concatenate"
        )

    compiled = flag_dnn.compile(
        concatenate_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(item, f"x{index}")
            for index, item in enumerate(inputs)
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["concatenate"]
    actual = compiled.run(*inputs)
    dnn_reference.synchronize()

    assert tuple(expected.shape) == tuple(actual.shape)
    assert expected.dtype == actual.dtype == dtype
    assert expected.device == actual.device == inputs[0].device
    utils.gems_assert_equal(actual, expected)


def run_gen_index_test(
    dnn_reference,
    dtype: torch.dtype,
    shape: Sequence[int],
    axis: int,
) -> None:
    x = torch.empty(tuple(shape), device=flag_dnn.device, dtype=dtype)
    assert dnn_reference.supports("gen_index", dtype)
    expected = dnn_reference.run(
        "gen_index", x, axis=axis, compute_data_type=dtype
    )

    @graph_decorator
    def gen_index_graph(x):
        return flag_dnn.gen_index(
            x,
            axis=axis,
            compute_data_type=x.dtype,
            name="gen_index",
        )

    compiled = flag_dnn.compile(
        gen_index_graph,
        inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["gen_index"]
    actual = compiled.run(x)
    dnn_reference.synchronize()

    assert tuple(expected.shape) == tuple(actual.shape) == tuple(x.shape)
    assert expected.dtype == actual.dtype == dtype
    assert expected.device == actual.device == x.device
    utils.gems_assert_equal(actual, expected)
