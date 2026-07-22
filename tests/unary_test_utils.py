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
import pytest

import flag_dnn
from devtools.dnn_reference.interfaces import DnnReferenceNotSupportedError
from flag_dnn.graph import graph as graph_decorator
from tests import accuracy_utils as utils
from tests import consts


UnaryInputDomain = Literal["real", "positive", "scaled", "tan", "logical"]


def make_unary_input(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    domain: UnaryInputDomain = "real",
) -> torch.Tensor:
    if domain == "logical":
        cpu = torch.rand(shape, device="cpu") > 0.5
    elif domain == "positive":
        cpu = torch.rand(shape, device="cpu", dtype=dtype) + 0.5
    elif domain == "scaled":
        cpu = torch.randn(shape, device="cpu", dtype=dtype) * 4.0
    elif domain == "tan":
        cpu = torch.rand(shape, device="cpu", dtype=dtype) - 0.5
    elif domain == "real":
        cpu = torch.randn(shape, device="cpu", dtype=dtype)
    else:
        raise ValueError(f"unsupported unary input domain: {domain}")
    return consts.pointwise_layout(cpu).to(flag_dnn.device)


def compile_unary_graph(
    op_name: str,
    x: torch.Tensor,
    operation_kwargs: dict[str, object] | None = None,
):
    op = getattr(flag_dnn, op_name)
    call_kwargs = dict(operation_kwargs or {})
    if op_name != "gelu":
        call_kwargs.update(
            compute_data_type="float32",
            name=op_name,
        )

    @graph_decorator
    def unary_graph(x):
        return op(x, **call_kwargs)

    compiled = flag_dnn.compile(
        unary_graph,
        inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == [op_name]
    attrs = compiled.graph.nodes[0].attrs
    if op_name != "gelu":
        assert attrs["compute_data_type"] == "float32"
        assert attrs["name"] == op_name
    for key, value in (operation_kwargs or {}).items():
        assert attrs[key] == value
    return compiled


def run_unary_test(
    dnn_reference,
    op_name: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    *,
    domain: UnaryInputDomain = "real",
    exact: bool = False,
    operation_kwargs: dict[str, object] | None = None,
) -> None:
    x = make_unary_input(shape, dtype, domain)
    assert dnn_reference.supports(op_name, dtype)

    kwargs = operation_kwargs or {}
    try:
        expected = dnn_reference.run(op_name, x, **kwargs)
    except DnnReferenceNotSupportedError as exc:
        pytest.skip(str(exc))
    actual = compile_unary_graph(op_name, x, operation_kwargs=kwargs).run(x)
    dnn_reference.synchronize()

    assert tuple(expected.shape) == tuple(actual.shape) == tuple(x.shape)
    assert expected.dtype == actual.dtype == dtype
    assert expected.device == actual.device == x.device
    if exact:
        utils.gems_assert_equal(actual, expected)
    else:
        atol = 5e-2 if dtype == torch.bfloat16 else 2e-2
        utils.gems_assert_close(actual, expected, dtype, atol=atol)
