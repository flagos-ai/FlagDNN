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

import torch
import pytest

import flag_dnn
from tests import accuracy_utils as utils
from tests import consts


def _make_input(shape, dtype):
    cpu = consts.pointwise_layout(
        torch.randn(shape, device="cpu", dtype=dtype)
    )
    return cpu.to(flag_dnn.device)


def _run_flag_dnn_add_graph(x, y, alpha=1):
    @flag_dnn.graph
    def flag_dnn_add_graph(x, y):
        return flag_dnn.add(
            x,
            y,
            alpha=alpha,
            compute_data_type="float32",
            name="add",
        )

    compiled = flag_dnn.compile(
        flag_dnn_add_graph,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(x, "x"),
            flag_dnn.TensorSpec.from_tensor(y, "y"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["add"]
    attrs = compiled.graph.nodes[0].attrs
    assert attrs["alpha"] == alpha
    assert attrs["compute_data_type"] == "float32"
    assert attrs["name"] == "add"
    return compiled.run(x, y)


def _assert_add_matches_reference(
    dnn_reference, dtype, x_shape, y_shape, alpha
):
    torch.manual_seed(0)
    x = _make_input(x_shape, dtype)
    y = _make_input(y_shape, dtype)
    assert dnn_reference.supports("add", dtype)

    expected = dnn_reference.run("add", x, y, alpha=alpha)
    actual = _run_flag_dnn_add_graph(x, y, alpha=alpha)
    dnn_reference.synchronize()

    output_shape = torch.broadcast_shapes(tuple(x.shape), tuple(y.shape))
    assert tuple(expected.shape) == output_shape
    assert tuple(actual.shape) == output_shape
    assert expected.dtype == actual.dtype == dtype
    assert expected.device == actual.device == x.device

    atol = 5e-2 if dtype == torch.bfloat16 else 2e-2
    utils.gems_assert_close(actual, expected, dtype, atol=atol)


@pytest.mark.add
@pytest.mark.graph
@pytest.mark.parametrize("dtype", consts.DNN_COMPARE_DTYPES)
@pytest.mark.parametrize("case", consts.get_add_test_cases())
def test_add(dnn_reference, dtype, case):
    x_shape, y_shape = case
    _assert_add_matches_reference(
        dnn_reference,
        dtype,
        x_shape,
        y_shape,
        alpha=1,
    )


@pytest.mark.add
@pytest.mark.graph
@pytest.mark.parametrize("dtype", consts.DNN_COMPARE_DTYPES)
@pytest.mark.parametrize("alpha", (0.5, -2.0))
def test_add_alpha(dnn_reference, dtype, alpha):
    _assert_add_matches_reference(
        dnn_reference,
        dtype,
        (2, 4, 8),
        (2, 4, 8),
        alpha=alpha,
    )
