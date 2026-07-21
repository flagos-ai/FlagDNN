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
import torch

import flag_dnn
from tests import accuracy_utils as utils
from tests import consts


def _make_input(shape, dtype):
    cpu = consts.pointwise_layout(
        torch.randn(shape, device="cpu", dtype=dtype)
    )
    return cpu.to(flag_dnn.device)


def _run_flag_dnn_abs_graph(x):
    @flag_dnn.graph
    def flag_dnn_abs_graph(x):
        return flag_dnn.abs(
            x,
            compute_data_type="float32",
            name="abs",
        )

    compiled = flag_dnn.compile(
        flag_dnn_abs_graph,
        inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == ["abs"]
    attrs = compiled.graph.nodes[0].attrs
    assert attrs["compute_data_type"] == "float32"
    assert attrs["name"] == "abs"
    return compiled.run(x)


@pytest.mark.abs
@pytest.mark.graph
@pytest.mark.parametrize("dtype", consts.DNN_COMPARE_DTYPES)
@pytest.mark.parametrize("shape", consts.ABS_SHAPES)
def test_abs(dnn_reference, dtype, shape):
    torch.manual_seed(0)
    x = _make_input(shape, dtype)
    assert dnn_reference.supports("abs", dtype)

    expected = dnn_reference.run("abs", x)
    actual = _run_flag_dnn_abs_graph(x)
    dnn_reference.synchronize()

    assert tuple(expected.shape) == tuple(x.shape)
    assert tuple(actual.shape) == tuple(x.shape)
    assert expected.dtype == actual.dtype == dtype
    assert expected.device == actual.device == x.device
    assert tuple(expected.stride()) == tuple(x.stride())
    assert tuple(actual.stride()) == tuple(x.stride())
    utils.gems_assert_equal(actual, expected)
