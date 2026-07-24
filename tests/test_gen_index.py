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

from tests import consts
from tests.utility_test_utils import run_gen_index_test


@pytest.mark.gen_index
@pytest.mark.graph
@pytest.mark.parametrize("dtype", consts.DNN_COMPARE_DTYPES)
@pytest.mark.parametrize("case", consts.GEN_INDEX_CASES)
def test_gen_index(dnn_reference, dtype, case):
    torch.manual_seed(0)
    shape, axis = case
    run_gen_index_test(dnn_reference, dtype, shape, axis)


@pytest.mark.gen_index
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gen_index_large_axis_two_exact_values():
    import flag_dnn

    x = torch.empty((32, 128, 256), device=flag_dnn.device)
    expected = torch.arange(256, device=x.device, dtype=torch.float32)
    expected = expected.reshape(1, 1, 256).expand_as(x)

    @flag_dnn.graph
    def gen_index_graph(x):
        return flag_dnn.gen_index(
            x,
            2,
            compute_data_type="float32",
            name="gen_index",
        )

    compiled = flag_dnn.compile(
        gen_index_graph,
        inputs=[flag_dnn.TensorSpec.from_tensor(x, "x")],
        options={"cache": None},
    )
    actual = compiled.run(x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
