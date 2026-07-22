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
from tests import consts
from tests.binary_test_utils import run_binary_test


@pytest.mark.cmp_lt
@pytest.mark.graph
@pytest.mark.parametrize("dtype", consts.DNN_COMPARE_DTYPES)
@pytest.mark.parametrize("case", consts.CMP_CASES)
def test_cmp_lt(dnn_reference, dtype, case):
    torch.manual_seed(0)
    run_binary_test(dnn_reference, "cmp_lt", dtype, case)


def test_cmp_lt_prepared_output_contract(dnn_reference):
    x = torch.arange(16, device=flag_dnn.device, dtype=torch.float16).reshape(
        1, 1, 16
    )
    y = torch.full_like(x, 8)
    prepared = dnn_reference.prepare("cmp_lt", x, y)
    try:
        first_output = prepared.output
        assert prepared.output.dtype == torch.bool
        assert prepared.run() is first_output
        torch.testing.assert_close(first_output, x < y)
        assert prepared.run() is first_output
    finally:
        prepared.close()
