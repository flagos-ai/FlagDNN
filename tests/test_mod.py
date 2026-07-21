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
from tests.binary_test_utils import (
    assert_binary_values,
    run_binary_test,
)


@pytest.mark.mod
@pytest.mark.graph
@pytest.mark.parametrize("dtype", consts.DNN_COMPARE_DTYPES)
@pytest.mark.parametrize("case", consts.MOD_CASES)
def test_mod(dnn_reference, dtype, case):
    torch.manual_seed(0)
    run_binary_test(dnn_reference, "mod", dtype, case, domain="mod")


@pytest.mark.mod
@pytest.mark.graph
@pytest.mark.parametrize("dtype", consts.DNN_COMPARE_DTYPES)
def test_mod_signed(dnn_reference, dtype):
    x = torch.tensor(
        [-3.0, -3.0, 3.0, 3.0, -5.5, 5.5],
        dtype=dtype,
        device=flag_dnn.device,
    ).reshape(1, 1, 6)
    y = torch.tensor(
        [2.0, -2.0, 2.0, -2.0, 2.25, -2.25],
        dtype=dtype,
        device=flag_dnn.device,
    ).reshape(1, 1, 6)
    assert_binary_values(dnn_reference, "mod", x, y)
