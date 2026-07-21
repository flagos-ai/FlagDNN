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
from tests.utility_test_utils import run_single_input_utility_test


@pytest.mark.reshape
@pytest.mark.graph
@pytest.mark.parametrize("dtype", consts.DNN_COMPARE_DTYPES)
@pytest.mark.parametrize("case", consts.RESHAPE_CASES)
def test_reshape(dnn_reference, dtype, case):
    torch.manual_seed(0)
    input_shape, new_shape = case
    run_single_input_utility_test(
        dnn_reference,
        "reshape",
        dtype,
        input_shape,
        operation_kwargs={"shape": new_shape},
    )
