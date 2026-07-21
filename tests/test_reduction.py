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
from tests.reduction_test_utils import run_reduction_test


@pytest.mark.reduction
@pytest.mark.graph
@pytest.mark.parametrize("dtype", consts.DNN_COMPARE_DTYPES)
@pytest.mark.parametrize("shape,dim,mode", consts.REDUCTION_CASES)
def test_reduction(dnn_reference, dtype, shape, dim, mode):
    torch.manual_seed(0)
    run_reduction_test(dnn_reference, dtype, shape, dim, mode)
