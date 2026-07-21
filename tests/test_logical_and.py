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
from tests.binary_test_utils import run_binary_test


@pytest.mark.logical_and
@pytest.mark.graph
@pytest.mark.parametrize("case", consts.LOGICAL_CASES)
def test_logical_and(dnn_reference, case):
    torch.manual_seed(0)
    run_binary_test(
        dnn_reference,
        "logical_and",
        torch.bool,
        case,
        domain="logical",
    )
