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
from tests.unary_test_utils import run_unary_test


@pytest.mark.logical_not
@pytest.mark.graph
@pytest.mark.parametrize("shape", consts.POINTWISE_UNARY_SHAPES)
def test_logical_not(dnn_reference, shape):
    torch.manual_seed(0)
    run_unary_test(
        dnn_reference,
        "logical_not",
        torch.bool,
        shape,
        domain="logical",
        exact=True,
    )
