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
from tests.unary_test_utils import run_unary_test


@pytest.mark.tanh
@pytest.mark.graph
@pytest.mark.parametrize("dtype", consts.DNN_COMPARE_DTYPES)
@pytest.mark.parametrize("shape", consts.TANH_SHAPES)
def test_tanh(dnn_reference, dtype, shape):
    torch.manual_seed(0)
    run_unary_test(dnn_reference, "tanh", dtype, shape)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", consts.DNN_COMPARE_DTYPES)
def test_tanh_preserves_channels_last_layout(dtype):
    x = torch.randn((4, 16, 64, 128), device=flag_dnn.device, dtype=dtype)
    x = x.contiguous(memory_format=torch.channels_last)

    actual = flag_dnn.tanh(x)

    assert actual.stride() == x.stride()
    torch.testing.assert_close(actual, torch.tanh(x))
