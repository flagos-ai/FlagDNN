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

from tests import consts
from tests.norm_test_utils import (
    run_batchnorm_test,
    run_layernorm_test,
    run_rmsnorm_test,
)


@pytest.mark.layernorm
@pytest.mark.graph
@pytest.mark.parametrize("dtype", consts.DNN_COMPARE_DTYPES)
def test_layernorm_multi_output(dnn_reference, dtype):
    run_layernorm_test(dnn_reference, dtype)


@pytest.mark.rmsnorm
@pytest.mark.graph
@pytest.mark.parametrize("dtype", consts.DNN_COMPARE_DTYPES)
def test_rmsnorm_multi_output(dnn_reference, dtype):
    run_rmsnorm_test(dnn_reference, dtype)


@pytest.mark.batchnorm
@pytest.mark.graph
@pytest.mark.parametrize("dtype", consts.DNN_COMPARE_DTYPES)
def test_batchnorm_multi_output(dnn_reference, dtype):
    run_batchnorm_test(dnn_reference, dtype)
