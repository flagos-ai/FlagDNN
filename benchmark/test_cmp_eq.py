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

from benchmark import consts
from benchmark.binary import BinaryBenchmark


class CmpEqBenchmark(BinaryBenchmark):
    op_name = "cmp_eq"
    shapes = consts.CMP_SHAPES
    shape_ids_env = "FLAGDNN_CMP_EQ_PERF_SHAPE_IDS"
    legacy_shape_ids_env = "FLAGDNN_CUDNN_CMP_EQ_PERF_SHAPE_IDS"


@pytest.mark.cmp_eq
@pytest.mark.graph
@pytest.mark.perf
@pytest.mark.parametrize("dtype", CmpEqBenchmark.dtypes)
def test_cmp_eq(dnn_baseline, dtype):
    torch.manual_seed(0)
    CmpEqBenchmark(dnn_baseline).run(dtype)
