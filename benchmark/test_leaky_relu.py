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
from benchmark.unary import UnaryBenchmark


class LeakyReluBenchmark(UnaryBenchmark):
    op_name = "leaky_relu"
    shapes = consts.LEAKY_RELU_SHAPES
    shape_ids_env = "FLAGDNN_LEAKY_RELU_PERF_SHAPE_IDS"
    legacy_shape_ids_env = "FLAGDNN_CUDNN_LEAKY_RELU_PERF_SHAPE_IDS"
    operation_kwargs = {"negative_slope": 0.2}


@pytest.mark.leaky_relu
@pytest.mark.graph
@pytest.mark.perf
@pytest.mark.parametrize("dtype", LeakyReluBenchmark.dtypes)
def test_leaky_relu(dnn_baseline, dtype):
    torch.manual_seed(0)
    LeakyReluBenchmark(dnn_baseline).run(dtype)
