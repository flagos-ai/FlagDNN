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

import torch

from flag_dnn.ops.leaky_relu import leaky_relu as leaky_relu_op


def leaky_relu_(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    return leaky_relu_op(x, negative_slope=negative_slope, inplace=True)
