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

from flag_dnn.ops.hardtanh import hardtanh as hardtanh_op


def hardtanh_(
    input: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0
) -> torch.Tensor:
    return hardtanh_op(input, min_val, max_val, inplace=True)
