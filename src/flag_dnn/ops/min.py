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

from typing import Optional, Union

import torch

from flag_dnn.ops.minimum import minimum


def min(
    input0: torch.Tensor | None = None,
    input1: Union[torch.Tensor, int, float, bool, None] = None,
    *,
    input: torch.Tensor | None = None,
    other: Union[torch.Tensor, int, float, bool, None] = None,
    a: torch.Tensor | None = None,
    b: Union[torch.Tensor, int, float, bool, None] = None,
    out: Optional[torch.Tensor] = None,
    compute_data_type=None,
    name: str = "",
) -> torch.Tensor:
    del compute_data_type, name
    left = input0 if input0 is not None else input
    left = left if left is not None else a
    right = input1 if input1 is not None else other
    right = right if right is not None else b
    if left is None or right is None:
        raise TypeError("min expects input0/input1 tensors")
    return minimum(left, right, out=out)
