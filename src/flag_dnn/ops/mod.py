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

from flag_dnn.ops.binary import binary


def mod(
    input: torch.Tensor | None = None,
    other: Union[torch.Tensor, int, float, None] = None,
    *,
    input0: torch.Tensor | None = None,
    input1: Union[torch.Tensor, int, float, None] = None,
    out: Optional[torch.Tensor] = None,
    compute_data_type=None,
    name: str = "",
) -> torch.Tensor:
    del compute_data_type, name
    left = input if input is not None else input0
    right = other if other is not None else input1
    if left is None:
        raise TypeError("mod missing input tensor")
    if right is None:
        raise TypeError("mod missing other operand")
    return binary(left, right, out=out, op_type="mod")
