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

from flag_dnn.ops.bitwise_or import bitwise_or


def _check_bool_pair(
    input: torch.Tensor, other: Union[torch.Tensor, bool], op_name: str
) -> None:
    if input.dtype != torch.bool:
        raise RuntimeError(f"{op_name} expects bool input tensors")
    if isinstance(other, torch.Tensor) and other.dtype != torch.bool:
        raise RuntimeError(f"{op_name} expects bool input tensors")


def logical_or(
    input: torch.Tensor | None = None,
    other: Union[torch.Tensor, bool, None] = None,
    *,
    a: torch.Tensor | None = None,
    b: Union[torch.Tensor, bool, None] = None,
    out: Optional[torch.Tensor] = None,
    compute_data_type=None,
    name: str = "",
) -> torch.Tensor:
    del compute_data_type, name
    left = input if input is not None else a
    right = other if other is not None else b
    if left is None or right is None:
        raise TypeError("logical_or expects input/other or a/b")
    _check_bool_pair(left, right, "logical_or")
    return bitwise_or(left, right, out=out)
