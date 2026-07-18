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

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Union

if TYPE_CHECKING:
    import torch


Number = Union[int, float]


class PreparedOperation(Protocol):
    output: torch.Tensor

    def run(self) -> torch.Tensor: ...

    def close(self) -> None: ...


class DnnProvider(Protocol):
    vendor_name: str
    implementation: str
    display_name: str

    def add(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        alpha: Number = 1,
    ) -> torch.Tensor: ...

    def abs(self, x: torch.Tensor) -> torch.Tensor: ...

    def prepare_add(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        alpha: Number = 1,
    ) -> PreparedOperation: ...

    def supports_dtype(self, dtype: torch.dtype) -> bool: ...

    def synchronize(self) -> None: ...

    def close(self) -> None: ...
