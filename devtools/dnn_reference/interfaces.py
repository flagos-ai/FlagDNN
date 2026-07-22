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

from typing import Any, TYPE_CHECKING, Protocol, Union

if TYPE_CHECKING:
    import torch


Number = Union[int, float]


class DnnProviderUnavailableError(RuntimeError):
    """Raised when a known provider cannot initialize its runtime."""


class DnnReferenceNotSupportedError(RuntimeError):
    """Raised when the selected DNN library cannot build an operation."""


class PreparedOperation(Protocol):
    output: Any

    def run(self) -> Any: ...

    def close(self) -> None: ...


class DnnReferenceOperation(Protocol):
    """One vendor DNN implementation registered under a FlagDNN op name."""

    name: str

    def supports_dtype(self, dtype: torch.dtype) -> bool: ...

    def run(self, *args: Any, **kwargs: Any) -> Any: ...

    def prepare(self, *args: Any, **kwargs: Any) -> PreparedOperation: ...


class DnnProvider(Protocol):
    vendor_name: str
    implementation: str
    display_name: str

    @property
    def operation_names(self) -> tuple[str, ...]: ...

    def get_operation(self, op_name: str) -> DnnReferenceOperation: ...

    def supports(self, op_name: str, dtype: torch.dtype) -> bool: ...

    def run(self, op_name: str, *args: Any, **kwargs: Any) -> Any: ...

    def prepare(
        self, op_name: str, *args: Any, **kwargs: Any
    ) -> PreparedOperation: ...

    def synchronize(self) -> None: ...

    def close(self) -> None: ...
