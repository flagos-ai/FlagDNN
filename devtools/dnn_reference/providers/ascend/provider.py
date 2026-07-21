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

from typing import Any, Callable, Union

import torch

from devtools.dnn_reference.operations import RegisteredOperationProvider

from . import common as _common
from .common import (
    AscendContext,
    load_library as _load_library,
    npu_module as _default_npu_module,
)
from .ops import (
    AscendAbsOperation,
    AscendAddOperation,
    AscendBinarySelectOperation,
    AscendSigmoidBackwardOperation,
    AscendReductionOperation,
    configure_abs as _configure_abs,
    configure_add as _configure_add,
    create_binary_operations,
    create_norm_operations,
    create_unary_operations,
    create_utility_operations,
)


Number = Union[int, float]

_DTYPE_CODES = _common.DTYPE_CODES
_ERROR_BUFFER_SIZE = _common.ERROR_BUFFER_SIZE
_INT64_POINTER = _common.INT64_POINTER
_metadata_array = _common.metadata_array


def _npu_module() -> Any:
    return _default_npu_module()


def _configure_library(library: Any) -> Any:
    _configure_add(library)
    _configure_abs(library)
    return library


class AscendDnnProvider(RegisteredOperationProvider):
    vendor_name = "ascend"
    implementation = "aclnn"
    display_name = "ACLNN"

    def __init__(
        self,
        library_loader: Callable[[], Any] = _load_library,
    ) -> None:
        self._context = AscendContext(
            library_loader,
            lambda: _npu_module(),
        )
        self._set_operations(
            (
                AscendAddOperation(self._context),
                AscendAbsOperation(self._context),
                AscendBinarySelectOperation(self._context),
                AscendSigmoidBackwardOperation(self._context),
                AscendReductionOperation(self._context),
                *create_unary_operations(self._context),
                *create_binary_operations(self._context),
                *create_utility_operations(self._context),
                *create_norm_operations(self._context),
            )
        )

    @property
    def _last_device(self) -> Any:
        return self._context.last_device

    @_last_device.setter
    def _last_device(self, value: Any) -> None:
        self._context.last_device = value

    def _get_library(self) -> Any:
        return self._context.get_library(_configure_library)

    def synchronize(self) -> None:
        self._context.synchronize()

    def close(self) -> None:
        return None

    # Compatibility shims for callers written before operation registration.
    def add(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        alpha: Number = 1,
    ) -> torch.Tensor:
        return self.run("add", x, y, alpha=alpha)

    def prepare_add(self, x, y, *, alpha: Number = 1):
        return self.prepare("add", x, y, alpha=alpha)

    def abs(self, x: torch.Tensor) -> torch.Tensor:
        return self.run("abs", x)


def create_provider() -> AscendDnnProvider:
    return AscendDnnProvider()


class AscendDnnOracle(AscendDnnProvider):
    """Compatibility name for existing correctness integrations."""

    implementation = "aclnnAdd"


def create_oracle() -> AscendDnnOracle:
    return AscendDnnOracle()
