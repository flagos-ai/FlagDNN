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

from typing import Union

import torch

from devtools.dnn_reference.operations import RegisteredOperationProvider

from .nvidia_ops import (
    NvidiaAbsOperation,
    NvidiaAddOperation,
    NvidiaBinarySelectOperation,
    NvidiaBatchNormInferenceOperation,
    NvidiaCausalConv1dOperation,
    NvidiaRmsNormRhtAmaxOperation,
    NvidiaSigmoidBackwardOperation,
    NvidiaReductionOperation,
    NvidiaContext,
    create_binary_operations,
    create_norm_operations,
    create_unary_operations,
    create_utility_operations,
)


Number = Union[int, float]


class NvidiaDnnProvider(RegisteredOperationProvider):
    vendor_name = "nvidia"
    implementation = "cudnn"
    display_name = "cuDNN"

    def __init__(self) -> None:
        self._context = NvidiaContext()
        self._set_operations(
            (
                NvidiaAddOperation(self._context),
                NvidiaAbsOperation(self._context),
                NvidiaBinarySelectOperation(self._context),
                NvidiaSigmoidBackwardOperation(self._context),
                NvidiaReductionOperation(self._context),
                NvidiaCausalConv1dOperation(self._context),
                NvidiaRmsNormRhtAmaxOperation(self._context),
                NvidiaBatchNormInferenceOperation(self._context),
                *create_unary_operations(self._context),
                *create_binary_operations(self._context),
                *create_utility_operations(self._context),
                *create_norm_operations(self._context),
            )
        )

    def synchronize(self) -> None:
        self._context.synchronize()

    def close(self) -> None:
        self._context.close()

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


def create_provider() -> NvidiaDnnProvider:
    return NvidiaDnnProvider()


class NvidiaDnnOracle(NvidiaDnnProvider):
    """Compatibility name for existing correctness integrations."""


def create_oracle() -> NvidiaDnnOracle:
    return NvidiaDnnOracle()
