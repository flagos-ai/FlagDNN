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

import torch

from ..common import DTYPE_CODES, AscendContext
from .binary import AscendBinaryOperation
from .unary import AscendUnaryOperation


class PreparedAscendSigmoidBackward:
    def __init__(self, sigmoid, backward) -> None:
        self._sigmoid = sigmoid
        self._backward = backward
        self.output = backward.output
        self._closed = False

    def run(self) -> torch.Tensor:
        if self._closed:
            raise RuntimeError(
                "prepared aclnn sigmoid_backward runner is closed"
            )
        self._sigmoid.run()
        return self._backward.run()

    def __call__(self) -> torch.Tensor:
        return self.run()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: Exception | None = None
        for prepared in (self._backward, self._sigmoid):
            try:
                prepared.close()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


class AscendSigmoidBackwardOperation:
    name = "sigmoid_backward"

    def __init__(self, context: AscendContext) -> None:
        self._context = context
        self._sigmoid = AscendUnaryOperation("sigmoid", context)
        self._backward = AscendBinaryOperation(
            "_sigmoid_backward_output", context
        )

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return dtype in DTYPE_CODES

    def _validate(
        self,
        loss: torch.Tensor,
        input: torch.Tensor,
    ) -> None:
        if not isinstance(loss, torch.Tensor) or not isinstance(
            input, torch.Tensor
        ):
            raise TypeError(
                "aclnn sigmoid_backward expects loss and input tensors"
            )
        if loss.shape != input.shape:
            raise ValueError(
                "aclnn sigmoid_backward requires equal input shapes, "
                f"got loss={tuple(loss.shape)}, input={tuple(input.shape)}"
            )
        if loss.dtype != input.dtype:
            raise TypeError("aclnn sigmoid_backward inputs must share a dtype")
        if loss.device != input.device:
            raise ValueError("aclnn sigmoid_backward inputs must share an NPU")
        if not self.supports_dtype(input.dtype):
            raise TypeError(
                f"aclnn sigmoid_backward does not support {input.dtype}"
            )

    def run(
        self,
        loss: torch.Tensor,
        input: torch.Tensor,
    ) -> torch.Tensor:
        prepared = self.prepare(loss, input)
        try:
            return prepared.run()
        finally:
            prepared.close()

    def prepare(
        self,
        loss: torch.Tensor,
        input: torch.Tensor,
    ) -> PreparedAscendSigmoidBackward:
        self._validate(loss, input)
        sigmoid = self._sigmoid.prepare(input)
        try:
            backward = self._backward.prepare(loss, sigmoid.output)
        except Exception:
            sigmoid.close()
            raise
        return PreparedAscendSigmoidBackward(sigmoid, backward)
