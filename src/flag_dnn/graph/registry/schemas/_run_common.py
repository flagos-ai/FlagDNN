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

"""Shared helpers for graph op eager-fallback run functions.

A run function is the correctness fallback executed at replay when no prepared
fast path applies; it calls the eager op. These helpers are used across op
families and live here so each family module stays self-contained.
"""

from __future__ import annotations

from typing import Any

import torch

from flag_dnn.graph.device import is_runtime_device_tensor


def _runtime_backend_available(inputs: list[Any]) -> bool:
    tensor_inputs = [
        value for value in inputs if isinstance(value, torch.Tensor)
    ]
    return bool(tensor_inputs) and all(
        is_runtime_device_tensor(value) for value in tensor_inputs
    )


def _require_runtime_backend(inputs: list[Any], op_type: str) -> None:
    if not _runtime_backend_available(inputs):
        raise NotImplementedError(
            f"FlagDNN graph {op_type} requires runtime device tensors; "
            "torch fallback is disabled"
        )


def _unsupported_triton_path(op_type: str, detail: str) -> None:
    raise NotImplementedError(
        f"FlagDNN graph {op_type} has no Triton path for {detail}; "
        "torch fallback is disabled"
    )


def _format_bias(
    input_tensor: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    if bias.dim() == 1 and input_tensor.dim() >= 2:
        shape = [1] * input_tensor.dim()
        shape[1] = bias.numel()
        return bias.reshape(shape)
    return bias


def _public_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in attrs.items() if not key.startswith("_")
    }


__all__ = (
    "_runtime_backend_available",
    "_require_runtime_backend",
    "_unsupported_triton_path",
    "_format_bias",
    "_public_attrs",
)
