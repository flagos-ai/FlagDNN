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

from typing import Any, Optional, Sequence

import torch

from flag_dnn import runtime
from flag_dnn.graph.device import is_runtime_device_tensor
from flag_dnn.graph.tensor import TensorSpec


def _require_runtime_backend(inputs: Sequence[Any], op_type: str) -> None:
    tensor_inputs = [
        value for value in inputs if isinstance(value, torch.Tensor)
    ]
    if not tensor_inputs or not all(
        is_runtime_device_tensor(value) for value in tensor_inputs
    ):
        raise NotImplementedError(
            f"FlagDNN graph {op_type} requires runtime device tensors; "
            "torch fallback is disabled"
        )


def _unsupported_triton_path(op_type: str, detail: str) -> None:
    raise NotImplementedError(
        f"FlagDNN graph {op_type} has no Triton path for {detail}; "
        "torch fallback is disabled"
    )


def _static_shape(spec: TensorSpec) -> Optional[tuple[int, ...]]:
    shape = tuple(spec.shape)
    if not all(isinstance(dim, int) for dim in shape):
        return None
    return tuple(int(dim) for dim in shape)


def _is_runtime_device_spec(spec: TensorSpec) -> bool:
    if spec.device is None:
        return False
    device = str(spec.device)
    runtime_name = runtime.device.name
    return device == runtime_name or device.startswith(runtime_name + ":")
