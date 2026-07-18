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

from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from flag_dnn.graph.tensor import TensorSpec


@dataclass
class GraphValue:
    id: int
    spec: TensorSpec
    producer: Optional[int] = None
    users: list[int] = field(default_factory=list)
    const_value: Any = None
    is_constant: bool = False

    def to_dict(self, include_const: bool = False) -> dict:
        spec = self.spec.signature()
        spec["name"] = self.spec.name
        data = {
            "id": self.id,
            "spec": spec,
            "producer": self.producer,
            "users": list(self.users),
            "is_constant": self.is_constant,
        }
        if include_const and self.is_constant:
            data["const"] = _const_signature(self.const_value)
        return data


@dataclass
class OpNode:
    id: int
    op_type: str
    inputs: list[int]
    outputs: list[int]
    attrs: dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "op_type": self.op_type,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "attrs": _jsonable_attrs(self.attrs),
            "name": self.name,
        }


def _jsonable_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in attrs.items():
        if isinstance(value, tuple):
            result[key] = list(value)
        elif isinstance(value, list):
            result[key] = value
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        else:
            result[key] = str(value)
    return result


def _const_signature(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": tuple(value.shape),
            "stride": tuple(value.stride()),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "data_ptr": (
                int(value.data_ptr()) if value.device.type != "meta" else 0
            ),
            "version": int(getattr(value, "_version", 0)),
            "id": id(value),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {"kind": "scalar", "value": value}
    return {"kind": "repr", "value": repr(value)}
