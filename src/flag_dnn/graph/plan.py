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

from flag_dnn.graph.graph import Graph
from flag_dnn.graph.memory import MemoryPlan
from flag_dnn.graph.tensor import TensorSpec


@dataclass
class ExecutionStep:
    op_type: str
    inputs: list[int]
    outputs: list[int]
    attrs: dict[str, Any] = field(default_factory=dict)
    backend: str = "auto"
    kernel_name: Optional[str] = None
    workspace: int = 0

    def to_dict(self) -> dict:
        return {
            "op_type": self.op_type,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "attrs": _jsonable(self.attrs),
            "backend": self.backend,
            "kernel_name": self.kernel_name,
            "workspace": self.workspace,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionStep":
        return cls(
            op_type=data["op_type"],
            inputs=list(data["inputs"]),
            outputs=list(data["outputs"]),
            attrs=dict(data.get("attrs", {})),
            backend=data.get("backend", "auto"),
            kernel_name=data.get("kernel_name"),
            workspace=int(data.get("workspace", 0)),
        )


@dataclass
class ExecutionPlan:
    graph_hash: str
    plan_id: str
    graph: Graph
    steps: list[ExecutionStep]
    input_specs: list[TensorSpec]
    output_specs: list[TensorSpec]
    workspace_size: int = 0
    memory_plan: Optional[MemoryPlan] = None
    tuned: bool = False
    debug_info: dict[str, Any] = field(default_factory=dict)

    def run(self, *inputs: Any) -> Any:
        from flag_dnn.graph.executor import execute_plan

        return execute_plan(self, inputs)

    def to_dict(self) -> dict:
        return {
            "graph_hash": self.graph_hash,
            "plan_id": self.plan_id,
            "steps": [step.to_dict() for step in self.steps],
            "input_specs": [_spec_to_dict(spec) for spec in self.input_specs],
            "output_specs": [
                _spec_to_dict(spec) for spec in self.output_specs
            ],
            "workspace_size": self.workspace_size,
            "memory_plan": (
                None
                if self.memory_plan is None
                else self.memory_plan.to_dict()
            ),
            "tuned": self.tuned,
            "debug_info": _jsonable(self.debug_info),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], graph: Graph) -> "ExecutionPlan":
        return cls(
            graph_hash=data["graph_hash"],
            plan_id=data["plan_id"],
            graph=graph,
            steps=[
                ExecutionStep.from_dict(step) for step in data.get("steps", [])
            ],
            input_specs=[
                _spec_from_dict(spec) for spec in data.get("input_specs", [])
            ],
            output_specs=[
                _spec_from_dict(spec) for spec in data.get("output_specs", [])
            ],
            workspace_size=int(data.get("workspace_size", 0)),
            memory_plan=(
                None
                if data.get("memory_plan") is None
                else MemoryPlan.from_dict(data["memory_plan"])
            ),
            tuned=bool(data.get("tuned", False)),
            debug_info=dict(data.get("debug_info", {})),
        )


def _spec_to_dict(spec: TensorSpec) -> dict:
    data = spec.signature()
    data["name"] = spec.name
    data["is_input"] = spec.is_input
    data["is_output"] = spec.is_output
    return data


def _spec_from_dict(data: dict[str, Any]) -> TensorSpec:
    return TensorSpec(
        name=data.get("name", ""),
        shape=tuple(data.get("shape", ())),
        dtype=data.get("dtype", "float32"),
        stride=data.get("stride"),
        layout=data.get("layout"),
        device=data.get("device"),
        is_input=bool(data.get("is_input", False)),
        is_output=bool(data.get("is_output", False)),
        contiguous=data.get("contiguous"),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
