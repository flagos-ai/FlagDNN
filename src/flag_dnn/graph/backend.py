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

from flag_dnn.graph.graph import Graph
from flag_dnn.graph.node import OpNode
from flag_dnn.graph.tensor import TensorSpec


@dataclass(frozen=True)
class BackendCapability:
    name: str
    supports_workspace: bool = True
    supports_autotune: bool = True
    supports_persistent_cache: bool = True
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class KernelCandidate:
    name: str
    backend: str
    op_type: str
    implementation: str
    priority: int = 100
    supported_dtypes: tuple[str, ...] = ()
    supported_layouts: tuple[str, ...] = ()
    constraints: dict[str, Any] = field(default_factory=dict)
    configs: tuple[dict[str, Any], ...] = ()
    estimated_cost: Optional[float] = None

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "op_type": self.op_type,
            "implementation": self.implementation,
            "priority": self.priority,
            "supported_dtypes": list(self.supported_dtypes),
            "supported_layouts": list(self.supported_layouts),
            "constraints": dict(self.constraints),
            "configs": [dict(config) for config in self.configs],
            "estimated_cost": self.estimated_cost,
        }


class GraphBackend:
    name = "base"

    def capability(self) -> BackendCapability:
        return BackendCapability(name=self.name)

    def is_available(self) -> bool:
        return True

    def supports(self, graph: Graph, input_specs: list[TensorSpec]) -> bool:
        return self.is_available()

    def candidates_for_node(
        self, node: OpNode, graph: Graph, input_specs: list[TensorSpec]
    ) -> list[KernelCandidate]:
        return [
            KernelCandidate(
                name=node.op_type,
                backend=self.name,
                op_type=node.op_type,
                implementation="default",
                priority=100,
            )
        ]


class TritonCudaBackend(GraphBackend):
    name = "triton_cuda"

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def capability(self) -> BackendCapability:
        return BackendCapability(
            name=self.name,
            notes=("FlagDNN Triton kernels",),
        )

    def supports(self, graph: Graph, input_specs: list[TensorSpec]) -> bool:
        if not self.is_available():
            return False
        return any(
            spec.device is None
            or spec.device == "cuda"
            or str(spec.device).startswith("cuda:")
            for spec in input_specs
        )

    def candidates_for_node(
        self, node: OpNode, graph: Graph, input_specs: list[TensorSpec]
    ) -> list[KernelCandidate]:
        if node.op_type == "fused_conv2d_bias_relu":
            return [
                KernelCandidate(
                    name="fused_conv2d_bias_relu.triton",
                    backend=self.name,
                    op_type=node.op_type,
                    implementation="triton_fused",
                    priority=0,
                    supported_dtypes=("float16", "bfloat16", "float32"),
                    supported_layouts=("contiguous", "nhwc", "strided"),
                    constraints={
                        "groups": 1,
                        "activation": "relu",
                    },
                    configs=(
                        {
                            "BLOCK_M": 64,
                            "BLOCK_N": 32,
                            "BLOCK_K": 32,
                            "GROUP_M": 8,
                        },
                    ),
                ),
            ]
        return super().candidates_for_node(node, graph, input_specs)


class TritonAscendBackend(GraphBackend):
    name = "triton_ascend"

    def is_available(self) -> bool:
        npu = getattr(torch, "npu", None)
        if npu is None:
            return False
        is_available = getattr(npu, "is_available", None)
        if not callable(is_available):
            return False
        try:
            return bool(is_available())
        except Exception:
            return False

    def capability(self) -> BackendCapability:
        return BackendCapability(
            name=self.name,
            notes=("FlagDNN Triton kernels compiled by FlagTree for Ascend",),
        )

    def supports(self, graph: Graph, input_specs: list[TensorSpec]) -> bool:
        if not self.is_available() or not input_specs:
            return False
        return all(
            spec.device is None
            or spec.device == "npu"
            or str(spec.device).startswith("npu:")
            for spec in input_specs
        )


class AutoBackend(GraphBackend):
    name = "auto"

    def __init__(self) -> None:
        self.backends: tuple[GraphBackend, ...] = (
            TritonCudaBackend(),
            TritonAscendBackend(),
        )

    def is_available(self) -> bool:
        return any(backend.is_available() for backend in self.backends)

    def capability(self) -> BackendCapability:
        return BackendCapability(
            name=self.name,
            notes=("Composite backend candidate selector",),
        )

    def supports(self, graph: Graph, input_specs: list[TensorSpec]) -> bool:
        return any(
            backend.supports(graph, input_specs) for backend in self.backends
        )

    def candidates_for_node(
        self, node: OpNode, graph: Graph, input_specs: list[TensorSpec]
    ) -> list[KernelCandidate]:
        candidates: list[KernelCandidate] = []
        for backend in self.backends:
            if not backend.supports(graph, input_specs):
                continue
            candidates.extend(
                backend.candidates_for_node(node, graph, input_specs)
            )
        candidates.sort(key=lambda candidate: candidate.priority)
        return candidates


def resolve_backend(name: str) -> GraphBackend:
    if name == "auto":
        return AutoBackend()
    if name in ("triton_cuda", "cuda", "triton"):
        return TritonCudaBackend()
    if name in ("triton_ascend", "ascend", "npu"):
        return TritonAscendBackend()
    if name == "torch":
        raise ValueError("FlagDNN graph no longer supports torch fallback")
    raise ValueError(f"unsupported FlagDNN graph backend: {name}")
