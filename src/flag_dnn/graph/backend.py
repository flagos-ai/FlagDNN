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


class TorchFallbackBackend(GraphBackend):
    name = "torch"

    def capability(self) -> BackendCapability:
        return BackendCapability(
            name=self.name,
            supports_autotune=True,
            notes=("CPU/CUDA semantic fallback backend",),
        )


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
                        "fallback": "composite_if_unsupported",
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
                KernelCandidate(
                    name="fused_conv2d_bias_relu.composite",
                    backend=self.name,
                    op_type=node.op_type,
                    implementation="composite",
                    priority=50,
                    supported_dtypes=(
                        "float16",
                        "bfloat16",
                        "float32",
                        "float64",
                    ),
                    constraints={"fallback": "conv2d_bias_then_activation"},
                ),
            ]
        return super().candidates_for_node(node, graph, input_specs)


def resolve_backend(name: str) -> GraphBackend:
    if name in ("auto", "triton_cuda", "cuda", "triton"):
        backend = TritonCudaBackend()
        if backend.is_available() or name != "auto":
            return backend
    return TorchFallbackBackend()
