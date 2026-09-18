"""Immutable execution-plan objects for the mthreads compiler provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ArgumentKind = Literal["tensor", "workspace", "scalar_i32", "scalar_f32"]


@dataclass(frozen=True)
class RuntimeArgument:
    kind: ArgumentKind
    semantic_name: str
    uid: int | None
    scalar_bits: str | None

    def to_json(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "semantic_name": self.semantic_name,
            "uid": self.uid,
            "scalar_bits": self.scalar_bits,
        }


@dataclass(frozen=True)
class KernelVariant:
    variant_id: str
    source: str
    source_sha256: str
    function: str
    full_signature: str
    grid: tuple[int, int, int]
    num_warps: int
    num_stages: int
    arguments: tuple[RuntimeArgument, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "variant_id": self.variant_id,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "function": self.function,
            "full_signature": self.full_signature,
            "grid": list(self.grid),
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "arguments": [argument.to_json() for argument in self.arguments],
        }


@dataclass(frozen=True)
class AutotuneSpec:
    enabled: bool
    warmup: int
    repetitions: int
    selection_cache: str

    def to_json(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "warmup": self.warmup,
            "repetitions": self.repetitions,
            "selection_cache": self.selection_cache,
        }


@dataclass(frozen=True)
class KernelStage:
    id: int
    node_id: int
    operation: str
    dependencies: tuple[int, ...]
    source: str
    function: str
    variants: tuple[KernelVariant, ...]
    autotune: AutotuneSpec

    def to_json(self) -> dict[str, object]:
        return {
            "id": self.id,
            "node_id": self.node_id,
            "operation": self.operation,
            "dependencies": list(self.dependencies),
            "source": self.source,
            "function": self.function,
            "variants": [variant.to_json() for variant in self.variants],
            "autotune": self.autotune.to_json(),
        }


@dataclass(frozen=True)
class ExecutionPlan:
    stages: tuple[KernelStage, ...]
    external_binding_uids: tuple[int, ...]
    workspace_size: int
    workspace_alignment: int
