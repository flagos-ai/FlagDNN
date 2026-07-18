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

from dataclasses import dataclass
from typing import Any, Optional

from flag_dnn.graph.graph import Graph


@dataclass(frozen=True)
class TensorLifetime:
    value_id: int
    birth: int
    last_use: int
    size_bytes: int


@dataclass(frozen=True)
class BufferBlock:
    id: int
    offset: int
    size_bytes: int
    alignment: int = 256


@dataclass(frozen=True)
class TensorAllocation:
    value_id: int
    buffer_id: int
    offset: int
    size_bytes: int


@dataclass(frozen=True)
class MemoryPlan:
    workspace_size: int
    lifetimes: tuple[TensorLifetime, ...]
    reusable_value_ids: tuple[int, ...]
    blocks: tuple[BufferBlock, ...] = ()
    allocations: Optional[dict[int, TensorAllocation]] = None

    def to_dict(self) -> dict:
        return {
            "workspace_size": self.workspace_size,
            "lifetimes": [
                {
                    "value_id": item.value_id,
                    "birth": item.birth,
                    "last_use": item.last_use,
                    "size_bytes": item.size_bytes,
                }
                for item in self.lifetimes
            ],
            "reusable_value_ids": list(self.reusable_value_ids),
            "blocks": [
                {
                    "id": block.id,
                    "offset": block.offset,
                    "size_bytes": block.size_bytes,
                    "alignment": block.alignment,
                }
                for block in self.blocks
            ],
            "allocations": {
                str(value_id): {
                    "value_id": allocation.value_id,
                    "buffer_id": allocation.buffer_id,
                    "offset": allocation.offset,
                    "size_bytes": allocation.size_bytes,
                }
                for value_id, allocation in (self.allocations or {}).items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryPlan":
        lifetimes = tuple(
            TensorLifetime(
                value_id=int(item["value_id"]),
                birth=int(item["birth"]),
                last_use=int(item["last_use"]),
                size_bytes=int(item["size_bytes"]),
            )
            for item in data.get("lifetimes", [])
        )
        blocks = tuple(
            BufferBlock(
                id=int(item["id"]),
                offset=int(item["offset"]),
                size_bytes=int(item["size_bytes"]),
                alignment=int(item.get("alignment", 256)),
            )
            for item in data.get("blocks", [])
        )
        allocations = {
            int(value_id): TensorAllocation(
                value_id=int(item["value_id"]),
                buffer_id=int(item["buffer_id"]),
                offset=int(item["offset"]),
                size_bytes=int(item["size_bytes"]),
            )
            for value_id, item in data.get("allocations", {}).items()
        }
        return cls(
            workspace_size=int(data.get("workspace_size", 0)),
            lifetimes=lifetimes,
            reusable_value_ids=tuple(
                int(value_id)
                for value_id in data.get("reusable_value_ids", [])
            ),
            blocks=blocks,
            allocations=allocations,
        )


def analyze_memory(graph: Graph) -> MemoryPlan:
    lifetimes = _compute_lifetimes(graph)
    workspace_size = sum(item.size_bytes for item in lifetimes)
    return MemoryPlan(
        workspace_size=workspace_size,
        lifetimes=tuple(sorted(lifetimes, key=lambda item: item.value_id)),
        reusable_value_ids=tuple(item.value_id for item in lifetimes),
    )


def allocate_memory(graph: Graph, alignment: int = 256) -> MemoryPlan:
    lifetimes = sorted(_compute_lifetimes(graph), key=lambda item: item.birth)
    blocks: list[BufferBlock] = []
    allocations: dict[int, TensorAllocation] = {}
    block_last_use: dict[int, int] = {}
    workspace_size = 0

    for lifetime in lifetimes:
        if lifetime.size_bytes <= 0:
            continue
        selected = None
        for block in blocks:
            if block.size_bytes < lifetime.size_bytes:
                continue
            if block_last_use.get(block.id, -1) < lifetime.birth:
                selected = block
                break

        if selected is None:
            offset = _align(workspace_size, alignment)
            selected = BufferBlock(
                id=len(blocks),
                offset=offset,
                size_bytes=_align(lifetime.size_bytes, alignment),
                alignment=alignment,
            )
            blocks.append(selected)
            workspace_size = selected.offset + selected.size_bytes

        block_last_use[selected.id] = lifetime.last_use
        allocations[lifetime.value_id] = TensorAllocation(
            value_id=lifetime.value_id,
            buffer_id=selected.id,
            offset=selected.offset,
            size_bytes=lifetime.size_bytes,
        )

    ordered_lifetimes = tuple(
        sorted(lifetimes, key=lambda item: item.value_id)
    )
    return MemoryPlan(
        workspace_size=workspace_size,
        lifetimes=ordered_lifetimes,
        reusable_value_ids=tuple(item.value_id for item in ordered_lifetimes),
        blocks=tuple(blocks),
        allocations=allocations,
    )


def _compute_lifetimes(graph: Graph) -> list[TensorLifetime]:
    node_index = {node.id: idx for idx, node in enumerate(graph.nodes)}
    output_set = set(graph.outputs)
    input_set = set(graph.inputs)
    lifetimes = []

    for value_id, value in graph.values.items():
        if (
            value_id in input_set
            or value_id in output_set
            or value.is_constant
        ):
            continue
        if value.producer is None:
            continue
        birth = node_index.get(value.producer)
        if birth is None:
            continue
        last_use = birth
        for user_id in value.users:
            if user_id in node_index:
                last_use = max(last_use, node_index[user_id])
        nbytes = value.spec.nbytes() or 0
        lifetimes.append(TensorLifetime(value_id, birth, last_use, nbytes))

    return lifetimes


def _align(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return ((value + alignment - 1) // alignment) * alignment
