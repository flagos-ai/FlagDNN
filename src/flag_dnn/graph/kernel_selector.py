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

import copy
import hashlib
import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

from flag_dnn.graph.backend import KernelCandidate, resolve_backend
from flag_dnn.graph.graph import Graph
from flag_dnn.graph.memory import allocate_memory
from flag_dnn.graph.plan import ExecutionPlan, ExecutionStep
from flag_dnn.graph.tensor import TensorSpec


@dataclass(frozen=True)
class _CandidateVariant:
    candidate: KernelCandidate
    config: Optional[dict[str, Any]] = None

    def to_debug_dict(self) -> dict[str, Any]:
        data = self.candidate.to_debug_dict()
        data["selected_config"] = (
            None if self.config is None else dict(self.config)
        )
        return data


class KernelSelector:
    def __init__(
        self, backend: str = "auto", options: Optional[dict[str, Any]] = None
    ):
        self.requested_backend = backend
        self.options = {} if options is None else dict(options)
        self.backend = resolve_backend(backend)

    def generate_candidates(
        self,
        graph: Graph,
        input_specs: list[TensorSpec],
        output_structure: Any = None,
    ) -> list[ExecutionPlan]:
        per_node_candidates = []
        for node in graph.topological_nodes():
            node_candidates = self.backend.candidates_for_node(
                node, graph, input_specs
            )
            per_node_candidates.append(
                (node, _expand_candidate_configs(node_candidates))
            )

        plans = []
        for variant_index, choices in enumerate(
            _limited_product(
                [candidates for _, candidates in per_node_candidates],
                limit=int(self.options.get("max_plan_candidates", 8)),
            )
        ):
            candidate_graph = copy.deepcopy(graph)
            steps = []
            for node, variant in zip(
                candidate_graph.topological_nodes(), choices
            ):
                candidate = variant.candidate
                attrs = dict(node.attrs)
                attrs["_implementation"] = candidate.implementation
                attrs["_kernel_candidate"] = candidate.name
                if variant.config is not None:
                    attrs["_kernel_config"] = dict(variant.config)
                steps.append(
                    ExecutionStep(
                        op_type=node.op_type,
                        inputs=list(node.inputs),
                        outputs=list(node.outputs),
                        attrs=attrs,
                        backend=candidate.backend,
                        kernel_name=candidate.name,
                        workspace=0,
                    )
                )
            memory_plan = allocate_memory(candidate_graph)
            graph_hash = candidate_graph.graph_hash()
            plan_id = _plan_id(graph_hash, choices, variant_index)
            output_specs = [
                candidate_graph.values[value_id].spec
                for value_id in candidate_graph.outputs
            ]
            plan = ExecutionPlan(
                graph_hash=graph_hash,
                plan_id=plan_id,
                graph=candidate_graph,
                steps=steps,
                input_specs=input_specs,
                output_specs=output_specs,
                workspace_size=memory_plan.workspace_size,
                memory_plan=memory_plan,
                tuned=False,
                debug_info={
                    "backend": self.backend.name,
                    "requested_backend": self.requested_backend,
                    "cache_hit": False,
                    "output_structure": output_structure,
                    "num_steps": len(steps),
                    "fusion": dict(candidate_graph.attrs.get("fusion", {})),
                    "kernel_candidates": [
                        variant.to_debug_dict() for variant in choices
                    ],
                    "memory": memory_plan.to_dict(),
                    "uses_workspace": False,
                    "selector": "kernel_selector_v1",
                },
            )
            plans.append(plan)

        plans.sort(
            key=lambda plan: sum(
                item.get("priority", 100)
                for item in plan.debug_info.get("kernel_candidates", [])
            )
        )
        return plans


def _limited_product(
    items: list[list[Any]], limit: int
) -> list[tuple[Any, ...]]:
    if limit <= 0:
        raise ValueError("max_plan_candidates must be positive")
    if not items:
        return [tuple()]
    if any(not choices for choices in items):
        return []

    initial = tuple(0 for _ in items)
    pending = deque([initial])
    visited = {initial}
    results: list[tuple[Any, ...]] = []
    while pending and len(results) < limit:
        indices = pending.popleft()
        results.append(
            tuple(choices[index] for choices, index in zip(items, indices))
        )
        for dimension, choices in enumerate(items):
            if indices[dimension] + 1 >= len(choices):
                continue
            next_indices = list(indices)
            next_indices[dimension] += 1
            next_key = tuple(next_indices)
            if next_key not in visited:
                visited.add(next_key)
                pending.append(next_key)
    return results


def _expand_candidate_configs(
    candidates: list[KernelCandidate],
) -> list[_CandidateVariant]:
    variants: list[_CandidateVariant] = []
    for candidate in sorted(candidates, key=lambda item: item.priority):
        if candidate.configs:
            variants.extend(
                _CandidateVariant(candidate, dict(config))
                for config in candidate.configs
            )
        else:
            variants.append(_CandidateVariant(candidate))
    return variants


def _plan_id(
    graph_hash: str,
    choices: tuple[_CandidateVariant, ...],
    variant_index: int,
) -> str:
    payload = graph_hash + str(variant_index)
    payload += "".join(
        variant.candidate.backend
        + variant.candidate.name
        + variant.candidate.implementation
        + json.dumps(
            variant.config,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        for variant in choices
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
