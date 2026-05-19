from __future__ import annotations

import copy
import hashlib
from typing import Any, Optional

from flag_dnn.graph.backend import KernelCandidate, resolve_backend
from flag_dnn.graph.graph import Graph
from flag_dnn.graph.memory import allocate_memory
from flag_dnn.graph.plan import ExecutionPlan, ExecutionStep
from flag_dnn.graph.tensor import TensorSpec


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
            if not node_candidates:
                node_candidates = [
                    KernelCandidate(
                        name=node.op_type,
                        backend=self.backend.name,
                        op_type=node.op_type,
                        implementation="default",
                    )
                ]
            per_node_candidates.append((node, node_candidates))

        plans = []
        for variant_index, choices in enumerate(
            _limited_product(
                [candidates for _, candidates in per_node_candidates],
                limit=int(self.options.get("max_plan_candidates", 8)),
            )
        ):
            candidate_graph = copy.deepcopy(graph)
            steps = []
            for node, candidate in zip(
                candidate_graph.topological_nodes(), choices
            ):
                attrs = dict(node.attrs)
                attrs["_implementation"] = candidate.implementation
                attrs["_kernel_candidate"] = candidate.name
                if candidate.configs:
                    attrs["_kernel_config"] = dict(candidate.configs[0])
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
                        candidate.to_debug_dict() for candidate in choices
                    ],
                    "memory": memory_plan.to_dict(),
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
    if not items:
        return [tuple()]
    results = [tuple()]
    for choices in items:
        next_results = []
        for prefix in results:
            for choice in choices:
                next_results.append(prefix + (choice,))
                if len(next_results) >= limit:
                    break
            if len(next_results) >= limit:
                break
        results = next_results
    return results


def _plan_id(
    graph_hash: str, choices: tuple[KernelCandidate, ...], variant_index: int
) -> str:
    payload = graph_hash + str(variant_index)
    payload += "".join(
        candidate.backend + candidate.name + candidate.implementation
        for candidate in choices
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
