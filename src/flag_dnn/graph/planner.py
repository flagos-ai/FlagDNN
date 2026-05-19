from __future__ import annotations

from typing import Any, Optional

from flag_dnn.graph.autotune import GraphAutotuner
from flag_dnn.graph.graph import Graph
from flag_dnn.graph.kernel_selector import KernelSelector
from flag_dnn.graph.plan import ExecutionPlan
from flag_dnn.graph.tensor import TensorSpec


class Planner:
    def __init__(
        self, backend: str = "auto", options: Optional[dict[str, Any]] = None
    ):
        self.backend = backend
        self.options = {} if options is None else dict(options)

    def build_plan(
        self,
        graph: Graph,
        input_specs: list[TensorSpec],
        output_structure: Any = None,
        runtime_inputs: Optional[tuple[Any, ...]] = None,
    ) -> ExecutionPlan:
        selector = KernelSelector(backend=self.backend, options=self.options)
        candidates = selector.generate_candidates(
            graph,
            input_specs,
            output_structure=output_structure,
        )
        plan = GraphAutotuner(self.options).select_best(
            candidates,
            runtime_inputs=runtime_inputs,
        )
        plan.debug_info["candidate_count"] = len(candidates)
        return plan
