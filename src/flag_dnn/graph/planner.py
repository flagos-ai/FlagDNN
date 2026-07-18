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
