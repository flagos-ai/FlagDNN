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

import pytest

from flag_dnn.graph.backend import KernelCandidate, TritonCudaBackend
from flag_dnn.graph.graph import Graph
from flag_dnn.graph.kernel_selector import KernelSelector, _limited_product
from flag_dnn.graph.tensor import TensorSpec


def test_limited_product_covers_each_dimension_before_deeper_variants():
    assert _limited_product([["a", "b"], ["x", "y"]], limit=2) == [
        ("a", "x"),
        ("b", "x"),
    ]


def test_limited_product_rejects_non_positive_limit():
    with pytest.raises(ValueError, match="positive"):
        _limited_product([[1]], limit=0)


def test_kernel_selector_expands_candidate_configs():
    graph = Graph()
    left = graph.add_input(TensorSpec("left", (4,), "float32"))
    right = graph.add_input(TensorSpec("right", (4,), "float32"))
    output = graph.add_op(
        "add",
        [left.value_id, right.value_id],
        {"op_type": "add", "alpha": 1, "rounding_mode": None},
    )
    graph.mark_outputs([output.value_id])

    candidate = KernelCandidate(
        name="configured-add",
        backend="test",
        op_type="add",
        implementation="test",
        configs=({"BLOCK": 64}, {"BLOCK": 128}),
    )

    class Backend:
        name = "test"

        def candidates_for_node(self, node, current_graph, input_specs):
            del node, current_graph, input_specs
            return [candidate]

    specs = [graph.values[value_id].spec for value_id in graph.inputs]
    selector = KernelSelector(options={"max_plan_candidates": 8})
    selector.backend = Backend()
    plans = selector.generate_candidates(graph, specs)

    assert [plan.steps[0].attrs["_kernel_config"] for plan in plans] == [
        {"BLOCK": 64},
        {"BLOCK": 128},
    ]
    assert len({plan.plan_id for plan in plans}) == 2


def test_cuda_backend_requires_every_input_to_be_cuda(monkeypatch):
    backend = TritonCudaBackend()
    monkeypatch.setattr(backend, "is_available", lambda: True)
    graph = Graph()
    cuda = TensorSpec("cuda", (1,), "float32", device="cuda:0")
    cpu = TensorSpec("cpu", (1,), "float32", device="cpu")
    unspecified = TensorSpec("unspecified", (1,), "float32", device=None)

    assert backend.supports(graph, [cuda, unspecified])
    assert not backend.supports(graph, [cuda, cpu])
    assert not backend.supports(graph, [])
