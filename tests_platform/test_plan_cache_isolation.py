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

"""Regression tests for graph plan cache isolation and disk safety."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

from flag_dnn.graph.cache import PlanCache, PlanCacheKey
from flag_dnn.graph.graph import Graph
from flag_dnn.graph.plan import ExecutionPlan


def _plan(graph):
    return ExecutionPlan(
        graph_hash=graph.graph_hash(),
        plan_id="test-plan",
        graph=graph,
        steps=[],
        input_specs=[],
        output_specs=[],
        debug_info={
            "output_structure": ("tuple", [("leaf",)]),
            "validate_inputs": True,
        },
    )


def _key(graph, **options):
    return PlanCacheKey.from_graph(
        graph,
        [],
        backend="auto",
        flagdnn_version="test",
        options=options,
    )


def test_memory_cache_returns_isolated_plans():
    graph = Graph()
    key = _key(graph)
    cache = PlanCache(enable_disk=False)
    original = _plan(graph)

    cache.put(key, original)
    original.debug_info["validate_inputs"] = False

    first = cache.get(key, graph)
    second = cache.get(key, graph)

    assert first is not None
    assert second is not None
    assert first is not second
    assert first is not original
    assert first.debug_info["validate_inputs"] is True

    first.debug_info["output_structure"] = ("list", [("leaf",)])
    first.debug_info["validate_inputs"] = False

    assert second.debug_info["output_structure"] == (
        "tuple",
        [("leaf",)],
    )
    assert second.debug_info["validate_inputs"] is True


def test_plan_cache_key_includes_planner_options():
    graph = Graph()

    no_autotune = _key(graph, autotune=False)
    autotune = _key(graph, autotune=True)
    limited = _key(graph, max_plan_candidates=1)
    comprehensive = _key(graph, max_plan_candidates=8)

    assert no_autotune.digest() != autotune.digest()
    assert limited.digest() != comprehensive.digest()


def test_disk_cache_treats_corrupt_json_as_a_miss(tmp_path):
    graph = Graph()
    key = _key(graph)
    cache = PlanCache(cache_dir=str(tmp_path))
    cache.put(key, _plan(graph))
    cache.clear_memory()

    cache_path = tmp_path / f"{key.digest()}.json"
    cache_path.write_text("{not valid json", encoding="utf-8")

    assert cache.get(key, graph) is None


def test_disk_cache_treats_invalid_plan_schema_as_a_miss(tmp_path):
    graph = Graph()
    key = _key(graph)
    cache = PlanCache(cache_dir=str(tmp_path))
    cache_path = tmp_path / f"{key.digest()}.json"
    cache_path.write_text(
        json.dumps({"cache_digest": key.digest(), "plan": {}}),
        encoding="utf-8",
    )

    assert cache.get(key, graph) is None


def test_disk_cache_replaces_complete_temporary_file_atomically(
    tmp_path, monkeypatch
):
    graph = Graph()
    key = _key(graph)
    cache = PlanCache(cache_dir=str(tmp_path))
    destination = tmp_path / f"{key.digest()}.json"
    real_replace = os.replace
    replacements = []

    def checked_replace(source, target):
        source_path = Path(source)
        target_path = Path(target)
        assert source_path.parent == tmp_path
        assert source_path != destination
        assert target_path == destination
        json.loads(source_path.read_text(encoding="utf-8"))
        replacements.append((source_path, target_path))
        real_replace(source_path, target_path)

    monkeypatch.setattr("flag_dnn.graph.cache.os.replace", checked_replace)
    cache.put(key, _plan(graph))

    assert replacements
    assert destination.is_file()
    assert not list(tmp_path.glob("*.tmp"))


def test_disk_cache_concurrent_writers_leave_a_complete_entry(tmp_path):
    graph = Graph()
    key = _key(graph)

    def write_entry(index):
        plan = _plan(graph)
        plan.plan_id = f"test-plan-{index}"
        PlanCache(cache_dir=str(tmp_path)).put(key, plan)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write_entry, range(16)))

    cache_path = tmp_path / f"{key.digest()}.json"
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["cache_digest"] == key.digest()
    assert payload["plan"]["plan_id"].startswith("test-plan-")
    assert not list(tmp_path.glob("*.tmp"))

    loaded = PlanCache(cache_dir=str(tmp_path)).get(key, graph)
    assert loaded is not None


def test_disk_cache_cleans_temporary_file_after_replace_failure(
    tmp_path, monkeypatch
):
    graph = Graph()
    key = _key(graph)
    plan = _plan(graph)
    cache = PlanCache(cache_dir=str(tmp_path))

    def fail_replace(source, target):
        del source, target
        raise OSError("simulated replace failure")

    monkeypatch.setattr("flag_dnn.graph.cache.os.replace", fail_replace)
    cache.put(key, plan)

    assert plan.debug_info["disk_cache_error"] == "write_failed"
    assert not list(tmp_path.glob("*.tmp"))
    assert not (tmp_path / f"{key.digest()}.json").exists()
    assert cache.get(key, graph) is not None
