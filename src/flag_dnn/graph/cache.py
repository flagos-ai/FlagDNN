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
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from flag_dnn.graph.graph import Graph
from flag_dnn.graph.plan import ExecutionPlan
from flag_dnn.graph.tensor import TensorSpec


@dataclass(frozen=True)
class PlanCacheKey:
    graph_hash: str
    shapes: tuple
    dtypes: tuple
    strides: tuple
    layouts: tuple
    devices: tuple
    backend: str
    triton_version: str
    flagdnn_version: str
    flags: tuple
    options: tuple

    @classmethod
    def from_graph(
        cls,
        graph: Graph,
        input_specs: list[TensorSpec],
        backend: str,
        flagdnn_version: str,
        options: Optional[Mapping[str, Any]] = None,
    ) -> "PlanCacheKey":
        return cls(
            graph_hash=graph.graph_hash(),
            shapes=tuple(tuple(spec.shape) for spec in input_specs),
            dtypes=tuple(spec.dtype for spec in input_specs),
            strides=tuple(spec.stride for spec in input_specs),
            layouts=tuple(spec.layout for spec in input_specs),
            devices=tuple(spec.device for spec in input_specs),
            backend=backend,
            triton_version=_triton_version(),
            flagdnn_version=flagdnn_version,
            flags=(
                (
                    "FLAGDNN_GRAPH_FUSION",
                    os.getenv("FLAGDNN_GRAPH_FUSION", "1"),
                ),
                ("FLAGDNN_AUTOTUNE", os.getenv("FLAGDNN_AUTOTUNE", "0")),
                (
                    "FLAGDNN_AUTOTUNE_WARMUP",
                    os.getenv("FLAGDNN_AUTOTUNE_WARMUP", "1"),
                ),
                (
                    "FLAGDNN_AUTOTUNE_REPEAT",
                    os.getenv("FLAGDNN_AUTOTUNE_REPEAT", "5"),
                ),
            ),
            options=_planner_option_signature(options),
        )

    def digest(self) -> str:
        encoded = json.dumps(
            self.__dict__, sort_keys=True, default=str, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class PlanCache:
    def __init__(
        self, cache_dir: Optional[str] = None, enable_disk: bool = True
    ):
        self._memory: dict[str, ExecutionPlan] = {}
        self._lock = threading.RLock()
        self.enable_disk = enable_disk
        if cache_dir is None:
            root = os.getenv("FLAGDNN_PLAN_CACHE_DIR")
            if root is None:
                xdg = os.getenv("XDG_CACHE_HOME")
                root = (
                    str(Path(xdg) / "flag_dnn" / "plans")
                    if xdg
                    else str(Path.home() / ".cache" / "flag_dnn" / "plans")
                )
            cache_dir = root
        self.cache_dir = Path(cache_dir)

    def get(self, key: PlanCacheKey, graph: Graph) -> Optional[ExecutionPlan]:
        digest = key.digest()
        with self._lock:
            cached_plan = self._memory.get(digest)
        if cached_plan is not None:
            plan = _clone_plan(cached_plan, graph)
            plan.debug_info["cache_hit"] = True
            plan.debug_info["cache_layer"] = "memory"
            return plan
        if not self.enable_disk:
            return None
        path = self._path(digest)
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                return None
            if data.get("cache_digest") != digest:
                return None
            plan_data = data["plan"]
            if not isinstance(plan_data, dict):
                return None
            plan = ExecutionPlan.from_dict(plan_data, graph)
        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ):
            return None
        plan.debug_info["cache_hit"] = True
        plan.debug_info["cache_layer"] = "disk"
        with self._lock:
            self._memory[digest] = _clone_plan(plan, graph)
        return plan

    def put(self, key: PlanCacheKey, plan: ExecutionPlan) -> None:
        digest = key.digest()
        plan.debug_info["cache_hit"] = False
        with self._lock:
            self._memory[digest] = _clone_plan(plan, plan.graph)
        if not self.enable_disk:
            return
        temporary_path: Optional[Path] = None
        file_descriptor: Optional[int] = None
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f"{digest}.",
                suffix=".tmp",
                dir=self.cache_dir,
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                file_descriptor = None
                json.dump(
                    {
                        "cache_digest": digest,
                        "cache_key": key.__dict__,
                        "plan": plan.to_dict(),
                    },
                    handle,
                    sort_keys=True,
                    indent=2,
                    default=str,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self._path(digest))
            temporary_path = None
            _fsync_directory(self.cache_dir)
        except (OSError, TypeError, ValueError):
            plan.debug_info["disk_cache_error"] = "write_failed"
        finally:
            if file_descriptor is not None:
                try:
                    os.close(file_descriptor)
                except OSError:
                    pass
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except OSError:
                    pass

    def clear_memory(self) -> None:
        with self._lock:
            self._memory.clear()

    def _path(self, digest: str) -> Path:
        return self.cache_dir / f"{digest}.json"


_DEFAULT_PLAN_CACHE = PlanCache()


def get_default_plan_cache() -> PlanCache:
    return _DEFAULT_PLAN_CACHE


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _triton_version() -> str:
    try:
        import triton

        return str(triton.__version__)
    except Exception:
        return "unknown"


_PLANNER_OPTION_NAMES = (
    "autotune",
    "autotune_warmup",
    "autotune_repeat",
    "fusion",
    "max_plan_candidates",
)


def _freeze_option(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze_option(item))
            for key, item in sorted(
                value.items(), key=lambda pair: str(pair[0])
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_option(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _planner_option_signature(
    options: Optional[Mapping[str, Any]],
) -> tuple:
    if not options:
        return ()
    return tuple(
        (name, _freeze_option(options[name]))
        for name in _PLANNER_OPTION_NAMES
        if name in options
    )


def _clone_plan(plan: ExecutionPlan, graph: Graph) -> ExecutionPlan:
    clone = copy.deepcopy(plan)
    clone.graph = graph
    if hasattr(clone, "_prepared_executor"):
        delattr(clone, "_prepared_executor")
    return clone
