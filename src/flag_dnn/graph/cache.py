from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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

    @classmethod
    def from_graph(
        cls,
        graph: Graph,
        input_specs: list[TensorSpec],
        backend: str,
        flagdnn_version: str,
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
            ),
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
        if digest in self._memory:
            plan = self._memory[digest]
            plan.debug_info["cache_hit"] = True
            plan.debug_info["cache_layer"] = "memory"
            return plan
        if not self.enable_disk:
            return None
        path = self._path(digest)
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except OSError:
            return None
        if data.get("cache_digest") != digest:
            return None
        plan = ExecutionPlan.from_dict(data["plan"], graph)
        plan.debug_info["cache_hit"] = True
        plan.debug_info["cache_layer"] = "disk"
        self._memory[digest] = plan
        return plan

    def put(self, key: PlanCacheKey, plan: ExecutionPlan) -> None:
        digest = key.digest()
        plan.debug_info["cache_hit"] = False
        self._memory[digest] = plan
        if not self.enable_disk:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with self._path(digest).open("w", encoding="utf-8") as handle:
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
        except OSError:
            plan.debug_info["disk_cache_error"] = "write_failed"

    def clear_memory(self) -> None:
        self._memory.clear()

    def _path(self, digest: str) -> Path:
        return self.cache_dir / f"{digest}.json"


_DEFAULT_PLAN_CACHE = PlanCache()


def get_default_plan_cache() -> PlanCache:
    return _DEFAULT_PLAN_CACHE


def _triton_version() -> str:
    try:
        import triton

        return str(triton.__version__)
    except Exception:
        return "unknown"
