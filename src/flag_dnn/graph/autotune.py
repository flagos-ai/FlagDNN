from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from flag_dnn.graph.device import (
    create_runtime_device_event,
    has_runtime_device_tensor,
    synchronize_current_runtime_device,
    synchronize_runtime_device,
)
from flag_dnn.graph.plan import ExecutionPlan


@dataclass(frozen=True)
class AutotuneResult:
    plan_id: str
    latency_ms: float
    warmup: int
    repeat: int
    workspace_size: int
    num_kernels: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "latency_ms": self.latency_ms,
            "warmup": self.warmup,
            "repeat": self.repeat,
            "workspace_size": self.workspace_size,
            "num_kernels": self.num_kernels,
        }


class GraphAutotuner:
    def __init__(self, options: Optional[dict[str, Any]] = None):
        self.options = {} if options is None else dict(options)
        self.enabled = self._enabled()
        self.warmup = int(
            self.options.get(
                "autotune_warmup", os.getenv("FLAGDNN_AUTOTUNE_WARMUP", 1)
            )
        )
        self.repeat = int(
            self.options.get(
                "autotune_repeat", os.getenv("FLAGDNN_AUTOTUNE_REPEAT", 5)
            )
        )

    def select_best(
        self,
        candidates: list[ExecutionPlan],
        runtime_inputs: Optional[tuple[Any, ...]] = None,
    ) -> ExecutionPlan:
        if not candidates:
            raise RuntimeError("planner produced no execution candidates")
        if not self.enabled or runtime_inputs is None or len(candidates) == 1:
            chosen = candidates[0]
            chosen.debug_info["autotune"] = {
                "enabled": self.enabled,
                "measured": False,
                "reason": self._skip_reason(runtime_inputs, candidates),
                "candidate_count": len(candidates),
            }
            return chosen

        results = []
        best_plan = candidates[0]
        best_latency = float("inf")
        for plan in candidates:
            try:
                latency = self._measure(plan, runtime_inputs)
            except Exception as exc:
                plan.debug_info.setdefault("autotune_errors", []).append(
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            result = AutotuneResult(
                plan_id=plan.plan_id,
                latency_ms=latency,
                warmup=self.warmup,
                repeat=self.repeat,
                workspace_size=plan.workspace_size,
                num_kernels=len(plan.steps),
            )
            results.append(result)
            if latency < best_latency:
                best_latency = latency
                best_plan = plan

        best_plan.tuned = bool(results)
        best_plan.debug_info["autotune"] = {
            "enabled": self.enabled,
            "measured": bool(results),
            "selected_plan_id": best_plan.plan_id,
            "results": [result.to_dict() for result in results],
            "candidate_count": len(candidates),
        }
        return best_plan

    def _measure(
        self, plan: ExecutionPlan, runtime_inputs: tuple[Any, ...]
    ) -> float:
        for _ in range(max(self.warmup, 0)):
            plan.run(*runtime_inputs)
        _synchronize(runtime_inputs)

        start = time.perf_counter()
        if _has_runtime_device(runtime_inputs):
            starter = create_runtime_device_event(enable_timing=True)
            ender = create_runtime_device_event(enable_timing=True)
            if starter is not None and ender is not None:
                starter.record()
                for _ in range(max(self.repeat, 1)):
                    plan.run(*runtime_inputs)
                ender.record()
                synchronize_current_runtime_device()
                return float(starter.elapsed_time(ender)) / max(
                    self.repeat, 1
                )

        for _ in range(max(self.repeat, 1)):
            plan.run(*runtime_inputs)
        elapsed = time.perf_counter() - start
        return elapsed * 1000.0 / max(self.repeat, 1)

    def _enabled(self) -> bool:
        option = self.options.get("autotune")
        if option is not None:
            return bool(option)
        return str(os.getenv("FLAGDNN_AUTOTUNE", "0")).lower() not in (
            "0",
            "false",
            "no",
        )

    def _skip_reason(
        self,
        runtime_inputs: Optional[tuple[Any, ...]],
        candidates: list[ExecutionPlan],
    ) -> str:
        if not self.enabled:
            return "disabled"
        if runtime_inputs is None:
            return "no_runtime_inputs"
        if len(candidates) == 1:
            return "single_candidate"
        return "unknown"


def _has_runtime_device(values: tuple[Any, ...]) -> bool:
    return has_runtime_device_tensor(values)


def _synchronize(values: tuple[Any, ...]) -> None:
    if _has_runtime_device(values):
        synchronize_runtime_device(values)
