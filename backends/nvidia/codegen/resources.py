"""Per-launch compiler scratch and sequential-program workspace accounting."""

from __future__ import annotations

from typing import Any
import math


def _effective_jit_warps(metadata: Any, requested: int, cached: bool) -> int:
    """Cached specialization uses the compiled total, not just producer warps."""
    effective = metadata.num_warps
    if (
        type(effective) is not int
        or not requested <= effective <= 32
        or (not cached and effective != requested)
    ):
        raise ValueError(
            "compiled warp count is incompatible with the JIT launch"
        )
    return effective


def _launch_scratch_size(
    metadata: Any, grid: tuple[int, int, int], minimum: int = 0
) -> int:
    # Triton reports bytes per CTA, not bytes per CUDA launch. TMA descriptors
    # are indexed with the linear CTA ID in the generated PTX.
    per_cta = getattr(metadata, "global_scratch_size", 0)
    profile = getattr(metadata, "profile_scratch_size", 0)
    alignment = getattr(metadata, "global_scratch_align", 1)
    if any(
        type(value) is not int or value < 0
        for value in (per_cta, profile, minimum)
    ):
        raise ValueError("invalid compiled scratch size")
    if (
        type(alignment) is not int
        or alignment < 1
        or alignment > 256
        or alignment & (alignment - 1)
    ):
        raise ValueError(
            "compiled scratch alignment exceeds workspace alignment"
        )
    if profile:
        raise ValueError("NVIDIA profile scratch is not supported")
    if len(grid) != 3 or any(
        type(extent) is not int or extent < 1 for extent in grid
    ):
        raise ValueError(
            "scratch launch grid must contain three positive integers"
        )
    size = max(minimum, per_cta * math.prod(grid))
    size = ((size + 255) // 256) * 256
    if size > (1 << 63) - 1:
        raise ValueError("compiled scratch size exceeds the artifact ABI")
    return size


def _program_scratch_size(stages: list[dict[str, Any]]) -> int:
    # Stages execute sequentially on the supplied stream. All variants/stages
    # can share one suffix; graph tensor workspace remains a disjoint prefix.
    return max(
        (
            variant["launch"]["global_scratch_size"]
            for stage in stages
            for variant in stage.get("variants", [stage])
        ),
        default=0,
    )
