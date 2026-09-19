"""Ascend codegen emit implementation."""

from __future__ import annotations

from ..dispatch.common import (
    LAUNCH_ABI,
    PointwiseStagePlan,
)
from ..dispatch.tuning import (
    _autotune_candidate_descriptor,
    _autotune_candidate_id,
    _order_stage_configurations,
    _stage_semantic_attributes,
)
from ..tuning_decoder import (
    TuningConfiguration,
    canonical_sha256,
    checked_grid,
)
from .launch import (
    _batchnorm_inference_grid,
    _candidate_payload,
)
from flagdnn_codegen.kernel_registry import (
    KernelCandidate,
)
from typing import (
    Any,
)


def _build_stage(
    *,
    plan: PointwiseStagePlan,
    candidate: KernelCandidate,
    configurations: tuple[TuningConfiguration, ...],
    tuning_source_sha256: str,
    enable_autotune: bool,
    capabilities: dict[str, Any],
    source_path: str,
    source_sha256: str,
    source_size: int,
    worker_count: int,
) -> dict[str, Any]:
    if not configurations:
        raise ValueError(f"Ascend {plan.operation} tuning produced no candidates")
    if not enable_autotune:
        configurations = _order_stage_configurations(plan, configurations)
    effective_configurations = configurations if enable_autotune else configurations[:1]
    tuning_enabled = enable_autotune and len(effective_configurations) > 1
    candidate_entries: list[dict[str, Any]] = []
    tuning_candidate_descriptors: list[dict[str, Any]] = []
    for configuration in effective_configurations:
        candidate_id = (
            _autotune_candidate_id(plan.operation, configuration)
            if tuning_enabled
            else "default"
        )
        candidate_entries.append(
            {
                "candidate_id": candidate_id,
                "launch_abi": LAUNCH_ABI,
                "payload": _candidate_payload(
                    stage=plan,
                    configuration=configuration,
                    capabilities=capabilities,
                    source_path=source_path,
                    source_sha256=source_sha256,
                    worker_count=worker_count,
                ),
            }
        )
        if tuning_enabled:
            descriptor = _autotune_candidate_descriptor(candidate_id, configuration)
            if plan.kernel_family in {
                "matmul",
                "convolution_fprop",
                "batchnorm",
                "batchnorm_inference",
                "rmsnorm",
                "layernorm",
            }:
                work_items = (
                    int(plan.meta["ROWS"])
                    if plan.kernel_family in {"rmsnorm", "layernorm"}
                    else (
                        int(plan.meta["CHANNELS"])
                        if plan.kernel_family == "batchnorm"
                        else plan.n_elements
                    )
                )
                if plan.kernel_family == "matmul":
                    block_size = int(configuration.meta["BLOCK_SIZE"])
                    tiles = ((int(plan.meta["M"]) + block_size - 1) // block_size) * (
                        (int(plan.meta["N"]) + block_size - 1) // block_size
                    )
                    descriptor["grid"] = [
                        min(tiles * int(plan.meta["BATCH"]), worker_count),
                        1,
                        1,
                    ]
                elif plan.kernel_family == "convolution_fprop":
                    grid = checked_grid(
                        work_items,
                        int(configuration.meta["BLOCK_SIZE"]),
                        capabilities,
                    )
                    descriptor["grid"] = [
                        min(grid[0], worker_count),
                        grid[1],
                        grid[2],
                    ]
                elif plan.kernel_family == "batchnorm_inference":
                    grid = _batchnorm_inference_grid(
                        plan,
                        int(configuration.meta["BLOCK_SIZE"]),
                        capabilities,
                    )
                    descriptor["grid"] = [
                        min(grid[0], worker_count),
                        grid[1],
                        grid[2],
                    ]
                else:
                    grid = checked_grid(work_items, 1, capabilities)
                    descriptor["grid"] = [
                        min(grid[0], worker_count),
                        grid[1],
                        grid[2],
                    ]
            tuning_candidate_descriptors.append(descriptor)

    if len({entry["candidate_id"] for entry in candidate_entries}) != len(
        candidate_entries
    ):
        raise ValueError(
            f"Ascend {plan.operation} tuning candidate identity is duplicated"
        )

    selection = {
        "state": "pending" if tuning_enabled else "fixed",
        "candidate_id": "" if tuning_enabled else "default",
    }
    autotune: dict[str, Any] = {
        "schema_version": 1,
        "enabled": tuning_enabled,
        "selection": selection,
    }
    if tuning_enabled:
        tuning = candidate.tuning
        if tuning is None:
            raise RuntimeError(
                f"Ascend {plan.operation} candidate lost its tuning metadata"
            )
        identity_payload = {
            "schema_version": 1,
            "backend": "ascend",
            "launch_abi": LAUNCH_ABI,
            "operation": plan.operation,
            "attributes": _stage_semantic_attributes(plan),
            "function": plan.function_name,
            "source_sha256": tuning_source_sha256,
            "key": tuning.key,
            "key_value": plan.n_elements,
            "strategy": tuning.strategy,
            "candidates": (
                sorted(
                    tuning_candidate_descriptors,
                    key=lambda descriptor: str(descriptor["candidate_id"]),
                )
                if plan.kernel_family
                in {
                    "matmul",
                    "convolution_fprop",
                    "batchnorm",
                    "batchnorm_inference",
                    "rmsnorm",
                    "layernorm",
                }
                else tuning_candidate_descriptors
            ),
        }
        if plan.kernel_family in {
            "matmul",
            "convolution_fprop",
            "batchnorm",
            "batchnorm_inference",
            "rmsnorm",
            "layernorm",
        }:
            identity_payload["kernel_source_sha256"] = source_sha256
        autotune.update(
            {
                "source": tuning.source,
                "source_sha256": tuning_source_sha256,
                "table": tuning.table,
                "key": tuning.key,
                "strategy": tuning.strategy,
                "warmup": tuning.warmup,
                "repetitions": tuning.repetitions,
                "candidate_identity": canonical_sha256(
                    identity_payload,
                    f"Ascend {plan.operation} tuning identity",
                ),
            }
        )

    return {
        "stage_id": plan.stage_id,
        "kind": "kernel",
        "source_node_ids": list(plan.source_node_ids),
        "dependencies": list(plan.dependencies),
        "operation": plan.operation,
        "kernel_family": plan.kernel_family,
        "kernel": {
            "provider": candidate.provider,
            "ownership": candidate.ownership,
            "source": candidate.source,
            "source_sha256": source_sha256,
            "entry_point": plan.function_name,
            "materialized_source": {
                "file": source_path,
                "size": source_size,
                "sha256": source_sha256,
            },
        },
        "workspace": dict(plan.workspace),
        "argument_sources": [dict(value) for value in plan.argument_sources],
        "candidates": candidate_entries,
        "autotune": autotune,
    }
