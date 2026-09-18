# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""THead codegen reduction lowering."""

from __future__ import annotations

from .artifacts import write_program_manifest

from ..codegen.io import compiler_entry_path
from ..compiler_identity import PROVIDER_NAME
from flagdnn_codegen import kernel_registry
from pathlib import Path
from typing import Any
from ..codegen.abi import (
    _manifest_tensor,
)
from ..codegen.io import (
    _atomic_write,
    _canonical_sha256,
    _sha256,
)
from ..dispatch.common import (
    SCHEMA_VERSION,
    _MAX_KERNEL_SOURCE_BYTES,
    _PPU_WARP_SIZE,
)
from ..dispatch.reduction import (
    _reduction_variant,
    _validate_reduction_graph,
)
from ..dispatch.selection import (
    _registry_sha256,
    _validate_reduction_candidate,
)
from ..dispatch.tuning import (
    _load_reduction_tuning,
)


def _compile_reduction(
    *,
    request: dict[str, Any],
    request_bytes: bytes,
    identity: dict[str, Any],
    target: str,
    output_directory: Path,
    enable_autotune: bool,
    operation: str,
) -> dict[str, Any]:
    plan = _validate_reduction_graph(request["graph"], operation)
    candidate = kernel_registry.select_kernel_candidate("thead", operation)
    _validate_reduction_candidate(candidate, operation)
    compiler_path = compiler_entry_path()
    source_path = kernel_registry.resolve_kernel_source(
        compiler_path, candidate
    )
    source_bytes = kernel_registry.materialize_kernel_source(
        source_path, candidate
    )
    if not source_bytes or len(source_bytes) > _MAX_KERNEL_SOURCE_BYTES:
        raise ValueError("THead reduction kernel source size is invalid")
    tuning_path = kernel_registry.resolve_tuning_source(
        compiler_path, candidate
    )
    if not tuning_path.is_file():
        raise ValueError("THead reduction tuning source is missing")
    registry_digest = _registry_sha256()
    default_configuration = {
        "META": {"BLOCK_M": 8, "BLOCK_N": 32},
        "maxnreg": None,
        "num_stages": 1,
        "num_warps": 4,
        "ppu_compiler_options": {},
    }
    variants = [_reduction_variant(plan, default_configuration, "default")]
    tuning_manifest: dict[str, Any] | None = None
    if enable_autotune:
        configurations, tuning_bytes = _load_reduction_tuning(
            tuning_path, candidate, operation
        )
        variants = [
            _reduction_variant(
                plan,
                configuration,
                "blockm"
                + str(configuration["META"]["BLOCK_M"])
                + "_blockn"
                + str(configuration["META"]["BLOCK_N"])
                + "_w"
                + str(configuration["num_warps"])
                + "_s"
                + str(configuration["num_stages"]),
            )
            for configuration in configurations
        ]
        candidate_identity = _canonical_sha256(
            {
                "schema_version": 1,
                "backend": "thead",
                "target": target,
                "engine": "libtriton_jit",
                "compiler_identity": identity["identity_sha256"],
                "kernel": {
                    "provider": candidate.provider,
                    "ownership": candidate.ownership,
                    "operation": operation,
                    "function": plan["function"],
                    "registry_sha256": registry_digest,
                    "source_sha256": _sha256(source_bytes),
                    "specialization": plan["constants"],
                },
                "tuning": {
                    "source_sha256": _sha256(tuning_bytes),
                    "table": candidate.tuning.table,
                    "key": candidate.tuning.key,
                    "key_value": plan["rows"],
                    "strategy": candidate.tuning.strategy,
                    "warmup": candidate.tuning.warmup,
                    "repetitions": candidate.tuning.repetitions,
                    "configurations": configurations,
                },
            }
        )
        tuning_manifest = {
            "warmup": candidate.tuning.warmup,
            "repetitions": candidate.tuning.repetitions,
            "candidate_identity": candidate_identity,
            "selection_cache": ".flagdnn-autotune-v1-stage-0.json",
        }
    materialized_name = "generated_stage_0.py"
    manifest = {
        "schema_version": 1,
        "artifact_kind": "flagdnn_execution_program",
        "flagdnn_version": request["flagdnn_version"],
        "graph_ir_schema_version": SCHEMA_VERSION,
        "backend_abi_version": 2,
        "backend": "thead",
        "target": target,
        "warp_size": _PPU_WARP_SIZE,
        "engine": "libtriton_jit",
        "request_sha256": _sha256(request_bytes),
        "compiler": {
            "provider": identity["provider"],
            "provider_version": identity["provider_version"],
            "identity_sha256": identity["identity_sha256"],
        },
        "external_uids": [tensor["uid"] for tensor in plan["tensors"]],
        "tensor_count": len(plan["tensors"]),
        "tensors": [_manifest_tensor(tensor) for tensor in plan["tensors"]],
        "workspace": {"size": 0, "alignment": 256},
        "program": {
            "schema_version": 1,
            "stage_count": 1,
            "stages": [
                {
                    "stage_id": 0,
                    "source_node_ids": [0],
                    "dependencies": [],
                    "operation": operation,
                    "kernel": {
                        "provider": candidate.provider,
                        "ownership": candidate.ownership,
                        "function": plan["function"],
                        "registry_sha256": registry_digest,
                        "materialized_source": {
                            "path": materialized_name,
                            "size": len(source_bytes),
                            "sha256": _sha256(source_bytes),
                        },
                    },
                    "variants": variants,
                    "tuning": tuning_manifest,
                }
            ],
        },
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    if output_directory.is_symlink() or not output_directory.is_dir():
        raise ValueError("THead artifact output directory is invalid")
    _atomic_write(output_directory / materialized_name, source_bytes)
    write_program_manifest(output_directory, manifest)
    return {
        "schema_version": 1,
        "status": "success",
        "backend": "thead",
        "provider": PROVIDER_NAME,
        "node_count": 1,
        "stage_count": 1,
        "target": target,
        "warp_size": _PPU_WARP_SIZE,
        "artifact_directory": str(output_directory),
        "workspace_size": 0,
        "workspace_alignment": 256,
        "execution_engine": "libtriton_jit",
    }
