# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""THead codegen pointwise lowering."""

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
    _BINARY_POINTWISE_MODES,
    _FIXED_NUM_STAGES,
    _FIXED_NUM_WARPS,
    _MAX_KERNEL_SOURCE_BYTES,
    _PPU_WARP_SIZE,
    _UNARY_POINTWISE_MODES,
    _pointwise_block_size,
)
from ..dispatch.pointwise import (
    _add_square_variant,
    _binary_pointwise_variant,
    _unary_pointwise_variant,
    _validate_add_square_graph,
    _validate_binary_pointwise_graph,
    _validate_binary_select_graph,
    _validate_unary_pointwise_graph,
)
from ..dispatch.selection import (
    _registry_sha256,
    _validate_add_square_candidate,
    _validate_binary_pointwise_candidate,
    _validate_unary_pointwise_candidate,
)
from ..dispatch.tuning import (
    _include_default_candidate,
    _load_pointwise_tuning,
)


def _compile_add_square(
    *,
    request: dict[str, Any],
    request_bytes: bytes,
    identity: dict[str, Any],
    target: str,
    output_directory: Path,
    enable_autotune: bool,
) -> dict[str, Any]:
    plan = _validate_add_square_graph(request["graph"])
    candidate = kernel_registry.select_kernel_candidate("thead", "add_square")
    _validate_add_square_candidate(candidate)
    compiler_path = compiler_entry_path()
    source_path = kernel_registry.resolve_kernel_source(
        compiler_path, candidate
    )
    source_bytes = kernel_registry.materialize_kernel_source(
        source_path, candidate
    )
    if not source_bytes or len(source_bytes) > _MAX_KERNEL_SOURCE_BYTES:
        raise ValueError("THead AddSquare kernel source size is invalid")
    tuning_path = kernel_registry.resolve_tuning_source(
        compiler_path, candidate
    )
    if not tuning_path.is_file():
        raise ValueError("THead AddSquare tuning source is missing")
    registry_digest = _registry_sha256()
    n_elements = int(plan["n_elements"])
    default_configuration = {
        "META": {"BLOCK_SIZE": _pointwise_block_size(n_elements)},
        "maxnreg": None,
        "num_stages": _FIXED_NUM_STAGES,
        "num_warps": _FIXED_NUM_WARPS,
        "ppu_compiler_options": {},
    }
    variants = [
        _add_square_variant(
            plan["argument_tensors"],
            n_elements,
            plan["stride_constants"],
            default_configuration,
            "default",
        )
    ]
    tuning_manifest: dict[str, Any] | None = None
    if enable_autotune:
        configurations, tuning_bytes = _load_pointwise_tuning(
            tuning_path, candidate, "add_square"
        )
        variants = [
            _add_square_variant(
                plan["argument_tensors"],
                n_elements,
                plan["stride_constants"],
                configuration,
                "block"
                + str(configuration["META"]["BLOCK_SIZE"])
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
                    "operation": "add_square",
                    "function": plan["function"],
                    "registry_sha256": registry_digest,
                    "source_sha256": _sha256(source_bytes),
                },
                "tuning": {
                    "source_sha256": _sha256(tuning_bytes),
                    "table": candidate.tuning.table,
                    "key": candidate.tuning.key,
                    "key_value": n_elements,
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

    # The single fused kernel consumes only the graph's external boundary
    # tensors.  Its virtual Mul output never crosses a stage boundary and
    # therefore has no backing workspace allocation.
    workspace_size = 0
    manifest_tensors = [_manifest_tensor(tensor) for tensor in plan["tensors"]]
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
        "external_uids": [
            tensor["uid"]
            for tensor in plan["tensors"]
            if not tensor["virtual"]
        ],
        "tensor_count": len(plan["tensors"]),
        "tensors": manifest_tensors,
        "workspace": {"size": workspace_size, "alignment": 256},
        "program": {
            "schema_version": 1,
            "stage_count": 1,
            "stages": [
                {
                    "stage_id": 0,
                    "source_node_ids": [0, 1],
                    "dependencies": [],
                    "operation": "add_square",
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
        "node_count": 2,
        "stage_count": 1,
        "target": target,
        "warp_size": _PPU_WARP_SIZE,
        "artifact_directory": str(output_directory),
        "workspace_size": workspace_size,
        "workspace_alignment": 256,
        "execution_engine": "libtriton_jit",
    }


def _compile_binary_pointwise(
    *,
    request: dict[str, Any],
    request_bytes: bytes,
    identity: dict[str, Any],
    target: str,
    output_directory: Path,
    enable_autotune: bool,
    operation: str,
) -> dict[str, Any]:
    operation_label = operation.capitalize()
    pointwise_mode = (
        41
        if operation == "binary_select"
        else _BINARY_POINTWISE_MODES[operation]
    )
    plan = (
        _validate_binary_select_graph(request["graph"])
        if operation == "binary_select"
        else _validate_binary_pointwise_graph(request["graph"], operation)
    )
    candidate = kernel_registry.select_kernel_candidate("thead", operation)
    _validate_binary_pointwise_candidate(candidate, operation)
    compiler_path = compiler_entry_path()
    source_path = kernel_registry.resolve_kernel_source(
        compiler_path, candidate
    )
    source_bytes = kernel_registry.materialize_kernel_source(
        source_path, candidate
    )
    if not source_bytes or len(source_bytes) > _MAX_KERNEL_SOURCE_BYTES:
        raise ValueError(
            f"THead {operation_label} kernel source size is invalid"
        )
    tuning_path = kernel_registry.resolve_tuning_source(
        compiler_path, candidate
    )
    if not tuning_path.is_file():
        raise ValueError(f"THead {operation_label} tuning source is missing")
    registry_digest = _registry_sha256()

    tensors_by_uid = {int(tensor["uid"]): tensor for tensor in plan["tensors"]}
    argument_tensors = [tensors_by_uid[uid] for uid in plan["ordered_uids"]]
    n_elements = int(plan["n_elements"])
    alpha = plan["alpha"]
    function = str(plan["function"])
    stride_constants = plan["stride_constants"]
    default_configuration = {
        "META": {"BLOCK_SIZE": _pointwise_block_size(n_elements)},
        "maxnreg": None,
        "num_stages": _FIXED_NUM_STAGES,
        "num_warps": (
            8
            if n_elements >= 65536
            and stride_constants is None
            and argument_tensors[0]["data_type"] == "float32"
            and pointwise_mode != 41
            else _FIXED_NUM_WARPS
        ),
        "ppu_compiler_options": {},
    }
    variants = [
        _binary_pointwise_variant(
            argument_tensors,
            n_elements,
            pointwise_mode,
            alpha,
            stride_constants,
            default_configuration,
            "default",
            plan.get("activation_parameters"),
        )
    ]
    tuning_manifest: dict[str, Any] | None = None
    if enable_autotune:
        configurations, tuning_bytes = _load_pointwise_tuning(
            tuning_path, candidate, operation
        )
        if n_elements >= 65536:
            configurations = _include_default_candidate(
                configurations, default_configuration
            )
        variants = [
            _binary_pointwise_variant(
                argument_tensors,
                n_elements,
                pointwise_mode,
                alpha,
                stride_constants,
                configuration,
                "block"
                + str(configuration["META"]["BLOCK_SIZE"])
                + "_w"
                + str(configuration["num_warps"])
                + "_s"
                + str(configuration["num_stages"]),
                plan.get("activation_parameters"),
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
                    "function": function,
                    "registry_sha256": registry_digest,
                    "source_sha256": _sha256(source_bytes),
                    "specialization": {"alpha": alpha},
                },
                "tuning": {
                    "source_sha256": _sha256(tuning_bytes),
                    "table": candidate.tuning.table,
                    "key": candidate.tuning.key,
                    "key_value": n_elements,
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
                        "function": function,
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


def _compile_unary_pointwise(
    *,
    request: dict[str, Any],
    request_bytes: bytes,
    identity: dict[str, Any],
    target: str,
    output_directory: Path,
    enable_autotune: bool,
    operation: str,
) -> dict[str, Any]:
    operation_label = operation.capitalize()
    pointwise_mode = _UNARY_POINTWISE_MODES[operation]
    plan = _validate_unary_pointwise_graph(request["graph"], operation)
    candidate = kernel_registry.select_kernel_candidate("thead", operation)
    _validate_unary_pointwise_candidate(candidate, operation)
    compiler_path = compiler_entry_path()
    source_path = kernel_registry.resolve_kernel_source(
        compiler_path, candidate
    )
    source_bytes = kernel_registry.materialize_kernel_source(
        source_path, candidate
    )
    if not source_bytes or len(source_bytes) > _MAX_KERNEL_SOURCE_BYTES:
        raise ValueError(
            f"THead {operation_label} kernel source size is invalid"
        )
    tuning_path = kernel_registry.resolve_tuning_source(
        compiler_path, candidate
    )
    if not tuning_path.is_file():
        raise ValueError(f"THead {operation_label} tuning source is missing")
    registry_digest = _registry_sha256()

    tensors_by_uid = {int(tensor["uid"]): tensor for tensor in plan["tensors"]}
    argument_tensors = [tensors_by_uid[uid] for uid in plan["ordered_uids"]]
    n_elements = int(plan["n_elements"])
    function = str(plan["function"])
    stride_constants = plan["stride_constants"]
    kernel_constants = plan["kernel_constants"]
    default_configuration = {
        "META": {"BLOCK_SIZE": _pointwise_block_size(n_elements)},
        "maxnreg": None,
        "num_stages": _FIXED_NUM_STAGES,
        "num_warps": (
            8
            if n_elements >= 65536
            and stride_constants is None
            and argument_tensors[0]["data_type"] == "float32"
            and pointwise_mode != 41
            else _FIXED_NUM_WARPS
        ),
        "ppu_compiler_options": {},
    }
    variants = [
        _unary_pointwise_variant(
            argument_tensors,
            n_elements,
            pointwise_mode,
            stride_constants,
            kernel_constants,
            default_configuration,
            "default",
        )
    ]
    tuning_manifest: dict[str, Any] | None = None
    if enable_autotune:
        configurations, tuning_bytes = _load_pointwise_tuning(
            tuning_path, candidate, operation
        )
        if n_elements >= 65536:
            configurations = _include_default_candidate(
                configurations, default_configuration
            )
        variants = [
            _unary_pointwise_variant(
                argument_tensors,
                n_elements,
                pointwise_mode,
                stride_constants,
                kernel_constants,
                configuration,
                "block"
                + str(configuration["META"]["BLOCK_SIZE"])
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
                    "function": function,
                    "registry_sha256": registry_digest,
                    "source_sha256": _sha256(source_bytes),
                    "specialization": kernel_constants,
                },
                "tuning": {
                    "source_sha256": _sha256(tuning_bytes),
                    "table": candidate.tuning.table,
                    "key": candidate.tuning.key,
                    "key_value": n_elements,
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
                        "function": function,
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
