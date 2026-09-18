# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""THead codegen convolution lowering."""

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
from ..dispatch.convolution import (
    _convolution_variant,
    _direct_dgrad,
    _validate_conv_bias_relu_graph,
    _validate_convolution_graph,
)
from ..dispatch.selection import (
    _registry_sha256,
    _validate_convolution_candidate,
)
from ..dispatch.tuning import (
    _include_default_candidate,
    _load_convolution_tuning,
)


def _compile_convolution(
    *,
    request: dict[str, Any],
    request_bytes: bytes,
    identity: dict[str, Any],
    target: str,
    output_directory: Path,
    enable_autotune: bool,
    operation: str,
) -> dict[str, Any]:
    plan = (
        _validate_conv_bias_relu_graph(request["graph"])
        if operation == "conv_bias_relu"
        else _validate_convolution_graph(request["graph"], operation)
    )
    candidate = kernel_registry.select_kernel_candidate("thead", operation)
    _validate_convolution_candidate(candidate, operation)
    compiler_path = compiler_entry_path()
    source_path = kernel_registry.resolve_kernel_source(
        compiler_path, candidate
    )
    source_bytes = kernel_registry.materialize_kernel_source(
        source_path, candidate
    )
    if not source_bytes or len(source_bytes) > _MAX_KERNEL_SOURCE_BYTES:
        raise ValueError(f"THead {operation} kernel source size is invalid")
    tuning_path = kernel_registry.resolve_tuning_source(
        compiler_path, candidate
    )
    if not tuning_path.is_file():
        raise ValueError(f"THead {operation} tuning source is missing")
    registry_digest = _registry_sha256()
    default_configuration: dict[str, Any] = {
        "META": {
            "BLOCK_M": 16,
            "BLOCK_OC": 16,
            "BLOCK_CI": 16,
            "BLOCK_K": 16,
            "BLOCK_HW": 16,
        },
        "maxnreg": None,
        "num_stages": 1,
        "num_warps": 4,
        "ppu_compiler_options": {},
    }
    # Keep the small tensor policy; amortize indexing and launch work on
    # larger NCHW convolutions. IEEE FP32 dot arithmetic is unchanged.
    if (
        operation == "convolution_wgrad"
        and plan["channels_per_group"] <= 8
        and plan["m"] >= 4096
    ):
        default_configuration["META"]["BLOCK_M"] = 1024
    elif (
        operation == "convolution_wgrad"
        and plan["argument_tensors"][0]["data_type"] != "float32"
        and plan["channels_per_group"] >= 32
        and plan["m"] >= 4096
        and all(
            plan["constants"][name] == 1
            for name in (
                "STRIDE_D",
                "STRIDE_H",
                "STRIDE_W",
                "X_STRIDE_W",
                "DY_STRIDE_W",
            )
        )
    ):
        # Amortize indexing over longer unit-stride reductions while retaining
        # 16x16 output tiles. Stride-two and IEEE FP32 profiles regress here.
        default_configuration["META"]["BLOCK_M"] = 128
    elif (
        operation == "convolution_dgrad"
        and min(plan["channels_per_group"], plan["outputs_per_group"]) >= 32
        and plan["m"] >= 256
    ):
        default_configuration["META"].update(
            BLOCK_M=(
                128
                if plan["argument_tensors"][0]["data_type"] == "float32"
                else 64
            ),
            BLOCK_CI=32,
            BLOCK_K=32,
        )
    elif (
        operation == "convolution_fprop"
        and plan["argument_tensors"][0]["data_type"] != "float32"
        and min(plan["channels_per_group"], plan["outputs_per_group"]) >= 32
        and plan["m"] >= 256
    ):
        default_configuration["META"].update(
            BLOCK_M=64, BLOCK_OC=32, BLOCK_K=32
        )
        width_only_nwc = all(
            plan["constants"][name] == 1
            for name in ("XD", "XH", "OD", "OH", "KD", "KH", "Y_STRIDE_C")
        )
        if plan["channels_per_group"] == 32 and not width_only_nwc:
            # PPU lowering of the 64x32x32 tile leaves output positions
            # unwritten for CI=32 NCHW 3x3 convolutions (FP16 and BF16).
            # Width-only NWC outputs have qualified complete stores and run
            # faster with the larger tile. Other CI=32 layouts use the
            # 32x32x32 registry candidate; complete metadata deduplicates it.
            default_configuration["META"] = {
                name: 32 for name in default_configuration["META"]
            }
    small_channels = (
        plan["channels_per_group"] <= 8
        and plan["outputs_per_group"] >= 16
        and plan["m"] >= 4096
    )
    if small_channels and operation == "convolution_fprop":
        # Wide spatial tiles are qualified for planar convolutions. Small
        # volumes regress in every dtype, so retain the original 16x16x16 tile.
        if plan["constants"]["XD"] == 1 and plan["constants"]["KD"] == 1:
            default_configuration["META"].update(
                BLOCK_M=128, BLOCK_OC=16, BLOCK_K=32
            )
    elif small_channels and operation == "convolution_dgrad":
        default_configuration["META"].update(
            BLOCK_M=128, BLOCK_CI=16, BLOCK_K=16
        )
        if not _direct_dgrad(plan["constants"], default_configuration["META"]):
            # Short half reductions need less scratch space; IEEE FP32 runs
            # faster with the original tile. Keep the qualified direct path.
            default_configuration["META"]["BLOCK_M"] = (
                16
                if plan["argument_tensors"][0]["data_type"] == "float32"
                else (64 if plan["m"] < 16384 else 256)
            )
    if (
        operation == "convolution_dgrad"
        and plan["argument_tensors"][0]["data_type"] == "float32"
        and not _direct_dgrad(plan["constants"], default_configuration["META"])
    ):
        meta = default_configuration["META"]
        blocks = (
            (plan["m"] + meta["BLOCK_M"] - 1)
            // meta["BLOCK_M"]
            * (
                (plan["channels_per_group"] + meta["BLOCK_CI"] - 1)
                // meta["BLOCK_CI"]
            )
            * plan["groups"]
        )
        if blocks < 64:
            # A large tile can leave ZW810's 64 SMs idle (for example the
            # FP32 P5 input gradient). Restore the qualified original tile.
            default_configuration["META"] = {name: 16 for name in meta}
    variants = [_convolution_variant(plan, default_configuration, "default")]
    tuning_manifest: dict[str, Any] | None = None
    if enable_autotune:
        configurations, tuning_bytes = _load_convolution_tuning(
            tuning_path, candidate, operation
        )
        if default_configuration["META"]["BLOCK_M"] > 16:
            configurations = _include_default_candidate(
                configurations, default_configuration
            )
        variants = [
            _convolution_variant(
                plan,
                configuration,
                "tile"
                + str(configuration["META"]["BLOCK_M"])
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
                    "key_value": plan["n_outputs"],
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
    # ConvBiasRelu is emitted as one fused kernel.  Its convolution and bias
    # intermediates remain internal SSA values, so materializing either graph
    # virtual tensor would reserve memory that the launch ABI never uses.
    workspace_size = 0
    manifest_tensors = [_manifest_tensor(tensor) for tensor in plan["tensors"]]
    source_node_ids = plan.get("source_node_ids", [0])
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
                    "source_node_ids": source_node_ids,
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
        "node_count": len(source_node_ids),
        "stage_count": 1,
        "target": target,
        "warp_size": _PPU_WARP_SIZE,
        "artifact_directory": str(output_directory),
        "workspace_size": workspace_size,
        "workspace_alignment": 256,
        "execution_engine": "libtriton_jit",
    }
