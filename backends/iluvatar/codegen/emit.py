# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Iluvatar codegen / emit implementation."""

from __future__ import annotations

from ..codegen.arguments import _argument_abi
from ..codegen.arguments import _artifact_arguments
from ..codegen.arguments import _nn_argument_abi
from ..codegen.io import _atomic_write
from ..codegen.io import _canonical
from ..codegen.io import _load_generated_module
from ..codegen.jit import _jit_full_signature
from ..codegen.jit import _jit_launch
from ..codegen.jit import _jit_runtime_signature
from ..dispatch import nn as nn_dispatch
from ..dispatch import tensor as tensor_dispatch
from ..dispatch.common import ILUVATAR_WARP_SIZE
from ..dispatch.common import _require_integer
from ..dispatch.pointwise import PointwiseSchema
from ..dispatch.pointwise import _pointwise_configuration
from ..dispatch.tuning import _load_nn_tuning
from ..dispatch.tuning import _load_pointwise_tuning
from ..dispatch.tuning import _load_tensor_tuning
from flagdnn_codegen.kernel_registry import materialize_kernel_source
from flagdnn_codegen.kernel_registry import resolve_kernel_source
from flagdnn_codegen.kernel_registry import select_kernel_candidate
from pathlib import Path
from typing import Any
import hashlib


def _specialize_pointwise_compute_source(
    generated_bytes: bytes, node: dict[str, Any]
) -> bytes:
    """Make the Graph compute type an explicit part of generated code.

    Triton's ordinary FP16/BF16 add, subtract and multiply expressions retain
    their input precision. FlagDNN's public pointwise graphs request FP32
    compute for numeric operations, so relying on implicit Triton promotion
    silently violates the Graph contract. Keep the common source
    platform-neutral and specialize only the per-artifact materialized copy.

    Boolean pointwise operations already execute in their declared boolean
    semantics. They still receive a deterministic source marker so the
    materialized source and candidate identities encode compute_data_type.
    """

    compute_data_type = node["compute_data_type"]
    schema: PointwiseSchema = node["schema"]
    try:
        source = generated_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("pointwise kernel source must be UTF-8") from error

    marker = (
        "# FlagDNN Iluvatar compute specialization: "
        f"{schema.data_type_policy}/{compute_data_type}\n"
    )
    if source.startswith(marker):
        raise ValueError("pointwise kernel source is already specialized")

    if node["operation"] == "identity" or node["tensors"][0]["data_type"] == "int32":
        # Identity performs no arithmetic.  Every supported input value is
        # already representable in FP32, and a direct copy additionally keeps
        # signed zero and NaN payload bits intact instead of round-tripping
        # through a conversion solely to annotate the compute type.
        pass
    elif compute_data_type == "float32":
        if schema.family == "binary":
            trigger_names = {"right"}
            cast_names = ("left", "right")
        elif schema.family == "unary":
            trigger_names = {"value"}
            cast_names = ("value",)
        elif schema.family == "ternary":
            trigger_names = {"right", "input1"}
            cast_names = ()
        else:
            raise ValueError(f"unknown pointwise family {schema.family!r}")

        lines: list[str] = []
        specialization_count = 0
        for line in source.splitlines(keepends=True):
            lines.append(line)
            stripped = line.lstrip()
            indentation = line[: len(line) - len(stripped)]
            loaded_name = next(
                (
                    name
                    for name in trigger_names
                    if stripped.startswith(f"{name} = tl.load(")
                ),
                None,
            )
            if loaded_name is None:
                continue
            names = cast_names
            if schema.family == "ternary":
                names = (
                    ("left", "right")
                    if loaded_name == "right"
                    else ("input0", "input1")
                )
            for name in names:
                lines.append(f"{indentation}{name} = {name}.to(tl.float32)\n")
            specialization_count += 1
        if specialization_count < 1:
            raise ValueError(
                "common pointwise kernel no longer exposes the expected "
                "load sites for FP32 specialization"
            )
        source = "".join(lines)
    elif compute_data_type != "boolean":
        # _validate_pointwise_data_types normally rejects this first. Keep a
        # defensive check here because this function is the precision
        # boundary for all future schema additions.
        raise ValueError(
            "Iluvatar pointwise materialization implements only float32 or "
            "boolean compute semantics"
        )

    return (marker + source).encode("utf-8")


def _compile_pointwise_stage(
    *,
    stage_id: int,
    node: dict[str, Any],
    dependencies: list[int],
    workspace: dict[int, tuple[int, int, int]],
    compiler_path: Path,
    output_directory: Path,
    enable_autotune: bool,
) -> dict[str, Any]:
    operation = node["operation"]
    parameters = node["parameters"]
    tensors = node["tensors"]
    candidate = select_kernel_candidate("iluvatar", operation)
    source_relative = Path(candidate.source)
    if (
        candidate.backend != "iluvatar"
        or candidate.operation != operation
        or candidate.source_format != "module"
        or source_relative.is_absolute()
        or ".." in source_relative.parts
        or source_relative.suffix != ".py"
    ):
        raise ValueError("Iluvatar pointwise kernel candidate is invalid")
    expected_candidate_contract = {
        "common": ("common_triton", "kernels"),
        "platform": ("iluvatar_triton", "platform"),
    }.get(candidate.ownership)
    if (
        expected_candidate_contract is None
        or (
            candidate.provider,
            candidate.source_layout,
        )
        != expected_candidate_contract
    ):
        raise ValueError(
            "Iluvatar pointwise kernel ownership/provider contract is invalid"
        )
    function_name, signature, constants, default_grid = _pointwise_configuration(node)
    source_path = resolve_kernel_source(compiler_path, candidate)
    source_bytes = source_path.read_bytes()
    generated_bytes = materialize_kernel_source(source_path, candidate)
    generated_bytes = _specialize_pointwise_compute_source(generated_bytes, node)
    if (
        not source_bytes
        or not generated_bytes
        or len(source_bytes) > 1 << 20
        or len(generated_bytes) > 1 << 20
    ):
        raise ValueError("pointwise kernel source size is invalid")
    generated_path = output_directory / f"generated_stage_{stage_id}.py"
    _atomic_write(generated_path, generated_bytes)
    module = _load_generated_module(generated_path, stage_id)

    if function_name not in candidate.functions:
        raise RuntimeError("kernel registry and Iluvatar compiler disagree")
    function = getattr(module, function_name)
    pointwise_layout = tuple(
        [("tensor", None)] * len(tensors) + [("scalar_i32", "n_elements")]
    )
    argument_abi = _argument_abi(
        pointwise_layout, tensors, node["tensor_roles"], parameters, workspace
    )
    runtime_signature = _jit_runtime_signature(function, signature, argument_abi)
    artifact_arguments = _artifact_arguments(argument_abi)
    registry_source_hash = hashlib.sha256(source_bytes).hexdigest()
    generated_source_hash = hashlib.sha256(generated_bytes).hexdigest()
    # The materialized source includes the Graph compute precision. Use that
    # digest for execution/candidate identity; the unmodified common source
    # digest remains recorded separately for ownership/audit purposes.
    kernel_source_hash = generated_source_hash
    stage: dict[str, Any] = {
        "stage_id": stage_id,
        "kind": "kernel",
        "engine": "libtriton_jit",
        "source_node_ids": [node["id"]],
        "dependencies": dependencies,
        "operation": operation,
        "source_sha256": kernel_source_hash,
        "kernel": {
            "provider": candidate.provider,
            "ownership": candidate.ownership,
            "source": candidate.source,
            "registry_source_sha256": registry_source_hash,
            "compute_data_type": node["compute_data_type"],
            "function": function_name,
            "materialized_source": {
                "path": generated_path.name,
                "size": len(generated_bytes),
                "sha256": generated_source_hash,
            },
        },
    }

    def make_variant(
        variant_id: str,
        variant_constants: dict[str, int | float],
        num_warps: int,
        num_stages: int,
        grid: tuple[int, int, int],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "variant_id": variant_id,
            "source_sha256": kernel_source_hash,
            "full_signature": _jit_full_signature(
                function, runtime_signature, variant_constants
            ),
            "compile_options": {
                "num_warps": num_warps,
                "num_stages": num_stages,
            },
            "argument_count": len(artifact_arguments),
            "arguments": artifact_arguments,
            "launch": _jit_launch(grid, num_warps),
        }
        if config is not None:
            result["config"] = config
        return result

    if enable_autotune:
        configurations, tuning_source_hash = _load_pointwise_tuning(
            compiler_path, candidate
        )
        elements = _require_integer(parameters, "n_elements")
        variants: list[dict[str, Any]] = []
        for index, configuration in enumerate(configurations):
            variant_constants = dict(constants)
            for name, value in configuration["META"].items():
                variant_constants[name] = int(value)
            block_size = int(variant_constants["BLOCK_SIZE"])
            tiles_per_program = int(variant_constants.get("TILES_PER_PROGRAM", 1))
            pack_factor = int(variant_constants.get("PACK_FACTOR", 1))
            elements_per_program = block_size * tiles_per_program * pack_factor
            grid = (
                (elements + elements_per_program - 1) // elements_per_program,
                1,
                1,
            )
            variants.append(
                make_variant(
                    f"config_{index}",
                    variant_constants,
                    int(configuration["num_warps"]),
                    int(configuration["num_stages"]),
                    grid,
                    configuration,
                )
            )
        base_identity = {
            "schema_version": 1,
            "backend": "iluvatar",
            "target_backend": "corex",
            "warp_size": ILUVATAR_WARP_SIZE,
            "source_sha256": tuning_source_hash,
            "kernel_ownership": candidate.ownership,
            "kernel_provider": candidate.provider,
            "operation": operation,
            "function": function_name,
            "table": candidate.tuning.table,
            "key": candidate.tuning.key,
            "key_value": elements,
            "strategy": candidate.tuning.strategy,
            "configurations": configurations,
        }
        base_candidate_identity = hashlib.sha256(_canonical(base_identity)).hexdigest()
        rendered_identity = {
            "schema_version": 1,
            "engine": "libtriton_jit",
            "base_candidate_identity": base_candidate_identity,
            "variants": [
                {
                    "variant_id": variant["variant_id"],
                    "source_sha256": variant["source_sha256"],
                    "full_signature": variant["full_signature"],
                    "compile_options": variant["compile_options"],
                    "launch": variant["launch"],
                }
                for variant in variants
            ],
        }
        stage["variants"] = variants
        stage["tuning"] = {
            "schema_version": 1,
            "source": candidate.tuning.source,
            "source_sha256": tuning_source_hash,
            "table": candidate.tuning.table,
            "key": candidate.tuning.key,
            "strategy": candidate.tuning.strategy,
            "warmup": candidate.tuning.warmup,
            "repetitions": candidate.tuning.repetitions,
            "base_candidate_identity": base_candidate_identity,
            "candidate_identity": hashlib.sha256(
                _canonical(rendered_identity)
            ).hexdigest(),
        }
    else:
        stage["variants"] = [
            make_variant(
                "default",
                constants,
                num_warps=4,
                num_stages=1,
                grid=default_grid,
            )
        ]
    return stage


def _compile_tensor_stage(
    *,
    stage_id: int,
    node: dict[str, Any],
    dependencies: list[int],
    workspace: dict[int, tuple[int, int, int]],
    compiler_path: Path,
    output_directory: Path,
    enable_autotune: bool,
) -> dict[str, Any]:
    operation = node["operation"]
    parameters = node["parameters"]
    tensors = node["tensors"]
    configuration = tensor_dispatch.kernel_configuration(node)
    candidate = select_kernel_candidate("iluvatar", operation)
    source_relative = Path(candidate.source)
    if (
        candidate.backend != "iluvatar"
        or candidate.operation != operation
        or candidate.source_format != "module"
        or source_relative.is_absolute()
        or ".." in source_relative.parts
        or source_relative.suffix != ".py"
    ):
        raise ValueError("Iluvatar tensor kernel candidate is invalid")
    expected_candidate_contract = {
        "common": ("common_triton", "kernels"),
        "platform": ("iluvatar_triton", "platform"),
    }.get(candidate.ownership)
    if (
        expected_candidate_contract is None
        or (
            candidate.provider,
            candidate.source_layout,
        )
        != expected_candidate_contract
    ):
        raise ValueError(
            "Iluvatar tensor kernel ownership/provider contract is invalid"
        )
    source_path = resolve_kernel_source(compiler_path, candidate)
    source_bytes = source_path.read_bytes()
    generated_bytes = materialize_kernel_source(source_path, candidate)
    if (
        not source_bytes
        or not generated_bytes
        or len(source_bytes) > 1 << 20
        or len(generated_bytes) > 1 << 20
    ):
        raise ValueError("Iluvatar tensor kernel source size is invalid")
    generated_path = output_directory / f"generated_stage_{stage_id}.py"
    _atomic_write(generated_path, generated_bytes)
    module = _load_generated_module(generated_path, stage_id)

    function_name = configuration.function_name
    if function_name not in candidate.functions:
        raise RuntimeError("kernel registry and Iluvatar tensor planner disagree")
    function = getattr(module, function_name)
    argument_abi = _argument_abi(
        configuration.argument_layout,
        tensors,
        node["tensor_roles"],
        parameters,
        workspace,
    )
    runtime_signature = _jit_runtime_signature(
        function, configuration.runtime_signature, argument_abi
    )
    artifact_arguments = _artifact_arguments(argument_abi)
    kernel_source_hash = hashlib.sha256(source_bytes).hexdigest()
    generated_source_hash = hashlib.sha256(generated_bytes).hexdigest()
    stage: dict[str, Any] = {
        "stage_id": stage_id,
        "kind": "kernel",
        "engine": "libtriton_jit",
        "source_node_ids": [node["id"]],
        "dependencies": dependencies,
        "operation": operation,
        "source_sha256": kernel_source_hash,
        "kernel": {
            "provider": candidate.provider,
            "ownership": candidate.ownership,
            "source": candidate.source,
            "function": function_name,
            "materialized_source": {
                "path": generated_path.name,
                "size": len(generated_bytes),
                "sha256": generated_source_hash,
            },
        },
    }

    def make_variant(
        variant_id: str,
        constants: dict[str, int | float | bool],
        num_warps: int,
        num_stages: int,
        grid: tuple[int, int, int],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "variant_id": variant_id,
            "source_sha256": kernel_source_hash,
            "full_signature": _jit_full_signature(
                function, runtime_signature, constants
            ),
            "compile_options": {
                "num_warps": num_warps,
                "num_stages": num_stages,
            },
            "argument_count": len(artifact_arguments),
            "arguments": artifact_arguments,
            "launch": _jit_launch(grid, num_warps),
        }
        if config is not None:
            result["config"] = config
        return result

    if enable_autotune:
        tuning = candidate.tuning
        if tuning is None:
            raise ValueError("tensor kernel candidate has no tuning metadata")
        if configuration.tuning.meta_keys == ("BLOCK_SIZE",):
            configurations, tuning_source_hash = _load_pointwise_tuning(
                compiler_path, candidate
            )
        else:
            configurations, tuning_source_hash = _load_tensor_tuning(
                compiler_path, candidate, configuration
            )
        variants: list[dict[str, Any]] = []
        for index, config in enumerate(configurations):
            constants, grid = configuration.variant(
                config["META"],
                num_warps=config["num_warps"],
                num_stages=config["num_stages"],
            )
            variants.append(
                make_variant(
                    f"config_{index}",
                    constants,
                    config["num_warps"],
                    config["num_stages"],
                    grid,
                    config,
                )
            )
        base_identity = {
            "schema_version": 1,
            "backend": "iluvatar",
            "target_backend": "corex",
            "warp_size": ILUVATAR_WARP_SIZE,
            "source_sha256": tuning_source_hash,
            "kernel_ownership": candidate.ownership,
            "kernel_provider": candidate.provider,
            "operation": operation,
            "function": function_name,
            "table": tuning.table,
            "key": tuning.key,
            "key_value": configuration.tuning_key_value,
            "strategy": tuning.strategy,
            "configurations": configurations,
        }
        base_candidate_identity = hashlib.sha256(_canonical(base_identity)).hexdigest()
        rendered_identity = {
            "schema_version": 1,
            "engine": "libtriton_jit",
            "base_candidate_identity": base_candidate_identity,
            "variants": [
                {
                    "variant_id": variant["variant_id"],
                    "source_sha256": variant["source_sha256"],
                    "full_signature": variant["full_signature"],
                    "compile_options": variant["compile_options"],
                    "launch": variant["launch"],
                }
                for variant in variants
            ],
        }
        stage["variants"] = variants
        stage["tuning"] = {
            "schema_version": 1,
            "source": tuning.source,
            "source_sha256": tuning_source_hash,
            "table": tuning.table,
            "key": tuning.key,
            "strategy": tuning.strategy,
            "warmup": tuning.warmup,
            "repetitions": tuning.repetitions,
            "base_candidate_identity": base_candidate_identity,
            "candidate_identity": hashlib.sha256(
                _canonical(rendered_identity)
            ).hexdigest(),
        }
    else:
        stage["variants"] = [
            make_variant(
                "default",
                configuration.constants,
                configuration.default_num_warps,
                configuration.default_num_stages,
                configuration.default_grid,
            )
        ]
    return stage


def _compile_nn_stage(
    *,
    stage_id: int,
    node: dict[str, Any],
    configuration: nn_dispatch.KernelStagePlan,
    dependencies: list[int],
    workspace: dict[int, tuple[int, int, int]],
    local_workspace: dict[str, dict[str, int]],
    compiler_path: Path,
    output_directory: Path,
    enable_autotune: bool,
) -> dict[str, Any]:
    operation = node["operation"]
    tensors = node["tensors"]
    candidate = select_kernel_candidate("iluvatar", operation)
    source_relative = Path(candidate.source)
    if (
        candidate.backend != "iluvatar"
        or candidate.operation != operation
        or candidate.source_format != "module"
        or source_relative.is_absolute()
        or ".." in source_relative.parts
        or source_relative.suffix != ".py"
    ):
        raise ValueError("Iluvatar NN kernel candidate is invalid")
    expected_candidate_contract = {
        "common": ("common_triton", "kernels"),
        "platform": ("iluvatar_triton", "platform"),
    }.get(candidate.ownership)
    if (
        expected_candidate_contract is None
        or (
            candidate.provider,
            candidate.source_layout,
        )
        != expected_candidate_contract
    ):
        raise ValueError("Iluvatar NN kernel ownership/provider contract is invalid")
    source_path = resolve_kernel_source(compiler_path, candidate)
    source_bytes = source_path.read_bytes()
    generated_bytes = materialize_kernel_source(source_path, candidate)
    if (
        not source_bytes
        or not generated_bytes
        or len(source_bytes) > 1 << 20
        or len(generated_bytes) > 1 << 20
    ):
        raise ValueError("Iluvatar NN kernel source size is invalid")
    generated_path = output_directory / f"generated_stage_{stage_id}.py"
    _atomic_write(generated_path, generated_bytes)
    module = _load_generated_module(generated_path, stage_id)

    function_name = configuration.function_name
    if function_name not in candidate.functions:
        raise RuntimeError("kernel registry and Iluvatar NN planner disagree")
    function = getattr(module, function_name)
    argument_abi = _nn_argument_abi(
        configuration,
        tensors,
        node["tensor_roles"],
        workspace,
        local_workspace,
    )
    runtime_signature = _jit_runtime_signature(
        function, configuration.runtime_signature, argument_abi
    )
    artifact_arguments = _artifact_arguments(argument_abi)
    kernel_source_hash = hashlib.sha256(source_bytes).hexdigest()
    generated_source_hash = hashlib.sha256(generated_bytes).hexdigest()
    stage: dict[str, Any] = {
        "stage_id": stage_id,
        "kind": "kernel",
        "engine": "libtriton_jit",
        "source_node_ids": list(node.get("source_node_ids", [node["id"]])),
        "dependencies": dependencies,
        "operation": operation,
        "substage": configuration.stage_name,
        "source_sha256": kernel_source_hash,
        "kernel": {
            "provider": candidate.provider,
            "ownership": candidate.ownership,
            "source": candidate.source,
            "function": function_name,
            "materialized_source": {
                "path": generated_path.name,
                "size": len(generated_bytes),
                "sha256": generated_source_hash,
            },
        },
    }

    def make_variant(
        variant_id: str,
        constants: dict[str, int | float | bool],
        num_warps: int,
        num_stages: int,
        grid: tuple[int, int, int],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "variant_id": variant_id,
            "source_sha256": kernel_source_hash,
            "full_signature": _jit_full_signature(
                function, runtime_signature, constants
            ),
            "compile_options": {
                "num_warps": num_warps,
                "num_stages": num_stages,
            },
            "argument_count": len(artifact_arguments),
            "arguments": artifact_arguments,
            "launch": _jit_launch(grid, num_warps),
        }
        if config is not None:
            result["config"] = config
        return result

    stage_autotune = (
        enable_autotune
        and bool(configuration.tuning.source)
        and bool(configuration.tuning.meta_keys)
    )
    if stage_autotune:
        tuning = configuration.tuning
        configurations, tuning_source_hash = _load_nn_tuning(
            compiler_path, candidate, configuration
        )
        variants: list[dict[str, Any]] = []
        for index, config in enumerate(configurations):
            constants, grid = configuration.variant(
                config["META"],
                num_warps=config["num_warps"],
                num_stages=config["num_stages"],
            )
            variants.append(
                make_variant(
                    f"config_{index}",
                    constants,
                    config["num_warps"],
                    config["num_stages"],
                    grid,
                    config,
                )
            )
        base_identity = {
            "schema_version": 1,
            "backend": "iluvatar",
            "target_backend": "corex",
            "warp_size": ILUVATAR_WARP_SIZE,
            "source_sha256": tuning_source_hash,
            "kernel_ownership": candidate.ownership,
            "kernel_provider": candidate.provider,
            "operation": operation,
            "substage": configuration.stage_name,
            "function": function_name,
            "table": tuning.table,
            "key": tuning.key,
            "key_value": configuration.tuning_key_value,
            "strategy": tuning.strategy,
            "configurations": configurations,
        }
        base_candidate_identity = hashlib.sha256(_canonical(base_identity)).hexdigest()
        rendered_identity = {
            "schema_version": 1,
            "engine": "libtriton_jit",
            "base_candidate_identity": base_candidate_identity,
            "variants": [
                {
                    "variant_id": variant["variant_id"],
                    "source_sha256": variant["source_sha256"],
                    "full_signature": variant["full_signature"],
                    "compile_options": variant["compile_options"],
                    "launch": variant["launch"],
                }
                for variant in variants
            ],
        }
        stage["variants"] = variants
        stage["tuning"] = {
            "schema_version": 1,
            "source": tuning.source,
            "source_sha256": tuning_source_hash,
            "table": tuning.table,
            "key": tuning.key,
            "strategy": tuning.strategy,
            "warmup": tuning.warmup,
            "repetitions": tuning.repetitions,
            "base_candidate_identity": base_candidate_identity,
            "candidate_identity": hashlib.sha256(
                _canonical(rendered_identity)
            ).hexdigest(),
        }
    else:
        stage["variants"] = [
            make_variant(
                "default",
                configuration.constants,
                configuration.default_num_warps,
                configuration.default_num_stages,
                configuration.default_grid,
            )
        ]
    return stage
