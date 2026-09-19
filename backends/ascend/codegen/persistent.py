"""Materialize persistent Ascend stages and their tuning candidates."""

from pathlib import Path
from typing import Any
from .emit import _build_stage
from .io import (
    _compiler_entry_path,
    _materialize_source,
    _validate_non_tle_kernel_source,
)
from .signature import _validate_kernel_function
from ..compiler_identity import aicore_count_from_target
from ..dispatch.common import _MAX_KERNEL_SOURCE_SIZE, require_object
from ..dispatch.graph import plan_graph
from ..dispatch.selection import _load_capabilities, _validate_platform_candidate
from ..tuning_decoder import (
    TuningConfiguration,
    load_tuning_table,
    validate_add_configuration,
)
from flagdnn_codegen.kernel_registry import (
    KernelCandidate,
    materialize_kernel_source,
    resolve_kernel_source,
    resolve_tuning_source,
    select_kernel_candidate,
)


def emit(graph, target_name, identity, enable_autotune, output_directory):
    ai_core_count = aicore_count_from_target(target_name)
    graph_plan = plan_graph(graph)
    capabilities = _load_capabilities()
    persistent_launch = require_object(
        capabilities["persistent_launch"], "persistent_launch"
    )
    if not (
        int(persistent_launch["minimum_ai_core_count"])
        <= ai_core_count
        <= int(persistent_launch["maximum_ai_core_count"])
    ):
        raise ValueError("Ascend target AI core count exceeds capability")
    worker_count = ai_core_count * int(persistent_launch["workers_per_ai_core"])
    candidate_by_operation: dict[str, KernelCandidate] = {}
    for operation in sorted({stage.operation for stage in graph_plan.stages}):
        candidate = select_kernel_candidate("ascend", operation)
        _validate_platform_candidate(operation, candidate)
        candidate_by_operation[operation] = candidate

    compiler_path = _compiler_entry_path()
    output_directory.mkdir(parents=True, exist_ok=True)
    stage_assets: dict[
        str,
        tuple[tuple[TuningConfiguration, ...], str, str, str, int],
    ] = {}
    for operation, candidate in candidate_by_operation.items():
        kernel_source_path = resolve_kernel_source(compiler_path, candidate)
        source_bytes = materialize_kernel_source(kernel_source_path, candidate)
        if not source_bytes or len(source_bytes) > _MAX_KERNEL_SOURCE_SIZE:
            raise ValueError("Ascend pointwise kernel source size is invalid")
        _validate_non_tle_kernel_source(source_bytes)
        for stage in graph_plan.stages:
            if stage.operation == operation:
                _validate_kernel_function(
                    source_bytes, candidate.source, stage.function_name
                )

        tuning = candidate.tuning
        if tuning is None:
            raise RuntimeError("validated Ascend pointwise candidate lost tuning")
        tuning_path = resolve_tuning_source(compiler_path, candidate)
        configurations, tuning_source_sha256 = load_tuning_table(
            tuning_path, tuning.table
        )
        for configuration in configurations:
            validate_add_configuration(
                configuration,
                capabilities,
                kernel_family=(
                    next(
                        stage.kernel_family
                        for stage in graph_plan.stages
                        if stage.operation == operation
                    )
                ),
            )
        source_path, source_sha256 = _materialize_source(
            output_directory=output_directory,
            source_bytes=source_bytes,
            compiler_identity_sha256=identity["identity_sha256"],
        )
        stage_assets[operation] = (
            configurations,
            tuning_source_sha256,
            source_path,
            source_sha256,
            len(source_bytes),
        )

    stages = []
    for stage in graph_plan.stages:
        (
            configurations,
            tuning_source_sha256,
            source_path,
            source_sha256,
            source_size,
        ) = stage_assets[stage.operation]
        stages.append(
            _build_stage(
                plan=stage,
                candidate=candidate_by_operation[stage.operation],
                configurations=configurations,
                tuning_source_sha256=tuning_source_sha256,
                enable_autotune=(
                    enable_autotune
                    and stage.function_name != "convolution_fprop_im2col_kernel"
                ),
                capabilities=capabilities,
                source_path=source_path,
                source_sha256=source_sha256,
                source_size=source_size,
                worker_count=worker_count,
            )
        )

    return stages, graph_plan.workspace_size
