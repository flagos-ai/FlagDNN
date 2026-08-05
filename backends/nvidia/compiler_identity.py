"""NVIDIA compiler identity and artifact-cache invalidation inputs."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import triton
import yaml
from triton.backends.nvidia.compiler import get_ptxas, get_ptxas_version

from flagdnn_codegen.kernel_registry import (
    iter_kernel_candidates,
    iter_kernel_registry_sources,
    resolve_kernel_source,
    resolve_tuning_source,
)


def build_compiler_identity(
    target_name: str,
    execution_engine: str,
    *,
    provider_path: Path,
    compiler_entry: Path,
    provider_name: str,
    provider_version: str,
    graph_schema_version: int,
    artifact_schema_version: int,
    execution_program_version: int,
) -> dict[str, Any]:
    """Hash every input that can change generated NVIDIA artifacts."""
    if not isinstance(target_name, str) or not target_name.startswith("sm_"):
        raise ValueError("NVIDIA target fingerprint is invalid")
    if execution_engine not in {"external_artifact", "libtriton_jit"}:
        raise ValueError("NVIDIA execution engine is invalid")

    provider_path = provider_path.resolve()
    compiler_entry = compiler_entry.resolve()
    identity_inputs: dict[str, Path] = {
        "provider:nvidia": provider_path,
        "provider_identity:nvidia": Path(__file__).resolve(),
        "driver": compiler_entry,
    }
    registry_source = compiler_entry.with_name("kernel_registry.py")
    if registry_source.is_file():
        identity_inputs["registry"] = registry_source
    loader_source = compiler_entry.with_name("provider_loader.py")
    if loader_source.is_file():
        identity_inputs["provider_loader"] = loader_source

    resource_root = compiler_entry.parents[2]
    for registry_path in iter_kernel_registry_sources("nvidia"):
        registry_label = registry_path.relative_to(resource_root).as_posix()
        identity_inputs[f"registry:{registry_label}"] = registry_path
    for candidate in iter_kernel_candidates("nvidia"):
        identity_inputs[f"kernel:{candidate.source}"] = resolve_kernel_source(
            compiler_entry, candidate
        ).resolve()
        if candidate.tuning is not None:
            identity_inputs[f"tuning:{candidate.tuning.source}"] = (
                resolve_tuning_source(compiler_entry, candidate).resolve()
            )

    source_files = {
        label: hashlib.sha256(source.read_bytes()).hexdigest()
        for label, source in sorted(identity_inputs.items())
    }
    target_architecture = int(target_name.removeprefix("sm_"))
    ptxas = get_ptxas(target_architecture)
    ptxas_path = Path(ptxas.path).resolve()
    if not ptxas_path.is_file():
        raise ValueError("Triton ptxas executable is missing")

    payload: dict[str, Any] = {
        "provider": provider_name,
        "provider_version": provider_version,
        "graph_schema_version": graph_schema_version,
        "artifact_schema_version": artifact_schema_version,
        "execution_program_version": execution_program_version,
        "execution_engine": execution_engine,
        "target": target_name,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_cache_tag": sys.implementation.cache_tag,
        "triton_version": triton.__version__,
        "yaml_version": yaml.__version__,
        "ptxas_version": get_ptxas_version(target_architecture).strip(),
        "ptxas_sha256": hashlib.sha256(ptxas_path.read_bytes()).hexdigest(),
        "source_files": source_files,
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload["identity_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload
