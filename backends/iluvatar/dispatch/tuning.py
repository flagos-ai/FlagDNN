# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Iluvatar dispatch / tuning implementation."""

from __future__ import annotations

from ..codegen.io import _canonical
from . import nn as nn_dispatch
from . import tensor as tensor_dispatch
from .common import ILUVATAR_MAX_SHARED_MEMORY_BYTES
from .common import ILUVATAR_WARP_SIZE
from .common import _WGRAD_DOT_ELEMENT_BYTES
from .common import _WGRAD_DOT_RIGHT_META
from flagdnn_codegen.kernel_registry import resolve_tuning_source
from pathlib import Path
from typing import Any
import hashlib
import itertools
import yaml


def _load_pointwise_tuning(
    compiler_path: Path, candidate: Any
) -> tuple[list[dict[str, Any]], str]:
    if candidate.tuning is None:
        raise ValueError("pointwise kernel candidate has no tuning metadata")
    source = resolve_tuning_source(compiler_path, candidate)
    document = yaml.safe_load(source.read_bytes())
    if not isinstance(document, dict):
        raise ValueError("Iluvatar tuning source must be a mapping")
    entries = document.get(candidate.tuning.table)
    if not isinstance(entries, list) or not entries:
        raise ValueError("Iluvatar pointwise tuning table is missing")

    expanded: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry:
            raise ValueError("Iluvatar tuning entry must be a mapping")
        if entry.get("gen") is True:
            param_map = entry.get("param_map")
            expected_map = {
                "META": {"BLOCK_SIZE": "block_size"},
                "num_warps": "warps",
                "num_stages": "stages",
            }
            if param_map != expected_map:
                raise ValueError("Iluvatar pointwise tuning param_map is invalid")
            dimensions: list[list[int]] = []
            for name in ("block_size", "warps", "stages"):
                values = entry.get(name)
                if (
                    not isinstance(values, list)
                    or not values
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value <= 0
                        for value in values
                    )
                ):
                    raise ValueError(f"Iluvatar tuning parameter {name} is invalid")
                dimensions.append(values)
            for block_size, warps, stages in itertools.product(*dimensions):
                expanded.append(
                    {
                        "META": {"BLOCK_SIZE": block_size},
                        "num_warps": warps,
                        "num_stages": stages,
                    }
                )
        elif "gen" in entry:
            raise ValueError("Iluvatar tuning gen flag must be true")
        else:
            expanded.append(entry)

    expected_meta_keys = {"BLOCK_SIZE"}

    configurations: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for entry in expanded:
        meta = entry.get("META")
        if not isinstance(meta, dict) or set(meta) != expected_meta_keys:
            raise ValueError("Iluvatar pointwise tuning META is invalid")
        block_size = meta["BLOCK_SIZE"]
        num_warps = entry.get("num_warps", 4)
        num_stages = entry.get("num_stages", 1)
        if (
            isinstance(block_size, bool)
            or not isinstance(block_size, int)
            or block_size < 32
            or block_size > 65536
            or block_size & (block_size - 1) != 0
        ):
            raise ValueError("Iluvatar BLOCK_SIZE must be a power of two")
        if (
            isinstance(num_warps, bool)
            or not isinstance(num_warps, int)
            or num_warps <= 0
            or (num_warps & (num_warps - 1)) != 0
            or num_warps * ILUVATAR_WARP_SIZE > 1024
        ):
            raise ValueError(
                "Iluvatar tuning num_warps must be a positive power of two "
                "within the workgroup limit"
            )
        if (
            isinstance(num_stages, bool)
            or not isinstance(num_stages, int)
            or not 1 <= num_stages <= 32
        ):
            raise ValueError("Iluvatar tuning num_stages is invalid")
        normalized_meta = {name: int(meta[name]) for name in sorted(meta)}
        configuration = {
            "META": normalized_meta,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
        encoded = _canonical(configuration)
        if encoded not in seen:
            seen.add(encoded)
            configurations.append(configuration)
    if not 2 <= len(configurations) <= 1024:
        raise ValueError("Iluvatar autotune needs between 2 and 1024 candidates")
    selected_payload = {
        "schema_version": 1,
        "table": candidate.tuning.table,
        "configurations": configurations,
    }
    return (
        configurations,
        hashlib.sha256(_canonical(selected_payload)).hexdigest(),
    )


def _load_tensor_tuning(
    compiler_path: Path,
    candidate: Any,
    kernel_configuration: tensor_dispatch.KernelConfiguration,
) -> tuple[list[dict[str, Any]], str]:
    tuning = candidate.tuning
    if tuning is None:
        raise ValueError("tensor kernel candidate has no tuning metadata")
    expected = kernel_configuration.tuning
    actual_contract = (
        tuning.source,
        tuning.table,
        tuning.key,
        tuning.strategy,
        tuning.warmup,
        tuning.repetitions,
    )
    expected_contract = (
        expected.source,
        expected.table,
        expected.key,
        expected.strategy,
        expected.warmup,
        expected.repetitions,
    )
    if actual_contract != expected_contract:
        raise ValueError("kernel registry and tensor tuning contract disagree")

    source = resolve_tuning_source(compiler_path, candidate)
    document = yaml.safe_load(source.read_bytes())
    if not isinstance(document, dict):
        raise ValueError("Iluvatar tuning source must be a mapping")
    entries = document.get(tuning.table)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Iluvatar tuning table {tuning.table!r} is missing")

    expected_meta = set(expected.meta_keys)
    configurations: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not entry or "gen" in entry:
            raise ValueError("Iluvatar tensor tuning entries must be explicit")
        meta = entry.get("META")
        if not isinstance(meta, dict) or set(meta) != expected_meta:
            raise ValueError("Iluvatar tensor tuning META is invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in meta.values()
        ):
            raise ValueError("Iluvatar tensor tuning META values are invalid")
        num_warps = entry.get("num_warps", 4)
        num_stages = entry.get("num_stages", 1)
        kernel_configuration.variant(meta, num_warps=num_warps, num_stages=num_stages)
        configuration = {
            "META": dict(meta),
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
        encoded = _canonical(configuration)
        if encoded not in seen:
            seen.add(encoded)
            configurations.append(configuration)
    if not 2 <= len(configurations) <= 1024:
        raise ValueError("Iluvatar autotune needs between 2 and 1024 candidates")
    selected_payload = {
        "schema_version": 1,
        "table": tuning.table,
        "configurations": configurations,
    }
    return (
        configurations,
        hashlib.sha256(_canonical(selected_payload)).hexdigest(),
    )


def _wgrad_dot_tuning_fits_shared_memory(
    configuration: nn_dispatch.KernelStagePlan,
    meta: dict[str, Any],
) -> bool:
    right_meta = _WGRAD_DOT_RIGHT_META.get(configuration.function_name)
    if right_meta is None:
        return True
    if configuration.operation != "convolution_wgrad":
        raise ValueError("Iluvatar WGrad dot function has the wrong operation")

    token = configuration.runtime_signature.get("dy_ptr")
    element_bytes = _WGRAD_DOT_ELEMENT_BYTES.get(token)
    if element_bytes is None:
        raise ValueError("Iluvatar WGrad dot stage has an invalid dy_ptr ABI token")

    required_meta = ("BLOCK_M", "BLOCK_OC", right_meta)
    missing_meta = [name for name in required_meta if name not in meta]
    if missing_meta:
        raise ValueError(
            "Iluvatar WGrad dot tuning META is missing: " + ", ".join(missing_meta)
        )
    values = [meta[name] for name in required_meta]
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        raise ValueError("Iluvatar WGrad dot tuning META values are invalid")
    block_m, block_oc, block_right = values
    required_bytes = block_m * (block_oc + block_right) * element_bytes
    return required_bytes <= ILUVATAR_MAX_SHARED_MEMORY_BYTES


def _load_nn_tuning(
    compiler_path: Path,
    candidate: Any,
    configuration: nn_dispatch.KernelStagePlan,
) -> tuple[list[dict[str, Any]], str]:
    tuning = candidate.tuning
    if tuning is None:
        raise ValueError("NN kernel candidate has no tuning metadata")
    expected = configuration.tuning
    actual_contract = (
        tuning.source,
        tuning.table,
        tuning.key,
        tuning.strategy,
        tuning.warmup,
        tuning.repetitions,
    )
    expected_contract = (
        expected.source,
        expected.table,
        expected.key,
        expected.strategy,
        expected.warmup,
        expected.repetitions,
    )
    if actual_contract != expected_contract:
        raise ValueError("kernel registry and NN tuning contract disagree")

    source = resolve_tuning_source(compiler_path, candidate)
    document = yaml.safe_load(source.read_bytes())
    if not isinstance(document, dict):
        raise ValueError("Iluvatar tuning source must be a mapping")
    entries = document.get(tuning.table)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Iluvatar tuning table {tuning.table!r} is missing")

    expected_meta = set(expected.meta_keys)
    if not expected_meta:
        raise ValueError("untunable Iluvatar NN stage requested autotuning")
    configurations: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not entry or "gen" in entry:
            raise ValueError("Iluvatar NN tuning entries must be explicit")
        raw_meta = entry.get("META")
        if not isinstance(raw_meta, dict):
            raise ValueError("Iluvatar NN tuning META is invalid")
        if not expected_meta.issubset(raw_meta):
            continue
        meta = {name: raw_meta[name] for name in expected.meta_keys}
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in meta.values()
        ):
            raise ValueError("Iluvatar NN tuning META values are invalid")
        if not _wgrad_dot_tuning_fits_shared_memory(configuration, meta):
            continue
        if (
            configuration.operation == "batchnorm"
            and configuration.function_name == "batch_norm_nchw_kernel"
            and "BLOCK_SIZE" in meta
            and meta["BLOCK_SIZE"] < configuration.constants.get("BLOCK_SIZE", 1)
        ):
            continue
        num_warps = entry.get("num_warps", 4)
        num_stages = entry.get("num_stages", 1)
        configuration.variant(meta, num_warps=num_warps, num_stages=num_stages)
        projected = {
            "META": meta,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
        encoded = _canonical(projected)
        if encoded not in seen:
            seen.add(encoded)
            configurations.append(projected)
    if not 2 <= len(configurations) <= 1024:
        raise ValueError("Iluvatar NN autotune needs between 2 and 1024 candidates")
    selected_payload = {
        "schema_version": 1,
        "table": tuning.table,
        "stage": configuration.stage_name,
        "configurations": configurations,
    }
    return (
        configurations,
        hashlib.sha256(_canonical(selected_payload)).hexdigest(),
    )
