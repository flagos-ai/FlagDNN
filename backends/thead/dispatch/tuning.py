# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Load and validate PPU tuning configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import yaml  # type: ignore[import-untyped]
from ..dispatch.common import (
    _APPROVED_PPU_COMPILER_OPTIONS,
    _TUNING_TABLES,
    _integer,
    _require_exact_fields,
    _require_object,
)


def _validate_ppu_option_value(value: object, description: str) -> None:
    if isinstance(value, bool) or isinstance(value, str):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    raise ValueError(f"{description} must be a boolean, integer, or string")


def _load_pointwise_tuning(
    tuning_path: Path,
    candidate: Any,
    operation: str,
) -> tuple[list[dict[str, Any]], bytes]:
    operation_label = operation.capitalize()
    tuning_bytes = tuning_path.read_bytes()
    if not tuning_bytes:
        raise ValueError(f"THead {operation_label} tuning source is empty")
    try:
        document = yaml.safe_load(tuning_bytes)
    except yaml.YAMLError as error:
        raise ValueError(
            f"THead {operation_label} tuning source is invalid: {error}"
        ) from error
    table = candidate.tuning.table
    if not isinstance(document, dict) or set(document) != _TUNING_TABLES:
        raise ValueError(
            f"THead {operation_label} tuning source has an invalid table set"
        )
    entries = document[table]
    if not isinstance(entries, list) or len(entries) != 2:
        raise ValueError(
            f"THead {operation_label} autotune requires exactly two candidates"
        )

    configurations: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict):
            raise ValueError(
                f"THead {operation_label} tuning candidate {index} "
                "must be a mapping"
            )
        _require_exact_fields(
            raw_entry,
            {
                "META",
                "num_warps",
                "num_stages",
                "maxnreg",
                "ppu_compiler_options",
            },
            set(),
            f"THead {operation_label} tuning candidate {index}",
        )
        meta = _require_object(
            raw_entry["META"], f"THead {operation_label} tuning META"
        )
        _require_exact_fields(
            meta,
            {"BLOCK_SIZE"},
            set(),
            f"THead {operation_label} tuning META",
        )
        block_size = _integer(
            meta["BLOCK_SIZE"], f"THead {operation_label} BLOCK_SIZE"
        )
        if (
            block_size < 32
            or block_size > 65536
            or block_size & (block_size - 1)
        ):
            raise ValueError(
                f"THead {operation_label} BLOCK_SIZE must be a power of two"
            )
        num_warps = _integer(
            raw_entry["num_warps"],
            f"THead {operation_label} tuning num_warps",
        )
        if num_warps <= 0 or num_warps > 32 or num_warps & (num_warps - 1):
            raise ValueError(
                f"THead {operation_label} tuning num_warps is invalid"
            )
        num_stages = _integer(
            raw_entry["num_stages"],
            f"THead {operation_label} tuning num_stages",
        )
        if not 1 <= num_stages <= 16:
            raise ValueError(
                f"THead {operation_label} tuning num_stages is invalid"
            )

        maxnreg = raw_entry["maxnreg"]
        if maxnreg is not None:
            maxnreg = _integer(
                maxnreg, f"THead {operation_label} tuning maxnreg"
            )
            if not 1 <= maxnreg <= 2**31 - 1:
                raise ValueError(
                    f"THead {operation_label} tuning maxnreg is invalid"
                )

        raw_options = _require_object(
            raw_entry["ppu_compiler_options"],
            f"THead {operation_label} ppu_compiler_options",
        )
        unknown_options = set(raw_options).difference(
            _APPROVED_PPU_COMPILER_OPTIONS
        )
        if unknown_options:
            raise ValueError(
                f"THead {operation_label} tuning has unapproved PPU "
                "compiler options: " + ", ".join(sorted(unknown_options))
            )
        ppu_options: dict[str, object] = {}
        for name in sorted(raw_options):
            _validate_ppu_option_value(
                raw_options[name], f"THead PPU compiler option {name}"
            )
            ppu_options[name] = raw_options[name]

        # CUDA-backend libtriton_jit's dynamic raw-argument entry point only
        # accepts num_warps and num_stages.  Keep all PPU-specific switches in
        # the schema and candidate identity, but reject values that the current
        # ABI could otherwise silently ignore.
        if maxnreg is not None or ppu_options:
            raise ValueError(
                "THead raw-argument libtriton_jit cannot convey maxnreg or "
                "PPU compiler options"
            )

        configuration = {
            "META": {"BLOCK_SIZE": block_size},
            "maxnreg": maxnreg,
            "num_stages": num_stages,
            "num_warps": num_warps,
            "ppu_compiler_options": ppu_options,
        }
        encoded = json.dumps(
            configuration,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if encoded in seen:
            raise ValueError(
                f"THead {operation_label} tuning candidates must be unique"
            )
        seen.add(encoded)
        configurations.append(configuration)
    return configurations, tuning_bytes


def _load_normalization_tuning(
    tuning_path: Path, candidate: Any, operation: str
) -> tuple[list[dict[str, Any]], bytes]:
    tuning_bytes = tuning_path.read_bytes()
    if not tuning_bytes:
        raise ValueError(f"THead {operation} tuning source is empty")
    try:
        document = yaml.safe_load(tuning_bytes)
    except yaml.YAMLError as error:
        raise ValueError(
            f"THead {operation} tuning source is invalid: {error}"
        ) from error
    if not isinstance(document, dict) or set(document) != _TUNING_TABLES:
        raise ValueError(
            f"THead {operation} tuning source has an invalid table set"
        )
    entries = document.get(candidate.tuning.table)
    if not isinstance(entries, list) or len(entries) != 2:
        raise ValueError(
            f"THead {operation} autotune requires exactly two candidates"
        )
    configurations: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for index, raw_entry in enumerate(entries):
        entry = _require_object(
            raw_entry, f"THead {operation} tuning candidate {index}"
        )
        _require_exact_fields(
            entry,
            {
                "META",
                "num_warps",
                "num_stages",
                "maxnreg",
                "ppu_compiler_options",
            },
            set(),
            f"THead {operation} tuning candidate {index}",
        )
        meta = _require_object(entry["META"], f"THead {operation} tuning META")
        _require_exact_fields(
            meta,
            {"BLOCK_SIZE", "ROWS_PER_PROGRAM"},
            set(),
            f"THead {operation} tuning META",
        )
        block_size = _integer(
            meta["BLOCK_SIZE"], f"THead {operation} BLOCK_SIZE"
        )
        rows_per_program = _integer(
            meta["ROWS_PER_PROGRAM"],
            f"THead {operation} ROWS_PER_PROGRAM",
        )
        num_warps = _integer(
            entry["num_warps"], f"THead {operation} num_warps"
        )
        num_stages = _integer(
            entry["num_stages"], f"THead {operation} num_stages"
        )
        options = _require_object(
            entry["ppu_compiler_options"],
            f"THead {operation} ppu_compiler_options",
        )
        if (
            block_size not in {128, 256}
            or rows_per_program != 1
            or num_warps != 4
            or num_stages != 1
            or entry["maxnreg"] is not None
            or options
        ):
            raise ValueError(
                f"THead {operation} tuning is outside the qualified PPU slice"
            )
        configuration = {
            "META": {
                "BLOCK_SIZE": block_size,
                "ROWS_PER_PROGRAM": rows_per_program,
            },
            "maxnreg": None,
            "num_stages": num_stages,
            "num_warps": num_warps,
            "ppu_compiler_options": {},
        }
        encoded = json.dumps(
            configuration, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if encoded in seen:
            raise ValueError(f"THead {operation} tuning candidates repeat")
        seen.add(encoded)
        configurations.append(configuration)
    return configurations, tuning_bytes


def _load_reduction_tuning(
    tuning_path: Path, candidate: Any, operation: str
) -> tuple[list[dict[str, Any]], bytes]:
    tuning_bytes = tuning_path.read_bytes()
    if not tuning_bytes:
        raise ValueError("THead reduction tuning source is empty")
    try:
        document = yaml.safe_load(tuning_bytes)
    except yaml.YAMLError as error:
        raise ValueError(
            f"THead reduction tuning is invalid: {error}"
        ) from error
    if not isinstance(document, dict) or set(document) != _TUNING_TABLES:
        raise ValueError("THead reduction tuning table set is invalid")
    entries = document.get(candidate.tuning.table)
    if not isinstance(entries, list) or len(entries) != 2:
        raise ValueError("THead reduction autotune requires two candidates")
    configurations: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for index, raw_entry in enumerate(entries):
        entry = _require_object(
            raw_entry, f"THead {operation} tuning candidate {index}"
        )
        _require_exact_fields(
            entry,
            {
                "META",
                "num_warps",
                "num_stages",
                "maxnreg",
                "ppu_compiler_options",
            },
            set(),
            f"THead {operation} tuning candidate {index}",
        )
        meta = _require_object(entry["META"], "THead reduction tuning META")
        _require_exact_fields(
            meta,
            {"BLOCK_M", "BLOCK_N"},
            set(),
            "THead reduction tuning META",
        )
        block_m = _integer(meta["BLOCK_M"], "THead reduction BLOCK_M")
        block_n = _integer(meta["BLOCK_N"], "THead reduction BLOCK_N")
        if any(
            value < 1 or value > 65536 or value & (value - 1)
            for value in (block_m, block_n)
        ):
            raise ValueError("THead reduction blocks must be powers of two")
        num_warps = _integer(entry["num_warps"], "THead reduction num_warps")
        num_stages = _integer(
            entry["num_stages"], "THead reduction num_stages"
        )
        if (
            num_warps < 1
            or num_warps > 32
            or num_warps & (num_warps - 1)
            or not 1 <= num_stages <= 16
            or entry["maxnreg"] is not None
            or _require_object(
                entry["ppu_compiler_options"],
                "THead reduction ppu_compiler_options",
            )
        ):
            raise ValueError("THead reduction tuning options are invalid")
        configuration = {
            "META": {"BLOCK_M": block_m, "BLOCK_N": block_n},
            "maxnreg": None,
            "num_stages": num_stages,
            "num_warps": num_warps,
            "ppu_compiler_options": {},
        }
        encoded = json.dumps(
            configuration, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if encoded in seen:
            raise ValueError("THead reduction tuning candidates repeat")
        seen.add(encoded)
        configurations.append(configuration)
    return configurations, tuning_bytes


def _load_matmul_tuning(
    tuning_path: Path, candidate: Any
) -> tuple[list[dict[str, Any]], bytes]:
    tuning_bytes = tuning_path.read_bytes()
    if not tuning_bytes:
        raise ValueError("THead MatMul tuning source is empty")
    try:
        document = yaml.safe_load(tuning_bytes)
    except yaml.YAMLError as error:
        raise ValueError(f"THead MatMul tuning is invalid: {error}") from error
    if not isinstance(document, dict) or set(document) != _TUNING_TABLES:
        raise ValueError("THead MatMul tuning table set is invalid")
    entries = document.get(candidate.tuning.table)
    if not isinstance(entries, list) or len(entries) != 2:
        raise ValueError("THead MatMul autotune requires two candidates")
    configurations: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for index, raw_entry in enumerate(entries):
        entry = _require_object(
            raw_entry, f"THead MatMul tuning candidate {index}"
        )
        _require_exact_fields(
            entry,
            {
                "META",
                "num_warps",
                "num_stages",
                "maxnreg",
                "ppu_compiler_options",
            },
            set(),
            f"THead MatMul tuning candidate {index}",
        )
        meta = _require_object(entry["META"], "THead MatMul tuning META")
        _require_exact_fields(
            meta,
            {"BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M"},
            set(),
            "THead MatMul tuning META",
        )
        block_m = _integer(meta["BLOCK_M"], "THead MatMul BLOCK_M")
        block_n = _integer(meta["BLOCK_N"], "THead MatMul BLOCK_N")
        block_k = _integer(meta["BLOCK_K"], "THead MatMul BLOCK_K")
        group_m = _integer(meta["GROUP_M"], "THead MatMul GROUP_M")
        if any(
            value < 1 or value > 256 or value & (value - 1)
            for value in (block_m, block_n, block_k, group_m)
        ):
            raise ValueError(
                "THead MatMul blocks and group must be bounded powers of two"
            )
        num_warps = _integer(entry["num_warps"], "THead MatMul num_warps")
        num_stages = _integer(entry["num_stages"], "THead MatMul num_stages")
        options = _require_object(
            entry["ppu_compiler_options"],
            "THead MatMul ppu_compiler_options",
        )
        if (
            num_warps < 1
            or num_warps > 32
            or num_warps & (num_warps - 1)
            or not 1 <= num_stages <= 16
            or entry["maxnreg"] is not None
            or options
        ):
            raise ValueError("THead MatMul tuning options are invalid")
        configuration = {
            "META": {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "GROUP_M": group_m,
            },
            "maxnreg": None,
            "num_stages": num_stages,
            "num_warps": num_warps,
            "ppu_compiler_options": {},
        }
        encoded = json.dumps(
            configuration, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if encoded in seen:
            raise ValueError("THead MatMul tuning candidates repeat")
        seen.add(encoded)
        configurations.append(configuration)
    return configurations, tuning_bytes


def _load_convolution_tuning(
    tuning_path: Path, candidate: Any, operation: str
) -> tuple[list[dict[str, Any]], bytes]:
    tuning_bytes = tuning_path.read_bytes()
    if not tuning_bytes:
        raise ValueError("THead convolution tuning source is empty")
    try:
        document = yaml.safe_load(tuning_bytes)
    except yaml.YAMLError as error:
        raise ValueError(
            f"THead convolution tuning is invalid: {error}"
        ) from error
    if not isinstance(document, dict) or set(document) != _TUNING_TABLES:
        raise ValueError("THead convolution tuning table set is invalid")
    entries = document.get(candidate.tuning.table)
    if not isinstance(entries, list) or len(entries) != 2:
        raise ValueError("THead convolution autotune requires two candidates")
    configurations: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    meta_names = {
        "BLOCK_M",
        "BLOCK_OC",
        "BLOCK_CI",
        "BLOCK_K",
        "BLOCK_HW",
    }
    for index, raw_entry in enumerate(entries):
        entry = _require_object(
            raw_entry, f"THead {operation} tuning candidate {index}"
        )
        _require_exact_fields(
            entry,
            {
                "META",
                "num_warps",
                "num_stages",
                "maxnreg",
                "ppu_compiler_options",
            },
            set(),
            f"THead {operation} tuning candidate {index}",
        )
        meta = _require_object(entry["META"], "THead convolution tuning META")
        _require_exact_fields(
            meta, meta_names, set(), "THead convolution tuning META"
        )
        parsed_meta = {
            name: _integer(meta[name], f"THead convolution {name}")
            for name in sorted(meta_names)
        }
        if any(
            value < 8 or value > 64 or value & (value - 1)
            for value in parsed_meta.values()
        ):
            raise ValueError(
                "THead convolution tiles must be bounded powers of two"
            )
        num_warps = _integer(entry["num_warps"], "THead convolution num_warps")
        num_stages = _integer(
            entry["num_stages"], "THead convolution num_stages"
        )
        options = _require_object(
            entry["ppu_compiler_options"],
            "THead convolution ppu_compiler_options",
        )
        if (
            num_warps < 1
            or num_warps > 32
            or num_warps & (num_warps - 1)
            or not 1 <= num_stages <= 16
            or entry["maxnreg"] is not None
            or options
        ):
            raise ValueError("THead convolution tuning options are invalid")
        configuration = {
            "META": parsed_meta,
            "maxnreg": None,
            "num_stages": num_stages,
            "num_warps": num_warps,
            "ppu_compiler_options": {},
        }
        encoded = json.dumps(
            configuration, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if encoded in seen:
            raise ValueError("THead convolution tuning candidates repeat")
        seen.add(encoded)
        configurations.append(configuration)
    return configurations, tuning_bytes


def _include_default_candidate(
    configurations: list[dict[str, Any]], default: dict[str, Any]
) -> list[dict[str, Any]]:
    return (
        configurations
        if default in configurations
        else [*configurations, default]
    )
