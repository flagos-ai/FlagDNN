"""Ascend tuning-table decoding and launch capability validation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


MAX_TUNING_SOURCE_SIZE = 1 << 20
MAX_TUNING_CANDIDATES = 1024
MAX_INT = 2**31 - 1
MAX_U32 = 2**32 - 1


@dataclass(frozen=True)
class TuningConfiguration:
    meta: dict[str, int | float | str | bool]
    num_warps: int
    num_stages: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "META": dict(self.meta),
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }


def canonical_json_bytes(value: Any, description: str) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{description} must contain finite JSON-compatible values"
        ) from error


def canonical_sha256(value: Any, description: str) -> str:
    return hashlib.sha256(canonical_json_bytes(value, description)).hexdigest()


def _flatten_param_map(
    value: object, path: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], str]]:
    if not isinstance(value, dict) or not value:
        raise ValueError("tuning param_map must be a nonempty mapping")
    result: list[tuple[tuple[str, ...], str]] = []
    for output_name, source in value.items():
        if not isinstance(output_name, str) or not output_name:
            raise ValueError("tuning output names must be nonempty strings")
        output_path = (*path, output_name)
        if isinstance(source, dict):
            result.extend(_flatten_param_map(source, output_path))
        elif isinstance(source, str) and source:
            result.append((output_path, source))
        else:
            raise ValueError("tuning param_map leaves must name a parameter")
    return result


def _assign_path(
    configuration: dict[str, Any], path: tuple[str, ...], value: Any
) -> None:
    current = configuration
    for name in path[:-1]:
        existing = current.get(name)
        if existing is None:
            nested: dict[str, Any] = {}
            current[name] = nested
            current = nested
        elif isinstance(existing, dict):
            current = existing
        else:
            raise ValueError(f"tuning output collides at {'.'.join(path)}")
    if path[-1] in current:
        raise ValueError(f"duplicate tuning output {'.'.join(path)}")
    current[path[-1]] = copy.deepcopy(value)


def _parameter_values(entry: dict[str, Any], name: str) -> list[Any]:
    if name not in entry:
        raise ValueError(
            f"tuning param_map references missing parameter {name}"
        )
    source = entry[name]
    values = source if isinstance(source, list) else [source]
    if not values:
        raise ValueError(f"tuning parameter {name} is empty")
    result: list[Any] = []
    seen: set[bytes] = set()
    for value in values:
        canonical = canonical_json_bytes(value, f"tuning parameter {name}")
        if canonical not in seen:
            seen.add(canonical)
            result.append(value)
    return result


def _expand_generated_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    leaves = _flatten_param_map(entry.get("param_map"))
    output_paths: set[tuple[str, ...]] = set()
    source_names: list[str] = []
    for output_path, source_name in leaves:
        if output_path in output_paths:
            raise ValueError(
                f"duplicate tuning output {'.'.join(output_path)}"
            )
        output_paths.add(output_path)
        if source_name not in source_names:
            source_names.append(source_name)
    dimensions = [_parameter_values(entry, name) for name in source_names]
    parameter_names = set(source_names)
    literals = {
        name: value
        for name, value in entry.items()
        if name not in {"gen", "param_map"} and name not in parameter_names
    }
    result: list[dict[str, Any]] = []
    for combination in itertools.product(*dimensions):
        selected = dict(zip(source_names, combination, strict=True))
        configuration: dict[str, Any] = {}
        for output_path, source_name in leaves:
            _assign_path(configuration, output_path, selected[source_name])
        for name, value in literals.items():
            _assign_path(configuration, (name,), value)
        result.append(configuration)
    return result


def _expand_entries(entries: list[object]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict) or not raw_entry:
            raise ValueError(
                f"tuning entry {index} must be a nonempty mapping"
            )
        if raw_entry.get("gen") is True:
            expanded = _expand_generated_entry(raw_entry)
        elif "gen" in raw_entry:
            raise ValueError(f"tuning entry {index}.gen must be true")
        else:
            expanded = [copy.deepcopy(raw_entry)]
        for configuration in expanded:
            canonical = canonical_json_bytes(
                configuration, f"tuning entry {index}"
            )
            if canonical not in seen:
                seen.add(canonical)
                result.append(configuration)
    if not result:
        raise ValueError("tuning table produced no configurations")
    if len(result) > MAX_TUNING_CANDIDATES:
        raise ValueError("tuning table exceeds the candidate limit")
    return result


def _positive_option(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > MAX_INT
    ):
        raise ValueError(f"tuning {name} must be an integer in [1, INT_MAX]")
    return value


def normalize_configuration(value: dict[str, Any]) -> TuningConfiguration:
    if set(value) != {"META", "num_warps", "num_stages"}:
        unknown = sorted(
            set(value).difference({"META", "num_warps", "num_stages"})
        )
        missing = sorted({"META", "num_warps", "num_stages"}.difference(value))
        detail = []
        if unknown:
            detail.append("unknown=" + ",".join(unknown))
        if missing:
            detail.append("missing=" + ",".join(missing))
        raise ValueError(
            "tuning configuration must contain only META, num_warps and "
            "num_stages (" + "; ".join(detail) + ")"
        )
    raw_meta = value.get("META")
    if not isinstance(raw_meta, dict) or not raw_meta:
        raise ValueError("tuning META must be a nonempty mapping")
    meta: dict[str, int | float | str | bool] = {}
    for name, item in raw_meta.items():
        if not isinstance(name, str) or not name:
            raise ValueError("tuning META names must be nonempty strings")
        if isinstance(item, (dict, list)) or item is None:
            raise ValueError(f"tuning META.{name} must be a scalar")
        canonical_json_bytes(item, f"tuning META.{name}")
        meta[name] = item
    return TuningConfiguration(
        meta=meta,
        num_warps=_positive_option(value.get("num_warps"), "num_warps"),
        num_stages=_positive_option(value.get("num_stages"), "num_stages"),
    )


def load_tuning_table(
    path: Path, table: str
) -> tuple[tuple[TuningConfiguration, ...], str]:
    data = path.read_bytes()
    if not data or len(data) > MAX_TUNING_SOURCE_SIZE:
        raise ValueError("tuning source is empty or exceeds its size limit")
    document = yaml.safe_load(data)
    if not isinstance(document, dict):
        raise ValueError("tuning source must be a YAML mapping")
    entries = document.get(table)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"missing nonempty tuning table {table!r}")
    configurations = tuple(
        normalize_configuration(value) for value in _expand_entries(entries)
    )
    return configurations, hashlib.sha256(data).hexdigest()


def validate_add_configuration(
    configuration: TuningConfiguration,
    capabilities: dict[str, Any],
    *,
    kernel_family: str = "binary",
) -> None:
    if set(configuration.meta) != {"BLOCK_SIZE"}:
        raise ValueError("Ascend Add tuning META must contain only BLOCK_SIZE")
    block_size = configuration.meta["BLOCK_SIZE"]
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise ValueError("Ascend Add BLOCK_SIZE must be an integer")
    compile_options = capabilities.get("compile_options")
    if not isinstance(compile_options, dict):
        raise ValueError("Ascend capabilities.compile_options is invalid")
    allowed_warps = compile_options.get("num_warps")
    allowed_stages = compile_options.get("num_stages")
    allowed_blocks = capabilities.get("block_sizes")
    if kernel_family == "matmul":
        matmul = capabilities.get("matmul")
        if not isinstance(matmul, dict):
            raise ValueError("Ascend matmul capability is invalid")
        allowed_blocks = matmul.get("block_sizes")
    elif kernel_family == "convolution_fprop":
        convolution = capabilities.get("convolution_fprop")
        if not isinstance(convolution, dict):
            raise ValueError("Ascend convolution_fprop capability is invalid")
        allowed_blocks = convolution.get("block_sizes")
    for value, allowed, name in (
        (configuration.num_warps, allowed_warps, "num_warps"),
        (configuration.num_stages, allowed_stages, "num_stages"),
        (block_size, allowed_blocks, "BLOCK_SIZE"),
    ):
        if (
            not isinstance(allowed, list)
            or not allowed
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item <= 0
                or item > MAX_INT
                for item in allowed
            )
            or len(set(allowed)) != len(allowed)
        ):
            raise ValueError(f"Ascend capability whitelist {name} is invalid")
        if value not in allowed:
            raise ValueError(
                f"Ascend tuning {name} is outside target capability"
            )


def checked_grid(
    n_elements: int,
    block_size: int,
    capabilities: dict[str, Any],
) -> tuple[int, int, int]:
    if (
        isinstance(n_elements, bool)
        or not isinstance(n_elements, int)
        or n_elements <= 0
        or isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size <= 0
    ):
        raise ValueError("Ascend grid inputs must be positive integers")
    grid = ((n_elements + block_size - 1) // block_size, 1, 1)
    limits = capabilities.get("grid")
    if not isinstance(limits, dict):
        raise ValueError("Ascend capabilities.grid is invalid")
    maximum_dimension = limits.get("max_dimension")
    maximum_product = limits.get("max_product")
    for value, name in (
        (maximum_dimension, "max_dimension"),
        (maximum_product, "max_product"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            or value > MAX_U32
        ):
            raise ValueError(f"Ascend grid capability {name} is invalid")
    product = 1
    for dimension in grid:
        if dimension <= 0 or dimension > maximum_dimension:
            raise ValueError("Ascend launch grid dimension is out of range")
        if product > maximum_product // dimension:
            raise ValueError("Ascend launch grid product exceeds uint32 limit")
        product *= dimension
    return grid
