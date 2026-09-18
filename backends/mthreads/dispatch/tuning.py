"""MThreads dispatch tuning implementation."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import yaml

_TUNING_ROOT_KEYS = {"schema_version", "backend", "defaults", "tables"}

_TUNING_DEFAULT_KEYS = {"block_size", "num_warps", "num_stages"}

_TUNING_TABLE_KEYS = {
    "strategy",
    "warmup",
    "repetitions",
    "dimensions",
}

_TUNING_DIMENSION_KEYS = {"block_size", "num_warps", "num_stages"}


def _require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    context: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{context} keys are invalid")


def _positive_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _integer_dimension(
    value: object,
    context: str,
) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must be a nonempty array")
    result = tuple(
        _positive_integer(item, f"{context}[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{context} contains duplicate candidates")
    return result


def _load_tuning(
    *,
    table_name: str,
    warp_size: int,
) -> tuple[
    tuple[int, int, int],
    tuple[tuple[int, int, int], ...],
    int,
    int,
]:
    source = Path(__file__).resolve().parents[1] / "tuning/mthreads.yaml"
    if not source.is_file() or source.stat().st_size > (1 << 20):
        raise ValueError("mthreads tuning policy is missing or oversized")
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("mthreads tuning policy must be an object")
    _require_exact_keys(value, _TUNING_ROOT_KEYS, "tuning")
    if value["schema_version"] != 1 or value["backend"] != "mthreads":
        raise ValueError("mthreads tuning policy metadata is invalid")
    defaults = value["defaults"]
    tables = value["tables"]
    if not isinstance(defaults, dict) or not isinstance(tables, dict):
        raise ValueError("mthreads tuning policy sections are invalid")
    _require_exact_keys(defaults, _TUNING_DEFAULT_KEYS, "tuning.defaults")
    if set(tables) != {
        "binary_contiguous",
        "binary_strided",
        "unary_contiguous",
        "unary_strided",
        "ternary_contiguous",
        "ternary_strided",
        "layout",
        "reduction",
        "matmul",
        "convolution",
        "normalization",
        "batchnorm",
        "batchnorm_inference",
        "attention",
    }:
        raise ValueError("mthreads tuning tables are invalid")
    default_block_size = _positive_integer(
        defaults["block_size"], "default block_size"
    )
    default = (
        (
            1
            if table_name == "reduction"
            else (
                32
                if table_name == "attention"
                else (
                    64
                    if table_name == "matmul"
                    else (
                        32
                        if table_name == "convolution"
                        else default_block_size
                    )
                )
            )
        ),
        _positive_integer(defaults["num_warps"], "default num_warps"),
        (
            2
            if table_name in {"matmul", "convolution", "attention"}
            else _positive_integer(
                defaults["num_stages"], "default num_stages"
            )
        ),
    )

    table = tables[table_name]
    if not isinstance(table, dict):
        raise ValueError(f"tuning table {table_name} must be an object")
    _require_exact_keys(table, _TUNING_TABLE_KEYS, f"tuning.{table_name}")
    if table["strategy"] != "cartesian":
        raise ValueError(
            "mthreads pointwise tuning strategy must be cartesian"
        )
    warmup = _positive_integer(table["warmup"], "tuning warmup")
    repetitions = _positive_integer(table["repetitions"], "tuning repetitions")
    dimensions = table["dimensions"]
    if not isinstance(dimensions, dict):
        raise ValueError("tuning dimensions must be an object")
    _require_exact_keys(
        dimensions,
        _TUNING_DIMENSION_KEYS,
        f"tuning.{table_name}.dimensions",
    )
    blocks = _integer_dimension(dimensions["block_size"], "tuning block_size")
    warps = _integer_dimension(dimensions["num_warps"], "tuning num_warps")
    stages = _integer_dimension(dimensions["num_stages"], "tuning num_stages")
    candidates = tuple(itertools.product(blocks, warps, stages))
    if len(set(candidates)) != len(candidates) or len(candidates) > 32:
        raise ValueError("mthreads pointwise tuning candidates are invalid")
    for block_size, num_warps, num_stages in (*candidates, default):
        if block_size > 1024 or num_warps * warp_size > 1024 or num_stages > 8:
            raise ValueError(
                "mthreads pointwise tuning candidate exceeds device limits"
            )
    if default not in candidates:
        raise ValueError(
            "mthreads pointwise default is not a tuning candidate"
        )
    return default, candidates, warmup, repetitions
