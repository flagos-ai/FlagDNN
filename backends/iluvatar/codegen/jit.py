# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Iluvatar codegen / jit implementation."""

from __future__ import annotations

from ..dispatch.common import ILUVATAR_WARP_SIZE
from ..dispatch.common import LIBTRITON_JIT_GLOBAL_SCRATCH_SIZE
from ..dispatch.common import MAX_I32
from typing import Any
import math
import re
import triton


def _triton_supports_pointer_range(version: str) -> bool:
    match = re.match(r"^(\d+)\.(\d+)(?:\.|$)", version)
    if match is None:
        raise RuntimeError(f"cannot parse Triton version {version!r}")
    return (int(match.group(1)), int(match.group(2))) >= (3, 3)


TRITON_SUPPORTS_POINTER_RANGE = _triton_supports_pointer_range(triton.__version__)


def _jit_runtime_signature(
    function: Any,
    runtime_signature: dict[str, str],
    argument_abi: list[dict[str, Any]],
) -> dict[str, str]:
    if (
        len(argument_abi) < 2
        or argument_abi[-2].get("kind") != "global_scratch_pointer"
        or argument_abi[-1].get("kind") != "profile_scratch_pointer"
    ):
        raise ValueError("libtriton_jit scratch ABI is invalid")
    runtime_names = [name for name in function.arg_names if name in runtime_signature]
    visible_arguments = argument_abi[:-2]
    if len(runtime_names) != len(visible_arguments):
        raise ValueError("kernel runtime signature and argument ABI disagree")
    result = dict(runtime_signature)
    for name, argument in zip(runtime_names, visible_arguments, strict=True):
        kind = argument.get("kind")
        if kind not in {"tensor", "workspace_tensor"}:
            continue
        token = result[name]
        if not token.startswith("*") or ":" in token:
            raise ValueError("JIT tensor argument is not a plain pointer")
        alignment = int(argument.get("alignment", 1))
        storage_size = int(argument.get("size", 0))
        specialization = "16" if alignment >= 16 else ""
        # Match IX CoreX Triton Tensor specialization exactly: buffer operations may
        # use 32-bit offsets only when the reachable storage range is proven to
        # fit in signed int32. The Iluvatar-private standalone compiler restores
        # this S marker as the tt.pointer_range=32 TTIR argument attribute.
        if TRITON_SUPPORTS_POINTER_RANGE and 0 < storage_size <= MAX_I32:
            specialization += "S"
        if specialization:
            result[name] = f"{token}:{specialization}"
    return result


def _jit_full_signature(
    function: Any,
    runtime_signature: dict[str, str],
    constants: dict[str, int | float],
) -> str:
    argument_names = list(function.arg_names)
    if set(runtime_signature).union(constants) != set(argument_names):
        raise ValueError("JIT signature does not cover every kernel argument")
    tokens: list[str] = []
    for name in argument_names:
        if name in runtime_signature:
            token = runtime_signature[name]
            if not isinstance(token, str) or not token or "," in token:
                raise ValueError("JIT runtime signature token is invalid")
            tokens.append(token)
            continue
        value = constants[name]
        if isinstance(value, bool):
            tokens.append("true" if value else "false")
        elif isinstance(value, int):
            tokens.append(str(value))
        elif isinstance(value, float) and math.isfinite(value):
            tokens.append(repr(value))
        else:
            raise ValueError("JIT constexpr must be a finite number")
    return ",".join(tokens)


def _jit_launch(grid: tuple[int, int, int], num_warps: int) -> dict[str, Any]:
    if (
        isinstance(num_warps, bool)
        or not isinstance(num_warps, int)
        or num_warps <= 0
        or (num_warps & (num_warps - 1)) != 0
        or num_warps * ILUVATAR_WARP_SIZE > 1024
    ):
        raise ValueError(
            "Iluvatar num_warps must be a positive power of two within the "
            "workgroup limit"
        )
    if any(value <= 0 or value > 2**32 - 1 for value in grid):
        raise ValueError("Iluvatar launch grid is invalid")
    return {
        "grid": list(grid),
        "block": [num_warps * ILUVATAR_WARP_SIZE, 1, 1],
        "cluster": [1, 1, 1],
        "shared_memory": 0,
        "num_ctas": 1,
        "global_scratch_size": LIBTRITON_JIT_GLOBAL_SCRATCH_SIZE,
        "profile_scratch_size": 0,
    }
