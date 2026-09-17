# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Hygon codegen abi."""

from __future__ import annotations

from ..dispatch import nn as compiler_nn
from ..dispatch.common import (
    FLOAT32_MAX,
    HYGON_WARP_SIZE,
    LIBTRITON_JIT_GLOBAL_SCRATCH_SIZE,
    MAX_I32,
    _require_number,
)
from ..dispatch.tensor_metadata import (
    _tensor_storage_size,
)
from typing import Any
import math
import re
import triton


def _triton_supports_pointer_range(version: str) -> bool:
    match = re.match(r"^(\d+)\.(\d+)(?:\.|$)", version)
    if match is None:
        raise RuntimeError(f"cannot parse Triton version {version!r}")
    return (int(match.group(1)), int(match.group(2))) >= (3, 3)


TRITON_SUPPORTS_POINTER_RANGE = _triton_supports_pointer_range(
    triton.__version__
)


def _tensor_argument(
    tensor: dict[str, Any], workspace: dict[int, tuple[int, int, int]]
) -> dict[str, Any]:
    uid = tensor["uid"]
    if tensor["virtual"]:
        offset, size, alignment = workspace[uid]
        return {
            "kind": "workspace_tensor",
            "uid": uid,
            "offset": offset,
            "size": size,
            "alignment": alignment,
        }
    return {
        "kind": "tensor",
        "uid": uid,
        "size": _tensor_storage_size(tensor),
        "alignment": tensor["alignment"],
    }


def _argument_abi(
    layout: tuple[tuple[str, str | int | None], ...],
    tensors: list[dict[str, Any]],
    tensor_roles: list[str],
    parameters: dict[str, Any],
    workspace: dict[int, tuple[int, int, int]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    tensor_index = 0
    consumed: set[int] = set()
    for kind, name in layout:
        if kind == "tensor":
            if tensor_index >= len(tensors):
                raise ValueError("kernel ABI requests too many tensors")
            argument = _tensor_argument(tensors[tensor_index], workspace)
            argument["role"] = tensor_roles[tensor_index]
            result.append(argument)
            consumed.add(tensor_index)
            tensor_index += 1
            continue
        if kind == "tensor_alias" and isinstance(name, int):
            if name < -len(tensors) or name >= len(tensors):
                raise ValueError("kernel ABI tensor alias is out of range")
            alias_index = name if name >= 0 else len(tensors) + name
            argument = _tensor_argument(tensors[alias_index], workspace)
            argument["role"] = tensor_roles[alias_index]
            result.append(argument)
            consumed.add(alias_index)
            continue
        if kind == "scalar_i32" and isinstance(name, str):
            value = parameters.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < -(2**31)
                or value > 2**31 - 1
            ):
                raise ValueError(
                    f"parameters.{name} must be representable as int32"
                )
            result.append({"kind": kind, "name": name, "value": value})
            continue
        if kind == "scalar_f32" and isinstance(name, str):
            result.append(
                {
                    "kind": kind,
                    "name": name,
                    "value": _require_number(parameters, name),
                }
            )
            continue
        raise ValueError("kernel ABI layout is invalid")
    if consumed != set(range(len(tensors))):
        raise ValueError("kernel ABI does not consume every tensor")
    result.extend(
        (
            {"kind": "global_scratch_pointer"},
            {"kind": "profile_scratch_pointer"},
        )
    )
    return result


def _nn_argument_abi(
    configuration: compiler_nn.KernelStagePlan,
    tensors: list[dict[str, Any]],
    tensor_roles: list[str],
    workspace: dict[int, tuple[int, int, int]],
    local_workspace: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for kind, payload in configuration.argument_layout:
        if kind == "tensor" and isinstance(payload, int):
            if payload < 0 or payload >= len(tensors):
                raise ValueError("Hygon NN kernel tensor index is invalid")
            argument = _tensor_argument(tensors[payload], workspace)
            argument["role"] = tensor_roles[payload]
            result.append(argument)
            continue
        if kind == "workspace_tensor" and isinstance(payload, str):
            try:
                argument = local_workspace[payload]
            except KeyError as error:
                raise ValueError(
                    f"Hygon NN workspace tensor {payload!r} is missing"
                ) from error
            argument = dict(argument)
            argument["role"] = payload
            result.append(argument)
            continue
        if kind == "scalar_i32" and isinstance(payload, str):
            value = configuration.runtime_values.get(payload)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < -(2**31)
                or value > 2**31 - 1
            ):
                raise ValueError(
                    f"Hygon NN runtime scalar {payload!r} is not int32"
                )
            result.append(
                {"kind": "scalar_i32", "name": payload, "value": value}
            )
            continue
        if kind == "scalar_f32" and isinstance(payload, str):
            value = configuration.runtime_values.get(payload)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or abs(float(value)) > FLOAT32_MAX
            ):
                raise ValueError(
                    f"Hygon NN runtime scalar {payload!r} is not float32"
                )
            result.append(
                {
                    "kind": "scalar_f32",
                    "name": payload,
                    "value": float(value),
                }
            )
            continue
        raise ValueError("Hygon NN kernel ABI layout is invalid")
    result.extend(
        (
            {"kind": "global_scratch_pointer"},
            {"kind": "profile_scratch_pointer"},
        )
    )
    return result


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
    runtime_names = [
        name for name in function.arg_names if name in runtime_signature
    ]
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
        # Match HCU Triton Tensor specialization exactly: buffer operations may
        # use 32-bit offsets only when the reachable storage range is proven to
        # fit in signed int32. The Hygon-private standalone compiler restores
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
    if num_warps <= 0 or num_warps * HYGON_WARP_SIZE > 1024:
        raise ValueError("Hygon num_warps exceeds the workgroup limit")
    if any(value <= 0 or value > 2**32 - 1 for value in grid):
        raise ValueError("Hygon launch grid is invalid")
    return {
        "grid": list(grid),
        "block": [num_warps * HYGON_WARP_SIZE, 1, 1],
        "cluster": [1, 1, 1],
        "shared_memory": 0,
        "num_ctas": 1,
        "global_scratch_size": LIBTRITON_JIT_GLOBAL_SCRATCH_SIZE,
        "profile_scratch_size": 0,
    }
