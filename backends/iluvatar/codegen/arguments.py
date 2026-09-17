# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Iluvatar codegen / arguments implementation."""

from __future__ import annotations

from ..dispatch import nn as nn_dispatch
from ..dispatch.common import FLOAT32_MAX
from ..dispatch.common import _require_number
from ..dispatch.graph_tensor import _tensor_storage_size
from typing import Any
import math


def _tensor_argument(
    tensor: dict[str, Any], workspace: dict[int, tuple[int, int, int]]
) -> dict[str, Any]:
    uid = tensor["uid"]
    if tensor["virtual"]:
        offset, size, _layout_alignment = workspace[uid]
        return {
            "kind": "workspace_tensor",
            "uid": uid,
            "offset": offset,
            "size": size,
            "alignment": tensor["alignment"],
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
                raise ValueError(f"parameters.{name} must be representable as int32")
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
    configuration: nn_dispatch.KernelStagePlan,
    tensors: list[dict[str, Any]],
    tensor_roles: list[str],
    workspace: dict[int, tuple[int, int, int]],
    local_workspace: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for kind, payload in configuration.argument_layout:
        if kind == "tensor" and isinstance(payload, int):
            if payload < 0 or payload >= len(tensors):
                raise ValueError("Iluvatar NN kernel tensor index is invalid")
            argument = _tensor_argument(tensors[payload], workspace)
            argument["role"] = tensor_roles[payload]
            result.append(argument)
            continue
        if kind == "workspace_tensor" and isinstance(payload, str):
            try:
                argument = local_workspace[payload]
            except KeyError as error:
                raise ValueError(
                    f"Iluvatar NN workspace tensor {payload!r} is missing"
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
                raise ValueError(f"Iluvatar NN runtime scalar {payload!r} is not int32")
            result.append({"kind": "scalar_i32", "name": payload, "value": value})
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
                    f"Iluvatar NN runtime scalar {payload!r} is not float32"
                )
            result.append(
                {
                    "kind": "scalar_f32",
                    "name": payload,
                    "value": float(value),
                }
            )
            continue
        raise ValueError("Iluvatar NN kernel ABI layout is invalid")
    result.extend(
        (
            {"kind": "global_scratch_pointer"},
            {"kind": "profile_scratch_pointer"},
        )
    )
    return result


def _artifact_arguments(
    argument_abi: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Serialize only arguments bound explicitly by the FlagDNN engine."""

    result: list[dict[str, Any]] = []
    for argument in argument_abi:
        kind = argument.get("kind")
        if kind in {"global_scratch_pointer", "profile_scratch_pointer"}:
            continue
        serialized = {
            key: value for key, value in argument.items() if key not in {"role", "name"}
        }
        if kind == "workspace_tensor":
            serialized["workspace_offset"] = serialized.pop("offset")
        result.append(serialized)
    return result
