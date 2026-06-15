"""Shared helpers for graph op eager-fallback run functions.

A run function is the correctness fallback executed at replay when no prepared
fast path applies; it calls the eager op. These helpers are used across op
families and live here so each family module stays self-contained.
"""

from __future__ import annotations

from typing import Any

import torch

from flag_dnn.graph.device import is_runtime_device_tensor


def _runtime_backend_available(inputs: list[Any]) -> bool:
    tensor_inputs = [
        value for value in inputs if isinstance(value, torch.Tensor)
    ]
    return bool(tensor_inputs) and all(
        is_runtime_device_tensor(value) for value in tensor_inputs
    )


def _require_runtime_backend(inputs: list[Any], op_type: str) -> None:
    if not _runtime_backend_available(inputs):
        raise NotImplementedError(
            f"FlagDNN graph {op_type} requires runtime device tensors; "
            "torch fallback is disabled"
        )


def _unsupported_triton_path(op_type: str, detail: str) -> None:
    raise NotImplementedError(
        f"FlagDNN graph {op_type} has no Triton path for {detail}; "
        "torch fallback is disabled"
    )


def _format_bias(
    input_tensor: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    if bias.dim() == 1 and input_tensor.dim() >= 2:
        shape = [1] * input_tensor.dim()
        shape[1] = bias.numel()
        return bias.reshape(shape)
    return bias


def _public_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in attrs.items() if not key.startswith("_")
    }


__all__ = (
    "_runtime_backend_available",
    "_require_runtime_backend",
    "_unsupported_triton_path",
    "_format_bias",
    "_public_attrs",
)
