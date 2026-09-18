# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Execution-program tensor bindings and Triton pointer signatures."""

from __future__ import annotations

from typing import Any
from ..dispatch.common import (
    _DATA_TYPE_BYTES,
)


def _element_count_signature(n_elements: int) -> str:
    # The count is artifact-owned and immutable at execution. Expose proven
    # divisibility to PPU lowering so masked stores can remain vectorized.
    return "i32:16" if n_elements >= 65536 and n_elements % 16 == 0 else "i32"


def _tensor_pointer_signature(
    tensor: dict[str, Any], *, copy_bits: bool = False
) -> str:
    scalar = {
        "float32": "fp32",
        "float16": "fp16",
        "bfloat16": "bf16",
        "int32": "i32",
        "fp8_e8m0": "i8",
        "boolean": "i8",
        "fp8_e4m3": "fp8e4nv",
        "fp8_e5m2": "fp8e5",
    }[str(tensor["data_type"])]
    if copy_bits and str(tensor["data_type"]).startswith("fp8_"):
        scalar = "i8"
    # In libtriton_jit's CUDA-compatible signature syntax ``:1`` means
    # value-equals-one specialization rather than one-byte alignment.  Leave
    # sub-16-byte pointers unspecialized and cap stronger guarantees at the
    # only supported divisibility hint.
    if int(tensor["alignment"]) < 16:
        return f"*{scalar}"
    return f"*{scalar}:16"


def _tensor_storage_size(tensor: dict[str, Any]) -> int:
    element_size = _DATA_TYPE_BYTES[str(tensor["data_type"])]
    elements = 1 + sum(
        (int(dimension) - 1) * int(stride)
        for dimension, stride in zip(
            tensor["dimensions"], tensor["strides"], strict=True
        )
    )
    return elements * element_size


def _manifest_tensor(tensor: dict[str, Any]) -> dict[str, Any]:
    return {
        "uid": tensor["uid"],
        "data_type": tensor["data_type"],
        "dimensions": tensor["dimensions"],
        "strides": tensor["strides"],
        "alignment": tensor["alignment"],
        "virtual": tensor["virtual"],
        "storage_size": _tensor_storage_size(tensor),
    }


def _tensor_argument(
    tensor: dict[str, Any], *, copy_bits: bool = False
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": "tensor",
        "uid": tensor["uid"],
        "size": _tensor_storage_size(tensor),
        "alignment": tensor["alignment"],
    }

    if copy_bits and str(tensor["data_type"]).startswith("fp8_"):
        result["storage_view"] = "fp8_bytes"
    return result
