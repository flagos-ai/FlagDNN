# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared metadata helpers for Hygon extended-operator plans."""
from .common import FLOAT_TYPES, POINTER_TYPES, _require_integer
from .tensor_metadata import _has_non_overlapping_strides

__all__ = [
    "FLOAT_DATA_TYPES",
    "FP8_DATA_TYPES",
    "TRITON_POINTER_TYPES",
    "_require_integer",
    "_has_non_overlapping_strides",
    "_is_row_major_contiguous",
]

FLOAT_DATA_TYPES = FLOAT_TYPES
FP8_DATA_TYPES = {"fp8_e4m3", "fp8_e5m2"}
TRITON_POINTER_TYPES = POINTER_TYPES


def _is_row_major_contiguous(tensor):
    stride = 1
    for size, actual in reversed(
        list(zip(tensor["dimensions"], tensor["strides"]))
    ):
        if size != 1 and actual != stride:
            return False
        stride *= size
    return True
