"""Metadata helpers for THead extended operator dispatch."""

from .tensor import _has_non_overlapping_strides  # noqa: F401

from .common import _integer

FLOAT_DATA_TYPES = {"float32", "float16", "bfloat16"}
FP8_DATA_TYPES = {"fp8_e4m3", "fp8_e5m2"}
TRITON_POINTER_TYPES = {
    "float32": "*fp32",
    "float16": "*fp16",
    "bfloat16": "*bf16",
    "int32": "*i32",
    "boolean": "*u8",
    "fp8_e4m3": "*fp8e4nv",
    "fp8_e5m2": "*fp8e5",
    "fp8_e8m0": "*u8",
}


def _require_integer(parameters, name, minimum=1, maximum=2**31 - 1):
    value = _integer(parameters.get(name), name)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside [{minimum}, {maximum}]")
    return value


def _is_row_major_contiguous(tensor):
    stride = 1
    for size, actual in reversed(
        list(zip(tensor["dimensions"], tensor["strides"]))
    ):
        if size != 1 and actual != stride:
            return False
        stride *= size
    return True
