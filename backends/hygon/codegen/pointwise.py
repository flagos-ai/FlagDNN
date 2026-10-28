# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Hygon codegen pointwise."""

from __future__ import annotations

from ..dispatch.pointwise import (
    PointwiseSchema,
    _can_use_dense_pointwise_kernel,
)
from typing import Any
from math import prod


def _specialize_pointwise_compute_source(
    generated_bytes: bytes, node: dict[str, Any]
) -> bytes:
    """Make the Graph compute type an explicit part of generated code.

    Triton's ordinary FP16/BF16 add, subtract and multiply expressions retain
    their input precision. FlagDNN's public pointwise graphs request FP32
    compute for numeric operations, so relying on implicit Triton promotion
    silently violates the Graph contract. Keep the common source
    platform-neutral and specialize only the per-artifact materialized copy.

    Boolean pointwise operations already execute in their declared boolean
    semantics. They still receive a deterministic source marker so the
    materialized source and candidate identities encode compute_data_type.
    """

    compute_data_type = node["compute_data_type"]
    schema: PointwiseSchema = node["schema"]
    try:
        source = generated_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("pointwise kernel source must be UTF-8") from error

    marker = (
        "# FlagDNN Hygon compute specialization: "
        f"{schema.data_type_policy}/{compute_data_type}\n"
    )
    if source.startswith(marker):
        raise ValueError("pointwise kernel source is already specialized")

    if compute_data_type == "float32" and node["operation"] == "identity":
        # Identity performs no arithmetic.  Every supported input value is
        # already representable in FP32, and a direct copy additionally keeps
        # signed zero and NaN payload bits intact instead of round-tripping
        # through a conversion solely to annotate the compute type.
        pass
    elif node["tensors"][0]["data_type"] == "int32":
        pass
    elif compute_data_type == "float32":
        if schema.family == "binary":
            trigger_names = {"right"}
            cast_names: tuple[str, ...] = ("left", "right")
        elif schema.family == "unary":
            trigger_names = {"value"}
            cast_names = ("value",)
        elif schema.family == "ternary":
            trigger_names = {"right", "input1"}
            cast_names = ()
        else:
            raise ValueError(f"unknown pointwise family {schema.family!r}")

        lines: list[str] = []
        specialization_count = 0
        for line in source.splitlines(keepends=True):
            lines.append(line)
            stripped = line.lstrip()
            indentation = line[: len(line) - len(stripped)]
            loaded_name = next(
                (
                    name
                    for name in trigger_names
                    if stripped.startswith(f"{name} = tl.load(")
                ),
                None,
            )
            if loaded_name is None:
                continue
            names = cast_names
            if schema.family == "ternary":
                names = (
                    ("left", "right")
                    if loaded_name == "right"
                    else ("input0", "input1")
                )
            for name in names:
                lines.append(f"{indentation}{name} = {name}.to(tl.float32)\n")
            specialization_count += 1
        if specialization_count < 1:
            raise ValueError(
                "common pointwise kernel no longer exposes the expected "
                "load sites for FP32 specialization"
            )
        source = "".join(lines)
    elif compute_data_type != "boolean":
        # _validate_pointwise_data_types normally rejects this first. Keep a
        # defensive check here because this function is the precision
        # boundary for all future schema additions.
        raise ValueError(
            "Hygon pointwise materialization implements only float32 or "
            "boolean compute semantics"
        )

    source = _specialize_streaming_memory_source(source, node)
    return (marker + source).encode("utf-8")


def _specialize_streaming_memory_source(source: str, node: dict[str, Any]) -> str:
    """Keep cache hints effective for large, dense pointwise working sets.

    The gfx936 buffer-op lowering drops load/store cache modifiers. Express
    these addresses as integer byte addresses so they retain global loads
    and stores; restore the proven alignment and contiguous lane information
    explicitly so this does not sacrifice vectorized memory instructions.
    The 8 MiB cutoff keeps cache-resident tensors on the ordinary path.
    """
    operations = {
        "add",
        "sub",
        "mul",
        "min",
        "max",
        "binary_select",
        "cmp_eq",
        "cmp_neq",
        "cmp_gt",
        "cmp_ge",
        "cmp_lt",
        "cmp_le",
    }
    tensors = node["tensors"]
    widths = {"float32": 4, "float16": 2, "bfloat16": 2, "boolean": 1}
    if (
        node["operation"] not in operations
        or not _can_use_dense_pointwise_kernel(tensors)
        or any(tensor["data_type"] not in widths for tensor in tensors)
        or any(tensor["alignment"] < 16 for tensor in tensors)
        or sum(
            prod(tensor["dimensions"]) * widths[tensor["data_type"]]
            for tensor in tensors
        )
        <= 8 * 1024 * 1024
    ):
        return source

    select = node["operation"] == "binary_select"
    function = "binary_select_tensor_kernel" if select else "binary_contiguous_kernel"
    first = source.index("@triton.jit\ndef " + function + "(")
    following = source.find("\n@triton.jit", first + 1)
    last = len(source) if following < 0 else following
    body = source[first:last]
    pointers = (
        ("input0_ptr", "input1_ptr", "mask_ptr", "out_ptr")
        if select
        else ("x_ptr", "y_ptr", "out_ptr")
    )
    for pointer in pointers:
        original = pointer + " + offsets"
        if body.count(original) != 1:
            raise ValueError("dense pointwise memory layout changed unexpectedly")
        body = body.replace(
            original, f"_hygon_streaming_pointer({pointer}, offsets, BLOCK_SIZE)"
        )

    lines = []
    loads = 0
    stores = 0
    for line in body.splitlines(keepends=True):
        if " = tl.load(" in line:
            if not line.rstrip().endswith(")"):
                raise ValueError("dense pointwise load is no longer a single line")
            line = line.rstrip()[:-1] + ', cache_modifier=".cg")\n'
            loads += 1
        elif "mask=mask," in line or "mask=active\n" in line:
            line = line.rstrip().rstrip(",") + ', cache_modifier=".cs",\n'
            stores += 1
        lines.append(line)
    if loads != len(pointers) - 1 or stores != 1:
        raise ValueError(
            "dense pointwise cache specialization missed a memory operation"
        )
    source = source[:first] + "".join(lines) + source[last:]
    return (
        source
        + """

@triton.jit
def _hygon_streaming_pointer(pointer, offsets, BLOCK_SIZE: tl.constexpr):
    # The host-side specialization proves 16-byte tensor alignment and
    # a dense layout. Every program starts at a BLOCK_SIZE-aligned element.
    address = pointer.to(tl.uint64) + offsets.to(tl.uint64) * (
        pointer.dtype.element_ty.primitive_bitwidth // 8
    )
    return tl.max_contiguous(
        tl.multiple_of(address.to(pointer.dtype), 16), BLOCK_SIZE
    )
"""
    )
