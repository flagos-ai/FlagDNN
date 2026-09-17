# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Hygon codegen pointwise."""

from __future__ import annotations

from ..dispatch.pointwise import (
    PointwiseSchema,
)
from typing import Any


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

    return (marker + source).encode("utf-8")
