# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate standalone index generation and concatenation plans."""

import math

from .extended_common import (
    TRITON_POINTER_TYPES,
    _has_non_overlapping_strides,
    _require_integer,
)


def _validate_tensor(tensor):
    if (
        not 1 <= len(tensor["dimensions"]) <= 8
        or tensor["data_type"] not in TRITON_POINTER_TYPES
        or not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        )
    ):
        raise ValueError(
            "index tensors require supported storage, rank 1..8 and "
            "non-overlapping strides"
        )


def _index_kernel_configuration(operation, parameters, tensors):
    if not tensors:
        raise ValueError("index operation requires an output")
    for tensor in tensors:
        _validate_tensor(tensor)
    output = tensors[-1]
    expected_fields = {"axis", "n_elements"}
    if operation == "concatenate":
        expected_fields.add("input_count")
    if set(parameters) != expected_fields:
        raise ValueError("index operation attributes are inconsistent")
    axis = _require_integer(
        parameters, "axis", minimum=0, maximum=len(output["dimensions"]) - 1
    )
    elements = _require_integer(parameters, "n_elements", maximum=2**63 - 1)
    if elements != math.prod(output["dimensions"]):
        raise ValueError("index output element count is inconsistent")
    block = 256
    grid = ((elements + block - 1) // block, 1, 1)
    if grid[0] > 2**31 - 1:
        raise ValueError("index launch grid exceeds THead limits")
    if operation == "gen_index":
        if len(tensors) != 1 or output["data_type"] not in {
            "float32",
            "int32",
        }:
            raise ValueError(
                "gen_index requires a single FP32 or INT32 output"
            )
        dimensions = [1] * (8 - len(output["dimensions"])) + output[
            "dimensions"
        ]
        strides = [0] * (8 - len(output["strides"])) + output["strides"]
        constants = {
            "N_ELEMENTS": elements,
            "AXIS_EXTENT": output["dimensions"][axis],
            "INNER": math.prod(output["dimensions"][axis + 1 :]),
            "BLOCK_SIZE": block,
        }
        for dimension in range(8):
            constants[f"DIM_{dimension}"] = dimensions[dimension]
            constants[f"OUTPUT_STRIDE_{dimension}"] = strides[dimension]
        return (
            "gen_index_kernel",
            {"output_ptr": TRITON_POINTER_TYPES[output["data_type"]]},
            constants,
            grid,
            [("tensor", None)],
        )
    if operation != "concatenate":
        raise ValueError("unsupported index operation")
    inputs = tensors[:-1]
    if _require_integer(parameters, "input_count", maximum=4095) != len(
        inputs
    ):
        raise ValueError("concatenate input count is inconsistent")
    extent = 0
    for source in inputs:
        if (
            source["data_type"] != output["data_type"]
            or len(source["dimensions"]) != len(output["dimensions"])
            or any(
                a != b
                for i, (a, b) in enumerate(
                    zip(
                        source["dimensions"], output["dimensions"], strict=True
                    )
                )
                if i != axis
            )
        ):
            raise ValueError("concatenate input dtype or dimensions mismatch")
        extent += source["dimensions"][axis]
    if extent != output["dimensions"][axis]:
        raise ValueError("concatenate output axis differs from summed inputs")
    # A repeated input shares one runtime pointer. Keep the artifact ABI's
    # unique boundary UID contract while preserving every logical region.
    unique_tensors = list(
        {tensor["uid"]: tensor for tensor in tensors}.values()
    )
    names = [f"input_{index}" for index in range(len(unique_tensors) - 1)] + [
        "output_ptr"
    ]
    return (
        "concatenate_kernel",
        dict(
            zip(
                names,
                (TRITON_POINTER_TYPES[t["data_type"]] for t in unique_tensors),
            )
        ),
        {"BLOCK_SIZE": block},
        grid,
        [("tensor", None)] * len(unique_tensors),
    )
