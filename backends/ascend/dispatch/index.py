"""Ascend dispatch for index."""

from __future__ import annotations
import math
from .common import (
    _has_non_overlapping_strides,
    _is_row_major_contiguous,
    _require_integer,
)

_STORAGE_POINTERS = {
    "float32": "*i32",
    "int32": "*i32",
    "float16": "*u16",
    "bfloat16": "*u16",
    "boolean": "*u8",
    "fp8_e4m3": "*u8",
    "fp8_e5m2": "*u8",
    "fp8_e8m0": "*u8",
}


def _validate_tensor(tensor):
    if not 1 <= len(
        tensor["dimensions"]
    ) <= 8 or not _has_non_overlapping_strides(
        tensor["dimensions"], tensor["strides"]
    ):
        raise ValueError(
            "index/layout tensors need rank 1..8 and non-overlapping strides"
        )
    if tensor["data_type"] not in _STORAGE_POINTERS:
        raise ValueError("unsupported index/layout storage data type")


def _shape_constants(dimensions, strides, prefix):
    return {
        f"{prefix}_{axis}": value
        for axis, value in enumerate([0] * (8 - len(strides)) + strides)
    } | {
        f"DIM_{axis}": value
        for axis, value in enumerate([1] * (8 - len(dimensions)) + dimensions)
    }


def _expand_concatenate_group(group):
    inputs, output = group["tensors"][:-1], group["tensors"][-1]
    _validate_tensor(output)
    parameters = group["parameters"]
    axis = _require_integer(
        parameters, "axis", minimum=0, maximum=len(output["dimensions"]) - 1
    )
    if not inputs or _require_integer(parameters, "input_count") != len(
        inputs
    ):
        raise ValueError("concatenate input count is inconsistent")
    if _require_integer(
        parameters, "n_elements", maximum=2**63 - 1
    ) != math.prod(output["dimensions"]):
        raise ValueError("concatenate element count is inconsistent")
    result, extent = [], 0
    for source in inputs:
        _validate_tensor(source)
        if source["data_type"] != output["data_type"] or len(
            source["dimensions"]
        ) != len(output["dimensions"]):
            raise ValueError(
                "concatenate input/output ranks and data types must match"
            )
        if any(
            a != b
            for i, (a, b) in enumerate(
                zip(source["dimensions"], output["dimensions"])
            )
            if i != axis
        ):
            raise ValueError("concatenate non-axis dimensions must match")
        result.append(
            {
                **group,
                "tensors": [source, output],
                "input_uids": [source["uid"]],
                "parameters": {
                    "axis": axis,
                    "axis_offset": extent,
                    "n_elements": math.prod(source["dimensions"]),
                    "_concatenate_copy": True,
                },
            }
        )
        extent += source["dimensions"][axis]
    if extent != output["dimensions"][axis]:
        raise ValueError(
            "concatenate output axis is not the sum of its inputs"
        )
    # Small row tiles preserve arbitrary strides while avoiding per-element
    # rank decoding. Up to four inputs share one launch; larger concatenations
    # keep the general per-input implementation.
    rows = max(math.prod(t["dimensions"][:-1]) for t in inputs)
    if (
        len(inputs) <= 4
        and max(t["dimensions"][-1] for t in inputs) <= 2048
        and rows <= 2**31 - 1
        # The runtime flattens all grid dimensions into one uint32 block count.
        and rows * len(inputs) <= 2**32 - 1
    ):
        return [{**group, "parameters": {**parameters, "_concatenate_rows": True}}]
    return result


def _row_windows_disjoint(tensor):
    dimensions, strides = tensor["dimensions"], tensor["strides"]
    span = 1 + (dimensions[-1] - 1) * strides[-1]
    for stride, dimension in sorted(
        (s, d) for s, d in zip(strides[:-1], dimensions[:-1]) if d > 1
    ):
        if stride < span:
            return False
        span += (dimension - 1) * stride
    return True


def _concatenate_rows_configuration(parameters, tensors):
    # Validate the unexpanded graph exactly as the general path does.
    _expand_concatenate_group({"parameters": parameters, "tensors": tensors})
    inputs, output = tensors[:-1], tensors[-1]
    if not 1 <= len(inputs) <= 4:
        raise ValueError("concatenate row tile needs one to four inputs")
    if max(t["dimensions"][-1] for t in inputs) > 2048:
        raise ValueError("concatenate row tile is too wide")
    rows = max(math.prod(t["dimensions"][:-1]) for t in inputs)
    if rows > 2**31 - 1 or rows * len(inputs) > 2**32 - 1:
        raise ValueError("concatenate row grid exceeds the device launch limit")
    rank = len(output["dimensions"])
    axis = parameters["axis"]
    constants = {
        "AXIS": 8 - rank + axis,
        "INPUT_COUNT": len(inputs),
        "ROW_WINDOWS_DISJOINT": _row_windows_disjoint(output),
        "BLOCK_SIZE": 1 << (max(t["dimensions"][-1] for t in inputs) - 1).bit_length(),
    }
    constants.update(_shape_constants(output["dimensions"], output["strides"], "OUTPUT_STRIDE"))
    for part in range(4):
        source = inputs[min(part, len(inputs) - 1)]
        constants[f"EXTENT_{part}"] = source["dimensions"][axis]
        for i, stride in enumerate([0] * (8 - rank) + source["strides"]):
            constants[f"INPUT{part}_STRIDE_{i}"] = stride
    pointer = _STORAGE_POINTERS[output["data_type"]]
    return (
        "concatenate_rows_kernel",
        {**{f"input{i}_ptr": pointer for i in range(4)}, "output_ptr": pointer},
        constants,
        (rows, len(inputs), 1),
        [("tensor_alias", min(i, len(inputs) - 1)) for i in range(4)]
        + [("tensor_alias", len(inputs))],
    )


def _index_kernel_configuration(operation, parameters, tensors):
    for tensor in tensors:
        _validate_tensor(tensor)
    if operation == "concatenate" and parameters.get("_concatenate_rows") is True:
        return _concatenate_rows_configuration(parameters, tensors)
    output = tensors[-1]
    block = 256
    elements = _require_integer(parameters, "n_elements", maximum=2**63 - 1)
    if (elements + block - 1) // block > 2**31 - 1:
        raise ValueError("index/layout grid exceeds the device launch limit")
    axis = _require_integer(
        parameters, "axis", minimum=0, maximum=len(output["dimensions"]) - 1
    )
    constants = {"N_ELEMENTS": elements, "BLOCK_SIZE": block}
    if operation == "gen_index":
        if (
            len(tensors) != 1
            or output["data_type"] not in {"int32", "float32"}
            or elements != math.prod(output["dimensions"])
        ):
            raise ValueError("invalid gen_index tensor metadata")
        if (
            output["data_type"] == "int32"
            and output["dimensions"][axis] > 2**31
        ):
            raise ValueError("gen_index axis does not fit INT32")
        if _is_row_major_contiguous(output):
            # Avoid vector coordinate reconstruction for a linear output.
            # A larger tile also keeps large ranges from launching hundreds
            # of tiny programs on the NPU.
            block = min(2048, 1 << (elements - 1).bit_length())
            return (
                "gen_index_dense_kernel",
                {"output_ptr": "*i32" if output["data_type"] == "int32" else "*fp32"},
                {
                    "N_ELEMENTS": elements,
                    "AXIS_EXTENT": output["dimensions"][axis],
                    "INNER": math.prod(output["dimensions"][axis + 1:]),
                    "OUTER": math.prod(output["dimensions"][:axis]),
                    "BLOCK_SIZE": block,
                },
                ((elements + block - 1) // block, 1, 1),
                [("tensor", None)],
            )
        constants.update(
            _shape_constants(
                output["dimensions"], output["strides"], "OUTPUT_STRIDE"
            )
        )
        constants.update(
            {
                "AXIS_EXTENT": output["dimensions"][axis],
                "INNER": math.prod(output["dimensions"][axis + 1 :]),
            }
        )
        return (
            "gen_index_kernel",
            {
                "output_ptr": (
                    "*i32" if output["data_type"] == "int32" else "*fp32"
                )
            },
            constants,
            ((elements + block - 1) // block, 1, 1),
            [("tensor", None)],
        )
    if (
        operation != "concatenate"
        or len(tensors) != 2
        or parameters.get("_concatenate_copy") is not True
    ):
        raise ValueError(
            "concatenate must be expanded before kernel selection"
        )
    source = tensors[0]
    offset = _require_integer(
        parameters, "axis_offset", minimum=0, maximum=2**63 - 1
    )
    if source["data_type"] != output["data_type"] or elements != math.prod(
        source["dimensions"]
    ):
        raise ValueError("invalid concatenate copy input")
    if (
        len(source["dimensions"]) != len(output["dimensions"])
        or any(
            a != b
            for i, (a, b) in enumerate(
                zip(source["dimensions"], output["dimensions"])
            )
            if i != axis
        )
        or offset + source["dimensions"][axis] > output["dimensions"][axis]
    ):
        raise ValueError("concatenate copy region is outside output")
    constants.update(
        _shape_constants(
            source["dimensions"], source["strides"], "INPUT_STRIDE"
        )
    )
    constants.update(
        _shape_constants(
            source["dimensions"], output["strides"], "OUTPUT_STRIDE"
        )
    )
    constants["OUTPUT_BASE"] = offset * output["strides"][axis]
    pointer = _STORAGE_POINTERS[source["data_type"]]
    return (
        "concatenate_copy_kernel",
        {"input_ptr": pointer, "output_ptr": pointer},
        constants,
        ((elements + block - 1) // block, 1, 1),
        [("tensor", None), ("tensor", None)],
    )
