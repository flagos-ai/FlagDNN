"""Shape, layout and launch selection for normalization forward/backward."""

import math
from .extended_common import (
    FLOAT_DATA_TYPES,
    TRITON_POINTER_TYPES,
    _has_non_overlapping_strides,
    _require_integer,
    _is_row_major_contiguous,
)


def _extended_normalization_configuration(operation, parameters, tensors):
    backward = operation.endswith("_backward")
    rms = operation == "rmsnorm_backward"
    count = (7 if rms else 8) if backward else 6
    if len(tensors) != count:
        raise ValueError("normalization tensor count is inconsistent")
    x = tensors[1 if backward else 0]
    scale = tensors[2 if backward else 1]
    shape = x["dimensions"]
    rank = len(shape)
    if not 1 <= rank <= 8:
        raise ValueError("normalization rank must be 1..8")
    for tensor in tensors:
        if tensor[
            "data_type"
        ] not in FLOAT_DATA_TYPES or not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        ):
            raise ValueError(
                "normalization requires floating tensors "
                "with non-overlapping strides"
            )
        if len(tensor["dimensions"]) > rank:
            raise ValueError("normalization tensor rank cannot exceed X rank")
    parameter_shape = [1] * (rank - len(scale["dimensions"])) + scale["dimensions"]
    if any(p not in (1, d) for p, d in zip(parameter_shape, shape)):
        raise ValueError("normalization scale cannot broadcast to X")
    if operation.startswith("batchnorm"):
        axes = [i for i in range(rank) if i != 1]
    elif operation.startswith("instancenorm"):
        axes = list(range(2, rank))
    else:
        axes = [
            i
            for i, p in enumerate(parameter_shape)
            if p != 1 and (i != 0 or not operation.startswith("adalayernorm"))
        ]
    if not axes and shape[-1] == 1:
        axes = [rank - 1]
    if parameters.get("axes") != axes or not axes:
        raise ValueError("normalization reduction axes are inconsistent")
    if operation.startswith(("batchnorm", "instancenorm")) and (
        rank < 2 or parameter_shape != [1, shape[1]] + [1] * (rank - 2)
    ):
        raise ValueError("channel normalization scale shape is invalid")
    reduction = math.prod(shape[i] for i in axes)
    groups = math.prod(shape) // reduction
    if (
        _require_integer(parameters, "groups") != groups
        or _require_integer(parameters, "reduction") != reduction
    ):
        raise ValueError("normalization group/reduction sizes are inconsistent")
    statistics_shape = [1 if i in axes else d for i, d in enumerate(shape)]

    def same(tensor, dimensions, data_type=None):
        if tensor["dimensions"] != dimensions or (
            data_type and tensor["data_type"] != data_type
        ):
            raise ValueError("normalization tensor metadata is inconsistent")

    if scale["data_type"] not in (x["data_type"], "float32"):
        raise ValueError("normalization scale must use X type or FP32")
    if backward:
        same(tensors[0], shape, x["data_type"])
        for tensor in tensors[3 : 4 if rms else 5]:
            same(tensor, statistics_shape, "float32")
        same(tensors[-3], shape, x["data_type"])
        for tensor in tensors[-2:]:
            same(tensor, scale["dimensions"])
            if tensor["data_type"] not in ("float32", scale["data_type"]):
                raise ValueError("normalization affine gradient type is invalid")
        roles = ["DY", "X", "SCALE", "MEAN", "INV", "DX", "DSCALE", "DBIAS"]
        abi_tensors = tensors[:3] + [tensors[3]] + tensors[3:] if rms else tensors
        arguments = (
            [("tensor", None)] * 3
            + ([("tensor_alias", 3)] if rms else [])
            + [("tensor", None)] * (len(tensors) - 3)
        )
    else:
        same(tensors[2], scale["dimensions"], scale["data_type"])
        same(tensors[3], shape, x["data_type"])
        for tensor in tensors[4:]:
            same(tensor, statistics_shape, "float32")
        roles = ["X", "SCALE", "BIAS", "Y", "MEAN", "INV"]
        abi_tensors = tensors
        arguments = [("tensor", None)] * len(tensors)
    if all(_is_row_major_contiguous(t) for t in tensors) and reduction <= 65536:
        signature = {
            role.lower() + "_ptr": TRITON_POINTER_TYPES[tensor["data_type"]]
            for role, tensor in zip(roles, abi_tensors)
        }
        block = 1 << (reduction - 1).bit_length()
        if operation == "batchnorm_backward":
            return (
                "compact_batchnorm_backward",
                signature,
                {
                    "CHANNELS": shape[1],
                    "SPATIAL": math.prod(shape[2:]),
                    "REDUCTION": reduction,
                    "BLOCK_SIZE": block,
                },
                (shape[1], 1, 1),
                arguments,
            )
        if not backward and (
            operation == "instancenorm" or axes == list(range(axes[0], rank))
        ):
            epsilon = parameters.get("epsilon")
            if (
                not isinstance(epsilon, (int, float))
                or not math.isfinite(epsilon)
                or epsilon <= 0
            ):
                raise ValueError("normalization epsilon must be finite and positive")
            mode = (
                0
                if operation == "instancenorm"
                else (2 if parameter_shape[0] != 1 else 1)
            )
            return (
                "compact_normalization_forward",
                signature,
                {
                    "CHANNELS": shape[1] if rank > 1 else 1,
                    "REDUCTION": reduction,
                    "BLOCK_SIZE": block,
                    "ROWS_PER_BATCH": groups // shape[0],
                    "AFFINE_MODE": mode,
                    "EPSILON": float(epsilon),
                },
                (groups, 1, 1),
                arguments,
            )
    parameter_count = math.prod(parameter_shape)
    affine_reduction = math.prod(shape) // parameter_count
    constants = {
        "GROUPS": groups,
        "REDUCTION": reduction,
        "AXES": sum(1 << (i + 8 - rank) for i in axes),
        "PARAM_AXES": sum(
            1 << (i + 8 - rank) for i, p in enumerate(parameter_shape) if p == 1
        ),
        "PARAMETERS": parameter_count,
        "AFFINE_REDUCTION": affine_reduction,
        "BLOCK_SIZE": min(
            1024,
            1 << (max(reduction, affine_reduction if backward else 1) - 1).bit_length(),
        ),
    }
    if backward:
        constants["RMS"] = rms
    else:
        epsilon = parameters.get("epsilon")
        if (
            not isinstance(epsilon, (int, float))
            or not math.isfinite(epsilon)
            or epsilon <= 0
        ):
            raise ValueError("normalization epsilon must be finite and positive")
        constants["EPSILON"] = float(epsilon)
    for i, dim in enumerate([1] * (8 - rank) + shape):
        constants[f"DIM_{i}"] = dim
    for role, tensor in zip(roles, abi_tensors):
        dims = [1] * (8 - len(tensor["dimensions"])) + tensor["dimensions"]
        strides = [0] * (8 - len(tensor["strides"])) + tensor["strides"]
        for i in range(8):
            constants[f"{role}_STRIDE_{i}"] = strides[i] if dims[i] != 1 else 0
    grid = groups + parameter_count if backward else groups
    if grid > 2**31 - 1:
        raise ValueError("normalization launch grid exceeds Iluvatar limits")
    return (
        (
            "extended_normalization_backward"
            if backward
            else "extended_normalization_forward"
        ),
        {
            role.lower() + "_ptr": TRITON_POINTER_TYPES[tensor["data_type"]]
            for role, tensor in zip(roles, abi_tensors)
        },
        constants,
        (grid, 1, 1),
        arguments,
    )
