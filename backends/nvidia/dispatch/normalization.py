"""Normalization validation and launch configurations."""

from __future__ import annotations

from typing import Any
import math

from .common import (
    FLOAT_DATA_TYPES,
    TRITON_POINTER_TYPES,
    _is_row_major_contiguous,
    _require_integer,
    _require_number,
    _unary_pointwise_tensor_constants,
)


def _normalization_forward_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
    *,
    rmsnorm: bool,
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | float | str | bool],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    expected_count = 5 if rmsnorm else 6
    if len(tensors) != expected_count:
        raise ValueError("normalization tensor count is invalid")
    x, scale, bias, y = tensors[:4]
    statistics = tensors[4:]
    if (
        x["data_type"] not in FLOAT_DATA_TYPES
        or y["data_type"] != x["data_type"]
        or scale["data_type"] != x["data_type"]
        or bias["data_type"] != x["data_type"]
    ):
        raise ValueError("normalization X/Y/scale/bias data types must match")
    if x["dimensions"] != y["dimensions"]:
        raise ValueError("normalization Y shape must match X")
    if not _is_row_major_contiguous(x) or not _is_row_major_contiguous(y):
        raise ValueError("normalization X/Y must be contiguous")
    if not _is_row_major_contiguous(scale) or not _is_row_major_contiguous(
        bias
    ):
        raise ValueError("normalization scale/bias must be contiguous")

    rows = _require_integer(parameters, "rows")
    normalized_elements = _require_integer(parameters, "normalized_elements")
    if math.prod(x["dimensions"]) != rows * normalized_elements:
        raise ValueError("normalization row/extent parameters are invalid")
    if (
        math.prod(scale["dimensions"]) != normalized_elements
        or math.prod(bias["dimensions"]) != normalized_elements
    ):
        raise ValueError("normalization scale/bias size is invalid")
    for statistic in statistics:
        if (
            statistic["data_type"] != "float32"
            or math.prod(statistic["dimensions"]) != rows
            or not _is_row_major_contiguous(statistic)
        ):
            raise ValueError("normalization statistic metadata is invalid")
    epsilon = _require_number(parameters, "epsilon")
    if epsilon <= 0.0:
        raise ValueError("normalization epsilon must be positive")
    block = 1 << (normalized_elements - 1).bit_length()
    if block > 65536:
        raise ValueError("normalization extent exceeds kernel limit")
    rows_per_program = 1
    pointer_types: list[str] = []
    for tensor in tensors:
        pointer_type = TRITON_POINTER_TYPES.get(tensor["data_type"])
        if pointer_type is None:
            raise ValueError("unsupported normalization data type")
        pointer_types.append(pointer_type)

    if (
        not rmsnorm
        and x["data_type"] == "float16"
        and 32 <= normalized_elements <= 512
        and rows >= 64
    ):
        return (
            "layer_norm_warp_kernel",
            {
                "x_ptr": pointer_types[0],
                "y_ptr": pointer_types[3],
                "mean_ptr": pointer_types[4],
                "inv_variance_ptr": pointer_types[5],
                "weight_ptr": pointer_types[1],
                "bias_ptr": pointer_types[2],
            },
            {
                "M": rows,
                "N": normalized_elements,
                "EPS": epsilon,
                "R": 4,
                "L": 32,
                "V": 4 if normalized_elements <= 128 else 8,
                "W": 4,
            },
            ((rows + 3) // 4, 1, 1),
            [("tensor_alias", i) for i in (0, 3, 4, 5, 1, 2)],
        )
    if (
        rmsnorm
        and x["data_type"] == "float16"
        and 32 <= normalized_elements <= 512
        and rows >= 64
    ):
        return (
            "rms_norm_warp_kernel",
            {
                "x_ptr": pointer_types[0],
                "y_ptr": pointer_types[3],
                "weight_ptr": pointer_types[1],
                "bias_ptr": pointer_types[2],
                "inv_variance_ptr": pointer_types[4],
            },
            {
                "M": rows,
                "N": normalized_elements,
                "EPS": epsilon,
                "V": 4 if normalized_elements <= 128 else 8,
            },
            ((rows + 3) // 4, 1, 1),
            [("tensor_alias", i) for i in (0, 3, 1, 2, 4)],
        )
    if rmsnorm:
        return (
            "rms_norm_kernel",
            {
                "x_ptr": pointer_types[0],
                "y_ptr": pointer_types[3],
                "weight_ptr": pointer_types[1],
                "bias_ptr": pointer_types[2],
                "inv_variance_ptr": pointer_types[4],
                "M": "i32",
            },
            {
                "N": normalized_elements,
                "eps": epsilon,
                "BLOCK_SIZE": block,
                "ROWS_PER_PROGRAM": rows_per_program,
                "STATIC_ROWS": rows,
                "HAS_WEIGHT": True,
                "HAS_BIAS": True,
                "RETURN_STATS": True,
            },
            ((rows + rows_per_program - 1) // rows_per_program, 1, 1),
            [
                ("tensor_alias", 0),
                ("tensor_alias", 3),
                ("tensor_alias", 1),
                ("tensor_alias", 2),
                ("tensor_alias", 4),
                ("scalar_i32", "rows"),
            ],
        )
    return (
        "layer_norm_kernel",
        {
            "x_ptr": pointer_types[0],
            "y_ptr": pointer_types[3],
            "mean_ptr": pointer_types[4],
            "inv_variance_ptr": pointer_types[5],
            "weight_ptr": pointer_types[1],
            "bias_ptr": pointer_types[2],
            "M": "i32",
        },
        {
            "eps": epsilon,
            "N": normalized_elements,
            "BLOCK_SIZE": block,
            "ROWS_PER_PROGRAM": rows_per_program,
            "STATIC_ROWS": rows,
            "PAIRED_REDUCTION": False,
            "EVICT_INPUT_FIRST": (
                x["data_type"] == "bfloat16" and normalized_elements <= 4096
            ),
            "HAS_WEIGHT": True,
            "HAS_BIAS": True,
            "RETURN_STATS": True,
        },
        ((rows + rows_per_program - 1) // rows_per_program, 1, 1),
        [
            ("tensor_alias", 0),
            ("tensor_alias", 3),
            ("tensor_alias", 4),
            ("tensor_alias", 5),
            ("tensor_alias", 1),
            ("tensor_alias", 2),
            ("scalar_i32", "rows"),
        ],
    )


def _batchnorm_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | float | str | bool],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    if len(tensors) != 10:
        raise ValueError("batchnorm tensor count is invalid")
    (
        x,
        scale,
        bias,
        previous_mean,
        previous_variance,
        y,
        mean,
        inv_variance,
        next_mean,
        next_variance,
    ) = tensors
    if (
        x["data_type"] != y["data_type"]
        or x["data_type"] not in FLOAT_DATA_TYPES
        or scale["data_type"] != x["data_type"]
        or bias["data_type"] != x["data_type"]
    ):
        raise ValueError(
            "batchnorm X/Y/scale/bias data types must match and be floating"
        )
    if x["dimensions"] != y["dimensions"]:
        raise ValueError("batchnorm Y shape must match X")
    if len(x["dimensions"]) < 2 or len(x["dimensions"]) > 8:
        raise ValueError("batchnorm X rank must be in [2, 8]")

    channels = x["dimensions"][1]
    for parameter_name, tensor in (("scale", scale), ("bias", bias)):
        if math.prod(tensor["dimensions"]) != channels:
            raise ValueError(
                f"batchnorm {parameter_name} size must match channels"
            )
        if not _is_row_major_contiguous(tensor):
            raise ValueError(f"batchnorm {parameter_name} must be contiguous")
    for statistic_name, tensor in (
        ("previous_running_mean", previous_mean),
        ("previous_running_variance", previous_variance),
        ("mean", mean),
        ("inv_variance", inv_variance),
        ("next_running_mean", next_mean),
        ("next_running_variance", next_variance),
    ):
        if tensor["data_type"] != "float32":
            raise ValueError(f"batchnorm {statistic_name} must use float32")
        if math.prod(tensor["dimensions"]) != channels:
            raise ValueError(
                f"batchnorm {statistic_name} size must match channels"
            )
        if not _is_row_major_contiguous(tensor):
            raise ValueError(f"batchnorm {statistic_name} must be contiguous")

    total_elements = _require_integer(parameters, "n_elements")
    batch = _require_integer(parameters, "batch")
    configured_channels = _require_integer(parameters, "channels")
    spatial = _require_integer(parameters, "spatial")
    rank = _require_integer(parameters, "rank")
    if total_elements != math.prod(x["dimensions"]):
        raise ValueError(
            "parameters.n_elements is inconsistent with batchnorm X"
        )
    if batch != x["dimensions"][0]:
        raise ValueError("parameters.batch is inconsistent with batchnorm X")
    if configured_channels != channels:
        raise ValueError(
            "parameters.channels is inconsistent with batchnorm X"
        )
    if spatial != math.prod(x["dimensions"][2:]):
        raise ValueError("parameters.spatial is inconsistent with batchnorm X")
    if rank != len(x["dimensions"]):
        raise ValueError("parameters.rank is inconsistent with batchnorm X")
    epsilon = _require_number(parameters, "epsilon")
    momentum = _require_number(parameters, "momentum")
    if epsilon <= 0.0:
        raise ValueError("parameters.epsilon must be positive")
    if momentum < 0.0 or momentum > 1.0:
        raise ValueError("parameters.momentum must be in [0, 1]")

    pointer_types = [
        TRITON_POINTER_TYPES.get(tensor["data_type"]) for tensor in tensors
    ]
    if any(pointer_type is None for pointer_type in pointer_types):
        raise ValueError("unsupported batchnorm data type")
    block = 256
    constants: dict[str, int | float | str | bool] = {
        "eps": epsilon,
        "momentum": momentum,
        "BLOCK_SIZE": block,
        "IS_TRAINING": True,
        "HAS_WEIGHT": True,
        "HAS_BIAS": True,
        "HAS_RUNNING_STATS": True,
        "RETURN_STATS": True,
    }
    constants.update(_unary_pointwise_tensor_constants([x, y]))
    # BatchNorm derives the parameter channel from the logical NCHW index.
    # A physically dense channels-last tensor therefore cannot use the
    # pointwise helper's physical-linear fast path even when X and Y have the
    # same mapping.
    constants["STRIDED"] = not (
        _is_row_major_contiguous(x) and _is_row_major_contiguous(y)
    )
    if (
        spatial >= 256
        and _is_row_major_contiguous(x)
        and _is_row_major_contiguous(y)
    ):
        return (
            "batch_norm_nchw_kernel",
            {
                "x_ptr": pointer_types[0],
                "y_ptr": pointer_types[5],
                "mean_ptr": pointer_types[3],
                "var_ptr": pointer_types[4],
                "weight_ptr": pointer_types[1],
                "bias_ptr": pointer_types[2],
                "saved_mean_ptr": pointer_types[6],
                "saved_inv_var_ptr": pointer_types[7],
                "next_running_mean_ptr": pointer_types[8],
                "next_running_var_ptr": pointer_types[9],
            },
            {
                "N": batch,
                "C": channels,
                "S": spatial,
                "eps": epsilon,
                "momentum": momentum,
                "BLOCK_SIZE": block,
                "IS_TRAINING": True,
                "HAS_WEIGHT": True,
                "HAS_BIAS": True,
                "HAS_RUNNING_STATS": True,
                "RETURN_STATS": True,
            },
            (channels, 1, 1),
            [
                ("tensor_alias", 0),
                ("tensor_alias", 5),
                ("tensor_alias", 3),
                ("tensor_alias", 4),
                ("tensor_alias", 1),
                ("tensor_alias", 2),
                ("tensor_alias", 6),
                ("tensor_alias", 7),
                ("tensor_alias", 8),
                ("tensor_alias", 9),
            ],
        )
    return (
        "batch_norm_kernel",
        {
            "x_ptr": pointer_types[0],
            "y_ptr": pointer_types[5],
            "mean_ptr": pointer_types[3],
            "var_ptr": pointer_types[4],
            "weight_ptr": pointer_types[1],
            "bias_ptr": pointer_types[2],
            "saved_mean_ptr": pointer_types[6],
            "saved_inv_var_ptr": pointer_types[7],
            "next_running_mean_ptr": pointer_types[8],
            "next_running_var_ptr": pointer_types[9],
            "N": "i32",
            "C": "i32",
            "S": "i32",
        },
        constants,
        (channels, 1, 1),
        [
            ("tensor_alias", 0),
            ("tensor_alias", 5),
            ("tensor_alias", 3),
            ("tensor_alias", 4),
            ("tensor_alias", 1),
            ("tensor_alias", 2),
            ("tensor_alias", 6),
            ("tensor_alias", 7),
            ("tensor_alias", 8),
            ("tensor_alias", 9),
            ("scalar_i32", "batch"),
            ("scalar_i32", "channels"),
            ("scalar_i32", "spatial"),
        ],
    )


def _batchnorm_inference_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | float | str | bool],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    if len(tensors) != 6:
        raise ValueError("batchnorm inference tensor count is invalid")
    x, mean, inv_variance, scale, bias, y = tensors
    if (
        x["data_type"] != y["data_type"]
        or x["data_type"] not in FLOAT_DATA_TYPES
    ):
        raise ValueError(
            "batchnorm inference X/Y data types must match and be floating"
        )
    if x["dimensions"] != y["dimensions"]:
        raise ValueError("batchnorm inference Y shape must match X")
    if len(x["dimensions"]) < 2 or len(x["dimensions"]) > 8:
        raise ValueError("batchnorm inference X rank must be in [2, 8]")

    channels = x["dimensions"][1]
    for parameter_name, tensor in (
        ("mean", mean),
        ("inv_variance", inv_variance),
        ("scale", scale),
        ("bias", bias),
    ):
        if tensor["data_type"] not in FLOAT_DATA_TYPES:
            raise ValueError(
                f"batchnorm inference {parameter_name} must be floating"
            )
        if math.prod(tensor["dimensions"]) != channels:
            raise ValueError(
                f"batchnorm inference {parameter_name} size must match "
                "channels"
            )
        if not _is_row_major_contiguous(tensor):
            raise ValueError(
                f"batchnorm inference {parameter_name} must be contiguous"
            )

    total_elements = _require_integer(parameters, "n_elements")
    configured_channels = _require_integer(parameters, "channels")
    spatial = _require_integer(parameters, "spatial")
    rank = _require_integer(parameters, "rank")
    if total_elements != math.prod(x["dimensions"]):
        raise ValueError(
            "parameters.n_elements is inconsistent with batchnorm inference X"
        )
    if configured_channels != channels:
        raise ValueError(
            "parameters.channels is inconsistent with batchnorm inference X"
        )
    if spatial != math.prod(x["dimensions"][2:]):
        raise ValueError(
            "parameters.spatial is inconsistent with batchnorm inference X"
        )
    if rank != len(x["dimensions"]):
        raise ValueError(
            "parameters.rank is inconsistent with batchnorm inference X"
        )

    pointer_types = [
        TRITON_POINTER_TYPES.get(tensor["data_type"]) for tensor in tensors
    ]
    if any(pointer_type is None for pointer_type in pointer_types):
        raise ValueError("unsupported batchnorm inference data type")
    block = 256
    constants: dict[str, int | float | str | bool] = {
        "eps": 0.0,
        "BLOCK_SIZE": block,
        "HAS_WEIGHT": True,
        "HAS_BIAS": True,
        "STAT_IS_INV_VARIANCE": True,
    }
    if _is_row_major_contiguous(x) and _is_row_major_contiguous(y):
        constants.update({"C": channels, "S": spatial})
        batch = total_elements // (channels * spatial)
        block_s = min(1 << (spatial - 1).bit_length(), block)
        block_c = block // block_s
        spatial_blocks = (spatial + block_s - 1) // block_s
        channel_blocks = (channels + block_c - 1) // block_c
        return (
            "batch_norm_inference_nchw_kernel",
            {
                "x_ptr": pointer_types[0],
                "mean_ptr": pointer_types[1],
                "stat_ptr": pointer_types[2],
                "weight_ptr": pointer_types[3],
                "bias_ptr": pointer_types[4],
                "y_ptr": pointer_types[5],
            },
            constants,
            (batch * channel_blocks * spatial_blocks, 1, 1),
            [("tensor", None)] * 6,
        )

    constants.update(_unary_pointwise_tensor_constants([x, y]))
    constants["STRIDED"] = not (
        _is_row_major_contiguous(x) and _is_row_major_contiguous(y)
    )
    return (
        "batch_norm_inference_kernel",
        {
            "x_ptr": pointer_types[0],
            "mean_ptr": pointer_types[1],
            "stat_ptr": pointer_types[2],
            "weight_ptr": pointer_types[3],
            "bias_ptr": pointer_types[4],
            "y_ptr": pointer_types[5],
            "total_elements": "i32",
            "C": "i32",
            "S": "i32",
        },
        constants,
        ((total_elements + block - 1) // block, 1, 1),
        [
            ("tensor", None),
            ("tensor", None),
            ("tensor", None),
            ("tensor", None),
            ("tensor", None),
            ("tensor", None),
            ("scalar_i32", "n_elements"),
            ("scalar_i32", "channels"),
            ("scalar_i32", "spatial"),
        ],
    )
