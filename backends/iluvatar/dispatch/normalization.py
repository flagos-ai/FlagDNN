# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Iluvatar dispatch / normalization implementation."""

from __future__ import annotations

from .nn_common import BATCH_NORM_INFERENCE_TUNING
from .nn_common import BATCH_NORM_TUNING
from .nn_common import FLOAT_TYPES
from .nn_common import GridSpec
from .nn_common import KernelStagePlan
from .nn_common import LAYER_NORM_TUNING
from .nn_common import MAX_I32
from .nn_common import MAX_I64
from .nn_common import MAX_RANK
from .nn_common import RMS_NORM_TUNING
from .nn_common import _checked_product
from .nn_common import _is_contiguous
from .nn_common import _make_stage
from .nn_common import _metadata_array
from .nn_common import _next_power_of_two
from .nn_common import _require_integer
from .nn_common import _require_number
from .nn_common import _require_object
from .nn_common import _same_dtype
from .nn_common import _tensor_pointer
from typing import Any
from typing import Mapping


def _validate_normalization(node: Mapping[str, Any]) -> dict[str, Any]:
    operation = str(node["operation"])
    p = _require_object(node["parameters"], "node.parameters")
    ports = _require_object(node["port_tensors"], "node.port_tensors")
    x = ports["x"]
    y = ports["y"]
    if x["data_type"] != y["data_type"] or x["data_type"] not in FLOAT_TYPES:
        raise ValueError("normalization X/Y must use one floating data type")
    if list(x["dimensions"]) != list(y["dimensions"]):
        raise ValueError("normalization Y shape must match X")

    if operation in ("layernorm", "rmsnorm"):
        scale, bias = ports["scale"], ports["bias"]
        _same_dtype((x, y, scale, bias), FLOAT_TYPES, "normalization")
        if not 1 <= len(x["dimensions"]) <= MAX_RANK:
            raise ValueError("normalization X rank must be in [1, 8]")
        if not _is_contiguous(x) or not _is_contiguous(y):
            raise ValueError("normalization X/Y must be contiguous")
        scale_dims = list(scale["dimensions"])
        x_dims = list(x["dimensions"])
        if not 1 <= len(scale_dims) <= len(x_dims):
            raise ValueError("normalization scale rank is invalid")
        leading = len(x_dims) - len(scale_dims)
        normalized_suffix = False
        normalized_elements = 1
        statistic_dims = list(x_dims)
        for axis, x_dim in enumerate(x_dims):
            scale_dim = 1 if axis < leading else scale_dims[axis - leading]
            if scale_dim != 1:
                if scale_dim != x_dim:
                    raise ValueError("normalization scale shape does not match X")
                normalized_suffix = True
            elif normalized_suffix and x_dim != 1:
                raise ValueError(
                    "normalization scale must describe a contiguous suffix"
                )
            if normalized_suffix:
                normalized_elements *= x_dim
                if normalized_elements > MAX_I64:
                    raise ValueError("normalization extent overflows int64")
                statistic_dims[axis] = 1
        if (
            not normalized_suffix
            or _checked_product(scale_dims, "normalization scale elements")
            != normalized_elements
            or _checked_product(list(bias["dimensions"]), "normalization bias elements")
            != normalized_elements
        ):
            raise ValueError("normalization scale/bias size is invalid")
        if not _is_contiguous(scale) or not _is_contiguous(bias):
            raise ValueError("normalization scale/bias must be contiguous")
        statistic_names = (
            ("inv_variance",) if operation == "rmsnorm" else ("mean", "inv_variance")
        )
        for name in statistic_names:
            statistic = ports[name]
            if (
                statistic["data_type"] != "float32"
                or list(statistic["dimensions"]) != statistic_dims
                or not _is_contiguous(statistic)
            ):
                raise ValueError(
                    f"normalization statistic {name!r} metadata is invalid"
                )
        rows = _checked_product(x_dims, "normalization X elements")
        rows //= normalized_elements
        if _require_integer(p, "rows", maximum=MAX_I64) != rows:
            raise ValueError("parameters.rows is inconsistent with X")
        if (
            _require_integer(p, "normalized_elements", maximum=MAX_I64)
            != normalized_elements
        ):
            raise ValueError(
                "parameters.normalized_elements is inconsistent with scale"
            )
        epsilon = _require_number(p, "epsilon", positive=True)
        if (
            "forward_phase" in p
            and _require_integer(p, "forward_phase", minimum=0, maximum=MAX_I32) != 2
        ):
            raise ValueError("normalization supports TRAINING phase only")
        return {
            "data_type": str(x["data_type"]),
            "rows": rows,
            "normalized_elements": normalized_elements,
            "epsilon": epsilon,
            "contiguous": True,
        }

    if not 2 <= len(x["dimensions"]) <= MAX_RANK:
        raise ValueError("batchnorm X rank must be in [2, 8]")
    dims = list(x["dimensions"])
    batch, channels = dims[:2]
    spatial = _checked_product(dims[2:] or [1], "batchnorm spatial extent")
    n_elements = _checked_product(dims, "batchnorm elements")
    expected = {
        "n_elements": n_elements,
        "channels": channels,
        "spatial": spatial,
        "rank": len(dims),
    }
    if operation == "batchnorm":
        expected["batch"] = batch
    for name, value in expected.items():
        if _require_integer(p, name, maximum=MAX_I64) != value:
            raise ValueError(f"parameters.{name} is inconsistent with X")
    _metadata_array(p, "dimensions", dims)
    _metadata_array(p, "x_strides", list(x["strides"]))
    _metadata_array(p, "y_strides", list(y["strides"]))

    if operation == "batchnorm":
        scale, bias = ports["scale"], ports["bias"]
        _same_dtype((x, y, scale, bias), FLOAT_TYPES, "batchnorm")
        for name in ("scale", "bias"):
            tensor = ports[name]
            if _checked_product(
                list(tensor["dimensions"]), f"batchnorm {name} elements"
            ) != channels or not _is_contiguous(tensor):
                raise ValueError(f"batchnorm {name} must be contiguous with C elements")
        for name in (
            "previous_running_mean",
            "previous_running_variance",
            "mean",
            "inv_variance",
            "next_running_mean",
            "next_running_variance",
        ):
            tensor = ports[name]
            if (
                tensor["data_type"] != "float32"
                or _checked_product(
                    list(tensor["dimensions"]),
                    f"batchnorm {name} elements",
                )
                != channels
                or not _is_contiguous(tensor)
            ):
                raise ValueError(
                    f"batchnorm statistic {name!r} must be contiguous " "float32[C]"
                )
        epsilon = _require_number(p, "epsilon", positive=True)
        momentum = _require_number(p, "momentum")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("batchnorm momentum must be in [0, 1]")
        return {
            "data_type": str(x["data_type"]),
            "batch": batch,
            "channels": channels,
            "spatial": spatial,
            "n_elements": n_elements,
            "epsilon": epsilon,
            "momentum": momentum,
            "contiguous": _is_contiguous(x) and _is_contiguous(y),
        }

    for name in ("mean", "inv_variance", "scale", "bias"):
        tensor = ports[name]
        if tensor["data_type"] not in FLOAT_TYPES:
            raise ValueError(f"batchnorm inference {name} must be floating point")
        if _checked_product(
            list(tensor["dimensions"]),
            f"batchnorm inference {name} elements",
        ) != channels or not _is_contiguous(tensor):
            raise ValueError(
                f"batchnorm inference {name} must be contiguous with " "C elements"
            )
    return {
        "data_type": str(x["data_type"]),
        "batch": batch,
        "channels": channels,
        "spatial": spatial,
        "n_elements": n_elements,
        "epsilon": 0.0,
        "contiguous": _is_contiguous(x) and _is_contiguous(y),
    }


def _padded_rank_metadata(
    tensor: Mapping[str, Any],
) -> tuple[list[int], list[int]]:
    leading = MAX_RANK - len(tensor["dimensions"])
    return (
        [1] * leading + list(tensor["dimensions"]),
        [0] * leading + list(tensor["strides"]),
    )


def _normalization_stage(node: Mapping[str, Any]) -> KernelStagePlan:
    operation = str(node["operation"])
    d = _require_object(node["derived"], "node.derived")
    if operation == "layernorm":
        block = min(_next_power_of_two(int(d["normalized_elements"])), 65536)
        constants: dict[str, int | float | bool] = {
            "M": int(d["rows"]),
            "eps": float(d["epsilon"]),
            "N": int(d["normalized_elements"]),
            "BLOCK_SIZE": block,
            "ROWS_PER_PROGRAM": 1,
            "HAS_WEIGHT": True,
            "HAS_BIAS": True,
            "RETURN_STATS": True,
        }
        pointers = (
            _tensor_pointer(node, "x_ptr", "x"),
            _tensor_pointer(node, "y_ptr", "y"),
            _tensor_pointer(node, "mean_ptr", "mean"),
            _tensor_pointer(node, "inv_variance_ptr", "inv_variance"),
            _tensor_pointer(node, "weight_ptr", "scale"),
            _tensor_pointer(node, "bias_ptr", "bias"),
        )
        return _make_stage(
            operation=operation,
            stage_name=operation,
            function_name="layer_norm_kernel",
            pointer_arguments=pointers,
            constants=constants,
            scalar_arguments=(("M", "i32"),),
            tuning=LAYER_NORM_TUNING,
            tuning_key_value=int(d["normalized_elements"]),
            grid_spec=GridSpec("norm_rows", (int(d["rows"]),)),
        )
    if operation == "rmsnorm":
        block = min(_next_power_of_two(int(d["normalized_elements"])), 65536)
        constants = {
            "M": int(d["rows"]),
            "N": int(d["normalized_elements"]),
            "eps": float(d["epsilon"]),
            "BLOCK_SIZE": block,
            "ROWS_PER_PROGRAM": 1,
            "HAS_WEIGHT": True,
            "HAS_BIAS": True,
            "RETURN_STATS": True,
        }
        pointers = (
            _tensor_pointer(node, "x_ptr", "x"),
            _tensor_pointer(node, "y_ptr", "y"),
            _tensor_pointer(node, "weight_ptr", "scale"),
            _tensor_pointer(node, "bias_ptr", "bias"),
            _tensor_pointer(node, "inv_variance_ptr", "inv_variance"),
        )
        return _make_stage(
            operation=operation,
            stage_name=operation,
            function_name="rms_norm_kernel",
            pointer_arguments=pointers,
            constants=constants,
            scalar_arguments=(("M", "i32"),),
            tuning=RMS_NORM_TUNING,
            tuning_key_value=int(d["normalized_elements"]),
            grid_spec=GridSpec("norm_rows", (int(d["rows"]),)),
        )

    if operation == "batchnorm":
        batch_block = _next_power_of_two(int(d["batch"]))
        use_nchw = bool(d["contiguous"]) and batch_block <= 512
        block = max(256, batch_block) if use_nchw else 256
        constants = {
            "N": int(d["batch"]),
            "C": int(d["channels"]),
            "S": int(d["spatial"]),
            "eps": float(d["epsilon"]),
            "momentum": float(d["momentum"]),
            "BLOCK_SIZE": block,
            "IS_TRAINING": True,
            "HAS_WEIGHT": True,
            "HAS_BIAS": True,
            "HAS_RUNNING_STATS": True,
            "RETURN_STATS": True,
        }
        pointers = (
            _tensor_pointer(node, "x_ptr", "x"),
            _tensor_pointer(node, "y_ptr", "y"),
            _tensor_pointer(node, "mean_ptr", "previous_running_mean"),
            _tensor_pointer(node, "var_ptr", "previous_running_variance"),
            _tensor_pointer(node, "weight_ptr", "scale"),
            _tensor_pointer(node, "bias_ptr", "bias"),
            _tensor_pointer(node, "saved_mean_ptr", "mean"),
            _tensor_pointer(node, "saved_inv_var_ptr", "inv_variance"),
            _tensor_pointer(node, "next_running_mean_ptr", "next_running_mean"),
            _tensor_pointer(node, "next_running_var_ptr", "next_running_variance"),
        )
        function = "batch_norm_nchw_kernel" if use_nchw else "batch_norm_kernel"
        if not use_nchw:
            x = node["port_tensors"]["x"]
            y = node["port_tensors"]["y"]
            dimensions, x_strides = _padded_rank_metadata(x)
            _, y_strides = _padded_rank_metadata(y)
            constants["STRIDED"] = not bool(d["contiguous"])
            for axis in range(MAX_RANK):
                constants[f"DIM_{axis}"] = dimensions[axis]
                constants[f"INPUT_STRIDE_{axis}"] = x_strides[axis]
                constants[f"OUTPUT_STRIDE_{axis}"] = y_strides[axis]
        return _make_stage(
            operation=operation,
            stage_name=operation,
            function_name=function,
            pointer_arguments=pointers,
            constants=constants,
            scalar_arguments=(
                (("N", "i32"), ("C", "i32"), ("S", "i32")) if not use_nchw else ()
            ),
            tuning=BATCH_NORM_TUNING,
            tuning_key_value=int(d["channels"]),
            grid_spec=GridSpec("batchnorm_channels", (int(d["channels"]),)),
        )

    x = node["port_tensors"]["x"]
    y = node["port_tensors"]["y"]
    constants = {
        "total_elements": int(d["n_elements"]),
        "C": int(d["channels"]),
        "S": int(d["spatial"]),
        "eps": 0.0,
        "BLOCK_SIZE": 256,
        "HAS_WEIGHT": True,
        "HAS_BIAS": True,
        "STAT_IS_INV_VARIANCE": True,
    }
    pointers = (
        _tensor_pointer(node, "x_ptr", "x"),
        _tensor_pointer(node, "mean_ptr", "mean"),
        _tensor_pointer(node, "stat_ptr", "inv_variance"),
        _tensor_pointer(node, "weight_ptr", "scale"),
        _tensor_pointer(node, "bias_ptr", "bias"),
        _tensor_pointer(node, "y_ptr", "y"),
    )

    if bool(d["contiguous"]):
        function = "batch_norm_inference_nchw_kernel"
        constants.pop("total_elements")
        constants["N"] = int(d["batch"])
        grid = GridSpec("linear", (int(d["n_elements"]),))
    else:
        function = "batch_norm_inference_kernel"
        dimensions, x_strides = _padded_rank_metadata(x)
        _, y_strides = _padded_rank_metadata(y)
        constants["STRIDED"] = True
        for axis in range(MAX_RANK):
            constants[f"DIM_{axis}"] = dimensions[axis]
            constants[f"INPUT_STRIDE_{axis}"] = x_strides[axis]
            constants[f"OUTPUT_STRIDE_{axis}"] = y_strides[axis]
        grid = GridSpec("linear", (int(d["n_elements"]),))
    return _make_stage(
        operation=operation,
        stage_name=operation,
        function_name=function,
        pointer_arguments=pointers,
        constants=constants,
        scalar_arguments=(
            (
                ("total_elements", "i32"),
                ("C", "i32"),
                ("S", "i32"),
            )
            if not bool(d["contiguous"])
            else ()
        ),
        tuning=BATCH_NORM_INFERENCE_TUNING,
        tuning_key_value=int(d["n_elements"]),
        grid_spec=grid,
    )
