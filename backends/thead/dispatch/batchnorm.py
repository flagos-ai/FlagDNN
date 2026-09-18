# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""THead dispatch batchnorm lowering."""

from __future__ import annotations

from typing import Any
import math
from ..codegen.abi import (
    _tensor_argument,
    _tensor_pointer_signature,
)
from ..dispatch.common import (
    _FIXED_BLOCK_SIZE,
    _FLOATING_DATA_TYPES,
    _PPU_WARP_SIZE,
    _integer,
    _require_exact_fields,
    _require_list,
    _require_object,
)
from ..dispatch.tensor import (
    _dense_strides,
    _has_non_overlapping_strides,
    _named_port_uids,
    _padded_pointwise_values,
)


def _validate_batchnorm_graph(
    graph: dict[str, Any], operation: str
) -> dict[str, Any]:
    inference = operation == "batchnorm_inference"
    tensors = _require_list(graph["tensors"], "graph.tensors")
    nodes = _require_list(graph["nodes"], "graph.nodes")
    expected_count = 6 if inference else 10
    if len(nodes) != 1 or len(tensors) != expected_count:
        raise ValueError(
            f"THead {operation} requires one node and {expected_count} tensors"
        )
    node = _require_object(nodes[0], f"{operation} node")
    if (
        node["id"] != 0
        or node["type"] != operation
        or node["compute_data_type"] != "float32"
    ):
        raise ValueError(f"THead {operation} requires a canonical fp32 node")
    if inference:
        input_names = ("x", "mean", "inv_variance", "scale", "bias")
        output_names: tuple[str, ...] = ("y",)
    else:
        input_names = (
            "x",
            "scale",
            "bias",
            "previous_running_mean",
            "previous_running_variance",
        )
        output_names = (
            "y",
            "mean",
            "inv_variance",
            "next_running_mean",
            "next_running_variance",
        )
    input_uids = _named_port_uids(node, "inputs", input_names, operation)
    output_uids = _named_port_uids(node, "outputs", output_names, operation)
    if len(set(input_uids + output_uids)) != expected_count:
        raise ValueError(
            f"THead {operation} requires distinct external tensors"
        )
    tensors_by_uid = {int(tensor["uid"]): tensor for tensor in tensors}
    ordered = [tensors_by_uid[uid] for uid in input_uids + output_uids]
    if any(
        tensor["virtual"] or int(tensor["alignment"]) < 16
        for tensor in ordered
    ):
        raise ValueError(
            f"THead {operation} requires aligned external tensors"
        )
    data_type = tensors_by_uid[input_uids[0]]["data_type"]
    if data_type not in _FLOATING_DATA_TYPES:
        raise ValueError(f"THead {operation} data type is unsupported")
    data_uids = (
        {input_uids[0], output_uids[0]}
        if inference
        else {input_uids[0], input_uids[1], input_uids[2], output_uids[0]}
    )
    statistic_uids = set(input_uids + output_uids).difference(data_uids)
    if any(tensors_by_uid[uid]["data_type"] != data_type for uid in data_uids):
        raise ValueError(f"THead {operation} requires matching data tensors")
    if any(
        tensors_by_uid[uid]["data_type"] != "float32" for uid in statistic_uids
    ):
        raise ValueError(f"THead {operation} statistics must use float32")
    x = tensors_by_uid[input_uids[0]]
    y = tensors_by_uid[output_uids[0]]
    dimensions = [int(value) for value in x["dimensions"]]
    if len(dimensions) < 2 or list(y["dimensions"]) != dimensions:
        raise ValueError(f"THead {operation} X/Y shape is invalid")
    if any(
        not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        )
        for tensor in (x, y)
    ):
        raise ValueError(
            f"THead {operation} requires non-overlapping X/Y tensors"
        )
    if any(
        list(tensor["strides"]) != _dense_strides(tensor["dimensions"])
        for tensor in ordered
        if int(tensor["uid"]) not in {int(x["uid"]), int(y["uid"])}
    ):
        raise ValueError(
            f"THead {operation} parameters and statistics must be contiguous"
        )
    strided = any(
        list(tensor["strides"]) != _dense_strides(tensor["dimensions"])
        for tensor in (x, y)
    )
    batch = dimensions[0]
    channels = dimensions[1]
    spatial = math.prod(dimensions[2:])
    n_elements = math.prod(dimensions)
    if any(
        value < 1 or value > 2**31 - 1
        for value in (batch, channels, spatial, n_elements)
    ):
        raise ValueError(f"THead {operation} dimensions are outside int32")
    parameter_dimensions = [int(value) for value in ordered[1]["dimensions"]]
    if math.prod(parameter_dimensions) != channels:
        raise ValueError(f"THead {operation} parameter shape is invalid")
    data_uids = {input_uids[0], output_uids[0]}
    if any(
        math.prod(tensor["dimensions"]) != channels
        for tensor in ordered
        if int(tensor["uid"]) not in data_uids
    ):
        raise ValueError(f"THead {operation} statistic shape is invalid")

    attributes = _require_object(node["attributes"], f"{operation} attributes")
    integer_fields = {
        "n_elements",
        "channels",
        "spatial",
        "rank",
    }
    if not inference:
        integer_fields.add("batch")
    real_fields = set() if inference else {"epsilon", "momentum"}
    vector_fields = {"dimensions", "x_strides", "y_strides"}
    _require_exact_fields(
        attributes,
        integer_fields | real_fields | vector_fields,
        set(),
        f"{operation} attributes",
    )
    expected_integers = {
        "n_elements": n_elements,
        "channels": channels,
        "spatial": spatial,
        "rank": len(dimensions),
    }
    if not inference:
        expected_integers["batch"] = batch
    if any(
        _integer(attributes[name], f"{operation} {name}") != expected
        for name, expected in expected_integers.items()
    ):
        raise ValueError(
            f"THead {operation} integer attributes are inconsistent"
        )
    if (
        list(attributes["dimensions"]) != dimensions
        or list(attributes["x_strides"]) != list(x["strides"])
        or list(attributes["y_strides"]) != list(y["strides"])
    ):
        raise ValueError(
            f"THead {operation} layout attributes are inconsistent"
        )
    epsilon = 0.0
    momentum = 0.0
    if not inference:
        raw_epsilon = attributes["epsilon"]
        raw_momentum = attributes["momentum"]
        if (
            isinstance(raw_epsilon, bool)
            or not isinstance(raw_epsilon, (int, float))
            or not math.isfinite(raw_epsilon)
            or raw_epsilon <= 0.0
            or isinstance(raw_momentum, bool)
            or not isinstance(raw_momentum, (int, float))
            or not math.isfinite(raw_momentum)
            or not 0.0 <= raw_momentum <= 1.0
        ):
            raise ValueError(f"THead {operation} real attributes are invalid")
        epsilon = float(raw_epsilon)
        momentum = float(raw_momentum)

    if inference:
        argument_tensors = [
            tensors_by_uid[uid] for uid in input_uids + output_uids
        ]
        function = (
            "batch_norm_inference_kernel"
            if strided
            else "batch_norm_inference_nchw_kernel"
        )
    else:
        # Kernel ABI: x, y, previous stats, affine parameters, saved stats,
        # and next running stats.
        kernel_uids = [
            input_uids[0],
            output_uids[0],
            input_uids[3],
            input_uids[4],
            input_uids[1],
            input_uids[2],
            output_uids[1],
            output_uids[2],
            output_uids[3],
            output_uids[4],
        ]
        argument_tensors = [tensors_by_uid[uid] for uid in kernel_uids]
        # The NCHW specialization forms a compile-time N x spatial tile whose
        # extent is BLOCK_SIZE.  Its default qualified block is 256, so large
        # batches must use the general loop kernel even for dense layouts.
        function = (
            "batch_norm_kernel"
            if strided or batch > _FIXED_BLOCK_SIZE
            else "batch_norm_nchw_kernel"
        )
    return {
        "tensors": tensors,
        "argument_tensors": argument_tensors,
        "function": function,
        "n_elements": n_elements,
        "batch": batch,
        "channels": channels,
        "spatial": spatial,
        "epsilon": epsilon,
        "momentum": momentum,
        "stride_constants": (
            [
                *_padded_pointwise_values(dimensions, 1),
                *_padded_pointwise_values(list(x["strides"]), 0),
                *_padded_pointwise_values(list(y["strides"]), 0),
            ]
            if function in {"batch_norm_kernel", "batch_norm_inference_kernel"}
            else None
        ),
    }


def _batchnorm_variant(
    plan: dict[str, Any],
    configuration: dict[str, Any],
    variant_id: str,
) -> dict[str, Any]:
    block_size = int(configuration["META"]["BLOCK_SIZE"])
    num_warps = int(configuration["num_warps"])
    inference_nchw = plan["function"] == "batch_norm_inference_nchw_kernel"
    inference_strided = plan["function"] == "batch_norm_inference_kernel"
    training_strided = plan["function"] == "batch_norm_kernel"
    scalar_arguments: list[dict[str, Any]] = []
    if inference_nchw:
        constants: list[float | int] = [
            plan["channels"],
            plan["spatial"],
            0.0,
            block_size,
            1,
            1,
            1,
        ]
        spatial = int(plan["spatial"])
        channels = int(plan["channels"])
        channel_tile = max(1, block_size // (1 << (spatial - 1).bit_length()))
        programs = (
            int(plan["batch"])
            * ((channels + channel_tile - 1) // channel_tile)
            * ((spatial + block_size - 1) // block_size)
        )
        shared_memory = 0
    elif inference_strided:
        constants = [
            0.0,
            block_size,
            1,
            1,
            1,
            1,
            *plan["stride_constants"],
        ]
        scalar_arguments = [
            {
                "kind": "scalar_i32",
                "name": "total_elements",
                "value": int(plan["n_elements"]),
            },
            {
                "kind": "scalar_i32",
                "name": "channels",
                "value": int(plan["channels"]),
            },
            {
                "kind": "scalar_i32",
                "name": "spatial",
                "value": int(plan["spatial"]),
            },
        ]
        programs = (int(plan["n_elements"]) + block_size - 1) // block_size
        shared_memory = 0
    else:
        constants = [
            plan["batch"],
            plan["channels"],
            plan["spatial"],
            plan["epsilon"],
            plan["momentum"],
            block_size,
            1,
            1,
            1,
            1,
            1,
        ]
        programs = int(plan["channels"])
        shared_memory = num_warps * 4
        if training_strided:
            constants = [
                plan["epsilon"],
                plan["momentum"],
                block_size,
                1,
                1,
                1,
                1,
                1,
                1,
                *plan["stride_constants"],
            ]
            scalar_arguments = [
                {
                    "kind": "scalar_i32",
                    "name": "batch",
                    "value": int(plan["batch"]),
                },
                {
                    "kind": "scalar_i32",
                    "name": "channels",
                    "value": int(plan["channels"]),
                },
                {
                    "kind": "scalar_i32",
                    "name": "spatial",
                    "value": int(plan["spatial"]),
                },
            ]
    return {
        "variant_id": variant_id,
        "full_signature": (
            ",".join(
                _tensor_pointer_signature(tensor)
                for tensor in plan["argument_tensors"]
            )
            + (",i32,i32,i32" if scalar_arguments else "")
            + ","
            + ",".join(str(value) for value in constants)
        ),
        "argument_count": len(plan["argument_tensors"])
        + len(scalar_arguments),
        "arguments": [
            *[_tensor_argument(tensor) for tensor in plan["argument_tensors"]],
            *scalar_arguments,
        ],
        "compile_options": {
            "num_warps": num_warps,
            "num_stages": int(configuration["num_stages"]),
            "maxnreg": configuration["maxnreg"],
            "ppu_compiler_options": configuration["ppu_compiler_options"],
        },
        "launch": {
            "grid": [programs, 1, 1],
            "block": [num_warps * _PPU_WARP_SIZE, 1, 1],
            "shared_memory": shared_memory,
        },
    }
