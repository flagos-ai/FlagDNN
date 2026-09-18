# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""THead dispatch convolution lowering."""

from __future__ import annotations

from typing import Any
import math
from ..codegen.abi import (
    _tensor_argument,
    _tensor_pointer_signature,
)
from ..dispatch.common import (
    _POINTWISE_ATTRIBUTE_DEFAULTS,
    _BINARY_POINTWISE_MODES,
    _FLOATING_DATA_TYPES,
    _PPU_WARP_SIZE,
    _UNARY_POINTWISE_MODES,
    _integer,
    _require_exact_fields,
    _require_list,
    _require_object,
)
from ..dispatch.tensor import (
    _has_non_overlapping_strides,
    _named_port_uids,
)


def _validate_convolution_graph(
    graph: dict[str, Any], operation: str
) -> dict[str, Any]:
    tensors = _require_list(graph["tensors"], "graph.tensors")
    nodes = _require_list(graph["nodes"], "graph.nodes")
    if len(nodes) != 1 or len(tensors) != 3:
        raise ValueError(
            f"THead {operation} requires one node and three tensors"
        )
    node = _require_object(nodes[0], f"{operation} node")
    if (
        node["id"] != 0
        or node["type"] != operation
        or node["compute_data_type"] != "float32"
    ):
        raise ValueError(
            f"THead {operation} requires a canonical float32 node"
        )
    port_names = {
        "convolution_fprop": (("input", "filter"), ("output",)),
        "convolution_dgrad": (("dy", "w"), ("dx",)),
        "convolution_wgrad": (("dy", "x"), ("dw",)),
    }[operation]
    input_uids = _named_port_uids(node, "inputs", port_names[0], operation)
    output_uids = _named_port_uids(node, "outputs", port_names[1], operation)
    argument_uids = input_uids + output_uids
    if len(set(argument_uids)) != 3:
        raise ValueError(f"THead {operation} tensor UIDs must be distinct")
    tensors_by_uid = {int(tensor["uid"]): tensor for tensor in tensors}
    arguments = [tensors_by_uid[uid] for uid in argument_uids]
    data_type = arguments[0]["data_type"]
    if (
        data_type not in _FLOATING_DATA_TYPES
        or any(tensor["data_type"] != data_type for tensor in arguments)
        or any(
            tensor["virtual"] or int(tensor["alignment"]) < 16
            for tensor in arguments
        )
    ):
        raise ValueError(
            f"THead {operation} requires aligned matching floating tensors"
        )

    attributes = _require_object(node["attributes"], f"{operation} attributes")
    required_attributes = {
        "spatial_rank",
        "groups",
        "n_outputs",
        "pre_padding",
        "post_padding",
        "stride",
        "dilation",
    }
    if operation != "convolution_fprop":
        required_attributes.add("convolution_mode")
    _require_exact_fields(
        attributes,
        required_attributes,
        {"input_precision"},
        f"{operation} attributes",
    )
    if _integer(
        attributes.get("input_precision", 0), "input_precision"
    ) not in {0, 1, 2}:
        raise ValueError(
            "THead input precision must be default, IEEE, or TF32"
        )
    spatial_rank = _integer(
        attributes["spatial_rank"], f"{operation} spatial_rank"
    )
    if not 1 <= spatial_rank <= 3:
        raise ValueError(f"THead {operation} spatial_rank must be in [1, 3]")
    rank = spatial_rank + 2
    if any(
        len(tensor["dimensions"]) != rank
        or not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        )
        for tensor in arguments
    ):
        raise ValueError(
            f"THead {operation} requires rank spatial_rank + 2 "
            "non-overlapping tensors"
        )

    groups = _integer(attributes["groups"], f"{operation} groups")
    if groups <= 0:
        raise ValueError(f"THead {operation} groups must be positive")

    def spatial_attribute(name: str, minimum: int) -> list[int]:
        raw_values = _require_list(attributes[name], f"{operation} {name}")
        values = [
            _integer(value, f"{operation} {name}[{index}]")
            for index, value in enumerate(raw_values)
        ]
        if len(values) != spatial_rank or any(
            value < minimum for value in values
        ):
            raise ValueError(f"THead {operation} {name} is invalid")
        return values

    pre_padding = spatial_attribute("pre_padding", 0)
    post_padding = spatial_attribute("post_padding", 0)
    stride = spatial_attribute("stride", 1)
    dilation = spatial_attribute("dilation", 1)
    convolution_mode = 0
    if operation != "convolution_fprop":
        convolution_mode = _integer(
            attributes["convolution_mode"],
            f"{operation} convolution_mode",
        )
        if convolution_mode not in (0, 1):
            raise ValueError(f"THead {operation} convolution_mode is invalid")

    if operation == "convolution_fprop":
        image, filter_tensor, loss = arguments
    elif operation == "convolution_dgrad":
        loss, filter_tensor, image = arguments
    else:
        loss, image, filter_tensor = arguments
    image_dimensions = [int(value) for value in image["dimensions"]]
    filter_dimensions = [int(value) for value in filter_tensor["dimensions"]]
    loss_dimensions = [int(value) for value in loss["dimensions"]]
    batch, channels = image_dimensions[:2]
    output_channels, filter_channels = filter_dimensions[:2]
    if channels % groups != 0 or output_channels % groups != 0:
        raise ValueError(
            f"THead {operation} channels must be divisible by groups"
        )
    channels_per_group = channels // groups
    outputs_per_group = output_channels // groups
    if filter_channels != channels_per_group:
        raise ValueError(
            f"THead {operation} filter channels disagree with groups"
        )
    expected_loss = [batch, output_channels]
    for axis in range(spatial_rank):
        effective_filter = (
            dilation[axis] * (filter_dimensions[axis + 2] - 1) + 1
        )
        padded_image = (
            image_dimensions[axis + 2] + pre_padding[axis] + post_padding[axis]
        )
        # Spatial coordinates are computed before the load mask. Python shape
        # arithmetic can fit while a device int64 intermediate would overflow.
        coordinate_limit = 2**63 - 1
        if (
            effective_filter - 1 > coordinate_limit
            or (loss_dimensions[axis + 2] - 1) * stride[axis]
            + effective_filter
            - 1
            > coordinate_limit
            or image_dimensions[axis + 2] - 1 + pre_padding[axis]
            > coordinate_limit
        ):
            raise ValueError(
                f"THead {operation} spatial coordinates exceed int64"
            )
        if padded_image < effective_filter:
            raise ValueError(f"THead {operation} filter exceeds padded input")
        expected_loss.append(
            (padded_image - effective_filter) // stride[axis] + 1
        )
    if loss_dimensions != expected_loss:
        raise ValueError(f"THead {operation} output shape is inconsistent")
    n_outputs = _integer(attributes["n_outputs"], f"{operation} n_outputs")
    graph_output = arguments[2]
    if n_outputs != math.prod(
        int(value) for value in graph_output["dimensions"]
    ):
        raise ValueError(
            f"THead {operation} n_outputs disagrees with output shape"
        )
    if not 1 <= n_outputs <= 2**31 - 1:
        raise ValueError(
            f"THead {operation} output element count exceeds int32"
        )

    def padded(values: list[Any], fill: int) -> list[int]:
        return [fill] * (3 - spatial_rank) + [int(value) for value in values]

    image_spatial = padded(image_dimensions[2:], 1)
    filter_spatial = padded(filter_dimensions[2:], 1)
    loss_spatial = padded(loss_dimensions[2:], 1)
    spatial_stride = padded(stride, 1)
    spatial_padding = padded(pre_padding, 0)
    spatial_dilation = padded(dilation, 1)
    image_strides = padded(image["strides"][2:], 0)
    filter_strides = padded(filter_tensor["strides"][2:], 0)
    loss_strides = padded(loss["strides"][2:], 0)
    constants: dict[str, int] = {
        "XD": image_spatial[0],
        "XH": image_spatial[1],
        "XW": image_spatial[2],
        "OD": loss_spatial[0],
        "OH": loss_spatial[1],
        "OW": loss_spatial[2],
        "KD": filter_spatial[0],
        "KH": filter_spatial[1],
        "KW": filter_spatial[2],
        "CIN_PER_GROUP": channels_per_group,
        "COUT_PER_GROUP": outputs_per_group,
        "GROUPS": groups,
        "STRIDE_D": spatial_stride[0],
        "STRIDE_H": spatial_stride[1],
        "STRIDE_W": spatial_stride[2],
        "PAD_FRONT": spatial_padding[0],
        "PAD_TOP": spatial_padding[1],
        "PAD_LEFT": spatial_padding[2],
        "DIL_D": spatial_dilation[0],
        "DIL_H": spatial_dilation[1],
        "DIL_W": spatial_dilation[2],
        "FLIP_FILTER": convolution_mode,
        "DY_STRIDE_N": int(loss["strides"][0]),
        "DY_STRIDE_C": int(loss["strides"][1]),
        "DY_STRIDE_D": loss_strides[0],
        "DY_STRIDE_H": loss_strides[1],
        "DY_STRIDE_W": loss_strides[2],
        "X_STRIDE_N": int(image["strides"][0]),
        "X_STRIDE_C": int(image["strides"][1]),
        "X_STRIDE_D": image_strides[0],
        "X_STRIDE_H": image_strides[1],
        "X_STRIDE_W": image_strides[2],
        "W_STRIDE_K": int(filter_tensor["strides"][0]),
        "W_STRIDE_C": int(filter_tensor["strides"][1]),
        "W_STRIDE_D": filter_strides[0],
        "W_STRIDE_H": filter_strides[1],
        "W_STRIDE_W": filter_strides[2],
        "INPUT_PRECISION": (
            2 if attributes.get("input_precision", 0) == 2 else 1
        ),
    }
    if operation == "convolution_fprop":
        function = "conv_fprop_nd_kernel"
        constants.update(
            {
                "Y_STRIDE_N": int(loss["strides"][0]),
                "Y_STRIDE_C": int(loss["strides"][1]),
                "Y_STRIDE_D": loss_strides[0],
                "Y_STRIDE_H": loss_strides[1],
                "Y_STRIDE_W": loss_strides[2],
            }
        )
        m = batch * math.prod(loss_spatial)
        constants["M"] = m
    elif operation == "convolution_dgrad":
        function = "conv_dgrad_nd_kernel"
        m = batch * math.prod(image_spatial)
        constants["M"] = m
    else:
        function = "conv_wgrad_nd_kernel"
        m = batch * math.prod(loss_spatial)
        constants["M"] = m
    reduction_extent = 1
    if operation == "convolution_fprop":
        reduction_extent = channels_per_group * math.prod(filter_spatial)
    elif operation == "convolution_dgrad":
        reduction_extent = outputs_per_group * math.prod(filter_spatial)
    if m > 2**31 - 1 or reduction_extent > 2**31 - 1:
        raise ValueError(f"THead {operation} iteration range exceeds int32")
    return {
        "operation": operation,
        "tensors": tensors,
        "argument_tensors": arguments,
        "constants": constants,
        "function": function,
        "batch": batch,
        "groups": groups,
        "m": m,
        "channels_per_group": channels_per_group,
        "outputs_per_group": outputs_per_group,
        "kernel_volume": math.prod(filter_spatial),
        "n_outputs": n_outputs,
    }


def _validate_conv_bias_relu_graph(graph: dict[str, Any]) -> dict[str, Any]:
    tensors = _require_list(graph["tensors"], "graph.tensors")
    nodes = sorted(
        (
            _require_object(value, "ConvBiasRelu node")
            for value in _require_list(graph["nodes"], "graph.nodes")
        ),
        key=lambda value: int(value["id"]),
    )
    if len(nodes) != 3 or len(tensors) != 6:
        raise ValueError(
            "THead ConvBiasRelu requires three nodes and six tensors"
        )
    convolution, add, relu = nodes
    if (
        convolution["id"] != 0
        or convolution["type"] != "convolution_fprop"
        or add["id"] != 1
        or add["type"] != "add"
        or relu["id"] != 2
        or relu["type"] != "relu"
        or any(node["compute_data_type"] != "float32" for node in nodes)
    ):
        raise ValueError(
            "THead ConvBiasRelu requires canonical float32 Conv-Add-ReLU nodes"
        )

    convolution_inputs = _named_port_uids(
        convolution, "inputs", ("input", "filter"), "ConvBiasRelu convolution"
    )
    convolution_output = _named_port_uids(
        convolution, "outputs", ("output",), "ConvBiasRelu convolution"
    )[0]
    add_inputs = _named_port_uids(
        add, "inputs", ("left", "right"), "ConvBiasRelu bias Add"
    )
    add_output = _named_port_uids(
        add, "outputs", ("output",), "ConvBiasRelu bias Add"
    )[0]
    relu_input = _named_port_uids(
        relu, "inputs", ("input",), "ConvBiasRelu ReLU"
    )[0]
    output_uid = _named_port_uids(
        relu, "outputs", ("output",), "ConvBiasRelu ReLU"
    )[0]
    if add_inputs[0] != convolution_output or relu_input != add_output:
        raise ValueError("THead ConvBiasRelu dataflow is not Conv-Add-ReLU")
    x_uid, w_uid = convolution_inputs
    bias_uid = add_inputs[1]
    if (
        len(
            {
                x_uid,
                w_uid,
                bias_uid,
                convolution_output,
                add_output,
                output_uid,
            }
        )
        != 6
    ):
        raise ValueError("THead ConvBiasRelu tensor roles must be distinct")

    tensors_by_uid = {int(tensor["uid"]): tensor for tensor in tensors}
    x = tensors_by_uid[x_uid]
    w = tensors_by_uid[w_uid]
    bias = tensors_by_uid[bias_uid]
    convolution_tensor = tensors_by_uid[convolution_output]
    biased_tensor = tensors_by_uid[add_output]
    output = tensors_by_uid[output_uid]
    role_tensors = [x, w, bias, convolution_tensor, biased_tensor, output]
    data_type = x["data_type"]
    if (
        data_type not in _FLOATING_DATA_TYPES
        or any(tensor["data_type"] != data_type for tensor in role_tensors)
        or any(int(tensor["alignment"]) < 16 for tensor in role_tensors)
    ):
        raise ValueError(
            "THead ConvBiasRelu requires aligned matching floating tensors"
        )
    if (
        x["virtual"]
        or w["virtual"]
        or bias["virtual"]
        or not convolution_tensor["virtual"]
        or not biased_tensor["virtual"]
        or output["virtual"]
    ):
        raise ValueError(
            "THead ConvBiasRelu requires two virtual intermediates"
        )
    if any(len(tensor["dimensions"]) != 4 for tensor in role_tensors):
        raise ValueError("THead ConvBiasRelu requires rank-four tensors")
    if any(
        tensor["dimensions"] != output["dimensions"]
        or tensor["strides"] != output["strides"]
        for tensor in (convolution_tensor, biased_tensor)
    ):
        raise ValueError(
            "THead ConvBiasRelu intermediate/output geometry must match"
        )
    output_dimensions = [int(value) for value in output["dimensions"]]
    if [int(value) for value in bias["dimensions"]] != [
        1,
        output_dimensions[1],
        1,
        1,
    ]:
        raise ValueError("THead ConvBiasRelu requires a channel bias")

    def channels_last_strides(dimensions: list[Any]) -> list[int]:
        n, channels, height, width = [int(value) for value in dimensions]
        del n
        return [channels * height * width, 1, width * channels, channels]

    if any(
        [int(value) for value in tensor["strides"]]
        != channels_last_strides(tensor["dimensions"])
        for tensor in role_tensors
    ):
        raise ValueError("THead ConvBiasRelu requires channels-last strides")

    add_attributes = _require_object(
        add["attributes"], "ConvBiasRelu Add attributes"
    )
    _require_exact_fields(
        add_attributes,
        {"alpha", "mode", "n_elements", "pointwise_mode"},
        set(_POINTWISE_ATTRIBUTE_DEFAULTS),
        "ConvBiasRelu Add attributes",
    )
    n_outputs = math.prod(output_dimensions)
    if (
        add_attributes["alpha"] != 1.0
        or add_attributes["mode"] != _BINARY_POINTWISE_MODES["add"]
        or add_attributes["pointwise_mode"] != _BINARY_POINTWISE_MODES["add"]
        or add_attributes["n_elements"] != n_outputs
    ):
        raise ValueError("THead ConvBiasRelu Add attributes are invalid")

    relu_attributes = _require_object(
        relu["attributes"], "ConvBiasRelu ReLU attributes"
    )
    expected_relu_attributes = {
        "elu_alpha": 1.0,
        "has_upper_clip": 0,
        "lower_clip": 0.0,
        "mode": _UNARY_POINTWISE_MODES["relu"],
        "n_elements": n_outputs,
        "negative_slope": 0.0,
        "relu_lower_clip": 0.0,
        "relu_lower_clip_slope": 0.0,
        "relu_upper_clip": 0.0,
        "relu_upper_clip_set": False,
        "softplus_beta": 1.0,
        "swish_beta": 1.0,
        "upper_clip": 0.0,
    }
    if relu_attributes != expected_relu_attributes:
        raise ValueError("THead ConvBiasRelu requires default ReLU attributes")

    synthetic_convolution = dict(convolution)
    synthetic_convolution["outputs"] = [{"name": "output", "uid": output_uid}]
    synthetic_tensors = []
    for tensor in (x, w, output):
        external = dict(tensor)
        external["virtual"] = False
        synthetic_tensors.append(external)
    plan = _validate_convolution_graph(
        {"tensors": synthetic_tensors, "nodes": [synthetic_convolution]},
        "convolution_fprop",
    )
    convolution_attributes = convolution["attributes"]
    if (
        convolution_attributes["groups"] != 1
        or convolution_attributes["pre_padding"]
        != convolution_attributes["post_padding"]
    ):
        raise ValueError(
            "THead ConvBiasRelu requires group one and symmetric padding"
        )
    plan.update(
        {
            "operation": "conv_bias_relu",
            "tensors": tensors,
            "argument_tensors": [x, w, bias, output],
            "virtual_tensors": [convolution_tensor, biased_tensor],
            "constants": {
                **plan["constants"],
                "BIAS_STRIDE_C": int(bias["strides"][1]),
            },
            "function": "conv2d_bias_relu_kernel",
            "source_node_ids": [0, 1, 2],
            "n_outputs": n_outputs,
        }
    )
    return plan


def _direct_dgrad(constants: dict[str, Any], meta: dict[str, Any]) -> bool:
    return (
        constants["CIN_PER_GROUP"] <= 8
        and constants["COUT_PER_GROUP"] >= 16
        and constants["M"] >= 4096
        and meta["BLOCK_M"] == 128
        and meta["BLOCK_K"] == 16
        and constants["XD"] == 1
        and constants["OD"] == 1
        and constants["KD"] == 1
        and constants["KH"] == 3
        and constants["KW"] == 3
        and constants["STRIDE_H"] == 2
        and constants["STRIDE_W"] == 2
        and constants["DIL_H"] == 1
        and constants["DIL_W"] == 1
        and constants["XH"] % 2 == 0
        and constants["XW"] % 2 == 0
        and constants["X_STRIDE_N"]
        == constants["CIN_PER_GROUP"] * constants["XH"] * constants["XW"]
        and constants["X_STRIDE_C"] == constants["XH"] * constants["XW"]
        and constants["X_STRIDE_H"] == constants["XW"]
        and constants["X_STRIDE_W"] == 1
        and constants["DY_STRIDE_N"]
        == constants["COUT_PER_GROUP"] * constants["OH"] * constants["OW"]
        and constants["DY_STRIDE_C"] == constants["OH"] * constants["OW"]
        and constants["DY_STRIDE_H"] == constants["OW"]
        and constants["DY_STRIDE_W"] == 1
        and constants["W_STRIDE_K"] == constants["CIN_PER_GROUP"] * 9
        and constants["W_STRIDE_C"] == 9
        and constants["W_STRIDE_H"] == 3
        and constants["W_STRIDE_W"] == 1
    )


def _convolution_variant(
    plan: dict[str, Any],
    configuration: dict[str, Any],
    variant_id: str,
) -> dict[str, Any]:
    constants = plan["constants"]
    meta = configuration["META"]
    function = plan["function"]
    operand_bytes = (
        4 if plan["argument_tensors"][0]["data_type"] == "float32" else 2
    )
    if function == "conv_fprop_nd_kernel":
        block_m = int(meta["BLOCK_M"])
        block_n = int(meta["BLOCK_OC"])
        block_k = int(meta["BLOCK_K"])
        ordered_constants = [
            *[
                constants[name]
                for name in (
                    "XD",
                    "XH",
                    "XW",
                    "OD",
                    "OH",
                    "OW",
                    "KD",
                    "KH",
                    "KW",
                    "CIN_PER_GROUP",
                    "COUT_PER_GROUP",
                    "GROUPS",
                    "STRIDE_D",
                    "STRIDE_H",
                    "STRIDE_W",
                    "PAD_FRONT",
                    "PAD_TOP",
                    "PAD_LEFT",
                    "DIL_D",
                    "DIL_H",
                    "DIL_W",
                    "X_STRIDE_N",
                    "X_STRIDE_C",
                    "X_STRIDE_D",
                    "X_STRIDE_H",
                    "X_STRIDE_W",
                    "W_STRIDE_K",
                    "W_STRIDE_C",
                    "W_STRIDE_D",
                    "W_STRIDE_H",
                    "W_STRIDE_W",
                    "Y_STRIDE_N",
                    "Y_STRIDE_C",
                    "Y_STRIDE_D",
                    "Y_STRIDE_H",
                    "Y_STRIDE_W",
                    "INPUT_PRECISION",
                    "M",
                )
            ],
            block_m,
            block_n,
            block_k,
        ]
        grid = [
            ((plan["m"] + block_m - 1) // block_m)
            * ((plan["outputs_per_group"] + block_n - 1) // block_n),
            plan["groups"],
            1,
        ]
        shared_memory = (block_m * block_k + block_k * block_n) * operand_bytes
    elif function == "conv2d_bias_relu_kernel":
        block_m = int(meta["BLOCK_HW"])
        block_n = int(meta["BLOCK_OC"])
        block_k = int(meta["BLOCK_K"])
        ordered_constants = [
            constants["XH"],
            constants["XW"],
            constants["OH"],
            constants["OW"],
            constants["CIN_PER_GROUP"],
            constants["COUT_PER_GROUP"],
            constants["GROUPS"],
            constants["STRIDE_H"],
            constants["STRIDE_W"],
            constants["PAD_TOP"],
            constants["PAD_LEFT"],
            constants["DIL_H"],
            constants["DIL_W"],
            constants["KH"],
            constants["KW"],
            constants["X_STRIDE_N"],
            constants["X_STRIDE_C"],
            constants["X_STRIDE_H"],
            constants["X_STRIDE_W"],
            constants["W_STRIDE_K"],
            constants["W_STRIDE_C"],
            constants["W_STRIDE_H"],
            constants["W_STRIDE_W"],
            constants["BIAS_STRIDE_C"],
            constants["Y_STRIDE_N"],
            constants["Y_STRIDE_C"],
            constants["Y_STRIDE_H"],
            constants["Y_STRIDE_W"],
            block_n,
            block_m,
            block_k,
            constants["INPUT_PRECISION"],
        ]
        grid = [
            ((constants["OH"] * constants["OW"] + block_m - 1) // block_m)
            * ((plan["outputs_per_group"] + block_n - 1) // block_n),
            plan["batch"] * plan["groups"],
            1,
        ]
        shared_memory = (block_m * block_k + block_k * block_n) * operand_bytes
    elif function == "conv_dgrad_nd_kernel":
        block_m = int(meta["BLOCK_M"])
        block_n = int(meta["BLOCK_CI"])
        block_k = int(meta["BLOCK_K"])
        ordered_constants = [
            *[
                constants[name]
                for name in (
                    "XD",
                    "XH",
                    "XW",
                    "OD",
                    "OH",
                    "OW",
                    "KD",
                    "KH",
                    "KW",
                    "CIN_PER_GROUP",
                    "COUT_PER_GROUP",
                    "STRIDE_D",
                    "STRIDE_H",
                    "STRIDE_W",
                    "PAD_FRONT",
                    "PAD_TOP",
                    "PAD_LEFT",
                    "DIL_D",
                    "DIL_H",
                    "DIL_W",
                    "FLIP_FILTER",
                    "DY_STRIDE_N",
                    "DY_STRIDE_C",
                    "DY_STRIDE_D",
                    "DY_STRIDE_H",
                    "DY_STRIDE_W",
                    "X_STRIDE_N",
                    "X_STRIDE_C",
                    "X_STRIDE_D",
                    "X_STRIDE_H",
                    "X_STRIDE_W",
                    "W_STRIDE_K",
                    "W_STRIDE_C",
                    "W_STRIDE_D",
                    "W_STRIDE_H",
                    "W_STRIDE_W",
                    "INPUT_PRECISION",
                    "M",
                )
            ],
            block_m,
            block_n,
            block_k,
            8,
        ]
        grid = [
            ((plan["m"] + block_m - 1) // block_m)
            * ((plan["channels_per_group"] + block_n - 1) // block_n),
            plan["groups"],
            1,
        ]
        shared_memory = (block_m * block_k + block_k * block_n) * operand_bytes
    else:
        block_m = int(meta["BLOCK_M"])
        block_n = int(meta["BLOCK_OC"])
        block_k = int(meta["BLOCK_CI"])
        ordered_constants = [
            *[
                constants[name]
                for name in (
                    "XD",
                    "XH",
                    "XW",
                    "OD",
                    "OH",
                    "OW",
                    "KD",
                    "KH",
                    "KW",
                    "CIN_PER_GROUP",
                    "COUT_PER_GROUP",
                    "STRIDE_D",
                    "STRIDE_H",
                    "STRIDE_W",
                    "PAD_FRONT",
                    "PAD_TOP",
                    "PAD_LEFT",
                    "DIL_D",
                    "DIL_H",
                    "DIL_W",
                    "FLIP_FILTER",
                    "DY_STRIDE_N",
                    "DY_STRIDE_C",
                    "DY_STRIDE_D",
                    "DY_STRIDE_H",
                    "DY_STRIDE_W",
                    "X_STRIDE_N",
                    "X_STRIDE_C",
                    "X_STRIDE_D",
                    "X_STRIDE_H",
                    "X_STRIDE_W",
                    "W_STRIDE_K",
                    "W_STRIDE_C",
                    "W_STRIDE_D",
                    "W_STRIDE_H",
                    "W_STRIDE_W",
                    "INPUT_PRECISION",
                    "M",
                )
            ],
            block_n,
            block_k,
            block_m,
        ]
        grid = [
            ((plan["outputs_per_group"] + block_n - 1) // block_n)
            * ((plan["channels_per_group"] + block_k - 1) // block_k),
            plan["kernel_volume"],
            plan["groups"],
        ]
        # WGrad multiplies [BLOCK_OC, BLOCK_M] by
        # [BLOCK_M, BLOCK_CI].  Keep the launch ABI tied to those actual
        # operand tiles instead of assuming all tuning dimensions are equal.
        shared_memory = (block_n * block_m + block_m * block_k) * operand_bytes
    if function == "conv_dgrad_nd_kernel" and _direct_dgrad(constants, meta):
        grid = [
            4 * ((plan["m"] // 4 + block_m - 1) // block_m),
            plan["channels_per_group"],
            plan["groups"],
        ]
        shared_memory = block_m * operand_bytes
    num_warps = int(configuration["num_warps"])
    if (
        function == "conv_wgrad_nd_kernel"
        and constants["CIN_PER_GROUP"] <= 8
        and constants["M"] >= 4096
        and block_m >= 256
    ):
        grid = [
            plan["outputs_per_group"]
            * plan["channels_per_group"]
            * plan["kernel_volume"],
            plan["groups"],
            1,
        ]
        shared_memory = num_warps * 4
    return {
        "variant_id": variant_id,
        "full_signature": (
            ",".join(
                _tensor_pointer_signature(tensor)
                for tensor in plan["argument_tensors"]
            )
            + ","
            + ",".join(str(value) for value in ordered_constants)
        ),
        "argument_count": len(plan["argument_tensors"]),
        "arguments": [
            _tensor_argument(tensor) for tensor in plan["argument_tensors"]
        ],
        "compile_options": {
            "num_warps": num_warps,
            "num_stages": int(configuration["num_stages"]),
            "maxnreg": configuration["maxnreg"],
            "ppu_compiler_options": configuration["ppu_compiler_options"],
        },
        "launch": {
            "grid": grid,
            "block": [num_warps * _PPU_WARP_SIZE, 1, 1],
            "shared_memory": shared_memory,
        },
    }
