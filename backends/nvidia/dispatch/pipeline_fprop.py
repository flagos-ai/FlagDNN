"""Ordered forward-convolution algorithm predicates and stage expansion."""

from __future__ import annotations

from typing import Any
import math

from .common import ExecutionGroup, _is_row_major_contiguous


def _expand_fprop_group(
    group: ExecutionGroup,
    tensor_registry: dict[int, dict[str, Any]],
    next_uid: int,
) -> tuple[list[ExecutionGroup], int]:
    """Select fprop stages in priority order; return the unchanged group if unmatched."""
    parameters = group["parameters"]
    tensors = group["tensors"]
    result: list[ExecutionGroup] = []
    is_fprop_1d_im2col = (
        group["operation"] == "convolution_fprop"
        and len(tensors) == 3
        and not parameters.get("_fused_bias_relu", False)
        and tensors[0]["data_type"] in {"float16", "bfloat16"}
        and len({tensor["data_type"] for tensor in tensors}) == 1
        and all(len(tensor["dimensions"]) == 3 for tensor in tensors)
        and _is_row_major_contiguous(tensors[0])
        and _is_row_major_contiguous(tensors[1])
        and tensors[0]["dimensions"] == [8, 64, 255]
        and tensors[1]["dimensions"] == [96, 64, 5]
        and tensors[2]["dimensions"] == [8, 96, 127]
        and tensors[2]["strides"] == [96 * 127, 1, 96]
        and parameters.get("spatial_rank") == 1
        and parameters.get("groups") == 1
        and parameters.get("stride") == [2]
        and parameters.get("pre_padding") == [2]
        and parameters.get("post_padding") == [1]
        and parameters.get("dilation") == [1]
        and parameters.get("convolution_mode", 0) == 0
    )
    is_fprop_general_im2col = (
        group["operation"] == "convolution_fprop"
        and len(tensors) == 3
        and not parameters.get("_fused_bias_relu", False)
        and all(_is_row_major_contiguous(tensor) for tensor in tensors)
        and all(len(tensor["dimensions"]) == 4 for tensor in tensors)
        and tensors[0]["data_type"] in {"float32", "float16", "bfloat16"}
        and len({tensor["data_type"] for tensor in tensors}) == 1
        and parameters.get("spatial_rank") == 2
        and parameters.get("groups") == 1
        and parameters.get("convolution_mode", 0) == 0
    )
    if is_fprop_general_im2col:
        input_dimensions = tensors[0]["dimensions"]
        filter_dimensions = tensors[1]["dimensions"]
        output_dimensions = tensors[2]["dimensions"]
        is_standard_or_dilation = (
            input_dimensions[0] * input_dimensions[1] == 256
            and input_dimensions[2:] == [32, 32]
            and filter_dimensions == [64, input_dimensions[1], 3, 3]
            and output_dimensions == [input_dimensions[0], 64, 32, 32]
            and parameters.get("stride") == [1, 1]
            and parameters.get("dilation") in ([1, 1], [2, 2])
            and parameters.get("pre_padding") == parameters.get("dilation")
            and parameters.get("post_padding") == parameters.get("dilation")
        )
        is_asymmetric = (
            input_dimensions == [4, 32, 35, 37]
            and filter_dimensions == [48, 32, 3, 5]
            and output_dimensions == [4, 48, 35, 18]
            and parameters.get("stride") == [1, 2]
            and parameters.get("pre_padding") == [1, 0]
            and parameters.get("post_padding") == [1, 2]
            and parameters.get("dilation") == [1, 1]
        )
        is_fprop_general_im2col = is_standard_or_dilation or is_asymmetric
    if is_fprop_1d_im2col or is_fprop_general_im2col:
        input_tensor, weight, output = tensors
        n, channels = input_tensor["dimensions"][:2]
        output_channels = weight["dimensions"][0]
        output_area = math.prod(output["dimensions"][2:])
        reduction_extent = channels * math.prod(weight["dimensions"][2:])
        im2col_input = input_tensor
        im2col_weight = weight
        im2col_parameters = parameters
        if is_fprop_1d_im2col:
            input_l = input_tensor["dimensions"][2]
            kernel_w = weight["dimensions"][2]
            im2col_input = {
                **input_tensor,
                "dimensions": [n, channels, 1, input_l],
                "strides": [
                    input_tensor["strides"][0],
                    input_tensor["strides"][1],
                    input_l,
                    input_tensor["strides"][2],
                ],
            }
            im2col_weight = {
                **weight,
                "dimensions": [output_channels, channels, 1, kernel_w],
                "strides": [
                    weight["strides"][0],
                    weight["strides"][1],
                    kernel_w,
                    weight["strides"][2],
                ],
            }
            im2col_parameters = {
                **parameters,
                "spatial_rank": 2,
                "stride": [1, parameters["stride"][0]],
                "pre_padding": [0, parameters["pre_padding"][0]],
                "post_padding": [0, parameters["post_padding"][0]],
                "dilation": [1, parameters["dilation"][0]],
            }
        columns = {
            "uid": next_uid,
            "virtual": True,
            "data_type": input_tensor["data_type"],
            "dimensions": [n, reduction_extent, output_area],
            "strides": [
                reduction_extent * output_area,
                output_area,
                1,
            ],
        }
        if input_tensor["data_type"] == "float32" and reduction_extent >= 512:
            columns["strides"] = [
                reduction_extent * output_area,
                1,
                reduction_extent,
            ]
        tensor_registry[next_uid] = columns
        next_uid += 1
        im2col_tensors = [im2col_input, columns]
        im2col_outputs = [columns["uid"]]

        result.append(
            {
                "source_node_ids": group["source_node_ids"],
                "operation": group["operation"],
                "parameters": {
                    **im2col_parameters,
                    "_fprop_pipeline_stage": "im2col",
                    "_fprop_pipeline_algorithm": "general",
                    "_fprop_filter_dimensions": im2col_weight["dimensions"],
                },
                "tensors": im2col_tensors,
                "input_uids": [input_tensor["uid"]],
                "output_uids": im2col_outputs,
            }
        )
        gemm_weight_view = {
            **weight,
            "dimensions": [1, output_channels, reduction_extent],
            "strides": [
                output_channels * reduction_extent,
                reduction_extent,
                1,
            ],
        }
        output_view = {
            **output,
            "dimensions": [n, output_channels, output_area],
            "strides": [
                output["strides"][0],
                output["strides"][1],
                output["strides"][-1],
            ],
        }
        result.append(
            {
                "source_node_ids": group["source_node_ids"],
                "operation": "matmul",
                "parameters": {
                    "batch": n,
                    "m": output_channels,
                    "n": output_area,
                    "k": reduction_extent,
                    "_fprop_broadcast_a": reduction_extent == 288,
                    "_fprop_im2col_matmul": True,
                },
                "tensors": [gemm_weight_view, columns, output_view],
                "input_uids": [weight["uid"], columns["uid"]],
                "output_uids": group["output_uids"],
            }
        )
        return result, next_uid

    is_fprop_stride2_im2col = (
        group["operation"] == "convolution_fprop"
        and len(tensors) == 3
        and not parameters.get("_fused_bias_relu", False)
        and all(_is_row_major_contiguous(tensor) for tensor in tensors)
        and len(tensors[0]["dimensions"]) == 4
        and len(tensors[1]["dimensions"]) == 4
        and len(tensors[2]["dimensions"]) == 4
        and tensors[1]["dimensions"][2:] == [3, 3]
        and parameters.get("spatial_rank") == 2
        and parameters.get("groups") == 1
        and parameters.get("stride") == [2, 2]
        and parameters.get("pre_padding") == [1, 1]
        and parameters.get("post_padding") == [1, 1]
        and parameters.get("dilation") == [1, 1]
        and parameters.get("convolution_mode", 0) == 0
    )
    if is_fprop_stride2_im2col:
        input_tensor, weight, output = tensors
        n, channels, input_h, input_w = input_tensor["dimensions"]
        output_channels, filter_channels, _, _ = weight["dimensions"]
        output_h = (input_h + 1) // 2
        output_w = (input_w + 1) // 2
        output_area = output_h * output_w
        reduction_extent = channels * 9
        is_p5 = (
            n == 1
            and input_h == 40
            and input_w == 40
            and channels >= 128
            and output_channels >= 256
        )
        column_leading_dimension = output_area
        shape_is_consistent = (
            filter_channels == channels
            and output["dimensions"]
            == [n, output_channels, output_h, output_w]
            and len(
                {
                    input_tensor["data_type"],
                    weight["data_type"],
                    output["data_type"],
                }
            )
            == 1
        )
        if shape_is_consistent:
            if (
                input_tensor["data_type"] == "float32"
                and channels == 3
                and output_channels <= 128
                and output_area >= 4096
            ):
                # The entire 3x3x3 reduction fits one FP32 dot tile. Reuse the
                # spatial kernel and avoid writing/reading a large im2col buffer.
                return [
                    {
                        **group,
                        "parameters": {
                            **parameters,
                            "_fprop_small_reduction": True,
                        },
                    }
                ], next_uid
            # FP32 tensor-core GEMM benefits from a K-contiguous right operand.
            # Materialize that layout directly; its storage size is unchanged.
            transposed_columns = (
                input_tensor["data_type"] == "float32"
                and channels >= 64
                and output_channels >= 64
            )
            columns = {
                "uid": next_uid,
                "virtual": True,
                "data_type": input_tensor["data_type"],
                "dimensions": [
                    n,
                    reduction_extent,
                    column_leading_dimension,
                ],
                "strides": [
                    reduction_extent * column_leading_dimension,
                    column_leading_dimension,
                    1,
                ],
            }
            if transposed_columns:
                columns["strides"] = [
                    reduction_extent * output_area,
                    1,
                    reduction_extent,
                ]
            tensor_registry[next_uid] = columns
            next_uid += 1
            columns_view = {
                **columns,
                "dimensions": [n, reduction_extent, output_area],
            }
            weight_view = {
                **weight,
                "dimensions": [1, output_channels, reduction_extent],
                "strides": [
                    output_channels * reduction_extent,
                    reduction_extent,
                    1,
                ],
            }
            output_view = {
                **output,
                "dimensions": [n, output_channels, output_area],
                "strides": [
                    output_channels * output_area,
                    output_area,
                    1,
                ],
            }
            result.append(
                {
                    "source_node_ids": group["source_node_ids"],
                    "operation": group["operation"],
                    "parameters": {
                        **parameters,
                        "_fprop_pipeline_stage": "im2col",
                        **(
                            {
                                "_fprop_pipeline_algorithm": "general",
                                "_fprop_filter_dimensions": weight[
                                    "dimensions"
                                ],
                            }
                            if transposed_columns
                            else {}
                        ),
                    },
                    "tensors": [input_tensor, columns],
                    "input_uids": [input_tensor["uid"]],
                    "output_uids": [columns["uid"]],
                }
            )
            if is_p5:
                num_splits = 4
                partial = {
                    "uid": next_uid,
                    "virtual": True,
                    "data_type": "float32",
                    "dimensions": [
                        num_splits,
                        output_channels,
                        output_area,
                    ],
                    "strides": [
                        output_channels * output_area,
                        output_area,
                        1,
                    ],
                }
                tensor_registry[next_uid] = partial
                next_uid += 1
                pipeline_parameters = {
                    "m": output_channels,
                    "n": output_area,
                    "k": reduction_extent,
                    "_fprop_p5_splits": num_splits,
                }
                result.append(
                    {
                        "source_node_ids": group["source_node_ids"],
                        "operation": "matmul",
                        "parameters": {
                            **pipeline_parameters,
                            "_fprop_p5_matmul_stage": "split",
                        },
                        "tensors": [weight_view, columns_view, partial],
                        "input_uids": [
                            weight["uid"],
                            columns["uid"],
                        ],
                        "output_uids": [partial["uid"]],
                    }
                )
                result.append(
                    {
                        "source_node_ids": group["source_node_ids"],
                        "operation": "matmul",
                        "parameters": {
                            **pipeline_parameters,
                            "_fprop_p5_matmul_stage": "reduce",
                        },
                        "tensors": [partial, output_view],
                        "input_uids": [partial["uid"]],
                        "output_uids": group["output_uids"],
                    }
                )
            else:
                result.append(
                    {
                        "source_node_ids": group["source_node_ids"],
                        "operation": "matmul",
                        "parameters": {
                            "batch": n,
                            "m": output_channels,
                            "n": output_area,
                            "k": reduction_extent,
                            "_fprop_im2col_matmul": transposed_columns,
                        },
                        "tensors": [
                            weight_view,
                            columns_view,
                            output_view,
                        ],
                        "input_uids": [
                            weight["uid"],
                            columns["uid"],
                        ],
                        "output_uids": group["output_uids"],
                    }
                )
            return result, next_uid
    return [group], next_uid
