"""Ordered data-gradient algorithm predicates and stage expansion."""

from __future__ import annotations

from typing import Any

from .common import ExecutionGroup, _is_row_major_contiguous
from .layout import _transpose_matrix_stage


def _expand_dgrad_group(
    group: ExecutionGroup,
    tensor_registry: dict[int, dict[str, Any]],
    next_uid: int,
) -> tuple[list[ExecutionGroup], int]:
    """Select dgrad stages in priority order; return the unchanged group if unmatched."""
    parameters = group["parameters"]
    tensors = group["tensors"]
    result: list[ExecutionGroup] = []
    is_fp32_3d_ci8_dot = (
        group["operation"] == "convolution_dgrad"
        and len(tensors) == 3
        and all(tensor["data_type"] == "float32" for tensor in tensors)
        and tensors[0]["dimensions"] == [2, 16, 8, 16, 16]
        and tensors[1]["dimensions"] == [16, 8, 3, 3, 3]
        and tensors[2]["dimensions"] == [2, 8, 8, 16, 16]
        and parameters.get("spatial_rank") == 3
        and parameters.get("groups") == 1
        and parameters.get("stride") == [1, 1, 1]
        and parameters.get("pre_padding") == [1, 1, 1]
        and parameters.get("post_padding") == [1, 1, 1]
        and parameters.get("dilation") == [1, 1, 1]
        and parameters.get("convolution_mode", 0) == 0
    )
    is_lowp_3d_packed_shape = (
        group["operation"] == "convolution_dgrad"
        and len(tensors) == 3
        and tensors[0]["data_type"] in {"float16", "bfloat16"}
        and len({tensor["data_type"] for tensor in tensors}) == 1
        and tensors[0]["dimensions"] == [2, 16, 8, 16, 16]
        and tensors[1]["dimensions"] == [16, 8, 3, 3, 3]
        and tensors[2]["dimensions"] == [2, 8, 8, 16, 16]
        and parameters.get("spatial_rank") == 3
        and parameters.get("groups") == 1
        and parameters.get("stride") == [1, 1, 1]
        and parameters.get("pre_padding") == [1, 1, 1]
        and parameters.get("post_padding") == [1, 1, 1]
        and parameters.get("dilation") == [1, 1, 1]
        and parameters.get("convolution_mode", 0) == 0
    )
    is_packed_dgrad_3d = (
        group["operation"] == "convolution_dgrad"
        and len(tensors) == 3
        and all(_is_row_major_contiguous(tensor) for tensor in tensors)
        and all(len(tensor["dimensions"]) == 5 for tensor in tensors)
        and len({tensor["data_type"] for tensor in tensors}) == 1
        and (is_lowp_3d_packed_shape or is_fp32_3d_ci8_dot)
        and tensors[1]["dimensions"][0] == tensors[0]["dimensions"][1]
        and tensors[1]["dimensions"][1] == tensors[2]["dimensions"][1]
        and tensors[0]["dimensions"][0] == tensors[2]["dimensions"][0]
        and parameters.get("spatial_rank") == 3
        and parameters.get("groups") == 1
        and parameters.get("convolution_mode", 0) == 0
    )
    if is_packed_dgrad_3d:
        loss, weight, output = tensors
        c_out, c_in, kernel_d, kernel_h, kernel_w = weight["dimensions"]
        packed = {
            "uid": next_uid,
            "virtual": True,
            "data_type": weight["data_type"],
            "dimensions": [
                kernel_d,
                kernel_h,
                kernel_w,
                c_out,
                c_in,
            ],
            "strides": [
                kernel_h * kernel_w * c_out * c_in,
                kernel_w * c_out * c_in,
                c_out * c_in,
                c_in,
                1,
            ],
        }
        tensor_registry[next_uid] = packed
        next_uid += 1
        result.append(
            {
                "source_node_ids": group["source_node_ids"],
                "operation": group["operation"],
                "parameters": {
                    **parameters,
                    "_dgrad_3d_pipeline_stage": "pack",
                },
                "tensors": [weight, packed],
                "input_uids": [weight["uid"]],
                "output_uids": [packed["uid"]],
            }
        )
        result.append(
            {
                "source_node_ids": group["source_node_ids"],
                "operation": group["operation"],
                "parameters": {
                    **parameters,
                    "_dgrad_3d_pipeline_stage": (
                        "compute_ci8_dot"
                        if is_fp32_3d_ci8_dot
                        else "compute_packed"
                    ),
                },
                "tensors": [loss, packed, output],
                "input_uids": [loss["uid"], packed["uid"]],
                "output_uids": group["output_uids"],
            }
        )
        return result, next_uid

    is_fp32_p5_tile4 = (
        group["operation"] == "convolution_dgrad"
        and len(tensors) == 3
        and all(tensor["data_type"] == "float32" for tensor in tensors)
        and all(len(tensor["dimensions"]) == 4 for tensor in tensors)
        and tensors[0]["dimensions"][0] == 1
        and tensors[0]["dimensions"][2:] == [20, 20]
        and tensors[1]["dimensions"][0] == tensors[0]["dimensions"][1]
        and tensors[1]["dimensions"][0] >= 512
        and tensors[1]["dimensions"][1] >= 256
        and tensors[1]["dimensions"][2:] == [3, 3]
        and tensors[2]["dimensions"]
        == [1, tensors[1]["dimensions"][1], 40, 40]
        and parameters.get("spatial_rank") == 2
        and parameters.get("groups") == 1
        and parameters.get("stride") == [2, 2]
        and parameters.get("pre_padding") == [1, 1]
        and parameters.get("post_padding") == [1, 1]
        and parameters.get("dilation") == [1, 1]
    )
    is_exact_p5_768 = (
        group["operation"] == "convolution_dgrad"
        and len(tensors) == 3
        and tensors[0]["dimensions"] == [1, 768, 20, 20]
        and tensors[1]["dimensions"] == [768, 768, 3, 3]
        and tensors[2]["dimensions"] == [1, 768, 40, 40]
        and parameters.get("spatial_rank") == 2
        and parameters.get("groups") == 1
        and parameters.get("stride") == [2, 2]
        and parameters.get("pre_padding") == [1, 1]
        and parameters.get("post_padding") == [1, 1]
        and parameters.get("dilation") == [1, 1]
        and parameters.get("convolution_mode") == 0
    )
    is_packed_dgrad_stride2 = (
        group["operation"] == "convolution_dgrad"
        and len(tensors) == 3
        and all(_is_row_major_contiguous(tensor) for tensor in tensors)
        and len(tensors[0]["dimensions"]) == 4
        and len(tensors[1]["dimensions"]) == 4
        and len(tensors[2]["dimensions"]) == 4
        and len({tensor["data_type"] for tensor in tensors}) == 1
        and tensors[0]["data_type"] in {"float32", "float16", "bfloat16"}
        and not (
            tensors[0]["data_type"] == "float32"
            and tensors[1]["dimensions"][0] >= 512
            and not is_fp32_p5_tile4
        )
        and tensors[1]["dimensions"][2:] == [3, 3]
        and parameters.get("spatial_rank") == 2
        and parameters.get("groups") == 1
        and parameters.get("stride") == [2, 2]
        and parameters.get("pre_padding") == [1, 1]
        and parameters.get("post_padding") == [1, 1]
        and parameters.get("dilation") == [1, 1]
        and parameters.get("convolution_mode") in {0, 1}
    )
    if is_packed_dgrad_stride2:
        loss, weight, output = tensors
        c_out, c_in, _, _ = weight["dimensions"]
        packed = {
            "uid": next_uid,
            "virtual": True,
            "data_type": weight["data_type"],
            "dimensions": [3, 3, c_out, c_in],
            "strides": [
                3 * c_out * c_in,
                c_out * c_in,
                c_in,
                1,
            ],
        }
        tensor_registry[next_uid] = packed
        next_uid += 1
        if is_exact_p5_768:
            # Physical layouts are [CI, KH, KW, CO] and [N, H, W, CO].
            # Both dot operands now have contiguous reduction channels.
            packed["strides"] = [3 * c_out, c_out, 1, 9 * c_out]
            packed_loss = {
                **loss,
                "uid": next_uid,
                "virtual": True,
                "strides": [c_out * 400, 1, c_out * 20, c_out],
            }
            tensor_registry[next_uid] = packed_loss
            next_uid += 1
            result.extend(
                [
                    _transpose_matrix_stage(
                        weight,
                        packed,
                        c_out,
                        c_in * 9,
                        group["source_node_ids"],
                    ),
                    _transpose_matrix_stage(
                        loss, packed_loss, c_out, 400, group["source_node_ids"]
                    ),
                ]
            )
            loss = packed_loss
        else:
            result.append(
                {
                    "source_node_ids": group["source_node_ids"],
                    "operation": group["operation"],
                    "parameters": {
                        **parameters,
                        "_dgrad_pipeline_stage": "pack",
                        "_dgrad_pack_round_tf32": (
                            is_fp32_p5_tile4 and not is_exact_p5_768
                        ),
                    },
                    "tensors": [weight, packed],
                    "input_uids": [weight["uid"]],
                    "output_uids": [packed["uid"]],
                }
            )
        if is_exact_p5_768:
            for parity_h in range(2):
                for parity_w in range(2):
                    result.append(
                        {
                            "source_node_ids": group["source_node_ids"],
                            "operation": group["operation"],
                            "parameters": {
                                **parameters,
                                "_dgrad_pipeline_stage": "compute",
                                "_dgrad_parity_h": parity_h,
                                "_dgrad_parity_w": parity_w,
                                "_dgrad_p5_parity": True,
                                "_dgrad_k_contiguous": True,
                            },
                            "tensors": [
                                loss,
                                packed,
                                output,
                            ],
                            "input_uids": [
                                loss["uid"],
                                packed["uid"],
                            ],
                            "output_uids": group["output_uids"],
                        }
                    )
            return result, next_uid
        if is_fp32_p5_tile4:
            result.append(
                {
                    "source_node_ids": group["source_node_ids"],
                    "operation": group["operation"],
                    "parameters": {
                        **parameters,
                        "_dgrad_pipeline_stage": "compute_tile4",
                        "_dgrad_round_tf32": True,
                    },
                    "tensors": [loss, packed, output],
                    "input_uids": [loss["uid"], packed["uid"]],
                    "output_uids": group["output_uids"],
                }
            )
            return result, next_uid

        use_tile4 = (
            128 <= c_in <= 768
            and loss["dimensions"][2] * loss["dimensions"][3] <= 1024
            and (loss["data_type"] != "float32" or c_in <= 256)
        )
        if use_tile4:
            result.append(
                {
                    "source_node_ids": group["source_node_ids"],
                    "operation": group["operation"],
                    "parameters": {
                        **parameters,
                        "_dgrad_pipeline_stage": "compute_tile4",
                    },
                    "tensors": [loss, packed, output],
                    "input_uids": [loss["uid"], packed["uid"]],
                    "output_uids": group["output_uids"],
                }
            )
        elif loss["data_type"] == "float32":
            for parity_h in range(2):
                result.append(
                    {
                        "source_node_ids": group["source_node_ids"],
                        "operation": group["operation"],
                        "parameters": {
                            **parameters,
                            "_dgrad_pipeline_stage": "compute_tile2w",
                            "_dgrad_parity_h": parity_h,
                            "_dgrad_small_ci": c_in < 32,
                        },
                        "tensors": [loss, packed, output],
                        "input_uids": [loss["uid"], packed["uid"]],
                        "output_uids": group["output_uids"],
                    }
                )
        else:
            for parity_h in range(2):
                for parity_w in range(2):
                    result.append(
                        {
                            "source_node_ids": group["source_node_ids"],
                            "operation": group["operation"],
                            "parameters": {
                                **parameters,
                                "_dgrad_pipeline_stage": "compute",
                                "_dgrad_parity_h": parity_h,
                                "_dgrad_parity_w": parity_w,
                            },
                            "tensors": [loss, packed, output],
                            "input_uids": [
                                loss["uid"],
                                packed["uid"],
                            ],
                            "output_uids": group["output_uids"],
                        }
                    )
        return result, next_uid
    return [group], next_uid
