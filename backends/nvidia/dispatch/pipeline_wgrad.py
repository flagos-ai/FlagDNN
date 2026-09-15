"""Ordered weight-gradient algorithm predicates and stage expansion."""

from __future__ import annotations

from typing import Any

from .common import (
    ExecutionGroup,
    FLOAT_DATA_TYPES,
    _is_row_major_contiguous,
    _next_power_of_two,
)


def _expand_wgrad_group(
    group: ExecutionGroup,
    tensor_registry: dict[int, dict[str, Any]],
    next_uid: int,
) -> tuple[list[ExecutionGroup], int]:
    """Select wgrad stages in priority order; return the unchanged group if unmatched."""
    parameters = group["parameters"]
    tensors = group["tensors"]
    result: list[ExecutionGroup] = []
    is_exact_wgrad_stem = (
        group["operation"] == "convolution_wgrad"
        and len(tensors) == 3
        and tensors[1]["dimensions"] == [1, 3, 640, 640]
        and tensors[0]["dimensions"]
        == [1, tensors[2]["dimensions"][0], 320, 320]
        and tensors[2]["dimensions"] == [tensors[2]["dimensions"][0], 3, 3, 3]
        and tensors[2]["dimensions"][0] in {16, 32, 64, 96}
        and all(_is_row_major_contiguous(tensor) for tensor in tensors)
        and parameters.get("spatial_rank") == 2
        and parameters.get("groups") == 1
        and parameters.get("stride") == [2, 2]
        and parameters.get("pre_padding") == [1, 1]
        and parameters.get("post_padding") == [1, 1]
        and parameters.get("dilation") == [1, 1]
        and parameters.get("convolution_mode") == 0
    )
    if is_exact_wgrad_stem:
        loss, image, output = tensors
        num_splits = 64
        c_out, cin_per_group, kernel_h, kernel_w = output["dimensions"]
        cik = cin_per_group * kernel_h * kernel_w
        partial = {
            "uid": next_uid,
            "virtual": True,
            "data_type": "float32",
            "dimensions": [num_splits, c_out, cik],
            "strides": [c_out * cik, cik, 1],
        }
        tensor_registry[next_uid] = partial
        next_uid += 1
        pipeline_parameters = {
            **parameters,
            "_wgrad_pipeline_algorithm": "stem_col",
            "_wgrad_num_splits": num_splits,
            "_wgrad_kernel_h": kernel_h,
            "_wgrad_kernel_w": kernel_w,
        }
        result.append(
            {
                "source_node_ids": group["source_node_ids"],
                "operation": group["operation"],
                "parameters": {
                    **pipeline_parameters,
                    "_wgrad_pipeline_stage": "split",
                },
                "tensors": [image, loss, partial],
                "input_uids": [image["uid"], loss["uid"]],
                "output_uids": [partial["uid"]],
            }
        )
        result.append(
            {
                "source_node_ids": group["source_node_ids"],
                "operation": group["operation"],
                "parameters": {
                    **pipeline_parameters,
                    "_wgrad_pipeline_stage": "reduce",
                },
                "tensors": [partial, output],
                "input_uids": [partial["uid"]],
                "output_uids": group["output_uids"],
            }
        )
        return result, next_uid

    is_exact_wgrad_p5 = (
        group["operation"] == "convolution_wgrad"
        and len(tensors) == 3
        and (
            tuple(tensors[1]["dimensions"]),
            tuple(tensors[0]["dimensions"]),
            tuple(tensors[2]["dimensions"]),
        )
        in {
            (
                (1, 128, 40, 40),
                (1, 256, 20, 20),
                (256, 128, 3, 3),
            ),
            (
                (1, 256, 40, 40),
                (1, 512, 20, 20),
                (512, 256, 3, 3),
            ),
            (
                (1, 512, 40, 40),
                (1, 512, 20, 20),
                (512, 512, 3, 3),
            ),
            (
                (1, 768, 40, 40),
                (1, 768, 20, 20),
                (768, 768, 3, 3),
            ),
        }
        and all(_is_row_major_contiguous(tensor) for tensor in tensors)
        and parameters.get("spatial_rank") == 2
        and parameters.get("groups") == 1
        and parameters.get("stride") == [2, 2]
        and parameters.get("pre_padding") == [1, 1]
        and parameters.get("post_padding") == [1, 1]
        and parameters.get("dilation") == [1, 1]
        and parameters.get("convolution_mode") == 0
    )
    if is_exact_wgrad_p5:
        loss, image, output = tensors
        c_in = image["dimensions"][1]
        cik = c_in * 9
        pipeline_parameters = {
            **parameters,
            "_wgrad_pipeline_algorithm": "p5",
        }
        packed = {
            "uid": next_uid,
            "virtual": True,
            "data_type": image["data_type"],
            "dimensions": [400, cik],
            "strides": [cik, 1],
        }
        tensor_registry[next_uid] = packed
        next_uid += 1
        result.append(
            {
                "source_node_ids": group["source_node_ids"],
                "operation": group["operation"],
                "parameters": {
                    **pipeline_parameters,
                    "_wgrad_pipeline_stage": "pack",
                },
                "tensors": [image, packed],
                "input_uids": [image["uid"]],
                "output_uids": [packed["uid"]],
            }
        )
        result.append(
            {
                "source_node_ids": group["source_node_ids"],
                "operation": group["operation"],
                "parameters": {
                    **pipeline_parameters,
                    "_wgrad_pipeline_stage": "matmul",
                },
                "tensors": [loss, packed, output],
                "input_uids": [loss["uid"], packed["uid"]],
                "output_uids": group["output_uids"],
            }
        )
        return result, next_uid

    is_exact_wgrad_stride2 = (
        group["operation"] == "convolution_wgrad"
        and len(tensors) == 3
        and tensors[0]["dimensions"] == [8, 128, 28, 28]
        and tensors[1]["dimensions"] == [8, 64, 56, 56]
        and tensors[2]["dimensions"] == [128, 64, 3, 3]
        and all(_is_row_major_contiguous(tensor) for tensor in tensors)
        and parameters.get("spatial_rank") == 2
        and parameters.get("groups") == 1
        and parameters.get("stride") == [2, 2]
        and parameters.get("pre_padding") == [1, 1]
        and parameters.get("post_padding") == [1, 1]
        and parameters.get("dilation") == [1, 1]
        and parameters.get("convolution_mode") == 0
    )
    if is_exact_wgrad_stride2:
        loss, image, output = tensors
        if image["data_type"] in FLOAT_DATA_TYPES:
            n, _, _, _ = image["dimensions"]
            _, c_out, loss_h, loss_w = loss["dimensions"]
            _, cin_per_group, kernel_h, kernel_w = output["dimensions"]
            output_area = loss_h * loss_w
            padded_area = _next_power_of_two(output_area)
            cik = cin_per_group * kernel_h * kernel_w
            columns = {
                "uid": next_uid,
                "virtual": True,
                "data_type": image["data_type"],
                "dimensions": [n, cik, padded_area],
                "strides": [cik * padded_area, padded_area, 1],
            }
            tensor_registry[next_uid] = columns
            next_uid += 1
            partial = {
                "uid": next_uid,
                "virtual": True,
                "data_type": image["data_type"],
                "dimensions": [n, c_out, cik],
                "strides": [c_out * cik, cik, 1],
            }
            tensor_registry[next_uid] = partial
            next_uid += 1
            pipeline_parameters = {
                **parameters,
                "_wgrad_pipeline_algorithm": "stride2_im2col",
                "_wgrad_num_splits": n,
                "_wgrad_kernel_h": kernel_h,
                "_wgrad_kernel_w": kernel_w,
            }
            result.append(
                {
                    "source_node_ids": group["source_node_ids"],
                    "operation": group["operation"],
                    "parameters": {
                        **pipeline_parameters,
                        "_wgrad_pipeline_stage": "im2col",
                    },
                    "tensors": [image, columns],
                    "input_uids": [image["uid"]],
                    "output_uids": [columns["uid"]],
                }
            )
            result.append(
                {
                    "source_node_ids": group["source_node_ids"],
                    "operation": group["operation"],
                    "parameters": {
                        **pipeline_parameters,
                        "_wgrad_pipeline_stage": "matmul",
                    },
                    "tensors": [loss, columns, partial],
                    "input_uids": [loss["uid"], columns["uid"]],
                    "output_uids": [partial["uid"]],
                }
            )
            result.append(
                {
                    "source_node_ids": group["source_node_ids"],
                    "operation": group["operation"],
                    "parameters": {
                        **pipeline_parameters,
                        "_wgrad_pipeline_stage": "reduce",
                    },
                    "tensors": [partial, output],
                    "input_uids": [partial["uid"]],
                    "output_uids": group["output_uids"],
                }
            )
            return result, next_uid
        num_splits = image["dimensions"][0]
        c_out, cin_per_group, kernel_h, kernel_w = output["dimensions"]
        kernel_elements = kernel_h * kernel_w
        partial = {
            "uid": next_uid,
            "virtual": True,
            "data_type": image["data_type"],
            "dimensions": [
                num_splits,
                c_out,
                cin_per_group,
                kernel_elements,
            ],
            "strides": [
                c_out * cin_per_group * kernel_elements,
                cin_per_group * kernel_elements,
                kernel_elements,
                1,
            ],
        }
        tensor_registry[next_uid] = partial
        next_uid += 1
        pipeline_parameters = {
            **parameters,
            "_wgrad_pipeline_algorithm": "stride2_row4",
            "_wgrad_num_splits": num_splits,
        }
        result.append(
            {
                "source_node_ids": group["source_node_ids"],
                "operation": group["operation"],
                "parameters": {
                    **pipeline_parameters,
                    "_wgrad_pipeline_stage": "split",
                },
                "tensors": [image, loss, partial],
                "input_uids": [image["uid"], loss["uid"]],
                "output_uids": [partial["uid"]],
            }
        )
        result.append(
            {
                "source_node_ids": group["source_node_ids"],
                "operation": group["operation"],
                "parameters": {
                    **pipeline_parameters,
                    "_wgrad_pipeline_stage": "reduce",
                },
                "tensors": [partial, output],
                "input_uids": [partial["uid"]],
                "output_uids": group["output_uids"],
            }
        )
        return result, next_uid

    is_wgrad_pipeline_candidate = (
        group["operation"] == "convolution_wgrad"
        and len(tensors) == 3
        and all(_is_row_major_contiguous(tensor) for tensor in tensors)
        and parameters.get("spatial_rank") == 2
        and parameters.get("groups") == 1
        and parameters.get("stride") == [1, 1]
        and parameters.get("dilation") == [1, 1]
        and parameters.get("convolution_mode") == 0
    )
    is_exact_wgrad_stride1 = (
        is_wgrad_pipeline_candidate
        and tensors[0]["dimensions"] == [8, 64, 32, 32]
        and tensors[1]["dimensions"] == [8, 32, 32, 32]
        and tensors[2]["dimensions"] == [64, 32, 3, 3]
        and parameters.get("pre_padding") == [1, 1]
        and parameters.get("post_padding") == [1, 1]
    )
    is_exact_wgrad_1x1 = (
        is_wgrad_pipeline_candidate
        and tensors[0]["dimensions"] == [8, 128, 28, 28]
        and tensors[1]["dimensions"] == [8, 64, 28, 28]
        and tensors[2]["dimensions"] == [128, 64, 1, 1]
        and parameters.get("pre_padding") == [0, 0]
        and parameters.get("post_padding") == [0, 0]
    )
    if not is_exact_wgrad_stride1 and not is_exact_wgrad_1x1:
        result.append(group)
        return result, next_uid

    loss, image, output = tensors
    if is_exact_wgrad_1x1 and image["data_type"] in {
        "float16",
        "bfloat16",
    }:
        n, c_in, image_h, image_w = image["dimensions"]
        c_out = output["dimensions"][0]
        image_area = image_h * image_w
        num_splits = 4 * n
        image_view = {
            **image,
            "dimensions": [n, c_in, image_area],
            "strides": [c_in * image_area, image_area, 1],
        }
        pipeline_parameters = {
            **parameters,
            "_wgrad_num_splits": num_splits,
            "_wgrad_kernel_h": 1,
            "_wgrad_kernel_w": 1,
            "_wgrad_pipeline_algorithm": "1x1_split",
        }
        partial = {
            "uid": next_uid,
            "virtual": True,
            "data_type": image["data_type"],
            "dimensions": [num_splits, c_out, c_in],
            "strides": [c_out * c_in, c_in, 1],
        }
        tensor_registry[next_uid] = partial
        next_uid += 1
        result.append(
            {
                "source_node_ids": group["source_node_ids"],
                "operation": group["operation"],
                "parameters": {
                    **pipeline_parameters,
                    "_wgrad_pipeline_stage": "matmul",
                },
                "tensors": [loss, image_view, partial],
                "input_uids": [loss["uid"], image["uid"]],
                "output_uids": [partial["uid"]],
            }
        )
        result.append(
            {
                "source_node_ids": group["source_node_ids"],
                "operation": group["operation"],
                "parameters": {
                    **pipeline_parameters,
                    "_wgrad_pipeline_stage": "reduce",
                },
                "tensors": [partial, output],
                "input_uids": [partial["uid"]],
                "output_uids": group["output_uids"],
            }
        )
        return result, next_uid

    pipeline_algorithm = "1x1" if is_exact_wgrad_1x1 else "3tap"
    num_splits = 32 if is_exact_wgrad_1x1 else 16
    c_out = tensors[2]["dimensions"][0]
    cin_per_group = tensors[2]["dimensions"][1]
    kernel_h = tensors[2]["dimensions"][2]
    kernel_w = tensors[2]["dimensions"][3]
    cik = cin_per_group * kernel_h * kernel_w
    partial = {
        "uid": next_uid,
        "virtual": True,
        "data_type": (
            "float16"
            if is_exact_wgrad_1x1 and tensors[1]["data_type"] == "float16"
            else "float32"
        ),
        "dimensions": [num_splits, c_out, cik],
        "strides": [c_out * cik, cik, 1],
    }
    tensor_registry[next_uid] = partial
    next_uid += 1
    pipeline_parameters = {
        **parameters,
        "_wgrad_num_splits": num_splits,
        "_wgrad_kernel_h": kernel_h,
        "_wgrad_kernel_w": kernel_w,
        "_wgrad_pipeline_algorithm": pipeline_algorithm,
    }
    result.append(
        {
            "source_node_ids": group["source_node_ids"],
            "operation": group["operation"],
            "parameters": {
                **pipeline_parameters,
                "_wgrad_pipeline_stage": "split",
            },
            "tensors": [image, loss, partial],
            "input_uids": [image["uid"], loss["uid"]],
            "output_uids": [partial["uid"]],
        }
    )
    result.append(
        {
            "source_node_ids": group["source_node_ids"],
            "operation": group["operation"],
            "parameters": {
                **pipeline_parameters,
                "_wgrad_pipeline_stage": "reduce",
            },
            "tensors": [partial, output],
            "input_uids": [partial["uid"]],
            "output_uids": group["output_uids"],
        }
    )
    return result, next_uid
