"""Expand attention operations into explicit dependency-ordered stages."""

from __future__ import annotations

from typing import Any

from .attention_common import _attention_flag
from .common import _is_row_major_contiguous


def _expand_sdpa_fp8_forward_group(
    group: dict[str, Any],
) -> list[dict[str, Any]]:
    parameters = group["parameters"]
    tensors = group["tensors"]
    has_bias = _attention_flag(parameters, "has_bias")
    if len(tensors) != 13 + int(has_bias):
        raise ValueError("FP8 SDPA forward group tensor count is invalid")
    output_index = 10 if has_bias else 9
    amax_s = tensors[output_index + 2]
    amax_o = tensors[output_index + 3]
    return [
        {
            "source_node_ids": group["source_node_ids"],
            "operation": "sdpa_fp8",
            "parameters": {
                **parameters,
                "_sdpa_fp8_stage": "zero_amax",
            },
            "tensors": [amax_s, amax_o],
            "input_uids": [],
            "output_uids": [amax_s["uid"], amax_o["uid"]],
        },
        {
            "source_node_ids": group["source_node_ids"],
            "operation": "sdpa_fp8",
            "parameters": {
                **parameters,
                "_sdpa_fp8_stage": "forward",
            },
            "tensors": tensors,
            "input_uids": [
                *group["input_uids"],
                amax_s["uid"],
                amax_o["uid"],
            ],
            "output_uids": group["output_uids"],
        },
    ]


def _expand_sdpa_fp8_backward_group(
    group: dict[str, Any],
) -> list[dict[str, Any]]:
    parameters = group["parameters"]
    tensors = group["tensors"]
    if len(tensors) != 25:
        raise ValueError("FP8 SDPA backward group tensor count is invalid")
    q, k, v, output, doutput, stats = tensors[:6]
    (
        descale_q,
        descale_k,
        descale_v,
        descale_o,
        descale_doutput,
        descale_s,
        descale_dp,
        scale_s,
        scale_dq,
        scale_dk,
        scale_dv,
        scale_dp,
    ) = tensors[6:18]
    dq, dk, dv = tensors[18:21]
    amax_dq, amax_dk, amax_dv, amax_dp = tensors[21:25]
    dq_inputs = [
        q,
        k,
        v,
        output,
        doutput,
        stats,
        descale_q,
        descale_k,
        descale_v,
        descale_o,
        descale_doutput,
        descale_dp,
        scale_dq,
        scale_dp,
    ]
    dkdv_inputs = [
        q,
        k,
        v,
        output,
        doutput,
        stats,
        descale_q,
        descale_k,
        descale_v,
        descale_o,
        descale_doutput,
        descale_s,
        descale_dp,
        scale_s,
        scale_dk,
        scale_dv,
        scale_dp,
    ]
    return [
        {
            "source_node_ids": group["source_node_ids"],
            "operation": "sdpa_fp8_backward",
            "parameters": {
                **parameters,
                "_sdpa_fp8_bwd_stage": "zero_amax",
            },
            "tensors": [amax_dq, amax_dk, amax_dv, amax_dp],
            "input_uids": [],
            "output_uids": [
                amax_dq["uid"],
                amax_dk["uid"],
                amax_dv["uid"],
                amax_dp["uid"],
            ],
        },
        {
            "source_node_ids": group["source_node_ids"],
            "operation": "sdpa_fp8_backward",
            "parameters": {
                **parameters,
                "_sdpa_fp8_bwd_stage": "dq",
            },
            "tensors": [*dq_inputs, dq, amax_dq],
            "input_uids": [
                *[tensor["uid"] for tensor in dq_inputs],
                amax_dq["uid"],
            ],
            "output_uids": [dq["uid"], amax_dq["uid"]],
        },
        {
            "source_node_ids": group["source_node_ids"],
            "operation": "sdpa_fp8_backward",
            "parameters": {
                **parameters,
                "_sdpa_fp8_bwd_stage": "dkdv",
            },
            "tensors": [
                *dkdv_inputs,
                dk,
                dv,
                amax_dk,
                amax_dv,
                amax_dp,
            ],
            "input_uids": [
                *[tensor["uid"] for tensor in dkdv_inputs],
                amax_dk["uid"],
                amax_dv["uid"],
                amax_dp["uid"],
            ],
            "output_uids": [
                dk["uid"],
                dv["uid"],
                amax_dk["uid"],
                amax_dv["uid"],
                amax_dp["uid"],
            ],
        },
    ]


def _expand_sdpa_backward_group(
    group: dict[str, Any],
    tensor_registry: dict[int, dict[str, Any]],
    next_uid: int,
) -> tuple[list[dict[str, Any]], int]:
    parameters = group["parameters"]
    tensors = group["tensors"]
    has_bias = _attention_flag(parameters, "has_bias")
    has_dbias = _attention_flag(parameters, "has_dbias")
    expected_count = 9 + int(has_bias) + int(has_dbias)
    if len(tensors) != expected_count:
        raise ValueError("SDPA backward group tensor count is invalid")

    q, k, v, output, doutput, stats = tensors[:6]
    offset = 6
    bias = tensors[offset] if has_bias else None
    offset += int(has_bias)
    dq, dk, dv = tensors[offset : offset + 3]
    offset += 3
    dbias = tensors[offset] if has_dbias else None

    batch, heads, sequence_q, _ = q["dimensions"]
    delta = {
        "uid": next_uid,
        "virtual": True,
        "alignment": 16,
        "data_type": "float32",
        "dimensions": [batch, heads, sequence_q],
        "strides": [heads * sequence_q, sequence_q, 1],
    }
    tensor_registry[next_uid] = delta
    next_uid += 1

    dbias_reduce = bool(
        dbias is not None
        and (
            dbias["dimensions"][0] != batch or dbias["dimensions"][1] != heads
        )
    )
    if dbias_reduce and not _is_row_major_contiguous(dbias):
        raise ValueError(
            "SDPA broadcast dBias reduction requires contiguous storage"
        )
    pipeline_parameters = {
        **parameters,
        "dbias_reduce": int(dbias_reduce),
    }
    common_inputs = [q, k, v]
    if bias is not None:
        common_inputs.append(bias)

    stages: list[dict[str, Any]] = []
    if dbias_reduce:
        stages.append(
            {
                "source_node_ids": group["source_node_ids"],
                "operation": "sdpa_backward",
                "parameters": {
                    **pipeline_parameters,
                    "_sdpa_bwd_stage": "zero_dbias",
                },
                "tensors": [dbias],
                "input_uids": [],
                "output_uids": [dbias["uid"]],
            }
        )

    dq_tensors = [*common_inputs, output, doutput, stats, delta, dq]
    if dbias is not None:
        dq_tensors.append(dbias)
    dq_inputs = [tensor["uid"] for tensor in common_inputs]
    dq_inputs.extend([output["uid"], doutput["uid"], stats["uid"]])
    if dbias_reduce:
        dq_inputs.append(dbias["uid"])
    dq_outputs = [delta["uid"], dq["uid"]]
    if dbias is not None:
        dq_outputs.append(dbias["uid"])
    stages.append(
        {
            "source_node_ids": group["source_node_ids"],
            "operation": "sdpa_backward",
            "parameters": {
                **pipeline_parameters,
                "_sdpa_bwd_stage": "dq",
            },
            "tensors": dq_tensors,
            "input_uids": dq_inputs,
            "output_uids": dq_outputs,
        }
    )

    if q["dimensions"][3] == v["dimensions"][3]:
        stages.append(
            {
                "source_node_ids": group["source_node_ids"],
                "operation": "sdpa_backward",
                "parameters": {
                    **pipeline_parameters,
                    "_sdpa_bwd_stage": "dkdv",
                },
                "tensors": [
                    *common_inputs,
                    doutput,
                    stats,
                    delta,
                    dk,
                    dv,
                ],
                "input_uids": [
                    *[tensor["uid"] for tensor in common_inputs],
                    doutput["uid"],
                    stats["uid"],
                    delta["uid"],
                ],
                "output_uids": [dk["uid"], dv["uid"]],
            }
        )
    else:
        stages.append(
            {
                "source_node_ids": group["source_node_ids"],
                "operation": "sdpa_backward",
                "parameters": {
                    **pipeline_parameters,
                    "_sdpa_bwd_stage": "dk",
                },
                "tensors": [
                    *common_inputs,
                    doutput,
                    stats,
                    delta,
                    dk,
                ],
                "input_uids": [
                    *[tensor["uid"] for tensor in common_inputs],
                    doutput["uid"],
                    stats["uid"],
                    delta["uid"],
                ],
                "output_uids": [dk["uid"]],
            }
        )
        dv_inputs = [q, k]
        if bias is not None:
            dv_inputs.append(bias)
        stages.append(
            {
                "source_node_ids": group["source_node_ids"],
                "operation": "sdpa_backward",
                "parameters": {
                    **pipeline_parameters,
                    "_sdpa_bwd_stage": "dv",
                },
                "tensors": [*dv_inputs, doutput, stats, dv],
                "input_uids": [
                    *[tensor["uid"] for tensor in dv_inputs],
                    doutput["uid"],
                    stats["uid"],
                ],
                "output_uids": [dv["uid"]],
            }
        )
    return stages, next_uid
