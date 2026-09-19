"""Ascend dispatch for attention pipeline."""

from typing import Any
from .attention_common import _attention_flag
from .common import _is_row_major_contiguous


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
        and (dbias["dimensions"][0] != batch or dbias["dimensions"][1] != heads)
    )
    if dbias_reduce and not _is_row_major_contiguous(dbias):
        raise ValueError("SDPA broadcast dBias reduction requires contiguous storage")
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
