"""Parallel partial attention gradients and deterministic device reduction."""

import math
from .attention_common import _require_fp8_scale_tensor
from .common import (
    TRITON_POINTER_TYPES,
    _is_row_major_contiguous,
    _has_non_overlapping_strides,
    _require_integer,
    _require_number,
)


def split_attention_gradients(stages, tensor_registry):
    result = []
    next_uid = max(tensor_registry, default=0) + 1
    for stage in stages:
        operation = stage["operation"]
        parameters = stage["parameters"]
        fp8 = operation == "sdpa_fp8_backward"
        key = "_sdpa_fp8_bwd_stage" if fp8 else "_sdpa_bwd_stage"
        if (
            operation not in {"sdpa_backward", "sdpa_fp8_backward"}
            or parameters.get(key) != "dkdv"
        ):
            result.append(stage)
            continue
        tensors = stage["tensors"]
        q, k, v = tensors[:3]
        batch, heads, sq, dimension = q["dimensions"]
        kv_heads, skv = k["dimensions"][1:3]
        chunk = max(128, ((sq + 3) // 4 + 127) // 128 * 128)
        chunks = (sq + chunk - 1) // chunk
        parts = heads // kv_heads * chunks
        dk_index = 17 if fp8 else len(tensors) - 2
        dk, dv = tensors[dk_index : dk_index + 2]
        if (
            parts == 1
            or sq < 128
            or q["data_type"] == "float32"
            or k["dimensions"] != v["dimensions"]
            or 8 * batch * heads * chunks * skv * dimension > 64 * 1024**2
        ):
            result.append(stage)
            continue
        partials = []
        for output in (dk, dv):
            partial = dict(
                output,
                uid=next_uid,
                virtual=True,
                alignment=16,
                data_type="float32",
                dimensions=[batch, heads * chunks, skv, dimension],
                strides=[
                    heads * chunks * skv * dimension,
                    skv * dimension,
                    dimension,
                    1,
                ],
            )
            tensor_registry[next_uid] = partial
            next_uid += 1
            partials.append(partial)
        replacements = {dk["uid"]: partials[0], dv["uid"]: partials[1]}
        modified = dict(
            stage,
            parameters={
                **parameters,
                "_sdpa_partial_count": parts,
                "_sdpa_query_chunk": chunk,
            },
            tensors=[replacements.get(t["uid"], t) for t in tensors],
            output_uids=[
                replacements[uid]["uid"] if uid in replacements else uid
                for uid in stage["output_uids"]
            ],
        )
        result.append(modified)
        extra = (
            [tensors[i] for i in (6, 10, 11, 12, 14, 15, 19, 20)]
            if fp8
            else []
        )
        result.append(
            dict(
                source_node_ids=stage["source_node_ids"],
                operation=operation,
                parameters={
                    **parameters,
                    "_sdpa_reduce_partials": True,
                    "_sdpa_partial_count": parts,
                },
                tensors=[*partials, dk, dv, *extra],
                input_uids=[t["uid"] for t in [*partials, *extra]],
                output_uids=[
                    dk["uid"],
                    dv["uid"],
                    *([tensors[19]["uid"], tensors[20]["uid"]] if fp8 else []),
                ],
            )
        )
    return result


def partial_gradient_reduction(operation, parameters, tensors):
    fp8 = operation == "sdpa_fp8_backward"
    if len(tensors) != (12 if fp8 else 4):
        raise ValueError("attention partial reduction tensor count is invalid")
    pk, pv, dk, dv = tensors[:4]
    parts = _require_integer(parameters, "_sdpa_partial_count", minimum=2)
    if any(len(t["dimensions"]) != 4 for t in (pk, pv, dk, dv)):
        raise ValueError(
            "attention partial reduction requires rank-four tensors"
        )
    output_types = {"fp8_e4m3", "fp8_e5m2"} if fp8 else {"float16", "bfloat16"}
    if (
        dk["data_type"] not in output_types
        or dv["data_type"] != dk["data_type"]
        or any(
            not _has_non_overlapping_strides(t["dimensions"], t["strides"])
            for t in (dk, dv)
        )
    ):
        raise ValueError("attention partial output metadata is invalid")
    if fp8:
        for t in tensors[4:]:
            _require_fp8_scale_tensor(t, "attention gradient scale/amax")
    batch, heads, sequence, dimension = dk["dimensions"]
    expected = [batch, heads * parts, sequence, dimension]
    if (
        not isinstance(parts, int)
        or parts <= 1
        or dv["dimensions"] != dk["dimensions"]
        or any(
            t["dimensions"] != expected
            or t["data_type"] != "float32"
            or not _is_row_major_contiguous(t)
            for t in (pk, pv)
        )
    ):
        raise ValueError("attention partial reduction metadata is invalid")
    pointer = TRITON_POINTER_TYPES[dk["data_type"]]
    signature = dict(
        pk_ptr="*fp32", pv_ptr="*fp32", dk_ptr=pointer, dv_ptr=pointer
    )
    names = [
        "descale_q",
        "descale_do",
        "descale_s",
        "descale_dp",
        "scale_dk",
        "scale_dv",
        "amax_dk",
        "amax_dv",
    ]
    signature.update({name + "_ptr": "*fp32" for name in names})
    arguments = [("tensor_alias", i) for i in range(4)]
    arguments += (
        [("tensor_alias", i) for i in range(4, 12)]
        if fp8
        else [("tensor_alias", 0)] * 8
    )
    constants = dict(
        ELEMENTS=math.prod(dk["dimensions"]),
        HEADS=heads,
        SEQUENCE=sequence,
        DIMENSION=dimension,
        PARTS=parts,
        FP8=fp8,
        ATTN_SCALE=_require_number(parameters, "attn_scale"),
        BLOCK=256,
    )
    for prefix, tensor in (("K", dk), ("V", dv)):
        for axis, stride in zip("BHSD", tensor["strides"]):
            constants[prefix + axis] = stride
    return (
        "attention_partial_gradient_reduce_kernel",
        signature,
        constants,
        ((constants["ELEMENTS"] + 255) // 256, 1, 1),
        arguments,
    )
