"""Grouped GEMM layout checks, routing ABI and NVIDIA launch geometry."""

from .common import (
    FP8_DATA_TYPES,
    FLOAT_DATA_TYPES,
    TRITON_POINTER_TYPES,
    _has_non_overlapping_strides,
    _require_integer,
)


def _moe_matmul_configuration(operation, parameters, tensors, architecture):
    backward = operation == "moe_grouped_matmul_bwd"
    mode = _require_integer(parameters, "mode", minimum=0, maximum=2)
    top_k = _require_integer(parameters, "top_k", maximum=2**31 - 1)
    if (backward and mode) or len(tensors) != 4 + mode:
        raise ValueError("inconsistent MoE routing ports")
    token, matrix = (tensors[1], tensors[0]) if backward else tensors[:2]
    offsets, output = tensors[2], tensors[-1]
    if (
        token["data_type"] not in {"float16", "bfloat16"} | FP8_DATA_TYPES
        or matrix["data_type"] != token["data_type"]
        or output["data_type"] not in FLOAT_DATA_TYPES
    ):
        raise ValueError(
            "MoE requires matching FP16/BF16/FP8 inputs and floating output"
        )
    if token["data_type"] in FP8_DATA_TYPES and architecture < 90:
        raise ValueError("NVIDIA FP8 MoE requires SM90 or newer")
    for tensor in tensors:
        if len(tensor["dimensions"]) != 3 or not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        ):
            raise ValueError("MoE requires non-overlapping rank-three tensors")
    experts = _require_integer(parameters, "experts")
    tokens = _require_integer(parameters, "tokens", maximum=2**31 - 1)
    routed = _require_integer(parameters, "routed_tokens", maximum=2**31 - 1)
    k, n = _require_integer(parameters, "k"), _require_integer(parameters, "n")
    if (
        token["dimensions"] != [1, tokens, k]
        or offsets["dimensions"] != [experts, 1, 1]
        or offsets["data_type"] != "int32"
    ):
        raise ValueError("inconsistent MoE token or offset shape")
    if matrix["dimensions"] != (
        [1, tokens, n] if backward else [experts, k, n]
    ) or output["dimensions"] != (
        [experts, k, n] if backward else [1, routed, n]
    ):
        raise ValueError("inconsistent MoE matrix or output shape")
    if mode != 1 and routed != tokens:
        raise ValueError(
            "MoE routed and input counts must match outside gather mode"
        )
    for tensor in tensors[3:-1]:
        if tensor["data_type"] != "int32" or tensor["dimensions"] != [
            1,
            routed,
            1,
        ]:
            raise ValueError(
                "MoE routing indices must be INT32 [1,routed_tokens,1]"
            )
    if mode == 2 and routed % top_k:
        raise ValueError("MoE scatter routed count must be divisible by top-k")
    constants = dict(
        EXPERTS=experts,
        TOKENS=tokens,
        ROUTED=routed,
        K=k,
        N=n,
        MODE=mode,
        TOP_K=top_k,
        TM=token["strides"][1],
        TK=token["strides"][2],
        ME=matrix["strides"][0],
        MK=matrix["strides"][1],
        MN=matrix["strides"][2],
        OE=output["strides"][0],
        OM=output["strides"][1],
        ON=output["strides"][2],
        OS=offsets["strides"][0],
        IS=tensors[3]["strides"][1] if mode else 0,
        KS=tensors[4]["strides"][1] if mode == 2 else 0,
        BACKWARD=backward,
    )
    signature = dict(
        token_ptr=TRITON_POINTER_TYPES[token["data_type"]],
        matrix_ptr=TRITON_POINTER_TYPES[matrix["data_type"]],
        offsets_ptr="*i32",
        index_ptr="*i32",
        ks_ptr="*i32",
        output_ptr=TRITON_POINTER_TYPES[output["data_type"]],
    )
    arguments = [
        ("tensor_alias", 1 if backward else 0),
        ("tensor_alias", 0 if backward else 1),
        ("tensor_alias", 2),
        ("tensor_alias", 3 if mode else 2),
        ("tensor_alias", 4 if mode == 2 else 2),
        ("tensor_alias", len(tensors) - 1),
    ]
    grid = (
        experts * (((k if backward else routed) + 15) // 16) * ((n + 31) // 32)
    )
    if grid > 2**31 - 1:
        raise ValueError("MoE launch grid exceeds NVIDIA limits")
    return "moe_matmul_kernel", signature, constants, (grid, 1, 1), arguments
