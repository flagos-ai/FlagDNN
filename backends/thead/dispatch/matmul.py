# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""THead dispatch matmul lowering."""

from __future__ import annotations

from typing import Any
import math
from ..codegen.abi import (
    _tensor_argument,
    _tensor_pointer_signature,
)
from ..dispatch.common import (
    _FLOATING_DATA_TYPES,
    _PPU_WARP_SIZE,
    _integer,
    _require_exact_fields,
    _require_list,
    _require_object,
)
from ..dispatch.tensor import (
    _has_non_overlapping_strides,
    _named_port_uids,
)


def _validate_matmul_graph(graph: dict[str, Any]) -> dict[str, Any]:
    tensors = _require_list(graph["tensors"], "graph.tensors")
    nodes = _require_list(graph["nodes"], "graph.nodes")
    if len(nodes) != 1 or len(tensors) != 3:
        raise ValueError("THead MatMul requires one node and three tensors")
    node = _require_object(nodes[0], "MatMul node")
    if (
        node["id"] != 0
        or node["type"] != "matmul"
        or node["compute_data_type"] != "float32"
    ):
        raise ValueError("THead MatMul requires a canonical float32 node")
    input_uids = _named_port_uids(node, "inputs", ("a", "b"), "MatMul")
    output_uids = _named_port_uids(node, "outputs", ("output",), "MatMul")
    ordered_uids = input_uids + output_uids
    if len(set(ordered_uids)) != 3:
        raise ValueError("THead MatMul tensor UIDs must be distinct")
    tensors_by_uid = {int(tensor["uid"]): tensor for tensor in tensors}
    a, b, output = [tensors_by_uid[uid] for uid in ordered_uids]
    if (
        a["data_type"] not in _FLOATING_DATA_TYPES
        or b["data_type"] != a["data_type"]
        or output["data_type"] != a["data_type"]
        or any(
            tensor["virtual"] or int(tensor["alignment"]) < 16
            for tensor in (a, b, output)
        )
    ):
        raise ValueError(
            "THead MatMul requires matching aligned external "
            "float32/float16/bfloat16 tensors"
        )
    if any(
        not 2 <= len(tensor["dimensions"]) <= 8
        or not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        )
        for tensor in (a, b, output)
    ):
        raise ValueError(
            "THead MatMul requires rank [2, 8] non-overlapping tensors"
        )

    a_dimensions = [int(value) for value in a["dimensions"]]
    b_dimensions = [int(value) for value in b["dimensions"]]
    output_dimensions = [int(value) for value in output["dimensions"]]
    m = a_dimensions[-2]
    k = a_dimensions[-1]
    if b_dimensions[-2] != k:
        raise ValueError("THead MatMul contraction dimensions do not match")
    n = b_dimensions[-1]
    a_batch = a_dimensions[:-2]
    b_batch = b_dimensions[:-2]
    batch_rank = max(len(a_batch), len(b_batch))
    if batch_rank > 6:
        raise ValueError("THead MatMul batch rank exceeds six")
    batch_dimensions = [1] * batch_rank
    for trailing in range(batch_rank):
        a_dimension = a_batch[-1 - trailing] if trailing < len(a_batch) else 1
        b_dimension = b_batch[-1 - trailing] if trailing < len(b_batch) else 1
        if (
            a_dimension != b_dimension
            and a_dimension != 1
            and b_dimension != 1
        ):
            raise ValueError(
                "THead MatMul batch dimensions are not broadcast-compatible"
            )
        batch_dimensions[-1 - trailing] = max(a_dimension, b_dimension)
    expected_output = [*batch_dimensions, m, n]
    if output_dimensions != expected_output:
        raise ValueError("THead MatMul output shape is inconsistent")
    batch = math.prod(batch_dimensions)
    if batch * m * n > 2**31 - 1:
        raise ValueError("THead MatMul output element count exceeds int32")

    attributes = _require_object(node["attributes"], "MatMul attributes")
    _require_exact_fields(
        attributes,
        {"batch", "m", "n", "k"},
        {"input_precision"},
        "MatMul attributes",
    )
    if _integer(
        attributes.get("input_precision", 0), "input_precision"
    ) not in {0, 1, 2}:
        raise ValueError(
            "THead input precision must be default, IEEE, or TF32"
        )
    encoded = {
        name: _integer(attributes[name], f"MatMul {name}")
        for name in ("batch", "m", "n", "k")
    }
    if encoded != {"batch": batch, "m": m, "n": n, "k": k}:
        raise ValueError("THead MatMul attributes disagree with tensors")

    leading = 6 - batch_rank
    padded_dimensions = [1] * leading + batch_dimensions

    def batch_strides(tensor: dict[str, Any]) -> list[int]:
        tensor_dimensions = [int(value) for value in tensor["dimensions"][:-2]]
        tensor_strides = [int(value) for value in tensor["strides"][:-2]]
        tensor_leading = batch_rank - len(tensor_dimensions)
        aligned_dimensions = [1] * tensor_leading + tensor_dimensions
        aligned_strides = [0] * tensor_leading + tensor_strides
        return [0] * leading + [
            0 if dimension == 1 else stride
            for dimension, stride in zip(
                aligned_dimensions, aligned_strides, strict=True
            )
        ]

    constants: dict[str, int] = {
        "M": m,
        "N": n,
        "K": k,
        "A_STRIDE_M": int(a["strides"][-2]),
        "A_STRIDE_K": int(a["strides"][-1]),
        "B_STRIDE_K": int(b["strides"][-2]),
        "B_STRIDE_N": int(b["strides"][-1]),
        "C_STRIDE_M": int(output["strides"][-2]),
        "C_STRIDE_N": int(output["strides"][-1]),
        "INPUT_IS_FLOAT32": 1 if a["data_type"] == "float32" else 0,
        "USE_TF32": int(attributes.get("input_precision", 0) == 2),
    }
    a_batch_strides = batch_strides(a)
    b_batch_strides = batch_strides(b)
    c_batch_strides = [0] * leading + [
        int(value) for value in output["strides"][:-2]
    ]
    for axis in range(6):
        constants[f"DIM_{axis}"] = padded_dimensions[axis]
        constants[f"A_BATCH_STRIDE_{axis}"] = a_batch_strides[axis]
        constants[f"B_BATCH_STRIDE_{axis}"] = b_batch_strides[axis]
        constants[f"C_BATCH_STRIDE_{axis}"] = c_batch_strides[axis]
    return {
        "tensors": tensors,
        "argument_tensors": [a, b, output],
        "constants": constants,
        "batch": batch,
        "m": m,
        "n": n,
        "k": k,
    }


def _matmul_variant(
    plan: dict[str, Any],
    configuration: dict[str, Any],
    variant_id: str,
) -> dict[str, Any]:
    meta = configuration["META"]
    block_m = int(meta["BLOCK_M"])
    block_n = int(meta["BLOCK_N"])
    block_k = int(meta["BLOCK_K"])
    group_m = int(meta["GROUP_M"])
    constants = plan["constants"]
    ordered_constants = [
        constants["M"],
        constants["N"],
        constants["K"],
        *[constants[f"DIM_{axis}"] for axis in range(6)],
        *[constants[f"A_BATCH_STRIDE_{axis}"] for axis in range(6)],
        *[constants[f"B_BATCH_STRIDE_{axis}"] for axis in range(6)],
        *[constants[f"C_BATCH_STRIDE_{axis}"] for axis in range(6)],
        constants["A_STRIDE_M"],
        constants["A_STRIDE_K"],
        constants["B_STRIDE_K"],
        constants["B_STRIDE_N"],
        constants["C_STRIDE_M"],
        constants["C_STRIDE_N"],
        constants["INPUT_IS_FLOAT32"],
        constants["USE_TF32"],
        block_m,
        block_n,
        block_k,
        group_m,
    ]
    num_warps = int(configuration["num_warps"])
    input_data_type = plan["argument_tensors"][0]["data_type"]
    # The PPU CUDA-compatible Triton lowering widens the staged operands of
    # eight-warp 128x128 tile to four-byte lanes even for FP16/BF16.
    # Four-warp tiles retain packed two-byte lanes. This value is
    # part of the launch ABI and must match the compiled cubin metadata.
    shared_scalar_bytes = (
        4
        if input_data_type == "float32"
        or ((block_m >= 128 or block_n >= 128) and num_warps >= 8)
        else 2
    )
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
        "argument_count": 3,
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
            "grid": [
                ((int(plan["m"]) + block_m - 1) // block_m)
                * ((int(plan["n"]) + block_n - 1) // block_n),
                int(plan["batch"]),
                1,
            ],
            "block": [num_warps * _PPU_WARP_SIZE, 1, 1],
            # Triton CUDA-backend lowering stages both operands in dynamic
            # shared memory for this IEEE dot-product tile.
            "shared_memory": (block_m * block_k + block_k * block_n)
            * shared_scalar_bytes
            * max(1, int(configuration["num_stages"]) - 1),
        },
    }
