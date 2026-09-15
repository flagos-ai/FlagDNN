"""FP32 MatMul packing policy; separate layout preparation from GEMM planning."""

from __future__ import annotations

from typing import Any

from .common import ExecutionGroup, _is_row_major_contiguous
from .layout import _transpose_matrix_stage
from .matmul_tf32 import _tf32_tma_direct_eligible, _tf32_tensor_map_eligible


def _expand_matmul_group(
    group: ExecutionGroup,
    tensor_registry: dict[int, dict[str, Any]],
    next_uid: int,
    *,
    architecture: int = 0,
) -> tuple[list[ExecutionGroup], int]:
    tensors = group["tensors"]
    if len(tensors) != 3 or not all(
        t["data_type"] == "float32"
        and len(t["dimensions"]) == 3
        and _is_row_major_contiguous(t)
        for t in tensors
    ):
        return [group], next_uid
    a, b, output = tensors
    batch, m, k = a["dimensions"]
    n = b["dimensions"][-1]
    if (
        batch < 4
        or min(m, n, k) < 512
        or b["dimensions"] != [batch, k, n]
        or output["dimensions"] != [batch, m, n]
        or group["parameters"].get("_fprop_broadcast_a", False)
    ):
        return [group], next_uid
    if _tf32_tma_direct_eligible(
        tensors, architecture
    ) or _tf32_tensor_map_eligible(tensors, architecture):
        return [group], next_uid
    # TF32 operands are rounded once while packing each batch independently.
    # Irregular IEEE-accuracy matrices retain the original pure-copy layout.
    tf32_packed = k % 4 == 0 and n % 4 == 0
    packed = {
        **b,
        "uid": next_uid,
        "virtual": True,
        "strides": [k * n, 1, k] if tf32_packed else [k, 1, batch * k],
    }
    tensor_registry[next_uid] = packed
    if tf32_packed:
        packing = {
            "source_node_ids": group["source_node_ids"],
            "operation": "matmul",
            "parameters": {**group["parameters"], "_matmul_tf32_pack": True},
            "tensors": [b, packed],
            "input_uids": [b["uid"]],
            "output_uids": [packed["uid"]],
        }
    else:
        packing = _transpose_matrix_stage(
            b, packed, batch * k, n, group["source_node_ids"]
        )
    compute = {
        **group,
        "parameters": {
            **group["parameters"],
            "_matmul_k_contiguous": True,
            "_matmul_tf32_packed": tf32_packed,
        },
        "tensors": [a, packed, output],
        "input_uids": [a["uid"], packed["uid"]],
    }
    return [packing, compute], next_uid + 1
