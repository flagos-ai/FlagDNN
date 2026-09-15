"""Contracts for FP32-storage, TF32-RNE packing and Hopper GEMM."""

from __future__ import annotations

from typing import Any

from .common import (
    KernelPlan,
    _ceil_div,
    _is_row_major_contiguous,
    _require_integer,
)


def _tf32_pack_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
    architecture: int,
) -> KernelPlan:
    batch = _require_integer(parameters, "batch", minimum=1)
    n = _require_integer(parameters, "n", minimum=1)
    k = _require_integer(parameters, "k", minimum=1)
    if (
        len(tensors) != 2
        or any(
            t["data_type"] != "float32" or t["dimensions"] != [batch, k, n]
            for t in tensors
        )
        or not _is_row_major_contiguous(tensors[0])
        or tensors[1]["strides"] != [k * n, 1, k]
        or k % 4
        or n % 4
    ):
        raise ValueError("TF32 MatMul packing metadata is invalid")
    return KernelPlan(
        "matmul_tf32_pack_b_kernel",
        {"b_ptr": "*fp32", "packed_ptr": "*fp32"},
        {
            "BATCH": batch,
            "N": n,
            "K": k,
            "BLOCK_SIZE": 32,
            "NATIVE_TF32_RNE": architecture >= 90,
        },
        (_ceil_div(k, 32) * _ceil_div(n, 32), batch, 1),
        [("tensor", None), ("tensor", None)],
    )


def _tf32_tma_direct_eligible(
    tensors: list[dict[str, Any]],
    architecture: int,
) -> bool:
    """Small Hopper matrices amortize packing poorly; use C.T = B.T @ A.T."""
    if (
        architecture < 90
        or len(tensors) != 3
        or any(
            t["data_type"] != "float32"
            or len(t["dimensions"]) != 3
            or not _is_row_major_contiguous(t)
            or t.get("alignment", 1) < 16
            for t in tensors
        )
    ):
        return False
    a, b, output = tensors
    batch, m, k = a["dimensions"]
    n = b["dimensions"][-1]
    return (
        batch >= 4
        and m == n == k == 512
        and b["dimensions"] == [batch, k, n]
        and output["dimensions"] == [batch, m, n]
    )


def _tf32_tma_direct_configuration(
    tensors: list[dict[str, Any]],
    architecture: int,
) -> KernelPlan | None:
    if not _tf32_tma_direct_eligible(tensors, architecture):
        return None
    batch, m, k = tensors[0]["dimensions"]
    n = tensors[1]["dimensions"][-1]
    return KernelPlan(
        "matmul_tf32_tma_direct_kernel",
        {"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32"},
        {
            "BATCH": batch,
            "M": m,
            "N": n,
            "K": k,
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 32,
            "PERSISTENT_GRID": 256,
            "GROUP_M": 8,
        },
        (min(256, batch * (m // 128) * (n // 128)), 1, 1),
        [("tensor", None), ("tensor", None), ("tensor", None)],
    )


def _tf32_tensor_map_eligible(
    tensors: list[dict[str, Any]],
    architecture: int,
) -> bool:
    """Whole-tile Hopper path; virtual inputs and tails retain pointer kernels."""
    if (
        architecture != 90
        or len(tensors) != 3
        or any(
            t["data_type"] != "float32"
            or len(t["dimensions"]) != 3
            or not _is_row_major_contiguous(t)
            or t.get("alignment", 1) < 16
            for t in tensors
        )
        or any(t["virtual"] for t in tensors[:2])
    ):
        return False
    a, b, output = tensors
    batch, m, k = a["dimensions"]
    n = b["dimensions"][-1]
    return (
        batch >= 4
        and min(m, n) >= 512
        and (k >= 1024 or (k == 512 and min(m, n) >= 1024))
        and m % 256 == 0
        and n % 128 == 0
        and k % 32 == 0
        and max(batch * m, batch * k, n, k) <= 2**31 - 1
        and b["dimensions"] == [batch, k, n]
        and output["dimensions"] == [batch, m, n]
    )


def _tf32_tensor_map_configuration(
    tensors: list[dict[str, Any]],
    architecture: int,
) -> KernelPlan | None:
    if not _tf32_tensor_map_eligible(tensors, architecture):
        return None
    batch, m, k = tensors[0]["dimensions"]
    n = tensors[1]["dimensions"][-1]
    if k == 512:
        return KernelPlan(
            "matmul_tf32_short_kernel",
            {"a_desc": "tensordesc", "b_desc": "tensordesc", "c_ptr": "*fp32"},
            {
                "BATCH": batch,
                "M": m,
                "N": n,
                "K": k,
                "BLOCK_M": 256,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "SLOTS": 4,
                "PERSISTENT_GRID": 132,
            },
            (min(132, batch * (m // 256) * (n // 128)), 1, 1),
            [
                ("tensor_map", "BLOCK_M,BLOCK_K"),
                ("tensor_map", "BLOCK_K,BLOCK_N"),
                ("tensor", None),
            ],
        )
    block_m, persistent_grid = 256, 132
    return KernelPlan(
        "matmul_tf32_tensor_map_kernel",
        {"a_desc": "tensordesc", "b_desc": "tensordesc", "c_ptr": "*fp32"},
        {
            "BATCH": batch,
            "M": m,
            "N": n,
            "K": k,
            "BLOCK_M": block_m,
            "BLOCK_N": 128,
            "BLOCK_K": 32,
            "GROUP_M": 8,
            "PERSISTENT_GRID": persistent_grid,
        },
        (min(persistent_grid, batch * (m // block_m) * (n // 128)), 1, 1),
        [
            ("tensor_map", "BLOCK_M,BLOCK_K"),
            ("tensor_map", "BLOCK_K,BLOCK_N"),
            ("tensor", None),
        ],
    )


def _tf32_tma_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
    architecture: int,
) -> KernelPlan | None:
    """Return None for valid packed layouts outside the Hopper tile contract."""
    if parameters.get("_matmul_tf32_packed") is not True:
        return None
    if len(tensors) != 3 or any(len(t["dimensions"]) != 3 for t in tensors):
        raise ValueError(
            "TF32 MatMul packed compute requires rank-three tensors"
        )
    a, b, output = tensors
    batch, m, k = a["dimensions"]
    n = b["dimensions"][-1]
    if (
        any(t["data_type"] != "float32" for t in tensors)
        or b["dimensions"] != [batch, k, n]
        or output["dimensions"] != [batch, m, n]
        or not all(_is_row_major_contiguous(t) for t in (a, output))
        or b["strides"] != [k * n, 1, k]
        or k % 4
        or n % 4
    ):
        raise ValueError("TF32 MatMul packed compute metadata is invalid")
    # Every tuning candidate uses BM <= 128 and BK <= 64. Flattened A TMA
    # loads must stay in their own batch; B tails are descriptor-bounded.
    if (
        architecture < 90
        or m % 128
        or k % 64
        or any(t.get("alignment", 1) < 16 for t in tensors)
    ):
        return None
    return KernelPlan(
        "matmul_tf32_tma_kernel",
        {"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32"},
        {
            "BATCH": batch,
            "M": m,
            "N": n,
            "K": k,
            "BLOCK_M": 128,
            "BLOCK_N": 256,
            "BLOCK_K": 32,
            "PERSISTENT_GRID": 132,
            "GROUP_M": 8,
        },
        (min(132, batch * _ceil_div(m, 128) * _ceil_div(n, 256)), 1, 1),
        [("tensor", None), ("tensor", None), ("tensor", None)],
    )
