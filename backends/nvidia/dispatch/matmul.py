"""Matrix multiplication validation and launch configurations."""

from __future__ import annotations

from typing import Any
import math

from .common import (
    FLOAT_DATA_TYPES,
    TRITON_POINTER_TYPES,
    _ceil_div,
    _has_non_overlapping_strides,
    _is_row_major_contiguous,
    _require_integer,
)
from .matmul_tf32 import (
    _tf32_pack_kernel_configuration,
    _tf32_tma_direct_configuration,
    _tf32_tensor_map_configuration,
    _tf32_tma_kernel_configuration,
)


def _matmul_p5_pipeline_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, str],
    dict[str, int],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    stage = parameters.get("_fprop_p5_matmul_stage")
    if stage not in {"split", "reduce"}:
        raise ValueError("the P5 MatMul pipeline stage is invalid")
    m = _require_integer(parameters, "m", minimum=1)
    n = _require_integer(parameters, "n", minimum=1)
    k = _require_integer(parameters, "k", minimum=1)
    splits = _require_integer(
        parameters, "_fprop_p5_splits", minimum=2, maximum=64
    )

    if stage == "split":
        if len(tensors) != 3:
            raise ValueError("the P5 split-K stage requires three tensors")
        a, b, partial = tensors
        if (
            a["data_type"] not in FLOAT_DATA_TYPES
            or b["data_type"] != a["data_type"]
            or partial["data_type"] != "float32"
            or a["dimensions"] != [1, m, k]
            or b["dimensions"] != [1, k, n]
            or partial["dimensions"] != [splits, m, n]
            or not all(
                _is_row_major_contiguous(tensor) for tensor in (a, partial)
            )
            or (
                not _is_row_major_contiguous(b)
                and not (
                    b["data_type"] == "float32"
                    and b["strides"] == [k * n, 1, k]
                )
            )
        ):
            raise ValueError("the P5 split-K tensor contract is invalid")
        block_m = 64
        block_n = 64
        block_k = 64
        return (
            "matmul_p5_split_k_kernel",
            {
                "a_ptr": TRITON_POINTER_TYPES[a["data_type"]],
                "b_ptr": TRITON_POINTER_TYPES[b["data_type"]],
                "partial_ptr": TRITON_POINTER_TYPES["float32"],
            },
            {
                "M": m,
                "N": n,
                "K": k,
                "B_STRIDE_K": b["strides"][-2],
                "B_STRIDE_N": b["strides"][-1],
                "SPLITS": splits,
                "INPUT_IS_FLOAT32": a["data_type"] == "float32",
                "USE_TF32": k % 4 == 0 and n % 4 == 0,
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "GROUP_M": 8,
            },
            (
                _ceil_div(m, block_m) * _ceil_div(n, block_n),
                splits,
                1,
            ),
            [("tensor", None), ("tensor", None), ("tensor", None)],
        )

    if len(tensors) != 2:
        raise ValueError("the P5 split-K reduce stage requires two tensors")
    partial, output = tensors
    if (
        partial["data_type"] != "float32"
        or output["data_type"] not in FLOAT_DATA_TYPES
        or partial["dimensions"] != [splits, m, n]
        or output["dimensions"] != [1, m, n]
        or not _is_row_major_contiguous(partial)
        or not _is_row_major_contiguous(output)
    ):
        raise ValueError("the P5 split-K reduce tensor contract is invalid")
    block_size = 1024
    total = m * n
    return (
        "matmul_p5_split_k_reduce_kernel",
        {
            "partial_ptr": TRITON_POINTER_TYPES["float32"],
            "output_ptr": TRITON_POINTER_TYPES[output["data_type"]],
        },
        {"TOTAL": total, "SPLITS": splits, "BLOCK_SIZE": block_size},
        (_ceil_div(total, block_size), 1, 1),
        [("tensor", None), ("tensor", None)],
    )


def _matmul_kernel_configuration(
    parameters: dict[str, Any],
    tensors: list[dict[str, Any]],
    architecture: int,
) -> tuple[
    str,
    dict[str, str],
    dict[str, int | bool],
    tuple[int, int, int],
    list[tuple[str, str | int | None]],
]:
    if parameters.get("_matmul_tf32_pack") is True:
        return _tf32_pack_kernel_configuration(
            parameters, tensors, architecture
        )
    if len(tensors) != 3:
        raise ValueError("MatMul tensor count is invalid")
    a, b, output = tensors
    data_types = [tensor["data_type"] for tensor in tensors]
    broadcast_a = parameters.get("_fprop_broadcast_a", False)
    if not isinstance(broadcast_a, bool):
        raise ValueError("internal broadcast-A MatMul flag must be boolean")
    valid_data_types = (
        len(set(data_types)) == 1 and data_types[0] in FLOAT_DATA_TYPES
    )
    if not valid_data_types:
        raise ValueError(
            "MatMul input/output data types must match and be floating"
        )
    if any(
        len(tensor["dimensions"]) < 2
        or len(tensor["dimensions"]) > 8
        or not _has_non_overlapping_strides(
            tensor["dimensions"], tensor["strides"]
        )
        for tensor in tensors
    ):
        raise ValueError(
            "MatMul tensors require rank [2, 8] and non-overlapping strides"
        )

    m = a["dimensions"][-2]
    k = a["dimensions"][-1]
    if k != b["dimensions"][-2]:
        raise ValueError("MatMul contraction dimensions do not match")
    n = b["dimensions"][-1]
    a_batch = a["dimensions"][:-2]
    b_batch = b["dimensions"][:-2]
    batch_rank = max(len(a_batch), len(b_batch))
    if batch_rank > 6:
        raise ValueError("MatMul batch rank exceeds six")
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
                "MatMul batch dimensions are not broadcast-compatible"
            )
        batch_dimensions[-1 - trailing] = max(a_dimension, b_dimension)
    expected_output = [*batch_dimensions, m, n]
    if output["dimensions"] != expected_output:
        raise ValueError("MatMul output metadata is inconsistent")

    batch = math.prod(batch_dimensions)
    for name, expected in (("batch", batch), ("m", m), ("n", n), ("k", k)):
        if _require_integer(parameters, name) != expected:
            raise ValueError(
                f"parameters.{name} is inconsistent with MatMul metadata"
            )

    tf32_plan = _tf32_tma_direct_configuration(tensors, architecture)
    if tf32_plan is not None:
        return tf32_plan
    tf32_plan = _tf32_tensor_map_configuration(tensors, architecture)
    if tf32_plan is not None:
        return tf32_plan
    tf32_plan = _tf32_tma_kernel_configuration(
        parameters, tensors, architecture
    )
    if tf32_plan is not None:
        return tf32_plan

    if broadcast_a:
        if (
            len(a["dimensions"]) != 3
            or len(b["dimensions"]) != 3
            or len(output["dimensions"]) != 3
            or a["dimensions"] != [1, m, k]
            or b["dimensions"] != [batch, k, n]
            or output["dimensions"] != [batch, m, n]
            or not all(
                _is_row_major_contiguous(tensor) for tensor in (a, b, output)
            )
        ):
            raise ValueError("broadcast-A MatMul metadata is inconsistent")
        block_m = 64
        block_n = 64
        block_k = 32
        return (
            "matmul_batched_broadcast_a_kernel",
            {
                "a_ptr": TRITON_POINTER_TYPES[data_types[0]],
                "b_ptr": TRITON_POINTER_TYPES[data_types[1]],
                "c_ptr": TRITON_POINTER_TYPES[data_types[2]],
            },
            {
                "BATCH": batch,
                "M": m,
                "N": n,
                "K": k,
                "INPUT_IS_FLOAT32": data_types[0] == "float32",
                "USE_TF32": k % 4 == 0 and n % 4 == 0,
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "GROUP_M": 8,
            },
            (
                _ceil_div(m, block_m) * _ceil_div(n, block_n),
                batch,
                1,
            ),
            [("tensor", None), ("tensor", None), ("tensor", None)],
        )

    is_batched_contiguous = (
        len(a["dimensions"]) == 3
        and len(b["dimensions"]) == 3
        and len(output["dimensions"]) == 3
        and a["dimensions"] == [batch, m, k]
        and b["dimensions"] == [batch, k, n]
        and output["dimensions"] == [batch, m, n]
        and all(_is_row_major_contiguous(tensor) for tensor in (a, b, output))
    )
    if (
        is_batched_contiguous
        and data_types[0] != "float32"
        and architecture >= 90
        and min(m, n, k) >= 512
        # Flattened TMA descriptors must never read/write into the next batch.
        # These divisors cover every tile in the persistent tuning table.
        and m % 256 == 0
        and k % 128 == 0
        and n % 8 == 0
        and all(tensor.get("alignment", 1) >= 16 for tensor in tensors)
    ):
        # The short schedule targets a small number of persistent waves.
        # Larger batches retain the prior schedule; small batches need more
        # output tiles to occupy the device instead of one large tile per CTA.
        short_k = max(m, n, k) <= 512 and batch <= 32
        small_short = short_k and batch < 8
        block_m = 64 if small_short else 128
        block_n = 256 if short_k and not small_short else 128
        block_k = 32 if short_k and not small_short else 64
        total_tiles = batch * _ceil_div(m, block_m) * _ceil_div(n, block_n)
        pointer_type = TRITON_POINTER_TYPES[data_types[0]]
        return (
            (
                "matmul_batched_tma_short_kernel"
                if short_k
                else "matmul_batched_tma_persistent_kernel"
            ),
            {
                "a_ptr": pointer_type,
                "b_ptr": pointer_type,
                "c_ptr": pointer_type,
            },
            {
                "BATCH": batch,
                "M": m,
                "N": n,
                "K": k,
                "PERSISTENT_GRID": 128,
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "GROUP_M": 8,
                "INPUT_IS_FLOAT32": data_types[0] == "float32",
            },
            (min(total_tiles, 128), 1, 1),
            [("tensor", None), ("tensor", None), ("tensor", None)],
        )

    if is_batched_contiguous:
        block_m = 32 if m < 64 else 128
        block_n = 32 if n < 64 else 128
        block_k = 32 if k < 64 else 64
        pointer_type = TRITON_POINTER_TYPES[data_types[0]]
        return (
            "matmul_batched_contiguous_kernel",
            {
                "a_ptr": pointer_type,
                "b_ptr": pointer_type,
                "c_ptr": pointer_type,
            },
            {
                "BATCH": batch,
                "M": m,
                "N": n,
                "K": k,
                "INPUT_IS_FLOAT32": data_types[0] == "float32",
                "NATIVE_TF32_RNE": architecture >= 90,
                "USE_TF32": k % 4 == 0 and n % 4 == 0,
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "GROUP_M": 8,
            },
            (
                _ceil_div(m, block_m) * _ceil_div(n, block_n),
                batch,
                1,
            ),
            [("tensor", None), ("tensor", None), ("tensor", None)],
        )

    def batch_strides(tensor: dict[str, Any]) -> list[int]:
        tensor_batch = tensor["dimensions"][:-2]
        leading = batch_rank - len(tensor_batch)
        dimensions = [1] * leading + tensor_batch
        strides = [0] * leading + tensor["strides"][:-2]
        effective = [
            0 if dimension == 1 else stride
            for dimension, stride in zip(dimensions, strides)
        ]
        return [0] * (6 - batch_rank) + effective

    padded_dimensions = [1] * (6 - batch_rank) + batch_dimensions
    a_batch_strides = batch_strides(a)
    b_batch_strides = batch_strides(b)
    output_batch_strides = [0] * (6 - batch_rank) + output["strides"][:-2]
    block_m = 32 if m < 64 else 64
    block_n = 32 if n < 64 else 64
    block_k = 32
    constants: dict[str, int | bool] = {
        "M": m,
        "N": n,
        "K": k,
        "A_STRIDE_M": a["strides"][-2],
        "A_STRIDE_K": a["strides"][-1],
        "B_STRIDE_K": b["strides"][-2],
        "B_STRIDE_N": b["strides"][-1],
        "C_STRIDE_M": output["strides"][-2],
        "C_STRIDE_N": output["strides"][-1],
        "NATIVE_TF32_RNE": architecture >= 90,
        "INPUT_IS_FLOAT32": data_types[0] == "float32",
        # cuDNN's vectorized TF32 MatMul needs four-element alignment.
        # Irregular matrices and unsupported multidimensional broadcast retain
        # the higher-accuracy path, as verified against the direct reference.
        "USE_TF32": data_types[0] == "float32"
        and batch_rank <= 1
        and k % 4 == 0
        and n % 4 == 0,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "GROUP_M": 8,
    }
    for axis in range(6):
        constants[f"DIM_{axis}"] = padded_dimensions[axis]
        constants[f"A_BATCH_STRIDE_{axis}"] = a_batch_strides[axis]
        constants[f"B_BATCH_STRIDE_{axis}"] = b_batch_strides[axis]
        constants[f"C_BATCH_STRIDE_{axis}"] = output_batch_strides[axis]

    return (
        "matmul_strided_kernel",
        {
            "a_ptr": TRITON_POINTER_TYPES[data_types[0]],
            "b_ptr": TRITON_POINTER_TYPES[data_types[1]],
            "c_ptr": TRITON_POINTER_TYPES[data_types[2]],
        },
        constants,
        (
            ((m + block_m - 1) // block_m) * ((n + block_n - 1) // block_n),
            batch,
            1,
        ),
        [("tensor", None), ("tensor", None), ("tensor", None)],
    )
