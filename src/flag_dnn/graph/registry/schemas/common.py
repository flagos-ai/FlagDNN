from __future__ import annotations

from typing import Any, Optional

import torch

from flag_dnn.graph.tensor import TensorSpec, canonical_dtype


def _shape_like_first(
    input_specs: list[TensorSpec], attrs: dict[str, Any]
) -> list[TensorSpec]:
    inp = input_specs[0]
    return [
        TensorSpec(
            name="",
            shape=inp.shape,
            dtype=inp.dtype,
            stride=None,
            layout=inp.layout,
            device=inp.device,
            contiguous=inp.contiguous,
        )
    ]


def _normalize_axis(axis: Any, rank: int, op_type: str = "op") -> int:
    axis = int(axis)
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise IndexError(
            f"graph {op_type} axis out of range for rank {rank}: {axis}"
        )
    return axis


def _rank_of(value: Any) -> int:
    if hasattr(value, "dim"):
        return int(value.dim())
    if hasattr(value, "shape"):
        return len(tuple(value.shape))
    raise TypeError(f"cannot infer rank from {type(value)!r}")


def _numel(shape: tuple[Any, ...]) -> Optional[int]:
    if any(not isinstance(dim, int) for dim in shape):
        return None
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def _tuple_n(value: Any, rank: int, name: str) -> tuple[int, ...]:
    if isinstance(value, int):
        return (int(value),) * rank
    result = tuple(int(v) for v in value)
    if len(result) != rank:
        raise RuntimeError(f"{name} must have length {rank}, got {value}")
    return result


def _float32_spec(shape: tuple[Any, ...], device: Optional[str]) -> TensorSpec:
    return TensorSpec(
        name="",
        shape=shape,
        dtype=canonical_dtype(torch.float32),
        device=device,
    )


def _pop_operand(params: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in params:
            return params.pop(name)
    return None


__all__ = (
    "_shape_like_first",
    "_normalize_axis",
    "_rank_of",
    "_numel",
    "_tuple_n",
    "_float32_spec",
    "_pop_operand",
)
