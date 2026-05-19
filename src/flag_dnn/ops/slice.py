from __future__ import annotations

import builtins
from typing import Any, Optional

import torch


def _as_slice(value: Any) -> builtins.slice:
    if isinstance(value, builtins.slice):
        return value
    if isinstance(value, (tuple, list)):
        if len(value) != 3:
            raise TypeError(
                "slice tuple specs must be (start, stop, step), "
                f"got {value}"
            )
        return builtins.slice(value[0], value[1], value[2])
    if value is None:
        return builtins.slice(None)
    raise TypeError(f"slice expects slice specs, got {type(value)!r}")


def _normalize_slices(slices: Any, ndim: int) -> tuple[builtins.slice, ...]:
    if slices is None:
        result: tuple[Any, ...] = ()
    elif isinstance(slices, builtins.slice):
        result = (slices,)
    else:
        result = tuple(slices)

    if len(result) > ndim:
        raise IndexError(
            f"too many slice specs for tensor of dimension {ndim}: "
            f"got {len(result)}"
        )
    normalized = tuple(_as_slice(item) for item in result)
    if len(normalized) < ndim:
        normalized = normalized + (builtins.slice(None),) * (
            ndim - len(normalized)
        )
    for item in normalized:
        step = 1 if item.step is None else int(item.step)
        if step <= 0:
            raise ValueError("slice step must be greater than zero")
    return normalized


def slice(
    input: torch.Tensor,
    slices: Any = (),
    *,
    out: Optional[torch.Tensor] = None,
    compute_data_type: Any = None,
    name: str = "",
) -> torch.Tensor:
    """Return ``input`` indexed by ``slices``.

    Missing dimensions default to ``slice(None)``. ``compute_data_type`` and
    ``name`` are accepted for cuDNN Frontend style graph compatibility.
    """
    del compute_data_type, name

    result = input[_normalize_slices(slices, input.dim())]
    if out is None:
        return result

    if out.shape != result.shape:
        raise RuntimeError(
            f"slice out shape {tuple(out.shape)} does not match result "
            f"shape {tuple(result.shape)}"
        )
    out.copy_(result)
    return out
