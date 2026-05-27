from typing import Any, Optional

import torch

from flag_dnn.ops.identity import _copy_dense_flat


def _normalize_shape(shape: Any) -> tuple[int, ...]:
    if shape is None:
        raise TypeError("reshape missing required argument: shape")
    if isinstance(shape, torch.Size):
        return tuple(int(dim) for dim in shape)
    if isinstance(shape, int):
        return (int(shape),)
    return tuple(int(dim) for dim in shape)


def _view_only(input: torch.Tensor, shape: Any) -> torch.Tensor:
    target_shape = _normalize_shape(shape)
    try:
        return input.view(target_shape)
    except RuntimeError as exc:
        raise NotImplementedError(
            "flag_dnn reshape is a view-only graph utility; materializing "
            "non-view reshapes is not enabled"
        ) from exc


def reshape(
    input: torch.Tensor,
    shape: Any,
    *,
    out: Optional[torch.Tensor] = None,
    name: str = "",
    reshape_mode: Any = "VIEW_ONLY",
) -> torch.Tensor:
    """Reshape ``input`` as a graph utility view.

    The no-``out`` path is intentionally view-only.  It does not use torch's
    materializing ``reshape`` fallback.  ``out`` requests
    materialization and is
    limited to dense contiguous views copied by a Triton kernel.
    """
    del name, reshape_mode

    result = _view_only(input, shape)
    if out is None:
        return result

    if out.shape != result.shape:
        raise RuntimeError(
            f"reshape out shape {tuple(out.shape)} does not match result "
            f"shape {tuple(result.shape)}"
        )
    if out.dtype != result.dtype:
        raise RuntimeError(
            f"reshape out dtype {out.dtype} does not match result dtype "
            f"{result.dtype}"
        )
    return _copy_dense_flat(result, out)
