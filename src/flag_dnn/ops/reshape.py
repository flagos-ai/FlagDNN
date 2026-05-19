from typing import Any, Optional

import torch


def _normalize_shape(shape: Any) -> tuple[int, ...]:
    if shape is None:
        raise TypeError("reshape missing required argument: shape")
    if isinstance(shape, torch.Size):
        return tuple(int(dim) for dim in shape)
    if isinstance(shape, int):
        return (int(shape),)
    return tuple(int(dim) for dim in shape)


def reshape(
    input: torch.Tensor,
    shape: Any,
    *,
    out: Optional[torch.Tensor] = None,
    name: str = "",
    reshape_mode: Any = "VIEW_ONLY",
) -> torch.Tensor:
    """Reshape ``input`` to ``shape``.

    ``name`` and ``reshape_mode`` are accepted for cuDNN Frontend style graph
    compatibility. Eager execution follows ``torch.reshape`` semantics.
    """
    del name, reshape_mode

    result = torch.reshape(input, _normalize_shape(shape))
    if out is None:
        return result

    if out.shape != result.shape:
        raise RuntimeError(
            f"reshape out shape {tuple(out.shape)} does not match result "
            f"shape {tuple(result.shape)}"
        )
    out.copy_(result)
    return out
