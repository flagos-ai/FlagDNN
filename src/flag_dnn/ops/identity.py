from typing import Any, Optional

import torch


def identity(
    input: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    compute_data_type: Any = None,
    name: str = "",
) -> torch.Tensor:
    """Return a tensor with the same value as ``input``.

    ``compute_data_type`` and ``name`` are accepted for cuDNN Frontend style
    graph compatibility. They do not affect eager identity semantics.
    """
    del compute_data_type, name

    if out is None:
        return input.clone()

    if out.shape != input.shape:
        raise RuntimeError(
            f"identity out shape {tuple(out.shape)} does not match input "
            f"shape {tuple(input.shape)}"
        )
    out.copy_(input)
    return out
