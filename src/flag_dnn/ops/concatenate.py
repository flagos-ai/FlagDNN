from typing import Any, Optional, Sequence

import torch


def concatenate(
    inputs: Sequence[torch.Tensor],
    axis: int,
    *,
    out: Optional[torch.Tensor] = None,
    in_place_index: Optional[int] = None,
    name: str = "",
) -> torch.Tensor:
    """Concatenate tensors along ``axis``.

    ``in_place_index`` and ``name`` are accepted for cuDNN Frontend style graph
    compatibility. They do not affect eager concatenation semantics.
    """
    del in_place_index, name

    if not inputs:
        raise RuntimeError("concatenate expects a non-empty input sequence")

    result = torch.cat(tuple(inputs), dim=int(axis))
    if out is None:
        return result

    if out.shape != result.shape:
        raise RuntimeError(
            f"concatenate out shape {tuple(out.shape)} does not match result "
            f"shape {tuple(result.shape)}"
        )
    out.copy_(result)
    return out
