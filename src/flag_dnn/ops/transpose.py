from typing import Any, Optional

import torch


def _normalize_dim(dim: int, ndim: int) -> int:
    dim = int(dim)
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise IndexError(
            f"dimension out of range (expected to be in range of "
            f"[-{ndim}, {ndim - 1}], but got {dim})"
        )
    return dim


def _normalize_permutation(permutation: Any, ndim: int) -> tuple[int, ...]:
    permutation = tuple(int(dim) for dim in permutation)
    if len(permutation) != ndim:
        raise RuntimeError(
            f"transpose permutation length {len(permutation)} does not match "
            f"input rank {ndim}"
        )
    normalized = tuple(_normalize_dim(dim, ndim) for dim in permutation)
    if len(set(normalized)) != ndim:
        raise RuntimeError(
            f"transpose permutation must contain each dimension once, "
            f"got {permutation}"
        )
    return normalized


def _swap_permutation(ndim: int, dim0: int, dim1: int) -> tuple[int, ...]:
    dim0 = _normalize_dim(dim0, ndim)
    dim1 = _normalize_dim(dim1, ndim)
    permutation = list(range(ndim))
    permutation[dim0], permutation[dim1] = permutation[dim1], permutation[dim0]
    return tuple(permutation)


def transpose(
    input: torch.Tensor,
    permutation: Any,
    dim1: Optional[int] = None,
    *,
    out: Optional[torch.Tensor] = None,
    compute_data_type: Any = None,
    name: str = "",
) -> torch.Tensor:
    """Permute ``input`` dimensions.

    ``permutation`` matches cuDNN Frontend's graph API. Passing ``dim1`` also
    supports the familiar two-dimension swap form.
    """
    del compute_data_type, name

    if dim1 is None:
        dims = _normalize_permutation(permutation, input.dim())
    else:
        dims = _swap_permutation(input.dim(), int(permutation), int(dim1))

    result = input.permute(dims)
    if out is None:
        return result

    raise NotImplementedError(
        "flag_dnn transpose is a view-only graph utility; materialized "
        "transpose(out=...) requires a dedicated Triton layout-copy kernel"
    )
