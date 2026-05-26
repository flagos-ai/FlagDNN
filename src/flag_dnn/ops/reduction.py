import torch

from flag_dnn.ops.abs import abs as flag_abs
from flag_dnn.ops.mean import mean
from flag_dnn.ops.mul import mul
from flag_dnn.ops.prod import prod
from flag_dnn.ops.sqrt import sqrt
from flag_dnn.ops.sum import sum


def _mode_name(mode) -> str:
    name = getattr(mode, "name", None)
    if name is None:
        name = str(mode).rsplit(".", 1)[-1]
    return str(name).upper()


def _dims(input: torch.Tensor, dim):
    rank = input.dim()
    if dim is None:
        return tuple(range(rank))
    if isinstance(dim, int):
        return dim if dim >= 0 else dim + rank
    return tuple(item if item >= 0 else item + rank for item in dim)


def _maybe_cast_for_fallback(
    input: torch.Tensor, dtype: torch.dtype | None
) -> torch.Tensor:
    # FlagDNN does not currently provide a standalone dtype-conversion op.
    # Keep this cast confined to torch fallback paths documented below.
    return input if dtype is None else input.to(dtype)


def _prod_reduce(
    input: torch.Tensor,
    dims,
    keepdim: bool,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if isinstance(dims, int):
        return prod(input, dim=dims, keepdim=keepdim, dtype=dtype)
    dim_tuple = tuple(dims)
    result = input
    ordered_dims = dim_tuple if keepdim else sorted(dim_tuple, reverse=True)
    for index, dim in enumerate(ordered_dims):
        result = prod(
            result,
            dim=dim,
            keepdim=keepdim,
            dtype=dtype if index == 0 else None,
        )
    return result


def reduction(
    input: torch.Tensor,
    mode,
    *,
    dim=None,
    keepdim: bool = True,
    dtype: torch.dtype | None = None,
    compute_data_type=None,
    name: str = "",
) -> torch.Tensor:
    del compute_data_type, name
    mode_name = _mode_name(mode)
    dims = _dims(input, dim)

    if mode_name in ("ADD", "SUM"):
        return sum(input, dim=dim, keepdim=keepdim, dtype=dtype)
    if mode_name in ("AVG", "MEAN"):
        return mean(input, dim=dim, keepdim=keepdim, dtype=dtype)
    if mode_name in ("MUL", "PROD"):
        return _prod_reduce(input, dims, keepdim, dtype=dtype)
    if mode_name == "NORM1":
        return sum(flag_abs(input), dim=dim, keepdim=keepdim, dtype=dtype)
    if mode_name == "NORM2":
        squared = mul(input, input)
        return sqrt(sum(squared, dim=dim, keepdim=keepdim, dtype=dtype))

    work = _maybe_cast_for_fallback(input, dtype)
    fallback_dims = _dims(work, dim)
    if mode_name == "MIN":
        # TODO: replace with a FlagDNN/Triton amin reduction when available.
        return torch.amin(work, dim=fallback_dims, keepdim=keepdim)
    if mode_name == "MAX":
        # TODO: replace with a FlagDNN/Triton amax reduction when available.
        return torch.amax(work, dim=fallback_dims, keepdim=keepdim)
    if mode_name == "AMAX":
        # TODO: replace the final amax with a FlagDNN/Triton reduction.
        return torch.amax(
            flag_abs(work), dim=fallback_dims, keepdim=keepdim
        )
    if mode_name == "MUL_NO_ZEROS":
        # TODO: replace torch.where after FlagDNN has a tensor where/select op.
        nonzero = torch.where(
            work == 0,
            torch.ones((), device=work.device, dtype=work.dtype),
            work,
        )
        return _prod_reduce(nonzero, fallback_dims, keepdim)
    raise NotImplementedError(
        f"flag_dnn reduction does not support mode={mode}"
    )
