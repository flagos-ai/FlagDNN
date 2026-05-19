from typing import Any, Optional

import torch

_DTYPE_ALIASES = {
    "boolean": torch.bool,
    "bool": torch.bool,
    "data_type.boolean": torch.bool,
    "data_type.int32": torch.int32,
    "data_type.int64": torch.int64,
    "data_type.float": torch.float32,
    "data_type.float16": torch.float16,
    "data_type.bfloat16": torch.bfloat16,
    "data_type.double": torch.float64,
    "double": torch.float64,
    "float": torch.float32,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "half": torch.float16,
    "int32": torch.int32,
    "int64": torch.int64,
    "long": torch.int64,
    "torch.bool": torch.bool,
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
}


def _normalize_axis(axis: int, ndim: int) -> int:
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise IndexError(
            f"axis out of range (expected to be in range of "
            f"[-{ndim}, {ndim - 1}], but got {axis})"
        )
    return axis


def _dtype_from_compute_data_type(compute_data_type: Any) -> torch.dtype:
    if compute_data_type is None:
        return torch.int32
    if isinstance(compute_data_type, torch.dtype):
        return compute_data_type
    key = str(compute_data_type).lower()
    if key in ("none", "not_set", "data_type.not_set"):
        return torch.int32
    if key in _DTYPE_ALIASES:
        return _DTYPE_ALIASES[key]
    tail = key.rsplit(".", 1)[-1]
    if tail in _DTYPE_ALIASES:
        return _DTYPE_ALIASES[tail]
    raise ValueError(
        f"unsupported gen_index compute_data_type: {compute_data_type}"
    )


def gen_index(
    input: torch.Tensor,
    axis: int,
    *,
    out: Optional[torch.Tensor] = None,
    compute_data_type: Any = None,
    name: str = "",
) -> torch.Tensor:
    """Generate per-element indices along ``axis`` with ``input`` shape."""
    del name

    axis = _normalize_axis(axis, input.dim())
    dtype = _dtype_from_compute_data_type(compute_data_type)
    index = torch.arange(input.shape[axis], device=input.device, dtype=dtype)
    view_shape = [1] * input.dim()
    view_shape[axis] = input.shape[axis]
    result = index.reshape(view_shape).expand(input.shape).clone()

    if out is None:
        return result

    if out.shape != result.shape:
        raise RuntimeError(
            f"gen_index out shape {tuple(out.shape)} does not match result "
            f"shape {tuple(result.shape)}"
        )
    out.copy_(result)
    return out
