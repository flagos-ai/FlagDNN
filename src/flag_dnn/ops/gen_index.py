# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner, triton_lang_extension as tle

_DTYPE_ALIASES = {
    "bfloat16": torch.bfloat16,
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


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("gen_index"),
    key=["n_elements", "axis_size", "inner_size"],
    strategy=["align32", "default", "default"],
    warmup=5,
    rep=10,
)
@triton.jit
def _gen_index_kernel(
    out_ptr,
    n_elements,
    axis_size: tl.constexpr,
    inner_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    axis_index = (offsets // inner_size) % axis_size
    tl.store(out_ptr + offsets, axis_index, mask=mask)


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


def _inner_size(shape: tuple[int, ...], axis: int) -> int:
    result = 1
    for dim in shape[axis + 1 :]:
        result *= int(dim)
    return result


def gen_index(
    input: torch.Tensor,
    axis: int,
    *,
    out: Optional[torch.Tensor] = None,
    compute_data_type: Any = None,
    name: str = "",
) -> torch.Tensor:
    """Generate per-element indices along ``axis`` with ``input`` shape.

    This is a real data-producing utility op, so the graph/eager path uses a
    Triton kernel instead of building an expanded framework tensor.
    """
    del name

    axis = _normalize_axis(axis, input.dim())
    dtype = _dtype_from_compute_data_type(compute_data_type)
    if out is None:
        out = torch.empty(tuple(input.shape), device=input.device, dtype=dtype)
    else:
        if tuple(out.shape) != tuple(input.shape):
            raise RuntimeError(
                f"gen_index out shape {tuple(out.shape)} does not match "
                f"input shape {tuple(input.shape)}"
            )
        if out.dtype != dtype:
            raise RuntimeError(
                f"gen_index out dtype {out.dtype} does not match requested "
                f"dtype {dtype}"
            )
        if not out.is_contiguous():
            raise NotImplementedError(
                "flag_dnn gen_index currently requires contiguous out"
            )

    n_elements = out.numel()
    if n_elements == 0:
        return out

    inner_size = _inner_size(tuple(input.shape), axis)
    axis_size = int(input.shape[axis])

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        _gen_index_kernel[grid](
            out,
            n_elements,
            axis_size,
            inner_size,
        )
    return out
