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

from typing import Optional, Sequence

import torch
import triton
import triton.language as tl

from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import triton_lang_extension as tle

_BLOCK_SIZE = 256
_MAX_DIMS = 6


@triton.jit
def _axis_coord(
    i0,
    i1,
    i2,
    i3,
    i4,
    i5,
    AXIS: tl.constexpr,
):
    coord = i0
    if AXIS == 1:
        coord = i1
    elif AXIS == 2:
        coord = i2
    elif AXIS == 3:
        coord = i3
    elif AXIS == 4:
        coord = i4
    elif AXIS == 5:
        coord = i5
    return coord


@triton.jit
def _input_offset(
    i0,
    i1,
    i2,
    i3,
    i4,
    i5,
    axis_index,
    AXIS: tl.constexpr,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
):
    off = i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3 + i4 * s4 + i5 * s5
    if AXIS == 0:
        off += (axis_index - i0) * s0
    elif AXIS == 1:
        off += (axis_index - i1) * s1
    elif AXIS == 2:
        off += (axis_index - i2) * s2
    elif AXIS == 3:
        off += (axis_index - i3) * s3
    elif AXIS == 4:
        off += (axis_index - i4) * s4
    elif AXIS == 5:
        off += (axis_index - i5) * s5
    return off


@triton.jit
def _concat2_kernel(
    x0,
    x1,
    out,
    n_elements,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    x0_axis,
    sx00,
    sx01,
    sx02,
    sx03,
    sx04,
    sx05,
    sx10,
    sx11,
    sx12,
    sx13,
    sx14,
    sx15,
    AXIS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    i5 = offsets % d5
    rem = offsets // d5
    i4 = rem % d4
    rem = rem // d4
    i3 = rem % d3
    rem = rem // d3
    i2 = rem % d2
    rem = rem // d2
    i1 = rem % d1
    i0 = rem // d1

    axis_out = _axis_coord(i0, i1, i2, i3, i4, i5, AXIS)
    use0 = axis_out < x0_axis
    off0 = _input_offset(
        i0,
        i1,
        i2,
        i3,
        i4,
        i5,
        axis_out,
        AXIS,
        sx00,
        sx01,
        sx02,
        sx03,
        sx04,
        sx05,
    )
    off1 = _input_offset(
        i0,
        i1,
        i2,
        i3,
        i4,
        i5,
        axis_out - x0_axis,
        AXIS,
        sx10,
        sx11,
        sx12,
        sx13,
        sx14,
        sx15,
    )
    v0 = tl.load(x0 + off0, mask=mask & use0, other=0.0)
    v1 = tl.load(x1 + off1, mask=mask & (~use0), other=0.0)
    values = tl.where(use0, v0, v1)
    tl.store(out + offsets, values, mask=mask)


@triton.jit
def _concat3_kernel(
    x0,
    x1,
    x2,
    out,
    n_elements,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    x0_axis,
    x1_axis_end,
    sx00,
    sx01,
    sx02,
    sx03,
    sx04,
    sx05,
    sx10,
    sx11,
    sx12,
    sx13,
    sx14,
    sx15,
    sx20,
    sx21,
    sx22,
    sx23,
    sx24,
    sx25,
    AXIS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    i5 = offsets % d5
    rem = offsets // d5
    i4 = rem % d4
    rem = rem // d4
    i3 = rem % d3
    rem = rem // d3
    i2 = rem % d2
    rem = rem // d2
    i1 = rem % d1
    i0 = rem // d1

    axis_out = _axis_coord(i0, i1, i2, i3, i4, i5, AXIS)
    use0 = axis_out < x0_axis
    use1 = (axis_out >= x0_axis) & (axis_out < x1_axis_end)
    use2 = axis_out >= x1_axis_end
    off0 = _input_offset(
        i0,
        i1,
        i2,
        i3,
        i4,
        i5,
        axis_out,
        AXIS,
        sx00,
        sx01,
        sx02,
        sx03,
        sx04,
        sx05,
    )
    off1 = _input_offset(
        i0,
        i1,
        i2,
        i3,
        i4,
        i5,
        axis_out - x0_axis,
        AXIS,
        sx10,
        sx11,
        sx12,
        sx13,
        sx14,
        sx15,
    )
    off2 = _input_offset(
        i0,
        i1,
        i2,
        i3,
        i4,
        i5,
        axis_out - x1_axis_end,
        AXIS,
        sx20,
        sx21,
        sx22,
        sx23,
        sx24,
        sx25,
    )
    v0 = tl.load(x0 + off0, mask=mask & use0, other=0.0)
    v1 = tl.load(x1 + off1, mask=mask & use1, other=0.0)
    v2 = tl.load(x2 + off2, mask=mask & use2, other=0.0)
    values = tl.where(use0, v0, tl.where(use1, v1, v2))
    tl.store(out + offsets, values, mask=mask)


def _normalize_axis(axis: int, rank: int) -> int:
    axis = int(axis)
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise IndexError(
            f"axis out of range (expected to be in range of "
            f"[-{rank}, {rank - 1}], but got {axis})"
        )
    return axis


def _pad(values: Sequence[int]) -> tuple[int, ...]:
    if len(values) > _MAX_DIMS:
        raise NotImplementedError(
            f"flag_dnn concatenate supports rank <= {_MAX_DIMS}"
        )
    return (0,) * (_MAX_DIMS - len(values)) + tuple(int(v) for v in values)


def _pad_shape(values: Sequence[int]) -> tuple[int, ...]:
    if len(values) > _MAX_DIMS:
        raise NotImplementedError(
            f"flag_dnn concatenate supports rank <= {_MAX_DIMS}"
        )
    return (1,) * (_MAX_DIMS - len(values)) + tuple(int(v) for v in values)


def _validate_inputs(
    inputs: Sequence[torch.Tensor], axis: int
) -> tuple[int, tuple[int, ...]]:
    if not inputs:
        raise RuntimeError("concatenate expects a non-empty input sequence")
    if len(inputs) > 3:
        raise NotImplementedError(
            "flag_dnn concatenate currently supports up to 3 inputs"
        )

    first = inputs[0]
    rank = first.dim()
    axis = _normalize_axis(axis, rank)
    dtype = first.dtype
    device = first.device
    out_shape = list(first.shape)
    out_axis = int(first.shape[axis])

    for item in inputs:
        if item.dim() != rank:
            raise RuntimeError("concatenate inputs must have the same rank")
        if item.dtype != dtype:
            raise RuntimeError("concatenate inputs must have the same dtype")
        if item.device != device:
            raise RuntimeError("concatenate inputs must be on the same device")
        for dim, (expected, actual) in enumerate(zip(first.shape, item.shape)):
            if dim == axis:
                continue
            if int(expected) != int(actual):
                raise RuntimeError(
                    "concatenate non-axis dimensions must match: "
                    f"{tuple(first.shape)} vs {tuple(item.shape)}"
                )
    for item in inputs[1:]:
        out_axis += int(item.shape[axis])
    out_shape[axis] = out_axis
    return axis, tuple(out_shape)


def concatenate(
    inputs: Sequence[torch.Tensor],
    axis: int,
    *,
    out: Optional[torch.Tensor] = None,
    in_place_index: Optional[int] = None,
    name: str = "",
) -> torch.Tensor:
    """Concatenate tensors along ``axis`` using a Triton data-move kernel."""
    del in_place_index, name

    inputs = tuple(inputs)
    axis, out_shape = _validate_inputs(inputs, axis)
    first = inputs[0]
    if len(inputs) == 1 and out is None:
        return first

    if out is None:
        out = torch.empty(out_shape, dtype=first.dtype, device=first.device)
    else:
        if tuple(out.shape) != out_shape:
            raise RuntimeError(
                f"concatenate out shape {tuple(out.shape)} does not match "
                f"result shape {out_shape}"
            )
        if out.dtype != first.dtype:
            raise RuntimeError(
                f"concatenate out dtype {out.dtype} does not match input "
                f"dtype {first.dtype}"
            )
        if out.device != first.device:
            raise RuntimeError("concatenate out must be on the input device")
        if not out.is_contiguous():
            raise NotImplementedError(
                "flag_dnn concatenate currently requires contiguous out"
            )

    n_elements = out.numel()
    if n_elements == 0:
        return out

    rank = first.dim()
    axis6 = axis + (_MAX_DIMS - rank)
    padded_shape = _pad_shape(out_shape)
    padded_strides = [_pad(item.stride()) for item in inputs]
    axis_sizes = [int(item.shape[axis]) for item in inputs]

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(first.device):
        if len(inputs) == 2:
            _concat2_kernel[grid](
                inputs[0],
                inputs[1],
                out,
                n_elements,
                *padded_shape,
                axis_sizes[0],
                *padded_strides[0],
                *padded_strides[1],
                AXIS=axis6,
                BLOCK_SIZE=_BLOCK_SIZE,
                num_warps=4,
            )
        elif len(inputs) == 3:
            _concat3_kernel[grid](
                inputs[0],
                inputs[1],
                inputs[2],
                out,
                n_elements,
                *padded_shape,
                axis_sizes[0],
                axis_sizes[0] + axis_sizes[1],
                *padded_strides[0],
                *padded_strides[1],
                *padded_strides[2],
                AXIS=axis6,
                BLOCK_SIZE=_BLOCK_SIZE,
                num_warps=4,
            )
        else:
            _concat2_kernel[grid](
                first,
                first,
                out,
                n_elements,
                *padded_shape,
                axis_sizes[0],
                *padded_strides[0],
                *padded_strides[0],
                AXIS=axis6,
                BLOCK_SIZE=_BLOCK_SIZE,
                num_warps=4,
            )
    return out
