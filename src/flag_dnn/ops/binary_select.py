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

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.ops.binary import (
    can_use_flat_output,
    empty_like_preserve_dense_layout,
    has_same_dense_flat_layout,
)
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


def _same_dense_flat_layout(
    input0: torch.Tensor, input1: torch.Tensor, mask: torch.Tensor
) -> bool:
    return has_same_dense_flat_layout(
        input0, input1
    ) and has_same_dense_flat_layout(input0, mask)


def collapse_dims_ternary(shape, strides_x, strides_y, strides_mask):
    if not shape:
        return [1], [0], [0], [0]

    c_shape, c_sx, c_sy, c_sm = [], [], [], []
    for i in reversed(range(len(shape))):
        size = shape[i]
        if size == 1:
            continue
        if not c_shape:
            c_shape.append(size)
            c_sx.append(strides_x[i])
            c_sy.append(strides_y[i])
            c_sm.append(strides_mask[i])
            continue

        prev_shape = c_shape[-1]
        is_contig_x = strides_x[i] == c_sx[-1] * prev_shape
        is_contig_y = strides_y[i] == c_sy[-1] * prev_shape
        is_contig_m = strides_mask[i] == c_sm[-1] * prev_shape
        if is_contig_x and is_contig_y and is_contig_m:
            c_shape[-1] *= size
        else:
            c_shape.append(size)
            c_sx.append(strides_x[i])
            c_sy.append(strides_y[i])
            c_sm.append(strides_mask[i])

    if not c_shape:
        return [1], [0], [0], [0]
    return c_shape[::-1], c_sx[::-1], c_sy[::-1], c_sm[::-1]


def pad_to_max_dims_ternary(
    shape, strides_x, strides_y, strides_mask, max_dims=6
):
    shape = list(shape)
    strides_x = list(strides_x)
    strides_y = list(strides_y)
    strides_mask = list(strides_mask)

    if len(shape) > max_dims:
        raise RuntimeError(f"Collapsed binary_select rank exceeds {max_dims}")

    while len(shape) < max_dims:
        shape.insert(0, 1)
        strides_x.insert(0, 0)
        strides_y.insert(0, 0)
        strides_mask.insert(0, 0)

    return shape, strides_x, strides_y, strides_mask


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("binary"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def binary_select_tensor_kernel(
    input0_ptr,
    input1_ptr,
    mask_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = offsets < n_elements

    input0 = tl.load(input0_ptr + offsets, mask=active)
    input1 = tl.load(input1_ptr + offsets, mask=active)
    mask_value = tl.load(mask_ptr + offsets, mask=active, other=0)
    result = tl.where(mask_value != 0, input0, input1)
    tl.store(
        out_ptr + offsets, result.to(out_ptr.dtype.element_ty), mask=active
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("binary"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def binary_select_broadcast_kernel(
    input0_ptr,
    input1_ptr,
    mask_ptr,
    out_ptr,
    n_elements,
    s1,
    s2,
    s3,
    s4,
    s5,
    sx0,
    sx1,
    sx2,
    sx3,
    sx4,
    sx5,
    sy0,
    sy1,
    sy2,
    sy3,
    sy4,
    sy5,
    sm0,
    sm1,
    sm2,
    sm3,
    sm4,
    sm5,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = offsets < n_elements

    idx5 = offsets % s5
    rem4 = offsets // s5

    idx4 = rem4 % s4
    rem3 = rem4 // s4

    idx3 = rem3 % s3
    rem2 = rem3 // s3

    idx2 = rem2 % s2
    rem1 = rem2 // s2

    idx1 = rem1 % s1
    idx0 = rem1 // s1

    input0_off = (
        idx0 * sx0
        + idx1 * sx1
        + idx2 * sx2
        + idx3 * sx3
        + idx4 * sx4
        + idx5 * sx5
    )
    input1_off = (
        idx0 * sy0
        + idx1 * sy1
        + idx2 * sy2
        + idx3 * sy3
        + idx4 * sy4
        + idx5 * sy5
    )
    mask_off = (
        idx0 * sm0
        + idx1 * sm1
        + idx2 * sm2
        + idx3 * sm3
        + idx4 * sm4
        + idx5 * sm5
    )

    input0 = tl.load(input0_ptr + input0_off, mask=active)
    input1 = tl.load(input1_ptr + input1_off, mask=active)
    mask_value = tl.load(mask_ptr + mask_off, mask=active, other=0)
    result = tl.where(mask_value != 0, input0, input1)
    tl.store(
        out_ptr + offsets, result.to(out_ptr.dtype.element_ty), mask=active
    )


def _validate_binary_select_args(
    input0: torch.Tensor, input1: torch.Tensor, mask: torch.Tensor
) -> None:
    if not isinstance(input0, torch.Tensor):
        raise TypeError("binary_select input0 must be a tensor")
    if not isinstance(input1, torch.Tensor):
        raise TypeError("binary_select input1 must be a tensor")
    if not isinstance(mask, torch.Tensor):
        raise TypeError("binary_select mask must be a tensor")
    if mask.dtype.is_complex:
        raise RuntimeError("binary_select mask must be bool or numeric")
    if input0.device != input1.device or input0.device != mask.device:
        raise RuntimeError("binary_select inputs must be on the same device")


def _binary_select_impl(
    input0: torch.Tensor,
    input1: torch.Tensor,
    mask: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logger.debug("FLAG_DNN BINARY_SELECT")
    _validate_binary_select_args(input0, input1, mask)

    out_shape = torch.broadcast_shapes(
        tuple(input0.shape), tuple(input1.shape), tuple(mask.shape)
    )
    out_dtype = torch.result_type(input0, input1)

    flat_layout_source = None
    if _same_dense_flat_layout(input0, input1, mask):
        flat_layout_source = input0

    if out is None:
        if flat_layout_source is not None and tuple(out_shape) == tuple(
            flat_layout_source.shape
        ):
            out = empty_like_preserve_dense_layout(
                flat_layout_source, out_dtype
            )
        else:
            out = torch.empty(out_shape, dtype=out_dtype, device=input0.device)
    else:
        assert tuple(out.shape) == tuple(out_shape), (
            f"out shape {out.shape} mismatch with broadcasted shape "
            f"{out_shape}"
        )

    n_elements = out.numel()
    if n_elements == 0:
        return out

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input0.device):
        if flat_layout_source is not None and can_use_flat_output(
            out, flat_layout_source
        ):
            binary_select_tensor_kernel[grid](
                input0, input1, mask, out, n_elements
            )
        else:
            if not out.is_contiguous():
                raise NotImplementedError(
                    "flag_dnn binary_select broadcast currently requires "
                    "contiguous out unless all operands share a dense flat "
                    "layout"
                )
            input0_exp = input0.expand(out_shape)
            input1_exp = input1.expand(out_shape)
            mask_exp = mask.expand(out_shape)

            c_shape, c_sx, c_sy, c_sm = collapse_dims_ternary(
                out_shape,
                input0_exp.stride(),
                input1_exp.stride(),
                mask_exp.stride(),
            )
            f_shape, f_sx, f_sy, f_sm = pad_to_max_dims_ternary(
                c_shape, c_sx, c_sy, c_sm, max_dims=6
            )

            binary_select_broadcast_kernel[grid](
                input0,
                input1,
                mask,
                out,
                n_elements,
                *f_shape[1:],
                *f_sx,
                *f_sy,
                *f_sm,
            )

    return out


def binary_select(
    input0: torch.Tensor | None = None,
    input1: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    *,
    out: Optional[torch.Tensor] = None,
    compute_data_type=None,
    name: str = "",
) -> torch.Tensor:
    del compute_data_type, name
    if input0 is None:
        raise TypeError("binary_select missing input0 tensor")
    if input1 is None:
        raise TypeError("binary_select missing input1 tensor")
    if mask is None:
        raise TypeError("binary_select missing mask tensor")
    return _binary_select_impl(input0, input1, mask, out=out)


def where(
    condition: torch.Tensor | None = None,
    input: torch.Tensor | None = None,
    other: torch.Tensor | None = None,
    *,
    out: Optional[torch.Tensor] = None,
    compute_data_type=None,
    name: str = "",
) -> torch.Tensor:
    del compute_data_type, name
    if condition is None:
        raise TypeError("where missing condition tensor")
    if input is None:
        raise TypeError("where missing input tensor")
    if other is None:
        raise TypeError("where missing other tensor")
    return _binary_select_impl(input, other, condition, out=out)
