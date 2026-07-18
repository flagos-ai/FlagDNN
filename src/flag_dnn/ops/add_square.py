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
    collapse_dims,
    empty_like_preserve_dense_layout,
    has_same_dense_flat_layout,
    pad_to_max_dims,
)
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


def _compute_uses_float32(compute_data_type) -> bool:
    if compute_data_type is None:
        return False
    if compute_data_type is torch.float32:
        return True
    if isinstance(compute_data_type, str):
        return compute_data_type.lower() in {"float", "float32", "fp32"}
    return str(compute_data_type).lower().endswith("float")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("add_square"),
    key=["n_elements", "COMPUTE_FLOAT32"],
    strategy=["align32", "default"],
    warmup=5,
    rep=10,
)
@triton.jit
def add_square_tensor_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    COMPUTE_FLOAT32: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
):
    pid = tle.program_id(0)
    base = pid * BLOCK_SIZE * TILES_PER_PROGRAM
    for i in tl.static_range(TILES_PER_PROGRAM):
        offsets = base + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        if COMPUTE_FLOAT32:
            a = a.to(tl.float32)
            b = b.to(tl.float32)
        result = a + b * b
        tl.store(
            out_ptr + offsets,
            result.to(out_ptr.dtype.element_ty),
            mask=mask,
        )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("add_square"),
    key=["n_elements", "COMPUTE_FLOAT32"],
    strategy=["align32", "default"],
    warmup=5,
    rep=10,
)
@triton.jit
def add_square_broadcast_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    s1,
    s2,
    s3,
    s4,
    s5,
    sa0,
    sa1,
    sa2,
    sa3,
    sa4,
    sa5,
    sb0,
    sb1,
    sb2,
    sb3,
    sb4,
    sb5,
    COMPUTE_FLOAT32: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
):
    pid = tle.program_id(0)
    base = pid * BLOCK_SIZE * TILES_PER_PROGRAM
    for i in tl.static_range(TILES_PER_PROGRAM):
        offsets = base + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

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

        a_offsets = (
            idx0 * sa0
            + idx1 * sa1
            + idx2 * sa2
            + idx3 * sa3
            + idx4 * sa4
            + idx5 * sa5
        )
        b_offsets = (
            idx0 * sb0
            + idx1 * sb1
            + idx2 * sb2
            + idx3 * sb3
            + idx4 * sb4
            + idx5 * sb5
        )

        a = tl.load(a_ptr + a_offsets, mask=mask)
        b = tl.load(b_ptr + b_offsets, mask=mask)
        if COMPUTE_FLOAT32:
            a = a.to(tl.float32)
            b = b.to(tl.float32)
        result = a + b * b
        tl.store(
            out_ptr + offsets,
            result.to(out_ptr.dtype.element_ty),
            mask=mask,
        )


def add_square(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    compute_data_type=None,
    name: str = "",
) -> torch.Tensor:
    del name
    logger.debug("FLAG_DNN ADD_SQUARE")

    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("add_square expects tensor inputs")
    if a.device != b.device:
        raise RuntimeError(
            f"Expected add_square inputs on the same device, got {a.device} "
            f"and {b.device}"
        )
    if not b.dtype.is_floating_point:
        raise RuntimeError(f"add_square is not implemented for '{b.dtype}'")

    out_shape = torch.broadcast_shapes(tuple(a.shape), tuple(b.shape))
    out_dtype = out.dtype if out is not None else torch.result_type(a, b)

    flat_layout_source = None
    if has_same_dense_flat_layout(a, b):
        flat_layout_source = a

    if out is None:
        if flat_layout_source is not None and tuple(out_shape) == tuple(
            flat_layout_source.shape
        ):
            out = empty_like_preserve_dense_layout(
                flat_layout_source, out_dtype
            )
        else:
            out = torch.empty(out_shape, dtype=out_dtype, device=a.device)
    else:
        assert tuple(out.shape) == tuple(out_shape), (
            f"out shape {out.shape} mismatch with broadcasted shape "
            f"{out_shape}"
        )

    n_elements = out.numel()
    if n_elements == 0:
        return out

    compute_float32 = _compute_uses_float32(compute_data_type)

    def grid(meta):
        return (
            triton.cdiv(
                n_elements, meta["BLOCK_SIZE"] * meta["TILES_PER_PROGRAM"]
            ),
        )

    with torch_device_fn.device(a.device):
        if flat_layout_source is not None and can_use_flat_output(
            out, flat_layout_source
        ):
            add_square_tensor_kernel[grid](
                a,
                b,
                out,
                n_elements,
                COMPUTE_FLOAT32=compute_float32,
            )
        else:
            if not out.is_contiguous():
                raise NotImplementedError(
                    "flag_dnn add_square broadcast currently requires "
                    "contiguous out unless all operands share a dense flat "
                    "layout"
                )
            a_exp = a.expand(out_shape)
            b_exp = b.expand(out_shape)
            c_shape, c_sa, c_sb = collapse_dims(
                out_shape, a_exp.stride(), b_exp.stride()
            )
            f_shape, f_sa, f_sb = pad_to_max_dims(
                c_shape, c_sa, c_sb, max_dims=6
            )

            add_square_broadcast_kernel[grid](
                a,
                b,
                out,
                n_elements,
                *f_shape[1:],
                *f_sa,
                *f_sb,
                COMPUTE_FLOAT32=compute_float32,
            )

    return out
