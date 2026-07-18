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
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils.triton_lang_helper import tl_extra_shim as libdevice
from flag_dnn.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


def is_dense_flat_tensor(tensor: torch.Tensor) -> bool:
    return tensor.is_contiguous() or (
        tensor.dim() == 4
        and tensor.is_contiguous(memory_format=torch.channels_last)
    )


def can_use_flat_output(output: torch.Tensor, source: torch.Tensor) -> bool:
    return (
        tuple(output.shape) == tuple(source.shape)
        and is_dense_flat_tensor(output)
        and tuple(output.stride()) == tuple(source.stride())
    )


def empty_like_preserve_dense_layout(
    source: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    return torch.empty_like(source, dtype=dtype)


def collapse_dims_unary(shape, strides):
    if not shape:
        return [1], [0]

    c_shape, c_str = [], []

    for i in reversed(range(len(shape))):
        s = shape[i]

        if s == 1:
            continue

        if not c_shape:
            c_shape.append(s)
            c_str.append(strides[i])
        else:
            prev_shape = c_shape[-1]
            is_contig = strides[i] == c_str[-1] * prev_shape

            if is_contig:
                c_shape[-1] *= s
            else:
                c_shape.append(s)
                c_str.append(strides[i])

    if not c_shape:
        return [1], [0]

    return c_shape[::-1], c_str[::-1]


def pad_to_max_dims_unary(shape, strides, max_dims=6):
    shape = list(shape)
    strides = list(strides)

    if len(shape) > max_dims:
        raise RuntimeError(f"坍缩后依然超过 {max_dims} 维，Not Support.")

    while len(shape) < max_dims:
        shape.insert(0, 1)
        strides.insert(0, 0)

    return shape, strides


@libentry()
@triton.jit
def unary_fill_false_kernel(
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(
        out_ptr + offsets, tl.zeros([BLOCK_SIZE], dtype=tl.int1), mask=mask
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("unary"),
    key=["n_elements", "OP_TYPE"],
    strategy=["align32", "default"],
    warmup=5,
    rep=10,
)
@triton.jit
def unary_contiguous_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    OP_TYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
):
    pid = tle.program_id(0)
    base = pid * BLOCK_SIZE * TILES_PER_PROGRAM
    for i in tl.static_range(TILES_PER_PROGRAM):
        offsets = base + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)

        if OP_TYPE == "isinf":
            res = (x == float("inf")) | (x == float("-inf"))
        elif OP_TYPE == "isnan":
            res = ~(x == x)
        elif OP_TYPE == "square":
            res = x * x
        elif OP_TYPE == "rsqrt":
            res = tl.math.rsqrt(x.to(tl.float32)).to(x.dtype)
        elif OP_TYPE == "positive":
            res = x
        elif OP_TYPE == "log":
            res = (tl.math.log2(x.to(tl.float32)) * 0.6931471805599453).to(
                x.dtype
            )
        elif OP_TYPE == "exp":
            res = tl.math.exp2(x.to(tl.float32) * 1.4426950408889634).to(
                x.dtype
            )
        elif OP_TYPE == "reciprocal":
            res = (1.0 / x.to(tl.float32)).to(x.dtype)
        elif OP_TYPE == "ceil":
            res = tl.math.ceil(x.to(tl.float32)).to(x.dtype)
        elif OP_TYPE == "floor":
            res = tl.math.floor(x.to(tl.float32)).to(x.dtype)
        elif OP_TYPE == "erf":
            res = tl.math.erf(x.to(tl.float32)).to(x.dtype)
        elif OP_TYPE == "sin":
            res = tl.math.sin(x.to(tl.float32)).to(x.dtype)
        elif OP_TYPE == "cos":
            res = tl.math.cos(x.to(tl.float32)).to(x.dtype)
        elif OP_TYPE == "tan":
            res = libdevice.tan(x.to(tl.float32)).to(x.dtype)
        elif OP_TYPE == "bitwise_not":
            res = ~x

        tl.store(
            out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask
        )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("unary"),
    key=["n_elements", "OP_TYPE"],
    strategy=["align32", "default"],
    warmup=5,
    rep=10,
)
@triton.jit
def unary_strided_kernel(
    x_ptr,
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
    OP_TYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
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

        x_off = (
            idx0 * sx0
            + idx1 * sx1
            + idx2 * sx2
            + idx3 * sx3
            + idx4 * sx4
            + idx5 * sx5
        )

        x = tl.load(x_ptr + x_off, mask=mask)

        if OP_TYPE == "isinf":
            res = (x == float("inf")) | (x == float("-inf"))
        elif OP_TYPE == "isnan":
            res = ~(x == x)
        elif OP_TYPE == "square":
            res = x * x
        elif OP_TYPE == "rsqrt":
            res = tl.math.rsqrt(x.to(tl.float32)).to(x.dtype)
        elif OP_TYPE == "positive":
            res = x
        elif OP_TYPE == "log":
            res = (tl.math.log2(x.to(tl.float32)) * 0.6931471805599453).to(
                x.dtype
            )
        elif OP_TYPE == "exp":
            res = tl.math.exp2(x.to(tl.float32) * 1.4426950408889634).to(
                x.dtype
            )
        elif OP_TYPE == "reciprocal":
            res = (1.0 / x.to(tl.float32)).to(x.dtype)
        elif OP_TYPE == "ceil":
            res = tl.math.ceil(x.to(tl.float32)).to(x.dtype)
        elif OP_TYPE == "floor":
            res = tl.math.floor(x.to(tl.float32)).to(x.dtype)
        elif OP_TYPE == "erf":
            res = tl.math.erf(x.to(tl.float32)).to(x.dtype)
        elif OP_TYPE == "sin":
            res = tl.math.sin(x.to(tl.float32)).to(x.dtype)
        elif OP_TYPE == "cos":
            res = tl.math.cos(x.to(tl.float32)).to(x.dtype)
        elif OP_TYPE == "tan":
            res = libdevice.tan(x.to(tl.float32)).to(x.dtype)
        elif OP_TYPE == "bitwise_not":
            res = ~x

        tl.store(
            out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask
        )


_FLOAT_OPS = {
    "rsqrt",
    "log",
    "exp",
    "reciprocal",
    "ceil",
    "floor",
    "erf",
    "sin",
    "cos",
    "tan",
    "square",
}
_INT_OPS = {"bitwise_not"}


def unary(
    input: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    op_type: str = "",
) -> torch.Tensor:
    logger.debug(f"FLAG_DNN {op_type.upper()}")

    # 输入合法性校验
    if op_type == "positive":
        if input.dtype == torch.bool:
            raise RuntimeError("torch.positive does not support bool tensors.")
        if out is None:
            return input
        else:
            # 用户传了 out，遵守语义把数据复制过去
            out.copy_(input)
            return out
    elif op_type in _INT_OPS:
        if input.dtype.is_floating_point:
            raise RuntimeError(
                f"{op_type} is not implemented for '{input.dtype}'"
            )
    elif op_type in _FLOAT_OPS:
        if not input.dtype.is_floating_point:
            raise RuntimeError(
                f"{op_type} is not implemented for '{input.dtype}'"
            )

    # 输出 dtype
    if op_type in ("isinf", "isnan"):
        out_dtype = torch.bool
    else:
        out_dtype = input.dtype

    if out is None:
        if is_dense_flat_tensor(input):
            out = empty_like_preserve_dense_layout(input, out_dtype)
        else:
            out = torch.empty(
                input.shape, dtype=out_dtype, device=input.device
            )

    n_elements = out.numel()
    if n_elements == 0:
        return out

    # 整数/bool 类型的 isnan / isinf 结果恒为 False，跳过 kernel
    if op_type in ("isnan", "isinf") and not input.dtype.is_floating_point:
        BLOCK_SIZE = 1024
        with torch_device_fn.device(input.device):
            unary_fill_false_kernel[(triton.cdiv(n_elements, BLOCK_SIZE),)](
                out, n_elements, BLOCK_SIZE=BLOCK_SIZE
            )
        return out

    def grid(meta):
        return (
            triton.cdiv(
                n_elements, meta["BLOCK_SIZE"] * meta["TILES_PER_PROGRAM"]
            ),
        )

    with torch_device_fn.device(input.device):
        if is_dense_flat_tensor(input) and can_use_flat_output(out, input):
            unary_contiguous_kernel[grid](
                input,
                out,
                n_elements,
                OP_TYPE=op_type,
            )
        else:
            if not out.is_contiguous():
                raise NotImplementedError(
                    "flag_dnn unary strided currently requires contiguous out "
                    "unless input and out share a dense flat layout"
                )
            c_shape, c_sx = collapse_dims_unary(input.shape, input.stride())
            f_shape, f_sx = pad_to_max_dims_unary(c_shape, c_sx, max_dims=6)

            unary_strided_kernel[grid](
                input,
                out,
                n_elements,
                *f_shape[1:],  # s1 到 s5
                *f_sx,  # sx0 到 sx5
                OP_TYPE=op_type,
            )

    return out
