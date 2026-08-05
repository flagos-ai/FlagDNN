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

"""NVIDIA binary kernels: Native ABI baseline plus optimized variants."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

# Stable FlagDNN pointwise mode values. These are FlagDNN-owned semantic IDs,
# not platform-library enum values.
POINTWISE_ADD = tl.constexpr(1)
POINTWISE_SUB = tl.constexpr(17)
POINTWISE_MUL = tl.constexpr(18)
POINTWISE_DIV = tl.constexpr(19)
POINTWISE_MIN = tl.constexpr(20)
POINTWISE_MAX = tl.constexpr(21)
POINTWISE_MOD = tl.constexpr(22)
POINTWISE_POW = tl.constexpr(23)
POINTWISE_CMP_EQ = tl.constexpr(25)
POINTWISE_CMP_NEQ = tl.constexpr(26)
POINTWISE_CMP_GT = tl.constexpr(27)
POINTWISE_CMP_GE = tl.constexpr(28)
POINTWISE_CMP_LT = tl.constexpr(29)
POINTWISE_CMP_LE = tl.constexpr(30)
POINTWISE_LOGICAL_AND = tl.constexpr(31)
POINTWISE_LOGICAL_OR = tl.constexpr(32)
POINTWISE_SIGMOID_BWD = tl.constexpr(40)


@triton.jit
def _apply_binary_operation(
    left,
    right,
    OP_KIND: tl.constexpr,
    ALPHA: tl.constexpr,
):
    if OP_KIND == POINTWISE_ADD:
        result = left + ALPHA * right
    elif OP_KIND == POINTWISE_SUB:
        result = left - ALPHA * right
    elif OP_KIND == POINTWISE_MUL:
        result = left * right
    elif OP_KIND == POINTWISE_SIGMOID_BWD:
        sigmoid = tl.sigmoid(right.to(tl.float32))
        result = left.to(tl.float32) * sigmoid * (1.0 - sigmoid)
    elif OP_KIND == POINTWISE_DIV:
        result = left / right
    elif OP_KIND == POINTWISE_MIN:
        result = tl.minimum(left, right)
    elif OP_KIND == POINTWISE_MAX:
        result = tl.maximum(left, right)
    elif OP_KIND == POINTWISE_MOD:
        result = libdevice.fmod(left.to(tl.float32), right.to(tl.float32))
    elif OP_KIND == POINTWISE_POW:
        result = libdevice.pow(left.to(tl.float32), right.to(tl.float32))
    elif OP_KIND == POINTWISE_CMP_EQ:
        result = left == right
    elif OP_KIND == POINTWISE_CMP_NEQ:
        result = left != right
    elif OP_KIND == POINTWISE_CMP_GT:
        result = left > right
    elif OP_KIND == POINTWISE_CMP_GE:
        result = left >= right
    elif OP_KIND == POINTWISE_CMP_LT:
        result = left < right
    elif OP_KIND == POINTWISE_CMP_LE:
        result = left <= right
    elif OP_KIND == POINTWISE_LOGICAL_AND:
        result = (left != 0) & (right != 0)
    elif OP_KIND == POINTWISE_LOGICAL_OR:
        result = (left != 0) | (right != 0)
    return result


@triton.jit
def binary_contiguous_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    OP_KIND: tl.constexpr,
    ALPHA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    left = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    right = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    result = _apply_binary_operation(left, right, OP_KIND, ALPHA)
    tl.store(
        out_ptr + offsets,
        result.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def binary_strided_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    DIM_0: tl.constexpr,
    DIM_1: tl.constexpr,
    DIM_2: tl.constexpr,
    DIM_3: tl.constexpr,
    DIM_4: tl.constexpr,
    DIM_5: tl.constexpr,
    DIM_6: tl.constexpr,
    DIM_7: tl.constexpr,
    LEFT_STRIDE_0: tl.constexpr,
    LEFT_STRIDE_1: tl.constexpr,
    LEFT_STRIDE_2: tl.constexpr,
    LEFT_STRIDE_3: tl.constexpr,
    LEFT_STRIDE_4: tl.constexpr,
    LEFT_STRIDE_5: tl.constexpr,
    LEFT_STRIDE_6: tl.constexpr,
    LEFT_STRIDE_7: tl.constexpr,
    RIGHT_STRIDE_0: tl.constexpr,
    RIGHT_STRIDE_1: tl.constexpr,
    RIGHT_STRIDE_2: tl.constexpr,
    RIGHT_STRIDE_3: tl.constexpr,
    RIGHT_STRIDE_4: tl.constexpr,
    RIGHT_STRIDE_5: tl.constexpr,
    RIGHT_STRIDE_6: tl.constexpr,
    RIGHT_STRIDE_7: tl.constexpr,
    OUTPUT_STRIDE_0: tl.constexpr,
    OUTPUT_STRIDE_1: tl.constexpr,
    OUTPUT_STRIDE_2: tl.constexpr,
    OUTPUT_STRIDE_3: tl.constexpr,
    OUTPUT_STRIDE_4: tl.constexpr,
    OUTPUT_STRIDE_5: tl.constexpr,
    OUTPUT_STRIDE_6: tl.constexpr,
    OUTPUT_STRIDE_7: tl.constexpr,
    OP_KIND: tl.constexpr,
    ALPHA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    remaining = offsets
    left_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    right_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    output_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

    coordinate = remaining % DIM_7
    remaining = remaining // DIM_7
    left_offsets += coordinate * LEFT_STRIDE_7
    right_offsets += coordinate * RIGHT_STRIDE_7
    output_offsets += coordinate * OUTPUT_STRIDE_7
    coordinate = remaining % DIM_6
    remaining = remaining // DIM_6
    left_offsets += coordinate * LEFT_STRIDE_6
    right_offsets += coordinate * RIGHT_STRIDE_6
    output_offsets += coordinate * OUTPUT_STRIDE_6
    coordinate = remaining % DIM_5
    remaining = remaining // DIM_5
    left_offsets += coordinate * LEFT_STRIDE_5
    right_offsets += coordinate * RIGHT_STRIDE_5
    output_offsets += coordinate * OUTPUT_STRIDE_5
    coordinate = remaining % DIM_4
    remaining = remaining // DIM_4
    left_offsets += coordinate * LEFT_STRIDE_4
    right_offsets += coordinate * RIGHT_STRIDE_4
    output_offsets += coordinate * OUTPUT_STRIDE_4
    coordinate = remaining % DIM_3
    remaining = remaining // DIM_3
    left_offsets += coordinate * LEFT_STRIDE_3
    right_offsets += coordinate * RIGHT_STRIDE_3
    output_offsets += coordinate * OUTPUT_STRIDE_3
    coordinate = remaining % DIM_2
    remaining = remaining // DIM_2
    left_offsets += coordinate * LEFT_STRIDE_2
    right_offsets += coordinate * RIGHT_STRIDE_2
    output_offsets += coordinate * OUTPUT_STRIDE_2
    coordinate = remaining % DIM_1
    remaining = remaining // DIM_1
    left_offsets += coordinate * LEFT_STRIDE_1
    right_offsets += coordinate * RIGHT_STRIDE_1
    output_offsets += coordinate * OUTPUT_STRIDE_1
    coordinate = remaining % DIM_0
    left_offsets += coordinate * LEFT_STRIDE_0
    right_offsets += coordinate * RIGHT_STRIDE_0
    output_offsets += coordinate * OUTPUT_STRIDE_0

    left = tl.load(x_ptr + left_offsets, mask=mask, other=0.0)
    right = tl.load(y_ptr + right_offsets, mask=mask, other=0.0)
    result = _apply_binary_operation(left, right, OP_KIND, ALPHA)
    tl.store(
        out_ptr + output_offsets,
        result.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


# -----------------------------------------------------------------------------
# Kernel algorithm variants. Native dispatch selects only registry-declared
# entry points; auxiliary kernels remain available to explicit compiler policy.
# -----------------------------------------------------------------------------


@triton.jit
def pow_tensor_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 向上转型到 float32，防止底层 libdevice 找不到 fp16/bf16 的 pow 签名
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    # res = libdevice.pow(x_f32, y_f32)
    log2_x = tl.math.log2(x_f32)
    res = tl.math.exp2(y_f32 * log2_x)

    # 写回时向下转型回目标数据类型
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def pow_scalar_exponent_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    exponent_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    x_f32 = x.to(tl.float32)

    exp_f32 = tl.cast(exponent_val, tl.float32)
    res = libdevice.pow(x_f32, exp_f32)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def pow_scalar_base_kernel(
    y_ptr,
    out_ptr,
    n_elements,
    base_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    y = tl.load(y_ptr + offsets, mask=mask)

    y_f32 = y.to(tl.float32)

    base_f32 = tl.cast(base_val, tl.float32)
    res = libdevice.pow(base_f32, y_f32)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def pow_broadcast_tensor_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    # 填充后的 6D 形状
    s1,
    s2,
    s3,
    s4,
    s5,
    # X 的 6D Strides
    sx0,
    sx1,
    sx2,
    sx3,
    sx4,
    sx5,
    # Y 的 6D Strides
    sy0,
    sy1,
    sy2,
    sy3,
    sy4,
    sy5,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 坐标还原（从内向外剥洋葱）
    # 由于做了坍缩，许多 sX 实际上是 1，Triton 编译器遇到 x % 1 或 x // 1 会直接优化掉?
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

    # 计算物理偏移并加载数据
    x_off = (
        idx0 * sx0
        + idx1 * sx1
        + idx2 * sx2
        + idx3 * sx3
        + idx4 * sx4
        + idx5 * sx5
    )
    y_off = (
        idx0 * sy0
        + idx1 * sy1
        + idx2 * sy2
        + idx3 * sy3
        + idx4 * sy4
        + idx5 * sy5
    )

    x = tl.load(x_ptr + x_off, mask=mask)
    y = tl.load(y_ptr + y_off, mask=mask)

    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    res = libdevice.pow(x_f32, y_f32)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def sigmoid_backward_kernel(
    loss_ptr,
    input_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    loss = tl.load(loss_ptr + offsets, mask=mask, other=0).to(tl.float32)
    x = tl.load(input_ptr + offsets, mask=mask, other=0).to(tl.float32)
    y = tl.sigmoid(x)
    dx = loss * y * (1.0 - y)

    tl.store(out_ptr + offsets, dx.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def sigmoid_backward_fp64_kernel(
    loss_ptr,
    input_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    loss = tl.load(loss_ptr + offsets, mask=mask, other=0).to(tl.float64)
    x = tl.load(input_ptr + offsets, mask=mask, other=0).to(tl.float64)
    y = 1.0 / (1.0 + libdevice.exp(-x))
    dx = loss * y * (1.0 - y)

    tl.store(out_ptr + offsets, dx.to(out_ptr.dtype.element_ty), mask=mask)
