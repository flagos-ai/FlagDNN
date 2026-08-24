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

"""Ascend-owned persistent binary kernels for the LTJ NPU contract."""

import triton
import triton.language as tl
from triton.language.extra.cann import libdevice as cann_libdevice


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
POINTWISE_SIGMOID_BACKWARD = tl.constexpr(40)


@triton.jit
def _normalize_finite_nonzero_binary32(bits):
    exponent_field = ((bits >> 23) & 0xff).to(tl.int32)
    significand = bits & 0x007fffff
    is_subnormal = exponent_field == 0
    significand = tl.where(
        is_subnormal,
        significand,
        significand | 0x00800000,
    )
    exponent = tl.where(is_subnormal, 1, exponent_field)
    for _ in tl.range(0, 23):
        needs_shift = is_subnormal & (significand < 0x00800000)
        significand = tl.where(needs_shift, significand << 1, significand)
        exponent = tl.where(needs_shift, exponent - 1, exponent)
    return significand, exponent


@triton.jit
def _ieee_fmodf(left, right):
    # CANN 9.0's native fmod overflows for wide exponent ratios.  This is the
    # binary32 integer long-division algorithm used by libc implementations.
    # int32 is intentional: the vector lowering cannot cast uint32 to uint64.
    left_bits = left.to(tl.int32, bitcast=True)
    right_bits = right.to(tl.int32, bitcast=True)
    sign = left_bits & -2147483648
    abs_left = left_bits & 0x7fffffff
    abs_right = right_bits & 0x7fffffff
    infinity = 0x7f800000

    invalid = (abs_right == 0) | (abs_right > infinity) | (abs_left >= infinity)
    algorithm_lane = (~invalid) & (abs_left > abs_right)

    # Keep special-case lanes harmless while the vectorized integer body runs.
    work_left = tl.where(algorithm_lane, abs_left, 0x3f800000).to(tl.int32)
    work_right = tl.where(algorithm_lane, abs_right, 0x3f000000).to(tl.int32)
    left_sig, left_exp = _normalize_finite_nonzero_binary32(work_left)
    right_sig, right_exp = _normalize_finite_nonzero_binary32(work_right)

    became_zero = tl.zeros(left.shape, tl.int1)
    # The maximum binary32 exponent distance is 254 - (-22) = 276.
    for _ in tl.range(0, 276):
        active = (left_exp > right_exp) & (~became_zero)
        difference = left_sig - right_sig
        subtract = active & (left_sig >= right_sig)
        became_zero = became_zero | (subtract & (difference == 0))
        reduced = tl.where(subtract, difference, left_sig)
        advance = active & (~became_zero)
        left_sig = tl.where(advance, reduced << 1, reduced)
        left_exp = tl.where(advance, left_exp - 1, left_exp)

    difference = left_sig - right_sig
    subtract = (left_sig >= right_sig) & (~became_zero)
    became_zero = became_zero | (subtract & (difference == 0))
    left_sig = tl.where(subtract, difference, left_sig)

    for _ in tl.range(0, 23):
        needs_shift = (~became_zero) & (left_sig < 0x00800000)
        left_sig = tl.where(needs_shift, left_sig << 1, left_sig)
        left_exp = tl.where(needs_shift, left_exp - 1, left_exp)

    is_normal = left_exp > 0
    normal_exponent = tl.where(is_normal, left_exp, 1)
    normal_bits = (left_sig - 0x00800000) | (normal_exponent << 23)
    subnormal_shift = tl.where(is_normal, 0, -left_exp + 1)
    subnormal_shift = tl.minimum(tl.maximum(subnormal_shift, 0), 31)
    subnormal_bits = left_sig >> subnormal_shift
    result_bits = tl.where(is_normal, normal_bits, subnormal_bits) | sign
    result_bits = tl.where(became_zero, sign, result_bits)

    # Match fmod classification.  NaN payload propagation is not promised;
    # invalid inputs return a canonical quiet NaN.
    result_bits = tl.where(abs_left < abs_right, left_bits, result_bits)
    result_bits = tl.where(abs_left == abs_right, sign, result_bits)
    result_bits = tl.where(invalid, 0x7fc00000, result_bits)
    return result_bits.to(tl.float32, bitcast=True)


@triton.jit
def _hybrid_fmodf(left, right):
    left_bits = left.to(tl.int32, bitcast=True)
    right_bits = right.to(tl.int32, bitcast=True)
    sign = left_bits & -2147483648
    abs_left = left_bits & 0x7fffffff
    abs_right = right_bits & 0x7fffffff
    infinity = 0x7f800000

    left_magnitude = abs_left.to(tl.float32, bitcast=True)
    right_magnitude = abs_right.to(tl.float32, bitcast=True)
    quotient = (left_magnitude / right_magnitude).to(tl.int32).to(tl.float32)
    native = tl.fma(-quotient, right_magnitude, left_magnitude)

    # Below 2^20 the rounded FP32 quotient can differ from the exact truncated
    # quotient only by one, so bringing a negative result back into [0, |y|)
    # repairs that boundary without an external math-library call.
    native = tl.where(native < 0.0, native + right_magnitude, native)
    native = tl.where(native >= right_magnitude, native - right_magnitude, native)
    result_bits = native.to(tl.int32, bitcast=True) | sign

    invalid = (abs_right == 0) | (abs_right > infinity) | (abs_left >= infinity)
    algorithm_lane = (~invalid) & (abs_left > abs_right)
    result_bits = tl.where(abs_left < abs_right, left_bits, result_bits)
    result_bits = tl.where(abs_left == abs_right, sign, result_bits)
    result_bits = tl.where(invalid, 0x7fc00000, result_bits)
    result = result_bits.to(tl.float32, bitcast=True)

    # Native fmod converts its quotient to int32 and therefore cannot represent
    # wide exponent ratios.  Keep the exact binary32 path off the common case,
    # but select it for every block that contains a ratio above the safe bound.
    requires_exact = algorithm_lane & (
        left_magnitude > right_magnitude * 1048576.0
    )
    if tl.max(requires_exact.to(tl.int32), axis=0) != 0:
        exact = _ieee_fmodf(left, right)
        result = tl.where(requires_exact, exact, result)
    return result


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
    elif OP_KIND == POINTWISE_DIV:
        result = left / right
    elif OP_KIND == POINTWISE_MIN:
        result = tl.minimum(left, right)
    elif OP_KIND == POINTWISE_MAX:
        result = tl.maximum(left, right)
    elif OP_KIND == POINTWISE_MOD:
        result = _hybrid_fmodf(left.to(tl.float32), right.to(tl.float32))
    elif OP_KIND == POINTWISE_POW:
        result = cann_libdevice.pow(left.to(tl.float32), right.to(tl.float32))
    elif OP_KIND == POINTWISE_CMP_EQ:
        result = (left == right).to(tl.int8)
    elif OP_KIND == POINTWISE_CMP_NEQ:
        result = (left != right).to(tl.int8)
    elif OP_KIND == POINTWISE_CMP_GT:
        result = (left > right).to(tl.int8)
    elif OP_KIND == POINTWISE_CMP_GE:
        result = (left >= right).to(tl.int8)
    elif OP_KIND == POINTWISE_CMP_LT:
        result = (left < right).to(tl.int8)
    elif OP_KIND == POINTWISE_CMP_LE:
        result = (left <= right).to(tl.int8)
    elif OP_KIND == POINTWISE_LOGICAL_AND:
        result = ((left != 0) & (right != 0)).to(tl.int8)
    elif OP_KIND == POINTWISE_LOGICAL_OR:
        result = ((left != 0) | (right != 0)).to(tl.int8)
    elif OP_KIND == POINTWISE_SIGMOID_BACKWARD:
        grad = left.to(tl.float32)
        logit = right.to(tl.float32)
        e = tl.exp(-tl.abs(logit))
        inv = 1.0 / (1.0 + e)
        result = grad * e * inv * inv
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
    WORKER_COUNT: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    if (
        OP_KIND == POINTWISE_MIN
        or OP_KIND == POINTWISE_MAX
        or OP_KIND == POINTWISE_CMP_EQ
        or OP_KIND == POINTWISE_CMP_NEQ
        or OP_KIND == POINTWISE_CMP_GT
        or OP_KIND == POINTWISE_CMP_GE
        or OP_KIND == POINTWISE_CMP_LT
        or OP_KIND == POINTWISE_CMP_LE
        or OP_KIND == POINTWISE_LOGICAL_AND
        or OP_KIND == POINTWISE_LOGICAL_OR
    ):
        vector_block_size: tl.constexpr = BLOCK_SIZE * 8
        if n_elements >= WORKER_COUNT * vector_block_size:
            start = program_id * vector_block_size
            worker_stride = WORKER_COUNT * vector_block_size
            while start + vector_block_size <= n_elements:
                offsets = start + tl.arange(0, vector_block_size)
                left = tl.load(x_ptr + offsets)
                right = tl.load(y_ptr + offsets)
                result = _apply_binary_operation(left, right, OP_KIND, ALPHA)
                tl.store(
                    out_ptr + offsets,
                    result.to(out_ptr.dtype.element_ty),
                )
                start += worker_stride

            tail_base = (n_elements // vector_block_size) * vector_block_size
            tail_start = tail_base + program_id * BLOCK_SIZE
            while tail_start < n_elements:
                offsets = tail_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                left = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                right = tl.load(y_ptr + offsets, mask=mask, other=1.0)
                result = _apply_binary_operation(left, right, OP_KIND, ALPHA)
                tl.store(
                    out_ptr + offsets,
                    result.to(out_ptr.dtype.element_ty),
                    mask=mask,
                )
                tail_start += WORKER_COUNT * BLOCK_SIZE
        else:
            start = program_id * BLOCK_SIZE
            worker_stride = WORKER_COUNT * BLOCK_SIZE
            while start < n_elements:
                for tile_index in tl.static_range(4):
                    tile_start = start + tile_index * worker_stride
                    if tile_start < n_elements:
                        offsets = tile_start + tl.arange(0, BLOCK_SIZE)
                        mask = offsets < n_elements
                        left = tl.load(
                            x_ptr + offsets, mask=mask, other=0.0
                        )
                        right = tl.load(
                            y_ptr + offsets, mask=mask, other=1.0
                        )
                        result = _apply_binary_operation(
                            left, right, OP_KIND, ALPHA
                        )
                        tl.store(
                            out_ptr + offsets,
                            result.to(out_ptr.dtype.element_ty),
                            mask=mask,
                        )
                start += 4 * worker_stride
    else:
        start = program_id * BLOCK_SIZE
        worker_stride = WORKER_COUNT * BLOCK_SIZE
        while start < n_elements:
            for tile_index in tl.static_range(4):
                tile_start = start + tile_index * worker_stride
                if tile_start < n_elements:
                    offsets = tile_start + tl.arange(0, BLOCK_SIZE)
                    mask = offsets < n_elements
                    left = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                    right = tl.load(y_ptr + offsets, mask=mask, other=1.0)
                    result = _apply_binary_operation(
                        left, right, OP_KIND, ALPHA
                    )
                    tl.store(
                        out_ptr + offsets,
                        result.to(out_ptr.dtype.element_ty),
                        mask=mask,
                    )
            start += 4 * worker_stride


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
    WORKER_COUNT: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    start = program_id * BLOCK_SIZE
    while start < n_elements:
        offsets = start + tl.arange(0, BLOCK_SIZE)
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
        right = tl.load(y_ptr + right_offsets, mask=mask, other=1.0)
        result = _apply_binary_operation(left, right, OP_KIND, ALPHA)
        tl.store(
            out_ptr + output_offsets,
            result.to(out_ptr.dtype.element_ty),
            mask=mask,
        )
        start += WORKER_COUNT * BLOCK_SIZE
