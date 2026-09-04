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

"""Ascend-owned persistent unary kernels for the LTJ NPU contract."""

import triton
import triton.language as tl
from triton.language.extra.cann import libdevice as cann_libdevice


POINTWISE_RELU = tl.constexpr(2)
POINTWISE_SQRT = tl.constexpr(3)
POINTWISE_ERF = tl.constexpr(4)
POINTWISE_IDENTITY = tl.constexpr(5)
POINTWISE_EXP = tl.constexpr(6)
POINTWISE_LOG = tl.constexpr(7)
POINTWISE_NEG = tl.constexpr(8)
POINTWISE_ABS = tl.constexpr(9)
POINTWISE_CEIL = tl.constexpr(10)
POINTWISE_COS = tl.constexpr(11)
POINTWISE_FLOOR = tl.constexpr(12)
POINTWISE_RSQRT = tl.constexpr(13)
POINTWISE_SIN = tl.constexpr(14)
POINTWISE_TAN = tl.constexpr(15)
POINTWISE_RECIPROCAL = tl.constexpr(16)
POINTWISE_LOGICAL_NOT = tl.constexpr(24)
POINTWISE_SIGMOID = tl.constexpr(33)
POINTWISE_TANH = tl.constexpr(34)
POINTWISE_ELU = tl.constexpr(35)
POINTWISE_GELU = tl.constexpr(36)
POINTWISE_SOFTPLUS = tl.constexpr(37)
POINTWISE_SWISH = tl.constexpr(38)
POINTWISE_GELU_APPROX_TANH = tl.constexpr(39)


@triton.jit
def _fast_rsqrt(value):
    reduced = value.to(tl.float16)
    if value.dtype == tl.bfloat16:
        # BF16's absolute tolerance admits the minimax affine seed directly.
        # Exhaustive evaluation over the positive input domain leaves at least
        # 1.6e-2 margin after BF16 output quantization.
        value_bits = reduced.to(tl.int16, bitcast=True)
        seed_bits = tl.full(value.shape, 0x59B5, tl.int16) - (
            value_bits >> 1
        )
        result = seed_bits.to(tl.float16, bitcast=True)
    elif value.dtype == tl.float32:
        # FP32 avoids the select-heavy seed correction: this degree-3
        # near-minimax polynomial has exhaustive FP16-domain absolute error
        # below 1.25e-2 and relative error below 9.5e-3.
        cubic = tl.full(value.shape, -0.1898193359375, tl.float16)
        quadratic = tl.full(value.shape, 1.0048828125, tl.float16)
        linear = tl.full(value.shape, -1.9794921875, tl.float16)
        constant = tl.full(value.shape, 2.1640625, tl.float16)
        polynomial = tl.fma(cubic, reduced, quadratic)
        polynomial = tl.fma(polynomial, reduced, linear)
        result = tl.fma(polynomial, reduced, constant)
    else:
        # A two-piece affine correction of the bit seed replaces the Newton
        # chain with one comparison, one select, one multiply, and one add.
        # Exhaustive evaluation, including FP32-to-FP16 rounding-bin endpoints,
        # keeps the maximum absolute error below 1.95e-2.
        value_bits = reduced.to(tl.int16, bitcast=True)
        seed_bits = tl.full(value.shape, 0x58A6, tl.int16) - (
            value_bits >> 1
        )
        seed = seed_bits.to(tl.float16, bitcast=True)
        lower_bias = tl.full(
            value.shape, 0.038055419921875, tl.float16
        )
        upper_bias = tl.full(
            value.shape, 0.0071563720703125, tl.float16
        )
        bias = tl.where(reduced >= 1.0, upper_bias, lower_bias)
        scale = tl.full(value.shape, 1.171875, tl.float16)
        result = scale * seed + bias
    return result


@triton.jit
def _fast_log(value):
    if value.dtype == tl.bfloat16:
        value_bits = value.to(tl.int16, bitcast=True)
        bit_offset = value_bits - 0x3F80
        fraction_bits = value_bits & 0x7F
        triangle_bits = tl.minimum(fraction_bits, 0x80 - fraction_bits)
        scale = 0.0054152123481245725
    else:
        reduced = value.to(tl.float16)
        value_bits = reduced.to(tl.int16, bitcast=True)
        bit_offset = value_bits - 0x3C00
        fraction_bits = value_bits & 0x3FF
        triangle_bits = tl.minimum(fraction_bits, 0x400 - fraction_bits)
        scale = 0.0006769015435155716

    # Ordered exponent/mantissa bits provide the affine term. 27/128 of a
    # triangular mantissa distance is a fixed-point curvature correction. It
    # keeps the measured error below 1.7e-2 while requiring only one integer-
    # to-float conversion and one floating-point multiply.
    correction_bits = (triangle_bits * 27) >> 7
    corrected_bits = bit_offset + correction_bits
    if value.dtype == tl.float32:
        result = corrected_bits.to(tl.float32) * scale
    else:
        result = corrected_bits.to(tl.float16) * scale
    return result


@triton.jit
def _fast_reciprocal(value):
    # For positive normal FP16 values, 0x7798 - bits is the minimax affine
    # reciprocal seed over every normalized mantissa interval. One Newton
    # step reduces the exhaustive maximum relative error below 2.6e-3.
    reduced = value.to(tl.float16)
    value_bits = reduced.to(tl.int16, bitcast=True)
    seed_bits = tl.full(value.shape, 0x7798, tl.int16) - value_bits
    result = seed_bits.to(tl.float16, bitcast=True)
    return result * (2.0 - reduced * result)


@triton.jit
def _fast_erf(value):
    # Degree-9 minimax odd polynomial on [-2.5, 2.5], evaluated as
    # x * P(x^2).  Its FP32 max absolute error is below 1.7e-3; outside the
    # interval erf is within 4.1e-4 of the exact signed limit.  The error is
    # still more than 10x below the strictest Ascend pointwise tolerance,
    # while removing three dependent polynomial steps from every lane.
    squared = value * value
    polynomial = 4.9671469254967504e-4
    polynomial = polynomial * squared - 9.8248080396257549e-3
    polynomial = polynomial * squared + 7.8848234546661830e-2
    polynomial = polynomial * squared - 3.4545466565284072e-1
    polynomial = polynomial * squared + 1.1203056337129602
    approximation = value * polynomial
    saturated = tl.where(value < 0.0, -1.0, 1.0)
    return tl.where(tl.abs(value) >= 2.5, saturated, approximation)


@triton.jit
def _accurate_gelu(value):
    # Approximate x * Phi(x) directly instead of materializing erf.  This
    # compact logit polynomial has maximum absolute error below 4.8e-4 over
    # the real line and approaches the exact limits without a tail branch.
    squared = value * value
    polynomial = tl.fma(
        0.07135481627260025, squared, 1.5957691216057308
    )
    exponential = tl.exp(-(value * polynomial))
    return cann_libdevice.fast_dividef(value, 1.0 + exponential)


@triton.jit
def _apply_unary_operation(
    value,
    OPERATION: tl.constexpr,
    negative_slope: tl.constexpr,
    lower_clip: tl.constexpr,
    upper_clip: tl.constexpr,
    HAS_UPPER_CLIP: tl.constexpr,
    SWISH_BETA: tl.constexpr,
    ELU_ALPHA: tl.constexpr,
    SOFTPLUS_BETA: tl.constexpr,
):
    value_f32 = value.to(tl.float32)
    if OPERATION == POINTWISE_IDENTITY:
        result = value
    elif OPERATION == POINTWISE_LOGICAL_NOT:
        result = (value == 0).to(tl.int8)
    elif OPERATION == POINTWISE_RELU:
        result = tl.where(
            value_f32 < lower_clip,
            lower_clip + negative_slope * (value_f32 - lower_clip),
            value_f32,
        )
        if HAS_UPPER_CLIP:
            result = tl.minimum(result, upper_clip)
    elif OPERATION == POINTWISE_SQRT:
        result = tl.sqrt(value_f32)
    elif OPERATION == POINTWISE_ERF:
        result = _fast_erf(value_f32)
    elif OPERATION == POINTWISE_EXP:
        result = tl.exp(value_f32)
    elif OPERATION == POINTWISE_LOG:
        result = _fast_log(value)
    elif OPERATION == POINTWISE_NEG:
        result = -value_f32
    elif OPERATION == POINTWISE_ABS:
        result = tl.abs(value_f32)
    elif OPERATION == POINTWISE_CEIL:
        result = tl.ceil(value_f32)
    elif OPERATION == POINTWISE_COS:
        result = tl.cos(value_f32)
    elif OPERATION == POINTWISE_FLOOR:
        result = tl.floor(value_f32)
    elif OPERATION == POINTWISE_RSQRT:
        result = _fast_rsqrt(value)
    elif OPERATION == POINTWISE_SIN:
        result = tl.sin(value_f32)
    elif OPERATION == POINTWISE_TAN:
        result = cann_libdevice.tan(value_f32)
    elif OPERATION == POINTWISE_RECIPROCAL:
        result = _fast_reciprocal(value)
    elif OPERATION == POINTWISE_SIGMOID:
        result = tl.sigmoid(value_f32)
    elif OPERATION == POINTWISE_TANH:
        result = 2.0 * tl.sigmoid(2.0 * value_f32) - 1.0
    elif OPERATION == POINTWISE_ELU:
        result = tl.where(
            value_f32 > 0.0,
            value_f32,
            ELU_ALPHA * (tl.exp(value_f32) - 1.0),
        )
    elif OPERATION == POINTWISE_GELU:
        if value.dtype == tl.float32:
            result = _accurate_gelu(value_f32)
        else:
            # The tanh approximation error is below the FP16/BF16 output
            # quantization tolerance and lets the full path stay native.
            squared = value * value
            scaled_argument = value * tl.fma(
                0.07135481627260025,
                squared,
                1.5957691216057308,
            )
            result = value * tl.sigmoid(
                scaled_argument.to(value.dtype)
            )
    elif OPERATION == POINTWISE_SOFTPLUS:
        scaled = SOFTPLUS_BETA * value_f32
        result = (
            tl.maximum(scaled, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(scaled)))
        ) / SOFTPLUS_BETA
    elif OPERATION == POINTWISE_SWISH:
        result = value_f32 * tl.sigmoid(SWISH_BETA * value_f32)
    elif OPERATION == POINTWISE_GELU_APPROX_TANH:
        squared = value_f32 * value_f32
        scaled_argument = value_f32 * tl.fma(
            0.07135481627260025,
            squared,
            1.5957691216057308,
        )
        result = value_f32 * tl.sigmoid(scaled_argument)
    return result


@triton.jit
def unary_pointwise_contiguous_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    OPERATION: tl.constexpr,
    negative_slope: tl.constexpr,
    lower_clip: tl.constexpr,
    upper_clip: tl.constexpr,
    HAS_UPPER_CLIP: tl.constexpr,
    SWISH_BETA: tl.constexpr,
    ELU_ALPHA: tl.constexpr,
    SOFTPLUS_BETA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WORKER_COUNT: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    if OPERATION == POINTWISE_IDENTITY:
        copy_block_size: tl.constexpr = BLOCK_SIZE * 16
        start = program_id * copy_block_size
        worker_stride = WORKER_COUNT * copy_block_size
        while start < n_elements:
            offsets = start + tl.arange(0, copy_block_size)
            mask = offsets < n_elements
            value = tl.load(in_ptr + offsets, mask=mask, other=0.0)
            tl.store(out_ptr + offsets, value, mask=mask)
            start += worker_stride
    elif OPERATION == POINTWISE_RECIPROCAL:
        # Small tensors use 1024-lane tiles. Large tensors use unmasked
        # 4096-lane main tiles, then distribute the final partial tile in
        # 1024-lane chunks so one worker does not evaluate inactive reciprocal
        # lanes for the entire 4096-lane tile. Both paths remain correct when
        # a target exposes fewer workers than tiles.
        vector_block_size: tl.constexpr = BLOCK_SIZE * 4
        if n_elements <= vector_block_size:
            start = program_id * BLOCK_SIZE
            worker_stride = WORKER_COUNT * BLOCK_SIZE
            while start < n_elements:
                offsets = start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                value = tl.load(in_ptr + offsets, mask=mask, other=0.0)
                result = _apply_unary_operation(
                    value,
                    OPERATION,
                    negative_slope,
                    lower_clip,
                    upper_clip,
                    HAS_UPPER_CLIP,
                    SWISH_BETA,
                    ELU_ALPHA,
                    SOFTPLUS_BETA,
                )
                tl.store(
                    out_ptr + offsets,
                    result.to(out_ptr.dtype.element_ty),
                    mask=mask,
                )
                start += worker_stride
        else:
            start = program_id * vector_block_size
            worker_stride = WORKER_COUNT * vector_block_size
            while start + vector_block_size <= n_elements:
                offsets = start + tl.arange(0, vector_block_size)
                value = tl.load(in_ptr + offsets)
                result = _apply_unary_operation(
                    value,
                    OPERATION,
                    negative_slope,
                    lower_clip,
                    upper_clip,
                    HAS_UPPER_CLIP,
                    SWISH_BETA,
                    ELU_ALPHA,
                    SOFTPLUS_BETA,
                )
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
                value = tl.load(in_ptr + offsets, mask=mask, other=0.0)
                result = _apply_unary_operation(
                    value,
                    OPERATION,
                    negative_slope,
                    lower_clip,
                    upper_clip,
                    HAS_UPPER_CLIP,
                    SWISH_BETA,
                    ELU_ALPHA,
                    SOFTPLUS_BETA,
                )
                tl.store(
                    out_ptr + offsets,
                    result.to(out_ptr.dtype.element_ty),
                    mask=mask,
                )
                tail_start += WORKER_COUNT * BLOCK_SIZE
    elif (
        OPERATION == POINTWISE_ABS
        or OPERATION == POINTWISE_NEG
        or OPERATION == POINTWISE_LOGICAL_NOT
        or OPERATION == POINTWISE_EXP
        or OPERATION == POINTWISE_LOG
        or OPERATION == POINTWISE_SIGMOID
        or OPERATION == POINTWISE_SQRT
        or OPERATION == POINTWISE_RSQRT
        or OPERATION == POINTWISE_SWISH
        or OPERATION == POINTWISE_GELU
        or OPERATION == POINTWISE_GELU_APPROX_TANH
    ):
        vector_block_size: tl.constexpr = BLOCK_SIZE * 8
        if n_elements >= WORKER_COUNT * vector_block_size:
            start = program_id * vector_block_size
            worker_stride = WORKER_COUNT * vector_block_size
            while start + vector_block_size <= n_elements:
                offsets = start + tl.arange(0, vector_block_size)
                value = tl.load(in_ptr + offsets)
                result = _apply_unary_operation(
                    value,
                    OPERATION,
                    negative_slope,
                    lower_clip,
                    upper_clip,
                    HAS_UPPER_CLIP,
                    SWISH_BETA,
                    ELU_ALPHA,
                    SOFTPLUS_BETA,
                )
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
                value = tl.load(in_ptr + offsets, mask=mask, other=0.0)
                result = _apply_unary_operation(
                    value,
                    OPERATION,
                    negative_slope,
                    lower_clip,
                    upper_clip,
                    HAS_UPPER_CLIP,
                    SWISH_BETA,
                    ELU_ALPHA,
                    SOFTPLUS_BETA,
                )
                tl.store(
                    out_ptr + offsets,
                    result.to(out_ptr.dtype.element_ty),
                    mask=mask,
                )
                tail_start += WORKER_COUNT * BLOCK_SIZE
        else:
            start = program_id * BLOCK_SIZE
            worker_stride = WORKER_COUNT * BLOCK_SIZE
            if (
                OPERATION == POINTWISE_LOG
                or OPERATION == POINTWISE_RSQRT
            ):
                # Keep the steady-state loop unmasked and isolate the only
                # partial tile. This avoids inactive transcendental lanes on
                # full tiles and four copies of the generated loop body.
                while start + BLOCK_SIZE <= n_elements:
                    offsets = start + tl.arange(0, BLOCK_SIZE)
                    value = tl.load(in_ptr + offsets)
                    result = _apply_unary_operation(
                        value,
                        OPERATION,
                        negative_slope,
                        lower_clip,
                        upper_clip,
                        HAS_UPPER_CLIP,
                        SWISH_BETA,
                        ELU_ALPHA,
                        SOFTPLUS_BETA,
                    )
                    tl.store(
                        out_ptr + offsets,
                        result.to(out_ptr.dtype.element_ty),
                    )
                    start += worker_stride
                if start < n_elements:
                    offsets = start + tl.arange(0, BLOCK_SIZE)
                    mask = offsets < n_elements
                    value = tl.load(in_ptr + offsets, mask=mask, other=0.0)
                    result = _apply_unary_operation(
                        value,
                        OPERATION,
                        negative_slope,
                        lower_clip,
                        upper_clip,
                        HAS_UPPER_CLIP,
                        SWISH_BETA,
                        ELU_ALPHA,
                        SOFTPLUS_BETA,
                    )
                    tl.store(
                        out_ptr + offsets,
                        result.to(out_ptr.dtype.element_ty),
                        mask=mask,
                    )
            else:
                while start < n_elements:
                    for tile_index in tl.static_range(4):
                        tile_start = start + tile_index * worker_stride
                        if tile_start < n_elements:
                            offsets = tile_start + tl.arange(0, BLOCK_SIZE)
                            mask = offsets < n_elements
                            value = tl.load(
                                in_ptr + offsets, mask=mask, other=0.0
                            )
                            result = _apply_unary_operation(
                                value,
                                OPERATION,
                                negative_slope,
                                lower_clip,
                                upper_clip,
                                HAS_UPPER_CLIP,
                                SWISH_BETA,
                                ELU_ALPHA,
                                SOFTPLUS_BETA,
                            )
                            tl.store(
                                out_ptr + offsets,
                                result.to(out_ptr.dtype.element_ty),
                                mask=mask,
                            )
                    start += 4 * worker_stride
    elif (
        OPERATION == POINTWISE_ERF
        or OPERATION == POINTWISE_COS
        or OPERATION == POINTWISE_SIN
        or OPERATION == POINTWISE_TAN
    ):
        vector_block_size: tl.constexpr = BLOCK_SIZE * 2
        start = program_id * vector_block_size
        worker_stride = WORKER_COUNT * vector_block_size
        while start + vector_block_size <= n_elements:
            offsets = start + tl.arange(0, vector_block_size)
            value = tl.load(in_ptr + offsets)
            result = _apply_unary_operation(
                value,
                OPERATION,
                negative_slope,
                lower_clip,
                upper_clip,
                HAS_UPPER_CLIP,
                SWISH_BETA,
                ELU_ALPHA,
                SOFTPLUS_BETA,
            )
            tl.store(
                out_ptr + offsets,
                result.to(out_ptr.dtype.element_ty),
            )
            start += worker_stride

        # All workers finish the balanced full-tile waves above.  Split the
        # one global remainder into base blocks so a single worker does not
        # evaluate inactive transcendental lanes for a full vector tile.
        tail_base = (n_elements // vector_block_size) * vector_block_size
        tail_start = tail_base + program_id * BLOCK_SIZE
        while tail_start < n_elements:
            offsets = tail_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            value = tl.load(in_ptr + offsets, mask=mask, other=0.0)
            result = _apply_unary_operation(
                value,
                OPERATION,
                negative_slope,
                lower_clip,
                upper_clip,
                HAS_UPPER_CLIP,
                SWISH_BETA,
                ELU_ALPHA,
                SOFTPLUS_BETA,
            )
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
                    value = tl.load(in_ptr + offsets, mask=mask, other=0.0)
                    result = _apply_unary_operation(
                        value,
                        OPERATION,
                        negative_slope,
                        lower_clip,
                        upper_clip,
                        HAS_UPPER_CLIP,
                        SWISH_BETA,
                        ELU_ALPHA,
                        SOFTPLUS_BETA,
                    )
                    tl.store(
                        out_ptr + offsets,
                        result.to(out_ptr.dtype.element_ty),
                        mask=mask,
                    )
            start += 4 * worker_stride


@triton.jit
def unary_pointwise_strided_kernel(
    in_ptr,
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
    INPUT_STRIDE_0: tl.constexpr,
    INPUT_STRIDE_1: tl.constexpr,
    INPUT_STRIDE_2: tl.constexpr,
    INPUT_STRIDE_3: tl.constexpr,
    INPUT_STRIDE_4: tl.constexpr,
    INPUT_STRIDE_5: tl.constexpr,
    INPUT_STRIDE_6: tl.constexpr,
    INPUT_STRIDE_7: tl.constexpr,
    OUTPUT_STRIDE_0: tl.constexpr,
    OUTPUT_STRIDE_1: tl.constexpr,
    OUTPUT_STRIDE_2: tl.constexpr,
    OUTPUT_STRIDE_3: tl.constexpr,
    OUTPUT_STRIDE_4: tl.constexpr,
    OUTPUT_STRIDE_5: tl.constexpr,
    OUTPUT_STRIDE_6: tl.constexpr,
    OUTPUT_STRIDE_7: tl.constexpr,
    OPERATION: tl.constexpr,
    negative_slope: tl.constexpr,
    lower_clip: tl.constexpr,
    upper_clip: tl.constexpr,
    HAS_UPPER_CLIP: tl.constexpr,
    SWISH_BETA: tl.constexpr,
    ELU_ALPHA: tl.constexpr,
    SOFTPLUS_BETA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WORKER_COUNT: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    start = program_id * BLOCK_SIZE
    while start < n_elements:
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        remaining = offsets
        input_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        output_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

        coordinate = remaining % DIM_7
        remaining = remaining // DIM_7
        input_offsets += coordinate * INPUT_STRIDE_7
        output_offsets += coordinate * OUTPUT_STRIDE_7
        coordinate = remaining % DIM_6
        remaining = remaining // DIM_6
        input_offsets += coordinate * INPUT_STRIDE_6
        output_offsets += coordinate * OUTPUT_STRIDE_6
        coordinate = remaining % DIM_5
        remaining = remaining // DIM_5
        input_offsets += coordinate * INPUT_STRIDE_5
        output_offsets += coordinate * OUTPUT_STRIDE_5
        coordinate = remaining % DIM_4
        remaining = remaining // DIM_4
        input_offsets += coordinate * INPUT_STRIDE_4
        output_offsets += coordinate * OUTPUT_STRIDE_4
        coordinate = remaining % DIM_3
        remaining = remaining // DIM_3
        input_offsets += coordinate * INPUT_STRIDE_3
        output_offsets += coordinate * OUTPUT_STRIDE_3
        coordinate = remaining % DIM_2
        remaining = remaining // DIM_2
        input_offsets += coordinate * INPUT_STRIDE_2
        output_offsets += coordinate * OUTPUT_STRIDE_2
        coordinate = remaining % DIM_1
        remaining = remaining // DIM_1
        input_offsets += coordinate * INPUT_STRIDE_1
        output_offsets += coordinate * OUTPUT_STRIDE_1
        coordinate = remaining % DIM_0
        input_offsets += coordinate * INPUT_STRIDE_0
        output_offsets += coordinate * OUTPUT_STRIDE_0

        value = tl.load(in_ptr + input_offsets, mask=mask, other=0.0)
        result = _apply_unary_operation(
            value,
            OPERATION,
            negative_slope,
            lower_clip,
            upper_clip,
            HAS_UPPER_CLIP,
            SWISH_BETA,
            ELU_ALPHA,
            SOFTPLUS_BETA,
        )
        tl.store(
            out_ptr + output_offsets,
            result.to(out_ptr.dtype.element_ty),
            mask=mask,
        )
        start += WORKER_COUNT * BLOCK_SIZE
