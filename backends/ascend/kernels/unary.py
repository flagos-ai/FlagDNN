"""Ascend kernels for unary."""

import triton
import triton.language as tl
from triton.language.extra.cann import libdevice as cann_libdevice
from triton.language.extra.cann import libdevice

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
        seed_bits = tl.full(value.shape, 0x59B5, tl.int16) - (value_bits >> 1)
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
        seed_bits = tl.full(value.shape, 0x58A6, tl.int16) - (value_bits >> 1)
        seed = seed_bits.to(tl.float16, bitcast=True)
        lower_bias = tl.full(value.shape, 0.038055419921875, tl.float16)
        upper_bias = tl.full(value.shape, 0.0071563720703125, tl.float16)
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
    polynomial = tl.fma(0.07135481627260025, squared, 1.5957691216057308)
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
            result = value * tl.sigmoid(scaled_argument.to(value.dtype))
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
            if OPERATION == POINTWISE_LOG or OPERATION == POINTWISE_RSQRT:
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


@triton.jit
def _apply_activation_gradient(
    left,
    right,
    OP_KIND: tl.constexpr,
    ALPHA: tl.constexpr,
    NEGATIVE_SLOPE: tl.constexpr = 0.0,
    LOWER_CLIP: tl.constexpr = 0.0,
    UPPER_CLIP: tl.constexpr = 0.0,
    HAS_UPPER_CLIP: tl.constexpr = False,
    SWISH_BETA: tl.constexpr = 1.0,
    ELU_ALPHA: tl.constexpr = 1.0,
    SOFTPLUS_BETA: tl.constexpr = 1.0,
):
    if OP_KIND == 40:
        sigmoid = tl.sigmoid(right.to(tl.float32))
        result = left.to(tl.float32) * sigmoid * (1.0 - sigmoid)
    elif OP_KIND >= 42 and OP_KIND <= 48:
        x = right.to(tl.float32)
        dy = left.to(tl.float32)
        if OP_KIND == 42:  # clipped/leaky ReLU
            gradient = tl.where(x > LOWER_CLIP, 1.0, NEGATIVE_SLOPE)
            if HAS_UPPER_CLIP:
                forward = tl.where(
                    x < LOWER_CLIP,
                    LOWER_CLIP + NEGATIVE_SLOPE * (x - LOWER_CLIP),
                    x,
                )
                gradient = tl.where(forward >= UPPER_CLIP, 0.0, gradient)
        elif OP_KIND == 43:  # tanh
            tail = tl.exp(-tl.abs(x))
            denominator = 1.0 + tail * tail
            # Split the exponential factors around DY so a small derivative
            # can still produce a representable gradient for a large DY.
            dy = dy * tail
            gradient = 4.0 * tail / (denominator * denominator)
        elif OP_KIND == 44:  # ELU
            gradient = tl.where(x > 0.0, 1.0, ELU_ALPHA * tl.exp(x))
        elif OP_KIND == 45:  # exact GELU
            density = tl.exp(-0.5 * x * x)
            # Preserve the limiting derivative when density underflows or X
            # is infinite; inf * 0 would otherwise contaminate the result.
            correction = tl.where(density == 0.0, 0.0, x * density)
            gradient = (
                0.5 * (1.0 + libdevice.erf(x * 0.7071067811865476))
                + correction * 0.3989422804014327
            )
        elif OP_KIND == 46:  # softplus
            gradient = tl.sigmoid(SOFTPLUS_BETA * x)
        elif OP_KIND == 47:  # swish
            value = tl.sigmoid(SWISH_BETA * x)
            correction = SWISH_BETA * x * value * (1.0 - value)
            gradient = tl.where(
                (value == 0.0) | (value == 1.0), value, value + correction
            )
        else:  # tanh approximation to GELU
            inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
            value = libdevice.tanh(inner)
            correction = (
                0.5
                * x
                * (1.0 - value * value)
                * 0.7978845608028654
                * (1.0 + 0.134145 * x * x)
            )
            gradient = 0.5 * (1.0 + value) + tl.where(
                tl.abs(value) == 1.0, 0.0, correction
            )
        result = dy * gradient
    return result


@triton.jit
def activation_backward_contiguous_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    OP_KIND: tl.constexpr,
    ALPHA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NEGATIVE_SLOPE: tl.constexpr = 0.0,
    LOWER_CLIP: tl.constexpr = 0.0,
    UPPER_CLIP: tl.constexpr = 0.0,
    HAS_UPPER_CLIP: tl.constexpr = False,
    SWISH_BETA: tl.constexpr = 1.0,
    ELU_ALPHA: tl.constexpr = 1.0,
    SOFTPLUS_BETA: tl.constexpr = 1.0,
    MASK_TAIL: tl.constexpr = True,
):
    program_id = tl.program_id(0).to(tl.int64)
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements if MASK_TAIL else True
    left = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    right = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    result = _apply_activation_gradient(
        left,
        right,
        OP_KIND,
        ALPHA,
        NEGATIVE_SLOPE,
        LOWER_CLIP,
        UPPER_CLIP,
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


@triton.jit
def activation_backward_strided_kernel(
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
    NEGATIVE_SLOPE: tl.constexpr = 0.0,
    LOWER_CLIP: tl.constexpr = 0.0,
    UPPER_CLIP: tl.constexpr = 0.0,
    HAS_UPPER_CLIP: tl.constexpr = False,
    SWISH_BETA: tl.constexpr = 1.0,
    ELU_ALPHA: tl.constexpr = 1.0,
    SOFTPLUS_BETA: tl.constexpr = 1.0,
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
    result = _apply_activation_gradient(
        left,
        right,
        OP_KIND,
        ALPHA,
        NEGATIVE_SLOPE,
        LOWER_CLIP,
        UPPER_CLIP,
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


# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0


@triton.jit
def gen_index_kernel(
    output_ptr,
    N_ELEMENTS: tl.constexpr,
    AXIS_EXTENT: tl.constexpr,
    INNER: tl.constexpr,
    DIM_0: tl.constexpr,
    DIM_1: tl.constexpr,
    DIM_2: tl.constexpr,
    DIM_3: tl.constexpr,
    DIM_4: tl.constexpr,
    DIM_5: tl.constexpr,
    DIM_6: tl.constexpr,
    DIM_7: tl.constexpr,
    OUTPUT_STRIDE_0: tl.constexpr,
    OUTPUT_STRIDE_1: tl.constexpr,
    OUTPUT_STRIDE_2: tl.constexpr,
    OUTPUT_STRIDE_3: tl.constexpr,
    OUTPUT_STRIDE_4: tl.constexpr,
    OUTPUT_STRIDE_5: tl.constexpr,
    OUTPUT_STRIDE_6: tl.constexpr,
    OUTPUT_STRIDE_7: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    logical = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    remaining = logical
    physical = tl.zeros((BLOCK_SIZE,), tl.int64)
    physical += (remaining % DIM_7) * OUTPUT_STRIDE_7
    remaining //= DIM_7
    physical += (remaining % DIM_6) * OUTPUT_STRIDE_6
    remaining //= DIM_6
    physical += (remaining % DIM_5) * OUTPUT_STRIDE_5
    remaining //= DIM_5
    physical += (remaining % DIM_4) * OUTPUT_STRIDE_4
    remaining //= DIM_4
    physical += (remaining % DIM_3) * OUTPUT_STRIDE_3
    remaining //= DIM_3
    physical += (remaining % DIM_2) * OUTPUT_STRIDE_2
    remaining //= DIM_2
    physical += (remaining % DIM_1) * OUTPUT_STRIDE_1
    remaining //= DIM_1
    physical += (remaining % DIM_0) * OUTPUT_STRIDE_0
    remaining //= DIM_0
    value = (logical // INNER) % AXIS_EXTENT
    tl.store(
        output_ptr + physical,
        value.to(output_ptr.dtype.element_ty),
        logical < N_ELEMENTS,
    )


@triton.jit
def concatenate_copy_kernel(
    input_ptr,
    output_ptr,
    N_ELEMENTS: tl.constexpr,
    OUTPUT_BASE: tl.constexpr,
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
    BLOCK_SIZE: tl.constexpr,
):
    logical = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    remaining = logical
    source = tl.zeros((BLOCK_SIZE,), tl.int64)
    destination = tl.full((BLOCK_SIZE,), OUTPUT_BASE, tl.int64)
    coordinate = remaining % DIM_7
    remaining //= DIM_7
    source += coordinate * INPUT_STRIDE_7
    destination += coordinate * OUTPUT_STRIDE_7
    coordinate = remaining % DIM_6
    remaining //= DIM_6
    source += coordinate * INPUT_STRIDE_6
    destination += coordinate * OUTPUT_STRIDE_6
    coordinate = remaining % DIM_5
    remaining //= DIM_5
    source += coordinate * INPUT_STRIDE_5
    destination += coordinate * OUTPUT_STRIDE_5
    coordinate = remaining % DIM_4
    remaining //= DIM_4
    source += coordinate * INPUT_STRIDE_4
    destination += coordinate * OUTPUT_STRIDE_4
    coordinate = remaining % DIM_3
    remaining //= DIM_3
    source += coordinate * INPUT_STRIDE_3
    destination += coordinate * OUTPUT_STRIDE_3
    coordinate = remaining % DIM_2
    remaining //= DIM_2
    source += coordinate * INPUT_STRIDE_2
    destination += coordinate * OUTPUT_STRIDE_2
    coordinate = remaining % DIM_1
    remaining //= DIM_1
    source += coordinate * INPUT_STRIDE_1
    destination += coordinate * OUTPUT_STRIDE_1
    coordinate = remaining % DIM_0
    remaining //= DIM_0
    source += coordinate * INPUT_STRIDE_0
    destination += coordinate * OUTPUT_STRIDE_0
    value = tl.load(input_ptr + source, logical < N_ELEMENTS, other=0)
    tl.store(output_ptr + destination, value, logical < N_ELEMENTS)


# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0


@triton.jit
def rope_kernel(
    x_ptr,
    freqs_ptr,
    y_ptr,
    ELEMENTS: tl.constexpr,
    HEADS: tl.constexpr,
    SEQUENCE: tl.constexpr,
    DIMENSION: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    SCALE: tl.constexpr,
    X_B: tl.constexpr,
    X_H: tl.constexpr,
    X_S: tl.constexpr,
    X_D: tl.constexpr,
    Y_B: tl.constexpr,
    Y_H: tl.constexpr,
    Y_S: tl.constexpr,
    Y_D: tl.constexpr,
    F_S: tl.constexpr,
    F_D: tl.constexpr,
    BACKWARD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    d = index % DIMENSION
    position = index // DIMENSION % SEQUENCE
    head = index // (DIMENSION * SEQUENCE) % HEADS
    batch = index // (DIMENSION * SEQUENCE * HEADS)
    source = batch * X_B + head * X_H + position * X_S + d * X_D
    destination = batch * Y_B + head * Y_H + position * Y_S + d * Y_D
    active = d >= DIMENSION - ROPE_DIM
    relative = d - (DIMENSION - ROPE_DIM)
    low = relative < ROPE_DIM // 2
    partner = tl.where(low, relative + ROPE_DIM // 2, relative - ROPE_DIM // 2)
    valid = index < ELEMENTS
    x = tl.load(x_ptr + source, valid, other=0).to(tl.float32)
    paired = tl.load(
        x_ptr + source + (partner - relative) * X_D, valid & active, other=0
    ).to(tl.float32)
    angle = tl.load(
        freqs_ptr + position * F_S + relative * F_D, valid & active, other=0
    ).to(tl.float32)
    if BACKWARD:
        paired_angle = tl.load(
            freqs_ptr + position * F_S + partner * F_D, valid & active, other=0
        ).to(tl.float32)
        sine = libdevice.sin(paired_angle)
        sign = tl.where(low, 1.0, -1.0)
    else:
        sine = libdevice.sin(angle)
        sign = tl.where(low, -1.0, 1.0)
    result = x * libdevice.cos(angle) + sign * paired * sine
    tl.store(y_ptr + destination, tl.where(active, result, x) * SCALE, valid)


# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0


@triton.jit
def rng_kernel(
    output_ptr,
    N_ELEMENTS: tl.constexpr,
    SEED: tl.constexpr,
    OFFSET: tl.constexpr,
    DISTRIBUTION: tl.constexpr,
    PROBABILITY: tl.constexpr,
    UNIFORM_BITS: tl.constexpr,
    DIM_0: tl.constexpr,
    DIM_1: tl.constexpr,
    DIM_2: tl.constexpr,
    DIM_3: tl.constexpr,
    DIM_4: tl.constexpr,
    DIM_5: tl.constexpr,
    DIM_6: tl.constexpr,
    DIM_7: tl.constexpr,
    OUTPUT_STRIDE_0: tl.constexpr,
    OUTPUT_STRIDE_1: tl.constexpr,
    OUTPUT_STRIDE_2: tl.constexpr,
    OUTPUT_STRIDE_3: tl.constexpr,
    OUTPUT_STRIDE_4: tl.constexpr,
    OUTPUT_STRIDE_5: tl.constexpr,
    OUTPUT_STRIDE_6: tl.constexpr,
    OUTPUT_STRIDE_7: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    logical = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    remaining = logical
    physical = tl.zeros((BLOCK_SIZE,), tl.int64)
    physical += (remaining % DIM_7) * OUTPUT_STRIDE_7
    remaining //= DIM_7
    physical += (remaining % DIM_6) * OUTPUT_STRIDE_6
    remaining //= DIM_6
    physical += (remaining % DIM_5) * OUTPUT_STRIDE_5
    remaining //= DIM_5
    physical += (remaining % DIM_4) * OUTPUT_STRIDE_4
    remaining //= DIM_4
    physical += (remaining % DIM_3) * OUTPUT_STRIDE_3
    remaining //= DIM_3
    physical += (remaining % DIM_2) * OUTPUT_STRIDE_2
    remaining //= DIM_2
    physical += (remaining % DIM_1) * OUTPUT_STRIDE_1
    remaining //= DIM_1
    physical += (remaining % DIM_0) * OUTPUT_STRIDE_0
    remaining //= DIM_0
    first, second, _, _ = tl.randint4x(SEED, (logical + OFFSET).to(tl.uint64))
    if DISTRIBUTION == 1:
        value = (first >> (32 - UNIFORM_BITS)).to(tl.float32) * (2.0 ** (-UNIFORM_BITS))
    elif DISTRIBUTION == 2:
        # Open-closed first uniform prevents log(0); second is in [0,1).
        u1 = ((first >> 8).to(tl.float32) + 1.0) * (2.0**-24)
        u2 = (second >> 8).to(tl.float32) * (2.0**-24)
        value = tl.sqrt(-2.0 * tl.log(u1)) * libdevice.cos(6.283185307179586 * u2)
    else:
        uniform = (first >> 8).to(tl.float32) * (2.0**-24)
        value = (uniform < PROBABILITY).to(tl.float32)
    tl.store(output_ptr + physical, value, logical < N_ELEMENTS)


# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0


@triton.jit
def resample_kernel(
    x_ptr,
    y_ptr,
    index_ptr,
    ELEMENTS: tl.constexpr,
    CHANNELS: tl.constexpr,
    ID: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SD: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PD: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    X_N: tl.constexpr,
    X_C: tl.constexpr,
    X_D: tl.constexpr,
    X_H: tl.constexpr,
    X_W: tl.constexpr,
    Y_N: tl.constexpr,
    Y_C: tl.constexpr,
    Y_D: tl.constexpr,
    Y_H: tl.constexpr,
    Y_W: tl.constexpr,
    I_N: tl.constexpr,
    I_C: tl.constexpr,
    I_D: tl.constexpr,
    I_H: tl.constexpr,
    I_W: tl.constexpr,
    MODE: tl.constexpr,
    PADDING: tl.constexpr,
    INDEX: tl.constexpr,
    ALIGN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_LAST: tl.constexpr = False,
):
    logical = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if CHANNELS_LAST:
        c = logical % CHANNELS
        spatial = logical // CHANNELS
        ow = spatial % OW
        oh = spatial // OW % OH
        od = spatial // (OW * OH) % OD
    else:
        ow = logical % OW
        oh = logical // OW % OH
        od = logical // (OW * OH) % OD
        c = logical // (OW * OH * OD) % CHANNELS
    n = logical // (OW * OH * OD * CHANNELS)
    valid = logical < ELEMENTS
    base = n * X_N + c * X_C
    destination = n * Y_N + c * Y_C + od * Y_D + oh * Y_H + ow * Y_W
    if MODE == 4:  # nearest, asymmetric floor mapping
        d = od * ID // OD
        h = oh * IH // OH
        w = ow * IW // OW
        result = tl.load(x_ptr + base + d * X_D + h * X_H + w * X_W, valid, other=0).to(
            tl.float32
        )
    elif MODE == 3:  # bilinear, edge padding
        if ALIGN:
            h = oh.to(tl.float32) * ((IH - 1) / (OH - 1) if OH > 1 else 0.0)
            w = ow.to(tl.float32) * ((IW - 1) / (OW - 1) if OW > 1 else 0.0)
        else:
            h = (oh.to(tl.float32) + 0.5) * (IH / OH) - 0.5
            w = (ow.to(tl.float32) + 0.5) * (IW / OW) - 0.5
        h = tl.minimum(tl.maximum(h, 0.0), IH - 1.0)
        w = tl.minimum(tl.maximum(w, 0.0), IW - 1.0)
        h0 = h.to(tl.int64)
        w0 = w.to(tl.int64)
        h1 = tl.minimum(h0 + 1, IH - 1)
        w1 = tl.minimum(w0 + 1, IW - 1)
        dh = h - h0.to(tl.float32)
        dw = w - w0.to(tl.float32)
        a = tl.load(x_ptr + base + h0 * X_H + w0 * X_W, valid, other=0).to(tl.float32)
        b = tl.load(x_ptr + base + h0 * X_H + w1 * X_W, valid, other=0).to(tl.float32)
        c0 = tl.load(x_ptr + base + h1 * X_H + w0 * X_W, valid, other=0).to(tl.float32)
        d0 = tl.load(x_ptr + base + h1 * X_H + w1 * X_W, valid, other=0).to(tl.float32)
        result = ((1.0 - dw) * a + dw * b) * (1.0 - dh) + (
            (1.0 - dw) * c0 + dw * d0
        ) * dh
    else:
        result = tl.full((BLOCK_SIZE,), float("-inf") if MODE == 5 else 0.0, tl.float32)
        best_index = tl.full((BLOCK_SIZE,), -1, tl.int32)
        samples = tl.zeros((BLOCK_SIZE,), tl.int32)
        for window in range(0, KD * KH * KW):
            d = od * SD - PD + window // (KH * KW)
            h = oh * SH - PH + window // KW % KH
            w = ow * SW - PW + window % KW
            inside = (d >= 0) & (d < ID) & (h >= 0) & (h < IH) & (w >= 0) & (w < IW)
            if PADDING == 1:
                d = tl.minimum(tl.maximum(d, 0), ID - 1)
                h = tl.minimum(tl.maximum(h, 0), IH - 1)
                w = tl.minimum(tl.maximum(w, 0), IW - 1)
                active = valid
            else:
                active = valid & inside
            sample = tl.load(
                x_ptr + base + d * X_D + h * X_H + w * X_W,
                active,
                other=float("-inf") if PADDING == 2 else 0.0,
            ).to(tl.float32)
            if MODE == 5:
                replace = (
                    (sample > result)
                    | (best_index == -1)
                    | ((sample != sample) & (result == result))
                )
                best_index = tl.where(replace, window, best_index)
                result = tl.maximum(result, sample)
            else:
                result += sample
                samples += inside.to(tl.int32)
        if MODE == 1:
            result = tl.where(samples > 0, result / tl.maximum(samples, 1), 0.0)
        elif MODE == 2:
            result /= KD * KH * KW
        if INDEX:
            tl.store(
                index_ptr + n * I_N + c * I_C + od * I_D + oh * I_H + ow * I_W,
                best_index,
                valid,
            )
    tl.store(y_ptr + destination, result, valid)


# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0


@triton.jit
def genstats_kernel(
    x_ptr,
    sum_ptr,
    sq_sum_ptr,
    CHANNELS: tl.constexpr,
    REDUCTION: tl.constexpr,
    SPATIAL: tl.constexpr,
    SUM_CHANNEL_STRIDE: tl.constexpr,
    SQ_SUM_CHANNEL_STRIDE: tl.constexpr,
    DIM_0: tl.constexpr,
    DIM_1: tl.constexpr,
    DIM_2: tl.constexpr,
    DIM_3: tl.constexpr,
    DIM_4: tl.constexpr,
    DIM_5: tl.constexpr,
    DIM_6: tl.constexpr,
    DIM_7: tl.constexpr,
    STRIDE_0: tl.constexpr,
    STRIDE_1: tl.constexpr,
    STRIDE_2: tl.constexpr,
    STRIDE_3: tl.constexpr,
    STRIDE_4: tl.constexpr,
    STRIDE_5: tl.constexpr,
    STRIDE_6: tl.constexpr,
    STRIDE_7: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    channel = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    partial_sum = tl.zeros((BLOCK_SIZE,), tl.float32)
    partial_square = tl.zeros((BLOCK_SIZE,), tl.float32)
    for start in range(0, REDUCTION, BLOCK_SIZE):
        reduced = start + offsets
        logical = (
            reduced.to(tl.int64) // SPATIAL * CHANNELS + channel
        ) * SPATIAL + reduced % SPATIAL
        physical = tl.zeros((BLOCK_SIZE,), tl.int64)
        physical += (logical % DIM_7) * STRIDE_7
        logical //= DIM_7
        physical += (logical % DIM_6) * STRIDE_6
        logical //= DIM_6
        physical += (logical % DIM_5) * STRIDE_5
        logical //= DIM_5
        physical += (logical % DIM_4) * STRIDE_4
        logical //= DIM_4
        physical += (logical % DIM_3) * STRIDE_3
        logical //= DIM_3
        physical += (logical % DIM_2) * STRIDE_2
        logical //= DIM_2
        physical += (logical % DIM_1) * STRIDE_1
        logical //= DIM_1
        physical += (logical % DIM_0) * STRIDE_0
        logical //= DIM_0
        x = tl.load(x_ptr + physical, reduced < REDUCTION, other=0).to(tl.float32)
        partial_sum += x
        partial_square += x * x
    tl.store(sum_ptr + channel * SUM_CHANNEL_STRIDE, tl.sum(partial_sum, 0))
    tl.store(sq_sum_ptr + channel * SQ_SUM_CHANNEL_STRIDE, tl.sum(partial_square, 0))


@triton.jit
def bn_finalize_kernel(
    sum_ptr,
    sq_sum_ptr,
    scale_ptr,
    bias_ptr,
    prev_mean_ptr,
    prev_var_ptr,
    eq_scale_ptr,
    eq_bias_ptr,
    mean_ptr,
    inv_ptr,
    next_mean_ptr,
    next_var_ptr,
    SUM_STRIDE: tl.constexpr,
    SQ_SUM_STRIDE: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    BIAS_STRIDE: tl.constexpr,
    PREV_MEAN_STRIDE: tl.constexpr,
    PREV_VAR_STRIDE: tl.constexpr,
    EQ_SCALE_STRIDE: tl.constexpr,
    EQ_BIAS_STRIDE: tl.constexpr,
    MEAN_STRIDE: tl.constexpr,
    INV_STRIDE: tl.constexpr,
    NEXT_MEAN_STRIDE: tl.constexpr,
    NEXT_VAR_STRIDE: tl.constexpr,
    CHANNELS: tl.constexpr,
    COUNT: tl.constexpr,
    EPSILON: tl.constexpr,
    MOMENTUM: tl.constexpr,
    HAS_RUNNING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    channel = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = channel < CHANNELS
    total = tl.load(sum_ptr + channel * SUM_STRIDE, valid, other=0).to(tl.float32)
    square = tl.load(sq_sum_ptr + channel * SQ_SUM_STRIDE, valid, other=0).to(
        tl.float32
    )
    scale = tl.load(scale_ptr + channel * SCALE_STRIDE, valid, other=0).to(tl.float32)
    bias = tl.load(bias_ptr + channel * BIAS_STRIDE, valid, other=0).to(tl.float32)
    # Ascend vector arithmetic finalizes FP32 moments in FP32, as ACLNN does.
    mean = total / COUNT
    variance = tl.maximum(square / COUNT - mean * mean, 0.0)
    inverse = 1.0 / tl.sqrt(variance + EPSILON)
    equivalent_scale = scale * inverse
    tl.store(eq_scale_ptr + channel * EQ_SCALE_STRIDE, equivalent_scale, valid)
    tl.store(
        eq_bias_ptr + channel * EQ_BIAS_STRIDE,
        bias - mean * equivalent_scale,
        valid,
    )
    tl.store(mean_ptr + channel * MEAN_STRIDE, mean, valid)
    tl.store(inv_ptr + channel * INV_STRIDE, inverse, valid)
    if HAS_RUNNING:
        previous_mean = tl.load(
            prev_mean_ptr + channel * PREV_MEAN_STRIDE, valid, other=0
        ).to(tl.float32)
        previous_variance = tl.load(
            prev_var_ptr + channel * PREV_VAR_STRIDE, valid, other=0
        ).to(tl.float32)
        unbiased_variance = variance * (COUNT / (COUNT - 1.0) if COUNT > 1.0 else 0.0)
        tl.store(
            next_mean_ptr + channel * NEXT_MEAN_STRIDE,
            (1.0 - MOMENTUM) * previous_mean + MOMENTUM * mean,
            valid,
        )
        tl.store(
            next_var_ptr + channel * NEXT_VAR_STRIDE,
            (1.0 - MOMENTUM) * previous_variance + MOMENTUM * unbiased_variance,
            valid,
        )


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


@triton.jit
def _apply_unary_elementwise_operation(
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
    if OPERATION == 2:
        result = tl.where(
            value_f32 < lower_clip,
            lower_clip + negative_slope * (value_f32 - lower_clip),
            value_f32,
        )
        if HAS_UPPER_CLIP:
            result = tl.minimum(result, upper_clip)
    elif OPERATION == 9:
        result = tl.abs(value_f32)
    elif OPERATION == 10:
        result = tl.ceil(value_f32)
    elif OPERATION == 11:
        result = tl.cos(value_f32)
    elif OPERATION == 4:
        result = libdevice.erf(value_f32)
    elif OPERATION == 6:
        result = tl.exp(value_f32)
    elif OPERATION == 12:
        result = tl.floor(value_f32)
    elif OPERATION == 5:
        result = value
    elif OPERATION == 7:
        result = tl.log(value_f32)
    elif OPERATION == 8:
        result = -value
    elif OPERATION == 16:
        result = 1.0 / value_f32
    elif OPERATION == 13:
        result = tl.rsqrt(value_f32)
    elif OPERATION == 14:
        result = tl.sin(value_f32)
    elif OPERATION == 3:
        result = tl.sqrt(value_f32)
    elif OPERATION == 15:
        result = libdevice.tan(value_f32)
    elif OPERATION == 24:
        result = value == 0
    elif OPERATION == 33:
        result = tl.sigmoid(value_f32)
    elif OPERATION == 34:
        result = 2.0 * tl.sigmoid(2.0 * value_f32) - 1.0
    elif OPERATION == 35:
        result = tl.where(
            value_f32 > 0.0,
            value_f32,
            ELU_ALPHA * (tl.exp(value_f32) - 1.0),
        )
    elif OPERATION == 36:
        result = 0.5 * value_f32 * (1.0 + libdevice.erf(value_f32 * 0.7071067811865476))
    elif OPERATION == 37:
        scaled = SOFTPLUS_BETA * value_f32
        # Factor the positive term before multiplying by beta: beta * x can
        # overflow although softplus(x, beta) is still representable.
        result = (
            tl.maximum(value_f32, 0.0)
            + libdevice.log1p(tl.exp(-tl.abs(scaled))) / SOFTPLUS_BETA
        )
    elif OPERATION == 38:
        result = value_f32 * tl.sigmoid(SWISH_BETA * value_f32)
    elif OPERATION == 39:
        inner = 0.7978845608028654 * (
            value_f32 + 0.044715 * value_f32 * value_f32 * value_f32
        )
        tanh_inner = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
        result = 0.5 * value_f32 * (1.0 + tanh_inner)
    return result


@triton.jit
def unary_elementwise_contiguous_kernel(
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
    TILES_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program_base = tl.program_id(0).to(tl.int64) * BLOCK_SIZE * TILES_PER_PROGRAM
    for tile_index in tl.static_range(TILES_PER_PROGRAM):
        offsets = program_base + tile_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        active = offsets < n_elements
        value = tl.load(in_ptr + offsets, mask=active, other=0.0)
        result = _apply_unary_elementwise_operation(
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
            mask=active,
        )


@triton.jit
def unary_elementwise_strided_kernel(
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
    STRIDED: tl.constexpr,
    OPERATION: tl.constexpr,
    negative_slope: tl.constexpr,
    lower_clip: tl.constexpr,
    upper_clip: tl.constexpr,
    HAS_UPPER_CLIP: tl.constexpr,
    SWISH_BETA: tl.constexpr,
    ELU_ALPHA: tl.constexpr,
    SOFTPLUS_BETA: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program_base = tl.program_id(0).to(tl.int64) * BLOCK_SIZE * TILES_PER_PROGRAM
    for tile_index in tl.static_range(TILES_PER_PROGRAM):
        offsets = program_base + tile_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        active = offsets < n_elements

        if STRIDED:
            remaining = offsets
            input_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
            output_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

            coordinate = remaining % DIM_7
            remaining //= DIM_7
            input_offsets += coordinate * INPUT_STRIDE_7
            output_offsets += coordinate * OUTPUT_STRIDE_7
            coordinate = remaining % DIM_6
            remaining //= DIM_6
            input_offsets += coordinate * INPUT_STRIDE_6
            output_offsets += coordinate * OUTPUT_STRIDE_6
            coordinate = remaining % DIM_5
            remaining //= DIM_5
            input_offsets += coordinate * INPUT_STRIDE_5
            output_offsets += coordinate * OUTPUT_STRIDE_5
            coordinate = remaining % DIM_4
            remaining //= DIM_4
            input_offsets += coordinate * INPUT_STRIDE_4
            output_offsets += coordinate * OUTPUT_STRIDE_4
            coordinate = remaining % DIM_3
            remaining //= DIM_3
            input_offsets += coordinate * INPUT_STRIDE_3
            output_offsets += coordinate * OUTPUT_STRIDE_3
            coordinate = remaining % DIM_2
            remaining //= DIM_2
            input_offsets += coordinate * INPUT_STRIDE_2
            output_offsets += coordinate * OUTPUT_STRIDE_2
            coordinate = remaining % DIM_1
            remaining //= DIM_1
            input_offsets += coordinate * INPUT_STRIDE_1
            output_offsets += coordinate * OUTPUT_STRIDE_1
            coordinate = remaining % DIM_0
            input_offsets += coordinate * INPUT_STRIDE_0
            output_offsets += coordinate * OUTPUT_STRIDE_0
        else:
            input_offsets = offsets
            output_offsets = offsets

        value = tl.load(in_ptr + input_offsets, mask=active, other=0.0)
        result = _apply_unary_elementwise_operation(
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
            mask=active,
        )
