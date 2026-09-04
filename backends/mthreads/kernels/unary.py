# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""MTGPU-specialized Triton kernels for unary pointwise operations."""

import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.language.extra.musa import libdevice as musa_libdevice


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
        result = tl.log2(value_f32) * 0.6931471805599453
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
        result = musa_libdevice.fast_tanh(value_f32)
    elif OPERATION == 35:
        result = tl.where(
            value_f32 > 0.0,
            value_f32,
            ELU_ALPHA * (tl.exp(value_f32) - 1.0),
        )
    elif OPERATION == 36:
        result = musa_libdevice.fast_gelu(value_f32)
    elif OPERATION == 37:
        scaled = SOFTPLUS_BETA * value_f32
        result = (
            tl.maximum(scaled, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(scaled)))
        ) / SOFTPLUS_BETA
    elif OPERATION == 38:
        result = value_f32 * tl.sigmoid(SWISH_BETA * value_f32)
    elif OPERATION == 39:
        inner = 0.7978845608028654 * (
            value_f32 + 0.044715 * value_f32 * value_f32 * value_f32
        )
        tanh_inner = musa_libdevice.fast_tanh(inner)
        result = 0.5 * value_f32 * (1.0 + tanh_inner)
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
    TILES_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # n_elements is an i32 runtime scalar. Keeping the dense linear index in
    # i32 avoids costly 64-bit integer arithmetic on MTGPU.
    program_base = tl.program_id(0) * BLOCK_SIZE * TILES_PER_PROGRAM
    full_program = (
        program_base + BLOCK_SIZE * TILES_PER_PROGRAM <= n_elements
    )
    for tile_index in tl.static_range(TILES_PER_PROGRAM):
        offsets = (
            program_base + tile_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        )
        active = offsets < n_elements
        if full_program:
            value = tl.load(in_ptr + offsets)
        else:
            value = tl.load(in_ptr + offsets, mask=active, other=0.0)
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
        if full_program:
            tl.store(out_ptr + offsets, result.to(out_ptr.dtype.element_ty))
        else:
            tl.store(
                out_ptr + offsets,
                result.to(out_ptr.dtype.element_ty),
                mask=active,
            )


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
    program_base = (
        tl.program_id(0).to(tl.int64) * BLOCK_SIZE * TILES_PER_PROGRAM
    )
    for tile_index in tl.static_range(TILES_PER_PROGRAM):
        offsets = (
            program_base + tile_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        )
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
            mask=active,
        )
