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

"""Platform-neutral Triton kernels for activation gradients."""

import triton
import triton.language as tl
from triton.language.extra import libdevice


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
