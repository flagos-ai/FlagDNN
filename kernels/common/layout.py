# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Materialization kernel for reshape, transpose, and slice Graph nodes.

The Python APIs in ``flag_dnn.ops`` express these operations as views.  Native
Graph execution may need a physical output, so lowering supplies one common
logical-shape/stride ABI consumed here.
"""

import triton
import triton.language as tl


@triton.jit
def layout_copy_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    INPUT_BASE: tl.constexpr,
    INPUT_DIM_0: tl.constexpr,
    INPUT_DIM_1: tl.constexpr,
    INPUT_DIM_2: tl.constexpr,
    INPUT_DIM_3: tl.constexpr,
    INPUT_DIM_4: tl.constexpr,
    INPUT_DIM_5: tl.constexpr,
    INPUT_DIM_6: tl.constexpr,
    INPUT_DIM_7: tl.constexpr,
    INPUT_STRIDE_0: tl.constexpr,
    INPUT_STRIDE_1: tl.constexpr,
    INPUT_STRIDE_2: tl.constexpr,
    INPUT_STRIDE_3: tl.constexpr,
    INPUT_STRIDE_4: tl.constexpr,
    INPUT_STRIDE_5: tl.constexpr,
    INPUT_STRIDE_6: tl.constexpr,
    INPUT_STRIDE_7: tl.constexpr,
    OUTPUT_DIM_0: tl.constexpr,
    OUTPUT_DIM_1: tl.constexpr,
    OUTPUT_DIM_2: tl.constexpr,
    OUTPUT_DIM_3: tl.constexpr,
    OUTPUT_DIM_4: tl.constexpr,
    OUTPUT_DIM_5: tl.constexpr,
    OUTPUT_DIM_6: tl.constexpr,
    OUTPUT_DIM_7: tl.constexpr,
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
    logical = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(
        0, BLOCK_SIZE
    )
    active = logical < n_elements
    input_remaining = logical
    output_remaining = logical
    input_offsets = tl.full((BLOCK_SIZE,), INPUT_BASE, dtype=tl.int64)
    output_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

    coordinate = input_remaining % INPUT_DIM_7
    input_remaining //= INPUT_DIM_7
    input_offsets += coordinate * INPUT_STRIDE_7
    coordinate = input_remaining % INPUT_DIM_6
    input_remaining //= INPUT_DIM_6
    input_offsets += coordinate * INPUT_STRIDE_6
    coordinate = input_remaining % INPUT_DIM_5
    input_remaining //= INPUT_DIM_5
    input_offsets += coordinate * INPUT_STRIDE_5
    coordinate = input_remaining % INPUT_DIM_4
    input_remaining //= INPUT_DIM_4
    input_offsets += coordinate * INPUT_STRIDE_4
    coordinate = input_remaining % INPUT_DIM_3
    input_remaining //= INPUT_DIM_3
    input_offsets += coordinate * INPUT_STRIDE_3
    coordinate = input_remaining % INPUT_DIM_2
    input_remaining //= INPUT_DIM_2
    input_offsets += coordinate * INPUT_STRIDE_2
    coordinate = input_remaining % INPUT_DIM_1
    input_remaining //= INPUT_DIM_1
    input_offsets += coordinate * INPUT_STRIDE_1
    input_offsets += (input_remaining % INPUT_DIM_0) * INPUT_STRIDE_0

    coordinate = output_remaining % OUTPUT_DIM_7
    output_remaining //= OUTPUT_DIM_7
    output_offsets += coordinate * OUTPUT_STRIDE_7
    coordinate = output_remaining % OUTPUT_DIM_6
    output_remaining //= OUTPUT_DIM_6
    output_offsets += coordinate * OUTPUT_STRIDE_6
    coordinate = output_remaining % OUTPUT_DIM_5
    output_remaining //= OUTPUT_DIM_5
    output_offsets += coordinate * OUTPUT_STRIDE_5
    coordinate = output_remaining % OUTPUT_DIM_4
    output_remaining //= OUTPUT_DIM_4
    output_offsets += coordinate * OUTPUT_STRIDE_4
    coordinate = output_remaining % OUTPUT_DIM_3
    output_remaining //= OUTPUT_DIM_3
    output_offsets += coordinate * OUTPUT_STRIDE_3
    coordinate = output_remaining % OUTPUT_DIM_2
    output_remaining //= OUTPUT_DIM_2
    output_offsets += coordinate * OUTPUT_STRIDE_2
    coordinate = output_remaining % OUTPUT_DIM_1
    output_remaining //= OUTPUT_DIM_1
    output_offsets += coordinate * OUTPUT_STRIDE_1
    output_offsets += (output_remaining % OUTPUT_DIM_0) * OUTPUT_STRIDE_0

    value = tl.load(input_ptr + input_offsets, mask=active, other=0.0)
    tl.store(output_ptr + output_offsets, value, mask=active)


# -----------------------------------------------------------------------------
# Kernel algorithm variants. Native dispatch selects only registry-declared
# entry points; auxiliary kernels remain available to explicit compiler policy.
# -----------------------------------------------------------------------------


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
    pid = tl.program_id(0)
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
    pid = tl.program_id(0)
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


@triton.jit
def _gen_index_kernel(
    out_ptr,
    n_elements,
    axis_size: tl.constexpr,
    inner_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    axis_index = (offsets // inner_size) % axis_size
    tl.store(out_ptr + offsets, axis_index, mask=mask)


@triton.jit
def _identity_copy_kernel(
    input_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, values, mask=mask)
