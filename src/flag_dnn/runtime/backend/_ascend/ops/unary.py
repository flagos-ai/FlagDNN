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

"""Ascend-only dense unary kernels used by graph prepared replay."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import torch
import triton
import triton.language as tl
from triton.language.extra.cann import extension as cann_extension

from flag_dnn.graph.device import is_runtime_device_tensor
from flag_dnn.utils import triton_lang_extension as tle
from flag_dnn.utils.libentry import libentry
from flag_dnn.utils.triton_lang_helper import tl_extra_shim as libdevice

from .binary import make_balanced_core_loop_grid, make_core_loop_grid

Grid = Callable[[dict[str, Any]], tuple[int, ...]]


def _fixed_grid(program_count: int) -> Grid:
    def grid(_meta: dict[str, Any]) -> tuple[int, ...]:
        return (program_count,)

    return grid


_FLOAT_DTYPES = {"float16", "bfloat16", "float32"}
_SIMPLE_UNARY_OPS = {
    "abs",
    "neg",
    "sqrt",
    "exp",
    "log",
    "rsqrt",
    "reciprocal",
    "ceil",
    "floor",
    "erf",
    "sin",
    "cos",
    "tan",
    "relu",
    "sigmoid",
    "tanh",
    "swish",
    "silu",
    "gelu",
    "gelu_approx_tanh",
    "leaky_relu",
    "elu",
    "softplus",
}
_MATH_HEAVY_OPS = {
    "exp",
    "log",
    "erf",
    "sin",
    "cos",
    "tan",
    "sigmoid",
    "tanh",
    "swish",
    "silu",
    "gelu",
    "gelu_tanh",
    "elu",
    "softplus",
}
_WIDE_TILE_OPS = {
    "abs",
    "neg",
    "sqrt",
    "rsqrt",
    "reciprocal",
    "ceil",
    "floor",
    "logical_not",
}


def get_dense_unary_block_size(
    op_type: str, n_elements: int, dtype: Any
) -> int:
    if n_elements <= 1024:
        return 1024
    if n_elements <= 4096:
        return 2048
    if n_elements <= 196608:
        max_block_size = 4096
    elif n_elements <= 401408:
        max_block_size = 8192
    elif n_elements <= 786432:
        max_block_size = 16384
    else:
        max_block_size = 8192
    if "float32" in str(dtype) and op_type not in _WIDE_TILE_OPS:
        max_block_size = min(max_block_size, 8192)
    if op_type in _MATH_HEAVY_OPS:
        max_block_size = min(max_block_size, 8192)
        if n_elements == 524288:
            max_block_size = min(max_block_size, 4096)
    if op_type in {"leaky_relu", "leaky_relu_max"}:
        max_block_size = min(max_block_size, 8192)
    if op_type in {"elu", "softplus"}:
        max_block_size = min(max_block_size, 4096)
    return max_block_size


def _can_use_aligned_loop(n_elements: int, block_size: int) -> bool:
    return n_elements >= 262144 and n_elements % block_size == 0


@triton.jit
def _unary_result(x, parameter0, parameter1, OP_TYPE: tl.constexpr):
    if OP_TYPE == "abs":
        result = tl.abs(x)
    elif OP_TYPE == "neg":
        result = -x
    elif OP_TYPE == "sqrt":
        result = tl.sqrt(x.to(tl.float32))
    elif OP_TYPE == "exp":
        result = libdevice.exp(x.to(tl.float32))
    elif OP_TYPE == "log":
        result = libdevice.log(x.to(tl.float32))
    elif OP_TYPE == "rsqrt":
        value = x.to(tl.float32)
        result = libdevice.rsqrt(value)
    elif OP_TYPE == "reciprocal":
        value = x.to(tl.float32)
        if tl.constexpr(x.dtype == tl.float32):
            result = libdevice.reciprocal(value)
        else:
            result = 1.0 / value
    elif OP_TYPE == "ceil":
        result = tl.math.ceil(x.to(tl.float32))
    elif OP_TYPE == "floor":
        result = tl.math.floor(x.to(tl.float32))
    elif OP_TYPE == "erf":
        result = libdevice.erf(x.to(tl.float32))
    elif OP_TYPE == "sin":
        result = libdevice.sin(x.to(tl.float32))
    elif OP_TYPE == "cos":
        result = libdevice.cos(x.to(tl.float32))
    elif OP_TYPE == "tan":
        result = libdevice.tan(x.to(tl.float32))
    elif OP_TYPE == "relu":
        if tl.constexpr(x.dtype == tl.bfloat16):
            result = libdevice.relu(x.to(tl.float32))
        else:
            result = libdevice.relu(x)
    elif OP_TYPE == "sigmoid":
        value = x.to(tl.float32)
        result = tl.sigmoid(value)
    elif OP_TYPE == "tanh":
        result = libdevice.tanh(x.to(tl.float32))
    elif OP_TYPE == "swish":
        value = x.to(tl.float32)
        sigmoid = 1.0 / (1.0 + libdevice.exp(-parameter0 * value))
        result = value * sigmoid
    elif OP_TYPE == "gelu":
        value = x.to(tl.float32)
        magnitude = tl.minimum(tl.abs(value), 5.75)
        square = magnitude * magnitude
        polynomial = tl.fma(
            0.0007174646351407614,
            square,
            -0.07410068966917274,
        )
        polynomial = tl.fma(polynomial, square, -1.5949169421333256)
        exponent = polynomial * value
        denominator = 1.0 + libdevice.exp(exponent)
        if tl.constexpr(x.dtype == tl.float32):
            result = value * libdevice.reciprocal(denominator)
        else:
            result = value / denominator
    elif OP_TYPE == "gelu_tanh":
        value = x.to(tl.float32)
        square = value * value
        exponent = value * tl.fma(
            -0.071354816245515,
            square,
            -1.595769121,
        )
        denominator = 1.0 + libdevice.exp(exponent)
        result = value * libdevice.reciprocal(denominator)
    elif OP_TYPE == "leaky_relu":
        slope = parameter0.to(x.dtype)
        result = tl.where(x > 0.0, x, x * slope)
    elif OP_TYPE == "leaky_relu_max":
        slope = parameter0.to(x.dtype)
        result = tl.maximum(x, x * slope)
    elif OP_TYPE == "elu":
        value = x.to(tl.float32)
        negative = parameter0 * (libdevice.exp(value) - 1.0)
        result = tl.where(value > 0.0, value, negative)
    elif OP_TYPE == "softplus":
        value = x.to(tl.float32)
        scaled = parameter0 * value
        abs_scaled = tl.abs(scaled)
        positive = tl.where(scaled > 0.0, scaled, 0.0)
        stable = (
            positive + libdevice.log1p(libdevice.exp(-abs_scaled))
        ) / parameter0
        result = tl.where(scaled > parameter1, value, stable)
    elif OP_TYPE == "logical_not":
        result = x == 0
    return result


@libentry()
@triton.jit
def unary_aligned_core_loop_kernel(
    input_ptr,
    output_ptr,
    parameter0,
    parameter1,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    OP_TYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    first_block = tle.program_id(0) * BLOCKS_PER_PROGRAM
    for local_block in range(0, BLOCKS_PER_PROGRAM):
        offsets = (first_block + local_block) * BLOCK_SIZE + tl.arange(
            0, BLOCK_SIZE
        )
        value = tl.load(input_ptr + offsets)
        result = _unary_result(value, parameter0, parameter1, OP_TYPE)
        tl.store(
            output_ptr + offsets,
            result.to(output_ptr.dtype.element_ty),
        )


@libentry()
@triton.jit
def unary_multibuffer_core_loop_kernel(
    input_ptr,
    output_ptr,
    parameter0,
    parameter1,
    N_BLOCKS: tl.constexpr,
    PROGRAM_COUNT: tl.constexpr,
    OP_TYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    for block_idx in tl.range(pid, N_BLOCKS, PROGRAM_COUNT):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        value = tl.load(input_ptr + offsets)
        cann_extension.multibuffer(value, 2)
        result = _unary_result(value, parameter0, parameter1, OP_TYPE)
        tl.store(
            output_ptr + offsets,
            result.to(output_ptr.dtype.element_ty),
        )


@libentry()
@triton.jit
def unary_neg_16bit_multibuffer_core_loop_kernel(
    input_ptr,
    output_ptr,
    parameter0,
    parameter1,
    N_BLOCKS: tl.constexpr,
    PROGRAM_COUNT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    for block_idx in tl.range(pid, N_BLOCKS, PROGRAM_COUNT):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        value = tl.load(input_ptr + offsets)
        cann_extension.multibuffer(value, 2)
        bits = value.to(tl.uint16, bitcast=True) ^ 0x8000
        result = bits.to(value.dtype, bitcast=True)
        tl.store(output_ptr + offsets, result)


@libentry()
@triton.jit
def unary_permuted_core_loop_kernel(
    input_ptr,
    output_ptr,
    parameter0,
    parameter1,
    N_BLOCKS: tl.constexpr,
    PROGRAM_COUNT: tl.constexpr,
    PHASES: tl.constexpr,
    STRIDE: tl.constexpr,
    SHIFT: tl.constexpr,
    USE_MULTIBUFFER: tl.constexpr,
    OP_TYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    for phase in range(0, PHASES):
        logical_pid = (pid * STRIDE + phase * SHIFT) % PROGRAM_COUNT
        block_idx = phase * PROGRAM_COUNT + logical_pid
        if block_idx < N_BLOCKS:
            offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            value = tl.load(input_ptr + offsets)
            if USE_MULTIBUFFER:
                cann_extension.multibuffer(value, 2)
            result = _unary_result(value, parameter0, parameter1, OP_TYPE)
            tl.store(
                output_ptr + offsets,
                result.to(output_ptr.dtype.element_ty),
            )


@libentry()
@triton.jit
def swish_constexpr_core_loop_kernel(
    input_ptr,
    output_ptr,
    BETA: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    PROGRAM_COUNT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    for block_idx in tl.range(pid, N_BLOCKS, PROGRAM_COUNT):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        value = tl.load(input_ptr + offsets).to(tl.float32)
        cann_extension.multibuffer(value, 2)
        denominator = 1.0 + libdevice.exp(-BETA * value)
        result = value / denominator
        tl.store(
            output_ptr + offsets,
            result.to(output_ptr.dtype.element_ty),
        )


@libentry()
@triton.jit
def unary_core_loop_kernel(
    input_ptr,
    output_ptr,
    parameter0,
    parameter1,
    N_ELEMENTS: tl.constexpr,
    PROGRAM_COUNT: tl.constexpr,
    OP_TYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_full_blocks: tl.constexpr = N_ELEMENTS // BLOCK_SIZE

    for block_idx in tl.range(pid, num_full_blocks, PROGRAM_COUNT):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        value = tl.load(input_ptr + offsets)
        result = _unary_result(value, parameter0, parameter1, OP_TYPE)
        tl.store(
            output_ptr + offsets,
            result.to(output_ptr.dtype.element_ty),
        )

    tail_elements: tl.constexpr = N_ELEMENTS - num_full_blocks * BLOCK_SIZE
    if tail_elements > 0:
        tail_pid: tl.constexpr = num_full_blocks % PROGRAM_COUNT
        if pid == tail_pid:
            offsets = num_full_blocks * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < N_ELEMENTS
            value = tl.load(input_ptr + offsets, mask=mask)
            result = _unary_result(value, parameter0, parameter1, OP_TYPE)
            tl.store(
                output_ptr + offsets,
                result.to(output_ptr.dtype.element_ty),
                mask=mask,
            )


@libentry()
@triton.jit
def unary_tiled_kernel(
    input_ptr,
    output_ptr,
    parameter0,
    parameter1,
    N_ELEMENTS: tl.constexpr,
    OP_TYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS
    value = tl.load(input_ptr + offsets, mask=mask)
    result = _unary_result(value, parameter0, parameter1, OP_TYPE)
    tl.store(
        output_ptr + offsets,
        result.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


@libentry()
@triton.jit
def unary_395523_balanced_kernel(
    input_ptr,
    output_ptr,
    parameter0,
    parameter1,
    OP_TYPE: tl.constexpr,
):
    pid = tle.program_id(0)

    main_offsets = pid * 8192 + tl.arange(0, 8192)
    main_value = tl.load(input_ptr + main_offsets)
    main_result = _unary_result(main_value, parameter0, parameter1, OP_TYPE)
    tl.store(
        output_ptr + main_offsets,
        main_result.to(output_ptr.dtype.element_ty),
    )

    tail_offsets = 393216 + pid * 64 + tl.arange(0, 64)
    tail_mask = tail_offsets < 395523
    tail_value = tl.load(input_ptr + tail_offsets, mask=tail_mask)
    tail_result = _unary_result(tail_value, parameter0, parameter1, OP_TYPE)
    tl.store(
        output_ptr + tail_offsets,
        tail_result.to(output_ptr.dtype.element_ty),
        mask=tail_mask,
    )


@libentry()
@triton.jit
def unary_395523_tail4096_kernel(
    input_ptr,
    output_ptr,
    parameter0,
    parameter1,
    OP_TYPE: tl.constexpr,
):
    pid = tle.program_id(0)

    main_offsets = pid * 8192 + tl.arange(0, 8192)
    main_value = tl.load(input_ptr + main_offsets)
    main_result = _unary_result(main_value, parameter0, parameter1, OP_TYPE)
    tl.store(
        output_ptr + main_offsets,
        main_result.to(output_ptr.dtype.element_ty),
    )

    if pid == 0:
        tail_offsets = 393216 + tl.arange(0, 4096)
        tail_mask = tail_offsets < 395523
        tail_value = tl.load(
            input_ptr + tail_offsets,
            mask=tail_mask,
        )
        tail_result = _unary_result(
            tail_value, parameter0, parameter1, OP_TYPE
        )
        tl.store(
            output_ptr + tail_offsets,
            tail_result.to(output_ptr.dtype.element_ty),
            mask=tail_mask,
        )


@triton.jit
def _unary_balanced_chunk(
    input_ptr,
    output_ptr,
    parameter0,
    parameter1,
    program_base,
    N_ELEMENTS: tl.constexpr,
    ELEMENTS_PER_PROGRAM: tl.constexpr,
    LOCAL_OFFSET: tl.constexpr,
    OP_TYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    local_offsets = LOCAL_OFFSET + tl.arange(0, BLOCK_SIZE)
    offsets = program_base + local_offsets
    mask = (local_offsets < ELEMENTS_PER_PROGRAM) & (offsets < N_ELEMENTS)
    value = tl.load(input_ptr + offsets, mask=mask)
    result = _unary_result(value, parameter0, parameter1, OP_TYPE)
    tl.store(
        output_ptr + offsets,
        result.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _unary_exact_chunk(
    input_ptr,
    output_ptr,
    parameter0,
    parameter1,
    offset,
    OP_TYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    value = tl.load(input_ptr + offsets)
    result = _unary_result(value, parameter0, parameter1, OP_TYPE)
    tl.store(
        output_ptr + offsets,
        result.to(output_ptr.dtype.element_ty),
    )


@triton.jit
def _rsqrt_float32_sqrt_reciprocal_exact_chunk(
    input_ptr,
    output_ptr,
    offset,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    value = tl.load(input_ptr + offsets)
    result = libdevice.reciprocal(tl.sqrt(value))
    tl.store(output_ptr + offsets, result)


@triton.jit
def _rsqrt_float32_sqrt_reciprocal_masked_chunk(
    input_ptr,
    output_ptr,
    offset,
    valid_elements,
    BLOCK_SIZE: tl.constexpr,
):
    lanes = tl.arange(0, BLOCK_SIZE)
    mask = lanes < valid_elements
    value = tl.load(
        input_ptr + offset + lanes,
        mask=mask,
        other=1.0,
    )
    result = libdevice.reciprocal(tl.sqrt(value))
    tl.store(output_ptr + offset + lanes, result, mask=mask)


@triton.jit
def _rsqrt_math_exact_chunk(
    input_ptr,
    output_ptr,
    offset,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    value = tl.load(input_ptr + offsets).to(tl.float32)
    result = libdevice.rsqrt(value)
    tl.store(
        output_ptr + offsets,
        result.to(output_ptr.dtype.element_ty),
    )


@libentry()
@triton.jit
def unary_1000_two_program_kernel(
    input_ptr,
    output_ptr,
    parameter0,
    parameter1,
    OP_TYPE: tl.constexpr,
):
    pid = tle.program_id(0)
    if pid == 0:
        _unary_exact_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            0,
            OP_TYPE,
            512,
        )
        _unary_exact_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            960,
            OP_TYPE,
            32,
        )
        _unary_exact_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            992,
            OP_TYPE,
            8,
        )
    else:
        _unary_exact_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            512,
            OP_TYPE,
            256,
        )
        _unary_exact_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            768,
            OP_TYPE,
            128,
        )
        _unary_exact_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            896,
            OP_TYPE,
            64,
        )


@libentry()
@triton.jit
def unary_176085_masked_tail_kernel(
    input_ptr,
    output_ptr,
    parameter0,
    parameter1,
    OP_TYPE: tl.constexpr,
):
    pid = tle.program_id(0)
    if pid < 42:
        _unary_exact_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            pid * 4096,
            OP_TYPE,
            4096,
        )
    else:
        lanes = tl.arange(0, 4096)
        mask = lanes < 4053
        value = tl.load(
            input_ptr + 172032 + lanes,
            mask=mask,
            other=1.0,
        )
        result = _unary_result(
            value,
            parameter0,
            parameter1,
            OP_TYPE,
        )
        tl.store(
            output_ptr + 172032 + lanes,
            result.to(output_ptr.dtype.element_ty),
            mask=mask,
        )


@libentry()
@triton.jit
def unary_293475_exact_split_kernel(
    input_ptr,
    output_ptr,
    parameter0,
    parameter1,
    OP_TYPE: tl.constexpr,
):
    pid = tle.program_id(0)
    if pid < 35:
        _unary_exact_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            pid * 8192,
            OP_TYPE,
            8192,
        )
    else:
        _unary_exact_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            286720,
            OP_TYPE,
            4096,
        )
        _unary_exact_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            290816,
            OP_TYPE,
            2048,
        )
        _unary_exact_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            292864,
            OP_TYPE,
            512,
        )
        _unary_exact_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            293376,
            OP_TYPE,
            64,
        )
        _unary_exact_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            293440,
            OP_TYPE,
            32,
        )
        _unary_exact_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            293472,
            OP_TYPE,
            2,
        )
        _unary_exact_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            293474,
            OP_TYPE,
            1,
        )


@libentry()
@triton.jit
def unary_395523_exact_tail_kernel(
    input_ptr,
    output_ptr,
    parameter0,
    parameter1,
    OP_TYPE: tl.constexpr,
):
    pid = tle.program_id(0)
    _unary_exact_chunk(
        input_ptr,
        output_ptr,
        parameter0,
        parameter1,
        pid * 8192,
        OP_TYPE,
        8192,
    )
    if pid < 36:
        _unary_exact_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            393216 + pid * 64,
            OP_TYPE,
            64,
        )
    elif pid == 36:
        _unary_exact_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            395520,
            OP_TYPE,
            2,
        )
        _unary_exact_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            395522,
            OP_TYPE,
            1,
        )


@libentry()
@triton.jit
def unary_395523_rsqrt_float32_tail512_kernel(
    input_ptr,
    output_ptr,
):
    pid = tle.program_id(0)
    for phase in range(0, 2):
        if phase == 0:
            mapped_pid = pid
        else:
            mapped_pid = (pid + 8) % 48
        offsets = phase * 196608 + mapped_pid * 4096 + tl.arange(0, 4096)
        value = tl.load(input_ptr + offsets)
        cann_extension.multibuffer(value, 2)
        result = libdevice.reciprocal(tl.sqrt(value))
        tl.store(output_ptr + offsets, result)

    if pid < 5:
        valid_elements = tl.where(pid == 4, 259, 512)
        _rsqrt_float32_sqrt_reciprocal_masked_chunk(
            input_ptr,
            output_ptr,
            393216 + pid * 512,
            valid_elements,
            512,
        )


@libentry()
@triton.jit
def unary_395523_rsqrt_float16_tail512_kernel(
    input_ptr,
    output_ptr,
):
    pid = tle.program_id(0)
    for phase in range(0, 2):
        offsets = phase * 196608 + pid * 4096 + tl.arange(0, 4096)
        value = tl.load(input_ptr + offsets).to(tl.float32)
        cann_extension.multibuffer(value, 2)
        result = libdevice.rsqrt(value)
        tl.store(
            output_ptr + offsets,
            result.to(output_ptr.dtype.element_ty),
        )

    if pid < 4:
        _rsqrt_math_exact_chunk(
            input_ptr,
            output_ptr,
            393216 + pid * 512,
            512,
        )
    elif pid == 4:
        _rsqrt_math_exact_chunk(
            input_ptr,
            output_ptr,
            395264,
            256,
        )
        _rsqrt_math_exact_chunk(
            input_ptr,
            output_ptr,
            395520,
            2,
        )
        _rsqrt_math_exact_chunk(
            input_ptr,
            output_ptr,
            395522,
            1,
        )


@libentry()
@triton.jit
def unary_524288_exact_48core_kernel(
    input_ptr,
    output_ptr,
    parameter0,
    parameter1,
    OP_TYPE: tl.constexpr,
):
    pid = tle.program_id(0)
    _unary_exact_chunk(
        input_ptr,
        output_ptr,
        parameter0,
        parameter1,
        pid * 8192,
        OP_TYPE,
        8192,
    )
    if pid < 32:
        _unary_exact_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            393216 + pid * 4096,
            OP_TYPE,
            4096,
        )


@triton.jit
def _sigmoid_exact_chunk(
    input_ptr,
    output_ptr,
    offset,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    value = tl.load(input_ptr + offsets)
    result = tl.sigmoid(value)
    tl.store(
        output_ptr + offsets,
        result.to(output_ptr.dtype.element_ty),
    )


@libentry()
@triton.jit
def unary_sigmoid_524288_aligned_32core_kernel(
    input_ptr,
    output_ptr,
):
    pid = tle.program_id(0)
    _sigmoid_exact_chunk(
        input_ptr,
        output_ptr,
        pid * 16384,
        16384,
    )


@libentry()
@triton.jit
def unary_sigmoid_524288_exact_48core_kernel(
    input_ptr,
    output_ptr,
):
    pid = tle.program_id(0)
    _sigmoid_exact_chunk(
        input_ptr,
        output_ptr,
        pid * 8192,
        8192,
    )
    if pid < 32:
        _sigmoid_exact_chunk(
            input_ptr,
            output_ptr,
            393216 + pid * 4096,
            4096,
        )


@libentry()
@triton.jit
def unary_sigmoid_1048576_exact_48core_kernel(
    input_ptr,
    output_ptr,
):
    pid = tle.program_id(0)
    _sigmoid_exact_chunk(
        input_ptr,
        output_ptr,
        pid * 8192,
        8192,
    )
    _sigmoid_exact_chunk(
        input_ptr,
        output_ptr,
        393216 + pid * 8192,
        8192,
    )
    _sigmoid_exact_chunk(
        input_ptr,
        output_ptr,
        786432 + pid * 4096,
        4096,
    )
    if pid < 32:
        _sigmoid_exact_chunk(
            input_ptr,
            output_ptr,
            983040 + pid * 2048,
            2048,
        )


@libentry()
@triton.jit
def unary_sigmoid_1048576_permuted_exact_kernel(
    input_ptr,
    output_ptr,
    STRIDE: tl.constexpr,
    SHIFT: tl.constexpr,
):
    pid = tle.program_id(0)
    logical_pid = (pid * STRIDE) % 48
    _sigmoid_exact_chunk(
        input_ptr,
        output_ptr,
        logical_pid * 8192,
        8192,
    )
    logical_pid = (pid * STRIDE + SHIFT) % 48
    _sigmoid_exact_chunk(
        input_ptr,
        output_ptr,
        393216 + logical_pid * 8192,
        8192,
    )
    logical_pid = (pid * STRIDE + 2 * SHIFT) % 48
    _sigmoid_exact_chunk(
        input_ptr,
        output_ptr,
        786432 + logical_pid * 4096,
        4096,
    )
    if pid < 32:
        tail_pid = (pid * 7 + 8) % 32
        _sigmoid_exact_chunk(
            input_ptr,
            output_ptr,
            983040 + tail_pid * 2048,
            2048,
        )


@libentry()
@triton.jit
def unary_1048576_exact_48core_kernel(
    input_ptr,
    output_ptr,
    parameter0,
    parameter1,
    OP_TYPE: tl.constexpr,
):
    pid = tle.program_id(0)
    _unary_exact_chunk(
        input_ptr,
        output_ptr,
        parameter0,
        parameter1,
        pid * 8192,
        OP_TYPE,
        8192,
    )
    _unary_exact_chunk(
        input_ptr,
        output_ptr,
        parameter0,
        parameter1,
        393216 + pid * 8192,
        OP_TYPE,
        8192,
    )
    _unary_exact_chunk(
        input_ptr,
        output_ptr,
        parameter0,
        parameter1,
        786432 + pid * 4096,
        OP_TYPE,
        4096,
    )
    if pid < 32:
        _unary_exact_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            983040 + pid * 2048,
            OP_TYPE,
            2048,
        )


@libentry()
@triton.jit
def unary_1048576_permuted_exact_kernel(
    input_ptr,
    output_ptr,
    parameter0,
    parameter1,
    STRIDE: tl.constexpr,
    SHIFT: tl.constexpr,
    OP_TYPE: tl.constexpr,
):
    pid = tle.program_id(0)
    logical_pid = (pid * STRIDE) % 48
    _unary_exact_chunk(
        input_ptr,
        output_ptr,
        parameter0,
        parameter1,
        logical_pid * 8192,
        OP_TYPE,
        8192,
    )
    logical_pid = (pid * STRIDE + SHIFT) % 48
    _unary_exact_chunk(
        input_ptr,
        output_ptr,
        parameter0,
        parameter1,
        393216 + logical_pid * 8192,
        OP_TYPE,
        8192,
    )
    logical_pid = (pid * STRIDE + 2 * SHIFT) % 48
    _unary_exact_chunk(
        input_ptr,
        output_ptr,
        parameter0,
        parameter1,
        786432 + logical_pid * 4096,
        OP_TYPE,
        4096,
    )
    if pid < 32:
        tail_pid = (pid * 7 + 8) % 32
        _unary_exact_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            983040 + tail_pid * 2048,
            OP_TYPE,
            2048,
        )


@libentry()
@triton.jit
def unary_1048576_three_phase_8192_kernel(
    input_ptr,
    output_ptr,
    parameter0,
    parameter1,
    OP_TYPE: tl.constexpr,
):
    pid = tle.program_id(0)
    _unary_exact_chunk(
        input_ptr,
        output_ptr,
        parameter0,
        parameter1,
        pid * 8192,
        OP_TYPE,
        8192,
    )
    _unary_exact_chunk(
        input_ptr,
        output_ptr,
        parameter0,
        parameter1,
        393216 + pid * 8192,
        OP_TYPE,
        8192,
    )
    if pid < 32:
        _unary_exact_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            786432 + pid * 8192,
            OP_TYPE,
            8192,
        )


@libentry()
@triton.jit
def unary_1048576_two_phase_16384_kernel(
    input_ptr,
    output_ptr,
    parameter0,
    parameter1,
    OP_TYPE: tl.constexpr,
):
    pid = tle.program_id(0)
    _unary_exact_chunk(
        input_ptr,
        output_ptr,
        parameter0,
        parameter1,
        pid * 16384,
        OP_TYPE,
        16384,
    )
    if pid < 16:
        _unary_exact_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            786432 + pid * 16384,
            OP_TYPE,
            16384,
        )


@triton.jit
def _leaky_relu_constexpr_chunk(
    input_ptr,
    output_ptr,
    offset,
    SLOPE: tl.constexpr,
    USE_MAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    value = tl.load(input_ptr + offsets)
    slope = tl.full((BLOCK_SIZE,), SLOPE, value.dtype)
    if USE_MAX:
        result = tl.maximum(value, value * slope)
    else:
        result = tl.where(value > 0.0, value, value * slope)
    tl.store(output_ptr + offsets, result)


@libentry()
@triton.jit
def unary_1048576_leaky_relu_constexpr_kernel(
    input_ptr,
    output_ptr,
    SLOPE: tl.constexpr,
    USE_MAX: tl.constexpr,
):
    pid = tle.program_id(0)
    _leaky_relu_constexpr_chunk(
        input_ptr,
        output_ptr,
        pid * 8192,
        SLOPE,
        USE_MAX,
        8192,
    )
    _leaky_relu_constexpr_chunk(
        input_ptr,
        output_ptr,
        393216 + pid * 8192,
        SLOPE,
        USE_MAX,
        8192,
    )
    if pid < 32:
        _leaky_relu_constexpr_chunk(
            input_ptr,
            output_ptr,
            786432 + pid * 8192,
            SLOPE,
            USE_MAX,
            8192,
        )


@triton.jit
def _leaky_relu_maximumf_chunk(
    input_ptr,
    output_ptr,
    offset,
    SLOPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    value = tl.load(input_ptr + offsets)
    slope = tl.full((BLOCK_SIZE,), SLOPE, value.dtype)
    result = tl.maximum(
        value,
        value * slope,
        propagate_nan=tl.PropagateNan.ALL,
    )
    tl.store(output_ptr + offsets, result)


@libentry()
@triton.jit
def unary_1048576_leaky_relu_maximumf_balanced_kernel(
    input_ptr,
    output_ptr,
    SLOPE: tl.constexpr,
):
    pid = tle.program_id(0)
    _leaky_relu_maximumf_chunk(input_ptr, output_ptr, pid * 8192, SLOPE, 8192)
    _leaky_relu_maximumf_chunk(
        input_ptr,
        output_ptr,
        393216 + pid * 8192,
        SLOPE,
        8192,
    )
    _leaky_relu_maximumf_chunk(
        input_ptr,
        output_ptr,
        786432 + pid * 4096,
        SLOPE,
        4096,
    )
    if pid < 32:
        _leaky_relu_maximumf_chunk(
            input_ptr,
            output_ptr,
            983040 + pid * 2048,
            SLOPE,
            2048,
        )


@libentry()
@triton.jit
def unary_1048576_leaky_relu_maximumf_permuted_kernel(
    input_ptr,
    output_ptr,
    SLOPE: tl.constexpr,
):
    pid = tle.program_id(0)
    logical_pid = (pid * 5) % 48
    _leaky_relu_maximumf_chunk(
        input_ptr,
        output_ptr,
        logical_pid * 8192,
        SLOPE,
        8192,
    )
    logical_pid = (pid * 5 + 4) % 48
    _leaky_relu_maximumf_chunk(
        input_ptr,
        output_ptr,
        393216 + logical_pid * 8192,
        SLOPE,
        8192,
    )
    logical_pid = (pid * 5 + 8) % 48
    _leaky_relu_maximumf_chunk(
        input_ptr,
        output_ptr,
        786432 + logical_pid * 4096,
        SLOPE,
        4096,
    )
    if pid < 32:
        logical_pid = (pid * 5 + 12) % 32
        _leaky_relu_maximumf_chunk(
            input_ptr,
            output_ptr,
            983040 + logical_pid * 2048,
            SLOPE,
            2048,
        )


@libentry()
@triton.jit
def unary_1048576_bfloat16_leaky_relu_maximumf_permuted_kernel(
    input_ptr,
    output_ptr,
    SLOPE: tl.constexpr,
):
    pid = tle.program_id(0)
    logical_pid = (pid * 19) % 48
    _leaky_relu_maximumf_chunk(
        input_ptr,
        output_ptr,
        logical_pid * 8192,
        SLOPE,
        8192,
    )
    logical_pid = (pid * 19 + 24) % 48
    _leaky_relu_maximumf_chunk(
        input_ptr,
        output_ptr,
        393216 + logical_pid * 8192,
        SLOPE,
        8192,
    )
    logical_pid = (pid * 19) % 48
    _leaky_relu_maximumf_chunk(
        input_ptr,
        output_ptr,
        786432 + logical_pid * 4096,
        SLOPE,
        4096,
    )
    if pid < 32:
        logical_pid = (pid * 19 + 8) % 32
        _leaky_relu_maximumf_chunk(
            input_ptr,
            output_ptr,
            983040 + logical_pid * 2048,
            SLOPE,
            2048,
        )


@libentry()
@triton.jit
def unary_524288_float32_leaky_relu_maximumf_balanced40_kernel(
    input_ptr,
    output_ptr,
    SLOPE: tl.constexpr,
):
    pid = tle.program_id(0)
    _leaky_relu_maximumf_chunk(
        input_ptr,
        output_ptr,
        pid * 8192,
        SLOPE,
        8192,
    )
    _leaky_relu_maximumf_chunk(
        input_ptr,
        output_ptr,
        327680 + pid * 4096,
        SLOPE,
        4096,
    )
    if pid < 32:
        _leaky_relu_maximumf_chunk(
            input_ptr,
            output_ptr,
            491520 + pid * 1024,
            SLOPE,
            1024,
        )


@libentry()
@triton.jit
def unary_leaky_relu_maximumf_aligned_core_loop_kernel(
    input_ptr,
    output_ptr,
    SLOPE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    first_block = tle.program_id(0) * BLOCKS_PER_PROGRAM
    for local_block in range(0, BLOCKS_PER_PROGRAM):
        _leaky_relu_maximumf_chunk(
            input_ptr,
            output_ptr,
            (first_block + local_block) * BLOCK_SIZE,
            SLOPE,
            BLOCK_SIZE,
        )


@libentry()
@triton.jit
def unary_leaky_relu_constexpr_aligned_core_loop_kernel(
    input_ptr,
    output_ptr,
    SLOPE: tl.constexpr,
    USE_MAX: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    first_block = tle.program_id(0) * BLOCKS_PER_PROGRAM
    for local_block in range(0, BLOCKS_PER_PROGRAM):
        offsets = (first_block + local_block) * BLOCK_SIZE + tl.arange(
            0, BLOCK_SIZE
        )
        value = tl.load(input_ptr + offsets)
        slope = tl.full((BLOCK_SIZE,), SLOPE, value.dtype)
        scaled = value * slope
        if USE_MAX:
            result = tl.maximum(value, scaled)
        else:
            result = tl.where(value > 0.0, value, scaled)
        tl.store(output_ptr + offsets, result)


@libentry()
@triton.jit
def unary_leaky_relu_abs_multibuffer_kernel(
    input_ptr,
    output_ptr,
    SLOPE: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    PROGRAM_COUNT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    for block_idx in tl.range(pid, N_BLOCKS, PROGRAM_COUNT):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        value = tl.load(input_ptr + offsets)
        cann_extension.multibuffer(value, 2)
        positive_coefficient = tl.full(
            (BLOCK_SIZE,),
            (1.0 + SLOPE) * 0.5,
            value.dtype,
        )
        absolute_coefficient = tl.full(
            (BLOCK_SIZE,),
            (1.0 - SLOPE) * 0.5,
            value.dtype,
        )
        result = tl.fma(
            value,
            positive_coefficient,
            tl.abs(value) * absolute_coefficient,
        )
        tl.store(output_ptr + offsets, result)


@libentry()
@triton.jit
def unary_balanced_chunks_kernel(
    input_ptr,
    output_ptr,
    parameter0,
    parameter1,
    N_ELEMENTS: tl.constexpr,
    ELEMENTS_PER_PROGRAM: tl.constexpr,
    CHUNK_LAYOUT: tl.constexpr,
    OP_TYPE: tl.constexpr,
):
    program_base = tle.program_id(0) * ELEMENTS_PER_PROGRAM
    if CHUNK_LAYOUT == 0:
        _unary_balanced_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            program_base,
            N_ELEMENTS,
            ELEMENTS_PER_PROGRAM,
            0,
            OP_TYPE,
            8192,
        )
        _unary_balanced_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            program_base,
            N_ELEMENTS,
            ELEMENTS_PER_PROGRAM,
            8192,
            OP_TYPE,
            2048,
        )
        _unary_balanced_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            program_base,
            N_ELEMENTS,
            ELEMENTS_PER_PROGRAM,
            10240,
            OP_TYPE,
            1024,
        )
    elif CHUNK_LAYOUT == 1:
        _unary_balanced_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            program_base,
            N_ELEMENTS,
            ELEMENTS_PER_PROGRAM,
            0,
            OP_TYPE,
            8192,
        )
        _unary_balanced_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            program_base,
            N_ELEMENTS,
            ELEMENTS_PER_PROGRAM,
            8192,
            OP_TYPE,
            8192,
        )
        _unary_balanced_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            program_base,
            N_ELEMENTS,
            ELEMENTS_PER_PROGRAM,
            16384,
            OP_TYPE,
            4096,
        )
        _unary_balanced_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            program_base,
            N_ELEMENTS,
            ELEMENTS_PER_PROGRAM,
            20480,
            OP_TYPE,
            2048,
        )
    else:
        _unary_balanced_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            program_base,
            N_ELEMENTS,
            ELEMENTS_PER_PROGRAM,
            0,
            OP_TYPE,
            16384,
        )
        _unary_balanced_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            program_base,
            N_ELEMENTS,
            ELEMENTS_PER_PROGRAM,
            16384,
            OP_TYPE,
            4096,
        )
        _unary_balanced_chunk(
            input_ptr,
            output_ptr,
            parameter0,
            parameter1,
            program_base,
            N_ELEMENTS,
            ELEMENTS_PER_PROGRAM,
            20480,
            OP_TYPE,
            2048,
        )


def prepare_dense_unary(
    *,
    op_type: str,
    attrs: dict[str, Any],
    input_specs: Sequence[Any],
    default_run_fn: Any,
) -> Optional[Any]:
    """Return an Ascend prepared unary replay when the dense contract holds."""
    if op_type not in _SIMPLE_UNARY_OPS and op_type != "logical_not":
        return None
    if len(input_specs) != 1:
        return None

    input_spec = input_specs[0]
    shape = tuple(input_spec.shape)
    if not all(isinstance(dim, int) for dim in shape):
        return None
    if input_spec.layout not in ("contiguous", "nhwc"):
        return None
    if input_spec.stride is None:
        return None
    if op_type == "logical_not":
        if input_spec.dtype != "bool":
            return None
    elif input_spec.dtype not in _FLOAT_DTYPES:
        return None
    if op_type == "relu" and any(
        attrs.get(name) is not None
        for name in ("negative_slope", "lower_clip", "upper_clip")
    ):
        return None
    if attrs.get("inplace"):
        return None

    kernel_op_type = op_type
    parameter0 = 0.0
    parameter1 = 0.0
    if op_type in {"swish", "silu"}:
        beta = attrs.get("swish_beta")
        parameter0 = 1.0 if beta is None else float(beta)
        kernel_op_type = "swish"
    elif op_type == "gelu":
        approximate = attrs.get("approximate", "none")
        if approximate not in ("none", "tanh"):
            return None
        kernel_op_type = "gelu_tanh" if approximate == "tanh" else "gelu"
    elif op_type == "gelu_approx_tanh":
        kernel_op_type = "gelu_tanh"
    elif op_type == "leaky_relu":
        slope = attrs.get("negative_slope")
        parameter0 = 0.01 if slope is None else float(slope)
        if input_spec.dtype == "float16" and 0.0 < parameter0 <= 1.0:
            kernel_op_type = "leaky_relu_max"
    elif op_type == "elu":
        alpha = attrs.get("alpha")
        parameter0 = 1.0 if alpha is None else float(alpha)
    elif op_type == "softplus":
        beta = attrs.get("beta")
        threshold = attrs.get("threshold")
        parameter0 = 1.0 if beta is None else float(beta)
        parameter1 = 20.0 if threshold is None else float(threshold)
        if parameter0 <= 0.0:
            return None

    from flag_dnn.graph.prepared import (
        PreparedSingleKernelRunSpec,
        PreparedSingleKernelSpec,
        get_prepared_output,
        make_single_kernel_run_fn,
        runtime_tensor_checks_from_specs,
    )
    from flag_dnn.graph.tensor import torch_dtype

    input_checks = runtime_tensor_checks_from_specs(
        input_specs,
        (0,),
        require_shape=True,
        require_stride=True,
        require_dtype=True,
    )
    if input_checks is None:
        return None

    static_shape = tuple(int(dim) for dim in shape)
    static_stride = tuple(int(item) for item in input_spec.stride)
    output_dtype = (
        torch.bool
        if op_type == "logical_not"
        else torch_dtype(input_spec.dtype)
    )
    n_elements = 1
    for dim in static_shape:
        n_elements *= dim
    if n_elements == 0:
        return None
    prepared_output_offset_bytes = 0
    if (
        n_elements == 1048576
        and kernel_op_type == "neg"
        and input_spec.layout == "contiguous"
    ):
        if input_spec.dtype == "float16":
            prepared_output_offset_bytes = 3 * 1024 * 1024
        elif input_spec.dtype == "bfloat16":
            prepared_output_offset_bytes = 1024 * 1024
    elif (
        n_elements == 524288
        and kernel_op_type == "neg"
        and input_spec.dtype == "float32"
        and input_spec.layout == "contiguous"
    ):
        prepared_output_offset_bytes = 1024 * 1024
    elif (
        n_elements == 524288
        and kernel_op_type in {"leaky_relu", "leaky_relu_max"}
        and input_spec.dtype == "float32"
        and input_spec.layout == "contiguous"
        and 0.0 < parameter0 <= 1.0
    ):
        prepared_output_offset_bytes = 4 * 1024 * 1024
    elif (
        n_elements == 1048576
        and kernel_op_type == "leaky_relu"
        and input_spec.dtype == "float32"
        and input_spec.layout == "contiguous"
        and 0.0 < parameter0 <= 1.0
    ):
        # Keep this exact replay away from an input/output address phase that
        # underutilizes HBM channels on Ascend 910.  The returned tensor stays
        # dense; only its private prepared storage has a nonzero offset.
        prepared_output_offset_bytes = 4 * 1024 * 1024 + 384 * 1024
    use_offset_prepared_output = prepared_output_offset_bytes != 0

    block_size = get_dense_unary_block_size(
        kernel_op_type, n_elements, input_spec.dtype
    )
    if (
        n_elements == 524288
        and input_spec.dtype == "float32"
        and kernel_op_type == "rsqrt"
    ):
        block_size = 8192
    if (
        kernel_op_type == "relu"
        and input_spec.dtype in {"float16", "bfloat16"}
        and n_elements == 1048576
    ):
        block_size = 16384
    aligned_grid = make_core_loop_grid(n_elements, input_spec.device)
    balanced_grid = make_balanced_core_loop_grid(n_elements, input_spec.device)

    def tiled_grid(meta: dict[str, Any]) -> tuple[int, ...]:
        return (triton.cdiv(n_elements, int(meta["BLOCK_SIZE"])),)

    grid: Grid
    constexpr_kwargs: dict[str, Any]
    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    output_element_size = torch.empty(
        (),
        dtype=output_dtype,
    ).element_size()

    def output_factory(inputs: Sequence[Any]) -> torch.Tensor:
        source = inputs[0]
        key = (
            source.device.type,
            source.device.index,
            output_dtype,
            static_shape,
            static_stride,
            prepared_output_offset_bytes,
        )

        def allocate_output() -> torch.Tensor:
            if not use_offset_prepared_output:
                return torch.empty_strided(
                    static_shape,
                    static_stride,
                    device=source.device,
                    dtype=output_dtype,
                )
            offset_elements = (
                prepared_output_offset_bytes // output_element_size
            )
            storage = torch.empty(
                n_elements + offset_elements,
                device=source.device,
                dtype=output_dtype,
            )
            return torch.as_strided(
                storage,
                static_shape,
                static_stride,
                offset_elements,
            )

        return get_prepared_output(
            output_cache,
            key,
            allocate_output,
        )

    def runtime_args(
        inputs: Sequence[Any], output: torch.Tensor
    ) -> tuple[Any, ...]:
        return (inputs[0], output, parameter0, parameter1)

    def extra_check(inputs: Sequence[Any]) -> bool:
        source = inputs[0]
        return isinstance(source, torch.Tensor) and is_runtime_device_tensor(
            source
        )

    aligned_program_count = aligned_grid({"BLOCK_SIZE": block_size})[0]
    balanced_program_count = balanced_grid({"BLOCK_SIZE": block_size})[0]
    chunk_layout: Optional[int] = None
    multibuffer_block_size: Optional[int] = None
    if (
        n_elements == 1048576
        and input_spec.dtype == "float32"
        and kernel_op_type == "elu"
    ):
        multibuffer_block_size = 4096
    use_1048576_lowp_swish_constexpr = (
        n_elements == 1048576
        and input_spec.dtype in {"float16", "bfloat16"}
        and kernel_op_type == "swish"
    )
    use_1048576_float32_swish_balanced = (
        n_elements == 1048576
        and input_spec.dtype == "float32"
        and kernel_op_type == "swish"
    )
    use_524288_sigmoid_dedicated = (
        n_elements == 524288
        and kernel_op_type == "sigmoid"
        and input_spec.dtype in _FLOAT_DTYPES
    )
    use_1048576_sigmoid_dedicated = (
        n_elements == 1048576
        and kernel_op_type == "sigmoid"
        and input_spec.dtype in _FLOAT_DTYPES
    )
    use_176085_exact_split = n_elements == 176085 and (
        kernel_op_type == "log" or kernel_op_type == "rsqrt"
    )
    use_293475_exact_split = n_elements == 293475 and (
        kernel_op_type == "log" or kernel_op_type == "rsqrt"
    )
    use_395523_exact_tail = n_elements == 395523 and (
        kernel_op_type == "reciprocal"
        or kernel_op_type == "log"
        or (kernel_op_type == "rsqrt")
    )
    use_395523_float32_ceil_tiled_16384 = (
        n_elements == 395523
        and kernel_op_type == "ceil"
        and input_spec.dtype == "float32"
    )
    use_395523_float32_rsqrt_tail512 = (
        n_elements == 395523
        and kernel_op_type == "rsqrt"
        and input_spec.dtype == "float32"
    )
    use_395523_float16_rsqrt_tail512 = (
        n_elements == 395523
        and kernel_op_type == "rsqrt"
        and input_spec.dtype == "float16"
    )
    use_524288_exact_48core = n_elements == 524288 and (
        (
            kernel_op_type in {"exp", "gelu_tanh", "sigmoid"}
            and input_spec.dtype == "float32"
        )
        or (
            kernel_op_type == "log"
            and input_spec.dtype in {"float16", "bfloat16"}
        )
        or (
            kernel_op_type == "swish"
            and input_spec.dtype in {"float16", "bfloat16"}
        )
        or (
            kernel_op_type == "leaky_relu"
            and input_spec.dtype in {"bfloat16", "float32"}
        )
        or (kernel_op_type == "relu" and input_spec.dtype == "float32")
    )
    use_524288_float32_neg_multibuffer_32x8192 = (
        n_elements == 524288
        and kernel_op_type == "neg"
        and input_spec.dtype == "float32"
    )
    use_524288_aligned_32x8192 = n_elements == 524288 and (
        (
            kernel_op_type == "exp"
            and input_spec.dtype in {"float16", "bfloat16"}
        )
        or (
            kernel_op_type == "leaky_relu"
            and input_spec.dtype in {"float16", "float32"}
        )
    )
    use_524288_lowp_sigmoid_aligned_32x16384 = (
        n_elements == 524288
        and kernel_op_type == "sigmoid"
        and input_spec.dtype in {"float16", "bfloat16"}
    )
    use_524288_float32_exp_aligned_32x16384 = (
        n_elements == 524288
        and kernel_op_type == "exp"
        and input_spec.dtype == "float32"
    )
    use_524288_bfloat16_rsqrt_aligned_32x8192 = (
        n_elements == 524288
        and kernel_op_type == "rsqrt"
        and input_spec.dtype == "bfloat16"
    )
    use_524288_bfloat16_log_aligned_32x16384 = (
        n_elements == 524288
        and kernel_op_type == "log"
        and input_spec.dtype == "bfloat16"
    )
    use_524288_float16_log_permuted_32x8192 = (
        n_elements == 524288
        and kernel_op_type == "log"
        and input_spec.dtype == "float16"
    )
    use_524288_float32_log_permuted_multibuffer_32x8192 = (
        n_elements == 524288
        and kernel_op_type == "log"
        and input_spec.dtype == "float32"
    )
    use_524288_float32_leaky_relu_constexpr_aligned = (
        n_elements == 524288
        and kernel_op_type == "leaky_relu"
        and input_spec.dtype == "float32"
        and not 0.0 < parameter0 <= 1.0
    )
    use_524288_float32_leaky_relu_maximumf_balanced40 = (
        n_elements == 524288
        and kernel_op_type == "leaky_relu"
        and input_spec.dtype == "float32"
        and 0.0 < parameter0 <= 1.0
    )
    use_524288_leaky_relu_maximumf_aligned = (
        n_elements == 524288
        and kernel_op_type in {"leaky_relu", "leaky_relu_max"}
        and input_spec.dtype in {"float16", "bfloat16"}
        and 0.0 < parameter0 <= 1.0
    )
    use_1048576_three_phase_8192 = n_elements == 1048576 and (
        (
            kernel_op_type in {"exp", "sigmoid", "sqrt"}
            and input_spec.dtype in {"float16", "bfloat16"}
        )
        or (kernel_op_type == "neg" and input_spec.dtype == "bfloat16")
    )
    use_1048576_neg_permuted_exact = (
        n_elements == 1048576
        and kernel_op_type == "neg"
        and input_spec.dtype == "float32"
    )
    use_1048576_rsqrt_permuted_exact = (
        n_elements == 1048576 and kernel_op_type == "rsqrt"
    )
    use_1048576_float32_abs_permuted_exact = (
        n_elements == 1048576
        and kernel_op_type == "abs"
        and input_spec.dtype == "float32"
    )
    use_1048576_bfloat16_neg_bitwise_multibuffer_32x16384 = (
        n_elements == 1048576
        and kernel_op_type == "neg"
        and input_spec.dtype == "bfloat16"
    )
    use_1048576_float16_neg_multibuffer_32x8192 = (
        n_elements == 1048576
        and kernel_op_type == "neg"
        and input_spec.dtype == "float16"
    )
    use_1048576_bfloat16_sigmoid_exact_48core = (
        n_elements == 1048576
        and kernel_op_type == "sigmoid"
        and input_spec.dtype == "bfloat16"
    )
    use_1048576_two_phase_16384 = (
        n_elements == 1048576
        and kernel_op_type == "relu"
        and input_spec.dtype in {"float16", "bfloat16"}
    )
    use_1048576_aligned_32x16384 = (
        n_elements == 1048576
        and kernel_op_type == "neg"
        and input_spec.dtype == "float16"
    )
    use_1048576_leaky_relu_constexpr = (
        n_elements == 1048576
        and kernel_op_type in {"leaky_relu", "leaky_relu_max"}
        and input_spec.dtype in {"float16", "bfloat16"}
    )
    use_1048576_float16_leaky_relu_abs = (
        n_elements == 1048576
        and kernel_op_type in {"leaky_relu", "leaky_relu_max"}
        and input_spec.dtype == "float16"
        and 0.0 <= parameter0 <= 1.0
    )
    use_1048576_float16_leaky_relu_maximumf_aligned = (
        n_elements == 1048576
        and kernel_op_type in {"leaky_relu", "leaky_relu_max"}
        and input_spec.dtype == "float16"
        and 0.0 < parameter0 <= 1.0
    )
    use_1048576_bfloat16_leaky_relu_maximumf_permuted = (
        n_elements == 1048576
        and kernel_op_type == "leaky_relu"
        and input_spec.dtype == "bfloat16"
        and 0.0 < parameter0 <= 1.0
    )
    use_1048576_float32_leaky_relu_maximumf_permuted = (
        n_elements == 1048576
        and kernel_op_type == "leaky_relu"
        and input_spec.dtype == "float32"
        and 0.0 < parameter0 <= 1.0
    )
    use_1048576_float32_leaky_relu_aligned = (
        n_elements == 1048576
        and kernel_op_type == "leaky_relu"
        and input_spec.dtype == "float32"
        and not 0.0 < parameter0 <= 1.0
    )
    use_1048576_exact_48core = (
        n_elements == 1048576
        and not (kernel_op_type == "neg" and input_spec.dtype == "float32")
        and (
            (
                kernel_op_type
                in {
                    "exp",
                    "gelu",
                    "gelu_tanh",
                    "sigmoid",
                    "sqrt",
                }
                and input_spec.dtype == "float32"
            )
            or (
                kernel_op_type in {"abs", "neg", "relu"}
                and input_spec.dtype in {"float16", "bfloat16", "float32"}
            )
            or kernel_op_type in {"log", "rsqrt"}
            or (
                kernel_op_type in {"leaky_relu", "swish"}
                and input_spec.dtype == "float32"
            )
        )
    )
    use_395523_balanced = (
        n_elements == 395523
        and balanced_program_count == 48
        and kernel_op_type
        in {
            "relu",
            "leaky_relu",
            "leaky_relu_max",
            "erf",
            "sin",
            "cos",
            "tan",
        }
    )
    use_395523_tail4096 = (
        n_elements == 395523
        and balanced_program_count == 48
        and (
            (
                kernel_op_type == "reciprocal"
                and input_spec.dtype in {"float16", "bfloat16"}
            )
            or (kernel_op_type == "sqrt" and input_spec.dtype == "float32")
            or (
                kernel_op_type in {"leaky_relu", "leaky_relu_max"}
                and input_spec.dtype in {"float16", "bfloat16", "float32"}
            )
        )
    )
    if balanced_program_count == 48:
        if kernel_op_type == "gelu_tanh" and (
            n_elements == 524288
            or (n_elements == 1048576 and input_spec.dtype == "float32")
        ):
            chunk_layout = 0 if n_elements == 524288 else 1
        elif (
            kernel_op_type == "ceil"
            and input_spec.dtype == "float32"
            and n_elements == 524288
        ):
            chunk_layout = 0
        elif (
            kernel_op_type in {"relu", "leaky_relu", "leaky_relu_max"}
            and n_elements == 524288
        ):
            chunk_layout = 0
        elif (
            kernel_op_type in {"exp", "sigmoid", "swish"}
            or (
                kernel_op_type == "log"
                and input_spec.dtype in {"float16", "bfloat16"}
            )
        ) and n_elements == 524288:
            chunk_layout = 0
        elif (
            kernel_op_type == "rsqrt"
            and n_elements == 1048576
            and input_spec.dtype in {"float16", "bfloat16"}
        ):
            chunk_layout = 1
        elif (
            kernel_op_type in {"exp", "sigmoid"}
            and n_elements == 1048576
            and input_spec.dtype == "float32"
        ):
            chunk_layout = 1
        elif n_elements == 1048576:
            if (
                kernel_op_type in {"abs", "neg"}
                and input_spec.dtype == "float32"
            ) or kernel_op_type in {"logical_not", "sqrt"}:
                chunk_layout = 2
            elif (
                kernel_op_type in {"relu", "leaky_relu", "leaky_relu_max"}
                and input_spec.dtype == "float32"
            ):
                chunk_layout = 1

    if use_524288_sigmoid_dedicated:
        if input_spec.dtype == "float32":
            kernel = unary_sigmoid_524288_exact_48core_kernel
            program_count = 48
        else:
            kernel = unary_sigmoid_524288_aligned_32core_kernel
            program_count = 32
        grid = _fixed_grid(program_count)
        constexpr_kwargs = {
            "num_warps": 4,
            "num_stages": 1,
        }

        def runtime_args(  # noqa: F811
            inputs: Sequence[Any], output: torch.Tensor
        ) -> tuple[Any, ...]:
            return (inputs[0], output)

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (program_count, 1, 1), ()

    elif use_1048576_sigmoid_dedicated:
        kernel = unary_sigmoid_1048576_exact_48core_kernel
        grid = _fixed_grid(48)
        constexpr_kwargs = {
            "num_warps": 4,
            "num_stages": 1,
        }

        def runtime_args(
            inputs: Sequence[Any], output: torch.Tensor
        ) -> tuple[Any, ...]:
            return (inputs[0], output)

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (48, 1, 1), ()

    elif use_1048576_lowp_swish_constexpr:
        kernel = swish_constexpr_core_loop_kernel
        grid = _fixed_grid(48)
        constexpr_beta = parameter0
        swish_block_size = 8192 if input_spec.dtype == "float16" else 4096
        swish_n_blocks = n_elements // swish_block_size
        constexpr_kwargs = {
            "BETA": constexpr_beta,
            "N_BLOCKS": swish_n_blocks,
            "PROGRAM_COUNT": 48,
            "BLOCK_SIZE": swish_block_size,
            "num_warps": 4,
            "num_stages": 1,
        }

        def runtime_args(
            inputs: Sequence[Any], output: torch.Tensor
        ) -> tuple[Any, ...]:
            return (inputs[0], output)

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (48, 1, 1), (
                constexpr_beta,
                swish_n_blocks,
                48,
                swish_block_size,
            )

    elif use_1048576_float32_swish_balanced:
        kernel = unary_balanced_chunks_kernel
        grid = _fixed_grid(48)
        swish_elements_per_program = 22016
        swish_chunk_layout = 1
        constexpr_kwargs = {
            "N_ELEMENTS": n_elements,
            "ELEMENTS_PER_PROGRAM": swish_elements_per_program,
            "CHUNK_LAYOUT": swish_chunk_layout,
            "OP_TYPE": kernel_op_type,
            "num_warps": 4,
            "num_stages": 1,
        }

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (48, 1, 1), (
                n_elements,
                swish_elements_per_program,
                swish_chunk_layout,
                kernel_op_type,
            )

    elif multibuffer_block_size is not None:
        kernel = unary_multibuffer_core_loop_kernel
        grid = _fixed_grid(48)
        n_blocks = n_elements // multibuffer_block_size
        constexpr_kwargs = {
            "N_BLOCKS": n_blocks,
            "PROGRAM_COUNT": 48,
            "OP_TYPE": kernel_op_type,
            "BLOCK_SIZE": multibuffer_block_size,
            "num_warps": 4,
            "num_stages": 1,
        }

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (48, 1, 1), (
                n_blocks,
                48,
                kernel_op_type,
                multibuffer_block_size,
            )

    elif use_1048576_float16_leaky_relu_maximumf_aligned:
        kernel = unary_leaky_relu_maximumf_aligned_core_loop_kernel
        grid = _fixed_grid(32)
        constexpr_slope = parameter0
        constexpr_kwargs = {
            "SLOPE": constexpr_slope,
            "BLOCKS_PER_PROGRAM": 2,
            "BLOCK_SIZE": 16384,
            "num_warps": 4,
            "num_stages": 1,
        }

        def runtime_args(
            inputs: Sequence[Any], output: torch.Tensor
        ) -> tuple[Any, ...]:
            return (inputs[0], output)

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (32, 1, 1), (
                constexpr_slope,
                2,
                16384,
            )

    elif use_1048576_float16_leaky_relu_abs:
        kernel = unary_leaky_relu_abs_multibuffer_kernel
        grid = _fixed_grid(48)
        constexpr_slope = parameter0
        constexpr_kwargs = {
            "SLOPE": constexpr_slope,
            "N_BLOCKS": 128,
            "PROGRAM_COUNT": 48,
            "BLOCK_SIZE": 8192,
            "num_warps": 4,
            "num_stages": 1,
        }

        def runtime_args(
            inputs: Sequence[Any], output: torch.Tensor
        ) -> tuple[Any, ...]:
            return (inputs[0], output)

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (48, 1, 1), (
                constexpr_slope,
                128,
                48,
                8192,
            )

    elif use_1048576_bfloat16_leaky_relu_maximumf_permuted:
        kernel = unary_1048576_bfloat16_leaky_relu_maximumf_permuted_kernel
        grid = _fixed_grid(48)
        constexpr_slope = parameter0
        constexpr_kwargs = {
            "SLOPE": constexpr_slope,
            "num_warps": 4,
            "num_stages": 1,
        }

        def runtime_args(
            inputs: Sequence[Any], output: torch.Tensor
        ) -> tuple[Any, ...]:
            return (inputs[0], output)

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (48, 1, 1), (constexpr_slope,)

    elif use_1048576_float32_leaky_relu_maximumf_permuted:
        kernel = unary_1048576_leaky_relu_maximumf_permuted_kernel
        grid = _fixed_grid(48)
        constexpr_slope = parameter0
        constexpr_kwargs = {
            "SLOPE": constexpr_slope,
            "num_warps": 4,
            "num_stages": 1,
        }

        def runtime_args(
            inputs: Sequence[Any], output: torch.Tensor
        ) -> tuple[Any, ...]:
            return (inputs[0], output)

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (48, 1, 1), (constexpr_slope,)

    elif use_1048576_float32_leaky_relu_aligned:
        kernel = unary_leaky_relu_constexpr_aligned_core_loop_kernel
        grid = _fixed_grid(32)
        constexpr_slope = parameter0
        constexpr_kwargs = {
            "SLOPE": constexpr_slope,
            "USE_MAX": False,
            "BLOCKS_PER_PROGRAM": 4,
            "BLOCK_SIZE": 8192,
            "num_warps": 4,
            "num_stages": 1,
        }

        def runtime_args(
            inputs: Sequence[Any], output: torch.Tensor
        ) -> tuple[Any, ...]:
            return (inputs[0], output)

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (32, 1, 1), (
                constexpr_slope,
                False,
                4,
                8192,
            )

    elif use_1048576_leaky_relu_constexpr:
        kernel = unary_1048576_leaky_relu_constexpr_kernel
        grid = _fixed_grid(48)
        use_max = kernel_op_type == "leaky_relu_max"
        constexpr_slope = parameter0
        constexpr_kwargs = {
            "SLOPE": constexpr_slope,
            "USE_MAX": use_max,
            "num_warps": 4,
            "num_stages": 1,
        }

        def runtime_args(
            inputs: Sequence[Any], output: torch.Tensor
        ) -> tuple[Any, ...]:
            return (inputs[0], output)

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (48, 1, 1), (constexpr_slope, use_max)

    elif use_176085_exact_split:
        kernel = unary_176085_masked_tail_kernel
        grid = _fixed_grid(43)
        constexpr_kwargs = {
            "OP_TYPE": kernel_op_type,
            "num_warps": 4,
            "num_stages": 1,
        }

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (43, 1, 1), (kernel_op_type,)

    elif use_293475_exact_split:
        kernel = unary_293475_exact_split_kernel
        grid = _fixed_grid(36)
        constexpr_kwargs = {
            "OP_TYPE": kernel_op_type,
            "num_warps": 4,
            "num_stages": 1,
        }

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (36, 1, 1), (kernel_op_type,)

    elif use_395523_float16_rsqrt_tail512:
        kernel = unary_395523_rsqrt_float16_tail512_kernel
        grid = _fixed_grid(48)
        constexpr_kwargs = {
            "num_warps": 4,
            "num_stages": 2,
            "multibuffer": True,
        }

        def runtime_args(
            inputs: Sequence[Any], output: torch.Tensor
        ) -> tuple[Any, ...]:
            return (inputs[0], output)

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (48, 1, 1), ()

    elif use_395523_float32_rsqrt_tail512:
        kernel = unary_395523_rsqrt_float32_tail512_kernel
        grid = _fixed_grid(48)
        constexpr_kwargs = {
            "num_warps": 4,
            "num_stages": 2,
            "multibuffer": True,
        }

        def runtime_args(
            inputs: Sequence[Any], output: torch.Tensor
        ) -> tuple[Any, ...]:
            return (inputs[0], output)

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (48, 1, 1), ()

    elif use_395523_exact_tail:
        kernel = unary_395523_exact_tail_kernel
        grid = _fixed_grid(48)
        constexpr_kwargs = {
            "OP_TYPE": kernel_op_type,
            "num_warps": 4,
            "num_stages": 1,
        }

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (48, 1, 1), (kernel_op_type,)

    elif use_395523_float32_ceil_tiled_16384:
        kernel = unary_tiled_kernel
        ceil_block_size = 16384
        ceil_program_count = triton.cdiv(n_elements, ceil_block_size)
        grid = _fixed_grid(ceil_program_count)
        constexpr_kwargs = {
            "N_ELEMENTS": n_elements,
            "OP_TYPE": kernel_op_type,
            "BLOCK_SIZE": ceil_block_size,
            "num_warps": 8,
            "num_stages": 1,
        }

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (ceil_program_count, 1, 1), (
                n_elements,
                kernel_op_type,
                ceil_block_size,
            )

    elif use_1048576_bfloat16_neg_bitwise_multibuffer_32x16384:
        kernel = unary_neg_16bit_multibuffer_core_loop_kernel
        grid = _fixed_grid(32)
        constexpr_kwargs = {
            "N_BLOCKS": 64,
            "PROGRAM_COUNT": 32,
            "BLOCK_SIZE": 16384,
            "num_warps": 4,
            "num_stages": 1,
        }

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (32, 1, 1), (64, 32, 16384)

    elif use_1048576_float16_neg_multibuffer_32x8192:
        kernel = unary_multibuffer_core_loop_kernel
        grid = _fixed_grid(32)
        constexpr_kwargs = {
            "N_BLOCKS": 128,
            "PROGRAM_COUNT": 32,
            "OP_TYPE": kernel_op_type,
            "BLOCK_SIZE": 8192,
            "num_warps": 4,
            "num_stages": 1,
        }

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (32, 1, 1), (
                128,
                32,
                kernel_op_type,
                8192,
            )

    elif (
        use_1048576_neg_permuted_exact
        or use_1048576_rsqrt_permuted_exact
        or use_1048576_float32_abs_permuted_exact
    ):
        kernel = unary_1048576_permuted_exact_kernel
        grid = _fixed_grid(48)
        if use_1048576_float32_abs_permuted_exact:
            permuted_stride, permuted_shift = 23, 4
        else:
            permuted_stride, permuted_shift = 19, 24
        constexpr_kwargs = {
            "STRIDE": permuted_stride,
            "SHIFT": permuted_shift,
            "OP_TYPE": kernel_op_type,
            "num_warps": 4,
            "num_stages": 1,
        }

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (48, 1, 1), (
                permuted_stride,
                permuted_shift,
                kernel_op_type,
            )

    elif use_1048576_bfloat16_sigmoid_exact_48core:
        kernel = unary_1048576_exact_48core_kernel
        grid = _fixed_grid(48)
        constexpr_kwargs = {
            "OP_TYPE": kernel_op_type,
            "num_warps": 4,
            "num_stages": 1,
        }

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (48, 1, 1), (kernel_op_type,)

    elif use_1048576_three_phase_8192:
        kernel = unary_1048576_three_phase_8192_kernel
        grid = _fixed_grid(48)
        constexpr_kwargs = {
            "OP_TYPE": kernel_op_type,
            "num_warps": 4,
            "num_stages": 1,
        }

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (48, 1, 1), (kernel_op_type,)

    elif use_1048576_two_phase_16384:
        kernel = unary_1048576_two_phase_16384_kernel
        grid = _fixed_grid(48)
        constexpr_kwargs = {
            "OP_TYPE": kernel_op_type,
            "num_warps": 4,
            "num_stages": 1,
        }

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (48, 1, 1), (kernel_op_type,)

    elif use_1048576_aligned_32x16384:
        kernel = unary_aligned_core_loop_kernel
        grid = _fixed_grid(32)
        constexpr_kwargs = {
            "BLOCKS_PER_PROGRAM": 2,
            "OP_TYPE": kernel_op_type,
            "BLOCK_SIZE": 16384,
            "num_warps": 4,
            "num_stages": 1,
        }

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (32, 1, 1), (2, kernel_op_type, 16384)

    elif use_524288_float32_leaky_relu_constexpr_aligned:
        kernel = unary_leaky_relu_constexpr_aligned_core_loop_kernel
        grid = _fixed_grid(32)
        constexpr_slope = parameter0
        constexpr_kwargs = {
            "SLOPE": constexpr_slope,
            "USE_MAX": False,
            "BLOCKS_PER_PROGRAM": 2,
            "BLOCK_SIZE": 8192,
            "num_warps": 1,
            "num_stages": 1,
        }

        def runtime_args(
            inputs: Sequence[Any], output: torch.Tensor
        ) -> tuple[Any, ...]:
            return (inputs[0], output)

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (32, 1, 1), (
                constexpr_slope,
                False,
                2,
                8192,
            )

    elif use_524288_float32_leaky_relu_maximumf_balanced40:
        kernel = unary_524288_float32_leaky_relu_maximumf_balanced40_kernel
        grid = _fixed_grid(40)
        constexpr_slope = parameter0
        constexpr_kwargs = {
            "SLOPE": constexpr_slope,
            "num_warps": 4,
            "num_stages": 1,
        }

        def runtime_args(
            inputs: Sequence[Any], output: torch.Tensor
        ) -> tuple[Any, ...]:
            return (inputs[0], output)

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (40, 1, 1), (constexpr_slope,)

    elif use_524288_leaky_relu_maximumf_aligned:
        kernel = unary_leaky_relu_maximumf_aligned_core_loop_kernel
        grid = _fixed_grid(32)
        constexpr_slope = parameter0
        maximum_block_size = 16384
        maximum_blocks_per_program = 1
        constexpr_kwargs = {
            "SLOPE": constexpr_slope,
            "BLOCKS_PER_PROGRAM": maximum_blocks_per_program,
            "BLOCK_SIZE": maximum_block_size,
            "num_warps": 4,
            "num_stages": 1,
        }

        def runtime_args(
            inputs: Sequence[Any], output: torch.Tensor
        ) -> tuple[Any, ...]:
            return (inputs[0], output)

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (32, 1, 1), (
                constexpr_slope,
                maximum_blocks_per_program,
                maximum_block_size,
            )

    elif use_524288_float32_exp_aligned_32x16384:
        kernel = unary_aligned_core_loop_kernel
        grid = _fixed_grid(32)
        constexpr_kwargs = {
            "BLOCKS_PER_PROGRAM": 1,
            "OP_TYPE": kernel_op_type,
            "BLOCK_SIZE": 16384,
            "num_warps": 4,
            "num_stages": 1,
        }

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (32, 1, 1), (1, kernel_op_type, 16384)

    elif use_524288_bfloat16_log_aligned_32x16384:
        kernel = unary_aligned_core_loop_kernel
        grid = _fixed_grid(32)
        constexpr_kwargs = {
            "BLOCKS_PER_PROGRAM": 1,
            "OP_TYPE": kernel_op_type,
            "BLOCK_SIZE": 16384,
            "num_warps": 4,
            "num_stages": 1,
        }

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (32, 1, 1), (1, kernel_op_type, 16384)

    elif (
        use_524288_float16_log_permuted_32x8192
        or use_524288_float32_log_permuted_multibuffer_32x8192
    ):
        kernel = unary_permuted_core_loop_kernel
        grid = _fixed_grid(32)
        use_explicit_multibuffer = (
            use_524288_float32_log_permuted_multibuffer_32x8192
        )
        constexpr_kwargs = {
            "N_BLOCKS": 64,
            "PROGRAM_COUNT": 32,
            "PHASES": 2,
            "STRIDE": 7,
            "SHIFT": 8,
            "USE_MULTIBUFFER": use_explicit_multibuffer,
            "OP_TYPE": kernel_op_type,
            "BLOCK_SIZE": 8192,
            "num_warps": 4,
            "num_stages": 1,
        }

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (32, 1, 1), (
                64,
                32,
                2,
                7,
                8,
                use_explicit_multibuffer,
                kernel_op_type,
                8192,
            )

    elif use_524288_float32_neg_multibuffer_32x8192:
        kernel = unary_multibuffer_core_loop_kernel
        grid = _fixed_grid(32)
        constexpr_kwargs = {
            "N_BLOCKS": 64,
            "PROGRAM_COUNT": 32,
            "OP_TYPE": kernel_op_type,
            "BLOCK_SIZE": 8192,
            "num_warps": 4,
            "num_stages": 1,
        }

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (32, 1, 1), (
                64,
                32,
                kernel_op_type,
                8192,
            )

    elif use_524288_bfloat16_rsqrt_aligned_32x8192:
        kernel = unary_aligned_core_loop_kernel
        grid = _fixed_grid(32)
        constexpr_kwargs = {
            "BLOCKS_PER_PROGRAM": 2,
            "OP_TYPE": kernel_op_type,
            "BLOCK_SIZE": 8192,
            "num_warps": 4,
            "num_stages": 1,
        }

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (32, 1, 1), (2, kernel_op_type, 8192)

    elif use_524288_lowp_sigmoid_aligned_32x16384:
        kernel = unary_aligned_core_loop_kernel
        grid = _fixed_grid(32)
        constexpr_kwargs = {
            "BLOCKS_PER_PROGRAM": 1,
            "OP_TYPE": kernel_op_type,
            "BLOCK_SIZE": 16384,
            "num_warps": 4,
            "num_stages": 1,
        }

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (32, 1, 1), (1, kernel_op_type, 16384)

    elif use_524288_aligned_32x8192:
        kernel = unary_aligned_core_loop_kernel
        grid = _fixed_grid(32)
        constexpr_kwargs = {
            "BLOCKS_PER_PROGRAM": 2,
            "OP_TYPE": kernel_op_type,
            "BLOCK_SIZE": 8192,
            "num_warps": 4,
            "num_stages": 1,
        }

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (32, 1, 1), (2, kernel_op_type, 8192)

    elif use_524288_exact_48core:
        kernel = unary_524288_exact_48core_kernel
        grid = _fixed_grid(48)
        constexpr_kwargs = {
            "OP_TYPE": kernel_op_type,
            "num_warps": 4,
            "num_stages": 1,
        }

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (48, 1, 1), (kernel_op_type,)

    elif use_1048576_exact_48core:
        kernel = unary_1048576_exact_48core_kernel
        grid = _fixed_grid(48)
        use_compiler_multibuffer = (
            kernel_op_type in {"log", "neg"} and input_spec.dtype == "float32"
        )
        constexpr_kwargs = {
            "OP_TYPE": kernel_op_type,
            "num_warps": 4,
            "num_stages": 2 if use_compiler_multibuffer else 1,
        }
        if use_compiler_multibuffer:
            constexpr_kwargs.update(
                {
                    "multibuffer": True,
                    "enable_ubuf_saving": True,
                }
            )

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (48, 1, 1), (kernel_op_type,)

    elif use_395523_tail4096:
        kernel = unary_395523_tail4096_kernel
        grid = _fixed_grid(48)
        constexpr_kwargs = {
            "OP_TYPE": kernel_op_type,
            "num_warps": 4,
            "num_stages": 1,
        }

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (48, 1, 1), (kernel_op_type,)

    elif use_395523_balanced:
        kernel = unary_395523_balanced_kernel
        grid = _fixed_grid(48)
        constexpr_kwargs = {
            "OP_TYPE": kernel_op_type,
            "num_warps": 4,
            "num_stages": 1,
        }

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (48, 1, 1), (kernel_op_type,)

    elif kernel_op_type in {"rsqrt", "log", "sqrt"} and n_elements == 1000:
        kernel = unary_1000_two_program_kernel
        grid = _fixed_grid(2)
        constexpr_kwargs = {
            "OP_TYPE": kernel_op_type,
            "num_warps": 4,
            "num_stages": 1,
        }

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (2, 1, 1), (kernel_op_type,)

    elif chunk_layout is not None:
        kernel = unary_balanced_chunks_kernel
        grid = _fixed_grid(balanced_program_count)
        alignment = 256
        elements_per_program = (
            triton.cdiv(
                triton.cdiv(n_elements, balanced_program_count),
                alignment,
            )
            * alignment
        )
        constexpr_kwargs = {
            "N_ELEMENTS": n_elements,
            "ELEMENTS_PER_PROGRAM": elements_per_program,
            "CHUNK_LAYOUT": chunk_layout,
            "OP_TYPE": kernel_op_type,
            "num_warps": 4,
            "num_stages": 1,
        }

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (*grid({}), 1, 1), (
                n_elements,
                elements_per_program,
                chunk_layout,
                kernel_op_type,
            )

    elif (
        _can_use_aligned_loop(n_elements, block_size)
        and aligned_program_count == balanced_program_count
    ):
        kernel = unary_aligned_core_loop_kernel
        grid = aligned_grid
        program_count = aligned_program_count
        blocks_per_program = n_elements // block_size // program_count
        constexpr_kwargs = {
            "BLOCKS_PER_PROGRAM": blocks_per_program,
            "OP_TYPE": kernel_op_type,
            "BLOCK_SIZE": block_size,
            "num_warps": 4,
            "num_stages": 1,
        }

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (*grid({"BLOCK_SIZE": block_size}), 1, 1), (
                blocks_per_program,
                kernel_op_type,
                block_size,
            )

    elif _can_use_aligned_loop(n_elements, block_size):
        kernel = unary_core_loop_kernel
        grid = balanced_grid
        program_count = balanced_program_count
        constexpr_kwargs = {
            "N_ELEMENTS": n_elements,
            "PROGRAM_COUNT": program_count,
            "OP_TYPE": kernel_op_type,
            "BLOCK_SIZE": block_size,
            "num_warps": 4,
            "num_stages": 1,
        }

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (*grid({"BLOCK_SIZE": block_size}), 1, 1), (
                n_elements,
                program_count,
                kernel_op_type,
                block_size,
            )

    else:
        kernel = unary_tiled_kernel
        grid = tiled_grid
        constexpr_kwargs = {
            "N_ELEMENTS": n_elements,
            "OP_TYPE": kernel_op_type,
            "BLOCK_SIZE": block_size,
            "num_warps": 4,
            "num_stages": 1,
        }

        def build_cached_call(
            constexprs: dict[str, Any],
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (*grid({"BLOCK_SIZE": block_size}), 1, 1), (
                n_elements,
                kernel_op_type,
                block_size,
            )

    return make_single_kernel_run_fn(
        PreparedSingleKernelRunSpec(
            kernel=PreparedSingleKernelSpec(
                kernel=kernel,
                grid=grid,
                static_args=(),
                constexpr_kwargs=constexpr_kwargs,
                build_cached_call=build_cached_call,
            ),
            input_checks=input_checks,
            output_factory=output_factory,
            runtime_args=runtime_args,
            extra_check=extra_check,
            validate_inputs=bool(attrs.get("_validate_inputs", True)),
        ),
        default_run_fn,
    )


__all__ = (
    "get_dense_unary_block_size",
    "prepare_dense_unary",
    "unary_aligned_core_loop_kernel",
    "unary_multibuffer_core_loop_kernel",
    "unary_neg_16bit_multibuffer_core_loop_kernel",
    "unary_permuted_core_loop_kernel",
    "swish_constexpr_core_loop_kernel",
    "unary_395523_balanced_kernel",
    "unary_395523_rsqrt_float16_tail512_kernel",
    "unary_395523_rsqrt_float32_tail512_kernel",
    "unary_395523_tail4096_kernel",
    "unary_1048576_leaky_relu_constexpr_kernel",
    "unary_1048576_bfloat16_leaky_relu_maximumf_permuted_kernel",
    "unary_1048576_leaky_relu_maximumf_balanced_kernel",
    "unary_1048576_leaky_relu_maximumf_permuted_kernel",
    "unary_524288_float32_leaky_relu_maximumf_balanced40_kernel",
    "unary_leaky_relu_maximumf_aligned_core_loop_kernel",
    "unary_1048576_three_phase_8192_kernel",
    "unary_1048576_permuted_exact_kernel",
    "unary_1048576_two_phase_16384_kernel",
    "unary_core_loop_kernel",
    "unary_tiled_kernel",
)
