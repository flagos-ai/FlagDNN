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

"""Ascend-only prepared kernels for dense two-input expressions."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import triton
import triton.language as tl

from flag_dnn.graph.device import is_runtime_device_tensor
from flag_dnn.utils import triton_lang_extension as tle
from flag_dnn.utils.libentry import libentry
from flag_dnn.utils.triton_lang_helper import tl_extra_shim as libdevice

from .binary import make_core_loop_grid

_FLOAT_DTYPES = {"float16", "bfloat16", "float32"}


def get_dense_pairwise_block_size(
    op_type: str, n_elements: int, dtype: Any
) -> int:
    if n_elements <= 1024:
        return 1024
    if n_elements <= 4096:
        return 2048
    if op_type in {"pow", "sigmoid_backward"}:
        return 4096
    return 4096 if "float32" in str(dtype) else 8192


def _can_use_aligned_loop(n_elements: int, block_size: int) -> bool:
    return n_elements >= 262144 and n_elements % block_size == 0


@triton.jit
def _pairwise_result(
    left,
    right,
    OP_TYPE: tl.constexpr,
    COMPUTE_FLOAT32: tl.constexpr,
):
    if OP_TYPE == "pow":
        result = libdevice.pow(
            left.to(tl.float32),
            right.to(tl.float32),
        )
    elif OP_TYPE == "add_square":
        if COMPUTE_FLOAT32:
            left = left.to(tl.float32)
            right = right.to(tl.float32)
        result = left + right * right
    elif OP_TYPE == "sigmoid_backward":
        loss = left.to(tl.float32)
        value = right.to(tl.float32)
        sigmoid = tl.sigmoid(value)
        result = loss * sigmoid * (1.0 - sigmoid)
    return result


@libentry()
@triton.jit
def pairwise_aligned_core_loop_kernel(
    left_ptr,
    right_ptr,
    output_ptr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    OP_TYPE: tl.constexpr,
    COMPUTE_FLOAT32: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    first_block = tle.program_id(0) * BLOCKS_PER_PROGRAM
    for local_block in range(0, BLOCKS_PER_PROGRAM):
        offsets = (first_block + local_block) * BLOCK_SIZE + tl.arange(
            0, BLOCK_SIZE
        )
        left = tl.load(left_ptr + offsets)
        right = tl.load(right_ptr + offsets)
        result = _pairwise_result(
            left,
            right,
            OP_TYPE,
            COMPUTE_FLOAT32,
        )
        tl.store(
            output_ptr + offsets,
            result.to(output_ptr.dtype.element_ty),
        )


@libentry()
@triton.jit
def pairwise_core_loop_kernel(
    left_ptr,
    right_ptr,
    output_ptr,
    N_ELEMENTS: tl.constexpr,
    OP_TYPE: tl.constexpr,
    COMPUTE_FLOAT32: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_programs = tle.num_programs(0)
    elements_per_program = tl.cdiv(N_ELEMENTS, num_programs)
    chunk_size = tl.cdiv(elements_per_program, 256) * 256
    chunk_start = pid * chunk_size
    chunk_end = tl.minimum(chunk_start + chunk_size, N_ELEMENTS)
    num_blocks = tl.cdiv(chunk_size, BLOCK_SIZE)

    for block_idx in range(0, num_blocks):
        offsets = (
            chunk_start + block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        )
        mask = offsets < chunk_end
        left = tl.load(left_ptr + offsets, mask=mask)
        right = tl.load(right_ptr + offsets, mask=mask)
        result = _pairwise_result(
            left,
            right,
            OP_TYPE,
            COMPUTE_FLOAT32,
        )
        tl.store(
            output_ptr + offsets,
            result.to(output_ptr.dtype.element_ty),
            mask=mask,
        )


def _compute_uses_float32(value: Any) -> bool:
    if value is None:
        return False
    if value is torch.float32:
        return True
    return str(value).lower() in {
        "float",
        "float32",
        "fp32",
        "torch.float32",
    }


def _prepare_dense_pairwise(
    *,
    kernel_op_type: str,
    attrs: dict[str, Any],
    input_specs: Sequence[Any],
    default_run_fn: Any,
) -> Optional[Any]:
    if len(input_specs) != 2:
        return None
    left_spec, right_spec = input_specs
    shape = tuple(left_spec.shape)
    if not all(isinstance(dim, int) for dim in shape):
        return None
    if shape != tuple(right_spec.shape):
        return None
    if (
        left_spec.layout not in ("contiguous", "nhwc")
        or right_spec.layout != left_spec.layout
        or left_spec.stride is None
        or right_spec.stride != left_spec.stride
        or left_spec.dtype not in _FLOAT_DTYPES
        or right_spec.dtype != left_spec.dtype
    ):
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
        (0, 1),
        require_shape=True,
        require_stride=True,
        require_dtype=True,
    )
    if input_checks is None:
        return None

    static_shape = tuple(int(dim) for dim in shape)
    static_stride = tuple(int(item) for item in left_spec.stride)
    output_dtype = torch_dtype(left_spec.dtype)
    compute_float32 = (
        True
        if kernel_op_type in {"pow", "sigmoid_backward"}
        else _compute_uses_float32(attrs.get("compute_data_type"))
    )
    n_elements = 1
    for dim in static_shape:
        n_elements *= dim
    if n_elements == 0:
        return None

    block_size = get_dense_pairwise_block_size(
        kernel_op_type, n_elements, left_spec.dtype
    )
    grid = make_core_loop_grid(n_elements, left_spec.device)
    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def output_factory(inputs: Sequence[Any]) -> torch.Tensor:
        source = inputs[0]
        key = (
            source.device.type,
            source.device.index,
            output_dtype,
            static_shape,
            static_stride,
        )
        return get_prepared_output(
            output_cache,
            key,
            lambda: torch.empty_strided(
                static_shape,
                static_stride,
                device=source.device,
                dtype=output_dtype,
            ),
        )

    def runtime_args(
        inputs: Sequence[Any], output: torch.Tensor
    ) -> tuple[Any, ...]:
        return (inputs[0], inputs[1], output)

    def extra_check(inputs: Sequence[Any]) -> bool:
        left, right = inputs
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and is_runtime_device_tensor(left)
            and is_runtime_device_tensor(right)
            and left.device == right.device
        )

    if _can_use_aligned_loop(n_elements, block_size):
        kernel = pairwise_aligned_core_loop_kernel
        program_count = grid({"BLOCK_SIZE": block_size})[0]
        blocks_per_program = n_elements // block_size // program_count
        constexpr_kwargs = {
            "BLOCKS_PER_PROGRAM": blocks_per_program,
            "OP_TYPE": kernel_op_type,
            "COMPUTE_FLOAT32": compute_float32,
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
                compute_float32,
                block_size,
            )

    else:
        kernel = pairwise_core_loop_kernel
        constexpr_kwargs = {
            "N_ELEMENTS": n_elements,
            "OP_TYPE": kernel_op_type,
            "COMPUTE_FLOAT32": compute_float32,
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
                compute_float32,
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


def prepare_dense_pow(**kwargs: Any) -> Optional[Any]:
    return _prepare_dense_pairwise(kernel_op_type="pow", **kwargs)


def prepare_dense_add_square(**kwargs: Any) -> Optional[Any]:
    return _prepare_dense_pairwise(kernel_op_type="add_square", **kwargs)


def prepare_dense_sigmoid_backward(**kwargs: Any) -> Optional[Any]:
    return _prepare_dense_pairwise(
        kernel_op_type="sigmoid_backward",
        **kwargs,
    )


__all__ = (
    "get_dense_pairwise_block_size",
    "pairwise_aligned_core_loop_kernel",
    "pairwise_core_loop_kernel",
    "prepare_dense_add_square",
    "prepare_dense_pow",
    "prepare_dense_sigmoid_backward",
)
