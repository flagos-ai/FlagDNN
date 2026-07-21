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

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Mapping, Optional, Sequence

import triton
import triton.language as tl
import triton.runtime.driver as driver

from flag_dnn.utils import triton_lang_extension as tle
from flag_dnn.utils.libentry import libentry

Grid = Callable[[dict[str, Any]], tuple[int, ...]]


def _device_index(device: Any) -> int:
    if isinstance(device, int):
        return device
    if isinstance(device, str) and ":" in device:
        return int(device.rsplit(":", 1)[1])
    index = getattr(device, "index", None)
    if index is not None and not callable(index):
        return int(index)
    return int(driver.active.get_current_device())


def _get_device_properties(device_index: int) -> Any:
    return driver.active.utils.get_device_properties(device_index)


def _property(properties: Any, name: str) -> Any:
    if isinstance(properties, Mapping):
        return properties.get(name)
    return getattr(properties, name, None)


@lru_cache(maxsize=None)
def get_vector_core_count(device: Any) -> int:
    """Return the number of Ascend vector cores for one logical device."""
    device_index = _device_index(device)
    properties = _get_device_properties(device_index)
    vector_cores = _property(properties, "num_vectorcore")
    if vector_cores is None:
        ai_cores = _property(properties, "num_aicore")
        if ai_cores is not None:
            vector_cores = int(ai_cores) * 2
    if vector_cores is None or int(vector_cores) <= 0:
        raise RuntimeError(
            "Ascend device properties do not expose a positive "
            f"num_vectorcore value: device={device_index}, "
            f"properties={properties!r}"
        )
    return int(vector_cores)


def make_core_loop_grid(n_elements: int, device: Any) -> Grid:
    """Cap launch tasks at physical vector cores for Ascend VV kernels."""
    vector_cores = get_vector_core_count(device)

    def grid(meta: dict[str, Any]) -> tuple[int, ...]:
        block_size = int(meta["BLOCK_SIZE"])
        num_blocks = (n_elements + block_size - 1) // block_size
        programs = max(1, min(vector_cores, num_blocks))
        if n_elements % block_size == 0:
            while num_blocks % programs != 0:
                programs -= 1
        return (programs,)

    return grid


def get_add_block_size(n_elements: int, dtype: Any, device: Any) -> int:
    """Choose one UB tile from the per-Vector-Core pointwise workload."""
    if n_elements <= 1024:
        return 1024
    if n_elements <= 4096:
        return 2048

    vector_cores = get_vector_core_count(device)
    per_core = (n_elements + vector_cores - 1) // vector_cores
    block_size = 1 << max(0, per_core - 1).bit_length()
    max_block_size = 8192 if "float32" in str(dtype) else 16384
    block_size = min(max_block_size, max(2048, block_size))
    while block_size > 2048:
        smaller = block_size // 2
        padded = ((per_core + block_size - 1) // block_size) * block_size
        smaller_padded = ((per_core + smaller - 1) // smaller) * smaller
        if smaller_padded * 4 > padded * 3:
            break
        block_size = smaller
    return block_size


def can_use_aligned_core_loop(n_elements: int, block_size: int) -> bool:
    """Return whether a large tensor can use an unmasked static core loop."""
    return n_elements >= 262144 and n_elements % block_size == 0


def launch_dense_binary(
    *,
    op_type: str,
    input: Any,
    other: Any,
    out: Any,
    n_elements: int,
    alpha: Any,
) -> bool:
    """Launch the Ascend dense Add specialization when it applies."""
    if op_type != "add":
        return False
    block_size = get_add_block_size(n_elements, input.dtype, input.device)
    grid = make_core_loop_grid(n_elements, input.device)
    alpha_is_one = float(alpha) == 1.0
    if alpha_is_one and can_use_aligned_core_loop(n_elements, block_size):
        program_count = grid({"BLOCK_SIZE": block_size})[0]
        blocks_per_program = n_elements // block_size // program_count
        add_tensor_aligned_core_loop_kernel[grid](
            input,
            other,
            out,
            BLOCKS_PER_PROGRAM=blocks_per_program,
            BLOCK_SIZE=block_size,
            num_warps=4,
            num_stages=1,
        )
    else:
        add_tensor_core_loop_kernel[grid](
            input,
            other,
            out,
            float(alpha),
            N_ELEMENTS=n_elements,
            ALPHA_IS_ONE=alpha_is_one,
            ALIGNED_BLOCKS=(
                n_elements >= 262144 and n_elements % block_size == 0
            ),
            BLOCK_SIZE=block_size,
            num_warps=4,
            num_stages=1,
        )
    return True


def prepare_dense_binary(
    *,
    kernel_op_type: str,
    left_spec: Any,
    input_checks: Any,
    output_factory: Any,
    default_run_fn: Any,
    extra_check: Any,
    n_elements: int,
    alpha: float,
    validate_inputs: bool,
) -> Optional[Any]:
    """Build the prepared Ascend Add replay.

    The implementation stays outside the common graph code.
    """
    if kernel_op_type != "add":
        return None

    from flag_dnn.graph.prepared import (
        PreparedSingleKernelRunSpec,
        PreparedSingleKernelSpec,
        make_single_kernel_run_fn,
    )

    block_size = get_add_block_size(
        n_elements, left_spec.dtype, left_spec.device
    )
    grid = make_core_loop_grid(n_elements, left_spec.device)
    alpha_is_one = alpha == 1.0

    if alpha_is_one and can_use_aligned_core_loop(n_elements, block_size):
        kernel = add_tensor_aligned_core_loop_kernel
        program_count = grid({"BLOCK_SIZE": block_size})[0]
        blocks_per_program = n_elements // block_size // program_count
        constexpr_kwargs = {
            "BLOCKS_PER_PROGRAM": blocks_per_program,
            "BLOCK_SIZE": block_size,
            "num_warps": 4,
            "num_stages": 1,
        }

        def runtime_args(
            inputs: Sequence[Any], output: Any
        ) -> tuple[Any, ...]:
            return (inputs[0], inputs[1], output)

        def build_cached_call(
            constexprs: dict[str, Any]
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            static_grid = (*grid({"BLOCK_SIZE": block_size}), 1, 1)
            return static_grid, (blocks_per_program, block_size)

    else:
        kernel = add_tensor_core_loop_kernel
        aligned_blocks = n_elements >= 262144 and n_elements % block_size == 0
        constexpr_kwargs = {
            "N_ELEMENTS": n_elements,
            "ALPHA_IS_ONE": alpha_is_one,
            "ALIGNED_BLOCKS": aligned_blocks,
            "BLOCK_SIZE": block_size,
            "num_warps": 4,
            "num_stages": 1,
        }

        def runtime_args(
            inputs: Sequence[Any], output: Any
        ) -> tuple[Any, ...]:
            return (inputs[0], inputs[1], output, alpha)

        def build_cached_call(
            constexprs: dict[str, Any]
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            static_grid = (*grid({"BLOCK_SIZE": block_size}), 1, 1)
            return static_grid, (
                n_elements,
                alpha_is_one,
                aligned_blocks,
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
            validate_inputs=validate_inputs,
        ),
        default_run_fn,
    )


@libentry()
@triton.jit
def add_tensor_aligned_core_loop_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    first_block = tle.program_id(0) * BLOCKS_PER_PROGRAM
    for local_block in range(0, BLOCKS_PER_PROGRAM):
        offsets = (first_block + local_block) * BLOCK_SIZE + tl.arange(
            0, BLOCK_SIZE
        )
        x = tl.load(x_ptr + offsets)
        y = tl.load(y_ptr + offsets)
        tl.store(out_ptr + offsets, x + y)


@libentry()
@triton.jit
def add_tensor_core_loop_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    alpha_val,
    N_ELEMENTS: tl.constexpr,
    ALPHA_IS_ONE: tl.constexpr,
    ALIGNED_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_programs = tle.num_programs(0)
    # Give each Vector Core one contiguous, nearly equal-sized region.  A
    # round-robin tile loop leaves some cores with one extra full UB tile,
    # which is a large imbalance for the 0.2--1M element pointwise range.
    elements_per_program = tl.cdiv(N_ELEMENTS, num_programs)
    chunk_size = tl.cdiv(elements_per_program, 256) * 256
    chunk_start = pid * chunk_size
    chunk_end = tl.minimum(chunk_start + chunk_size, N_ELEMENTS)
    num_blocks = tl.cdiv(chunk_size, BLOCK_SIZE)

    for block_idx in range(0, num_blocks):
        offsets = (
            chunk_start + block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        )
        if ALIGNED_BLOCKS:
            x = tl.load(x_ptr + offsets)
            y = tl.load(y_ptr + offsets)
        else:
            mask = offsets < chunk_end
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
        if ALPHA_IS_ONE:
            result = x + y
        else:
            result = x + alpha_val * y
        if ALIGNED_BLOCKS:
            tl.store(out_ptr + offsets, result)
        else:
            tl.store(out_ptr + offsets, result, mask=mask)


__all__: list[str] = [
    "add_tensor_aligned_core_loop_kernel",
    "add_tensor_core_loop_kernel",
    "can_use_aligned_core_loop",
    "get_add_block_size",
    "get_vector_core_count",
    "launch_dense_binary",
    "make_core_loop_grid",
    "prepare_dense_binary",
]
