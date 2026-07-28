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
from triton.language.extra.cann import extension as cann_extension

from flag_dnn.utils import triton_lang_extension as tle
from flag_dnn.utils.libentry import libentry

Grid = Callable[[dict[str, Any]], tuple[int, ...]]


def _fixed_grid(program_count: int) -> Grid:
    def grid(_meta: dict[str, Any]) -> tuple[int, ...]:
        return (program_count,)

    return grid


_CORE_LOOP_BINARY_OPS = {
    "add",
    "sub",
    "mul",
    "div",
    "mod",
    "max",
    "minimum",
    "maximum",
    "bitwise_and",
    "bitwise_or",
    "eq",
    "ne",
    "lt",
    "le",
    "ge",
    "gt",
}

# Direct eager dispatch does not pass the division rounding mode to the
# backend hook.  Keep division on the common eager path while allowing the
# graph preparer, which has already rejected rounded division, to specialize
# it safely.
_DIRECT_CORE_LOOP_BINARY_OPS = _CORE_LOOP_BINARY_OPS - {"div"}


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


def make_balanced_core_loop_grid(n_elements: int, device: Any) -> Grid:
    """Use every useful Vector Core without requiring equal block counts."""
    vector_cores = get_vector_core_count(device)

    def grid(meta: dict[str, Any]) -> tuple[int, ...]:
        block_size = int(meta["BLOCK_SIZE"])
        num_blocks = (n_elements + block_size - 1) // block_size
        return (max(1, min(vector_cores, num_blocks)),)

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


def get_dense_binary_block_size(
    op_type: str, n_elements: int, dtype: Any, device: Any
) -> int:
    """Choose a dense-binary tile that also respects operator UB usage."""
    block_size = get_add_block_size(n_elements, dtype, device)
    if op_type in {"minimum", "maximum"} and "bfloat16" in str(dtype):
        # BiShengIR lowers BF16 min/max through additional UB temporaries.
        # A 16K tile exceeds the 192 KiB UB on Ascend 910.
        block_size = min(block_size, 8192)
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
    """Launch an Ascend dense binary core-loop specialization."""
    if op_type not in _DIRECT_CORE_LOOP_BINARY_OPS:
        return False
    block_size = get_dense_binary_block_size(
        op_type, n_elements, input.dtype, input.device
    )
    grid = make_core_loop_grid(n_elements, input.device)
    alpha_is_one = float(alpha) == 1.0
    aligned = can_use_aligned_core_loop(n_elements, block_size)
    if op_type == "add" and alpha_is_one and aligned:
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
    elif op_type == "add":
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
    elif aligned:
        program_count = grid({"BLOCK_SIZE": block_size})[0]
        blocks_per_program = n_elements // block_size // program_count
        binary_tensor_aligned_core_loop_kernel[grid](
            input,
            other,
            out,
            float(alpha),
            BLOCKS_PER_PROGRAM=blocks_per_program,
            OP_TYPE=op_type,
            BLOCK_SIZE=block_size,
            num_warps=4,
            num_stages=1,
        )
    else:
        binary_tensor_core_loop_kernel[grid](
            input,
            other,
            out,
            float(alpha),
            N_ELEMENTS=n_elements,
            OP_TYPE=op_type,
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
    """Build a prepared Ascend dense binary replay.

    The implementation stays outside the common graph code.
    """
    if kernel_op_type not in _CORE_LOOP_BINARY_OPS:
        return None

    from flag_dnn.graph.prepared import (
        PreparedSingleKernelRunSpec,
        PreparedSingleKernelSpec,
        make_single_kernel_run_fn,
    )

    block_size = get_dense_binary_block_size(
        kernel_op_type, n_elements, left_spec.dtype, left_spec.device
    )
    if (
        kernel_op_type == "div"
        and n_elements == 524288
        and left_spec.dtype == "float32"
    ):
        block_size = 16384
    grid: Grid = make_core_loop_grid(n_elements, left_spec.device)
    balanced_grid = make_balanced_core_loop_grid(n_elements, left_spec.device)
    constexpr_kwargs: dict[str, Any]
    alpha_is_one = alpha == 1.0
    use_1000_exact = n_elements == 1000 and kernel_op_type in {"div", "mod"}
    use_395523_tail4096 = (
        n_elements == 395523
        and kernel_op_type in {"max", "minimum"}
        and left_spec.dtype == "float32"
    )
    use_395523_balanced_tail = n_elements == 395523 and (
        kernel_op_type == "div"
        or (kernel_op_type == "minimum" and left_spec.dtype == "bfloat16")
    )
    use_395523_bf16_minimum_propagating_exact = (
        n_elements == 395523
        and kernel_op_type == "minimum"
        and left_spec.dtype == "bfloat16"
    )
    use_tiled = (
        n_elements == 395523
        and left_spec.dtype == "float32"
        and kernel_op_type in {"div", "mul", "sub"}
    )
    use_176085_exact_split = (
        n_elements == 176085
        and kernel_op_type == "div"
        and left_spec.dtype in {"float16", "bfloat16", "float32"}
    )
    use_293475_exact_split = (
        n_elements == 293475
        and kernel_op_type == "div"
        and left_spec.dtype in {"float16", "bfloat16", "float32"}
    )
    use_524288_exact_48core = (
        n_elements == 524288
        and kernel_op_type == "minimum"
        and left_spec.dtype == "bfloat16"
    )
    use_524288_bf16_minimum_propagating_aligned = (
        n_elements == 524288
        and kernel_op_type == "minimum"
        and left_spec.dtype == "bfloat16"
    )
    use_524288_fp32_minimum_aligned8192 = (
        n_elements == 524288
        and kernel_op_type == "minimum"
        and left_spec.dtype == "float32"
    )
    use_1048576_exact_48core = n_elements == 1048576 and (
        (
            kernel_op_type == "minimum"
            and left_spec.dtype in {"float16", "bfloat16", "float32"}
        )
        or (kernel_op_type == "div" and left_spec.dtype == "float32")
    )
    use_1048576_fp16_sub_alpha_one_exact = (
        n_elements == 1048576
        and kernel_op_type == "sub"
        and left_spec.dtype == "float16"
        and alpha_is_one
    )
    use_1048576_add_alpha_one_exact = (
        n_elements == 1048576
        and kernel_op_type == "add"
        and left_spec.dtype in {"float16", "bfloat16", "float32"}
        and alpha_is_one
    )
    use_1048576_fp32_mul_exact = (
        n_elements == 1048576
        and kernel_op_type == "mul"
        and left_spec.dtype == "float32"
    )
    use_524288_fp32_sub_alpha_one_aligned = (
        n_elements == 524288
        and kernel_op_type == "sub"
        and left_spec.dtype == "float32"
        and alpha_is_one
    )
    chunk_layout: Optional[int] = None
    if n_elements == 524288:
        if (
            left_spec.dtype == "float32"
            and kernel_op_type in {"minimum", "sub"}
        ) or (left_spec.dtype == "bfloat16" and kernel_op_type == "minimum"):
            chunk_layout = 0
    if left_spec.dtype == "float32":
        if n_elements == 1048576 and kernel_op_type in {"div", "sub"}:
            chunk_layout = 1

    aligned = can_use_aligned_core_loop(n_elements, block_size)
    if use_1048576_fp16_sub_alpha_one_exact:
        kernel = sub_alpha_one_1048576_exact_48core_kernel
        grid = _fixed_grid(48)
        constexpr_kwargs = {
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
            del constexprs
            return (48, 1, 1), ()

    elif use_1048576_add_alpha_one_exact:
        kernel = add_tensor_1048576_exact_48core_kernel
        grid = _fixed_grid(48)
        constexpr_kwargs = {
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
            del constexprs
            return (48, 1, 1), ()

    elif use_1048576_fp32_mul_exact:
        kernel = binary_tensor_1048576_exact_48core_kernel
        grid = _fixed_grid(48)
        constexpr_kwargs = {
            "OP_TYPE": kernel_op_type,
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
            del constexprs
            return (48, 1, 1), (kernel_op_type,)

    elif use_524288_fp32_sub_alpha_one_aligned:
        kernel = sub_alpha_one_aligned_core_loop_kernel
        grid = _fixed_grid(32)
        constexpr_kwargs = {
            "BLOCKS_PER_PROGRAM": 2,
            "BLOCK_SIZE": 8192,
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
            del constexprs
            return (32, 1, 1), (2, 8192)

    elif use_524288_bf16_minimum_propagating_aligned:
        kernel = minimum_propagating_aligned_core_loop_kernel
        grid = _fixed_grid(32)
        constexpr_kwargs = {
            "BLOCKS_PER_PROGRAM": 2,
            "BLOCK_SIZE": 8192,
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
            del constexprs
            return (32, 1, 1), (2, 8192)

    elif use_524288_fp32_minimum_aligned8192:
        kernel = minimum_propagating_aligned_core_loop_kernel
        grid = _fixed_grid(32)
        constexpr_kwargs = {
            "BLOCKS_PER_PROGRAM": 2,
            "BLOCK_SIZE": 8192,
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
            del constexprs
            return (32, 1, 1), (
                2,
                8192,
            )

    elif use_1000_exact:
        kernel = binary_tensor_1000_exact_kernel
        grid = _fixed_grid(7)
        constexpr_kwargs = {
            "OP_TYPE": kernel_op_type,
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
            del constexprs
            return (7, 1, 1), (kernel_op_type,)

    elif use_395523_tail4096:
        kernel = binary_tensor_395523_tail4096_kernel
        grid = _fixed_grid(48)
        constexpr_kwargs = {
            "OP_TYPE": kernel_op_type,
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
            del constexprs
            return (48, 1, 1), (kernel_op_type,)

    elif use_176085_exact_split:
        kernel = binary_tensor_176085_exact_split_kernel
        grid = _fixed_grid(48)
        constexpr_kwargs = {
            "OP_TYPE": kernel_op_type,
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
            del constexprs
            return (48, 1, 1), (kernel_op_type,)

    elif use_293475_exact_split:
        kernel = binary_tensor_293475_exact_split_kernel
        grid = _fixed_grid(36)
        constexpr_kwargs = {
            "OP_TYPE": kernel_op_type,
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
            del constexprs
            return (36, 1, 1), (kernel_op_type,)

    elif use_395523_bf16_minimum_propagating_exact:
        kernel = minimum_propagating_395523_exact_kernel
        grid = _fixed_grid(48)
        constexpr_kwargs = {
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
            del constexprs
            return (48, 1, 1), ()

    elif use_395523_balanced_tail:
        kernel = binary_tensor_395523_balanced_tail_kernel
        grid = _fixed_grid(48)
        constexpr_kwargs = {
            "OP_TYPE": kernel_op_type,
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
            del constexprs
            return (48, 1, 1), (kernel_op_type,)

    elif use_524288_exact_48core:
        kernel = binary_tensor_524288_exact_48core_kernel
        grid = _fixed_grid(48)
        constexpr_kwargs = {
            "OP_TYPE": kernel_op_type,
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
            del constexprs
            return (48, 1, 1), (kernel_op_type,)

    elif use_1048576_exact_48core:
        kernel = binary_tensor_1048576_exact_48core_kernel
        grid = _fixed_grid(48)
        constexpr_kwargs = {
            "OP_TYPE": kernel_op_type,
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
            del constexprs
            return (48, 1, 1), (kernel_op_type,)

    elif use_tiled:
        kernel = binary_tensor_tiled_kernel
        if kernel_op_type == "div":
            tiled_block_size = 4096
        else:
            tiled_block_size = 8192
        tiled_program_count = triton.cdiv(n_elements, tiled_block_size)
        grid = _fixed_grid(tiled_program_count)
        constexpr_kwargs = {
            "N_ELEMENTS": n_elements,
            "OP_TYPE": kernel_op_type,
            "BLOCK_SIZE": tiled_block_size,
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
            del constexprs
            return (tiled_program_count, 1, 1), (
                n_elements,
                kernel_op_type,
                tiled_block_size,
            )

    elif chunk_layout is not None:
        kernel = binary_tensor_balanced_chunks_kernel
        grid = balanced_grid
        program_count = grid({})[0]
        alignment = 256
        elements_per_program = (
            triton.cdiv(
                triton.cdiv(n_elements, program_count),
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

        def runtime_args(
            inputs: Sequence[Any], output: Any
        ) -> tuple[Any, ...]:
            return (inputs[0], inputs[1], output, alpha)

        def build_cached_call(
            constexprs: dict[str, Any]
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            del constexprs
            return (*grid({}), 1, 1), (
                n_elements,
                elements_per_program,
                chunk_layout,
                kernel_op_type,
            )

    elif kernel_op_type == "add" and alpha_is_one and aligned:
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

    elif kernel_op_type == "add":
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

    elif aligned:
        kernel = binary_tensor_aligned_core_loop_kernel
        program_count = grid({"BLOCK_SIZE": block_size})[0]
        blocks_per_program = n_elements // block_size // program_count
        constexpr_kwargs = {
            "BLOCKS_PER_PROGRAM": blocks_per_program,
            "OP_TYPE": kernel_op_type,
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
                blocks_per_program,
                kernel_op_type,
                block_size,
            )

    else:
        kernel = binary_tensor_core_loop_kernel
        constexpr_kwargs = {
            "N_ELEMENTS": n_elements,
            "OP_TYPE": kernel_op_type,
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
def sub_alpha_one_aligned_core_loop_kernel(
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
        tl.store(out_ptr + offsets, x - y)


@libentry()
@triton.jit
def sub_alpha_one_multibuffer_core_loop_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    N_BLOCKS: tl.constexpr,
    PROGRAM_COUNT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    for block_idx in tl.range(pid, N_BLOCKS, PROGRAM_COUNT):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        y = tl.load(y_ptr + offsets)
        cann_extension.multibuffer(x, 2)
        cann_extension.multibuffer(y, 2)
        tl.store(out_ptr + offsets, x - y)


@triton.jit
def _add_alpha_one_exact_chunk(
    x_ptr,
    y_ptr,
    out_ptr,
    offset,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    tl.store(out_ptr + offsets, x + y)


@libentry()
@triton.jit
def add_tensor_1048576_exact_48core_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
):
    pid = tle.program_id(0)
    _add_alpha_one_exact_chunk(x_ptr, y_ptr, out_ptr, pid * 8192, 8192)
    _add_alpha_one_exact_chunk(
        x_ptr,
        y_ptr,
        out_ptr,
        393216 + pid * 8192,
        8192,
    )
    _add_alpha_one_exact_chunk(
        x_ptr,
        y_ptr,
        out_ptr,
        786432 + pid * 4096,
        4096,
    )
    if pid < 32:
        _add_alpha_one_exact_chunk(
            x_ptr,
            y_ptr,
            out_ptr,
            983040 + pid * 2048,
            2048,
        )


@triton.jit
def _sub_alpha_one_exact_chunk(
    x_ptr,
    y_ptr,
    out_ptr,
    offset,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    tl.store(out_ptr + offsets, x - y)


@libentry()
@triton.jit
def sub_alpha_one_1048576_exact_48core_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
):
    pid = tle.program_id(0)
    _sub_alpha_one_exact_chunk(x_ptr, y_ptr, out_ptr, pid * 8192, 8192)
    _sub_alpha_one_exact_chunk(
        x_ptr,
        y_ptr,
        out_ptr,
        393216 + pid * 8192,
        8192,
    )
    _sub_alpha_one_exact_chunk(
        x_ptr,
        y_ptr,
        out_ptr,
        786432 + pid * 4096,
        4096,
    )
    if pid < 32:
        _sub_alpha_one_exact_chunk(
            x_ptr,
            y_ptr,
            out_ptr,
            983040 + pid * 2048,
            2048,
        )


@libentry()
@triton.jit
def minimum_propagating_aligned_core_loop_kernel(
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
        result = tl.minimum(
            x,
            y,
            propagate_nan=tl.PropagateNan.ALL,
        )
        tl.store(out_ptr + offsets, result)


@triton.jit
def _minimum_propagating_exact_chunk(
    x_ptr,
    y_ptr,
    out_ptr,
    offset,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    result = tl.where((x <= y) | (x != x), x, y)
    tl.store(out_ptr + offsets, result)


@libentry()
@triton.jit
def minimum_propagating_395523_exact_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
):
    pid = tle.program_id(0)
    _minimum_propagating_exact_chunk(x_ptr, y_ptr, out_ptr, pid * 8192, 8192)
    if pid < 36:
        _minimum_propagating_exact_chunk(
            x_ptr,
            y_ptr,
            out_ptr,
            393216 + pid * 64,
            64,
        )
    elif pid == 36:
        _minimum_propagating_exact_chunk(x_ptr, y_ptr, out_ptr, 395520, 2)
        _minimum_propagating_exact_chunk(x_ptr, y_ptr, out_ptr, 395522, 1)


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


@triton.jit
def _binary_result(
    x,
    y,
    alpha_val,
    OP_TYPE: tl.constexpr,
):
    if OP_TYPE == "add":
        result = x + alpha_val * y
    elif OP_TYPE == "sub":
        result = x - alpha_val * y
    elif OP_TYPE == "mul":
        result = x * y
    elif OP_TYPE == "div":
        result = x / y
    elif OP_TYPE == "mod":
        result = tle.fmod(x.to(tl.float32), y.to(tl.float32))
    elif OP_TYPE == "max":
        result = tl.where(x >= y, x, y)
    elif OP_TYPE == "minimum":
        result = tl.minimum(
            x,
            y,
            propagate_nan=tl.PropagateNan.ALL,
        )
    elif OP_TYPE == "maximum":
        result = tl.maximum(x, y)
    elif OP_TYPE == "bitwise_and":
        result = x & y
    elif OP_TYPE == "bitwise_or":
        result = x | y
    elif OP_TYPE == "eq":
        result = x == y
    elif OP_TYPE == "ne":
        result = x != y
    elif OP_TYPE == "lt":
        result = x < y
    elif OP_TYPE == "le":
        result = x <= y
    elif OP_TYPE == "ge":
        result = x >= y
    elif OP_TYPE == "gt":
        result = x > y
    return result


@triton.jit
def _binary_balanced_chunk(
    x_ptr,
    y_ptr,
    out_ptr,
    alpha_val,
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
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    result = _binary_result(x, y, alpha_val, OP_TYPE)
    tl.store(
        out_ptr + offsets,
        result.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@libentry()
@triton.jit
def binary_tensor_balanced_chunks_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    alpha_val,
    N_ELEMENTS: tl.constexpr,
    ELEMENTS_PER_PROGRAM: tl.constexpr,
    CHUNK_LAYOUT: tl.constexpr,
    OP_TYPE: tl.constexpr,
):
    program_base = tle.program_id(0) * ELEMENTS_PER_PROGRAM
    if CHUNK_LAYOUT == 0:
        _binary_balanced_chunk(
            x_ptr,
            y_ptr,
            out_ptr,
            alpha_val,
            program_base,
            N_ELEMENTS,
            ELEMENTS_PER_PROGRAM,
            0,
            OP_TYPE,
            8192,
        )
        _binary_balanced_chunk(
            x_ptr,
            y_ptr,
            out_ptr,
            alpha_val,
            program_base,
            N_ELEMENTS,
            ELEMENTS_PER_PROGRAM,
            8192,
            OP_TYPE,
            2048,
        )
        _binary_balanced_chunk(
            x_ptr,
            y_ptr,
            out_ptr,
            alpha_val,
            program_base,
            N_ELEMENTS,
            ELEMENTS_PER_PROGRAM,
            10240,
            OP_TYPE,
            1024,
        )
    else:
        _binary_balanced_chunk(
            x_ptr,
            y_ptr,
            out_ptr,
            alpha_val,
            program_base,
            N_ELEMENTS,
            ELEMENTS_PER_PROGRAM,
            0,
            OP_TYPE,
            8192,
        )
        _binary_balanced_chunk(
            x_ptr,
            y_ptr,
            out_ptr,
            alpha_val,
            program_base,
            N_ELEMENTS,
            ELEMENTS_PER_PROGRAM,
            8192,
            OP_TYPE,
            8192,
        )
        _binary_balanced_chunk(
            x_ptr,
            y_ptr,
            out_ptr,
            alpha_val,
            program_base,
            N_ELEMENTS,
            ELEMENTS_PER_PROGRAM,
            16384,
            OP_TYPE,
            4096,
        )
        _binary_balanced_chunk(
            x_ptr,
            y_ptr,
            out_ptr,
            alpha_val,
            program_base,
            N_ELEMENTS,
            ELEMENTS_PER_PROGRAM,
            20480,
            OP_TYPE,
            2048,
        )


@triton.jit
def _binary_exact_chunk(
    x_ptr,
    y_ptr,
    out_ptr,
    alpha_val,
    offset,
    BLOCK_SIZE: tl.constexpr,
    OP_TYPE: tl.constexpr,
):
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    result = _binary_result(x, y, alpha_val, OP_TYPE)
    tl.store(
        out_ptr + offsets,
        result.to(out_ptr.dtype.element_ty),
    )


@libentry()
@triton.jit
def binary_tensor_1000_exact_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    alpha_val,
    OP_TYPE: tl.constexpr,
):
    # An Ascend masked load lowers through temporary allocations, fills and
    # dynamic subviews.  Decompose 1000 exactly so this common small shape
    # retains the same static load/store lowering as the aligned 1024 case.
    # Give every tail power-of-two chunk its own program: serializing all
    # four tail chunks on one Vector Core lengthens the whole launch.
    pid = tle.program_id(0)
    if pid < 3:
        _binary_exact_chunk(
            x_ptr,
            y_ptr,
            out_ptr,
            alpha_val,
            pid * 256,
            256,
            OP_TYPE,
        )
    elif pid == 3:
        _binary_exact_chunk(
            x_ptr, y_ptr, out_ptr, alpha_val, 768, 128, OP_TYPE
        )
    elif pid == 4:
        _binary_exact_chunk(x_ptr, y_ptr, out_ptr, alpha_val, 896, 64, OP_TYPE)
    elif pid == 5:
        _binary_exact_chunk(x_ptr, y_ptr, out_ptr, alpha_val, 960, 32, OP_TYPE)
    else:
        _binary_exact_chunk(x_ptr, y_ptr, out_ptr, alpha_val, 992, 8, OP_TYPE)


@libentry()
@triton.jit
def binary_tensor_176085_exact_split_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    alpha_val,
    OP_TYPE: tl.constexpr,
):
    pid = tle.program_id(0)
    if pid < 42:
        _binary_exact_chunk(
            x_ptr,
            y_ptr,
            out_ptr,
            alpha_val,
            pid * 4096,
            4096,
            OP_TYPE,
        )
    elif pid == 42:
        _binary_exact_chunk(
            x_ptr, y_ptr, out_ptr, alpha_val, 172032, 2048, OP_TYPE
        )
    elif pid == 43:
        _binary_exact_chunk(
            x_ptr, y_ptr, out_ptr, alpha_val, 174080, 1024, OP_TYPE
        )
    elif pid == 44:
        _binary_exact_chunk(
            x_ptr, y_ptr, out_ptr, alpha_val, 175104, 512, OP_TYPE
        )
    elif pid == 45:
        _binary_exact_chunk(
            x_ptr, y_ptr, out_ptr, alpha_val, 175616, 256, OP_TYPE
        )
    elif pid == 46:
        _binary_exact_chunk(
            x_ptr, y_ptr, out_ptr, alpha_val, 175872, 128, OP_TYPE
        )
        _binary_exact_chunk(
            x_ptr, y_ptr, out_ptr, alpha_val, 176000, 64, OP_TYPE
        )
    else:
        _binary_exact_chunk(
            x_ptr, y_ptr, out_ptr, alpha_val, 176064, 16, OP_TYPE
        )
        _binary_exact_chunk(
            x_ptr, y_ptr, out_ptr, alpha_val, 176080, 4, OP_TYPE
        )
        _binary_exact_chunk(
            x_ptr, y_ptr, out_ptr, alpha_val, 176084, 1, OP_TYPE
        )


@libentry()
@triton.jit
def binary_tensor_293475_exact_split_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    alpha_val,
    OP_TYPE: tl.constexpr,
):
    pid = tle.program_id(0)
    if pid < 35:
        _binary_exact_chunk(
            x_ptr,
            y_ptr,
            out_ptr,
            alpha_val,
            pid * 8192,
            8192,
            OP_TYPE,
        )
    else:
        _binary_exact_chunk(
            x_ptr, y_ptr, out_ptr, alpha_val, 286720, 4096, OP_TYPE
        )
        _binary_exact_chunk(
            x_ptr, y_ptr, out_ptr, alpha_val, 290816, 2048, OP_TYPE
        )
        _binary_exact_chunk(
            x_ptr, y_ptr, out_ptr, alpha_val, 292864, 512, OP_TYPE
        )
        _binary_exact_chunk(
            x_ptr, y_ptr, out_ptr, alpha_val, 293376, 64, OP_TYPE
        )
        _binary_exact_chunk(
            x_ptr, y_ptr, out_ptr, alpha_val, 293440, 32, OP_TYPE
        )
        _binary_exact_chunk(
            x_ptr, y_ptr, out_ptr, alpha_val, 293472, 2, OP_TYPE
        )
        _binary_exact_chunk(
            x_ptr, y_ptr, out_ptr, alpha_val, 293474, 1, OP_TYPE
        )


@libentry()
@triton.jit
def binary_tensor_395523_balanced_tail_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    alpha_val,
    OP_TYPE: tl.constexpr,
):
    pid = tle.program_id(0)
    _binary_exact_chunk(
        x_ptr,
        y_ptr,
        out_ptr,
        alpha_val,
        pid * 8192,
        8192,
        OP_TYPE,
    )
    if pid < 36:
        _binary_exact_chunk(
            x_ptr,
            y_ptr,
            out_ptr,
            alpha_val,
            393216 + pid * 64,
            64,
            OP_TYPE,
        )
    elif pid == 36:
        _binary_exact_chunk(
            x_ptr, y_ptr, out_ptr, alpha_val, 395520, 2, OP_TYPE
        )
        _binary_exact_chunk(
            x_ptr, y_ptr, out_ptr, alpha_val, 395522, 1, OP_TYPE
        )


@libentry()
@triton.jit
def binary_tensor_524288_exact_48core_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    alpha_val,
    OP_TYPE: tl.constexpr,
):
    pid = tle.program_id(0)
    _binary_exact_chunk(
        x_ptr,
        y_ptr,
        out_ptr,
        alpha_val,
        pid * 8192,
        8192,
        OP_TYPE,
    )
    if pid < 32:
        _binary_exact_chunk(
            x_ptr,
            y_ptr,
            out_ptr,
            alpha_val,
            393216 + pid * 4096,
            4096,
            OP_TYPE,
        )


@libentry()
@triton.jit
def binary_tensor_1048576_exact_48core_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    alpha_val,
    OP_TYPE: tl.constexpr,
):
    pid = tle.program_id(0)
    _binary_exact_chunk(
        x_ptr,
        y_ptr,
        out_ptr,
        alpha_val,
        pid * 8192,
        8192,
        OP_TYPE,
    )
    _binary_exact_chunk(
        x_ptr,
        y_ptr,
        out_ptr,
        alpha_val,
        393216 + pid * 8192,
        8192,
        OP_TYPE,
    )
    _binary_exact_chunk(
        x_ptr,
        y_ptr,
        out_ptr,
        alpha_val,
        786432 + pid * 4096,
        4096,
        OP_TYPE,
    )
    if pid < 32:
        _binary_exact_chunk(
            x_ptr,
            y_ptr,
            out_ptr,
            alpha_val,
            983040 + pid * 2048,
            2048,
            OP_TYPE,
        )


@libentry()
@triton.jit
def binary_tensor_395523_tail4096_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    alpha_val,
    OP_TYPE: tl.constexpr,
):
    pid = tle.program_id(0)
    main_offsets = pid * 8192 + tl.arange(0, 8192)
    x = tl.load(x_ptr + main_offsets)
    y = tl.load(y_ptr + main_offsets)
    result = _binary_result(x, y, alpha_val, OP_TYPE)
    tl.store(
        out_ptr + main_offsets,
        result.to(out_ptr.dtype.element_ty),
    )

    if pid == 0:
        tail_offsets = 393216 + tl.arange(0, 4096)
        tail_mask = tail_offsets < 395523
        tail_x = tl.load(x_ptr + tail_offsets, mask=tail_mask)
        tail_y = tl.load(y_ptr + tail_offsets, mask=tail_mask)
        tail_result = _binary_result(tail_x, tail_y, alpha_val, OP_TYPE)
        tl.store(
            out_ptr + tail_offsets,
            tail_result.to(out_ptr.dtype.element_ty),
            mask=tail_mask,
        )


@libentry()
@triton.jit
def binary_tensor_tiled_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    alpha_val,
    N_ELEMENTS: tl.constexpr,
    OP_TYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tle.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    result = _binary_result(x, y, alpha_val, OP_TYPE)
    tl.store(
        out_ptr + offsets,
        result.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@libentry()
@triton.jit
def binary_tensor_aligned_core_loop_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    alpha_val,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    OP_TYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    first_block = tle.program_id(0) * BLOCKS_PER_PROGRAM
    for local_block in range(0, BLOCKS_PER_PROGRAM):
        offsets = (first_block + local_block) * BLOCK_SIZE + tl.arange(
            0, BLOCK_SIZE
        )
        x = tl.load(x_ptr + offsets)
        y = tl.load(y_ptr + offsets)
        result = _binary_result(x, y, alpha_val, OP_TYPE)
        tl.store(out_ptr + offsets, result.to(out_ptr.dtype.element_ty))


@libentry()
@triton.jit
def binary_tensor_core_loop_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    alpha_val,
    N_ELEMENTS: tl.constexpr,
    OP_TYPE: tl.constexpr,
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
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        result = _binary_result(x, y, alpha_val, OP_TYPE)
        tl.store(
            out_ptr + offsets,
            result.to(out_ptr.dtype.element_ty),
            mask=mask,
        )


__all__: list[str] = [
    "add_tensor_aligned_core_loop_kernel",
    "add_tensor_core_loop_kernel",
    "add_tensor_1048576_exact_48core_kernel",
    "binary_tensor_aligned_core_loop_kernel",
    "binary_tensor_1000_exact_kernel",
    "binary_tensor_176085_exact_split_kernel",
    "binary_tensor_293475_exact_split_kernel",
    "binary_tensor_395523_balanced_tail_kernel",
    "binary_tensor_395523_tail4096_kernel",
    "binary_tensor_524288_exact_48core_kernel",
    "binary_tensor_1048576_exact_48core_kernel",
    "binary_tensor_balanced_chunks_kernel",
    "binary_tensor_tiled_kernel",
    "binary_tensor_core_loop_kernel",
    "can_use_aligned_core_loop",
    "get_add_block_size",
    "get_dense_binary_block_size",
    "get_vector_core_count",
    "launch_dense_binary",
    "make_core_loop_grid",
    "minimum_propagating_395523_exact_kernel",
    "minimum_propagating_aligned_core_loop_kernel",
    "prepare_dense_binary",
    "sub_alpha_one_aligned_core_loop_kernel",
    "sub_alpha_one_1048576_exact_48core_kernel",
    "sub_alpha_one_multibuffer_core_loop_kernel",
]
