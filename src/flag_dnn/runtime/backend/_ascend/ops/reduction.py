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

"""Ascend-only prepared kernels for small-axis reductions."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import triton
import triton.language as tl

from flag_dnn.utils import triton_lang_extension as tle
from flag_dnn.utils.libentry import libentry

_SUPPORTED_MODES = {"ADD", "AVG", "MUL"}


def get_reduction_rows_block_size(rows: int) -> int:
    if rows <= 128:
        return max(16, triton.next_power_of_2(rows))
    return 128


@libentry()
@triton.jit
def reduction_rows_kernel(
    input_ptr,
    output_ptr,
    ROWS: tl.constexpr,
    REDUCED: tl.constexpr,
    INNER: tl.constexpr,
    STRIDE_OUTER: tl.constexpr,
    STRIDE_REDUCED: tl.constexpr,
    STRIDE_INNER: tl.constexpr,
    OP: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    row_offsets = tle.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    mask = row_offsets < ROWS
    outer_offsets = row_offsets // INNER
    inner_offsets = row_offsets % INNER
    base_offsets = outer_offsets * STRIDE_OUTER + inner_offsets * STRIDE_INNER

    if OP == "MUL":
        accumulator = tl.full((BLOCK_ROWS,), 1.0, dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)

    for reduction_index in range(0, REDUCED):
        value = tl.load(
            input_ptr + base_offsets + reduction_index * STRIDE_REDUCED,
            mask=mask,
            other=1.0 if OP == "MUL" else 0.0,
        ).to(tl.float32)
        if OP == "MUL":
            accumulator *= value
        else:
            accumulator += value

    if OP == "AVG":
        accumulator /= REDUCED
    tl.store(
        output_ptr + row_offsets,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


@libentry()
@triton.jit
def reduction_outer_inner_kernel(
    input_ptr,
    output_ptr,
    OUTER: tl.constexpr,
    REDUCED: tl.constexpr,
    INNER: tl.constexpr,
    STRIDE_OUTER: tl.constexpr,
    STRIDE_REDUCED: tl.constexpr,
    STRIDE_INNER: tl.constexpr,
    OP: tl.constexpr,
    BLOCK_INNER: tl.constexpr,
):
    inner_offsets = tle.program_id(0) * BLOCK_INNER + tl.arange(0, BLOCK_INNER)
    outer_index = tle.program_id(1)
    mask = inner_offsets < INNER
    base_offsets = outer_index * STRIDE_OUTER + inner_offsets * STRIDE_INNER
    accumulator = tl.zeros((BLOCK_INNER,), dtype=tl.float32)
    for reduction_index in range(0, REDUCED):
        accumulator += tl.load(
            input_ptr + base_offsets + reduction_index * STRIDE_REDUCED,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
    if OP == "AVG":
        accumulator /= REDUCED
    tl.store(
        output_ptr + outer_index * INNER + inner_offsets,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


def prepare_dense_reduction(
    *,
    mode: str,
    input_spec: Any,
    input_checks: Any,
    output_factory: Any,
    default_run_fn: Any,
    rows: int,
    reduced: int,
    inner: int,
    stride_outer: int,
    stride_reduced: int,
    stride_inner: int,
    validate_inputs: bool,
) -> Optional[Any]:
    if (
        mode not in _SUPPORTED_MODES
        or input_spec.dtype not in ("float16", "bfloat16", "float32")
        or rows <= 0
        or reduced <= 0
        or reduced > 64
    ):
        return None

    from flag_dnn.graph.prepared import (
        PreparedSingleKernelRunSpec,
        PreparedSingleKernelSpec,
        make_single_kernel_run_fn,
    )

    def runtime_args(inputs: Sequence[Any], output: Any) -> tuple[Any, ...]:
        return inputs[0], output

    static_grid: tuple[Any, ...]
    constexpr_kwargs: dict[str, Any]
    cached_args: tuple[Any, ...]
    if mode in {"ADD", "AVG"}:
        outer = rows // inner
        block_inner = min(256, triton.next_power_of_2(inner))
        static_grid = (triton.cdiv(inner, block_inner), outer)
        kernel = reduction_outer_inner_kernel
        constexpr_kwargs = {
            "OUTER": outer,
            "REDUCED": reduced,
            "INNER": inner,
            "STRIDE_OUTER": stride_outer,
            "STRIDE_REDUCED": stride_reduced,
            "STRIDE_INNER": stride_inner,
            "OP": mode,
            "BLOCK_INNER": block_inner,
            "num_warps": 4,
            "num_stages": 1,
        }
        cached_args = (
            outer,
            reduced,
            inner,
            stride_outer,
            stride_reduced,
            stride_inner,
            mode,
            block_inner,
        )
    else:
        block_rows = get_reduction_rows_block_size(rows)
        static_grid = (triton.cdiv(rows, block_rows),)
        kernel = reduction_rows_kernel
        constexpr_kwargs = {
            "ROWS": rows,
            "REDUCED": reduced,
            "INNER": inner,
            "STRIDE_OUTER": stride_outer,
            "STRIDE_REDUCED": stride_reduced,
            "STRIDE_INNER": stride_inner,
            "OP": mode,
            "BLOCK_ROWS": block_rows,
            "num_warps": 4,
            "num_stages": 1,
        }
        cached_args = (
            rows,
            reduced,
            inner,
            stride_outer,
            stride_reduced,
            stride_inner,
            mode,
            block_rows,
        )

    def grid(_meta: dict[str, Any]) -> tuple[int, ...]:
        return static_grid

    def build_cached_call(
        constexprs: dict[str, Any],
    ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        del constexprs
        return (*static_grid, *(1,) * (3 - len(static_grid))), cached_args

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
            validate_inputs=validate_inputs,
        ),
        default_run_fn,
    )


__all__ = (
    "get_reduction_rows_block_size",
    "prepare_dense_reduction",
    "reduction_outer_inner_kernel",
    "reduction_rows_kernel",
)
