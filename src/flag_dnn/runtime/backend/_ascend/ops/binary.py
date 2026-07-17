from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Mapping

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
    "make_core_loop_grid",
]
