from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from flag_dnn.runtime import torch_device_fn

_DEFAULT_NUM_THREADS_BY_N = {
    2048: 128,
    4096: 256,
    7168: 128,
    8192: 512,
    16384: 1024,
    32768: 512,
}
_RPC_CANDIDATES = (2, 4, 8)
_TARGET_MIN_CTAS = 148
_HADAMARD_BLOCK = 16


def _best_num_threads(n: int) -> Optional[int]:
    for num_threads in (1024, 512, 256, 128, 64):
        if n % num_threads != 0:
            continue
        ept = n // num_threads
        if ept >= 8 and ept % 8 == 0:
            return num_threads
    return None


def _pick_rows_per_cta(m: int) -> int:
    for rows_per_cta in reversed(_RPC_CANDIDATES):
        if m % rows_per_cta != 0:
            continue
        if m // rows_per_cta >= _TARGET_MIN_CTAS:
            return rows_per_cta
    return _RPC_CANDIDATES[0]


@triton.jit
def _hadamard_sign(cols, k: tl.constexpr):
    bits = cols & k
    parity = (
        (bits & 1) ^ ((bits >> 1) & 1) ^ ((bits >> 2) & 1) ^ ((bits >> 3) & 1)
    )
    return tl.where(parity == 0, 1.0, -1.0)


@triton.jit
def _rmsnorm_rht_rows_kernel(
    x_ptr,
    w_ptr,
    o_ptr,
    row_amax_ptr,
    eps,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    x_row = x_ptr + row * N
    o_row = o_ptr + row * N

    sum_squares = 0.0
    for start in range(0, N, BLOCK_N):
        offsets = start + cols
        mask = offsets < N
        x = tl.load(x_row + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_squares += tl.sum(x * x, axis=0)

    rrms = tl.rsqrt(sum_squares / N + eps)
    max_abs = 0.0

    for start in range(0, N, BLOCK_N):
        offsets = start + cols
        mask = offsets < N
        had_cols = offsets & 15
        group_base = offsets - had_cols
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k in tl.static_range(0, 16):
            source_cols = group_base + k
            x = tl.load(x_row + source_cols, mask=mask, other=0.0).to(
                tl.float32
            )
            w = tl.load(w_ptr + source_cols, mask=mask, other=0.0).to(
                tl.float32
            )
            sign = _hadamard_sign(had_cols, k)
            acc += x * rrms * w * sign

        out = acc * 0.25
        max_abs = tl.maximum(max_abs, tl.max(tl.abs(out), axis=0))
        tl.store(o_row + offsets, out.to(o_ptr.dtype.element_ty), mask=mask)

    tl.store(row_amax_ptr + row, max_abs)


@triton.jit
def _rows_to_cta_amax_kernel(
    row_amax_ptr,
    amax_ptr,
    ROWS_PER_CTA: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    cta = tl.program_id(0)
    offsets = cta * ROWS_PER_CTA + tl.arange(0, BLOCK_R)
    mask = tl.arange(0, BLOCK_R) < ROWS_PER_CTA
    values = tl.load(row_amax_ptr + offsets, mask=mask, other=0.0)
    tl.store(amax_ptr + cta, tl.max(values, axis=0))


def _validate_inputs(
    x_tensor: torch.Tensor,
    w_tensor: torch.Tensor,
    rows_per_cta: int,
    num_threads: int,
) -> tuple[int, int]:
    if x_tensor.dim() != 2:
        raise ValueError("x_tensor must be 2D after optional trailing squeeze")
    if w_tensor.dim() != 1:
        raise ValueError("w_tensor must be 1D after optional trailing squeeze")
    if x_tensor.dtype != torch.bfloat16:
        raise ValueError("x_tensor must be torch.bfloat16")
    if w_tensor.dtype != torch.bfloat16:
        raise ValueError("w_tensor must be torch.bfloat16")
    if x_tensor.device != w_tensor.device:
        raise ValueError("x_tensor and w_tensor must be on the same device")
    if not x_tensor.is_contiguous():
        raise ValueError("x_tensor must be row-major contiguous")
    if not w_tensor.is_contiguous():
        raise ValueError("w_tensor must be contiguous")

    m, n = int(x_tensor.shape[0]), int(x_tensor.shape[1])
    if int(w_tensor.shape[0]) != n:
        raise ValueError(
            "w_tensor length must match x_tensor hidden dimension,"
            f"got {w_tensor.shape[0]} and {n}"
        )
    if n % _HADAMARD_BLOCK != 0:
        raise ValueError(f"N must be divisible by 16, got {n}")
    if num_threads <= 0 or num_threads % 32 != 0 or num_threads > 1024:
        raise ValueError(f"invalid num_threads={num_threads}")
    if n % num_threads != 0:
        raise ValueError(
            f"N={n} must be divisible by num_threads={num_threads}"
        )
    ept = n // num_threads
    if ept < 8 or ept % 8 != 0:
        raise ValueError(f"EPT={ept} must be >= 8 and divisible by 8")
    if rows_per_cta <= 0 or m % rows_per_cta != 0:
        raise ValueError(
            f"M must be divisible by rows_per_cta, got M={m},"
            "rows_per_cta={rows_per_cta}"
        )
    return m, n


def rmsnorm_rht_amax_wrapper_sm100(
    x_tensor: torch.Tensor,
    w_tensor: torch.Tensor,
    eps: float = 1e-5,
    num_threads: Optional[int] = None,
    rows_per_cta: Optional[int] = None,
    current_stream=None,
):
    del current_stream
    if x_tensor.ndim == 3 and x_tensor.shape[-1] == 1:
        x_tensor = x_tensor.squeeze(-1)
    if w_tensor.ndim == 2 and w_tensor.shape[-1] == 1:
        w_tensor = w_tensor.squeeze(-1)

    if x_tensor.dim() != 2:
        raise ValueError("x_tensor must be 2D after optional trailing squeeze")
    m, n = int(x_tensor.shape[0]), int(x_tensor.shape[1])
    resolved_num_threads = (
        num_threads
        if num_threads is not None
        else _DEFAULT_NUM_THREADS_BY_N.get(n, _best_num_threads(n))
    )
    if resolved_num_threads is None:
        raise ValueError(f"No valid num_threads found for N={n}")
    resolved_rows_per_cta = (
        rows_per_cta if rows_per_cta is not None else _pick_rows_per_cta(m)
    )

    m, n = _validate_inputs(
        x_tensor, w_tensor, resolved_rows_per_cta, resolved_num_threads
    )
    o_tensor = torch.empty_like(x_tensor)
    row_amax = torch.empty((m,), dtype=torch.float32, device=x_tensor.device)
    amax_tensor = torch.empty(
        (m // resolved_rows_per_cta,),
        dtype=torch.float32,
        device=x_tensor.device,
    )

    block_n = min(max(triton.next_power_of_2(n), 16), 1024)
    with torch_device_fn.device(x_tensor.device):
        _rmsnorm_rht_rows_kernel[(m,)](
            x_tensor,
            w_tensor,
            o_tensor,
            row_amax,
            float(eps),
            N=n,
            BLOCK_N=block_n,
            num_warps=8,
        )
        block_r = triton.next_power_of_2(resolved_rows_per_cta)
        _rows_to_cta_amax_kernel[(m // resolved_rows_per_cta,)](
            row_amax,
            amax_tensor,
            ROWS_PER_CTA=resolved_rows_per_cta,
            BLOCK_R=block_r,
            num_warps=1,
        )

    return {"o_tensor": o_tensor, "amax_tensor": amax_tensor}
