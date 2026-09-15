# Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0
"""FP8 GEMM with tensor descales or OCP MXFP8 contraction-block scales."""
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice


@tr.jit
def fp8_matmul_kernel(
    a_ptr,
    b_ptr,
    sa_ptr,
    sb_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SCALE_MODE: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    CM: tl.constexpr,
    CN: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SBK: tl.constexpr,
    SBN: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    D4: tl.constexpr,
    D5: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    A2: tl.constexpr,
    A3: tl.constexpr,
    A4: tl.constexpr,
    A5: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    B3: tl.constexpr,
    B4: tl.constexpr,
    B5: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    C4: tl.constexpr,
    C5: tl.constexpr,
    SA0: tl.constexpr,
    SA1: tl.constexpr,
    SA2: tl.constexpr,
    SA3: tl.constexpr,
    SA4: tl.constexpr,
    SA5: tl.constexpr,
    SB0: tl.constexpr,
    SB1: tl.constexpr,
    SB2: tl.constexpr,
    SB3: tl.constexpr,
    SB4: tl.constexpr,
    SB5: tl.constexpr,
    BLOCK_M: tl.constexpr = 16,
    BLOCK_N: tl.constexpr = 32,
    BLOCK_K: tl.constexpr = 32,
):
    tl.static_assert(SCALE_MODE != 2 or BLOCK_K == 32)
    tile = tl.program_id(0).to(tl.int64)
    tiles_m: tl.constexpr = tr.cdiv(M, BLOCK_M)
    tiles_n: tl.constexpr = tr.cdiv(N, BLOCK_N)
    batch = tile // (tiles_m * tiles_n)
    row = tile // tiles_n % tiles_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tile % tiles_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ao = tl.full((), 0, tl.int64)
    bo = tl.full((), 0, tl.int64)
    co = tl.full((), 0, tl.int64)
    sao = tl.full((), 0, tl.int64)
    sbo = tl.full((), 0, tl.int64)
    coordinate = batch % D5
    batch = batch // D5
    ao += coordinate * A5
    bo += coordinate * B5
    co += coordinate * C5
    sao += coordinate * SA5
    sbo += coordinate * SB5
    coordinate = batch % D4
    batch = batch // D4
    ao += coordinate * A4
    bo += coordinate * B4
    co += coordinate * C4
    sao += coordinate * SA4
    sbo += coordinate * SB4
    coordinate = batch % D3
    batch = batch // D3
    ao += coordinate * A3
    bo += coordinate * B3
    co += coordinate * C3
    sao += coordinate * SA3
    sbo += coordinate * SB3
    coordinate = batch % D2
    batch = batch // D2
    ao += coordinate * A2
    bo += coordinate * B2
    co += coordinate * C2
    sao += coordinate * SA2
    sbo += coordinate * SB2
    coordinate = batch % D1
    batch = batch // D1
    ao += coordinate * A1
    bo += coordinate * B1
    co += coordinate * C1
    sao += coordinate * SA1
    sbo += coordinate * SB1
    coordinate = batch % D0
    batch = batch // D0
    ao += coordinate * A0
    bo += coordinate * B0
    co += coordinate * C0
    sao += coordinate * SA0
    sbo += coordinate * SB0
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start in range(tr.cdiv(K, BLOCK_K)):
        reduction = start * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_ptr + ao + row[:, None] * AM + reduction[None, :] * AK,
            (row[:, None] < M) & (reduction[None, :] < K),
            0.0,
        )
        b = tl.load(
            b_ptr + bo + reduction[:, None] * BK + col[None, :] * BN,
            (reduction[:, None] < K) & (col[None, :] < N),
            0.0,
        )
        if SCALE_MODE == 2:
            sa = tl.load(
                sa_ptr + sao + row * SAM + start * SAK, row < M, 127
            ).to(tl.int32)
            sb = tl.load(
                sb_ptr + sbo + start * SBK + col * SBN, col < N, 127
            ).to(tl.int32)
            partial = tl.dot(a, b, out_dtype=tl.float32)
            # Combine exponents before scaling. Sequential multiplications can
            # overflow or underflow despite a representable final product.
            scaled = libdevice.ldexp(partial, sa[:, None] + sb[None, :] - 254)
            accumulator += tl.where(
                (sa[:, None] == 255) | (sb[None, :] == 255),
                float("nan"),
                scaled,
            )
        else:
            accumulator = tl.dot(a, b, accumulator)
    if SCALE_MODE == 1:
        # Preserve a finite result when the product of the FP32 descales alone
        # would overflow. Dot accumulation and the stored result remain FP32.
        sa = tl.load(sa_ptr).to(tl.float64)
        sb = tl.load(sb_ptr).to(tl.float64)
        accumulator = (accumulator.to(tl.float64) * sa * sb).to(tl.float32)
    tl.store(
        c_ptr + co + row[:, None] * CM + col[None, :] * CN,
        accumulator,
        (row[:, None] < M) & (col[None, :] < N),
    )
