# Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0
"""Deterministic grouped GEMM with device-resident expert routing."""
import triton as tr
import triton.language as tl


@tr.jit
def moe_matmul_kernel(
    token_ptr,
    matrix_ptr,
    offsets_ptr,
    index_ptr,
    ks_ptr,
    output_ptr,
    EXPERTS: tl.constexpr,
    TOKENS: tl.constexpr,
    ROUTED: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    MODE: tl.constexpr,
    TOP_K: tl.constexpr,
    TM: tl.constexpr,
    TK: tl.constexpr,
    ME: tl.constexpr,
    MK: tl.constexpr,
    MN: tl.constexpr,
    OE: tl.constexpr,
    OM: tl.constexpr,
    ON: tl.constexpr,
    OS: tl.constexpr,
    IS: tl.constexpr,
    KS: tl.constexpr,
    BACKWARD: tl.constexpr,
):
    tile = tl.program_id(0).to(tl.int64)
    tiles_n: tl.constexpr = tr.cdiv(N, 32)
    tiles_m: tl.constexpr = tr.cdiv(K if BACKWARD else ROUTED, 16)
    expert = tile // (tiles_m * tiles_n)
    row = tile // tiles_n % tiles_m * 16 + tl.arange(0, 16)
    col = tile % tiles_n * 32 + tl.arange(0, 32)
    # Guard every address even for invalid routing metadata. Valid
    # callers supply
    # monotone offsets beginning at zero and a permutation for scatter slots.
    begin = tl.minimum(
        tl.maximum(tl.load(offsets_ptr + expert * OS).to(tl.int64), 0), ROUTED
    )
    end = tl.minimum(
        tl.maximum(
            tl.load(
                offsets_ptr + (expert + 1) * OS, expert + 1 < EXPERTS, ROUTED
            ).to(tl.int64),
            begin,
        ),
        ROUTED,
    )
    if BACKWARD:
        accumulator = tl.zeros((16, 32), tl.float32)
        for start in range(begin, end, 32):
            reduction = start + tl.arange(0, 32)
            x = tl.load(
                token_ptr + reduction[None, :] * TM + row[:, None] * TK,
                (reduction[None, :] < end) & (row[:, None] < K),
                0.0,
            )
            dy = tl.load(
                matrix_ptr + reduction[:, None] * MK + col[None, :] * MN,
                (reduction[:, None] < end) & (col[None, :] < N),
                0.0,
            )
            accumulator = tl.dot(x, dy, accumulator)
        tl.store(
            output_ptr + expert * OE + row[:, None] * OM + col[None, :] * ON,
            accumulator,
            (row[:, None] < K) & (col[None, :] < N),
        )
    elif tile // tiles_n % tiles_m * 16 < end - begin:
        slot = begin + row
        valid = slot < end
        if MODE == 1:
            source = tl.load(index_ptr + slot * IS, valid, -1).to(tl.int64)
            valid &= (source >= 0) & (source < TOKENS)
        else:
            source = slot
        accumulator = tl.zeros((16, 32), tl.float32)
        for start in range(tr.cdiv(K, 32)):
            reduction = start * 32 + tl.arange(0, 32)
            x = tl.load(
                token_ptr + source[:, None] * TM + reduction[None, :] * TK,
                valid[:, None] & (reduction[None, :] < K),
                0.0,
            )
            w = tl.load(
                matrix_ptr
                + expert * ME
                + reduction[:, None] * MK
                + col[None, :] * MN,
                (reduction[:, None] < K) & (col[None, :] < N),
                0.0,
            )
            accumulator = tl.dot(x, w, accumulator)
        if MODE == 2:
            token = tl.load(index_ptr + slot * IS, valid, -1).to(tl.int64)
            rank = tl.load(ks_ptr + slot * KS, valid, -1).to(tl.int64)
            destination = token * TOP_K + rank
            valid &= (
                (token >= 0)
                & (token < ROUTED // TOP_K)
                & (rank >= 0)
                & (rank < TOP_K)
            )
        else:
            destination = slot
        tl.store(
            output_ptr + destination[:, None] * OM + col[None, :] * ON,
            accumulator,
            valid[:, None] & (col[None, :] < N),
        )
