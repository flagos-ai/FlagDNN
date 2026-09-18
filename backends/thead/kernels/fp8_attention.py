# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""PPU FP8 attention using an explicit byte storage ABI.

PPU cc80 supports the E4M3 bias-15 conversion format. Its finite byte codes
represent E4M3 bias-7 values divided by 256, so a power-of-two adjustment gives
an exact storage conversion without requiring unsupported native fp8e4nv.
One program owns all heads and amax outputs, including every Graph replay.
"""
import triton
import triton.language as tl


@triton.jit
def fp8_decode_bytes(values, E4: tl.constexpr):
    if E4:
        return (
            values.to(tl.float8e4b15, bitcast=True)
            .to(tl.float16)
            .to(tl.float32)
            * 256.0
        )
    return values.to(tl.float8e5, bitcast=True).to(tl.float16).to(tl.float32)


@triton.jit
def fp8_encode_bytes(values, E4: tl.constexpr):
    if E4:
        bounded = tl.minimum(tl.maximum(values, -448.0), 448.0)
        return (
            (bounded * (1.0 / 256.0))
            .to(tl.float8e4b15)
            .to(tl.int8, bitcast=True)
        )
    bounded = tl.minimum(tl.maximum(values, -57344.0), 57344.0)
    return bounded.to(tl.float8e5).to(tl.int8, bitcast=True)


@triton.jit
def fp8_load_tile(
    pointer,
    head,
    start,
    ROWS: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    E4: tl.constexpr,
):
    rows = start + tl.arange(0, BLOCK)
    features = tl.arange(0, D)
    raw = tl.load(
        pointer + head * ROWS * D + rows[:, None] * D + features[None, :],
        rows[:, None] < ROWS,
        other=0,
    )
    return fp8_decode_bytes(raw, E4).to(tl.float16)


@triton.jit
def fp8_sdpa_forward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    descale_q_ptr,
    descale_k_ptr,
    descale_v_ptr,
    descale_s_ptr,
    scale_s_ptr,
    scale_o_ptr,
    o_ptr,
    stats_ptr,
    amax_s_ptr,
    amax_o_ptr,
    BATCH: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    D: tl.constexpr,
    E4: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    STATS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    scale_qk = tl.load(descale_q_ptr) * tl.load(descale_k_ptr) * SCALE
    scale_sv = tl.load(descale_s_ptr) * tl.load(descale_v_ptr)
    scale_s = tl.load(scale_s_ptr)
    scale_o = tl.load(scale_o_ptr)
    features = tl.arange(0, D)
    amax_s = tl.full((), 0.0, tl.float32)
    amax_o = tl.full((), 0.0, tl.float32)
    for head in range(BATCH * HQ):
        kv_head = (head // HQ) * HK + (head % HQ) // (HQ // HK)
        for row_start in range(0, SQ, BLOCK_M):
            rows = row_start + tl.arange(0, BLOCK_M)
            q = fp8_load_tile(q_ptr, head, row_start, SQ, D, BLOCK_M, E4)
            maximum = tl.full((BLOCK_M,), float("-inf"), tl.float32)
            denominator = tl.zeros((BLOCK_M,), tl.float32)
            # Quantization is relative to the final row maximum, so first
            # compute the softmax normalizer and then accumulate quantized P*V.
            for column_start in range(0, SK, BLOCK_N):
                columns = column_start + tl.arange(0, BLOCK_N)
                k = fp8_load_tile(
                    k_ptr, kv_head, column_start, SK, D, BLOCK_N, E4
                )
                logits = tl.dot(q, tl.trans(k)).to(tl.float32) * scale_qk
                visible = columns[None, :] < SK
                if CAUSAL:
                    visible = visible & (columns[None, :] <= rows[:, None])
                logits = tl.where(visible, logits, float("-inf"))
                next_maximum = tl.maximum(maximum, tl.max(logits, 1))
                denominator = denominator * tl.exp(
                    maximum - next_maximum
                ) + tl.sum(tl.exp(logits - next_maximum[:, None]), 1)
                maximum = next_maximum
            output = tl.zeros((BLOCK_M, D), tl.float32)
            for column_start in range(0, SK, BLOCK_N):
                columns = column_start + tl.arange(0, BLOCK_N)
                k = fp8_load_tile(
                    k_ptr, kv_head, column_start, SK, D, BLOCK_N, E4
                )
                v = fp8_load_tile(
                    v_ptr, kv_head, column_start, SK, D, BLOCK_N, E4
                )
                logits = tl.dot(q, tl.trans(k)).to(tl.float32) * scale_qk
                visible = columns[None, :] < SK
                if CAUSAL:
                    visible = visible & (columns[None, :] <= rows[:, None])
                p = tl.where(visible, tl.exp(logits - maximum[:, None]), 0.0)
                quantized = fp8_decode_bytes(
                    fp8_encode_bytes(p * scale_s, E4), E4
                ).to(tl.float16)
                output += tl.dot(quantized, v).to(tl.float32)
            output *= scale_sv / denominator[:, None]
            amax_s = tl.maximum(
                amax_s, tl.max(tl.where(rows < SQ, 1.0 / denominator, 0.0), 0)
            )
            valid_output = rows[:, None] < SQ
            amax_o = tl.maximum(
                amax_o,
                tl.max(
                    tl.max(tl.where(valid_output, tl.abs(output), 0.0), 1), 0
                ),
            )
            tl.store(
                o_ptr + head * SQ * D + rows[:, None] * D + features[None, :],
                fp8_encode_bytes(output * scale_o, E4),
                valid_output,
            )
            if STATS:
                tl.store(
                    stats_ptr + head * SQ + rows,
                    maximum + tl.log(denominator),
                    rows < SQ,
                )
    tl.store(amax_s_ptr, amax_s)
    tl.store(amax_o_ptr, amax_o)


@triton.jit
def fp8_backward_tile(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    do_ptr,
    stats_ptr,
    head,
    kv_head,
    row_start,
    column_start,
    descale_q,
    descale_k,
    descale_v,
    descale_o,
    descale_do,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    D: tl.constexpr,
    E4: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = row_start + tl.arange(0, BLOCK_M)
    columns = column_start + tl.arange(0, BLOCK_N)
    q = fp8_load_tile(q_ptr, head, row_start, SQ, D, BLOCK_M, E4)
    k = fp8_load_tile(k_ptr, kv_head, column_start, SK, D, BLOCK_N, E4)
    v = fp8_load_tile(v_ptr, kv_head, column_start, SK, D, BLOCK_N, E4)
    o = fp8_load_tile(o_ptr, head, row_start, SQ, D, BLOCK_M, E4).to(
        tl.float32
    )
    do = fp8_load_tile(do_ptr, head, row_start, SQ, D, BLOCK_M, E4)
    stats = tl.load(stats_ptr + head * SQ + rows, rows < SQ, other=0.0)
    logits = tl.dot(q, tl.trans(k)).to(tl.float32) * (
        descale_q * descale_k * SCALE
    )
    visible = (rows[:, None] < SQ) & (columns[None, :] < SK)
    if CAUSAL:
        visible = visible & (columns[None, :] <= rows[:, None])
    p = tl.where(visible, tl.exp(logits - stats[:, None]), 0.0)
    dp = tl.dot(do, tl.trans(v)).to(tl.float32) * (descale_do * descale_v)
    delta = tl.sum(o * do.to(tl.float32), 1) * (descale_o * descale_do)
    ds = p * (dp - delta[:, None]) * SCALE
    return q, k, do, p, ds


@triton.jit
def fp8_sdpa_backward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    do_ptr,
    stats_ptr,
    descale_q_ptr,
    descale_k_ptr,
    descale_v_ptr,
    descale_o_ptr,
    descale_do_ptr,
    descale_s_ptr,
    descale_dp_ptr,
    scale_s_ptr,
    scale_dq_ptr,
    scale_dk_ptr,
    scale_dv_ptr,
    scale_dp_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    amax_dq_ptr,
    amax_dk_ptr,
    amax_dv_ptr,
    amax_dp_ptr,
    BATCH: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    D: tl.constexpr,
    E4: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    descale_q = tl.load(descale_q_ptr)
    descale_k = tl.load(descale_k_ptr)
    descale_v = tl.load(descale_v_ptr)
    descale_o = tl.load(descale_o_ptr)
    descale_do = tl.load(descale_do_ptr)
    descale_s = tl.load(descale_s_ptr)
    descale_dp = tl.load(descale_dp_ptr)
    scale_s = tl.load(scale_s_ptr)
    scale_dq = tl.load(scale_dq_ptr)
    scale_dk = tl.load(scale_dk_ptr)
    scale_dv = tl.load(scale_dv_ptr)
    scale_dp = tl.load(scale_dp_ptr)
    features = tl.arange(0, D)
    amax_dq = tl.full((), 0.0, tl.float32)
    amax_dk = tl.full((), 0.0, tl.float32)
    amax_dv = tl.full((), 0.0, tl.float32)
    amax_dp = tl.full((), 0.0, tl.float32)
    for kv_head in range(BATCH * HK):
        for group in range(HQ // HK):
            head = (kv_head // HK) * HQ + (kv_head % HK) * (HQ // HK) + group
            for row_start in range(0, SQ, BLOCK_M):
                rows = row_start + tl.arange(0, BLOCK_M)
                dq = tl.zeros((BLOCK_M, D), tl.float32)
                for column_start in range(0, SK, BLOCK_N):
                    q, k, do, p, ds = fp8_backward_tile(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        o_ptr,
                        do_ptr,
                        stats_ptr,
                        head,
                        kv_head,
                        row_start,
                        column_start,
                        descale_q,
                        descale_k,
                        descale_v,
                        descale_o,
                        descale_do,
                        SQ,
                        SK,
                        D,
                        E4,
                        SCALE,
                        CAUSAL,
                        BLOCK_M,
                        BLOCK_N,
                    )
                    amax_dp = tl.maximum(
                        amax_dp, tl.max(tl.max(tl.abs(ds), 1), 0)
                    )
                    ds_quantized = fp8_decode_bytes(
                        fp8_encode_bytes(ds * scale_dp, E4), E4
                    ).to(tl.float16)
                    dq += tl.dot(ds_quantized, k).to(tl.float32) * (
                        descale_dp * descale_k
                    )
                valid_q = rows[:, None] < SQ
                amax_dq = tl.maximum(
                    amax_dq,
                    tl.max(tl.max(tl.where(valid_q, tl.abs(dq), 0.0), 1), 0),
                )
                tl.store(
                    dq_ptr
                    + head * SQ * D
                    + rows[:, None] * D
                    + features[None, :],
                    fp8_encode_bytes(dq * scale_dq, E4),
                    valid_q,
                )
        for column_start in range(0, SK, BLOCK_N):
            columns = column_start + tl.arange(0, BLOCK_N)
            dk = tl.zeros((BLOCK_N, D), tl.float32)
            dv = tl.zeros((BLOCK_N, D), tl.float32)
            for group in range(HQ // HK):
                head = (
                    (kv_head // HK) * HQ + (kv_head % HK) * (HQ // HK) + group
                )
                for row_start in range(0, SQ, BLOCK_M):
                    q, k, do, p, ds = fp8_backward_tile(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        o_ptr,
                        do_ptr,
                        stats_ptr,
                        head,
                        kv_head,
                        row_start,
                        column_start,
                        descale_q,
                        descale_k,
                        descale_v,
                        descale_o,
                        descale_do,
                        SQ,
                        SK,
                        D,
                        E4,
                        SCALE,
                        CAUSAL,
                        BLOCK_M,
                        BLOCK_N,
                    )
                    ds_quantized = fp8_decode_bytes(
                        fp8_encode_bytes(ds * scale_dp, E4), E4
                    ).to(tl.float16)
                    p_quantized = fp8_decode_bytes(
                        fp8_encode_bytes(p * scale_s, E4), E4
                    ).to(tl.float16)
                    dk += tl.dot(tl.trans(ds_quantized), q).to(tl.float32) * (
                        descale_dp * descale_q
                    )
                    dv += tl.dot(tl.trans(p_quantized), do).to(tl.float32) * (
                        descale_s * descale_do
                    )
            valid_kv = columns[:, None] < SK
            amax_dk = tl.maximum(
                amax_dk,
                tl.max(tl.max(tl.where(valid_kv, tl.abs(dk), 0.0), 1), 0),
            )
            amax_dv = tl.maximum(
                amax_dv,
                tl.max(tl.max(tl.where(valid_kv, tl.abs(dv), 0.0), 1), 0),
            )
            tl.store(
                dk_ptr
                + kv_head * SK * D
                + columns[:, None] * D
                + features[None, :],
                fp8_encode_bytes(dk * scale_dk, E4),
                valid_kv,
            )
            tl.store(
                dv_ptr
                + kv_head * SK * D
                + columns[:, None] * D
                + features[None, :],
                fp8_encode_bytes(dv * scale_dv, E4),
                valid_kv,
            )
    tl.store(amax_dq_ptr, amax_dq)
    tl.store(amax_dk_ptr, amax_dk)
    tl.store(amax_dv_ptr, amax_dv)
    tl.store(amax_dp_ptr, amax_dp)
