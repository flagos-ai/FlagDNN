# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Platform-neutral Triton kernels for attention operations."""

from __future__ import annotations

import triton
import triton.language as tl

_sdpa_LOG2E_KERNEL_variant = tl.constexpr(1.4426950408889634)


@triton.jit
def _sdpa_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    k_base,
    v_base,
    bias_base,
    qk_scale,
    offs_m,
    offs_d,
    offs_dv,
    lo,
    hi,
    SQ,
    SKV,
    min_diag,
    max_diag,
    stride_kn,
    stride_kd,
    stride_vn,
    stride_vd,
    stride_bias_m,
    stride_bias_n,
    HEAD_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PADDED_D: tl.constexpr,
    PADDED_DV: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BANDED: tl.constexpr,
    MASKED: tl.constexpr,
):
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_mask = None
        if MASKED and PADDED_D:
            k_mask = (offs_d[:, None] < HEAD_DIM) & (offs_n[None, :] < SKV)
        elif MASKED:
            k_mask = offs_n[None, :] < SKV
        elif PADDED_D:
            k_mask = offs_d[:, None] < HEAD_DIM
        if k_mask is not None:
            k = tl.load(
                k_base
                + offs_d[:, None] * stride_kd
                + offs_n[None, :] * stride_kn,
                mask=k_mask,
                other=0.0,
            )
        else:
            k = tl.load(
                k_base
                + offs_d[:, None] * stride_kd
                + offs_n[None, :] * stride_kn
            )

        qk = tl.dot(q, k)
        score = qk.to(tl.float32) * qk_scale
        if HAS_BIAS:
            bias_mask = (offs_m[:, None] < SQ) & (offs_n[None, :] < SKV)
            bias_tile = tl.load(
                bias_base
                + offs_m[:, None] * stride_bias_m
                + offs_n[None, :] * stride_bias_n,
                mask=bias_mask,
                other=0.0,
            )
            score += bias_tile.to(tl.float32) * _sdpa_LOG2E_KERNEL_variant

        if MASKED:
            visible = offs_n[None, :] < SKV
            if BANDED:
                diag = offs_n[None, :] - offs_m[:, None]
                visible = visible & (diag >= min_diag) & (diag <= max_diag)
            score = tl.where(visible, score, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(score, 1))
        if MASKED:
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        else:
            m_safe = m_new
        p = tl.exp2(score - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v_mask = None
        if MASKED and PADDED_DV:
            v_mask = (offs_n[:, None] < SKV) & (offs_dv[None, :] < V_DIM)
        elif MASKED:
            v_mask = offs_n[:, None] < SKV
        elif PADDED_DV:
            v_mask = offs_dv[None, :] < V_DIM
        if v_mask is not None:
            v = tl.load(
                v_base
                + offs_n[:, None] * stride_vn
                + offs_dv[None, :] * stride_vd,
                mask=v_mask,
                other=0.0,
            )
        else:
            v = tl.load(
                v_base
                + offs_n[:, None] * stride_vn
                + offs_dv[None, :] * stride_vd
            )
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new
    return acc, l_i, m_i


@triton.jit
def _sdpa_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    bias_ptr,
    o_ptr,
    stats_ptr,
    qk_scale,
    HQ,
    SQ,
    SKV,
    q_per_k,
    q_per_v,
    min_diag,
    max_diag,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_bias_b,
    stride_bias_h,
    stride_bias_m,
    stride_bias_n,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_sb,
    stride_sh,
    stride_sm,
    HEAD_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    # ELEM_SIZE only feeds the autotune key so fp16/bf16 and fp32 inputs
    # never share one tuned tile config (their smem budgets differ).
    ELEM_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BANDED: tl.constexpr,
    GENERATE_STATS: tl.constexpr,
    REVERSE_CAUSAL: tl.constexpr,
):
    raw_pid_m = tl.program_id(0)
    if REVERSE_CAUSAL:
        pid_m = tl.cdiv(SQ, BLOCK_M) - 1 - raw_pid_m
    else:
        pid_m = raw_pid_m
    pid_bh = tl.program_id(1)
    off_b = pid_bh // HQ
    off_h = pid_bh % HQ
    off_kh = off_h // q_per_k
    off_vh = off_h // q_per_v

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_m = offs_m < SQ

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_kh * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_vh * stride_vh
    bias_base = bias_ptr + off_b * stride_bias_b + off_h * stride_bias_h

    PADDED_D: tl.constexpr = BLOCK_D != HEAD_DIM
    PADDED_DV: tl.constexpr = BLOCK_DV != V_DIM

    q_mask = mask_m[:, None]
    if PADDED_D:
        q_mask = q_mask & (offs_d[None, :] < HEAD_DIM)
    q = tl.load(
        q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=q_mask,
        other=0.0,
    )

    # Visible columns for row i satisfy min_diag <= j - i <= max_diag.
    # Derive the column range covered by this row tile and split it into
    # a fully visible interior plus masked boundary tiles.
    lo = tl.maximum(start_m + min_diag, 0)
    lo_block = (lo // BLOCK_N) * BLOCK_N
    hi = tl.minimum(start_m + BLOCK_M - 1 + max_diag + 1, SKV)
    hi = tl.maximum(hi, lo_block)

    full_lo = tl.maximum(start_m + BLOCK_M - 1 + min_diag, 0)
    full_lo_block = tl.cdiv(full_lo, BLOCK_N) * BLOCK_N
    full_hi = tl.minimum(start_m + max_diag + 1, SKV)
    full_hi_block = (full_hi // BLOCK_N) * BLOCK_N

    phase_a_end = tl.minimum(full_lo_block, hi)
    phase_b_end = tl.maximum(tl.minimum(full_hi_block, hi), phase_a_end)

    acc = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)

    if lo_block < phase_a_end:
        acc, l_i, m_i = _sdpa_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            k_base,
            v_base,
            bias_base,
            qk_scale,
            offs_m,
            offs_d,
            offs_dv,
            lo_block,
            phase_a_end,
            SQ,
            SKV,
            min_diag,
            max_diag,
            stride_kn,
            stride_kd,
            stride_vn,
            stride_vd,
            stride_bias_m,
            stride_bias_n,
            HEAD_DIM=HEAD_DIM,
            V_DIM=V_DIM,
            BLOCK_N=BLOCK_N,
            PADDED_D=PADDED_D,
            PADDED_DV=PADDED_DV,
            HAS_BIAS=HAS_BIAS,
            BANDED=BANDED,
            MASKED=True,
        )
    if phase_a_end < phase_b_end:
        acc, l_i, m_i = _sdpa_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            k_base,
            v_base,
            bias_base,
            qk_scale,
            offs_m,
            offs_d,
            offs_dv,
            phase_a_end,
            phase_b_end,
            SQ,
            SKV,
            min_diag,
            max_diag,
            stride_kn,
            stride_kd,
            stride_vn,
            stride_vd,
            stride_bias_m,
            stride_bias_n,
            HEAD_DIM=HEAD_DIM,
            V_DIM=V_DIM,
            BLOCK_N=BLOCK_N,
            PADDED_D=PADDED_D,
            PADDED_DV=PADDED_DV,
            HAS_BIAS=HAS_BIAS,
            BANDED=BANDED,
            MASKED=False,
        )
    if phase_b_end < hi:
        acc, l_i, m_i = _sdpa_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            k_base,
            v_base,
            bias_base,
            qk_scale,
            offs_m,
            offs_d,
            offs_dv,
            phase_b_end,
            hi,
            SQ,
            SKV,
            min_diag,
            max_diag,
            stride_kn,
            stride_kd,
            stride_vn,
            stride_vd,
            stride_bias_m,
            stride_bias_n,
            HEAD_DIM=HEAD_DIM,
            V_DIM=V_DIM,
            BLOCK_N=BLOCK_N,
            PADDED_D=PADDED_D,
            PADDED_DV=PADDED_DV,
            HAS_BIAS=HAS_BIAS,
            BANDED=BANDED,
            MASKED=True,
        )

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]

    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh
    o_mask = mask_m[:, None]
    if PADDED_DV:
        o_mask = o_mask & (offs_dv[None, :] < V_DIM)
    tl.store(
        o_base + offs_m[:, None] * stride_om + offs_dv[None, :] * stride_od,
        acc.to(o_ptr.dtype.element_ty),
        mask=o_mask,
    )

    if GENERATE_STATS:
        stats = m_i / _sdpa_LOG2E_KERNEL_variant + tl.log(l_safe)
        stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh
        tl.store(stats_base + offs_m * stride_sm, stats, mask=mask_m)


@triton.jit
def _sdpa_bwd_dq_dbias_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    bias_ptr,
    o_ptr,
    do_ptr,
    stats_ptr,
    delta_ptr,
    dq_ptr,
    dbias_ptr,
    attn_scale,
    HQ: tl.constexpr,
    SQ,
    SKV,
    q_per_k: tl.constexpr,
    q_per_v: tl.constexpr,
    min_diag,
    max_diag,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_bias_b: tl.constexpr,
    stride_bias_h: tl.constexpr,
    stride_bias_m: tl.constexpr,
    stride_bias_n: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_od: tl.constexpr,
    stride_dob: tl.constexpr,
    stride_doh: tl.constexpr,
    stride_dom: tl.constexpr,
    stride_dod: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_sh: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_delta_b: tl.constexpr,
    stride_delta_h: tl.constexpr,
    stride_delta_m: tl.constexpr,
    stride_dqb: tl.constexpr,
    stride_dqh: tl.constexpr,
    stride_dqm: tl.constexpr,
    stride_dqd: tl.constexpr,
    stride_dbias_b: tl.constexpr,
    stride_dbias_h: tl.constexpr,
    stride_dbias_m: tl.constexpr,
    stride_dbias_n: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    DBIAS_BATCHES: tl.constexpr,
    DBIAS_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D_FULL: tl.constexpr,
    BLOCK_D_OUT: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    FULL_ATTENTION: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_DBIAS: tl.constexpr,
    DBIAS_REDUCE: tl.constexpr,
    BANDED: tl.constexpr,
    CAUSAL_TOP_LEFT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_bh = tl.program_id(2)
    off_b = pid_bh // HQ
    off_h = pid_bh % HQ
    off_kh = off_h // q_per_k
    off_vh = off_h // q_per_v

    start_m = pid_m * BLOCK_M
    start_d = pid_d * BLOCK_D_OUT
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d_full = tl.arange(0, BLOCK_D_FULL)
    offs_d = start_d + tl.arange(0, BLOCK_D_OUT)
    offs_dv = tl.arange(0, BLOCK_DV)

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_kh * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_vh * stride_vh
    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh
    do_base = do_ptr + off_b * stride_dob + off_h * stride_doh
    stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh
    delta_base = delta_ptr + off_b * stride_delta_b + off_h * stride_delta_h

    q_full = tl.load(
        q_base
        + offs_m[:, None] * stride_qm
        + offs_d_full[None, :] * stride_qd,
        mask=(offs_m[:, None] < SQ) & (offs_d_full[None, :] < HEAD_DIM),
        other=0.0,
    )
    do_tile = tl.load(
        do_base + offs_m[:, None] * stride_dom + offs_dv[None, :] * stride_dod,
        mask=(offs_m[:, None] < SQ) & (offs_dv[None, :] < V_DIM),
        other=0.0,
    )
    stats = tl.load(
        stats_base + offs_m * stride_sm,
        mask=offs_m < SQ,
        other=float("-inf"),
    ).to(tl.float32)
    o_tile = tl.load(
        o_base + offs_m[:, None] * stride_om + offs_dv[None, :] * stride_od,
        mask=(offs_m[:, None] < SQ) & (offs_dv[None, :] < V_DIM),
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o_tile * do_tile.to(tl.float32), axis=1)
    tl.store(
        delta_base + offs_m * stride_delta_m,
        delta,
        mask=offs_m < SQ,
    )

    dq = tl.zeros((BLOCK_M, BLOCK_D_OUT), dtype=tl.float32)

    loop_end_n = SKV
    if CAUSAL_TOP_LEFT:
        loop_end_n = tl.minimum(SKV, start_m + BLOCK_M)
    for start_n in tl.range(0, loop_end_n, BLOCK_N):
        cols = start_n + offs_n
        k_full = tl.load(
            k_base
            + cols[:, None] * stride_kn
            + offs_d_full[None, :] * stride_kd,
            mask=(cols[:, None] < SKV) & (offs_d_full[None, :] < HEAD_DIM),
            other=0.0,
        )
        score = tl.dot(q_full, tl.trans(k_full)).to(tl.float32) * attn_scale
        if HAS_BIAS:
            bias_tile = tl.load(
                bias_ptr
                + off_b * stride_bias_b
                + off_h * stride_bias_h
                + offs_m[:, None] * stride_bias_m
                + cols[None, :] * stride_bias_n,
                mask=(offs_m[:, None] < SQ) & (cols[None, :] < SKV),
                other=0.0,
            )
            score += bias_tile.to(tl.float32)

        if FULL_ATTENTION:
            p = tl.exp(score - stats[:, None])
        else:
            valid = (offs_m[:, None] < SQ) & (cols[None, :] < SKV)
            if BANDED:
                diag = cols[None, :] - offs_m[:, None]
                valid = valid & (diag >= min_diag) & (diag <= max_diag)
            p = tl.where(valid, tl.exp(score - stats[:, None]), 0.0)

        v_tile = tl.load(
            v_base + cols[:, None] * stride_vn + offs_dv[None, :] * stride_vd,
            mask=(cols[:, None] < SKV) & (offs_dv[None, :] < V_DIM),
            other=0.0,
        )
        dp = tl.dot(do_tile, tl.trans(v_tile)).to(tl.float32)
        ds = p * (dp - delta[:, None])

        k_out = tl.load(
            k_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=(cols[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )
        dq += tl.dot(ds.to(k_out.dtype), k_out)

        if HAS_DBIAS and pid_d == 0:
            dbias_b = 0 if DBIAS_BATCHES == 1 else off_b
            dbias_h = 0 if DBIAS_HEADS == 1 else off_h
            dbias_offsets = (
                dbias_ptr
                + dbias_b * stride_dbias_b
                + dbias_h * stride_dbias_h
                + offs_m[:, None] * stride_dbias_m
                + cols[None, :] * stride_dbias_n
            )
            dbias_mask = (offs_m[:, None] < SQ) & (cols[None, :] < SKV)
            if DBIAS_REDUCE:
                tl.atomic_add(
                    dbias_offsets,
                    ds.to(dbias_ptr.dtype.element_ty),
                    sem="relaxed",
                    mask=dbias_mask,
                )
            else:
                tl.store(
                    dbias_offsets,
                    ds.to(dbias_ptr.dtype.element_ty),
                    mask=dbias_mask,
                )

    tl.store(
        dq_ptr
        + off_b * stride_dqb
        + off_h * stride_dqh
        + offs_m[:, None] * stride_dqm
        + offs_d[None, :] * stride_dqd,
        (dq * attn_scale).to(dq_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM),
    )


@triton.jit
def _sdpa_bwd_dk_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    bias_ptr,
    do_ptr,
    stats_ptr,
    delta_ptr,
    dk_ptr,
    attn_scale,
    HKV: tl.constexpr,
    SQ,
    SKV,
    min_diag,
    max_diag,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_bias_b: tl.constexpr,
    stride_bias_h: tl.constexpr,
    stride_bias_m: tl.constexpr,
    stride_bias_n: tl.constexpr,
    stride_dob: tl.constexpr,
    stride_doh: tl.constexpr,
    stride_dom: tl.constexpr,
    stride_dod: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_sh: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_delta_b: tl.constexpr,
    stride_delta_h: tl.constexpr,
    stride_delta_m: tl.constexpr,
    stride_dkb: tl.constexpr,
    stride_dkh: tl.constexpr,
    stride_dkn: tl.constexpr,
    stride_dkd: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    Q_PER: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D_FULL: tl.constexpr,
    BLOCK_D_OUT: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    FULL_ATTENTION: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BANDED: tl.constexpr,
    CAUSAL_TOP_LEFT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_bh = tl.program_id(2)
    off_b = pid_bh // HKV
    off_kh = pid_bh % HKV

    start_n = pid_n * BLOCK_N
    start_d = pid_d * BLOCK_D_OUT
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d_full = tl.arange(0, BLOCK_D_FULL)
    offs_d = start_d + tl.arange(0, BLOCK_D_OUT)
    offs_dv = tl.arange(0, BLOCK_DV)

    k_base = k_ptr + off_b * stride_kb + off_kh * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_kh * stride_vh
    k_full = tl.load(
        k_base
        + offs_n[:, None] * stride_kn
        + offs_d_full[None, :] * stride_kd,
        mask=(offs_n[:, None] < SKV) & (offs_d_full[None, :] < HEAD_DIM),
        other=0.0,
    )
    v_full = tl.load(
        v_base + offs_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd,
        mask=(offs_n[:, None] < SKV) & (offs_dv[None, :] < V_DIM),
        other=0.0,
    )
    dk = tl.zeros((BLOCK_N, BLOCK_D_OUT), dtype=tl.float32)

    for group_idx in tl.static_range(0, Q_PER):
        off_h = off_kh * Q_PER + group_idx
        q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
        do_base = do_ptr + off_b * stride_dob + off_h * stride_doh
        stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh
        delta_base = (
            delta_ptr + off_b * stride_delta_b + off_h * stride_delta_h
        )
        loop_start_m = 0
        if CAUSAL_TOP_LEFT:
            loop_start_m = (start_n // BLOCK_M) * BLOCK_M
        for start_m in tl.range(loop_start_m, SQ, BLOCK_M):
            rows = start_m + offs_m
            q_full = tl.load(
                q_base
                + rows[:, None] * stride_qm
                + offs_d_full[None, :] * stride_qd,
                mask=(rows[:, None] < SQ) & (offs_d_full[None, :] < HEAD_DIM),
                other=0.0,
            )
            do_tile = tl.load(
                do_base
                + rows[:, None] * stride_dom
                + offs_dv[None, :] * stride_dod,
                mask=(rows[:, None] < SQ) & (offs_dv[None, :] < V_DIM),
                other=0.0,
            )
            stats = tl.load(
                stats_base + rows * stride_sm,
                mask=rows < SQ,
                other=float("-inf"),
            ).to(tl.float32)
            delta = tl.load(
                delta_base + rows * stride_delta_m,
                mask=rows < SQ,
                other=0.0,
            ).to(tl.float32)

            score = (
                tl.dot(q_full, tl.trans(k_full)).to(tl.float32) * attn_scale
            )
            if HAS_BIAS:
                bias_tile = tl.load(
                    bias_ptr
                    + off_b * stride_bias_b
                    + off_h * stride_bias_h
                    + rows[:, None] * stride_bias_m
                    + offs_n[None, :] * stride_bias_n,
                    mask=(rows[:, None] < SQ) & (offs_n[None, :] < SKV),
                    other=0.0,
                )
                score += bias_tile.to(tl.float32)

            if FULL_ATTENTION:
                p = tl.exp(score - stats[:, None])
            else:
                valid = (rows[:, None] < SQ) & (offs_n[None, :] < SKV)
                if BANDED:
                    diag = offs_n[None, :] - rows[:, None]
                    valid = valid & (diag >= min_diag) & (diag <= max_diag)
                p = tl.where(valid, tl.exp(score - stats[:, None]), 0.0)
            dp = tl.dot(do_tile, tl.trans(v_full)).to(tl.float32)
            ds = p * (dp - delta[:, None])

            q_out = tl.load(
                q_base
                + rows[:, None] * stride_qm
                + offs_d[None, :] * stride_qd,
                mask=(rows[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM),
                other=0.0,
            )
            dk += tl.dot(tl.trans(ds).to(q_out.dtype), q_out)

    tl.store(
        dk_ptr
        + off_b * stride_dkb
        + off_kh * stride_dkh
        + offs_n[:, None] * stride_dkn
        + offs_d[None, :] * stride_dkd,
        (dk * attn_scale).to(dk_ptr.dtype.element_ty),
        mask=(offs_n[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM),
    )


@triton.jit
def _sdpa_bwd_dkdv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    bias_ptr,
    do_ptr,
    stats_ptr,
    delta_ptr,
    dk_ptr,
    dv_ptr,
    attn_scale,
    HKV: tl.constexpr,
    SQ,
    SKV,
    min_diag,
    max_diag,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_bias_b: tl.constexpr,
    stride_bias_h: tl.constexpr,
    stride_bias_m: tl.constexpr,
    stride_bias_n: tl.constexpr,
    stride_dob: tl.constexpr,
    stride_doh: tl.constexpr,
    stride_dom: tl.constexpr,
    stride_dod: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_sh: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_delta_b: tl.constexpr,
    stride_delta_h: tl.constexpr,
    stride_delta_m: tl.constexpr,
    stride_dkb: tl.constexpr,
    stride_dkh: tl.constexpr,
    stride_dkn: tl.constexpr,
    stride_dkd: tl.constexpr,
    stride_dvb: tl.constexpr,
    stride_dvh: tl.constexpr,
    stride_dvn: tl.constexpr,
    stride_dvd: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    Q_PER: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D_FULL: tl.constexpr,
    BLOCK_D_OUT: tl.constexpr,
    FULL_ATTENTION: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BANDED: tl.constexpr,
    CAUSAL_TOP_LEFT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_bh = tl.program_id(2)
    off_b = pid_bh // HKV
    off_kh = pid_bh % HKV

    start_n = pid_n * BLOCK_N
    start_d = pid_d * BLOCK_D_OUT
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d_full = tl.arange(0, BLOCK_D_FULL)
    offs_d = start_d + tl.arange(0, BLOCK_D_OUT)

    k_base = k_ptr + off_b * stride_kb + off_kh * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_kh * stride_vh
    k_full = tl.load(
        k_base
        + offs_n[:, None] * stride_kn
        + offs_d_full[None, :] * stride_kd,
        mask=(offs_n[:, None] < SKV) & (offs_d_full[None, :] < HEAD_DIM),
        other=0.0,
    )
    v_full = tl.load(
        v_base
        + offs_n[:, None] * stride_vn
        + offs_d_full[None, :] * stride_vd,
        mask=(offs_n[:, None] < SKV) & (offs_d_full[None, :] < HEAD_DIM),
        other=0.0,
    )
    dk = tl.zeros((BLOCK_N, BLOCK_D_OUT), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, BLOCK_D_OUT), dtype=tl.float32)

    for group_idx in tl.static_range(0, Q_PER):
        off_h = off_kh * Q_PER + group_idx
        q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
        do_base = do_ptr + off_b * stride_dob + off_h * stride_doh
        stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh
        delta_base = (
            delta_ptr + off_b * stride_delta_b + off_h * stride_delta_h
        )
        loop_start_m = 0
        if CAUSAL_TOP_LEFT:
            loop_start_m = (start_n // BLOCK_M) * BLOCK_M
        for start_m in tl.range(loop_start_m, SQ, BLOCK_M):
            rows = start_m + offs_m
            q_full = tl.load(
                q_base
                + rows[:, None] * stride_qm
                + offs_d_full[None, :] * stride_qd,
                mask=(rows[:, None] < SQ) & (offs_d_full[None, :] < HEAD_DIM),
                other=0.0,
            )
            do_full = tl.load(
                do_base
                + rows[:, None] * stride_dom
                + offs_d_full[None, :] * stride_dod,
                mask=(rows[:, None] < SQ) & (offs_d_full[None, :] < HEAD_DIM),
                other=0.0,
            )
            stats = tl.load(
                stats_base + rows * stride_sm,
                mask=rows < SQ,
                other=float("-inf"),
            ).to(tl.float32)
            delta = tl.load(
                delta_base + rows * stride_delta_m,
                mask=rows < SQ,
                other=0.0,
            ).to(tl.float32)

            score = (
                tl.dot(q_full, tl.trans(k_full)).to(tl.float32) * attn_scale
            )
            if HAS_BIAS:
                bias_tile = tl.load(
                    bias_ptr
                    + off_b * stride_bias_b
                    + off_h * stride_bias_h
                    + rows[:, None] * stride_bias_m
                    + offs_n[None, :] * stride_bias_n,
                    mask=(rows[:, None] < SQ) & (offs_n[None, :] < SKV),
                    other=0.0,
                )
                score += bias_tile.to(tl.float32)

            if FULL_ATTENTION:
                p_attn = tl.exp(score - stats[:, None])
            else:
                valid = (rows[:, None] < SQ) & (offs_n[None, :] < SKV)
                if BANDED:
                    diag = offs_n[None, :] - rows[:, None]
                    valid = valid & (diag >= min_diag) & (diag <= max_diag)
                p_attn = tl.where(valid, tl.exp(score - stats[:, None]), 0.0)
            dp = tl.dot(do_full, tl.trans(v_full)).to(tl.float32)
            ds = p_attn * (dp - delta[:, None])

            q_out = tl.load(
                q_base
                + rows[:, None] * stride_qm
                + offs_d[None, :] * stride_qd,
                mask=(rows[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM),
                other=0.0,
            )
            do_out = tl.load(
                do_base
                + rows[:, None] * stride_dom
                + offs_d[None, :] * stride_dod,
                mask=(rows[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM),
                other=0.0,
            )
            dk += tl.dot(tl.trans(ds).to(q_out.dtype), q_out)
            dv += tl.dot(tl.trans(p_attn).to(do_out.dtype), do_out)

    mask = (offs_n[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM)
    tl.store(
        dk_ptr
        + off_b * stride_dkb
        + off_kh * stride_dkh
        + offs_n[:, None] * stride_dkn
        + offs_d[None, :] * stride_dkd,
        (dk * attn_scale).to(dk_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        dv_ptr
        + off_b * stride_dvb
        + off_kh * stride_dvh
        + offs_n[:, None] * stride_dvn
        + offs_d[None, :] * stride_dvd,
        dv.to(dv_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _sdpa_bwd_dv_kernel(
    q_ptr,
    k_ptr,
    bias_ptr,
    do_ptr,
    stats_ptr,
    dv_ptr,
    attn_scale,
    HKV: tl.constexpr,
    SQ,
    SKV,
    min_diag,
    max_diag,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_bias_b: tl.constexpr,
    stride_bias_h: tl.constexpr,
    stride_bias_m: tl.constexpr,
    stride_bias_n: tl.constexpr,
    stride_dob: tl.constexpr,
    stride_doh: tl.constexpr,
    stride_dom: tl.constexpr,
    stride_dod: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_sh: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_dvb: tl.constexpr,
    stride_dvh: tl.constexpr,
    stride_dvn: tl.constexpr,
    stride_dvd: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    Q_PER: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D_FULL: tl.constexpr,
    BLOCK_DV_OUT: tl.constexpr,
    FULL_ATTENTION: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BANDED: tl.constexpr,
    CAUSAL_TOP_LEFT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_dv = tl.program_id(1)
    pid_bh = tl.program_id(2)
    off_b = pid_bh // HKV
    off_kh = pid_bh % HKV

    start_n = pid_n * BLOCK_N
    start_dv = pid_dv * BLOCK_DV_OUT
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d_full = tl.arange(0, BLOCK_D_FULL)
    offs_dv = start_dv + tl.arange(0, BLOCK_DV_OUT)

    k_base = k_ptr + off_b * stride_kb + off_kh * stride_kh
    k_full = tl.load(
        k_base
        + offs_n[:, None] * stride_kn
        + offs_d_full[None, :] * stride_kd,
        mask=(offs_n[:, None] < SKV) & (offs_d_full[None, :] < HEAD_DIM),
        other=0.0,
    )
    dv = tl.zeros((BLOCK_N, BLOCK_DV_OUT), dtype=tl.float32)

    for group_idx in tl.static_range(0, Q_PER):
        off_h = off_kh * Q_PER + group_idx
        q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
        do_base = do_ptr + off_b * stride_dob + off_h * stride_doh
        stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh
        loop_start_m = 0
        if CAUSAL_TOP_LEFT:
            loop_start_m = (start_n // BLOCK_M) * BLOCK_M
        for start_m in tl.range(loop_start_m, SQ, BLOCK_M):
            rows = start_m + offs_m
            q_full = tl.load(
                q_base
                + rows[:, None] * stride_qm
                + offs_d_full[None, :] * stride_qd,
                mask=(rows[:, None] < SQ) & (offs_d_full[None, :] < HEAD_DIM),
                other=0.0,
            )
            score = (
                tl.dot(q_full, tl.trans(k_full)).to(tl.float32) * attn_scale
            )
            if HAS_BIAS:
                bias_tile = tl.load(
                    bias_ptr
                    + off_b * stride_bias_b
                    + off_h * stride_bias_h
                    + rows[:, None] * stride_bias_m
                    + offs_n[None, :] * stride_bias_n,
                    mask=(rows[:, None] < SQ) & (offs_n[None, :] < SKV),
                    other=0.0,
                )
                score += bias_tile.to(tl.float32)

            stats = tl.load(
                stats_base + rows * stride_sm,
                mask=rows < SQ,
                other=float("-inf"),
            ).to(tl.float32)
            if FULL_ATTENTION:
                p = tl.exp(score - stats[:, None])
            else:
                valid = (rows[:, None] < SQ) & (offs_n[None, :] < SKV)
                if BANDED:
                    diag = offs_n[None, :] - rows[:, None]
                    valid = valid & (diag >= min_diag) & (diag <= max_diag)
                p = tl.where(valid, tl.exp(score - stats[:, None]), 0.0)
            do_tile = tl.load(
                do_base
                + rows[:, None] * stride_dom
                + offs_dv[None, :] * stride_dod,
                mask=(rows[:, None] < SQ) & (offs_dv[None, :] < V_DIM),
                other=0.0,
            )
            dv += tl.dot(tl.trans(p).to(do_tile.dtype), do_tile)

    tl.store(
        dv_ptr
        + off_b * stride_dvb
        + off_kh * stride_dvh
        + offs_n[:, None] * stride_dvn
        + offs_dv[None, :] * stride_dvd,
        dv.to(dv_ptr.dtype.element_ty),
        mask=(offs_n[:, None] < SKV) & (offs_dv[None, :] < V_DIM),
    )


@triton.jit
def _zero_contiguous_kernel(ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(
        ptr + offsets,
        tl.zeros((BLOCK,), dtype=tl.float32),
        mask=offsets < n_elements,
    )


_LOG2E_KERNEL = tl.constexpr(1.4426950408889634)

_LN2_KERNEL = tl.constexpr(0.6931471805599453)


@triton.jit
def _sdpa_fp8_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    k_base,
    v_base,
    bias_base,
    qk_scale,
    s_scale,
    offs_m,
    offs_d,
    offs_dv,
    lo,
    hi,
    SQ,
    SKV,
    min_diag,
    max_diag,
    stride_kn,
    stride_kd,
    stride_vn,
    stride_vd,
    stride_bias_m,
    stride_bias_n,
    HEAD_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PADDED_D: tl.constexpr,
    PADDED_DV: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BANDED: tl.constexpr,
    MASKED: tl.constexpr,
):
    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_mask = None
        if MASKED and PADDED_D:
            k_mask = (offs_d[:, None] < HEAD_DIM) & (offs_n[None, :] < SKV)
        elif MASKED:
            k_mask = offs_n[None, :] < SKV
        elif PADDED_D:
            k_mask = offs_d[:, None] < HEAD_DIM
        if k_mask is not None:
            k = tl.load(
                k_base
                + offs_d[:, None] * stride_kd
                + offs_n[None, :] * stride_kn,
                mask=k_mask,
                other=0.0,
            )
        else:
            k = tl.load(
                k_base
                + offs_d[:, None] * stride_kd
                + offs_n[None, :] * stride_kn
            )

        # Q (fp8) @ K^T (fp8) -> fp32, then scaled into log2 units.
        qk = tl.dot(q, k)
        score = qk.to(tl.float32) * qk_scale
        if HAS_BIAS:
            bias_mask = (offs_m[:, None] < SQ) & (offs_n[None, :] < SKV)
            bias_tile = tl.load(
                bias_base
                + offs_m[:, None] * stride_bias_m
                + offs_n[None, :] * stride_bias_n,
                mask=bias_mask,
                other=0.0,
            )
            score += bias_tile.to(tl.float32) * _LOG2E_KERNEL

        if MASKED:
            visible = offs_n[None, :] < SKV
            if BANDED:
                diag = offs_n[None, :] - offs_m[:, None]
                visible = visible & (diag >= min_diag) & (diag <= max_diag)
            score = tl.where(visible, score, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(score, 1))
        if MASKED:
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        else:
            m_safe = m_new
        p = tl.math.exp2(score - m_safe[:, None])
        alpha = tl.math.exp2(m_i - m_safe)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v_mask = None
        if MASKED and PADDED_DV:
            v_mask = (offs_n[:, None] < SKV) & (offs_dv[None, :] < V_DIM)
        elif MASKED:
            v_mask = offs_n[:, None] < SKV
        elif PADDED_DV:
            v_mask = offs_dv[None, :] < V_DIM
        if v_mask is not None:
            v = tl.load(
                v_base
                + offs_n[:, None] * stride_vn
                + offs_dv[None, :] * stride_vd,
                mask=v_mask,
                other=0.0,
            )
        else:
            v = tl.load(
                v_base
                + offs_n[:, None] * stride_vn
                + offs_dv[None, :] * stride_vd
            )
        # P (fp32) -> P (fp8) using scale_s, then P (fp8) @ V (fp8) -> fp32.
        p_fp8 = (p * s_scale).to(v.dtype)
        acc = tl.dot(p_fp8, v, acc)
        l_i = l_i * alpha + l_ij
        m_i = m_new
    return acc, l_i, m_i


@triton.jit
def _zero_sdpa_fp8_fwd_amax_kernel(amax_s_ptr, amax_o_ptr):
    tl.store(amax_s_ptr, 0.0)
    tl.store(amax_o_ptr, 0.0)


@triton.jit
def _sdpa_fp8_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    bias_ptr,
    o_ptr,
    stats_ptr,
    amax_s_ptr,
    amax_o_ptr,
    descale_q_ptr,
    descale_k_ptr,
    descale_v_ptr,
    descale_s_ptr,
    scale_s_ptr,
    scale_o_ptr,
    attn_scale,
    HQ,
    SQ,
    SKV,
    q_per_k,
    q_per_v,
    min_diag,
    max_diag,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_bias_b,
    stride_bias_h,
    stride_bias_m,
    stride_bias_n,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_sb,
    stride_sh,
    stride_sm,
    HEAD_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BANDED: tl.constexpr,
    GENERATE_STATS: tl.constexpr,
    REVERSE_CAUSAL: tl.constexpr,
):
    descale_q = tl.load(descale_q_ptr)
    descale_k = tl.load(descale_k_ptr)
    descale_v = tl.load(descale_v_ptr)
    descale_s = tl.load(descale_s_ptr)
    s_scale = tl.load(scale_s_ptr)
    o_scale = tl.load(scale_o_ptr)
    qk_scale = attn_scale * descale_q * descale_k * _LOG2E_KERNEL
    sv_descale = descale_s * descale_v

    raw_pid_m = tl.program_id(0)
    if REVERSE_CAUSAL:
        pid_m = tl.cdiv(SQ, BLOCK_M) - 1 - raw_pid_m
    else:
        pid_m = raw_pid_m
    pid_bh = tl.program_id(1)
    off_b = pid_bh // HQ
    off_h = pid_bh % HQ
    off_kh = off_h // q_per_k
    off_vh = off_h // q_per_v

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_m = offs_m < SQ

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_kh * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_vh * stride_vh
    bias_base = bias_ptr + off_b * stride_bias_b + off_h * stride_bias_h

    PADDED_D: tl.constexpr = BLOCK_D != HEAD_DIM
    PADDED_DV: tl.constexpr = BLOCK_DV != V_DIM

    q_mask = mask_m[:, None]
    if PADDED_D:
        q_mask = q_mask & (offs_d[None, :] < HEAD_DIM)
    q = tl.load(
        q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=q_mask,
        other=0.0,
    )

    lo = tl.maximum(start_m + min_diag, 0)
    lo_block = (lo // BLOCK_N) * BLOCK_N
    hi = tl.minimum(start_m + BLOCK_M - 1 + max_diag + 1, SKV)
    hi = tl.maximum(hi, lo_block)

    full_lo = tl.maximum(start_m + BLOCK_M - 1 + min_diag, 0)
    full_lo_block = tl.cdiv(full_lo, BLOCK_N) * BLOCK_N
    full_hi = tl.minimum(start_m + max_diag + 1, SKV)
    full_hi_block = (full_hi // BLOCK_N) * BLOCK_N

    phase_a_end = tl.minimum(full_lo_block, hi)
    phase_b_end = tl.maximum(tl.minimum(full_hi_block, hi), phase_a_end)

    acc = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)

    if lo_block < phase_a_end:
        acc, l_i, m_i = _sdpa_fp8_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            k_base,
            v_base,
            bias_base,
            qk_scale,
            s_scale,
            offs_m,
            offs_d,
            offs_dv,
            lo_block,
            phase_a_end,
            SQ,
            SKV,
            min_diag,
            max_diag,
            stride_kn,
            stride_kd,
            stride_vn,
            stride_vd,
            stride_bias_m,
            stride_bias_n,
            HEAD_DIM=HEAD_DIM,
            V_DIM=V_DIM,
            BLOCK_N=BLOCK_N,
            PADDED_D=PADDED_D,
            PADDED_DV=PADDED_DV,
            HAS_BIAS=HAS_BIAS,
            BANDED=BANDED,
            MASKED=True,
        )
    if phase_a_end < phase_b_end:
        acc, l_i, m_i = _sdpa_fp8_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            k_base,
            v_base,
            bias_base,
            qk_scale,
            s_scale,
            offs_m,
            offs_d,
            offs_dv,
            phase_a_end,
            phase_b_end,
            SQ,
            SKV,
            min_diag,
            max_diag,
            stride_kn,
            stride_kd,
            stride_vn,
            stride_vd,
            stride_bias_m,
            stride_bias_n,
            HEAD_DIM=HEAD_DIM,
            V_DIM=V_DIM,
            BLOCK_N=BLOCK_N,
            PADDED_D=PADDED_D,
            PADDED_DV=PADDED_DV,
            HAS_BIAS=HAS_BIAS,
            BANDED=BANDED,
            MASKED=False,
        )
    if phase_b_end < hi:
        acc, l_i, m_i = _sdpa_fp8_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            k_base,
            v_base,
            bias_base,
            qk_scale,
            s_scale,
            offs_m,
            offs_d,
            offs_dv,
            phase_b_end,
            hi,
            SQ,
            SKV,
            min_diag,
            max_diag,
            stride_kn,
            stride_kd,
            stride_vn,
            stride_vd,
            stride_bias_m,
            stride_bias_n,
            HEAD_DIM=HEAD_DIM,
            V_DIM=V_DIM,
            BLOCK_N=BLOCK_N,
            PADDED_D=PADDED_D,
            PADDED_DV=PADDED_DV,
            HAS_BIAS=HAS_BIAS,
            BANDED=BANDED,
            MASKED=True,
        )

    l_safe = tl.maximum(l_i, 1.0)
    o_val = acc * (sv_descale / l_safe[:, None])

    o_valid = mask_m[:, None]
    if PADDED_DV:
        o_valid = o_valid & (offs_dv[None, :] < V_DIM)
    local_amax_o = tl.max(tl.where(o_valid, tl.abs(o_val), 0.0))
    tl.atomic_max(amax_o_ptr, local_amax_o, sem="relaxed")
    # The maximum softmax probability in each row is exp(max_score - LSE),
    # which is exactly 1 / l_i in this online-softmax representation.
    amax_s_val = tl.max(tl.where(mask_m, 1.0 / l_safe, 0.0))
    tl.atomic_max(amax_s_ptr, amax_s_val, sem="relaxed")

    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh
    tl.store(
        o_base + offs_m[:, None] * stride_om + offs_dv[None, :] * stride_od,
        (o_val * o_scale).to(o_ptr.dtype.element_ty),
        mask=o_valid,
    )

    if GENERATE_STATS:
        stats = (m_i + tl.log2(l_safe)) * _LN2_KERNEL
        stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh
        tl.store(stats_base + offs_m * stride_sm, stats, mask=mask_m)


@triton.jit
def _zero_sdpa_fp8_bwd_amax_kernel(
    amax_dq_ptr, amax_dk_ptr, amax_dv_ptr, amax_dp_ptr
):
    tl.store(amax_dq_ptr, 0.0)
    tl.store(amax_dk_ptr, 0.0)
    tl.store(amax_dv_ptr, 0.0)
    tl.store(amax_dp_ptr, 0.0)


@triton.jit
def _sdpa_fp8_bwd_dq_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    do_ptr,
    stats_ptr,
    dq_ptr,
    amax_dq_ptr,
    descale_q_ptr,
    descale_k_ptr,
    descale_v_ptr,
    descale_o_ptr,
    descale_do_ptr,
    descale_dp_ptr,
    scale_dq_ptr,
    scale_dp_ptr,
    attn_scale,
    HQ,
    SQ,
    SKV,
    q_per_k: tl.constexpr,
    q_per_v: tl.constexpr,
    min_diag,
    max_diag,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_od: tl.constexpr,
    stride_dob: tl.constexpr,
    stride_doh: tl.constexpr,
    stride_dom: tl.constexpr,
    stride_dod: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_sh: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_dqb: tl.constexpr,
    stride_dqh: tl.constexpr,
    stride_dqm: tl.constexpr,
    stride_dqd: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BANDED: tl.constexpr,
    FULL_BLOCKS: tl.constexpr,
    CAUSAL_TOP_LEFT: tl.constexpr,
):
    descale_q = tl.load(descale_q_ptr)
    descale_k = tl.load(descale_k_ptr)
    descale_v = tl.load(descale_v_ptr)
    descale_o = tl.load(descale_o_ptr)
    descale_do = tl.load(descale_do_ptr)
    descale_dp = tl.load(descale_dp_ptr)
    scale_dq = tl.load(scale_dq_ptr)
    scale_dp = tl.load(scale_dp_ptr)
    qk_scale = descale_q * descale_k * attn_scale
    ov_descale = descale_o * descale_do
    do_v_descale = descale_do * descale_v
    dq_descale = descale_dp * descale_k

    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // HQ
    off_h = pid_bh % HQ
    off_kh = off_h // q_per_k
    off_vh = off_h // q_per_v

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_kh * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_vh * stride_vh
    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh
    do_base = do_ptr + off_b * stride_dob + off_h * stride_doh
    stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh

    q_offsets = offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    o_offsets = offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    do_offsets = offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
    if FULL_BLOCKS:
        q = tl.load(q_base + q_offsets)
        o = tl.load(o_base + o_offsets).to(tl.float32)
        do = tl.load(do_base + do_offsets)
        stats = tl.load(stats_base + offs_m * stride_sm).to(tl.float32)
    else:
        valid_md = (offs_m[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM)
        q = tl.load(q_base + q_offsets, mask=valid_md, other=0.0)
        o = tl.load(o_base + o_offsets, mask=valid_md, other=0.0).to(
            tl.float32
        )
        do = tl.load(do_base + do_offsets, mask=valid_md, other=0.0)
        stats = tl.load(
            stats_base + offs_m * stride_sm,
            mask=offs_m < SQ,
            other=float("-inf"),
        ).to(tl.float32)
    do_f32 = do.to(tl.float32)
    row_delta = tl.sum(o * do_f32, axis=1) * ov_descale

    dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    loop_skv = SKV
    if CAUSAL_TOP_LEFT:
        loop_skv = tl.minimum(SKV, start_m + BLOCK_M)

    for start_n in tl.range(0, loop_skv, BLOCK_N):
        cols = start_n + offs_n
        k_offsets = cols[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_offsets = cols[:, None] * stride_vn + offs_d[None, :] * stride_vd
        if FULL_BLOCKS:
            k = tl.load(k_base + k_offsets)
            v = tl.load(v_base + v_offsets)
        else:
            valid_nd = (cols[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM)
            k = tl.load(k_base + k_offsets, mask=valid_nd, other=0.0)
            v = tl.load(v_base + v_offsets, mask=valid_nd, other=0.0)

        score = tl.dot(q, tl.trans(k)).to(tl.float32) * qk_scale
        if BANDED:
            diag = cols[None, :] - offs_m[:, None]
            valid = (diag >= min_diag) & (diag <= max_diag)
            if not FULL_BLOCKS:
                valid = valid & (offs_m[:, None] < SQ) & (cols[None, :] < SKV)
            p = tl.where(
                valid,
                tl.exp2((score - stats[:, None]) * 1.4426950408889634),
                0.0,
            )
        elif FULL_BLOCKS:
            p = tl.exp2((score - stats[:, None]) * 1.4426950408889634)
        else:
            valid = (offs_m[:, None] < SQ) & (cols[None, :] < SKV)
            p = tl.where(
                valid,
                tl.exp2((score - stats[:, None]) * 1.4426950408889634),
                0.0,
            )

        dp = tl.dot(do, tl.trans(v)).to(tl.float32) * do_v_descale
        ds = p * (dp - row_delta[:, None]) * attn_scale
        ds_quant = (ds * scale_dp).to(q.dtype)
        dq += tl.dot(ds_quant, k)

    dq_val = dq * dq_descale
    dq_out_ptrs = (
        dq_ptr
        + off_b * stride_dqb
        + off_h * stride_dqh
        + offs_m[:, None] * stride_dqm
        + offs_d[None, :] * stride_dqd
    )
    if FULL_BLOCKS:
        local_amax = tl.max(tl.abs(dq_val))
        tl.atomic_max(amax_dq_ptr, local_amax, sem="relaxed")
        tl.store(dq_out_ptrs, (dq_val * scale_dq).to(dq_ptr.dtype.element_ty))
    else:
        local_amax = tl.max(tl.where(valid_md, tl.abs(dq_val), 0.0))
        tl.atomic_max(amax_dq_ptr, local_amax, sem="relaxed")
        tl.store(
            dq_out_ptrs,
            (dq_val * scale_dq).to(dq_ptr.dtype.element_ty),
            mask=valid_md,
        )


@triton.jit
def _sdpa_fp8_bwd_dkdv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    do_ptr,
    stats_ptr,
    dk_ptr,
    dv_ptr,
    amax_dk_ptr,
    amax_dv_ptr,
    amax_dp_ptr,
    descale_q_ptr,
    descale_k_ptr,
    descale_v_ptr,
    descale_o_ptr,
    descale_do_ptr,
    descale_s_ptr,
    descale_dp_ptr,
    scale_s_ptr,
    scale_dk_ptr,
    scale_dv_ptr,
    scale_dp_ptr,
    attn_scale,
    HKV: tl.constexpr,
    SQ,
    SKV,
    min_diag,
    max_diag,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_od: tl.constexpr,
    stride_dob: tl.constexpr,
    stride_doh: tl.constexpr,
    stride_dom: tl.constexpr,
    stride_dod: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_sh: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_dkb: tl.constexpr,
    stride_dkh: tl.constexpr,
    stride_dkn: tl.constexpr,
    stride_dkd: tl.constexpr,
    stride_dvb: tl.constexpr,
    stride_dvh: tl.constexpr,
    stride_dvn: tl.constexpr,
    stride_dvd: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    Q_PER: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BANDED: tl.constexpr,
    FULL_BLOCKS: tl.constexpr,
    CAUSAL_TOP_LEFT: tl.constexpr,
):
    descale_q = tl.load(descale_q_ptr)
    descale_k = tl.load(descale_k_ptr)
    descale_v = tl.load(descale_v_ptr)
    descale_o = tl.load(descale_o_ptr)
    descale_do = tl.load(descale_do_ptr)
    descale_s = tl.load(descale_s_ptr)
    descale_dp = tl.load(descale_dp_ptr)
    scale_s = tl.load(scale_s_ptr)
    scale_dk = tl.load(scale_dk_ptr)
    scale_dv = tl.load(scale_dv_ptr)
    scale_dp = tl.load(scale_dp_ptr)
    qk_scale = descale_q * descale_k * attn_scale
    ov_descale = descale_o * descale_do
    do_v_descale = descale_do * descale_v
    dk_descale = descale_dp * descale_q
    dv_descale = descale_s * descale_do

    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // HKV
    off_kh = pid_bh % HKV

    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    k_base = k_ptr + off_b * stride_kb + off_kh * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_kh * stride_vh
    kv_offsets = offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    vv_offsets = offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    if FULL_BLOCKS:
        k = tl.load(k_base + kv_offsets)
        v = tl.load(v_base + vv_offsets)
    else:
        valid_nd = (offs_n[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM)
        k = tl.load(k_base + kv_offsets, mask=valid_nd, other=0.0)
        v = tl.load(v_base + vv_offsets, mask=valid_nd, other=0.0)
    dk = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    local_amax_dp = tl.full((), 0.0, dtype=tl.float32)

    for group_idx in tl.static_range(0, Q_PER):
        off_h = off_kh * Q_PER + group_idx
        q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
        o_base = o_ptr + off_b * stride_ob + off_h * stride_oh
        do_base = do_ptr + off_b * stride_dob + off_h * stride_doh
        stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh

        loop_start_m = 0
        if CAUSAL_TOP_LEFT:
            loop_start_m = start_n

        for start_m in tl.range(loop_start_m, SQ, BLOCK_M):
            rows = start_m + offs_m
            q_offsets = rows[:, None] * stride_qm + offs_d[None, :] * stride_qd
            o_offsets = rows[:, None] * stride_om + offs_d[None, :] * stride_od
            do_offsets = (
                rows[:, None] * stride_dom + offs_d[None, :] * stride_dod
            )
            if FULL_BLOCKS:
                q = tl.load(q_base + q_offsets)
                o = tl.load(o_base + o_offsets).to(tl.float32)
                do = tl.load(do_base + do_offsets)
                stats = tl.load(stats_base + rows * stride_sm).to(tl.float32)
            else:
                valid_md = (rows[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM)
                q = tl.load(q_base + q_offsets, mask=valid_md, other=0.0)
                o = tl.load(o_base + o_offsets, mask=valid_md, other=0.0).to(
                    tl.float32
                )
                do = tl.load(do_base + do_offsets, mask=valid_md, other=0.0)
                stats = tl.load(
                    stats_base + rows * stride_sm,
                    mask=rows < SQ,
                    other=float("-inf"),
                ).to(tl.float32)
            do_f32 = do.to(tl.float32)
            row_delta = tl.sum(o * do_f32, axis=1) * ov_descale

            score = tl.dot(q, tl.trans(k)).to(tl.float32) * qk_scale
            if BANDED:
                diag = offs_n[None, :] - rows[:, None]
                valid = (diag >= min_diag) & (diag <= max_diag)
                if not FULL_BLOCKS:
                    valid = (
                        valid & (rows[:, None] < SQ) & (offs_n[None, :] < SKV)
                    )
                p = tl.where(
                    valid,
                    tl.exp2((score - stats[:, None]) * 1.4426950408889634),
                    0.0,
                )
            elif FULL_BLOCKS:
                p = tl.exp2((score - stats[:, None]) * 1.4426950408889634)
            else:
                valid = (rows[:, None] < SQ) & (offs_n[None, :] < SKV)
                p = tl.where(
                    valid,
                    tl.exp2((score - stats[:, None]) * 1.4426950408889634),
                    0.0,
                )

            p_quant = (p * scale_s).to(q.dtype)
            dp = tl.dot(do, tl.trans(v)).to(tl.float32) * do_v_descale
            ds = p * (dp - row_delta[:, None]) * attn_scale
            if BANDED:
                local_amax_dp = tl.maximum(
                    local_amax_dp, tl.max(tl.where(valid, tl.abs(ds), 0.0))
                )
            elif FULL_BLOCKS:
                local_amax_dp = tl.maximum(local_amax_dp, tl.max(tl.abs(ds)))
            else:
                local_amax_dp = tl.maximum(
                    local_amax_dp, tl.max(tl.where(valid, tl.abs(ds), 0.0))
                )
            ds_quant = (ds * scale_dp).to(q.dtype)

            dk += tl.dot(tl.trans(ds_quant), q)
            dv += tl.dot(tl.trans(p_quant), do)

    dk_val = dk * dk_descale
    dv_val = dv * dv_descale
    dk_out_ptrs = (
        dk_ptr
        + off_b * stride_dkb
        + off_kh * stride_dkh
        + offs_n[:, None] * stride_dkn
        + offs_d[None, :] * stride_dkd
    )
    dv_out_ptrs = (
        dv_ptr
        + off_b * stride_dvb
        + off_kh * stride_dvh
        + offs_n[:, None] * stride_dvn
        + offs_d[None, :] * stride_dvd
    )
    if FULL_BLOCKS:
        local_amax_dk = tl.max(tl.abs(dk_val))
        local_amax_dv = tl.max(tl.abs(dv_val))
        tl.atomic_max(amax_dk_ptr, local_amax_dk, sem="relaxed")
        tl.atomic_max(amax_dv_ptr, local_amax_dv, sem="relaxed")
        tl.atomic_max(amax_dp_ptr, local_amax_dp, sem="relaxed")
        tl.store(dk_out_ptrs, (dk_val * scale_dk).to(dk_ptr.dtype.element_ty))
        tl.store(dv_out_ptrs, (dv_val * scale_dv).to(dv_ptr.dtype.element_ty))
    else:
        local_amax_dk = tl.max(tl.where(valid_nd, tl.abs(dk_val), 0.0))
        local_amax_dv = tl.max(tl.where(valid_nd, tl.abs(dv_val), 0.0))
        tl.atomic_max(amax_dk_ptr, local_amax_dk, sem="relaxed")
        tl.atomic_max(amax_dv_ptr, local_amax_dv, sem="relaxed")
        tl.atomic_max(amax_dp_ptr, local_amax_dp, sem="relaxed")
        tl.store(
            dk_out_ptrs,
            (dk_val * scale_dk).to(dk_ptr.dtype.element_ty),
            mask=valid_nd,
        )
        tl.store(
            dv_out_ptrs,
            (dv_val * scale_dv).to(dv_ptr.dtype.element_ty),
            mask=valid_nd,
        )
