# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA Triton kernels for attention operations.

This platform module contains the generic NVIDIA fallback entries together
with CUDA/Hopper-specialized decode, descriptor/TMA, and persistent-cache
variants. The NVIDIA registry owns dispatch for these operations; other
platforms use the compact portable module under ``kernels/common``.
"""

from __future__ import annotations

import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Attention algorithm variants. Native dispatch selects only registry-declared
# entry points; auxiliary kernels remain available to explicit compiler policy.
# -----------------------------------------------------------------------------


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
def _sdpa_fwd_dense_exact_inner(
    acc,
    l_i,
    m_i,
    q,
    k_base,
    v_base,
    qk_scale: tl.constexpr,
    offs_d,
    offs_dv,
    SKV: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vd: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    for start_n in range(0, SKV, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(
            k_base + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn
        )
        score = tl.dot(q, k).to(tl.float32) * qk_scale
        m_new = tl.maximum(m_i, tl.max(score, 1))
        p = tl.exp2(score - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(
            v_base + offs_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    return acc, l_i, m_i


@triton.jit
def _sdpa_fwd_dense_exact_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    qk_scale: tl.constexpr,
    HQ: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    q_per_k: tl.constexpr,
    q_per_v: tl.constexpr,
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
    HEAD_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    ELEM_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // HQ
    off_h = pid_bh % HQ
    off_kh = off_h // q_per_k
    off_vh = off_h // q_per_v

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_kh * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_vh * stride_vh

    q = tl.load(
        q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    )
    acc = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)

    acc, l_i, m_i = _sdpa_fwd_dense_exact_inner(
        acc,
        l_i,
        m_i,
        q,
        k_base,
        v_base,
        qk_scale,
        offs_d,
        offs_dv,
        SKV,
        stride_kn,
        stride_kd,
        stride_vn,
        stride_vd,
        BLOCK_N=BLOCK_N,
    )

    acc = acc / l_i[:, None]
    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh
    tl.store(
        o_base + offs_m[:, None] * stride_om + offs_dv[None, :] * stride_od,
        acc.to(o_ptr.dtype.element_ty),
    )


@triton.jit
def _sdpa_fwd_gqa_causal_desc_inner(
    acc,
    l_i,
    m_i,
    q,
    k_desc,
    v_desc,
    qk_scale,
    offs_m,
    lo,
    hi,
    BLOCK_N: tl.constexpr,
    MASKED: tl.constexpr,
):
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        start_n_i32 = start_n.to(tl.int32)  # type: ignore[attr-defined]
        offs_n = start_n_i32 + tl.arange(0, BLOCK_N)
        k = tl.trans(k_desc.load([start_n_i32, 0]))
        score = tl.dot(q, k).to(tl.float32) * qk_scale
        if MASKED:
            score = tl.where(
                offs_n[None, :] <= offs_m[:, None], score, float("-inf")
            )

        m_new = tl.maximum(m_i, tl.max(score, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(score - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = v_desc.load([start_n_i32, 0])
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new
    return acc, l_i, m_i


@triton.jit
def _sdpa_fwd_gqa_causal_kdesc_inner(
    acc,
    l_i,
    m_i,
    q,
    k_desc,
    v_base,
    qk_scale: tl.constexpr,
    offs_m,
    offs_dv,
    lo,
    hi,
    stride_vn: tl.constexpr,
    stride_vd: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MASKED: tl.constexpr,
):
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        start_n_i32 = start_n.to(tl.int32)  # type: ignore[attr-defined]
        offs_n = start_n_i32 + tl.arange(0, BLOCK_N)
        k = tl.trans(k_desc.load([start_n_i32, 0]))
        score = tl.dot(q, k).to(tl.float32) * qk_scale
        if MASKED:
            score = tl.where(
                offs_n[None, :] <= offs_m[:, None], score, float("-inf")
            )

        m_new = tl.maximum(m_i, tl.max(score, 1))
        m_safe = m_new
        p = tl.exp2(score - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = tl.load(
            v_base + offs_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd
        )
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new
    return acc, l_i, m_i


@triton.jit
def _sdpa_fwd_causal_host_kdesc_inner(
    acc,
    l_i,
    m_i,
    q,
    k_desc,
    k_row,
    v_base,
    qk_scale: tl.constexpr,
    offs_m,
    offs_dv,
    lo,
    hi,
    stride_vn: tl.constexpr,
    stride_vd: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MASKED: tl.constexpr,
):
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        start_n_i32 = start_n.to(tl.int32)  # type: ignore[attr-defined]
        offs_n = start_n_i32 + tl.arange(0, BLOCK_N)
        k = tl.trans(k_desc.load([k_row + start_n_i32, 0]))
        score = tl.dot(q, k).to(tl.float32) * qk_scale
        if MASKED:
            score = tl.where(
                offs_n[None, :] <= offs_m[:, None],
                score,
                float("-inf"),
            )
        m_new = tl.maximum(m_i, tl.max(score, 1))
        p = tl.exp2(score - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(
            v_base + offs_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd
        )
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new
    return acc, l_i, m_i


@triton.jit
def _sdpa_fwd_mha_causal_hostdesc_kernel(
    q_ptr,
    k_desc,
    v_ptr,
    o_ptr,
    stats_ptr,
    qk_scale: tl.constexpr,
    HQ: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_od: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_sh: tl.constexpr,
    stride_sm: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    ELEM_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    raw_pid_m = tl.program_id(1)
    pid_m = tl.cdiv(SQ, BLOCK_M) - 1 - raw_pid_m
    off_b = pid_bh // HQ
    off_h = pid_bh % HQ
    start_m = pid_m * BLOCK_M
    offs_m = tl.max_contiguous(start_m + tl.arange(0, BLOCK_M), BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    v_base = v_ptr + off_b * stride_vb + off_h * stride_vh
    q = tl.load(
        q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    )
    acc = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    hi = start_m + BLOCK_M
    full_hi = tl.minimum((start_m // BLOCK_N) * BLOCK_N, hi)
    k_row = pid_bh * SKV
    if 0 < full_hi:
        acc, l_i, m_i = _sdpa_fwd_causal_host_kdesc_inner(
            acc,
            l_i,
            m_i,
            q,
            k_desc,
            k_row,
            v_base,
            qk_scale,
            offs_m,
            offs_dv,
            0,
            full_hi,
            stride_vn,
            stride_vd,
            BLOCK_N=BLOCK_N,
            MASKED=False,
        )
    if full_hi < hi:
        acc, l_i, m_i = _sdpa_fwd_causal_host_kdesc_inner(
            acc,
            l_i,
            m_i,
            q,
            k_desc,
            k_row,
            v_base,
            qk_scale,
            offs_m,
            offs_dv,
            full_hi,
            hi,
            stride_vn,
            stride_vd,
            BLOCK_N=BLOCK_N,
            MASKED=True,
        )
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    tl.store(
        o_ptr
        + off_b * stride_ob
        + off_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_dv[None, :] * stride_od,
        acc.to(o_ptr.dtype.element_ty),
    )
    tl.store(
        stats_ptr + off_b * stride_sb + off_h * stride_sh + offs_m * stride_sm,
        m_i / _sdpa_LOG2E_KERNEL_variant + tl.log(l_safe),
    )


@triton.jit
def _sdpa_fwd_gqa_causal_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    stats_ptr,
    qk_scale,
    HKV,
    SQ,
    SKV,
    GROUP,
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
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_sb,
    stride_sh,
    stride_sm,
    HEAD_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    ELEM_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    raw_pid_m = tl.program_id(0)
    pid_m = tl.cdiv(SQ, BLOCK_M) - 1 - raw_pid_m
    pid_bkv = tl.program_id(1)
    pid_hg = tl.program_id(2)
    off_b = pid_bkv // HKV
    off_kh = pid_bkv % HKV

    start_m = pid_m * BLOCK_M
    offs_mh = tl.arange(0, BLOCK_M * BLOCK_H)
    offs_h = pid_hg * BLOCK_H + offs_mh // BLOCK_M
    offs_m = start_m + (offs_mh % BLOCK_M)
    q_head = off_kh * GROUP + offs_h
    row_mask = (offs_h < GROUP) & (offs_m < SQ)
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)

    q = tl.load(
        q_ptr
        + off_b * stride_qb
        + q_head[:, None] * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd,
        mask=row_mask[:, None],
        other=0.0,
    )

    k_base = k_ptr + off_b * stride_kb + off_kh * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_kh * stride_vh
    acc = tl.zeros((BLOCK_M * BLOCK_H, BLOCK_DV), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M * BLOCK_H,), dtype=tl.float32)
    m_i = tl.full((BLOCK_M * BLOCK_H,), float("-inf"), dtype=tl.float32)

    hi = tl.minimum(start_m + BLOCK_M, SKV)
    full_hi = ((start_m + 1) // BLOCK_N) * BLOCK_N
    full_hi = tl.minimum(full_hi, hi)

    if 0 < full_hi:
        acc, l_i, m_i = _sdpa_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            k_base,
            v_base,
            q_ptr,
            qk_scale,
            offs_m,
            offs_d,
            offs_dv,
            0,
            full_hi,
            SQ,
            SKV,
            -1073741824,
            0,
            stride_kn,
            stride_kd,
            stride_vn,
            stride_vd,
            0,
            0,
            HEAD_DIM=HEAD_DIM,
            V_DIM=V_DIM,
            BLOCK_N=BLOCK_N,
            PADDED_D=False,
            PADDED_DV=False,
            HAS_BIAS=False,
            BANDED=True,
            MASKED=False,
        )
    if full_hi < hi:
        acc, l_i, m_i = _sdpa_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            k_base,
            v_base,
            q_ptr,
            qk_scale,
            offs_m,
            offs_d,
            offs_dv,
            full_hi,
            hi,
            SQ,
            SKV,
            -1073741824,
            0,
            stride_kn,
            stride_kd,
            stride_vn,
            stride_vd,
            0,
            0,
            HEAD_DIM=HEAD_DIM,
            V_DIM=V_DIM,
            BLOCK_N=BLOCK_N,
            PADDED_D=False,
            PADDED_DV=False,
            HAS_BIAS=False,
            BANDED=True,
            MASKED=True,
        )

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    tl.store(
        o_ptr
        + off_b * stride_ob
        + q_head[:, None] * stride_oh
        + offs_m[:, None] * stride_om
        + offs_dv[None, :] * stride_od,
        acc.to(o_ptr.dtype.element_ty),
        mask=row_mask[:, None],
    )

    stats = m_i / _sdpa_LOG2E_KERNEL_variant + tl.log(l_safe)
    tl.store(
        stats_ptr
        + off_b * stride_sb
        + q_head * stride_sh
        + offs_m * stride_sm,
        stats,
        mask=row_mask,
    )


@triton.jit
def _sdpa_fwd_mha_causal_desc_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    stats_ptr,
    qk_scale: tl.constexpr,
    HQ: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
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
    stride_sb: tl.constexpr,
    stride_sh: tl.constexpr,
    stride_sm: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    ELEM_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    raw_pid_m = tl.program_id(1)
    pid_m = tl.cdiv(SQ, BLOCK_M) - 1 - raw_pid_m
    off_b = pid_bh // HQ
    off_h = pid_bh % HQ

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_m = tl.max_contiguous(offs_m, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_h * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_h * stride_vh

    q = tl.load(
        q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    )
    k_desc = tl.make_tensor_descriptor(
        k_base,
        shape=[SKV, HEAD_DIM],
        strides=[stride_kn, stride_kd],
        block_shape=[BLOCK_N, BLOCK_D],
    )
    acc = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)

    hi = start_m + BLOCK_M
    full_hi = (start_m // BLOCK_N) * BLOCK_N
    full_hi = tl.minimum(full_hi, hi)

    if 0 < full_hi:
        acc, l_i, m_i = _sdpa_fwd_gqa_causal_kdesc_inner(
            acc,
            l_i,
            m_i,
            q,
            k_desc,
            v_base,
            qk_scale,
            offs_m,
            offs_dv,
            0,
            full_hi,
            stride_vn,
            stride_vd,
            BLOCK_N=BLOCK_N,
            MASKED=False,
        )
    if full_hi < hi:
        acc, l_i, m_i = _sdpa_fwd_gqa_causal_kdesc_inner(
            acc,
            l_i,
            m_i,
            q,
            k_desc,
            v_base,
            qk_scale,
            offs_m,
            offs_dv,
            full_hi,
            hi,
            stride_vn,
            stride_vd,
            BLOCK_N=BLOCK_N,
            MASKED=True,
        )

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh
    tl.store(
        o_base + offs_m[:, None] * stride_om + offs_dv[None, :] * stride_od,
        acc.to(o_ptr.dtype.element_ty),
    )

    stats = m_i / _sdpa_LOG2E_KERNEL_variant + tl.log(l_safe)
    stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh
    tl.store(stats_base + offs_m * stride_sm, stats)


@triton.jit
def _sdpa_fwd_gqa_causal_desc_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    stats_ptr,
    qk_scale: tl.constexpr,
    HKV: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    GROUP: tl.constexpr,
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
    stride_sb: tl.constexpr,
    stride_sh: tl.constexpr,
    stride_sm: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    ELEM_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    raw_pid_m = tl.program_id(0)
    pid_m = tl.cdiv(SQ, BLOCK_M) - 1 - raw_pid_m
    pid_bkv = tl.program_id(1)
    pid_hg = tl.program_id(2)
    off_b = pid_bkv // HKV
    off_kh = pid_bkv % HKV

    start_m = pid_m * BLOCK_M
    offs_mh = tl.arange(0, BLOCK_M * BLOCK_H)
    offs_h = pid_hg * BLOCK_H + offs_mh // BLOCK_M
    offs_m = start_m + (offs_mh % BLOCK_M)
    q_head = off_kh * GROUP + offs_h
    row_mask = (offs_h < GROUP) & (offs_m < SQ)
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)

    q = tl.load(
        q_ptr
        + off_b * stride_qb
        + q_head[:, None] * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd,
        mask=row_mask[:, None],
        other=0.0,
    )

    k_base = k_ptr + off_b * stride_kb + off_kh * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_kh * stride_vh
    k_desc = tl.make_tensor_descriptor(
        k_base,
        shape=[SKV, HEAD_DIM],
        strides=[stride_kn, stride_kd],
        block_shape=[BLOCK_N, BLOCK_D],
    )
    acc = tl.zeros((BLOCK_M * BLOCK_H, BLOCK_DV), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M * BLOCK_H,), dtype=tl.float32)
    m_i = tl.full((BLOCK_M * BLOCK_H,), float("-inf"), dtype=tl.float32)

    hi = tl.minimum(start_m + BLOCK_M, SKV)
    full_hi = ((start_m + 1) // BLOCK_N) * BLOCK_N
    full_hi = tl.minimum(full_hi, hi)

    if 0 < full_hi:
        acc, l_i, m_i = _sdpa_fwd_gqa_causal_kdesc_inner(
            acc,
            l_i,
            m_i,
            q,
            k_desc,
            v_base,
            qk_scale,
            offs_m,
            offs_dv,
            0,
            full_hi,
            stride_vn,
            stride_vd,
            BLOCK_N=BLOCK_N,
            MASKED=False,
        )
    if full_hi < hi:
        acc, l_i, m_i = _sdpa_fwd_gqa_causal_kdesc_inner(
            acc,
            l_i,
            m_i,
            q,
            k_desc,
            v_base,
            qk_scale,
            offs_m,
            offs_dv,
            full_hi,
            hi,
            stride_vn,
            stride_vd,
            BLOCK_N=BLOCK_N,
            MASKED=True,
        )

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    tl.store(
        o_ptr
        + off_b * stride_ob
        + q_head[:, None] * stride_oh
        + offs_m[:, None] * stride_om
        + offs_dv[None, :] * stride_od,
        acc.to(o_ptr.dtype.element_ty),
        mask=row_mask[:, None],
    )

    stats = m_i / _sdpa_LOG2E_KERNEL_variant + tl.log(l_safe)
    tl.store(
        stats_ptr
        + off_b * stride_sb
        + q_head * stride_sh
        + offs_m * stride_sm,
        stats,
        mask=row_mask,
    )


@triton.jit
def _sdpa_decode_split_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    part_ptr,
    qk_scale,
    HKV,
    SKV,
    CHUNK,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_pb,
    stride_ph,
    stride_ps,
    stride_pd,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    ELEM_SIZE: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    # Flash-decoding partial pass for seq_q == 1: one program covers a
    # whole GQA head group (the group is the tl.dot M dimension, so the
    # K/V chunk is loaded once per group) over one KV chunk.
    pid_s = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // HKV
    off_kvh = pid_bh % HKV

    offs_g = tl.arange(0, BLOCK_G)
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_g = offs_g < GROUP
    q_head = off_kvh * GROUP + offs_g

    q_mask = mask_g[:, None]
    if BLOCK_D != HEAD_DIM:
        q_mask = q_mask & (offs_d[None, :] < HEAD_DIM)
    q = tl.load(
        q_ptr
        + off_b * stride_qb
        + q_head[:, None] * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=q_mask,
        other=0.0,
    )
    k_base = k_ptr + off_b * stride_kb + off_kvh * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_kvh * stride_vh

    lo = pid_s * CHUNK
    hi = tl.minimum(lo + CHUNK, SKV)

    acc = tl.zeros((BLOCK_G, BLOCK_DV), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_G,), dtype=tl.float32)
    m_i = tl.full((BLOCK_G,), float("-inf"), dtype=tl.float32)

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_mask = offs_n[None, :] < hi
        if BLOCK_D != HEAD_DIM:
            k_mask = k_mask & (offs_d[:, None] < HEAD_DIM)
        k = tl.load(
            k_base + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn,
            mask=k_mask,
            other=0.0,
        )
        score = tl.dot(q, k) * qk_scale
        score = tl.where(offs_n[None, :] < hi, score, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(score, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(score - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        v_mask = offs_n[:, None] < hi
        if BLOCK_DV != V_DIM:
            v_mask = v_mask & (offs_dv[None, :] < V_DIM)
        v = tl.load(
            v_base
            + offs_n[:, None] * stride_vn
            + offs_dv[None, :] * stride_vd,
            mask=v_mask,
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # Partial layout per (b, h, split): [acc[0:V_DIM], m, l] along the
    # last axis of the float32 workspace.
    part_base = (
        part_ptr + off_b * stride_pb + q_head * stride_ph + pid_s * stride_ps
    )
    acc_mask = mask_g[:, None]
    if BLOCK_DV != V_DIM:
        acc_mask = acc_mask & (offs_dv[None, :] < V_DIM)
    tl.store(
        part_base[:, None] + offs_dv[None, :] * stride_pd,
        acc,
        mask=acc_mask,
    )
    tl.store(part_base + V_DIM * stride_pd, m_i, mask=mask_g)
    tl.store(part_base + (V_DIM + 1) * stride_pd, l_i, mask=mask_g)


@triton.jit
def _sdpa_decode_combine_kernel(
    part_ptr,
    o_ptr,
    stats_ptr,
    HQ,
    SPLITS,
    stride_pb,
    stride_ph,
    stride_ps,
    stride_pd,
    stride_ob,
    stride_oh,
    stride_od,
    stride_sb,
    stride_sh,
    V_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    GENERATE_STATS: tl.constexpr,
):
    pid = tl.program_id(0)
    off_b = pid // HQ
    off_h = pid % HQ
    base = part_ptr + off_b * stride_pb + off_h * stride_ph

    offs_s = tl.arange(0, BLOCK_S)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_s = offs_s < SPLITS
    m_s = tl.load(
        base + offs_s * stride_ps + V_DIM * stride_pd,
        mask=mask_s,
        other=float("-inf"),
    )
    l_s = tl.load(
        base + offs_s * stride_ps + (V_DIM + 1) * stride_pd,
        mask=mask_s,
        other=0.0,
    )
    m = tl.max(m_s, 0)
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    scale = tl.exp2(m_s - m_safe)
    l_sum = tl.sum(l_s * scale, 0)

    acc_mask = mask_s[:, None]
    if BLOCK_DV != V_DIM:
        acc_mask = acc_mask & (offs_dv[None, :] < V_DIM)
    acc = tl.load(
        base[None, None]
        + offs_s[:, None] * stride_ps
        + offs_dv[None, :] * stride_pd,
        mask=acc_mask,
        other=0.0,
    )
    out = tl.sum(acc * scale[:, None], 0)
    l_safe = tl.where(l_sum == 0.0, 1.0, l_sum)
    out = out / l_safe

    o_mask = None
    if BLOCK_DV != V_DIM:
        o_mask = offs_dv < V_DIM
    tl.store(
        o_ptr + off_b * stride_ob + off_h * stride_oh + offs_dv * stride_od,
        out.to(o_ptr.dtype.element_ty),
        mask=o_mask,
    )
    if GENERATE_STATS:
        stats = m / _sdpa_LOG2E_KERNEL_variant + tl.log(l_safe)
        tl.store(stats_ptr + off_b * stride_sb + off_h * stride_sh, stats)


@triton.jit
def _sdpa_bwd_delta_kernel(
    o_ptr,
    do_ptr,
    delta_ptr,
    HQ: tl.constexpr,
    SQ: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_od: tl.constexpr,
    stride_dob: tl.constexpr,
    stride_doh: tl.constexpr,
    stride_dom: tl.constexpr,
    stride_dod: tl.constexpr,
    stride_db: tl.constexpr,
    stride_dh: tl.constexpr,
    stride_dm: tl.constexpr,
    V_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // HQ
    off_h = pid_bh % HQ

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask = (offs_m[:, None] < SQ) & (offs_dv[None, :] < V_DIM)

    o = tl.load(
        o_ptr
        + off_b * stride_ob
        + off_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_dv[None, :] * stride_od,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        do_ptr
        + off_b * stride_dob
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_dv[None, :] * stride_dod,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(
        delta_ptr + off_b * stride_db + off_h * stride_dh + offs_m * stride_dm,
        delta,
        mask=offs_m < SQ,
    )


@triton.jit
def _sdpa_bwd_owner_causal_d128_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    stats_ptr,
    delta_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    attn_scale,
    HQ: tl.constexpr,
    Q_PER: tl.constexpr,
    SQ: tl.constexpr,
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
    stride_dkb: tl.constexpr,
    stride_dkh: tl.constexpr,
    stride_dkn: tl.constexpr,
    stride_dkd: tl.constexpr,
    stride_dvb: tl.constexpr,
    stride_dvh: tl.constexpr,
    stride_dvn: tl.constexpr,
    stride_dvd: tl.constexpr,
    NUM_N_BLOCKS: tl.constexpr,
    NUM_M_BLOCKS: tl.constexpr,
    BLOCK_M_DKDV: tl.constexpr,
    BLOCK_N_DKDV: tl.constexpr,
    BLOCK_M_DQ: tl.constexpr,
    BLOCK_N_DQ: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tl.static_assert(BLOCK_M_DKDV == BLOCK_N_DKDV)
    tl.static_assert(BLOCK_M_DQ == BLOCK_N_DQ)
    pid = tl.program_id(0)
    off_b = tl.program_id(1)
    off_kh = tl.program_id(2)
    offs_d = tl.arange(0, BLOCK_D)

    if pid < NUM_N_BLOCKS:
        start_n = pid * BLOCK_N_DKDV
        cols = start_n + tl.arange(0, BLOCK_N_DKDV)
        k_base = k_ptr + off_b * stride_kb + off_kh * stride_kh
        v_base = v_ptr + off_b * stride_vb + off_kh * stride_vh
        k_tile = tl.load(
            k_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            eviction_policy="evict_last",
        )
        v_tile = tl.load(
            v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            eviction_policy="evict_last",
        )
        dk = tl.zeros((BLOCK_N_DKDV, BLOCK_D), dtype=tl.float32)
        dv = tl.zeros((BLOCK_N_DKDV, BLOCK_D), dtype=tl.float32)
        rows_base = tl.arange(0, BLOCK_M_DKDV)
        for off_g in range(0, Q_PER):
            off_h = off_kh * Q_PER + off_g
            q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
            do_base = do_ptr + off_b * stride_dob + off_h * stride_doh
            stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh
            delta_base = (
                delta_ptr + off_b * stride_delta_b + off_h * stride_delta_h
            )
            for start_m in tl.range(start_n, SQ, BLOCK_M_DKDV):
                rows = start_m + rows_base
                q_tile = tl.load(
                    q_base
                    + rows[:, None] * stride_qm
                    + offs_d[None, :] * stride_qd,
                    eviction_policy="evict_last",
                )
                do_tile = tl.load(
                    do_base
                    + rows[:, None] * stride_dom
                    + offs_d[None, :] * stride_dod,
                    eviction_policy="evict_last",
                )
                stats = tl.load(
                    stats_base + rows * stride_sm,
                    eviction_policy="evict_last",
                ).to(tl.float32)
                delta = tl.load(
                    delta_base + rows * stride_delta_m,
                    eviction_policy="evict_last",
                ).to(tl.float32)
                score = tl.dot(q_tile, tl.trans(k_tile)).to(tl.float32) * (
                    attn_scale * 1.4426950408889634
                )
                if start_m == start_n:
                    valid = cols[None, :] <= rows[:, None]
                    p = tl.where(
                        valid,
                        tl.exp2(score - stats[:, None] * 1.4426950408889634),
                        0.0,
                    )
                else:
                    p = tl.exp2(score - stats[:, None] * 1.4426950408889634)
                dp = tl.dot(do_tile, tl.trans(v_tile)).to(tl.float32)
                ds = p * (dp - delta[:, None])
                dk += tl.dot(tl.trans(ds).to(q_tile.dtype), q_tile)
                dv += tl.dot(tl.trans(p).to(do_tile.dtype), do_tile)
        tl.store(
            dk_ptr
            + off_b * stride_dkb
            + off_kh * stride_dkh
            + cols[:, None] * stride_dkn
            + offs_d[None, :] * stride_dkd,
            (dk * attn_scale).to(dk_ptr.dtype.element_ty),
        )
        tl.store(
            dv_ptr
            + off_b * stride_dvb
            + off_kh * stride_dvh
            + cols[:, None] * stride_dvn
            + offs_d[None, :] * stride_dvd,
            dv.to(dv_ptr.dtype.element_ty),
        )
    else:
        query_pid = pid - NUM_N_BLOCKS
        off_g = query_pid // NUM_M_BLOCKS
        pid_m = query_pid % NUM_M_BLOCKS
        off_h = off_kh * Q_PER + off_g
        start_m = pid_m * BLOCK_M_DQ
        rows = start_m + tl.arange(0, BLOCK_M_DQ)
        q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
        k_base = k_ptr + off_b * stride_kb + off_kh * stride_kh
        v_base = v_ptr + off_b * stride_vb + off_kh * stride_vh
        do_base = do_ptr + off_b * stride_dob + off_h * stride_doh
        stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh
        delta_base = (
            delta_ptr + off_b * stride_delta_b + off_h * stride_delta_h
        )
        q_tile = tl.load(
            q_base + rows[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            eviction_policy="evict_last",
        )
        do_tile = tl.load(
            do_base
            + rows[:, None] * stride_dom
            + offs_d[None, :] * stride_dod,
            eviction_policy="evict_last",
        )
        stats = tl.load(
            stats_base + rows * stride_sm,
            eviction_policy="evict_last",
        ).to(tl.float32)
        delta = tl.load(
            delta_base + rows * stride_delta_m,
            eviction_policy="evict_last",
        ).to(tl.float32)
        dq = tl.zeros((BLOCK_M_DQ, BLOCK_D), dtype=tl.float32)
        cols_base = tl.arange(0, BLOCK_N_DQ)
        for start_n in tl.range(0, start_m + BLOCK_M_DQ, BLOCK_N_DQ):
            cols = start_n + cols_base
            k_tile = tl.load(
                k_base
                + cols[:, None] * stride_kn
                + offs_d[None, :] * stride_kd,
                eviction_policy="evict_last",
            )
            v_tile = tl.load(
                v_base
                + cols[:, None] * stride_vn
                + offs_d[None, :] * stride_vd,
                eviction_policy="evict_last",
            )
            score = tl.dot(q_tile, tl.trans(k_tile)).to(tl.float32) * (
                attn_scale * 1.4426950408889634
            )
            if start_n + BLOCK_N_DQ <= start_m:
                p = tl.exp2(score - stats[:, None] * 1.4426950408889634)
            else:
                valid = cols[None, :] <= rows[:, None]
                p = tl.where(
                    valid,
                    tl.exp2(score - stats[:, None] * 1.4426950408889634),
                    0.0,
                )
            dp = tl.dot(do_tile, tl.trans(v_tile)).to(tl.float32)
            ds = p * (dp - delta[:, None])
            dq += tl.dot(ds.to(k_tile.dtype), k_tile)
        tl.store(
            dq_ptr
            + off_b * stride_dqb
            + off_h * stride_dqh
            + rows[:, None] * stride_dqm
            + offs_d[None, :] * stride_dqd,
            (dq * attn_scale).to(dq_ptr.dtype.element_ty),
        )


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


@triton.jit
def _zero_two_contiguous_kernel(
    b_ptr,
    c_ptr,
    b_elements: tl.constexpr,
    c_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    zeros = tl.zeros((BLOCK,), dtype=tl.float32)
    tl.store(b_ptr + offsets, zeros, mask=offsets < b_elements)
    tl.store(c_ptr + offsets, zeros, mask=offsets < c_elements)


@triton.jit
def _zero_three_contiguous_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_elements: tl.constexpr,
    b_elements: tl.constexpr,
    c_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    zeros = tl.zeros((BLOCK,), dtype=tl.float32)
    tl.store(a_ptr + offsets, zeros, mask=offsets < a_elements)
    tl.store(b_ptr + offsets, zeros, mask=offsets < b_elements)
    tl.store(c_ptr + offsets, zeros, mask=offsets < c_elements)


@triton.jit
def _zero_three_and_delta_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    o_ptr,
    do_ptr,
    delta_ptr,
    a_elements: tl.constexpr,
    b_elements: tl.constexpr,
    c_elements: tl.constexpr,
    total_rows: tl.constexpr,
    HQ: tl.constexpr,
    SQ: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_od: tl.constexpr,
    stride_dob: tl.constexpr,
    stride_doh: tl.constexpr,
    stride_dom: tl.constexpr,
    stride_dod: tl.constexpr,
    stride_delta_b: tl.constexpr,
    stride_delta_h: tl.constexpr,
    stride_delta_m: tl.constexpr,
    BLOCK_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_ZERO + tl.arange(0, BLOCK_ZERO)
    zeros = tl.zeros((BLOCK_ZERO,), dtype=tl.float32)
    tl.store(a_ptr + offsets, zeros, mask=offsets < a_elements)
    tl.store(b_ptr + offsets, zeros, mask=offsets < b_elements)
    tl.store(c_ptr + offsets, zeros, mask=offsets < c_elements)

    if pid * BLOCK_M < total_rows:
        row_ids = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        valid_rows = row_ids < total_rows
        rows_per_batch = HQ * SQ
        off_b = row_ids // rows_per_batch
        rem = row_ids - off_b * rows_per_batch
        off_h = rem // SQ
        offs_m = rem - off_h * SQ
        o = tl.load(
            o_ptr
            + off_b[:, None] * stride_ob
            + off_h[:, None] * stride_oh
            + offs_m[:, None] * stride_om
            + offs_d[None, :] * stride_od,
            mask=valid_rows[:, None] & (offs_d[None, :] < BLOCK_D),
            other=0.0,
        ).to(tl.float32)
        do = tl.load(
            do_ptr
            + off_b[:, None] * stride_dob
            + off_h[:, None] * stride_doh
            + offs_m[:, None] * stride_dom
            + offs_d[None, :] * stride_dod,
            mask=valid_rows[:, None] & (offs_d[None, :] < BLOCK_D),
            other=0.0,
        ).to(tl.float32)
        delta = tl.sum(o * do, axis=1)
        tl.store(
            delta_ptr
            + off_b * stride_delta_b
            + off_h * stride_delta_h
            + offs_m * stride_delta_m,
            delta,
            mask=valid_rows,
        )


@triton.jit
def _zero_three_equal_and_delta_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    o_ptr,
    do_ptr,
    delta_ptr,
    total_rows,
    HQ: tl.constexpr,
    SQ: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_od: tl.constexpr,
    stride_dob: tl.constexpr,
    stride_doh: tl.constexpr,
    stride_dom: tl.constexpr,
    stride_dod: tl.constexpr,
    stride_delta_b: tl.constexpr,
    stride_delta_h: tl.constexpr,
    stride_delta_m: tl.constexpr,
    BLOCK_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_ZERO + tl.arange(0, BLOCK_ZERO)
    zeros = tl.zeros((BLOCK_ZERO,), dtype=tl.float32)
    tl.store(a_ptr + offsets, zeros)
    tl.store(b_ptr + offsets, zeros)
    tl.store(c_ptr + offsets, zeros)

    if pid * BLOCK_M < total_rows:
        row_ids = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        valid_rows = row_ids < total_rows
        rows_per_batch = HQ * SQ
        off_b = row_ids // rows_per_batch
        rem = row_ids - off_b * rows_per_batch
        off_h = rem // SQ
        offs_m = rem - off_h * SQ
        o = tl.load(
            o_ptr
            + off_b[:, None] * stride_ob
            + off_h[:, None] * stride_oh
            + offs_m[:, None] * stride_om
            + offs_d[None, :] * stride_od,
            mask=valid_rows[:, None],
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        do = tl.load(
            do_ptr
            + off_b[:, None] * stride_dob
            + off_h[:, None] * stride_doh
            + offs_m[:, None] * stride_dom
            + offs_d[None, :] * stride_dod,
            mask=valid_rows[:, None],
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        delta = tl.sum(o * do, axis=1)
        tl.store(
            delta_ptr
            + off_b * stride_delta_b
            + off_h * stride_delta_h
            + offs_m * stride_delta_m,
            delta,
            mask=valid_rows,
        )


@triton.jit
def _sdpa_bwd_fused_atomic_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    do_ptr,
    stats_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    attn_scale,
    HQ: tl.constexpr,
    SQ,
    SKV,
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
    stride_dkb: tl.constexpr,
    stride_dkh: tl.constexpr,
    stride_dkn: tl.constexpr,
    stride_dkd: tl.constexpr,
    stride_dvb: tl.constexpr,
    stride_dvh: tl.constexpr,
    stride_dvn: tl.constexpr,
    stride_dvd: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bh = tl.program_id(2)
    off_b = pid_bh // HQ
    off_h = pid_bh % HQ

    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N
    rows = start_m + tl.arange(0, BLOCK_M)
    cols = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_h * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_h * stride_vh
    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh
    do_base = do_ptr + off_b * stride_dob + off_h * stride_doh
    stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh

    q_tile = tl.load(
        q_base + rows[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=(rows[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    k_tile = tl.load(
        k_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd,
        mask=(cols[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    v_tile = tl.load(
        v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd,
        mask=(cols[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    o_tile = tl.load(
        o_base + rows[:, None] * stride_om + offs_d[None, :] * stride_od,
        mask=(rows[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    do_tile = tl.load(
        do_base + rows[:, None] * stride_dom + offs_d[None, :] * stride_dod,
        mask=(rows[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    stats = tl.load(
        stats_base + rows * stride_sm,
        mask=rows < SQ,
        other=float("-inf"),
    ).to(tl.float32)
    stats_log2 = stats * 1.4426950408889634

    score = tl.dot(q_tile, tl.trans(k_tile)).to(tl.float32) * (
        attn_scale * 1.4426950408889634
    )
    valid = (rows[:, None] < SQ) & (cols[None, :] < SKV)
    p = tl.where(valid, tl.exp2(score - stats_log2[:, None]), 0.0)
    dp = tl.dot(do_tile, tl.trans(v_tile)).to(tl.float32)
    delta = tl.sum(o_tile * do_tile.to(tl.float32), axis=1)
    ds = p * (dp - delta[:, None])

    dq = tl.dot(ds.to(k_tile.dtype), k_tile) * attn_scale
    dk = tl.dot(tl.trans(ds).to(q_tile.dtype), q_tile) * attn_scale
    dv = tl.dot(tl.trans(p).to(do_tile.dtype), do_tile)

    tl.atomic_add(
        dq_ptr
        + off_b * stride_dqb
        + off_h * stride_dqh
        + rows[:, None] * stride_dqm
        + offs_d[None, :] * stride_dqd,
        dq.to(dq_ptr.dtype.element_ty),
        sem="relaxed",
        mask=(rows[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM),
    )
    tl.atomic_add(
        dk_ptr
        + off_b * stride_dkb
        + off_h * stride_dkh
        + cols[:, None] * stride_dkn
        + offs_d[None, :] * stride_dkd,
        dk.to(dk_ptr.dtype.element_ty),
        sem="relaxed",
        mask=(cols[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM),
    )
    tl.atomic_add(
        dv_ptr
        + off_b * stride_dvb
        + off_h * stride_dvh
        + cols[:, None] * stride_dvn
        + offs_d[None, :] * stride_dvd,
        dv.to(dv_ptr.dtype.element_ty),
        sem="relaxed",
        mask=(cols[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM),
    )


@triton.jit
def _sdpa_bwd_fused_atomic_causal_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    do_ptr,
    stats_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    attn_scale,
    HQ: tl.constexpr,
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
    stride_dqb: tl.constexpr,
    stride_dqh: tl.constexpr,
    stride_dqm: tl.constexpr,
    stride_dqd: tl.constexpr,
    stride_dkb: tl.constexpr,
    stride_dkh: tl.constexpr,
    stride_dkn: tl.constexpr,
    stride_dkd: tl.constexpr,
    stride_dvb: tl.constexpr,
    stride_dvh: tl.constexpr,
    stride_dvn: tl.constexpr,
    stride_dvd: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BANDED: tl.constexpr,
    CAUSAL_TOP_LEFT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bh = tl.program_id(2)
    off_b = pid_bh // HQ
    off_h = pid_bh % HQ

    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N
    if CAUSAL_TOP_LEFT and start_n > start_m + BLOCK_M - 1:
        return

    rows = start_m + tl.arange(0, BLOCK_M)
    cols = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_h * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_h * stride_vh
    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh
    do_base = do_ptr + off_b * stride_dob + off_h * stride_doh
    stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh

    q_tile = tl.load(
        q_base + rows[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=(rows[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    k_tile = tl.load(
        k_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd,
        mask=(cols[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    v_tile = tl.load(
        v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd,
        mask=(cols[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    o_tile = tl.load(
        o_base + rows[:, None] * stride_om + offs_d[None, :] * stride_od,
        mask=(rows[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    do_tile = tl.load(
        do_base + rows[:, None] * stride_dom + offs_d[None, :] * stride_dod,
        mask=(rows[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    stats = tl.load(
        stats_base + rows * stride_sm,
        mask=rows < SQ,
        other=float("-inf"),
    ).to(tl.float32)
    stats_log2 = stats * 1.4426950408889634

    score = tl.dot(q_tile, tl.trans(k_tile)).to(tl.float32) * (
        attn_scale * 1.4426950408889634
    )
    full_tile = start_n + BLOCK_N <= start_m
    full_tile = full_tile & (start_m + BLOCK_M <= SQ)
    full_tile = full_tile & (start_n + BLOCK_N <= SKV)
    if CAUSAL_TOP_LEFT and not BANDED and full_tile:
        p = tl.exp2(score - stats_log2[:, None])
    else:
        valid = (rows[:, None] < SQ) & (cols[None, :] < SKV)
        if BANDED:
            diag = cols[None, :] - rows[:, None]
            valid = valid & (diag >= min_diag) & (diag <= max_diag)
        p = tl.where(valid, tl.exp2(score - stats_log2[:, None]), 0.0)
    dp = tl.dot(do_tile, tl.trans(v_tile)).to(tl.float32)
    delta = tl.sum(o_tile * do_tile.to(tl.float32), axis=1)
    ds = p * (dp - delta[:, None])

    dq = tl.dot(ds.to(k_tile.dtype), k_tile) * attn_scale
    dk = tl.dot(tl.trans(ds).to(q_tile.dtype), q_tile) * attn_scale
    dv = tl.dot(tl.trans(p).to(do_tile.dtype), do_tile)

    tl.atomic_add(
        dq_ptr
        + off_b * stride_dqb
        + off_h * stride_dqh
        + rows[:, None] * stride_dqm
        + offs_d[None, :] * stride_dqd,
        dq.to(dq_ptr.dtype.element_ty),
        sem="relaxed",
        mask=(rows[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM),
    )
    tl.atomic_add(
        dk_ptr
        + off_b * stride_dkb
        + off_h * stride_dkh
        + cols[:, None] * stride_dkn
        + offs_d[None, :] * stride_dkd,
        dk.to(dk_ptr.dtype.element_ty),
        sem="relaxed",
        mask=(cols[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM),
    )
    tl.atomic_add(
        dv_ptr
        + off_b * stride_dvb
        + off_h * stride_dvh
        + cols[:, None] * stride_dvn
        + offs_d[None, :] * stride_dvd,
        dv.to(dv_ptr.dtype.element_ty),
        sem="relaxed",
        mask=(cols[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM),
    )


@triton.jit
def _sdpa_bwd_fused_atomic_causal_exact_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    do_ptr,
    stats_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    attn_scale,
    HQ: tl.constexpr,
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
    stride_dqb: tl.constexpr,
    stride_dqh: tl.constexpr,
    stride_dqm: tl.constexpr,
    stride_dqd: tl.constexpr,
    stride_dkb: tl.constexpr,
    stride_dkh: tl.constexpr,
    stride_dkn: tl.constexpr,
    stride_dkd: tl.constexpr,
    stride_dvb: tl.constexpr,
    stride_dvh: tl.constexpr,
    stride_dvn: tl.constexpr,
    stride_dvd: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BANDED: tl.constexpr,
    CAUSAL_TOP_LEFT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bh = tl.program_id(2)
    off_b = pid_bh // HQ
    off_h = pid_bh % HQ

    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N
    if start_n > start_m + BLOCK_M - 1:
        return

    rows = start_m + tl.arange(0, BLOCK_M)
    cols = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_h * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_h * stride_vh
    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh
    do_base = do_ptr + off_b * stride_dob + off_h * stride_doh
    stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh

    q_tile = tl.load(
        q_base + rows[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        eviction_policy="evict_last",
    )
    k_tile = tl.load(
        k_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd,
        eviction_policy="evict_last",
    )
    v_tile = tl.load(
        v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd,
        eviction_policy="evict_last",
    )
    o_tile = tl.load(
        o_base + rows[:, None] * stride_om + offs_d[None, :] * stride_od,
        eviction_policy="evict_last",
    ).to(tl.float32)
    do_tile = tl.load(
        do_base + rows[:, None] * stride_dom + offs_d[None, :] * stride_dod,
        eviction_policy="evict_last",
    )
    stats = tl.load(
        stats_base + rows * stride_sm, eviction_policy="evict_last"
    ).to(tl.float32)
    stats_log2 = stats * 1.4426950408889634

    score = tl.dot(q_tile, tl.trans(k_tile)).to(tl.float32) * (
        attn_scale * 1.4426950408889634
    )
    if start_n + BLOCK_N <= start_m:
        p = tl.exp2(score - stats_log2[:, None])
    else:
        valid = cols[None, :] <= rows[:, None]
        p = tl.where(valid, tl.exp2(score - stats_log2[:, None]), 0.0)
    dp = tl.dot(do_tile, tl.trans(v_tile)).to(tl.float32)
    delta = tl.sum(o_tile * do_tile.to(tl.float32), axis=1)
    ds = p * (dp - delta[:, None])

    dq = tl.dot(ds.to(k_tile.dtype), k_tile) * attn_scale
    dk = tl.dot(tl.trans(ds).to(q_tile.dtype), q_tile) * attn_scale
    dv = tl.dot(tl.trans(p).to(do_tile.dtype), do_tile)

    tl.atomic_add(
        dq_ptr
        + off_b * stride_dqb
        + off_h * stride_dqh
        + rows[:, None] * stride_dqm
        + offs_d[None, :] * stride_dqd,
        dq.to(dq_ptr.dtype.element_ty),
        sem="relaxed",
    )
    tl.atomic_add(
        dk_ptr
        + off_b * stride_dkb
        + off_h * stride_dkh
        + cols[:, None] * stride_dkn
        + offs_d[None, :] * stride_dkd,
        dk.to(dk_ptr.dtype.element_ty),
        sem="relaxed",
    )
    tl.atomic_add(
        dv_ptr
        + off_b * stride_dvb
        + off_h * stride_dvh
        + cols[:, None] * stride_dvn
        + offs_d[None, :] * stride_dvd,
        dv.to(dv_ptr.dtype.element_ty),
        sem="relaxed",
    )


@triton.jit
def _sdpa_bwd_fused_atomic_causal_exact_delta_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    do_ptr,
    stats_ptr,
    delta_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    attn_scale,
    HQ: tl.constexpr,
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
    stride_delta_b: tl.constexpr,
    stride_delta_h: tl.constexpr,
    stride_delta_m: tl.constexpr,
    stride_dqb: tl.constexpr,
    stride_dqh: tl.constexpr,
    stride_dqm: tl.constexpr,
    stride_dqd: tl.constexpr,
    stride_dkb: tl.constexpr,
    stride_dkh: tl.constexpr,
    stride_dkn: tl.constexpr,
    stride_dkd: tl.constexpr,
    stride_dvb: tl.constexpr,
    stride_dvh: tl.constexpr,
    stride_dvn: tl.constexpr,
    stride_dvd: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BANDED: tl.constexpr,
    CAUSAL_TOP_LEFT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bh = tl.program_id(2)
    off_b = pid_bh // HQ
    off_h = pid_bh % HQ

    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N
    if start_n > start_m + BLOCK_M - 1:
        return

    rows = start_m + tl.arange(0, BLOCK_M)
    cols = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_h * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_h * stride_vh
    do_base = do_ptr + off_b * stride_dob + off_h * stride_doh
    stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh

    q_tile = tl.load(
        q_base + rows[:, None] * stride_qm + offs_d[None, :] * stride_qd
    )
    k_tile = tl.load(
        k_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd
    )
    v_tile = tl.load(
        v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd
    )
    do_tile = tl.load(
        do_base + rows[:, None] * stride_dom + offs_d[None, :] * stride_dod
    )
    stats = tl.load(stats_base + rows * stride_sm).to(tl.float32)
    stats_log2 = stats * 1.4426950408889634

    score = tl.dot(q_tile, tl.trans(k_tile)).to(tl.float32) * (
        attn_scale * 1.4426950408889634
    )
    if start_n + BLOCK_N <= start_m:
        p = tl.exp2(score - stats_log2[:, None])
    else:
        valid = cols[None, :] <= rows[:, None]
        p = tl.where(valid, tl.exp2(score - stats_log2[:, None]), 0.0)
    dp = tl.dot(do_tile, tl.trans(v_tile)).to(tl.float32)
    delta = tl.load(
        delta_ptr
        + off_b * stride_delta_b
        + off_h * stride_delta_h
        + rows * stride_delta_m
    ).to(tl.float32)
    ds = p * (dp - delta[:, None])

    dq = tl.dot(ds.to(k_tile.dtype), k_tile) * attn_scale
    dk = tl.dot(tl.trans(ds).to(q_tile.dtype), q_tile) * attn_scale
    dv = tl.dot(tl.trans(p).to(do_tile.dtype), do_tile)

    tl.atomic_add(
        dq_ptr
        + off_b * stride_dqb
        + off_h * stride_dqh
        + rows[:, None] * stride_dqm
        + offs_d[None, :] * stride_dqd,
        dq.to(dq_ptr.dtype.element_ty),
        sem="relaxed",
    )
    tl.atomic_add(
        dk_ptr
        + off_b * stride_dkb
        + off_h * stride_dkh
        + cols[:, None] * stride_dkn
        + offs_d[None, :] * stride_dkd,
        dk.to(dk_ptr.dtype.element_ty),
        sem="relaxed",
    )
    tl.atomic_add(
        dv_ptr
        + off_b * stride_dvb
        + off_h * stride_dvh
        + cols[:, None] * stride_dvn
        + offs_d[None, :] * stride_dvd,
        dv.to(dv_ptr.dtype.element_ty),
        sem="relaxed",
    )


@triton.jit
def _sdpa_bwd_fused_atomic_gqa_causal_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    do_ptr,
    stats_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    attn_scale,
    HQ: tl.constexpr,
    Q_PER: tl.constexpr,
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
    stride_dqb: tl.constexpr,
    stride_dqh: tl.constexpr,
    stride_dqm: tl.constexpr,
    stride_dqd: tl.constexpr,
    stride_dkb: tl.constexpr,
    stride_dkh: tl.constexpr,
    stride_dkn: tl.constexpr,
    stride_dkd: tl.constexpr,
    stride_dvb: tl.constexpr,
    stride_dvh: tl.constexpr,
    stride_dvn: tl.constexpr,
    stride_dvd: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BANDED: tl.constexpr,
    CAUSAL_TOP_LEFT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bh = tl.program_id(2)
    off_b = pid_bh // HQ
    off_h = pid_bh % HQ
    off_kh = off_h // Q_PER

    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N
    if CAUSAL_TOP_LEFT and start_n > start_m + BLOCK_M - 1:
        return

    rows = start_m + tl.arange(0, BLOCK_M)
    cols = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_kh * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_kh * stride_vh
    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh
    do_base = do_ptr + off_b * stride_dob + off_h * stride_doh
    stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh

    q_tile = tl.load(
        q_base + rows[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=(rows[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    k_tile = tl.load(
        k_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd,
        mask=(cols[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    v_tile = tl.load(
        v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd,
        mask=(cols[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    o_tile = tl.load(
        o_base + rows[:, None] * stride_om + offs_d[None, :] * stride_od,
        mask=(rows[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    do_tile = tl.load(
        do_base + rows[:, None] * stride_dom + offs_d[None, :] * stride_dod,
        mask=(rows[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    stats = tl.load(
        stats_base + rows * stride_sm,
        mask=rows < SQ,
        other=float("-inf"),
    ).to(tl.float32)
    stats_log2 = stats * 1.4426950408889634

    score = tl.dot(q_tile, tl.trans(k_tile)).to(tl.float32) * (
        attn_scale * 1.4426950408889634
    )
    full_tile = start_n + BLOCK_N <= start_m
    full_tile = full_tile & (start_m + BLOCK_M <= SQ)
    full_tile = full_tile & (start_n + BLOCK_N <= SKV)
    if CAUSAL_TOP_LEFT and not BANDED and full_tile:
        p = tl.exp2(score - stats_log2[:, None])
    else:
        valid = (rows[:, None] < SQ) & (cols[None, :] < SKV)
        if BANDED:
            diag = cols[None, :] - rows[:, None]
            valid = valid & (diag >= min_diag) & (diag <= max_diag)
        p = tl.where(valid, tl.exp2(score - stats_log2[:, None]), 0.0)
    dp = tl.dot(do_tile, tl.trans(v_tile)).to(tl.float32)
    delta = tl.sum(o_tile * do_tile.to(tl.float32), axis=1)
    ds = p * (dp - delta[:, None])

    dq = tl.dot(ds.to(k_tile.dtype), k_tile) * attn_scale
    dk = tl.dot(tl.trans(ds).to(q_tile.dtype), q_tile) * attn_scale
    dv = tl.dot(tl.trans(p).to(do_tile.dtype), do_tile)

    tl.atomic_add(
        dq_ptr
        + off_b * stride_dqb
        + off_h * stride_dqh
        + rows[:, None] * stride_dqm
        + offs_d[None, :] * stride_dqd,
        dq.to(dq_ptr.dtype.element_ty),
        sem="relaxed",
        mask=(rows[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM),
    )
    tl.atomic_add(
        dk_ptr
        + off_b * stride_dkb
        + off_kh * stride_dkh
        + cols[:, None] * stride_dkn
        + offs_d[None, :] * stride_dkd,
        dk.to(dk_ptr.dtype.element_ty),
        sem="relaxed",
        mask=(cols[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM),
    )
    tl.atomic_add(
        dv_ptr
        + off_b * stride_dvb
        + off_kh * stride_dvh
        + cols[:, None] * stride_dvn
        + offs_d[None, :] * stride_dvd,
        dv.to(dv_ptr.dtype.element_ty),
        sem="relaxed",
        mask=(cols[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM),
    )


@triton.jit
def _sdpa_bwd_fused_atomic_causal_tri_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    do_ptr,
    stats_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    attn_scale,
    HQ: tl.constexpr,
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
    stride_dqb: tl.constexpr,
    stride_dqh: tl.constexpr,
    stride_dqm: tl.constexpr,
    stride_dqd: tl.constexpr,
    stride_dkb: tl.constexpr,
    stride_dkh: tl.constexpr,
    stride_dkn: tl.constexpr,
    stride_dkd: tl.constexpr,
    stride_dvb: tl.constexpr,
    stride_dvh: tl.constexpr,
    stride_dvn: tl.constexpr,
    stride_dvd: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BANDED: tl.constexpr,
    CAUSAL_TOP_LEFT: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_m = tl.full((), 0, tl.int64)
    pid_n = tl.full((), 0, tl.int64)
    row_base = 0
    for row in tl.static_range(0, NUM_BLOCKS):
        in_row = (pid_t >= row_base) & (pid_t < row_base + row + 1)
        pid_m = tl.where(in_row, row, pid_m)
        pid_n = tl.where(in_row, pid_t - row_base, pid_n)
        row_base += row + 1
    off_b = pid_bh // HQ
    off_h = pid_bh % HQ

    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N
    rows = start_m + tl.arange(0, BLOCK_M)
    cols = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_h * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_h * stride_vh
    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh
    do_base = do_ptr + off_b * stride_dob + off_h * stride_doh
    stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh

    q_tile = tl.load(
        q_base + rows[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=(rows[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    k_tile = tl.load(
        k_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd,
        mask=(cols[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    v_tile = tl.load(
        v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd,
        mask=(cols[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    o_tile = tl.load(
        o_base + rows[:, None] * stride_om + offs_d[None, :] * stride_od,
        mask=(rows[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    do_tile = tl.load(
        do_base + rows[:, None] * stride_dom + offs_d[None, :] * stride_dod,
        mask=(rows[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    stats = tl.load(
        stats_base + rows * stride_sm,
        mask=rows < SQ,
        other=float("-inf"),
    ).to(tl.float32)
    stats_log2 = stats * 1.4426950408889634

    score = tl.dot(q_tile, tl.trans(k_tile)).to(tl.float32) * (
        attn_scale * 1.4426950408889634
    )
    full_tile = start_n + BLOCK_N <= start_m
    full_tile = full_tile & (start_m + BLOCK_M <= SQ)
    full_tile = full_tile & (start_n + BLOCK_N <= SKV)
    if CAUSAL_TOP_LEFT and not BANDED and full_tile:
        p = tl.exp2(score - stats_log2[:, None])
    else:
        valid = (rows[:, None] < SQ) & (cols[None, :] < SKV)
        if BANDED:
            diag = cols[None, :] - rows[:, None]
            valid = valid & (diag >= min_diag) & (diag <= max_diag)
        p = tl.where(valid, tl.exp2(score - stats_log2[:, None]), 0.0)
    dp = tl.dot(do_tile, tl.trans(v_tile)).to(tl.float32)
    delta = tl.sum(o_tile * do_tile.to(tl.float32), axis=1)
    ds = p * (dp - delta[:, None])

    dq = tl.dot(ds.to(k_tile.dtype), k_tile) * attn_scale
    dk = tl.dot(tl.trans(ds).to(q_tile.dtype), q_tile) * attn_scale
    dv = tl.dot(tl.trans(p).to(do_tile.dtype), do_tile)

    tl.atomic_add(
        dq_ptr
        + off_b * stride_dqb
        + off_h * stride_dqh
        + rows[:, None] * stride_dqm
        + offs_d[None, :] * stride_dqd,
        dq.to(dq_ptr.dtype.element_ty),
        sem="relaxed",
        mask=(rows[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM),
    )
    tl.atomic_add(
        dk_ptr
        + off_b * stride_dkb
        + off_h * stride_dkh
        + cols[:, None] * stride_dkn
        + offs_d[None, :] * stride_dkd,
        dk.to(dk_ptr.dtype.element_ty),
        sem="relaxed",
        mask=(cols[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM),
    )
    tl.atomic_add(
        dv_ptr
        + off_b * stride_dvb
        + off_h * stride_dvh
        + cols[:, None] * stride_dvn
        + offs_d[None, :] * stride_dvd,
        dv.to(dv_ptr.dtype.element_ty),
        sem="relaxed",
        mask=(cols[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM),
    )


@triton.jit
def _sdpa_bwd_fused_atomic_causal_delta_tri_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    do_ptr,
    stats_ptr,
    delta_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    attn_scale,
    HQ: tl.constexpr,
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
    stride_delta_b: tl.constexpr,
    stride_delta_h: tl.constexpr,
    stride_delta_m: tl.constexpr,
    stride_dqb: tl.constexpr,
    stride_dqh: tl.constexpr,
    stride_dqm: tl.constexpr,
    stride_dqd: tl.constexpr,
    stride_dkb: tl.constexpr,
    stride_dkh: tl.constexpr,
    stride_dkn: tl.constexpr,
    stride_dkd: tl.constexpr,
    stride_dvb: tl.constexpr,
    stride_dvh: tl.constexpr,
    stride_dvn: tl.constexpr,
    stride_dvd: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BANDED: tl.constexpr,
    CAUSAL_TOP_LEFT: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_m = tl.full((), 0, tl.int64)
    pid_n = tl.full((), 0, tl.int64)
    row_base = 0
    for row in tl.static_range(0, NUM_BLOCKS):
        in_row = (pid_t >= row_base) & (pid_t < row_base + row + 1)
        pid_m = tl.where(in_row, row, pid_m)
        pid_n = tl.where(in_row, pid_t - row_base, pid_n)
        row_base += row + 1
    off_b = pid_bh // HQ
    off_h = pid_bh % HQ

    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N
    rows = start_m + tl.arange(0, BLOCK_M)
    cols = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_h * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_h * stride_vh
    do_base = do_ptr + off_b * stride_dob + off_h * stride_doh
    stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh

    q_tile = tl.load(
        q_base + rows[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=(rows[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    k_tile = tl.load(
        k_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd,
        mask=(cols[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    v_tile = tl.load(
        v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd,
        mask=(cols[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    do_tile = tl.load(
        do_base + rows[:, None] * stride_dom + offs_d[None, :] * stride_dod,
        mask=(rows[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    stats = tl.load(
        stats_base + rows * stride_sm,
        mask=rows < SQ,
        other=float("-inf"),
    ).to(tl.float32)
    stats_log2 = stats * 1.4426950408889634

    score = tl.dot(q_tile, tl.trans(k_tile)).to(tl.float32) * (
        attn_scale * 1.4426950408889634
    )
    full_tile = start_n + BLOCK_N <= start_m
    full_tile = full_tile & (start_m + BLOCK_M <= SQ)
    full_tile = full_tile & (start_n + BLOCK_N <= SKV)
    if CAUSAL_TOP_LEFT and not BANDED and full_tile:
        p = tl.exp2(score - stats_log2[:, None])
    else:
        valid = (rows[:, None] < SQ) & (cols[None, :] < SKV)
        if BANDED:
            diag = cols[None, :] - rows[:, None]
            valid = valid & (diag >= min_diag) & (diag <= max_diag)
        p = tl.where(valid, tl.exp2(score - stats_log2[:, None]), 0.0)
    dp = tl.dot(do_tile, tl.trans(v_tile)).to(tl.float32)
    delta = tl.load(
        delta_ptr
        + off_b * stride_delta_b
        + off_h * stride_delta_h
        + rows * stride_delta_m,
        mask=rows < SQ,
        other=0.0,
    ).to(tl.float32)
    ds = p * (dp - delta[:, None])

    dq = tl.dot(ds.to(k_tile.dtype), k_tile) * attn_scale
    dk = tl.dot(tl.trans(ds).to(q_tile.dtype), q_tile) * attn_scale
    dv = tl.dot(tl.trans(p).to(do_tile.dtype), do_tile)

    tl.atomic_add(
        dq_ptr
        + off_b * stride_dqb
        + off_h * stride_dqh
        + rows[:, None] * stride_dqm
        + offs_d[None, :] * stride_dqd,
        dq.to(dq_ptr.dtype.element_ty),
        sem="relaxed",
        mask=(rows[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM),
    )
    tl.atomic_add(
        dk_ptr
        + off_b * stride_dkb
        + off_h * stride_dkh
        + cols[:, None] * stride_dkn
        + offs_d[None, :] * stride_dkd,
        dk.to(dk_ptr.dtype.element_ty),
        sem="relaxed",
        mask=(cols[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM),
    )
    tl.atomic_add(
        dv_ptr
        + off_b * stride_dvb
        + off_h * stride_dvh
        + cols[:, None] * stride_dvn
        + offs_d[None, :] * stride_dvd,
        dv.to(dv_ptr.dtype.element_ty),
        sem="relaxed",
        mask=(cols[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM),
    )


@triton.jit
def _sdpa_bwd_fused_atomic_gqa_causal_tri_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    do_ptr,
    stats_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    attn_scale,
    HQ: tl.constexpr,
    Q_PER: tl.constexpr,
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
    stride_dqb: tl.constexpr,
    stride_dqh: tl.constexpr,
    stride_dqm: tl.constexpr,
    stride_dqd: tl.constexpr,
    stride_dkb: tl.constexpr,
    stride_dkh: tl.constexpr,
    stride_dkn: tl.constexpr,
    stride_dkd: tl.constexpr,
    stride_dvb: tl.constexpr,
    stride_dvh: tl.constexpr,
    stride_dvn: tl.constexpr,
    stride_dvd: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BANDED: tl.constexpr,
    CAUSAL_TOP_LEFT: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_m = tl.full((), 0, tl.int64)
    pid_n = tl.full((), 0, tl.int64)
    row_base = 0
    for row in tl.static_range(0, NUM_BLOCKS):
        in_row = (pid_t >= row_base) & (pid_t < row_base + row + 1)
        pid_m = tl.where(in_row, row, pid_m)
        pid_n = tl.where(in_row, pid_t - row_base, pid_n)
        row_base += row + 1
    off_b = pid_bh // HQ
    off_h = pid_bh % HQ
    off_kh = off_h // Q_PER

    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N
    rows = start_m + tl.arange(0, BLOCK_M)
    cols = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_kh * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_kh * stride_vh
    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh
    do_base = do_ptr + off_b * stride_dob + off_h * stride_doh
    stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh

    q_tile = tl.load(
        q_base + rows[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=(rows[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    k_tile = tl.load(
        k_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd,
        mask=(cols[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    v_tile = tl.load(
        v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd,
        mask=(cols[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    o_tile = tl.load(
        o_base + rows[:, None] * stride_om + offs_d[None, :] * stride_od,
        mask=(rows[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    do_tile = tl.load(
        do_base + rows[:, None] * stride_dom + offs_d[None, :] * stride_dod,
        mask=(rows[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    stats = tl.load(
        stats_base + rows * stride_sm,
        mask=rows < SQ,
        other=float("-inf"),
    ).to(tl.float32)
    stats_log2 = stats * 1.4426950408889634

    score = tl.dot(q_tile, tl.trans(k_tile)).to(tl.float32) * (
        attn_scale * 1.4426950408889634
    )
    full_tile = start_n + BLOCK_N <= start_m
    full_tile = full_tile & (start_m + BLOCK_M <= SQ)
    full_tile = full_tile & (start_n + BLOCK_N <= SKV)
    if CAUSAL_TOP_LEFT and not BANDED and full_tile:
        p = tl.exp2(score - stats_log2[:, None])
    else:
        valid = (rows[:, None] < SQ) & (cols[None, :] < SKV)
        if BANDED:
            diag = cols[None, :] - rows[:, None]
            valid = valid & (diag >= min_diag) & (diag <= max_diag)
        p = tl.where(valid, tl.exp2(score - stats_log2[:, None]), 0.0)
    dp = tl.dot(do_tile, tl.trans(v_tile)).to(tl.float32)
    delta = tl.sum(o_tile * do_tile.to(tl.float32), axis=1)
    ds = p * (dp - delta[:, None])

    dq = tl.dot(ds.to(k_tile.dtype), k_tile) * attn_scale
    dk = tl.dot(tl.trans(ds).to(q_tile.dtype), q_tile) * attn_scale
    dv = tl.dot(tl.trans(p).to(do_tile.dtype), do_tile)

    tl.atomic_add(
        dq_ptr
        + off_b * stride_dqb
        + off_h * stride_dqh
        + rows[:, None] * stride_dqm
        + offs_d[None, :] * stride_dqd,
        dq.to(dq_ptr.dtype.element_ty),
        sem="relaxed",
        mask=(rows[:, None] < SQ) & (offs_d[None, :] < HEAD_DIM),
    )
    tl.atomic_add(
        dk_ptr
        + off_b * stride_dkb
        + off_kh * stride_dkh
        + cols[:, None] * stride_dkn
        + offs_d[None, :] * stride_dkd,
        dk.to(dk_ptr.dtype.element_ty),
        sem="relaxed",
        mask=(cols[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM),
    )
    tl.atomic_add(
        dv_ptr
        + off_b * stride_dvb
        + off_kh * stride_dvh
        + cols[:, None] * stride_dvn
        + offs_d[None, :] * stride_dvd,
        dv.to(dv_ptr.dtype.element_ty),
        sem="relaxed",
        mask=(cols[:, None] < SKV) & (offs_d[None, :] < HEAD_DIM),
    )


@triton.jit
def _sdpa_bwd_mloop_causal_d128_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    do_ptr,
    stats_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    attn_scale,
    HQ: tl.constexpr,
    Q_PER: tl.constexpr,
    SQ: tl.constexpr,
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
    stride_dkb: tl.constexpr,
    stride_dkh: tl.constexpr,
    stride_dkn: tl.constexpr,
    stride_dkd: tl.constexpr,
    stride_dvb: tl.constexpr,
    stride_dvh: tl.constexpr,
    stride_dvn: tl.constexpr,
    stride_dvd: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // HQ
    off_h = pid_bh % HQ
    off_kh = off_h // Q_PER
    start_m = pid_m * BLOCK_M
    rows = start_m + tl.arange(0, BLOCK_M)
    cols_base = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_kh * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_kh * stride_vh
    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh
    do_base = do_ptr + off_b * stride_dob + off_h * stride_doh
    stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh

    q_tile = tl.load(
        q_base + rows[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        eviction_policy="evict_last",
    )
    do_tile = tl.load(
        do_base + rows[:, None] * stride_dom + offs_d[None, :] * stride_dod,
        eviction_policy="evict_last",
    )
    o_tile = tl.load(
        o_base + rows[:, None] * stride_om + offs_d[None, :] * stride_od,
        eviction_policy="evict_last",
    ).to(tl.float32)
    delta = tl.sum(o_tile * do_tile.to(tl.float32), axis=1)
    stats = tl.load(
        stats_base + rows * stride_sm, eviction_policy="evict_last"
    ).to(tl.float32)
    stats_log2 = stats * 1.4426950408889634
    dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in tl.range(0, start_m + BLOCK_M, BLOCK_N, disable_licm=True):
        cols = start_n + cols_base
        k_tile = tl.load(
            k_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            eviction_policy="evict_last",
        )
        v_tile = tl.load(
            v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            eviction_policy="evict_last",
        )
        score = tl.dot(q_tile, tl.trans(k_tile)).to(tl.float32) * (
            attn_scale * 1.4426950408889634
        )
        if start_n + BLOCK_N <= start_m:
            p = tl.exp2(score - stats_log2[:, None])
        else:
            valid = cols[None, :] <= rows[:, None]
            p = tl.where(valid, tl.exp2(score - stats_log2[:, None]), 0.0)
        dp = tl.dot(do_tile, tl.trans(v_tile)).to(tl.float32)
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k_tile.dtype), k_tile)
        dk = tl.dot(tl.trans(ds).to(q_tile.dtype), q_tile) * attn_scale
        dv = tl.dot(tl.trans(p).to(do_tile.dtype), do_tile)
        tl.atomic_add(
            dk_ptr
            + off_b * stride_dkb
            + off_kh * stride_dkh
            + cols[:, None] * stride_dkn
            + offs_d[None, :] * stride_dkd,
            dk.to(dk_ptr.dtype.element_ty),
            sem="relaxed",
        )
        tl.atomic_add(
            dv_ptr
            + off_b * stride_dvb
            + off_kh * stride_dvh
            + cols[:, None] * stride_dvn
            + offs_d[None, :] * stride_dvd,
            dv.to(dv_ptr.dtype.element_ty),
            sem="relaxed",
        )

    tl.store(
        dq_ptr
        + off_b * stride_dqb
        + off_h * stride_dqh
        + rows[:, None] * stride_dqm
        + offs_d[None, :] * stride_dqd,
        (dq * attn_scale).to(dq_ptr.dtype.element_ty),
    )


@triton.jit
def _sdpa_bwd_dense_mloop_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    do_ptr,
    stats_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    attn_scale,
    HQ: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
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
    stride_dkb: tl.constexpr,
    stride_dkh: tl.constexpr,
    stride_dkn: tl.constexpr,
    stride_dkd: tl.constexpr,
    stride_dvb: tl.constexpr,
    stride_dvh: tl.constexpr,
    stride_dvn: tl.constexpr,
    stride_dvd: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // HQ
    off_h = pid_bh % HQ
    start_m = pid_m * BLOCK_M
    rows = start_m + tl.arange(0, BLOCK_M)
    cols_base = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_h * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_h * stride_vh
    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh
    do_base = do_ptr + off_b * stride_dob + off_h * stride_doh
    stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh

    q_tile = tl.load(
        q_base + rows[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        eviction_policy="evict_last",
    )
    do_tile = tl.load(
        do_base + rows[:, None] * stride_dom + offs_d[None, :] * stride_dod,
        eviction_policy="evict_last",
    )
    o_tile = tl.load(
        o_base + rows[:, None] * stride_om + offs_d[None, :] * stride_od,
        eviction_policy="evict_last",
    ).to(tl.float32)
    delta = tl.sum(o_tile * do_tile.to(tl.float32), axis=1)
    stats = tl.load(
        stats_base + rows * stride_sm, eviction_policy="evict_last"
    ).to(tl.float32)
    stats_log2 = stats * 1.4426950408889634
    dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in tl.range(0, SKV, BLOCK_N):
        cols = start_n + cols_base
        k_tile = tl.load(
            k_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            eviction_policy="evict_last",
        )
        v_tile = tl.load(
            v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            eviction_policy="evict_last",
        )
        score = tl.dot(q_tile, tl.trans(k_tile)).to(tl.float32) * (
            attn_scale * 1.4426950408889634
        )
        p = tl.exp2(score - stats_log2[:, None])
        dp = tl.dot(do_tile, tl.trans(v_tile)).to(tl.float32)
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k_tile.dtype), k_tile)
        dk = tl.dot(tl.trans(ds).to(q_tile.dtype), q_tile) * attn_scale
        dv = tl.dot(tl.trans(p).to(do_tile.dtype), do_tile)
        tl.atomic_add(
            dk_ptr
            + off_b * stride_dkb
            + off_h * stride_dkh
            + cols[:, None] * stride_dkn
            + offs_d[None, :] * stride_dkd,
            dk.to(dk_ptr.dtype.element_ty),
            sem="relaxed",
        )
        tl.atomic_add(
            dv_ptr
            + off_b * stride_dvb
            + off_h * stride_dvh
            + cols[:, None] * stride_dvn
            + offs_d[None, :] * stride_dvd,
            dv.to(dv_ptr.dtype.element_ty),
            sem="relaxed",
        )

    tl.store(
        dq_ptr
        + off_b * stride_dqb
        + off_h * stride_dqh
        + rows[:, None] * stride_dqm
        + offs_d[None, :] * stride_dqd,
        (dq * attn_scale).to(dq_ptr.dtype.element_ty),
    )


@triton.jit
def _sdpa_bwd_gqa_dq_delta_causal_d128_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    stats_ptr,
    delta_ptr,
    dq_ptr,
    attn_scale,
    HQ: tl.constexpr,
    Q_PER: tl.constexpr,
    SQ: tl.constexpr,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // HQ
    off_h = pid_bh % HQ
    off_kh = off_h // Q_PER
    start_m = pid_m * BLOCK_M
    rows = start_m + tl.arange(0, BLOCK_M)
    cols_base = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_kh * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_kh * stride_vh
    do_base = do_ptr + off_b * stride_dob + off_h * stride_doh
    stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh
    delta_base = delta_ptr + off_b * stride_delta_b + off_h * stride_delta_h
    q_tile = tl.load(
        q_base + rows[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        eviction_policy="evict_last",
    )
    do_tile = tl.load(
        do_base + rows[:, None] * stride_dom + offs_d[None, :] * stride_dod,
        eviction_policy="evict_last",
    )
    delta = tl.load(
        delta_base + rows * stride_delta_m, eviction_policy="evict_last"
    ).to(tl.float32)
    stats = tl.load(
        stats_base + rows * stride_sm, eviction_policy="evict_last"
    ).to(tl.float32)
    stats_log2 = stats * 1.4426950408889634
    dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for start_n in tl.range(0, start_m + BLOCK_M, BLOCK_N):
        cols = start_n + cols_base
        k_tile = tl.load(
            k_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            eviction_policy="evict_last",
        )
        v_tile = tl.load(
            v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            eviction_policy="evict_last",
        )
        score = tl.dot(q_tile, tl.trans(k_tile)).to(tl.float32) * (
            attn_scale * 1.4426950408889634
        )
        if start_n + BLOCK_N <= start_m:
            p = tl.exp2(score - stats_log2[:, None])
        else:
            valid = cols[None, :] <= rows[:, None]
            p = tl.where(valid, tl.exp2(score - stats_log2[:, None]), 0.0)
        dp = tl.dot(do_tile, tl.trans(v_tile)).to(tl.float32)
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k_tile.dtype), k_tile)
    tl.store(
        dq_ptr
        + off_b * stride_dqb
        + off_h * stride_dqh
        + rows[:, None] * stride_dqm
        + offs_d[None, :] * stride_dqd,
        (dq * attn_scale).to(dq_ptr.dtype.element_ty),
    )


@triton.jit
def _sdpa_bwd_gqa_dq_store_delta_causal_d128_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    do_ptr,
    stats_ptr,
    delta_ptr,
    dq_ptr,
    attn_scale,
    HQ: tl.constexpr,
    Q_PER: tl.constexpr,
    SQ: tl.constexpr,
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
    stride_delta_b: tl.constexpr,
    stride_delta_h: tl.constexpr,
    stride_delta_m: tl.constexpr,
    stride_dqb: tl.constexpr,
    stride_dqh: tl.constexpr,
    stride_dqm: tl.constexpr,
    stride_dqd: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // HQ
    off_h = pid_bh % HQ
    off_kh = off_h // Q_PER
    start_m = pid_m * BLOCK_M
    rows = start_m + tl.arange(0, BLOCK_M)
    cols_base = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_kh * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_kh * stride_vh
    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh
    do_base = do_ptr + off_b * stride_dob + off_h * stride_doh
    stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh
    delta_base = delta_ptr + off_b * stride_delta_b + off_h * stride_delta_h
    q_tile = tl.load(
        q_base + rows[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        eviction_policy="evict_last",
    )
    do_tile = tl.load(
        do_base + rows[:, None] * stride_dom + offs_d[None, :] * stride_dod,
        eviction_policy="evict_last",
    )
    o_tile = tl.load(
        o_base + rows[:, None] * stride_om + offs_d[None, :] * stride_od,
        eviction_policy="evict_last",
    ).to(tl.float32)
    delta = tl.sum(o_tile * do_tile.to(tl.float32), axis=1)
    tl.store(delta_base + rows * stride_delta_m, delta)
    stats = tl.load(
        stats_base + rows * stride_sm, eviction_policy="evict_last"
    ).to(tl.float32)
    stats_log2 = stats * 1.4426950408889634
    dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for start_n in tl.range(0, start_m + BLOCK_M, BLOCK_N):
        cols = start_n + cols_base
        k_tile = tl.load(
            k_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            eviction_policy="evict_last",
        )
        v_tile = tl.load(
            v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            eviction_policy="evict_last",
        )
        score = tl.dot(q_tile, tl.trans(k_tile)).to(tl.float32) * (
            attn_scale * 1.4426950408889634
        )
        if start_n + BLOCK_N <= start_m:
            p = tl.exp2(score - stats_log2[:, None])
        else:
            valid = cols[None, :] <= rows[:, None]
            p = tl.where(valid, tl.exp2(score - stats_log2[:, None]), 0.0)
        dp = tl.dot(do_tile, tl.trans(v_tile)).to(tl.float32)
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k_tile.dtype), k_tile)
    tl.store(
        dq_ptr
        + off_b * stride_dqb
        + off_h * stride_dqh
        + rows[:, None] * stride_dqm
        + offs_d[None, :] * stride_dqd,
        (dq * attn_scale).to(dq_ptr.dtype.element_ty),
    )


@triton.jit
def _sdpa_bwd_gqa_dkdv_atomic_causal_d128_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    stats_ptr,
    delta_ptr,
    dk_ptr,
    dv_ptr,
    attn_scale,
    HQ: tl.constexpr,
    Q_PER: tl.constexpr,
    SQ: tl.constexpr,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // HQ
    off_h = pid_bh % HQ
    off_kh = off_h // Q_PER
    start_n = pid_n * BLOCK_N
    cols = start_n + tl.arange(0, BLOCK_N)
    rows_base = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    k_base = k_ptr + off_b * stride_kb + off_kh * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_kh * stride_vh
    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    do_base = do_ptr + off_b * stride_dob + off_h * stride_doh
    stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh
    delta_base = delta_ptr + off_b * stride_delta_b + off_h * stride_delta_h
    k_tile = tl.load(
        k_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd,
        eviction_policy="evict_last",
    )
    v_tile = tl.load(
        v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd,
        eviction_policy="evict_last",
    )
    dk = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    for start_m in tl.range((start_n // BLOCK_M) * BLOCK_M, SQ, BLOCK_M):
        rows = start_m + rows_base
        q_tile = tl.load(
            q_base + rows[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            eviction_policy="evict_last",
        )
        do_tile = tl.load(
            do_base
            + rows[:, None] * stride_dom
            + offs_d[None, :] * stride_dod,
            eviction_policy="evict_last",
        )
        stats = tl.load(
            stats_base + rows * stride_sm, eviction_policy="evict_last"
        ).to(tl.float32)
        stats_log2 = stats * 1.4426950408889634
        delta = tl.load(
            delta_base + rows * stride_delta_m, eviction_policy="evict_last"
        ).to(tl.float32)
        score = tl.dot(q_tile, tl.trans(k_tile)).to(tl.float32) * (
            attn_scale * 1.4426950408889634
        )
        if start_m >= start_n + BLOCK_N:
            p = tl.exp2(score - stats_log2[:, None])
        else:
            valid = cols[None, :] <= rows[:, None]
            p = tl.where(valid, tl.exp2(score - stats_log2[:, None]), 0.0)
        dp = tl.dot(do_tile, tl.trans(v_tile)).to(tl.float32)
        ds = p * (dp - delta[:, None])
        dk += tl.dot(tl.trans(ds).to(q_tile.dtype), q_tile)
        dv += tl.dot(tl.trans(p).to(do_tile.dtype), do_tile)
    tl.atomic_add(
        dk_ptr
        + off_b * stride_dkb
        + off_kh * stride_dkh
        + cols[:, None] * stride_dkn
        + offs_d[None, :] * stride_dkd,
        (dk * attn_scale).to(dk_ptr.dtype.element_ty),
        sem="relaxed",
    )
    tl.atomic_add(
        dv_ptr
        + off_b * stride_dvb
        + off_kh * stride_dvh
        + cols[:, None] * stride_dvn
        + offs_d[None, :] * stride_dvd,
        dv.to(dv_ptr.dtype.element_ty),
        sem="relaxed",
    )


@triton.jit
def _sdpa_bwd_decode_dkdv_dq_atomic_kernel(
    q,
    k,
    v,
    o,
    do,
    stats,
    dq,
    dk,
    dv,
    attn_scale,
    HQ: tl.constexpr,
    SKV: tl.constexpr,
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
    stride_dkb: tl.constexpr,
    stride_dkh: tl.constexpr,
    stride_dkn: tl.constexpr,
    stride_dkd: tl.constexpr,
    stride_dvb: tl.constexpr,
    stride_dvh: tl.constexpr,
    stride_dvn: tl.constexpr,
    stride_dvd: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // HQ
    off_h = pid_bh % HQ
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = q + off_b * stride_qb + off_h * stride_qh
    k_base = k + off_b * stride_kb + off_h * stride_kh
    v_base = v + off_b * stride_vb + off_h * stride_vh
    o_base = o + off_b * stride_ob + off_h * stride_oh
    do_base = do + off_b * stride_dob + off_h * stride_doh

    qv = tl.load(q_base + offs_d * stride_qd, eviction_policy="evict_last")
    dov = tl.load(do_base + offs_d * stride_dod, eviction_policy="evict_last")
    ov = tl.load(o_base + offs_d * stride_od, eviction_policy="evict_last").to(
        tl.float32
    )
    delta = tl.sum(ov * dov.to(tl.float32), axis=0)
    st = tl.load(stats + off_b * stride_sb + off_h * stride_sh).to(tl.float32)
    st_log2 = st * 1.4426950408889634

    kt = tl.load(
        k_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd,
        mask=cols[:, None] < SKV,
        other=0.0,
        eviction_policy="evict_last",
    )
    vt = tl.load(
        v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd,
        mask=cols[:, None] < SKV,
        other=0.0,
        eviction_policy="evict_last",
    )
    score = tl.sum(kt.to(tl.float32) * qv[None, :].to(tl.float32), axis=1) * (
        attn_scale * 1.4426950408889634
    )
    p = tl.where(cols < SKV, tl.exp2(score - st_log2), 0.0)
    dp = tl.sum(vt.to(tl.float32) * dov[None, :].to(tl.float32), axis=1)
    ds = p * (dp - delta)
    dq_partial = tl.sum(ds[:, None] * kt.to(tl.float32), axis=0) * attn_scale
    mask = cols[:, None] < SKV
    tl.store(
        dk
        + off_b * stride_dkb
        + off_h * stride_dkh
        + cols[:, None] * stride_dkn
        + offs_d[None, :] * stride_dkd,
        (ds[:, None] * qv[None, :].to(tl.float32) * attn_scale).to(
            dk.dtype.element_ty
        ),
        mask=mask,
    )
    tl.store(
        dv
        + off_b * stride_dvb
        + off_h * stride_dvh
        + cols[:, None] * stride_dvn
        + offs_d[None, :] * stride_dvd,
        (p[:, None] * dov[None, :].to(tl.float32)).to(dv.dtype.element_ty),
        mask=mask,
    )
    tl.atomic_add(
        dq + off_b * stride_dqb + off_h * stride_dqh + offs_d * stride_dqd,
        dq_partial.to(dq.dtype.element_ty),
        sem="relaxed",
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
def _sdpa_fp8_fast_inner(
    acc,
    l_i,
    m_i,
    q,
    k_base,
    v_base,
    qk_scale,
    s_scale,
    offs_m,
    offs_d,
    offs_dv,
    lo,
    hi,
    SKV,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vd: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL_MASK: tl.constexpr,
    TAIL_MASK: tl.constexpr,
):
    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        if CAUSAL_MASK or TAIL_MASK:
            k = tl.load(
                k_base
                + offs_d[:, None] * stride_kd
                + offs_n[None, :] * stride_kn,
                mask=offs_n[None, :] < SKV,
                other=0.0,
            )
        else:
            k = tl.load(
                k_base
                + offs_d[:, None] * stride_kd
                + offs_n[None, :] * stride_kn
            )
        score = tl.dot(q, k).to(tl.float32) * qk_scale
        if CAUSAL_MASK:
            score = tl.where(
                offs_n[None, :] <= offs_m[:, None], score, float("-inf")
            )
        if TAIL_MASK:
            score = tl.where(offs_n[None, :] < SKV, score, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(score, 1))
        if CAUSAL_MASK or TAIL_MASK:
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        else:
            m_safe = m_new
        p = tl.math.exp2(score - m_safe[:, None])
        alpha = tl.math.exp2(m_i - m_safe)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]
        if CAUSAL_MASK or TAIL_MASK:
            v = tl.load(
                v_base
                + offs_n[:, None] * stride_vn
                + offs_dv[None, :] * stride_vd,
                mask=offs_n[:, None] < SKV,
                other=0.0,
            )
        else:
            v = tl.load(
                v_base
                + offs_n[:, None] * stride_vn
                + offs_dv[None, :] * stride_vd
            )
        acc = tl.dot((p * s_scale).to(v.dtype), v, acc)
        l_i = l_i * alpha + l_ij
        m_i = m_new
    return acc, l_i, m_i


@triton.jit
def _sdpa_fp8_fwd_fast_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    stats_ptr,
    amax_s_ptr,
    amax_o_ptr,
    qk_scale,
    s_scale,
    sv_descale,
    o_scale,
    HQ: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    Q_PER_K: tl.constexpr,
    Q_PER_V: tl.constexpr,
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
    stride_sb: tl.constexpr,
    stride_sh: tl.constexpr,
    stride_sm: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    CAUSAL: tl.constexpr,
    GENERATE_STATS: tl.constexpr,
):
    raw_pid_m = tl.program_id(0)
    if CAUSAL:
        pid_m = tl.cdiv(SQ, BLOCK_M) - 1 - raw_pid_m
    else:
        pid_m = raw_pid_m
    pid_bh = tl.program_id(1)
    off_b = pid_bh // HQ
    off_h = pid_bh % HQ
    off_kh = off_h // Q_PER_K
    off_vh = off_h // Q_PER_V

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_m = offs_m < SQ

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_kh * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_vh * stride_vh
    if SQ % BLOCK_M == 0:
        q = tl.load(
            q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        )
    else:
        q = tl.load(
            q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=mask_m[:, None],
            other=0.0,
        )

    acc = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)

    if CAUSAL:
        hi = tl.minimum(start_m + BLOCK_M, SKV)
        full_hi = tl.minimum((start_m // BLOCK_N) * BLOCK_N, hi)
        if 0 < full_hi:
            acc, l_i, m_i = _sdpa_fp8_fast_inner(
                acc,
                l_i,
                m_i,
                q,
                k_base,
                v_base,
                qk_scale,
                s_scale,
                offs_m,
                offs_d,
                offs_dv,
                0,
                full_hi,
                SKV,
                stride_kn,
                stride_kd,
                stride_vn,
                stride_vd,
                BLOCK_N=BLOCK_N,
                CAUSAL_MASK=False,
                TAIL_MASK=False,
            )
        if full_hi < hi:
            acc, l_i, m_i = _sdpa_fp8_fast_inner(
                acc,
                l_i,
                m_i,
                q,
                k_base,
                v_base,
                qk_scale,
                s_scale,
                offs_m,
                offs_d,
                offs_dv,
                full_hi,
                hi,
                SKV,
                stride_kn,
                stride_kd,
                stride_vn,
                stride_vd,
                BLOCK_N=BLOCK_N,
                CAUSAL_MASK=True,
                TAIL_MASK=False,
            )
    else:
        full = (SKV // BLOCK_N) * BLOCK_N
        if 0 < full:
            acc, l_i, m_i = _sdpa_fp8_fast_inner(
                acc,
                l_i,
                m_i,
                q,
                k_base,
                v_base,
                qk_scale,
                s_scale,
                offs_m,
                offs_d,
                offs_dv,
                0,
                full,
                SKV,
                stride_kn,
                stride_kd,
                stride_vn,
                stride_vd,
                BLOCK_N=BLOCK_N,
                CAUSAL_MASK=False,
                TAIL_MASK=False,
            )
        if full < SKV:
            acc, l_i, m_i = _sdpa_fp8_fast_inner(
                acc,
                l_i,
                m_i,
                q,
                k_base,
                v_base,
                qk_scale,
                s_scale,
                offs_m,
                offs_d,
                offs_dv,
                full,
                SKV,
                SKV,
                stride_kn,
                stride_kd,
                stride_vn,
                stride_vd,
                BLOCK_N=BLOCK_N,
                CAUSAL_MASK=False,
                TAIL_MASK=True,
            )

    l_safe = tl.maximum(l_i, 1.0)
    o_val = acc * (sv_descale / l_safe[:, None])
    if SQ % BLOCK_M == 0:
        local_amax_o = tl.max(tl.abs(o_val))
    else:
        local_amax_o = tl.max(tl.where(mask_m[:, None], tl.abs(o_val), 0.0))
    tl.atomic_max(amax_o_ptr, local_amax_o, sem="relaxed")
    if SQ % BLOCK_M == 0:
        amax_s_val = tl.max(tl.abs(m_i)) * _LN2_KERNEL
    else:
        amax_s_val = tl.max(tl.where(mask_m, tl.abs(m_i), 0.0)) * _LN2_KERNEL
    tl.atomic_max(amax_s_ptr, amax_s_val, sem="relaxed")

    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh
    if SQ % BLOCK_M == 0:
        tl.store(
            o_base
            + offs_m[:, None] * stride_om
            + offs_dv[None, :] * stride_od,
            (o_val * o_scale).to(o_ptr.dtype.element_ty),
        )
    else:
        tl.store(
            o_base
            + offs_m[:, None] * stride_om
            + offs_dv[None, :] * stride_od,
            (o_val * o_scale).to(o_ptr.dtype.element_ty),
            mask=mask_m[:, None],
        )
    if GENERATE_STATS:
        stats = (m_i + tl.log2(l_safe)) * _LN2_KERNEL
        stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh
        if SQ % BLOCK_M == 0:
            tl.store(stats_base + offs_m * stride_sm, stats)
        else:
            tl.store(stats_base + offs_m * stride_sm, stats, mask=mask_m)


@triton.jit
def _sdpa_fp8_tma_inner(
    acc,
    l_i,
    m_i,
    q,
    k_desc,
    v_desc,
    qk_scale,
    s_scale,
    offs_m,
    lo,
    hi,
    SKV,
    BLOCK_N: tl.constexpr,
    CAUSAL_MASK: tl.constexpr,
    TAIL_MASK: tl.constexpr,
):
    for start_n in tl.range(lo, hi, BLOCK_N, disable_licm=True):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        start_n_i32 = start_n.to(tl.int32)  # type: ignore[attr-defined]
        offs_n = start_n_i32 + tl.arange(0, BLOCK_N)
        # TMA loads: hardware addressing, no manual pointer / transpose math.
        k = tl.trans(k_desc.load([start_n_i32, 0]))
        score = tl.dot(q, k).to(tl.float32) * qk_scale
        if CAUSAL_MASK:
            score = tl.where(
                offs_n[None, :] <= offs_m[:, None], score, float("-inf")
            )
        if TAIL_MASK:
            score = tl.where(offs_n[None, :] < SKV, score, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(score, 1))
        if CAUSAL_MASK or TAIL_MASK:
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        else:
            m_safe = m_new
        p = tl.math.exp2(score - m_safe[:, None])
        alpha = tl.math.exp2(m_i - m_safe)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = v_desc.load([start_n_i32, 0])
        acc = tl.dot((p * s_scale).to(v.dtype), v, acc)
        l_i = l_i * alpha + l_ij
        m_i = m_new
    return acc, l_i, m_i


@triton.jit
def _sdpa_fp8_fwd_dense_nostats_hostdesc_tma_kernel(
    q_ptr,
    k_desc,
    v_desc,
    o_ptr,
    amax_s_ptr,
    amax_o_ptr,
    qk_scale: tl.constexpr,
    s_scale: tl.constexpr,
    sv_descale: tl.constexpr,
    o_scale: tl.constexpr,
    SQ: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    COMPUTE_AMAX: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    start_m = pid_m * BLOCK_M
    offs_m = tl.max_contiguous(start_m + tl.arange(0, BLOCK_M), BLOCK_M)
    offs_d = tl.arange(0, 128)
    head_offset = pid_bh * SQ * 128
    q = tl.load(q_ptr + head_offset + offs_m[:, None] * 128 + offs_d[None, :])

    acc = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    head_row = pid_bh * SQ

    for start_n in tl.range(0, SQ, BLOCK_N, disable_licm=True):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        start_n_i32 = start_n.to(tl.int32)  # type: ignore[attr-defined]
        k = tl.trans(k_desc.load([head_row + start_n_i32, 0]))
        score = tl.dot(q, k).to(tl.float32) * qk_scale
        m_new = tl.maximum(m_i, tl.max(score, 1))
        p = tl.math.exp2(score - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = v_desc.load([head_row + start_n_i32, 0])
        acc = tl.dot((p * s_scale).to(v.dtype), v, acc)
        l_i = l_i * alpha + l_ij
        m_i = m_new

    o_val = acc * (sv_descale / l_i[:, None])
    if COMPUTE_AMAX:
        local_amax_o = tl.max(tl.abs(o_val))
        tl.atomic_max(
            amax_o_ptr.to(tl.pointer_type(tl.uint32)),
            local_amax_o.to(tl.uint32, bitcast=True),
            sem="relaxed",
        )
        local_amax_s = tl.max(tl.abs(m_i)) * _LN2_KERNEL
        tl.atomic_max(
            amax_s_ptr.to(tl.pointer_type(tl.uint32)),
            local_amax_s.to(tl.uint32, bitcast=True),
            sem="relaxed",
        )

    tl.store(
        o_ptr + head_offset + offs_m[:, None] * 128 + offs_d[None, :],
        (o_val * o_scale).to(o_ptr.dtype.element_ty),
    )


@triton.jit
def _sdpa_fp8_fwd_causal_nostats_hostdesc_tma_kernel(
    q_ptr,
    k_desc,
    v_desc,
    o_ptr,
    amax_s_ptr,
    amax_o_ptr,
    qk_scale: tl.constexpr,
    s_scale: tl.constexpr,
    sv_descale: tl.constexpr,
    o_scale: tl.constexpr,
    SQ: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    COMPUTE_AMAX: tl.constexpr,
):
    raw_pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_m = tl.cdiv(SQ, BLOCK_M) - 1 - raw_pid_m

    start_m = pid_m * BLOCK_M
    offs_m = tl.max_contiguous(start_m + tl.arange(0, BLOCK_M), BLOCK_M)
    offs_d = tl.arange(0, 128)
    head_offset = pid_bh * SQ * 128
    q = tl.load(q_ptr + head_offset + offs_m[:, None] * 128 + offs_d[None, :])

    acc = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    head_row = pid_bh * SQ
    hi = start_m + BLOCK_M
    full_hi = (start_m // BLOCK_N) * BLOCK_N

    if 0 < full_hi:
        for start_n in tl.range(0, full_hi, BLOCK_N, disable_licm=True):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            start_n_i32 = start_n.to(tl.int32)  # type: ignore[attr-defined]
            k = tl.trans(k_desc.load([head_row + start_n_i32, 0]))
            score = tl.dot(q, k).to(tl.float32) * qk_scale
            m_new = tl.maximum(m_i, tl.max(score, 1))
            p = tl.math.exp2(score - m_new[:, None])
            alpha = tl.math.exp2(m_i - m_new)
            l_ij = tl.sum(p, 1)
            acc = acc * alpha[:, None]
            v = v_desc.load([head_row + start_n_i32, 0])
            acc = tl.dot((p * s_scale).to(v.dtype), v, acc)
            l_i = l_i * alpha + l_ij
            m_i = m_new
    if full_hi < hi:
        for start_n in tl.range(full_hi, hi, BLOCK_N, disable_licm=True):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            start_n_i32 = start_n.to(tl.int32)  # type: ignore[attr-defined]
            offs_n = start_n_i32 + tl.arange(0, BLOCK_N)
            k = tl.trans(k_desc.load([head_row + start_n_i32, 0]))
            score = tl.dot(q, k).to(tl.float32) * qk_scale
            score = tl.where(
                offs_n[None, :] <= offs_m[:, None], score, float("-inf")
            )
            m_new = tl.maximum(m_i, tl.max(score, 1))
            p = tl.math.exp2(score - m_new[:, None])
            alpha = tl.math.exp2(m_i - m_new)
            l_ij = tl.sum(p, 1)
            acc = acc * alpha[:, None]
            v = v_desc.load([head_row + start_n_i32, 0])
            acc = tl.dot((p * s_scale).to(v.dtype), v, acc)
            l_i = l_i * alpha + l_ij
            m_i = m_new

    o_val = acc * (sv_descale / l_i[:, None])
    if COMPUTE_AMAX:
        local_amax_o = tl.max(tl.abs(o_val))
        tl.atomic_max(
            amax_o_ptr.to(tl.pointer_type(tl.uint32)),
            local_amax_o.to(tl.uint32, bitcast=True),
            sem="relaxed",
        )
        local_amax_s = tl.max(tl.abs(m_i)) * _LN2_KERNEL
        tl.atomic_max(
            amax_s_ptr.to(tl.pointer_type(tl.uint32)),
            local_amax_s.to(tl.uint32, bitcast=True),
            sem="relaxed",
        )

    tl.store(
        o_ptr + head_offset + offs_m[:, None] * 128 + offs_d[None, :],
        (o_val * o_scale).to(o_ptr.dtype.element_ty),
    )


@triton.jit
def _sdpa_fp8_fwd_dense512_hostdesc_tma_kernel(
    q_ptr,
    k_desc,
    v_desc,
    o_ptr,
    amax_s_ptr,
    amax_o_ptr,
    qk_scale: tl.constexpr,
    s_scale: tl.constexpr,
    sv_descale: tl.constexpr,
    o_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    COMPUTE_AMAX: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_m = tl.max_contiguous(offs_m, BLOCK_M)
    offs_d = tl.arange(0, 128)

    # Row0 is exact dense contiguous BHSD with H=16/S=D=128. Flatten B*H
    # for K/V descriptors so TMA descriptor construction happens on host.
    head_offset = pid_bh * 65536
    q = tl.load(q_ptr + head_offset + offs_m[:, None] * 128 + offs_d[None, :])

    acc = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)

    head_row = pid_bh * 512
    for start_n in tl.range(0, 512, BLOCK_N, disable_licm=True):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        start_n_i32 = start_n.to(tl.int32)  # type: ignore[attr-defined]
        k = tl.trans(k_desc.load([head_row + start_n_i32, 0]))
        score = tl.dot(q, k).to(tl.float32) * qk_scale
        m_new = tl.maximum(m_i, tl.max(score, 1))
        p = tl.math.exp2(score - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = v_desc.load([head_row + start_n_i32, 0])
        acc = tl.dot((p * s_scale).to(v.dtype), v, acc)
        l_i = l_i * alpha + l_ij
        m_i = m_new

    o_val = acc * (sv_descale / l_i[:, None])
    if COMPUTE_AMAX:
        local_amax_o = tl.max(tl.abs(o_val))
        tl.atomic_max(
            amax_o_ptr.to(tl.pointer_type(tl.uint32)),
            local_amax_o.to(tl.uint32, bitcast=True),
            sem="relaxed",
        )
        local_amax_s = tl.max(tl.abs(m_i)) * _LN2_KERNEL
        tl.atomic_max(
            amax_s_ptr.to(tl.pointer_type(tl.uint32)),
            local_amax_s.to(tl.uint32, bitcast=True),
            sem="relaxed",
        )

    tl.store(
        o_ptr + head_offset + offs_m[:, None] * 128 + offs_d[None, :],
        (o_val * o_scale).to(o_ptr.dtype.element_ty),
    )


@triton.jit
def _sdpa_fp8_fwd_row1_causal_hostdesc_tma_kernel(
    q_ptr,
    k_desc,
    v_desc,
    o_ptr,
    stats_ptr,
    amax_s_ptr,
    amax_o_ptr,
    qk_scale: tl.constexpr,
    s_scale: tl.constexpr,
    sv_descale: tl.constexpr,
    o_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    COMPUTE_AMAX: tl.constexpr,
):
    raw_pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_m = tl.cdiv(1024, BLOCK_M) - 1 - raw_pid_m

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_m = tl.max_contiguous(offs_m, BLOCK_M)
    offs_d = tl.arange(0, 128)

    head_offset = pid_bh * 131072
    q = tl.load(q_ptr + head_offset + offs_m[:, None] * 128 + offs_d[None, :])

    acc = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)

    head_row = pid_bh * 1024
    hi = start_m + BLOCK_M
    full_hi = (start_m // BLOCK_N) * BLOCK_N
    if 0 < full_hi:
        for start_n in tl.range(0, full_hi, BLOCK_N, disable_licm=True):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            start_n_i32 = start_n.to(tl.int32)  # type: ignore[attr-defined]
            k = tl.trans(k_desc.load([head_row + start_n_i32, 0]))
            score = tl.dot(q, k).to(tl.float32) * qk_scale
            m_new = tl.maximum(m_i, tl.max(score, 1))
            p = tl.math.exp2(score - m_new[:, None])
            alpha = tl.math.exp2(m_i - m_new)
            l_ij = tl.sum(p, 1)
            acc = acc * alpha[:, None]
            v = v_desc.load([head_row + start_n_i32, 0])
            acc = tl.dot((p * s_scale).to(v.dtype), v, acc)
            l_i = l_i * alpha + l_ij
            m_i = m_new
    if full_hi < hi:
        for start_n in tl.range(full_hi, hi, BLOCK_N, disable_licm=True):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            start_n_i32 = start_n.to(tl.int32)  # type: ignore[attr-defined]
            offs_n = start_n_i32 + tl.arange(0, BLOCK_N)
            k = tl.trans(k_desc.load([head_row + start_n_i32, 0]))
            score = tl.dot(q, k).to(tl.float32) * qk_scale
            score = tl.where(
                offs_n[None, :] <= offs_m[:, None], score, float("-inf")
            )
            m_new = tl.maximum(m_i, tl.max(score, 1))
            p = tl.math.exp2(score - m_new[:, None])
            alpha = tl.math.exp2(m_i - m_new)
            l_ij = tl.sum(p, 1)
            acc = acc * alpha[:, None]
            v = v_desc.load([head_row + start_n_i32, 0])
            acc = tl.dot((p * s_scale).to(v.dtype), v, acc)
            l_i = l_i * alpha + l_ij
            m_i = m_new

    o_val = acc * (sv_descale / l_i[:, None])
    if COMPUTE_AMAX:
        local_amax_o = tl.max(tl.abs(o_val))
        tl.atomic_max(
            amax_o_ptr.to(tl.pointer_type(tl.uint32)),
            local_amax_o.to(tl.uint32, bitcast=True),
            sem="relaxed",
        )
        local_amax_s = tl.max(tl.abs(m_i)) * _LN2_KERNEL
        tl.atomic_max(
            amax_s_ptr.to(tl.pointer_type(tl.uint32)),
            local_amax_s.to(tl.uint32, bitcast=True),
            sem="relaxed",
        )

    tl.store(
        o_ptr + head_offset + offs_m[:, None] * 128 + offs_d[None, :],
        (o_val * o_scale).to(o_ptr.dtype.element_ty),
    )
    stats = (m_i + tl.log2(l_i)) * _LN2_KERNEL
    tl.store(stats_ptr + pid_bh * 1024 + offs_m, stats)


@triton.jit
def _sdpa_fp8_fwd_row1_causal_pcache_full_kernel(
    q_ptr,
    k_desc,
    v_desc,
    p_ptr,
    alpha_ptr,
    final_l_ptr,
    o_ptr,
    stats_ptr,
    amax_s_ptr,
    amax_o_ptr,
    qk_scale: tl.constexpr,
    s_scale: tl.constexpr,
    sv_descale: tl.constexpr,
    o_scale: tl.constexpr,
):
    raw_pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_m = 15 - raw_pid_m
    start_m = pid_m * 64
    offs_m = tl.max_contiguous(start_m + tl.arange(0, 64), 64)
    offs_d = tl.arange(0, 128)

    head_offset = pid_bh * 131072
    q = tl.load(q_ptr + head_offset + offs_m[:, None] * 128 + offs_d[None, :])

    acc = tl.zeros((64, 128), dtype=tl.float32)
    l_i = tl.zeros((64,), dtype=tl.float32)
    m_i = tl.full((64,), float("-inf"), dtype=tl.float32)
    head_row = pid_bh * 1024
    p_base = p_ptr + pid_bh * 1048576
    alpha_base = alpha_ptr + pid_bh * 16384

    for start_n in tl.range(0, start_m, 64, disable_licm=True):
        start_n = tl.multiple_of(start_n, 64)
        start_n_i32 = start_n.to(tl.int32)  # type: ignore[attr-defined]
        offs_n = start_n_i32 + tl.arange(0, 64)
        k = tl.trans(k_desc.load([head_row + start_n_i32, 0]))
        score = tl.dot(q, k).to(tl.float32) * qk_scale
        m_new = tl.maximum(m_i, tl.max(score, 1))
        p = tl.math.exp2(score - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        p_fp8 = (p * s_scale).to(p_ptr.dtype.element_ty)
        tl.store(p_base + offs_m[:, None] * 1024 + offs_n[None, :], p_fp8)
        acc = acc * alpha[:, None]
        v = v_desc.load([head_row + start_n_i32, 0])
        acc = tl.dot(p_fp8, v, acc)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new
        tl.store(alpha_base + (start_n_i32 // 64) * 1024 + offs_m, alpha)

    start_n_i32 = start_m
    offs_n = start_n_i32 + tl.arange(0, 64)
    k = tl.trans(k_desc.load([head_row + start_n_i32, 0]))
    score = tl.dot(q, k).to(tl.float32) * qk_scale
    score = tl.where(offs_n[None, :] <= offs_m[:, None], score, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(score, 1))
    p = tl.math.exp2(score - m_new[:, None])
    alpha = tl.math.exp2(m_i - m_new)
    p_fp8 = (p * s_scale).to(p_ptr.dtype.element_ty)
    tl.store(p_base + offs_m[:, None] * 1024 + offs_n[None, :], p_fp8)
    acc = acc * alpha[:, None]
    v = v_desc.load([head_row + start_n_i32, 0])
    acc = tl.dot(p_fp8, v, acc)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_new
    tl.store(alpha_base + pid_m * 1024 + offs_m, alpha)

    out_descale = sv_descale / l_i
    o_unscaled = acc * out_descale[:, None]
    local_amax_o = tl.max(tl.abs(o_unscaled))
    tl.atomic_max(
        amax_o_ptr.to(tl.pointer_type(tl.uint32)),
        local_amax_o.to(tl.uint32, bitcast=True),
        sem="relaxed",
    )
    local_amax_s = tl.max(tl.abs(m_i)) * _LN2_KERNEL
    tl.atomic_max(
        amax_s_ptr.to(tl.pointer_type(tl.uint32)),
        local_amax_s.to(tl.uint32, bitcast=True),
        sem="relaxed",
    )
    out_scale = out_descale * o_scale
    tl.store(
        o_ptr + head_offset + offs_m[:, None] * 128 + offs_d[None, :],
        (acc * out_scale[:, None]).to(o_ptr.dtype.element_ty),
    )
    stats = (m_i + tl.log2(l_i)) * _LN2_KERNEL
    tl.store(stats_ptr + pid_bh * 1024 + offs_m, stats)
    tl.store(final_l_ptr + pid_bh * 1024 + offs_m, out_scale)


@triton.jit
def _sdpa_fp8_fwd_row1_causal_pcache_replay_kernel(
    v_desc,
    p_desc,
    alpha_ptr,
    final_l_ptr,
    o_ptr,
):
    raw_pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_m = 15 - raw_pid_m
    start_m = pid_m * 64
    offs_m = tl.max_contiguous(start_m + tl.arange(0, 64), 64)
    offs_d = tl.arange(0, 128)

    head_offset = pid_bh * 131072
    acc = tl.zeros((64, 128), dtype=tl.float32)
    head_row = pid_bh * 1024
    alpha_base = alpha_ptr + pid_bh * 16384

    for start_n in tl.range(0, start_m, 64, disable_licm=True):
        start_n = tl.multiple_of(start_n, 64)
        start_n_i32 = start_n.to(tl.int32)  # type: ignore[attr-defined]
        alpha = tl.load(alpha_base + (start_n_i32 // 64) * 1024 + offs_m)
        acc = acc * alpha[:, None]
        p = p_desc.load([pid_bh * 1024 + start_m, start_n_i32])
        v = v_desc.load([head_row + start_n_i32, 0])
        acc = tl.dot(p, v, acc)

    start_n_i32 = start_m
    alpha = tl.load(alpha_base + pid_m * 1024 + offs_m)
    acc = acc * alpha[:, None]
    p = p_desc.load([pid_bh * 1024 + start_m, start_n_i32])
    v = v_desc.load([head_row + start_n_i32, 0])
    acc = tl.dot(p, v, acc)

    out_scale = tl.load(final_l_ptr + pid_bh * 1024 + offs_m)
    tl.store(
        o_ptr + head_offset + offs_m[:, None] * 128 + offs_d[None, :],
        (acc * out_scale[:, None]).to(o_ptr.dtype.element_ty),
    )


@triton.jit
def _sdpa_fp8_fwd_row2_causal_pcache_full_kernel(
    q_ptr,
    k_desc,
    v_desc,
    p_ptr,
    alpha_ptr,
    final_l_ptr,
    o_ptr,
    stats_ptr,
    amax_s_ptr,
    amax_o_ptr,
    qk_scale: tl.constexpr,
    s_scale: tl.constexpr,
    sv_descale: tl.constexpr,
    o_scale: tl.constexpr,
):
    raw_pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_m = 31 - raw_pid_m
    start_m = pid_m * 64
    offs_m = tl.max_contiguous(start_m + tl.arange(0, 64), 64)
    offs_d = tl.arange(0, 128)

    head_offset = pid_bh * 262144
    q = tl.load(q_ptr + head_offset + offs_m[:, None] * 128 + offs_d[None, :])

    acc = tl.zeros((64, 128), dtype=tl.float32)
    l_i = tl.zeros((64,), dtype=tl.float32)
    m_i = tl.full((64,), float("-inf"), dtype=tl.float32)
    head_row = pid_bh * 2048
    p_base = p_ptr + pid_bh * 4194304
    alpha_base = alpha_ptr + pid_bh * 65536

    for start_n in tl.range(0, start_m, 64, disable_licm=True):
        start_n = tl.multiple_of(start_n, 64)
        start_n_i32 = start_n.to(tl.int32)  # type: ignore[attr-defined]
        offs_n = start_n_i32 + tl.arange(0, 64)
        k = tl.trans(k_desc.load([head_row + start_n_i32, 0]))
        score = tl.dot(q, k).to(tl.float32) * qk_scale
        m_new = tl.maximum(m_i, tl.max(score, 1))
        p = tl.math.exp2(score - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        p_fp8 = (p * s_scale).to(p_ptr.dtype.element_ty)
        tl.store(p_base + offs_m[:, None] * 2048 + offs_n[None, :], p_fp8)
        acc = acc * alpha[:, None]
        v = v_desc.load([head_row + start_n_i32, 0])
        acc = tl.dot(p_fp8, v, acc)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new
        tl.store(alpha_base + (start_n_i32 // 64) * 2048 + offs_m, alpha)

    for start_n in tl.range(start_m, start_m + 64, 64, disable_licm=True):
        start_n = tl.multiple_of(start_n, 64)
        start_n_i32 = start_n.to(tl.int32)  # type: ignore[attr-defined]
        offs_n = start_n_i32 + tl.arange(0, 64)
        k = tl.trans(k_desc.load([head_row + start_n_i32, 0]))
        score = tl.dot(q, k).to(tl.float32) * qk_scale
        score = tl.where(
            offs_n[None, :] <= offs_m[:, None], score, float("-inf")
        )
        m_new = tl.maximum(m_i, tl.max(score, 1))
        p = tl.math.exp2(score - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        p_fp8 = (p * s_scale).to(p_ptr.dtype.element_ty)
        tl.store(p_base + offs_m[:, None] * 2048 + offs_n[None, :], p_fp8)
        acc = acc * alpha[:, None]
        v = v_desc.load([head_row + start_n_i32, 0])
        acc = tl.dot(p_fp8, v, acc)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new
        tl.store(alpha_base + (start_n_i32 // 64) * 2048 + offs_m, alpha)

    out_descale = sv_descale / l_i
    o_unscaled = acc * out_descale[:, None]
    local_amax_o = tl.max(tl.abs(o_unscaled))
    tl.atomic_max(
        amax_o_ptr.to(tl.pointer_type(tl.uint32)),
        local_amax_o.to(tl.uint32, bitcast=True),
        sem="relaxed",
    )
    local_amax_s = tl.max(tl.abs(m_i)) * _LN2_KERNEL
    tl.atomic_max(
        amax_s_ptr.to(tl.pointer_type(tl.uint32)),
        local_amax_s.to(tl.uint32, bitcast=True),
        sem="relaxed",
    )
    out_scale = out_descale * o_scale
    tl.store(
        o_ptr + head_offset + offs_m[:, None] * 128 + offs_d[None, :],
        (acc * out_scale[:, None]).to(o_ptr.dtype.element_ty),
    )
    stats = (m_i + tl.log2(l_i)) * _LN2_KERNEL
    tl.store(stats_ptr + pid_bh * 2048 + offs_m, stats)
    tl.store(final_l_ptr + pid_bh * 2048 + offs_m, out_scale)


@triton.jit
def _sdpa_fp8_fwd_row2_causal_pcache_prefix_kernel(
    v_desc,
    p_desc,
    alpha_ptr,
    prefix_ptr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    start_m = pid_m * 64
    offs_r = tl.arange(0, 64)
    offs_d = tl.arange(0, 128)

    acc = tl.zeros((64, 128), dtype=tl.float32)
    head_row = pid_bh * 2048
    alpha_base = alpha_ptr + pid_bh * 65536
    hi = tl.minimum(start_m + 64, 640)

    for start_n in tl.range(0, hi, 64, num_stages=4):
        start_n_i32 = start_n.to(tl.int32)  # type: ignore[attr-defined]
        alpha = tl.load(
            alpha_base + (start_n_i32 // 64) * 2048 + start_m + offs_r
        )
        acc = acc * alpha[:, None]
        p = p_desc.load([pid_bh * 2048 + start_m, start_n_i32])
        v = v_desc.load([head_row + start_n_i32, 0])
        acc = tl.dot(p, v, acc)

    tl.store(
        prefix_ptr
        + (pid_bh * 32 + pid_m) * 8192
        + offs_r[:, None] * 128
        + offs_d[None, :],
        acc,
    )


@triton.jit
def _sdpa_fp8_fwd_row2_causal_pcache_prefix_replay_kernel(
    v_desc,
    p_desc,
    alpha_ptr,
    final_l_ptr,
    prefix_ptr,
    o_ptr,
):
    raw_pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_m = raw_pid_m
    start_m = pid_m * 64
    offs_m = tl.max_contiguous(start_m + tl.arange(0, 64), 64)
    offs_r = tl.arange(0, 64)
    offs_d = tl.arange(0, 128)

    head_offset = pid_bh * 262144
    acc = tl.load(
        prefix_ptr
        + (pid_bh * 32 + pid_m) * 8192
        + offs_r[:, None] * 128
        + offs_d[None, :]
    )
    head_row = pid_bh * 2048
    alpha_base = alpha_ptr + pid_bh * 65536

    for start_n in tl.range(640, start_m + 64, 64, num_stages=4):
        start_n_i32 = start_n.to(tl.int32)  # type: ignore[attr-defined]
        alpha = tl.load(alpha_base + (start_n_i32 // 64) * 2048 + offs_m)
        acc = acc * alpha[:, None]
        p = p_desc.load([pid_bh * 2048 + start_m, start_n_i32])
        v = v_desc.load([head_row + start_n_i32, 0])
        acc = tl.dot(p, v, acc)

    out_scale = tl.load(final_l_ptr + pid_bh * 2048 + offs_m)
    tl.store(
        o_ptr + head_offset + offs_m[:, None] * 128 + offs_d[None, :],
        (acc * out_scale[:, None]).to(o_ptr.dtype.element_ty),
    )


@triton.jit
def _sdpa_fp8_fwd_mha_nostats_pcache_full_kernel(
    q_ptr,
    k_desc,
    v_desc,
    p_ptr,
    alpha_ptr,
    final_l_ptr,
    o_ptr,
    amax_s_ptr,
    amax_o_ptr,
    qk_scale: tl.constexpr,
    s_scale: tl.constexpr,
    sv_descale: tl.constexpr,
    o_scale: tl.constexpr,
    SQ: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    raw_pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    if CAUSAL:
        pid_m = tl.cdiv(SQ, 64) - 1 - raw_pid_m
    else:
        pid_m = raw_pid_m
    start_m = pid_m * 64
    offs_m = tl.max_contiguous(start_m + tl.arange(0, 64), 64)
    offs_d = tl.arange(0, 128)

    head_offset = pid_bh * SQ * 128
    q = tl.load(q_ptr + head_offset + offs_m[:, None] * 128 + offs_d[None, :])

    acc = tl.zeros((64, 128), dtype=tl.float32)
    l_i = tl.zeros((64,), dtype=tl.float32)
    m_i = tl.full((64,), float("-inf"), dtype=tl.float32)
    head_row = pid_bh * SQ
    p_base = p_ptr + pid_bh * SQ * SQ
    alpha_base = alpha_ptr + pid_bh * (SQ // 64) * SQ
    full_hi = start_m if CAUSAL else SQ

    for start_n in tl.range(0, full_hi, 64, disable_licm=True):
        start_n = tl.multiple_of(start_n, 64)
        start_n_i32 = start_n.to(tl.int32)  # type: ignore[attr-defined]
        offs_n = start_n_i32 + tl.arange(0, 64)
        k = tl.trans(k_desc.load([head_row + start_n_i32, 0]))
        score = tl.dot(q, k).to(tl.float32) * qk_scale
        m_new = tl.maximum(m_i, tl.max(score, 1))
        p = tl.math.exp2(score - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        p_fp8 = (p * s_scale).to(p_ptr.dtype.element_ty)
        tl.store(p_base + offs_m[:, None] * SQ + offs_n[None, :], p_fp8)
        acc = acc * alpha[:, None]
        v = v_desc.load([head_row + start_n_i32, 0])
        acc = tl.dot(p_fp8, v, acc)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new
        tl.store(alpha_base + (start_n_i32 // 64) * SQ + offs_m, alpha)

    if CAUSAL:
        start_n_i32 = start_m
        offs_n = start_n_i32 + tl.arange(0, 64)
        k = tl.trans(k_desc.load([head_row + start_n_i32, 0]))
        score = tl.dot(q, k).to(tl.float32) * qk_scale
        score = tl.where(
            offs_n[None, :] <= offs_m[:, None], score, float("-inf")
        )
        m_new = tl.maximum(m_i, tl.max(score, 1))
        p = tl.math.exp2(score - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        p_fp8 = (p * s_scale).to(p_ptr.dtype.element_ty)
        tl.store(p_base + offs_m[:, None] * SQ + offs_n[None, :], p_fp8)
        acc = acc * alpha[:, None]
        v = v_desc.load([head_row + start_n_i32, 0])
        acc = tl.dot(p_fp8, v, acc)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new
        tl.store(alpha_base + pid_m * SQ + offs_m, alpha)
    out_descale = sv_descale / l_i
    o_unscaled = acc * out_descale[:, None]
    local_amax_o = tl.max(tl.abs(o_unscaled))
    tl.atomic_max(
        amax_o_ptr.to(tl.pointer_type(tl.uint32)),
        local_amax_o.to(tl.uint32, bitcast=True),
        sem="relaxed",
    )
    local_amax_s = tl.max(tl.abs(m_i)) * _LN2_KERNEL
    tl.atomic_max(
        amax_s_ptr.to(tl.pointer_type(tl.uint32)),
        local_amax_s.to(tl.uint32, bitcast=True),
        sem="relaxed",
    )
    out_scale = out_descale * o_scale
    tl.store(
        o_ptr + head_offset + offs_m[:, None] * 128 + offs_d[None, :],
        (acc * out_scale[:, None]).to(o_ptr.dtype.element_ty),
    )
    tl.store(final_l_ptr + pid_bh * SQ + offs_m, out_scale)


@triton.jit
def _sdpa_fp8_fwd_mha_nostats_pcache_prefix_kernel(
    v_desc,
    p_desc,
    alpha_ptr,
    prefix_ptr,
    SQ: tl.constexpr,
    PREFIX_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    start_m = pid_m * 64
    offs_r = tl.arange(0, 64)
    offs_d = tl.arange(0, 128)

    acc = tl.zeros((64, 128), dtype=tl.float32)
    head_row = pid_bh * SQ
    alpha_base = alpha_ptr + pid_bh * (SQ // 64) * SQ
    hi = tl.minimum(start_m + 64, PREFIX_N) if CAUSAL else PREFIX_N

    for start_n in tl.range(0, hi, 64, num_stages=4):
        start_n_i32 = start_n.to(tl.int32)  # type: ignore[attr-defined]
        alpha = tl.load(
            alpha_base + (start_n_i32 // 64) * SQ + start_m + offs_r
        )
        acc = acc * alpha[:, None]
        p = p_desc.load([pid_bh * SQ + start_m, start_n_i32])
        v = v_desc.load([head_row + start_n_i32, 0])
        acc = tl.dot(p, v, acc)

    tl.store(
        prefix_ptr
        + (pid_bh * (SQ // 64) + pid_m) * 8192
        + offs_r[:, None] * 128
        + offs_d[None, :],
        acc,
    )


@triton.jit
def _sdpa_fp8_fwd_mha_nostats_pcache_prefix_replay_kernel(
    v_desc,
    p_desc,
    alpha_ptr,
    final_l_ptr,
    prefix_ptr,
    o_ptr,
    SQ: tl.constexpr,
    PREFIX_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    start_m = pid_m * 64
    offs_m = tl.max_contiguous(start_m + tl.arange(0, 64), 64)
    offs_r = tl.arange(0, 64)
    offs_d = tl.arange(0, 128)

    acc = tl.load(
        prefix_ptr
        + (pid_bh * (SQ // 64) + pid_m) * 8192
        + offs_r[:, None] * 128
        + offs_d[None, :]
    )
    head_row = pid_bh * SQ
    alpha_base = alpha_ptr + pid_bh * (SQ // 64) * SQ
    hi = start_m + 64 if CAUSAL else SQ

    for start_n in tl.range(PREFIX_N, hi, 64, num_stages=4):
        start_n_i32 = start_n.to(tl.int32)  # type: ignore[attr-defined]
        alpha = tl.load(alpha_base + (start_n_i32 // 64) * SQ + offs_m)
        acc = acc * alpha[:, None]
        p = p_desc.load([pid_bh * SQ + start_m, start_n_i32])
        v = v_desc.load([head_row + start_n_i32, 0])
        acc = tl.dot(p, v, acc)

    out_scale = tl.load(final_l_ptr + pid_bh * SQ + offs_m)
    tl.store(
        o_ptr + pid_bh * SQ * 128 + offs_m[:, None] * 128 + offs_d[None, :],
        (acc * out_scale[:, None]).to(o_ptr.dtype.element_ty),
    )


@triton.jit
def _sdpa_fp8_fwd_gqa_causal_pcache_full_kernel(
    q_ptr,
    k_desc,
    v_desc,
    p_ptr,
    alpha_ptr,
    final_l_ptr,
    o_ptr,
    stats_ptr,
    amax_s_ptr,
    amax_o_ptr,
    qk_scale: tl.constexpr,
    s_scale: tl.constexpr,
    sv_descale: tl.constexpr,
    o_scale: tl.constexpr,
    SQ: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    GROUP: tl.constexpr,
):
    raw_pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_m = tl.cdiv(SQ, 64) - 1 - raw_pid_m
    start_m = pid_m * 64
    offs_m = tl.max_contiguous(start_m + tl.arange(0, 64), 64)
    offs_d = tl.arange(0, 128)

    off_b = pid_bh // HQ
    off_h = pid_bh - off_b * HQ
    off_kh = off_h // GROUP
    q_head_offset = (off_b * HQ + off_h) * SQ * 128
    kv_head_row = (off_b * HKV + off_kh) * SQ
    q = tl.load(
        q_ptr + q_head_offset + offs_m[:, None] * 128 + offs_d[None, :]
    )

    acc = tl.zeros((64, 128), dtype=tl.float32)
    l_i = tl.zeros((64,), dtype=tl.float32)
    m_i = tl.full((64,), float("-inf"), dtype=tl.float32)
    p_base = p_ptr + pid_bh * SQ * SQ
    alpha_base = alpha_ptr + pid_bh * (SQ // 64) * SQ

    for start_n in tl.range(0, start_m, 64, disable_licm=True):
        start_n = tl.multiple_of(start_n, 64)
        start_n_i32 = start_n.to(tl.int32)  # type: ignore[attr-defined]
        offs_n = start_n_i32 + tl.arange(0, 64)
        k = tl.trans(k_desc.load([kv_head_row + start_n_i32, 0]))
        score = tl.dot(q, k).to(tl.float32) * qk_scale
        m_new = tl.maximum(m_i, tl.max(score, 1))
        p = tl.math.exp2(score - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        p_fp8 = (p * s_scale).to(p_ptr.dtype.element_ty)
        tl.store(p_base + offs_m[:, None] * SQ + offs_n[None, :], p_fp8)
        acc = acc * alpha[:, None]
        v = v_desc.load([kv_head_row + start_n_i32, 0])
        acc = tl.dot(p_fp8, v, acc)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new
        tl.store(alpha_base + (start_n_i32 // 64) * SQ + offs_m, alpha)

    start_n_i32 = start_m
    offs_n = start_n_i32 + tl.arange(0, 64)
    k = tl.trans(k_desc.load([kv_head_row + start_n_i32, 0]))
    score = tl.dot(q, k).to(tl.float32) * qk_scale
    score = tl.where(offs_n[None, :] <= offs_m[:, None], score, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(score, 1))
    p = tl.math.exp2(score - m_new[:, None])
    alpha = tl.math.exp2(m_i - m_new)
    p_fp8 = (p * s_scale).to(p_ptr.dtype.element_ty)
    tl.store(p_base + offs_m[:, None] * SQ + offs_n[None, :], p_fp8)
    acc = acc * alpha[:, None]
    v = v_desc.load([kv_head_row + start_n_i32, 0])
    acc = tl.dot(p_fp8, v, acc)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_new
    tl.store(alpha_base + pid_m * SQ + offs_m, alpha)

    out_descale = sv_descale / l_i
    o_unscaled = acc * out_descale[:, None]
    local_amax_o = tl.max(tl.abs(o_unscaled))
    tl.atomic_max(
        amax_o_ptr.to(tl.pointer_type(tl.uint32)),
        local_amax_o.to(tl.uint32, bitcast=True),
        sem="relaxed",
    )
    local_amax_s = tl.max(tl.abs(m_i)) * _LN2_KERNEL
    tl.atomic_max(
        amax_s_ptr.to(tl.pointer_type(tl.uint32)),
        local_amax_s.to(tl.uint32, bitcast=True),
        sem="relaxed",
    )
    out_scale = out_descale * o_scale
    tl.store(
        o_ptr + q_head_offset + offs_m[:, None] * 128 + offs_d[None, :],
        (acc * out_scale[:, None]).to(o_ptr.dtype.element_ty),
    )
    stats = (m_i + tl.log2(l_i)) * _LN2_KERNEL
    tl.store(stats_ptr + pid_bh * SQ + offs_m, stats)
    tl.store(final_l_ptr + pid_bh * SQ + offs_m, out_scale)


@triton.jit
def _sdpa_fp8_fwd_gqa_causal_pcache_prefix_kernel(
    v_desc,
    p_desc,
    alpha_ptr,
    prefix_ptr,
    SQ: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    GROUP: tl.constexpr,
    PREFIX_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    start_m = pid_m * 64
    offs_r = tl.arange(0, 64)
    offs_d = tl.arange(0, 128)

    off_b = pid_bh // HQ
    off_h = pid_bh - off_b * HQ
    off_kh = off_h // GROUP
    kv_head_row = (off_b * HKV + off_kh) * SQ
    alpha_base = alpha_ptr + pid_bh * (SQ // 64) * SQ
    acc = tl.zeros((64, 128), dtype=tl.float32)
    hi = tl.minimum(start_m + 64, PREFIX_N)

    for start_n in tl.range(0, hi, 64, num_stages=4):
        start_n_i32 = start_n.to(tl.int32)  # type: ignore[attr-defined]
        alpha = tl.load(
            alpha_base + (start_n_i32 // 64) * SQ + start_m + offs_r
        )
        acc = acc * alpha[:, None]
        p = p_desc.load([pid_bh * SQ + start_m, start_n_i32])
        v = v_desc.load([kv_head_row + start_n_i32, 0])
        acc = tl.dot(p, v, acc)

    tl.store(
        prefix_ptr
        + (pid_bh * (SQ // 64) + pid_m) * 8192
        + offs_r[:, None] * 128
        + offs_d[None, :],
        acc,
    )


@triton.jit
def _sdpa_fp8_fwd_gqa_causal_pcache_prefix_replay_kernel(
    v_desc,
    p_desc,
    alpha_ptr,
    final_l_ptr,
    prefix_ptr,
    o_ptr,
    SQ: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    GROUP: tl.constexpr,
    PREFIX_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    start_m = pid_m * 64
    offs_m = tl.max_contiguous(start_m + tl.arange(0, 64), 64)
    offs_r = tl.arange(0, 64)
    offs_d = tl.arange(0, 128)

    off_b = pid_bh // HQ
    off_h = pid_bh - off_b * HQ
    off_kh = off_h // GROUP
    q_head_offset = (off_b * HQ + off_h) * SQ * 128
    kv_head_row = (off_b * HKV + off_kh) * SQ
    alpha_base = alpha_ptr + pid_bh * (SQ // 64) * SQ

    acc = tl.load(
        prefix_ptr
        + (pid_bh * (SQ // 64) + pid_m) * 8192
        + offs_r[:, None] * 128
        + offs_d[None, :]
    )

    for start_n in tl.range(PREFIX_N, start_m + 64, 64, num_stages=4):
        start_n_i32 = start_n.to(tl.int32)  # type: ignore[attr-defined]
        alpha = tl.load(alpha_base + (start_n_i32 // 64) * SQ + offs_m)
        acc = acc * alpha[:, None]
        p = p_desc.load([pid_bh * SQ + start_m, start_n_i32])
        v = v_desc.load([kv_head_row + start_n_i32, 0])
        acc = tl.dot(p, v, acc)

    out_scale = tl.load(final_l_ptr + pid_bh * SQ + offs_m)
    tl.store(
        o_ptr + q_head_offset + offs_m[:, None] * 128 + offs_d[None, :],
        (acc * out_scale[:, None]).to(o_ptr.dtype.element_ty),
    )


@triton.jit
def _sdpa_fp8_fwd_tma_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    stats_ptr,
    amax_s_ptr,
    amax_o_ptr,
    qk_scale,
    s_scale,
    sv_descale,
    o_scale,
    HQ: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    Q_PER_K: tl.constexpr,
    Q_PER_V: tl.constexpr,
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
    stride_sb: tl.constexpr,
    stride_sh: tl.constexpr,
    stride_sm: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    CAUSAL: tl.constexpr,
    GENERATE_STATS: tl.constexpr,
):
    raw_pid_m = tl.program_id(0)
    if CAUSAL:
        pid_m = tl.cdiv(SQ, BLOCK_M) - 1 - raw_pid_m
    else:
        pid_m = raw_pid_m
    pid_bh = tl.program_id(1)
    off_b = pid_bh // HQ
    off_h = pid_bh % HQ
    off_kh = off_h // Q_PER_K
    off_vh = off_h // Q_PER_V

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_m = offs_m < SQ

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    if SQ % BLOCK_M == 0:
        q = tl.load(
            q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        )
    else:
        q = tl.load(
            q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=mask_m[:, None],
            other=0.0,
        )
    k_desc = tl.make_tensor_descriptor(
        k_ptr + off_b * stride_kb + off_kh * stride_kh,
        shape=[SKV, HEAD_DIM],
        strides=[stride_kn, stride_kd],
        block_shape=[BLOCK_N, BLOCK_D],
    )
    v_desc = tl.make_tensor_descriptor(
        v_ptr + off_b * stride_vb + off_vh * stride_vh,
        shape=[SKV, V_DIM],
        strides=[stride_vn, stride_vd],
        block_shape=[BLOCK_N, BLOCK_DV],
    )

    acc = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)

    if CAUSAL:
        hi = tl.minimum(start_m + BLOCK_M, SKV)
        full_hi = tl.minimum((start_m // BLOCK_N) * BLOCK_N, hi)
        if 0 < full_hi:
            acc, l_i, m_i = _sdpa_fp8_tma_inner(
                acc,
                l_i,
                m_i,
                q,
                k_desc,
                v_desc,
                qk_scale,
                s_scale,
                offs_m,
                0,
                full_hi,
                SKV,
                BLOCK_N=BLOCK_N,
                CAUSAL_MASK=False,
                TAIL_MASK=False,
            )
        if full_hi < hi:
            acc, l_i, m_i = _sdpa_fp8_tma_inner(
                acc,
                l_i,
                m_i,
                q,
                k_desc,
                v_desc,
                qk_scale,
                s_scale,
                offs_m,
                full_hi,
                hi,
                SKV,
                BLOCK_N=BLOCK_N,
                CAUSAL_MASK=True,
                TAIL_MASK=False,
            )
    else:
        full = (SKV // BLOCK_N) * BLOCK_N
        if 0 < full:
            acc, l_i, m_i = _sdpa_fp8_tma_inner(
                acc,
                l_i,
                m_i,
                q,
                k_desc,
                v_desc,
                qk_scale,
                s_scale,
                offs_m,
                0,
                full,
                SKV,
                BLOCK_N=BLOCK_N,
                CAUSAL_MASK=False,
                TAIL_MASK=False,
            )
        if full < SKV:
            acc, l_i, m_i = _sdpa_fp8_tma_inner(
                acc,
                l_i,
                m_i,
                q,
                k_desc,
                v_desc,
                qk_scale,
                s_scale,
                offs_m,
                full,
                SKV,
                SKV,
                BLOCK_N=BLOCK_N,
                CAUSAL_MASK=False,
                TAIL_MASK=True,
            )

    l_safe = tl.maximum(l_i, 1.0)
    o_val = acc * (sv_descale / l_safe[:, None])
    if SQ % BLOCK_M == 0:
        local_amax_o = tl.max(tl.abs(o_val))
    else:
        local_amax_o = tl.max(tl.where(mask_m[:, None], tl.abs(o_val), 0.0))
    tl.atomic_max(amax_o_ptr, local_amax_o, sem="relaxed")
    if SQ % BLOCK_M == 0:
        amax_s_val = tl.max(tl.abs(m_i)) * _LN2_KERNEL
    else:
        amax_s_val = tl.max(tl.where(mask_m, tl.abs(m_i), 0.0)) * _LN2_KERNEL
    tl.atomic_max(amax_s_ptr, amax_s_val, sem="relaxed")

    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh
    if SQ % BLOCK_M == 0:
        tl.store(
            o_base
            + offs_m[:, None] * stride_om
            + offs_dv[None, :] * stride_od,
            (o_val * o_scale).to(o_ptr.dtype.element_ty),
        )
    else:
        tl.store(
            o_base
            + offs_m[:, None] * stride_om
            + offs_dv[None, :] * stride_od,
            (o_val * o_scale).to(o_ptr.dtype.element_ty),
            mask=mask_m[:, None],
        )
    if GENERATE_STATS:
        stats = (m_i + tl.log2(l_safe)) * _LN2_KERNEL
        stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh
        if SQ % BLOCK_M == 0:
            tl.store(stats_base + offs_m * stride_sm, stats)
        else:
            tl.store(stats_base + offs_m * stride_sm, stats, mask=mask_m)


@triton.jit
def _sdpa_fp8_pack_vt_kernel(
    v_ptr,
    vt_ptr,
    SKV: tl.constexpr,
    V_DIM: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vd: tl.constexpr,
    HKV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // HKV
    off_h = pid_bh % HKV
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    vals = tl.load(
        v_ptr
        + off_b * stride_vb
        + off_h * stride_vh
        + offs_n[:, None] * stride_vn
        + offs_d[None, :] * stride_vd,
        mask=offs_n[:, None] < SKV,
        other=0.0,
    )
    vt_base = vt_ptr + (off_b * HKV + off_h) * V_DIM * SKV
    tl.store(
        vt_base + offs_d[None, :] * SKV + offs_n[:, None],
        vals,
        mask=offs_n[:, None] < SKV,
    )


@triton.jit
def _sdpa_fp8_fwd_gqa_causal_vt_kernel(
    q_ptr,
    k_ptr,
    vt_ptr,
    o_ptr,
    stats_ptr,
    amax_s_ptr,
    amax_o_ptr,
    qk_scale,
    s_scale,
    sv_descale,
    o_scale,
    HKV: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    GROUP: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_od: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_sh: tl.constexpr,
    stride_sm: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GENERATE_STATS: tl.constexpr,
):
    raw_pid_m = tl.program_id(0)
    pid_m = tl.cdiv(SQ, BLOCK_M) - 1 - raw_pid_m
    pid_bkv = tl.program_id(1)
    pid_hg = tl.program_id(2)
    off_b = pid_bkv // HKV
    off_kh = pid_bkv % HKV

    start_m = pid_m * BLOCK_M
    offs_mh = tl.arange(0, BLOCK_M * BLOCK_H)
    offs_h = pid_hg * BLOCK_H + offs_mh // BLOCK_M
    offs_m = start_m + (offs_mh % BLOCK_M)
    q_head = off_kh * GROUP + offs_h
    row_mask = (offs_h < GROUP) & (offs_m < SQ)
    offs_d = tl.arange(0, BLOCK_D)

    if SQ % BLOCK_M == 0 and GROUP % BLOCK_H == 0:
        q = tl.load(
            q_ptr
            + off_b * stride_qb
            + q_head[:, None] * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_d[None, :] * stride_qd,
        )
    else:
        q = tl.load(
            q_ptr
            + off_b * stride_qb
            + q_head[:, None] * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_d[None, :] * stride_qd,
            mask=row_mask[:, None],
            other=0.0,
        )
    k_desc = tl.make_tensor_descriptor(
        k_ptr + off_b * stride_kb + off_kh * stride_kh,
        shape=[SKV, HEAD_DIM],
        strides=[stride_kn, stride_kd],
        block_shape=[BLOCK_N, BLOCK_D],
    )
    vt_desc = tl.make_tensor_descriptor(
        vt_ptr + (off_b * HKV + off_kh) * V_DIM * SKV,
        shape=[V_DIM, SKV],
        strides=[SKV, 1],
        block_shape=[BLOCK_D, BLOCK_N],
    )

    acc_t = tl.zeros((BLOCK_D, BLOCK_M * BLOCK_H), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M * BLOCK_H,), dtype=tl.float32)
    m_i = tl.full((BLOCK_M * BLOCK_H,), float("-inf"), dtype=tl.float32)

    hi = tl.minimum(start_m + BLOCK_M, SKV)
    full_hi = tl.minimum((start_m // BLOCK_N) * BLOCK_N, hi)
    if 0 < full_hi:
        for start_n in tl.range(0, full_hi, BLOCK_N, disable_licm=True):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            start_n_i32 = start_n.to(tl.int32)  # type: ignore[attr-defined]
            k = tl.trans(k_desc.load([start_n_i32, 0]))
            score = tl.dot(q, k).to(tl.float32) * qk_scale
            m_new = tl.maximum(m_i, tl.max(score, 1))
            p = tl.math.exp2(score - m_new[:, None])
            alpha = tl.math.exp2(m_i - m_new)
            l_ij = tl.sum(p, 1)
            acc_t = acc_t * alpha[None, :]
            vt = vt_desc.load([0, start_n_i32])
            p_fp8 = (p * s_scale).to(vt.dtype)
            acc_t = tl.dot(vt, tl.trans(p_fp8), acc_t)
            l_i = l_i * alpha + l_ij
            m_i = m_new
    if full_hi < hi:
        for start_n in tl.range(full_hi, hi, BLOCK_N, disable_licm=True):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            start_n_i32 = start_n.to(tl.int32)  # type: ignore[attr-defined]
            offs_n = start_n_i32 + tl.arange(0, BLOCK_N)
            k = tl.trans(k_desc.load([start_n_i32, 0]))
            score = tl.dot(q, k).to(tl.float32) * qk_scale
            score = tl.where(
                offs_n[None, :] <= offs_m[:, None], score, float("-inf")
            )
            m_new = tl.maximum(m_i, tl.max(score, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.math.exp2(score - m_safe[:, None])
            alpha = tl.math.exp2(m_i - m_safe)
            l_ij = tl.sum(p, 1)
            acc_t = acc_t * alpha[None, :]
            vt = vt_desc.load([0, start_n_i32])
            p_fp8 = (p * s_scale).to(vt.dtype)
            acc_t = tl.dot(vt, tl.trans(p_fp8), acc_t)
            l_i = l_i * alpha + l_ij
            m_i = m_new

    l_safe = tl.maximum(l_i, 1.0)
    o_val = tl.trans(acc_t) * (sv_descale / l_safe[:, None])
    if SQ % BLOCK_M == 0 and GROUP % BLOCK_H == 0:
        local_amax_o = tl.max(tl.abs(o_val))
    else:
        local_amax_o = tl.max(tl.where(row_mask[:, None], tl.abs(o_val), 0.0))
    tl.atomic_max(amax_o_ptr, local_amax_o, sem="relaxed")
    if SQ % BLOCK_M == 0 and GROUP % BLOCK_H == 0:
        amax_s_val = tl.max(tl.abs(m_i)) * _LN2_KERNEL
    else:
        amax_s_val = tl.max(tl.where(row_mask, tl.abs(m_i), 0.0)) * _LN2_KERNEL
    tl.atomic_max(amax_s_ptr, amax_s_val, sem="relaxed")

    if SQ % BLOCK_M == 0 and GROUP % BLOCK_H == 0:
        tl.store(
            o_ptr
            + off_b * stride_ob
            + q_head[:, None] * stride_oh
            + offs_m[:, None] * stride_om
            + offs_d[None, :] * stride_od,
            (o_val * o_scale).to(o_ptr.dtype.element_ty),
        )
    else:
        tl.store(
            o_ptr
            + off_b * stride_ob
            + q_head[:, None] * stride_oh
            + offs_m[:, None] * stride_om
            + offs_d[None, :] * stride_od,
            (o_val * o_scale).to(o_ptr.dtype.element_ty),
            mask=row_mask[:, None],
        )
    if GENERATE_STATS:
        stats = (m_i + tl.log2(l_safe)) * _LN2_KERNEL
        if SQ % BLOCK_M == 0 and GROUP % BLOCK_H == 0:
            tl.store(
                stats_ptr
                + off_b * stride_sb
                + q_head * stride_sh
                + offs_m * stride_sm,
                stats,
            )
        else:
            tl.store(
                stats_ptr
                + off_b * stride_sb
                + q_head * stride_sh
                + offs_m * stride_sm,
                stats,
                mask=row_mask,
            )


@triton.jit
def _sdpa_fp8_fwd_gqa_causal_tma_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    stats_ptr,
    amax_s_ptr,
    amax_o_ptr,
    qk_scale,
    s_scale,
    sv_descale,
    o_scale,
    HKV: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    GROUP: tl.constexpr,
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
    stride_sb: tl.constexpr,
    stride_sh: tl.constexpr,
    stride_sm: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    GENERATE_STATS: tl.constexpr,
):
    raw_pid_m = tl.program_id(0)
    pid_m = tl.cdiv(SQ, BLOCK_M) - 1 - raw_pid_m
    pid_bkv = tl.program_id(1)
    pid_hg = tl.program_id(2)
    off_b = pid_bkv // HKV
    off_kh = pid_bkv % HKV

    start_m = pid_m * BLOCK_M
    offs_mh = tl.arange(0, BLOCK_M * BLOCK_H)
    offs_h = pid_hg * BLOCK_H + offs_mh // BLOCK_M
    offs_m = start_m + (offs_mh % BLOCK_M)
    q_head = off_kh * GROUP + offs_h
    row_mask = (offs_h < GROUP) & (offs_m < SQ)
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)

    if SQ % BLOCK_M == 0 and GROUP % BLOCK_H == 0:
        q = tl.load(
            q_ptr
            + off_b * stride_qb
            + q_head[:, None] * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_d[None, :] * stride_qd,
        )
    else:
        q = tl.load(
            q_ptr
            + off_b * stride_qb
            + q_head[:, None] * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_d[None, :] * stride_qd,
            mask=row_mask[:, None],
            other=0.0,
        )
    k_desc = tl.make_tensor_descriptor(
        k_ptr + off_b * stride_kb + off_kh * stride_kh,
        shape=[SKV, HEAD_DIM],
        strides=[stride_kn, stride_kd],
        block_shape=[BLOCK_N, BLOCK_D],
    )
    v_desc = tl.make_tensor_descriptor(
        v_ptr + off_b * stride_vb + off_kh * stride_vh,
        shape=[SKV, V_DIM],
        strides=[stride_vn, stride_vd],
        block_shape=[BLOCK_N, BLOCK_DV],
    )

    acc = tl.zeros((BLOCK_M * BLOCK_H, BLOCK_DV), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M * BLOCK_H,), dtype=tl.float32)
    m_i = tl.full((BLOCK_M * BLOCK_H,), float("-inf"), dtype=tl.float32)

    hi = tl.minimum(start_m + BLOCK_M, SKV)
    full_hi = tl.minimum((start_m // BLOCK_N) * BLOCK_N, hi)
    if 0 < full_hi:
        acc, l_i, m_i = _sdpa_fp8_tma_inner(
            acc,
            l_i,
            m_i,
            q,
            k_desc,
            v_desc,
            qk_scale,
            s_scale,
            offs_m,
            0,
            full_hi,
            SKV,
            BLOCK_N=BLOCK_N,
            CAUSAL_MASK=False,
            TAIL_MASK=False,
        )
    if full_hi < hi:
        acc, l_i, m_i = _sdpa_fp8_tma_inner(
            acc,
            l_i,
            m_i,
            q,
            k_desc,
            v_desc,
            qk_scale,
            s_scale,
            offs_m,
            full_hi,
            hi,
            SKV,
            BLOCK_N=BLOCK_N,
            CAUSAL_MASK=True,
            TAIL_MASK=False,
        )

    l_safe = tl.maximum(l_i, 1.0)
    o_val = acc * (sv_descale / l_safe[:, None])
    if SQ % BLOCK_M == 0 and GROUP % BLOCK_H == 0:
        local_amax_o = tl.max(tl.abs(o_val))
    else:
        local_amax_o = tl.max(tl.where(row_mask[:, None], tl.abs(o_val), 0.0))
    tl.atomic_max(amax_o_ptr, local_amax_o, sem="relaxed")
    if SQ % BLOCK_M == 0 and GROUP % BLOCK_H == 0:
        amax_s_val = tl.max(tl.abs(m_i)) * _LN2_KERNEL
    else:
        amax_s_val = tl.max(tl.where(row_mask, tl.abs(m_i), 0.0)) * _LN2_KERNEL
    tl.atomic_max(amax_s_ptr, amax_s_val, sem="relaxed")

    if SQ % BLOCK_M == 0 and GROUP % BLOCK_H == 0:
        tl.store(
            o_ptr
            + off_b * stride_ob
            + q_head[:, None] * stride_oh
            + offs_m[:, None] * stride_om
            + offs_dv[None, :] * stride_od,
            (o_val * o_scale).to(o_ptr.dtype.element_ty),
        )
    else:
        tl.store(
            o_ptr
            + off_b * stride_ob
            + q_head[:, None] * stride_oh
            + offs_m[:, None] * stride_om
            + offs_dv[None, :] * stride_od,
            (o_val * o_scale).to(o_ptr.dtype.element_ty),
            mask=row_mask[:, None],
        )
    if GENERATE_STATS:
        stats = (m_i + tl.log2(l_safe)) * _LN2_KERNEL
        if SQ % BLOCK_M == 0 and GROUP % BLOCK_H == 0:
            tl.store(
                stats_ptr
                + off_b * stride_sb
                + q_head * stride_sh
                + offs_m * stride_sm,
                stats,
            )
        else:
            tl.store(
                stats_ptr
                + off_b * stride_sb
                + q_head * stride_sh
                + offs_m * stride_sm,
                stats,
                mask=row_mask,
            )


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


@triton.jit
def _sdpa_fp8_bwd_gqa_accum_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    do_ptr,
    stats_ptr,
    dk_accum_ptr,
    dv_accum_ptr,
    amax_dp_ptr,
    qk_scale,
    ov_descale,
    do_v_descale,
    scale_s,
    scale_dp,
    attn_scale,
    HQ: tl.constexpr,
    SQ,
    SKV,
    q_per_k: tl.constexpr,
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
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BANDED: tl.constexpr,
    CAUSAL_TOP_LEFT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // HQ
    off_h = pid_bh % HQ
    off_kh = off_h // q_per_k

    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_kh * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_kh * stride_vh
    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh
    do_base = do_ptr + off_b * stride_dob + off_h * stride_doh
    stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh

    k = tl.load(
        k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    )
    v = tl.load(
        v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    )
    dk = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    local_amax_dp = tl.full((), 0.0, dtype=tl.float32)

    loop_start_m = 0
    if CAUSAL_TOP_LEFT:
        loop_start_m = start_n

    for start_m in tl.range(loop_start_m, SQ, BLOCK_M):
        rows = start_m + offs_m
        q = tl.load(
            q_base + rows[:, None] * stride_qm + offs_d[None, :] * stride_qd
        )
        o = tl.load(
            o_base + rows[:, None] * stride_om + offs_d[None, :] * stride_od
        ).to(tl.float32)
        do = tl.load(
            do_base + rows[:, None] * stride_dom + offs_d[None, :] * stride_dod
        )
        do_f32 = do.to(tl.float32)
        row_delta = tl.sum(o * do_f32, axis=1) * ov_descale
        stats = tl.load(stats_base + rows * stride_sm).to(tl.float32)

        score = tl.dot(q, tl.trans(k)).to(tl.float32) * qk_scale
        if BANDED:
            diag = offs_n[None, :] - rows[:, None]
            valid = (diag >= min_diag) & (diag <= max_diag)
            p_tile = tl.where(
                valid,
                tl.exp2((score - stats[:, None]) * 1.4426950408889634),
                0.0,
            )
        else:
            p_tile = tl.exp2((score - stats[:, None]) * 1.4426950408889634)

        p_quant = (p_tile * scale_s).to(q.dtype)
        dp = tl.dot(do, tl.trans(v)).to(tl.float32) * do_v_descale
        ds = p_tile * (dp - row_delta[:, None]) * attn_scale
        if BANDED:
            local_amax_dp = tl.maximum(
                local_amax_dp, tl.max(tl.where(valid, tl.abs(ds), 0.0))
            )
        else:
            local_amax_dp = tl.maximum(local_amax_dp, tl.max(tl.abs(ds)))
        ds_quant = (ds * scale_dp).to(q.dtype)

        dk += tl.dot(tl.trans(ds_quant), q)
        dv += tl.dot(tl.trans(p_quant), do)

    scratch_offsets = (
        (off_b * HQ + off_h) * SKV + offs_n[:, None]
    ) * HEAD_DIM + offs_d[None, :]
    tl.store(dk_accum_ptr + scratch_offsets, dk)
    tl.store(dv_accum_ptr + scratch_offsets, dv)
    tl.atomic_max(amax_dp_ptr, local_amax_dp, sem="relaxed")


@triton.jit
def _sdpa_fp8_bwd_gqa_reduce_kernel(
    dk_accum_ptr,
    dv_accum_ptr,
    dk_ptr,
    dv_ptr,
    amax_dk_ptr,
    amax_dv_ptr,
    dk_descale,
    dv_descale,
    scale_dk,
    scale_dv,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    SKV,
    q_per_k: tl.constexpr,
    stride_dkb: tl.constexpr,
    stride_dkh: tl.constexpr,
    stride_dkn: tl.constexpr,
    stride_dkd: tl.constexpr,
    stride_dvb: tl.constexpr,
    stride_dvh: tl.constexpr,
    stride_dvn: tl.constexpr,
    stride_dvd: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // HKV
    off_kh = pid_bh % HKV

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    dk = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for group_idx in tl.static_range(0, q_per_k):
        off_h = off_kh * q_per_k + group_idx
        scratch_offsets = (
            (off_b * HQ + off_h) * SKV + offs_n[:, None]
        ) * HEAD_DIM + offs_d[None, :]
        dk += tl.load(dk_accum_ptr + scratch_offsets)
        dv += tl.load(dv_accum_ptr + scratch_offsets)

    dk_val = dk * dk_descale
    dv_val = dv * dv_descale
    local_amax_dk = tl.max(tl.abs(dk_val))
    local_amax_dv = tl.max(tl.abs(dv_val))
    tl.atomic_max(amax_dk_ptr, local_amax_dk, sem="relaxed")
    tl.atomic_max(amax_dv_ptr, local_amax_dv, sem="relaxed")

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
    tl.store(dk_out_ptrs, (dk_val * scale_dk).to(dk_ptr.dtype.element_ty))
    tl.store(dv_out_ptrs, (dv_val * scale_dv).to(dv_ptr.dtype.element_ty))


@triton.jit
def _sdpa_fp8_bwd_materialize_p_ds_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    do_ptr,
    stats_ptr,
    p_ptr,
    ds_ptr,
    qk_scale,
    ov_descale,
    do_v_descale,
    scale_s,
    scale_dp,
    attn_scale,
    HQ: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    Q_PER: tl.constexpr,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // HQ
    off_h = pid_bh - off_b * HQ
    off_kh = off_h // Q_PER

    start_m = pid_m * BLOCK_M
    offs_m = tl.max_contiguous(start_m + tl.arange(0, BLOCK_M), BLOCK_M)
    rel_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_kh * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_kh * stride_vh
    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh
    do_base = do_ptr + off_b * stride_dob + off_h * stride_doh
    stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh

    q = tl.load(
        q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    )
    o = tl.load(
        o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    ).to(tl.float32)
    do = tl.load(
        do_base + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
    )
    stats = tl.load(stats_base + offs_m * stride_sm).to(tl.float32)
    row_delta = tl.sum(o * do.to(tl.float32), axis=1) * ov_descale

    loop_skv = SKV
    if CAUSAL:
        loop_skv = tl.minimum(SKV, start_m + BLOCK_M)

    p_base = p_ptr + pid_bh * SQ * SKV + start_m * SKV
    ds_base = ds_ptr + pid_bh * SQ * SKV + start_m * SKV
    for start_n in tl.range(0, loop_skv, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        cols = start_n + offs_n
        k = tl.load(
            k_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd
        )
        v = tl.load(
            v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd
        )
        score = tl.dot(q, tl.trans(k)).to(tl.float32) * qk_scale
        p = tl.exp2((score - stats[:, None]) * 1.4426950408889634)
        if CAUSAL:
            p = tl.where(cols[None, :] <= offs_m[:, None], p, 0.0)
        dp = tl.dot(do, tl.trans(v)).to(tl.float32) * do_v_descale
        ds = p * (dp - row_delta[:, None]) * attn_scale
        tl.store(
            p_base + rel_m[:, None] * SKV + cols[None, :],
            (p * scale_s).to(p_ptr.dtype.element_ty),
        )
        tl.store(
            ds_base + rel_m[:, None] * SKV + cols[None, :],
            (ds * scale_dp).to(ds_ptr.dtype.element_ty),
        )


@triton.jit
def _sdpa_fp8_bwd_replay_dq_kernel(
    ds_ptr,
    k_ptr,
    dq_ptr,
    dq_descale,
    scale_dq,
    HQ: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    Q_PER: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_dqb: tl.constexpr,
    stride_dqh: tl.constexpr,
    stride_dqm: tl.constexpr,
    stride_dqd: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // HQ
    off_h = pid_bh - off_b * HQ
    off_kh = off_h // Q_PER

    start_m = pid_m * BLOCK_M
    offs_m = tl.max_contiguous(start_m + tl.arange(0, BLOCK_M), BLOCK_M)
    rel_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    loop_skv = SKV
    if CAUSAL:
        loop_skv = tl.minimum(SKV, start_m + BLOCK_M)

    k_base = k_ptr + off_b * stride_kb + off_kh * stride_kh
    ds_base = ds_ptr + pid_bh * SQ * SKV + start_m * SKV
    dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for start_n in tl.range(0, loop_skv, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        cols = start_n + offs_n
        ds = tl.load(ds_base + rel_m[:, None] * SKV + cols[None, :])
        k = tl.load(
            k_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd
        )
        dq += tl.dot(ds, k)

    dq_val = dq * dq_descale
    dq_out = (
        dq_ptr
        + off_b * stride_dqb
        + off_h * stride_dqh
        + offs_m[:, None] * stride_dqm
        + offs_d[None, :] * stride_dqd
    )
    tl.store(dq_out, (dq_val * scale_dq).to(dq_ptr.dtype.element_ty))


@triton.jit
def _sdpa_fp8_bwd_replay_dkdv_kernel(
    p_ptr,
    ds_ptr,
    q_ptr,
    do_ptr,
    dk_ptr,
    dv_ptr,
    dk_descale,
    dv_descale,
    scale_dk,
    scale_dv,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    Q_PER: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_dob: tl.constexpr,
    stride_doh: tl.constexpr,
    stride_dom: tl.constexpr,
    stride_dod: tl.constexpr,
    stride_dkb: tl.constexpr,
    stride_dkh: tl.constexpr,
    stride_dkn: tl.constexpr,
    stride_dkd: tl.constexpr,
    stride_dvb: tl.constexpr,
    stride_dvh: tl.constexpr,
    stride_dvn: tl.constexpr,
    stride_dvd: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
    REPLAY_DK: tl.constexpr,
    REPLAY_DV: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // HKV
    off_kh = pid_bh - off_b * HKV

    start_n = pid_n * BLOCK_N
    offs_n = tl.max_contiguous(start_n + tl.arange(0, BLOCK_N), BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    if REPLAY_DK:
        dk = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    if REPLAY_DV:
        dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for group_idx in tl.static_range(0, Q_PER):
        off_h = off_kh * Q_PER + group_idx
        q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
        do_base = do_ptr + off_b * stride_dob + off_h * stride_doh
        cache_head = (off_b * HQ + off_h) * SQ * SKV
        loop_start_m = 0
        if CAUSAL:
            loop_start_m = (start_n // BLOCK_M) * BLOCK_M
        for start_m in tl.range(loop_start_m, SQ, BLOCK_M):
            rows = start_m + offs_m
            q = tl.load(
                q_base
                + rows[:, None] * stride_qm
                + offs_d[None, :] * stride_qd
            )
            do = tl.load(
                do_base
                + rows[:, None] * stride_dom
                + offs_d[None, :] * stride_dod
            )
            p = tl.load(
                p_ptr + cache_head + rows[:, None] * SKV + offs_n[None, :]
            )
            ds = tl.load(
                ds_ptr + cache_head + rows[:, None] * SKV + offs_n[None, :]
            )
            if REPLAY_DK:
                dk += tl.dot(tl.trans(ds), q)
            if REPLAY_DV:
                dv += tl.dot(tl.trans(p), do)

    if REPLAY_DK:
        dk_val = dk * dk_descale
    if REPLAY_DV:
        dv_val = dv * dv_descale
    dk_out = (
        dk_ptr
        + off_b * stride_dkb
        + off_kh * stride_dkh
        + offs_n[:, None] * stride_dkn
        + offs_d[None, :] * stride_dkd
    )
    dv_out = (
        dv_ptr
        + off_b * stride_dvb
        + off_kh * stride_dvh
        + offs_n[:, None] * stride_dvn
        + offs_d[None, :] * stride_dvd
    )
    if REPLAY_DK:
        tl.store(dk_out, (dk_val * scale_dk).to(dk_ptr.dtype.element_ty))
    if REPLAY_DV:
        tl.store(dv_out, (dv_val * scale_dv).to(dv_ptr.dtype.element_ty))
