import logging
import math
from typing import Optional, Union

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle

from flag_dnn.ops.sdpa import (
    _BOTTOM_RIGHT,
    _LOG2E,
    _TOP_LEFT,
    _UNBOUNDED_DIAG,
    _ensure_triton_tma_allocator,
    _resolve_band,
    _resolve_generate_stats,
    _validate_dropout,
)

logger = logging.getLogger(__name__)

# fp8 io dtypes mirror cuDNN frontend sdpa_fp8 (FP8_E4M3 / FP8_E5M2).
_SUPPORTED_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

_LOG2E_KERNEL = tl.constexpr(1.4426950408889634)
# Natural-log of 2; converts a value expressed in log2 units back to natural
# units (1 / log2(e)). Used only to report amax_s in natural score units.
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
        # P (fp32) -> P (fp8) using scale_s, then P (fp8) @ V (fp8) -> fp32.
        p_fp8 = (p * s_scale).to(v.dtype)
        acc += tl.dot(p_fp8, v)
        m_i = m_new
    return acc, l_i, m_i


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sdpa_fp8"),
    key=[
        "SQ",
        "SKV",
        "HEAD_DIM",
        "V_DIM",
        "HAS_BIAS",
        "BANDED",
        "GENERATE_STATS",
        "REVERSE_CAUSAL",
    ],
    strategy=[
        "log",
        "log",
        "default",
        "default",
        "default",
        "default",
        "default",
        "default",
    ],
    warmup=5,
    rep=10,
)
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
    qk_scale,
    s_scale,
    sv_descale,
    o_scale,
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
    raw_pid_m = tle.program_id(0)
    if REVERSE_CAUSAL:
        pid_m = tl.cdiv(SQ, BLOCK_M) - 1 - raw_pid_m
    else:
        pid_m = raw_pid_m
    pid_bh = tle.program_id(1)
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
    tl.atomic_max(amax_o_ptr, local_amax_o)
    # amax_s proxy: largest row-max score magnitude (cheap; derived from the
    # online-softmax running max instead of a per-block abs+reduction). Used
    # only as a delayed-scaling calibration hint, not asserted for equality.
    amax_s_val = tl.max(tl.where(mask_m, tl.abs(m_i), 0.0)) * _LN2_KERNEL
    tl.atomic_max(amax_s_ptr, amax_s_val)

    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh
    tl.store(
        o_base + offs_m[:, None] * stride_om + offs_dv[None, :] * stride_od,
        (o_val * o_scale).to(o_ptr.dtype.element_ty),
        mask=o_valid,
    )

    if GENERATE_STATS:
        stats = m_i / _LOG2E_KERNEL + tl.log(l_safe)
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
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(
            k_base + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn
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
        p = tl.exp2(score - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(
            v_base + offs_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd
        )
        acc += tl.dot((p * s_scale).to(v.dtype), v)
        m_i = m_new
    return acc, l_i, m_i


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sdpa_fp8_fast"),
    key=[
        "SQ",
        "SKV",
        "HEAD_DIM",
        "V_DIM",
        "Q_PER_K",
        "CAUSAL",
        "GENERATE_STATS",
    ],
    strategy=[
        "log",
        "log",
        "default",
        "default",
        "default",
        "default",
        "default",
    ],
    warmup=5,
    rep=10,
)
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
    raw_pid_m = tle.program_id(0)
    if CAUSAL:
        pid_m = tl.cdiv(SQ, BLOCK_M) - 1 - raw_pid_m
    else:
        pid_m = raw_pid_m
    pid_bh = tle.program_id(1)
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
    local_amax_o = tl.max(tl.where(mask_m[:, None], tl.abs(o_val), 0.0))
    tl.atomic_max(amax_o_ptr, local_amax_o)
    amax_s_val = tl.max(tl.where(mask_m, tl.abs(m_i), 0.0)) * _LN2_KERNEL
    tl.atomic_max(amax_s_ptr, amax_s_val)

    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh
    tl.store(
        o_base + offs_m[:, None] * stride_om + offs_dv[None, :] * stride_od,
        (o_val * o_scale).to(o_ptr.dtype.element_ty),
        mask=mask_m[:, None],
    )
    if GENERATE_STATS:
        stats = m_i / _LOG2E_KERNEL + tl.log(l_safe)
        stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh
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
    for start_n in range(lo, hi, BLOCK_N):
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
        p = tl.exp2(score - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = v_desc.load([start_n_i32, 0])
        acc += tl.dot((p * s_scale).to(v.dtype), v)
        m_i = m_new
    return acc, l_i, m_i


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sdpa_fp8_tma"),
    key=[
        "SQ",
        "SKV",
        "HEAD_DIM",
        "V_DIM",
        "Q_PER_K",
        "CAUSAL",
        "GENERATE_STATS",
    ],
    strategy=[
        "log",
        "log",
        "default",
        "default",
        "default",
        "default",
        "default",
    ],
    warmup=5,
    rep=10,
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
    raw_pid_m = tle.program_id(0)
    if CAUSAL:
        pid_m = tl.cdiv(SQ, BLOCK_M) - 1 - raw_pid_m
    else:
        pid_m = raw_pid_m
    pid_bh = tle.program_id(1)
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
    local_amax_o = tl.max(tl.where(mask_m[:, None], tl.abs(o_val), 0.0))
    tl.atomic_max(amax_o_ptr, local_amax_o)
    amax_s_val = tl.max(tl.where(mask_m, tl.abs(m_i), 0.0)) * _LN2_KERNEL
    tl.atomic_max(amax_s_ptr, amax_s_val)

    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh
    tl.store(
        o_base + offs_m[:, None] * stride_om + offs_dv[None, :] * stride_od,
        (o_val * o_scale).to(o_ptr.dtype.element_ty),
        mask=mask_m[:, None],
    )
    if GENERATE_STATS:
        stats = m_i / _LOG2E_KERNEL + tl.log(l_safe)
        stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh
        tl.store(stats_base + offs_m * stride_sm, stats, mask=mask_m)


def _as_scalar(value, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeError(
                f"sdpa_fp8 {name} must be a scalar (1-element) tensor "
                f"or float, got shape {tuple(value.shape)}"
            )
        return float(value.detach().reshape(()).item())
    return float(value)


def _check_sdpa_fp8_inputs(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> None:
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.dim() != 4:
            raise RuntimeError(
                f"sdpa_fp8 {name} must be a 4D (B, H, S, D) tensor, got "
                f"rank {tensor.dim()}"
            )
    if q.dtype not in _SUPPORTED_FP8_DTYPES:
        raise NotImplementedError(
            "flag_dnn sdpa_fp8 supports float8_e4m3fn and float8_e5m2 inputs, "
            f"got {q.dtype}"
        )
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise RuntimeError(
            "sdpa_fp8 expects q, k, and v to share one fp8 dtype, got "
            f"{q.dtype}, {k.dtype}, {v.dtype}"
        )
    if k.device != q.device or v.device != q.device:
        raise RuntimeError(
            "sdpa_fp8 expects q, k, and v on one device, got "
            f"{q.device}, {k.device}, {v.device}"
        )
    if k.shape[0] != q.shape[0] or v.shape[0] != q.shape[0]:
        raise RuntimeError(
            "sdpa_fp8 q, k, and v batch sizes must match, got "
            f"{q.shape[0]}, {k.shape[0]}, {v.shape[0]}"
        )
    if k.shape[3] != q.shape[3]:
        raise RuntimeError(
            "sdpa_fp8 q and k head dimensions must match, got "
            f"{q.shape[3]} and {k.shape[3]}"
        )
    if v.shape[2] != k.shape[2]:
        raise RuntimeError(
            "sdpa_fp8 k and v sequence lengths must match, got "
            f"{k.shape[2]} and {v.shape[2]}"
        )
    if q.shape[1] % k.shape[1] != 0 or q.shape[1] % v.shape[1] != 0:
        raise RuntimeError(
            "sdpa_fp8 query head count must be a multiple of key/value head "
            f"counts, got {q.shape[1]}, {k.shape[1]}, {v.shape[1]}"
        )


def _check_sdpa_fp8_bias(
    bias: torch.Tensor, q: torch.Tensor, skv: int
) -> None:
    if bias.dim() != 4:
        raise RuntimeError(
            f"sdpa_fp8 bias must be a 4D tensor, got rank {bias.dim()}"
        )
    batch, heads, sq = q.shape[0], q.shape[1], q.shape[2]
    if bias.shape[0] not in (1, batch) or bias.shape[1] not in (1, heads):
        raise RuntimeError(
            "sdpa_fp8 bias batch/head dimensions must be 1 or match q, got "
            f"{tuple(bias.shape)}"
        )
    if bias.shape[2] != sq or bias.shape[3] != skv:
        raise RuntimeError(
            "sdpa_fp8 bias trailing dimensions must be (seq_q, seq_kv) = "
            f"({sq}, {skv}), got {tuple(bias.shape)}"
        )
    if bias.device != q.device:
        raise RuntimeError("sdpa_fp8 bias must be on the same device as q")


def sdpa_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    descale_q,
    descale_k,
    descale_v,
    descale_s,
    scale_s,
    scale_o,
    is_inference: Optional[bool] = None,
    *,
    attn_scale: Optional[float] = None,
    bias: Optional[torch.Tensor] = None,
    use_causal_mask: bool = False,
    use_causal_mask_bottom_right: bool = False,
    sliding_window_length: Optional[int] = None,
    diagonal_alignment: Union[str, int, None] = _TOP_LEFT,
    diagonal_band_left_bound: Optional[int] = None,
    diagonal_band_right_bound: Optional[int] = None,
    dropout=None,
    generate_stats: Optional[bool] = None,
    compute_data_type=None,
    name: str = "",
):
    """Scaled dot product flash attention forward with fp8 inputs/outputs.

    Mirrors the cuDNN frontend ``pygraph.sdpa_fp8`` semantics for the dense
    BHSD layout using per-tensor (current) scaling. ``q``/``k``/``v`` are fp8
    (``float8_e4m3fn`` or ``float8_e5m2``); ``descale_*`` and ``scale_*`` are
    scalar (Python float or 1-element tensor) calibration factors.

    Returns ``(output, amax_s, amax_o)`` when ``generate_stats`` is falsy,
    otherwise ``(output, stats, amax_s, amax_o)`` where ``stats`` holds the
    per-row logsumexp of the masked scores in float32. ``output`` is fp8
    (same dtype as ``q``), ``amax_s``/``amax_o`` are float32 scalars (1,1,1,1).
    """
    del compute_data_type, name
    _check_sdpa_fp8_inputs(q, k, v)
    _validate_dropout(dropout)
    stats_requested = _resolve_generate_stats(generate_stats, is_inference)
    alignment, left, right = _resolve_band(
        use_causal_mask,
        use_causal_mask_bottom_right,
        sliding_window_length,
        diagonal_alignment,
        diagonal_band_left_bound,
        diagonal_band_right_bound,
    )

    descale_q = _as_scalar(descale_q, "descale_q")
    descale_k = _as_scalar(descale_k, "descale_k")
    descale_v = _as_scalar(descale_v, "descale_v")
    descale_s = _as_scalar(descale_s, "descale_s")
    scale_s = _as_scalar(scale_s, "scale_s")
    scale_o = _as_scalar(scale_o, "scale_o")

    batch, heads, sq, head_dim = (
        int(q.shape[0]),
        int(q.shape[1]),
        int(q.shape[2]),
        int(q.shape[3]),
    )
    skv = int(k.shape[2])
    v_dim = int(v.shape[3])

    if bias is not None:
        _check_sdpa_fp8_bias(bias, q, skv)

    if attn_scale is None:
        attn_scale = 1.0 / math.sqrt(head_dim) if head_dim > 0 else 1.0
    attn_scale = float(attn_scale)

    o = torch.empty((batch, heads, sq, v_dim), device=q.device, dtype=q.dtype)
    amax_s = torch.zeros((1, 1, 1, 1), device=q.device, dtype=torch.float32)
    amax_o = torch.zeros((1, 1, 1, 1), device=q.device, dtype=torch.float32)
    stats = None
    if stats_requested:
        stats = torch.empty(
            (batch, heads, sq, 1), device=q.device, dtype=torch.float32
        )

    if o.numel() != 0 and head_dim != 0:
        shift = skv - sq if alignment == _BOTTOM_RIGHT else 0
        min_diag = 1 - left + shift if left is not None else -_UNBOUNDED_DIAG
        max_diag = right + shift if right is not None else _UNBOUNDED_DIAG
        banded = left is not None or right is not None
        reverse_causal = alignment == _TOP_LEFT and left is None and right == 0

        if bias is not None:
            bias_arg = bias
            stride_bias = (
                bias.stride(0) if bias.shape[0] != 1 else 0,
                bias.stride(1) if bias.shape[1] != 1 else 0,
                bias.stride(2),
                bias.stride(3),
            )
        else:
            bias_arg = q
            stride_bias = (0, 0, 0, 0)
        if stats is not None:
            stats_arg = stats
            stride_stats = (stats.stride(0), stats.stride(1), stats.stride(2))
        else:
            stats_arg = o
            stride_stats = (0, 0, 0)

        qk_scale = attn_scale * descale_q * descale_k * _LOG2E
        sv_descale = descale_s * descale_v

        # Lean fast path: dense or top-left causal, no bias, power-of-two
        # head_dim (so BLOCK_D == head_dim needs no padding mask). Strides are
        # baked in as constexpr to fold address arithmetic, mirroring the fp16
        # dense-exact / causal-desc fast kernels that reach cuDNN parity.
        pure_causal = (
            banded and left is None and right == 0 and alignment == _TOP_LEFT
        )
        head_pow2 = head_dim >= 16 and (head_dim & (head_dim - 1)) == 0
        v_pow2 = v_dim >= 16 and (v_dim & (v_dim - 1)) == 0
        fast_ok = (
            bias is None
            and head_dim == v_dim
            and head_pow2
            and v_pow2
            and (not banded or pure_causal)
        )

        # TMA path needs a contiguous innermost dim and a 16B-aligned box.
        # Its descriptor setup only amortizes for large KV work, so restrict it
        # to long sequences (>=2048) or dense medium sequences (>=1024); short
        # / medium-causal cases stay on the lower-overhead lean kernel.
        tma_amortizes = skv >= 2048 or (skv >= 1024 and not pure_causal)
        tma_ok = (
            fast_ok
            and tma_amortizes
            and q.stride(3) == 1
            and k.stride(3) == 1
            and v.stride(3) == 1
            and head_dim % 16 == 0
            and v_dim % 16 == 0
        )

        def fast_grid(meta):
            return (triton.cdiv(sq, meta["BLOCK_M"]), batch * heads)

        used_fast = False
        if tma_ok:
            _ensure_triton_tma_allocator()
            with torch_device_fn.device(q.device):
                _sdpa_fp8_fwd_tma_kernel[fast_grid](
                    q,
                    k,
                    v,
                    o,
                    stats_arg,
                    amax_s,
                    amax_o,
                    qk_scale,
                    scale_s,
                    sv_descale,
                    scale_o,
                    heads,
                    sq,
                    skv,
                    heads // int(k.shape[1]),
                    heads // int(v.shape[1]),
                    q.stride(0),
                    q.stride(1),
                    q.stride(2),
                    q.stride(3),
                    k.stride(0),
                    k.stride(1),
                    k.stride(2),
                    k.stride(3),
                    v.stride(0),
                    v.stride(1),
                    v.stride(2),
                    v.stride(3),
                    o.stride(0),
                    o.stride(1),
                    o.stride(2),
                    o.stride(3),
                    stride_stats[0],
                    stride_stats[1],
                    stride_stats[2],
                    HEAD_DIM=head_dim,
                    V_DIM=v_dim,
                    BLOCK_D=head_dim,
                    BLOCK_DV=v_dim,
                    CAUSAL=pure_causal,
                    GENERATE_STATS=stats is not None,
                )
            used_fast = True
        elif fast_ok:
            with torch_device_fn.device(q.device):
                _sdpa_fp8_fwd_fast_kernel[fast_grid](
                    q,
                    k,
                    v,
                    o,
                    stats_arg,
                    amax_s,
                    amax_o,
                    qk_scale,
                    scale_s,
                    sv_descale,
                    scale_o,
                    heads,
                    sq,
                    skv,
                    heads // int(k.shape[1]),
                    heads // int(v.shape[1]),
                    q.stride(0),
                    q.stride(1),
                    q.stride(2),
                    q.stride(3),
                    k.stride(0),
                    k.stride(1),
                    k.stride(2),
                    k.stride(3),
                    v.stride(0),
                    v.stride(1),
                    v.stride(2),
                    v.stride(3),
                    o.stride(0),
                    o.stride(1),
                    o.stride(2),
                    o.stride(3),
                    stride_stats[0],
                    stride_stats[1],
                    stride_stats[2],
                    HEAD_DIM=head_dim,
                    V_DIM=v_dim,
                    BLOCK_D=head_dim,
                    BLOCK_DV=v_dim,
                    CAUSAL=pure_causal,
                    GENERATE_STATS=stats is not None,
                )
            used_fast = True

        def grid(meta):
            return (triton.cdiv(sq, meta["BLOCK_M"]), batch * heads)

        if not used_fast:
            with torch_device_fn.device(q.device):
                _sdpa_fp8_fwd_kernel[grid](
                    q,
                    k,
                    v,
                    bias_arg,
                    o,
                    stats_arg,
                    amax_s,
                    amax_o,
                    qk_scale,
                    scale_s,
                    sv_descale,
                    scale_o,
                    heads,
                    sq,
                    skv,
                    heads // int(k.shape[1]),
                    heads // int(v.shape[1]),
                    min_diag,
                    max_diag,
                    q.stride(0),
                    q.stride(1),
                    q.stride(2),
                    q.stride(3),
                    k.stride(0),
                    k.stride(1),
                    k.stride(2),
                    k.stride(3),
                    v.stride(0),
                    v.stride(1),
                    v.stride(2),
                    v.stride(3),
                    stride_bias[0],
                    stride_bias[1],
                    stride_bias[2],
                    stride_bias[3],
                    o.stride(0),
                    o.stride(1),
                    o.stride(2),
                    o.stride(3),
                    stride_stats[0],
                    stride_stats[1],
                    stride_stats[2],
                    HEAD_DIM=head_dim,
                    V_DIM=v_dim,
                    BLOCK_D=max(16, triton.next_power_of_2(head_dim)),
                    BLOCK_DV=max(16, triton.next_power_of_2(v_dim)),
                    HAS_BIAS=bias is not None,
                    BANDED=banded,
                    GENERATE_STATS=stats is not None,
                    REVERSE_CAUSAL=reverse_causal,
                )
    elif stats is not None:
        stats.fill_(float("-inf"))

    if stats is not None:
        return o, stats, amax_s, amax_o
    return o, amax_s, amax_o
