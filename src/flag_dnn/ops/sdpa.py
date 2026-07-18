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

import logging
import math
from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_LOG2E: float = 1.4426950408889634

_TOP_LEFT = "TOP_LEFT"
_BOTTOM_RIGHT = "BOTTOM_RIGHT"
_DIAGONAL_ALIGNMENTS = (_TOP_LEFT, _BOTTOM_RIGHT)

# Sentinel diagonal offsets meaning "unbounded" inside the kernel. They are
# far larger than any realistic sequence length while staying safely inside
# int32 once combined with block offsets.
_UNBOUNDED_DIAG = 1 << 30

_LOG2E_KERNEL = tl.constexpr(1.4426950408889634)

_DECODE_CHUNK_SIZE = 1024


_TMA_ALLOCATOR_SET = False


def _triton_tma_alloc(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def _ensure_triton_tma_allocator() -> None:
    global _TMA_ALLOCATOR_SET
    if not _TMA_ALLOCATOR_SET:
        triton.set_allocator(_triton_tma_alloc)
        _TMA_ALLOCATOR_SET = True


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
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new
    return acc, l_i, m_i


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sdpa"),
    key=[
        "SQ",
        "SKV",
        "HEAD_DIM",
        "V_DIM",
        "ELEM_SIZE",
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
        "default",
    ],
    warmup=5,
    rep=10,
)
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
        stats = m_i / _LOG2E_KERNEL + tl.log(l_safe)
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


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sdpa_dense_exact"),
    key=["SQ", "SKV", "HEAD_DIM", "V_DIM", "ELEM_SIZE"],
    strategy=["log", "log", "default", "default", "default"],
    warmup=5,
    rep=10,
)
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
    pid_m = tle.program_id(0)
    pid_bh = tle.program_id(1)
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


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sdpa_mha_causal_hostdesc"),
    key=["SQ", "SKV", "HEAD_DIM", "V_DIM", "ELEM_SIZE"],
    strategy=["log", "log", "default", "default", "default"],
    warmup=5,
    rep=10,
)
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
        m_i / _LOG2E_KERNEL + tl.log(l_safe),
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sdpa_gqa_causal"),
    key=["SQ", "SKV", "HEAD_DIM", "V_DIM", "ELEM_SIZE", "GROUP"],
    strategy=["log", "log", "default", "default", "default", "default"],
    warmup=5,
    rep=10,
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
    raw_pid_m = tle.program_id(0)
    pid_m = tl.cdiv(SQ, BLOCK_M) - 1 - raw_pid_m
    pid_bkv = tle.program_id(1)
    pid_hg = tle.program_id(2)
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

    stats = m_i / _LOG2E_KERNEL + tl.log(l_safe)
    tl.store(
        stats_ptr
        + off_b * stride_sb
        + q_head * stride_sh
        + offs_m * stride_sm,
        stats,
        mask=row_mask,
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sdpa_mha_causal_desc"),
    key=["SQ", "SKV", "HEAD_DIM", "V_DIM", "ELEM_SIZE"],
    strategy=["log", "log", "default", "default", "default"],
    warmup=5,
    rep=10,
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
    pid_bh = tle.program_id(0)
    raw_pid_m = tle.program_id(1)
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

    stats = m_i / _LOG2E_KERNEL + tl.log(l_safe)
    stats_base = stats_ptr + off_b * stride_sb + off_h * stride_sh
    tl.store(stats_base + offs_m * stride_sm, stats)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sdpa_gqa_causal_desc"),
    key=["SQ", "SKV", "HEAD_DIM", "V_DIM", "ELEM_SIZE", "GROUP"],
    strategy=["log", "log", "default", "default", "default", "default"],
    warmup=5,
    rep=10,
)
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
    raw_pid_m = tle.program_id(0)
    pid_m = tl.cdiv(SQ, BLOCK_M) - 1 - raw_pid_m
    pid_bkv = tle.program_id(1)
    pid_hg = tle.program_id(2)
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

    stats = m_i / _LOG2E_KERNEL + tl.log(l_safe)
    tl.store(
        stats_ptr
        + off_b * stride_sb
        + q_head * stride_sh
        + offs_m * stride_sm,
        stats,
        mask=row_mask,
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sdpa_decode"),
    key=["SKV", "CHUNK", "HEAD_DIM", "ELEM_SIZE"],
    strategy=["log", "log", "default", "default"],
    warmup=5,
    rep=10,
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
    pid_s = tle.program_id(0)
    pid_bh = tle.program_id(1)
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
    pid = tle.program_id(0)
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
        stats = m / _LOG2E_KERNEL + tl.log(l_safe)
        tl.store(stats_ptr + off_b * stride_sb + off_h * stride_sh, stats)


def _resolve_generate_stats(
    generate_stats: Optional[bool], is_inference: Optional[bool]
) -> bool:
    if generate_stats is None and is_inference is None:
        return False
    if generate_stats is None:
        return not bool(is_inference)
    generate_stats = bool(generate_stats)
    if is_inference is not None and bool(is_inference) == generate_stats:
        raise ValueError(
            "sdpa got conflicting generate_stats and is_inference; "
            "generate_stats must equal (not is_inference)"
        )
    return generate_stats


def _resolve_alignment(diagonal_alignment: Union[str, int, None]) -> str:
    if diagonal_alignment is None:
        return _TOP_LEFT
    if isinstance(diagonal_alignment, str):
        alignment = diagonal_alignment.upper()
    elif isinstance(diagonal_alignment, int):
        alignment = _DIAGONAL_ALIGNMENTS[diagonal_alignment]
    else:
        name = getattr(diagonal_alignment, "name", None)
        if name is None:
            raise TypeError(
                "sdpa diagonal_alignment must be 'TOP_LEFT' or "
                f"'BOTTOM_RIGHT', got {diagonal_alignment!r}"
            )
        alignment = str(name).upper()
    if alignment not in _DIAGONAL_ALIGNMENTS:
        raise ValueError(
            "sdpa diagonal_alignment must be 'TOP_LEFT' or 'BOTTOM_RIGHT', "
            f"got {diagonal_alignment!r}"
        )
    return alignment


def _resolve_band(
    use_causal_mask: bool,
    use_causal_mask_bottom_right: bool,
    sliding_window_length: Optional[int],
    diagonal_alignment: Union[str, int, None],
    diagonal_band_left_bound: Optional[int],
    diagonal_band_right_bound: Optional[int],
) -> Tuple[str, Optional[int], Optional[int]]:
    alignment = _resolve_alignment(diagonal_alignment)
    left = diagonal_band_left_bound
    right = diagonal_band_right_bound

    if use_causal_mask and use_causal_mask_bottom_right:
        raise ValueError(
            "sdpa cannot combine use_causal_mask and "
            "use_causal_mask_bottom_right"
        )
    if use_causal_mask or use_causal_mask_bottom_right:
        if right is not None:
            raise ValueError(
                "sdpa cannot combine deprecated causal flags with "
                "diagonal_band_right_bound"
            )
        right = 0
        if use_causal_mask_bottom_right:
            alignment = _BOTTOM_RIGHT
    if sliding_window_length is not None:
        if left is not None:
            raise ValueError(
                "sdpa cannot combine deprecated sliding_window_length with "
                "diagonal_band_left_bound"
            )
        left = int(sliding_window_length)

    if left is not None:
        left = int(left)
        if left < 1:
            raise ValueError(
                f"sdpa diagonal_band_left_bound must be >= 1, got {left}"
            )
    if right is not None:
        right = int(right)
        if right < 0:
            raise ValueError(
                f"sdpa diagonal_band_right_bound must be >= 0, got {right}"
            )
    return alignment, left, right


def _check_sdpa_inputs(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> None:
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.dim() != 4:
            raise RuntimeError(
                f"sdpa {name} must be a 4D (B, H, S, D) tensor, got "
                f"rank {tensor.dim()}"
            )
    if q.dtype not in _SUPPORTED_DTYPES:
        raise NotImplementedError(
            "flag_dnn sdpa supports fp16, bf16, and fp32 inputs, got "
            f"{q.dtype}"
        )
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise RuntimeError(
            "sdpa expects q, k, and v to share one dtype, got "
            f"{q.dtype}, {k.dtype}, {v.dtype}"
        )
    if k.device != q.device or v.device != q.device:
        raise RuntimeError(
            "sdpa expects q, k, and v on one device, got "
            f"{q.device}, {k.device}, {v.device}"
        )
    if k.shape[0] != q.shape[0] or v.shape[0] != q.shape[0]:
        raise RuntimeError(
            "sdpa q, k, and v batch sizes must match, got "
            f"{q.shape[0]}, {k.shape[0]}, {v.shape[0]}"
        )
    if k.shape[3] != q.shape[3]:
        raise RuntimeError(
            "sdpa q and k head dimensions must match, got "
            f"{q.shape[3]} and {k.shape[3]}"
        )
    if v.shape[2] != k.shape[2]:
        raise RuntimeError(
            "sdpa k and v sequence lengths must match, got "
            f"{k.shape[2]} and {v.shape[2]}"
        )
    if q.shape[1] % k.shape[1] != 0 or q.shape[1] % v.shape[1] != 0:
        raise RuntimeError(
            "sdpa query head count must be a multiple of key/value head "
            f"counts, got {q.shape[1]}, {k.shape[1]}, {v.shape[1]}"
        )


def _check_sdpa_bias(bias: torch.Tensor, q: torch.Tensor, skv: int) -> None:
    if bias.dim() != 4:
        raise RuntimeError(
            f"sdpa bias must be a 4D tensor, got rank {bias.dim()}"
        )
    batch, heads, sq = q.shape[0], q.shape[1], q.shape[2]
    if bias.shape[0] not in (1, batch) or bias.shape[1] not in (1, heads):
        raise RuntimeError(
            "sdpa bias batch/head dimensions must be 1 or match q, got "
            f"{tuple(bias.shape)}"
        )
    if bias.shape[2] != sq or bias.shape[3] != skv:
        raise RuntimeError(
            "sdpa bias trailing dimensions must be (seq_q, seq_kv) = "
            f"({sq}, {skv}), got {tuple(bias.shape)}"
        )
    if bias.dtype != q.dtype:
        raise RuntimeError(
            f"sdpa bias dtype must match q dtype, got {bias.dtype}"
        )
    if bias.device != q.device:
        raise RuntimeError("sdpa bias must be on the same device as q")


def _validate_dropout(dropout) -> None:
    if dropout is None:
        return
    if isinstance(dropout, (int, float)) and float(dropout) == 0.0:
        return
    raise NotImplementedError(
        "flag_dnn sdpa currently supports dropout=None (probability 0) only"
    )


def sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
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
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Scaled dot product flash attention forward.

    Mirrors the cuDNN frontend ``pygraph.sdpa`` semantics for the dense
    BHSD layout: ``softmax(attn_scale * q @ k^T + bias + mask) @ v``.
    Returns ``output`` when ``generate_stats`` is falsy, otherwise the
    ``(output, stats)`` pair where ``stats`` holds the per-row logsumexp
    of the masked attention scores in float32.
    """
    del compute_data_type, name
    _check_sdpa_inputs(q, k, v)
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

    batch, heads, sq, head_dim = (
        int(q.shape[0]),
        int(q.shape[1]),
        int(q.shape[2]),
        int(q.shape[3]),
    )
    skv = int(k.shape[2])
    v_dim = int(v.shape[3])

    if bias is not None:
        _check_sdpa_bias(bias, q, skv)

    if attn_scale is None:
        attn_scale = 1.0 / math.sqrt(head_dim) if head_dim > 0 else 1.0
    attn_scale = float(attn_scale)

    o = torch.empty((batch, heads, sq, v_dim), device=q.device, dtype=q.dtype)
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

        used_decode = False
        if (
            sq == 1
            and bias is None
            and not banded
            and q.dtype in (torch.float16, torch.bfloat16)
            and k.shape[1] == v.shape[1]
            and heads > int(k.shape[1])
        ):
            hkv = int(k.shape[1])
            group = heads // hkv
            chunk = min(skv, _DECODE_CHUNK_SIZE)
            splits = triton.cdiv(skv, chunk)
            part = torch.empty(
                (batch, heads, splits, v_dim + 2),
                device=q.device,
                dtype=torch.float32,
            )
            block_g = max(1, triton.next_power_of_2(group))
            block_d = max(16, triton.next_power_of_2(head_dim))
            block_dv = max(16, triton.next_power_of_2(v_dim))
            with torch_device_fn.device(q.device):
                _sdpa_decode_split_kernel[(splits, batch * hkv)](
                    q,
                    k,
                    v,
                    part,
                    attn_scale * _LOG2E,
                    hkv,
                    skv,
                    chunk,
                    q.stride(0),
                    q.stride(1),
                    q.stride(3),
                    k.stride(0),
                    k.stride(1),
                    k.stride(2),
                    k.stride(3),
                    v.stride(0),
                    v.stride(1),
                    v.stride(2),
                    v.stride(3),
                    part.stride(0),
                    part.stride(1),
                    part.stride(2),
                    part.stride(3),
                    GROUP=group,
                    HEAD_DIM=head_dim,
                    V_DIM=v_dim,
                    ELEM_SIZE=q.element_size(),
                    BLOCK_G=block_g,
                    BLOCK_D=block_d,
                    BLOCK_DV=block_dv,
                )
                _sdpa_decode_combine_kernel[(batch * heads,)](
                    part,
                    o,
                    stats_arg,
                    heads,
                    splits,
                    part.stride(0),
                    part.stride(1),
                    part.stride(2),
                    part.stride(3),
                    o.stride(0),
                    o.stride(1),
                    o.stride(3),
                    stride_stats[0],
                    stride_stats[1],
                    V_DIM=v_dim,
                    BLOCK_S=max(1, triton.next_power_of_2(splits)),
                    BLOCK_DV=block_dv,
                    GENERATE_STATS=stats is not None,
                )
            used_decode = True

        if (
            not used_decode
            and sq == 2048
            and skv == 2048
            and bias is None
            and stats is not None
            and banded
            and alignment == _TOP_LEFT
            and left is None
            and right == 0
            and q.dtype in (torch.float16, torch.bfloat16)
            and head_dim == 128
            and v_dim == 128
            and heads == int(k.shape[1])
            and heads == int(v.shape[1])
            and q.stride(3) == 1
            and k.stride(3) == 1
            and v.stride(3) == 1
        ):
            _ensure_triton_tma_allocator()

            def mha_desc_grid(meta):
                return (batch * heads, triton.cdiv(sq, meta["BLOCK_M"]))

            with torch_device_fn.device(q.device):
                _sdpa_fwd_mha_causal_desc_kernel[mha_desc_grid](
                    q,
                    k,
                    v,
                    o,
                    stats,
                    attn_scale * _LOG2E,
                    heads,
                    sq,
                    skv,
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
                    ELEM_SIZE=q.element_size(),
                    BLOCK_D=head_dim,
                    BLOCK_DV=v_dim,
                )
            used_decode = True

        if (
            not used_decode
            and sq == 4096
            and skv == 4096
            and bias is None
            and stats is not None
            and banded
            and alignment == _TOP_LEFT
            and left is None
            and right == 0
            and q.dtype in (torch.float16, torch.bfloat16)
            and head_dim == 128
            and v_dim == 128
            and k.shape[1] == v.shape[1]
            and heads > int(k.shape[1])
            and heads // int(k.shape[1]) <= 4
            and q.stride(3) == 1
            and k.stride(3) == 1
            and v.stride(3) == 1
        ):
            hkv = int(k.shape[1])
            group = heads // hkv
            _ensure_triton_tma_allocator()

            def gqa_desc_grid(meta):
                return (
                    triton.cdiv(sq, meta["BLOCK_M"]),
                    batch * hkv,
                    triton.cdiv(group, meta["BLOCK_H"]),
                )

            with torch_device_fn.device(q.device):
                _sdpa_fwd_gqa_causal_desc_kernel[gqa_desc_grid](
                    q,
                    k,
                    v,
                    o,
                    stats,
                    attn_scale * _LOG2E,
                    hkv,
                    sq,
                    skv,
                    group,
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
                    ELEM_SIZE=q.element_size(),
                    BLOCK_D=head_dim,
                    BLOCK_DV=v_dim,
                )
            used_decode = True

        if (
            not used_decode
            and sq > 1
            and sq == skv
            and bias is None
            and stats is not None
            and banded
            and alignment == _TOP_LEFT
            and left is None
            and right == 0
            and q.dtype in (torch.float16, torch.bfloat16)
            and head_dim == 128
            and v_dim == 128
            and k.shape[1] == v.shape[1]
            and heads > int(k.shape[1])
            and heads // int(k.shape[1]) <= 4
            and sq % 64 == 0
            and skv % 64 == 0
        ):
            hkv = int(k.shape[1])
            group = heads // hkv

            def gqa_grid(meta):
                return (
                    triton.cdiv(sq, meta["BLOCK_M"]),
                    batch * hkv,
                    triton.cdiv(group, meta["BLOCK_H"]),
                )

            with torch_device_fn.device(q.device):
                _sdpa_fwd_gqa_causal_kernel[gqa_grid](
                    q,
                    k,
                    v,
                    o,
                    stats,
                    attn_scale * _LOG2E,
                    hkv,
                    sq,
                    skv,
                    group,
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
                    ELEM_SIZE=q.element_size(),
                    BLOCK_D=head_dim,
                    BLOCK_DV=v_dim,
                )
            used_decode = True

        if (
            not used_decode
            and sq > 1
            and bias is None
            and stats is None
            and not banded
            and q.dtype in (torch.float16, torch.bfloat16)
            and head_dim == 128
            and v_dim == 128
            and sq % 64 == 0
            and skv % 64 == 0
        ):

            def dense_grid(meta):
                return (triton.cdiv(sq, meta["BLOCK_M"]), batch * heads)

            with torch_device_fn.device(q.device):
                _sdpa_fwd_dense_exact_kernel[dense_grid](
                    q,
                    k,
                    v,
                    o,
                    attn_scale * _LOG2E,
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
                    HEAD_DIM=head_dim,
                    V_DIM=v_dim,
                    ELEM_SIZE=q.element_size(),
                    BLOCK_D=head_dim,
                    BLOCK_DV=v_dim,
                )
            used_decode = True

        if not used_decode:

            def grid(meta):
                return (triton.cdiv(sq, meta["BLOCK_M"]), batch * heads)

            with torch_device_fn.device(q.device):
                _sdpa_fwd_kernel[grid](
                    q,
                    k,
                    v,
                    bias_arg,
                    o,
                    stats_arg,
                    attn_scale * _LOG2E,
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
                    ELEM_SIZE=q.element_size(),
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
        return o, stats
    return o
