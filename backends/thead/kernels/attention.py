# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""PPU attention kernels, derived from the common attention implementation.

Backward programs own disjoint query or key tiles within a batch/KV-head.
Broadcast bias gradients are reduced by their owning query tile. Bounded
tiles keep shared memory independent of sequence length without atomics.
"""
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
        qk = tl.dot(q, k, input_precision="ieee")
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
        acc += tl.dot(p.to(v.dtype), v, input_precision="ieee")
        m_i = m_new
    return (acc, l_i, m_i)


@triton.jit
def thead_sdpa_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    bias_ptr,
    o_ptr,
    stats_ptr,
    qk_scale: tl.constexpr,
    HQ: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    q_per_k: tl.constexpr,
    q_per_v: tl.constexpr,
    min_diag: tl.constexpr,
    max_diag: tl.constexpr,
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
    if HAS_BIAS:
        bias_base = bias_ptr + off_b * stride_bias_b + off_h * stride_bias_h
    else:
        bias_base = q_ptr
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
    lo_block = lo // BLOCK_N * BLOCK_N
    hi = tl.minimum(start_m + BLOCK_M - 1 + max_diag + 1, SKV)
    hi = tl.maximum(hi, lo_block)
    full_lo = tl.maximum(start_m + BLOCK_M - 1 + min_diag, 0)
    full_lo_block = tl.cdiv(full_lo, BLOCK_N) * BLOCK_N
    full_hi = tl.minimum(start_m + max_diag + 1, SKV)
    full_hi_block = full_hi // BLOCK_N * BLOCK_N
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
def _thead_sdpa_backward_tiles(
    q_ptr,
    k_ptr,
    v_ptr,
    bias_ptr,
    o_ptr,
    do_ptr,
    stats_ptr,
    batch,
    head,
    kv_head,
    row_start,
    column_start,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    SCALE: tl.constexpr,
    BIAS_BATCHES: tl.constexpr,
    BIAS_HEADS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    rows = row_start + tl.arange(0, BLOCK_M)
    columns = column_start + tl.arange(0, BLOCK_N)
    features = tl.arange(0, BLOCK_D)
    values = tl.arange(0, BLOCK_DV)
    q = tl.load(
        q_ptr
        + (batch * HQ + head) * SQ * D
        + rows[:, None] * D
        + features[None, :],
        (rows[:, None] < SQ) & (features[None, :] < D),
        other=0.0,
    )
    k = tl.load(
        k_ptr
        + (batch * HKV + kv_head) * SKV * D
        + columns[:, None] * D
        + features[None, :],
        (columns[:, None] < SKV) & (features[None, :] < D),
        other=0.0,
    )
    v = tl.load(
        v_ptr
        + (batch * HKV + kv_head) * SKV * DV
        + columns[:, None] * DV
        + values[None, :],
        (columns[:, None] < SKV) & (values[None, :] < DV),
        other=0.0,
    )
    do = tl.load(
        do_ptr
        + (batch * HQ + head) * SQ * DV
        + rows[:, None] * DV
        + values[None, :],
        (rows[:, None] < SQ) & (values[None, :] < DV),
        other=0.0,
    )
    o = tl.load(
        o_ptr
        + (batch * HQ + head) * SQ * DV
        + rows[:, None] * DV
        + values[None, :],
        (rows[:, None] < SQ) & (values[None, :] < DV),
        other=0.0,
    ).to(tl.float32)
    stats = tl.load(
        stats_ptr + (batch * HQ + head) * SQ + rows, rows < SQ, other=0.0
    )
    logits = (
        tl.dot(q, tl.trans(k), input_precision="ieee").to(tl.float32) * SCALE
    )
    if HAS_BIAS:
        bias_batch = 0 if BIAS_BATCHES == 1 else batch
        bias_head = 0 if BIAS_HEADS == 1 else head
        bias = tl.load(
            bias_ptr
            + (bias_batch * BIAS_HEADS + bias_head) * SQ * SKV
            + rows[:, None] * SKV
            + columns[None, :],
            (rows[:, None] < SQ) & (columns[None, :] < SKV),
            other=0.0,
        )
        logits += bias.to(tl.float32)
    visible = (rows[:, None] < SQ) & (columns[None, :] < SKV)
    if CAUSAL:
        visible = visible & (columns[None, :] <= rows[:, None])
    p = tl.where(visible, tl.exp(logits - stats[:, None]), 0.0)
    dp = tl.dot(do, tl.trans(v), input_precision="ieee").to(tl.float32)
    delta = tl.sum(o * do.to(tl.float32), 1)
    ds = p * (dp - delta[:, None])
    return q, k, do, p, ds


@triton.jit
def thead_sdpa_bwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    bias_ptr,
    o_ptr,
    do_ptr,
    stats_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    dbias_ptr,
    BATCH: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    SCALE: tl.constexpr,
    BIAS_BATCHES: tl.constexpr,
    BIAS_HEADS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_DBIAS: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    tile = tl.program_id(0)
    owner = tl.program_id(1)
    batch = owner // HKV
    kv_head = owner % HKV
    features = tl.arange(0, BLOCK_D)
    values = tl.arange(0, BLOCK_DV)
    if tile < tl.cdiv(SQ, BLOCK_M):
        row_start = tile * BLOCK_M
        rows = row_start + tl.arange(0, BLOCK_M)
        for group in range(HQ // HKV):
            head = kv_head * (HQ // HKV) + group
            dq = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
            for column_start in range(0, SKV, BLOCK_N):
                columns = column_start + tl.arange(0, BLOCK_N)
                q, k, do, p, ds = _thead_sdpa_backward_tiles(
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    bias_ptr,
                    o_ptr,
                    do_ptr,
                    stats_ptr,
                    batch,
                    head,
                    kv_head,
                    row_start,
                    column_start,
                    HQ,
                    HKV,
                    SQ,
                    SKV,
                    D,
                    DV,
                    SCALE,
                    BIAS_BATCHES,
                    BIAS_HEADS,
                    HAS_BIAS,
                    CAUSAL,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_D,
                    BLOCK_DV,
                )
                dq += (
                    tl.dot(ds.to(k.dtype), k, input_precision="ieee").to(
                        tl.float32
                    )
                    * SCALE
                )
                if HAS_DBIAS:
                    if BIAS_BATCHES == 1:
                        if batch == 0:
                            dbias = ds
                            for other_batch in range(1, BATCH):
                                _, _, _, _, other_ds = (
                                    _thead_sdpa_backward_tiles(
                                        q_ptr,
                                        k_ptr,
                                        v_ptr,
                                        bias_ptr,
                                        o_ptr,
                                        do_ptr,
                                        stats_ptr,
                                        other_batch,
                                        head,
                                        kv_head,
                                        row_start,
                                        column_start,
                                        HQ,
                                        HKV,
                                        SQ,
                                        SKV,
                                        D,
                                        DV,
                                        SCALE,
                                        BIAS_BATCHES,
                                        BIAS_HEADS,
                                        HAS_BIAS,
                                        CAUSAL,
                                        BLOCK_M,
                                        BLOCK_N,
                                        BLOCK_D,
                                        BLOCK_DV,
                                    )
                                )
                                dbias += other_ds
                            tl.store(
                                dbias_ptr
                                + head * SQ * SKV
                                + rows[:, None] * SKV
                                + columns[None, :],
                                dbias,
                                (rows[:, None] < SQ)
                                & (columns[None, :] < SKV),
                            )
                    else:
                        tl.store(
                            dbias_ptr
                            + (batch * HQ + head) * SQ * SKV
                            + rows[:, None] * SKV
                            + columns[None, :],
                            ds,
                            (rows[:, None] < SQ) & (columns[None, :] < SKV),
                        )
            tl.store(
                dq_ptr
                + (batch * HQ + head) * SQ * D
                + rows[:, None] * D
                + features[None, :],
                dq,
                (rows[:, None] < SQ) & (features[None, :] < D),
            )
    else:
        column_start = (tile - tl.cdiv(SQ, BLOCK_M)) * BLOCK_N
        columns = column_start + tl.arange(0, BLOCK_N)
        dk = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
        dv = tl.zeros((BLOCK_N, BLOCK_DV), tl.float32)
        for group in range(HQ // HKV):
            head = kv_head * (HQ // HKV) + group
            for row_start in range(0, SQ, BLOCK_M):
                q, k, do, p, ds = _thead_sdpa_backward_tiles(
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    bias_ptr,
                    o_ptr,
                    do_ptr,
                    stats_ptr,
                    batch,
                    head,
                    kv_head,
                    row_start,
                    column_start,
                    HQ,
                    HKV,
                    SQ,
                    SKV,
                    D,
                    DV,
                    SCALE,
                    BIAS_BATCHES,
                    BIAS_HEADS,
                    HAS_BIAS,
                    CAUSAL,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_D,
                    BLOCK_DV,
                )
                dk += (
                    tl.dot(
                        tl.trans(ds).to(q.dtype), q, input_precision="ieee"
                    ).to(tl.float32)
                    * SCALE
                )
                dv += tl.dot(
                    tl.trans(p).to(do.dtype), do, input_precision="ieee"
                ).to(tl.float32)
        tl.store(
            dk_ptr
            + (batch * HKV + kv_head) * SKV * D
            + columns[:, None] * D
            + features[None, :],
            dk,
            (columns[:, None] < SKV) & (features[None, :] < D),
        )
        tl.store(
            dv_ptr
            + (batch * HKV + kv_head) * SKV * DV
            + columns[:, None] * DV
            + values[None, :],
            dv,
            (columns[:, None] < SKV) & (values[None, :] < DV),
        )
