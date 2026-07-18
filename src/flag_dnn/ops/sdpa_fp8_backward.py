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

import math
from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle

from flag_dnn.ops.sdpa import (
    _BOTTOM_RIGHT,
    _TOP_LEFT,
    _UNBOUNDED_DIAG,
    _resolve_band,
    _validate_dropout,
)
from flag_dnn.ops.sdpa_fp8 import _SUPPORTED_FP8_DTYPES, _as_scalar


@triton.jit
def _zero_sdpa_fp8_bwd_amax_kernel(
    amax_dq_ptr, amax_dk_ptr, amax_dv_ptr, amax_dp_ptr
):
    tl.store(amax_dq_ptr, 0.0)
    tl.store(amax_dk_ptr, 0.0)
    tl.store(amax_dv_ptr, 0.0)
    tl.store(amax_dp_ptr, 0.0)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sdpa_fp8_backward_dq"),
    key=["SQ", "SKV", "HEAD_DIM", "q_per_k", "BANDED"],
    strategy=["log", "log", "default", "default", "default"],
    warmup=5,
    rep=10,
)
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
    qk_scale,
    ov_descale,
    do_v_descale,
    dq_descale,
    scale_dq,
    scale_dp,
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
    pid_m = tle.program_id(0)
    pid_bh = tle.program_id(1)
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


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sdpa_fp8_backward_dkdv"),
    key=["SQ", "SKV", "HEAD_DIM", "Q_PER", "BANDED"],
    strategy=["log", "log", "default", "default", "default"],
    warmup=5,
    rep=10,
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
    qk_scale,
    ov_descale,
    do_v_descale,
    p_do_descale,
    dk_descale,
    dv_descale,
    scale_dk,
    scale_dv,
    scale_s,
    scale_dp,
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
    pid_n = tle.program_id(0)
    pid_bh = tle.program_id(1)
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


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sdpa_fp8_backward_gqa_accum"),
    key=["SQ", "SKV", "HEAD_DIM", "q_per_k", "BANDED"],
    strategy=["log", "log", "default", "default", "default"],
    warmup=5,
    rep=10,
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
    pid_n = tle.program_id(0)
    pid_bh = tle.program_id(1)
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


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sdpa_fp8_backward_gqa_reduce"),
    key=["SKV", "HEAD_DIM", "q_per_k"],
    strategy=["log", "default", "default"],
    warmup=5,
    rep=10,
)
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
    pid_n = tle.program_id(0)
    pid_bh = tle.program_id(1)
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


@libentry()
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
    pid_m = tle.program_id(0)
    pid_bh = tle.program_id(1)
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


@libentry()
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
    pid_m = tle.program_id(0)
    pid_bh = tle.program_id(1)
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


@libentry()
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
    pid_n = tle.program_id(0)
    pid_bh = tle.program_id(1)
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


def _reject_unsupported(
    use_padding_mask: bool,
    seq_len_q,
    seq_len_kv,
    dropout,
    sink_token,
    dSink_token,
) -> None:
    if use_padding_mask or seq_len_q is not None or seq_len_kv is not None:
        raise NotImplementedError(
            "sdpa_fp8_backward does not support padding or variable lengths"
        )
    _validate_dropout(dropout)
    if sink_token is not None or dSink_token is not None:
        raise NotImplementedError(
            "sdpa_fp8_backward does not support sink tokens"
        )


def _check_sdpa_fp8_backward_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    dO: torch.Tensor,
    stats: torch.Tensor,
) -> None:
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.dim() != 4:
            raise RuntimeError(
                f"sdpa_fp8_backward {name} must be a 4D (B, H, S, D) "
                f"tensor, got rank {tensor.dim()}"
            )
    if q.dtype not in _SUPPORTED_FP8_DTYPES:
        raise NotImplementedError(
            "flag_dnn sdpa_fp8_backward supports float8_e4m3fn and "
            f"float8_e5m2 inputs, got {q.dtype}"
        )
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise RuntimeError(
            "sdpa_fp8_backward expects q, k, and v to share one fp8 dtype, "
            f"got {q.dtype}, {k.dtype}, {v.dtype}"
        )
    if k.device != q.device or v.device != q.device:
        raise RuntimeError("sdpa_fp8_backward q, k, and v must share device")
    if k.shape[0] != q.shape[0] or v.shape[0] != q.shape[0]:
        raise RuntimeError("sdpa_fp8_backward q, k, and v batch mismatch")
    if k.shape[3] != q.shape[3]:
        raise RuntimeError(
            "sdpa_fp8_backward q and k head dimensions must match"
        )
    if k.shape[1] != v.shape[1]:
        raise NotImplementedError(
            "sdpa_fp8_backward currently requires k and v to have the same "
            "head count"
        )
    if k.shape[2] != v.shape[2]:
        raise RuntimeError(
            "sdpa_fp8_backward k and v sequence lengths must match"
        )
    if v.shape[3] != q.shape[3]:
        raise NotImplementedError(
            "sdpa_fp8_backward currently requires v head dimension to match q"
        )
    if q.shape[1] % k.shape[1] != 0:
        raise RuntimeError(
            "sdpa_fp8_backward query head count must be a multiple of "
            "key/value head count"
        )

    expected_o = (q.shape[0], q.shape[1], q.shape[2], v.shape[3])
    if tuple(o.shape) != expected_o or tuple(dO.shape) != expected_o:
        raise RuntimeError(
            "sdpa_fp8_backward expects o and dO shape "
            f"{expected_o}, got {tuple(o.shape)} and {tuple(dO.shape)}"
        )
    if o.dtype != q.dtype or dO.dtype != q.dtype:
        raise RuntimeError(
            "sdpa_fp8_backward expects q, o, and dO to share fp8 dtype, "
            f"got {q.dtype}, {o.dtype}, {dO.dtype}"
        )
    if o.device != q.device or dO.device != q.device:
        raise RuntimeError("sdpa_fp8_backward inputs must be on one device")

    expected_stats = (q.shape[0], q.shape[1], q.shape[2], 1)
    if tuple(stats.shape) != expected_stats:
        raise RuntimeError(
            "sdpa_fp8_backward expects stats shape "
            f"{expected_stats}, got {tuple(stats.shape)}"
        )
    if stats.dtype != torch.float32:
        raise RuntimeError(
            f"sdpa_fp8_backward stats must be float32, got {stats.dtype}"
        )
    if stats.device != q.device:
        raise RuntimeError(
            "sdpa_fp8_backward stats must be on the same device"
        )


def sdpa_fp8_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    dO: torch.Tensor,
    stats: torch.Tensor,
    descale_q,
    descale_k,
    descale_v,
    descale_o,
    descale_dO,
    descale_s,
    descale_dP,
    scale_s,
    scale_dQ,
    scale_dK,
    scale_dV,
    scale_dP,
    *,
    attn_scale: Optional[float] = None,
    use_padding_mask: bool = False,
    seq_len_q=None,
    seq_len_kv=None,
    use_causal_mask: bool = False,
    use_causal_mask_bottom_right: bool = False,
    sliding_window_length: Optional[int] = None,
    diagonal_alignment: Union[str, int, None] = _TOP_LEFT,
    diagonal_band_left_bound: Optional[int] = None,
    diagonal_band_right_bound: Optional[int] = None,
    dropout=None,
    use_deterministic_algorithm: bool = False,
    compute_data_type=None,
    name: str = "",
    sink_token=None,
    dSink_token=None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Scaled dot product attention backward with fp8 inputs and outputs."""
    del use_deterministic_algorithm, compute_data_type, name
    _reject_unsupported(
        use_padding_mask,
        seq_len_q,
        seq_len_kv,
        dropout,
        sink_token,
        dSink_token,
    )
    _check_sdpa_fp8_backward_inputs(q, k, v, o, dO, stats)

    descale_q = _as_scalar(descale_q, "descale_q")
    descale_k = _as_scalar(descale_k, "descale_k")
    descale_v = _as_scalar(descale_v, "descale_v")
    descale_o = _as_scalar(descale_o, "descale_o")
    descale_dO = _as_scalar(descale_dO, "descale_dO")
    descale_s = _as_scalar(descale_s, "descale_s")
    descale_dP = _as_scalar(descale_dP, "descale_dP")
    scale_s = _as_scalar(scale_s, "scale_s")
    scale_dQ = _as_scalar(scale_dQ, "scale_dQ")
    scale_dK = _as_scalar(scale_dK, "scale_dK")
    scale_dV = _as_scalar(scale_dV, "scale_dV")
    scale_dP = _as_scalar(scale_dP, "scale_dP")

    batch = int(q.shape[0])
    heads = int(q.shape[1])
    kv_heads = int(k.shape[1])
    sq = int(q.shape[2])
    skv = int(k.shape[2])
    head_dim = int(q.shape[3])
    if head_dim > 128:
        raise NotImplementedError(
            "sdpa_fp8_backward currently supports head_dim <= 128"
        )

    if attn_scale is None:
        attn_scale = 1.0 / math.sqrt(head_dim) if head_dim > 0 else 1.0
    attn_scale = float(attn_scale)

    dQ = torch.empty_like(q)
    dK = torch.empty_like(k)
    dV = torch.empty_like(v)
    amax_dQ = torch.empty((1, 1, 1, 1), device=q.device, dtype=torch.float32)
    amax_dK = torch.empty((1, 1, 1, 1), device=q.device, dtype=torch.float32)
    amax_dV = torch.empty((1, 1, 1, 1), device=q.device, dtype=torch.float32)
    amax_dP = torch.empty((1, 1, 1, 1), device=q.device, dtype=torch.float32)
    if dQ.numel() == 0 or dK.numel() == 0 or dV.numel() == 0:
        amax_dQ.zero_()
        amax_dK.zero_()
        amax_dV.zero_()
        amax_dP.zero_()
        return dQ, dK, dV, amax_dQ, amax_dK, amax_dV, amax_dP

    alignment, left, right = _resolve_band(
        use_causal_mask,
        use_causal_mask_bottom_right,
        sliding_window_length,
        diagonal_alignment,
        diagonal_band_left_bound,
        diagonal_band_right_bound,
    )
    shift = skv - sq if alignment == _BOTTOM_RIGHT else 0
    min_diag = 1 - left + shift if left is not None else -_UNBOUNDED_DIAG
    max_diag = right + shift if right is not None else _UNBOUNDED_DIAG
    banded = left is not None or right is not None
    causal_top_left = alignment == _TOP_LEFT and left is None and right == 0

    q_per_k = heads // kv_heads
    q_per_v = q_per_k
    qk_scale = descale_q * descale_k * attn_scale
    ov_descale = descale_o * descale_dO
    do_v_descale = descale_dO * descale_v
    p_do_descale = descale_s * descale_dO
    dq_descale = descale_dP * descale_k
    dk_descale = descale_dP * descale_q
    dv_descale = p_do_descale
    full_blocks = head_dim == 128 and sq % 128 == 0 and skv % 128 == 0

    with torch_device_fn.device(q.device):
        _zero_sdpa_fp8_bwd_amax_kernel[(1,)](
            amax_dQ, amax_dK, amax_dV, amax_dP
        )

        def grid_dq(meta):
            return (triton.cdiv(sq, meta["BLOCK_M"]), batch * heads)

        _sdpa_fp8_bwd_dq_kernel[grid_dq](
            q,
            k,
            v,
            o,
            dO,
            stats,
            dQ,
            amax_dQ,
            qk_scale,
            ov_descale,
            do_v_descale,
            dq_descale,
            scale_dQ,
            scale_dP,
            attn_scale,
            heads,
            sq,
            skv,
            q_per_k,
            q_per_v,
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
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            dO.stride(0),
            dO.stride(1),
            dO.stride(2),
            dO.stride(3),
            stats.stride(0),
            stats.stride(1),
            stats.stride(2),
            dQ.stride(0),
            dQ.stride(1),
            dQ.stride(2),
            dQ.stride(3),
            HEAD_DIM=head_dim,
            BANDED=banded,
            FULL_BLOCKS=full_blocks,
            CAUSAL_TOP_LEFT=causal_top_left,
        )

        use_gqa_split = q_per_k > 1 and full_blocks
        if use_gqa_split:
            dK_accum = torch.empty(
                (batch, heads, skv, head_dim),
                device=q.device,
                dtype=torch.float32,
            )
            dV_accum = torch.empty_like(dK_accum)

            def grid_gqa_accum(meta):
                return (triton.cdiv(skv, meta["BLOCK_N"]), batch * heads)

            _sdpa_fp8_bwd_gqa_accum_kernel[grid_gqa_accum](
                q,
                k,
                v,
                o,
                dO,
                stats,
                dK_accum,
                dV_accum,
                amax_dP,
                qk_scale,
                ov_descale,
                do_v_descale,
                scale_s,
                scale_dP,
                attn_scale,
                heads,
                sq,
                skv,
                q_per_k,
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
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),
                dO.stride(0),
                dO.stride(1),
                dO.stride(2),
                dO.stride(3),
                stats.stride(0),
                stats.stride(1),
                stats.stride(2),
                HEAD_DIM=head_dim,
                BANDED=banded,
                CAUSAL_TOP_LEFT=causal_top_left,
            )

            def grid_gqa_reduce(meta):
                return (triton.cdiv(skv, meta["BLOCK_N"]), batch * kv_heads)

            _sdpa_fp8_bwd_gqa_reduce_kernel[grid_gqa_reduce](
                dK_accum,
                dV_accum,
                dK,
                dV,
                amax_dK,
                amax_dV,
                dk_descale,
                dv_descale,
                scale_dK,
                scale_dV,
                heads,
                kv_heads,
                skv,
                q_per_k,
                dK.stride(0),
                dK.stride(1),
                dK.stride(2),
                dK.stride(3),
                dV.stride(0),
                dV.stride(1),
                dV.stride(2),
                dV.stride(3),
                HEAD_DIM=head_dim,
            )
        else:

            def grid_dkdv(meta):
                return (triton.cdiv(skv, meta["BLOCK_N"]), batch * kv_heads)

            _sdpa_fp8_bwd_dkdv_kernel[grid_dkdv](
                q,
                k,
                v,
                o,
                dO,
                stats,
                dK,
                dV,
                amax_dK,
                amax_dV,
                amax_dP,
                qk_scale,
                ov_descale,
                do_v_descale,
                p_do_descale,
                dk_descale,
                dv_descale,
                scale_dK,
                scale_dV,
                scale_s,
                scale_dP,
                attn_scale,
                kv_heads,
                sq,
                skv,
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
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),
                dO.stride(0),
                dO.stride(1),
                dO.stride(2),
                dO.stride(3),
                stats.stride(0),
                stats.stride(1),
                stats.stride(2),
                dK.stride(0),
                dK.stride(1),
                dK.stride(2),
                dK.stride(3),
                dV.stride(0),
                dV.stride(1),
                dV.stride(2),
                dV.stride(3),
                HEAD_DIM=head_dim,
                Q_PER=q_per_k,
                BANDED=banded,
                FULL_BLOCKS=full_blocks,
                CAUSAL_TOP_LEFT=causal_top_left,
            )

    return dQ, dK, dV, amax_dQ, amax_dK, amax_dV, amax_dP
