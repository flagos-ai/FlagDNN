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
from flag_dnn.utils import triton_lang_extension as tle
from flag_dnn.utils.device_info import get_device_capability_for
from flag_dnn.ops.sdpa import (
    _BOTTOM_RIGHT,
    _TOP_LEFT,
    _UNBOUNDED_DIAG,
    _check_sdpa_bias,
    _check_sdpa_inputs,
    _resolve_band,
    _validate_dropout,
)


_SM90_SAFE_TUNED_CONFIGS = {
    "sdpa_backward_fused_atomic_gqa_causal_d128": {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_D": 128,
        "num_warps": 4,
        "num_stages": 1,
    },
    "sdpa_backward_gqa_dq_delta_d128": {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_D": 128,
        "num_warps": 4,
        "num_stages": 1,
    },
    "sdpa_backward_owner_mha_causal_d128": {
        "BLOCK_M_DKDV": 64,
        "BLOCK_N_DKDV": 64,
        "BLOCK_M_DQ": 64,
        "BLOCK_N_DQ": 64,
        "BLOCK_D": 128,
        "num_warps": 4,
        "num_stages": 1,
    },
}

_NVIDIA_DEFAULT_TUNED_CONFIGS = {
    name: {**config, "num_stages": 2, "num_ctas": 1}
    for name, config in _SM90_SAFE_TUNED_CONFIGS.items()
}


def _single_tuned_config_kwargs(op_name: str, device=None) -> dict:
    configs = runtime.get_tuned_config(op_name)
    if len(configs) != 1:
        raise RuntimeError(
            f"sdpa_backward expected exactly one tuned config for {op_name}, "
            f"got {len(configs)}"
        )
    config = configs[0]
    kwargs = dict(config.kwargs)
    kwargs["num_warps"] = config.num_warps
    kwargs["num_stages"] = config.num_stages
    if hasattr(config, "num_ctas"):
        kwargs["num_ctas"] = config.num_ctas
    if (
        device is not None
        and runtime.device.vendor_name == "nvidia"
        and op_name in _SM90_SAFE_TUNED_CONFIGS
    ):
        capability = get_device_capability_for(device)
        if capability == (9, 0):
            kwargs = {
                **_SM90_SAFE_TUNED_CONFIGS[op_name],
                "num_ctas": 1,
            }
        elif capability != (0, 0):
            kwargs = dict(_NVIDIA_DEFAULT_TUNED_CONFIGS[op_name])
    return kwargs


def _tuned_config_supported_on_device(
    op_name: str, config: dict, device
) -> bool:
    if runtime.device.vendor_name != "nvidia":
        return True
    expected = _SM90_SAFE_TUNED_CONFIGS.get(op_name)
    if expected is None:
        return True
    capability = get_device_capability_for(device)
    if capability == (0, 0):
        return False
    if capability != (9, 0):
        return True
    return (
        all(config.get(key) == value for key, value in expected.items())
        and config.get("num_ctas", 1) == 1
    )


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
    pid_m = tle.program_id(0)
    pid_d = tle.program_id(1)
    pid_bh = tle.program_id(2)
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
    pid_n = tle.program_id(0)
    pid_d = tle.program_id(1)
    pid_bh = tle.program_id(2)
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
    pid_n = tle.program_id(0)
    pid_d = tle.program_id(1)
    pid_bh = tle.program_id(2)
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
    pid_n = tle.program_id(0)
    pid_dv = tle.program_id(1)
    pid_bh = tle.program_id(2)
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
    offsets = tle.program_id(0) * BLOCK + tl.arange(0, BLOCK)
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
    offsets = tle.program_id(0) * BLOCK + tl.arange(0, BLOCK)
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
    offsets = tle.program_id(0) * BLOCK + tl.arange(0, BLOCK)
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
    pid = tle.program_id(0)
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
    pid = tle.program_id(0)
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
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)
    pid_bh = tle.program_id(2)
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
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)
    pid_bh = tle.program_id(2)
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
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)
    pid_bh = tle.program_id(2)
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
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)
    pid_bh = tle.program_id(2)
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
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)
    pid_bh = tle.program_id(2)
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
    pid_t = tle.program_id(0)
    pid_bh = tle.program_id(1)
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
    pid_t = tle.program_id(0)
    pid_bh = tle.program_id(1)
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
    pid_t = tle.program_id(0)
    pid_bh = tle.program_id(1)
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
    pid_m = tle.program_id(0)
    pid_bh = tle.program_id(1)
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
    pid_m = tle.program_id(0)
    pid_bh = tle.program_id(1)
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
    pid_m = tle.program_id(0)
    pid_bh = tle.program_id(1)
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
    pid_m = tle.program_id(0)
    pid_bh = tle.program_id(1)
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
    pid_n = tle.program_id(0)
    pid_bh = tle.program_id(1)
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
    pid_n = tle.program_id(0)
    pid_bh = tle.program_id(1)
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


def _reject_unsupported(
    use_alibi_mask: bool,
    use_padding_mask: bool,
    seq_len_q,
    seq_len_kv,
    max_total_seq_len_q,
    max_total_seq_len_kv,
    rng_dump,
    score_mod,
    score_mod_bprop,
    sink_token,
    dSink_token,
) -> None:
    if use_alibi_mask:
        raise NotImplementedError("sdpa_backward does not support alibi mask")
    if use_padding_mask or seq_len_q is not None or seq_len_kv is not None:
        raise NotImplementedError(
            "sdpa_backward does not support padding or variable lengths"
        )
    if max_total_seq_len_q is not None or max_total_seq_len_kv is not None:
        raise NotImplementedError(
            "sdpa_backward does not support max_total_seq_len attributes"
        )
    if rng_dump is not None:
        raise NotImplementedError("sdpa_backward does not support rng_dump")
    if score_mod is not None or score_mod_bprop is not None:
        raise NotImplementedError(
            "sdpa_backward does not support score callbacks"
        )
    if sink_token is not None or dSink_token is not None:
        raise NotImplementedError("sdpa_backward does not support sink tokens")


def _check_sdpa_backward_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    dO: torch.Tensor,
    stats: torch.Tensor,
    dBias: Optional[torch.Tensor],
) -> None:
    _check_sdpa_inputs(q, k, v)
    if k.shape[1] != v.shape[1]:
        raise NotImplementedError(
            "sdpa_backward currently requires k and v to have the same "
            "head count"
        )
    expected_o = (q.shape[0], q.shape[1], q.shape[2], v.shape[3])
    if tuple(o.shape) != expected_o or tuple(dO.shape) != expected_o:
        raise RuntimeError(
            "sdpa_backward expects o and dO shape "
            f"{expected_o}, got {tuple(o.shape)} and {tuple(dO.shape)}"
        )
    if o.dtype != q.dtype or dO.dtype != q.dtype:
        raise RuntimeError(
            "sdpa_backward expects q, o, and dO to share dtype, got "
            f"{q.dtype}, {o.dtype}, {dO.dtype}"
        )
    if o.device != q.device or dO.device != q.device:
        raise RuntimeError("sdpa_backward inputs must be on one device")
    expected_stats = (q.shape[0], q.shape[1], q.shape[2], 1)
    if tuple(stats.shape) != expected_stats:
        raise RuntimeError(
            "sdpa_backward expects stats shape "
            f"{expected_stats}, got {tuple(stats.shape)}"
        )
    if stats.dtype != torch.float32:
        raise RuntimeError(
            f"sdpa_backward stats must be float32, got {stats.dtype}"
        )
    if stats.device != q.device:
        raise RuntimeError("sdpa_backward stats must be on the same device")
    if dBias is not None:
        if dBias.dim() != 4:
            raise RuntimeError(
                f"sdpa_backward dBias must be 4D, got rank {dBias.dim()}"
            )
        if dBias.shape[0] not in (1, q.shape[0]) or dBias.shape[1] not in (
            1,
            q.shape[1],
        ):
            raise RuntimeError(
                "sdpa_backward dBias batch/head dimensions must be 1 or "
                f"match q, got {tuple(dBias.shape)}"
            )
        if dBias.shape[2] != q.shape[2] or dBias.shape[3] != k.shape[2]:
            raise RuntimeError(
                "sdpa_backward dBias trailing dimensions must be "
                f"({q.shape[2]}, {k.shape[2]}), got {tuple(dBias.shape)}"
            )
        if dBias.dtype != q.dtype:
            raise RuntimeError(
                f"sdpa_backward dBias dtype must match q, got {dBias.dtype}"
            )
        if dBias.device != q.device:
            raise RuntimeError(
                "sdpa_backward dBias must be on the same device"
            )


def sdpa_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    dO: torch.Tensor,
    stats: torch.Tensor,
    *,
    attn_scale: Optional[float] = None,
    bias: Optional[torch.Tensor] = None,
    dBias: Optional[torch.Tensor] = None,
    use_alibi_mask: bool = False,
    use_padding_mask: bool = False,
    seq_len_q=None,
    seq_len_kv=None,
    max_total_seq_len_q=None,
    max_total_seq_len_kv=None,
    use_causal_mask: bool = False,
    use_causal_mask_bottom_right: bool = False,
    sliding_window_length: Optional[int] = None,
    diagonal_alignment: Union[str, int, None] = _TOP_LEFT,
    diagonal_band_left_bound: Optional[int] = None,
    diagonal_band_right_bound: Optional[int] = None,
    dropout=None,
    rng_dump=None,
    use_deterministic_algorithm: bool = False,
    compute_data_type=None,
    name: str = "",
    score_mod=None,
    score_mod_bprop=None,
    sink_token=None,
    dSink_token=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scaled dot product flash attention backward for dense BHSD tensors."""
    del compute_data_type, name
    _reject_unsupported(
        use_alibi_mask,
        use_padding_mask,
        seq_len_q,
        seq_len_kv,
        max_total_seq_len_q,
        max_total_seq_len_kv,
        rng_dump,
        score_mod,
        score_mod_bprop,
        sink_token,
        dSink_token,
    )
    _validate_dropout(dropout)
    del use_deterministic_algorithm
    _check_sdpa_backward_inputs(q, k, v, o, dO, stats, dBias)
    if bias is not None:
        _check_sdpa_bias(bias, q, int(k.shape[2]))

    batch = int(q.shape[0])
    heads = int(q.shape[1])
    kv_heads = int(k.shape[1])
    sq = int(q.shape[2])
    skv = int(k.shape[2])
    head_dim = int(q.shape[3])
    v_dim = int(v.shape[3])

    if attn_scale is None:
        attn_scale = 1.0 / math.sqrt(head_dim) if head_dim > 0 else 1.0
    attn_scale = float(attn_scale)

    dQ = torch.empty_like(q)
    dK = torch.empty_like(k)
    dV = torch.empty_like(v)
    if dQ.numel() == 0 or dK.numel() == 0 or dV.numel() == 0:
        return dQ, dK, dV

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
    causal_top_left = (
        alignment == _TOP_LEFT and left is None and right == 0 and sq == skv
    )

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

    has_dbias = dBias is not None
    dbias_reduce = False
    if has_dbias:
        dbias_reduce = dBias.shape[0] != batch or dBias.shape[1] != heads
        if dbias_reduce and not dBias.is_contiguous():
            raise NotImplementedError(
                "sdpa_backward dBias broadcast reduction requires a "
                "contiguous dBias tensor"
            )
        dbias_arg = dBias
        stride_dbias = (
            dBias.stride(0),
            dBias.stride(1),
            dBias.stride(2),
            dBias.stride(3),
        )
    else:
        dbias_arg = q
        stride_dbias = (0, 0, 0, 0)

    q_per_k = heads // kv_heads
    q_per_v = heads // int(v.shape[1])
    block_d_full = max(16, triton.next_power_of_2(head_dim))
    block_dv_full = max(16, triton.next_power_of_2(v_dim))
    dq_config = _single_tuned_config_kwargs("sdpa_backward_dq")
    dkdv_config = _single_tuned_config_kwargs("sdpa_backward_dkdv")
    dk_config = _single_tuned_config_kwargs("sdpa_backward_dk")
    dv_config = _single_tuned_config_kwargs("sdpa_backward_dv")
    use_fused_gqa = (
        causal_top_left
        and q_per_k == q_per_v
        and q_per_k > 1
        and head_dim == v_dim
        and head_dim <= 128
        and sq <= 4096
        and skv <= 4096
    )
    if (
        use_fused_gqa
        and runtime.device.vendor_name == "nvidia"
        and q.dtype == torch.bfloat16
        and head_dim == 128
        and v_dim == 128
        and sq == 4096
        and skv == 4096
        and get_device_capability_for(q.device) == (9, 0)
    ):
        # The SM90 fused atomic kernel has sparse dK accuracy outliers here.
        use_fused_gqa = False
    fused_config_name = "sdpa_backward_fused_atomic"
    if use_fused_gqa:
        fused_config_name = "sdpa_backward_fused_atomic_gqa_causal_d128"
    elif causal_top_left:
        fused_config_name = (
            "sdpa_backward_fused_atomic_causal_d128"
            if head_dim > 64
            else "sdpa_backward_fused_atomic_causal"
        )
    elif head_dim <= 32:
        fused_config_name = "sdpa_backward_fused_atomic_d32"
    elif head_dim > 64:
        fused_config_name = "sdpa_backward_fused_atomic_d128"
    fused_atomic_config = _single_tuned_config_kwargs(
        fused_config_name, device=q.device
    )
    if use_fused_gqa and not _tuned_config_supported_on_device(
        fused_config_name, fused_atomic_config, q.device
    ):
        use_fused_gqa = False
        fused_config_name = (
            "sdpa_backward_fused_atomic_causal_d128"
            if head_dim > 64
            else "sdpa_backward_fused_atomic_causal"
        )
        fused_atomic_config = _single_tuned_config_kwargs(fused_config_name)
    full_attention = False

    small_supported = sq <= 1024 and skv <= 1024
    long_causal_supported = causal_top_left and sq <= 4096 and skv <= 4096
    dense_d32_supported = (
        not causal_top_left and head_dim <= 32 and sq <= 2048 and skv <= 2048
    )
    decode_supported = not causal_top_left and sq == 1 and skv <= 2048
    use_fused_atomic = (
        (not banded or causal_top_left)
        and bias is None
        and not has_dbias
        and head_dim == v_dim
        and ((q_per_k == 1 and q_per_v == 1) or use_fused_gqa)
        and head_dim <= 128
        and (
            small_supported
            or long_causal_supported
            or dense_d32_supported
            or decode_supported
        )
        and q.dtype in (torch.float16, torch.bfloat16)
    )

    delta = torch.empty(
        (batch, heads, sq), device=q.device, dtype=torch.float32
    )

    with torch_device_fn.device(q.device):
        if use_fused_atomic:
            block_m = int(fused_atomic_config["BLOCK_M"])
            block_n = int(fused_atomic_config["BLOCK_N"])
            zero_grid = (
                triton.cdiv(max(dQ.numel(), dK.numel(), dV.numel()), 1024),
            )
            _zero_three_contiguous_kernel[zero_grid](
                dQ,
                dK,
                dV,
                dQ.numel(),
                dK.numel(),
                dV.numel(),
                BLOCK=1024,
            )
            fused_grid = (
                triton.cdiv(sq, block_m),
                triton.cdiv(skv, block_n),
                batch * heads,
            )
            if causal_top_left and use_fused_gqa:
                _sdpa_bwd_fused_atomic_gqa_causal_kernel[fused_grid](
                    q,
                    k,
                    v,
                    o,
                    dO,
                    stats,
                    dQ,
                    dK,
                    dV,
                    attn_scale,
                    heads,
                    q_per_k,
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
                    dQ.stride(0),
                    dQ.stride(1),
                    dQ.stride(2),
                    dQ.stride(3),
                    dK.stride(0),
                    dK.stride(1),
                    dK.stride(2),
                    dK.stride(3),
                    dV.stride(0),
                    dV.stride(1),
                    dV.stride(2),
                    dV.stride(3),
                    HEAD_DIM=head_dim,
                    BANDED=banded,
                    CAUSAL_TOP_LEFT=causal_top_left,
                    **fused_atomic_config,
                )
            elif causal_top_left:
                _sdpa_bwd_fused_atomic_causal_kernel[fused_grid](
                    q,
                    k,
                    v,
                    o,
                    dO,
                    stats,
                    dQ,
                    dK,
                    dV,
                    attn_scale,
                    heads,
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
                    dQ.stride(0),
                    dQ.stride(1),
                    dQ.stride(2),
                    dQ.stride(3),
                    dK.stride(0),
                    dK.stride(1),
                    dK.stride(2),
                    dK.stride(3),
                    dV.stride(0),
                    dV.stride(1),
                    dV.stride(2),
                    dV.stride(3),
                    HEAD_DIM=head_dim,
                    BANDED=banded,
                    CAUSAL_TOP_LEFT=causal_top_left,
                    **fused_atomic_config,
                )
            else:
                _sdpa_bwd_fused_atomic_kernel[fused_grid](
                    q,
                    k,
                    v,
                    o,
                    dO,
                    stats,
                    dQ,
                    dK,
                    dV,
                    attn_scale,
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
                    dK.stride(0),
                    dK.stride(1),
                    dK.stride(2),
                    dK.stride(3),
                    dV.stride(0),
                    dV.stride(1),
                    dV.stride(2),
                    dV.stride(3),
                    HEAD_DIM=head_dim,
                    **fused_atomic_config,
                )
            return dQ, dK, dV
        if dbias_reduce:
            grid_zero = (triton.cdiv(dBias.numel(), 1024),)
            _zero_contiguous_kernel[grid_zero](
                dBias, dBias.numel(), BLOCK=1024
            )

        def grid_dq(meta):
            return (
                triton.cdiv(sq, meta["BLOCK_M"]),
                triton.cdiv(head_dim, meta["BLOCK_D_OUT"]),
                batch * heads,
            )

        _sdpa_bwd_dq_dbias_kernel[grid_dq](
            q,
            k,
            v,
            bias_arg,
            o,
            dO,
            stats,
            delta,
            dQ,
            dbias_arg,
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
            stride_bias[0],
            stride_bias[1],
            stride_bias[2],
            stride_bias[3],
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
            delta.stride(0),
            delta.stride(1),
            delta.stride(2),
            dQ.stride(0),
            dQ.stride(1),
            dQ.stride(2),
            dQ.stride(3),
            stride_dbias[0],
            stride_dbias[1],
            stride_dbias[2],
            stride_dbias[3],
            HEAD_DIM=head_dim,
            V_DIM=v_dim,
            DBIAS_BATCHES=int(dBias.shape[0]) if has_dbias else 1,
            DBIAS_HEADS=int(dBias.shape[1]) if has_dbias else 1,
            BLOCK_D_FULL=block_d_full,
            BLOCK_DV=block_dv_full,
            FULL_ATTENTION=full_attention,
            HAS_BIAS=bias is not None,
            HAS_DBIAS=has_dbias,
            DBIAS_REDUCE=dbias_reduce,
            BANDED=banded,
            CAUSAL_TOP_LEFT=causal_top_left,
            **dq_config,
        )

        if head_dim == v_dim:

            def grid_dkdv(meta):
                return (
                    triton.cdiv(skv, meta["BLOCK_N"]),
                    triton.cdiv(head_dim, meta["BLOCK_D_OUT"]),
                    batch * kv_heads,
                )

            _sdpa_bwd_dkdv_kernel[grid_dkdv](
                q,
                k,
                v,
                bias_arg,
                dO,
                stats,
                delta,
                dK,
                dV,
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
                stride_bias[0],
                stride_bias[1],
                stride_bias[2],
                stride_bias[3],
                dO.stride(0),
                dO.stride(1),
                dO.stride(2),
                dO.stride(3),
                stats.stride(0),
                stats.stride(1),
                stats.stride(2),
                delta.stride(0),
                delta.stride(1),
                delta.stride(2),
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
                BLOCK_D_FULL=block_d_full,
                FULL_ATTENTION=full_attention,
                HAS_BIAS=bias is not None,
                BANDED=banded,
                CAUSAL_TOP_LEFT=causal_top_left,
                **dkdv_config,
            )
        else:

            def grid_dk(meta):
                return (
                    triton.cdiv(skv, meta["BLOCK_N"]),
                    triton.cdiv(head_dim, meta["BLOCK_D_OUT"]),
                    batch * kv_heads,
                )

            _sdpa_bwd_dk_kernel[grid_dk](
                q,
                k,
                v,
                bias_arg,
                dO,
                stats,
                delta,
                dK,
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
                stride_bias[0],
                stride_bias[1],
                stride_bias[2],
                stride_bias[3],
                dO.stride(0),
                dO.stride(1),
                dO.stride(2),
                dO.stride(3),
                stats.stride(0),
                stats.stride(1),
                stats.stride(2),
                delta.stride(0),
                delta.stride(1),
                delta.stride(2),
                dK.stride(0),
                dK.stride(1),
                dK.stride(2),
                dK.stride(3),
                HEAD_DIM=head_dim,
                V_DIM=v_dim,
                Q_PER=q_per_k,
                BLOCK_D_FULL=block_d_full,
                BLOCK_DV=block_dv_full,
                FULL_ATTENTION=full_attention,
                HAS_BIAS=bias is not None,
                BANDED=banded,
                CAUSAL_TOP_LEFT=causal_top_left,
                **dk_config,
            )

            def grid_dv(meta):
                return (
                    triton.cdiv(skv, meta["BLOCK_N"]),
                    triton.cdiv(v_dim, meta["BLOCK_DV_OUT"]),
                    batch * kv_heads,
                )

            _sdpa_bwd_dv_kernel[grid_dv](
                q,
                k,
                bias_arg,
                dO,
                stats,
                dV,
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
                stride_bias[0],
                stride_bias[1],
                stride_bias[2],
                stride_bias[3],
                dO.stride(0),
                dO.stride(1),
                dO.stride(2),
                dO.stride(3),
                stats.stride(0),
                stats.stride(1),
                stats.stride(2),
                dV.stride(0),
                dV.stride(1),
                dV.stride(2),
                dV.stride(3),
                HEAD_DIM=head_dim,
                V_DIM=v_dim,
                Q_PER=q_per_v,
                BLOCK_D_FULL=block_d_full,
                HAS_BIAS=bias is not None,
                BANDED=banded,
                CAUSAL_TOP_LEFT=causal_top_left,
                **dv_config,
            )

    return dQ, dK, dV
