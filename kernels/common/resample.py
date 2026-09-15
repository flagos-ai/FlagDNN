# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Spatial pooling and resize with arbitrary NCDHW storage strides."""
import triton
import triton.language as tl


@triton.jit
def resample_kernel(
    x_ptr,
    y_ptr,
    index_ptr,
    ELEMENTS: tl.constexpr,
    CHANNELS: tl.constexpr,
    ID: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SD: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PD: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    X_N: tl.constexpr,
    X_C: tl.constexpr,
    X_D: tl.constexpr,
    X_H: tl.constexpr,
    X_W: tl.constexpr,
    Y_N: tl.constexpr,
    Y_C: tl.constexpr,
    Y_D: tl.constexpr,
    Y_H: tl.constexpr,
    Y_W: tl.constexpr,
    I_N: tl.constexpr,
    I_C: tl.constexpr,
    I_D: tl.constexpr,
    I_H: tl.constexpr,
    I_W: tl.constexpr,
    MODE: tl.constexpr,
    PADDING: tl.constexpr,
    INDEX: tl.constexpr,
    ALIGN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_LAST: tl.constexpr = False,
):
    logical = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(
        0, BLOCK_SIZE
    )
    if CHANNELS_LAST:
        c = logical % CHANNELS
        spatial = logical // CHANNELS
        ow = spatial % OW
        oh = spatial // OW % OH
        od = spatial // (OW * OH) % OD
    else:
        ow = logical % OW
        oh = logical // OW % OH
        od = logical // (OW * OH) % OD
        c = logical // (OW * OH * OD) % CHANNELS
    n = logical // (OW * OH * OD * CHANNELS)
    valid = logical < ELEMENTS
    base = n * X_N + c * X_C
    destination = n * Y_N + c * Y_C + od * Y_D + oh * Y_H + ow * Y_W
    if MODE == 4:  # nearest, asymmetric floor mapping
        d = od * ID // OD
        h = oh * IH // OH
        w = ow * IW // OW
        result = tl.load(
            x_ptr + base + d * X_D + h * X_H + w * X_W, valid, other=0
        ).to(tl.float32)
    elif MODE == 3:  # bilinear, edge padding
        if ALIGN:
            h = oh.to(tl.float32) * ((IH - 1) / (OH - 1) if OH > 1 else 0.0)
            w = ow.to(tl.float32) * ((IW - 1) / (OW - 1) if OW > 1 else 0.0)
        else:
            h = (oh.to(tl.float32) + 0.5) * (IH / OH) - 0.5
            w = (ow.to(tl.float32) + 0.5) * (IW / OW) - 0.5
        h = tl.minimum(tl.maximum(h, 0.0), IH - 1.0)
        w = tl.minimum(tl.maximum(w, 0.0), IW - 1.0)
        h0 = h.to(tl.int64)
        w0 = w.to(tl.int64)
        h1 = tl.minimum(h0 + 1, IH - 1)
        w1 = tl.minimum(w0 + 1, IW - 1)
        dh = h - h0.to(tl.float32)
        dw = w - w0.to(tl.float32)
        a = tl.load(x_ptr + base + h0 * X_H + w0 * X_W, valid, other=0).to(
            tl.float32
        )
        b = tl.load(x_ptr + base + h0 * X_H + w1 * X_W, valid, other=0).to(
            tl.float32
        )
        c0 = tl.load(x_ptr + base + h1 * X_H + w0 * X_W, valid, other=0).to(
            tl.float32
        )
        d0 = tl.load(x_ptr + base + h1 * X_H + w1 * X_W, valid, other=0).to(
            tl.float32
        )
        result = ((1.0 - dw) * a + dw * b) * (1.0 - dh) + (
            (1.0 - dw) * c0 + dw * d0
        ) * dh
    else:
        result = tl.full(
            (BLOCK_SIZE,), float("-inf") if MODE == 5 else 0.0, tl.float32
        )
        best_index = tl.full((BLOCK_SIZE,), -1, tl.int32)
        samples = tl.zeros((BLOCK_SIZE,), tl.int32)
        for window in range(0, KD * KH * KW):
            d = od * SD - PD + window // (KH * KW)
            h = oh * SH - PH + window // KW % KH
            w = ow * SW - PW + window % KW
            inside = (
                (d >= 0) & (d < ID) & (h >= 0) & (h < IH) & (w >= 0) & (w < IW)
            )
            if PADDING == 1:
                d = tl.minimum(tl.maximum(d, 0), ID - 1)
                h = tl.minimum(tl.maximum(h, 0), IH - 1)
                w = tl.minimum(tl.maximum(w, 0), IW - 1)
                active = valid
            else:
                active = valid & inside
            sample = tl.load(
                x_ptr + base + d * X_D + h * X_H + w * X_W,
                active,
                other=float("-inf") if PADDING == 2 else 0.0,
            ).to(tl.float32)
            if MODE == 5:
                replace = (
                    (sample > result)
                    | (best_index == -1)
                    | ((sample != sample) & (result == result))
                )
                best_index = tl.where(replace, window, best_index)
                result = tl.maximum(result, sample)
            else:
                result += sample
                samples += inside.to(tl.int32)
        if MODE == 1:
            result = tl.where(
                samples > 0, result / tl.maximum(samples, 1), 0.0
            )
        elif MODE == 2:
            result /= KD * KH * KW
        if INDEX:
            tl.store(
                index_ptr + n * I_N + c * I_C + od * I_D + oh * I_H + ow * I_W,
                best_index,
                valid,
            )
    tl.store(y_ptr + destination, result, valid)
