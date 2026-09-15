# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA warp layouts for short contiguous normalization rows."""

import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout


@g.jit
def layer_norm_warp_kernel(
    x_ptr,
    y_ptr,
    mean_ptr,
    inv_variance_ptr,
    weight_ptr,
    bias_ptr,
    M: gl.constexpr,
    N: gl.constexpr,
    EPS: gl.constexpr,
    R: gl.constexpr,
    L: gl.constexpr,
    V: gl.constexpr,
    W: gl.constexpr,
):
    layout: gl.constexpr = BlockedLayout([1, V], [32 // L, L], [W, 1], [1, 0])
    row = gl.program_id(0).to(gl.int64) * R + gl.arange(
        0, R, layout=SliceLayout(1, layout)
    )
    col = gl.arange(
        0, triton.next_power_of_2(N), layout=SliceLayout(0, layout)
    )
    x = gl.load(
        x_ptr + row[:, None] * N + col[None, :],
        (row[:, None] < M) & (col[None, :] < N),
        other=0,
    ).to(gl.float32)
    scale = gl.load(weight_ptr + col, col < N, other=0).to(gl.float32)
    bias = gl.load(bias_ptr + col, col < N, other=0).to(gl.float32)
    mean = gl.sum(x, 1) / N
    var = gl.maximum(gl.sum(x * x, 1) / N - mean * mean, 0)
    inv = gl.rsqrt(var + EPS)
    y = (x - mean[:, None]) * inv[:, None] * scale[None, :] + bias[None, :]
    gl.store(
        y_ptr + row[:, None] * N + col[None, :],
        y,
        (row[:, None] < M) & (col[None, :] < N),
    )
    gl.store(mean_ptr + row, mean, row < M)
    gl.store(inv_variance_ptr + row, inv, row < M)


@g.jit
def rms_norm_warp_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    inv_variance_ptr,
    M: gl.constexpr,
    N: gl.constexpr,
    EPS: gl.constexpr,
    V: gl.constexpr,
):
    # One row per 32-lane warp; four independent rows share each CTA.
    layout: gl.constexpr = BlockedLayout([1, V], [1, 32], [4, 1], [1, 0])
    row = gl.program_id(0).to(gl.int64) * 4 + gl.arange(
        0, 4, layout=SliceLayout(1, layout)
    )
    col = gl.arange(
        0, triton.next_power_of_2(N), layout=SliceLayout(0, layout)
    )
    active = (row[:, None] < M) & (col[None, :] < N)
    x = gl.load(x_ptr + row[:, None] * N + col[None, :], active, other=0).to(
        gl.float32
    )
    scale = gl.load(weight_ptr + col, col < N, other=0).to(gl.float32)
    bias = gl.load(bias_ptr + col, col < N, other=0).to(gl.float32)
    inv = gl.rsqrt(gl.sum(x * x, 1) / N + EPS)
    y = x * inv[:, None] * scale[None, :] + bias[None, :]
    gl.store(y_ptr + row[:, None] * N + col[None, :], y, active)
    gl.store(inv_variance_ptr + row, inv, row < M)
