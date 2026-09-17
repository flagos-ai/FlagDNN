#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Check every FProp tuning configuration with poisoned convolution tails."""

import argparse
import importlib.util
from pathlib import Path

import torch
import yaml

# N, Cin, Cout, H, W, kernel, padding, groups. Exercise both tails,
# spatially aligned group boundaries, multiple batches, and aligned K.
PROBLEMS = (
    (1, 4, 6, 7, 7, 3, 1, 2),
    (1, 4, 4, 10, 18, 3, 0, 2),
    (2, 4, 4, 10, 18, 3, 0, 2),
    (1, 16, 6, 7, 7, 2, 0, 2),
)


def check_problem(kernel, configurations, problem):
    batch, cin, cout, height, width, kh, pad, groups = problem
    oh, ow = height + 2 * pad - kh + 1, width + 2 * pad - kh + 1
    nx, nw, ny = (
        batch * cin * height * width,
        cout * (cin // groups) * kh * kh,
        batch * cout * oh * ow,
    )
    constants = dict(
        XH=height,
        XW=width,
        OH=oh,
        OW=ow,
        C_IN=cin,
        C_OUT=cout,
        BATCH=batch,
        CIN_PER_GROUP=cin // groups,
        COUT_PER_GROUP=cout // groups,
        GROUPS=groups,
        STRIDE_H=1,
        STRIDE_W=1,
        PAD_TOP=pad,
        PAD_LEFT=pad,
        DIL_H=1,
        DIL_W=1,
        KH=kh,
        KW=kh,
        HAS_BIAS=False,
        APPLY_RELU=False,
        BIAS_STRIDE_C=1,
        INPUT_PRECISION=1,
        X_STRIDE_N=cin * height * width,
        X_STRIDE_C=height * width,
        X_STRIDE_H=width,
        X_STRIDE_W=1,
        W_STRIDE_K=(cin // groups) * kh * kh,
        W_STRIDE_C=kh * kh,
        W_STRIDE_R=kh,
        W_STRIDE_S=1,
        Y_STRIDE_N=cout * oh * ow,
        Y_STRIDE_C=oh * ow,
        Y_STRIDE_H=ow,
        Y_STRIDE_W=1,
    )
    checked = 0
    for dtype_id, (dtype, atol) in enumerate(
        ((torch.float32, 1e-5), (torch.float16, 2e-2), (torch.bfloat16, 5e-2))
    ):
        host_x = torch.tensor(
            [((i * 17) % 41 - 20) / 13 for i in range(nx)], dtype=dtype
        ).reshape(batch, cin, height, width)
        host_w = torch.tensor(
            [((i * 17 + 11) % 41 - 20) / 14 for i in range(nw)], dtype=dtype
        ).reshape(cout, cin // groups, kh, kh)
        expected = torch.nn.functional.conv2d(
            host_x.float(), host_w.float(), padding=pad, groups=groups
        ).to(dtype)
        # Poison invalid loads deterministically instead of depending on the
        # contents of an adjacent allocator block.
        x_storage = torch.full((nx + 8192,), 17, dtype=dtype, device="cuda")
        w_storage = torch.full((nw + 8192,), 19, dtype=dtype, device="cuda")
        y_storage = torch.full((ny + 8192,), 23, dtype=dtype, device="cuda")
        x, w = x_storage[:nx].view_as(host_x), w_storage[:nw].view_as(host_w)
        y = y_storage[:ny].reshape(batch, cout, oh, ow)
        x.copy_(host_x)
        w.copy_(host_w)
        for meta, options in configurations:
            y_storage.fill_(23)
            torch.cuda.synchronize()
            tiles = ((oh * ow + meta["BLOCK_HW"] - 1) // meta["BLOCK_HW"]) * (
                (cout // groups + meta["BLOCK_OC"] - 1) // meta["BLOCK_OC"]
            )
            kernel[(tiles, batch * groups, 1)](
                x, w, x, y, **constants, DTYPE_ID=dtype_id, **meta, **options
            )
            torch.cuda.synchronize()
            label = f"problem={problem} dtype={dtype} META={meta} options={options}"
            torch.testing.assert_close(
                y.cpu(), expected, atol=atol, rtol=atol, msg=label
            )
            if not bool(torch.all(y_storage[ny:] == 23).item()):
                raise AssertionError(f"output guard overwritten: {label}")
            if not torch.equal(x.cpu(), host_x) or not torch.equal(w.cpu(), host_w):
                raise AssertionError(f"input modified: {label}")
            checked += 1
    return checked


def main():
    backend = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel-source", type=Path, default=backend / "kernels/convolution.py"
    )
    parser.add_argument("--problem", type=int, choices=range(len(PROBLEMS)))
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "iluvatar_convolution_contract", args.kernel_source
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entries = yaml.safe_load((backend / "tuning/convolution.yaml").read_text())[
        "conv_fprop"
    ]
    configurations, seen = [], set()
    for entry in entries:
        meta = {
            key: entry["META"][key]
            for key in ("BLOCK_HW", "BLOCK_OC", "BLOCK_K", "GROUP_M")
        }
        options = {
            key: entry.get(key, default)
            for key, default in (("num_warps", 4), ("num_stages", 1))
        }
        identity = tuple(sorted((meta | options).items()))
        if identity not in seen:
            seen.add(identity)
            configurations.append((meta, options))
    problems = PROBLEMS if args.problem is None else (PROBLEMS[args.problem],)
    checked = sum(
        check_problem(module.conv2d_spatial_nchw_kernel, configurations, problem)
        for problem in problems
    )
    print(f"PASS: grouped convolution tuning configurations={checked}")


if __name__ == "__main__":
    main()
