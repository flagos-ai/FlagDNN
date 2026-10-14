#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Reproducible dense GEMM throughput sweep; no sparse or advertised peaks.

CUDA events time batches on the GPU timeline, excluding input preparation and
correctness checks. FP8 uses the allocating _scaled_mm API; any inter-launch
gap is retained. Other paths preallocate output. Five batches are retained;
select
best median across sizes rather than the fastest individual sample. See
docs/equivalent-performance-baseline.md.
"""
import argparse
import datetime
import importlib.metadata
import json
import statistics
import subprocess
from pathlib import Path

import torch


def make_case(kind, n, device):
    # This installation rejects FP16/BF16 cublasGemmEx; explicit cuBLASLt
    # works.
    torch.backends.cuda.preferred_blas_library(
        "cublaslt" if kind in ("fp16", "bf16") else "cublas"
    )
    torch.backends.cuda.matmul.allow_tf32 = kind == "tf32"
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dtype = {
        "fp64": torch.float64,
        "fp32": torch.float32,
        "tf32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp8_e4m3": torch.float8_e4m3fn,
        "fp8_e5m2_e4m3": torch.float8_e5m2,
        "int8": torch.int8,
    }[kind]
    if kind == "int8":
        a = torch.randint(-3, 4, (n, n), device=device, dtype=dtype)
        b = torch.randint(-3, 4, (n, n), device=device, dtype=dtype).t()
        out = torch.empty((n, n), device=device, dtype=torch.int32)

        def run():
            return torch._int_mm(a, b, out=out)

    else:
        a = (torch.rand((n, n), device=device) - 0.5).to(dtype)
        b_dtype = torch.float8_e4m3fn if kind.startswith("fp8") else dtype
        b = (torch.rand((n, n), device=device) - 0.5).to(b_dtype).t()
        if kind.startswith("fp8"):
            scale = torch.ones((), device=device, dtype=torch.float32)

            def run():
                return torch._scaled_mm(
                    a,
                    b,
                    scale_a=scale,
                    scale_b=scale,
                    out_dtype=torch.float16,
                    use_fast_accum=False,
                )

        else:
            out = torch.empty((n, n), device=device, dtype=dtype)

            def run():
                return torch.mm(a, b, out=out)

    return a, b, run


def validate(a, b, out, kind):
    # A distributed 8x8 sample using already-quantized inputs, calculated
    # on CPU in float64. This is a sanity check, not a proof for every output
    # element.
    n = a.shape[0]
    idx = torch.linspace(0, n - 1, 8, dtype=torch.long, device=a.device)
    # FP8 CUDA advanced indexing is not available in every PyTorch release.
    a_rows = torch.stack([a[int(i)].float() for i in idx.cpu()]).cpu().double()
    b_cols = (
        torch.stack([b[:, int(i)].float() for i in idx.cpu()], dim=1)
        .cpu()
        .double()
    )
    reference = a_rows @ b_cols
    observed = out[idx[:, None], idx[None, :]].cpu().double()
    if (
        not torch.isfinite(reference).all()
        or not torch.isfinite(observed).all()
    ):
        raise RuntimeError(
            f"{kind} correctness sample contains NaN or infinity"
        )
    scale = max(reference.abs().max().item(), 1.0)
    error = (observed - reference).abs().max().item()
    tolerances = {
        "fp64": 1e-10,
        "fp32": 2e-5,
        "tf32": 2e-3,
        "fp16": 1e-3,
        "bf16": 8e-3,
        "fp8_e4m3": 2e-3,
        "fp8_e5m2_e4m3": 2e-3,
        "int8": 0.0,
    }
    if error > tolerances[kind] * scale:
        raise RuntimeError(
            f"{kind} correctness sample failed: {error=}, {scale=}"
        )
    return {
        "sample": "8x8 distributed entries vs CPU float64 of quantized inputs",
        "max_abs_error": error,
        "reference_max_abs": scale,
        "relative_to_max_tolerance": tolerances[kind],
        "passed": True,
    }


def kernel_names(run):
    # CUPTI activity tracing requires no hardware performance counters.
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        run()
        torch.cuda.synchronize()
    return sorted(
        {
            e.name
            for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[4096, 8192, 12288]
    )
    parser.add_argument(
        "--kinds",
        nargs="+",
        default=[
            "fp64",
            "fp32",
            "tf32",
            "fp16",
            "bf16",
            "fp8_e4m3",
            "fp8_e5m2_e4m3",
            "int8",
        ],
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    if min(args.sizes) <= 0 or args.rounds <= 0 or args.warmup < 0:
        parser.error("sizes/rounds must be positive and warmup nonnegative")
    torch.cuda.set_device(args.device)
    torch.manual_seed(20260920)
    props = torch.cuda.get_device_properties(args.device)
    try:
        flagtree = importlib.metadata.version("flagtree")
    except importlib.metadata.PackageNotFoundError:
        flagtree = None
    report = {
        "start_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": str(props),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "flagtree": flagtree,
        "seed": 20260920,
        "arguments": vars(args) | {"output": str(args.output)},
        "nvidia_smi_before": subprocess.check_output(
            ["nvidia-smi"], text=True
        ),
        "method": (
            "CUDA events; per-batch mean, then median of batches; "
            "2*M*N*K operations; dense"
        ),
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for kind in args.kinds:
        for n in args.sizes:
            a, b, run = make_case(kind, n, f"cuda:{args.device}")
            out = run()
            torch.cuda.synchronize()
            correctness = validate(a, b, out, kind)
            for _ in range(args.warmup):
                run()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run()
            end.record()
            end.synchronize()
            probe_ms = start.elapsed_time(end)
            repeats = max(3, min(100, int(150 / max(probe_ms, 0.001))))
            samples = []
            for _ in range(args.rounds):
                start.record()
                for _ in range(repeats):
                    run()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) / repeats)
            median_ms = statistics.median(samples)
            operations = 2 * n**3
            result = {
                "kind": kind,
                "n": n,
                "blas_preference": str(
                    torch.backends.cuda.preferred_blas_library()
                ),
                "allow_tf32": kind == "tf32",
                "allow_reduced_precision_reduction": False,
                "fp8_use_fast_accum": False,
                "layout": "A=row-major B=column-major C=row-major",
                "input_dtype_a": str(a.dtype),
                "input_dtype_b": str(b.dtype),
                "output_dtype": str(out.dtype),
                "operations": operations,
                "repeats_per_round": repeats,
                "round_ms": samples,
                "median_ms": median_ms,
                "min_ms": min(samples),
                "max_ms": max(samples),
                "throughput": operations / median_ms / 1e9,
                "unit": "TOPS" if kind == "int8" else "TFLOP/s",
                "correctness": correctness,
            }
            try:
                result["observed_gpu_kernels"] = kernel_names(run)
            except Exception as exc:
                result["trace_error"] = str(exc)
            report["results"].append(result)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"{kind:16} n={n:5} {median_ms:9.4f} ms "
                f"{result['throughput']:9.3f} {result['unit']} check=pass",
                flush=True,
            )
            del a, b, out, run
            torch.cuda.empty_cache()
    report["end_utc"] = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()
    report["nvidia_smi_after"] = subprocess.check_output(
        ["nvidia-smi"], text=True
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
