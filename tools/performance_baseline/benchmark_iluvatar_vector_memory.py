#!/usr/bin/env python3
"""Measure Iluvatar FP32 vector and logical HBM bandwidth peaks.

Example (run on an otherwise idle allocated device):
  python3 tools/performance_baseline/benchmark_iluvatar_vector_memory.py \
      --device 0 --output-dir build/iluvatar/peak-measurement

The collector never edits performance-baseline.json. Its summary and raw
samples are evidence for a separately reviewed baseline update. Throughput
is the maximum across configurations of work divided by median elapsed time,
matching NVIDIA/Hygon collectors. Memory traffic counts logical FP32 bytes,
using 256/1024 MiB buffers; the compute kernel contains only explicit FP32
FMA chains (2 FLOPs per element). Graph replay avoids Python dispatch gaps
in CUDA event intervals. All allocation, compilation, initialization,
transfers, validation and capture are outside timing.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys


OPERATIONS = ("vector_fp32", "hbm_copy", "hbm_triad", "hbm_write")


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument(
        "--operations", nargs="+", choices=OPERATIONS, default=list(OPERATIONS)
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sizes-mib", nargs="+", type=positive_int, default=[256, 1024]
    )
    parser.add_argument(
        "--memory-blocks",
        nargs="+",
        type=positive_int,
        default=[4096, 8192, 16384],
    )
    parser.add_argument(
        "--compute-blocks",
        nargs="+",
        type=positive_int,
        default=[256, 512, 1024],
    )
    parser.add_argument(
        "--compute-programs",
        nargs="+",
        type=positive_int,
        default=[64, 128, 256],
    )
    parser.add_argument("--iterations", type=positive_int, default=256)
    parser.add_argument("--warmup", type=positive_int, default=20)
    parser.add_argument("--rounds", type=positive_int, default=5)
    parser.add_argument("--samples", type=positive_int, default=20)
    parser.add_argument(
        "--graph-replays",
        type=positive_int,
        default=10,
        help="kernel invocations in each captured graph sample",
    )
    args = parser.parse_args(argv)
    if args.device < 0:
        parser.error("device must be nonnegative")
    if any(size < 256 for size in args.sizes_mib):
        parser.error("HBM measurements require at least 256 MiB per buffer")
    if any(
        block & (block - 1)
        for block in args.memory_blocks + args.compute_blocks
    ):
        parser.error("block sizes must be powers of two")
    return args


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_kernels(triton, tl):
    # tl must be visible as a module global to the installed Triton JIT.
    globals()["tl"] = tl

    @triton.jit
    def initialize(
        A,
        B,
        C,
        N: tl.constexpr,  # type: ignore[name-defined]
        BLOCK: tl.constexpr,  # type: ignore[name-defined]
        PROGRAMS: tl.constexpr,  # type: ignore[name-defined]
    ):
        tile = tl.program_id(0)
        # Keep both element offsets and the final loop increment in range.
        if (
            N > 0x7FFFFFFF
            or (N + BLOCK - 1) // BLOCK - 1 + PROGRAMS > 0x7FFFFFFF
        ):
            tile = tile.to(tl.int64)
        while tile < tl.cdiv(N, BLOCK):
            index = tile * BLOCK + tl.arange(0, BLOCK)
            x = 0.5 + (index % 1024).to(tl.float32) * (1.0 / 2048.0)
            # Masking preserves modulo-1024 semantics even if int32 index*17
            # wraps.
            y = 0.5 + ((index * 17 + 13) & 1023).to(tl.float32) * (
                1.0 / 2048.0
            )
            tl.store(A + index, x, index < N)
            tl.store(B + index, y, index < N)
            tl.store(C + index, -123.0, index < N)
            tile += PROGRAMS

    @triton.jit
    def memory(
        A,
        B,
        C,
        N: tl.constexpr,  # type: ignore[name-defined]
        OP: tl.constexpr,  # type: ignore[name-defined]
        BLOCK: tl.constexpr,  # type: ignore[name-defined]
        PROGRAMS: tl.constexpr,  # type: ignore[name-defined]
    ):
        tile = tl.program_id(0)
        if (
            N > 0x7FFFFFFF
            or (N + BLOCK - 1) // BLOCK - 1 + PROGRAMS > 0x7FFFFFFF
        ):
            tile = tile.to(tl.int64)
        while tile < tl.cdiv(N, BLOCK):
            index = tile * BLOCK + tl.arange(0, BLOCK)
            if OP == "hbm_write":
                value = 0.5 + (index % 1024).to(tl.float32) * (1.0 / 2048.0)
            else:
                value = tl.load(A + index, index < N, other=0.0)
                if OP == "hbm_triad":
                    other = tl.load(B + index, index < N, other=0.0)
                    value = tl.fma(other, 0.75, value)
            tl.store(C + index, value, index < N)
            tile += PROGRAMS

    @triton.jit
    def compute(
        COEFF,
        OUT,
        ITERATIONS: tl.constexpr,  # type: ignore[name-defined]
        BLOCK: tl.constexpr,  # type: ignore[name-defined]
    ):
        lane = tl.arange(0, BLOCK)
        # Each logical lane writes eight FP32 elements.
        index = tl.program_id(0).to(tl.int64) * BLOCK + lane
        # Coefficients are device inputs, not compiler constants. Each chain
        # alternates exactly between x and -x+0.125, returning to its seed.
        a = tl.load(COEFF)
        b = tl.load(COEFF + 1)
        seed = 1.0 + (index % 8).to(tl.float32) * 0.0625
        x0, x1, x2, x3 = seed, seed + 0.0625, seed + 0.125, seed + 0.1875
        x4, x5, x6, x7 = (
            seed + 0.25,
            seed + 0.3125,
            seed + 0.375,
            seed + 0.4375,
        )
        iteration = 0
        while iteration < ITERATIONS:
            for _ in tl.static_range(8):
                x0 = tl.fma(x0, a, b)
                x1 = tl.fma(x1, a, b)
                x2 = tl.fma(x2, a, b)
                x3 = tl.fma(x3, a, b)
                x4 = tl.fma(x4, a, b)
                x5 = tl.fma(x5, a, b)
                x6 = tl.fma(x6, a, b)
                x7 = tl.fma(x7, a, b)
            iteration += 1
        tl.store(OUT + index * 8 + 0, x0)
        tl.store(OUT + index * 8 + 1, x1)
        tl.store(OUT + index * 8 + 2, x2)
        tl.store(OUT + index * 8 + 3, x3)
        tl.store(OUT + index * 8 + 4, x4)
        tl.store(OUT + index * 8 + 5, x5)
        tl.store(OUT + index * 8 + 6, x6)
        tl.store(OUT + index * 8 + 7, x7)

    return initialize, memory, compute


def main(argv=None):
    args = parse_args(argv)
    collector_source = Path(__file__).read_bytes()
    import torch
    import triton
    import triton.language as tl

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    properties = torch.cuda.get_device_properties(device)
    runtime_attributes = {
        "compute_units": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "l2_bytes": properties.L2_cache_size,
        "architecture": f"corex_{properties.major}{properties.minor}",
    }
    if any(
        (size << 20) <= runtime_attributes["l2_bytes"]
        for size in args.sizes_mib
    ):
        raise ValueError(
            "every memory buffer must exceed the runtime-reported L2 capacity"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "samples.csv"
    summary_path = args.output_dir / "summary.json"
    if csv_path.exists() or summary_path.exists():
        raise FileExistsError(
            "refusing to overwrite an existing measurement; "
            "select a new output directory"
        )
    artifact_dir = args.output_dir / "kernels"
    artifact_dir.mkdir()
    collector_snapshot = args.output_dir / "collector-source.py"
    collector_snapshot.write_bytes(collector_source)
    initialize, memory, compute = make_kernels(triton, tl)
    configs = []
    packages = {}
    for name in ("torch", "triton"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    packages["triton"] = triton.__version__

    def save_kernel(kernel, key, require_fma=False):
        artifacts = {}
        for name, data in kernel.asm.items():
            if not isinstance(data, (str, bytes)):
                continue
            path = artifact_dir / f"{key}.{name}"
            path.write_bytes(data.encode() if isinstance(data, str) else data)
            artifacts[name] = {"file": str(path), "sha256": digest(path)}
        if require_fma:
            ttir = kernel.asm.get("ttir", "")
            if "math.fma" not in ttir or "tt.dot" in ttir:
                raise RuntimeError(
                    "compute artifact must retain vector FMA operations "
                    "and contain no tensor dot"
                )
        return artifacts

    with csv_path.open("x", newline="") as raw:
        writer = csv.DictWriter(
            raw,
            fieldnames=[
                "operation",
                "configuration",
                "work_per_launch",
                "round",
                "sample",
                "elapsed_ms",
                "throughput",
                "unit",
                "validation",
            ],
        )
        writer.writeheader()

        def measure(op, config, work, launch, validate, reset):
            key = f"{op}-{len(configs):03d}"
            print(f"Measuring {key}: {config}", file=sys.stderr, flush=True)
            kernel = launch()  # Compilation is outside all timed regions.
            torch.cuda.synchronize()
            validate()
            artifacts = save_kernel(kernel, key, op == "vector_fp32")
            for _ in range(args.warmup):
                launch()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(args.graph_replays):
                    launch()
            graph.replay()
            torch.cuda.synchronize()
            validate()
            begin = [
                torch.cuda.Event(enable_timing=True)
                for _ in range(args.samples)
            ]
            end = [
                torch.cuda.Event(enable_timing=True)
                for _ in range(args.samples)
            ]
            times = []
            unit = "TFLOP/s" if op == "vector_fp32" else "TB/s"
            for round_index in range(args.rounds):
                for start, stop in zip(begin, end):
                    start.record()
                    graph.replay()
                    stop.record()
                end[-1].synchronize()
                round_times = [
                    start.elapsed_time(stop) / args.graph_replays
                    for start, stop in zip(begin, end)
                ]
                if any(
                    not math.isfinite(value) or value <= 0
                    for value in round_times
                ):
                    raise RuntimeError(
                        "nonpositive or nonfinite CUDA event duration"
                    )
                validate()
                # A poisoned output must be completely rewritten in all checked
                # locations; this also rejects empty/nonfunctional graph
                # replays.
                reset()
                graph.replay()
                torch.cuda.synchronize()
                validate()
                for sample_index, elapsed in enumerate(round_times):
                    writer.writerow(
                        dict(
                            operation=op,
                            configuration=json.dumps(config, sort_keys=True),
                            work_per_launch=work,
                            round=round_index,
                            sample=sample_index,
                            elapsed_ms=elapsed,
                            throughput=work / (elapsed * 1e9),
                            unit=unit,
                            validation="pass",
                        )
                    )
                raw.flush()
                times.extend(round_times)
            median_ms = statistics.median(times)
            configs.append(
                dict(
                    operation=op,
                    configuration=config,
                    work_per_launch=work,
                    unit=unit,
                    median_ms=median_ms,
                    min_ms=min(times),
                    max_ms=max(times),
                    throughput=work / (median_ms * 1e9),
                    sample_count=len(times),
                    validation="pass",
                    artifacts=artifacts,
                )
            )
            graph.reset()

        if "vector_fp32" in args.operations:
            coefficients = torch.tensor(
                [-1.0, 0.125], dtype=torch.float32, device=device
            )
            for block in args.compute_blocks:
                for programs in args.compute_programs:
                    count = block * programs
                    out = torch.empty(
                        count * 8, dtype=torch.float32, device=device
                    )
                    expected = (
                        1.0
                        + torch.arange(count, dtype=torch.int64)
                        .remainder(8)
                        .float()[:, None]
                        * 0.0625
                        + torch.arange(8, dtype=torch.float32)[None, :]
                        * 0.0625
                    ).flatten()

                    # measure invokes callbacks before buffers are deleted.
                    def validate_compute():
                        torch.testing.assert_close(
                            out.cpu(), expected, rtol=0, atol=0  # noqa: F821
                        )

                    measure(
                        "vector_fp32",
                        dict(
                            block=block,
                            programs=programs,
                            chains=8,
                            unroll=8,
                            iterations=args.iterations,
                        ),
                        count * 8 * 8 * args.iterations * 2,
                        lambda: compute[(programs,)](
                            coefficients,
                            out,  # noqa: F821
                            args.iterations,
                            block,
                            num_warps=4,
                            num_stages=1,
                        ),
                        validate_compute,
                        lambda: out.fill_(-123.0),  # noqa: F821
                    )
            del out, coefficients

        memory_ops = [op for op in args.operations if op.startswith("hbm_")]
        for size in args.sizes_mib if memory_ops else []:
            count = (size << 20) // 4
            left = torch.empty(count, dtype=torch.float32, device=device)
            right = torch.empty_like(left)
            out = torch.empty_like(left)
            init_programs = min(32768, triton.cdiv(count, 4096))
            initialize[(init_programs,)](
                left,
                right,
                out,
                count,
                4096,
                init_programs,
                num_warps=4,
                num_stages=1,
            )
            torch.cuda.synchronize()
            # Deterministic windows at the beginning, end and throughout every
            # buffer. Transfers are bounded, outside timing, with no RNG state.
            starts = sorted(
                {
                    0,
                    count - 1024,
                    *(i * (count - 1024) // 31 for i in range(32)),
                }
            )
            indexes = torch.cat(
                [
                    torch.arange(start, start + 1024, dtype=torch.int64)
                    for start in starts
                ]
            )
            sample_device = indexes.to(device)
            expected_left = 0.5 + indexes.remainder(1024).float() / 2048.0
            expected_right = (
                0.5 + (indexes * 17 + 13).remainder(1024).float() / 2048.0
            )
            for op in memory_ops:
                expected = (
                    expected_left + 0.75 * expected_right
                    if op == "hbm_triad"
                    else expected_left
                )

                # Buffers remain live until all measure callbacks return.
                def validate_memory():
                    torch.testing.assert_close(
                        out[sample_device].cpu(),  # noqa: F821
                        expected,
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        left[sample_device].cpu(),  # noqa: F821
                        expected_left,
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        right[sample_device].cpu(),  # noqa: F821
                        expected_right,
                        rtol=0,
                        atol=0,
                    )

                for block in args.memory_blocks:
                    programs = min(32768, triton.cdiv(count, block))
                    measure(
                        op,
                        dict(
                            block=block,
                            programs=programs,
                            size_mib=size,
                            elements=count,
                        ),
                        (size << 20)
                        * {"hbm_write": 1, "hbm_copy": 2, "hbm_triad": 3}[op],
                        lambda: memory[(programs,)](
                            left,  # noqa: F821
                            right,  # noqa: F821
                            out,  # noqa: F821
                            count,
                            op,
                            block,
                            programs,
                            num_warps=4,
                            num_stages=1,
                        ),
                        validate_memory,
                        lambda: out.fill_(-123.0),  # noqa: F821
                    )
            del left, right, out, sample_device
            torch.cuda.empty_cache()

    best = {
        op: max(
            (entry for entry in configs if entry["operation"] == op),
            key=lambda entry: entry["throughput"],
        )
        for op in args.operations
    }
    validation = []
    if "vector_fp32" in args.operations:
        validation.append("All FP32 FMA outputs checked exactly")
    if memory_ops:
        validation.append(
            "Memory output and both inputs checked exactly in "
            "32 distributed 1024-element windows"
        )
    validation.append(
        "Validation before/after timing and after output reset "
        "plus graph replay"
    )
    result = dict(
        schema_version=1,
        platform="iluvatar",
        device=torch.cuda.get_device_name(device),
        measured_on=datetime.now(timezone.utc).date().isoformat(),
        device_ordinal=args.device,
        device_properties=str(properties),
        runtime_attributes=runtime_attributes,
        packages=packages,
        collector=str(Path(__file__).resolve()),
        collector_sha256=digest(collector_snapshot),
        collector_source_snapshot=str(collector_snapshot),
        operations=args.operations,
        raw_csv=str(csv_path),
        raw_csv_sha256=digest(csv_path),
        timing_method="cuda_events_graph_replay",
        warmup_iterations=args.warmup,
        graph_warmup_replays=1,
        launches_per_sample=args.graph_replays,
        rounds=args.rounds,
        samples_per_round=args.samples,
        statistic=(
            "maximum_across_configurations_of_work_divided_by_"
            "all_sample_median_elapsed_time"
        ),
        configuration_count=len(configs),
        sample_count=sum(c["sample_count"] for c in configs),
        validation="; ".join(validation),
        scope="Measured operations: "
        + ", ".join(args.operations)
        + ". Other peaks remain unmeasured.",
        environment={
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "TRITON_BACKEND")
        },
        peaks={op: entry["throughput"] for op, entry in best.items()},
        best_configurations={
            op: entry["configuration"] for op, entry in best.items()
        },
        configurations=configs,
    )
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"summary": str(summary_path), "peaks": result["peaks"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
