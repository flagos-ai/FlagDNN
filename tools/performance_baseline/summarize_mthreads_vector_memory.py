#!/usr/bin/env python3
"""
Validate a default MUSA peak sweep and summarize it without changing baselines.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(
    raw_csv: Path,
    raw_log: Path,
    collector: Path,
    device_ir: Path,
    measured_on: str,
) -> dict:
    peaks: dict[str, float]
    rows = list(csv.DictReader(raw_csv.open()))
    kinds = collections.Counter(row["record"] for row in rows)
    if kinds != {"sample": 5400, "round": 270, "summary": 54}:
        raise ValueError(f"Incomplete default sweep: {kinds}")
    if any(row["validation"] != "pass" for row in rows):
        raise ValueError(
            "Every sample/configuration must pass collector validation"
        )
    metadata = re.search(
        r"device=(\d+) name=(.*?) arch=(mp_\d+) compute_units=(\d+) "
        r"wavefront_size=(\d+) L2_bytes=(\d+) total_memory_bytes=(\d+) "
        r"runtime=(\d+) driver=(\d+) warmup=(\d+) rounds=(\d+) repeats=(\d+)",
        raw_log.read_text(),
    )
    if (
        metadata is None
        or (
            "Completed: all compute outputs and sampled memory outputs passed "
            "validation."
        )
        not in raw_log.read_text()
    ):
        raise ValueError("Missing device metadata or collector completion")
    (
        device,
        name,
        arch,
        cus,
        warp,
        l2,
        memory,
        runtime,
        driver,
        warmup,
        rounds,
        repeats,
    ) = metadata.groups()
    if (int(warmup), int(rounds), int(repeats)) != (20, 5, 20):
        raise ValueError(
            "Peak contract requires 20 warmups and 5 rounds of 20 samples"
        )
    body = (
        device_ir.read_text()
        .split("define protected mtgpu_kernel void @_Z11compute_fma", 1)[1]
        .split("\n}", 1)[0]
    )
    if (
        body.count("call contract float @llvm.musa.fma.rte.f") != 256
        or "mma" in body
    ):
        raise ValueError(
            (
                "Compute device IR must retain 256 scalar FMA calls and no "
                "matrix intrinsic"
            )
        )
    grouped = collections.defaultdict(list)
    summaries = {}
    for row in rows:
        key = row["resource"], row["configuration"]
        if row["record"] == "sample":
            grouped[key].append(row)
        elif row["record"] == "summary":
            if key in summaries:
                raise ValueError("Duplicate configuration")
            summaries[key] = row
    configs, peaks = [], {}
    for (metric, configuration), row in summaries.items():
        samples = grouped[metric, configuration]
        if {(int(s["round"]), int(s["repeat"])) for s in samples} != {
            (r, j) for r in range(5) for j in range(20)
        }:
            raise ValueError(f"Missing/repeated samples: {configuration}")
        times = [float(s["elapsed_ms"]) for s in samples]
        if any(not math.isfinite(t) or t <= 0 for t in times):
            raise ValueError("Invalid event time")
        median = statistics.median(times)
        work = int(row["work_per_launch"])
        spec = dict(part.split("=", 1) for part in configuration.split(";"))
        threads, blocks = int(spec["threads"]), int(spec["blocks_per_cu"])
        if threads not in (128, 256) or blocks not in (2, 4, 8):
            raise ValueError("Unexpected default sweep configuration")
        if metric == "vector_fp32":
            if (spec["iters"], spec["chains"], spec["unroll"]) != (
                "256",
                "8",
                "32",
            ):
                raise ValueError("Unexpected FMA work configuration")
            expected_work = int(cus) * blocks * threads * 256 * 8 * 32 * 2
        else:
            size_mib = int(spec["buffer_mib"])
            if size_mib not in (256, 1024) or (size_mib << 20) <= int(l2):
                raise ValueError("Memory working set must exceed device L2")
            streams = {
                "hbm_copy": 2,
                "hbm_read": 1,
                "hbm_write": 1,
                "hbm_triad": 3,
            }[metric]
            expected_work = (size_mib << 20) * streams
        if work != expected_work or any(
            int(s["work_per_launch"]) != work for s in samples
        ):
            raise ValueError("FMA or logical-byte work mismatch")
        if not math.isclose(median, float(row["median_ms"]), rel_tol=1e-8):
            raise ValueError("CSV summary median mismatch")
        throughput = float(row["throughput"])
        divisor = 1e12 if metric == "vector_fp32" else 1e9
        if not math.isclose(
            throughput, work / (median * 1e-3) / divisor, rel_tol=1e-8
        ):
            raise ValueError("CSV throughput arithmetic mismatch")
        if metric != "vector_fp32":
            throughput = float(f"{throughput / 1000:.10g}")
        peaks[metric] = max(peaks.get(metric, 0), throughput)
        configs.append(
            dict(
                metric=metric,
                configuration=configuration,
                work_per_launch=work,
                sample_count=100,
                median_ms=float(row["median_ms"]),
                min_ms=min(times),
                max_ms=max(times),
                throughput=throughput,
                unit="TFLOP/s" if metric == "vector_fp32" else "TB/s",
                validation="pass",
            )
        )
    return dict(
        schema_version=1,
        platform="mthreads",
        device=name,
        architecture=arch,
        measured_on=measured_on,
        device_ordinal=int(device),
        compute_units=int(cus),
        wavefront_size=int(warp),
        l2_bytes=int(l2),
        total_memory_bytes=int(memory),
        musa_runtime_version=int(runtime),
        driver_version=int(driver),
        collector=str(collector),
        collector_sha256=sha256(collector),
        raw_csv=str(raw_csv),
        raw_csv_sha256=sha256(raw_csv),
        raw_log=str(raw_log),
        raw_log_sha256=sha256(raw_log),
        device_ir=str(device_ir),
        device_ir_sha256=sha256(device_ir),
        device_ir_validation=(
            "256 scalar musa.fma.rte.f calls in the runtime compute loop; no "
            "matrix intrinsic."
        ),
        selection=(
            "Maximum across configurations of throughput computed from all "
            "100 event samples' median; scalar FMA counts 2 FLOPs, memory "
            "counts logical read/write bytes."
        ),
        warmup_iterations=20,
        rounds=5,
        samples_per_round=20,
        configuration_count=54,
        sample_count=5400,
        validation=(
            "All FP32 output elements checked exactly after every round. "
            "Memory head/middle/tail windows and 64 read checksums sampled "
            "after every round; all passed."
        ),
        peaks=peaks,
        configurations=configs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--raw-log", type=Path, required=True)
    parser.add_argument("--device-ir", type=Path, required=True)
    parser.add_argument(
        "--collector",
        type=Path,
        default=Path(
            "tools/performance_baseline/benchmark_mthreads_vector_memory.mu"
        ),
    )
    parser.add_argument("--measured-on", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.raw_csv,
        args.raw_log,
        args.collector,
        args.device_ir,
        args.measured_on,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            {"configurations": 54, "samples": 5400, "peaks": result["peaks"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
