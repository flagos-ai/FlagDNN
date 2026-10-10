#!/usr/bin/env python3
"""Build/run native cuDNN reference catalog; isolate failures per case.

python tools/performance_baseline/benchmark_nvidia_extended.py --build-only
python tools/performance_baseline/benchmark_nvidia_extended.py \\
    --shared-context \\
    --output /tmp/flagdnn-baseline-expansion/extended.jsonl

All subprocesses are sequential. The C++ --catalog mode does not initialize
CUDA. Actual runs require an idle exclusive GPU; arrange this externally.
"""
import argparse
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def run_shared(args, selected, completed, output, logs):
    """Native shared context, flushed per-case results, per-case timeout.

    Any failed case makes C++ return 3 after emitting its result; this also
    recovers CUDA errors hidden inside cuDNN/frontend status messages. The
    remaining cases resume in a fresh process with their original settings.
    """
    metadata = {row["case_name"]: row for row in selected}

    def save(row, wall_seconds):
        row["wall_seconds"] = wall_seconds
        row["process_mode"] = "shared_cuda_context"
        output.write(json.dumps(row, separators=(",", ":")) + "\n")
        output.flush()
        completed.add(row["case_name"])
        print(
            f"{len(completed)} completed {row['status']} {row['case_name']}",
            file=sys.stderr,
            flush=True,
        )

    for row in selected:
        if row["case_name"] not in completed and row["status"] == "blocked":
            save(dict(row), 0)
    attempt = 0
    while remaining := [
        row["case_name"]
        for row in selected
        if row["case_name"] not in completed
    ]:
        attempt += 1
        case_file = logs / f"shared-{attempt:03d}.cases"
        case_file.write_text("\n".join(remaining) + "\n")
        with (logs / f"shared-{attempt:03d}.stderr").open("wb") as stderr:
            process = subprocess.Popen(
                [
                    str(args.binary),
                    *(["--extra-variants"] if args.extra_variants else []),
                    "--case-file",
                    str(case_file),
                ],
                stdout=subprocess.PIPE,
                stderr=stderr,
            )
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            pending = b""
            start = time.monotonic()
            timed_out = False
            try:
                while True:
                    elapsed = time.monotonic() - start
                    if elapsed >= args.timeout:
                        timed_out = True
                        process.kill()
                        break
                    if not selector.select(args.timeout - elapsed):
                        continue
                    chunk = os.read(process.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    pending += chunk
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        if not line.startswith(b"{"):
                            continue
                        row = json.loads(line)
                        name = row["case_name"]
                        if name not in metadata or name in completed:
                            process.kill()
                            raise RuntimeError(
                                f"unexpected or duplicate case result: {name}"
                            )
                        now = time.monotonic()
                        save(row, now - start)
                        start = now
                try:
                    remaining_timeout = max(
                        0, args.timeout - (time.monotonic() - start)
                    )
                    returncode = process.wait(timeout=remaining_timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    process.kill()
                    returncode = process.wait()
            finally:
                selector.close()
                if process.poll() is None:
                    process.kill()
                process.wait()
                process.stdout.close()
        unrecorded = [name for name in remaining if name not in completed]
        no_progress = len(unrecorded) == len(remaining)
        if unrecorded and (timed_out or returncode != 3 or no_progress):
            row = dict(metadata[unrecorded[0]])
            row.update(
                status="failed",
                correctness="NOT_RUN",
                stage="timeout" if timed_out else "process",
            )
            row["reason"] = (
                f"native reference exceeded {args.timeout}s "
                "per-case process timeout"
                if timed_out
                else (
                    f"shared native process exit={returncode}; "
                    "missing result; "
                    f"see shared-{attempt:03d}.stderr"
                )
            )
            save(row, time.monotonic() - start)


def build(binary):
    cuda = Path(os.environ.get("CUDA_ROOT", "/usr/local/cuda"))
    if not os.environ.get("CUDNN_ROOT"):
        raise SystemExit(
            "CUDNN_ROOT must be set to the cuDNN installation directory"
        )
    cudnn = Path(os.environ["CUDNN_ROOT"]).expanduser().resolve()
    binary.parent.mkdir(parents=True, exist_ok=True)
    frontend_setting = os.environ.get("CUDNN_FRONTEND_INCLUDE_DIR")
    if frontend_setting:
        frontend = Path(frontend_setting).expanduser().resolve()
    else:
        frontend_root = cudnn.parents[1] if len(cudnn.parents) > 1 else cudnn
        frontend = frontend_root / "include"
    adapter = ROOT / "build/nvidia/backends/nvidia/validation"
    includes = [
        ROOT / p
        for p in (
            "include",
            "tests",
            "benchmark",
            "backends/nvidia",
            "build/nvidia/generated/include",
        )
    ]
    includes += [cuda / "include", cudnn / "include", frontend]
    libraries = [ROOT / "build/nvidia/src", cuda / "lib64", cudnn / "lib"]
    cmd = [
        os.environ.get("CXX", "c++"),
        "-O1",
        "-std=c++20",
        "-ffunction-sections",
        "-fdata-sections",
    ]
    cmd += ["-I" + str(p) for p in includes]
    cmd += [
        str(ROOT / "tools/performance_baseline/benchmark_nvidia_extended.cpp")
    ]
    cmd += [
        str(adapter / ("libflagdnn_validation_nvidia_" + name + ".a"))
        for name in ("extended", "attention", "tensor_io")
    ]
    cmd += ["-L" + str(p) for p in libraries] + [
        "-Wl,-rpath," + str(p) for p in libraries
    ]
    cmd += [
        "-Wl,--gc-sections",
        "-lflagdnn",
        "-l:libcudnn.so.9",
        "-l:libcudnn_ext.so.9",
        "-lcuda",
        "-lcudart",
        "-lnvrtc",
        "-ldl",
        "-o",
        str(binary),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", type=Path, default=Path("/tmp/benchmark_nvidia_extended")
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Defaults to extended.jsonl, or extra_variants.jsonl for "
            "capability probes, under /tmp/flagdnn-baseline-expansion"
        ),
    )
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--op", action="append")
    parser.add_argument("--case")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--extra-variants",
        action="store_true",
        help=(
            "Probe only existing functional FP32/dBias attention variants "
            "excluded from the native performance catalog"
        ),
    )
    parser.add_argument(
        "--shared-context",
        action="store_true",
        help=(
            "Reuse the native CUDA context, retaining per-case warmup "
            "and graph samples"
        ),
    )
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a positive finite number")
    args.binary = args.binary.expanduser().resolve()
    if args.output is None:
        args.output = Path("/tmp/flagdnn-baseline-expansion") / (
            "extra_variants.jsonl" if args.extra_variants else "extended.jsonl"
        )
    if not args.skip_build:
        build(args.binary)
    catalog_run = subprocess.run(
        [
            str(args.binary),
            "--catalog",
            *(["--extra-variants"] if args.extra_variants else []),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    catalog = [json.loads(line) for line in catalog_run.stdout.splitlines()]
    if len({row["case_name"] for row in catalog}) != len(catalog):
        raise RuntimeError("duplicate catalog case names")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".catalog.jsonl").write_text(catalog_run.stdout)
    print(f"catalog cases={len(catalog)}", file=sys.stderr, flush=True)
    if args.build_only:
        return
    completed = set()
    if args.resume and args.output.exists():
        completed = {
            json.loads(line)["case_name"]
            for line in args.output.read_text().splitlines()
        }
    selected = [
        row
        for row in catalog
        if (not args.op or row["op"] in args.op)
        and (not args.case or row["case_name"] == args.case)
    ]
    logs = args.output.parent / (args.output.stem + "-logs")
    logs.mkdir(exist_ok=True)
    with args.output.open("a" if args.resume else "w") as output:
        if args.shared_context:
            run_shared(args, selected, completed, output, logs)
            return
        for index, meta in enumerate(selected):
            if meta["case_name"] in completed:
                continue
            start = time.monotonic()
            result = dict(meta)
            if meta["status"] != "blocked":
                try:
                    proc = subprocess.run(
                        [
                            str(args.binary),
                            *(
                                ["--extra-variants"]
                                if args.extra_variants
                                else []
                            ),
                            "--case",
                            meta["case_name"],
                        ],
                        text=True,
                        capture_output=True,
                        timeout=args.timeout,
                    )
                    (logs / f"{index:05d}.stderr").write_text(proc.stderr)
                    rows = [
                        json.loads(line)
                        for line in proc.stdout.splitlines()
                        if line.startswith("{")
                    ]
                    if (
                        len(rows) == 1
                        and rows[0]["case_name"] == meta["case_name"]
                        and (
                            proc.returncode == 0
                            or rows[0].get("status") == "failed"
                        )
                    ):
                        result = rows[0]
                        result["process_returncode"] = proc.returncode
                    else:
                        result.update(
                            status="failed",
                            stage="process",
                            reason=(
                                f"exit={proc.returncode}; "
                                "no single matching JSON result; "
                                f"stderr={proc.stderr[-1500:]}"
                            ),
                        )
                except subprocess.TimeoutExpired:
                    result.update(
                        status="failed",
                        stage="timeout",
                        reason=(
                            f"native reference exceeded {args.timeout}s "
                            "process timeout"
                        ),
                    )
            result["wall_seconds"] = time.monotonic() - start
            output.write(json.dumps(result, separators=(",", ":")) + "\n")
            output.flush()
            print(
                f"{index + 1}/{len(selected)} {result['status']} "
                f"{meta['case_name']}",
                file=sys.stderr,
                flush=True,
            )


if __name__ == "__main__":
    main()
