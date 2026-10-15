#!/usr/bin/env python3
"""Collect the 90 active activation-backward cases.

The retired 85-case scope is retained only for explicit historical audits.

Run with an idle, exclusively reserved H100 80GB / 700 W GPU. The runner is
sequential and never changes power settings. It does not import results.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]


def required_names(include_retired=False):
    shapes = (
        "1x1x16 2x4x8 1x4x8x16 2x4x8x16 1x3x17 3x5x7 "
        "1x3x5x7 2x3x5x7 1x1x31 1x1x32 1x1x33 2x7x16 "
        "1x1x1023 1x1x1025"
    ).split()
    names = set()
    for op in ("elu_backward", "softplus_backward"):
        for dtype in ("fp32", "fp16", "bfloat16"):
            names.update(f"{op}_{dtype}_{shape}" for shape in shapes)
            names.add(f"{op}_attributes_{dtype}_1x1x65")
    if include_retired:
        names.update(f"pow_int32_{shape}" for shape in shapes)
        names.add("pow_int32_broadcast_padded_2x3x17")
        suffixes = (
            "1_axis0 17_axis-1 33_axis0 2_3_axis-1 3_7_axis0 "
            "2_3_5_axis-1 1_3_17_axis0 2_4_33_axis-2 2_3_4_5_axis0 "
            "1_7_3_9_axis-3 2_2_2_2_3_axis0 "
            "1_2_1_3_1_2_1_5_axis-5 3_257_axis0 2_1025_axis-1"
        ).split()
        for dtype in ("int32", "bool", "fp8_e4m3", "fp8_e5m2", "fp8_e8m0"):
            names.update(
                f"concatenate_{dtype}_{suffix}" for suffix in suffixes
            )
    assert len(names) == (175 if include_retired else 90)
    return names


def run_case(binary, row, output, variant, flags):
    name = row["case_name"]
    stem = output / "raw" / variant / name
    stem.parent.mkdir(parents=True, exist_ok=True)
    command = [str(binary), "--case", name, *flags]
    start = time.monotonic()
    with stem.with_suffix(".jsonl").open("wb") as stdout, stem.with_suffix(
        ".stderr"
    ).open("wb") as stderr:
        try:
            process = subprocess.run(
                command,
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
                timeout=120,
                check=False,
            )
            code = process.returncode
        except subprocess.TimeoutExpired:
            code = 124
    stem.with_suffix(".exit").write_text(f"{code}\n")
    lines = stem.with_suffix(".jsonl").read_text().splitlines()
    parsed = [json.loads(line) for line in lines if line.startswith("{")]
    if len(parsed) != 1 or parsed[0].get("case_name") != name:
        result = dict(
            row,
            status="ERROR",
            stage="process",
            reason=f"exit={code}; expected one matching JSON result",
            timing={},
        )
    else:
        result = parsed[0]
    result.update(
        process_exit=code,
        command=command,
        variant=variant,
        wall_seconds=time.monotonic() - start,
        raw_stdout=str(stem.with_suffix(".jsonl").relative_to(ROOT)),
        raw_stderr=str(stem.with_suffix(".stderr").relative_to(ROOT)),
    )
    if code and result["status"] == "PASS":
        result.update(
            status="ERROR",
            stage="process",
            reason=f"process exited {code} after PASS",
        )
    print(variant, result["status"], name, result["stage"], flush=True)
    return result


def capture_sources(binary, output):
    # Freeze evidence before collection; later workspace edits cannot
    # invalidate
    # the measured source or substitute a different collector in the archive.
    files = subprocess.check_output(
        [
            "git",
            "ls-files",
            "include",
            "tests/common",
            "reference/cpu",
            "backends/nvidia/validation",
            "benchmark/common",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    files += [
        "tools/performance_baseline/" + name
        for name in (
            "benchmark_nvidia_supplement.cpp",
            "build_benchmark_nvidia_supplement.sh",
            "run_benchmark_nvidia_supplement.py",
            "audit_nvidia_supplement.py",
        )
    ]
    hashes = {}
    with tarfile.open(output / "source-snapshot.tar.gz", "w:gz") as archive:
        for name in sorted(set(files)):
            path = ROOT / name
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.add(path, arcname=name)
    (output / "source-sha256.json").write_text(
        json.dumps(hashes, indent=2) + "\n"
    )
    frozen_binary = output / "collector.bin"
    shutil.copy2(binary, frozen_binary)
    (output / "binary-sha256.json").write_text(
        json.dumps(
            {
                "collector.bin": hashlib.sha256(
                    frozen_binary.read_bytes()
                ).hexdigest()
            },
            indent=2,
        )
        + "\n"
    )
    for filename, command in (
        ("source-head.txt", ["git", "rev-parse", "HEAD"]),
        ("status-before.txt", ["git", "status", "--short"]),
        ("working-tree.patch", ["git", "diff", "--binary"]),
    ):
        (output / filename).write_bytes(
            subprocess.check_output(command, cwd=ROOT)
        )
    return frozen_binary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binary, output = args.binary.resolve(), args.output.resolve()
    try:
        output.relative_to(ROOT)
    except ValueError:
        parser.error("--output must be inside the FlagDNN repository")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output must be a new or empty directory")
    if not binary.is_file():
        parser.error("--binary must name an existing collector")
    output.mkdir(parents=True, exist_ok=True)
    # Execute the frozen copy so a concurrent rebuild cannot change the
    # collector halfway through a run or disagree with the archived hash.
    binary = capture_sources(binary, output)
    catalog_text = subprocess.check_output(
        [str(binary), "--catalog"], text=True, cwd=ROOT
    )
    catalog = [json.loads(line) for line in catalog_text.splitlines()]
    expected = required_names()
    if (
        len(catalog) != len(expected)
        or {r["case_name"] for r in catalog} != expected
    ):
        parser.error(
            (
                "collector catalog differs from the active 90-case scope; "
                "rebuild the collector"
            )
        )
    (output / "catalog.jsonl").write_text(catalog_text)
    (output / "manifest.txt").write_text(
        "\n".join(sorted(required_names())) + "\n"
    )
    test = subprocess.run(
        [str(binary), "--self-test"], capture_output=True, text=True, cwd=ROOT
    )
    (output / "self-test.jsonl").write_text(test.stdout)
    (output / "self-test.stderr").write_text(test.stderr)
    test.check_returncode()
    (output / "nvidia-smi-before.txt").write_bytes(
        subprocess.check_output(["nvidia-smi", "-q"])
    )
    rows = []
    with (output / "results.jsonl").open("w") as combined:
        for row in catalog:
            result = run_case(binary, row, output, "default", [])
            rows.append(result)
            combined.write(json.dumps(result, separators=(",", ":")) + "\n")
            combined.flush()
    (output / "nvidia-smi-after.txt").write_bytes(
        subprocess.check_output(["nvidia-smi", "-q"])
    )
    summary = {}
    for op in sorted({r["op"] for r in rows}):
        names = {r["case_name"] for r in rows if r["op"] == op}
        good = {
            r["case_name"]
            for r in rows
            if r["op"] == op and r["status"] == "PASS"
        }
        summary[op] = {
            "requested": len(names),
            "passed": len(good),
            "unavailable": len(names - good),
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
