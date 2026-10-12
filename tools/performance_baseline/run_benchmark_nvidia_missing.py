#!/usr/bin/env python3
"""Collect the exact Appendix 1 cases from the NVIDIA missing-case report.

Both binaries must already be built. Collection is sequential and restarts
CUDA after a failed case. This command never edits the public baseline.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from benchmark_nvidia_extended import run_shared

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = ROOT / "docs/nvidia-performance-missing-cases-20261012.md"


def requested_cases(report):
    section = report.read_text().split("## 附录一", 1)[1]
    section = section.split("## 附录二", 1)[0]
    result = []
    operator = None
    in_fence = False
    for line in section.splitlines():
        if line.startswith("### "):
            operator = line.split()[1]
        elif line.startswith("```"):
            in_fence = not in_fence
        elif in_fence and line.strip():
            if operator is None:
                raise ValueError("case precedes its operator heading")
            result.append((operator, line.strip()))
    if not result or len(result) != len(set(result)):
        raise ValueError("empty or duplicate requested case IDs")
    if len({name for _, name in result}) != len(result):
        raise ValueError("case names must be unique across collectors")
    return result


def select_catalogs(requested, catalogs):
    wanted = set(requested)
    selected = {}
    found = set()
    for group, catalog in catalogs.items():
        selected[group] = []
        for row in catalog:
            key = (row["op"], row["case_name"])
            if key not in wanted:
                continue
            if key in found:
                raise ValueError(f"duplicate collector case: {key}")
            found.add(key)
            selected[group].append(dict(row, status="ready"))
    if missing := wanted - found:
        raise ValueError(f"requested cases absent from catalogs: {missing}")
    return selected


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--core-binary", type=Path, required=True)
    parser.add_argument("--aux-binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if not 0 < args.timeout < float("inf"):
        parser.error("--timeout must be positive and finite")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    binaries = {
        "core": args.core_binary.resolve(),
        "aux": args.aux_binary.resolve(),
    }
    requested = requested_cases(args.report)
    catalogs = {}
    for group, binary in binaries.items():
        result = subprocess.run(
            [str(binary), "--catalog"],
            check=True,
            text=True,
            capture_output=True,
        )
        catalogs[group] = [
            json.loads(line) for line in result.stdout.splitlines()
        ]
    selected = select_catalogs(requested, catalogs)
    manifest = {
        "report": str(args.report),
        "report_sha256": sha256(args.report),
        "requested_count": len(requested),
        "collectors": {},
    }
    for group, rows in selected.items():
        binary = binaries[group]
        manifest["collectors"][group] = {
            "binary": str(binary),
            "binary_sha256": sha256(binary),
            "count": len(rows),
        }
        (args.output_dir / f"{group}.cases").write_text(
            "".join(row["case_name"] + "\n" for row in rows)
        )
        (args.output_dir / f"{group}.catalog.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    (args.output_dir / "collection-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest), flush=True)
    if args.list_only:
        return
    for group, rows in selected.items():
        logs = args.output_dir / f"{group}.logs"
        logs.mkdir()
        options = SimpleNamespace(
            binary=binaries[group],
            timeout=args.timeout,
            extra_variants=False,
            process_mode=(
                "shared_cuda_context"
                if group == "core"
                else "shared_process_per_case_primary_context"
            ),
        )
        with (args.output_dir / f"{group}.jsonl").open("x") as output:
            run_shared(options, rows, set(), output, logs)

    results = [
        json.loads(line)
        for group in selected
        for line in (args.output_dir / f"{group}.jsonl")
        .read_text()
        .splitlines()
    ]
    passed = sum(row["status"] == "PASS" for row in results)
    print(json.dumps({"passed": passed, "failed": len(results) - passed}))
    if passed != len(requested):
        sys.exit(1)


if __name__ == "__main__":
    main()
