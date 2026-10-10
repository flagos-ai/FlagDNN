#!/usr/bin/env python3
"""Measure shared native cuDNN gap cases, with fresh-process error recovery."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument(
        "--shared-groups",
        action="store_true",
        help=(
            "Reuse a context until an execution error; recover remaining "
            "cases in fresh processes."
        ),
    )
    args = parser.parse_args()
    catalog = [
        json.loads(line)
        for line in subprocess.check_output(
            [str(args.binary), "--catalog"], text=True
        ).splitlines()
    ]
    if args.group:
        catalog = [case for case in catalog if case["group"] in args.group]
    order = {"attributes": 0, "identity_fp8": 1, "integer": 2}
    catalog.sort(key=lambda case: order[case["group"]])
    if not catalog:
        parser.error("case filter matched no cases")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logs = args.output.with_suffix(".logs")
    logs.mkdir(exist_ok=True)
    counts = {"PASS": 0, "ERROR": 0}

    def attempt(command, cases, label):
        started = time.monotonic()
        stdout, stderr, returncode = "", "", None
        failure = ""
        try:
            process = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=args.timeout,
                check=False,
            )
            stdout, stderr, returncode = (
                process.stdout,
                process.stderr,
                process.returncode,
            )
        except subprocess.TimeoutExpired as error:
            failure = str(error)
            stdout, stderr = error.stdout or "", error.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
        (logs / (label + ".log")).write_text(
            stdout + "\n--- stderr ---\n" + stderr
        )
        expected = {case["case_name"] for case in cases}
        records = {}
        for line in stdout.splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                not isinstance(record, dict)
                or record.get("case_name") not in expected
            ):
                continue
            name = record["case_name"]
            if name in records:
                raise RuntimeError(f"duplicate result: {name}")
            records[name] = record
        if len(cases) == 1:
            case = cases[0]
            record = records.get(case["case_name"])
            if record is None or (
                returncode != 0 and record.get("status") == "PASS"
            ):
                records[case["case_name"]] = dict(
                    case,
                    status="ERROR",
                    stage="isolated_process",
                    error=failure
                    or (
                        f"exit={returncode}; "
                        "no complete successful process result"
                    ),
                    correctness={"status": "NOT_COMPLETED"},
                )
        for record in records.values():
            record["process_group_wall_s"] = time.monotonic() - started
            record["process_group_size"] = len(cases)
        return records

    with args.output.open("x") as output:
        pending = {}
        ordinal = 0
        for group in order:
            cases = [case for case in catalog if case["group"] == group]
            if not cases:
                continue
            if args.shared_groups:
                pending = attempt(
                    [str(args.binary), "--group", group], cases, group
                )
            for case in cases:
                name = case["case_name"]
                record = pending.get(name)
                if record is None:
                    record = attempt(
                        [str(args.binary), "--case", name], [case], name
                    )[name]
                counts[record["status"]] += 1
                output.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False)
                    + "\n"
                )
                output.flush()
                ordinal += 1
                print(
                    f"{ordinal}/{len(catalog)} {record['status']} {name}",
                    flush=True,
                )
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
