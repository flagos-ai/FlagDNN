#!/usr/bin/env python3

"""Serial native CTest runner for FlagDNN functional tests and benchmarks."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "cmake" / "Operators.cmake"
VALID_SUITES = ("functional", "benchmark")
DEFAULT_OPERATORS = (
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "max",
    "min",
    "mod",
    "add_square",
    "cmp_eq",
)


def operator_manifests() -> dict[str, list[str]]:
    source = MANIFEST.read_text(encoding="utf-8")
    set_names = {
        "benchmark": "FLAGDNN_BENCHMARK_OPERATORS",
        "functional": "FLAGDNN_FUNCTIONAL_OPERATORS",
    }
    raw_sets: dict[str, list[str]] = {}
    for set_name in set_names.values():
        match = re.search(
            rf"set\({re.escape(set_name)}(?P<body>.*?)\)",
            source,
            flags=re.DOTALL,
        )
        if match is None:
            raise RuntimeError(
                f"Cannot parse {set_name} from operator manifest: {MANIFEST}"
            )
        body = re.sub(r"#.*", "", match.group("body"))
        raw_sets[set_name] = re.findall(
            r"\$\{[A-Z0-9_]+\}|[a-z][a-z0-9_]*", body
        )

    resolved: dict[str, list[str]] = {}

    def resolve(set_name: str, stack: tuple[str, ...] = ()) -> list[str]:
        if set_name in resolved:
            return resolved[set_name]
        if set_name in stack:
            raise RuntimeError(
                "Operator manifest contains a cyclic set expansion: "
                + " -> ".join((*stack, set_name))
            )
        if set_name not in raw_sets:
            raise RuntimeError(
                f"Operator manifest references unknown set {set_name}"
            )
        operators: list[str] = []
        for token in raw_sets[set_name]:
            if token.startswith("${"):
                operators.extend(resolve(token[2:-1], (*stack, set_name)))
            else:
                operators.append(token)
        if not operators or len(operators) != len(set(operators)):
            raise RuntimeError(
                f"{set_name} is empty or contains duplicate operators"
            )
        resolved[set_name] = operators
        return operators

    return {
        suite: list(resolve(set_name)) for suite, set_name in set_names.items()
    }


def requested_suites(value: str) -> list[str]:
    raw = [item.strip() for item in value.split(",") if item.strip()]
    if not raw or raw == ["all"] or "all" in raw:
        return list(VALID_SUITES)
    unsupported = set(raw) - set(VALID_SUITES)
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"Unsupported suite(s): {names}")
    return list(dict.fromkeys(raw))


def requested_operators(
    manifests: dict[str, list[str]],
    suites: list[str],
    value: str | None,
    list_file: Path | None,
) -> dict[str, list[str]]:
    if value and list_file:
        raise ValueError("--ops and --op-list-file are mutually exclusive")
    if value == "all":
        return {suite: list(manifests[suite]) for suite in suites}
    if value:
        selected = [item.strip() for item in value.split(",") if item.strip()]
    elif list_file:
        selected = [
            line.strip()
            for line in list_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        selected = list(DEFAULT_OPERATORS)

    selected = list(dict.fromkeys(selected))
    unavailable = {
        suite: sorted(set(selected) - set(manifests[suite]))
        for suite in suites
    }
    unavailable = {
        suite: operators
        for suite, operators in unavailable.items()
        if operators
    }
    if unavailable:
        details = "; ".join(
            f"{suite}: {', '.join(operators)}"
            for suite, operators in unavailable.items()
        )
        raise ValueError(
            "Operators are not registered for every requested suite ("
            + details
            + ")"
        )
    return {suite: list(selected) for suite in suites}


def device_environment(platform: str, device: str | None) -> dict[str, str]:
    environment = os.environ.copy()
    if device is None:
        return environment
    if platform == "ascend":
        environment["ASCEND_RT_VISIBLE_DEVICES"] = device
        environment["NPU_VISIBLE_DEVICES"] = device
    else:
        environment["CUDA_VISIBLE_DEVICES"] = device
    return environment


def benchmark_records(output: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for raw_line in output.splitlines():
        line = re.sub(r"^\s*\d+:\s?", "", raw_line).strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") != "steady_state":
            continue
        case = record.get("case")
        provider = record.get("provider")
        if isinstance(case, str) and isinstance(provider, str):
            records.setdefault(case, {})[provider] = record
    return records


def ctest_command(
    build_dir: Path, operator: str, suite: str, platform: str
) -> list[str]:
    base = [
        "ctest",
        "--test-dir",
        str(build_dir),
        "-j1",
        "--output-on-failure",
    ]
    if suite == "functional":
        return base + ["-L", "^functional$", "-L", f"^{operator}$"]
    test_name = f"benchmark.{platform}.{operator}"
    return base + ["-R", f"^{re.escape(test_name)}$", "-V"]


def run_one(
    build_dir: Path,
    operator: str,
    suite: str,
    platform: str,
    environment: dict[str, str],
    timeout: int,
    verbose: bool,
) -> dict[str, Any]:
    command = ctest_command(build_dir, operator, suite, platform)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        exit_code = completed.returncode
        if "No tests were found" in stdout + stderr:
            status = "not_found"
        elif exit_code == 0:
            status = "passed"
        else:
            status = "failed"
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        exit_code = None
        status = "timeout"

    duration = time.monotonic() - started
    if verbose or status != "passed":
        if stdout:
            print(stdout, end="" if stdout.endswith("\n") else "\n")
        if stderr:
            print(
                stderr,
                file=sys.stderr,
                end="" if stderr.endswith("\n") else "\n",
            )

    result: dict[str, Any] = {
        "status": status,
        "duration_seconds": duration,
        "exit_code": exit_code,
        "command": command,
    }
    if suite == "benchmark":
        result["records"] = benchmark_records(stdout)
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FlagDNN native functional tests and benchmarks serially"
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(os.environ.get("FLAGDNN_BUILD_DIR", "build")),
        help="configured CMake build directory",
    )
    parser.add_argument(
        "--ops",
        help=(
            "comma-separated operators, or 'all' for every manifest operator "
            f"(default: {','.join(DEFAULT_OPERATORS)})"
        ),
    )
    parser.add_argument("--op-list-file", type=Path)
    parser.add_argument(
        "--suites",
        default="functional",
        help="functional, benchmark, comma-separated values, or all",
    )
    parser.add_argument(
        "--platform",
        default=os.environ.get("FLAGDNN_BENCHMARK_PLATFORM", "nvidia"),
        help="benchmark CTest platform component",
    )
    parser.add_argument(
        "--device",
        help="visible device id; leaves the current environment unchanged if omitted",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="timeout in seconds for each operator/suite",
    )
    parser.add_argument(
        "--output", type=Path, help="optional JSON summary path"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list every manifest operator and exit",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        manifests = operator_manifests()
        suites = requested_suites(arguments.suites)
        if arguments.list:
            listed = dict.fromkeys(
                operator for suite in suites for operator in manifests[suite]
            )
            print("\n".join(listed))
            return 0
        suite_operators = requested_operators(
            manifests,
            suites,
            arguments.ops,
            arguments.op_list_file,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    build_dir = arguments.build_dir.expanduser().resolve()
    if not (build_dir / "CTestTestfile.cmake").is_file():
        print(
            f"error: {build_dir} is not a configured CTest build directory",
            file=sys.stderr,
        )
        return 2
    if arguments.timeout <= 0:
        print("error: --timeout must be positive", file=sys.stderr)
        return 2

    environment = device_environment(arguments.platform, arguments.device)
    results: dict[str, dict[str, Any]] = {}
    failed = False
    operators = list(
        dict.fromkeys(
            operator for suite in suites for operator in suite_operators[suite]
        )
    )
    total = sum(len(suite_operators[suite]) for suite in suites)
    completed_count = 0

    for suite in suites:
        for operator in suite_operators[suite]:
            completed_count += 1
            print(
                f"[{completed_count}/{total}] {suite} {operator}",
                flush=True,
            )
            result = run_one(
                build_dir=build_dir,
                operator=operator,
                suite=suite,
                platform=arguments.platform,
                environment=environment,
                timeout=arguments.timeout,
                verbose=arguments.verbose,
            )
            results.setdefault(operator, {})[suite] = result
            status = result["status"]
            duration = result["duration_seconds"]
            print(f"  {status}: {duration:.2f}s", flush=True)
            failed = failed or status != "passed"

    summary = {
        "schema_version": 1,
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "build_dir": str(build_dir),
        "platform": arguments.platform,
        "operators": operators,
        "suites": suites,
        "suite_operators": suite_operators,
        "results": results,
    }
    if arguments.output:
        output = arguments.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"summary: {output}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
