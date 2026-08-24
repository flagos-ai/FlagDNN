#!/usr/bin/env python3

"""Serial native CTest runner for FlagDNN functional tests and benchmarks."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "cmake" / "Operators.cmake"
VALID_SUITES = ("functional", "benchmark")
BUILD_CONFIGURATION_FILE = ".flagdnn-build-config"
PLATFORM_ADAPTER_FILE = "run_tests_adapter.py"
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
PLATFORM_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
CTEST_SKIPPED_PATTERN = re.compile(
    r"(?:\*\*\*Skipped|\bNot Run\b.*\bSkipped\b|\(Skipped\))",
    flags=re.IGNORECASE,
)
CTEST_DISABLED_PATTERN = re.compile(
    r"(?:\*\*\*Not Run|\(Disabled\))",
    flags=re.IGNORECASE,
)
CTEST_AGGREGATE_SKIPPED_PATTERN = re.compile(
    r"^\s*\d+/\d+\s+Test\s+#\d+:.*(?:\*\*\*Skipped|\(Skipped\))",
    flags=re.IGNORECASE | re.MULTILINE,
)
CTEST_AGGREGATE_DISABLED_PATTERN = re.compile(
    r"^\s*\d+/\d+\s+Test\s+#\d+:.*(?:\*\*\*Not Run|\(Disabled\))",
    flags=re.IGNORECASE | re.MULTILINE,
)
PROCESS_TERMINATION_GRACE_SECONDS = 5.0
PROCESS_KILL_GRACE_SECONDS = 5.0
PROCESS_CLEANUP_POLL_SECONDS = 0.05
_PLATFORM_ADAPTERS: dict[str, ModuleType | None] = {}


def load_platform_adapter(platform: str) -> ModuleType | None:
    """Load optional test policy owned by backends/<platform>/validation."""
    if PLATFORM_PATTERN.fullmatch(platform) is None:
        raise ValueError("platform name must match [a-z][a-z0-9_]*")
    if platform in _PLATFORM_ADAPTERS:
        return _PLATFORM_ADAPTERS[platform]
    path = (
        ROOT
        / "backends"
        / platform
        / "validation"
        / PLATFORM_ADAPTER_FILE
    )
    if not path.is_file():
        _PLATFORM_ADAPTERS[platform] = None
        return None
    spec = importlib.util.spec_from_file_location(
        f"_flagdnn_run_tests_adapter_{platform}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load platform test adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        previous_bytecode_setting = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous_bytecode_setting
    except Exception as error:
        raise RuntimeError(
            f"Cannot initialize platform test adapter {path}: {error}"
        ) from error
    _PLATFORM_ADAPTERS[platform] = module
    return module


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


def registered_manifests(
    build_dir: Path,
    platform: str,
    manifests: dict[str, list[str]],
    suites: list[str],
    environment: dict[str, str],
    timeout: int,
    configuration: str | None = None,
) -> dict[str, list[str]]:
    command = [
        "ctest",
        "--test-dir",
        str(build_dir),
        "--show-only=json-v1",
    ]
    if configuration is not None:
        command.extend(["-C", configuration])
    stdout, stderr, exit_code, timed_out = run_process_group(
        command, environment, min(timeout, 60)
    )
    if timed_out:
        raise RuntimeError("CTest test discovery timed out")
    if exit_code != 0:
        detail = stderr.strip() or stdout.strip()
        raise RuntimeError(
            f"Cannot inspect configured {platform} tests"
            + (f": {detail}" if detail else "")
        )
    try:
        document = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"CTest returned invalid JSON while inspecting {platform} tests"
        ) from error

    tests = document.get("tests")
    if not isinstance(tests, list):
        raise RuntimeError("CTest JSON does not contain a test inventory")
    names = {
        test.get("name")
        for test in tests
        if isinstance(test, dict) and isinstance(test.get("name"), str)
    }

    result = dict(manifests)
    for suite in suites:
        prefix = f"{suite}.{platform}."
        result[suite] = [
            operator
            for operator in manifests[suite]
            if prefix + operator in names
        ]
        if not result[suite]:
            raise RuntimeError(
                f"No {suite} tests are registered for platform {platform}"
            )
    return result


def requested_suites(value: str) -> list[str]:
    raw = [item.strip() for item in value.split(",") if item.strip()]
    if not raw:
        raise ValueError("--suites must select at least one suite")
    if raw == ["all"]:
        return list(VALID_SUITES)
    if "all" in raw:
        raise ValueError("--suites 'all' cannot be combined with other values")
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
    if value is not None and list_file is not None:
        raise ValueError("--ops and --op-list-file are mutually exclusive")
    if value == "all":
        return {suite: list(manifests[suite]) for suite in suites}
    if value is not None:
        selected = [item.strip() for item in value.split(",") if item.strip()]
        if not selected:
            raise ValueError("--ops must select at least one operator")
    elif list_file is not None:
        selected = [
            line.strip()
            for line in list_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not selected:
            raise ValueError(
                "--op-list-file must contain at least one operator"
            )
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


def device_environment(
    platform: str,
    device: str | None,
    base_environment: dict[str, str] | None = None,
    adapter: ModuleType | None = None,
) -> dict[str, str]:
    environment = dict(
        os.environ if base_environment is None else base_environment
    )
    if adapter is None:
        adapter = load_platform_adapter(platform)
    configure = (
        None
        if adapter is None
        else getattr(adapter, "configure_environment", None)
    )
    if configure is None:
        if device is not None:
            raise ValueError(
                f"platform {platform} does not define device visibility"
            )
        return environment
    configure(environment, device)
    return environment


def resolve_build_directory(requested: Path | None, platform: str) -> Path:
    if requested is None:
        configured = os.environ.get("FLAGDNN_BUILD_DIR")
        requested = (
            Path(configured) if configured else Path("build") / platform
        )
    requested = requested.expanduser()
    if not requested.is_absolute():
        requested = ROOT / requested
    return requested.resolve()


def validate_build_directory(build_dir: Path) -> None:
    if (build_dir / "CTestTestfile.cmake").is_file():
        return

    installed_package = any(
        build_dir.glob("lib*/cmake/FlagDNN/FlagDNNConfig.cmake")
    )
    if installed_package or (build_dir / "share" / "flagdnn").is_dir():
        suggestion = "a configured build tree containing CTestTestfile.cmake"
        parent_build = build_dir.parent
        if (
            build_dir.name == "install"
            and (parent_build / "CTestTestfile.cmake").is_file()
        ):
            suggestion = str(parent_build)
        raise ValueError(
            f"{build_dir} is an installed FlagDNN SDK; validation tests are "
            f"not installed. Use --build-dir {suggestion}"
        )

    raise ValueError(
        f"{build_dir} is not a configured CTest build directory; "
        "run tools/build.sh with tests enabled or pass --build-dir"
    )


def resolve_build_configuration(
    build_dir: Path, requested: str | None
) -> str | None:
    """Resolve the CTest configuration for single- and multi-config builds."""

    def valid(value: str) -> bool:
        return re.fullmatch(r"[A-Za-z0-9_.+-]+", value) is not None

    if requested is not None:
        requested = requested.strip()
        if not requested or not valid(requested):
            raise ValueError(
                "--config must contain a CMake configuration name"
            )
        return requested

    cache_values: dict[str, str] = {}
    cache = build_dir / "CMakeCache.txt"
    if cache.is_file():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.startswith("CMAKE_BUILD_TYPE:") or line.startswith(
                "CMAKE_CONFIGURATION_TYPES:"
            ):
                key, _, value = line.partition("=")
                cache_values[key.partition(":")[0]] = value.strip()
    build_type = cache_values.get("CMAKE_BUILD_TYPE", "")
    if build_type:
        return build_type
    marker = build_dir / BUILD_CONFIGURATION_FILE
    if marker.is_file():
        value = marker.read_text(encoding="utf-8").strip()
        if not value or not valid(value):
            raise ValueError(f"invalid build configuration marker: {marker}")
        return value
    if cache_values.get("CMAKE_CONFIGURATION_TYPES"):
        return "Release"
    return None


def benchmark_records(
    output: str,
    adapter: ModuleType | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    records: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    v1_keys = {
        "schema_version",
        "kind",
        "provider",
        "case",
        "unit",
        "median",
        "p90",
        "samples",
    }

    def positive_number(value: Any) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and value > 0
        )

    def valid_metric(value: Any) -> bool:
        return (
            isinstance(value, dict)
            and set(value) == {"median", "p90", "samples"}
            and positive_number(value.get("median"))
            and positive_number(value.get("p90"))
            and isinstance(value.get("samples"), list)
            and bool(value["samples"])
            and all(positive_number(sample) for sample in value["samples"])
        )

    def valid_summary(metric: dict[str, Any]) -> bool:
        ordered_samples = sorted(float(sample) for sample in metric["samples"])

        def nearest_rank(fraction: float) -> float:
            index = max(0, math.ceil(fraction * len(ordered_samples)) - 1)
            return ordered_samples[index]

        return math.isclose(
            float(metric["median"]),
            nearest_rank(0.5),
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        ) and math.isclose(
            float(metric["p90"]),
            nearest_rank(0.9),
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        )

    for raw_line in output.splitlines():
        line = re.sub(r"^\s*\d+:\s?", "", raw_line).strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(
                "malformed benchmark JSON "
                f"(line {error.lineno}, column {error.colno}): {line}"
            )
            continue
        if not isinstance(record, dict):
            errors.append(f"benchmark record is not an object: {line}")
            continue
        # Backends and launchers may emit their own structured diagnostics.
        # Only records explicitly claiming the steady-state timing protocol
        # belong to this parser.
        if record.get("kind") != "steady_state":
            continue
        version = record.get("schema_version")
        version_is_v1 = (
            isinstance(version, int)
            and not isinstance(version, bool)
            and version == 1
        )
        case = record.get("case")
        provider = record.get("provider")
        if (
            not isinstance(case, str)
            or not case
            or not isinstance(provider, str)
            or not provider
        ):
            errors.append(
                "benchmark record has an invalid case or provider: " + line
            )
            continue
        if version_is_v1:
            actual_keys = set(record)
            if actual_keys != v1_keys:
                missing = sorted(v1_keys - actual_keys)
                extra = sorted(actual_keys - v1_keys)
                errors.append(
                    "benchmark record fields do not match schema"
                    f"; missing={missing}; extra={extra}"
                )
                continue
            metric = {
                "median": record.get("median"),
                "p90": record.get("p90"),
                "samples": record.get("samples"),
            }
            if record.get("unit") != "us" or not valid_metric(metric):
                errors.append(
                    "benchmark record violates result.schema.json: " + line
                )
                continue
            metrics = {"timing": metric}
        else:
            validator = (
                None
                if adapter is None
                else getattr(adapter, "validate_benchmark_record", None)
            )
            if validator is None:
                errors.append(
                    f"unsupported benchmark schema_version={version!r}: "
                    + line
                )
                continue
            try:
                metrics = validator(record)
            except ValueError as error:
                errors.append(
                    "benchmark record violates result.schema.json: "
                    f"{error}: {line}"
                )
                continue
            if (
                not isinstance(metrics, dict)
                or not metrics
                or not all(
                    isinstance(name, str) and isinstance(metric, dict)
                    for name, metric in metrics.items()
                )
            ):
                errors.append(
                    "platform adapter returned invalid benchmark metrics: "
                    + line
                )
                continue
        invalid_metrics = [
            name
            for name, metric in metrics.items()
            if not valid_metric(metric) or not valid_summary(metric)
        ]
        if invalid_metrics:
            errors.append(
                f"benchmark summary does not match samples for "
                f"case={case} provider={provider} "
                f"metrics={','.join(invalid_metrics)}"
            )
            continue
        case_records = records.setdefault(case, {})
        if provider in case_records:
            errors.append(
                f"duplicate benchmark record for case={case} "
                f"provider={provider}"
            )
            continue
        case_records[provider] = record
    return records, errors


def ctest_status(exit_code: int, output: str) -> str:
    if "No tests were found" in output:
        return "not_found"
    if exit_code != 0:
        return "failed"
    if CTEST_DISABLED_PATTERN.search(output) is not None:
        return "failed"
    if CTEST_SKIPPED_PATTERN.search(output) is not None:
        return "skipped"
    return "passed"


def ctest_aggregate_status(exit_code: int, output: str) -> str:
    """Classify an outer CTest run without consuming nested test output."""
    if "No tests were found" in output:
        return "not_found"
    if exit_code != 0:
        return "failed"
    if CTEST_AGGREGATE_DISABLED_PATTERN.search(output) is not None:
        return "failed"
    if CTEST_AGGREGATE_SKIPPED_PATTERN.search(output) is not None:
        return "skipped"
    return "passed"


def verbose_ctest_command(
    build_dir: Path,
    expression: str,
    configuration: str | None = None,
) -> list[str]:
    command = [
        "ctest",
        "--test-dir",
        str(build_dir),
        "-j1",
        "-R",
        expression,
        "-V",
    ]
    if configuration is not None:
        command.extend(["-C", configuration])
    return command


def ctest_command(
    build_dir: Path,
    operator: str,
    suite: str,
    platform: str,
    configuration: str | None = None,
    adapter: ModuleType | None = None,
) -> list[str]:
    test_name = f"{suite}.{platform}.{operator}"
    expression = f"^{re.escape(test_name)}$"
    if adapter is None:
        adapter = load_platform_adapter(platform)
    expression_hook = (
        None if adapter is None else getattr(adapter, "test_expression", None)
    )
    if expression_hook is not None:
        expression = expression_hook(suite, operator) or expression
    return verbose_ctest_command(build_dir, expression, configuration)


class _TerminationSignal(SystemExit):
    def __init__(self, signum: int):
        super().__init__(128 + signum)
        self.signum = signum


def _linux_process_identity(pid: int) -> tuple[str, int, int] | None:
    """Return (state, process-group, session) from procfs, if available."""
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text(
            encoding="utf-8"
        )
    except (OSError, UnicodeError):
        return None
    # The comm field is parenthesized and may itself contain whitespace or
    # parentheses.  Splitting after its final ')' keeps the fixed stat fields
    # aligned: state, ppid, pgrp, session, ...
    comm_end = stat.rfind(")")
    if comm_end < 0:
        return None
    fields = stat[comm_end + 1 :].split()
    if len(fields) < 4:
        return None
    try:
        return fields[0], int(fields[2]), int(fields[3])
    except ValueError:
        return None


def _session_process_groups(session_id: int) -> dict[int, set[int]]:
    """Snapshot live process groups belonging to one Linux/POSIX session."""
    if session_id <= 0:
        return {}
    try:
        if session_id == os.getsid(0):
            # Never allow cleanup of the runner's own session.
            return {}
    except OSError:
        return {}

    groups: dict[int, set[int]] = {}
    proc = Path("/proc")
    if proc.is_dir():
        try:
            entries = tuple(proc.iterdir())
        except OSError:
            entries = ()
        for entry in entries:
            if not entry.name.isdecimal():
                continue
            pid = int(entry.name)
            identity = _linux_process_identity(pid)
            if identity is None:
                continue
            state, process_group, process_session = identity
            if (
                process_session == session_id
                and process_group > 0
                and state not in {"Z", "X"}
            ):
                groups.setdefault(process_group, set()).add(pid)
        return groups

    # procfs is the race-resistant implementation used on Linux.  Retain a
    # conservative POSIX fallback for other hosts: start_new_session=True
    # guarantees that the direct child initially has PGID == SID == PID.
    try:
        if (
            os.getsid(session_id) == session_id
            and os.getpgid(session_id) == session_id
        ):
            groups[session_id] = {session_id}
    except (OSError, ProcessLookupError):
        pass
    return groups


def _signal_session_process_groups(
    session_id: int,
    signum: signal.Signals,
    previously_signaled: dict[int, set[int]],
) -> None:
    """Signal each newly observed group in a private child session."""
    groups = _session_process_groups(session_id)
    own_process_group = os.getpgrp()
    # Descendant groups are signaled first.  For SIGKILL this gives their
    # supervisor a brief opportunity to reap them before the session leader
    # is killed as well.
    ordered_groups = sorted(groups, key=lambda group: group == session_id)
    killed_descendant_group = False
    for process_group in ordered_groups:
        members = groups[process_group]
        if process_group == own_process_group:
            continue
        if members.issubset(previously_signaled.get(process_group, set())):
            continue
        # Re-snapshot immediately before killpg.  This narrows the unavoidable
        # PID/PGID reuse race and prevents a stale group id from escaping the
        # target session.
        current_members = _session_process_groups(session_id).get(
            process_group, set()
        )
        if not current_members:
            continue
        if (
            signum == signal.SIGKILL
            and process_group == session_id
            and killed_descendant_group
        ):
            time.sleep(PROCESS_CLEANUP_POLL_SECONDS)
        try:
            os.killpg(process_group, signum)
        except ProcessLookupError:
            continue
        previously_signaled.setdefault(process_group, set()).update(
            current_members
        )
        if signum == signal.SIGKILL and process_group != session_id:
            killed_descendant_group = True


def _terminate_process_session(
    process: subprocess.Popen[str], session_id: int
) -> tuple[str, str]:
    """Terminate and reap every process group in a spawned test session."""
    latest_output = ""
    latest_error = ""

    def cleanup_phase(signum: signal.Signals, grace_seconds: float) -> bool:
        nonlocal latest_output, latest_error
        deadline = time.monotonic() + grace_seconds
        signaled: dict[int, set[int]] = {}
        while True:
            _signal_session_process_groups(session_id, signum, signaled)
            remaining = deadline - time.monotonic()
            wait_seconds = max(
                0.001,
                min(PROCESS_CLEANUP_POLL_SECONDS, max(0.0, remaining)),
            )
            try:
                latest_output, latest_error = process.communicate(
                    timeout=wait_seconds
                )
            except subprocess.TimeoutExpired:
                pass
            if (
                process.poll() is not None
                and not _session_process_groups(session_id)
            ):
                if not latest_output and not latest_error:
                    latest_output, latest_error = process.communicate()
                return True
            if remaining <= 0.0:
                return False

    if cleanup_phase(signal.SIGTERM, PROCESS_TERMINATION_GRACE_SECONDS):
        return latest_output, latest_error
    if cleanup_phase(signal.SIGKILL, PROCESS_KILL_GRACE_SECONDS):
        return latest_output, latest_error

    # An uninterruptible descendant can keep an inherited pipe open.  Do not
    # hang the runner forever after both bounded cleanup phases.  The direct
    # child is still forcibly reaped, while captured prefix output remains
    # available for diagnostics.
    try:
        process.kill()
    except ProcessLookupError:
        pass
    try:
        return process.communicate(timeout=PROCESS_CLEANUP_POLL_SECONDS)
    except subprocess.TimeoutExpired as error:
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        try:
            process.wait(timeout=PROCESS_CLEANUP_POLL_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        output = error.output if isinstance(error.output, str) else ""
        stderr = error.stderr if isinstance(error.stderr, str) else ""
        return output or latest_output, stderr or latest_error


def run_process_group(
    command: list[str],
    environment: dict[str, str],
    timeout: int,
) -> tuple[str, str, int | None, bool]:
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    previous_handlers: dict[signal.Signals, Any] = {}

    def handle_termination(signum: int, _frame: Any) -> None:
        raise _TerminationSignal(signum)

    for signum in (signal.SIGTERM, signal.SIGHUP):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handle_termination)

    session_id = process.pid

    def terminate() -> tuple[str, str]:
        return _terminate_process_session(process, session_id)

    try:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            return stdout, stderr, process.returncode, False
        except subprocess.TimeoutExpired:
            stdout, stderr = terminate()
            return stdout, stderr, None, True
        except BaseException:
            try:
                terminate()
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise

    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def run_one(
    build_dir: Path,
    operator: str,
    suite: str,
    platform: str,
    environment: dict[str, str],
    timeout: int,
    verbose: bool,
    manifest_operators: list[str],
    configuration: str | None = None,
    adapter: ModuleType | None = None,
) -> dict[str, Any]:
    if adapter is None:
        adapter = load_platform_adapter(platform)
    command = ctest_command(
        build_dir,
        operator,
        suite,
        platform,
        configuration,
        adapter,
    )
    started = time.monotonic()
    stdout, stderr, exit_code, timed_out = run_process_group(
        command, environment, timeout
    )
    if timed_out:
        status = "timeout"
    else:
        status = ctest_status(exit_code, stdout + "\n" + stderr)
    ctest_reported_status = status

    duration = time.monotonic() - started
    result: dict[str, Any] = {
        "status": status,
        "duration_seconds": duration,
        "exit_code": exit_code,
        "command": command,
    }
    combined_output = stdout + "\n" + stderr
    records: dict[str, dict[str, Any]] = {}
    if suite == "benchmark":
        records, record_errors = benchmark_records(combined_output, adapter)
        if ctest_reported_status == "passed" and not records:
            record_errors.append(
                "passed benchmark emitted no steady-state timing records"
            )
        if ctest_reported_status == "skipped" and records:
            record_errors.append(
                "skipped benchmark emitted timing provider records"
            )
        result["records"] = records
        if record_errors:
            result["record_errors"] = record_errors
            if status in ("passed", "skipped"):
                status = "failed"
                result["status"] = status
    postprocess = (
        None
        if adapter is None
        else getattr(adapter, "postprocess_result", None)
    )
    if postprocess is not None:
        postprocess(
            result=result,
            ctest_reported_status=ctest_reported_status,
            output=combined_output,
            operator=operator,
            suite=suite,
            records=records,
            manifest_operators=manifest_operators,
        )
    status = result["status"]
    if verbose or status != "passed":
        if stdout:
            print(stdout, end="" if stdout.endswith("\n") else "\n")
        if stderr:
            print(
                stderr,
                file=sys.stderr,
                end="" if stderr.endswith("\n") else "\n",
            )
        for error in result.get("record_errors", []):
            print(f"benchmark record error: {error}", file=sys.stderr)
        diagnostics = (
            []
            if adapter is None
            else getattr(adapter, "result_diagnostics", lambda _result: [])(
                result
            )
        )
        for label, error in diagnostics:
            print(f"{label}: {error}", file=sys.stderr)
    return result


def required_preflight_tests(
    platform: str,
    suites: list[str] | tuple[str, ...] = VALID_SUITES,
    adapter: ModuleType | None = None,
) -> set[str]:
    required_tests = {
        "core.json_contract",
        "core.backend_dependency_boundary",
        "core.backend_contract",
        "core.c_header",
        "core.c_api",
        "core.convolution_backward_api",
        "core.batchnorm_inference_api",
        "core.cpp_header",
        "core.frontend_api",
        "core.test_architecture",
        "core.run_tests_contract",
        "core.default_backend_contract",
    }
    if "benchmark" in suites:
        required_tests.update(
            {
                "benchmark.catalog_contract",
                "benchmark.catalog_dependency_boundary",
            }
        )
    if adapter is None:
        adapter = load_platform_adapter(platform)
    platform_tests = (
        None if adapter is None else getattr(adapter, "preflight_tests", None)
    )
    if platform_tests is not None:
        required_tests.update(platform_tests(suites))
    return required_tests


def run_preflight(
    build_dir: Path,
    platform: str,
    environment: dict[str, str],
    timeout: int,
    verbose: bool,
    suites: list[str],
    configuration: str | None = None,
    adapter: ModuleType | None = None,
) -> dict[str, Any]:
    if adapter is None:
        adapter = load_platform_adapter(platform)
    metadata: dict[str, Any] = {"visibility_masks": {}}
    metadata_hook = (
        None
        if adapter is None
        else getattr(adapter, "preflight_metadata", None)
    )
    if metadata_hook is not None:
        metadata.update(metadata_hook(environment))
    required_tests = required_preflight_tests(platform, suites, adapter)

    listing_command = [
        "ctest",
        "--test-dir",
        str(build_dir),
        "--show-only=json-v1",
    ]
    if configuration is not None:
        listing_command.extend(["-C", configuration])
    listing_stdout, listing_stderr, listing_exit, listing_timeout = (
        run_process_group(listing_command, environment, min(timeout, 60))
    )
    listing_error: str | None = None
    discovered_tests: set[str] = set()
    if listing_timeout:
        listing_error = "CTest catalog listing timed out"
    elif listing_exit != 0:
        listing_error = "CTest catalog listing failed: " + (
            listing_stderr.strip() or listing_stdout.strip()
        )
    else:
        try:
            catalog = json.loads(listing_stdout)
            discovered_tests = {
                test["name"]
                for test in catalog.get("tests", [])
                if isinstance(test, dict) and isinstance(test.get("name"), str)
            }
        except (AttributeError, json.JSONDecodeError, TypeError) as error:
            listing_error = f"cannot parse CTest JSON catalog: {error}"
    missing_tests = sorted(required_tests - discovered_tests)
    if listing_error is not None or missing_tests:
        errors = []
        if listing_error is not None:
            errors.append(listing_error)
        if missing_tests:
            errors.append(
                "required preflight tests are missing: "
                + ", ".join(missing_tests)
            )
        for error in errors:
            print(f"preflight error: {error}", file=sys.stderr)
        return {
            "status": "failed",
            "duration_seconds": 0.0,
            "exit_code": listing_exit,
            "command": listing_command,
            "required_tests": sorted(required_tests),
            "missing_tests": missing_tests,
            "errors": errors,
            **metadata,
        }

    patterns = [r"core\."]
    if "benchmark" in suites:
        patterns.append(r"benchmark\.catalog_")
    patterns.append(rf"integration\.{re.escape(platform)}\.")
    expression = "^(" + "|".join(patterns) + ")"
    command = verbose_ctest_command(build_dir, expression, configuration)
    started = time.monotonic()
    stdout, stderr, exit_code, timed_out = run_process_group(
        command, environment, timeout
    )
    status = (
        "timeout"
        if timed_out
        else ctest_aggregate_status(exit_code, stdout + "\n" + stderr)
    )
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
    return {
        "status": status,
        "duration_seconds": duration,
        "exit_code": exit_code,
        "command": command,
        "required_tests": sorted(required_tests),
        "missing_tests": [],
        "errors": [],
        **metadata,
    }


def summary_status_fields(
    overall_status: str,
    exit_code: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "overall_status": overall_status,
        "exit_code": exit_code,
    }


def summary_envelope(
    arguments: argparse.Namespace,
    overall_status: str,
    exit_code: int | None,
) -> dict[str, Any]:
    """Build the stable top-level fields shared by every summary state."""
    return {
        **summary_status_fields(overall_status, exit_code),
        "platform": arguments.platform,
        "device": arguments.device,
    }


def preliminary_output_argument(arguments: list[str]) -> Path | None:
    """Find the last explicit --output before full argparse validation."""
    requested: Path | None = None
    for index, argument in enumerate(arguments):
        if argument.startswith("--output="):
            requested = Path(argument.partition("=")[2])
        elif (
            argument == "--output"
            and index + 1 < len(arguments)
            and not arguments[index + 1].startswith("-")
        ):
            requested = Path(arguments[index + 1])
    return requested


def atomic_write_summary(output: Path, summary: dict[str, Any]) -> None:
    """Publish one complete JSON document without exposing partial writes."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(summary, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Run FlagDNN native functional tests and benchmarks " "serially"
        )
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        help=(
            "configured CMake build directory "
            "(default: FLAGDNN_BUILD_DIR or build/<platform>)"
        ),
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
        "--min-speedup",
        type=float,
        help=(
            "platform-defined benchmark speedup threshold; available only "
            "when supported by the selected platform adapter"
        ),
    )
    parser.add_argument(
        "--platform",
        default=os.environ.get("FLAGDNN_BENCHMARK_PLATFORM", "nvidia"),
        help="CTest platform component",
    )
    parser.add_argument(
        "--device",
        help=(
            "visible device id; leaves the current environment unchanged "
            "if omitted"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help=(
            "timeout in seconds for each operator/suite "
            "(default: selected platform policy, or 1800)"
        ),
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("FLAGDNN_BUILD_TYPE"),
        help=(
            "CMake build configuration for CTest (default: the configuration "
            "recorded by tools/build.sh, the single-config CMAKE_BUILD_TYPE, "
            "or Release for a multi-config tree)"
        ),
    )
    preflight = parser.add_mutually_exclusive_group()
    preflight.add_argument(
        "--preflight",
        dest="preflight",
        action="store_true",
        help="run core and platform integration contracts before operators",
    )
    preflight.add_argument(
        "--no-preflight",
        dest="preflight",
        action="store_false",
        help="skip core and platform integration contracts",
    )
    parser.set_defaults(preflight=None)
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


def _run_main() -> int:
    preliminary_output = preliminary_output_argument(sys.argv[1:])
    output: Path | None = None
    if preliminary_output is not None:
        try:
            output = preliminary_output.expanduser().resolve()
            atomic_write_summary(
                output,
                {
                    **summary_status_fields("running", None),
                    "phase": "argument_validation",
                },
            )
        except (OSError, RuntimeError) as error:
            print(
                f"error: cannot initialize --output summary: {error}",
                file=sys.stderr,
            )
            return 2
    try:
        arguments = parse_arguments()
    except SystemExit as error:
        exit_code = error.code if isinstance(error.code, int) else 2
        if output is not None:
            try:
                atomic_write_summary(
                    output,
                    {
                        **summary_status_fields(
                            "passed" if exit_code == 0 else "failed",
                            exit_code,
                        ),
                        "phase": "argument_validation",
                        **({"mode": "help"} if exit_code == 0 else {}),
                    },
                )
            except OSError as summary_error:
                print(
                    "error: cannot finalize --output summary: "
                    f"{summary_error}",
                    file=sys.stderr,
                )
                return 2
        return exit_code
    if arguments.output is not None:
        parsed_output = arguments.output.expanduser().resolve()
        if output is None or parsed_output != output:
            print(
                "error: could not safely identify --output before argument "
                "validation",
                file=sys.stderr,
            )
            return 2
        atomic_write_summary(
            output,
            summary_envelope(arguments, "running", None),
        )

    def finish_early(
        exit_code: int,
        overall_status: str,
        **fields: Any,
    ) -> int:
        if output is not None:
            summary = summary_envelope(
                arguments, overall_status, exit_code
            )
            summary.update(fields)
            try:
                atomic_write_summary(output, summary)
            except OSError as error:
                print(
                    f"error: cannot finalize --output summary: {error}",
                    file=sys.stderr,
                )
                return 2
        return exit_code

    def validation_error(message: str) -> int:
        print(f"error: {message}", file=sys.stderr)
        return finish_early(2, "failed", error=message)

    if PLATFORM_PATTERN.fullmatch(arguments.platform) is None:
        return validation_error(
            "--platform must match [a-z][a-z0-9_]*"
        )
    try:
        adapter = load_platform_adapter(arguments.platform)
        manifests = operator_manifests()
        suites = requested_suites(arguments.suites)
        if arguments.list:
            listed = dict.fromkeys(
                operator for suite in suites for operator in manifests[suite]
            )
            print("\n".join(listed))
            return finish_early(
                0,
                "passed",
                mode="list",
                suites=suites,
                operators=list(listed),
            )
        filter_registered = bool(
            adapter is not None
            and getattr(adapter, "FILTER_REGISTERED_TESTS", False)
        )
        suite_operators = (
            None
            if filter_registered
            else requested_operators(
                manifests,
                suites,
                arguments.ops,
                arguments.op_list_file,
            )
        )
    except (OSError, RuntimeError, ValueError) as error:
        return validation_error(str(error))

    build_dir = resolve_build_directory(
        arguments.build_dir, arguments.platform
    )
    try:
        validate_build_directory(build_dir)
        build_configuration = resolve_build_configuration(
            build_dir, arguments.config
        )
        default_timeout = (
            1800
            if adapter is None
            else getattr(adapter, "DEFAULT_TIMEOUT", 1800)
        )
        timeout = (
            default_timeout if arguments.timeout is None else arguments.timeout
        )
        if (
            not isinstance(timeout, int)
            or isinstance(timeout, bool)
            or timeout <= 0
        ):
            raise ValueError("--timeout must be positive")
        environment = device_environment(
            arguments.platform,
            arguments.device,
            adapter=adapter,
        )
        if filter_registered:
            manifests = registered_manifests(
                build_dir,
                arguments.platform,
                manifests,
                suites,
                environment,
                timeout,
                build_configuration,
            )
            suite_operators = requested_operators(
                manifests,
                suites,
                arguments.ops,
                arguments.op_list_file,
            )
    except (OSError, RuntimeError, ValueError) as error:
        return validation_error(str(error))

    assert suite_operators is not None
    if arguments.min_speedup is not None and (
        not math.isfinite(arguments.min_speedup)
        or arguments.min_speedup <= 0.0
    ):
        return validation_error("--min-speedup must be finite and positive")
    if arguments.min_speedup is not None and "benchmark" not in suites:
        return validation_error("--min-speedup requires the benchmark suite")
    supports_min_speedup = bool(
        adapter is not None
        and getattr(adapter, "SUPPORTS_MIN_SPEEDUP", False)
    )
    if arguments.min_speedup is not None and not supports_min_speedup:
        return validation_error(
            f"--min-speedup is not supported by platform "
            f"{arguments.platform}"
        )

    prepare = None if adapter is None else getattr(adapter, "prepare", None)
    try:
        platform_state = (
            {} if prepare is None else prepare(manifests, suites)
        )
    except (OSError, RuntimeError, ValueError) as error:
        return validation_error(str(error))
    if not isinstance(platform_state, dict):
        return validation_error("platform adapter returned invalid state")

    manifest_operators = list(
        dict.fromkeys(
            operator
            for manifest in manifests.values()
            for operator in manifest
        )
    )
    operators = list(
        dict.fromkeys(
            operator for suite in suites for operator in suite_operators[suite]
        )
    )
    total = sum(len(suite_operators[suite]) for suite in suites)
    if total == 0:
        return validation_error(
            "operator selection produced zero operator/suite runs"
        )

    prepare_run = (
        None if adapter is None else getattr(adapter, "prepare_run", None)
    )
    try:
        run_state = (
            {}
            if prepare_run is None
            else prepare_run(
                environment=environment,
                verbose=arguments.verbose,
            )
        )
    except (OSError, RuntimeError, ValueError) as error:
        return validation_error(str(error))
    if not isinstance(run_state, dict):
        return validation_error("platform adapter returned invalid run state")
    duplicate_state = set(platform_state).intersection(run_state)
    if duplicate_state:
        return validation_error(
            "platform adapter returned duplicate state fields: "
            + ", ".join(sorted(duplicate_state))
        )
    platform_state.update(run_state)

    results: dict[str, dict[str, Any]] = {}
    failed = False
    preflight_result: dict[str, Any] | None = None
    preflight_default = bool(
        adapter is not None
        and getattr(adapter, "PREFLIGHT_BY_DEFAULT", False)
    )
    should_run_preflight = (
        preflight_default
        if arguments.preflight is None
        else arguments.preflight
    )
    if should_run_preflight:
        print(
            "[preflight] core and platform integration contracts", flush=True
        )
        preflight_result = run_preflight(
            build_dir=build_dir,
            platform=arguments.platform,
            environment=environment,
            timeout=timeout,
            verbose=arguments.verbose,
            suites=suites,
            configuration=build_configuration,
            adapter=adapter,
        )
        preflight_status = preflight_result["status"]
        print(
            "  "
            f"{preflight_status}: "
            f"{preflight_result['duration_seconds']:.2f}s",
            flush=True,
        )
        failed = preflight_status != "passed"

    status_is_success = (
        (lambda status: status == "passed")
        if adapter is None
        else getattr(
            adapter,
            "status_is_success",
            lambda status: status == "passed",
        )
    )
    completed_count = 0
    if not failed:
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
                    timeout=timeout,
                    verbose=arguments.verbose,
                    manifest_operators=manifest_operators,
                    configuration=build_configuration,
                    adapter=adapter,
                )
                results.setdefault(operator, {})[suite] = result
                status = result["status"]
                duration = result["duration_seconds"]
                print(f"  {status}: {duration:.2f}s", flush=True)
                failed = failed or not status_is_success(status)

    preflight_passed = (
        preflight_result is None or preflight_result["status"] == "passed"
    )
    finalize = None if adapter is None else getattr(adapter, "finalize", None)
    platform_outcome = (
        {"failed": False, "summary": {}, "coverage": {}}
        if finalize is None
        else finalize(
            results=results,
            suite_operators=suite_operators,
            suites=suites,
            state=platform_state,
            min_speedup=arguments.min_speedup,
            preflight_passed=preflight_passed,
        )
    )
    if not isinstance(platform_outcome, dict):
        raise RuntimeError("platform adapter returned an invalid outcome")
    failed = failed or bool(platform_outcome.get("failed", False))

    status_counts = {
        status: sum(
            result["status"] == status
            for operator_results in results.values()
            for result in operator_results.values()
        )
        for status in ("passed", "failed", "skipped", "timeout", "not_found")
    }
    benchmark_case_pairs = 0
    benchmark_provider_records = 0
    benchmark_record_errors = 0
    for operator_results in results.values():
        for result in operator_results.values():
            if result["status"] == "passed":
                records = result.get("records", {})
                if isinstance(records, dict):
                    benchmark_provider_records += sum(
                        len(providers)
                        for providers in records.values()
                        if isinstance(providers, dict)
                    )
                    benchmark_case_pairs += sum(
                        isinstance(providers, dict) and len(providers) >= 2
                        for providers in records.values()
                    )
            benchmark_record_errors += len(result.get("record_errors", []))

    coverage = {
        "benchmark_case_pairs": benchmark_case_pairs,
        "benchmark_provider_records": benchmark_provider_records,
        "benchmark_record_errors": benchmark_record_errors,
    }
    platform_coverage = platform_outcome.get("coverage", {})
    if not isinstance(platform_coverage, dict):
        raise RuntimeError("platform adapter returned invalid coverage")
    coverage.update(platform_coverage)
    platform_summary = platform_outcome.get("summary", {})
    if not isinstance(platform_summary, dict):
        raise RuntimeError("platform adapter returned invalid summary")

    final_exit_code = 1 if failed else 0
    summary = {
        **summary_envelope(
            arguments,
            "failed" if failed else "passed",
            final_exit_code,
        ),
        "build_dir": str(build_dir),
        "build_configuration": build_configuration,
        "operators": operators,
        "suites": suites,
        "suite_operators": suite_operators,
        "preflight": preflight_result,
        "status_counts": status_counts,
        "coverage": coverage,
        "comparable_coverage": None,
        "performance": None,
        "results": results,
        **platform_summary,
    }
    if output is not None:
        try:
            atomic_write_summary(output, summary)
        except OSError as error:
            print(
                f"error: cannot finalize --output summary: {error}",
                file=sys.stderr,
            )
            return 2
        print(f"summary: {output}")

    return final_exit_code

def main() -> int:
    previous_handlers: dict[signal.Signals, Any] = {}

    def handle_termination(signum: int, _frame: Any) -> None:
        raise _TerminationSignal(signum)

    for signum in (signal.SIGTERM, signal.SIGHUP):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handle_termination)
    try:
        return _run_main()
    except _TerminationSignal as termination:
        exit_code = 128 + termination.signum
        requested_output = preliminary_output_argument(sys.argv[1:])
        if requested_output is not None:
            try:
                atomic_write_summary(
                    requested_output.expanduser().resolve(),
                    {
                        **summary_status_fields("failed", exit_code),
                        "phase": "signal",
                        "signal": signal.Signals(termination.signum).name,
                    },
                )
            except (OSError, RuntimeError) as error:
                print(
                    f"error: cannot finalize --output summary: {error}",
                    file=sys.stderr,
                )
        return exit_code
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
