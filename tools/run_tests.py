#!/usr/bin/env python3

"""Run FlagDNN accuracy and performance tests with per-device workers."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import importlib.metadata
import importlib.util
import json
import math
import multiprocessing as mp
import os
import platform
import queue
from pathlib import Path
import re
import signal
import shutil
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


SUITE_NAMES = {"functional": "accuracy", "benchmark": "performance"}
STATUS_NAMES = {
    "passed": "Passed",
    "failed": "Failed",
    "skipped": "Skipped",
    "timeout": "Timeout",
    "not_found": "NotFound",
}
COUNT_NAMES = ("passed", "failed", "skipped", "errors", "xfailed", "xpassed")
DTYPE_NAMES = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "float32": "fp32",
    "float64": "fp64",
    "e4m3": "f8-e4m3fn",
    "e5m2": "f8-e5m2",
    "e8m0": "fp8_e8m0",
}
DTYPE_PATTERN = re.compile(
    r"(?:^|_)(bfloat16|float16|float32|float64|bf16|fp16|fp32|fp64|"
    r"tf32|int8|int16|int32|int64|uint8|bool|e4m3|e5m2|e8m0)(?:_|$)"
)


def diagnostic_path(output: Path) -> Path:
    return output.with_name(output.stem + ".native.json")


def case_dtype(case: str) -> str:
    # FP8 matrix names specify both input formats before any output dtype.
    fp8 = re.search(r"_a(e4m3|e5m2|e8m0)_b(e4m3|e5m2|e8m0)_", case)
    if fp8:
        return DTYPE_NAMES[fp8.group(1)]
    match = DTYPE_PATTERN.search(case)
    return DTYPE_NAMES.get(match[1], match[1]) if match else "unknown"


def functional_counts(output: str, status: str) -> dict[str, int]:
    """Count native cases across CTest groups, including partial failures."""
    groups: dict[str, dict[str, Any]] = {}
    for raw in output.splitlines():
        prefix = re.match(r"^\s*(\d+):\s?(.*)$", raw)
        group_id, line = (prefix[1], prefix[2]) if prefix else ("", raw)
        group = groups.setdefault(
            group_id, {"totals": {}, "passed": set(), "skipped": set()}
        )
        total = re.match(r"^(\S+): (PASS|SKIP) cases=(\d+)\b", line)
        if total:
            accounting = dict(
                re.findall(r"\b(executed|skipped)=(\d+)\b", line)
            )
            if "executed" in accounting and "skipped" in accounting:
                group["totals"][(total[1], "PASS")] = int(
                    accounting["executed"]
                )
                group["totals"][(total[1], "SKIP")] = int(
                    accounting["skipped"]
                )
            else:
                group["totals"][(total[1], total[2])] = int(total[3])
        elif skip := re.match(r"^SKIP case=(\S+)\b", line):
            group["skipped"].add(skip[1])
        elif re.match(r"^\S+: .*\bPASS\b", line):
            group["passed"].add(line.split(":", 1)[0])
    counts = dict.fromkeys(COUNT_NAMES, 0)
    for group in groups.values():
        counts["passed"] += max(
            len(group["passed"]),
            sum(
                value
                for (_, kind), value in group["totals"].items()
                if kind == "PASS"
            ),
        )
        counts["skipped"] += max(
            len(group["skipped"]),
            sum(
                value
                for (_, kind), value in group["totals"].items()
                if kind == "SKIP"
            ),
        )
    # An aborted native executable does not expose a pytest-style failing
    # parameter count. Report its execution error, without inventing cases.
    if status == "failed":
        counts["errors"] = 1
    return {"total": sum(counts.values()), **counts}


def performance_data(task: dict[str, Any]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for case, providers in task.get("records", {}).items():
        if "flagdnn" not in providers or len(providers) != 2:
            continue
        actual = providers["flagdnn"]
        reference = next(
            value for key, value in providers.items() if key != "flagdnn"
        )
        values = (actual.get("median"), reference.get("median"))
        if any(record.get("unit") != "us" for record in (actual, reference)):
            raise ValueError(f"unsupported latency unit for {case}")
        if not all(
            type(value) in (int, float) and math.isfinite(value) and value > 0
            for value in values
        ):
            raise ValueError(f"invalid latency for {case}")
        dtype = case_dtype(case)
        entry = data.setdefault(dtype, {"result": "OK", "details": {}})
        # The old parser accepts arbitrary string shape_detail values. Keep
        # the complete native case identity so layout/mode variants cannot
        # overwrite each other when their input dimensions are equal.
        entry["details"][case] = {
            "base": float(reference["median"]) / 1000.0,
            "flag_dnn": float(actual["median"]) / 1000.0,
            "gems": float(actual["median"]) / 1000.0,
            "speedup": float(reference["median"]) / float(actual["median"]),
        }
    for entry in data.values():
        details = entry["details"]
        total_speedup = 0.0
        for row in details.values():
            total_speedup += row["speedup"]
        entry["speedup"] = total_speedup / len(details)
    return data


def suite_result(task: dict[str, Any], suite: str) -> dict[str, Any]:
    status = task.get("status", "error")
    exit_code = task.get("exit_code")
    if status == "timeout":
        exit_code = -100
    elif status == "not_found":
        exit_code = 5
    elif exit_code is None:
        exit_code = 0 if status in {"passed", "skipped"} else 1
    result: dict[str, Any] = {
        "status": STATUS_NAMES.get(status, "Error"),
        "duration": float(task.get("duration_seconds", 0.0)),
        "exit_code": exit_code,
    }
    if suite == "functional":
        counts = task.get("case_counts", dict.fromkeys(COUNT_NAMES, 0))
        result.update({name: int(counts.get(name, 0)) for name in COUNT_NAMES})
        result["total"] = sum(result[name] for name in COUNT_NAMES)
    else:
        result["data"] = performance_data(task) if status == "passed" else {}
        if status not in {"timeout", "not_found"}:
            result["test_case"] = task.get("test_case", "Unknown")
        if status not in {"passed", "timeout", "not_found"}:
            result["reason"] = (
                "; ".join(task.get("record_errors", [])) or status
            )
    return result


def summary_document(summary: dict[str, Any]) -> dict[str, Any]:
    timestamp = summary.get("timestamp_utc")
    instant = (
        dt.datetime.fromisoformat(timestamp)
        if timestamp
        else dt.datetime.now().astimezone()
    )
    selected: dict[str, list[str]] = {}
    for suite, operators in summary.get("suite_operators", {}).items():
        for operator in operators:
            selected.setdefault(operator, []).append(SUITE_NAMES[suite])
    results = {
        operator: {
            # Like the old hasattr check, this tracks public operator
            # availability, not test success or device support.
            "implemented": True,
            **{
                SUITE_NAMES[suite]: suite_result(task, suite)
                for suite, task in suites.items()
            },
        }
        for operator, suites in summary.get("results", {}).items()
    }
    # psum_text indexes both suites, including for single-suite runs.
    # Keep selection metadata and case counts faithful to what was executed.
    for suites in results.values():
        for suite, name in SUITE_NAMES.items():
            if name not in suites:
                suites[name] = {
                    **suite_result({"status": "skipped"}, suite),
                    "reason": "Suite not selected",
                }
    environment = {**empty_environment(), **(summary.get("env") or {})}
    if not isinstance(environment.get("triton"), dict):
        environment["triton"] = {"version": "unknown"}
    # FlagGems' psum_text uses these aliases for the tested implementation.
    environment["flag_gems"] = dict(environment["flag_dnn"])
    return {
        "timestamp": instant.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
        "env": environment,
        "selected_suites": selected,
        "result": results,
    }


def performance_file(
    operator: str, task: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    dtype_names = {
        "fp16": "torch.float16",
        "fp32": "torch.float32",
        "bf16": "torch.bfloat16",
        "fp64": "torch.float64",
        "bool": "torch.bool",
        "int32": "torch.int32",
    }
    return {
        operator: {
            "result": task["status"],
            "test_case": result.get("test_case", "Unknown"),
            "reason": result.get("reason"),
            "details": [
                {
                    "dtype": dtype_names.get(dtype, dtype),
                    "result": [
                        {
                            "shape_detail": case,
                            "latency_base": row["base"],
                            "latency": row["flag_dnn"],
                            "speedup": row["speedup"],
                        }
                        for case, row in entry["details"].items()
                    ],
                }
                for dtype, entry in result["data"].items()
            ],
        }
    }


def empty_environment() -> dict[str, Any]:
    return {
        "architecture": platform.machine(),
        "os_name": platform.system(),
        "os_release": platform.release(),
        "python": platform.python_version(),
        "torch": {
            "version": "unknown",
            "cuda_available": False,
            "device_name": "N/A",
            "device_count": 0,
        },
        "flagtree": None,
        "triton": {"version": "unknown"},
        "flag_dnn": {
            "version": "unknown",
            "vendor": "unknown",
            "device": "unknown",
        },
    }


def probe_environment() -> dict[str, Any]:
    """Collect legacy env fields without requiring a Python FlagDNN."""
    result = empty_environment()
    try:
        result["flagtree"] = importlib.metadata.version("flagtree")
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        import torch

        result["torch"]["version"] = str(torch.__version__)
        result["torch"]["cuda_available"] = torch.cuda.is_available()
        result["torch"]["device_count"] = torch.cuda.device_count()
        if result["torch"]["cuda_available"]:
            result["torch"]["device_name"] = torch.cuda.get_device_name()
    except (ImportError, RuntimeError, AssertionError):
        pass
    try:
        import triton

        result["triton"] = {
            "version": str(triton.__version__),
            "has_config": hasattr(triton, "Config"),
        }
    except ImportError:
        pass
    return result


def report_environment(
    root: Path,
    build_dir: Path,
    vendor: str,
    environment: dict[str, str],
    device: str,
) -> dict[str, Any]:
    cache = build_dir / "CMakeCache.txt"
    text = cache.read_text() if cache.exists() else ""
    python = re.search(r"^FLAGDNN_CODEGEN_PYTHON:[^=]+=(.+)$", text, re.M)
    try:
        process = subprocess.run(
            [
                python[1] if python else sys.executable,
                str(Path(__file__)),
                "--report-environment",
            ],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        result = json.loads(process.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        result = empty_environment()
        result["python"] = "unknown"
    version = re.search(
        r"\bVERSION\s+(\d+\.\d+\.\d+)", (root / "CMakeLists.txt").read_text()
    )
    result["flag_dnn"] = {
        "version": version[1] if version else "unknown",
        "vendor": vendor,
        "device": device,
    }
    return result


def load_platform_adapter(platform: str) -> ModuleType | None:
    """Load optional test policy owned by backends/<platform>/validation."""
    if PLATFORM_PATTERN.fullmatch(platform) is None:
        raise ValueError("platform name must match [a-z][a-z0-9_]*")
    if platform in _PLATFORM_ADAPTERS:
        return _PLATFORM_ADAPTERS[platform]
    path = ROOT / "backends" / platform / "validation" / PLATFORM_ADAPTER_FILE
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
    for set_name in dict.fromkeys(
        re.findall(r"set\((FLAGDNN_[A-Z0-9_]+)\s", source)
    ):
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
        if (
            not configured
            and (ROOT / "build" / "CTestTestfile.cmake").is_file()
        ):
            requested = ROOT / "build"
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
        stat = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
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
            if process.poll() is not None and not _session_process_groups(
                session_id
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
    log_paths: tuple[Path, Path] | None = None,
) -> tuple[str, str, int | None, bool]:
    with contextlib.ExitStack() as stack:
        streams = (
            [
                stack.enter_context(path.open("w", encoding="utf-8"))
                for path in log_paths
            ]
            if log_paths is not None
            else [subprocess.PIPE, subprocess.PIPE]
        )

        def captured(
            stdout: str | None, stderr: str | None
        ) -> tuple[str, str]:
            if log_paths is None:
                return stdout or "", stderr or ""
            return (
                log_paths[0].read_text(encoding="utf-8", errors="replace"),
                log_paths[1].read_text(encoding="utf-8", errors="replace"),
            )

        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=streams[0],
            stderr=streams[1],
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
                return *captured(stdout, stderr), process.returncode, False
            except subprocess.TimeoutExpired:
                stdout, stderr = terminate()
                return *captured(stdout, stderr), None, True
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
    capture_output: bool = False,
    log_directory: Path | None = None,
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
    if log_directory is None:
        stdout, stderr, exit_code, timed_out = run_process_group(
            command, environment, timeout
        )
    else:
        log_directory.mkdir(parents=True, exist_ok=True)
        stdout, stderr, exit_code, timed_out = run_process_group(
            command,
            environment,
            timeout,
            (
                log_directory / f"{SUITE_NAMES[suite]}_stdout.log",
                log_directory / f"{SUITE_NAMES[suite]}_stderr.log",
            ),
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
    if suite == "functional":
        result["case_counts"] = functional_counts(combined_output, status)
        accounting = result.get("case_accounting", {})
        if accounting:
            counts = result["case_counts"]
            counts["skipped"] = accounting.get(
                "reference_skipped",
                accounting.get("skipped", counts["skipped"]),
            )
            counts["passed"] = accounting.get(
                "reference_executed",
                accounting.get("executed", counts["passed"]),
            )
            counts["total"] = sum(counts[name] for name in COUNT_NAMES)
    else:
        result["test_case"] = f"benchmark/test_{operator}.cpp"
    if capture_output:
        result.update(stdout=stdout, stderr=stderr)
        return result
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
        "core.kernel_registry_contract",
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


def publish_summary(output: Path, summary: dict[str, Any]) -> None:
    document = summary_document(summary)
    snapshot: Path | None = None
    published = False
    try:
        for operator, suites in document["result"].items():
            result = suites.get("performance")
            if result is None or not result["data"]:
                continue
            if snapshot is None:
                output.parent.mkdir(parents=True, exist_ok=True)
                snapshot = Path(
                    tempfile.mkdtemp(
                        dir=output.parent, prefix=f"{output.name}.data-"
                    )
                )
            relative = (
                Path(snapshot.name) / operator / "performance_result.json"
            )
            raw = performance_file(
                operator, summary["results"][operator]["benchmark"], result
            )
            atomic_write_summary(output.parent / relative, raw)
            result["data_file"] = relative.as_posix()
        # Commit only after all referenced files exist. Each publication owns
        # an immutable snapshot so earlier reports keep their original data.
        atomic_write_summary(output, document)
        published = True
        # Do not mark diagnostics complete before the public report commits.
        atomic_write_summary(diagnostic_path(output), summary)
    finally:
        if snapshot is not None and not published:
            shutil.rmtree(snapshot)


class LiveDisplay:
    """Manages terminal output with a pinned footer for GPU status lines."""

    def __init__(self, gpu_ids, op_count, op_width=20):
        self.gpu_ids = gpu_ids
        self.op_count = op_count
        self.op_width = op_width
        self.gpu_index = {gid: i + 1 for i, gid in enumerate(gpu_ids)}
        # Match the scrolling log width (55 + op_width visible characters).
        # Progress line: "[Progress] [" (12) + bar + "]  " (3) + nums_str
        nums_width = len(f"{op_count}/{op_count} ops")
        self.bar_width = max(20, 55 + op_width - 12 - 3 - nums_width)
        self.nums_width = nums_width
        progress_line = self._fmt_progress(0)
        gpu_lines = [f"{DIM}[GPU {gid:2d}] idle{NC}" for gid in gpu_ids]
        self.footer = [progress_line] + gpu_lines
        self.n = len(self.footer)
        self.footer_drawn = False

    def _fmt_progress(self, tests_done):
        total_tests = self.op_count * 2
        color = GREEN if tests_done >= total_tests else CYAN
        bar = (
            _progress_bar(tests_done, total_tests, self.bar_width, color=color)
            if self.op_count
            else " " * self.bar_width
        )
        ops_done = tests_done // 2
        nums = f"{ops_done}/{self.op_count} ops"
        return f"[Progress] [{color}{bar}{NC}]  {nums:>{self.nums_width}}"

    def _draw_footer(self):
        if not IS_TTY:
            return
        for line in self.footer:
            sys.stdout.write(line + "\n")
        sys.stdout.flush()
        self.footer_drawn = True

    def _erase_footer(self):
        if not IS_TTY or not self.footer_drawn:
            return
        for _ in range(self.n):
            sys.stdout.write("\033[A\033[2K")

    def init(self):
        if IS_TTY:
            self._draw_footer()

    def log(self, msg):
        """Print a scrolling log line above the footer."""
        if IS_TTY:
            self._erase_footer()
            sys.stdout.write(msg + "\n")
            self._draw_footer()
        else:
            sys.stdout.write(msg + "\n")
            sys.stdout.flush()

    def update_gpu(self, gpu_id, status_line):
        """Update a GPU's footer line."""
        idx = self.gpu_index.get(gpu_id)
        if idx is None:
            return
        self.footer[idx] = status_line
        if IS_TTY:
            self._erase_footer()
            self._draw_footer()

    def update_progress(self, tests_done):
        """Update the global progress bar."""
        self.footer[0] = self._fmt_progress(tests_done)
        if IS_TTY:
            self._erase_footer()
            self._draw_footer()
        else:
            sys.stdout.write(self.footer[0] + "\n")
            sys.stdout.flush()

    def finish(self):
        """Clear the footer when done."""
        if IS_TTY:
            self._erase_footer()
            sys.stdout.flush()


def _progress_bar(done, total, width=40, color=""):
    if not total:
        return " " * width
    frac = done * width / total
    full = int(frac)
    has_half = (frac - full) >= 0.5 and full < width
    empty = width - full - (1 if has_half else 0)
    bar = "█" * full
    if has_half:
        bar += f"{DIM}█{NC}{color}"
    bar += " " * empty
    return bar


def _format_status(status, dur):
    STATUS_MAP = {
        "Passed": (GREEN, "OK"),
        "Failed": (RED, "FAILED"),
        "Timeout": (RED, "TIMEOUT"),
        "Error": (RED, "ERROR"),
        "NotFound": (YELLOW, "NOTFOUND"),
        "Skipped": (YELLOW, "SKIPPED"),
    }
    color, label = STATUS_MAP.get(status, (YELLOW, status.upper()))
    return f"{color}[{label:<8} {dur:>6.1f}s]{NC}"


# The console and public report protocol match FlagBLAS tools/run_tests.py.
IS_TTY = sys.stdout.isatty()
RED = GREEN = YELLOW = CYAN = DIM = NC = ""


def configure_colors(mode: str) -> None:
    global IS_TTY, RED, GREEN, YELLOW, CYAN, DIM, NC
    IS_TTY = sys.stdout.isatty()
    if mode == "always" or (mode == "auto" and IS_TTY):
        RED, GREEN, YELLOW, CYAN, DIM, NC = (
            "\033[31m",
            "\033[32m",
            "\033[93m",
            "\033[36m",
            "\033[2m",
            "\033[0m",
        )
    else:
        RED = GREEN = YELLOW = CYAN = DIM = NC = ""


def detect_platform(build_dir: Path | None) -> str:
    """Prefer configured build metadata over assumptions about CUDA vendors."""
    explicit = os.environ.get("FLAGDNN_BENCHMARK_PLATFORM")
    if explicit:
        return explicit
    configured = build_dir or os.environ.get("FLAGDNN_BUILD_DIR")
    candidates = [Path(configured)] if configured else [ROOT / "build"]
    if configured is None:
        candidates.extend(sorted((ROOT / "build").glob("*")))
    found: set[str] = set()
    for directory in candidates:
        if not directory.is_absolute():
            directory = ROOT / directory
        cache = directory / "CMakeCache.txt"
        if not cache.is_file():
            continue
        values = dict(
            re.findall(
                r"^(FLAGDNN_BACKENDS|FLAGDNN_DEFAULT_BACKEND):[^=]+=(.*)$",
                cache.read_text(),
                re.M,
            )
        )
        default = values.get("FLAGDNN_DEFAULT_BACKEND", "auto")
        backends = values.get("FLAGDNN_BACKENDS", "").split(";")
        if default != "auto":
            found.add(default)
        else:
            found.update(backend for backend in backends if backend)
    if len(found) > 1:
        raise ValueError("multiple configured backends; specify --platform")
    return next(iter(found), "nvidia")


def gpu_ids(value: str) -> list[int]:
    parts = value.split(",")
    if not parts or any(
        not re.fullmatch(r"\d+", part.strip()) for part in parts
    ):
        raise ValueError(
            "--gpus must be a comma-separated list of nonnegative GPU IDs"
        )
    result = [int(part) for part in parts]
    if len(set(result)) != len(result):
        raise ValueError("--gpus must not contain duplicate GPU IDs")
    return result


def accuracy_records(operator: str, task: dict[str, Any]) -> dict[str, Any]:
    """Translate native cases while retaining CTest sub-suite identities."""
    records: dict[str, Any] = {}
    for raw in (
        task.get("stdout", "") + "\n" + task.get("stderr", "")
    ).splitlines():
        prefix = re.match(r"^\s*(\d+):\s?(.*)$", raw)
        if prefix is None:
            continue
        group, line = prefix.groups()
        if re.search(r"\b(?:PASS|SKIP)\s+cases=", line):
            continue
        match = re.match(r"^(\S+): .*\b(PASS|FAIL)\b", line)
        skip = re.search(
            r"(?:^SKIP |^\[SKIP\].*?\s)case=(\S+).*?reason=(\S+)", line
        )
        if match:
            case, outcome = match.groups()
            status = "passed" if outcome == "PASS" else "failed"
            reason = line if status == "failed" else None
        elif skip:
            case, reason = skip.groups()
            status = "skipped"
        else:
            continue
        key = f"tests/test_{operator}.cpp::ctest_{group}[{case}]"
        records[key] = {
            "params": {"case": case},
            "result": status,
            "opname": [operator],
            "skipped_reason": reason if status == "skipped" else None,
        }
        if reason is not None:
            records[key]["reason"] = reason
    # Some native suites only print aggregate accounting. Expose explicitly
    # unnamed cases rather than silently losing their reported case counts.
    counts = task.get("case_counts", {})
    for status in ("passed", "skipped", "failed"):
        observed = sum(item["result"] == status for item in records.values())
        expected = counts.get(status, 0)
        for index in range(observed, expected):
            key = (
                f"tests/test_{operator}.cpp::native_accounting"
                f"[{status}_{index + 1}]"
            )
            records[key] = {
                "params": {"native_case_index": index + 1},
                "result": status,
                "opname": [operator],
                "skipped_reason": None,
            }
    return records


def batch_suite_result(
    operator: str, suite: str, task: dict[str, Any], output: Path
) -> dict[str, Any]:
    directory = output / operator
    directory.mkdir(parents=True, exist_ok=True)
    status = task["status"]
    result = {
        "status": STATUS_NAMES.get(status, "Error"),
        "exit_code": -100 if status == "timeout" else task.get("exit_code", 1),
        "duration": task.get("duration_seconds", 0.0),
    }
    if suite == "functional":
        records = accuracy_records(operator, task)
        counts = {
            name: sum(item["result"] == name for item in records.values())
            for name in ("passed", "failed", "skipped")
        }
        details: dict[str, Any] = {}
        outcome = "failed" if counts["failed"] else "skipped"
        for case, item in records.items():
            if item["result"] != outcome:
                continue
            # Match FlagBLAS parse_accuracy_data's case:parameter identity.
            parameters = [case.split("[", 1)[0]] + [
                str(value).replace(" ", "")
                for value in item["params"].values()
            ]
            group = details.setdefault(outcome, {}).setdefault(
                item.get("reason", "Unknown"), []
            )
            identity = ":".join(parameters)
            if identity not in group:
                group.append(identity)
        result.update(total=sum(counts.values()), **counts, details=details)
        if status == "passed":
            result["status"] = (
                "Failed"
                if counts["failed"]
                else (
                    "Skipped"
                    if counts["skipped"]
                    else "Passed" if counts["passed"] else "NotFound"
                )
            )
        if status in {"failed", "error"} and not counts["failed"]:
            # A crashed executable or invalid native accounting is an execution
            # error, not an invented failed parameter case.
            result["errors"] = 1
            result["details"]["error"] = (
                "; ".join(task.get("record_errors", [])) or status
            )
        relative = Path(operator) / "accuracy_result.json"
        atomic_write_summary(output / relative, records)
    else:
        dtype_aliases = {
            "f8-e4m3fn": "float8_e4m3fn",
            "f8-e5m2": "float8_e5m2",
            "fp64": "torch.float64",
        }
        data = {
            dtype_aliases.get(dtype, dtype): entry
            for dtype, entry in performance_data(task).items()
        }
        for entry in data.values():
            for row in entry["details"].values():
                row.pop("flag_dnn", None)
        result["data"] = data
        relative = Path(operator) / "performance_result.log"
        dtype_names = {
            "fp16": "torch.float16",
            "fp32": "torch.float32",
            "bf16": "torch.bfloat16",
            "torch.float64": "torch.float64",
            "float8_e4m3fn": "torch.float8_e4m3fn",
            "float8_e5m2": "torch.float8_e5m2",
            **{
                name: "torch." + name
                for name in (
                    "int8",
                    "int16",
                    "int32",
                    "int64",
                    "uint8",
                    "bool",
                )
            },
        }
        with (output / relative).open("w", encoding="utf-8") as stream:
            stream.write("[INFO] Benchmark record logger enabled\n")
            for dtype, entry in data.items():
                rows = []
                for case, row in entry["details"].items():
                    rows.append(
                        {
                            "legacy_shape": None,
                            "shape_detail": case,
                            "latency_base": row["base"],
                            "latency": row["gems"],
                            "gbps_base": None,
                            "gbps": None,
                            "speedup": row["speedup"],
                            "accuracy": None,
                            "tflops": None,
                            "utilization": None,
                            "compared_speedup": None,
                            "error_msg": None,
                        }
                    )
                stream.write(
                    "[INFO] "
                    + json.dumps(
                        {
                            "op_name": operator,
                            "dtype": dtype_names.get(dtype, dtype),
                            "mode": "kernel",
                            "level": "core",
                            "result": rows,
                        }
                    )
                    + "\n"
                )
        if data:
            result["test_case"] = operator
    if status == "timeout":
        if suite == "functional":
            return {
                "status": "Timeout",
                "exit_code": -100,
                "duration": result["duration"],
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 0,
            }
        return {
            "status": "Timeout",
            "exit_code": -100,
            "duration": result["duration"],
            "data": {},
        }
    result["data_file"] = relative.as_posix()
    return result


def batch_environment(environment: dict[str, Any]) -> dict[str, Any]:
    result = dict(environment)
    result.pop("flag_gems", None)
    try:
        release = platform.freedesktop_os_release()
        result["os_name"] = release.get("ID", result["os_name"])
        result["os_release"] = release.get("VERSION_ID", result["os_release"])
    except (OSError, AttributeError):
        pass
    return result


def write_batch_summary(
    output: Path, environment: dict[str, Any], gpu_list: list[int]
) -> None:
    results = {}
    for gpu in gpu_list:
        path = output / f"summary{gpu}.json"
        if path.exists():
            results.update(json.loads(path.read_text()))
    atomic_write_summary(
        output / "summary.json",
        {
            "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "env": batch_environment(environment),
            "result": results,
        },
    )


def batch_worker(
    gpu: int, work_queue: Any, events: Any, settings: dict[str, Any]
) -> None:
    # Each process runs both phases on its GPU before taking another op.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    result = {}
    output = settings["output"]
    adapter = load_platform_adapter(settings["platform"])
    environment = device_environment(
        settings["platform"], str(gpu), settings["environment"], adapter
    )
    try:
        with contextlib.ExitStack() as stack:
            sink = stack.enter_context(open(os.devnull, "w"))
            stack.enter_context(contextlib.redirect_stdout(sink))
            stack.enter_context(contextlib.redirect_stderr(sink))
            while True:
                operator = work_queue.get()
                if operator is None:
                    break
                public: dict[str, Any] = {"customized": adapter is not None}
                native = {}
                for suite in VALID_SUITES:
                    phase = (
                        "accuracy" if suite == "functional" else "benchmark"
                    )
                    if operator not in settings["suite_operators"].get(
                        suite, []
                    ):
                        task = {"status": "skipped", "exit_code": 0}
                    else:
                        events.put(("start", gpu, phase, operator))
                        try:
                            task = run_one(
                                build_dir=settings["build_dir"],
                                operator=operator,
                                suite=suite,
                                platform=settings["platform"],
                                environment=environment,
                                timeout=settings["timeout"],
                                verbose=False,
                                manifest_operators=settings[
                                    "manifest_operators"
                                ],
                                configuration=settings["configuration"],
                                adapter=adapter,
                                capture_output=True,
                                log_directory=(
                                    output / operator
                                    if settings["dump_output"]
                                    else None
                                ),
                            )
                        except Exception as error:
                            task = {
                                "status": "error",
                                "exit_code": 1,
                                "stderr": str(error),
                            }
                    directory = output / operator
                    directory.mkdir(parents=True, exist_ok=True)
                    if settings["dump_output"]:
                        for channel in ("stdout", "stderr"):
                            (
                                directory
                                / f"{SUITE_NAMES[suite]}_{channel}.log"
                            ).write_text(
                                task.get(channel, ""), encoding="utf-8"
                            )
                    public[SUITE_NAMES[suite]] = batch_suite_result(
                        operator, suite, task, output
                    )
                    task.pop("stdout", None)
                    task.pop("stderr", None)
                    native[suite] = task
                    events.put(
                        (
                            "done",
                            gpu,
                            phase,
                            operator,
                            public[SUITE_NAMES[suite]]["status"],
                            task.get("duration_seconds", 0.0),
                        )
                    )
                result[operator] = public
                atomic_write_summary(output / f"summary{gpu}.json", result)
                events.put(("result", gpu, operator, native))
    except BaseException as error:
        events.put(("error", gpu, str(error)))
        raise
    finally:
        events.put(("exit", gpu))


def isolated_ctest_directory(
    destination: Path, tests: list[dict[str, Any]]
) -> None:
    """Keep concurrent CTest runs from sharing Testing/Temporary files."""

    def quote(value: Any) -> str:
        if isinstance(value, list):
            value = ";".join(str(item).replace(";", r"\;") for item in value)
        elif isinstance(value, bool):
            value = "TRUE" if value else "FALSE"
        value = str(value)
        delimiter = "="
        while "]" + delimiter + "]" in value:
            delimiter += "="
        return "[" + delimiter + "[" + value + "]" + delimiter + "]"

    lines = []
    for test in tests:
        name = quote(test["name"])
        command = test.get("command", ["flagdnn-missing-test-executable"])
        lines.append(
            f"add_test({name} {' '.join(quote(arg) for arg in command)})"
        )
        for prop in test.get("properties", []):
            lines.append(
                f"set_tests_properties({name} PROPERTIES "
                f"{prop['name']} {quote(prop['value'])})"
            )
    destination.mkdir(parents=True)
    (destination / "CTestTestfile.cmake").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_batch(
    settings: dict[str, Any], operators: list[str], gpu_list: list[int]
) -> tuple[dict[str, Any], bool]:
    # Inherit backend policies without importing GPU packages in each worker.
    # Native test processes still get isolated sessions.
    command = [
        "ctest",
        "--test-dir",
        str(settings["build_dir"]),
        "--show-only=json-v1",
    ]
    if settings["configuration"]:
        command.extend(["-C", settings["configuration"]])
    stdout, stderr, code, timed_out = run_process_group(
        command, settings["environment"], min(settings["timeout"], 60)
    )
    if code != 0 or timed_out:
        raise RuntimeError("Cannot discover CTest inventory: " + stderr)
    tests = json.loads(stdout)["tests"]
    temporary = tempfile.TemporaryDirectory(prefix="flagdnn-ctest-workers-")
    context = mp.get_context("fork")
    work_queue, events = context.Queue(), context.Queue()
    for operator in operators:
        work_queue.put(operator)
    for _ in gpu_list:
        work_queue.put(None)
    workers = []
    for gpu in gpu_list:
        directory = Path(temporary.name) / str(gpu)
        isolated_ctest_directory(directory, tests)
        workers.append(
            context.Process(
                target=batch_worker,
                args=(
                    gpu,
                    work_queue,
                    events,
                    {**settings, "build_dir": directory},
                ),
            )
        )
    display = LiveDisplay(
        gpu_list, len(operators), op_width=min(max(map(len, operators)), 40)
    )
    results: dict[str, Any] = {}
    exited: set[int] = set()
    per_gpu_done = dict.fromkeys(gpu_list, 0)
    tests_done = 0
    failed = False
    try:
        for worker in workers:
            worker.start()
        display.init()
        while len(exited) < len(workers):
            try:
                message = events.get(timeout=0.2)
            except queue.Empty:
                for gpu, worker in zip(gpu_list, workers):
                    if worker.exitcode is not None and gpu not in exited:
                        exited.add(gpu)
                        failed = failed or worker.exitcode != 0
                continue
            kind, gpu, *payload = message
            if kind == "exit":
                exited.add(gpu)
                display.update_gpu(
                    gpu,
                    f"{DIM}[GPU {gpu:2d}] done ({per_gpu_done[gpu]} ops){NC}",
                )
            elif kind == "error":
                failed = True
                display.log(f"{RED}[ERROR]{NC} GPU {gpu}: {payload[0]}")
            elif kind == "result":
                operator, native = payload
                results[operator] = native
            else:
                phase, operator, *outcome = payload
                label = "accuracy " if phase == "accuracy" else "benchmark"
                op_column = (
                    operator
                    if len(operator) <= display.op_width
                    else operator[: display.op_width - 3] + "..."
                ).ljust(display.op_width)
                timestamp = dt.datetime.now().strftime("%H:%M:%S")
                if kind == "start":
                    if IS_TTY:
                        display.update_gpu(
                            gpu,
                            f"[GPU {gpu:2d}] ({per_gpu_done[gpu]:>3} done)  "
                            f"{label} {op_column}",
                        )
                    else:
                        display.log(
                            f"[INFO] [{timestamp}][GPU {gpu:2d}] "
                            f"{label} {op_column} ..."
                        )
                elif kind == "done":
                    status, duration = outcome
                    tests_done += 1
                    if phase == "benchmark":
                        per_gpu_done[gpu] += 1
                    line = (
                        f"{GREEN}[INFO]{NC} [{timestamp}][GPU {gpu:2d}] "
                        f"{label} {op_column} "
                        f"{_format_status(status, duration)}"
                    )
                    if not IS_TTY:
                        done = tests_done // 2
                        width = len(str(len(operators)))
                        line += (
                            f"  ({done * 100 // len(operators):>3}% "
                            f"{done:>{width}}/{len(operators)} ops)"
                        )
                    display.footer[0] = display._fmt_progress(tests_done)
                    display.log(line)
        for worker in workers:
            worker.join()
            failed = failed or worker.exitcode != 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            if worker.pid is not None:
                worker.join(
                    timeout=PROCESS_TERMINATION_GRACE_SECONDS
                    + PROCESS_KILL_GRACE_SECONDS
                    + 1
                )
                if worker.is_alive():
                    worker.kill()
                    worker.join()
        display.finish()
        work_queue.cancel_join_thread()
        work_queue.close()
        events.close()
        temporary.cleanup()
    return results, failed or set(results) != set(operators)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=("Run FlagDNN accuracy and performance tests across GPUs"),
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
        default=None,
        help=(
            "functional, benchmark, comma-separated values, or all "
            "(default: all)"
        ),
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
        default=None,
        help="CTest backend (default: infer from the configured build)",
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
            "(default: 4800; legacy --output uses platform policy)"
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
        "--output",
        type=Path,
        help=(
            "legacy-compatible JSON summary path "
            "(also writes <stem>.native.json diagnostics)"
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list every manifest operator and exit",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--gpus", help="comma-separated GPU IDs (default: 0)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="test data directory (default: results)",
    )
    parser.add_argument(
        "--dump-output",
        action="store_true",
        help="dump stdout/stderr of each test to log files",
    )
    parser.add_argument(
        "--color", choices=("auto", "always", "never"), default="auto"
    )
    parser.add_argument("--start", help="the ID of the first default operator")
    parser.add_argument(
        "--stages",
        default="stable",
        help="operator stages (the native catalog is stable)",
    )
    arguments = parser.parse_args()
    arguments.batch = arguments.output is None
    if not arguments.batch and (
        arguments.output_dir is not None
        or arguments.gpus is not None
        or arguments.dump_output
    ):
        parser.error(
            "--output cannot be combined with --output-dir, --gpus "
            "or --dump-output"
        )
    if arguments.device is not None and arguments.gpus is not None:
        parser.error("--device and --gpus are mutually exclusive")
    try:
        arguments.platform = arguments.platform or (
            detect_platform(arguments.build_dir)
            if arguments.batch
            else os.environ.get("FLAGDNN_BENCHMARK_PLATFORM", "nvidia")
        )
        arguments.gpu_ids = (
            gpu_ids(
                arguments.gpus
                if arguments.gpus is not None
                else arguments.device or "0"
            )
            if arguments.batch
            else []
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    arguments.suites = arguments.suites or (
        "all" if arguments.batch else "functional"
    )
    if arguments.batch:
        arguments.output_dir = (
            (arguments.output_dir or Path("results")).expanduser().resolve()
        )
        if arguments.timeout is None:
            arguments.timeout = 4800
    configure_colors(arguments.color)
    return arguments


def _run_main() -> int:
    preliminary_output = preliminary_output_argument(sys.argv[1:])
    output: Path | None = None
    if preliminary_output is not None:
        try:
            output = preliminary_output.expanduser().resolve()
            publish_summary(
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
                publish_summary(
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
        publish_summary(
            output,
            summary_envelope(arguments, "running", None),
        )

    def finish_early(
        exit_code: int,
        overall_status: str,
        **fields: Any,
    ) -> int:
        if output is not None:
            summary = summary_envelope(arguments, overall_status, exit_code)
            summary.update(fields)
            try:
                publish_summary(output, summary)
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

    if arguments.batch and not arguments.list:
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        # Invalidate summaries before running anything; never merge stale GPUs.
        for stale in arguments.output_dir.glob("summary[0-9]*.json"):
            stale.unlink()
        for gpu in arguments.gpu_ids:
            atomic_write_summary(
                arguments.output_dir / f"summary{gpu}.json", {}
            )
        write_batch_summary(
            arguments.output_dir, empty_environment(), arguments.gpu_ids
        )

    if PLATFORM_PATTERN.fullmatch(arguments.platform) is None:
        return validation_error("--platform must match [a-z][a-z0-9_]*")
    try:
        adapter = load_platform_adapter(arguments.platform)
        manifests = operator_manifests()
        select_manifests = (
            None
            if adapter is None
            else getattr(adapter, "select_operator_manifests", None)
        )
        if select_manifests is not None:
            manifests = select_manifests(manifests)
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
            str(arguments.gpu_ids[0]) if arguments.batch else arguments.device,
            adapter=adapter,
        )
        configure_batch = (
            None
            if adapter is None
            else getattr(adapter, "configure_batch_environment", None)
        )
        if arguments.batch and configure_batch is not None:
            configure_batch(environment, build_dir)
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
    if (
        arguments.start
        and arguments.ops is None
        and arguments.op_list_file is None
    ):
        suite_operators = {
            suite: [op for op in selected if op >= arguments.start]
            for suite, selected in suite_operators.items()
        }
    if arguments.stages != "stable":
        stages = {stage.strip() for stage in arguments.stages.split(",")}
        if not stages <= {"alpha", "beta", "stable", "all", "removed"}:
            return validation_error("unsupported --stages value")
        if (
            not stages.intersection({"stable", "all"})
            and arguments.ops is None
            and arguments.op_list_file is None
        ):
            return validation_error(
                "the native operator catalog only contains stable operators"
            )
    if arguments.min_speedup is not None and (
        not math.isfinite(arguments.min_speedup)
        or arguments.min_speedup <= 0.0
    ):
        return validation_error("--min-speedup must be finite and positive")
    if arguments.min_speedup is not None and "benchmark" not in suites:
        return validation_error("--min-speedup requires the benchmark suite")
    supports_min_speedup = bool(
        adapter is not None and getattr(adapter, "SUPPORTS_MIN_SPEEDUP", False)
    )
    if arguments.min_speedup is not None and not supports_min_speedup:
        return validation_error(
            f"--min-speedup is not supported by platform "
            f"{arguments.platform}"
        )

    prepare = None if adapter is None else getattr(adapter, "prepare", None)
    try:
        platform_state = {} if prepare is None else prepare(manifests, suites)
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
        not arguments.batch
        and adapter is not None
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
    batch_env = {}
    if arguments.batch:
        batch_env = report_environment(
            ROOT,
            build_dir,
            arguments.platform,
            device_environment(arguments.platform, None, adapter=adapter),
            getattr(adapter, "REPORT_DEVICE", arguments.platform),
        )
        print(
            f"{GREEN}[INFO]{NC} Testing {len(operators)} operators ...",
            flush=True,
        )
        # Remove this run's old artifacts, including optional logs left by a
        # previous --dump-output invocation.
        for operator in operators:
            directory = arguments.output_dir / operator
            directory.mkdir(parents=True, exist_ok=True)
            for name in (
                "accuracy_result.json",
                "performance_result.log",
                "accuracy_stdout.log",
                "accuracy_stderr.log",
                "performance_stdout.log",
                "performance_stderr.log",
            ):
                (directory / name).unlink(missing_ok=True)
        try:
            if not failed:
                results, worker_failed = run_batch(
                    {
                        "output": arguments.output_dir,
                        "platform": arguments.platform,
                        "environment": environment,
                        "build_dir": build_dir,
                        "suite_operators": suite_operators,
                        "timeout": timeout,
                        "manifest_operators": manifest_operators,
                        "configuration": build_configuration,
                        "dump_output": arguments.dump_output,
                    },
                    operators,
                    arguments.gpu_ids,
                )
                failed = worker_failed or any(
                    not status_is_success(task["status"])
                    for tasks in results.values()
                    for suite, task in tasks.items()
                    if suite in suites
                )
        except BaseException:
            finalize = (
                None if adapter is None else getattr(adapter, "finalize", None)
            )
            if finalize is not None:
                finalize(
                    results=results,
                    suite_operators=suite_operators,
                    suites=suites,
                    state=platform_state,
                    min_speedup=arguments.min_speedup,
                    preflight_passed=False,
                )
            raise
        finally:
            write_batch_summary(
                arguments.output_dir, batch_env, arguments.gpu_ids
            )
    else:
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
        "env": (
            report_environment(
                ROOT,
                build_dir,
                arguments.platform,
                environment,
                getattr(adapter, "REPORT_DEVICE", arguments.platform),
            )
            if output is not None
            else {}
        ),
        **platform_summary,
    }
    if output is not None:
        try:
            publish_summary(output, summary)
        except OSError as error:
            print(
                f"error: cannot finalize --output summary: {error}",
                file=sys.stderr,
            )
            return 2
        print(f"summary: {output}")

    if arguments.batch:
        print(f"{GREEN}[INFO]{NC} Test completed.", flush=True)
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
    except KeyboardInterrupt:
        print("[WARN] Interrupted. Cleanup done.", file=sys.stderr)
        return 130
    except _TerminationSignal as termination:
        exit_code = 128 + termination.signum
        requested_output = preliminary_output_argument(sys.argv[1:])
        if requested_output is not None:
            try:
                publish_summary(
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
    if sys.argv[1:] == ["--report-environment"]:
        print(json.dumps(probe_environment()))
    else:
        raise SystemExit(main())
