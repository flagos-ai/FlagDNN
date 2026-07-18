#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
"""
Run FlagDNN operator tests from conf/operators.yaml.

The runner discovers operator ids from the inventory, checks which pytest marks
exist in the test directories, and then runs only the suites declared by
operator labels.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import queue as queue_module
import re
import shlex
import signal
import subprocess
import sys
import time
import types
from importlib import metadata
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import flag_dnn  # noqa: E402

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[93m"
CYAN = "\033[36m"
DIM = "\033[2m"
NC = "\033[0m"

OPTS = argparse.Namespace()
CFG = types.SimpleNamespace()
ENV_INFO: dict[str, Any] = {}
MARKS_BY_SUITE: dict[str, set[str]] = {}

TIMEOUT = -100
WORKER_PROCESSES: list[Process] = []
INTERRUPTED = False

IS_TTY = sys.stdout.isatty()
USE_COLORS = IS_TTY

if not USE_COLORS:
    RED = GREEN = YELLOW = CYAN = DIM = NC = ""

SUITES = {
    "accuracy": {
        "label": "accuracy",
        "directory": "tests",
        "kind": "accuracy",
    },
    "benchmark": {
        "label": "benchmark",
        "directory": "benchmark",
        "kind": "benchmark",
    },
}

BUILTIN_MARKS = {
    "parametrize",
    "skip",
    "skipif",
    "xfail",
    "usefixtures",
    "filterwarnings",
    "timeout",
    "tryfirst",
    "trylast",
    "graph",
    "perf",
    "cudnn_legacy",
}

DTYPE_MAP = {
    "torch.float16": "fp16",
    "torch.bfloat16": "bf16",
    "torch.float32": "fp32",
    "torch.float64": "fp64",
    "torch.int16": "int16",
    "torch.int32": "int32",
    "torch.int64": "int64",
    "torch.bool": "bool",
}


def pinfo(msg: str, **kwargs: Any) -> None:
    print(f"{GREEN}[INFO]{NC} {msg}", flush=True, **kwargs)


def perror(msg: str, **kwargs: Any) -> None:
    print(f"{RED}[ERROR]{NC} {msg}", flush=True, **kwargs)


def pwarn(msg: str, **kwargs: Any) -> None:
    print(f"{YELLOW}[WARN]{NC} {msg}", flush=True, **kwargs)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class LiveDisplay:
    """Terminal output with a pinned progress footer when stdout is a TTY."""

    def __init__(
        self, gpu_ids: list[int], task_count: int, op_width: int = 20
    ):
        self.gpu_ids = gpu_ids
        self.task_count = task_count
        self.op_width = op_width
        self.gpu_index = {gid: i + 1 for i, gid in enumerate(gpu_ids)}
        nums_width = len(f"{task_count}/{task_count} tasks")
        self.bar_width = max(20, 55 + op_width - 12 - 3 - nums_width)
        self.nums_width = nums_width
        progress_line = self._fmt_progress(0)
        gpu_lines = [f"{DIM}[GPU {gid:2d}] idle{NC}" for gid in gpu_ids]
        self.footer = [progress_line] + gpu_lines
        self.n = len(self.footer)
        self.footer_drawn = False

    def _fmt_progress(self, tasks_done: int) -> str:
        color = GREEN if tasks_done >= self.task_count else CYAN
        bar = _progress_bar(tasks_done, self.task_count, self.bar_width)
        nums = f"{tasks_done}/{self.task_count} tasks"
        return f"[Progress] [{color}{bar}{NC}]  {nums:>{self.nums_width}}"

    def _draw_footer(self) -> None:
        if not IS_TTY:
            return
        for line in self.footer:
            sys.stdout.write(line + "\n")
        sys.stdout.flush()
        self.footer_drawn = True

    def _erase_footer(self) -> None:
        if not IS_TTY or not self.footer_drawn:
            return
        for _ in range(self.n):
            sys.stdout.write("\033[A\033[2K")

    def init(self) -> None:
        if IS_TTY:
            self._draw_footer()

    def log(self, msg: str) -> None:
        if IS_TTY:
            self._erase_footer()
            sys.stdout.write(msg + "\n")
            self._draw_footer()
        else:
            sys.stdout.write(msg + "\n")
            sys.stdout.flush()

    def update_gpu(self, gpu_id: int, status_line: str) -> None:
        idx = self.gpu_index.get(gpu_id)
        if idx is None:
            return
        self.footer[idx] = status_line
        if IS_TTY:
            self._erase_footer()
            self._draw_footer()

    def set_progress(self, tasks_done: int) -> None:
        self.footer[0] = self._fmt_progress(tasks_done)

    def finish(self) -> None:
        if IS_TTY:
            self._erase_footer()
            sys.stdout.flush()


def _progress_bar(done: int, total: int, width: int = 40) -> str:
    if not total:
        return " " * width
    full = int(done * width / total)
    return "#" * full + "-" * (width - full)


def _format_status(status: str, dur: float) -> str:
    status_map = {
        "Passed": (GREEN, "OK"),
        "Failed": (RED, "FAILED"),
        "Timeout": (RED, "TIMEOUT"),
        "Error": (RED, "ERROR"),
        "NotFound": (YELLOW, "NOTFOUND"),
        "Skipped": (YELLOW, "SKIPPED"),
    }
    color, label = status_map.get(status, (YELLOW, status.upper()))
    return f"{color}[{label:<8} {dur:>6.1f}s]{NC}"


def get_ops_from_inventory() -> list[dict[str, Any]]:
    op_inventory = ROOT / "conf" / "operators.yaml"
    try:
        with op_inventory.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("ops", [])
    except Exception as exc:
        perror(f"Failed to load operator inventory: {exc}")
        return []


def discover_marks(directory: Path) -> set[str]:
    marks: set[str] = set()
    pattern = re.compile(r"pytest\.mark\.([A-Za-z_][A-Za-z0-9_]*)")
    for path in directory.glob("test*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for mark in pattern.findall(text):
            if mark not in BUILTIN_MARKS:
                marks.add(mark)
    return marks


def discover_all_marks() -> dict[str, set[str]]:
    return {
        name: discover_marks(ROOT / spec["directory"])
        for name, spec in SUITES.items()
    }


def _probe_torch() -> None:
    ENV_INFO.setdefault("torch", {})
    try:
        import torch

        ENV_INFO["torch"]["version"] = torch.__version__
        pinfo(f"PyTorch detected ... {torch.__version__}")
    except Exception as exc:
        perror(f"PyTorch not installed, please fix it - {exc}")
        sys.exit(1)

    try:
        cuda_available = torch.cuda.is_available()
        ENV_INFO["torch"]["cuda_available"] = cuda_available
        pinfo(f"PyTorch CUDA support ... {cuda_available}")
    except Exception:
        ENV_INFO["torch"]["cuda_available"] = False

    try:
        dev_name = torch.cuda.get_device_name()
        ENV_INFO["torch"]["device_name"] = dev_name
        pinfo(f"PyTorch device name ... {dev_name}")
    except Exception:
        ENV_INFO["torch"]["device_name"] = "N/A"

    try:
        dev_count = torch.cuda.device_count()
        ENV_INFO["torch"]["device_count"] = dev_count
        pinfo(f"PyTorch device count ... {dev_count}")
    except Exception:
        ENV_INFO["torch"]["device_count"] = 0


def _probe_triton() -> None:
    try:
        version = metadata.version("flagtree")
        ENV_INFO["flagtree"] = version
        pinfo(f"FlagTree (flagtree) detected ... {version}")
        has_flagtree = True
    except Exception:
        ENV_INFO["flagtree"] = None
        has_flagtree = False
        pwarn("FlagTree (flagtree) not installed, testing Triton ...")

    try:
        import triton

        ENV_INFO["triton"] = {"version": triton.__version__}
        pinfo(f"Triton (triton) detected ... {triton.__version__}")
    except Exception:
        ENV_INFO["triton"] = None
        if not has_flagtree:
            perror("Neither FlagTree nor Triton is installed, please fix it.")
            sys.exit(1)


def _git_sha() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        return proc.stdout.strip()
    except Exception:
        return None


def _probe_flagdnn() -> None:
    try:
        version = getattr(flag_dnn, "__version__", "unknown")
        sha = _git_sha()
        ver_str = f"{version}+git{sha[:8]}" if sha else str(version)
        ENV_INFO["flag_dnn"] = {"version": ver_str}
        pinfo(f"flag_dnn detected ... {ver_str}")
    except Exception as exc:
        perror(f"{exc}")
        perror(
            "flag_dnn has not been installed, please run `pip install -e .`"
        )
        sys.exit(1)

    try:
        vendor = flag_dnn.vendor_name
        ENV_INFO["flag_dnn"]["vendor"] = vendor
        pinfo(f"flag_dnn vendor detection ... {vendor}")
    except Exception as exc:
        perror(f"flag_dnn failed to detect vendor info: {exc}")
        sys.exit(1)

    try:
        device = flag_dnn.device
        ENV_INFO["flag_dnn"]["device"] = device
        pinfo(f"flag_dnn device detection ... {device}")
    except Exception as exc:
        perror(f"flag_dnn failed to detect device info: {exc}")
        sys.exit(1)


def probe_env() -> None:
    ENV_INFO["architecture"] = platform.machine()
    ENV_INFO["os_name"] = platform.system()
    ENV_INFO["os_release"] = platform.release()
    ENV_INFO["python"] = platform.python_version()

    _probe_torch()
    _probe_triton()
    _probe_flagdnn()


def get_env(gpu_ids: str) -> dict[str, str]:
    env = os.environ.copy()
    vendor = ENV_INFO.get("flag_dnn", {}).get("vendor", "")
    vendor_env_map = {
        "ascend": ["ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"],
        "hygon": ["HIP_VISIBLE_DEVICES"],
        "iluvatar": ["ILUVATAR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"],
        "kunlunxin": ["CUDA_VISIBLE_DEVICES"],
        "metax": ["MACA_VISIBLE_DEVICES"],
        "mthreads": ["MUSA_VISIBLE_DEVICES"],
        "thead": ["CUDA_VISIBLE_DEVICES"],
        "tsingmicro": ["TXDA_VISIBLE_DEVICES"],
    }
    for var in vendor_env_map.get(vendor, ["CUDA_VISIBLE_DEVICES"]):
        env[var] = gpu_ids
    return env


def run_cmd(
    op: str,
    suite: str,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    timeout: int,
) -> dict[str, Any]:
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
        code = TIMEOUT
    except Exception as exc:
        stdout, stderr = "", str(exc)
        code = -1

    if CFG.dump_output:
        op_dir = CFG.output_dir / op
        ensure_dir(op_dir)
        (op_dir / f"{suite}_stdout.log").write_text(
            stdout or "", encoding="utf-8"
        )
        (op_dir / f"{suite}_stderr.log").write_text(
            stderr or "", encoding="utf-8"
        )
        (op_dir / f"{suite}_cmd.txt").write_text(
            shlex.join(cmd) + "\n", encoding="utf-8"
        )

    return {"exit_code": code, "stdout": stdout or "", "stderr": stderr or ""}


def parse_pytest_counts(
    stdout: str, stderr: str, exit_code: int
) -> dict[str, Any]:
    text = stdout + "\n" + stderr
    counts = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
    }
    pattern = re.compile(
        r"(\d+)\s+"
        r"(passed|failed|skipped|error|errors|xfailed|xpassed|deselected)"
    )
    for number, name in pattern.findall(text):
        key = "errors" if name in {"error", "errors"} else name
        if key in counts:
            counts[key] += int(number)

    total = sum(counts.values())
    if exit_code == TIMEOUT:
        status = "Timeout"
    elif exit_code == 5 and total == 0:
        status = "NotFound"
    elif counts["failed"] or counts["errors"] or exit_code not in (0, 5):
        status = "Failed"
    elif counts["passed"] == 0 and counts["skipped"] > 0:
        status = "Skipped"
    elif total == 0:
        status = "NotFound"
    else:
        status = "Passed"

    result = {"total": total, "status": status}
    result.update(counts)
    return result


def parse_benchmark_stdout(
    op: str, stdout: str, stderr: str, exit_code: int
) -> dict[str, Any]:
    pytest_counts = parse_pytest_counts(stdout, stderr, exit_code)
    if exit_code == TIMEOUT:
        return {"status": "Timeout", "data": {}, "counts": pytest_counts}
    if exit_code == 5:
        status = pytest_counts["status"]
        if status == "Skipped":
            return {"status": "Skipped", "data": {}, "counts": pytest_counts}
        return {"status": "NotFound", "data": {}, "counts": pytest_counts}

    records: dict[str, Any] = {}
    current_dtype = "unknown"
    current_op = op
    op_pattern = re.compile(
        r"Operator:\s*(?P<operator>[A-Za-z0-9_]+)\s+.*Performance Test"
        r"(?:\s*\((?P<meta>[^)]*)\))?"
    )
    for line in stdout.splitlines():
        op_match = op_pattern.search(line)
        if op_match:
            current_op = op_match.group("operator")
            meta = op_match.group("meta") or ""
            dtype_match = re.search(r"dtype=([^,\s]+)", meta)
            current_dtype = dtype_match.group(1) if dtype_match else "unknown"
            current_dtype = DTYPE_MAP.get(current_dtype, current_dtype)
            continue
        if not line.startswith("SUCCESS"):
            continue
        parts = line.split(maxsplit=6)
        if len(parts) < 4 or current_op != op:
            continue
        try:
            base = float(parts[1])
            flag_dnn_latency = float(parts[2])
            speedup = float(parts[3])
        except ValueError:
            continue
        shape = parts[6].strip() if len(parts) > 6 else ""
        records.setdefault(current_dtype, {"details": {}, "speedups": []})
        records[current_dtype]["details"][shape] = {
            "base": base,
            "flag_dnn": flag_dnn_latency,
            "speedup": speedup,
        }
        records[current_dtype]["speedups"].append(speedup)

    data = {}
    for dtype, item in records.items():
        speedups = item.pop("speedups")
        data[dtype] = {
            "result": "OK",
            "details": item["details"],
            "speedup": sum(speedups) / len(speedups) if speedups else 0.0,
        }

    if exit_code != 0:
        return {"status": "Failed", "data": data, "counts": pytest_counts}
    if not data:
        if pytest_counts["status"] in {"Skipped", "NotFound"}:
            return {
                "status": pytest_counts["status"],
                "data": {},
                "counts": pytest_counts,
            }
        return {"status": "Passed", "data": {}, "counts": pytest_counts}
    return {"status": "Passed", "data": data, "counts": pytest_counts}


def suite_is_available(op: str, suite: str) -> bool:
    return op in MARKS_BY_SUITE.get(suite, set())


def run_accuracy_suite(gpu_id: int, op: str) -> dict[str, Any]:
    suite = "accuracy"
    if not suite_is_available(op, suite):
        return {"status": "NotFound", "exit_code": 5, "duration": 0.0}

    env = get_env(str(gpu_id))
    directory = ROOT / SUITES[suite]["directory"]
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        f"{op} and graph",
        "-q",
        "-rs",
        "--tb=short",
    ]

    start = time.time()
    run = run_cmd(op, suite, cmd, cwd=directory, env=env, timeout=CFG.timeout)
    duration = time.time() - start
    result = parse_pytest_counts(
        run["stdout"], run["stderr"], run["exit_code"]
    )
    result["exit_code"] = run["exit_code"]
    result["duration"] = duration
    return result


def run_benchmark_suite(gpu_id: int, op: str) -> dict[str, Any]:
    suite = "benchmark"
    if not suite_is_available(op, suite):
        return {
            "status": "NotFound",
            "exit_code": 5,
            "duration": 0.0,
            "data": {},
        }

    env = get_env(str(gpu_id))
    directory = ROOT / "benchmark"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        f"{op} and graph and perf",
        "-q",
        "-s",
        "-rs",
        "--tb=short",
    ]

    start = time.time()
    run = run_cmd(op, suite, cmd, cwd=directory, env=env, timeout=CFG.timeout)
    duration = time.time() - start
    record = parse_benchmark_stdout(
        op, run["stdout"], run["stderr"], run["exit_code"]
    )
    record["duration"] = duration
    record["exit_code"] = run["exit_code"]
    return record


def run_suite(gpu_id: int, op: str, suite: str) -> dict[str, Any]:
    if suite == "accuracy":
        return run_accuracy_suite(gpu_id, op)
    if suite == "benchmark":
        return run_benchmark_suite(gpu_id, op)
    return {"status": "Skipped", "duration": 0.0, "exit_code": 0}


def worker_proc(gpu_id: int, work_queue: Queue, display_queue: Queue) -> None:
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

    worker_result: dict[str, Any] = {}
    while True:
        try:
            op = work_queue.get_nowait()
        except queue_module.Empty:
            break
        if not op:
            continue

        op_result: dict[str, Any] = {
            "implemented": CFG.implemented_ops.get(op, False),
        }
        suites = CFG.op_suites.get(op, [])
        for suite in suites:
            display_queue.put(("start", gpu_id, suite, op))
            result = run_suite(gpu_id, op, suite)
            display_queue.put(
                (
                    "done",
                    gpu_id,
                    suite,
                    op,
                    result.get("status", "Error"),
                    result.get("duration", 0.0),
                )
            )
            op_result[suite] = result

        worker_result[op] = op_result
        json_path = CFG.output_dir / f"summary{gpu_id}.json"
        tmp_path = json_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(worker_result, f, indent=2)
        os.replace(tmp_path, json_path)

    display_queue.put(("exit", gpu_id))


def display_loop(queue: Queue, display: LiveDisplay, n_workers: int) -> None:
    exited = 0
    tasks_done = 0
    per_gpu_done = {gid: 0 for gid in display.gpu_ids}

    while exited < n_workers:
        try:
            msg = queue.get(timeout=1)
        except Exception:
            continue

        kind = msg[0]
        if kind == "exit":
            gpu_id = msg[1]
            n = per_gpu_done.get(gpu_id, 0)
            display.update_gpu(
                gpu_id, f"{DIM}[GPU {gpu_id:2d}] done ({n} tasks){NC}"
            )
            exited += 1
            continue

        if kind == "start":
            _, gpu_id, suite, op = msg
            op_col = _short_op(op, display.op_width)
            n = per_gpu_done.get(gpu_id, 0)
            if IS_TTY:
                display.update_gpu(
                    gpu_id,
                    f"[GPU {gpu_id:2d}] ({n:>3} done)  {suite:<18} {op_col}",
                )
            else:
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                display.log(
                    f"[INFO] [{ts}][GPU {gpu_id:2d}] {suite:<18} {op_col} ..."
                )
            continue

        if kind == "done":
            _, gpu_id, suite, op, status, dur = msg
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            op_col = _short_op(op, display.op_width)
            status_str = _format_status(status, dur)

            tasks_done += 1
            per_gpu_done[gpu_id] = per_gpu_done.get(gpu_id, 0) + 1
            display.set_progress(tasks_done)

            total = display.task_count
            pct = tasks_done * 100 // total if total else 100
            total_w = len(str(total))
            log_line = (
                f"{GREEN}[INFO]{NC} [{ts}][GPU {gpu_id:2d}] "
                f"{suite:<18} {op_col} {status_str}"
            )
            if not IS_TTY:
                log_line += (
                    f"  ({pct:>3}% {tasks_done:>{total_w}}/{total} tasks)"
                )
            display.log(log_line)


def _short_op(op: str, width: int) -> str:
    op_display = op if len(op) <= width else op[: width - 3] + "..."
    return op_display.ljust(width)


def cleanup_intermediate_files() -> None:
    if not hasattr(CFG, "output_dir"):
        return
    for path in CFG.output_dir.glob("summary*.tmp"):
        try:
            path.unlink()
        except OSError:
            pass


def terminate_workers() -> None:
    for proc in WORKER_PROCESSES:
        if proc.is_alive():
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
    for proc in WORKER_PROCESSES:
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()


def handle_interrupt(signum: int, frame: Any) -> None:
    global INTERRUPTED
    if INTERRUPTED:
        return
    INTERRUPTED = True
    if IS_TTY:
        sys.stdout.write("\n")
    pwarn("Interrupted. Cleaning up ...")
    terminate_workers()
    cleanup_intermediate_files()
    pwarn("Cleanup done.")
    sys.exit(1)


def _stage_matches(op: dict[str, Any], effective_stages: set[str]) -> bool:
    stages = op.get("stages", [])
    if not stages:
        return False
    present = {next(iter(stage.keys()), None) for stage in stages if stage}
    return bool(present & effective_stages)


def _effective_stages(value: str) -> set[str]:
    effective: set[str] = set()
    for raw_stage in value.split(","):
        stage = raw_stage.strip()
        if stage not in {"alpha", "beta", "stable", "all", "removed"}:
            pwarn(f"ignoring unsupported stage name '{stage}'...")
            continue
        if stage == "all":
            return {"alpha", "beta", "stable"}
        effective.add(stage)
    return effective or {"stable"}


def _requested_suites(value: str) -> set[str] | None:
    requested = {item.strip() for item in value.split(",") if item.strip()}
    if not requested or requested == {"auto"}:
        return None
    if "all" in requested:
        return set(SUITES)
    unsupported = requested - set(SUITES)
    if unsupported:
        raise ValueError(
            f"Unsupported suite(s): {', '.join(sorted(unsupported))}"
        )
    return requested


def get_ops_to_test() -> list[str]:
    catalog = get_ops_from_inventory()
    CFG.catalog_by_id = {op["id"]: op for op in catalog if "id" in op}

    if OPTS.ops:
        return [
            op.strip().lstrip("_") for op in OPTS.ops.split(",") if op.strip()
        ]

    if OPTS.op_list_file:
        try:
            lines = (
                Path(OPTS.op_list_file)
                .read_text(encoding="utf-8")
                .splitlines()
            )
        except Exception as exc:
            perror(f"Failed reading the specified op list file: {exc}")
            return []
        return [
            line.strip().lstrip("_")
            for line in lines
            if line.strip() and not line.strip().startswith("#")
        ]

    stages = _effective_stages(OPTS.stages)
    ops = []
    for op in catalog:
        op_id = op.get("id")
        if not op_id:
            continue
        if not _stage_matches(op, stages):
            continue
        if OPTS.start is not None and op_id < OPTS.start:
            continue
        ops.append(op_id)
    return ops


def build_op_suites(ops: list[str]) -> dict[str, list[str]]:
    requested = _requested_suites(OPTS.suites)
    op_suites: dict[str, list[str]] = {}
    for op in ops:
        metadata = CFG.catalog_by_id.get(op, {})
        labels = set(metadata.get("labels", []))
        suites = []
        for suite, spec in SUITES.items():
            if spec["label"] not in labels:
                continue
            if requested is not None and suite not in requested:
                continue
            suites.append(suite)
        op_suites[op] = suites
    return op_suites


def main() -> None:
    global OPTS
    global USE_COLORS, RED, GREEN, YELLOW, CYAN, DIM, NC
    global MARKS_BY_SUITE

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", required=False, help="comma-separated op ids")
    parser.add_argument(
        "--op-list-file", required=False, help="operator list file"
    )
    parser.add_argument("--start", required=False, help="first operator id")
    parser.add_argument("--gpus", default="0", help="comma-separated GPU ids")
    parser.add_argument(
        "--output-dir",
        default="results",
        help="relative path to root for test data",
    )
    parser.add_argument(
        "--stages",
        default="stable",
        help="comma-separated stages: alpha,beta,stable,all",
    )
    parser.add_argument(
        "--suites",
        default="auto",
        help=(
            "comma-separated suites, 'all', or 'auto'. "
            "Suites: accuracy,benchmark"
        ),
    )
    parser.add_argument(
        "--timeout", type=int, default=600, help="per-suite timeout"
    )
    parser.add_argument(
        "--dump-output",
        action="store_true",
        default=False,
        help="dump stdout/stderr of each pytest run to output-dir",
    )
    parser.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default="auto",
        help="control ANSI color output",
    )
    OPTS = parser.parse_args()

    if OPTS.color == "always":
        USE_COLORS = True
        RED, GREEN, YELLOW, CYAN, DIM, NC = (
            "\033[31m",
            "\033[32m",
            "\033[93m",
            "\033[36m",
            "\033[2m",
            "\033[0m",
        )
    elif OPTS.color == "never":
        USE_COLORS = False
        RED = GREEN = YELLOW = CYAN = DIM = NC = ""

    CFG.dump_output = OPTS.dump_output
    CFG.timeout = OPTS.timeout
    output_dir = Path(OPTS.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    CFG.output_dir = output_dir
    ensure_dir(CFG.output_dir)

    MARKS_BY_SUITE = discover_all_marks()
    probe_env()

    ops = get_ops_to_test()
    if not ops:
        pwarn("No operators to test. Please specify at least one operator.")
        sys.exit(1)

    CFG.ops = ops
    CFG.op_suites = build_op_suites(ops)
    CFG.implemented_ops = {
        op: hasattr(flag_dnn, op)
        or op in dict(getattr(flag_dnn, "_FULL_CONFIG", []))
        for op in ops
    }

    task_count = sum(len(suites) for suites in CFG.op_suites.values())
    if task_count == 0:
        pwarn("No test suites selected from operators.yaml labels.")
        sys.exit(1)

    pinfo(f"Testing {len(ops)} operators, {task_count} suite tasks ...")

    gpu_ids = [
        int(item) for item in OPTS.gpus.strip().split(",") if item.strip()
    ]
    if not gpu_ids:
        pwarn("Empty GPU list specified.")
        sys.exit(1)

    op_width = min(max(len(op) for op in ops), 40) if ops else 20
    work_queue: Queue = Queue()
    for op in ops:
        work_queue.put(op)

    display_queue: Queue = Queue()
    display = LiveDisplay(gpu_ids, task_count, op_width=op_width)

    for gpu_id in gpu_ids:
        proc = Process(
            target=worker_proc, args=(gpu_id, work_queue, display_queue)
        )
        proc.start()
        WORKER_PROCESSES.append(proc)

    display.init()
    display_loop(display_queue, display, len(gpu_ids))

    for proc in WORKER_PROCESSES:
        proc.join()

    display.finish()

    op_data: dict[str, Any] = {}
    for gpu_id in gpu_ids:
        gpu_file = CFG.output_dir / f"summary{gpu_id}.json"
        if not gpu_file.exists():
            perror(
                f"GPU {gpu_id} failed to produce a summary, recovery needed."
            )
            continue
        try:
            op_data.update(json.loads(gpu_file.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, ValueError):
            perror(f"GPU {gpu_id} summary is invalid JSON, skipping.")

    final_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "env": ENV_INFO,
        "selected_suites": CFG.op_suites,
        "result": op_data,
    }
    (CFG.output_dir / "summary.json").write_text(
        json.dumps(final_data, indent=2), encoding="utf-8"
    )

    cleanup_intermediate_files()
    pinfo("Test completed.")


if __name__ == "__main__":
    main()
