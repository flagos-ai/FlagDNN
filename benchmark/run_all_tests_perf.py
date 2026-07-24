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

import argparse
import glob
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence, TypedDict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
import triton  # noqa: E402

from benchmark import consts  # noqa: E402
from benchmark.base import (  # noqa: E402
    PERF_RECORD_PREFIX,
    UNSUPPORTED_RECORD_PREFIX,
)

# ================= 配置区 =================

# 目标算子列表 (白名单)
# 例如: ["relu", "add"]。如果留空 []，则自动测试目录下所有的 test_*.py
TARGET_OPERATORS: list[str] = []

# benchmark/run_all_tests_perf.py 只负责 benchmark 目录。
TEST_DIR = SCRIPT_DIR
LOG_DIR = os.path.join(REPO_ROOT, "benchmark_logs")  # 单个测试日志的存放目录

# 运行状态汇总，例如 total / passed / failed / details
REPORT_FILE = os.path.join(REPO_ROOT, "benchmark_summary.json")

# 核心性能数据，例如 operator / dtype / shape / speedup
DATA_FILE = os.path.join(REPO_ROOT, "benchmark_data.json")

# ==========================================


class BenchmarkDetail(TypedDict):
    file: str
    operator: str
    status: str
    return_code: int
    duration_seconds: float
    log_path: str
    collected_items: int
    passed_items: int
    failed_items: int
    skipped_items: int
    error_items: int
    data_points_collected: int
    dtypes_collected: list[str]
    operators_collected: list[str]


class _RequiredBenchmarkSummary(TypedDict):
    total: int
    passed: int
    failed: int
    skipped_or_unsupported: int
    errored_or_interrupted: int
    details: list[BenchmarkDetail]
    start_time: str


class BenchmarkSummary(_RequiredBenchmarkSummary, total=False):
    total_duration_seconds: float


def get_operator_name(filename):
    """从文件名中提取算子名，例如 test_relu.py -> relu"""
    basename = os.path.basename(filename)
    if basename.startswith("test_") and basename.endswith(".py"):
        return basename[5:-3]
    return basename


def repo_relative_path(path):
    return os.path.relpath(path, REPO_ROOT)


def parse_float(value):
    """
    安全解析 float。
    支持普通小数、科学计数法、nan、inf。
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_meta_args(meta_text):
    """
    解析 Operator 行括号里的参数。

    输入示例：
        dtype=torch.float16, mode=kernel, level=comprehensive

    输出：
        {
            "dtype": "torch.float16",
            "mode": "kernel",
            "level": "comprehensive"
        }
    """
    meta = {}

    if not meta_text:
        return meta

    for item in meta_text.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue

        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()

        if key:
            meta[key] = value

    return meta


def short_dtype(dtype):
    """
    torch.float16 -> float16
    torch.bfloat16 -> bfloat16
    float16 -> float16
    """
    if not dtype:
        return "unknown"

    dtype = str(dtype).strip()

    if dtype.startswith("torch."):
        return dtype[len("torch.") :]

    return dtype


def parse_legacy_operator_dtype(operator_name):
    """
    兼容老格式：
        Operator: relu_fp16 Performance Test

    如果日志里没有 dtype=(...)，但 operator 名字带了后缀，
    可以尽量拆成：
        operator = relu
        dtype = torch.float16
    """
    legacy_map = {
        "fp16": "torch.float16",
        "float16": "torch.float16",
        "bf16": "torch.bfloat16",
        "bfloat16": "torch.bfloat16",
        "fp32": "torch.float32",
        "float32": "torch.float32",
        "fp64": "torch.float64",
        "float64": "torch.float64",
        "int32": "torch.int32",
        "int64": "torch.int64",
        "bool": "torch.bool",
    }

    for suffix, dtype in legacy_map.items():
        marker = "_" + suffix
        if operator_name.endswith(marker):
            return operator_name[: -len(marker)], dtype

    return operator_name, None


def parse_pytest_collected_count(stdout_text):
    """Return pytest collected item count parsed from stdout."""
    for line in stdout_text.splitlines():
        match = re.search(r"collected\s+(\d+)\s+items?", line)
        if match:
            return int(match.group(1))
    return 0


def parse_pytest_skipped_count(stdout_text):
    """Return pytest skipped item count parsed from the summary line."""
    skipped_count = 0
    for line in stdout_text.splitlines():
        match = re.search(r"(\d+)\s+skipped", line)
        if match:
            skipped_count = int(match.group(1))
    return skipped_count


def parse_pytest_outcome_count(stdout_text, outcome):
    """Return an outcome count parsed from pytest's terminal summary."""
    labels = {
        "passed": ("passed",),
        "failed": ("failed",),
        "skipped": ("skipped",),
        "errors": ("error", "errors"),
        "xfailed": ("xfailed",),
        "xpassed": ("xpassed",),
    }
    if outcome not in labels:
        raise ValueError(f"Unsupported pytest outcome: {outcome}")

    count = 0
    for line in stdout_text.splitlines():
        for label in labels[outcome]:
            match = re.search(rf"(\d+)\s+{label}\b", line)
            if match:
                count = int(match.group(1))
                break
    return count


def all_pytest_items_skipped(stdout_text):
    collected_count = parse_pytest_collected_count(stdout_text)
    skipped_count = parse_pytest_skipped_count(stdout_text)
    return skipped_count > 0 and (
        collected_count == 0 or skipped_count >= collected_count
    )


def _parse_legacy_perf_output(stdout_text):
    """
    从 pytest 的输出中解析出性能数据。

    重点改动：
    1. 从 Operator 行解析 dtype / mode / level。
    2. 每一条 SUCCESS 记录都带上当前 dtype。
    3. 这样 perf_data.json 后续就可以按 dtype 分组统计平均 speedup。
    """

    records = []

    current_context = {
        "operator": None,
        "dtype": "unknown",
        "dtype_short": "unknown",
        "mode": None,
        "level": None,
    }

    for line in stdout_text.splitlines():
        line = line.rstrip("\n")

        # 例子：
        # Operator: conv2d  Performance Test
        # (dtype=torch.float16, mode=kernel, level=comprehensive)
        op_match = re.search(
            r"Operator:\s*(?P<operator>[A-Za-z0-9_]+)\s*"
            r"(?:[A-Za-z0-9_ ]+)?Performance Test"
            r"(?:\s*\((?P<meta>[^)]*)\))?",
            line,
        )

        if op_match:
            raw_operator = op_match.group("operator")
            meta_text = op_match.group("meta") or ""
            meta = parse_meta_args(meta_text)

            operator, legacy_dtype = parse_legacy_operator_dtype(raw_operator)

            dtype = meta.get("dtype") or legacy_dtype or "unknown"
            mode = meta.get("mode")
            level = meta.get("level")

            current_context = {
                "operator": operator,
                "dtype": dtype,
                "dtype_short": short_dtype(dtype),
                "mode": mode,
                "level": level,
            }
            continue

        # 只解析 SUCCESS 行
        if not line.startswith("SUCCESS"):
            continue

        current_op = current_context.get("operator")
        if not current_op:
            continue

        parts = line.split(maxsplit=6)

        # SUCCESS CudnnLatency FlagDNNLatency Speedup CudnnGBPS FlagDNNGBPS ...
        if len(parts) < 4:
            continue

        cudnn_latency = parse_float(parts[1])
        flagdnn_latency = parse_float(parts[2])
        speedup = parse_float(parts[3])

        if cudnn_latency is None or flagdnn_latency is None or speedup is None:
            continue

        record = {
            "operator": current_context["operator"],
            "dtype": current_context["dtype"],
            "dtype_short": current_context["dtype_short"],
            "cudnn_latency": cudnn_latency,
            "flagdnn_latency": flagdnn_latency,
            "speedup": speedup,
        }

        # mode / level 不是所有日志都有，有就写入，没有就不写
        if current_context.get("mode") is not None:
            record["mode"] = current_context["mode"]

        if current_context.get("level") is not None:
            record["level"] = current_context["level"]

        # 如果日志中有 GBPS 数据，也一并提取
        if len(parts) >= 6:
            cudnn_gbps = parse_float(parts[4])
            flagdnn_gbps = parse_float(parts[5])

            if cudnn_gbps is not None and flagdnn_gbps is not None:
                record["cudnn_gbps"] = cudnn_gbps
                record["flagdnn_gbps"] = flagdnn_gbps

        if len(parts) == 7 and parts[6].strip():
            record["size_detail"] = parts[6].strip()

        records.append(record)

    return records


def parse_perf_output(
    stdout_text: str,
    *,
    source_file: str,
    source_log: str | None = None,
    allow_legacy: bool = False,
) -> list[dict[str, Any]]:
    raw_records: list[dict[str, Any]] = []
    success_count = sum(
        line.startswith("SUCCESS") for line in stdout_text.splitlines()
    )
    required = {
        "schema_version",
        "operator",
        "dtype",
        "mode",
        "level",
        "execution_path",
        "size_detail",
        "baseline_latency_ms",
        "flagdnn_latency_ms",
        "speedup",
    }
    for line in stdout_text.splitlines():
        marker = line.find(PERF_RECORD_PREFIX)
        if marker < 0:
            continue
        payload = line[marker + len(PERF_RECORD_PREFIX) :]
        try:
            record = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("malformed raw performance record") from exc
        if not isinstance(record, dict):
            raise ValueError("raw performance record must be an object")
        missing = required - record.keys()
        if missing or record.get("schema_version") != 1:
            raise ValueError(f"invalid raw schema; missing={sorted(missing)}")
        for field in (
            "operator",
            "dtype",
            "mode",
            "level",
            "execution_path",
            "size_detail",
        ):
            if not str(record[field]).strip():
                raise ValueError(f"raw field {field} must be nonempty")
        baseline = float(record["baseline_latency_ms"])
        flagdnn = float(record["flagdnn_latency_ms"])
        speedup = float(record["speedup"])
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (baseline, flagdnn, speedup)
        ):
            raise ValueError("raw latency/speedup must be finite and positive")
        if not math.isclose(
            speedup, baseline / flagdnn, rel_tol=1e-12, abs_tol=0.0
        ):
            raise ValueError("raw speedup does not match raw latencies")
        normalized = dict(record)
        normalized["source_file"] = source_file
        normalized["source_log"] = source_log
        normalized["source_row_index"] = len(raw_records)
        raw_records.append(normalized)
    if raw_records:
        if len(raw_records) != success_count:
            raise ValueError(
                f"raw/SUCCESS count mismatch: {len(raw_records)} != "
                f"{success_count}"
            )
        return raw_records
    if success_count and not allow_legacy:
        raise ValueError("raw performance records are required")
    if not allow_legacy:
        return []

    legacy = _parse_legacy_perf_output(stdout_text)
    normalized_legacy = []
    for index, row in enumerate(legacy):
        baseline = float(row["cudnn_latency"])
        flagdnn = float(row["flagdnn_latency"])
        normalized_legacy.append(
            {
                **row,
                "schema_version": 0,
                "execution_path": "legacy_text_parser",
                "baseline_latency_ms": baseline,
                "flagdnn_latency_ms": flagdnn,
                "speedup": baseline / flagdnn,
                "source_file": source_file,
                "source_log": source_log,
                "source_row_index": index,
            }
        )
    return normalized_legacy


def parse_unsupported_output(
    stdout_text: str, *, source_file: str
) -> list[dict[str, Any]]:
    lines = stdout_text.splitlines()
    human_count = sum("UNSUPPORTED shape=" in line for line in lines)
    records: list[dict[str, Any]] = []
    required = {
        "schema_version",
        "operator",
        "dtype",
        "size_detail",
        "reason",
    }
    for line in lines:
        marker = line.find(UNSUPPORTED_RECORD_PREFIX)
        if marker < 0:
            continue
        payload = line[marker + len(UNSUPPORTED_RECORD_PREFIX) :]
        try:
            record = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("malformed unsupported record") from exc
        if not isinstance(record, dict):
            raise ValueError("unsupported record must be an object")
        missing = required - record.keys()
        if missing or record.get("schema_version") != 1:
            raise ValueError(
                f"invalid unsupported schema; missing={sorted(missing)}"
            )
        normalized = {
            "schema_version": 1,
            "operator": str(record["operator"]),
            "dtype": str(record["dtype"]),
            "size_detail": str(record["size_detail"]),
            "reason": str(record["reason"]),
            "source_file": source_file,
        }
        if not all(str(value).strip() for value in normalized.values()):
            raise ValueError("unsupported record contains an empty field")
        records.append(normalized)
    if len(records) != human_count:
        raise ValueError(
            f"unsupported raw/human count mismatch: "
            f"{len(records)} != {human_count}"
        )
    identities = [
        (row["source_file"], row["dtype"], row["size_detail"])
        for row in records
    ]
    if len(identities) != len(set(identities)):
        raise ValueError(
            "duplicate unsupported identity in one benchmark file"
        )
    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-legacy-perf", action="store_true")
    return parser.parse_args(argv)


def _legacy_main() -> int:
    os.makedirs(LOG_DIR, exist_ok=True)

    # 收集并过滤测试文件
    all_test_files = sorted(glob.glob(os.path.join(TEST_DIR, "test_*.py")))

    if not all_test_files:
        print(
            f"未在 {repo_relative_path(TEST_DIR)} 目录下找到任何 "
            "test_*.py 文件。"
        )
        return 0

    test_files = []

    if TARGET_OPERATORS:
        for candidate_file in all_test_files:
            op_name = get_operator_name(candidate_file)
            if op_name in TARGET_OPERATORS:
                test_files.append(candidate_file)

        print(f"🔍 已启用算子过滤，目标算子数量: {len(TARGET_OPERATORS)}")
    else:
        test_files = all_test_files
        print("🔍 未设置过滤，将执行 benchmark 目录下所有性能测试。")

    if not test_files:
        print(
            "过滤后没有需要执行的测试文件，请检查 TARGET_OPERATORS 是否拼写正确。"
        )
        return 0

    print(f"🚀 共发现 {len(test_files)} 个待测性能文件，开始提交测试任务...\n")
    print("-" * 60)

    summary: BenchmarkSummary = {
        "total": len(test_files),
        "passed": 0,
        "failed": 0,
        "skipped_or_unsupported": 0,
        "errored_or_interrupted": 0,
        "details": [],
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 存储所有算子的所有性能测试记录
    all_perf_data = []

    start_time_total = time.time()

    for idx, file_path in enumerate(test_files, 1):
        file_name = os.path.basename(file_path)
        rel_file_path = repo_relative_path(file_path)
        log_name = rel_file_path.replace(os.sep, "_")
        log_file = os.path.join(LOG_DIR, f"{log_name}.log")

        print(
            f"[{idx}/{len(test_files)}] 正在测速: {file_name:<35}",
            end="",
            flush=True,
        )

        # 构建命令
        # 如果你需要 yhrun，把下面注释打开即可。
        cmd = [
            # "yhrun",
            # "-p", "h100x",
            # "-G", "1",
            sys.executable,
            "-m",
            "pytest",
            "-v",
            "-s",
            "-rs",
            rel_file_path,
        ]

        start_time = time.time()

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )

        duration = time.time() - start_time

        # ==== 核心解析步骤 ====
        extracted_data = parse_perf_output(
            result.stdout,
            source_file=file_name,
            allow_legacy=True,
        )
        all_perf_data.extend(extracted_data)

        # 分析退出状态码
        if result.returncode == 0 and all_pytest_items_skipped(result.stdout):
            status = "⏭️ SKIPPED/UNSUPPORTED"
            status_label = "SKIPPED/UNSUPPORTED"
            summary["skipped_or_unsupported"] += 1
        elif result.returncode == 0:
            status = "✅ PASS"
            status_label = "PASS"
            summary["passed"] += 1
        elif result.returncode == 1:
            status = "❌ FAIL"
            status_label = "FAIL"
            summary["failed"] += 1
        elif result.returncode == 5 and all_pytest_items_skipped(
            result.stdout
        ):
            status = "⏭️ SKIPPED/UNSUPPORTED"
            status_label = "SKIPPED/UNSUPPORTED"
            summary["skipped_or_unsupported"] += 1
        elif result.returncode == 5:
            status = "⚠️ NO TESTS"
            status_label = "NO TESTS"
            summary["errored_or_interrupted"] += 1
        else:
            status = f"💥 ERROR (Code: {result.returncode})"
            status_label = f"ERROR (Code: {result.returncode})"
            summary["errored_or_interrupted"] += 1

        print(f" -> {status} ({duration:.2f}s)")

        # 将标准输出和错误写入独立日志文件
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"=== Command: {' '.join(cmd)} ===\n")
            f.write(f"=== Status: {status} ===\n")
            f.write(f"=== Duration: {duration:.2f}s ===\n\n")
            f.write("--- STDOUT ---\n")
            f.write(result.stdout)
            f.write("\n")

            if result.stderr:
                f.write("--- STDERR ---\n")
                f.write(result.stderr)
                f.write("\n")

        # 当前测试文件中解析到的 dtype 集合
        dtypes_collected = sorted(
            {item.get("dtype", "unknown") for item in extracted_data}
        )

        # 当前测试文件中解析到的算子集合
        operators_collected = sorted(
            {item.get("operator", "unknown") for item in extracted_data}
        )

        summary["details"].append(
            {
                "file": file_name,
                "operator": get_operator_name(file_name),
                "status": status_label,
                "return_code": result.returncode,
                "duration_seconds": round(duration, 2),
                "log_path": log_file,
                "collected_items": parse_pytest_collected_count(result.stdout),
                "passed_items": parse_pytest_outcome_count(
                    result.stdout, "passed"
                ),
                "failed_items": parse_pytest_outcome_count(
                    result.stdout, "failed"
                ),
                "skipped_items": parse_pytest_outcome_count(
                    result.stdout, "skipped"
                ),
                "error_items": parse_pytest_outcome_count(
                    result.stdout, "errors"
                ),
                "data_points_collected": len(extracted_data),
                "dtypes_collected": dtypes_collected,
                "operators_collected": operators_collected,
            }
        )

    summary["total_duration_seconds"] = round(
        time.time() - start_time_total,
        2,
    )

    # 写运行状态汇总
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    # 写性能明细数据
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(all_perf_data, f, indent=4, ensure_ascii=False)

    print("-" * 60)
    print("📊 性能测试执行完毕！")
    print(
        f"总计脚本: {summary['total']} | "
        f"通过: {summary['passed']} | "
        f"跳过/不支持: {summary['skipped_or_unsupported']} | "
        f"异常: {summary['failed'] + summary['errored_or_interrupted']}"
    )
    print(
        f"共收集到 {len(all_perf_data)} 条性能数据记录，已保存至 {DATA_FILE}"
    )
    print(f"运行状态汇总已保存至 {REPORT_FILE}")
    print(f"总耗时: {summary['total_duration_seconds']} 秒")

    has_failures = summary["failed"] or summary["errored_or_interrupted"]
    return 1 if has_failures else 0


def _classify_result(returncode: int, stdout: str) -> tuple[str, str]:
    if returncode == 0 and all_pytest_items_skipped(stdout):
        return "SKIPPED/UNSUPPORTED", "⏭️ SKIPPED/UNSUPPORTED"
    if returncode == 0:
        return "PASS", "✅ PASS"
    if returncode == 1:
        return "FAIL", "❌ FAIL"
    if returncode == 5 and all_pytest_items_skipped(stdout):
        return "SKIPPED/UNSUPPORTED", "⏭️ SKIPPED/UNSUPPORTED"
    if returncode == 5:
        return "NO TESTS", "⚠️ NO TESTS"
    return f"ERROR (Code: {returncode})", f"💥 ERROR (Code: {returncode})"


def _environment_snapshot() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "device_name": (
            torch.cuda.get_device_name() if cuda_available else None
        ),
        "device_capability": (
            list(torch.cuda.get_device_capability())
            if cuda_available
            else None
        ),
        "flagdnn_cache_dir": os.environ.get("FLAGGEMS_CACHE_DIR"),
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        "triton_cache_autotuning": os.environ.get("TRITON_CACHE_AUTOTUNING"),
        "perf_warmup": consts.bench_warmup(),
        "perf_repeat": consts.bench_repeat(),
    }


def _case_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return tuple(
        str(row[field])
        for field in ("operator", "dtype", "mode", "level", "size_detail")
    )  # type: ignore[return-value]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        print(
            f"refusing to overwrite output directory: {output_dir}",
            file=sys.stderr,
        )
        return 2

    all_test_files = sorted(Path(TEST_DIR).glob("test_*.py"))
    by_operator = {
        get_operator_name(path.name): path for path in all_test_files
    }
    requested = list(args.operator)
    if len(requested) != len(set(requested)):
        print("--operator values must be unique", file=sys.stderr)
        return 2
    invalid = [
        name for name in requested if re.fullmatch(r"[a-z0-9_]+", name) is None
    ]
    unknown = sorted(set(requested).difference(by_operator))
    if invalid or unknown:
        print(
            f"invalid operators={invalid}, unknown operators={unknown}",
            file=sys.stderr,
        )
        return 2
    test_files = (
        [by_operator[name] for name in requested]
        if requested
        else all_test_files
    )
    if not test_files:
        print("no benchmark test files selected", file=sys.stderr)
        return 2

    log_dir = output_dir / "benchmark_logs"
    log_dir.mkdir(parents=True)
    report_file = output_dir / "benchmark_summary.json"
    data_file = output_dir / "benchmark_data.json"
    summary: dict[str, Any] = {
        "total": len(test_files),
        "passed": 0,
        "failed": 0,
        "skipped_or_unsupported": 0,
        "errored_or_interrupted": 0,
        "details": [],
        "unsupported_records": [],
        "environment": _environment_snapshot(),
        "start_time": datetime.now().isoformat(timespec="seconds"),
    }
    all_perf_data: list[dict[str, Any]] = []
    all_unsupported_records: list[dict[str, Any]] = []
    total_start = time.time()

    for index, file_path in enumerate(test_files, 1):
        file_name = file_path.name
        relative_test = str(file_path.relative_to(REPO_ROOT))
        log_file = log_dir / f"benchmark_{file_name}.log"
        source_log = str(log_file.relative_to(output_dir))
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-v",
            "-s",
            "-rs",
            relative_test,
        ]
        print(
            f"[{index}/{len(test_files)}] benchmark: {file_name:<35}",
            end="",
            flush=True,
        )
        started = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        duration = time.time() - started
        status_label, status_display = _classify_result(
            result.returncode, result.stdout
        )
        extracted_data: list[dict[str, Any]] = []
        unsupported_data: list[dict[str, Any]] = []
        if status_label in {"PASS", "SKIPPED/UNSUPPORTED"}:
            try:
                extracted_data = parse_perf_output(
                    result.stdout,
                    source_file=file_name,
                    source_log=source_log,
                    allow_legacy=args.allow_legacy_perf,
                )
                unsupported_data = parse_unsupported_output(
                    result.stdout, source_file=file_name
                )
            except ValueError as exc:
                status_label = "RAW DATA ERROR"
                status_display = f"💥 RAW DATA ERROR ({exc})"
                extracted_data = []
                unsupported_data = []

        accepted_data = extracted_data if status_label == "PASS" else []
        if status_label == "PASS":
            summary["passed"] += 1
        elif status_label == "SKIPPED/UNSUPPORTED":
            summary["skipped_or_unsupported"] += 1
        elif status_label == "FAIL":
            summary["failed"] += 1
        else:
            summary["errored_or_interrupted"] += 1
        all_perf_data.extend(accepted_data)
        all_unsupported_records.extend(unsupported_data)
        print(f" -> {status_display} ({duration:.2f}s)")

        with log_file.open("x", encoding="utf-8") as handle:
            handle.write(f"=== Command: {' '.join(cmd)} ===\n")
            handle.write(f"=== Status: {status_display} ===\n")
            handle.write(f"=== Duration: {duration:.6f}s ===\n\n")
            handle.write("--- STDOUT ---\n")
            handle.write(result.stdout)
            handle.write("\n")
            if result.stderr:
                handle.write("--- STDERR ---\n")
                handle.write(result.stderr)
                handle.write("\n")

        summary["details"].append(
            {
                "file": file_name,
                "operator": get_operator_name(file_name),
                "status": status_label,
                "return_code": result.returncode,
                "duration_seconds": duration,
                "log_path": source_log,
                "collected_items": parse_pytest_collected_count(result.stdout),
                "passed_items": parse_pytest_outcome_count(
                    result.stdout, "passed"
                ),
                "failed_items": parse_pytest_outcome_count(
                    result.stdout, "failed"
                ),
                "skipped_items": parse_pytest_outcome_count(
                    result.stdout, "skipped"
                ),
                "error_items": parse_pytest_outcome_count(
                    result.stdout, "errors"
                ),
                "data_points_collected": len(accepted_data),
                "raw_records_collected": sum(
                    row["schema_version"] == 1 for row in accepted_data
                ),
                "legacy_records_collected": sum(
                    row["schema_version"] == 0 for row in accepted_data
                ),
                "unsupported_records_collected": len(unsupported_data),
                "dtypes_collected": sorted(
                    {str(row["dtype"]) for row in accepted_data}
                ),
                "operators_collected": sorted(
                    {str(row["operator"]) for row in accepted_data}
                ),
            }
        )

    all_perf_data.sort(key=_case_key)
    keys = [_case_key(row) for row in all_perf_data]
    if len(keys) != len(set(keys)):
        summary["errored_or_interrupted"] += 1
        summary["duplicate_case_keys"] = [
            list(key)
            for key in sorted({key for key in keys if keys.count(key) > 1})
        ]
    all_unsupported_records.sort(
        key=lambda row: (
            row["source_file"],
            row["dtype"],
            row["size_detail"],
            row["operator"],
            row["reason"],
        )
    )
    summary["unsupported_records"] = all_unsupported_records
    summary["total_duration_seconds"] = time.time() - total_start
    with report_file.open("x", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
    with data_file.open("x", encoding="utf-8") as handle:
        json.dump(
            all_perf_data,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")

    print(
        f"completed: files={summary['total']} passed={summary['passed']} "
        f"skipped={summary['skipped_or_unsupported']} "
        f"errors={summary['failed'] + summary['errored_or_interrupted']} "
        f"records={len(all_perf_data)} output={output_dir}"
    )
    has_failures = summary["failed"] or summary["errored_or_interrupted"]
    return 1 if has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
