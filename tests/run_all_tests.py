import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime


# ================= Configuration =================

# Target operator whitelist.
# If empty, all test_*.py files under TEST_DIR will be executed.
# If populated, only files named test_<operator>.py will be executed.
TARGET_OPERATORS: list[str] = []

# Non-operator test files to always skip. These are framework / self-check
# tests, not per-operator coverage, so the operator runner ignores them.
# Use the test_<name>.py suffix (e.g. "prepared" -> test_prepared.py).
# Entries here are skipped even if also listed in TARGET_OPERATORS.
EXCLUDED_OPERATORS: list[str] = [
    "prepared",  # test_prepared.py: prepared fast-path framework test
    "registry",  # test_registry.py: registry / capture-wrapper self-check
    "graph",  # test_graph.py: graph capture/compile core test
    "partial_completion",  # test_partial_completion.py
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
TEST_DIR = SCRIPT_DIR
LOG_DIR = os.path.join(REPO_ROOT, "test_logs")
REPORT_FILE = os.path.join(REPO_ROOT, "test_summary.json")

# ================================================


def get_operator_name(filename):
    """Extract operator name, e.g. test_batch_norm.py -> batch_norm."""
    basename = os.path.basename(filename)
    if basename.startswith("test_") and basename.endswith(".py"):
        return basename[5:-3]
    return basename


def pytest_reported_skip(stdout, stderr):
    output = f"{stdout}\n{stderr}"
    return (
        "SKIPPED [" in output
        or re.search(r"\b\d+\s+skipped\b", output) is not None
        or re.search(
            r"collected\s+0\s+items\s+/\s+\d+\s+skipped",
            output,
        )
        is not None
    )


def classify_pytest_result(return_code, stdout, stderr):
    if return_code == 0:
        return "PASS", "passed"
    if return_code == 1:
        return "FAIL", "failed"
    if return_code == 5:
        if pytest_reported_skip(stdout, stderr):
            return "SKIPPED", "skipped"
        return "NO TESTS", "no_tests"
    return f"ERROR (Code: {return_code})", "errored_or_interrupted"


def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    all_test_files = sorted(glob.glob(os.path.join(TEST_DIR, "test_*.py")))
    if not all_test_files:
        print("No test_*.py files found.")
        return

    test_files = []
    if TARGET_OPERATORS:
        for file_path in all_test_files:
            op_name = get_operator_name(file_path)
            if op_name in TARGET_OPERATORS:
                test_files.append(file_path)
        print(
            "Operator filter enabled, target operator count: "
            f"{len(TARGET_OPERATORS)}"
        )
    else:
        test_files = all_test_files
        print("No operator filter configured; running all tests.")

    if EXCLUDED_OPERATORS:
        excluded = set(EXCLUDED_OPERATORS)
        kept = [
            file_path
            for file_path in test_files
            if get_operator_name(file_path) not in excluded
        ]
        skipped = len(test_files) - len(kept)
        if skipped:
            print(
                f"Excluding {skipped} non-operator test file(s): "
                f"{', '.join(sorted(excluded))}"
            )
        test_files = kept

    if not test_files:
        print("No test files matched TARGET_OPERATORS.")
        return

    print(f"Found {len(test_files)} test files. Starting pytest runs...\n")
    print("-" * 60)

    summary = {
        "total": len(test_files),
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "no_tests": 0,
        "errored_or_interrupted": 0,
        "details": [],
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    start_time_total = time.time()

    for idx, file_path in enumerate(test_files, 1):
        file_name = os.path.basename(file_path)
        log_file = os.path.join(LOG_DIR, f"{file_name}.log")

        print(
            f"[{idx}/{len(test_files)}] Testing: {file_name:<30}",
            end="",
            flush=True,
        )

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-v",
            "-s",
            "-rs",
            file_path,
        ]

        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time

        status, summary_key = classify_pytest_result(
            result.returncode,
            result.stdout,
            result.stderr,
        )
        summary[summary_key] += 1

        print(f" -> {status} ({duration:.2f}s)")

        with open(log_file, "w", encoding="utf-8") as file_obj:
            file_obj.write(f"=== Command: {' '.join(cmd)} ===\n")
            file_obj.write(f"=== Status: {status} ===\n")
            file_obj.write(f"=== Duration: {duration:.2f}s ===\n\n")
            file_obj.write("--- STDOUT ---\n" + result.stdout + "\n")
            if result.stderr:
                file_obj.write("--- STDERR ---\n" + result.stderr + "\n")

        summary["details"].append(
            {
                "file": file_name,
                "status": status,
                "return_code": result.returncode,
                "duration_seconds": round(duration, 2),
                "log_path": log_file,
            }
        )

    summary["total_duration_seconds"] = round(
        time.time() - start_time_total, 2
    )

    with open(REPORT_FILE, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=4, ensure_ascii=False)

    print("-" * 60)
    print("Test run completed.")
    print(
        f"Total: {summary['total']} | "
        f"Passed: {summary['passed']} | "
        f"Failed: {summary['failed']} | "
        f"Skipped: {summary['skipped']} | "
        f"No tests: {summary['no_tests']} | "
        f"Errors/interrupted: {summary['errored_or_interrupted']}"
    )
    print(f"Total duration: {summary['total_duration_seconds']} seconds")
    print(f"Detailed logs saved under '{LOG_DIR}'.")


if __name__ == "__main__":
    main()
