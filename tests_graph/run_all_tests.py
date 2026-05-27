import glob
import json
import os
import subprocess
import time
from datetime import datetime


# ================= Configuration =================

# Target operator whitelist.
# If empty, all test_*.py files under TEST_DIR will be executed.
# If populated, only files named test_<operator>.py will be executed.
TARGET_OPERATORS = []

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
TEST_DIR = SCRIPT_DIR
LOG_DIR = os.path.join(REPO_ROOT, "test_graph_logs")
REPORT_FILE = os.path.join(REPO_ROOT, "test_graph_summary.json")

# ================================================


def get_operator_name(filename):
    """Extract operator name, e.g. test_batch_norm.py -> batch_norm."""
    basename = os.path.basename(filename)
    if basename.startswith("test_") and basename.endswith(".py"):
        return basename[5:-3]
    return basename


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

    if not test_files:
        print("No test files matched TARGET_OPERATORS.")
        return

    print(f"Found {len(test_files)} test files. Starting pytest runs...\n")
    print("-" * 60)

    summary = {
        "total": len(test_files),
        "passed": 0,
        "failed": 0,
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
            "python3",
            "-m",
            "pytest",
            "-v",
            "-s",
            file_path,
        ]

        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time

        if result.returncode == 0:
            status = "PASS"
            summary["passed"] += 1
        elif result.returncode == 1:
            status = "FAIL"
            summary["failed"] += 1
        elif result.returncode == 5:
            status = "NO TESTS"
            summary["no_tests"] += 1
        else:
            status = f"ERROR (Code: {result.returncode})"
            summary["errored_or_interrupted"] += 1

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
        f"No tests: {summary['no_tests']} | "
        f"Errors/interrupted: {summary['errored_or_interrupted']}"
    )
    print(f"Total duration: {summary['total_duration_seconds']} seconds")
    print(f"Detailed logs saved under '{LOG_DIR}'.")


if __name__ == "__main__":
    main()
