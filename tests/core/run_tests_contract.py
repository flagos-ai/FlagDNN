#!/usr/bin/env python3

"""Core contracts for tools/run_tests.py result validation."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import re
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location(
        "flagdnn_run_tests_contract_subject", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load runner: {path}")
    module = importlib.util.module_from_spec(spec)
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
    return module


def timing(case: str, provider: str) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "kind": "steady_state",
            "provider": provider,
            "case": case,
            "unit": "us",
            "median": 2.0,
            "p90": 3.0,
            "samples": [1.0, 2.0, 3.0],
        },
        separators=(",", ":"),
    )


def invoke_main(runner, arguments: list[str]) -> tuple[int, str, str]:
    original_argv = sys.argv
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        sys.argv = [str(runner.__file__), *arguments]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
            stderr
        ):
            exit_code = runner.main()
    finally:
        sys.argv = original_argv
    return exit_code, stdout.getvalue(), stderr.getvalue()


def check_real_ctest_selectors(adapter) -> None:
    ctest = shutil.which("ctest")
    if ctest is None:
        adjacent = Path(sys.executable).with_name("ctest")
        if adjacent.is_file():
            ctest = str(adjacent)
    require(ctest is not None, "CTest is required to verify test selectors")
    # CTest uses its own regular-expression engine. Python re accepting a
    # selector does not prove that CTest can parse it or discover its tests.
    functional = [
        "functional.nvidia.matmul" + suffix
        for suffix in (
            "",
            ".ieee",
            ".tf32",
            ".fp8",
            ".host",
            ".precision",
            "_fp8",
        )
    ]
    suffixes = (
        "",
        ".boolean",
        ".copy",
        ".ieee",
        ".tf32",
        ".fp32_output",
        ".fp8",
        ".int32",
        ".host",
        ".precision",
        ".unknown",
    )
    operators = ("add", "matmul", "identity", "reduction")
    benchmark = [
        "benchmark.nvidia." + operator + suffix
        for operator in operators
        for suffix in suffixes
    ]
    with tempfile.TemporaryDirectory(prefix="flagdnn-ctest-selectors-") as tmp:
        directory = Path(tmp)
        (directory / "CTestTestfile.cmake").write_text(
            "\n".join(
                f'add_test({name} "{sys.executable}" "-c" "pass")'
                for name in functional + benchmark
            )
            + "\n",
            encoding="utf-8",
        )
        checks = [
            ("functional", "matmul", set(functional[:4])),
            *[
                (
                    "benchmark",
                    operator,
                    {
                        "benchmark.nvidia." + operator + suffix
                        for suffix in suffixes[:7]
                    },
                )
                for operator in operators
            ],
        ]
        for suite, operator, expected in checks:
            output = subprocess.run(
                [
                    ctest,
                    "--test-dir",
                    str(directory),
                    "--show-only=json-v1",
                    "-R",
                    adapter.test_expression(suite, operator),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=20,
            )
            try:
                selected = {
                    test["name"] for test in json.loads(output.stdout)["tests"]
                }
            except (ValueError, KeyError) as error:
                raise RuntimeError(
                    f"CTest rejected {suite}/{operator} selector: "
                    + output.stdout
                    + output.stderr
                ) from error
            require(
                selected == expected,
                f"CTest selected wrong tests for {suite}/{operator}: "
                f"{sorted(selected)} != {sorted(expected)}",
            )


def check_legacy_summary(runner) -> None:
    # These keys and values are the public contract of the former runner,
    # independent of the native diagnostic document and its internal names.
    report = runner
    case = "add_perf_bfloat16_2x3_by_2x3"
    pair = {
        name: {"median": value, "unit": "us"}
        for name, value in (("flagdnn", 2.0), ("reference", 6.0))
    }
    functional = {
        "status": "passed",
        "duration_seconds": 0.25,
        "exit_code": 0,
        "case_counts": {
            "passed": 12,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "xfailed": 0,
            "xpassed": 0,
        },
    }
    benchmark = {
        "status": "passed",
        "duration_seconds": 1.5,
        "exit_code": 0,
        "test_case": "benchmark/test_add.cpp",
        "records": {case: pair, case + "_strided": pair},
    }
    native = {
        "timestamp_utc": "2026-09-14T00:00:00+00:00",
        "overall_status": "passed",
        "exit_code": 0,
        "env": {"python": "3.12.0"},
        "suite_operators": {"functional": ["add"], "benchmark": ["add"]},
        "results": {"add": {"functional": functional, "benchmark": benchmark}},
    }
    public = report.summary_document(native)
    require(
        set(public) == {"timestamp", "env", "selected_suites", "result"},
        "legacy report top-level contract changed",
    )
    require(
        re.fullmatch(
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", public["timestamp"]
        )
        is not None,
        "legacy timestamp is not a local second-resolution string",
    )
    require(
        public["selected_suites"] == {"add": ["accuracy", "performance"]},
        "legacy suite mapping changed",
    )
    op = public["result"]["add"]
    require(
        op["implemented"] is True
        and set(op) == {"implemented", "accuracy", "performance"},
        "legacy operator keys changed",
    )
    require(
        op["accuracy"]
        == {
            "status": "Passed",
            "duration": 0.25,
            "exit_code": 0,
            "total": 12,
            **functional["case_counts"],
        },
        "legacy accuracy counts/status/duration changed",
    )
    perf = op["performance"]
    require(
        set(perf) == {"status", "duration", "exit_code", "test_case", "data"},
        "native performance fields leaked into legacy suite",
    )
    dtype = perf["data"]["bf16"]
    require(
        set(dtype) == {"result", "details", "speedup"}
        and dtype["result"] == "OK"
        and dtype["speedup"] == 3.0,
        "legacy dtype grouping or arithmetic mean changed",
    )
    require(
        len(dtype["details"]) == 2,
        "same-shape layout variants were overwritten",
    )
    require(
        dtype["details"][case]
        == {"base": 0.006, "flag_dnn": 0.002, "gems": 0.002, "speedup": 3.0},
        "legacy latency names, millisecond units or speedup changed",
    )
    rounding_records = {
        f"add_fp32_case{index}": {
            "flagdnn": {"median": 1.0, "unit": "us"},
            "reference": {"median": speedup, "unit": "us"},
        }
        for index, speedup in enumerate((1.0e16, 1.0, 1.0))
    }
    require(
        report.performance_data({"records": rounding_records})["fp32"][
            "speedup"
        ]
        == 1.0e16 / 3,
        "legacy sequential floating-point average changed",
    )
    for state, expected in (
        ("failed", "Failed"),
        ("skipped", "Skipped"),
        ("timeout", "Timeout"),
        ("not_found", "NotFound"),
    ):
        row = report.suite_result(
            {"status": state, "duration_seconds": 2.0, "exit_code": None},
            "benchmark",
        )
        require(
            row["status"] == expected and row["data"] == {},
            "legacy non-success status changed",
        )
        if state == "timeout":
            require(
                row["exit_code"] == -100, "legacy timeout sentinel changed"
            )
        elif state == "not_found":
            require(row["exit_code"] == 5, "legacy no-tests sentinel changed")
    require(
        report.case_dtype("reduction_fp16_to_fp32_2_3") == "fp16"
        and report.case_dtype("matmul_ae4m3_be4m3_mode3_16_16_32_out_bf16")
        == "f8-e4m3fn",
        "output dtype was mistaken for input dtype",
    )
    require(
        public["env"]["flag_gems"] == public["env"]["flag_dnn"]
        and isinstance(public["env"]["triton"], dict),
        "psum_text environment aliases or fallback are incompatible",
    )
    for suite, task, missing in (
        ("functional", functional, "performance"),
        ("benchmark", benchmark, "accuracy"),
    ):
        partial = report.summary_document(
            {
                **native,
                "env": {"triton": None},
                "suite_operators": {suite: ["add"]},
                "results": {"add": {suite: task}},
            }
        )
        suites = partial["result"]["add"]
        require(
            suites[missing]["status"] == "Skipped"
            and suites[missing]["reason"] == "Suite not selected"
            and partial["selected_suites"]
            == {"add": [report.SUITE_NAMES[suite]]}
            and partial["env"]["triton"]["version"] == "unknown",
            "single-suite report fabricated execution or cannot be rendered",
        )
        # Match psum_text's mandatory multi-operator lookups.
        require(
            all(
                isinstance(suites["accuracy"][key], int)
                for key in ("passed", "failed", "skipped")
            )
            and isinstance(suites["performance"]["data"], dict),
            "single-suite report is missing mandatory psum_text fields",
        )
        if missing == "accuracy":
            require(
                suites[missing]["total"] == 0,
                "unselected functional suite was counted as executed",
            )
    fp8_data = report.performance_data(
        {
            "records": {
                f"matmul_a{fmt}_b{fmt}_16_16_32": pair
                for fmt in ("e4m3", "e5m2")
            }
        }
    )
    require(
        set(fp8_data) == {"f8-e4m3fn", "f8-e5m2"},
        "FP8 measurements would be omitted by psum_text's dtype columns",
    )
    output = (
        "12: a: reference PASS\n12: b: reference PASS\n"
        "12: FLAGDNN_SUITE: PASS cases=2\n"
        "12: FLAGDNN_SUITE: PASS cases=2\n"
        "13: c: reference PASS\n13: ERROR: stopped\n"
    )
    counts = report.functional_counts(output, "failed")
    require(
        counts["passed"] == 3
        and counts["errors"] == 1
        and counts["total"] == 4,
        "partial functional output was lost or counted twice",
    )
    for output, expected in (
        ("7: SUITE: PASS cases=10 executed=7 skipped=3\n", (7, 3)),
        ("7: SUITE: SKIP cases=10 executed=0 skipped=10\n", (0, 10)),
        (
            "8: SKIP case=unsupported_fp16 reason=not_supported\n"
            "8: SKIP case=unsupported_fp16 reason=not_supported\n"
            "8: SUITE: PASS cases=2 catalog_cases=3\n",
            (2, 1),
        ),
    ):
        counts = report.functional_counts(output, "passed")
        require(
            (counts["passed"], counts["skipped"]) == expected
            and counts["total"] == sum(expected),
            "native partial coverage was miscounted as successful cases",
        )
    with tempfile.TemporaryDirectory(prefix="flagdnn-legacy-report-") as tmp:
        path = Path(tmp) / "summary.json"
        runner.publish_summary(path, native)
        published = json.loads(path.read_text())
        require(
            published["result"]["add"]["performance"]["data"] == perf["data"],
            "published legacy data differs from the contract",
        )
        raw_path = (
            path.parent
            / published["result"]["add"]["performance"]["data_file"]
        )
        raw = json.loads(raw_path.read_text())["add"]
        require(
            raw["result"] == "passed"
            and raw["details"][0]["dtype"] == "torch.bfloat16"
            and raw["details"][0]["result"][0]["latency_base"] == 0.006,
            "legacy performance data_file is missing or incompatible",
        )
        require(
            json.loads(report.diagnostic_path(path).read_text()) == native,
            "native diagnostics lost information",
        )
        original_raw = raw_path.read_bytes()
        changed = json.loads(json.dumps(native))
        changed["results"]["add"]["benchmark"]["records"][case]["flagdnn"][
            "median"
        ] = 4.0
        # Separate reports and repeated publication of the same report must
        # both preserve the data referenced by an earlier copied summary.
        for destination in (path.with_name("second.json"), path):
            runner.publish_summary(destination, changed)
            require(
                raw_path.read_bytes() == original_raw,
                "publishing another report overwrote historical measurements",
            )

        runner.publish_summary(path, {"overall_status": "running"})
        old_public = path.read_bytes()
        old_native = report.diagnostic_path(path).read_bytes()
        original_write = runner.atomic_write_summary
        for failure_target in ("performance_result.json", path.name):
            directories = set(path.parent.iterdir())

            def fail_write(destination, document):
                if destination.name == failure_target:
                    raise OSError("injected publication failure")
                original_write(destination, document)

            runner.atomic_write_summary = fail_write
            try:
                try:
                    runner.publish_summary(path, native)
                except OSError:
                    pass
                else:
                    raise RuntimeError("publication failure was ignored")
            finally:
                runner.atomic_write_summary = original_write
            require(
                path.read_bytes() == old_public
                and report.diagnostic_path(path).read_bytes() == old_native
                and set(path.parent.iterdir()) == directories,
                "failed publication exposed success or left unused data files",
            )
        runner.publish_summary(path, {"overall_status": "running"})
        require(
            json.loads(path.read_text())["result"] == {},
            "starting another run left stale public success",
        )


def check_batch_runner(runner) -> None:
    """Exercise concurrent CTest processes and the public archive protocol."""
    with tempfile.TemporaryDirectory(prefix="flagdnn-batch-contract-") as tmp:
        root = Path(tmp)
        build = root / "build"
        build.mkdir()
        fixture = root / "case.py"
        fixture.write_text("""import json, os, sys, time
op, suite = sys.argv[1:]
gpu = os.environ['CUDA_VISIBLE_DEVICES']
print('GPU=' + gpu, flush=True)
print('fixture stderr ' + op, file=sys.stderr, flush=True)
if os.environ.get('FLAGDNN_CONTRACT_BLOCK') == op and suite == 'functional':
    with open(os.environ['FLAGDNN_CONTRACT_READY'], 'w') as stream:
        stream.write(str(os.getpid()))
    time.sleep(30)
time.sleep(0.05)
if suite == 'functional':
    print(op + '_fp32: reference PASS', flush=True)
    print(op + '_fp16: reference PASS', flush=True)
    print('FLAGDNN_' + op.upper() +
          '_FUNCTIONAL: PASS cases=2 executed=2 skipped=0')
else:
    for provider, latency in [('flagdnn', 2.0), ('cudnn', 6.0)]:
        print(json.dumps({'schema_version': 1, 'kind': 'steady_state',
            'provider': provider, 'case': op + '_fp32', 'unit': 'us',
            'median': latency, 'p90': latency, 'samples': [latency]}))
""")
        (build / "CMakeCache.txt").write_text(
            "FLAGDNN_BACKENDS:STRING=nvidia\n"
        )
        tests = []
        for operator in runner.DEFAULT_OPERATORS:
            for suite in runner.VALID_SUITES:
                tests.append(
                    {
                        "name": f"{suite}.nvidia.{operator}",
                        "command": [
                            sys.executable,
                            str(fixture),
                            operator,
                            suite,
                        ],
                        "properties": [
                            {"name": "WORKING_DIRECTORY", "value": str(root)}
                        ],
                    }
                )
        # Use a second directory so the fixture exercises inventory discovery
        # and recreation, not shared CTest logs between the GPU workers.
        source = root / "source"
        runner.isolated_ctest_directory(source, tests)
        shutil.copy(
            source / "CTestTestfile.cmake", build / "CTestTestfile.cmake"
        )
        output = root / "results"
        command = [
            sys.executable,
            str(Path(runner.__file__).resolve()),
            "--build-dir",
            str(build),
            "--output-dir",
            str(output),
            "--gpus",
            "1,2",
            "--color",
            "never",
        ]
        environment = dict(os.environ)
        environment.pop("FLAGDNN_BENCHMARK_PLATFORM", None)
        process = subprocess.run(
            command + ["--dump-output"],
            capture_output=True,
            text=True,
            env=environment,
            timeout=60,
        )
        require(
            process.returncode == 0,
            "batch run failed: " + process.stdout + process.stderr,
        )
        require(
            "\033[" not in process.stdout,
            "redirected batch output contains ANSI codes",
        )
        document = json.loads((output / "summary.json").read_text())
        require(
            set(document) == {"timestamp", "env", "result"},
            "batch summary envelope differs from FlagBLAS",
        )
        require(
            "flag_dnn" in document["env"]
            and "flag_gems" not in document["env"],
            "batch environment uses the wrong implementation key",
        )
        require(
            set(document["result"]) == set(runner.DEFAULT_OPERATORS),
            "batch defaults changed",
        )
        merged = {}
        for gpu in (1, 2):
            partition = json.loads((output / f"summary{gpu}.json").read_text())
            require(bool(partition), "GPU worker did not receive work")
            require(
                not (set(merged) & set(partition)),
                "operator ran on multiple GPUs",
            )
            merged.update(partition)
            for operator, result in partition.items():
                require(
                    set(result) == {"customized", "accuracy", "performance"},
                    "operator summary fields differ from FlagBLAS",
                )
                acc, perf = result["accuracy"], result["performance"]
                require(
                    set(acc)
                    == {
                        "total",
                        "passed",
                        "failed",
                        "skipped",
                        "details",
                        "status",
                        "exit_code",
                        "duration",
                        "data_file",
                    },
                    "accuracy summary fields differ from FlagBLAS",
                )
                require(
                    acc["total"] == acc["passed"] == 2
                    and acc["status"] == "Passed",
                    "native case counts were lost",
                )
                require(
                    set(perf)
                    == {
                        "duration",
                        "exit_code",
                        "data_file",
                        "data",
                        "status",
                        "test_case",
                    },
                    "performance summary fields differ from FlagBLAS",
                )
                row = perf["data"]["fp32"]["details"][operator + "_fp32"]
                require(
                    row == {"base": 0.006, "gems": 0.002, "speedup": 3.0},
                    "performance protocol or microsecond conversion changed",
                )
                names = {f.name for f in (output / operator).iterdir()}
                require(
                    names
                    == {
                        "accuracy_result.json",
                        "performance_result.log",
                        "accuracy_stdout.log",
                        "accuracy_stderr.log",
                        "performance_stdout.log",
                        "performance_stderr.log",
                    },
                    "operator archive file names differ from FlagBLAS",
                )
                for flavor in ("accuracy", "performance"):
                    log = (
                        output / operator / f"{flavor}_stdout.log"
                    ).read_text()
                    require(
                        "GPU=" + str(gpu) in log,
                        "worker GPU visibility leaked",
                    )
                raw = json.loads((output / acc["data_file"]).read_text())
                require(
                    len(raw) == acc["total"]
                    and all(
                        item["result"] == "passed" for item in raw.values()
                    ),
                    "raw accuracy data contradicts summary",
                )
                lines = (output / perf["data_file"]).read_text().splitlines()
                require(
                    lines[0] == "[INFO] Benchmark record logger enabled",
                    "performance preamble changed",
                )
                record = json.loads(lines[1][len("[INFO] ") :])
                require(
                    record["op_name"] == operator
                    and record["dtype"] == "torch.float32"
                    and record["result"][0]["latency"] == row["gems"],
                    "performance archive contradicts summary",
                )
        require(
            merged == document["result"],
            "final summary differs from GPU partitions",
        )
        require(
            "functional.nvidia."
            not in (
                build / "Testing" / "Temporary" / "LastTest.log"
            ).read_text(),
            "workers wrote shared CTest state",
        )
        # Reuse the output directory, leaving one idle GPU and removing dump
        # files from the selected operator. No stale operator may be merged.
        process = subprocess.run(
            command + ["--ops", "add"],
            capture_output=True,
            text=True,
            env=environment,
            timeout=60,
        )
        require(
            process.returncode == 0,
            "second batch run failed: " + process.stderr,
        )
        document = json.loads((output / "summary.json").read_text())
        require(
            set(document["result"]) == {"add"},
            "stale summary rows survived a rerun",
        )
        require(
            not list((output / "add").glob("*stdout.log")),
            "dump-output is not optional",
        )
        require(
            sum(
                bool(json.loads((output / f"summary{gpu}.json").read_text()))
                for gpu in (1, 2)
            )
            == 1,
            "idle GPU summary is missing or stale",
        )
        # An operator timeout must preserve output and allow its benchmark and
        # subsequent operators to run.
        ready = root / "ready"
        blocked_env = {
            **environment,
            "FLAGDNN_CONTRACT_BLOCK": "add",
            "FLAGDNN_CONTRACT_READY": str(ready),
        }
        process = subprocess.run(
            command + ["--ops", "add,sub", "--dump-output", "--timeout", "1"],
            capture_output=True,
            text=True,
            env=blocked_env,
            timeout=60,
        )
        document = json.loads((output / "summary.json").read_text())
        require(
            process.returncode == 1
            and document["result"]["add"]["accuracy"]["status"] == "Timeout",
            "batch timeout status was lost",
        )
        require(
            document["result"]["add"]["performance"]["status"] == "Passed"
            and document["result"]["sub"]["accuracy"]["status"] == "Passed",
            "timeout stopped unrelated work",
        )
        require(
            "GPU=" in (output / "add" / "accuracy_stdout.log").read_text(),
            "timeout lost partial output",
        )
        # Interrupt after a completed operator; retain that result and kill
        # the blocked native test while preserving its live stdout archive.
        ready.unlink(missing_ok=True)
        blocked_env["FLAGDNN_CONTRACT_BLOCK"] = "sub"
        interrupted = subprocess.Popen(
            command + ["--gpus", "1", "--ops", "add,sub", "--dump-output"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=blocked_env,
        )
        try:
            deadline = time.monotonic() + 30
            while not ready.exists() and time.monotonic() < deadline:
                require(
                    interrupted.poll() is None, "batch exited before interrupt"
                )
                time.sleep(0.05)
            require(ready.exists(), "native test did not start")
            child_pid = int(ready.read_text())
            live_log = output / "sub" / "accuracy_stdout.log"
            require(
                live_log.exists() and "GPU=1" in live_log.read_text(),
                "dump-output is not written while a command is running",
            )
            interrupted.terminate()
            stdout, stderr = interrupted.communicate(timeout=20)
            require(interrupted.returncode == 143, "batch lost SIGTERM status")
            document = json.loads((output / "summary.json").read_text())
            require(
                set(document["result"]) == {"add"},
                "interrupt lost completed results or merged stale rows",
            )
            identity = runner._linux_process_identity(child_pid)
            require(
                identity is None or identity[0] == "Z",
                "native process survived batch interruption",
            )
        finally:
            if interrupted.poll() is None:
                interrupted.kill()
                interrupted.communicate(timeout=20)

        # Invalid GPU lists must fail before launching any test.
        for value in ("", "1,", "1,1", "-1", "a"):
            process = subprocess.run(
                command + ["--gpus", value],
                capture_output=True,
                text=True,
                env=environment,
                timeout=10,
            )
            require(
                process.returncode == 2,
                "invalid GPU selection accepted: " + repr(value),
            )


def main() -> int:
    if len(sys.argv) != 2:
        raise RuntimeError("usage: run_tests_contract.py RUN_TESTS_PY")
    runner = load_runner(Path(sys.argv[1]).resolve())
    check_legacy_summary(runner)
    check_batch_runner(runner)
    hygon = runner.load_platform_adapter("hygon")
    ascend = runner.load_platform_adapter("ascend")
    nvidia = runner.load_platform_adapter("nvidia")
    check_real_ctest_selectors(nvidia)
    iluvatar = runner.load_platform_adapter("iluvatar")
    require(
        hygon is not None
        and ascend is not None
        and nvidia is not None
        and iluvatar is not None,
        "repository platform test adapters are incomplete",
    )
    runtime_record = json.loads(timing("runtime_profile", "flagdnn"))
    runtime_record.update(
        {
            "schema_version": 3,
            "host_submit_us": {
                "median": 2.0,
                "p90": 3.0,
                "samples": [1.0, 2.0, 3.0],
            },
            "build_us": 1000.0,
            "warm_build_us": 100.0,
            "workspace_bytes": 0,
            "build_cache": "fresh_artifact_cache",
        }
    )
    records, errors = runner.benchmark_records(
        json.dumps(runtime_record), nvidia
    )
    require(
        not errors and records["runtime_profile"]["flagdnn"] == runtime_record,
        "NVIDIA runtime metrics were rejected or lost",
    )
    native_record = {
        **runtime_record,
        "case": "native/test",
        "comparison": "none",
    }
    records, errors = runner.benchmark_records(
        json.dumps(native_record), nvidia
    )
    require(
        bool(errors) and not records,
        "NVIDIA accepted a benchmark record without a cuDNN comparison",
    )
    for field, invalid in (
        ("case", "ordinary"),
        ("case", "native/"),
        ("provider", "cudnn"),
        ("comparison", "unknown"),
        ("median", 0.0),
    ):
        _, errors = runner.benchmark_records(
            json.dumps({**native_record, field: invalid}), nvidia
        )
        require(bool(errors), f"invalid native record {field} was accepted")
    for operator in (
        "logical_not",
        "add",
        "matmul",
        "reduction",
        "matmul_fp8",
    ):
        expression = nvidia.test_expression("benchmark", operator)
        require(
            re.fullmatch(expression, f"benchmark.nvidia.{operator}")
            is not None,
            "base benchmark is missing from the NVIDIA test expression",
        )
        for suffix in (
            "boolean",
            "copy",
            "ieee",
            "tf32",
            "fp32_output",
            "fp8",
        ):
            require(
                re.fullmatch(
                    expression, f"benchmark.nvidia.{operator}.{suffix}"
                )
                is not None,
                "dtype benchmark is missing from the NVIDIA test expression",
            )
        require(
            re.fullmatch(
                expression, f"benchmark.nvidia.{operator}.unregistered"
            )
            is None,
            "NVIDIA test expression accepted an unknown dtype suffix",
        )
    for field, invalid in (
        ("schema_version", 99),
        ("schema_version", 3.0),
        ("workspace_bytes", -1),
        ("workspace_bytes", True),
        ("build_us", 0),
        ("warm_build_us", float("nan")),
        ("build_cache", "unknown"),
        ("build_cache", []),
        ("build_cache", {}),
        ("host_submit_us", {"median": 1.0, "p90": 1.0, "samples": [2.0]}),
    ):
        malformed = {**runtime_record, field: invalid}
        _, errors = runner.benchmark_records(json.dumps(malformed), nvidia)
        require(bool(errors), f"invalid runtime metric {field} was accepted")

    require(nvidia.SUPPORTS_MIN_SPEEDUP, "NVIDIA speedup gate is unavailable")

    def nvidia_outcome(
        records,
        threshold=0.9,
        *,
        reasons=None,
        status="passed",
        preflight=True,
    ):
        return nvidia.finalize(
            results={
                "add": {
                    "benchmark": {
                        "status": status,
                        "records": records,
                        "reference_unsupported": reasons or {},
                    }
                }
            },
            suite_operators={"benchmark": ["add"]},
            suites=["benchmark"],
            state={},
            min_speedup=threshold,
            preflight_passed=preflight,
        )

    pair = {
        "flagdnn": {"median": 10.0, "unit": "us"},
        "cudnn": {"median": 9.0, "unit": "us"},
    }
    exact = nvidia_outcome({"exact": pair})
    require(
        not exact["failed"] and exact["summary"]["performance"]["gate_passed"],
        "speedup exactly at 0.9 must pass",
    )
    native_providers = {"flagdnn": native_record}
    native_outcome = nvidia_outcome({"native/test": native_providers})
    require(
        native_outcome["failed"]
        and native_outcome["summary"]["comparable_coverage"]["incomplete"],
        "NVIDIA accepted a single-provider benchmark outcome",
    )
    mixed = nvidia_outcome({"exact": pair, "native/test": native_providers})
    require(
        mixed["failed"]
        and mixed["summary"]["performance"]["case_count"] == 1
        and not mixed["summary"]["comparable_coverage"]["complete"],
        "a valid pair concealed an unpaired NVIDIA benchmark",
    )
    for invalid in (0, float("nan"), float("inf"), True):
        require(
            nvidia_outcome(
                {
                    "native/test": {
                        "flagdnn": {**native_record, "median": invalid}
                    }
                }
            )["failed"],
            "invalid native timing passed finalization",
        )
    task = {"status": "passed"}
    nvidia.postprocess_result(
        result=task,
        ctest_reported_status="passed",
        output="",
        operator="add",
        suite="benchmark",
        records={"native/test": native_providers},
        manifest_operators=["add"],
    )
    require(
        task["status"] == "failed",
        "single-provider NVIDIA benchmark bypassed postprocessing",
    )
    slow_pair = {**pair, "cudnn": {"median": 8.0, "unit": "us"}}
    slow = nvidia_outcome(
        {
            "slow": slow_pair,
            "fast": {**pair, "cudnn": {"median": 90.0, "unit": "us"}},
        }
    )
    require(
        slow["failed"]
        and slow["summary"]["performance"]["failed_case_count"] == 1
        and slow["summary"]["performance"]["failures"][0]["case"] == "slow",
        "fast cases must not hide a below-target NVIDIA case",
    )
    unpaired = {"flagdnn": pair["flagdnn"]}
    mixed = nvidia_outcome(
        {"exact": pair, "unsupported": unpaired},
        reasons={"unsupported": "layout"},
    )
    require(
        mixed["failed"]
        and mixed["summary"]["performance"]["case_count"] == 1
        and mixed["summary"]["comparable_coverage"]["unsupported_case_count"]
        == 1
        and not mixed["summary"]["comparable_coverage"][
            "all_cases_comparable"
        ],
        "unsupported references must fail the NVIDIA benchmark run",
    )
    require(
        nvidia_outcome(
            {"exact": pair, "unsupported": unpaired},
            threshold=None,
            reasons={"unsupported": "layout"},
        )["failed"],
        "missing NVIDIA references must fail even without a speedup threshold",
    )
    for records, reasons in (
        ({}, {}),
        ({"missing": unpaired}, {}),
        ({"unsupported": unpaired}, {"unsupported": "layout"}),
    ):
        require(
            nvidia_outcome(records, reasons=reasons)["failed"],
            "empty/missing/unsupported-only NVIDIA timings "
            "produced a false-green gate",
        )
    for status in ("failed", "timeout", "skipped", "not_found"):
        require(
            nvidia_outcome({"exact": pair}, status=status)["failed"],
            "unsuccessful NVIDIA benchmark passed its speedup gate",
        )
    require(
        nvidia_outcome({"exact": pair}, preflight=False)["failed"],
        "NVIDIA speedup gate ignored failed preflight",
    )
    for invalid in (0.0, -1.0, float("nan"), float("inf"), True):
        require(
            nvidia_outcome(
                {
                    "invalid": {
                        **pair,
                        "flagdnn": {"median": invalid, "unit": "us"},
                    }
                }
            )["failed"],
            "invalid NVIDIA GPU timing passed its speedup gate",
        )
    require(
        nvidia_outcome(
            {
                "overflow": {
                    "flagdnn": {"median": 1.0e-300, "unit": "us"},
                    "cudnn": {"median": 1.0e300, "unit": "us"},
                }
            }
        )["failed"],
        "nonfinite NVIDIA speedup produced a false-green gate",
    )
    informational = nvidia_outcome({"slow": slow_pair}, threshold=None)
    require(
        not informational["failed"]
        and informational["summary"]["performance"]["gate_passed"] is None,
        "NVIDIA informational performance reporting must not "
        "invent a threshold",
    )
    for output, providers, expected in (
        ("1: case: cuDNN UNSUPPORTED: layout\n", unpaired, "failed"),
        ("case: cuDNN UNSUPPORTED: layout\n", unpaired, "failed"),
        ("", unpaired, "failed"),
        ("case: cuDNN UNSUPPORTED: layout\n", pair, "failed"),
    ):
        task = {"status": "passed"}
        nvidia.postprocess_result(
            result=task,
            ctest_reported_status="passed",
            output=output,
            operator="add",
            suite="benchmark",
            records={"case": providers},
            manifest_operators=["add"],
        )
        require(
            task["status"] == expected,
            "NVIDIA missing-reference evidence was mishandled",
        )
    require(
        hygon.PREFLIGHT_BY_DEFAULT
        and iluvatar.PREFLIGHT_BY_DEFAULT
        and not ascend.PREFLIGHT_BY_DEFAULT
        and not nvidia.PREFLIGHT_BY_DEFAULT
        and ascend.DEFAULT_TIMEOUT == 7200
        and nvidia.DEFAULT_TIMEOUT == 3600,
        "platform adapter defaults changed unexpectedly",
    )
    runner_source = Path(runner.__file__).read_text(encoding="utf-8").lower()
    for platform_detail in (
        "hygon",
        "hipdnn",
        "ascend",
        "aclnn",
        "hip_visible_devices",
        "npu_visible_devices",
        "iluvatar",
        "corex",
        "cuda_visible_devices",
    ):
        require(
            platform_detail not in runner_source,
            f"generic runner contains platform policy: {platform_detail}",
        )
    manifests = runner.operator_manifests()
    nvidia_manifests = nvidia.select_operator_manifests(manifests)
    unsupported_nvidia = {
        "genstats",
        "rng",
        "rope",
        "rope_backward",
        "moe_grouped_matmul_bwd",
    }
    require(
        set(nvidia_manifests["functional"])
        == set(manifests["functional"]) - unsupported_nvidia
        and set(nvidia_manifests["benchmark"])
        == set(manifests["functional"]) - unsupported_nvidia
        and not nvidia.FILTER_REGISTERED_TESTS,
        "NVIDIA cuDNN qualification must exclude unsupported standalone "
        "operators and retain missing-test errors",
    )
    manifest_operators = list(
        dict.fromkeys(
            operator
            for manifest in manifests.values()
            for operator in manifest
        )
    )
    repository_comparable_catalog = hygon.load_hygon_comparable_case_catalog(
        manifests["benchmark"]
    )
    require(
        repository_comparable_catalog["metric"] == hygon.SPEEDUP_METRIC
        and repository_comparable_catalog["declared_operator_count"] > 0
        and repository_comparable_catalog["declared_case_count"] > 0
        and repository_comparable_catalog["declared_case_count"]
        == sum(
            len(cases)
            for cases in repository_comparable_catalog["operators"].values()
        ),
        "repository Hygon comparable-case catalog is empty or inconsistent",
    )

    def provider_pair(case: str) -> dict[str, dict]:
        return {
            provider: json.loads(timing(case, provider))
            for provider in ("flagdnn", "hipdnn")
        }

    synthetic_catalog = {
        "schema_version": 1,
        "metric": hygon.SPEEDUP_METRIC,
        "source": "synthetic comparable coverage contract",
        "declared_operator_count": 2,
        "declared_case_count": 3,
        "operators": {
            "add": ["add_perf_required_a", "add_perf_required_b"],
            "mul": ["mul_perf_required"],
        },
    }
    partial_results = {
        "add": {
            "benchmark": {
                "status": "passed",
                "records": {
                    "add_perf_required_a": provider_pair("add_perf_required_a")
                },
            }
        }
    }
    missing_coverage = hygon.hygon_comparable_coverage(
        partial_results, ["add"], synthetic_catalog
    )
    require(
        not missing_coverage["verified"]
        and missing_coverage["required_case_count"] == 2
        and missing_coverage["observed_required_case_count"] == 1
        and missing_coverage["missing_case_count"] == 1
        and missing_coverage["missing_cases"][0]["case"]
        == "add_perf_required_b",
        "missing one declared comparable case did not fail coverage",
    )
    missing_performance = hygon.benchmark_speedup_summary(
        partial_results, 0.9, missing_coverage
    )
    require(
        missing_performance["ratio_gate_passed"]
        and not missing_performance["comparable_coverage_verified"]
        and not missing_performance["gate_passed"],
        "missing declared comparable case did not fail the speedup gate",
    )
    no_threshold_missing_performance = hygon.benchmark_speedup_summary(
        partial_results, None, missing_coverage
    )
    require(
        no_threshold_missing_performance["ratio_gate_passed"]
        and not no_threshold_missing_performance["coverage_gate_passed"]
        and not no_threshold_missing_performance["gate_passed"],
        "performance summary contradicted failed coverage without a ratio "
        "threshold",
    )
    omitted_coverage_performance = hygon.benchmark_speedup_summary(
        partial_results, 0.9
    )
    require(
        omitted_coverage_performance["ratio_gate_passed"]
        and not omitted_coverage_performance["coverage_gate_passed"]
        and not omitted_coverage_performance["gate_passed"],
        "speedup gate accepted an omitted comparable-coverage result",
    )

    complete_with_extra_results = {
        "add": {
            "benchmark": {
                "status": "passed",
                "records": {
                    case: provider_pair(case)
                    for case in (
                        "add_perf_required_a",
                        "add_perf_required_b",
                        "add_perf_newly_supported",
                    )
                },
            }
        }
    }
    complete_coverage = hygon.hygon_comparable_coverage(
        complete_with_extra_results, ["add"], synthetic_catalog
    )
    require(
        complete_coverage["verified"]
        and complete_coverage["required_case_count"] == 2
        and complete_coverage["observed_required_case_count"] == 2
        and complete_coverage["observed_pair_case_count"] == 3
        and complete_coverage["extra_pair_case_count"] == 1,
        "an extra comparable case was rejected or miscounted",
    )
    require(
        hygon.hygon_comparable_coverage(
            complete_with_extra_results, ["add"], synthetic_catalog
        )["verified"],
        "an unselected declared operator affected comparable coverage",
    )
    selected_missing = hygon.hygon_comparable_coverage(
        complete_with_extra_results, ["add", "mul"], synthetic_catalog
    )
    require(
        not selected_missing["verified"]
        and selected_missing["missing_case_count"] == 1
        and selected_missing["missing_cases"][0]["operator"] == "mul",
        "selected declared operator subset was not required exactly",
    )
    require(
        hygon.hygon_comparable_coverage(
            {"pow": {"benchmark": {"status": "skipped", "records": {}}}},
            ["pow"],
            synthetic_catalog,
        )["verified"],
        "selected operator absent from the declaration caused a false failure",
    )

    flagdnn_record = json.loads(timing("add_perf", "flagdnn"))
    hipdnn_record = json.loads(timing("add_perf", "hipdnn"))
    flagdnn_record["median"] = 2.0
    hipdnn_record["median"] = 1.5
    performance = hygon.benchmark_speedup_summary(
        {
            "add": {
                "benchmark": {
                    "status": "passed",
                    "records": {
                        "add_perf": {
                            "flagdnn": flagdnn_record,
                            "hipdnn": hipdnn_record,
                        }
                    },
                }
            }
        },
        0.9,
    )
    require(
        performance["case_count"] == 1
        and performance["failed_case_count"] == 1
        and not performance["gate_passed"]
        and performance["minimum_speedup"] == 0.75
        and performance["failures"][0]["case"] == "add_perf",
        "per-case speedup gate did not fail closed below threshold",
    )
    failed_performance = hygon.benchmark_speedup_summary(
        {
            "add": {
                "benchmark": {
                    "status": "failed",
                    "records": {
                        "add_perf": {
                            "flagdnn": flagdnn_record,
                            "hipdnn": hipdnn_record,
                        }
                    },
                }
            }
        },
        0.9,
    )
    require(
        failed_performance["case_count"] == 0
        and not failed_performance["gate_passed"],
        "speedup gate accepted partial records from a failed benchmark",
    )
    empty_performance = hygon.benchmark_speedup_summary({}, 0.9)
    require(
        empty_performance["case_count"] == 0
        and not empty_performance["gate_passed"],
        "speedup gate accepted an empty comparable-case set",
    )

    for empty_ops in ("", ",", " , , "):
        try:
            runner.requested_operators(
                manifests, ["functional"], empty_ops, None
            )
        except ValueError as error:
            require(
                "at least one operator" in str(error),
                f"empty --ops produced an unclear error: {error}",
            )
        else:
            raise RuntimeError(f"empty --ops accepted: {empty_ops!r}")

    for invalid_suites in (
        "",
        ",",
        "all,functional",
        "functional,all",
        "benchmrak",
    ):
        try:
            runner.requested_suites(invalid_suites)
        except ValueError:
            pass
        else:
            raise RuntimeError(
                f"invalid --suites accepted: {invalid_suites!r}"
            )

    with tempfile.TemporaryDirectory(
        prefix="flagdnn-runner-contract-"
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)

        def catalog_document(operators: dict[str, list[str]]) -> dict:
            return {
                "schema_version": 1,
                "platform": "hygon",
                "suite": "benchmark",
                "metric": hygon.SPEEDUP_METRIC,
                "source": "synthetic catalog parser contract",
                "declared_operator_count": len(operators),
                "declared_case_count": sum(map(len, operators.values())),
                "operators": operators,
            }

        valid_catalog_path = temporary_root / "valid-comparable-cases.json"
        valid_catalog_path.write_text(
            json.dumps(
                catalog_document(
                    {
                        "add": ["add_perf_required"],
                        "mul": ["mul_perf_required"],
                    }
                )
            ),
            encoding="utf-8",
        )
        parsed_catalog = hygon.load_hygon_comparable_case_catalog(
            manifests["benchmark"], valid_catalog_path
        )
        require(
            parsed_catalog["declared_operator_count"] == 2
            and parsed_catalog["declared_case_count"] == 2,
            "valid comparable-case catalog was rejected",
        )
        require(
            parsed_catalog["catalog_path"]
            == str(valid_catalog_path.resolve()),
            "parsed comparable-case catalog lost its source path",
        )

        malformed_catalogs = {
            "wrong-owner": catalog_document({"add": ["mul_perf_wrong_owner"]}),
            "duplicate-case": catalog_document(
                {"add": ["add_perf_duplicate", "add_perf_duplicate"]}
            ),
            "unknown-operator": catalog_document(
                {"not_a_manifest_operator": ["not_a_manifest_operator_perf"]}
            ),
            "empty-source": {
                **catalog_document({"add": ["add_perf_required"]}),
                "source": "",
            },
            "wrong-metric": {
                **catalog_document({"add": ["add_perf_required"]}),
                "metric": "flagdnn/hipdnn",
            },
            "boolean-schema-version": {
                **catalog_document({"add": ["add_perf_required"]}),
                "schema_version": True,
            },
            "empty-catalog": catalog_document({}),
            "count-mismatch": {
                **catalog_document({"add": ["add_perf_required"]}),
                "declared_case_count": 2,
            },
        }
        for label, malformed_catalog in malformed_catalogs.items():
            malformed_path = temporary_root / f"{label}.json"
            malformed_path.write_text(
                json.dumps(malformed_catalog), encoding="utf-8"
            )
            try:
                hygon.load_hygon_comparable_case_catalog(
                    manifests["benchmark"], malformed_path
                )
            except RuntimeError:
                pass
            else:
                raise RuntimeError(
                    f"malformed comparable-case catalog accepted: {label}"
                )

        duplicate_key_path = temporary_root / "duplicate-key.json"
        duplicate_key_path.write_text(
            '{"schema_version":1,"schema_version":1}', encoding="utf-8"
        )
        try:
            hygon.load_hygon_comparable_case_catalog(
                manifests["benchmark"], duplicate_key_path
            )
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "comparable-case catalog with duplicate JSON key was accepted"
            )

        empty_list = temporary_root / "empty-ops.txt"
        empty_list.write_text("\n  # comment only\n# also a comment\n")
        try:
            runner.requested_operators(
                manifests, ["functional"], None, empty_list
            )
        except ValueError as error:
            require(
                "at least one operator" in str(error),
                f"empty op list produced an unclear error: {error}",
            )
        else:
            raise RuntimeError("empty/comment-only op list was accepted")

        summary_path = temporary_root / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "overall_status": "passed",
                    "exit_code": 0,
                    "stale": True,
                }
            )
        )
        observed_states: list[dict] = []
        original_replace = runner.os.replace

        def recording_replace(source, destination) -> None:
            original_replace(source, destination)
            if Path(destination) == runner.diagnostic_path(summary_path):
                observed_states.append(
                    json.loads(Path(destination).read_text(encoding="utf-8"))
                )

        runner.os.replace = recording_replace
        try:
            exit_code, _, stderr = invoke_main(
                runner,
                ["--ops", ",", "--output", str(summary_path)],
            )
        finally:
            runner.os.replace = original_replace
        require(exit_code == 2, "empty --ops main path did not fail")
        require("at least one operator" in stderr, "empty --ops error hidden")
        require(
            len(observed_states) >= 2
            and all(
                state["overall_status"] == "running"
                for state in observed_states[:-1]
            )
            and observed_states[-1]["overall_status"] == "failed",
            "summary did not atomically invalidate stale success before "
            "publishing early failure",
        )
        early_failure = json.loads(
            runner.diagnostic_path(summary_path).read_text(encoding="utf-8")
        )
        require(
            early_failure["schema_version"] == 2
            and early_failure["overall_status"] == "failed"
            and early_failure["exit_code"] == 2
            and not early_failure.get("stale", False),
            "early validation summary retained stale success state",
        )
        require(
            not list(temporary_root.glob(".summary.json.*.tmp")),
            "atomic summary publication leaked a temporary file",
        )

        exit_code, _, _ = invoke_main(
            runner, ["--list", "--output", str(summary_path)]
        )
        listed_summary = json.loads(
            runner.diagnostic_path(summary_path).read_text(encoding="utf-8")
        )
        require(
            exit_code == 0
            and listed_summary["schema_version"] == 2
            and listed_summary["overall_status"] == "passed"
            and listed_summary["exit_code"] == 0,
            "successful terminal summary lacks schema-v2 status fields",
        )

        summary_path.write_text(
            '{"schema_version":2,"overall_status":"passed","exit_code":0}\n'
        )
        exit_code, _, stderr = invoke_main(
            runner,
            [
                "--timeout",
                "not-an-integer",
                "--output",
                str(summary_path),
            ],
        )
        argument_failure = json.loads(
            runner.diagnostic_path(summary_path).read_text(encoding="utf-8")
        )
        require(
            exit_code == 2
            and "invalid int value" in stderr
            and argument_failure["overall_status"] == "failed"
            and argument_failure["exit_code"] == 2,
            "argparse failure left a stale successful summary",
        )

        exit_code, _, _ = invoke_main(
            runner, ["--help", "--output", str(summary_path)]
        )
        help_summary = json.loads(
            runner.diagnostic_path(summary_path).read_text(encoding="utf-8")
        )
        require(
            exit_code == 0
            and help_summary["overall_status"] == "passed"
            and help_summary["exit_code"] == 0
            and help_summary.get("mode") == "help",
            "--help summary can be confused with a completed test run",
        )

        fake_build = temporary_root / "build"
        fake_build.mkdir()
        (fake_build / "CTestTestfile.cmake").write_text("# contract\n")

        configuration_root = temporary_root / "configuration"
        configuration_root.mkdir()
        (configuration_root / "CMakeCache.txt").write_text(
            "CMAKE_BUILD_TYPE:STRING=Debug\n",
            encoding="utf-8",
        )
        (configuration_root / runner.BUILD_CONFIGURATION_FILE).write_text(
            "Release\n", encoding="utf-8"
        )
        require(
            runner.resolve_build_configuration(configuration_root, None)
            == "Debug",
            "single-config CMAKE_BUILD_TYPE did not override a stale marker",
        )
        (configuration_root / "CMakeCache.txt").write_text(
            "CMAKE_BUILD_TYPE:STRING=\n"
            "CMAKE_CONFIGURATION_TYPES:STRING=Debug;Release\n",
            encoding="utf-8",
        )
        require(
            runner.resolve_build_configuration(configuration_root, None)
            == "Release"
            and runner.resolve_build_configuration(
                configuration_root, "RelWithDebInfo"
            )
            == "RelWithDebInfo",
            "multi-config build configuration was not propagated",
        )

        original_catalog_loader = hygon.load_hygon_comparable_case_catalog

        def reject_catalog(*_arguments, **_keywords):
            raise RuntimeError("synthetic malformed comparable catalog")

        hygon.load_hygon_comparable_case_catalog = reject_catalog
        try:
            exit_code, _, stderr = invoke_main(
                runner,
                [
                    "--platform",
                    "hygon",
                    "--build-dir",
                    str(fake_build),
                    "--suites",
                    "benchmark",
                    "--ops",
                    "add",
                    "--no-preflight",
                    "--output",
                    str(summary_path),
                ],
            )
        finally:
            hygon.load_hygon_comparable_case_catalog = original_catalog_loader
        malformed_main_summary = json.loads(
            runner.diagnostic_path(summary_path).read_text(encoding="utf-8")
        )
        require(
            exit_code == 2
            and "malformed comparable catalog" in stderr
            and malformed_main_summary["overall_status"] == "failed"
            and malformed_main_summary["exit_code"] == 2,
            "main did not fail closed when comparable catalog loading failed",
        )

        original_run_one = runner.run_one
        required_add_case = repository_comparable_catalog["operators"]["add"][
            0
        ]

        def partial_add_benchmark(**_arguments):
            return {
                "status": "passed",
                "duration_seconds": 0.0,
                "exit_code": 0,
                "command": [],
                "records": {
                    required_add_case: provider_pair(required_add_case)
                },
            }

        runner.run_one = partial_add_benchmark
        try:
            exit_code, _, _ = invoke_main(
                runner,
                [
                    "--platform",
                    "hygon",
                    "--build-dir",
                    str(fake_build),
                    "--suites",
                    "benchmark",
                    "--ops",
                    "add",
                    "--no-preflight",
                    "--output",
                    str(summary_path),
                ],
            )
        finally:
            runner.run_one = original_run_one
        partial_main_summary = json.loads(
            runner.diagnostic_path(summary_path).read_text(encoding="utf-8")
        )
        require(
            exit_code == 1
            and partial_main_summary["overall_status"] == "failed"
            and not partial_main_summary["comparable_coverage"]["verified"]
            and partial_main_summary["performance"]["threshold"] is None,
            "main accepted incomplete comparable coverage without "
            "--min-speedup",
        )

        def skipped_undeclared_benchmark(**_arguments):
            return {
                "status": "skipped",
                "duration_seconds": 0.0,
                "exit_code": 0,
                "command": [],
                "records": {},
            }

        runner.run_one = skipped_undeclared_benchmark
        try:
            exit_code, _, _ = invoke_main(
                runner,
                [
                    "--platform",
                    "hygon",
                    "--build-dir",
                    str(fake_build),
                    "--suites",
                    "benchmark",
                    "--ops",
                    "pow",
                    "--no-preflight",
                    "--output",
                    str(summary_path),
                ],
            )
        finally:
            runner.run_one = original_run_one
        undeclared_main_summary = json.loads(
            runner.diagnostic_path(summary_path).read_text(encoding="utf-8")
        )
        undeclared_coverage = undeclared_main_summary["comparable_coverage"]
        require(
            exit_code == 0
            and undeclared_main_summary["overall_status"] == "passed"
            and undeclared_coverage["verified"]
            and undeclared_coverage["required_case_count"] == 0,
            "main rejected an undeclared unsupported operator",
        )

        def passed_hygon_functional(**_arguments):
            return {
                "status": "passed",
                "duration_seconds": 0.0,
                "exit_code": 0,
                "command": [],
            }

        runner.run_one = passed_hygon_functional
        try:
            exit_code, _, _ = invoke_main(
                runner,
                [
                    "--platform",
                    "hygon",
                    "--build-dir",
                    str(fake_build),
                    "--suites",
                    "functional",
                    "--ops",
                    "add",
                    "--no-preflight",
                    "--output",
                    str(summary_path),
                ],
            )
        finally:
            runner.run_one = original_run_one
        functional_only_summary = json.loads(
            runner.diagnostic_path(summary_path).read_text(encoding="utf-8")
        )
        require(
            exit_code == 0
            and functional_only_summary["overall_status"] == "passed"
            and functional_only_summary["comparable_coverage"] is None
            and functional_only_summary["performance"] is None,
            "Hygon functional-only run emitted a benchmark performance "
            "summary",
        )

        def skipped_nvidia_functional(**_arguments):
            return {
                "status": "skipped",
                "duration_seconds": 0.0,
                "exit_code": 0,
                "command": [],
            }

        runner.run_one = skipped_nvidia_functional
        try:
            exit_code, _, _ = invoke_main(
                runner,
                [
                    "--platform",
                    "nvidia",
                    "--build-dir",
                    str(fake_build),
                    "--ops",
                    "add",
                    "--no-preflight",
                    "--output",
                    str(summary_path),
                ],
            )
        finally:
            runner.run_one = original_run_one
        nvidia_skip_summary = json.loads(
            runner.diagnostic_path(summary_path).read_text(encoding="utf-8")
        )
        require(
            exit_code == 1
            and nvidia_skip_summary["overall_status"] == "failed"
            and nvidia_skip_summary["performance"] is None,
            "non-Hygon skipped suite produced a false-green result",
        )

        original_run_preflight = runner.run_preflight
        run_one_called = False

        def failed_preflight(**_arguments):
            return {
                "status": "failed",
                "duration_seconds": 0.0,
                "exit_code": 1,
                "command": [],
                "required_tests": [],
                "missing_tests": ["synthetic.contract"],
                "errors": ["synthetic preflight failure"],
                "visibility_masks": {},
            }

        def forbidden_after_preflight(**_arguments):
            nonlocal run_one_called
            run_one_called = True
            raise RuntimeError("operator ran after failed preflight")

        runner.run_preflight = failed_preflight
        runner.run_one = forbidden_after_preflight
        try:
            exit_code, _, _ = invoke_main(
                runner,
                [
                    "--platform",
                    "nvidia",
                    "--build-dir",
                    str(fake_build),
                    "--ops",
                    "add",
                    "--preflight",
                    "--output",
                    str(summary_path),
                ],
            )
        finally:
            runner.run_preflight = original_run_preflight
            runner.run_one = original_run_one
        require(
            exit_code == 1 and not run_one_called,
            "operator suites ran after a failed preflight",
        )

        original_requested_operators = runner.requested_operators
        runner.requested_operators = (
            lambda selected_manifests, selected_suites, value, list_file: {
                suite: [] for suite in selected_suites
            }
        )
        try:
            exit_code, _, stderr = invoke_main(
                runner,
                [
                    "--build-dir",
                    str(fake_build),
                    "--no-preflight",
                    "--output",
                    str(summary_path),
                ],
            )
        finally:
            runner.requested_operators = original_requested_operators
        require(exit_code == 2, "zero-run defensive check did not fail")
        require(
            "zero operator/suite runs" in stderr,
            "zero-run defensive failure was not explicit",
        )

        summary_path.write_text(
            '{"schema_version":2,"overall_status":"passed","exit_code":0}\n'
        )
        original_operator_manifests = runner.operator_manifests

        def unexpected_failure():
            raise AssertionError("synthetic unexpected failure")

        runner.operator_manifests = unexpected_failure
        try:
            try:
                invoke_main(runner, ["--output", str(summary_path)])
            except AssertionError:
                pass
            else:
                raise RuntimeError(
                    "synthetic unexpected failure was swallowed"
                )
        finally:
            runner.operator_manifests = original_operator_manifests
        interrupted_summary = json.loads(
            runner.diagnostic_path(summary_path).read_text(encoding="utf-8")
        )
        require(
            interrupted_summary["overall_status"] == "running"
            and interrupted_summary["exit_code"] is None,
            "unexpected failure left a stale successful summary",
        )

    hygon_preflight = runner.required_preflight_tests("hygon")
    for required_contract in (
        "core.kernel_registry_contract",
        "integration.hygon.jit_candidate_compatibility_contract",
        "integration.hygon.jit_global_state_contract",
        "integration.hygon.convolution_validation_static_contract",
        "integration.hygon.normalization_stability",
    ):
        require(
            required_contract in hygon_preflight,
            f"Hygon preflight omits {required_contract}",
        )
    require(
        len(hygon_preflight) == 29,
        "Hygon preflight required-test count is not 29",
    )
    functional_preflight = runner.required_preflight_tests(
        "hygon", ["functional"]
    )
    require(
        "benchmark.catalog_contract" not in functional_preflight
        and "benchmark.catalog_dependency_boundary"
        not in functional_preflight,
        "functional-only preflight still requires benchmark targets",
    )
    nvidia_preflight = runner.required_preflight_tests(
        "nvidia", ["functional"]
    )
    for required_contract in (
        "core.autotune_policy",
        "core.library_abi",
        "core.build_environment.parallel",
        "core.build_environment.parallel_handles",
        "core.build_environment.parallel_cold_handles",
        "core.build_environment.fail",
        "core.build_environment.not_ready",
        "integration.nvidia.compiler_contract",
        "integration.nvidia.artifact_contract",
        "integration.nvidia.tensor_map_first",
        "integration.nvidia.short_map_first",
        "integration.nvidia.execution_contract",
        "integration.nvidia.dependency_boundary",
        "integration.nvidia.reference_dependency_boundary",
        "integration.nvidia.runtime",
        "integration.nvidia.graph",
    ):
        require(
            required_contract in nvidia_preflight,
            f"NVIDIA preflight omits {required_contract}",
        )

    operator_command = runner.ctest_command(
        Path("/tmp/flagdnn-build"), "add", "benchmark", "hygon"
    )
    preflight_command = runner.verbose_ctest_command(
        Path("/tmp/flagdnn-build"),
        r"^(core\.|integration\.hygon\.)",
        "Debug",
    )
    for label, command in (
        ("operator", operator_command),
        ("preflight", preflight_command),
    ):
        require(command.count("-V") == 1, f"{label} CTest is not verbose")
        require(
            "--output-on-failure" not in command,
            f"{label} CTest combines -V with --output-on-failure",
        )
    require(
        preflight_command[-2:] == ["-C", "Debug"],
        "CTest configuration was not propagated",
    )
    nvidia_matmul_command = runner.ctest_command(
        Path("/tmp/flagdnn-build"),
        "matmul",
        "functional",
        "nvidia",
    )
    nvidia_matmul_expression = nvidia_matmul_command[
        nvidia_matmul_command.index("-R") + 1
    ]
    require(
        all(
            re.fullmatch(
                nvidia_matmul_expression, "functional.nvidia.matmul" + suffix
            )
            for suffix in ("", ".ieee", ".tf32", ".fp8")
        ),
        "NVIDIA matmul selection omits a functional dtype branch",
    )

    discovery_commands: list[list[str]] = []
    original_run_process_group = runner.run_process_group

    def synthetic_inventory(command, _environment, _timeout):
        discovery_commands.append(command)
        return (
            json.dumps(
                {
                    "tests": [
                        {"name": "functional.synthetic.add"},
                        {"name": "benchmark.synthetic.matmul"},
                    ]
                }
            ),
            "",
            0,
            False,
        )

    runner.run_process_group = synthetic_inventory
    try:
        filtered_manifests = runner.registered_manifests(
            Path("/tmp/flagdnn-build"),
            "synthetic",
            manifests,
            ["functional", "benchmark"],
            {},
            1800,
            "Debug",
        )
    finally:
        runner.run_process_group = original_run_process_group
    require(
        filtered_manifests
        == {
            "functional": ["add"],
            "benchmark": ["matmul"],
        }
        and discovery_commands[0][-2:] == ["-C", "Debug"],
        "configured CTest inventory was not filtered generically",
    )

    records, errors = runner.benchmark_records(
        timing("add_fp32_2x3", "flagdnn")
        + "\n"
        + timing("add_fp32_2x3", "hipdnn")
    )
    require(not errors, f"valid timing records rejected: {errors}")
    diagnostic_records, diagnostic_errors = runner.benchmark_records(
        '{"kind":"diagnostic","message":"backend trace"}\n'
        + timing("add_fp32_2x3", "flagdnn")
    )
    require(
        not diagnostic_errors and set(diagnostic_records) == {"add_fp32_2x3"},
        "non-timing structured diagnostic was treated as benchmark data",
    )
    _, malformed_timing_errors = runner.benchmark_records(
        '{"kind":"steady_state","message":"missing timing fields"}'
    )
    require(
        malformed_timing_errors,
        "malformed steady-state timing record was silently ignored",
    )

    ascend_metric = {
        "median": 2.0,
        "p90": 3.0,
        "samples": [1.0, 2.0, 3.0],
    }
    ascend_record = {
        "schema_version": 2,
        "kind": "steady_state",
        "provider": "flagdnn",
        "case": "add_fp32_2x3",
        "environment": {
            "soc_fingerprint": "synthetic-soc",
            "cann_package_version": "synthetic-cann",
            "ascendcl_build_id": "synthetic-ascendcl",
            "runtime_build_id": "synthetic-runtime",
        },
        "provider_identity": {
            "libtriton_jit_sha256": "0" * 64,
            "compiler_identity_sha256": "1" * 64,
            "artifact_request_sha256": "2" * 64,
            "launch_abi": "ltj_npu_raw_v1",
            "selected_candidate": "synthetic-candidate",
        },
        "benchmark_config": {
            "warmup_iterations": 0,
            "sample_count": 3,
            "iterations_per_sample": 1,
        },
        "stream_us": ascend_metric,
        "submit_us": ascend_metric,
        "end_to_end_us": ascend_metric,
    }
    ascend_records, ascend_errors = runner.benchmark_records(
        json.dumps(ascend_record), ascend
    )
    require(
        not ascend_errors and set(ascend_records) == {"add_fp32_2x3"},
        f"valid Ascend timing record rejected: {ascend_errors}",
    )
    invalid_identity = {
        **ascend_record,
        "provider_identity": {"unexpected": "identity"},
    }
    invalid_records, invalid_errors = runner.benchmark_records(
        json.dumps(invalid_identity), ascend
    )
    require(
        not invalid_records and invalid_errors,
        "invalid Ascend provider identity was accepted",
    )
    unowned_records, unowned_errors = runner.benchmark_records(
        json.dumps(ascend_record)
    )
    require(
        not unowned_records and unowned_errors,
        "generic runner accepted a platform schema without its adapter",
    )

    original_run_process_group = runner.run_process_group
    runner.run_process_group = lambda *_args, **_kwargs: ("", "", 0, False)
    try:
        empty_nvidia_benchmark = runner.run_one(
            build_dir=Path("/tmp/flagdnn-build"),
            operator="add",
            suite="benchmark",
            platform="nvidia",
            environment={},
            timeout=1,
            verbose=False,
            manifest_operators=manifest_operators,
        )
    finally:
        runner.run_process_group = original_run_process_group
    require(
        empty_nvidia_benchmark["status"] == "failed"
        and empty_nvidia_benchmark.get("record_errors"),
        "passed generic benchmark without timing records was accepted",
    )
    require(
        not hygon.validate_hygon_benchmark_pairs(
            records, "add", manifest_operators
        ),
        "valid Hygon benchmark pair rejected",
    )
    require(
        hygon.validate_hygon_benchmark_pairs(
            records, "mul", manifest_operators
        ),
        "wrong-operator benchmark case accepted",
    )
    for longer, shorter in (
        ("add_square", "add"),
        ("batchnorm_inference", "batchnorm"),
        ("sdpa_backward", "sdpa"),
    ):
        overlapping: dict[str, dict[str, dict[str, object]]] = {
            f"{longer}_fp32_case": {
                "flagdnn": {},
                "hipdnn": {},
            }
        }
        require(
            hygon.validate_hygon_benchmark_pairs(
                overlapping, shorter, manifest_operators
            ),
            f"{shorter} accepted a {longer} benchmark case",
        )

    for direction in ("fprop", "dgrad", "wgrad"):
        operator = f"conv_{direction}"
        for spatial_rank in (1, 2, 3):
            case = f"conv{spatial_rank}d_{direction}_fp32_case"
            convolution_records: dict[str, dict[str, dict[str, object]]] = {
                case: {"flagdnn": {}, "hipdnn": {}}
            }
            require(
                not hygon.validate_hygon_benchmark_pairs(
                    convolution_records, operator, manifest_operators
                ),
                f"{operator} rejected rank-qualified case {case}",
            )
            wrong_operator = (
                "conv_wgrad" if direction != "wgrad" else "conv_dgrad"
            )
            require(
                hygon.validate_hygon_benchmark_pairs(
                    convolution_records,
                    wrong_operator,
                    manifest_operators,
                ),
                f"{wrong_operator} accepted case {case}",
            )
            convolution_skip = hygon.hipdnn_skip_records(
                f"[SKIP][hipdnn] op={operator} case={case} "
                "reason=HIPDNN_STATUS_NOT_SUPPORTED"
            )
            require(
                not hygon.validate_hygon_skip_records(
                    convolution_skip, operator, manifest_operators
                ),
                f"{operator} rejected rank-qualified skip case {case}",
            )
            wrong_convolution_skip = hygon.hipdnn_skip_records(
                f"[SKIP][hipdnn] op={wrong_operator} case={case} "
                "reason=HIPDNN_STATUS_NOT_SUPPORTED"
            )
            require(
                hygon.validate_hygon_skip_records(
                    wrong_convolution_skip,
                    wrong_operator,
                    manifest_operators,
                ),
                f"{wrong_operator} accepted skip case {case}",
            )

    for invalid_case in (
        "conv4d_dgrad_fp32_case",
        "conv2d_dgradient_fp32_case",
        "xconv2d_dgrad_fp32_case",
    ):
        require(
            hygon.benchmark_case_operator(invalid_case, manifest_operators)
            is None,
            f"invalid convolution case alias accepted: {invalid_case}",
        )

    skip_line = (
        "[SKIP][hipdnn] op=add case=add_fp32_2x3 "
        "reason=HIPDNN_STATUS_NOT_SUPPORTED"
    )
    skips = hygon.hipdnn_skip_records(skip_line)
    require(
        not hygon.validate_hygon_skip_records(
            skips, "add", manifest_operators
        ),
        "valid Hygon skip rejected",
    )
    require(
        hygon.validate_hygon_skip_records(skips, "mul", manifest_operators),
        "wrong-operator Hygon skip accepted",
    )
    require(
        hygon.validate_hygon_skip_records(
            skips + skips, "add", manifest_operators
        ),
        "duplicate Hygon skip case accepted",
    )
    for malformed_skip in (
        "[SKIP][hipdnn] case=add_fp32_2x3 op=add "
        "reason=HIPDNN_STATUS_NOT_SUPPORTED",
        "[SKIP][hipdnn] reason=HIPDNN_STATUS_NOT_SUPPORTED "
        "op=add case=add_fp32_2x3",
        "[SKIP][hipdnn] op=add reason=missing-case " "case=add_fp32_2x3",
    ):
        malformed_records = hygon.hipdnn_skip_records(malformed_skip)
        require(
            malformed_records
            and hygon.validate_hygon_skip_records(
                malformed_records, "add", manifest_operators
            ),
            f"malformed structured SKIP was accepted: {malformed_skip}",
        )

    add_square_skip = hygon.hipdnn_skip_records(
        "[SKIP][hipdnn] op=add case=add_square_fp32_2x3 "
        "reason=HIPDNN_STATUS_NOT_SUPPORTED"
    )
    require(
        hygon.validate_hygon_skip_records(
            add_square_skip, "add", manifest_operators
        ),
        "add accepted an add_square skip case by prefix",
    )
    add_square_skip = hygon.hipdnn_skip_records(
        "[SKIP][hipdnn] op=add_square case=add_square_fp32_2x3 "
        "reason=HIPDNN_STATUS_NOT_SUPPORTED"
    )
    require(
        not hygon.validate_hygon_skip_records(
            add_square_skip, "add_square", manifest_operators
        ),
        "add_square rejected its own longest-manifest skip case",
    )

    benchmark_output = (
        timing("add_fp32_2x3", "flagdnn")
        + "\n"
        + timing("add_fp32_2x3", "hipdnn")
        + "\nFLAGDNN_ADD_BENCHMARK: PASS cases=1 executed=1 skipped=0\n"
    )
    require(
        not hygon.validate_hygon_case_accounting(
            benchmark_output,
            "add",
            "benchmark",
            "passed",
            records,
            [],
        ),
        "valid Hygon benchmark accounting rejected",
    )
    require(
        hygon.validate_hygon_case_accounting(
            "FLAGDNN_ADD_FUNCTIONAL: PASS executed=1 skipped=0",
            "add",
            "functional",
            "passed",
            {},
            [],
        ),
        "functional accounting without cases=<count> was accepted",
    )
    require(
        not hygon.validate_hygon_case_accounting(
            "FLAGDNN_ADD_FUNCTIONAL: PASS cases=1 executed=1 skipped=0",
            "add",
            "functional",
            "passed",
            {},
            [],
        ),
        "valid functional case accounting rejected",
    )
    require(
        hygon.validate_hygon_case_accounting(
            "FLAGDNN_ADD_FUNCTIONAL: PASS cases=0 executed=0 skipped=0",
            "add",
            "functional",
            "passed",
            {},
            [],
        ),
        "zero-case PASS accounting was accepted",
    )
    require(
        hygon.validate_hygon_case_accounting(
            "FLAGDNN_ADD_FUNCTIONAL: PASS cases=1 executed=0 skipped=1",
            "add",
            "functional",
            "passed",
            {},
            skips,
        ),
        "PASS accounting with no executed case was accepted",
    )
    require(
        not hygon.validate_hygon_case_accounting(
            "FLAGDNN_ADD_FUNCTIONAL: SKIP cases=1 executed=0 skipped=1",
            "add",
            "functional",
            "skipped",
            {},
            skips,
        ),
        "valid all-SKIP accounting was rejected",
    )
    require(
        hygon.validate_hygon_case_accounting(
            "FLAGDNN_ADD_FUNCTIONAL: SKIP cases=1 executed=1 skipped=0",
            "add",
            "functional",
            "skipped",
            {},
            [],
        ),
        "SKIP accounting with an executed case was accepted",
    )
    require(
        hygon.validate_hygon_case_accounting(
            "FLAGDNN_ADD_BENCHMARK: PASS cases=2 executed=2 skipped=0",
            "add",
            "benchmark",
            "passed",
            records,
            [],
        ),
        "incomplete Hygon benchmark coverage accepted",
    )
    require(
        hygon.validate_hygon_case_accounting(
            "FLAGDNN_MUL_FUNCTIONAL: PASS cases=1 executed=1 skipped=0",
            "add",
            "functional",
            "passed",
            {},
            [],
        ),
        "functional accounting accepted another operator's marker",
    )
    require(
        hygon.validate_hygon_case_accounting(
            "FLAGDNN_ADD_BENCHMARK: PASS cases=1 executed=1 skipped=0",
            "add",
            "functional",
            "passed",
            {},
            [],
        ),
        "functional accounting accepted the benchmark marker",
    )
    require(
        hygon.validate_hygon_case_accounting(
            "FLAGDNN_ADD_FUNCTIONAL: PASS cases=1 executed=1 skipped=0\n"
            "FLAGDNN_MUL_FUNCTIONAL: PASS cases=1 executed=1 skipped=0",
            "add",
            "functional",
            "passed",
            {},
            [],
        ),
        "functional accounting ignored an additional foreign marker",
    )
    require(
        hygon.validate_hygon_case_accounting(
            "FLAGDNN_ADD_SQUARE_FUNCTIONAL: PASS cases=1 "
            "executed=1 skipped=0",
            "add",
            "functional",
            "passed",
            {},
            [],
        ),
        "add accounting accepted the add_square marker by prefix",
    )
    require(
        not hygon.validate_hygon_case_accounting(
            "FLAGDNN_ADD_SQUARE_FUNCTIONAL: PASS cases=1 "
            "executed=1 skipped=0",
            "add_square",
            "functional",
            "passed",
            {},
            [],
        ),
        "add_square accounting rejected its exact marker",
    )
    require(
        not hygon.validate_hygon_case_accounting(
            "FLAGDNN_CONVOLUTION_FUNCTIONAL: PASS cases=1 "
            "executed=1 skipped=0",
            "conv_fprop",
            "functional",
            "passed",
            {},
            [],
        ),
        "NVIDIA-aligned convolution family marker was rejected",
    )
    require(
        hygon.validate_hygon_case_accounting(
            "FLAGDNN_CONV_FPROP_FUNCTIONAL: PASS cases=1 "
            "executed=1 skipped=0",
            "conv_fprop",
            "functional",
            "passed",
            {},
            [],
        ),
        "conv_fprop accepted a non-contract per-operator marker",
    )
    require(
        not hygon.validate_hygon_case_accounting(
            "FLAGDNN_CONV_FPROP_BENCHMARK: PASS cases=1 "
            "executed=1 skipped=0",
            "conv_fprop",
            "benchmark",
            "passed",
            {"conv2d_fprop_fp32_case": {"flagdnn": {}, "hipdnn": {}}},
            [],
        ),
        "conv_fprop benchmark lost its per-operator marker",
    )

    require(
        runner.ctest_status(0, "test ***Not Run (Disabled)") == "failed",
        "disabled CTest was accepted as passed",
    )
    require(
        runner.ctest_status(0, "test ***Skipped") == "skipped",
        "ordinary skipped CTest lost its classification",
    )

    nested_skip_output = (
        "11: 1/1 Test #1: integration.iluvatar.jit ***Skipped\n"
        "11/30 Test #11: core.run_tests_contract ... Passed\n"
    )
    require(
        runner.ctest_aggregate_status(0, nested_skip_output) == "passed",
        "nested contract output incorrectly skipped an aggregate CTest run",
    )
    require(
        runner.ctest_aggregate_status(
            0, "1/1 Test #1: integration.iluvatar.jit ***Skipped"
        )
        == "skipped",
        "top-level aggregate CTest skip lost its classification",
    )
    require(
        runner.ctest_aggregate_status(
            0, "1/1 Test #1: integration.iluvatar.jit ***Not Run (Disabled)"
        )
        == "failed",
        "top-level disabled aggregate CTest was accepted",
    )

    operator_environment = {
        "HIP_VISIBLE_DEVICES": "0",
        "CUDA_VISIBLE_DEVICES": "stale",
        "FLAGDNN_POINTWISE_CASE": "fp32",
        "FLAGDNN_SDPA_FP8_BACKWARD_CASE": "small",
        "FLAGDNN_BENCHMARK_CASE": "add_fp32_case",
    }
    explicit = runner.device_environment("hygon", "0", operator_environment)
    require(
        explicit.get("HIP_VISIBLE_DEVICES") == "0"
        and not any(
            variable != "HIP_VISIBLE_DEVICES" and variable in explicit
            for variable in hygon.VISIBILITY_VARIABLES
        ),
        "explicit Hygon device selection is not isolated",
    )
    unmasked = runner.device_environment("hygon", None, operator_environment)
    require(
        not any(hygon.CASE_FILTER_PATTERN.fullmatch(name) for name in unmasked)
        and unmasked.get("HIP_VISIBLE_DEVICES") == "0"
        and unmasked.get("CUDA_VISIBLE_DEVICES") == "stale",
        "Hygon runner retained a FLAGDNN_*_CASE filter",
    )
    ascend_environment = runner.device_environment(
        "ascend", "2", {}, adapter=ascend
    )
    require(
        ascend_environment
        == {
            "ASCEND_RT_VISIBLE_DEVICES": "2",
            "NPU_VISIBLE_DEVICES": "2",
        },
        "Ascend device visibility policy was not delegated",
    )
    nvidia_environment = runner.device_environment(
        "nvidia", "3", {}, adapter=nvidia
    )
    require(
        nvidia_environment == {"CUDA_VISIBLE_DEVICES": "3"},
        "NVIDIA device visibility policy was not delegated",
    )

    if os.name == "posix" and Path("/proc/self/stat").is_file():
        with tempfile.TemporaryDirectory(
            prefix="flagdnn-runner-session-contract-"
        ) as session_directory:
            session_root = Path(session_directory)
            middle_script = session_root / "middle.py"
            top_script = session_root / "top.py"
            middle_script.write_text(
                """
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
signal.signal(signal.SIGHUP, signal.SIG_IGN)
leaf_source = '''
import signal
import time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
signal.signal(signal.SIGHUP, signal.SIG_IGN)
while True:
    time.sleep(1)
'''
leaf = subprocess.Popen(
    [sys.executable, "-c", leaf_source],
    preexec_fn=lambda: os.setpgid(0, 0),
)
record = {
    "session_id": os.getsid(0),
    "middle_pid": os.getpid(),
    "middle_pgid": os.getpgrp(),
    "leaf_pid": leaf.pid,
    "leaf_pgid": os.getpgid(leaf.pid),
}
Path(sys.argv[1]).write_text(json.dumps(record), encoding="utf-8")
while True:
    time.sleep(1)
""".lstrip(),
                encoding="utf-8",
            )
            top_script.write_text(
                """
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
signal.signal(signal.SIGHUP, signal.SIG_IGN)
subprocess.Popen(
    [sys.executable, sys.argv[1], sys.argv[2]],
    preexec_fn=lambda: os.setpgid(0, 0),
)
record = Path(sys.argv[2])
while not record.is_file():
    time.sleep(0.01)
print("READY", flush=True)
while True:
    time.sleep(1)
""".lstrip(),
                encoding="utf-8",
            )

            original_term_grace = runner.PROCESS_TERMINATION_GRACE_SECONDS
            original_kill_grace = runner.PROCESS_KILL_GRACE_SECONDS
            original_poll = runner.PROCESS_CLEANUP_POLL_SECONDS
            runner.PROCESS_TERMINATION_GRACE_SECONDS = 0.1
            runner.PROCESS_KILL_GRACE_SECONDS = 2.0
            runner.PROCESS_CLEANUP_POLL_SECONDS = 0.02

            def process_is_live(pid: int, session_id: int) -> bool:
                identity = runner._linux_process_identity(pid)
                return (
                    identity is not None
                    and identity[0] not in {"Z", "X"}
                    and identity[2] == session_id
                )

            def exercise_session_cleanup(
                label: str, trigger: signal.Signals | None
            ) -> None:
                record_path = session_root / f"{label}.json"
                command = [
                    sys.executable,
                    str(top_script),
                    str(middle_script),
                    str(record_path),
                ]
                trigger_thread: threading.Thread | None = None
                if trigger is not None:

                    def send_signal_when_ready() -> None:
                        deadline = time.monotonic() + 5.0
                        while (
                            not record_path.is_file()
                            and time.monotonic() < deadline
                        ):
                            time.sleep(0.01)
                        os.kill(os.getpid(), trigger)

                    trigger_thread = threading.Thread(
                        target=send_signal_when_ready,
                        name=f"run-tests-{label}-trigger",
                        daemon=True,
                    )
                    trigger_thread.start()
                try:
                    if trigger is None:
                        _, _, return_code, timed_out = (
                            runner.run_process_group(
                                command,
                                dict(os.environ),
                                timeout=1,
                            )
                        )
                        require(
                            timed_out and return_code is None,
                            "timeout cleanup contract did not time out",
                        )
                    else:
                        try:
                            runner.run_process_group(
                                command,
                                dict(os.environ),
                                timeout=10,
                            )
                        except runner._TerminationSignal as termination:
                            require(
                                termination.signum == trigger,
                                f"{label} propagated the wrong signal",
                            )
                        else:
                            raise RuntimeError(
                                f"{label} did not propagate termination"
                            )
                finally:
                    if trigger_thread is not None:
                        trigger_thread.join(timeout=1.0)

                require(
                    record_path.is_file(),
                    f"{label} hierarchy did not publish process ids",
                )
                record = json.loads(record_path.read_text(encoding="utf-8"))
                session_id = int(record["session_id"])
                require(
                    int(record["middle_pgid"]) != session_id
                    and int(record["leaf_pgid"])
                    not in {session_id, int(record["middle_pgid"])},
                    f"{label} did not create nested process groups",
                )
                deadline = time.monotonic() + 2.0
                pids = (
                    session_id,
                    int(record["middle_pid"]),
                    int(record["leaf_pid"]),
                )
                while (
                    any(process_is_live(pid, session_id) for pid in pids)
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.02)
                require(
                    not any(process_is_live(pid, session_id) for pid in pids)
                    and not runner._session_process_groups(session_id),
                    f"{label} left a live nested process-group orphan",
                )

            try:
                exercise_session_cleanup("timeout", None)
                exercise_session_cleanup("sigterm", signal.SIGTERM)
                exercise_session_cleanup("sighup", signal.SIGHUP)
            finally:
                runner.PROCESS_TERMINATION_GRACE_SECONDS = original_term_grace
                runner.PROCESS_KILL_GRACE_SECONDS = original_kill_grace
                runner.PROCESS_CLEANUP_POLL_SECONDS = original_poll

    print("PASS tools/run_tests.py parsing, accounting, and device contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
