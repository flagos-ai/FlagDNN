#!/usr/bin/env python3

"""Core contracts for tools/run_tests.py result validation."""

from __future__ import annotations

from collections import Counter
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


def check_platform_single_report_hooks(runner) -> None:
    """Optional single-report processing must preserve other platforms."""

    class DisabledHook:
        postprocess_report = None

    native = {
        "timestamp_utc": "2026-10-10T00:00:00+00:00",
        "overall_status": "passed",
        "exit_code": 0,
        "suite_operators": {"benchmark": ["add"]},
        "results": {
            "add": {
                "benchmark": {
                    "status": "passed",
                    "exit_code": 0,
                    "records": {
                        "add_fp32": {
                            provider: {"median": latency, "unit": "us"}
                            for provider, latency in (
                                ("flagdnn", 2.0),
                                ("reference", 6.0),
                            )
                        }
                    },
                }
            }
        },
    }
    expected = runner.summary_document(native)
    adapters = [("none", None), ("no-hook", object())]
    adapters.append(("disabled-hook", DisabledHook()))
    for platform in (
        "nvidia",
        "ascend",
        "thead",
        "iluvatar",
        "mthreads",
        "hygon",
    ):
        adapter = runner.load_platform_adapter(platform)
        require(adapter is not None, f"missing {platform} test adapter")
        adapters.append((platform, adapter))
    for label, adapter in adapters:
        require(
            runner.summary_document(native, adapter=adapter) == expected,
            f"{label} changed the native-only single-report contract",
        )


def check_platform_batch_report_hooks(runner) -> None:
    """Platforms opt into report changes; the default archive stays intact."""

    class NoHook:
        pass

    class DisabledHook:
        postprocess_batch_report = None

    adapters = [
        ("no-adapter", None),
        ("no-hook", NoHook()),
        ("disabled-hook", DisabledHook()),
    ]
    for platform in ("nvidia", "ascend", "thead", "iluvatar"):
        adapter = runner.load_platform_adapter(platform)
        require(adapter is not None, f"missing {platform} test adapter")
        require(
            getattr(adapter, "postprocess_batch_report", None) is None,
            f"{platform} unexpectedly opted into batch report rewriting",
        )
        adapters.append((platform, adapter))
    environment = {
        "os_name": "fixture",
        "os_release": "fixture",
        "flag_dnn": "fixture",
        "flag_gems": "legacy alias",
    }
    native_case = "add_fp32"
    flagdnn = json.loads(timing(native_case, "flagdnn"))
    reference = json.loads(timing(native_case, "native_reference"))
    reference.update(median=6.0, p90=6.0, samples=[6.0])
    native_task = {
        "status": "passed",
        "exit_code": 0,
        "duration_seconds": 0.25,
        "test_case": "benchmark/test_add.cpp",
        "records": {
            native_case: {"flagdnn": flagdnn, "native_reference": reference}
        },
    }
    with tempfile.TemporaryDirectory(
        prefix="flagdnn-platform-reports-"
    ) as tmp:
        output = Path(tmp)
        performance = runner.batch_suite_result(
            "add", "benchmark", native_task, output
        )
        row = performance["data"]["fp32"]["details"][native_case]
        require(
            row == {"base": 0.006, "gems": 0.002, "speedup": 3.0},
            "generic native-pair serialization changed",
        )
        partitions = {
            0: {"add": {"customized": False, "performance": performance}},
            1: {
                "sub": {
                    "customized": True,
                    "performance": {
                        "status": "Failed",
                        "exit_code": 1,
                        "duration": 0.5,
                        "data": {},
                        "test_case": "benchmark/test_sub.cpp",
                    },
                }
            },
            9: {},
        }
        originals = {
            gpu: json.dumps(partition, indent=1) + "\n"
            for gpu, partition in partitions.items()
        }
        expected_results = {**partitions[0], **partitions[1]}
        archive = output / "add/performance_result.log"
        archive_bytes = archive.read_bytes()

        def reset_partitions():
            for gpu, content in originals.items():
                (output / f"summary{gpu}.json").write_text(content)

        def require_unchanged(label):
            for gpu, content in originals.items():
                require(
                    (output / f"summary{gpu}.json").read_text() == content,
                    f"{label} rewrote a per-GPU report without opting in",
                )
            require(
                not (output / "summary2.json").exists()
                and archive.read_bytes() == archive_bytes,
                f"{label} created a missing partition "
                "or rewrote the native archive",
            )

        # A final platform outcome with equivalent-looking keys must not
        # activate common-runner policy for platforms without the hook.
        unrelated_summary = {
            "equivalent_performance": {
                "case_count": 1,
                "metric": "must_not_be_interpreted",
                "cases": [
                    {
                        "operator": "add",
                        "case": native_case,
                        "passed": False,
                        "relative_efficiency": 0.1,
                        "required_relative_efficiency": 0.8,
                    }
                ],
            },
            "platform_owned_field": "untouched",
        }
        for label, adapter in adapters:
            reset_partitions()
            runner.write_batch_summary(
                output,
                environment,
                [0, 1, 2, 9],
                adapter=adapter,
                platform_summary=unrelated_summary,
                exit_code=1,
            )
            report = json.loads((output / "summary.json").read_text())
            require(
                set(report) == {"timestamp", "env", "result"}
                and report["result"] == expected_results
                and report["env"] == runner.batch_environment(environment),
                f"{label} inherited another platform's "
                "report fields or status",
            )
            require_unchanged(label)

        # The original three-argument API remains usable without an adapter.
        reset_partitions()
        runner.write_batch_summary(output, environment, [0, 1, 2, 9])
        require_unchanged("legacy batch writer")
        hygon = runner.load_platform_adapter("hygon")
        require(
            callable(getattr(hygon, "postprocess_batch_report", None)),
            "Hygon equivalent reporting did not opt into the generic hook",
        )
        for summary in (
            {},
            {"equivalent_performance": {"case_count": 0, "cases": []}},
        ):
            reset_partitions()
            runner.write_batch_summary(
                output,
                environment,
                [0, 1, 2, 9],
                adapter=hygon,
                platform_summary=summary,
                exit_code=1,
            )
            report = json.loads((output / "summary.json").read_text())
            require(
                set(report) == {"timestamp", "env", "result"}
                and report["result"] == expected_results,
                "Hygon native-only reporting changed "
                "through the equivalent hook",
            )
            require_unchanged("Hygon native-only hook")

        class GenericHook:
            def postprocess_batch_report(
                self, *, partitions, report, platform_summary, exit_code
            ):
                require(
                    set(partitions) == {0, 1, 9},
                    "generic hook received the wrong GPU partitions",
                )
                require(
                    exit_code == 7
                    and platform_summary == {"vendor_gate": "failed"},
                    "generic hook lost final platform state",
                )
                target = partitions[0]["add"]["performance"]
                target.update(status="Failed", exit_code=7)
                target["data"]["fp32"]["result"] = "FAIL"
                target["vendor_gate"] = {"score": 0.25, "passed": False}
                report["result"]["add"] = partitions[0]["add"]
                report["vendor_gate"] = platform_summary["vendor_gate"]
                report["exit_code"] = exit_code
                return True

        reset_partitions()
        runner.write_batch_summary(
            output,
            environment,
            [0, 1, 2, 9],
            adapter=GenericHook(),
            platform_summary={"vendor_gate": "failed"},
            exit_code=7,
        )
        report = json.loads((output / "summary.json").read_text())
        stored = json.loads((output / "summary0.json").read_text())
        require(
            report["vendor_gate"] == "failed"
            and report["exit_code"] == 7
            and stored["add"]["performance"]["status"] == "Failed"
            and stored["add"] == report["result"]["add"]
            and stored["add"]["performance"]["vendor_gate"]["score"] == 0.25
            and report["result"]["sub"] == expected_results["sub"],
            "generic report hook failed to persist "
            "non-Hygon fields and statuses",
        )

        class MetadataHook:
            def postprocess_batch_report(self, *, report, **_kwargs):
                report["vendor_note"] = "metadata only"
                return False

        reset_partitions()
        runner.write_batch_summary(
            output, environment, [0, 1, 2, 9], adapter=MetadataHook()
        )
        report = json.loads((output / "summary.json").read_text())
        require(
            report["vendor_note"] == "metadata only"
            and report["result"] == expected_results,
            "metadata-only report hook was ignored or changed native results",
        )
        require_unchanged("metadata-only hook")

        class InvalidHook:
            def postprocess_batch_report(self, **_kwargs):
                return {"changed": True}

        merged_before = (output / "summary.json").read_bytes()
        try:
            runner.write_batch_summary(
                output, environment, [0, 1, 9], adapter=InvalidHook()
            )
        except RuntimeError as error:
            require(
                "invalid batch report update" in str(error),
                "invalid generic hook failed with an unrelated error",
            )
        else:
            raise RuntimeError(
                "non-boolean report-hook return value was accepted"
            )
        require(
            (output / "summary.json").read_bytes() == merged_before,
            "invalid report-hook output overwrote the published summary",
        )
        require_unchanged("invalid hook")


def check_operator_presets(runner) -> None:
    """Both presets retain their order and use the shared manifest validation."""
    manifests = runner.load_platform_adapter(
        "nvidia"
    ).select_operator_manifests(runner.operator_manifests())
    for appoint, preset in (
        (False, runner.DEFAULT_OPERATORS),
        (True, runner.DEFAULT_OPERATORS_APPOINT),
    ):
        expected = list(dict.fromkeys(preset))
        selected = runner.requested_operators(
            manifests, list(runner.VALID_SUITES), None, None, appoint=appoint
        )
        require(
            selected == {suite: expected for suite in runner.VALID_SUITES},
            f"operator preset lost its order or selected the wrong list: {appoint=}",
        )
        missing = expected[-1]
        incomplete = {
            **manifests,
            "benchmark": [
                operator
                for operator in manifests["benchmark"]
                if operator != missing
            ],
        }
        try:
            runner.requested_operators(
                incomplete,
                list(runner.VALID_SUITES),
                None,
                None,
                appoint=appoint,
            )
        except ValueError as error:
            require(
                "benchmark" in str(error) and missing in str(error),
                "preset manifest error omitted the missing operator/suite",
            )
        else:
            raise RuntimeError("preset bypassed manifest support validation")

    require(
        runner.requested_operators(manifests, ["functional"], None, None)[
            "functional"
        ]
        == list(dict.fromkeys(runner.DEFAULT_OPERATORS)),
        "omitting appoint changed the default operator list",
    )
    # Include empty explicit selectors and 'all': an explicit --appoint must
    # not silently win over any --ops value or attempt to read a list file.
    conflicts = [
        ("add", None),
        ("all", None),
        ("", None),
        (None, Path("/nonexistent/flagdnn-appoint-contract.txt")),
    ]
    for value, list_file in conflicts:
        try:
            runner.requested_operators(
                manifests,
                ["functional"],
                value,
                list_file,
                appoint=True,
            )
        except ValueError as error:
            require(
                "--appoint" in str(error)
                and ("--ops" if value is not None else "--op-list-file")
                in str(error),
                "appoint conflict produced an unrelated error",
            )
        else:
            raise RuntimeError(
                "appoint accepted an explicit operator selector"
            )

    original_argv = sys.argv
    try:
        for extra, expected in (([], False), (["--appoint"], True)):
            sys.argv = [runner.__file__, "--platform", "nvidia", *extra]
            arguments = runner.parse_arguments()
            require(
                arguments.appoint is expected,
                "--appoint is not an optional boolean switch",
            )
        for extra in (
            ["--appoint", "--ops", "add"],
            ["--appoint", "--ops", "all"],
            ["--appoint", "--ops", ""],
            ["--appoint", "--op-list-file", "not-read.txt"],
        ):
            sys.argv = [runner.__file__, "--platform", "nvidia", *extra]
            stderr = io.StringIO()
            try:
                with contextlib.redirect_stderr(stderr):
                    runner.parse_arguments()
            except SystemExit as error:
                require(
                    error.code == 2 and "--appoint" in stderr.getvalue(),
                    "CLI operator-selector conflict has the wrong diagnostic",
                )
            else:
                raise RuntimeError(
                    "CLI accepted conflicting operator selectors"
                )
    finally:
        sys.argv = original_argv


def check_batch_runner(runner) -> None:
    """Exercise concurrent CTest processes and the public archive protocol."""
    for marker in ("[skip]", "[SKIP]", "SKIP"):
        case = "relu_backward_attributes_fp32_1x1x65"
        reason = "muDNN cannot express clipped ReLU gradients"
        records = runner.accuracy_records("relu_backward", {
            "stdout": f"96: {marker} case={case} provider=mudnn reason={reason}\n",
            "stderr": "", "case_counts": {"passed": 0, "failed": 0, "skipped": 1},
        })
        require(len(records) == 1, "named skip duplicated as native accounting")
        record = next(iter(records.values()))
        require(record["params"] == {"case": case}
                and record["skipped_reason"] == reason,
                "skip parser lost case identity or truncated its reason")
    with tempfile.TemporaryDirectory(prefix="flagdnn-batch-contract-") as tmp:
        root = Path(tmp)
        build = root / "build"
        build.mkdir()
        fixture = root / "case.py"
        fixture.write_text(
            """import json, os, sys, time
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
"""
        )
        (build / "CMakeCache.txt").write_text(
            "FLAGDNN_BACKENDS:STRING=nvidia\n"
        )
        tests = []
        for operator in dict.fromkeys(
            (*runner.DEFAULT_OPERATORS, *runner.DEFAULT_OPERATORS_APPOINT)
        ):
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
        # The flag changes actual worker scheduling, not only CLI parsing.
        # One mock GPU makes completion order deterministic; these fixtures
        # run Python/CTest only and never access a GPU.
        process = subprocess.run(
            command + ["--appoint", "--gpus", "1"],
            capture_output=True,
            text=True,
            env=environment,
            timeout=60,
        )
        require(
            process.returncode == 0,
            "appointed batch run failed: " + process.stdout + process.stderr,
        )
        document = json.loads((output / "summary.json").read_text())
        require(
            set(document["result"]) == set(runner.DEFAULT_OPERATORS_APPOINT),
            "--appoint batch did not schedule DEFAULT_OPERATORS_APPOINT",
        )
        scheduled = re.findall(r"\baccuracy\s+(\w+)\s+\[", process.stdout)
        require(
            scheduled == list(dict.fromkeys(runner.DEFAULT_OPERATORS_APPOINT)),
            "--appoint worker scheduling lost preset order",
        )
        require(
            all(
                result["accuracy"]["status"] == "Passed"
                and result["performance"]["status"] == "Passed"
                for result in document["result"].values()
            ),
            "appointed batch did not run both selected suites successfully",
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


def check_batch_cache_environment(runner) -> None:
    """Check cache and THead device policy through workers and CTest."""
    with tempfile.TemporaryDirectory(prefix="flagdnn-batch-cache-") as tmp:
        root = Path(tmp)
        build = root / "build"
        build.mkdir()
        fixture = root / "cache_case.py"
        fixture.write_text(
            """import json, os, sys, time
from pathlib import Path
print('CACHE_ENV=' + json.dumps(os.environ.get('FLAGDNN_CACHE_PATH')),
      flush=True)
cuda_mask = os.environ.get('CUDA_VISIBLE_DEVICES')
print('DEVICE_ENV=' + json.dumps({
    'CUDA_VISIBLE_DEVICES': cuda_mask,
    'HGGC_VISIBLE_DEVICES': os.environ.get('HGGC_VISIBLE_DEVICES'),
    # Model the PPU driver's CUDA mask for process-local device zero.
    'simulated_physical_device': (cuda_mask or '0').split(',')[0],
}), flush=True)
barrier = os.environ.get('FLAGDNN_CONTRACT_DEVICE_BARRIER')
if barrier:
    barrier = Path(barrier)
    (barrier / sys.argv[1]).touch()
    deadline = time.monotonic() + 20
    while not all((barrier / operator).is_file()
                  for operator in ('add', 'sub')):
        if time.monotonic() >= deadline:
            raise RuntimeError(
                'both GPU workers must start before either exits')
        time.sleep(0.01)
time.sleep(0.05)
print(sys.argv[1] + '_fp32: reference PASS', flush=True)
print('FLAGDNN_' + sys.argv[1].upper() +
      '_FUNCTIONAL: PASS cases=1 executed=1 skipped=0', flush=True)
"""
        )
        (build / "CMakeCache.txt").write_text(
            "FLAGDNN_BACKENDS:STRING=nvidia;thead\n"
        )
        tests = [
            {
                "name": f"functional.{platform}.{operator}",
                "command": [sys.executable, str(fixture), operator],
                "properties": [
                    {"name": "WORKING_DIRECTORY", "value": str(root)}
                ],
            }
            for platform in ("nvidia", "thead")
            for operator in ("add", "sub")
        ]
        runner.isolated_ctest_directory(root / "inventory", tests)
        shutil.copy(
            root / "inventory" / "CTestTestfile.cmake",
            build / "CTestTestfile.cmake",
        )
        environment = dict(os.environ)
        environment.pop("FLAGDNN_CACHE_PATH", None)
        default_cache = str(build / "cache" / "run_tests")
        explicit_cache = str(root / "explicit cache")
        for label, platform, configured, expected in (
            ("default", "thead", None, default_cache),
            ("repeat", "thead", None, default_cache),
            ("empty", "thead", "", default_cache),
            ("override", "thead", explicit_cache, explicit_cache),
            ("other-platform", "nvidia", None, None),
            ("other-override", "nvidia", explicit_cache, explicit_cache),
        ):
            output = root / label
            worker_environment = dict(environment)
            if configured is not None:
                worker_environment["FLAGDNN_CACHE_PATH"] = configured
            if label == "default":
                # Force both native subprocesses to coexist: a quick first
                # task must not consume the queue before worker 1 starts.
                barrier = root / "device-barrier"
                barrier.mkdir()
                worker_environment.update(
                    {
                        "CUDA_VISIBLE_DEVICES": "7",
                        "HGGC_VISIBLE_DEVICES": "6",
                        "FLAGDNN_CONTRACT_DEVICE_BARRIER": str(barrier),
                    }
                )
            process = subprocess.run(
                [
                    sys.executable,
                    str(Path(runner.__file__).resolve()),
                    "--build-dir",
                    str(build),
                    "--platform",
                    platform,
                    "--output-dir",
                    str(output),
                    "--ops",
                    "add,sub",
                    "--suites",
                    "functional",
                    "--gpus",
                    "0,1",
                    "--dump-output",
                    "--color",
                    "never",
                ],
                capture_output=True,
                text=True,
                env=worker_environment,
                timeout=60,
            )
            require(
                process.returncode == 0,
                f"{label} batch failed: " + process.stdout + process.stderr,
            )
            for operator in ("add", "sub"):
                log = (output / operator / "accuracy_stdout.log").read_text()
                observed = re.findall(r"CACHE_ENV=(.*)", log)
                require(
                    len(observed) == 1 and json.loads(observed[0]) == expected,
                    f"{label}/{operator} worker cache environment differs: "
                    f"{observed!r} != {expected!r}",
                )
            if label == "default":
                assigned = set()
                for gpu in (0, 1):
                    partition = json.loads(
                        (output / f"summary{gpu}.json").read_text()
                    )
                    require(
                        len(partition) == 1
                        and not assigned.intersection(partition),
                        "THead native tasks did not run on both GPU workers",
                    )
                    assigned.update(partition)
                    operator = next(iter(partition))
                    log = (
                        output / operator / "accuracy_stdout.log"
                    ).read_text()
                    observed = re.findall(r"DEVICE_ENV=(.*)", log)
                    expected_device = {
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        "HGGC_VISIBLE_DEVICES": str(gpu),
                        "simulated_physical_device": str(gpu),
                    }
                    require(
                        len(observed) == 1
                        and json.loads(observed[0]) == expected_device,
                        f"THead worker {gpu} native device mapping differs: "
                        f"{observed!r} != {expected_device!r}",
                    )
                require(
                    assigned == {"add", "sub"},
                    "THead device-routing regression lost an operator",
                )
        # The serial compatibility mode keeps native cache policy unchanged.
        process = subprocess.run(
            [
                sys.executable,
                str(Path(runner.__file__).resolve()),
                "--build-dir",
                str(build),
                "--platform",
                "thead",
                "--output",
                str(root / "legacy.json"),
                "--ops",
                "add",
                "--suites",
                "functional",
                "--no-preflight",
                "--verbose",
            ],
            capture_output=True,
            text=True,
            env=environment,
            timeout=60,
        )
        require(
            process.returncode == 0 and "CACHE_ENV=null" in process.stdout,
            "batch cache default leaked into serial mode: "
            + process.stdout
            + process.stderr,
        )


def check_hygon_native_timing(runner, hygon) -> None:
    case = "add_perf_fp32_8x16x32_by_8x16x32"
    metadata_keys = ("timing_mode", "timing_reason", "benchmark_config")

    def native(provider="flagdnn", *, direct=False):
        record = json.loads(timing(case, provider))
        record.update(
            schema_version=5,
            timing_mode=(
                "hip_event_direct_batch" if direct else "hip_graph_batch"
            ),
            timing_reason=(
                "hipdnn_graph_tail_workaround" if direct else "standard"
            ),
            benchmark_config={
                "warmup_iterations": 0,
                "sample_count": 3,
                "iterations_per_sample": 50,
            },
        )
        if provider == "hipdnn":
            record.update(median=1.0, p90=1.5, samples=[0.5, 1.0, 1.5])
        return record

    for direct in (False, True):
        pair = {
            provider: native(provider, direct=direct)
            for provider in ("flagdnn", "hipdnn")
        }
        output = "\n".join(json.dumps(record) for record in pair.values())
        records, errors = runner.benchmark_records(output, hygon)
        require(
            not errors
            and records == {case: pair}
            and not hygon.validate_hygon_benchmark_pairs(
                records, "add", ["add"]
            ),
            "valid Hygon native timing provenance was rejected",
        )
        unsupported, errors = runner.benchmark_records(output)
        require(
            errors and not unsupported, "v5 leaked into the generic parser"
        )
        task = {"status": "passed", "exit_code": 0, "records": records}
        hygon.postprocess_result(
            result=task,
            ctest_reported_status="passed",
            output="FLAGDNN_ADD_BENCHMARK: PASS cases=1 executed=1 skipped=0\n",
            operator="add",
            suite="benchmark",
            records=records,
            manifest_operators=["add"],
        )
        public = runner.suite_result(task, "benchmark")
        row = public["data"]["fp32"]["details"][case]
        exported = runner.performance_file("add", task, public)["add"][
            "details"
        ][0]["result"][0]
        with tempfile.TemporaryDirectory(prefix="flagdnn-native-v5-") as tmp:
            batch = runner.batch_suite_result(
                "add", "benchmark", task, Path(tmp)
            )
            batch_row = batch["data"]["fp32"]["details"][case]
            log = (Path(tmp) / "add/performance_result.log").read_text()
            logged = json.loads(log.splitlines()[1].removeprefix("[INFO] "))[
                "result"
            ][0]
        performance = hygon.benchmark_speedup_summary(
            {"add": {"benchmark": task}}, None
        )
        gated = hygon.benchmark_speedup_summary(
            {"add": {"benchmark": task}}, 0.9
        )
        for rendered in (
            row,
            exported,
            batch_row,
            logged,
            gated["failures"][0],
        ):
            require(
                all(
                    rendered.get(key) == pair["flagdnn"][key]
                    for key in metadata_keys
                ),
                "native timing provenance was lost in a report",
            )
        require(
            task["status"] == "passed"
            and public["status"] == "Passed"
            and public["data"]["fp32"]["result"] == "OK"
            and row["speedup"] == 0.5
            and performance["minimum_speedup"] == 0.5,
            "native v5 metadata changed speedup math or enabled a default gate",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            default = hygon.finalize(
                results={"add": {"benchmark": task}},
                suite_operators={"benchmark": ["add"]},
                suites=["benchmark"],
                state={},
                min_speedup=None,
                preflight_passed=True,
            )
        require(
            not default["failed"],
            "default slow native v5 became a failed task",
        )
        failed_task = {**task, "status": "failed", "exit_code": 7}
        hygon.postprocess_result(
            result=failed_task,
            ctest_reported_status="failed",
            output="FLAGDNN_ADD_BENCHMARK_FAILED: precision mismatch\n",
            operator="add",
            suite="benchmark",
            records=records,
            manifest_operators=["add"],
        )
        with contextlib.redirect_stdout(io.StringIO()):
            failed = hygon.finalize(
                results={"add": {"benchmark": failed_task}},
                suite_operators={"benchmark": ["add"]},
                suites=["benchmark"],
                state={},
                min_speedup=None,
                preflight_passed=True,
            )
        require(
            failed["failed"]
            and failed_task["status"] == "failed"
            and failed_task["exit_code"] == 7,
            "native v5 metadata masked a real execution/precision failure",
        )

    valid = native()
    malformed = [
        {**valid, key: value}
        for key, value in (
            ("schema_version", 5.0),
            ("schema_version", True),
            ("provider", "cudnn"),
            ("case", ""),
            ("unit", "ms"),
            ("timing_mode", "cuda_graph"),
            ("timing_mode", []),
            ("timing_reason", "hipdnn_graph_tail_workaround"),
            ("median", 0),
            ("median", 10**1000),
            ("p90", float("inf")),
            ("samples", [float("nan")] * 3),
            ("samples", [1.0, 2.0]),
            ("p90", 2.0),
            ("extra", True),
            ("benchmark_config", None),
        )
    ]
    for key in valid:
        malformed.append(
            {name: value for name, value in valid.items() if name != key}
        )
    for key in valid["benchmark_config"]:
        for value in (-1, True, 1.5, None):
            malformed.append(
                {
                    **valid,
                    "benchmark_config": {
                        **valid["benchmark_config"],
                        key: value,
                    },
                }
            )
        malformed.append(
            {
                **valid,
                "benchmark_config": {
                    name: value
                    for name, value in valid["benchmark_config"].items()
                    if name != key
                },
            }
        )
    for key in ("sample_count", "iterations_per_sample"):
        malformed.append(
            {
                **valid,
                "benchmark_config": {**valid["benchmark_config"], key: 0},
            }
        )
    malformed.append(
        {
            **valid,
            "benchmark_config": {**valid["benchmark_config"], "extra": 1},
        }
    )
    malformed.append({**native(direct=True), "timing_reason": "standard"})
    for record in malformed:
        rejected, errors = runner.benchmark_records(json.dumps(record), hygon)
        # A record not claiming the timing protocol remains unrelated diagnostics.
        require(
            not rejected and (errors or "kind" not in record),
            "malformed native v5 timing was accepted: " + repr(record),
        )
    for other in (
        json.loads(timing(case, "hipdnn")),
        native("hipdnn", direct=True),
        {
            **native("hipdnn"),
            "benchmark_config": {
                **valid["benchmark_config"],
                "warmup_iterations": 1,
            },
        },
        {**native("hipdnn"), "case": case + "_wrong"},
        {**native("hipdnn"), "provider": "flagdnn"},
    ):
        require(
            hygon.validate_hygon_benchmark_pairs(
                {case: {"flagdnn": valid, "hipdnn": other}}, "add", ["add"]
            ),
            "mixed native timing provenance or identity was accepted",
        )
    incompatible_pairs = [
        (native(), native("hipdnn", direct=True)),
        (native(), json.loads(timing(case, "hipdnn"))),
        (json.loads(timing(case, "flagdnn")), native("hipdnn")),
        (
            native(),
            {
                **native("hipdnn"),
                "benchmark_config": {
                    **valid["benchmark_config"],
                    "warmup_iterations": 1,
                },
            },
        ),
        (native(), {**native("hipdnn"), "timing_reason": "different_reason"}),
    ]
    for key in metadata_keys:
        incompatible_pairs.append(
            (
                native(),
                {
                    name: value
                    for name, value in native("hipdnn").items()
                    if name != key
                },
            )
        )
        if key != "timing_mode":
            incompatible_pairs.append(
                tuple(
                    {
                        name: value
                        for name, value in native(provider).items()
                        if name != key
                    }
                    for provider in ("flagdnn", "hipdnn")
                )
            )
    with tempfile.TemporaryDirectory(prefix="flagdnn-incompatible-v5-") as tmp:
        for index, (actual, reference) in enumerate(incompatible_pairs):
            raw = {case: {"flagdnn": actual, "hipdnn": reference}}
            errors = hygon.validate_hygon_benchmark_pairs(raw, "add", ["add"])
            require(errors, "incompatible fixture was not rejected by adapter")
            task = {
                "status": "failed",
                "exit_code": 1,
                "records": raw,
                "record_errors": errors,
            }
            snapshot = json.dumps(task, sort_keys=True)
            batch = runner.batch_suite_result(
                "add", "benchmark", task, Path(tmp) / str(index)
            )
            log = (
                Path(tmp) / str(index) / "add/performance_result.log"
            ).read_text()
            require(
                runner.performance_data(task) == {}
                and batch["status"] == "Failed"
                and batch["data"] == {}
                and "speedup" not in log
                and json.dumps(task, sort_keys=True) == snapshot,
                "incompatible timing pair produced a ratio/fake OK or lost raw diagnostics",
            )
        good_case = case + "_compatible"
        compatible = {
            provider: {**native(provider), "case": good_case}
            for provider in ("flagdnn", "hipdnn")
        }
        task["records"] = {**raw, good_case: compatible}
        details = runner.performance_data(task)["fp32"]["details"]
        require(
            set(details) == {good_case}
            and details[good_case]["speedup"] == 0.5,
            "an incompatible timing pair erased unrelated valid comparisons",
        )
    legacy = "\n".join(
        timing(case, provider) for provider in ("flagdnn", "hipdnn")
    )
    legacy_records, errors = runner.benchmark_records(legacy, hygon)
    require(
        not errors
        and not hygon.validate_hygon_benchmark_pairs(
            legacy_records, "add", ["add"]
        ),
        "native v5 support broke legacy v1 provider pairs",
    )
    legacy_row = runner.performance_data({"records": legacy_records})["fp32"][
        "details"
    ][case]
    require(
        not any(key in legacy_row for key in metadata_keys),
        "v1 invented timing provenance",
    )
    # Older adapters may carry their own benchmark_config without declaring
    # this timing provenance. Do not change their public row format.
    legacy_records[case]["flagdnn"]["benchmark_config"] = {"legacy": True}
    legacy_row = runner.performance_data({"records": legacy_records})["fp32"][
        "details"
    ][case]
    require(
        not any(key in legacy_row for key in metadata_keys),
        "native timing metadata passthrough changed an older adapter's output",
    )


def check_equivalent_performance(runner, hygon) -> None:
    case = "div_perf_fp32_1x1x1024_by_1x1x1024"

    def measured(median=2.0):
        return {
            "schema_version": 4,
            "kind": "steady_state",
            "provider": "flagdnn",
            "case": case,
            "unit": "us",
            "median": median,
            "p90": median + 1.0,
            "samples": [1.0] * 9 + [median] + [median + 1.0] * 10,
            "equivalent_baseline": {
                "provider": "nvidia_equivalent",
                "platform": "nvidia",
                "target_platform": "hygon",
                "scale_metric": "vector_fp32",
                "timing_profile": "native",
                "reference_time_us": 2.0,
                "reference_peak": 4.0,
                "target_peak": 5.0,
                "min_relative_efficiency": 0.8,
                "equivalent_time_us": 1.6,
                "limit_us": 2.0,
                "relative_efficiency": 1.6 / median,
                "passed": median <= 2.0,
            },
        }

    valid = measured()
    records, errors = runner.benchmark_records(json.dumps(valid), hygon)
    require(
        not errors and records == {case: {"flagdnn": valid}},
        "equivalent baseline lost the real samples "
        "or invented reference samples",
    )
    for key, value in (
        ("schema_version", 4.0),
        ("case", "add_perf_fp32_1_by_1"),
        ("case", "div_perf_fp64_1_by_1"),
        ("provider", "nvidia_equivalent"),
        ("unit", "ms"),
        ("median", 0.0),
        ("median", float("nan")),
        ("p90", 2.0),
        ("samples", [2.0]),
        ("samples", [float("inf")] * 20),
        ("unexpected", 1),
    ):
        malformed = {**valid, key: value}
        rejected, errors = runner.benchmark_records(
            json.dumps(malformed), hygon
        )
        require(
            errors and not rejected,
            f"invalid equivalent record {key} accepted",
        )
    numeric = (
        "reference_time_us",
        "reference_peak",
        "target_peak",
        "min_relative_efficiency",
        "equivalent_time_us",
        "limit_us",
        "relative_efficiency",
    )
    mutations = [
        (key, value)
        for key in numeric
        for value in (0.0, -1.0, float("nan"), float("inf"), True, None)
    ] + [
        ("target_peak", 10**1000),
        ("min_relative_efficiency", 1.1),
        ("equivalent_time_us", 2.0),
        ("limit_us", 2.1),
        ("relative_efficiency", 1.0),
        ("reference_time_us", 1.0e308),
        ("passed", False),
        ("passed", 1),
        ("provider", "hipdnn"),
        ("platform", "hygon"),
        ("target_platform", "nvidia"),
        ("scale_metric", "vector_fp16"),
        ("timing_profile", "attention"),
        ("samples", [1.6]),
    ]
    for key, value in mutations:
        malformed = {
            **valid,
            "equivalent_baseline": {
                **valid["equivalent_baseline"],
                key: value,
            },
        }
        rejected, errors = runner.benchmark_records(
            json.dumps(malformed), hygon
        )
        require(
            errors and not rejected,
            f"invalid equivalent baseline {key} accepted",
        )
    for key in valid["equivalent_baseline"]:
        baseline = dict(valid["equivalent_baseline"])
        del baseline[key]
        rejected, errors = runner.benchmark_records(
            json.dumps({**valid, "equivalent_baseline": baseline}), hygon
        )
        require(
            errors and not rejected, f"missing equivalent field {key} accepted"
        )
    overflow = {
        **valid,
        "equivalent_baseline": {
            **valid["equivalent_baseline"],
            "reference_time_us": 10**308,
            "reference_peak": 10**308,
        },
    }
    rejected, errors = runner.benchmark_records(json.dumps(overflow), hygon)
    require(
        errors and not rejected, "integer equivalent formula overflow escaped"
    )
    require(
        not hygon.validate_hygon_benchmark_pairs(
            records, "div", ["div", "add"]
        )
        and hygon.validate_hygon_benchmark_pairs(
            records, "add", ["div", "add"]
        )
        and hygon.validate_hygon_benchmark_pairs(
            {case: {"flagdnn": valid, "hipdnn": {}}}, "div", ["div"]
        )
        and hygon.validate_hygon_benchmark_pairs(
            {"add_perf": {"flagdnn": {}}}, "add", ["add"]
        ),
        "single-provider exception leaked outside equivalent div records",
    )

    def processed(median=2.0, *, reported=None, output=None):
        record = measured(median)
        status = reported or "passed"
        task = {
            "status": status,
            "exit_code": int(status != "passed"),
            "records": {case: {"flagdnn": record}},
        }
        marker = "PASS" if status == "passed" else "FAIL"
        hygon.postprocess_result(
            result=task,
            ctest_reported_status=status,
            output=(
                output
                if output is not None
                else (
                    f"FLAGDNN_DIV_BENCHMARK: {marker} "
                    "cases=1 executed=1 skipped=0\n"
                )
            ),
            operator="div",
            suite="benchmark",
            records=task["records"],
            manifest_operators=["div"],
        )
        return task

    passed = processed()
    below_baseline = processed(2.000001)
    failed = processed(2.000001, reported="failed")
    require(
        passed["status"] == "passed"
        and not passed.get("performance_errors")
        and failed["status"] == "failed"
        and failed.get("performance_errors")
        and not failed.get("case_accounting_errors")
        and not failed.get("record_errors"),
        "equivalent threshold boundary or completed-failure "
        "accounting is wrong",
    )
    require(
        below_baseline["status"] == "passed"
        and below_baseline["exit_code"] == 0
        and below_baseline.get("performance_warnings")
        and not below_baseline.get("performance_errors")
        and hygon.complete_benchmark(below_baseline),
        "informational baseline evaluation failed a successful default run",
    )
    incomplete = processed(2.000001, output="")
    timed_out = processed(2.000001, reported="timeout", output="")
    require(
        incomplete.get("case_accounting_errors")
        and not incomplete.get("performance_errors")
        and hygon.equivalent_performance_summary(
            {"div": {"benchmark": incomplete}}, None
        )["case_count"]
        == 0,
        "aborted equivalent execution was treated as a complete "
        "performance failure",
    )
    require(
        timed_out["status"] == "timeout"
        and not timed_out.get("performance_errors")
        and not timed_out.get("case_accounting_errors"),
        "partial equivalent records replaced the original timeout status",
    )
    public = runner.suite_result(failed, "benchmark")
    row = public["data"]["fp32"]["details"][case]
    require(
        public["status"] == "Failed"
        and public["data"]["fp32"]["result"] == "OK"
        and row["base"] == 0.0016
        and row["equivalent_baseline"]
        == measured(2.000001)["equivalent_baseline"]
        and row["speedup"] == 1.6 / 2.000001,
        "an execution failure lost baseline data "
        "or invented an implicit dtype performance gate",
    )
    exported = runner.performance_file("div", failed, public)
    require(
        exported["div"]["details"][0]["result"][0]["equivalent_baseline"]
        == row["equivalent_baseline"],
        "legacy performance JSON erased the equivalent reference identity",
    )
    with tempfile.TemporaryDirectory(
        prefix="flagdnn-equivalent-report-"
    ) as tmp:
        batch = runner.batch_suite_result(
            "div", "benchmark", failed, Path(tmp)
        )
        log = (Path(tmp) / "div" / "performance_result.log").read_text()
        exported = json.loads(log.splitlines()[1].removeprefix("[INFO] "))
        require(
            batch["data"]["fp32"]["details"][case]["equivalent_baseline"]
            == row["equivalent_baseline"]
            and exported["result"][0]["equivalent_baseline"]
            == row["equivalent_baseline"],
            "batch performance report erased the equivalent "
            "reference identity",
        )

    def outcome(task, threshold):
        with contextlib.redirect_stdout(io.StringIO()):
            return hygon.finalize(
                results={"div": {"benchmark": task}},
                suite_operators={"benchmark": ["div"]},
                suites=["benchmark"],
                state={},
                min_speedup=threshold,
                preflight_passed=True,
            )

    for task, threshold, expect_failure in (
        (passed, None, False),
        (passed, 0.8, False),
        (passed, 0.9, True),
        (below_baseline, None, False),
        (below_baseline, 0.1, True),
        (failed, None, True),
        (failed, 0.1, True),
    ):
        result = outcome(task, threshold)
        equivalent = result["summary"]["equivalent_performance"]
        native = result["summary"]["performance"]
        require(
            result["failed"] == expect_failure
            and equivalent["case_count"] == 1
            and equivalent["gate_enabled"] == (threshold is not None)
            and equivalent["gate_passed"]
            is (None if threshold is None else not expect_failure)
            and equivalent["metric"] == hygon.EQUIVALENT_METRIC
            and native["metric"] == hygon.SPEEDUP_METRIC
            and native["case_count"] == 0
            and native["gate_passed"] is None,
            "default evaluation enabled a gate, execution failure was hidden, "
            "or explicit --min-speedup relaxed 0.8",
        )
    with tempfile.TemporaryDirectory(
        prefix="flagdnn-equivalent-batch-"
    ) as tmp:
        root = Path(tmp)
        build, output = root / "build", root / "results"
        fixture = root / "case.py"
        fixture.write_text(
            "print(" + repr(json.dumps(valid)) + ")\n"
            "print('FLAGDNN_DIV_BENCHMARK: "
            "PASS cases=1 executed=1 skipped=0')\n"
        )
        runner.isolated_ctest_directory(
            build,
            [
                {
                    "name": "benchmark.hygon.div",
                    "command": [sys.executable, str(fixture)],
                    "properties": [],
                }
            ],
        )
        (build / "CMakeCache.txt").write_text(
            "FLAGDNN_BACKENDS:STRING=hygon\n"
        )
        process = subprocess.run(
            [
                sys.executable,
                str(Path(runner.__file__).resolve()),
                "--build-dir",
                str(build),
                "--platform",
                "hygon",
                "--output-dir",
                str(output),
                "--gpus",
                "0",
                "--ops",
                "div",
                "--suites",
                "benchmark",
                "--no-preflight",
                "--min-speedup",
                "0.9",
                "--color",
                "never",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        require(
            process.returncode == 1,
            "strict equivalent batch gate did not fail: "
            + process.stdout
            + process.stderr,
        )
        merged = json.loads((output / "summary.json").read_text())
        partition = json.loads((output / "summary0.json").read_text())
        performance = merged["result"]["div"]["performance"]
        detail = performance["data"]["fp32"]["details"][case]
        require(
            merged["overall_status"] == "failed"
            and merged["exit_code"] == 1
            and merged["equivalent_performance"]["requested_min_speedup"]
            == 0.9
            and not merged["equivalent_performance"]["gate_passed"]
            and merged["equivalent_performance"]["gate_enabled"]
            and performance["status"] == "Failed"
            and performance["data"]["fp32"]["result"] == "FAIL"
            and detail["equivalent_baseline"]["passed"]
            and detail["performance_gate"]["enabled"]
            and not detail["performance_gate"]["passed"]
            and detail["performance_gate"]["required_relative_efficiency"]
            == 0.9
            and partition["div"]["performance"] == performance,
            "batch finalization lost the stricter threshold "
            "or kept a false pass",
        )
        single_output = root / "single.json"
        process = subprocess.run(
            [
                sys.executable,
                str(Path(runner.__file__).resolve()),
                "--build-dir",
                str(build),
                "--platform",
                "hygon",
                "--output",
                str(single_output),
                "--ops",
                "div",
                "--suites",
                "benchmark",
                "--no-preflight",
                "--min-speedup",
                "0.9",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        require(
            process.returncode == 1,
            "strict equivalent single gate did not fail: "
            + process.stdout
            + process.stderr,
        )
        single = json.loads(single_output.read_text())
        performance = single["result"]["div"]["performance"]
        detail = performance["data"]["fp32"]["details"][case]
        data_file = json.loads(
            (single_output.parent / performance["data_file"]).read_text()
        )["div"]
        exported = data_file["details"][0]["result"][0]
        diagnostic = json.loads(
            runner.diagnostic_path(single_output).read_text()
        )
        require(
            single["overall_status"] == "failed"
            and single["exit_code"] == 1
            and performance["status"] == "Failed"
            and performance["exit_code"] == 1
            and performance["data"]["fp32"]["result"] == "FAIL"
            and detail["performance_gate"]["enabled"]
            and not detail["performance_gate"]["passed"]
            and detail["performance_gate"]["required_relative_efficiency"]
            == 0.9
            and detail["equivalent_baseline"]["passed"]
            and data_file["result"] == "failed"
            and exported["performance_gate"] == detail["performance_gate"]
            and exported["equivalent_baseline"]
            == detail["equivalent_baseline"]
            and diagnostic["overall_status"] == "failed"
            and diagnostic["results"]["div"]["benchmark"]["records"][case][
                "flagdnn"
            ]
            == valid,
            "single-mode final gate disagrees with its data file "
            "or overwrites original baseline/samples",
        )
        low_record = measured(2.5)

        def fixture_source(record, trailer):
            return (
                "print("
                + repr(json.dumps(record))
                + ", flush=True)\n"
                + trailer
            )

        passing_trailer = (
            "print('FLAGDNN_DIV_BENCHMARK: "
            "PASS cases=1 executed=1 skipped=0', flush=True)\n"
        )
        fixture.write_text(fixture_source(low_record, passing_trailer))
        common_arguments = [
            sys.executable,
            str(Path(runner.__file__).resolve()),
            "--build-dir",
            str(build),
            "--platform",
            "hygon",
            "--ops",
            "div",
            "--suites",
            "benchmark",
            "--no-preflight",
        ]
        for mode in ("single", "batch"):
            target = root / f"default-{mode}"
            mode_arguments = (
                ["--output", str(target)]
                if mode == "single"
                else [
                    "--output-dir",
                    str(target),
                    "--gpus",
                    "0",
                    "--dump-output",
                ]
            )
            process = subprocess.run(
                common_arguments + mode_arguments,
                capture_output=True,
                text=True,
                timeout=60,
            )
            require(
                process.returncode == 0,
                f"default below-baseline {mode} run failed: "
                + process.stdout
                + process.stderr,
            )
            require(
                "equivalent performance gate:" not in process.stdout
                and "  FAIL op=" not in process.stdout
                and "informational" in process.stdout
                and "BELOW_BASELINE" in process.stdout,
                f"default {mode} console output mislabeled evaluation "
                "as an enabled performance gate",
            )
            report = json.loads(
                (
                    target if mode == "single" else target / "summary.json"
                ).read_text()
            )
            performance = report["result"]["div"]["performance"]
            detail = performance["data"]["fp32"]["details"][case]
            gate = detail["performance_gate"]
            evaluation = report["equivalent_performance"]
            require(
                report["overall_status"] == "passed"
                and report["exit_code"] == 0
                and performance["status"] == "Passed"
                and performance["exit_code"] == 0
                and performance["data"]["fp32"]["result"] == "OK"
                and not evaluation["gate_enabled"]
                and evaluation["gate_passed"] is None
                and evaluation["failed_case_count"] == 1
                and not evaluation["cases"][0]["passed"]
                and not gate["enabled"]
                and gate["passed"] is None
                and gate["required_relative_efficiency"] is None
                and detail["equivalent_baseline"]
                == low_record["equivalent_baseline"],
                f"default {mode} report turned baseline evaluation "
                "into an implicit failure",
            )
            if mode == "single":
                diagnostic = json.loads(
                    runner.diagnostic_path(target).read_text()
                )
                measured_task = diagnostic["results"]["div"]["benchmark"]
                raw_record = measured_task["records"][case]["flagdnn"]
                exported = json.loads(
                    (target.parent / performance["data_file"]).read_text()
                )["div"]
                require(
                    measured_task.get("performance_warnings")
                    and not measured_task.get("performance_errors")
                    and exported["result"] == "passed"
                    and exported["details"][0]["result"][0]["performance_gate"]
                    == gate,
                    "default single-run diagnostics/data file lost "
                    "informational gate semantics",
                )
            else:
                partition = json.loads((target / "summary0.json").read_text())
                require(
                    partition["div"]["performance"] == performance,
                    "default batch partition disagrees with merged report",
                )
                raw_records, errors = runner.benchmark_records(
                    (target / "div/performance_stdout.log").read_text(), hygon
                )
                require(not errors, "default batch raw record became invalid")
                raw_record = raw_records[case]["flagdnn"]
            require(
                raw_record == low_record
                and raw_record["equivalent_baseline"]["passed"] is False,
                f"default {mode} reporting overwrote baseline or real samples",
            )

        explicit_output = root / "explicit-low.json"
        process = subprocess.run(
            common_arguments
            + ["--output", str(explicit_output), "--min-speedup", "0.1"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        report = json.loads(explicit_output.read_text())
        performance = report["result"]["div"]["performance"]
        require(
            process.returncode == 1
            and performance["status"] == "Failed"
            and performance["data"]["fp32"]["result"] == "FAIL"
            and report["equivalent_performance"]["gate_enabled"]
            and report["equivalent_performance"]["gate_passed"] is False,
            "an explicit low threshold relaxed the minimum baseline gate",
        )
        for label, source, extra_arguments, expected_status in (
            (
                "nonzero",
                fixture_source(
                    low_record,
                    "print('FLAGDNN_DIV_BENCHMARK: FAIL "
                    "cases=1 executed=1 skipped=0', flush=True)\n"
                    "raise SystemExit(7)\n",
                ),
                [],
                "Failed",
            ),
            (
                "precision",
                fixture_source(
                    low_record,
                    "print('[FAIL][cpu_reference] precision mismatch', "
                    "flush=True)\n"
                    "raise SystemExit(1)\n",
                ),
                [],
                "Failed",
            ),
            (
                "malformed",
                fixture_source(
                    {**low_record, "samples": [1.0]}, passing_trailer
                ),
                [],
                "Failed",
            ),
            (
                "timeout",
                fixture_source(low_record, "import time\ntime.sleep(30)\n"),
                ["--timeout", "1"],
                "Timeout",
            ),
        ):
            fixture.write_text(source)
            target = root / f"default-error-{label}.json"
            process = subprocess.run(
                common_arguments + ["--output", str(target)] + extra_arguments,
                capture_output=True,
                text=True,
                timeout=60,
            )
            report = json.loads(target.read_text())
            diagnostic = json.loads(runner.diagnostic_path(target).read_text())
            require(
                process.returncode == 1
                and diagnostic["overall_status"] == "failed"
                and report["result"]["div"]["performance"]["status"]
                == expected_status,
                f"disabled performance gate concealed a real {label} failure",
            )
    require(
        "integration.hygon.div_performance_baseline"
        in hygon.preflight_tests(["benchmark"])
        and "integration.hygon.div_performance_baseline"
        not in hygon.preflight_tests(["functional"]),
        "div equivalent-baseline preflight escaped benchmark scope",
    )


def check_hygon_baseline_skip_scope(runner, hygon) -> None:
    """Only known auxiliary cases can claim a missing NVIDIA baseline."""
    boolean_shapes = (
        "1x1x16",
        "2x8x8",
        "1x8x4x8",
        "2x8x4x8",
        "1x1x8",
        "1x1x24",
        "1x1x128",
        "1x1x256",
        "3x5x32",
        "1x16x3x5",
        "2x16x7x9",
        "3x32x5x7",
    )
    index_shapes = (
        (1,),
        (17,),
        (33,),
        (2, 3),
        (3, 7),
        (2, 3, 5),
        (1, 3, 17),
        (2, 4, 33),
        (2, 3, 4, 5),
        (1, 7, 3, 9),
        (2, 2, 2, 2, 3),
        (1, 2, 1, 3, 1, 2, 1, 5),
        (3, 257),
        (2, 1025),
    )
    allowed = {"concatenate": []}
    for index, shape in enumerate(index_shapes):
        axis = index % len(shape) - (len(shape) if index % 2 else 0)
        dimensions = "_".join(map(str, shape))
        for dtype in ("int32", "bool", "fp8_e4m3", "fp8_e5m2", "fp8_e8m0"):
            allowed["concatenate"].append(
                f"concatenate_{dtype}_{dimensions}_axis{axis}"
            )
    require(
        {operator: len(cases) for operator, cases in allowed.items()}
        == {"concatenate": 70},
        "NVIDIA auxiliary exclusion fixture coverage changed",
    )

    def processed(operator, cases):
        marker = f"FLAGDNN_{operator.upper()}_BENCHMARK"
        suffix = ".boolean" if operator.startswith("logical_") else ""
        output = "".join(
            f"1: [SKIP][nvidia_baseline] op={operator} "
            f"case={case} reason=no matching NVIDIA reference\n"
            for case in cases
        ) + (
            f"1: {marker}: SKIP cases={len(cases)} "
            f"executed=0 skipped={len(cases)}\n"
            f"1/1 Test #1: benchmark.hygon.{operator}{suffix} "
            ".... ***Skipped 0.01 sec\n"
        )
        task = {"status": "skipped", "exit_code": 0, "records": {}}
        hygon.postprocess_result(
            result=task,
            ctest_reported_status="skipped",
            output=output,
            operator=operator,
            suite="benchmark",
            records={},
            manifest_operators=[operator],
        )
        with contextlib.redirect_stdout(io.StringIO()):
            outcome = hygon.finalize(
                results={operator: {"benchmark": task}},
                suite_operators={"benchmark": [operator]},
                suites=["benchmark"],
                state={},
                min_speedup=None,
                preflight_passed=True,
            )
        return task, outcome

    for operator, cases in allowed.items():
        task, outcome = processed(operator, cases)
        require(
            task["status"] == "skipped"
            and not task.get("skip_record_errors")
            and not task.get("case_accounting_errors")
            and outcome["coverage"]["nvidia_baseline_skips"] == len(cases),
            f"legitimate {operator} auxiliary exclusions were rejected",
        )
    rejected = (
        ("div", "div_perf_fp32_1x1x1024_by_1x1x1024"),
        ("div", "div_no_such_case"),
        ("logical_and", "logical_and_perf_bool_1x1x1024_by_1x1x1024"),
        ("logical_or", "logical_or_perf_bool_1x1x1024_by_1x1x1024"),
        ("logical_and", "logical_and_bool_1x1x17"),
        ("logical_and", "logical_and_fp32_1x1x16"),
        ("logical_not", "logical_not_bool_1x1x16"),
        ("concatenate", "concatenate_fp32_1_axis0"),
        ("concatenate", "concatenate_bf16_2_1025_axis-1"),
        ("concatenate", "concatenate_int32_7_axis0"),
        ("concatenate", "concatenate_int32_1_axis1"),
        ("concatenate", "concatenate_int64_1_axis0"),
        ("add", "add_fp32_1"),
    )
    for operator, case in rejected:
        task, outcome = processed(operator, [case])
        require(
            task["status"] == "failed"
            and task.get("skip_record_errors")
            and outcome["coverage"]["nvidia_baseline_skips"] == 0,
            f"unproven NVIDIA baseline absence was accepted for {case}",
        )
    for operator in ("logical_and", "logical_or"):
        task, outcome = processed(
            operator, [f"{operator}_bool_{shape}" for shape in boolean_shapes]
        )
        require(
            task["status"] == "failed"
            and task.get("skip_record_errors")
            and outcome["coverage"]["nvidia_baseline_skips"] == 0,
            "newly measured auxiliary BOOL cases retained a missing-NVIDIA exemption",
        )
    variable = "FLAGDNN_PERFORMANCE_BASELINE"
    previous_baseline = os.environ.get(variable)
    try:
        with tempfile.TemporaryDirectory(
            prefix="flagdnn-baseline-skip-evidence-"
        ) as tmp:
            override = Path(tmp) / "baseline.json"
            os.environ[variable] = str(override)
            task, _ = processed("concatenate", allowed["concatenate"][:1])
            require(
                task["status"] == "failed" and task.get("skip_record_errors"),
                "missing override baseline silently "
                "used default skip evidence",
            )
            for contents in (
                "{invalid JSON",
                json.dumps({"reference": {"cases": {}}}),
                json.dumps(
                    {"reference": {"cases": [], "unavailable_cases": {}}}
                ),
                json.dumps(
                    {
                        "reference": {
                            "cases": {allowed["concatenate"][0]: {}},
                            "unavailable_cases": {},
                        }
                    }
                ),
            ):
                override.write_text(contents)
                task, _ = processed("concatenate", allowed["concatenate"][:1])
                require(
                    task["status"] == "failed"
                    and task.get("skip_record_errors"),
                    "malformed/contradictory override baseline "
                    "was ignored by skip validation",
                )
            os.environ[variable] = ""
            task, _ = processed("concatenate", allowed["concatenate"][:1])
            require(
                task["status"] == "skipped",
                "empty baseline override did not select the default evidence",
            )
    finally:
        if previous_baseline is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous_baseline


def check_performance_baseline_policy_isolation(runner, hygon) -> None:
    """Target scaling policies must not redefine the shared NVIDIA data."""
    root = Path(runner.__file__).resolve().parents[1]
    catalog = json.loads((root / "docs/performance-baseline.json").read_text())
    reference = catalog["reference"]
    cases = reference["cases"]
    require(
        reference["platform"] == "nvidia"
        and len(cases) == reference["coverage"]["case_count"] == 3264
        and len({case["op"] for case in cases.values()})
        == reference["coverage"]["operator_count"]
        == 82,
        "shared NVIDIA reference coverage changed during policy isolation",
    )
    require(
        "case_policies" not in reference["scaling"]
        and reference["scaling"]["status"] == "unverified"
        and all(
            "scale_metric" in case and case["scale_metric"] is None
            for case in cases.values()
        ),
        "a target-specific scaling policy leaked into shared NVIDIA data",
    )
    expected_groups = {
        "vector_fp32": {
            "add_square",
            "elu",
            "sigmoid",
            "sigmoid_backward",
            "swish",
            "swish_backward",
            "tanh",
            "tanh_backward",
            "batchnorm",
            "reduction",
            "cos",
            "div",
            "elu_backward",
            "erf",
            "exp",
            "gelu",
            "gelu_approx_tanh",
            "gelu_approx_tanh_backward",
            "gelu_backward",
            "log",
            "mod",
            "pow",
            "reciprocal",
            "rsqrt",
            "sin",
            "softplus",
            "softplus_backward",
            "sqrt",
            "tan",
        },
        "hbm_copy": {
            "abs",
            "relu",
            "identity",
            "ceil",
            "concatenate",
            "floor",
            "leaky_relu",
            "logical_not",
            "neg",
            "reshape",
            "slice",
            "transpose",
        },
        "hbm_triad": {
            "add",
            "sub",
            "mul",
            "min",
            "max",
            "scale",
            "relu_backward",
            "binary_select",
            "cmp_eq",
            "cmp_ge",
            "cmp_gt",
            "cmp_le",
            "cmp_lt",
            "cmp_neq",
            "leaky_relu_backward",
            "logical_and",
            "logical_or",
        },
        "hbm_write": {"gen_index"},
    }
    expected_metrics = {
        operator: metric
        for metric, operators in expected_groups.items()
        for operator in operators
    }
    platforms = catalog["platforms"]
    policies = platforms["hygon"]["scaling"]["case_policies"]
    require(
        len(policies) == 59
        and set(policies) == hygon.EQUIVALENT_OPERATORS
        and {op: policy["scale_metric"] for op, policy in policies.items()}
        == expected_metrics,
        "Hygon-owned policies changed their operator/metric assignments",
    )
    require(
        Counter(
            policies[case["op"]]["scale_metric"]
            for case in cases.values()
            if case["op"] in policies
        )
        == {
            "vector_fp32": 1022,
            "hbm_copy": 444,
            "hbm_triad": 480,
            "hbm_write": 28,
        },
        # This is policy-by-op reference coverage (1974), not the runtime
        # catalog (1941). Six old backward-attribute records remain outside
        # the new BF16-perf-only scope; 27 older reference-only cases were
        # already outside the generated runtime catalog.
        "Hygon-owned policy reference coverage changed during isolation",
    )
    ascend = platforms["ascend"]
    ascend_policies = ascend.get("scaling", {}).get("case_policies", {})
    require(
        set(ascend_policies) == {"identity"}
        and ascend_policies["identity"]["scale_metric"] == "hbm_copy"
        and set(ascend_policies["identity"]["supported_dtypes"])
        == {"fp8_e4m3", "fp8_e5m2"}
        and ascend_policies["identity"].get("reason")
        and ascend["device"] != platforms["hygon"]["device"]
        and ascend["architecture"]
        and ascend["measured_on"]
        and ascend["peaks"]["hbm_copy"] > 0
        and all(value is None for key, value in ascend["peaks"].items()
                if key != "hbm_copy"),
        "Ascend must own its measured copy peak and FP8-only identity policy",
    )
    mthreads = platforms["mthreads"]
    mthreads_policies = mthreads["scaling"]["case_policies"]
    require(
        {op: policy["scale_metric"] for op, policy in mthreads_policies.items()}
        == {
            "elu_backward": "vector_fp32",
            "softplus_backward": "vector_fp32",
            "swish_backward": "vector_fp32",
            "relu_backward": "hbm_triad",
            "leaky_relu_backward": "hbm_triad",
        },
        "MThreads-owned policies changed their operator/metric assignments",
    )
    require(
        all(
            set(policy["supported_dtypes"]) == {"fp32", "fp16", "bfloat16"}
            for policy in mthreads_policies.values()
        )
        and mthreads["device"] == "MTT S5000"
        and mthreads["architecture"] == "mp_31"
        and all(
            isinstance(mthreads["peaks"][metric], (int, float))
            and mthreads["peaks"][metric] > 0
            for metric in {"vector_fp32", "hbm_triad"}
        )
        and mthreads["measurement"]["sample_count"] > 0
        and mthreads["measurement"]["raw_archive_sha256"],
        "MThreads policies must retain their measured device evidence and dtype scope",
    )
    for name, platform in platforms.items():
        if name in {"hygon", "ascend", "mthreads"}:
            continue
        require(
            not platform.get("scaling", {}).get("case_policies"),
            f"{name} inherited Hygon-specific scaling policies",
        )


def check_general_equivalent_performance(runner, hygon) -> None:
    """Exercise bandwidth normalization and mixed native/equivalent suites."""

    def measured(
        case,
        metric="vector_fp32",
        traffic=1.0,
        validation="cpu_full",
        efficiency=1.0,
    ):
        equivalent = 2.0 * 4.0 / 5.0 * traffic
        median = equivalent / efficiency
        return {
            "schema_version": 4,
            "kind": "steady_state",
            "provider": "flagdnn",
            "case": case,
            "unit": "us",
            "median": median,
            "p90": median,
            "samples": [median]
            * (10 if case.startswith("batchnorm_") else 20),
            "equivalent_baseline": {
                "provider": "nvidia_equivalent",
                "platform": "nvidia",
                "target_platform": "hygon",
                "scale_metric": metric,
                "timing_profile": (
                    "batchnorm_training"
                    if case.startswith("batchnorm_")
                    else "native"
                ),
                "reference_time_us": 2.0,
                "reference_peak": 4.0,
                "target_peak": 5.0,
                "min_relative_efficiency": 0.8,
                "traffic_ratio": traffic,
                "reference_validation": validation,
                "equivalent_time_us": equivalent,
                "limit_us": equivalent / 0.8,
                "relative_efficiency": equivalent / median,
                "passed": median <= equivalent / 0.8,
            },
        }

    root = Path(runner.__file__).resolve().parents[1]
    catalog = json.loads((root / "docs/performance-baseline.json").read_text())
    policies = catalog["platforms"]["hygon"]["scaling"]["case_policies"]
    schema = json.loads((root / "benchmark/result.schema.json").read_text())
    case_pattern = schema["$defs"]["v4"]["properties"]["case"]["pattern"]
    seen = set()
    accepted_references = 0
    supplemental_references = 0
    for case, reference in catalog["reference"]["cases"].items():
        operator = reference["op"]
        if operator not in hygon.EQUIVALENT_OPERATORS:
            continue
        if hygon.EQUIVALENT_CASE_PATTERN.fullmatch(case) is None:
            require(
                operator in {"relu_backward", "swish_backward"}
                and case.startswith(operator + "_attributes_"),
                f"unexpected policy/reference case outside runtime scope: {case}",
            )
            continue
        seen.add(operator)
        accepted_references += 1
        supplemental_references += (
            reference.get("measurement_id") == "nvidia-missing-20261012"
        )
        record = measured(
            case,
            metric=policies[operator]["scale_metric"],
            validation=reference["validation"],
            traffic=8.0 if case.startswith("identity_bool_") else 1.0,
        )
        records, errors = runner.benchmark_records(json.dumps(record), hygon)
        require(
            not errors
            and re.fullmatch(case_pattern, case) is not None
            and not hygon.validate_hygon_benchmark_pairs(
                records, operator, sorted(hygon.EQUIVALENT_OPERATORS)
            ),
            f"reference case is excluded by generalized v4 identity: {case}",
        )
    require(
        len(seen) == 59
        and seen == hygon.EQUIVALENT_OPERATORS
        and accepted_references == 1968
        and supplemental_references == 371,
        "equivalent validation must cover 1968 reference records across 59 ops, including all 371 supplemental records",
    )

    examples = (
        ("identity_fp8_e4m3_2x3x4", "hbm_copy", 1.0),
        ("identity_fp8_e5m2_2x3x4", "hbm_copy", 1.0),
        ("identity_perf_bfloat16_2x3x4", "hbm_copy", 1.0),
        ("identity_bfloat16_2x3x4", "hbm_copy", 1.0),
        ("identity_int32_2x3x4", "hbm_copy", 1.0),
        ("identity_bool_1x1x16", "hbm_copy", 8.0),
        ("add_perf_bfloat16_1x1x1024_by_1x1x1024", "hbm_triad", 1.0),
        ("relu_backward_perf_bfloat16_1x1x1024_by_1x1x1024", "hbm_triad", 1.0),
        ("batchnorm_perf_fp16_1x128x8x8", "vector_fp32", 1.0),
        ("batchnorm_perf_bfloat16_1x128x8x8", "vector_fp32", 1.0),
        ("reduction_sum_bfloat16_to_fp32_3_7", "vector_fp32", 1.0),
        ("reduction_avg_fp16_to_fp32_3_5_7", "vector_fp32", 1.0),
        ("logical_and_bool_1x1x16", "hbm_triad", 8.0),
        ("logical_or_bool_1x1x16", "hbm_triad", 8.0),
        ("logical_and_perf_bool_1x1x1024_by_1x1x1024", "hbm_triad", 8.0),
        ("cmp_eq_perf_fp32_1x1x1024_by_1x1x1024", "hbm_triad", 72 / 65),
        ("cmp_eq_perf_fp16_1x1x1024_by_1x1x1024", "hbm_triad", 40 / 33),
        ("reshape_fp32_2x3x4_to_6x4", "hbm_copy", 1.0),
        ("gen_index_int32_3_axis0", "hbm_write", 1.0),
        ("concatenate_bf16_2_1025_axis-1", "hbm_copy", 1.0),
        (
            "gelu_approx_tanh_backward_perf_fp32_1x1x1024_by_1x1x1024",
            "vector_fp32",
            1.0,
        ),
    )
    for case, metric, ratio in examples:
        operator = hygon.EQUIVALENT_CASE_PATTERN.fullmatch(case).group(
            "operator"
        )
        validations = (
            ("cpu_full",)
            if operator
            in hygon.EQUIVALENT_BFLOAT16_OPERATORS | {"batchnorm", "reduction"}
            or case.startswith("identity_perf_")
            else (
                ("bit_exact",)
                if case.startswith("identity_")
                else ("cpu_full", "cpu_sampled", "bit_exact", "sanity")
            )
        )
        for validation in validations:
            record = measured(case, metric, ratio, validation)
            parsed, errors = runner.benchmark_records(
                json.dumps(record), hygon
            )
            require(
                not errors and parsed[case]["flagdnn"] == record,
                f"traffic or reference validation was lost for {case}",
            )
            task = {"status": "passed", "records": parsed}
            detail = next(iter(runner.performance_data(task).values()))[
                "details"
            ][case]
            require(
                detail["equivalent_baseline"] == record["equivalent_baseline"]
                and detail["speedup"] == 1.0,
                f"reported bandwidth ratio is not normalized for {case}",
            )

    # Profiles are operator-specific; accepting ten samples under the native
    # profile would silently change the established 10/20/50 timing contract.
    for case in (
        "batchnorm_perf_fp16_1x128x8x8",
        "batchnorm_perf_bfloat16_1x128x8x8",
        "add_perf_bfloat16_1x1x1024_by_1x1x1024",
        "reduction_sum_bfloat16_to_fp32_3_7",
    ):
        valid = measured(case)
        expected_profile = (
            "batchnorm_training" if case.startswith("batchnorm_") else "native"
        )
        for profile in (
            "attention",
            "unknown",
            None,
            True,
            (
                "native"
                if expected_profile == "batchnorm_training"
                else "batchnorm_training"
            ),
        ):
            changed = {
                **valid,
                "equivalent_baseline": {
                    **valid["equivalent_baseline"],
                    "timing_profile": profile,
                },
            }
            parsed, errors = runner.benchmark_records(
                json.dumps(changed), hygon
            )
            require(
                errors and not parsed,
                f"incorrect {profile} profile accepted for {case}",
            )
        for count in (
            0,
            1,
            9,
            11,
            19,
            21,
            20 if expected_profile == "batchnorm_training" else 10,
        ):
            parsed, errors = runner.benchmark_records(
                json.dumps({**valid, "samples": [valid["median"]] * count}),
                hygon,
            )
            require(
                errors and not parsed,
                f"incorrect sample count {count} accepted for {case}",
            )
        for validation in ("bit_exact", "sanity", "cpu_sampled"):
            changed = {
                **valid,
                "equivalent_baseline": {
                    **valid["equivalent_baseline"],
                    "reference_validation": validation,
                },
            }
            parsed, errors = runner.benchmark_records(
                json.dumps(changed), hygon
            )
            require(
                errors and not parsed,
                f"unmeasured validation level accepted for {case}",
            )
    for case, validation, traffic in (
        ("identity_perf_bfloat16_2x3x4", "bit_exact", 1.0),
        ("identity_bfloat16_2x3x4", "cpu_full", 1.0),
        ("identity_int32_2x3x4", "cpu_full", 1.0),
        ("identity_bool_1x1x16", "bit_exact", 1.0),
        ("identity_int32_2x3x4", "bit_exact", 8.0),
    ):
        parsed, errors = runner.benchmark_records(
            json.dumps(measured(case, "hbm_copy", traffic, validation)), hygon
        )
        require(
            errors and not parsed,
            f"identity validation/traffic mismatch accepted for {case}",
        )

    for field, value in (
        ("reference_validation", "cpu_full"),
        ("reference_validation", "sanity"),
        ("scale_metric", "vector_fp32"),
        ("traffic_ratio", 8.0),
    ):
        identity = measured(
            "identity_fp8_e4m3_2x3x4", "hbm_copy", 1.0, "bit_exact"
        )
        identity["equivalent_baseline"][field] = value
        parsed, errors = runner.benchmark_records(json.dumps(identity), hygon)
        require(
            errors and not parsed, f"invalid FP8 identity {field} accepted"
        )

    # BatchNorm's native FP32 pair and new low-precision equivalent results
    # coexist in one suite, with independent metrics and the shorter profile.
    bn_native_case = "batchnorm_perf_fp32_1x128x8x8"
    bn_equivalent_case = "batchnorm_perf_fp16_1x128x8x8"
    bn_equivalent = measured(bn_equivalent_case, efficiency=0.79)
    bn_rows = [bn_equivalent]
    for provider in ("flagdnn", "hipdnn"):
        bn_rows.append(
            {
                "schema_version": 5,
                "kind": "steady_state",
                "provider": provider,
                "case": bn_native_case,
                "unit": "us",
                "median": 2.0,
                "p90": 2.0,
                "samples": [2.0] * 10,
                "timing_mode": "hip_graph_batch",
                "timing_reason": "standard",
                "benchmark_config": {
                    "warmup_iterations": 5,
                    "sample_count": 10,
                    "iterations_per_sample": 20,
                },
            }
        )
    bn_records, errors = runner.benchmark_records(
        "\n".join(map(json.dumps, bn_rows)), hygon
    )
    require(not errors, "mixed BatchNorm profiles were rejected")
    bn_task = {"status": "passed", "exit_code": 0, "records": bn_records}
    bn_output = (
        "FLAGDNN_BATCHNORM_BENCHMARK: PASS cases=2 executed=2 skipped=0\n"
    )
    hygon.postprocess_result(
        result=bn_task,
        ctest_reported_status="passed",
        output=bn_output,
        operator="batchnorm",
        suite="benchmark",
        records=bn_records,
        manifest_operators=["batchnorm"],
    )
    require(
        bn_task["status"] == "passed"
        and hygon.complete_benchmark(bn_task)
        and len(bn_task.get("performance_warnings", [])) == 1,
        "below-baseline BatchNorm became an execution failure without a CLI gate",
    )
    bn_results = {"batchnorm": {"benchmark": bn_task}}
    bn_native = hygon.benchmark_speedup_summary(bn_results, None)
    bn_normalized = hygon.equivalent_performance_summary(bn_results, None)
    bn_gated = hygon.equivalent_performance_summary(bn_results, 0.9)
    require(
        bn_native["case_count"] == bn_normalized["case_count"] == 1
        and bn_native["gate_passed"]
        and bn_normalized["failed_case_count"] == 1
        and bn_normalized["gate_passed"] is None
        and bn_gated["gate_passed"] is False,
        "mixed BatchNorm native/equivalent statistics or opt-in gating differ",
    )
    bn_public = runner.suite_result(bn_task, "benchmark")
    bn_detail = bn_public["data"]["fp16"]["details"][bn_equivalent_case]
    require(
        bn_public["status"] == "Passed"
        and bn_detail["equivalent_baseline"]
        == bn_equivalent["equivalent_baseline"]
        and len(bn_records[bn_equivalent_case]["flagdnn"]["samples"]) == 10,
        "BatchNorm profile in the public report or measured samples in raw records were lost",
    )
    bn_legacy = runner.performance_file("batchnorm", bn_task, bn_public)
    require(
        "batchnorm_training" in json.dumps(bn_legacy),
        "BatchNorm profile was lost in the performance datafile",
    )
    broken_bn = dict(bn_task, status="failed", exit_code=1)
    hygon.postprocess_result(
        result=broken_bn,
        ctest_reported_status="failed",
        output=bn_output,
        operator="batchnorm",
        suite="benchmark",
        records=bn_records,
        manifest_operators=["batchnorm"],
    )
    require(
        broken_bn["status"] == "failed" and broken_bn["exit_code"] == 1,
        "BatchNorm equivalent records concealed a real execution failure",
    )

    case = "cmp_eq_perf_fp32_1x1x1024_by_1x1x1024"
    valid = measured(case, "hbm_triad", 72 / 65)
    mutations = [
        ("traffic_ratio", value)
        for value in (0, -1, None, True, "8", float("nan"), float("inf"))
    ] + [
        ("traffic_ratio", 1),  # Derived values must change with traffic.
        ("reference_validation", "unknown"),
        ("reference_validation", []),
        ("reference_validation", None),
        ("scale_metric", "hbm_read"),
        ("scale_metric", []),
        ("scale_metric", "vector_fp32"),  # Non-unit traffic is forbidden.
    ]
    for key, value in mutations:
        record = {
            **valid,
            "equivalent_baseline": {
                **valid["equivalent_baseline"],
                key: value,
            },
        }
        parsed, errors = runner.benchmark_records(json.dumps(record), hygon)
        require(
            errors and not parsed, f"invalid generalized field {key} accepted"
        )
    for key in ("traffic_ratio", "reference_validation"):
        baseline = dict(valid["equivalent_baseline"])
        del baseline[key]
        parsed, errors = runner.benchmark_records(
            json.dumps({**valid, "equivalent_baseline": baseline}), hygon
        )
        require(errors and not parsed, f"new operator omitted mandatory {key}")
    for invalid_case in (
        "add_perf_fp32_1_by_1",
        "pow_unknown_fp32_1_by_1",
        "gelu_other_perf_fp32_1024",
        "logical_or_perf_fp64_1_by_1",
        "identity_fp32_2x3x4",
        "identity_fp16_2x3x4",
        "identity_fp8_e8m0_2x3x4",
        "identity_perf_fp8_e4m3_2x3x4",
        "neg_fp8_e4m3_2x3x4",
        "add_perf_fp16_1x1x1024_by_1x1x1024",
        "relu_backward_attributes_bfloat16_1x1x65",
        "swish_backward_attributes_bfloat16_1x1x65",
        "identity_perf_int32_2x3x4",
        "batchnorm_perf_fp32_1x128x8x8",
        "batchnorm_backward_perf_fp16_1x128x8x8",
        "batchnorm_attributes_fp16_1x128x8x8",
        "reduction_sum_fp32_to_fp32_3_7",
        "reduction_avg_bfloat16_to_fp32_3_7",
        "reduction_mul_bfloat16_to_fp32_3_7",
        "reduction_sum_fp16_to_fp16_3_7",
    ):
        parsed, errors = runner.benchmark_records(
            json.dumps({**valid, "case": invalid_case}), hygon
        )
        require(
            errors and not parsed, f"invalid operator identity {invalid_case}"
        )
    nested_case = examples[-1][0]
    nested = {nested_case: {"flagdnn": measured(nested_case)}}
    require(
        hygon.validate_hygon_benchmark_pairs(nested, "gelu", ["gelu"])
        and hygon.validate_hygon_benchmark_pairs(
            {case: {"flagdnn": measured(nested_case)}}, "cmp_eq", ["cmp_eq"]
        ),
        "longer operator prefixes or mismatched "
        "record identities were accepted",
    )

    native_case = "cmp_eq_perf_fp16_1x1x1024_by_1x1x1024"
    equivalent = measured(case, "hbm_triad", 72 / 65, efficiency=0.79)
    records, errors = runner.benchmark_records(
        "\n".join(
            (
                timing(native_case, "flagdnn"),
                timing(native_case, "hipdnn"),
                json.dumps(equivalent),
            )
        ),
        hygon,
    )
    require(not errors, "valid mixed native/equivalent records were rejected")
    task = {"status": "failed", "exit_code": 1, "records": records}
    output = (
        "1: FLAGDNN_CMP_EQ_BENCHMARK: PASS cases=1 executed=1 skipped=0\n"
        "1/2 Test #1: benchmark.hygon.cmp_eq .... Passed 0.01 sec\n"
        "2: FLAGDNN_CMP_EQ_BENCHMARK: FAIL cases=1 executed=1 skipped=0\n"
        "2/2 Test #2: benchmark.hygon.cmp_eq.boolean .... ***Failed 0.01 sec\n"
    )
    hygon.postprocess_result(
        result=task,
        ctest_reported_status="failed",
        output=output,
        operator="cmp_eq",
        suite="benchmark",
        records=records,
        manifest_operators=["cmp_eq"],
    )
    require(
        task.get("performance_errors") and hygon.complete_benchmark(task),
        "mixed PASS/FAIL children lost completed equivalent failure records",
    )
    public = runner.suite_result(task, "benchmark")
    require(
        public["status"] == "Failed"
        and public["data"]["fp16"]["result"] == "OK"
        and public["data"]["fp32"]["result"] == "OK"
        and set(public["data"]["fp16"]["details"]) == {native_case},
        "equivalent failure discarded or rewrote a valid native pair",
    )
    declared = {
        "schema_version": 1,
        "metric": hygon.SPEEDUP_METRIC,
        "source": "fixture",
        "declared_operator_count": 1,
        "declared_case_count": 1,
        "operators": {"cmp_eq": [native_case]},
    }
    results = {"cmp_eq": {"benchmark": task}}
    coverage = hygon.hygon_comparable_coverage(results, ["cmp_eq"], declared)
    native = hygon.benchmark_speedup_summary(results, 1.1, coverage)
    normalized = hygon.equivalent_performance_summary(results, 0.1)
    require(
        coverage["verified"]
        and coverage["observed_pair_case_count"] == 1
        and native["case_count"] == native["failed_case_count"] == 1
        and normalized["case_count"] == normalized["failed_case_count"] == 1
        and normalized["cases"][0]["required_relative_efficiency"] == 0.8,
        "mixed native/equivalent summaries conflated gates "
        "or dropped native data",
    )
    for broken in (
        output.replace(
            "2: FLAGDNN_CMP_EQ_BENCHMARK: FAIL cases=1 executed=1 skipped=0\n",
            "",
        ),
        output.replace("FAIL cases=1 executed=1", "FAIL cases=2 executed=2"),
        output.replace("***Failed", "Passed"),
    ):
        require(
            hygon.validate_hygon_case_accounting(
                broken, "cmp_eq", "benchmark", "failed", records, []
            ),
            "incomplete or inconsistent mixed CTest accounting was accepted",
        )

    layout_case = "concatenate_fp32_1_axis0"
    layout_records = {
        layout_case: {"flagdnn": measured(layout_case, "hbm_copy")}
    }
    auxiliary_output = (
        "2: [SKIP][nvidia_baseline] op=concatenate "
        "case=concatenate_int32_1_axis0 reason=no matching NVIDIA reference\n"
        "2: [SKIP][nvidia_baseline] op=concatenate "
        "case=concatenate_bool_1_axis0 reason=no matching NVIDIA reference\n"
        "2: FLAGDNN_CONCATENATE_BENCHMARK: SKIP cases=2 executed=0 skipped=2\n"
        "2/2 Test #2: benchmark.hygon.concatenate.copy "
        ".... ***Skipped 0.01 sec\n"
    )
    mixed_output = (
        "1: [EXCLUDED][nvidia_baseline] op=concatenate "
        "case=concatenate_excluded reason=not a measured case\n"
        "1: FLAGDNN_CONCATENATE_BENCHMARK: PASS cases=1 executed=1 skipped=0\n"
        "1/2 Test #1: benchmark.hygon.concatenate .... Passed 0.01 sec\n"
        + auxiliary_output
    )
    mixed_task = {
        "status": "skipped",
        "exit_code": 0,
        "records": layout_records,
        "record_errors": ["skipped benchmark emitted timing provider records"],
    }
    hygon.postprocess_result(
        result=mixed_task,
        ctest_reported_status="skipped",
        output=mixed_output,
        operator="concatenate",
        suite="benchmark",
        records=layout_records,
        manifest_operators=["concatenate"],
    )
    require(
        mixed_task["status"] == "passed"
        and hygon.complete_benchmark(mixed_task)
        and len(mixed_task["baseline_skip_records"]) == 2
        and not mixed_task.get("skip_records"),
        "explicit NVIDIA auxiliary exclusions masked the measured main suite",
    )
    with contextlib.redirect_stdout(io.StringIO()):
        outcome = hygon.finalize(
            results={"concatenate": {"benchmark": mixed_task}},
            suite_operators={"benchmark": ["concatenate"]},
            suites=["benchmark"],
            state={},
            min_speedup=None,
            preflight_passed=True,
        )
    require(
        not outcome["failed"]
        and outcome["coverage"]["nvidia_baseline_skips"] == 2
        and outcome["coverage"]["hipdnn_reference_skips"] == 0
        and outcome["summary"]["equivalent_performance"]["case_count"] == 1,
        "NVIDIA baseline exclusions were reported as hipDNN capability skips",
    )
    only_auxiliary = {"status": "skipped", "exit_code": 0, "records": {}}
    hygon.postprocess_result(
        result=only_auxiliary,
        ctest_reported_status="skipped",
        output=auxiliary_output,
        operator="concatenate",
        suite="benchmark",
        records={},
        manifest_operators=["concatenate"],
    )
    require(
        only_auxiliary["status"] == "skipped"
        and not only_auxiliary.get("case_accounting_errors"),
        "an entirely excluded NVIDIA auxiliary catalog was misclassified",
    )
    failed_records = {
        layout_case: {
            "flagdnn": measured(layout_case, "hbm_copy", efficiency=0.79)
        }
    }
    failed_output = mixed_output.replace(
        "BENCHMARK: PASS", "BENCHMARK: FAIL"
    ).replace(".... Passed 0.01 sec", ".... ***Failed 0.01 sec")

    def failed_with_skips(output):
        task = {"status": "failed", "exit_code": 1, "records": failed_records}
        hygon.postprocess_result(
            result=task,
            ctest_reported_status="failed",
            output=output,
            operator="concatenate",
            suite="benchmark",
            records=failed_records,
            manifest_operators=["concatenate"],
        )
        with contextlib.redirect_stdout(io.StringIO()):
            outcome = hygon.finalize(
                results={"concatenate": {"benchmark": task}},
                suite_operators={"benchmark": ["concatenate"]},
                suites=["benchmark"],
                state={},
                min_speedup=None,
                preflight_passed=True,
            )
        return task, outcome

    for provider, counter in (
        ("nvidia_baseline", "nvidia_baseline_skips"),
        ("hipdnn", "hipdnn_reference_skips"),
    ):
        task, outcome = failed_with_skips(
            failed_output.replace("[nvidia_baseline]", f"[{provider}]")
        )
        require(
            hygon.complete_benchmark(task)
            and outcome["failed"]
            and outcome["coverage"][counter] == 2
            and outcome["summary"]["equivalent_performance"]["case_count"] == 1
            and outcome["summary"]["equivalent_performance"][
                "failed_case_count"
            ]
            == 1,
            f"complete performance failure discarded valid {provider} skips",
        )
    for malformed in (
        failed_output.replace(
            "cases=2 executed=0 skipped=2", "cases=3 " "executed=0 skipped=3"
        ),
        failed_output.replace(
            "concatenate_bool_1_axis0", "concatenate_unknown"
        ),
        failed_output.replace(
            "reason=no matching NVIDIA reference", "reason="
        ),
    ):
        task, outcome = failed_with_skips(malformed)
        require(
            not hygon.complete_benchmark(task)
            and (
                task.get("case_accounting_errors")
                or task.get("skip_record_errors")
            )
            and outcome["coverage"]["nvidia_baseline_skips"] == 0
            and outcome["coverage"]["hipdnn_reference_skips"] == 0
            and outcome["summary"]["equivalent_performance"]["case_count"]
            == 0,
            "malformed accounting/skip evidence contributed to coverage",
        )


def main() -> int:
    if len(sys.argv) != 2:
        raise RuntimeError("usage: run_tests_contract.py RUN_TESTS_PY")
    runner = load_runner(Path(sys.argv[1]).resolve())
    check_legacy_summary(runner)
    check_platform_single_report_hooks(runner)
    check_platform_batch_report_hooks(runner)
    check_operator_presets(runner)
    check_batch_runner(runner)
    check_batch_cache_environment(runner)
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
    check_hygon_native_timing(runner, hygon)
    check_equivalent_performance(runner, hygon)
    check_hygon_baseline_skip_scope(runner, hygon)
    check_performance_baseline_policy_isolation(runner, hygon)
    check_general_equivalent_performance(runner, hygon)
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
    diagnostic_catalog = {
        **synthetic_catalog,
        "declared_case_count": 4,
        "operators": {
            "add": [
                "add_perf_complete",
                "add_perf_partial",
                "add_perf_absent",
            ],
            "mul": ["mul_perf_not_run"],
        },
    }
    failed_with_partial_results = {
        "add": {
            "benchmark": {
                "status": "failed",
                "records": {
                    "add_perf_complete": provider_pair("add_perf_complete"),
                    "add_perf_partial": {
                        "flagdnn": json.loads(
                            timing("add_perf_partial", "flagdnn")
                        )
                    },
                },
            }
        }
    }
    diagnosed = hygon.hygon_comparable_coverage(
        failed_with_partial_results, ["add", "mul"], diagnostic_catalog
    )
    require(
        not diagnosed["verified"]
        and diagnosed["observed_required_case_count"] == 0
        and diagnosed["raw_complete_required_case_count"] == 1
        and diagnosed["raw_complete_pair_case_count"] == 1
        and diagnosed["excluded_complete_required_case_count"] == 1
        and diagnosed["missing_case_count"] == 4
        and diagnosed["missing_reason_counts"]
        == {
            "pair_present_but_suite_failed": 1,
            "partial_pair": 1,
            "pair_not_emitted": 1,
            "not_run": 1,
        }
        and [entry["reason"] for entry in diagnosed["missing_cases"]]
        == [
            "pair_present_but_suite_failed",
            "partial_pair",
            "pair_not_emitted",
            "not_run",
        ],
        "comparable coverage confused complete-but-excluded, partial, absent, and unrun cases",
    )
    diagnostics = io.StringIO()
    with contextlib.redirect_stdout(diagnostics):
        diagnostic_outcome = hygon.finalize(
            results=failed_with_partial_results,
            suite_operators={"benchmark": ["add", "mul"]},
            suites=["benchmark"],
            state={"comparable_catalog": diagnostic_catalog},
            min_speedup=None,
            preflight_passed=True,
        )
    require(
        diagnostic_outcome["failed"]
        and "raw_complete=1" in diagnostics.getvalue()
        and "reason=pair_present_but_suite_failed" in diagnostics.getvalue()
        and "missing comparable pair" not in diagnostics.getvalue(),
        "coverage diagnostics still describe complete-but-excluded pairs as missing output",
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

        def empty_selection(
            selected_manifests,
            selected_suites,
            value,
            list_file,
            *,
            appoint=False,
        ):
            return {suite: [] for suite in selected_suites}

        runner.requested_operators = empty_selection
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
        "integration.hygon.python_runtime_lifecycle_contract",
        "integration.hygon.convolution_validation_static_contract",
        "integration.hygon.normalization_stability",
        "integration.hygon.tensor_reference_contract",
        "integration.hygon.div_performance_baseline",
        "integration.hygon.benchmark_validation_guards",
        "integration.hygon.benchmark_timing_policy",
    ):
        require(
            required_contract in hygon_preflight,
            f"Hygon preflight omits {required_contract}",
        )
    require(
        len(hygon_preflight) == 34,
        "Hygon preflight required-test count is not 34",
    )
    functional_preflight = runner.required_preflight_tests(
        "hygon", ["functional"]
    )
    require(
        "benchmark.catalog_contract" not in functional_preflight
        and "benchmark.catalog_dependency_boundary" not in functional_preflight
        and "integration.hygon.benchmark_validation_guards"
        not in functional_preflight
        and "integration.hygon.benchmark_timing_policy"
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
