import json
import logging
import os
from typing import Any

import pytest
import torch

import flag_dnn
from benchmark.attri_util import (
    ALL_AVAILABLE_METRICS,
    BOOL_DTYPES,
    DEFAULT_ITER_COUNT,
    DEFAULT_WARMUP_COUNT,
    FLOAT_DTYPES,
    INT_DTYPES,
    BenchLevel,
    BenchMode,
    OperationAttribute,
    get_recommended_shapes,
)
from flag_dnn.runtime import torch_device_fn

device = flag_dnn.device
vendor_name = flag_dnn.vendor_name
recordLogger = logging.getLogger("flag_dnn_benchmark")
recordLogger.propagate = False
REGISTERED_MARKS: set[str] = set()
TEST_RESULTS: dict[str, Any] = {}
REPORT_FILE = "benchmark_result.json"


def update_result(op, data):
    if not Config.record_json:
        return

    TEST_RESULTS.setdefault(op, {})
    TEST_RESULTS[op].setdefault("details", [])
    TEST_RESULTS[op]["details"].append(data)


def emit_record_logger(message: str) -> None:
    if not Config.record_log:
        return

    if recordLogger.handlers:
        handler = recordLogger.handlers[0]
        if getattr(handler, "stream", None) is None:
            handler.acquire()
            try:
                handler.stream = handler._open()  # type: ignore[attr-defined]
            finally:
                handler.release()
    recordLogger.info(message)


class BenchConfig:
    def __init__(self):
        self.mode = BenchMode.KERNEL
        self.bench_level = BenchLevel.COMPREHENSIVE
        self.warm_up = DEFAULT_WARMUP_COUNT
        self.repetition = DEFAULT_ITER_COUNT
        if (
            vendor_name == "kunlunxin"
        ):  # Speed Up Benchmark Test, Big Shape Will Cause Timeout
            self.warm_up = 1
            self.repetition = 1
        self.record_log = False
        self.record_json = False
        self.output = REPORT_FILE
        self.user_desired_dtypes = None
        self.user_desired_metrics = None
        self.shape_file = os.path.join(
            os.path.dirname(__file__), "core_shapes.yaml"
        )
        self.query = False


Config = BenchConfig()


def pytest_addoption(parser):
    parser.addoption(
        (
            "--mode" if vendor_name != "kunlunxin" else "--fg_mode"
        ),  # TODO: fix pytest-* common --mode args
        action="store",
        default="kernel",
        required=False,
        choices=["kernel", "operator", "wrapper"],
        help=(
            "Specify how to measure latency, 'kernel' for device kernel, "
            "'operator' for end2end operator or 'wrapper' for runtime wrapper."
        ),
    )

    parser.addoption(
        "--level",
        action="store",
        default="comprehensive",
        required=False,
        choices=[level.value for level in BenchLevel],
        help="Specify the benchmark level: comprehensive, or core.",
    )

    parser.addoption(
        "--warmup",
        default=DEFAULT_WARMUP_COUNT,
        help="Number of warmup runs before benchmark run.",
    )

    parser.addoption(
        "--iter",
        default=DEFAULT_ITER_COUNT,
        help="Number of reps for each benchmark run.",
    )

    parser.addoption(
        "--query", action="store_true", default=False, help="Enable query mode"
    )

    parser.addoption(
        "--metrics",
        action="append",
        default=None,
        required=False,
        choices=ALL_AVAILABLE_METRICS,
        help=(
            "Specify the metrics we want to benchmark. "
            "If not specified, the metric items will "
            "vary according to the specified "
            "operation's category and name."
        ),
    )

    parser.addoption(
        "--dtypes",
        action="append",
        default=None,
        required=False,
        choices=[
            str(ele).split(".")[-1]
            for ele in FLOAT_DTYPES + INT_DTYPES + BOOL_DTYPES + [torch.cfloat]
        ],
        help=(
            "Specify the data types for benchmarks. "
            "If not specified, the dtype items will "
            "vary according to the specified "
            "operation's category and name."
        ),
    )

    parser.addoption(
        "--shape_file",
        action="store",
        default=os.path.join(os.path.dirname(__file__), "core_shapes.yaml"),
        required=False,
        help=(
            "Specify the shape file name for "
            "benchmarks. If not specified, a "
            "default shape list will be used."
        ),
    )

    try:
        parser.addoption(
            "--record",
            action="store",
            default="none",
            required=False,
            choices=["none", "log", "json"],
            help="Benchmark info recorded in log/json files or not",
        )
        parser.addoption(
            "--output",
            default=REPORT_FILE,
            help="Path to benchmark JSON report.",
        )
    except ValueError:
        # Mixed test+benchmark pytest runs may already register --record in
        # tests/conftest.py. Reuse the existing option in that case.
        pass

    try:
        parser.addoption(
            "--collect-marks",
            action="store_true",
            help=(
                "Collect benchmark marker information without "
                "executing tests."
            ),
        )
    except ValueError:
        pass


def pytest_configure(config):
    global Config  # noqa: F824
    global REGISTERED_MARKS
    global REPORT_FILE

    REGISTERED_MARKS = {
        marker.split(":")[0].strip() for marker in config.getini("markers")
    }

    mode_value = config.getoption(
        "--mode" if vendor_name != "kunlunxin" else "--fg_mode"
    )
    Config.mode = BenchMode(mode_value)

    Config.query = config.getoption("--query")

    level_value = config.getoption("--level")
    Config.bench_level = BenchLevel(level_value)

    warmup_value = config.getoption("--warmup")
    Config.warm_up = int(warmup_value)

    iter_value = config.getoption("--iter")
    Config.repetition = int(iter_value)

    types_str = config.getoption("--dtypes")
    dtypes = (
        [getattr(torch, dtype) for dtype in types_str]
        if types_str
        else types_str
    )
    Config.user_desired_dtypes = dtypes

    metrics = config.getoption("--metrics")
    Config.user_desired_metrics = metrics

    shape_file_str = config.getoption("--shape_file")
    Config.shape_file = shape_file_str

    Config.record_log = config.getoption("--record") == "log"
    Config.record_json = config.getoption("--record") == "json"
    if Config.record_json:
        Config.output = config.getoption("--output")
        REPORT_FILE = Config.output

    if Config.record_log:
        cmd_args = [
            arg.replace(".py", "").replace("=", "_").replace("/", "_")
            for arg in config.invocation_params.args
        ]

        log_file = "result_{}.log".format("_".join(cmd_args)).replace(
            "_-", "-"
        )

        for h in list(recordLogger.handlers):
            recordLogger.removeHandler(h)
            try:
                h.close()
            except Exception as e:
                import warnings

                warnings.warn(f"Failed to close handler: {e}")

        handler = logging.FileHandler(
            log_file, mode="w", encoding="utf-8", delay=False
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        recordLogger.addHandler(handler)
        recordLogger.setLevel(logging.INFO)
        emit_record_logger("Benchmark record logger enabled")


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
}


@pytest.fixture(scope="session", autouse=True)
def setup_once(request):
    if request.config.getoption("--query"):
        print("\nThis is query mode; all benchmark functions will be skipped.")
    # else:
    #     note_info = (
    #         "\n\nNote: The 'size' field below is for "
    #         "backward compatibility with previous "
    #         "versions of the benchmark. "
    #         "\nThis field will be removed in a future release."
    #     )
    #     print(note_info)


@pytest.fixture(scope="function", autouse=True)
def clear_function_cache():
    yield
    torch_device_fn.empty_cache()


@pytest.fixture(scope="module", autouse=True)
def clear_module_cache():
    yield
    torch_device_fn.empty_cache()


@pytest.fixture()
def extract_and_log_op_attributes(request):
    print("")
    op_attributes = []

    # Extract the 'recommended_shapes' attribute
    # from the pytest marker decoration.
    for mark in request.node.iter_markers():
        if mark.name in BUILTIN_MARKS:
            continue
        op_specified_shapes = mark.kwargs.get("recommended_shapes")
        shape_desc = mark.kwargs.get("shape_desc", "M, N")
        rec_core_shapes = get_recommended_shapes(
            mark.name, op_specified_shapes
        )

        if rec_core_shapes:
            attri = OperationAttribute(
                op_name=mark.name,
                recommended_core_shapes=rec_core_shapes,
                shape_desc=shape_desc,
            )
            print(attri)
            op_attributes.append(attri.to_dict())

    if request.config.getoption("--query"):
        # Skip the real benchmark functions
        pytest.skip("Skipping benchmark due to the query parameter.")

    yield
    if Config.record_log and op_attributes:
        emit_record_logger(json.dumps(op_attributes, indent=2))


def get_reason(report):
    if hasattr(report.longrepr, "reprcrash"):
        return report.longrepr.reprcrash.message
    if isinstance(report.longrepr, tuple):
        return report.longrepr[2]
    return str(report.longrepr)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    out = yield
    report = out.get_result()
    all_marks = [mark.name for mark in item.iter_markers()]
    marks = [mark for mark in all_marks if mark not in BUILTIN_MARKS]
    report.opid = marks[0] if marks else item.nodeid


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report):
    if not Config.record_json:
        return

    op = getattr(report, "opid", report.nodeid)
    TEST_RESULTS.setdefault(op, {})

    if report.when == "setup":
        if report.outcome == "skipped":
            TEST_RESULTS[op]["result"] = "skipped"
            TEST_RESULTS[op]["reason"] = get_reason(report)
            TEST_RESULTS[op]["test_case"] = report.nodeid
    elif report.when == "call":
        TEST_RESULTS[op]["result"] = report.outcome
        TEST_RESULTS[op]["test_case"] = report.nodeid
        if report.outcome in ["skipped", "failed"]:
            TEST_RESULTS[op]["reason"] = get_reason(report)
        else:
            TEST_RESULTS[op]["reason"] = None


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not Config.record_json:
        return

    data = TEST_RESULTS
    if os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, "r") as file:
            existing_data = json.load(file)
        existing_data.update(TEST_RESULTS)
        data = existing_data

    with open(REPORT_FILE, "w") as file:
        json.dump(data, file, indent=2, default=str)


def pytest_collection_modifyitems(session, config, items):
    if not config.getoption("--collect-marks"):
        return

    report = []
    for item in items:
        data = {
            "test_case": item.name,
            "file": item.location[0],
        }
        if item.cls:
            data["class"] = item.cls.__name__
        if item.originalname:
            data["function"] = item.originalname

        all_marks = list(item.iter_markers())
        op_marks = [
            mark.name
            for mark in all_marks
            if mark.name not in BUILTIN_MARKS
            and mark.name not in REGISTERED_MARKS
        ]
        data["marks"] = op_marks
        report.append(data)

    print(json.dumps(report, indent=2))
    pytest.exit("Collected benchmark marks.", returncode=0)
