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

import ctypes
import inspect
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict

import pytest
import torch

from devtools.dnn_reference.providers.ascend import provider as ascend


CPP_ROOT = (
    Path(__file__).parents[1]
    / "devtools"
    / "dnn_reference"
    / "providers"
    / "ascend"
    / "csrc"
)
COMMON_HEADER = CPP_ROOT / "common" / "oracle_common.h"
COMMON_SOURCE = CPP_ROOT / "common" / "oracle_common.cpp"
ADD_SOURCE = CPP_ROOT / "ops" / "add.cpp"
ABS_SOURCE = CPP_ROOT / "ops" / "abs.cpp"
NATIVE_SOURCES = (COMMON_SOURCE, ADD_SOURCE, ABS_SOURCE)
FAKE_CANN_ROOT = (
    Path(__file__).parents[1]
    / "tests"
    / "oracles"
    / "csrc"
    / "ascend"
    / "fake_cann"
)
_CANN_DEPENDENCY = re.compile(
    r"(?:/(?:ascend|cann)(?:/|$)|lib(?:ascend|acl|opapi|hccl|ge|runtime|"
    r"nnop|profapi|graph|metadef|register|opp|lowering|platform))",
    re.IGNORECASE,
)


class FakeStatus:
    SET_REPEATABLE = 1
    MALLOC = 2
    ADD = 3
    SYNCHRONIZE = 4
    DESTROY_EXECUTOR = 5
    DESTROY_X = 6
    DESTROY_Y = 7
    DESTROY_OUTPUT = 8
    DESTROY_SCALAR = 9
    FREE = 10
    CREATE_X = 11
    CREATE_Y = 12
    CREATE_OUTPUT = 13
    CREATE_SCALAR = 14
    GET_WORKSPACE = 15
    ABS_GET_WORKSPACE = 16
    ABS = 17


class _AbsCallOptions(TypedDict, total=False):
    input_data: int | None
    output_data: int | None
    input_shape: bool
    input_strides: bool
    input_rank: int
    output_shape: bool
    output_strides: bool
    output_rank: int
    dtype_code: int
    stream: int | None


def test_native_sources_are_split_by_responsibility():
    assert COMMON_HEADER.is_file()
    assert COMMON_SOURCE.is_file()
    assert ADD_SOURCE.is_file()
    assert ABS_SOURCE.is_file()
    assert not (CPP_ROOT / "aclnn_oracle.cpp").exists()

    common = COMMON_SOURCE.read_text(encoding="utf-8")
    add = ADD_SOURCE.read_text(encoding="utf-8")
    abs_source = ABS_SOURCE.read_text(encoding="utf-8")
    assert "aclnnAddGetWorkspaceSize" not in common
    assert "aclnnAdd(" not in common
    assert "aclnnAbsGetWorkspaceSize" not in common
    assert "aclnnAbs(" not in common
    assert 'extern "C" int flagdnn_test_aclnn_add' in add
    assert 'extern "C" int flagdnn_test_aclnn_abs' in abs_source


def _required_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        pytest.fail(f"{name} is required for fake-CANN isolation checks")
    return path


def _run_isolation_check(command: list[str], description: str) -> str:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        pytest.fail(
            f"{description} failed (exit {completed.returncode}):\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.stdout


def _assert_fake_library_is_isolated(library_path: Path) -> None:
    nm = _required_tool("nm")
    undefined_symbols = _run_isolation_check(
        [nm, "-D", "--undefined-only", str(library_path)],
        "nm undefined-symbol inspection",
    )
    unresolved_cann = [
        line
        for line in undefined_symbols.splitlines()
        if line.split() and line.split()[-1].split("@", 1)[0].startswith("acl")
    ]
    assert (
        not unresolved_cann
    ), "fake ACLNN library has unresolved ACL/ACLNN symbols:\n" + "\n".join(
        unresolved_cann
    )

    readelf = _required_tool("readelf")
    dynamic_section = _run_isolation_check(
        [readelf, "-d", str(library_path)],
        "readelf dynamic-section inspection",
    )
    needed_libraries = re.findall(
        r"Shared library: \[([^]]+)\]", dynamic_section, flags=re.IGNORECASE
    )
    cann_needed = [
        library
        for library in needed_libraries
        if _CANN_DEPENDENCY.search(library)
    ]
    assert (
        not cann_needed
    ), "fake ACLNN library has CANN DT_NEEDED dependencies: " + ", ".join(
        cann_needed
    )

    ldd = shutil.which("ldd")
    if ldd is not None:
        resolved_dependencies = _run_isolation_check(
            [ldd, str(library_path)], "ldd dependency inspection"
        )
        cann_paths = [
            line
            for line in resolved_dependencies.splitlines()
            if _CANN_DEPENDENCY.search(line)
        ]
        assert (
            not cann_paths
        ), "fake ACLNN library resolves CANN dependencies:\n" + "\n".join(
            cann_paths
        )


@pytest.fixture(scope="session")
def fake_aclnn_library(tmp_path_factory: pytest.TempPathFactory):
    """Build the production wrapper with the test-only fake CANN ABI."""
    compiler = _required_tool("c++")
    assert FAKE_CANN_ROOT.is_dir(), "fake CANN fixture is missing"

    build_dir = tmp_path_factory.mktemp("fake-cann-aclnn")
    library_path = build_dir / "libflagdnn_fake_aclnn.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-Wl,-Bsymbolic",
            "-Wl,-z,defs",
            "-Wall",
            "-Wextra",
            "-I",
            str(FAKE_CANN_ROOT / "include"),
            "-I",
            str(CPP_ROOT),
            str(COMMON_SOURCE),
            str(ADD_SOURCE),
            str(ABS_SOURCE),
            str(FAKE_CANN_ROOT / "fake_cann.cpp"),
            "-o",
            str(library_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    _assert_fake_library_is_isolated(library_path)
    library = ctypes.CDLL(str(library_path))
    library.flagdnn_test_aclnn_add.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    library.flagdnn_test_aclnn_add.restype = ctypes.c_int
    library.flagdnn_aclnn_add_create.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    library.flagdnn_aclnn_add_create.restype = ctypes.c_int
    for name in ("flagdnn_aclnn_add_run", "flagdnn_aclnn_add_destroy"):
        function = getattr(library, name)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        function.restype = ctypes.c_int
    library.flagdnn_test_aclnn_abs.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    library.flagdnn_test_aclnn_abs.restype = ctypes.c_int
    library.fake_cann_reset.argtypes = []
    library.fake_cann_set_operation.argtypes = [ctypes.c_int]
    library.fake_cann_set_status.argtypes = [ctypes.c_int, ctypes.c_int]
    library.fake_cann_events.restype = ctypes.c_char_p
    library.fake_cann_count.argtypes = [ctypes.c_char_p]
    library.fake_cann_count.restype = ctypes.c_int
    library.fake_cann_last_free.restype = ctypes.c_void_p
    library.fake_cann_workspace.restype = ctypes.c_void_p
    library.fake_cann_executor_destroy_count.restype = ctypes.c_int
    yield library
    library.fake_cann_reset()


def _call_fake_wrapper(library: ctypes.CDLL) -> tuple[int, str]:
    shape = (ctypes.c_int64 * 1)(1)
    strides = (ctypes.c_int64 * 1)(1)
    error = ctypes.create_string_buffer(1024)
    status = library.flagdnn_test_aclnn_add(
        ctypes.c_void_p(0x1010),
        shape,
        strides,
        1,
        ctypes.c_void_p(0x2020),
        shape,
        strides,
        1,
        ctypes.c_void_p(0x3030),
        shape,
        strides,
        1,
        2,
        1.0,
        ctypes.c_void_p(0x4040),
        error,
        len(error),
    )
    return status, error.value.decode()


def _create_fake_prepared_add(
    library: ctypes.CDLL,
) -> tuple[int, str, ctypes.c_void_p]:
    shape = (ctypes.c_int64 * 1)(1)
    strides = (ctypes.c_int64 * 1)(1)
    error = ctypes.create_string_buffer(1024)
    handle = ctypes.c_void_p()
    status = library.flagdnn_aclnn_add_create(
        ctypes.c_void_p(0x1010),
        shape,
        strides,
        1,
        ctypes.c_void_p(0x2020),
        shape,
        strides,
        1,
        ctypes.c_void_p(0x3030),
        shape,
        strides,
        1,
        2,
        1.0,
        ctypes.c_void_p(0x4040),
        ctypes.byref(handle),
        error,
        len(error),
    )
    return status, error.value.decode(), handle


def test_fake_cann_prepared_add_separates_setup_run_and_cleanup(
    fake_aclnn_library,
):
    library = fake_aclnn_library
    library.fake_cann_reset()

    status, detail, handle = _create_fake_prepared_add(library)

    assert status == 0, detail
    assert handle.value is not None
    assert library.fake_cann_count(b"get_workspace") == 1
    assert library.fake_cann_count(b"set_repeatable") == 1
    assert library.fake_cann_count(b"malloc") == 1
    assert library.fake_cann_count(b"add") == 0
    assert library.fake_cann_count(b"synchronize") == 0

    for _ in range(2):
        error = ctypes.create_string_buffer(1024)
        status = library.flagdnn_aclnn_add_run(handle, error, len(error))
        assert status == 0, error.value.decode()

    assert library.fake_cann_count(b"add") == 2
    assert library.fake_cann_count(b"synchronize") == 0

    error = ctypes.create_string_buffer(1024)
    status = library.flagdnn_aclnn_add_destroy(handle, error, len(error))
    assert status == 0, error.value.decode()
    assert library.fake_cann_count(b"synchronize") == 1
    assert library.fake_cann_executor_destroy_count() == 1
    assert library.fake_cann_count(b"free") == 1


def _call_fake_abs_wrapper(
    library: ctypes.CDLL,
    *,
    input_data: int | None = 0x1010,
    output_data: int | None = 0x3030,
    input_shape: bool = True,
    input_strides: bool = True,
    input_rank: int = 1,
    output_shape: bool = True,
    output_strides: bool = True,
    output_rank: int = 1,
    dtype_code: int = 2,
    stream: int | None = 0x4040,
) -> tuple[int, str]:
    shape = (ctypes.c_int64 * 1)(1)
    strides = (ctypes.c_int64 * 1)(1)
    error = ctypes.create_string_buffer(1024)
    library.fake_cann_set_operation(1)
    status = library.flagdnn_test_aclnn_abs(
        ctypes.c_void_p(input_data),
        shape if input_shape else None,
        strides if input_strides else None,
        input_rank,
        ctypes.c_void_p(output_data),
        shape if output_shape else None,
        strides if output_strides else None,
        output_rank,
        dtype_code,
        ctypes.c_void_p(stream),
        error,
        len(error),
    )
    return status, error.value.decode()


def _events(library: ctypes.CDLL) -> list[str]:
    encoded = library.fake_cann_events()
    return encoded.decode().split(",") if encoded else []


def _configure_failure(
    library: ctypes.CDLL, failure: int, status: int
) -> None:
    library.fake_cann_set_status(failure, status)


def _configure_workspace_size(library: ctypes.CDLL, size: int) -> None:
    library.fake_cann_set_workspace_size.argtypes = [ctypes.c_uint64]
    library.fake_cann_set_workspace_size(size)


def _assert_fake_wiring_is_valid(
    library: ctypes.CDLL, workspace_size: int = 64
) -> None:
    library.fake_cann_wiring_is_valid.restype = ctypes.c_int
    library.fake_cann_wiring_error.restype = ctypes.c_char_p
    library.fake_cann_last_add_workspace.restype = ctypes.c_void_p
    library.fake_cann_last_add_workspace_size.restype = ctypes.c_uint64
    library.fake_cann_last_add_executor.restype = ctypes.c_void_p
    library.fake_cann_last_add_stream.restype = ctypes.c_void_p
    library.fake_cann_last_synchronize_stream.restype = ctypes.c_void_p
    library.fake_cann_executor.restype = ctypes.c_void_p

    assert (
        library.fake_cann_wiring_is_valid() == 1
    ), library.fake_cann_wiring_error().decode()
    assert (
        library.fake_cann_last_add_workspace() == library.fake_cann_workspace()
    )
    assert library.fake_cann_last_add_workspace_size() == workspace_size
    assert (
        library.fake_cann_last_add_executor() == library.fake_cann_executor()
    )
    assert library.fake_cann_last_add_stream() == 0x4040
    assert library.fake_cann_last_synchronize_stream() == 0x4040


def _assert_fake_abs_wiring_is_valid(
    library: ctypes.CDLL, workspace_size: int = 64
) -> None:
    library.fake_cann_wiring_is_valid.restype = ctypes.c_int
    library.fake_cann_wiring_error.restype = ctypes.c_char_p
    library.fake_cann_last_abs_workspace.restype = ctypes.c_void_p
    library.fake_cann_last_abs_workspace_size.restype = ctypes.c_uint64
    library.fake_cann_last_abs_executor.restype = ctypes.c_void_p
    library.fake_cann_last_abs_stream.restype = ctypes.c_void_p
    library.fake_cann_last_synchronize_stream.restype = ctypes.c_void_p
    library.fake_cann_executor.restype = ctypes.c_void_p

    assert (
        library.fake_cann_wiring_is_valid() == 1
    ), library.fake_cann_wiring_error().decode()
    assert (
        library.fake_cann_last_abs_workspace() == library.fake_cann_workspace()
    )
    assert library.fake_cann_last_abs_workspace_size() == workspace_size
    assert (
        library.fake_cann_last_abs_executor() == library.fake_cann_executor()
    )
    assert library.fake_cann_last_abs_stream() == 0x4040
    assert library.fake_cann_last_synchronize_stream() == 0x4040


def _assert_complete_cleanup(events: list[str]) -> None:
    assert events[-6:] == [
        "destroy_executor",
        "destroy_x",
        "destroy_y",
        "destroy_output",
        "destroy_scalar",
        "free",
    ]


def test_fake_cann_success_uses_expected_execution_and_cleanup_order(
    fake_aclnn_library: ctypes.CDLL,
):
    fake_aclnn_library.fake_cann_reset()

    status, error = _call_fake_wrapper(fake_aclnn_library)
    events = _events(fake_aclnn_library)

    assert status == 0
    assert error == ""
    assert events == [
        "create_x",
        "create_y",
        "create_output",
        "create_scalar",
        "get_workspace",
        "set_repeatable",
        "malloc",
        "add",
        "synchronize",
        "destroy_executor",
        "destroy_x",
        "destroy_y",
        "destroy_output",
        "destroy_scalar",
        "free",
    ]
    assert fake_aclnn_library.fake_cann_executor_destroy_count() == 1
    assert fake_aclnn_library.fake_cann_last_free() == (
        fake_aclnn_library.fake_cann_workspace()
    )
    assert fake_aclnn_library.fake_cann_last_free() not in {
        0x1010,
        0x2020,
        0x3030,
    }
    _assert_fake_wiring_is_valid(fake_aclnn_library)


def test_fake_cann_abs_success_uses_expected_execution_and_cleanup_order(
    fake_aclnn_library: ctypes.CDLL,
):
    fake_aclnn_library.fake_cann_reset()

    status, error = _call_fake_abs_wrapper(fake_aclnn_library)
    events = _events(fake_aclnn_library)

    assert status == 0
    assert error == ""
    assert events == [
        "create_x",
        "create_output",
        "get_abs_workspace",
        "set_repeatable",
        "malloc",
        "abs",
        "synchronize",
        "destroy_executor",
        "destroy_x",
        "destroy_output",
        "free",
    ]
    assert fake_aclnn_library.fake_cann_executor_destroy_count() == 1
    assert fake_aclnn_library.fake_cann_last_free() == (
        fake_aclnn_library.fake_cann_workspace()
    )
    assert fake_aclnn_library.fake_cann_last_free() not in {0x1010, 0x3030}
    _assert_fake_abs_wiring_is_valid(fake_aclnn_library)


@pytest.mark.parametrize(
    ("call_options", "expected_status", "stage"),
    [
        ({"input_data": None}, -1, "input/output device pointer is null"),
        ({"output_data": None}, -1, "input/output device pointer is null"),
        ({"input_shape": False}, -2, "shape/stride metadata"),
        ({"input_strides": False}, -2, "shape/stride metadata"),
        ({"input_rank": 0}, -2, "shape/stride metadata"),
        ({"output_shape": False}, -2, "shape/stride metadata"),
        ({"output_strides": False}, -2, "shape/stride metadata"),
        ({"output_rank": 0}, -2, "shape/stride metadata"),
        ({"stream": None}, -3, "stream pointer is null"),
        ({"dtype_code": 99}, -4, "unsupported dtype code"),
    ],
)
def test_fake_cann_abs_rejects_invalid_wrapper_arguments_before_acquisition(
    fake_aclnn_library: ctypes.CDLL,
    call_options: _AbsCallOptions,
    expected_status: int,
    stage: str,
):
    fake_aclnn_library.fake_cann_reset()

    status, error = _call_fake_abs_wrapper(fake_aclnn_library, **call_options)

    assert status == expected_status
    assert stage in error
    assert f"status={expected_status}" in error
    assert _events(fake_aclnn_library) == []


@pytest.mark.parametrize(
    (
        "failure",
        "injected_status",
        "expected_status",
        "stage",
        "expected_events",
        "executor_destroy_count",
    ),
    [
        (
            FakeStatus.CREATE_X,
            61,
            -5,
            "aclCreateTensor(x)",
            ["create_x"],
            0,
        ),
        (
            FakeStatus.CREATE_OUTPUT,
            63,
            -6,
            "aclCreateTensor(output)",
            ["create_x", "create_output", "destroy_x"],
            0,
        ),
        (
            FakeStatus.ABS_GET_WORKSPACE,
            65,
            65,
            "aclnnAbsGetWorkspaceSize",
            [
                "create_x",
                "create_output",
                "get_abs_workspace",
                "destroy_x",
                "destroy_output",
            ],
            0,
        ),
        (
            FakeStatus.SET_REPEATABLE,
            31,
            31,
            "aclSetAclOpExecutorRepeatable",
            [
                "create_x",
                "create_output",
                "get_abs_workspace",
                "set_repeatable",
                "destroy_x",
                "destroy_output",
            ],
            0,
        ),
        (
            FakeStatus.MALLOC,
            32,
            32,
            "aclrtMalloc(workspace)",
            [
                "create_x",
                "create_output",
                "get_abs_workspace",
                "set_repeatable",
                "malloc",
                "destroy_executor",
                "destroy_x",
                "destroy_output",
            ],
            1,
        ),
    ],
)
def test_fake_cann_abs_acquisition_failures_stop_and_cleanup_owned_resources(
    fake_aclnn_library: ctypes.CDLL,
    failure: int,
    injected_status: int,
    expected_status: int,
    stage: str,
    expected_events: list[str],
    executor_destroy_count: int,
):
    fake_aclnn_library.fake_cann_reset()
    _configure_failure(fake_aclnn_library, failure, injected_status)

    status, error = _call_fake_abs_wrapper(fake_aclnn_library)
    events = _events(fake_aclnn_library)

    assert status == expected_status
    assert f"{stage} failed, status={expected_status}" in error
    assert events == expected_events
    assert fake_aclnn_library.fake_cann_count(b"abs") == 0
    assert fake_aclnn_library.fake_cann_count(b"synchronize") == 0
    assert fake_aclnn_library.fake_cann_count(b"free") == 0
    assert (
        fake_aclnn_library.fake_cann_executor_destroy_count()
        == executor_destroy_count
    )


def test_fake_cann_abs_zero_workspace_executes_without_allocation_or_free(
    fake_aclnn_library: ctypes.CDLL,
):
    fake_aclnn_library.fake_cann_reset()
    _configure_workspace_size(fake_aclnn_library, 0)

    status, error = _call_fake_abs_wrapper(fake_aclnn_library)
    events = _events(fake_aclnn_library)

    assert status == 0
    assert error == ""
    assert events == [
        "create_x",
        "create_output",
        "get_abs_workspace",
        "set_repeatable",
        "abs",
        "synchronize",
        "destroy_executor",
        "destroy_x",
        "destroy_output",
    ]
    assert fake_aclnn_library.fake_cann_count(b"malloc") == 0
    assert fake_aclnn_library.fake_cann_count(b"free") == 0
    assert fake_aclnn_library.fake_cann_executor_destroy_count() == 1
    _assert_fake_abs_wiring_is_valid(fake_aclnn_library, workspace_size=0)


def test_fake_cann_abs_failure_still_synchronizes_and_cleans_up(
    fake_aclnn_library: ctypes.CDLL,
):
    fake_aclnn_library.fake_cann_reset()
    _configure_failure(fake_aclnn_library, FakeStatus.ABS, 41)

    status, error = _call_fake_abs_wrapper(fake_aclnn_library)
    events = _events(fake_aclnn_library)

    assert status == 41
    assert "aclnnAbs failed, status=41" in error
    assert events[5:7] == ["abs", "synchronize"]
    assert events[-4:] == [
        "destroy_executor",
        "destroy_x",
        "destroy_output",
        "free",
    ]
    assert fake_aclnn_library.fake_cann_executor_destroy_count() == 1


def test_fake_cann_abs_failure_remains_first_after_synchronize_failure(
    fake_aclnn_library: ctypes.CDLL,
):
    fake_aclnn_library.fake_cann_reset()
    _configure_failure(fake_aclnn_library, FakeStatus.ABS, 41)
    _configure_failure(fake_aclnn_library, FakeStatus.SYNCHRONIZE, 42)

    status, error = _call_fake_abs_wrapper(fake_aclnn_library)
    events = _events(fake_aclnn_library)

    assert status == 41
    assert events[5:7] == ["abs", "synchronize"]
    assert "aclnnAbs failed, status=41" in error
    assert "aclrtSynchronizeStream failed, status=42" in error
    assert error.index("status=41") < error.index("status=42")
    assert events[-4:] == [
        "destroy_executor",
        "destroy_x",
        "destroy_output",
        "free",
    ]


def test_fake_cann_abs_cleanup_failures_are_appended_and_attempted_once(
    fake_aclnn_library: ctypes.CDLL,
):
    fake_aclnn_library.fake_cann_reset()
    failures = [
        (FakeStatus.DESTROY_EXECUTOR, 51, "destroy_executor"),
        (FakeStatus.DESTROY_X, 52, "destroy_x"),
        (FakeStatus.DESTROY_OUTPUT, 54, "destroy_output"),
        (FakeStatus.FREE, 56, "free"),
    ]
    for failure, status, _ in failures:
        _configure_failure(fake_aclnn_library, failure, status)

    status, error = _call_fake_abs_wrapper(fake_aclnn_library)
    events = _events(fake_aclnn_library)

    assert status == 51
    assert events[-4:] == [
        "destroy_executor",
        "destroy_x",
        "destroy_output",
        "free",
    ]
    for _, injected_status, event in failures:
        assert f"status={injected_status}" in error
        assert fake_aclnn_library.fake_cann_count(event.encode()) == 1
    assert fake_aclnn_library.fake_cann_executor_destroy_count() == 1


@pytest.mark.parametrize(
    (
        "failure",
        "injected_status",
        "expected_status",
        "stage",
        "expected_events",
    ),
    [
        (FakeStatus.CREATE_X, 61, -5, "aclCreateTensor(x)", ["create_x"]),
        (
            FakeStatus.CREATE_Y,
            62,
            -6,
            "aclCreateTensor(y)",
            ["create_x", "create_y", "destroy_x"],
        ),
        (
            FakeStatus.CREATE_OUTPUT,
            63,
            -7,
            "aclCreateTensor(output)",
            [
                "create_x",
                "create_y",
                "create_output",
                "destroy_x",
                "destroy_y",
            ],
        ),
        (
            FakeStatus.CREATE_SCALAR,
            64,
            -8,
            "aclCreateScalar(alpha)",
            [
                "create_x",
                "create_y",
                "create_output",
                "create_scalar",
                "destroy_x",
                "destroy_y",
                "destroy_output",
            ],
        ),
        (
            FakeStatus.GET_WORKSPACE,
            65,
            65,
            "aclnnAddGetWorkspaceSize",
            [
                "create_x",
                "create_y",
                "create_output",
                "create_scalar",
                "get_workspace",
                "destroy_x",
                "destroy_y",
                "destroy_output",
                "destroy_scalar",
            ],
        ),
    ],
)
def test_fake_cann_acquisition_failure_returns_wrapper_status_and_stops(
    fake_aclnn_library: ctypes.CDLL,
    failure: int,
    injected_status: int,
    expected_status: int,
    stage: str,
    expected_events: list[str],
):
    fake_aclnn_library.fake_cann_reset()
    _configure_failure(fake_aclnn_library, failure, injected_status)

    actual_status, error = _call_fake_wrapper(fake_aclnn_library)
    events = _events(fake_aclnn_library)

    assert actual_status == expected_status
    assert f"{stage} failed, status={expected_status}" in error
    assert events == expected_events
    assert fake_aclnn_library.fake_cann_count(b"set_repeatable") == 0
    assert fake_aclnn_library.fake_cann_count(b"malloc") == 0
    assert fake_aclnn_library.fake_cann_count(b"add") == 0
    assert fake_aclnn_library.fake_cann_count(b"synchronize") == 0
    assert fake_aclnn_library.fake_cann_executor_destroy_count() == 0


def test_fake_cann_zero_workspace_executes_without_workspace_allocation(
    fake_aclnn_library: ctypes.CDLL,
):
    fake_aclnn_library.fake_cann_reset()
    _configure_workspace_size(fake_aclnn_library, 0)

    status, error = _call_fake_wrapper(fake_aclnn_library)
    events = _events(fake_aclnn_library)

    assert status == 0
    assert error == ""
    assert events == [
        "create_x",
        "create_y",
        "create_output",
        "create_scalar",
        "get_workspace",
        "set_repeatable",
        "add",
        "synchronize",
        "destroy_executor",
        "destroy_x",
        "destroy_y",
        "destroy_output",
        "destroy_scalar",
    ]
    assert fake_aclnn_library.fake_cann_count(b"malloc") == 0
    assert fake_aclnn_library.fake_cann_count(b"free") == 0
    assert fake_aclnn_library.fake_cann_executor_destroy_count() == 1

    library = fake_aclnn_library
    library.fake_cann_last_add_workspace.restype = ctypes.c_void_p
    library.fake_cann_last_add_workspace_size.restype = ctypes.c_uint64
    assert library.fake_cann_last_add_workspace() is None
    assert library.fake_cann_last_add_workspace_size() == 0
    _assert_fake_wiring_is_valid(library, workspace_size=0)


def test_fake_cann_set_repeatable_failure_keeps_executor_borrowed(
    fake_aclnn_library: ctypes.CDLL,
):
    fake_aclnn_library.fake_cann_reset()
    _configure_failure(fake_aclnn_library, FakeStatus.SET_REPEATABLE, 31)

    status, error = _call_fake_wrapper(fake_aclnn_library)
    events = _events(fake_aclnn_library)

    assert status == 31
    assert "aclSetAclOpExecutorRepeatable failed, status=31" in error
    assert events == [
        "create_x",
        "create_y",
        "create_output",
        "create_scalar",
        "get_workspace",
        "set_repeatable",
        "destroy_x",
        "destroy_y",
        "destroy_output",
        "destroy_scalar",
    ]
    assert fake_aclnn_library.fake_cann_executor_destroy_count() == 0
    assert fake_aclnn_library.fake_cann_count(b"malloc") == 0
    assert fake_aclnn_library.fake_cann_count(b"add") == 0
    assert fake_aclnn_library.fake_cann_count(b"synchronize") == 0


def test_fake_cann_malloc_failure_cleans_owned_executor_without_free(
    fake_aclnn_library: ctypes.CDLL,
):
    fake_aclnn_library.fake_cann_reset()
    _configure_failure(fake_aclnn_library, FakeStatus.MALLOC, 32)

    status, error = _call_fake_wrapper(fake_aclnn_library)
    events = _events(fake_aclnn_library)

    assert status == 32
    assert "aclrtMalloc(workspace) failed, status=32" in error
    assert events[:7] == [
        "create_x",
        "create_y",
        "create_output",
        "create_scalar",
        "get_workspace",
        "set_repeatable",
        "malloc",
    ]
    assert events[-5:] == [
        "destroy_executor",
        "destroy_x",
        "destroy_y",
        "destroy_output",
        "destroy_scalar",
    ]
    assert fake_aclnn_library.fake_cann_count(b"add") == 0
    assert fake_aclnn_library.fake_cann_count(b"synchronize") == 0
    assert fake_aclnn_library.fake_cann_count(b"free") == 0
    assert fake_aclnn_library.fake_cann_executor_destroy_count() == 1


def test_fake_cann_add_failure_preserves_add_status_and_records_later_failures(
    fake_aclnn_library: ctypes.CDLL,
):
    fake_aclnn_library.fake_cann_reset()
    _configure_failure(fake_aclnn_library, FakeStatus.ADD, 41)
    _configure_failure(fake_aclnn_library, FakeStatus.SYNCHRONIZE, 42)
    _configure_failure(fake_aclnn_library, FakeStatus.DESTROY_EXECUTOR, 51)

    status, error = _call_fake_wrapper(fake_aclnn_library)
    events = _events(fake_aclnn_library)

    assert status == 41
    assert events[7:9] == ["add", "synchronize"]
    _assert_complete_cleanup(events)
    for expected in ("status=41", "status=42", "status=51"):
        assert expected in error


def test_fake_cann_synchronize_failure_is_returned_after_successful_add(
    fake_aclnn_library: ctypes.CDLL,
):
    fake_aclnn_library.fake_cann_reset()
    _configure_failure(fake_aclnn_library, FakeStatus.SYNCHRONIZE, 42)

    status, error = _call_fake_wrapper(fake_aclnn_library)
    events = _events(fake_aclnn_library)

    assert status == 42
    assert "aclrtSynchronizeStream failed, status=42" in error
    assert events[7:9] == ["add", "synchronize"]
    _assert_complete_cleanup(events)
    assert fake_aclnn_library.fake_cann_count(b"add") == 1
    assert fake_aclnn_library.fake_cann_executor_destroy_count() == 1


@pytest.mark.parametrize(
    ("failure", "status", "event"),
    [
        (FakeStatus.DESTROY_EXECUTOR, 51, "destroy_executor"),
        (FakeStatus.DESTROY_X, 52, "destroy_x"),
        (FakeStatus.DESTROY_Y, 53, "destroy_y"),
        (FakeStatus.DESTROY_OUTPUT, 54, "destroy_output"),
        (FakeStatus.DESTROY_SCALAR, 55, "destroy_scalar"),
        (FakeStatus.FREE, 56, "free"),
    ],
)
def test_fake_cann_first_cleanup_failure_is_returned_and_later_cleanup_runs(
    fake_aclnn_library: ctypes.CDLL,
    failure: int,
    status: int,
    event: str,
):
    fake_aclnn_library.fake_cann_reset()
    _configure_failure(fake_aclnn_library, failure, status)

    actual_status, error = _call_fake_wrapper(fake_aclnn_library)
    events = _events(fake_aclnn_library)

    assert actual_status == status
    assert f"status={status}" in error
    _assert_complete_cleanup(events)
    assert fake_aclnn_library.fake_cann_count(event.encode()) == 1
    assert fake_aclnn_library.fake_cann_executor_destroy_count() == 1


def _code_without_comments_or_strings(source: str) -> str:
    lexical_tokens = re.compile(
        r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\])*\"|" r"'(?:\\.|[^'\\])*'",
        re.DOTALL,
    )

    def preserve_offsets(match: re.Match[str]) -> str:
        return re.sub(r"[^\n]", " ", match.group(0))

    return lexical_tokens.sub(preserve_offsets, source)


def _call_positions(source: str, identifier: str) -> list[int]:
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(identifier)}\s*\(")
    code = _code_without_comments_or_strings(source)
    return [match.start() for match in pattern.finditer(code)]


def _first_call_position(source: str, identifier: str) -> int:
    positions = _call_positions(source, identifier)
    assert positions, f"missing function call: {identifier}"
    return positions[0]


def _function_body(source: str, identifier: str) -> str:
    signature = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(identifier)}\s*"
        rf"\([^;{{}}]*\)\s*(?:noexcept\s*)?\{{"
    )
    match = signature.search(source)
    assert match is not None, f"missing function definition: {identifier}"

    depth = 1
    for index in range(match.end(), len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[match.end() : index]
    raise AssertionError(f"unterminated function definition: {identifier}")


def test_aclnn_wrapper_reuses_torch_npu_context_and_stream():
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in NATIVE_SOURCES
    )

    for required in (
        "aclCreateTensor",
        "aclCreateScalar",
        "aclnnAddGetWorkspaceSize",
        "aclSetAclOpExecutorRepeatable",
        "aclrtSynchronizeStream",
        "aclDestroyAclOpExecutor",
        "aclDestroyTensor",
        "aclDestroyScalar",
        "aclrtFree",
    ):
        assert _call_positions(source, required), required

    assert re.search(
        r"ExecuteAndSynchronize\s*\([^;]*\baclnnAdd\b", source, re.DOTALL
    )

    for forbidden in (
        "aclInit",
        "aclFinalize",
        "aclrtSetDevice",
        "aclrtResetDevice",
        "aclrtCreateContext",
        "aclrtDestroyContext",
        "aclrtCreateStream",
        "aclrtDestroyStream",
    ):
        assert not _call_positions(source, forbidden), forbidden


def test_aclnn_wrapper_makes_executor_repeatable_before_execution():
    common = COMMON_SOURCE.read_text(encoding="utf-8")
    add = ADD_SOURCE.read_text(encoding="utf-8")
    call_order = (
        "aclnnAddGetWorkspaceSize",
        "MakeExecutorRepeatable",
        "AllocateWorkspace",
        "ExecuteAndSynchronize",
    )
    positions = [_first_call_position(add, name) for name in call_order]

    assert positions == sorted(positions)
    repeatable = _function_body(common, "MakeExecutorRepeatable")
    set_repeatable = _first_call_position(
        repeatable, "aclSetAclOpExecutorRepeatable"
    )
    assert "executor_owned = true" in repeatable[set_repeatable:]

    execution = _function_body(common, "ExecuteAndSynchronize")
    execute = _first_call_position(execution, "execute")
    synchronize = _first_call_position(execution, "aclrtSynchronizeStream")
    assert execute < synchronize
    assert "return" not in execution[execute:synchronize]


def test_aclnn_wrapper_checks_each_created_resource_immediately():
    source = _function_body(
        ADD_SOURCE.read_text(encoding="utf-8"),
        "flagdnn_test_aclnn_add",
    )
    tensor_calls = _call_positions(source, "aclCreateTensor")
    scalar_call = _first_call_position(source, "aclCreateScalar")
    workspace_call = _first_call_position(source, "aclnnAddGetWorkspaceSize")
    assert len(tensor_calls) == 3

    starts = [*tensor_calls, scalar_call]
    ends = [tensor_calls[1], tensor_calls[2], scalar_call, workspace_call]
    stages = (
        '"aclCreateTensor(x)"',
        '"aclCreateTensor(y)"',
        '"aclCreateTensor(output)"',
        '"aclCreateScalar(alpha)"',
    )
    for start, end, stage in zip(starts, ends, stages):
        segment = source[start:end]
        assert "nullptr" in segment
        assert stage in segment


def test_aclnn_wrapper_uses_explicit_checked_cleanup_without_stl_errors():
    common = COMMON_SOURCE.read_text(encoding="utf-8")
    add = ADD_SOURCE.read_text(encoding="utf-8")
    wrapper = _function_body(add, "flagdnn_test_aclnn_add")

    assert len(_call_positions(wrapper, "DestroyExecutor")) == 1
    assert len(_call_positions(wrapper, "DestroyTensor")) == 3
    assert len(_call_positions(wrapper, "DestroyScalar")) == 1
    assert len(_call_positions(wrapper, "FreeWorkspace")) == 1

    expected_cleanup_helpers = {
        "DestroyExecutor": "aclDestroyAclOpExecutor",
        "DestroyTensor": "aclDestroyTensor",
        "DestroyScalar": "aclDestroyScalar",
        "FreeWorkspace": "aclrtFree",
    }
    for helper, function in expected_cleanup_helpers.items():
        cleanup = _function_body(common, helper)
        assert len(_call_positions(cleanup, function)) == 1
        checked_call = re.compile(
            rf"RecordAclFailure\s*\([^;]*?"
            rf"(?<![A-Za-z0-9_]){re.escape(function)}\s*\(",
            re.DOTALL,
        )
        assert len(checked_call.findall(cleanup)) == 1

    source = common + "\n" + add
    for forbidden in (
        "#include <string>",
        "#include <sstream>",
        "std::string",
        "std::ostringstream",
        "~Resources(",
    ):
        assert forbidden not in source


def test_aclnn_wrapper_retains_first_failure_and_contains_cpp_exceptions():
    common = COMMON_SOURCE.read_text(encoding="utf-8")
    add = ADD_SOURCE.read_text(encoding="utf-8")
    wrapper = _function_body(add, "flagdnn_test_aclnn_add")
    execution = _function_body(common, "ExecuteAndSynchronize")
    execute_position = _first_call_position(execution, "execute")
    synchronize_position = _first_call_position(
        execution, "aclrtSynchronizeStream"
    )
    execution_position = _first_call_position(wrapper, "ExecuteAndSynchronize")
    cleanup_position = _first_call_position(wrapper, "DestroyExecutor")

    assert execution_position < cleanup_position
    assert (
        "RecordAclFailure" in execution[execute_position:synchronize_position]
    )
    assert "RecordAclFailure" in execution[synchronize_position:]
    assert "return" not in _code_without_comments_or_strings(
        execution[execute_position:synchronize_position]
    )
    assert "if (state->first_status == ACL_SUCCESS)" in common
    assert re.search(
        r'extern\s+"C"\s+int\s+flagdnn_test_aclnn_add\s*'
        r"\([^{}]*\)\s+noexcept\s*\{",
        add,
        re.DOTALL,
    )
    assert "catch" not in _code_without_comments_or_strings(common + add)


class _FakeCFunction:
    def __init__(self):
        self.argtypes = None
        self.restype = None


def test_ascend_oracle_configure_library_uses_exact_c_abi_signatures():
    add_function = _FakeCFunction()
    create_function = _FakeCFunction()
    run_function = _FakeCFunction()
    destroy_function = _FakeCFunction()
    abs_function = _FakeCFunction()
    library = SimpleNamespace(
        flagdnn_test_aclnn_add=add_function,
        flagdnn_aclnn_add_create=create_function,
        flagdnn_aclnn_add_run=run_function,
        flagdnn_aclnn_add_destroy=destroy_function,
        flagdnn_test_aclnn_abs=abs_function,
    )

    assert ascend._configure_library(library) is library
    assert add_function.argtypes == [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    assert add_function.restype is ctypes.c_int
    assert len(add_function.argtypes) == 17
    assert create_function.argtypes == [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    assert create_function.restype is ctypes.c_int
    for function in (run_function, destroy_function):
        assert function.argtypes == [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        assert function.restype is ctypes.c_int
    assert abs_function.argtypes == [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    assert abs_function.restype is ctypes.c_int
    assert len(abs_function.argtypes) == 12


class _FakeDevice:
    def __init__(self, device_type="npu", index=2):
        self.type = device_type
        self.index = index

    def __str__(self):
        return f"{self.type}:{self.index}"


class _FakeTensor:
    def __init__(
        self,
        shape=(2, 3),
        stride=(5, 1),
        *,
        layout=torch.strided,
        device=None,
        dtype=torch.float32,
        storage_offset=0,
        data_pointer=0x1234,
    ):
        self.shape = tuple(shape)
        self._stride = tuple(stride)
        self.layout = layout
        self.device = device or _FakeDevice()
        self.dtype = dtype
        self._storage_offset = storage_offset
        self._data_pointer = data_pointer

    def stride(self):
        return self._stride

    def dim(self):
        return len(self.shape)

    def storage_offset(self):
        return self._storage_offset

    def data_ptr(self):
        return self._data_pointer

    def is_contiguous(self, memory_format=torch.contiguous_format):
        if memory_format == torch.channels_last:
            order = (1, 3, 2, 0) if self.dim() == 4 else ()
        else:
            order = tuple(reversed(range(self.dim())))
        if not order:
            return False

        expected_stride = 1
        for dimension in order:
            size = self.shape[dimension]
            if size != 1:
                if self._stride[dimension] != expected_stride:
                    return False
                expected_stride *= size
        return True


class _FakeDeviceContext:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        self.events.append("enter-device")

    def __exit__(self, exc_type, exc_value, traceback):
        self.events.append("exit-device")


class _FakeNpu:
    def __init__(self, events, stream_pointer=0x4567):
        self.events = events
        self.stream_pointer = stream_pointer

    def device(self, device):
        self.events.append(("device", device))
        return _FakeDeviceContext(self.events)

    def current_stream(self, *, device):
        self.events.append(("current-stream", device))
        return SimpleNamespace(npu_stream=self.stream_pointer)


class _RecordingAbsFunction:
    def __init__(self, *, status=0, detail=b""):
        self.status = status
        self.detail = detail
        self.calls = []

    def __call__(self, *arguments):
        self.calls.append(arguments)
        arguments[-2].value = self.detail
        return self.status


def _install_fake_abs_runtime(monkeypatch, events, *, stream_pointer=0x4567):
    monkeypatch.setattr(ascend.torch, "Tensor", _FakeTensor)
    monkeypatch.setattr(
        ascend.torch,
        "empty_strided",
        lambda shape, stride, *, device, dtype: _FakeTensor(
            shape,
            stride,
            device=device,
            dtype=dtype,
            data_pointer=0x5678,
        ),
    )
    monkeypatch.setattr(
        ascend,
        "_npu_module",
        lambda: _FakeNpu(events, stream_pointer=stream_pointer),
    )


def test_python_ascend_oracle_abs_passes_exact_unary_abi_and_preserves_layout(
    monkeypatch,
):
    events = []
    _install_fake_abs_runtime(monkeypatch, events)
    function = _RecordingAbsFunction()
    library = SimpleNamespace(flagdnn_test_aclnn_abs=function)
    x = _FakeTensor(
        shape=(2, 3, 4, 5),
        stride=(60, 1, 15, 3),
        data_pointer=0x1234,
    )
    oracle = ascend.AscendDnnOracle(library_loader=lambda: library)

    output = oracle.abs(x)

    assert tuple(output.shape) == (2, 3, 4, 5)
    assert tuple(output.stride()) == (60, 1, 15, 3)
    assert output.dtype is torch.float32
    assert output.device is x.device
    assert oracle._last_device is x.device
    assert len(function.calls) == 1
    arguments = function.calls[0]
    assert len(arguments) == 12
    assert arguments[0].value == 0x1234
    assert tuple(arguments[1]) == (2, 3, 4, 5)
    assert tuple(arguments[2]) == (60, 1, 15, 3)
    assert arguments[3].value == 4
    assert arguments[4].value == 0x5678
    assert tuple(arguments[5]) == (2, 3, 4, 5)
    assert tuple(arguments[6]) == (60, 1, 15, 3)
    assert arguments[7].value == 4
    assert arguments[8].value == 2
    assert arguments[9].value == 0x4567
    assert len(arguments[10]) == ascend._ERROR_BUFFER_SIZE
    assert arguments[11].value == ascend._ERROR_BUFFER_SIZE
    assert events == [
        ("device", x.device),
        "enter-device",
        ("current-stream", x.device),
        "exit-device",
    ]


@pytest.mark.parametrize(
    ("x", "error_type", "message"),
    [
        (object(), TypeError, "torch.Tensor"),
        (
            _FakeTensor(layout=torch.sparse_coo),
            ValueError,
            "strided",
        ),
        (_FakeTensor(device=_FakeDevice("cpu")), ValueError, "NPU"),
        (_FakeTensor(dtype=torch.float64), TypeError, "does not support"),
        (_FakeTensor(shape=(), stride=()), ValueError, "rank-0"),
        (_FakeTensor(shape=(1,) * 9, stride=(1,) * 9), ValueError, "rank"),
        (_FakeTensor(storage_offset=1), ValueError, "storage offset"),
        (
            _FakeTensor(shape=(2, 3), stride=(0, 1)),
            ValueError,
            "contiguous or 4D channels-last",
        ),
        (
            _FakeTensor(shape=(2, 3), stride=(4, 1)),
            ValueError,
            "contiguous or 4D channels-last",
        ),
    ],
    ids=(
        "non-tensor",
        "non-strided",
        "cpu",
        "unsupported-dtype",
        "rank-zero",
        "rank-nine",
        "nonzero-storage-offset",
        "overlapping-stride",
        "strided-with-holes",
    ),
)
def test_python_ascend_oracle_abs_rejects_invalid_input_before_loading_cann(
    monkeypatch, x, error_type, message
):
    _install_fake_abs_runtime(monkeypatch, [])

    def fail_loader():
        pytest.fail("library must remain lazy for invalid Abs input")

    oracle = ascend.AscendDnnOracle(library_loader=fail_loader)

    with pytest.raises(error_type, match=message):
        oracle.abs(x)


def test_python_abs_native_error_after_device_context_restores(
    monkeypatch,
):
    events = []
    _install_fake_abs_runtime(monkeypatch, events, stream_pointer=0xABCD)
    function = _RecordingAbsFunction(status=27, detail=b"native detail")
    library = SimpleNamespace(flagdnn_test_aclnn_abs=function)
    x = _FakeTensor(shape=(4,), stride=(1,), data_pointer=0x1234)
    oracle = ascend.AscendDnnOracle(library_loader=lambda: library)

    with pytest.raises(RuntimeError) as caught:
        oracle.abs(x)

    assert events[-1] == "exit-device"
    assert str(caught.value) == (
        "aclnnAbs oracle failed: status=27, detail=native detail, "
        "x_shape=(4,), x_stride=(1,), output_shape=(4,), "
        "output_stride=(1,), dtype=torch.float32, device=npu:2, "
        "stream=0xabcd"
    )


def test_python_ascend_oracle_abs_has_no_framework_arithmetic_fallback():
    source = inspect.getsource(ascend.AscendDnnOracle.abs)

    for forbidden in ("torch.abs(", ".abs(", ".cpu("):
        assert forbidden not in source
    assert "flagdnn_test_aclnn_abs" in source
    assert "with npu.device(x.device):" in source


def test_ascend_oracle_constructor_and_dtype_checks_are_lazy():
    load_calls = []
    oracle = ascend.AscendDnnOracle(
        library_loader=lambda: load_calls.append("load")
    )

    assert oracle.vendor_name == "ascend"
    assert oracle.implementation == "aclnnAdd"
    assert oracle.supports_dtype(torch.float16)
    assert oracle.supports_dtype(torch.bfloat16)
    assert oracle.supports_dtype(torch.float32)
    assert not oracle.supports_dtype(torch.float64)
    assert not oracle.supports_dtype(torch.int32)
    assert load_calls == []


def test_ascend_oracle_rejects_cpu_before_loading_cann():
    def fail_loader():
        pytest.fail("library must remain lazy for invalid input")

    oracle = ascend.AscendDnnOracle(library_loader=fail_loader)

    with pytest.raises(ValueError, match="NPU tensors"):
        oracle.add(torch.ones(8), torch.ones(8))


def test_ascend_oracle_has_no_torch_arithmetic_fallback():
    source = inspect.getsource(ascend.AscendDnnOracle.add)

    for forbidden in (
        "torch.add(",
        ".add(",
        "operator.add(",
        ".cpu(",
    ):
        assert forbidden not in source
    assert "flagdnn_test_aclnn_add" in source


@pytest.mark.parametrize(
    ("alpha", "error_type"),
    ((True, TypeError), (10**10000, ValueError)),
    ids=("bool", "overflow"),
)
def test_ascend_oracle_rejects_invalid_alpha_before_loading_cann(
    alpha, error_type
):
    def fail_loader():
        pytest.fail("library must remain lazy for invalid alpha")

    oracle = ascend.AscendDnnOracle(library_loader=fail_loader)

    with pytest.raises(error_type, match="alpha"):
        oracle.add(torch.ones(8), torch.ones(8), alpha=alpha)


def test_ascend_oracle_rejects_non_strided_layout_before_loading_cann():
    def fail_loader():
        pytest.fail("library must remain lazy for invalid layout")

    oracle = ascend.AscendDnnOracle(library_loader=fail_loader)
    sparse = torch.ones(8).to_sparse()

    with pytest.raises(ValueError, match="strided"):
        oracle.add(sparse, sparse)


def test_ascend_oracle_uses_restoring_device_guard_not_set_device():
    source = inspect.getsource(ascend.AscendDnnOracle.add)

    assert "npu.set_device(" not in source
    assert "with npu.device(x.device):" in source


def test_ascend_oracle_synchronizes_last_operation_device(monkeypatch):
    calls = []
    fake_npu = SimpleNamespace(
        synchronize=lambda device=None: calls.append(device)
    )
    monkeypatch.setattr(ascend, "_npu_module", lambda: fake_npu)
    oracle = ascend.AscendDnnOracle()
    oracle._last_device = torch.device("npu:3")

    oracle.synchronize()

    assert calls == [torch.device("npu:3")]
