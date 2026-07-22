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

import importlib
from types import SimpleNamespace

import pytest
import torch

from benchmark.base import (
    CudnnCompareBenchmark,
    DnnCompareBenchmark,
    skip_unsupported_cudnn_graph,
)
from devtools.dnn_reference.interfaces import DnnReferenceNotSupportedError
from benchmark.reduction import ReductionBenchmarkBase
from benchmark.run_all_tests_perf import (
    parse_perf_output,
    parse_pytest_outcome_count,
)


def test_parse_perf_output_preserves_arbitrary_shape_detail():
    output = (
        "Operator: sdpa  cuDNN Compare Performance Test "
        "(dtype=torch.float16, mode=kernel, "
        "level=comprehensive)\n"
        "SUCCESS 0.500000 1.000000 0.500 3.000 4.000 "
        "[(1, 32, 1024, 64), (1, 8, 1024, 64), "
        "'causal=True']\n"
    )

    records = parse_perf_output(output)

    assert records == [
        {
            "operator": "sdpa",
            "dtype": "torch.float16",
            "dtype_short": "float16",
            "cudnn_latency": 0.5,
            "flagdnn_latency": 1.0,
            "speedup": 0.5,
            "mode": "kernel",
            "level": "comprehensive",
            "cudnn_gbps": 3.0,
            "flagdnn_gbps": 4.0,
            "size_detail": (
                "[(1, 32, 1024, 64), (1, 8, 1024, 64), " "'causal=True']"
            ),
        }
    ]


def test_bad_param_is_not_treated_as_an_unsupported_benchmark():
    error = RuntimeError("cudnn_status: CUDNN_STATUS_BAD_PARAM")

    with pytest.raises(RuntimeError) as caught:
        skip_unsupported_cudnn_graph(error, "malformed")

    assert caught.value is error


def test_not_supported_is_skipped_by_benchmark():
    error = RuntimeError("cudnn_status: CUDNN_STATUS_NOT_SUPPORTED")

    with pytest.raises(pytest.skip.Exception, match="does not support"):
        skip_unsupported_cudnn_graph(error, "unsupported")


def test_parse_pytest_outcome_count_handles_partial_skips():
    output = (
        "collected 3 items\n"
        "================ 2 passed, 1 skipped in 1.00s ================\n"
    )

    assert parse_pytest_outcome_count(output, "passed") == 2
    assert parse_pytest_outcome_count(output, "skipped") == 1
    assert parse_pytest_outcome_count(output, "failed") == 0


@pytest.mark.parametrize(
    "return_code,stdout,expected",
    (
        (
            0,
            "collected 1 item\n1 passed in 0.01s\n",
            0,
        ),
        (
            0,
            "collected 1 item\n1 skipped in 0.01s\n",
            0,
        ),
        (
            1,
            "collected 1 item\n1 failed in 0.01s\n",
            1,
        ),
        (
            2,
            "collected 1 item\n1 error in 0.01s\n",
            1,
        ),
    ),
)
def test_benchmark_runner_exit_code_reflects_child_failures(
    monkeypatch, tmp_path, return_code, stdout, expected
):
    module = importlib.import_module("benchmark.run_all_tests_perf")
    test_file = tmp_path / "test_fake.py"
    monkeypatch.setattr(module, "LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(module, "REPORT_FILE", str(tmp_path / "summary.json"))
    monkeypatch.setattr(module, "DATA_FILE", str(tmp_path / "data.json"))
    monkeypatch.setattr(module, "TARGET_OPERATORS", [])
    monkeypatch.setattr(module.glob, "glob", lambda pattern: [str(test_file)])
    result = SimpleNamespace(returncode=return_code, stdout=stdout, stderr="")
    monkeypatch.setattr(
        module.subprocess, "run", lambda *args, **kwargs: result
    )

    assert module.main() == expected


def test_reduction_benchmark_shape_detail_includes_operation():
    benchmark = object.__new__(ReductionBenchmarkBase)
    benchmark.case = ((8, 8, 32, 32), 1, "AVG")
    x = torch.empty((8, 8, 32, 32))

    assert benchmark.shape_detail((x,)) == {
        "input": (8, 8, 32, 32),
        "dim": 1,
        "mode": "AVG",
        "keepdim": True,
    }


class _FakePrepared:
    reference_name = "fake"

    def __init__(self, closed, name):
        self._closed = closed
        self._name = name

    def run(self):
        return None

    __call__ = run

    def close(self):
        self._closed.append(self._name)


class _FakeBaseline:
    vendor_name = "nvidia"
    display_name = "fake"

    def supports(self, _op_name, _dtype):
        return True


class _FakeTimer:
    def measure_pair(self, baseline_run, flag_dnn_run):
        baseline_run()
        flag_dnn_run()
        return 1.0, 0.5


class _PartialUnsupportedBenchmark(DnnCompareBenchmark):
    op_name = "partial"
    shapes = ("unsupported", "supported")

    def __init__(self):
        self.closed = []
        super().__init__(_FakeBaseline(), timer=_FakeTimer())

    def make_inputs(self, shape, _dtype):
        return (shape,)

    def build_baseline_runner(self, inputs):
        if inputs == ("unsupported",):
            raise DnnReferenceNotSupportedError("unsupported shape")
        return _FakePrepared(self.closed, "baseline")

    def build_flag_dnn_runner(self, _inputs):
        return _FakePrepared(self.closed, "flag_dnn")

    def shape_detail(self, inputs):
        return inputs[0]

    def transfer_bytes(self, _inputs):
        return 0


def test_compare_benchmark_continues_after_unsupported_shape(capsys):
    benchmark = _PartialUnsupportedBenchmark()

    benchmark.run(torch.float32)

    output = capsys.readouterr().out
    assert "UNSUPPORTED" in output
    assert "unsupported shape" in output
    assert "SUCCESS" in output
    assert benchmark.closed == ["baseline"]


class _PartialUnsupportedCudnnBenchmark(CudnnCompareBenchmark):
    op_name = "partial_cudnn"
    shapes = ("unsupported", "supported")

    def __init__(self):
        super().__init__(cudnn_handle=None)

    def make_inputs(self, shape, _dtype):
        return (shape,)

    def build_cudnn_runner(self, inputs):
        if inputs == ("unsupported",):
            pytest.skip("cuDNN has no engine for this shape")
        return lambda: None

    def build_flag_dnn_runner(self, _inputs):
        return lambda: None

    def shape_detail(self, inputs):
        return inputs[0]

    def transfer_bytes(self, _inputs):
        return 0


def test_cudnn_compare_benchmark_continues_after_unsupported_shape(
    monkeypatch, capsys
):
    benchmark = _PartialUnsupportedCudnnBenchmark()
    monkeypatch.setattr("benchmark.base.torch.cuda.synchronize", lambda: None)
    monkeypatch.setattr("benchmark.base.bench_ms", lambda _runner: 1.0)

    benchmark.run(torch.float32)

    output = capsys.readouterr().out
    assert "UNSUPPORTED" in output
    assert "cuDNN has no engine for this shape" in output
    assert "SUCCESS" in output
