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
import json
import sys
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
    size_detail = "[(1, 32, 1024, 64), (1, 8, 1024, 64), 'causal=True']"
    raw_record = {
        "schema_version": 1,
        "operator": "sdpa",
        "dtype": "torch.float16",
        "mode": "kernel",
        "level": "comprehensive",
        "execution_path": "compiled_graph.bind",
        "size_detail": size_detail,
        "baseline_latency_ms": 0.5,
        "flagdnn_latency_ms": 1.0,
        "speedup": 0.5,
    }
    output = (
        "Operator: sdpa  cuDNN Compare Performance Test "
        "(dtype=torch.float16, mode=kernel, "
        "level=comprehensive)\n"
        "SUCCESS 0.500000 1.000000 0.500 3.000 4.000 "
        f"{size_detail}\n"
        f"FLAGDNN_PERF_JSON {json.dumps(raw_record)}\n"
    )

    records = parse_perf_output(output, source_file="test_sdpa.py")

    assert len(records) == 1
    assert records[0]["size_detail"] == size_detail
    assert records[0]["source_file"] == "test_sdpa.py"
    assert records[0]["source_row_index"] == 0


def test_parse_perf_output_allows_explicit_legacy_compatibility():
    output = (
        "Operator: add_fp16 Performance Test\n"
        "SUCCESS 0.500000 1.000000 0.500 3.000 4.000 [(16,)]\n"
    )

    records = parse_perf_output(
        output,
        source_file="test_add.py",
        allow_legacy=True,
    )

    assert len(records) == 1
    assert records[0]["schema_version"] == 0
    assert records[0]["operator"] == "add"
    assert records[0]["dtype"] == "torch.float16"
    assert records[0]["execution_path"] == "legacy_text_parser"


def _install_fake_cudnn(monkeypatch):
    fake_error = type("cudnnGraphNotSupportedError", (RuntimeError,), {})
    monkeypatch.setattr(
        "benchmark.base.get_cudnn",
        lambda: SimpleNamespace(cudnnGraphNotSupportedError=fake_error),
    )


def test_bad_param_is_not_treated_as_an_unsupported_benchmark(monkeypatch):
    _install_fake_cudnn(monkeypatch)
    error = RuntimeError("cudnn_status: CUDNN_STATUS_BAD_PARAM")

    with pytest.raises(RuntimeError) as caught:
        skip_unsupported_cudnn_graph(error, "malformed")

    assert caught.value is error


def test_not_supported_is_skipped_by_benchmark(monkeypatch):
    _install_fake_cudnn(monkeypatch)
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


def test_environment_snapshot_keeps_npu_fields_out_of_nvidia(monkeypatch):
    module = importlib.import_module("benchmark.run_all_tests_perf")
    monkeypatch.setenv("DNN_VENDOR", "nvidia")
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        module.torch.cuda,
        "get_device_name",
        lambda: "H100",
    )
    monkeypatch.setattr(
        module.torch.cuda,
        "get_device_capability",
        lambda: (9, 0),
    )

    snapshot = module._environment_snapshot()

    assert snapshot["cuda_available"] is True
    assert snapshot["device_name"] == "H100"
    assert snapshot["device_capability"] == [9, 0]
    assert "npu_available" not in snapshot
    assert "torch_npu" not in snapshot
    assert "cann_runtime" not in snapshot


def test_environment_snapshot_adds_ascend_runtime_fields(monkeypatch):
    module = importlib.import_module("benchmark.run_all_tests_perf")
    fake_npu = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 16,
        get_device_name=lambda: "Ascend 910",
    )
    monkeypatch.setenv("DNN_VENDOR", "ascend")
    monkeypatch.setenv("ASCEND_HOME_PATH", "/opt/cann")
    monkeypatch.setenv("ASCEND_OPP_PATH", "/opt/cann/opp")
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(module.torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(module.torch.version, "cann", "9.0.0", raising=False)

    snapshot = module._environment_snapshot()

    assert snapshot["npu_available"] is True
    assert snapshot["npu_device_count"] == 16
    assert snapshot["npu_device_name"] == "Ascend 910"
    assert snapshot["torch_npu"]
    assert snapshot["cann_runtime"] == "9.0.0"
    assert snapshot["ascend_home_path"] == "/opt/cann"
    assert snapshot["ascend_opp_path"] == "/opt/cann/opp"


def test_environment_snapshot_accepts_torch_npu_device_api(monkeypatch):
    module = importlib.import_module("benchmark.run_all_tests_perf")
    fake_npu = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 8,
        get_device_name=lambda: "Ascend 910B",
    )
    fake_torch_npu = SimpleNamespace(
        __version__="test",
        npu=fake_npu,
    )
    monkeypatch.setenv("DNN_VENDOR", "ascend")
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.delattr(module.torch, "npu", raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)

    snapshot = module._environment_snapshot()

    assert snapshot["npu_available"] is True
    assert snapshot["npu_device_count"] == 8
    assert snapshot["npu_device_name"] == "Ascend 910B"
    assert snapshot["torch_npu"] == "test"


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
    test_dir = tmp_path / "benchmark"
    test_dir.mkdir()
    (test_dir / "test_fake.py").write_text("", encoding="utf-8")
    output_dir = tmp_path / "output"
    monkeypatch.setattr(module, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(module, "TEST_DIR", str(test_dir))
    monkeypatch.setattr(module, "_environment_snapshot", lambda: {})
    result = SimpleNamespace(returncode=return_code, stdout=stdout, stderr="")
    monkeypatch.setattr(
        module.subprocess, "run", lambda *args, **kwargs: result
    )

    assert module.main(["--output-dir", str(output_dir)]) == expected
    summary = json.loads(
        (output_dir / "benchmark_summary.json").read_text(encoding="utf-8")
    )
    assert summary["total"] == 1
    assert summary["details"][0]["return_code"] == return_code


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
