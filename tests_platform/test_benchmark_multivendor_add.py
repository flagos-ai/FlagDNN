from pathlib import Path
from types import SimpleNamespace

import torch

from benchmark import base
from benchmark import consts


class _PreparedAdd:
    def __init__(self):
        self.output = torch.empty(1)
        self.run_count = 0
        self.closed = False

    def run(self):
        self.run_count += 1
        return self.output

    def close(self):
        self.closed = True


class _Baseline:
    vendor_name = "ascend"
    implementation = "aclnn"
    display_name = "ACLNN"

    def __init__(self):
        self.prepared = None

    def supports_dtype(self, dtype):
        return dtype == torch.float32

    def prepare_add(self, x, y, *, alpha=1):
        assert alpha == 1
        self.prepared = _PreparedAdd()
        return self.prepared


class _AddBenchmark(base.DnnCompareBenchmark):
    op_name = "add"
    shapes = (((4,), (4,)),)
    shape_ids_env = "FLAGDNN_TEST_ADD_SHAPE_IDS"

    def make_inputs(self, shape, dtype):
        x_shape, y_shape = shape
        return torch.ones(x_shape, dtype=dtype), torch.ones(
            y_shape, dtype=dtype
        )

    def build_baseline_runner(self, inputs):
        return self.baseline.prepare_add(*inputs, alpha=1)

    def build_flag_dnn_runner(self, inputs):
        x, y = inputs
        return lambda: x + y


def test_generic_benchmark_runs_and_closes_selected_baseline(
    monkeypatch, capsys
):
    measurements = iter((2.0, 1.0))

    def fake_bench(function):
        function()
        return next(measurements)

    monkeypatch.setattr(base, "bench_ms", fake_bench)
    baseline = _Baseline()

    _AddBenchmark(baseline).run(torch.float32)

    assert baseline.prepared is not None
    assert baseline.prepared.run_count == 1
    assert baseline.prepared.closed
    output = capsys.readouterr().out
    assert "ACLNN Compare Performance Test" in output
    assert "ACLNN Latency (ms)" in output


def test_add_benchmark_source_has_no_vendor_specific_control_flow():
    source = (
        Path(__file__).parents[1] / "benchmark" / "test_add.py"
    ).read_text(encoding="utf-8")

    for forbidden in (
        "import cudnn",
        "get_cudnn",
        "cudnn_handle",
        "torch.cuda",
        "torch.npu",
    ):
        assert forbidden not in source


def test_generic_performance_environment_overrides_legacy_names(monkeypatch):
    monkeypatch.setenv("FLAGDNN_CUDNN_PERF_WARMUP", "11")
    monkeypatch.setenv("FLAGDNN_CUDNN_PERF_REPEAT", "12")
    monkeypatch.setenv("FLAGDNN_CUDNN_PERF_MIN_SPEEDUP", "0.8")
    monkeypatch.setenv("FLAGDNN_PERF_WARMUP", "21")
    monkeypatch.setenv("FLAGDNN_PERF_REPEAT", "22")
    monkeypatch.setenv("FLAGDNN_PERF_MIN_SPEEDUP", "0.9")

    assert consts.bench_warmup() == 21
    assert consts.bench_repeat() == 22
    assert consts.min_speedup() == 0.9


class _FakePointwiseTensor:
    def __init__(self, device_type):
        self.device = SimpleNamespace(type=device_type)
        self.memory_format = None

    def dim(self):
        return 4

    def contiguous(self, *, memory_format):
        self.memory_format = memory_format
        return self


def test_pointwise_layout_uses_channels_last_only_for_cuda():
    cuda = _FakePointwiseTensor("cuda")
    npu = _FakePointwiseTensor("npu")

    assert consts.pointwise_layout(cuda) is cuda
    assert cuda.memory_format == torch.channels_last
    assert consts.pointwise_layout(npu) is npu
    assert npu.memory_format is None


def test_lower_percentile_uses_an_observed_event_sample():
    samples = [9.0, 1.0, 7.0, 2.0, 8.0, 3.0]

    assert base._lower_percentile(samples) == 2.0
