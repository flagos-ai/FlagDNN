from __future__ import annotations

import os
import statistics
from functools import lru_cache
from typing import Any

import pytest

import torch  # noqa: E402
import triton  # noqa: E402

from benchmark.attri_util import (  # noqa: E402
    BenchLevel,
    BenchMode,
    BenchmarkMetrics,
)
from benchmark import consts  # noqa: E402


@lru_cache(maxsize=1)
def get_cudnn():
    return pytest.importorskip("cudnn", exc_type=ImportError)


def cudnn_data_type(dtype):
    cudnn = get_cudnn()
    if dtype == torch.float16:
        return cudnn.data_type.HALF
    if dtype == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    if dtype == torch.float32:
        return cudnn.data_type.FLOAT
    if dtype == torch.float8_e4m3fn:
        return cudnn.data_type.FP8_E4M3
    if dtype == torch.float8_e5m2:
        return cudnn.data_type.FP8_E5M2
    if dtype == torch.float64:
        return cudnn.data_type.DOUBLE
    if dtype == torch.bool:
        return cudnn.data_type.BOOLEAN
    raise TypeError(f"Unsupported dtype for cuDNN frontend: {dtype}")


def skip_unsupported_cudnn_graph(exc, op_name):
    cudnn = get_cudnn()
    message = str(exc)
    if (
        isinstance(exc, cudnn.cudnnGraphNotSupportedError)
        or "CUDNN_STATUS_BAD_PARAM" in message
        or "CUDNN_STATUS_NOT_SUPPORTED" in message
        or "No valid engine configs" in message
    ):
        pytest.skip(f"cuDNN frontend does not support {op_name}: {exc}")
    raise exc


def bench_ms(fn):
    return triton.testing.do_bench(
        fn,
        warmup=consts.bench_warmup(),
        rep=consts.bench_repeat(),
        return_mode="median",
    )


def _lower_percentile(values, quantile=0.2):
    """Return a measured sample without interpolating event timestamps."""
    if not values:
        raise ValueError("cannot summarize an empty timing sample")
    ordered = sorted(values)
    index = int((len(ordered) - 1) * quantile)
    return ordered[index]


def _format_timing_distribution(values):
    ordered = sorted(values)
    return (
        f"min={ordered[0]:.6f}, "
        f"p20={_lower_percentile(ordered):.6f}, "
        f"median={statistics.median(ordered):.6f}, "
        f"max={ordered[-1]:.6f}"
    )


def bench_pair_ms(first, second, *, use_ascend_events=False):
    if not use_ascend_events:
        return bench_ms(first), bench_ms(second)

    device_interface = triton.runtime.driver.active.get_device_interface()
    active_driver = triton.runtime.driver.active
    first()
    second()
    device_interface.synchronize()

    for _ in range(max(1, consts.bench_warmup())):
        first()
        second()
    device_interface.synchronize()

    repeat = max(1, consts.bench_repeat())
    cache = active_driver.get_empty_cache_for_benchmark()
    # Queue enough device work before a timed region that the NPU cannot reach
    # its start event while the heavily loaded host is still submitting the
    # measured kernel.  One 192 MiB clear is occasionally too short on 910C.
    queue_guard_clears = 4
    for _ in range(queue_guard_clears):
        active_driver.clear_cache(cache)
    device_interface.synchronize()
    first_start = [
        device_interface.Event(enable_timing=True) for _ in range(repeat)
    ]
    first_end = [
        device_interface.Event(enable_timing=True) for _ in range(repeat)
    ]
    second_start = [
        device_interface.Event(enable_timing=True) for _ in range(repeat)
    ]
    second_end = [
        device_interface.Event(enable_timing=True) for _ in range(repeat)
    ]

    def record(fn, start, end):
        for _ in range(queue_guard_clears):
            active_driver.clear_cache(cache)
        start.record()
        fn()
        end.record()

    for index in range(repeat):
        if index % 2 == 0:
            record(first, first_start[index], first_end[index])
            record(second, second_start[index], second_end[index])
        else:
            record(second, second_start[index], second_end[index])
            record(first, first_start[index], first_end[index])

    device_interface.synchronize()
    first_times = [
        start.elapsed_time(end) for start, end in zip(first_start, first_end)
    ]
    second_times = [
        start.elapsed_time(end) for start, end in zip(second_start, second_end)
    ]
    if os.getenv("FLAGDNN_PERF_DEBUG_TIMING") == "1":
        print(
            "Ascend timing samples: "
            f"baseline({_format_timing_distribution(first_times)}), "
            f"FlagDNN({_format_timing_distribution(second_times)})"
        )
    # The queue guard removes host-submission bubbles. Keep the median as the
    # final statistic because Ascend events can also contain implausibly short
    # outliers (for example 0.76 us among otherwise stable 1.7 us ACLNN runs).
    # Cache clearing remains outside every timed interval, so this is still a
    # cold-cache kernel measurement rather than a warm-cache throughput loop.
    return statistics.median(first_times), statistics.median(second_times)


def format_perf_result(op_name, dtype, metrics, reference_name="cuDNN"):
    title = (
        f"\nOperator: {op_name}  {reference_name} Compare Performance Test "
        f"(dtype={dtype}, mode={BenchMode.KERNEL.value}, "
        f"level={BenchLevel.COMPREHENSIVE.value})\n"
    )
    col_names = [
        f"{'Status':<10}",
        f"{f'{reference_name} Latency (ms)':>20}",
        f"{'FlagDNN Graph Latency (ms)':>28}",
        f"{'FlagDNN Speedup':>20}",
        f"{f'{reference_name} GBPS':>20}",
        f"{'FlagDNN GBPS':>20}",
        f"{'Size Detail':>20}\n",
    ]
    header = " ".join(col_names)
    lines = [title, header, "-" * len(header) + "\n"]
    for item in metrics:
        status = "SUCCESS" if item.error_msg is None else "FAILED"
        lines.append(
            f"{status:<10}"
            f"{item.latency_base:>20.6f}"
            f"{item.latency:>28.6f}"
            f"{item.speedup:>20.3f}"
            f"{item.gbps_base:>20.3f}"
            f"{item.gbps:>20.3f}"
            f"          {item.shape_detail}\n"
        )
    return "".join(lines)


class DnnCompareBenchmark:
    op_name: str = ""
    dtypes: tuple[Any, ...] = consts.COMPARE_FLOAT_DTYPES
    shapes: Any = ()
    shape_ids_env: str = ""
    legacy_shape_ids_env: str = ""
    enforce_min_speedup: bool = False

    def __init__(self, baseline):
        self.baseline = baseline

    def selected_shapes(self):
        legacy = (
            (self.legacy_shape_ids_env,) if self.legacy_shape_ids_env else ()
        )
        return consts.selected_shapes(
            self.shapes,
            self.shape_ids_env,
            legacy_env_names=legacy,
        )

    def make_inputs(self, shape, dtype):
        raise NotImplementedError

    def build_baseline_runner(self, inputs):
        raise NotImplementedError

    def build_flag_dnn_runner(self, inputs):
        raise NotImplementedError

    def transfer_bytes(self, inputs):
        return (
            sum(
                item.numel() * item.element_size()
                for item in inputs
                if isinstance(item, torch.Tensor)
            )
            * 2
        )

    def shape_detail(self, inputs):
        return [
            item.size() for item in inputs if isinstance(item, torch.Tensor)
        ]

    def run(self, dtype):
        if not self.baseline.supports_dtype(dtype):
            pytest.skip(
                f"{self.baseline.display_name} does not support {dtype}"
            )

        metrics = []
        for shape in self.selected_shapes():
            inputs = self.make_inputs(shape, dtype)
            baseline_run = self.build_baseline_runner(inputs)
            try:
                flag_dnn_run = self.build_flag_dnn_runner(inputs)
                use_ascend_events = (
                    self.baseline.vendor_name == "ascend"
                    and any(
                        isinstance(item, torch.Tensor)
                        and item.device.type == "npu"
                        for item in inputs
                    )
                )
                baseline_ms, flag_dnn_ms = bench_pair_ms(
                    baseline_run.run,
                    flag_dnn_run,
                    use_ascend_events=use_ascend_events,
                )
                bytes_moved = self.transfer_bytes(inputs)
            finally:
                baseline_run.close()

            metrics.append(
                BenchmarkMetrics(
                    shape_detail=self.shape_detail(inputs),
                    latency_base=baseline_ms,
                    latency=flag_dnn_ms,
                    speedup=(
                        baseline_ms / flag_dnn_ms
                        if flag_dnn_ms > 0
                        else float("inf")
                    ),
                    gbps_base=(
                        bytes_moved / baseline_ms / 1e6
                        if baseline_ms > 0
                        else None
                    ),
                    gbps=(
                        bytes_moved / flag_dnn_ms / 1e6
                        if flag_dnn_ms > 0
                        else None
                    ),
                )
            )

        print(
            format_perf_result(
                self.op_name,
                dtype,
                metrics,
                reference_name=self.baseline.display_name,
            )
        )
        assert all(
            item.latency_base and item.latency_base > 0 for item in metrics
        )
        assert all(item.latency and item.latency > 0 for item in metrics)
        if self.enforce_min_speedup:
            min_speedup = consts.min_speedup()
            if min_speedup > 0:
                slow = [item for item in metrics if item.speedup < min_speedup]
                assert not slow, (
                    f"{self.op_name} FlagDNN graph speedup below "
                    f"{min_speedup}: "
                    + ", ".join(
                        f"{item.shape_detail}={item.speedup:.3f}"
                        for item in slow
                    )
                )


class CudnnCompareBenchmark:
    op_name: str = ""
    dtypes: tuple[Any, ...] = consts.COMPARE_FLOAT_DTYPES
    shapes: Any = ()
    shape_ids_env: str = ""
    enforce_min_speedup: bool = False

    def __init__(self, cudnn_handle):
        self.cudnn_handle = cudnn_handle

    def selected_shapes(self):
        return consts.selected_shapes(self.shapes, self.shape_ids_env)

    def make_inputs(self, shape, dtype):
        raise NotImplementedError

    def build_cudnn_runner(self, inputs):
        raise NotImplementedError

    def build_flag_dnn_runner(self, inputs):
        raise NotImplementedError

    def transfer_bytes(self, inputs):
        return (
            sum(
                item.numel() * item.element_size()
                for item in inputs
                if isinstance(item, torch.Tensor)
            )
            * 2
        )

    def shape_detail(self, inputs):
        return [
            item.size() for item in inputs if isinstance(item, torch.Tensor)
        ]

    def run(self, dtype):
        metrics = []
        for shape in self.selected_shapes():
            inputs = self.make_inputs(shape, dtype)
            cudnn_run = self.build_cudnn_runner(inputs)
            flag_dnn_run = self.build_flag_dnn_runner(inputs)

            torch.cuda.synchronize()
            cudnn_ms = bench_ms(cudnn_run)
            flag_dnn_ms = bench_ms(flag_dnn_run)
            bytes_moved = self.transfer_bytes(inputs)

            metrics.append(
                BenchmarkMetrics(
                    shape_detail=self.shape_detail(inputs),
                    latency_base=cudnn_ms,
                    latency=flag_dnn_ms,
                    speedup=(
                        cudnn_ms / flag_dnn_ms
                        if flag_dnn_ms > 0
                        else float("inf")
                    ),
                    gbps_base=(
                        bytes_moved / cudnn_ms / 1e6 if cudnn_ms > 0 else None
                    ),
                    gbps=(
                        bytes_moved / flag_dnn_ms / 1e6
                        if flag_dnn_ms > 0
                        else None
                    ),
                )
            )

        print(format_perf_result(self.op_name, dtype, metrics))
        assert all(
            item.latency_base and item.latency_base > 0 for item in metrics
        )
        assert all(item.latency and item.latency > 0 for item in metrics)
        if self.enforce_min_speedup:
            min_speedup = consts.min_speedup()
            if min_speedup > 0:
                slow = [item for item in metrics if item.speedup < min_speedup]
                assert not slow, (
                    f"{self.op_name} FlagDNN graph speedup below "
                    f"{min_speedup}: "
                    + ", ".join(
                        f"{item.shape_detail}={item.speedup:.3f}"
                        for item in slow
                    )
                )
