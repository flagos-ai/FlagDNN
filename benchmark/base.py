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

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pytest

import torch  # noqa: E402

from benchmark.attri_util import (  # noqa: E402
    BenchLevel,
    BenchMode,
    BenchmarkMetrics,
)
from benchmark import consts  # noqa: E402
from benchmark.timers import BenchmarkTimer, create_timer  # noqa: E402
from benchmark.timers.ascend import (  # noqa: E402
    AscendEventTimer,
    lower_percentile,
)
from benchmark.timers.default import (  # noqa: E402
    TritonBenchmarkTimer,
    bench_ms,
)


_lower_percentile = lower_percentile


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


def bench_pair_ms(first, second, *, use_ascend_events=False):
    timer = AscendEventTimer() if use_ascend_events else TritonBenchmarkTimer()
    return timer.measure_pair(first, second)


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

    def __init__(self, baseline, timer: BenchmarkTimer | None = None) -> None:
        self.baseline = baseline
        self.timer = timer

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
        if not self.baseline.supports(self.op_name, dtype):
            pytest.skip(
                f"{self.baseline.display_name} does not support {dtype}"
            )

        metrics = []
        for shape in self.selected_shapes():
            inputs = self.make_inputs(shape, dtype)
            baseline_run = self.build_baseline_runner(inputs)
            try:
                flag_dnn_run = self.build_flag_dnn_runner(inputs)
                timer = self.timer or create_timer(
                    self.baseline.vendor_name, inputs
                )
                baseline_ms, flag_dnn_ms = timer.measure_pair(
                    baseline_run.run, flag_dnn_run
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


class CudnnCompareBenchmark:
    op_name: str = ""
    dtypes: tuple[Any, ...] = consts.COMPARE_FLOAT_DTYPES
    shapes: Any = ()
    shape_ids_env: str = ""

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
