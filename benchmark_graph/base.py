from __future__ import annotations

from typing import Any

import pytest

cudnn = pytest.importorskip("cudnn", exc_type=ImportError)

import torch  # noqa: E402
import triton  # noqa: E402

from benchmark.attri_util import (  # noqa: E402
    BenchLevel,
    BenchMode,
    BenchmarkMetrics,
)
from benchmark_graph import consts  # noqa: E402


def get_cudnn():
    return cudnn


def cudnn_data_type(dtype):
    cudnn = get_cudnn()
    if dtype == torch.float16:
        return cudnn.data_type.HALF
    if dtype == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    if dtype == torch.float32:
        return cudnn.data_type.FLOAT
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


def format_perf_result(op_name, dtype, metrics):
    title = (
        f"\nOperator: {op_name}  cuDNN Compare Performance Test "
        f"(dtype={dtype}, mode={BenchMode.KERNEL.value}, "
        f"level={BenchLevel.COMPREHENSIVE.value})\n"
    )
    col_names = [
        f"{'Status':<10}",
        f"{'cuDNN Latency (ms)':>20}",
        f"{'FlagDNN Graph Latency (ms)':>28}",
        f"{'FlagDNN Speedup':>20}",
        f"{'cuDNN GBPS':>20}",
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
