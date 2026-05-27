from typing import Generator

import numpy as np
import pytest
import torch

import flag_dnn

from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_log(x):
    return torch.log(x)


def gems_log_wrapper(x):
    return flag_dnn.ops.log(x)


class LogBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        configs = [
            (1,),
            (16,),
            (64,),
            (127,),
            (1023, 1025),
            (7, 31, 109),
            (33, 129, 257),
            (1, 2048, 4096),
            (8, 128, 12288),
            (4, 4096, 4096),
            (1, 3, 224, 224),
            (32, 256, 56, 56),
            (16, 1024, 14, 14),
            (2, 16, 32, 64, 64),
            (1024 * 256,),
            (1024 * 1024 * 16,),
            (8192, 8192),
            (1024 * 1024 * 64,),
            (2, 8192, 8192),
        ]
        self.shapes = [(shape,) for shape in configs]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        MAX_TENSOR_BYTES = 8 * 1024**3

        for (shape_x,) in self.shapes:
            element_size = torch.tensor([], dtype=cur_dtype).element_size()
            total_bytes = np.prod(shape_x) * element_size * 2

            if total_bytes > MAX_TENSOR_BYTES:
                continue

            x = (
                torch.abs(
                    torch.randn(shape_x, dtype=cur_dtype, device=self.device)
                )
                + 0.1
            )
            if x.numel() == 0:
                continue
            yield (x,)

    def get_gbps(self, args, latency):
        x = args[0]
        io_amount = shape_utils.size_in_bytes(x) * 2
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.log
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_log(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = LogBenchmark(
        op_name="log",
        torch_op=torch_log,
        gems_op=gems_log_wrapper,
        dtypes=[dtype],
    )
    bench.run()
