from typing import Generator

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import flag_dnn

from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_log_softmax(x, y=None):
    return F.log_softmax(x, dim=-1)


def gems_log_softmax_wrapper(x, y=None):
    return flag_dnn.ops.log_softmax(x, dim=-1)


class LogSoftmaxBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        shapes = [
            (1024,),
            (65536,),
            (256, 1000),
            (1024, 100),
            (32, 32000),
            (16, 128256),
            (16, 12, 1024, 1024),
            (8, 32, 2048, 2048),
        ]
        self.shapes = shapes
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        MAX_TENSOR_BYTES = 8 * 1024**3
        for shape in self.shapes:
            numel = np.prod(shape)
            element_size = torch.tensor([], dtype=cur_dtype).element_size()
            if numel * element_size > MAX_TENSOR_BYTES:
                continue
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            if inp.numel() == 0:
                continue
            yield inp, None

    def get_gbps(self, args, latency):
        inp = args[0]
        io_amount = shape_utils.size_in_bytes(inp) * 2
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.log_softmax
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_log_softmax(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = LogSoftmaxBenchmark(
        op_name="log_softmax",
        torch_op=torch_log_softmax,
        gems_op=gems_log_softmax_wrapper,
        dtypes=[dtype],
    )
    bench.run()
