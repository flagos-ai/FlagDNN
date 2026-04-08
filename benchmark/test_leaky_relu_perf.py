from typing import Generator

import pytest
import torch
import torch.nn.functional as F

import flag_dnn

from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


# 默认 negative_slope=0.01，直接调用 F.leaky_relu
def torch_leaky_relu(x, y=None):
    return F.leaky_relu(x)


def gems_leaky_relu_wrapper(x, y=None):
    return flag_dnn.ops.leaky_relu(x)


class LeakyReluBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        shapes = [
            (1024,),
            (5333,),
            (65536,),
            (100000,),
            (1048576,),
            (3000000,),
            (4194304,),
            (10000000,),
            (16777216,),
            (33554432,),
            (50000000,),
            (67108864,),
            (134217728,),
        ]
        self.shapes = shapes
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp1 = torch.randn(shape, dtype=cur_dtype, device=self.device)
            if inp1.numel() > 0:
                yield inp1, None

    def get_gbps(self, args, latency):
        inp1 = args[0]
        # Leaky ReLU 是 Element-wise 操作，读取一次输入，写入一次输出
        io_amount = shape_utils.size_in_bytes(
            inp1
        ) + shape_utils.size_in_bytes(inp1)
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.leaky_relu
def test_perf_leaky_relu():
    bench = LeakyReluBenchmark(
        op_name="leaky_relu",
        torch_op=torch_leaky_relu,
        gems_op=gems_leaky_relu_wrapper,
        dtypes=[torch.float16, torch.bfloat16, torch.float32, torch.float64],
    )
    bench.run()
