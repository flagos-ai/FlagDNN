from typing import Generator

import pytest
import torch
import torch.nn.functional as F

import flag_dnn

from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


# 默认 negative_slope=0.01，直接调用 F.leaky_relu
def torch_leaky_relu_(x, y=None):
    return F.leaky_relu_(x)


def gems_leaky_relu__wrapper(x, y=None):
    return flag_dnn.ops.leaky_relu_(x)


class LeakyReluBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        shapes = [
            (1024,),  # 小 1D
            (65536,),  # 中等 1D
            (1048576,),  # 大 1D
            (1024, 1024),  # 典型 2D
            (4096, 4096),  # 大 2D
            (32, 128, 768),  # Transformer 常见 3D
            (32, 512, 768),  # 更长序列
            (32, 64, 56, 56),  # CV 4D
            (32, 256, 28, 28),  # 更深层特征图
            (5333,),  # 非对齐 1D
            (17, 31),  # 非对齐 2D
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
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_leaky_relu(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = LeakyReluBenchmark(
        op_name="leaky_relu_",
        torch_op=torch_leaky_relu_,
        gems_op=gems_leaky_relu__wrapper,
        dtypes=[dtype],
    )
    bench.run()
