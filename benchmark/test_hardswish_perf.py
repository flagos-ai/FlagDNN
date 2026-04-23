from math import prod
from typing import Generator

import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from benchmark.performance_utils import Benchmark


def torch_hardswish(x):
    return F.hardswish(x)


def gems_hardswish_wrapper(x):
    return flag_dnn.ops.hardswish(x)


class HardSwishBenchmark(Benchmark):
    IO_FACTOR = 2
    MAX_PEAK_BYTES = 6 * 1024**3

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        self.shapes = [
            # 1D: 纯 elementwise 吞吐测试
            (1024,),
            (4096,),
            # 2D: 常规矩阵形状
            (32, 128),
            (128, 512),
            (512, 1024),
            (1024, 4096),
            # 4D: 更贴近 hardswish 常见 CNN / MobileNet 使用场景
            (1, 16, 112, 112),
            (1, 16, 56, 56),
            # batched feature maps
            (8, 16, 112, 112),
            (8, 24, 56, 56),
            (8, 80, 14, 14),
            (8, 160, 7, 7),
        ]
        return None

    @staticmethod
    def _tensor_nbytes(shape, dtype):
        return prod(shape) * torch.empty((), dtype=dtype).element_size()

    def _estimate_peak_bytes(self, shape, dtype):
        input_bytes = self._tensor_nbytes(shape, dtype)
        return input_bytes * 2

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            if (
                self._estimate_peak_bytes(shape, cur_dtype)
                > self.MAX_PEAK_BYTES
            ):
                continue

            numel = prod(shape)
            if numel == 0:
                continue

            x = torch.empty(
                shape, dtype=cur_dtype, device=self.device
            ).uniform_(-5.0, 5.0)
            yield (x,)

    def get_gbps(self, args, latency):
        x = args[0]
        io_amount = x.numel() * x.element_size() * self.IO_FACTOR
        return io_amount / (latency * 1e-3) / 1e9


@pytest.mark.hardswish
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_hardswish(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = HardSwishBenchmark(
        op_name="hardswish",
        torch_op=torch_hardswish,
        gems_op=gems_hardswish_wrapper,
        dtypes=[dtype],
    )
    bench.run()
