from math import prod
from typing import Generator

import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from benchmark.performance_utils import Benchmark


def torch_threshold(x, threshold_val, value_val):
    return F.threshold(x, threshold_val, value_val)


def gems_threshold_wrapper(x, threshold_val, value_val):
    return flag_dnn.ops.threshold(x, threshold_val, value_val)


class ThresholdBenchmark(Benchmark):
    IO_FACTOR = 2
    MAX_PEAK_BYTES = 6 * 1024**3

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        self.shapes = [
            (4096,),  # 小向量
            (65536,),  # 中等向量
            (1048576,),  # 大向量
            (4096, 4096),  # 大 2D
            (32, 128, 768),  # Transformer: B x S x H
            (32, 512, 768),  # Transformer 长序列
            (32, 128, 1024),  # 更大 hidden size
            (32, 64, 56, 56),  # CV: ResNet 中间层激活
            (32, 256, 28, 28),  # CV: 更深层特征图
        ]
        return None

    @staticmethod
    def _tensor_nbytes(shape, dtype):
        return prod(shape) * torch.empty((), dtype=dtype).element_size()

    def _estimate_peak_bytes(self, shape, dtype):
        input_bytes = self._tensor_nbytes(shape, dtype)
        return input_bytes * 2

    def get_input_iter(self, cur_dtype) -> Generator:
        threshold_val = 0.0
        value_val = 0.0

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
            ).uniform_(-1.0, 1.0)
            yield x, threshold_val, value_val

    def get_gbps(self, args, latency):
        x = args[0]
        io_amount = x.numel() * x.element_size() * self.IO_FACTOR
        return io_amount / (latency * 1e-3) / 1e9


@pytest.mark.threshold
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_threshold(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = ThresholdBenchmark(
        op_name="threshold",
        torch_op=torch_threshold,
        gems_op=gems_threshold_wrapper,
        dtypes=[dtype],
    )
    bench.run()
