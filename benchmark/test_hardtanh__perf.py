from math import prod
from typing import Generator

import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from benchmark.performance_utils import Benchmark


def torch_hardtanh_(x, min_val, max_val):
    return F.hardtanh_(x, min_val=min_val, max_val=max_val)


def gems_hardtanh__wrapper(x, min_val, max_val):
    return flag_dnn.ops.hardtanh_(x, min_val, max_val)


class Hardtanh_Benchmark(Benchmark):
    IO_FACTOR = 2
    MAX_PEAK_BYTES = 6 * 1024**3

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        self.shapes = [
            # 1D
            (1024,),
            (4096,),
            (16384,),
            (65536,),
            (262144,),
            (1048576,),
            # 2D
            (256, 256),
            (512, 512),
            (1024, 1024),
            (2048, 2048),
            # 4D
            (1, 3, 224, 224),
            (8, 3, 224, 224),
            (32, 3, 224, 224),
            (64, 3, 224, 224),
            (1, 64, 112, 112),
            (8, 64, 112, 112),
            (32, 64, 112, 112),
            (1, 128, 56, 56),
            (8, 128, 56, 56),
            (32, 128, 56, 56),
        ]
        return None

    @staticmethod
    def _tensor_nbytes(shape, dtype):
        return prod(shape) * torch.empty((), dtype=dtype).element_size()

    def _estimate_peak_bytes(self, shape, dtype):
        input_bytes = self._tensor_nbytes(shape, dtype)
        return input_bytes * 2

    def get_input_iter(self, cur_dtype) -> Generator:
        min_val = -1.0
        max_val = 1.0

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
            ).uniform_(-2.0, 2.0)
            yield x, min_val, max_val

    def get_gbps(self, args, latency):
        x = args[0]
        io_amount = x.numel() * x.element_size() * self.IO_FACTOR
        return io_amount / (latency * 1e-3) / 1e9


@pytest.mark.hardtanh_
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_hardtanh_(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = Hardtanh_Benchmark(
        op_name="hardtanh_",
        torch_op=torch_hardtanh_,
        gems_op=gems_hardtanh__wrapper,
        dtypes=[dtype],
    )
    bench.run()
