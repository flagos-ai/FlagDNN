from typing import Generator

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import flag_dnn

from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_kl_div(x, y):
    return F.kl_div(x, y, reduction="batchmean")


def gems_kl_div_wrapper(x, y):
    return flag_dnn.ops.kl_div(x, y, reduction="batchmean")


class KlDivBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        configs = [
            (1024,),
            (32, 1000),
            (8, 32000),
            (16, 128256),
            (32, 256, 512),
        ]
        self.shapes = [(s,) for s in configs]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        MAX_TENSOR_BYTES = 8 * 1024**3
        for (shape,) in self.shapes:
            numel = np.prod(shape)
            element_size = torch.tensor([], dtype=cur_dtype).element_size()
            if numel * element_size * 2 > MAX_TENSOR_BYTES:
                continue
            # log-probabilities for input
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            x = F.log_softmax(x.float(), dim=-1).to(cur_dtype)
            # probabilities for target
            y = torch.rand(shape, dtype=cur_dtype, device=self.device)
            y = (y / y.sum(dim=-1, keepdim=True))
            if x.numel() == 0:
                continue
            yield x, y

    def get_gbps(self, args, latency):
        x = args[0]
        io_amount = shape_utils.size_in_bytes(x) * 2
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.kl_div
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64]
)
def test_perf_kl_div(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = KlDivBenchmark(
        op_name="kl_div",
        torch_op=torch_kl_div,
        gems_op=gems_kl_div_wrapper,
        dtypes=[dtype],
    )
    bench.run()
