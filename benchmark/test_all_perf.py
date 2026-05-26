from typing import Generator

import numpy as np
import pytest
import torch

import flag_dnn

from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_all(x, dim, keepdim):
    if dim is None:
        return torch.all(x)
    return torch.all(x, dim=dim, keepdim=keepdim)


def gems_all_wrapper(x, dim, keepdim):
    if dim is None:
        return flag_dnn.ops.all(x)
    return flag_dnn.ops.all(x, dim=dim, keepdim=keepdim)


class AllBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        configs = [
            ((1024 * 1024 * 16,), None, False),
            ((32, 256, 1024), None, False),
            ((1024, 1024), 1, False),
            ((32, 1024, 1024), 2, False),
            ((8, 128, 4096), 2, False),
            ((1024, 1024), 0, False),
        ]
        self.shapes = configs
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        MAX_TENSOR_BYTES = 8 * 1024**3
        for config in self.shapes:
            shape, dim, keepdim = config
            numel = np.prod(shape)
            element_size = torch.tensor([], dtype=cur_dtype).element_size()
            if numel * element_size > MAX_TENSOR_BYTES:
                continue
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            if inp.numel() == 0:
                continue
            yield inp, dim, keepdim

    def get_gbps(self, args, latency):
        inp = args[0]
        io_amount = shape_utils.size_in_bytes(inp) + inp.element_size()
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.all
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
def test_perf_all(dtype):
    bench = AllBenchmark(
        op_name="all",
        torch_op=torch_all,
        gems_op=gems_all_wrapper,
        dtypes=[dtype],
    )
    bench.run()
