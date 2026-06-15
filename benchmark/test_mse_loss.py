from typing import Generator

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import flag_dnn

from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_mse_loss(x, y):
    return F.mse_loss(x, y, reduction="mean")


def gems_mse_loss_wrapper(x, y):
    return flag_dnn.ops.mse_loss(x, y, reduction="mean")


class MseLossBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        configs = [
            (1024,),
            (65536,),
            (1024 * 1024,),
            (32, 1000),
            (8, 4096),
            (32, 256, 1024),
            (16, 64, 224, 224),
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
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            y = torch.randn(shape, dtype=cur_dtype, device=self.device)
            if x.numel() == 0:
                continue
            yield x, y

    def get_gbps(self, args, latency):
        x = args[0]
        io_amount = shape_utils.size_in_bytes(x) * 2
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.mse_loss
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_mse_loss(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = MseLossBenchmark(
        op_name="mse_loss",
        torch_op=torch_mse_loss,
        gems_op=gems_mse_loss_wrapper,
        dtypes=[dtype],
    )
    bench.run()
