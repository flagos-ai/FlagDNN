from typing import Generator

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import flag_dnn

from benchmark.performance_utils import Benchmark


def torch_one_hot(x, num_classes):
    return F.one_hot(x, num_classes=num_classes)


def gems_one_hot_wrapper(x, num_classes):
    return flag_dnn.ops.one_hot(x, num_classes=num_classes)


class OneHotBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # (input_shape, num_classes)
        configs = [
            ((1024,), 10),
            ((1024,), 1000),
            ((1024 * 1024,), 100),
            ((32, 512), 10),
            ((8, 2048), 32000),
            ((16, 4096), 128256),
        ]
        self.shapes = configs
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        MAX_TENSOR_BYTES = 8 * 1024**3
        for shape, num_classes in self.shapes:
            out_numel = np.prod(shape) * num_classes
            if out_numel * 8 > MAX_TENSOR_BYTES:
                continue
            x = torch.randint(
                0, num_classes, shape, dtype=torch.long, device=self.device
            )
            if x.numel() == 0:
                continue
            yield x, num_classes

    def get_gbps(self, args, latency):
        x, num_classes = args[0], args[1]
        out_bytes = x.numel() * num_classes * 8  # int64
        io_amount = x.numel() * 8 + out_bytes
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.one_hot
@pytest.mark.parametrize("num_classes", [10, 1000, 32000])
def test_one_hot(num_classes):
    bench = OneHotBenchmark(
        op_name="one_hot",
        torch_op=torch_one_hot,
        gems_op=gems_one_hot_wrapper,
        dtypes=[torch.int64],  # placeholder dtype, not used for int ops
    )
    bench.run()
