from typing import Generator

import numpy as np
import pytest
import torch

import flag_dnn

from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_bitwise_xor(x, y):
    return torch.bitwise_xor(x, y)


def gems_bitwise_xor_wrapper(x, y):
    return flag_dnn.ops.bitwise_xor(x, y)


class Bitwise_xorBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        configs = [
            (1,),
            (16,),
            (64,),
            (127,),
            (1023, 1025),
            (7, 31, 109),
            (33, 129, 257),
            (1, 2048, 4096),
            (8, 128, 12288),
            (4, 4096, 4096),
            (1, 3, 224, 224),
            (32, 256, 56, 56),
            (16, 1024, 14, 14),
            (2, 16, 32, 64, 64),
            (1024 * 256,),
            (1024 * 1024 * 16,),
            (8192, 8192),
            (1024 * 1024 * 64,),
            (2, 8192, 8192),
        ]
        self.shapes = [(shape, shape) for shape in configs]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        MAX_TENSOR_BYTES = 8 * 1024**3

        for shape_x, shape_y in self.shapes:
            element_size = torch.tensor([], dtype=cur_dtype).element_size()
            total_bytes = (np.prod(shape_x) + np.prod(shape_y) + np.prod(shape_x)) * element_size

            if total_bytes > MAX_TENSOR_BYTES:
                continue

            x = torch.randint(-100, 100, shape_x, dtype=cur_dtype, device=self.device)
            y = torch.randint(-100, 100, shape_y, dtype=cur_dtype, device=self.device)
            if x.numel() == 0:
                continue
            yield x, y

    def get_gbps(self, args, latency):
        x = args[0]
        y = args[1]
        io_amount = shape_utils.size_in_bytes(x) + shape_utils.size_in_bytes(y) + shape_utils.size_in_bytes(x)
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.bitwise_xor
@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64]
)
def test_perf_bitwise_xor(dtype):
    bench = Bitwise_xorBenchmark(
        op_name="bitwise_xor",
        torch_op=torch_bitwise_xor,
        gems_op=gems_bitwise_xor_wrapper,
        dtypes=[dtype],
    )
    bench.run()
