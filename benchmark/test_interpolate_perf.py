from typing import Generator

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import flag_dnn

from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_interpolate_nearest(x, out_size):
    return F.interpolate(x, size=out_size, mode="nearest")


def gems_interpolate_nearest_wrapper(x, out_size):
    return flag_dnn.ops.interpolate(x, size=out_size, mode="nearest")


def torch_interpolate_bilinear(x, out_size):
    return F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)


def gems_interpolate_bilinear_wrapper(x, out_size):
    return flag_dnn.ops.interpolate(x, size=out_size, mode="bilinear", align_corners=False)


class InterpolateNearestBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # (input_shape, output_spatial_size)
        configs = [
            ((1, 3, 224, 224), (448, 448)),
            ((8, 3, 128, 128), (256, 256)),
            ((16, 64, 56, 56), (112, 112)),
            ((4, 256, 32, 32), (64, 64)),
            ((32, 3, 32, 32), (16, 16)),  # downscale
        ]
        self.shapes = configs
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        MAX_TENSOR_BYTES = 8 * 1024**3
        for inp_shape, out_size in self.shapes:
            numel = np.prod(inp_shape)
            element_size = torch.tensor([], dtype=cur_dtype).element_size()
            if numel * element_size > MAX_TENSOR_BYTES:
                continue
            x = torch.randn(inp_shape, dtype=cur_dtype, device=self.device)
            if x.numel() == 0:
                continue
            yield x, out_size

    def get_gbps(self, args, latency):
        x, out_size = args[0], args[1]
        out_numel = x.shape[0] * x.shape[1] * out_size[0] * out_size[1]
        io_amount = shape_utils.size_in_bytes(x) + out_numel * x.element_size()
        return io_amount * 1e-9 / (latency * 1e-3)


class InterpolateBilinearBenchmark(InterpolateNearestBenchmark):
    pass


@pytest.mark.interpolate
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32]
)
def test_perf_interpolate_nearest(dtype):
    bench = InterpolateNearestBenchmark(
        op_name="interpolate_nearest",
        torch_op=torch_interpolate_nearest,
        gems_op=gems_interpolate_nearest_wrapper,
        dtypes=[dtype],
    )
    bench.run()


@pytest.mark.interpolate
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32]
)
def test_perf_interpolate_bilinear(dtype):
    bench = InterpolateBilinearBenchmark(
        op_name="interpolate_bilinear",
        torch_op=torch_interpolate_bilinear,
        gems_op=gems_interpolate_bilinear_wrapper,
        dtypes=[dtype],
    )
    bench.run()
