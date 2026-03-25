from typing import Generator

import numpy as np
import pytest
import torch

import flag_dnn
from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_cumprod(x, dim):
    return torch.cumprod(x, dim=dim)

def gems_cumprod_wrapper(x, dim):
    return flag_dnn.ops.cumprod(x, dim=dim)


class CumprodBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # cumprod 必须指定单个 dim，输出 shape 永远与输入一致
        configs = [
            ((1024 * 1024 * 16,), 0),                 # 1D 超长向量 Scan
            ((32, 256, 1024), 2),                     # Inner Dim (Row) Scan, 易合并
            ((32, 256, 1024), 0),                     # Outer Dim (Column) Scan, 最难合并
            ((32, 256, 1024), 1),                     # Middle Dim Scan
            ((1024, 1024), 1),                        # 方阵 Inner
            ((1024, 1024), 0),                        # 方阵 Outer
            ((32, 256, 56, 56), 3),                   # CV 典型 Inner
            ((32, 256, 56, 56), 1),                   # CV 典型 Channel Scan
        ]
        self.shapes = configs
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        MAX_TENSOR_BYTES = 8 * 1024**3
        for config in self.shapes:
            shape, dim = config
            numel = np.prod(shape)
            element_size = torch.tensor([], dtype=cur_dtype).element_size()
            tensor_bytes = numel * element_size

            if tensor_bytes > MAX_TENSOR_BYTES:
                continue

            # 使用 rand() * 0.5 + 0.75 使得生成的值在 0.75~1.25 之间，避免极端的 Inf 和 0 
            inp = torch.rand(shape, dtype=cur_dtype, device=self.device) * 0.5 + 0.75
            if inp.numel() == 0:
                continue
                
            yield inp, dim

    def get_gbps(self, args, latency):
        inp = args[0]
        out_numel = inp.numel() # cumprod 输出与输入相等
                
        io_amount = shape_utils.size_in_bytes(inp) + (out_numel * inp.element_size())
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.cumprod
def test_perf_cumprod_fp16():
    bench = CumprodBenchmark(op_name="cumprod_fp16", torch_op=torch_cumprod, gems_op=gems_cumprod_wrapper, dtypes=[torch.float16])
    bench.run()

@pytest.mark.cumprod
def test_perf_cumprod_bf16():
    bench = CumprodBenchmark(op_name="cumprod_bf16", torch_op=torch_cumprod, gems_op=gems_cumprod_wrapper, dtypes=[torch.bfloat16])
    bench.run()

@pytest.mark.cumprod
def test_perf_cumprod_fp32():
    bench = CumprodBenchmark(op_name="cumprod_fp32", torch_op=torch_cumprod, gems_op=gems_cumprod_wrapper, dtypes=[torch.float32])
    bench.run()

@pytest.mark.cumprod
def test_perf_cumprod_fp64():
    if not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = CumprodBenchmark(op_name="cumprod_fp64", torch_op=torch_cumprod, gems_op=gems_cumprod_wrapper, dtypes=[torch.float64])
    bench.run()