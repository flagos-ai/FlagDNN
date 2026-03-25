from typing import Generator

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_adaptive_avg_pool1d(x, output_size):
    return F.adaptive_avg_pool1d(x, output_size=output_size)

def gems_adaptive_avg_pool1d_wrapper(x, output_size):
    return flag_dnn.ops.adaptive_avg_pool1d(x, output_size=output_size)


class AdaptiveAvgPool1dBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # 配置格式为: (shape, output_size)
        configs = [
            ((32, 256, 1024), 1),             
            ((8, 512, 4096), 1),              
            ((32, 128, 1024), 32),            
            ((64, 64, 512), 16),              
            ((4, 128, 16000), 100),           
            ((1, 256, 48000), 256),           
        ]
        self.shapes = configs
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        MAX_TENSOR_BYTES = 8 * 1024**3

        for config in self.shapes:
            shape, output_size = config
            numel = np.prod(shape)
            element_size = torch.tensor([], dtype=cur_dtype).element_size()
            tensor_bytes = numel * element_size

            if tensor_bytes > MAX_TENSOR_BYTES:
                continue

            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            if inp.numel() == 0:
                continue
                
            yield inp, output_size

    def get_gbps(self, args, latency):
        inp, output_size = args
        
        # Adaptive Pooling 输出直接由指定的 output_size 决定
        out_numel = inp.shape[0] * inp.shape[1] * output_size
                
        io_amount = shape_utils.size_in_bytes(inp) + (out_numel * inp.element_size())
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.adaptive_avg_pool1d
def test_perf_adaptive_avg_pool1d_fp16():
    bench = AdaptiveAvgPool1dBenchmark(op_name="adaptive_avg_pool1d_fp16", torch_op=torch_adaptive_avg_pool1d, gems_op=gems_adaptive_avg_pool1d_wrapper, dtypes=[torch.float16])
    bench.run()

@pytest.mark.adaptive_avg_pool1d
def test_perf_adaptive_avg_pool1d_bf16():
    bench = AdaptiveAvgPool1dBenchmark(op_name="adaptive_avg_pool1d_bf16", torch_op=torch_adaptive_avg_pool1d, gems_op=gems_adaptive_avg_pool1d_wrapper, dtypes=[torch.bfloat16])
    bench.run()

@pytest.mark.adaptive_avg_pool1d
def test_perf_adaptive_avg_pool1d_fp32():
    bench = AdaptiveAvgPool1dBenchmark(op_name="adaptive_avg_pool1d_fp32", torch_op=torch_adaptive_avg_pool1d, gems_op=gems_adaptive_avg_pool1d_wrapper, dtypes=[torch.float32])
    bench.run()

@pytest.mark.adaptive_avg_pool1d
def test_perf_adaptive_avg_pool1d_fp64():
    if not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AdaptiveAvgPool1dBenchmark(op_name="adaptive_avg_pool1d_fp64", torch_op=torch_adaptive_avg_pool1d, gems_op=gems_adaptive_avg_pool1d_wrapper, dtypes=[torch.float64])
    bench.run()