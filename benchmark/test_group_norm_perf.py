from typing import Generator

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_group_norm(x, num_groups):
    return F.group_norm(x, num_groups)

def gems_group_norm_wrapper(x, num_groups):
    return flag_dnn.ops.group_norm(x, num_groups)


class GroupNormBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # 配置格式为: (shape, num_groups)
        # 必须保证 shape[1] (Channels) % num_groups == 0
        configs = [
            # 1. 典型 2D 卷积/视觉模型中的特征图 (Batch, Channels, H, W)
            ((32, 256, 56, 56), 32),                 # 常见配置: 32个组，每组 8 个通道
            ((16, 512, 28, 28), 32),                 
            ((8, 1024, 14, 14), 32),                 
            
            # 2. Diffusion Models (如 SD) 中的典型 UNet 分辨率
            ((4, 320, 64, 64), 32),                  
            ((2, 1280, 16, 16), 32),                 
            
            # 3. 1D 时序/音频特征 (Batch, Channels, Length)
            ((32, 128, 1024), 8),                    
            ((16, 256, 4096), 16),                   
        ]
        self.shapes = configs
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        MAX_TENSOR_BYTES = 8 * 1024**3

        for config in self.shapes:
            shape, num_groups = config
            numel = np.prod(shape)
            element_size = torch.tensor([], dtype=cur_dtype).element_size()
            tensor_bytes = numel * element_size

            if tensor_bytes > MAX_TENSOR_BYTES:
                continue

            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            if inp.numel() == 0:
                continue
                
            yield inp, num_groups

    def get_gbps(self, args, latency):
        inp = args[0]
        io_amount = shape_utils.size_in_bytes(inp) * 2 
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.group_norm
def test_perf_group_norm_fp16():
    bench = GroupNormBenchmark(op_name="group_norm_fp16", torch_op=torch_group_norm, gems_op=gems_group_norm_wrapper, dtypes=[torch.float16])
    bench.run()

@pytest.mark.group_norm
def test_perf_group_norm_bf16():
    bench = GroupNormBenchmark(op_name="group_norm_bf16", torch_op=torch_group_norm, gems_op=gems_group_norm_wrapper, dtypes=[torch.bfloat16])
    bench.run()

@pytest.mark.group_norm
def test_perf_group_norm_fp32():
    bench = GroupNormBenchmark(op_name="group_norm_fp32", torch_op=torch_group_norm, gems_op=gems_group_norm_wrapper, dtypes=[torch.float32])
    bench.run()

@pytest.mark.group_norm
def test_perf_group_norm_fp64():
    if not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = GroupNormBenchmark(op_name="group_norm_fp64", torch_op=torch_group_norm, gems_op=gems_group_norm_wrapper, dtypes=[torch.float64])
    bench.run()