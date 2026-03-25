from typing import Generator

import numpy as np
import pytest
import torch

import flag_dnn
from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_prod(x, dim, keepdim):
    if dim is None:
        return torch.prod(x)
    return torch.prod(x, dim=dim, keepdim=keepdim)

def gems_prod_wrapper(x, dim, keepdim):
    if dim is None:
        return flag_dnn.ops.prod(x)
    return flag_dnn.ops.prod(x, dim=dim, keepdim=keepdim)


class ProdBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # 注意: torch.prod 不支持传入 tuple 作为 dim, 所以移除了 (2, 3) 这类配置
        configs = [
            ((1024 * 1024 * 16,), None, False),       # 1D 超长向量
            ((32, 256, 1024), None, False),           # 3D 全局归约
            ((1024, 1024), 1, False),                 # 典型方阵 Inner
            ((32, 1024, 1024), 2, False),             # NLP Seq_len 维度归约
            ((8, 128, 4096), 2, False),               # NLP 大词表/长序列
            ((1024, 1024), 0, False),                 # 跨步距 Outer 访存
            ((32, 1024, 1024), 0, False),             # Reduce Batch 维度
            ((32, 256, 56, 56), 1, False),            # Reduce 通道维度
            ((1, 16, 2048, 2048), 2, False),          # 空间维度单个方向
            ((64, 512, 512), 2, True),                # keepdim 测试
            ((128, 256, 256), 1, True),
        ]
        self.shapes = configs
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        MAX_TENSOR_BYTES = 8 * 1024**3
        for config in self.shapes:
            shape, dim, keepdim = config
            numel = np.prod(shape)
            element_size = torch.tensor([], dtype=cur_dtype).element_size()
            tensor_bytes = numel * element_size

            if tensor_bytes > MAX_TENSOR_BYTES:
                continue

            # 对于 Prod, 为了防止极端的 Inf 或 0 溢出导致硬件性能偏差，改用正态分布范围限制
            inp = torch.rand(shape, dtype=cur_dtype, device=self.device) * 2.0 - 1.0 
            if inp.numel() == 0:
                continue
                
            yield inp, dim, keepdim

    def get_gbps(self, args, latency):
        inp = args[0]
        dim = args[1]
        
        if dim is None:
            out_numel = 1
        else:
            out_numel = inp.numel() // inp.shape[dim]
                
        io_amount = shape_utils.size_in_bytes(inp) + (out_numel * inp.element_size())
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.prod
def test_perf_prod_fp16():
    bench = ProdBenchmark(op_name="prod_fp16", torch_op=torch_prod, gems_op=gems_prod_wrapper, dtypes=[torch.float16])
    bench.run()

@pytest.mark.prod
def test_perf_prod_bf16():
    bench = ProdBenchmark(op_name="prod_bf16", torch_op=torch_prod, gems_op=gems_prod_wrapper, dtypes=[torch.bfloat16])
    bench.run()

@pytest.mark.prod
def test_perf_prod_fp32():
    bench = ProdBenchmark(op_name="prod_fp32", torch_op=torch_prod, gems_op=gems_prod_wrapper, dtypes=[torch.float32])
    bench.run()

@pytest.mark.prod
def test_perf_prod_fp64():
    if not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = ProdBenchmark(op_name="prod_fp64", torch_op=torch_prod, gems_op=gems_prod_wrapper, dtypes=[torch.float64])
    bench.run()