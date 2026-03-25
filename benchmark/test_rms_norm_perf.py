from typing import Generator

import numpy as np
import pytest
import torch

import flag_dnn
from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_rms_norm(x, normalized_shape, eps=1e-5):
    # PyTorch 早期版本未直接提供 F.rms_norm，因此这里使用原生算子组合模拟作为基准
    # 若你的 PyTorch 版本支持，可以直接替换为 torch.nn.functional.rms_norm
    dims = tuple(range(-len(normalized_shape), 0))
    variance = x.pow(2).mean(dim=dims, keepdim=True)
    return x * torch.rsqrt(variance + eps)

def gems_rms_norm_wrapper(x, normalized_shape):
    # 假设 gems 提供了 rms_norm
    return flag_dnn.ops.rms_norm(x, normalized_shape)


class RmsNormBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # 配置格式为: (shape, normalized_shape)
        # RMSNorm 的典型场景与 LayerNorm 高度重合，集中在 LLM
        configs = [
            ((32, 1024, 1024), (1024,)),             # (Batch, SeqLen, Hidden)
            ((16, 4096, 4096), (4096,)),             # LLaMA 典型 shape
            ((8, 8192, 4096), (4096,)),              
            ((1, 32768, 4096), (4096,)),             # 极限长序列推理
            ((128, 197, 768), (768,)),               
        ]
        self.shapes = configs
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        MAX_TENSOR_BYTES = 8 * 1024**3

        for config in self.shapes:
            shape, normalized_shape = config
            numel = np.prod(shape)
            element_size = torch.tensor([], dtype=cur_dtype).element_size()
            tensor_bytes = numel * element_size

            if tensor_bytes > MAX_TENSOR_BYTES:
                continue

            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            if inp.numel() == 0:
                continue
                
            yield inp, normalized_shape

    def get_gbps(self, args, latency):
        inp = args[0]
        io_amount = shape_utils.size_in_bytes(inp) * 2 
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.rms_norm
def test_perf_rms_norm_fp16():
    bench = RmsNormBenchmark(op_name="rms_norm_fp16", torch_op=torch_rms_norm, gems_op=gems_rms_norm_wrapper, dtypes=[torch.float16])
    bench.run()

@pytest.mark.rms_norm
def test_perf_rms_norm_bf16():
    bench = RmsNormBenchmark(op_name="rms_norm_bf16", torch_op=torch_rms_norm, gems_op=gems_rms_norm_wrapper, dtypes=[torch.bfloat16])
    bench.run()

@pytest.mark.rms_norm
def test_perf_rms_norm_fp32():
    bench = RmsNormBenchmark(op_name="rms_norm_fp32", torch_op=torch_rms_norm, gems_op=gems_rms_norm_wrapper, dtypes=[torch.float32])
    bench.run()

@pytest.mark.rms_norm
def test_perf_rms_norm_fp64():
    if not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = RmsNormBenchmark(op_name="rms_norm_fp64", torch_op=torch_rms_norm, gems_op=gems_rms_norm_wrapper, dtypes=[torch.float64])
    bench.run()