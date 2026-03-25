from typing import Generator

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_layer_norm(x, normalized_shape):
    return F.layer_norm(x, normalized_shape)

def gems_layer_norm_wrapper(x, normalized_shape):
    return flag_dnn.ops.layer_norm(x, normalized_shape)


class LayerNormBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # 配置格式为: (shape, normalized_shape)
        # normalized_shape 必须匹配输入张量最后几个维度
        configs = [
            # 1. 典型 NLP/LLM 模型中的 Hidden Size 归一化
            ((32, 1024, 1024), (1024,)),             # (Batch, SeqLen, Hidden)
            ((16, 4096, 4096), (4096,)),             # 大模型长序列
            ((8, 8192, 4096), (4096,)),              # 极长上下文
            
            # 2. 视觉任务 (CV) 中的 LayerNorm (如 Vision Transformer)
            ((128, 197, 768), (768,)),               # ViT Base
            ((64, 1370, 1024), (1024,)),             # ViT Large
            
            # 3. 跨多维度归一化
            ((32, 256, 56, 56), (256, 56, 56)),      # InstanceNorm 的等价变体
            ((32, 256, 56, 56), (56, 56)),           # 对空间维度归一化
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
        # LayerNorm 读取一次输入 x，写出一次输出 y (暂时忽略极小的 weight/bias)
        io_amount = shape_utils.size_in_bytes(inp) * 2 
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.layer_norm
def test_perf_layer_norm_fp16():
    bench = LayerNormBenchmark(op_name="layer_norm_fp16", torch_op=torch_layer_norm, gems_op=gems_layer_norm_wrapper, dtypes=[torch.float16])
    bench.run()

@pytest.mark.layer_norm
def test_perf_layer_norm_bf16():
    bench = LayerNormBenchmark(op_name="layer_norm_bf16", torch_op=torch_layer_norm, gems_op=gems_layer_norm_wrapper, dtypes=[torch.bfloat16])
    bench.run()

@pytest.mark.layer_norm
def test_perf_layer_norm_fp32():
    bench = LayerNormBenchmark(op_name="layer_norm_fp32", torch_op=torch_layer_norm, gems_op=gems_layer_norm_wrapper, dtypes=[torch.float32])
    bench.run()

@pytest.mark.layer_norm
def test_perf_layer_norm_fp64():
    if not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = LayerNormBenchmark(op_name="layer_norm_fp64", torch_op=torch_layer_norm, gems_op=gems_layer_norm_wrapper, dtypes=[torch.float64])
    bench.run()