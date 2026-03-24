import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from .accuracy_utils import gems_assert_close

# adaptive_avg_pool1d 参数格式：(shape, output_size)
PARAMS = [
    ((2, 3, 32), 16),              # 标准降采样
    ((1, 8, 14), 14),              # output == input (原样输出)
    ((2, 4, 15), 7),               # 不规则的奇数下采样
    ((1, 2, 8), 12),               # 上采样 (输出尺寸大于输入，PyTorch 是支持的)
    ((4, 5, 20), 1),               # 全局平均池化 (Global Average Pooling)
    ((16, 14), 5),                 # 2D 张量输入 (无 Batch(N) 维度)
]

@pytest.mark.adaptive_avg_pool1d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape, output_size", PARAMS)
def test_accuracy_adaptive_avg_pool1d(dtype, shape, output_size):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 使用 randn 生成测试数据
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_out = F.adaptive_avg_pool1d(x, output_size)
 
    out = flag_dnn.ops.adaptive_avg_pool1d(x, output_size)

    # 容差设置：精确对齐 bfloat16 和 float16 的累加特点
    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 2e-3
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 1e-5, 1e-5

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.adaptive_avg_pool1d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_accuracy_adaptive_avg_pool1d_empty_tensor(dtype):
    shape = (0, 3, 32)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_out = F.adaptive_avg_pool1d(x, 2)
    out = flag_dnn.ops.adaptive_avg_pool1d(x, 2)

    assert out.shape == ref_out.shape
    assert out.numel() == 0