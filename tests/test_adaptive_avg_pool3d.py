import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from .accuracy_utils import gems_assert_close

# adaptive_avg_pool3d 参数格式：(shape, output_size)
PARAMS = [
    ((2, 3, 8, 16, 16), (4, 8, 8)),        # 标准 3D 降采样
    ((1, 8, 5, 14, 14), 7),                # 输出尺寸为单 int
    ((2, 4, 7, 15, 15), (3, 5, 7)),        # 三个维度不同尺寸
    ((1, 2, 4, 8, 8), (5, 10, 10)),        # 上采样 (输出尺寸大于输入)
    ((4, 5, 10, 20, 20), 1),               # 3D 全局平均池化 (Global Average Pooling)
    ((3, 8, 14, 14), (4, 7, 7)),           # 4D 张量输入 (无 Batch 维度 N)
    ((1, 2, 8, 8, 8), (8, 8, 8)),          # output == input (原样输出)
]

@pytest.mark.adaptive_avg_pool3d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape, output_size", PARAMS)
def test_accuracy_adaptive_avg_pool3d(dtype, shape, output_size):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 使用 randn 生成测试数据
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_out = F.adaptive_avg_pool3d(x, output_size)
    
    out = flag_dnn.ops.adaptive_avg_pool3d(x, output_size)

    # 容差设置：精确对齐 bfloat16 和 float16 的累加特点
    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 2e-3
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 1e-5, 1e-5

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.adaptive_avg_pool3d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_accuracy_adaptive_avg_pool3d_empty_tensor(dtype):
    # D, H, W 至少一个维度的尺寸导致输出 M=0 的情况
    shape = (0, 3, 4, 32, 32)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_out = F.adaptive_avg_pool3d(x, 2)
    out = flag_dnn.ops.adaptive_avg_pool3d(x, 2)

    assert out.shape == ref_out.shape
    assert out.numel() == 0