import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from .accuracy_utils import gems_assert_close

# avg_pool1d 参数格式：(shape, kernel_size, stride, padding)
PARAMS = [
    ((2, 3, 32), 2, 2, 0),             # 标准 2 降采样
    ((1, 16, 28), 3, 1, 1),            # 保持原图尺寸 (Padding=1)
    ((4, 8, 15), 3, 2, 1),             # 奇数尺寸的步长跨越
    ((2, 4, 32), 5, 1, 0),             # 较大的一维卷积核测试
    ((2, 3, 10), 3, 1, 1),             # 强边缘补零测试
    ((16, 14), 2, 2, 0),               # 2D 张量输入 (无 Batch(N) 维度)
]

@pytest.mark.avg_pool1d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape, kernel_size, stride, padding", PARAMS)
@pytest.mark.parametrize("ceil_mode", [False, True])
@pytest.mark.parametrize("count_include_pad", [False, True])
def test_accuracy_avg_pool1d(dtype, shape, kernel_size, stride, padding, ceil_mode, count_include_pad):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 使用 randn 生成测试数据
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_out = F.avg_pool1d(
        x, kernel_size, stride=stride, padding=padding,
        ceil_mode=ceil_mode, count_include_pad=count_include_pad
    )
    
    out = flag_dnn.ops.avg_pool1d(
        x, kernel_size, stride=stride, padding=padding,
        ceil_mode=ceil_mode, count_include_pad=count_include_pad
    )

    # 容差设置：Average Pooling 存在浮点累加和除法
    # 对于 fp16/bf16 放宽一点，fp32/fp64 可以保持 1e-5
    if dtype in [torch.float16, torch.bfloat16]:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 1e-5, 1e-5

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.avg_pool1d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_accuracy_avg_pool1d_empty_tensor(dtype):
    shape = (0, 3, 32)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_out = F.avg_pool1d(x, 2, 2)
    out = flag_dnn.ops.avg_pool1d(x, 2, 2)

    assert out.shape == ref_out.shape
    assert out.numel() == 0