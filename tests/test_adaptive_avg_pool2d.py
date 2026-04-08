import pytest
import torch
import torch.nn.functional as F
import flag_dnn


# (shape, output_size)
PARAMS = [
    ((2, 3, 32, 32), (1, 1)),                 # 全局平均池化 (Global Average Pooling)
    ((1, 16, 28, 28), 14),                    # 降维到 14x14 (输入单整数)
    ((4, 8, 15, 15), (7, 5)),                 # 非对称目标尺寸
    ((2, 4, 32, 32), (None, 16)),             # 保持 H 尺寸不变，W 降到 16
    ((16, 14, 14), (2, 2)),                   # 3D 张量输入
]


@pytest.mark.adaptive_avg_pool2d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape, output_size", PARAMS)
def test_accuracy_adaptive_avg_pool2d(dtype, shape, output_size):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    # 容差设置
    if dtype == torch.bfloat16:
        rtol, atol = 1e-2, 1e-2
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 1e-5, 1e-5

    ref_y = F.adaptive_avg_pool2d(x, output_size)
    with flag_dnn.use_dnn():
        y = F.adaptive_avg_pool2d(x, output_size)

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.adaptive_avg_pool2d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_accuracy_adaptive_avg_pool2d_empty_tensor(dtype):
    shape = (0, 3, 32, 32)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_y = F.adaptive_avg_pool2d(x, (2, 2))
    with flag_dnn.use_dnn():
        y = F.adaptive_avg_pool2d(x, (2, 2))

    assert y.shape == ref_y.shape
    assert y.numel() == 0


@pytest.mark.adaptive_avg_pool2d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_accuracy_adaptive_avg_pool2d_large_values(dtype):
    shape = (2, 3, 32, 32)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 1000.0

    if dtype == torch.bfloat16:
        rtol, atol = 1e-2, 1e-1
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-2
    else:
        rtol, atol = 1e-5, 1e-4

    ref_y = F.adaptive_avg_pool2d(x, (2, 2))
    with flag_dnn.use_dnn():
        y = F.adaptive_avg_pool2d(x, (2, 2))

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.adaptive_avg_pool2d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_accuracy_adaptive_avg_pool2d_mixed_values(dtype):
    shape = (2, 3, 32, 32)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    
    x[..., ::2, ::2] *= 1000.0
    x[..., 1::2, 1::2] *= 0.001

    if dtype == torch.bfloat16:
        rtol, atol = 1e-2, 1e-1
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-2
    else:
        rtol, atol = 1e-5, 1e-4

    ref_y = F.adaptive_avg_pool2d(x, (3, 3))
    with flag_dnn.use_dnn():
        y = F.adaptive_avg_pool2d(x, (3, 3))

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)