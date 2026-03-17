import pytest
import torch
import torch.nn.functional as F

import flag_dnn

from .accuracy_utils import gems_assert_close

# 专门为 PReLU 扩展了多维 Shape，以测试通道维度 (dim=1)
SHAPES = [(32,), (1024,), (2, 16), (4, 8, 32), (2, 4, 16, 16)]
MODES = ['single', 'channel']

@pytest.mark.prelu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("mode", MODES)
def test_accuracy_prelu(dtype, shape, mode):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 如果是逐通道模式，但维度不够（只有 1 维），则跳过当前测试组合
    if mode == 'channel' and len(shape) < 2:
        pytest.skip("Channel mode requires at least 2 dimensions")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    # 针对不同数据类型动态设置容差 (Tolerance)
    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2  # BF16 精度极低，需要较宽松的容差
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3    # FP16 中等宽松
    else:
        rtol, atol = 1e-5, 1e-5    # FP32 和 FP64 保持严格
    
    # 根据模式初始化 weight 参数
    if mode == 'single':
        num_parameters = 1
    else:
        num_parameters = shape[1] # PyTorch 约定 dim=1 是通道维度
        
    weight = torch.full((num_parameters,), 0.25, dtype=dtype, device=flag_dnn.device)
    
    ref_y = F.prelu(x, weight)
    y = flag_dnn.ops.prelu(x, weight)

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.prelu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", MODES)
def test_accuracy_prelu_empty_tensor(dtype, mode):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 测试空张量，为了能测试 channel 模式，给一个含 0 的多维 shape
    shape = (0, 4, 16) if mode == 'channel' else (0,)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    # 针对不同数据类型动态设置容差 (Tolerance)
    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2  # BF16 精度极低，需要较宽松的容差
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3    # FP16 中等宽松
    else:
        rtol, atol = 1e-5, 1e-5    # FP32 和 FP64 保持严格
    
    num_parameters = shape[1] if mode == 'channel' else 1
    weight = torch.full((num_parameters,), 0.25, dtype=dtype, device=flag_dnn.device)
    
    ref_y = F.prelu(x, weight)
    y = flag_dnn.ops.prelu(x, weight)

    assert y.shape == shape
    assert y.dtype == dtype
    assert y.device == x.device
    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.prelu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", MODES)
def test_accuracy_prelu_negative_values(dtype, mode):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    shape = (4, 8, 16) # 固定一个多维形状方便测试
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) - 2.0

    # 针对不同数据类型动态设置容差 (Tolerance)
    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2  # BF16 精度极低，需要较宽松的容差
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3    # FP16 中等宽松
    else:
        rtol, atol = 1e-5, 1e-5    # FP32 和 FP64 保持严格
    
    num_parameters = shape[1] if mode == 'channel' else 1
    # 随机生成 weight 而不是全 0.25，更能测出计算的准确性
    weight = torch.randn(num_parameters, dtype=dtype, device=flag_dnn.device) * 0.1
    
    ref_y = F.prelu(x, weight)
    y = flag_dnn.ops.prelu(x, weight)

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.prelu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", MODES)
def test_accuracy_prelu_positive_values(dtype, mode):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    shape = (4, 8, 16)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) + 2.0

    # 针对不同数据类型动态设置容差 (Tolerance)
    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2  # BF16 精度极低，需要较宽松的容差
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3    # FP16 中等宽松
    else:
        rtol, atol = 1e-5, 1e-5    # FP32 和 FP64 保持严格
    
    num_parameters = shape[1] if mode == 'channel' else 1
    weight = torch.randn(num_parameters, dtype=dtype, device=flag_dnn.device) * 0.1
    
    ref_y = F.prelu(x, weight)
    y = flag_dnn.ops.prelu(x, weight)

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.prelu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", MODES)
def test_accuracy_prelu_mixed_values(dtype, mode):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    shape = (4, 8, 16)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    # 针对不同数据类型动态设置容差 (Tolerance)
    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2  # BF16 精度极低，需要较宽松的容差
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3    # FP16 中等宽松
    else:
        rtol, atol = 1e-5, 1e-5    # FP32 和 FP64 保持严格
    
    num_parameters = shape[1] if mode == 'channel' else 1
    weight = torch.randn(num_parameters, dtype=dtype, device=flag_dnn.device) * 0.1
    
    ref_y = F.prelu(x, weight)
    y = flag_dnn.ops.prelu(x, weight)

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)