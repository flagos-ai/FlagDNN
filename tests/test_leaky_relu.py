import pytest
import torch
import torch.nn.functional as F

import flag_dnn

from .accuracy_utils import gems_assert_close


SHAPES = [(32,), (1024,), (5333,), (16384,), (1024 * 1024,)]
NEGATIVE_SLOPES = [0.01, 0.2]  # 测试默认斜率和较大的斜率


@pytest.mark.leaky_relu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("negative_slope", NEGATIVE_SLOPES)
def test_accuracy_leaky_relu(dtype, shape, inplace, negative_slope):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    # 针对不同数据类型动态设置容差 (Tolerance)
    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2  # BF16 精度极低，需要较宽松的容差
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3    # FP16 中等宽松
    else:
        rtol, atol = 1e-5, 1e-5    # FP32 和 FP64 保持严格
    
    # 必须 clone，防止 inplace=True 时原生算子破坏输入数据
    ref_x = x.clone() 
    ref_y = F.leaky_relu(ref_x, negative_slope=negative_slope, inplace=inplace)

    y = flag_dnn.ops.leaky_relu(x, negative_slope=negative_slope, inplace=inplace)

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.leaky_relu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("negative_slope", NEGATIVE_SLOPES)
def test_accuracy_leaky_relu_empty_tensor(dtype, inplace, negative_slope):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 测试空张量 (shape 为 0)
    x = torch.randn(0, dtype=dtype, device=flag_dnn.device)

    # 针对不同数据类型动态设置容差 (Tolerance)
    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2  # BF16 精度极低，需要较宽松的容差
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3    # FP16 中等宽松
    else:
        rtol, atol = 1e-5, 1e-5    # FP32 和 FP64 保持严格
    
    ref_x = x.clone()
    ref_y = F.leaky_relu(ref_x, negative_slope=negative_slope, inplace=inplace)
    
    y = flag_dnn.ops.leaky_relu(x, negative_slope=negative_slope, inplace=inplace)

    assert y.shape == (0,)
    assert y.dtype == dtype
    assert y.device == x.device
    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.leaky_relu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("negative_slope", NEGATIVE_SLOPES)
def test_accuracy_leaky_relu_negative_values(dtype, inplace, negative_slope):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 偏移使其绝大多数为负数，专注测试 negative_slope 逻辑
    x = torch.randn(100, dtype=dtype, device=flag_dnn.device) - 2.0

    # 针对不同数据类型动态设置容差 (Tolerance)
    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2  # BF16 精度极低，需要较宽松的容差
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3    # FP16 中等宽松
    else:
        rtol, atol = 1e-5, 1e-5    # FP32 和 FP64 保持严格
    
    ref_x = x.clone()
    ref_y = F.leaky_relu(ref_x, negative_slope=negative_slope, inplace=inplace)
    
    y = flag_dnn.ops.leaky_relu(x, negative_slope=negative_slope, inplace=inplace)

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.leaky_relu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("negative_slope", NEGATIVE_SLOPES)
def test_accuracy_leaky_relu_positive_values(dtype, inplace, negative_slope):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 偏移使其绝大多数为正数，专注测试恒等映射逻辑
    x = torch.randn(100, dtype=dtype, device=flag_dnn.device) + 2.0

    # 针对不同数据类型动态设置容差 (Tolerance)
    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2  # BF16 精度极低，需要较宽松的容差
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3    # FP16 中等宽松
    else:
        rtol, atol = 1e-5, 1e-5    # FP32 和 FP64 保持严格
    
    ref_x = x.clone()
    ref_y = F.leaky_relu(ref_x, negative_slope=negative_slope, inplace=inplace)
    
    y = flag_dnn.ops.leaky_relu(x, negative_slope=negative_slope, inplace=inplace)

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.leaky_relu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("negative_slope", NEGATIVE_SLOPES)
def test_accuracy_leaky_relu_mixed_values(dtype, inplace, negative_slope):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 混合正负数
    x = torch.randn(100, dtype=dtype, device=flag_dnn.device)

    # 针对不同数据类型动态设置容差 (Tolerance)
    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2  # BF16 精度极低，需要较宽松的容差
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3    # FP16 中等宽松
    else:
        rtol, atol = 1e-5, 1e-5    # FP32 和 FP64 保持严格
    
    ref_x = x.clone()
    ref_y = F.leaky_relu(ref_x, negative_slope=negative_slope, inplace=inplace)
    
    y = flag_dnn.ops.leaky_relu(x, negative_slope=negative_slope, inplace=inplace)

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)