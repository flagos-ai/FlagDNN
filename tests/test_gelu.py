import pytest
import torch
import torch.nn.functional as F

import flag_dnn

from .accuracy_utils import gems_assert_close


SHAPES = [(32,), (1024,), (5333,), (16384,), (1024 * 1024,)]


@pytest.mark.gelu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("approximate", ['none', 'tanh'])
def test_accuracy_gelu(dtype, shape, approximate):
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

    ref_x = x.clone()
    ref_y = F.gelu(ref_x, approximate=approximate)

    with flag_dnn.use_dnn():
        y = F.gelu(x, approximate=approximate)

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.gelu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("approximate", ['none', 'tanh'])
def test_accuracy_gelu_empty_tensor(dtype, approximate):
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

    ref_y = F.gelu(x, approximate=approximate)
    with flag_dnn.use_dnn():
        y = F.gelu(x, approximate=approximate)

    assert y.shape == (0,)
    assert y.dtype == dtype
    assert y.device == x.device
    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.gelu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("approximate", ['none', 'tanh'])
def test_accuracy_gelu_negative_values(dtype, approximate):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 偏移使其绝大多数为负数
    x = torch.randn(100, dtype=dtype, device=flag_dnn.device) - 2.0

    # 针对不同数据类型动态设置容差 (Tolerance)
    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2  # BF16 精度极低，需要较宽松的容差
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3    # FP16 中等宽松
    else:
        rtol, atol = 1e-5, 1e-5    # FP32 和 FP64 保持严格

    ref_y = F.gelu(x, approximate=approximate)
    with flag_dnn.use_dnn():
        y = F.gelu(x, approximate=approximate)

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.gelu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("approximate", ['none', 'tanh'])
def test_accuracy_gelu_positive_values(dtype, approximate):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 偏移使其绝大多数为正数
    x = torch.randn(100, dtype=dtype, device=flag_dnn.device) + 2.0

    # 针对不同数据类型动态设置容差 (Tolerance)
    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2  # BF16 精度极低，需要较宽松的容差
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3    # FP16 中等宽松
    else:
        rtol, atol = 1e-5, 1e-5    # FP32 和 FP64 保持严格

    ref_y = F.gelu(x, approximate=approximate)
    with flag_dnn.use_dnn():
        y = F.gelu(x, approximate=approximate)

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.gelu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("approximate", ['none', 'tanh'])
def test_accuracy_gelu_mixed_values(dtype, approximate):
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

    ref_y = F.gelu(x, approximate=approximate)
    with flag_dnn.use_dnn():
        y = F.gelu(x, approximate=approximate)

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)