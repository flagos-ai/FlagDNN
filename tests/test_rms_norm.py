import pytest
import torch
import torch.nn.functional as F
import flag_dnn


# 定义常用的形状和它们对应的归一化形状 (shape, normalized_shape)
SHAPES_AND_NORM_SHAPES = [
    ((32,), (32,)),  # 1D 张量
    ((1024,), (1024,)),  # 1D 大张量
    ((2, 16), (16,)),  # 2D 张量
    ((4, 8, 32), (32,)),  # 3D 张量 (类似 LLaMA 等 LLM 的序列处理)
    ((4, 8, 32), (8, 32)),  # 3D 张量，归一化最后两维
    ((2, 4, 16, 16), (16, 16)),  # 4D 张量
]


@pytest.mark.rms_norm
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape, normalized_shape", SHAPES_AND_NORM_SHAPES)
@pytest.mark.parametrize("elementwise_affine", [False, True])
def test_accuracy_rms_norm(dtype, shape, normalized_shape, elementwise_affine):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    # 动态设置容差
    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 1e-5, 1e-5

    weight = None
    if elementwise_affine:
        weight = torch.randn(normalized_shape, dtype=dtype, device=flag_dnn.device)

    # PyTorch 官方 API 调用
    ref_y = F.rms_norm(x, normalized_shape, weight=weight)

    # 自定义算子调用
    with flag_dnn.use_dnn():
        y = F.rms_norm(x, normalized_shape, weight=weight)

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.rms_norm
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("elementwise_affine", [False, True])
def test_accuracy_rms_norm_empty_tensor(dtype, elementwise_affine):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    shape = (0, 4, 16)
    normalized_shape = (16,)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 1e-5, 1e-5

    weight = None
    if elementwise_affine:
        weight = torch.randn(normalized_shape, dtype=dtype, device=flag_dnn.device)

    ref_y = F.rms_norm(x, normalized_shape, weight=weight)
    with flag_dnn.use_dnn():
        y = F.rms_norm(x, normalized_shape, weight=weight)

    assert y.shape == shape
    assert y.dtype == dtype
    assert y.device == x.device
    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.rms_norm
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("elementwise_affine", [False, True])
def test_accuracy_rms_norm_large_values(dtype, elementwise_affine):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    shape = (4, 8, 32)
    normalized_shape = (32,)

    # 相比于 LayerNorm 的均值平移，RMSNorm 对大数值的平方和极度敏感
    # 在 FP16/BF16 下更容易溢出，因此需严控数据范围
    if dtype in [torch.float16, torch.bfloat16]:
        x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 5.0 + 20.0
    else:
        x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 100.0 + 1000.0

    if dtype == torch.bfloat16:
        rtol, atol = 5e-2, 5e-2  # BF16 宽容度加大
    elif dtype == torch.float16:
        rtol, atol = 1e-2, 1e-2
    else:
        rtol, atol = 1e-5, 1e-5

    weight = None
    if elementwise_affine:
        if dtype in [torch.float16, torch.bfloat16]:
            weight = (
                torch.randn(normalized_shape, dtype=dtype, device=flag_dnn.device) * 2.0
            )
        else:
            weight = (
                torch.randn(normalized_shape, dtype=dtype, device=flag_dnn.device)
                * 10.0
            )

    ref_y = F.rms_norm(x, normalized_shape, weight=weight)
    with flag_dnn.use_dnn():
        y = F.rms_norm(x, normalized_shape, weight=weight)

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.rms_norm
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("elementwise_affine", [False, True])
def test_accuracy_rms_norm_mixed_values(dtype, elementwise_affine):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 混合常规数据测试，并测试多维度的 normalized_shape
    shape = (4, 8, 16)
    normalized_shape = (8, 16)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    # 动态设置容差
    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 1e-5, 1e-5

    weight = None
    if elementwise_affine:
        weight = torch.randn(normalized_shape, dtype=dtype, device=flag_dnn.device)

    ref_y = F.rms_norm(x, normalized_shape, weight=weight)
    with flag_dnn.use_dnn():
        y = F.rms_norm(x, normalized_shape, weight=weight)

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)
