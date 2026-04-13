import pytest
import torch
import torch.nn.functional as F
import flag_dnn


SHAPES = [
    (1,),  # 单元素
    (32,),  # 小 1D
    (1024,),  # 对齐 1D
    (5333,),  # 非对齐 1D
    (17, 31),  # 小 2D，非对齐
    (128, 256),  # 常见 2D
    (4, 8, 16),  # 3D
    (2, 3, 32, 32),  # 常见 4D
    (16, 64, 56, 56),  # 更接近实际模型场景
    (1024 * 1024,),  # 大 1D
]

NEGATIVE_SLOPES = [0.01, 0.2]  # 测试默认斜率和较大的斜率


def get_tol(dtype):
    if dtype == torch.float16:
        return dict(rtol=1e-3, atol=1e-3)
    if dtype == torch.bfloat16:
        return dict(rtol=1e-2, atol=1e-2)
    if dtype == torch.float32:
        return dict(rtol=1e-6, atol=1e-6)
    return dict(rtol=1e-12, atol=1e-12)


@pytest.mark.leaky_relu
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("negative_slope", NEGATIVE_SLOPES)
def test_accuracy_leaky_relu(dtype, shape, inplace, negative_slope):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    # 必须 clone，防止 inplace=True 时原生算子破坏输入数据
    ref_x = x.clone()
    ref_y = F.leaky_relu(ref_x, negative_slope=negative_slope, inplace=inplace)

    with flag_dnn.use_dnn():
        y = F.leaky_relu(x, negative_slope=negative_slope, inplace=inplace)

    torch.testing.assert_close(y, ref_y, **get_tol(dtype))


@pytest.mark.leaky_relu
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("negative_slope", NEGATIVE_SLOPES)
def test_accuracy_leaky_relu_empty_tensor(dtype, inplace, negative_slope):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(0, dtype=dtype, device=flag_dnn.device)

    ref_x = x.clone()
    ref_y = F.leaky_relu(ref_x, negative_slope=negative_slope, inplace=inplace)

    with flag_dnn.use_dnn():
        y = F.leaky_relu(x, negative_slope=negative_slope, inplace=inplace)

    assert y.shape == (0,)
    assert y.dtype == dtype
    assert y.device == x.device
    torch.testing.assert_close(y, ref_y, **get_tol(dtype))


@pytest.mark.leaky_relu
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("negative_slope", NEGATIVE_SLOPES)
def test_accuracy_leaky_relu_negative_values(dtype, inplace, negative_slope):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(100, dtype=dtype, device=flag_dnn.device) - 2.0

    ref_x = x.clone()
    ref_y = F.leaky_relu(ref_x, negative_slope=negative_slope, inplace=inplace)

    with flag_dnn.use_dnn():
        y = F.leaky_relu(x, negative_slope=negative_slope, inplace=inplace)

    torch.testing.assert_close(y, ref_y, **get_tol(dtype))


@pytest.mark.leaky_relu
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("negative_slope", NEGATIVE_SLOPES)
def test_accuracy_leaky_relu_positive_values(dtype, inplace, negative_slope):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(100, dtype=dtype, device=flag_dnn.device) + 2.0

    ref_x = x.clone()
    ref_y = F.leaky_relu(ref_x, negative_slope=negative_slope, inplace=inplace)

    with flag_dnn.use_dnn():
        y = F.leaky_relu(x, negative_slope=negative_slope, inplace=inplace)

    torch.testing.assert_close(y, ref_y, **get_tol(dtype))


@pytest.mark.leaky_relu
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("negative_slope", NEGATIVE_SLOPES)
def test_accuracy_leaky_relu_mixed_values(dtype, inplace, negative_slope):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(100, dtype=dtype, device=flag_dnn.device)

    ref_x = x.clone()
    ref_y = F.leaky_relu(ref_x, negative_slope=negative_slope, inplace=inplace)

    with flag_dnn.use_dnn():
        y = F.leaky_relu(x, negative_slope=negative_slope, inplace=inplace)

    torch.testing.assert_close(y, ref_y, **get_tol(dtype))
