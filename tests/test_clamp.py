import pytest
import torch
import flag_dnn


SHAPES = [(32,), (1024,), (5333,), (16384,), (1024 * 1024,), (2, 3, 4, 5)]

# 测试组合：(min_val, max_val)
CLAMP_BOUNDS = [
    (-0.5, 0.5),  # 正常双边界
    (0.0, None),  # 只有下界 (类似于 ReLU)
    (None, 0.0),  # 只有上界
    (0.5, -0.5),  # 异常边界：min > max，预期全部被 clamp 到 max (-0.5)
]


def _get_tolerances(dtype):
    if dtype == torch.bfloat16:
        return 1.6e-2, 1e-2
    elif dtype == torch.float16:
        return 1e-3, 1e-3
    else:
        return 1e-5, 1e-5


@pytest.mark.clamp
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("min_val, max_val", CLAMP_BOUNDS)
def test_accuracy_clamp(dtype, shape, min_val, max_val):
    """最基础的全域测试"""
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    rtol, atol = _get_tolerances(dtype)
    ref_out = torch.clamp(x, min=min_val, max=max_val)
    with flag_dnn.use_dnn():
        out = torch.clamp(x, min=min_val, max=max_val)

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.clamp
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("min_val, max_val", CLAMP_BOUNDS)
def test_accuracy_clamp_mixed_values(dtype, shape, min_val, max_val):
    """细粒度测试：显式测试包含正负数的混合情况"""
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    rtol, atol = _get_tolerances(dtype)
    ref_out = torch.clamp(x, min=min_val, max=max_val)
    with flag_dnn.use_dnn():
        out = torch.clamp(x, min=min_val, max=max_val)

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.clamp
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("min_val, max_val", [(0.1, 0.5), (0.5, None), (None, 0.2)])
def test_accuracy_clamp_positive_values(dtype, shape, min_val, max_val):
    """细粒度测试：纯正数的情况"""
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.abs(torch.randn(shape, dtype=dtype, device=flag_dnn.device)) + 0.1

    rtol, atol = _get_tolerances(dtype)
    ref_out = torch.clamp(x, min=min_val, max=max_val)
    with flag_dnn.use_dnn():
        out = torch.clamp(x, min=min_val, max=max_val)

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.clamp
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("min_val, max_val", [(-0.5, -0.1), (-0.5, None), (None, -0.2)])
def test_accuracy_clamp_negative_values(dtype, shape, min_val, max_val):
    """细粒度测试：纯负数的情况"""
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = -torch.abs(torch.randn(shape, dtype=dtype, device=flag_dnn.device)) - 0.1

    rtol, atol = _get_tolerances(dtype)
    ref_out = torch.clamp(x, min=min_val, max=max_val)
    with flag_dnn.use_dnn():
        out = torch.clamp(x, min=min_val, max=max_val)

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.clamp
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("min_val, max_val", CLAMP_BOUNDS)
def test_accuracy_clamp_empty_tensor(dtype, min_val, max_val):
    """边界情况：空张量测试"""
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(0, dtype=dtype, device=flag_dnn.device)

    rtol, atol = _get_tolerances(dtype)
    ref_out = torch.clamp(x, min=min_val, max=max_val)
    with flag_dnn.use_dnn():
        out = torch.clamp(x, min=min_val, max=max_val)

    assert out.shape == (0,)
    assert out.dtype == dtype
    assert out.device == x.device
    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.clamp
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_clamp_tensor_bounds_same_shape(dtype, shape):
    """测试边界为相同形状 Tensor 的情况"""
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    # 构造同形状的 min 和 max Tensor
    min_t = torch.randn(shape, dtype=dtype, device=flag_dnn.device) - 1.0
    max_t = min_t + 2.0  # 确保 max > min

    rtol, atol = _get_tolerances(dtype)

    ref_out = torch.clamp(x, min=min_t, max=max_t)
    with flag_dnn.use_dnn():
        out = torch.clamp(x, min=min_t, max=max_t)

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)

    # 测试仅有 Tensor min
    ref_out_min = torch.clamp(x, min=min_t)
    with flag_dnn.use_dnn():
        out_min = torch.clamp(x, min=min_t)

    torch.testing.assert_close(out_min, ref_out_min, rtol=rtol, atol=atol)


@pytest.mark.clamp
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_accuracy_clamp_tensor_bounds_broadcast(dtype):
    """测试边界为需要广播的 Tensor (例如标量 Tensor 或 1D Tensor)"""
    shape = (4, 16, 32)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    # 1. 标量 Tensor 广播
    min_scalar_t = torch.tensor(-0.5, dtype=dtype, device=flag_dnn.device)
    max_scalar_t = torch.tensor(0.5, dtype=dtype, device=flag_dnn.device)

    rtol, atol = _get_tolerances(dtype)

    ref_out = torch.clamp(x, min=min_scalar_t, max=max_scalar_t)
    with flag_dnn.use_dnn():
        out = torch.clamp(x, min=min_scalar_t, max=max_scalar_t)

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)

    # 2. 尾部维度广播 (例如 1D Tensor [32] 广播到 [4, 16, 32])
    min_1d_t = torch.randn(32, dtype=dtype, device=flag_dnn.device) - 1.0
    max_1d_t = min_1d_t + 2.0

    ref_out = torch.clamp(x, min=min_1d_t, max=max_1d_t)
    with flag_dnn.use_dnn():
        out = torch.clamp(x, min=min_1d_t, max=max_1d_t)

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)
