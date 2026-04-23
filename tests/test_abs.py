import pytest
import torch
import flag_dnn


SHAPES = [(32,), (1024,), (5333,), (16384,), (1024 * 1024,), (2, 3, 4, 5)]


def _get_tolerances(dtype):
    if dtype == torch.bfloat16:
        return 1.6e-2, 1e-2
    elif dtype == torch.float16:
        return 1e-3, 1e-3
    else:
        return 1e-5, 1e-5


@pytest.mark.abs
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_abs(dtype, shape):
    """最基础的全域测试"""
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    rtol, atol = _get_tolerances(dtype)
    ref_out = torch.abs(x)
    with flag_dnn.use_dnn():
        out = torch.abs(x)

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.abs
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_abs_mixed_values(dtype, shape):
    """细粒度测试：显式测试包含正负数的混合情况"""
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 显式构造混合张量（天然的正态分布即可保证混合）
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    rtol, atol = _get_tolerances(dtype)
    ref_out = torch.abs(x)
    with flag_dnn.use_dnn():
        out = torch.abs(x)

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.abs
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_abs_positive_values(dtype, shape):
    """细粒度测试：纯正数的情况"""
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = (
        torch.abs(torch.randn(shape, dtype=dtype, device=flag_dnn.device))
        + 0.1
    )

    rtol, atol = _get_tolerances(dtype)
    ref_out = torch.abs(x)
    with flag_dnn.use_dnn():
        out = torch.abs(x)

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.abs
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_abs_negative_values(dtype, shape):
    """细粒度测试：纯负数的情况"""
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = (
        -torch.abs(torch.randn(shape, dtype=dtype, device=flag_dnn.device))
        - 0.1
    )

    rtol, atol = _get_tolerances(dtype)
    ref_out = torch.abs(x)
    with flag_dnn.use_dnn():
        out = torch.abs(x)

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.abs
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
def test_accuracy_abs_empty_tensor(dtype):
    """边界情况：空张量测试"""
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(0, dtype=dtype, device=flag_dnn.device)

    rtol, atol = _get_tolerances(dtype)
    ref_out = torch.abs(x)
    with flag_dnn.use_dnn():
        out = torch.abs(x)

    assert out.shape == (0,)
    assert out.dtype == dtype
    assert out.device == x.device
    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.abs
@pytest.mark.parametrize(
    "dtype", [torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_accuracy_abs_integer(dtype):
    x = torch.randint(-9, 10, (257,), dtype=dtype, device=flag_dnn.device)

    ref_out = torch.abs(x)
    with flag_dnn.use_dnn():
        out = torch.abs(x)

    assert out.dtype == ref_out.dtype
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
