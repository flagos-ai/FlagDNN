import pytest
import torch
import flag_dnn
from tests import accuracy_utils as utils

SHAPES = list(utils.POINTWISE_SHAPES) + [(32,), (1024,), (5333,), (65536,), (1024 * 1024,)]

FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _skip_fp64(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")


@pytest.mark.isnan
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_isnan(shape, dtype):
    """基础测试：纯正常浮点数"""
    _skip_fp64(dtype)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    ref_inp = utils.to_reference(x)
    ref_out = torch.isnan(ref_inp)
    with flag_dnn.use_dnn():
        res_out = torch.isnan(x)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.isnan
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_isnan_with_nan(shape, dtype):
    """包含 NaN 值"""
    _skip_fp64(dtype)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    x.view(-1)[::10] = float("nan")
    ref_inp = utils.to_reference(x)
    ref_out = torch.isnan(ref_inp)
    with flag_dnn.use_dnn():
        res_out = torch.isnan(x)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.isnan
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_isnan_with_inf(shape, dtype):
    """包含 inf：isnan(inf) 应为 False"""
    _skip_fp64(dtype)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    x.view(-1)[::10] = float("inf")
    ref_inp = utils.to_reference(x)
    ref_out = torch.isnan(ref_inp)
    with flag_dnn.use_dnn():
        res_out = torch.isnan(x)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.isnan
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_isnan_with_mixed_special(shape, dtype):
    """同时包含 nan 和 inf"""
    _skip_fp64(dtype)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    flat = x.view(-1)
    flat[::10] = float("nan")
    flat[1::10] = float("inf")
    ref_inp = utils.to_reference(x)
    ref_out = torch.isnan(ref_inp)
    with flag_dnn.use_dnn():
        res_out = torch.isnan(x)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.isnan
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_isnan_empty_tensor(dtype):
    """边界：空张量"""
    _skip_fp64(dtype)
    x = torch.randn(0, dtype=dtype, device=flag_dnn.device)
    ref_inp = utils.to_reference(x)
    ref_out = torch.isnan(ref_inp)
    with flag_dnn.use_dnn():
        res_out = torch.isnan(x)
    assert res_out.shape == (0,)
    assert res_out.dtype == torch.bool
    utils.gems_assert_equal(res_out, ref_out)
