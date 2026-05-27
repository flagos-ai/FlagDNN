import pytest
import torch
import flag_dnn
from tests import accuracy_utils as utils

SHAPES = list(utils.POINTWISE_SHAPES) + [
    (32,),
    (1024,),
    (5333,),
    (65536,),
    (1024 * 1024,),
]

FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _skip_fp64(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")


@pytest.mark.exp
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_exp(shape, dtype):
    """基础测试：正负混合"""
    _skip_fp64(dtype)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    ref_inp = utils.to_reference(x)
    ref_out = torch.exp(ref_inp)
    with flag_dnn.use_dnn():
        res_out = torch.exp(x)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.exp
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_exp_positive_values(shape, dtype):
    """正数输入（exp > 1）"""
    _skip_fp64(dtype)
    x = (
        torch.abs(torch.randn(shape, dtype=dtype, device=flag_dnn.device))
        + 0.1
    )
    ref_inp = utils.to_reference(x)
    ref_out = torch.exp(ref_inp)
    with flag_dnn.use_dnn():
        res_out = torch.exp(x)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.exp
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_exp_negative_values(shape, dtype):
    """负数输入（exp 结果在 (0,1) 之间）"""
    _skip_fp64(dtype)
    x = (
        -torch.abs(torch.randn(shape, dtype=dtype, device=flag_dnn.device))
        - 0.1
    )
    ref_inp = utils.to_reference(x)
    ref_out = torch.exp(ref_inp)
    with flag_dnn.use_dnn():
        res_out = torch.exp(x)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.exp
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_exp_zeros(shape, dtype):
    """全零张量（exp(0) = 1）"""
    _skip_fp64(dtype)
    x = torch.zeros(shape, dtype=dtype, device=flag_dnn.device)
    ref_inp = utils.to_reference(x)
    ref_out = torch.exp(ref_inp)
    with flag_dnn.use_dnn():
        res_out = torch.exp(x)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.exp
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_exp_empty_tensor(dtype):
    """边界：空张量"""
    _skip_fp64(dtype)
    x = torch.randn(0, dtype=dtype, device=flag_dnn.device)
    ref_inp = utils.to_reference(x)
    ref_out = torch.exp(ref_inp)
    with flag_dnn.use_dnn():
        res_out = torch.exp(x)
    assert res_out.shape == (0,)
    assert res_out.dtype == dtype
    utils.gems_assert_close(res_out, ref_out, dtype)
