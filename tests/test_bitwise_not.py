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

INT_DTYPES = [torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64]


@pytest.mark.bitwise_not
@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_bitwise_not(shape, dtype):
    """整数类型基础测试"""
    info = torch.iinfo(dtype)
    low = max(-100, info.min)
    high = min(100, info.max)
    x = torch.randint(low, high, shape, dtype=dtype, device=flag_dnn.device)
    ref_inp = utils.to_reference(x)
    ref_out = torch.bitwise_not(ref_inp)
    with flag_dnn.use_dnn():
        res_out = torch.bitwise_not(x)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bitwise_not
@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_bitwise_not_zeros(shape, dtype):
    """全零张量"""
    x = torch.zeros(shape, dtype=dtype, device=flag_dnn.device)
    ref_inp = utils.to_reference(x)
    ref_out = torch.bitwise_not(ref_inp)
    with flag_dnn.use_dnn():
        res_out = torch.bitwise_not(x)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bitwise_not
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_bitwise_not_bool(shape):
    """bool 类型"""
    x = torch.randint(0, 2, shape, dtype=torch.bool, device=flag_dnn.device)
    ref_inp = utils.to_reference(x)
    ref_out = torch.bitwise_not(ref_inp)
    with flag_dnn.use_dnn():
        res_out = torch.bitwise_not(x)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bitwise_not
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_accuracy_bitwise_not_empty_tensor(dtype):
    """边界：空张量"""
    x = torch.zeros(0, dtype=dtype, device=flag_dnn.device)
    ref_inp = utils.to_reference(x)
    ref_out = torch.bitwise_not(ref_inp)
    with flag_dnn.use_dnn():
        res_out = torch.bitwise_not(x)
    assert res_out.shape == (0,)
    assert res_out.dtype == dtype
    utils.gems_assert_equal(res_out, ref_out)
