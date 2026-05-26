import pytest
import torch
import flag_dnn
from tests import accuracy_utils as utils

SHAPES = list(utils.POINTWISE_SHAPES) + [(32,), (1024,), (5333,), (65536,), (1024 * 1024,)]

BROADCAST_SHAPES = [
    ((4, 4), (4,)),
    ((2, 3, 4), (3, 1)),
    ((1, 5), (5, 5)),
]

INT_DTYPES = [torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64]


@pytest.mark.bitwise_or
@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_bitwise_or(shape, dtype):
    info = torch.iinfo(dtype)
    low = max(-100, info.min)
    high = min(100, info.max)
    x = torch.randint(low, high, shape, dtype=dtype, device=flag_dnn.device)
    y = torch.randint(low, high, shape, dtype=dtype, device=flag_dnn.device)
    ref_x = utils.to_reference(x)
    ref_y = utils.to_reference(y)
    ref_out = torch.bitwise_or(ref_x, ref_y)
    with flag_dnn.use_dnn():
        res_out = torch.bitwise_or(x, y)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bitwise_or
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_bitwise_or_bool(shape):
    x = torch.randint(0, 2, shape, dtype=torch.bool, device=flag_dnn.device)
    y = torch.randint(0, 2, shape, dtype=torch.bool, device=flag_dnn.device)
    ref_x = utils.to_reference(x)
    ref_y = utils.to_reference(y)
    ref_out = torch.bitwise_or(ref_x, ref_y)
    with flag_dnn.use_dnn():
        res_out = torch.bitwise_or(x, y)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bitwise_or
@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_bitwise_or_zeros(shape, dtype):
    x = torch.zeros(shape, dtype=dtype, device=flag_dnn.device)
    y = torch.zeros(shape, dtype=dtype, device=flag_dnn.device)
    ref_x = utils.to_reference(x)
    ref_y = utils.to_reference(y)
    ref_out = torch.bitwise_or(ref_x, ref_y)
    with flag_dnn.use_dnn():
        res_out = torch.bitwise_or(x, y)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bitwise_or
@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("input_shape, other_shape", BROADCAST_SHAPES)
def test_accuracy_bitwise_or_broadcast(input_shape, other_shape, dtype):
    info = torch.iinfo(dtype)
    low = max(-100, info.min)
    high = min(100, info.max)
    x = torch.randint(low, high, input_shape, dtype=dtype, device=flag_dnn.device)
    y = torch.randint(low, high, other_shape, dtype=dtype, device=flag_dnn.device)
    ref_x = utils.to_reference(x)
    ref_y = utils.to_reference(y)
    ref_out = torch.bitwise_or(ref_x, ref_y)
    with flag_dnn.use_dnn():
        res_out = torch.bitwise_or(x, y)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bitwise_or
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_accuracy_bitwise_or_empty_tensor(dtype):
    x = torch.zeros(0, dtype=dtype, device=flag_dnn.device)
    y = torch.zeros(0, dtype=dtype, device=flag_dnn.device)
    ref_x = utils.to_reference(x)
    ref_y = utils.to_reference(y)
    ref_out = torch.bitwise_or(ref_x, ref_y)
    with flag_dnn.use_dnn():
        res_out = torch.bitwise_or(x, y)
    assert res_out.shape == (0,)
    assert res_out.dtype == dtype
    utils.gems_assert_equal(res_out, ref_out)
