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

BROADCAST_SHAPES = [
    ((4, 4), (4,)),
    ((2, 3, 4), (3, 1)),
    ((1, 5), (5, 5)),
]

FLOAT_DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
INT_DTYPES = [torch.int8, torch.int16, torch.int32, torch.int64]


@pytest.mark.fmax
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_fmax(shape, dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    y = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    ref_x = utils.to_reference(x)
    ref_y = utils.to_reference(y)
    ref_out = torch.fmax(ref_x, ref_y)
    with flag_dnn.use_dnn():
        res_out = torch.fmax(x, y)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.fmax
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_fmax_with_nan(shape, dtype):
    """Test NaN-aware behavior: fmax ignores NaN."""
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    y = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    if x.numel() > 0:
        x.view(-1)[0] = float("nan")
    ref_x = utils.to_reference(x)
    ref_y = utils.to_reference(y)
    ref_out = torch.fmax(ref_x, ref_y)
    with flag_dnn.use_dnn():
        res_out = torch.fmax(x, y)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.fmax
@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_fmax_int(shape, dtype):
    info = torch.iinfo(dtype)
    low = max(-100, info.min)
    high = min(100, info.max)
    x = torch.randint(low, high, shape, dtype=dtype, device=flag_dnn.device)
    y = torch.randint(low, high, shape, dtype=dtype, device=flag_dnn.device)
    ref_x = utils.to_reference(x)
    ref_y = utils.to_reference(y)
    ref_out = torch.fmax(ref_x, ref_y)
    with flag_dnn.use_dnn():
        res_out = torch.fmax(x, y)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.fmax
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("input_shape, other_shape", BROADCAST_SHAPES)
def test_accuracy_fmax_broadcast(input_shape, other_shape, dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    x = torch.randn(input_shape, dtype=dtype, device=flag_dnn.device)
    y = torch.randn(other_shape, dtype=dtype, device=flag_dnn.device)
    ref_x = utils.to_reference(x)
    ref_y = utils.to_reference(y)
    ref_out = torch.fmax(ref_x, ref_y)
    with flag_dnn.use_dnn():
        res_out = torch.fmax(x, y)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.fmax
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_fmax_empty_tensor(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    x = torch.randn(0, dtype=dtype, device=flag_dnn.device)
    y = torch.randn(0, dtype=dtype, device=flag_dnn.device)
    _ = torch.fmax(x, y)
    with flag_dnn.use_dnn():
        res_out = torch.fmax(x, y)
    assert res_out.shape == (0,)
    assert res_out.dtype == dtype
