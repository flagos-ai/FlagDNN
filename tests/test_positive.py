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


@pytest.mark.positive
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_positive(shape, dtype):
    """基础测试：identity 不改变值"""
    _skip_fp64(dtype)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    ref_inp = utils.to_reference(x)
    ref_out = torch.positive(ref_inp)
    with flag_dnn.use_dnn():
        res_out = torch.positive(x)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.positive
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_positive_negative_values(shape, dtype):
    """纯负数（identity 不改变值）"""
    _skip_fp64(dtype)
    x = (
        -torch.abs(torch.randn(shape, dtype=dtype, device=flag_dnn.device))
        - 0.1
    )
    ref_inp = utils.to_reference(x)
    ref_out = torch.positive(ref_inp)
    with flag_dnn.use_dnn():
        res_out = torch.positive(x)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.positive
def test_accuracy_positive_bool_raises():
    """bool 输入应抛出 RuntimeError"""
    x = torch.tensor([True, False], device=flag_dnn.device)
    with pytest.raises(RuntimeError):
        with flag_dnn.use_dnn():
            torch.positive(x)


@pytest.mark.positive
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_positive_empty_tensor(dtype):
    """边界：空张量"""
    _skip_fp64(dtype)
    x = torch.randn(0, dtype=dtype, device=flag_dnn.device)
    ref_inp = utils.to_reference(x)
    ref_out = torch.positive(ref_inp)
    with flag_dnn.use_dnn():
        res_out = torch.positive(x)
    assert res_out.shape == (0,)
    assert res_out.dtype == dtype
    utils.gems_assert_close(res_out, ref_out, dtype)
