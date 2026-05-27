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


@pytest.mark.rsqrt
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_rsqrt(shape, dtype):
    """正数输入（标准情况）"""
    _skip_fp64(dtype)
    x = (
        torch.abs(torch.randn(shape, dtype=dtype, device=flag_dnn.device))
        + 0.1
    )
    ref_inp = utils.to_reference(x)
    ref_out = torch.rsqrt(ref_inp)
    with flag_dnn.use_dnn():
        res_out = torch.rsqrt(x)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.rsqrt
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_rsqrt_small_values(shape, dtype):
    """小正数（rsqrt 结果较大）"""
    _skip_fp64(dtype)
    x = (
        torch.abs(torch.randn(shape, dtype=dtype, device=flag_dnn.device))
        * 0.01
        + 1e-4
    )
    ref_inp = utils.to_reference(x)
    ref_out = torch.rsqrt(ref_inp)
    with flag_dnn.use_dnn():
        res_out = torch.rsqrt(x)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.rsqrt
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_rsqrt_large_values(shape, dtype):
    """大正数（rsqrt 结果较小）"""
    _skip_fp64(dtype)
    x = (
        torch.abs(torch.randn(shape, dtype=dtype, device=flag_dnn.device))
        * 100.0
        + 1.0
    )
    ref_inp = utils.to_reference(x)
    ref_out = torch.rsqrt(ref_inp)
    with flag_dnn.use_dnn():
        res_out = torch.rsqrt(x)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.rsqrt
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rsqrt_empty_tensor(dtype):
    """边界：空张量"""
    _skip_fp64(dtype)
    x = torch.randn(0, dtype=dtype, device=flag_dnn.device)
    ref_inp = utils.to_reference(x)
    ref_out = torch.rsqrt(ref_inp)
    with flag_dnn.use_dnn():
        res_out = torch.rsqrt(x)
    assert res_out.shape == (0,)
    assert res_out.dtype == dtype
    utils.gems_assert_close(res_out, ref_out, dtype)
