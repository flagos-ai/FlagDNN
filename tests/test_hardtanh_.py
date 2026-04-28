import pytest
import torch
import torch.nn.functional as F
import flag_dnn


# (shape, min_val, max_val) 的组合测试用例
HARDTANH_CASES = [
    ((1,), -1.0, 1.0),
    ((16,), -1.0, 1.0),
    ((1024,), -2.0, 2.0),
    ((2, 3), -1.0, 1.0),
    ((4, 8, 16), -0.5, 0.5),
    ((2, 3, 32, 32), -1.0, 1.0),
    ((1, 128, 64, 64), -3.0, 3.0),
    ((0,), -1.0, 1.0),
    ((0, 3), -1.0, 1.0),
]

INTEGER_HARDTANH_CASES = [
    ((), -1, 1),
    ((1,), -1, 1),
    ((17,), -2, 2),
    ((17, 31), -3, 3),
    ((2, 3, 4, 5), -1, 1),
    ((0, 3), -1, 1),
]

INTEGER_DTYPES = [torch.int8, torch.int16, torch.int32, torch.int64]


def get_tol(dtype):
    if dtype == torch.float16:
        return dict(rtol=1e-3, atol=1e-3)
    if dtype == torch.bfloat16:
        return dict(rtol=1e-2, atol=1e-2)
    if dtype == torch.float32:
        return dict(rtol=1e-6, atol=1e-6)
    if dtype == torch.float64:
        return dict(rtol=1e-12, atol=1e-12)
    return dict(rtol=1e-12, atol=1e-12)


@pytest.mark.hardtanh_
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape, min_val, max_val", HARDTANH_CASES)
def test_accuracy_hardtanh_(dtype, shape, min_val, max_val):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 5.0

    x_ref = x.clone()
    x_custom = x.clone()

    out_ref = F.hardtanh_(x_ref, min_val=min_val, max_val=max_val)

    with flag_dnn.use_dnn():
        out_custom = F.hardtanh_(x_custom, min_val=min_val, max_val=max_val)

    torch.testing.assert_close(out_custom, out_ref, **get_tol(dtype))

    assert out_custom.data_ptr() == x_custom.data_ptr(), (
        "output is not modifying " "the input tensor directly."
    )
    torch.testing.assert_close(x_custom, x_ref, **get_tol(dtype))


@pytest.mark.hardtanh_
@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
@pytest.mark.parametrize("shape, min_val, max_val", INTEGER_HARDTANH_CASES)
def test_accuracy_hardtanh__integer_dtype(dtype, shape, min_val, max_val):
    x = torch.randint(-5, 6, shape, dtype=dtype, device=flag_dnn.device)

    x_ref = x.clone()
    x_custom = x.clone()

    out_ref = F.hardtanh_(x_ref, min_val=min_val, max_val=max_val)
    with flag_dnn.use_dnn():
        out_custom = F.hardtanh_(x_custom, min_val=min_val, max_val=max_val)

    assert out_custom.dtype == dtype
    assert out_custom.data_ptr() == x_custom.data_ptr()
    torch.testing.assert_close(out_custom, out_ref, rtol=0, atol=0)
    torch.testing.assert_close(x_custom, x_ref, rtol=0, atol=0)
