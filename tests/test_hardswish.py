import pytest
import torch
import torch.nn.functional as F
import flag_dnn


# (shape, inplace) 的组合测试用例
HARDSWISH_CASES = [
    ((0,), False),
    ((1,), False),
    ((17,), False),
    ((1024,), False),
    ((4096,), False),
    ((2, 3), False),
    ((4, 5, 6), False),
    ((2, 3, 32, 32), False),
    ((1, 64, 112, 112), False),
    ((0,), True),
    ((1,), True),
    ((17,), True),
    ((1024,), True),
    ((4096,), True),
    ((2, 3), True),
    ((4, 5, 6), True),
    ((2, 3, 32, 32), True),
    ((1, 64, 112, 112), True),
]


def get_tol(dtype):
    if dtype == torch.float16:
        return dict(rtol=1e-3, atol=1e-3)
    if dtype == torch.bfloat16:
        return dict(rtol=1e-2, atol=1e-2)
    if dtype == torch.float32:
        return dict(rtol=1e-6, atol=1e-6)
    return dict(rtol=1e-12, atol=1e-12)


@pytest.mark.hardswish
@pytest.mark.parametrize("dtype", [torch.float64])
@pytest.mark.parametrize("shape, inplace", HARDSWISH_CASES)
def test_accuracy_hardswish(dtype, shape, inplace):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 5.0

    x_ref = x.clone()
    x_custom = x.clone()

    out_ref = F.hardswish(x_ref, inplace=inplace)
    with flag_dnn.use_dnn():
        out_custom = F.hardswish(x_custom, inplace=inplace)

    torch.testing.assert_close(out_custom, out_ref, **get_tol(dtype))

    if inplace:
        assert out_custom.data_ptr() == x_custom.data_ptr(), (
            "Inplace flag is True, but output is not modifying "
            "the input tensor directly."
        )
        torch.testing.assert_close(x_custom, x_ref, **get_tol(dtype))
    else:
        if x.numel() > 0:
            assert out_custom.data_ptr() != x_custom.data_ptr(), (
                "Inplace flag is False, but output is modifying "
                "the input tensor memory."
            )
            torch.testing.assert_close(x_custom, x, **get_tol(dtype))
