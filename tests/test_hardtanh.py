import pytest
import torch
import torch.nn.functional as F
import flag_dnn


# (shape, min_val, max_val, inplace) 的组合测试用例
HARDTANH_CASES = [
    ((1,), -1.0, 1.0, False),
    ((1,), -1.0, 1.0, True),
    ((16,), -1.0, 1.0, False),
    ((16,), -1.0, 1.0, True),
    ((1024,), -2.0, 2.0, False),
    ((1024,), -2.0, 2.0, True),
    ((2, 3), -1.0, 1.0, False),
    ((2, 3), -1.0, 1.0, True),
    ((4, 8, 16), -0.5, 0.5, False),
    ((4, 8, 16), -0.5, 0.5, True),
    ((2, 3, 32, 32), -1.0, 1.0, False),
    ((2, 3, 32, 32), -1.0, 1.0, True),
    ((1, 128, 64, 64), -3.0, 3.0, False),
    ((1, 128, 64, 64), -3.0, 3.0, True),
    ((0,), -1.0, 1.0, False),
    ((0,), -1.0, 1.0, True),
    ((0, 3), -1.0, 1.0, False),
    ((0, 3), -1.0, 1.0, True),
]


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


@pytest.mark.hardtanh
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape, min_val, max_val, inplace", HARDTANH_CASES)
def test_accuracy_hardtanh(dtype, shape, min_val, max_val, inplace):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 5.0

    x_ref = x.clone()
    x_custom = x.clone()

    out_ref = F.hardtanh(
        x_ref, min_val=min_val, max_val=max_val, inplace=inplace
    )

    with flag_dnn.use_dnn():
        out_custom = F.hardtanh(
            x_custom, min_val=min_val, max_val=max_val, inplace=inplace
        )

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
