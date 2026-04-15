import pytest
import torch
import torch.nn.functional as F
import flag_dnn


# (shape, alpha)
ELU_CASES = [
    ((1,), 1.0),
    ((16,), 1.0),
    ((1024,), 0.5),
    ((1024,), 2.0),
    ((2, 3), 1.0),
    ((4, 8, 16), 1.0),
    ((2, 3, 32, 32), 1.0),
    ((1, 128, 64, 64), 0.75),
    ((0,), 1.0),
    ((0, 3), 1.0),
]


def get_tol(dtype):
    if dtype == torch.float16:
        return dict(rtol=2e-3, atol=2e-3)
    if dtype == torch.bfloat16:
        return dict(rtol=2e-2, atol=2e-2)
    if dtype == torch.float32:
        return dict(rtol=1e-6, atol=1e-6)
    if dtype == torch.float64:
        return dict(rtol=1e-12, atol=1e-12)
    return dict(rtol=1e-12, atol=1e-12)


@pytest.mark.elu_
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape, alpha", ELU_CASES)
def test_accuracy_elu_(dtype, shape, alpha):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 3.0

    x_ref = x.clone()
    x_custom = x.clone()

    out_ref = F.elu_(x_ref, alpha=alpha)

    with flag_dnn.use_dnn():
        out_custom = F.elu_(x_custom, alpha=alpha)

    torch.testing.assert_close(out_custom, out_ref, **get_tol(dtype))

    assert out_custom.data_ptr() == x_custom.data_ptr(), (
        "output is not modifying " "the input tensor directly."
    )
    torch.testing.assert_close(x_custom, x_ref, **get_tol(dtype))
