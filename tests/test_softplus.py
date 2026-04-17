import pytest
import torch
import torch.nn.functional as F

import flag_dnn


SOFTPLUS_CASES = [
    ((0,), 1.0, 20.0),
    ((1,), 1.0, 20.0),
    ((17,), 1.0, 20.0),
    ((1024,), 1.0, 20.0),
    ((4096,), 1.0, 20.0),
    ((2, 3), 1.0, 20.0),
    ((4, 5, 6), 1.0, 20.0),
    ((2, 3, 32, 32), 1.0, 20.0),
    ((1, 64, 112, 112), 1.0, 20.0),
    ((1024,), 0.5, 20.0),
    ((1024,), 2.0, 20.0),
    ((1024,), 1.0, 10.0),
    ((1024,), 1.0, 30.0),
    ((2, 3, 32, 32), 0.5, 10.0),
    ((2, 3, 32, 32), 2.0, 30.0),
]


def get_tol(dtype):
    if dtype == torch.float16:
        return dict(rtol=1e-3, atol=1e-3)
    if dtype == torch.bfloat16:
        return dict(rtol=1e-2, atol=1e-2)
    if dtype == torch.float32:
        return dict(rtol=1e-6, atol=1e-6)
    return dict(rtol=1e-12, atol=1e-12)


@pytest.mark.softplus
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape, beta, threshold", SOFTPLUS_CASES)
def test_accuracy_softplus(dtype, shape, beta, threshold):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 5.0

    x_ref = x.clone()
    x_custom = x.clone()

    out_ref = F.softplus(x_ref, beta=beta, threshold=threshold)

    with flag_dnn.use_dnn():
        out_custom = F.softplus(x_custom, beta=beta, threshold=threshold)

    torch.testing.assert_close(out_custom, out_ref, **get_tol(dtype))

    if x.numel() > 0:
        assert (
            out_custom.data_ptr() != x_custom.data_ptr()
        ), "softplus should be out-of-place, but output shares input memory."
        torch.testing.assert_close(x_custom, x, **get_tol(dtype))


@pytest.mark.softplus
def test_softplus_invalid_beta():
    x = torch.randn((16,), dtype=torch.float32, device=flag_dnn.device)

    with flag_dnn.use_dnn():
        with pytest.raises(ValueError):
            F.softplus(x, beta=0.0, threshold=20.0)
