import pytest
import torch
import torch.nn.functional as F
import flag_dnn


GLU_CASES = [
    ((2, 4), 1),
    ((3, 8), -1),
    ((2, 4, 8), 1),
    ((2, 4, 8), 2),
    ((2, 4, 8), -1),
    ((1, 6, 7, 8), 1),
    ((1, 6, 7, 8), -1),
    ((2, 8, 16, 32), 2),
    ((2, 8, 16, 32), -1),
]


def get_tol(dtype):
    if dtype == torch.float16:
        return dict(rtol=2e-3, atol=2e-3)
    if dtype == torch.bfloat16:
        return dict(rtol=2e-2, atol=2e-2)
    if dtype == torch.float32:
        return dict(rtol=1e-6, atol=1e-6)
    return dict(rtol=1e-12, atol=1e-12)


@pytest.mark.glu
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape, dim", GLU_CASES)
def test_accuracy_glu(dtype, shape, dim):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 5.0

    out_ref = F.glu(x, dim=dim)

    with flag_dnn.use_dnn():
        out_custom = F.glu(x.clone(), dim=dim)

    torch.testing.assert_close(out_custom, out_ref, **get_tol(dtype))


@pytest.mark.glu
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
def test_glu_invalid_odd_dim(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn((2, 5, 8), dtype=dtype, device=flag_dnn.device)

    with pytest.raises(RuntimeError):
        with flag_dnn.use_dnn():
            F.glu(x, dim=1)
