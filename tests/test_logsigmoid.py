import pytest
import torch
import torch.nn.functional as F
import flag_dnn


LOGSIGMOID_CASES = [
    (0,),
    (1,),
    (17,),
    (1024,),
    (4096,),
    (2, 3),
    (4, 5, 6),
    (2, 3, 32, 32),
    (1, 64, 112, 112),
]


def get_tol(dtype):
    if dtype == torch.float16:
        return dict(rtol=2e-3, atol=2e-3)
    if dtype == torch.bfloat16:
        return dict(rtol=2e-2, atol=2e-2)
    if dtype == torch.float32:
        return dict(rtol=1e-6, atol=1e-6)
    return dict(rtol=1e-12, atol=1e-12)


@pytest.mark.logsigmoid
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape", LOGSIGMOID_CASES)
def test_accuracy_logsigmoid(dtype, shape):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 8.0
    x_custom = x.clone()
    out_ref = F.logsigmoid(x)

    with flag_dnn.use_dnn():
        out_custom = F.logsigmoid(x_custom)

    torch.testing.assert_close(out_custom, out_ref, **get_tol(dtype))
