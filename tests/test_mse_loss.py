import pytest
import torch
import torch.nn.functional as F
import flag_dnn
from tests import accuracy_utils as utils

SHAPES = list(utils.POINTWISE_SHAPES) + [(32,), (1024,), (5333,), (2, 3, 64)]
REDUCTION_MODES = ["none", "mean", "sum"]
FLOAT_DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


def _get_tolerances(dtype):
    if dtype == torch.bfloat16:
        return 1.6e-2, 1e-2
    elif dtype == torch.float16:
        return 1e-2, 1e-2
    else:
        return 1e-4, 1e-4


@pytest.mark.mse_loss
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", REDUCTION_MODES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_mse_loss(shape, reduction, dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    inp = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    tgt = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    ref_inp = utils.to_reference(inp)
    ref_tgt = utils.to_reference(tgt)

    rtol, atol = _get_tolerances(dtype)
    ref_out = F.mse_loss(ref_inp, ref_tgt, reduction=reduction)
    with flag_dnn.use_dnn():
        out = F.mse_loss(inp, tgt, reduction=reduction)
    torch.testing.assert_close(out, ref_out.to(out.dtype), rtol=rtol, atol=atol)


@pytest.mark.mse_loss
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mse_loss_empty_tensor(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    inp = torch.randn(0, dtype=dtype, device=flag_dnn.device)
    tgt = torch.randn(0, dtype=dtype, device=flag_dnn.device)
    ref_out = F.mse_loss(inp, tgt, reduction="none")
    with flag_dnn.use_dnn():
        out = F.mse_loss(inp, tgt, reduction="none")
    assert out.shape == (0,)
    assert out.dtype == dtype
