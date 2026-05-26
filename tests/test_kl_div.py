import pytest
import torch
import torch.nn.functional as F
import flag_dnn
from tests import accuracy_utils as utils

SHAPES = [(1024,), (2, 512), (4, 8, 32)]
REDUCTION_MODES = ["none", "mean", "sum", "batchmean"]
FLOAT_DTYPES = [torch.float32, torch.float64]


def _get_tolerances(dtype):
    if dtype == torch.float64:
        return 1e-4, 1e-4
    return 1e-3, 1e-3


@pytest.mark.kl_div
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", REDUCTION_MODES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_kl_div(shape, reduction, dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    # input is log-probabilities (use log_softmax to create valid log-probs)
    inp = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    inp = F.log_softmax(inp, dim=-1)
    # target is probabilities (non-negative, sum to 1)
    tgt = torch.rand(shape, dtype=dtype, device=flag_dnn.device)
    tgt = tgt / tgt.sum(dim=-1, keepdim=True)

    ref_inp = utils.to_reference(inp)
    ref_tgt = utils.to_reference(tgt)

    rtol, atol = _get_tolerances(dtype)
    ref_out = F.kl_div(ref_inp, ref_tgt, reduction=reduction)
    with flag_dnn.use_dnn():
        out = F.kl_div(inp, tgt, reduction=reduction)
    torch.testing.assert_close(out, ref_out.to(out.dtype), rtol=rtol, atol=atol)


@pytest.mark.kl_div
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", REDUCTION_MODES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_kl_div_log_target(shape, reduction, dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    inp = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    inp = F.log_softmax(inp, dim=-1)
    tgt = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    tgt = F.log_softmax(tgt, dim=-1)

    ref_inp = utils.to_reference(inp)
    ref_tgt = utils.to_reference(tgt)

    rtol, atol = _get_tolerances(dtype)
    ref_out = F.kl_div(ref_inp, ref_tgt, reduction=reduction, log_target=True)
    with flag_dnn.use_dnn():
        out = F.kl_div(inp, tgt, reduction=reduction, log_target=True)
    torch.testing.assert_close(out, ref_out.to(out.dtype), rtol=rtol, atol=atol)
