import pytest
import torch
import torch.nn.functional as F
import flag_dnn
from tests import accuracy_utils as utils

SHAPES = list(utils.POINTWISE_SHAPES) + [
    (32,),
    (1024,),
    (2, 16),
    (4, 8, 32),
    (2, 4, 16, 16),
]
DIMS = [-1, 0, 1, 2]


@pytest.mark.log_softmax
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dim", DIMS)
def test_accuracy_log_softmax(dtype, shape, dim):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    if dim >= len(shape) or dim < -len(shape):
        pytest.skip(f"dim {dim} out of bounds for shape {shape}")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 1e-5, 1e-5

    ref_y = F.log_softmax(x, dim=dim)
    with flag_dnn.use_dnn():
        y = F.log_softmax(x, dim=dim)

    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)


@pytest.mark.log_softmax
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
def test_accuracy_log_softmax_empty_tensor(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    shape = (0, 4, 16)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    _ = F.log_softmax(x, dim=-1)
    with flag_dnn.use_dnn():
        y = F.log_softmax(x, dim=-1)
    assert y.shape == shape
    assert y.dtype == dtype


@pytest.mark.log_softmax
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("dim", [-1, 1])
def test_accuracy_log_softmax_large_values(dtype, dim):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    shape = (4, 8, 32)
    if dim >= len(shape):
        pytest.skip()
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 10.0 + 100.0
    if dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-2
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 1e-5, 1e-5
    ref_y = F.log_softmax(x, dim=dim)
    with flag_dnn.use_dnn():
        y = F.log_softmax(x, dim=dim)
    torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol)
